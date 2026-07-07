import os
import torch
import json
from torch.utils.data import Dataset, DataLoader
from tokenizers import Tokenizer
from model.esm.utils.structure.protein_chain import ProteinChain
from model.esm.tokenization.sequence_tokenizer import EsmSequenceTokenizer
from model.esm.utils.encoding import tokenize_sequence
import numpy as np
from Bio.PDB import PDBParser, MMCIFParser

# 例：64 个距离 bin，从 2Å 到 22Å
BIN_EDGES = np.linspace(2.0, 22.0, num=65)  # 65 个边界 => 64 个区间
NUM_BINS  = len(BIN_EDGES) - 1


class ProteinPeptideDataset(Dataset):
    def __init__(self, id_list, pdb_dir, json_files, interface_json: str | None = None):  # === NEW: interface_json
        """
        Args:
        - id_list: like ['1a0m_A', '2b0n_B']
        - pdb_dir: root folder that contains <ID>/peptide.pdb and <ID>/receptor.pdb
        - json_files: list[str], attribute JSONs you already use
        - interface_json: path to interface_labels.json produced earlier (可为 None)
        """
        self.id_list = id_list
        self.pdb_dir = pdb_dir

        # 聚合多个属性 JSON
        self.peptide_attributes: dict[str, dict] = {}
        for json_file in json_files:
            with open(json_file, 'r') as f:
                data = json.load(f)
            for k, v in data.items():
                self.peptide_attributes.setdefault(k, {}).update(v)

        # === NEW: 载入 interface_labels JSON（如提供）
        self.interface_db: dict[str, dict] = {}
        if interface_json is not None and os.path.isfile(interface_json):
            with open(interface_json, "r", encoding="utf-8") as f:
                self.interface_db = json.load(f)
        # 如果没给/找不到，就保持为空 dict，后续自动生成 -100 占位

    def __len__(self):
        return len(self.id_list)

    def __getitem__(self, idx):
        sample_id = self.id_list[idx]
        peptide_pdb_file = os.path.join(self.pdb_dir, f"{sample_id}/peptide.pdb")
        protein_pdb_file = os.path.join(self.pdb_dir, f"{sample_id}/receptor.pdb")

        # 提取序列（None 则抛错）
        receptor_seq = extract_sequence_from_pdb(protein_pdb_file)
        peptide_seq  = extract_sequence_from_pdb(peptide_pdb_file)
        if peptide_seq is None:
            raise ValueError(f"Missing peptide sequence for index {sample_id}")
        peptide_seq = "<|bos|>" + peptide_seq + "<|eos|>"

        # tokens
        sequence_tokenizer = EsmSequenceTokenizer()
        receptor_seq_tensor = tokenize_sequence(receptor_seq, sequence_tokenizer)
        peptide_seq_tensor  = tokenize_sequence(peptide_seq,  sequence_tokenizer)

        # 属性特征
        peptide_attr = self.peptide_attributes.get(sample_id, {})
        binding_pocket_tensor = torch.tensor(peptide_attr.get('interaction_windows', [[0.0, 0.0, 0.0]]))
        vina_affinity_tensor  = torch.tensor(peptide_attr.get('score', -100.0)).unsqueeze(0)
        stability_tensor      = torch.tensor(peptide_attr.get('stability_score', 40)).unsqueeze(0)
        solubility_tensor     = torch.tensor(peptide_attr.get('solubility_score', 40)).unsqueeze(0)

        # 结构标签（多肽自距阵）
        distogram = extract_distogram_labels(peptide_pdb_file)         # [L_pep, L_pep]
        L_pep = distogram.size(0)
        # === NEW: 受体 CA 长度，用于对齐 interface_labels
        L_rec = count_ca_len(protein_pdb_file)                          # [int]

        # === NEW: 取 interface_labels，裁剪/填充到 [L_pep, L_rec]
        interface_labels = self._fetch_and_align_interface(sample_id, L_pep, L_rec)

        return {
            "id": sample_id,
            "peptide_seq": peptide_seq,
            "receptor_seq": receptor_seq,
            "peptide_seq_tensor": peptide_seq_tensor,
            "receptor_seq_tensor": receptor_seq_tensor,
            "binding_pocket": binding_pocket_tensor.to(torch.float),
            "vina_affinity": vina_affinity_tensor.to(torch.float),
            "stability": stability_tensor.to(torch.float),
            "solubility": solubility_tensor.to(torch.float),
            "structure_labels": distogram,                   # [L_pep, L_pep] (bins)
            "interface_labels": interface_labels,            # === NEW: [L_pep, L_rec] (0/1/-100)
        }

    # === NEW: 取并对齐 interface 矩阵
    def _fetch_and_align_interface(self, sample_id: str, L_pep: int, L_rec: int) -> torch.Tensor:
        """
        返回 [L_pep, L_rec] 的 long tensor：
          - 若 JSON 有该样本，则裁剪/对齐并填 -100；
          - 若没有该样本或没提供 JSON，返回全 -100（训练时被忽略）。
        """
        # 初始化全 -100（忽略）
        tgt = torch.full((L_pep, L_rec), -100, dtype=torch.long)

        entry = self.interface_db.get(sample_id)
        if not entry:
            return tgt  # 没有该样本，忽略

        labels = entry.get("labels", None)
        if labels is None:
            return tgt

        lab_np = np.asarray(labels, dtype=np.int64)  # JSON: Lp_json x Lr_json
        if lab_np.ndim != 2:
            return tgt

        Lp_json, Lr_json = lab_np.shape
        h = min(L_pep, Lp_json)
        w = min(L_rec, Lr_json)

        if h > 0 and w > 0:
            tgt[:h, :w] = torch.from_numpy(lab_np[:h, :w])

        return tgt


# ---- helpers ----

def extract_sequence_from_pdb(pdb_path):
    try:
        protein_chain = ProteinChain.from_pdb(pdb_path)
        if protein_chain is None or not protein_chain.sequence:
            return None
        return protein_chain.sequence
    except Exception as e:
        print(f"Error reading {pdb_path}: {e}")
        return None

def _load_structure(pdb_or_cif_path):
    ext = os.path.splitext(pdb_or_cif_path)[1].lower()
    if ext in [".cif", ".mmcif"]:
        parser = MMCIFParser(QUIET=True)
    else:
        parser = PDBParser(QUIET=True)
    return parser.get_structure("pep", pdb_or_cif_path)

def _pick_chain(model, target_chain_id=None):
    chains = list(model.get_chains())
    if not chains:
        raise ValueError("No chains found in model.")
    if target_chain_id is None:
        return chains[0]
    for ch in chains:
        if ch.id == target_chain_id:
            return ch
    tc = target_chain_id.strip()
    for ch in chains:
        if ch.id.strip() == tc:
            return ch
    if len(tc) == 1:
        for ch in chains:
            if ch.id == tc + " ":
                return ch
    if len(tc) == 2:
        for ch in chains:
            if ch.id == tc[0]:
                return ch
    return chains[0]

def extract_distogram_labels(pdb_file, chain_id=None):
    structure = _load_structure(pdb_file)
    model = next(structure.get_models())
    chain = _pick_chain(model, chain_id)

    coords = []
    for res in chain:
        if getattr(res, "id", None) and res.id[0] != " ":
            continue
        if "CA" in res:
            coords.append(res["CA"].get_vector().get_array())

    coords = np.asarray(coords, dtype=np.float32)
    if coords.size == 0:
        # 返回 0x0，外层会处理（不会参与 loss）
        return torch.zeros((0, 0), dtype=torch.long)

    dmat = np.linalg.norm(coords[:, None] - coords[None, :], axis=-1)
    dist_labels = np.digitize(dmat, BIN_EDGES) - 1
    dist_labels = np.clip(dist_labels, 0, NUM_BINS - 1)
    return torch.tensor(np.ascontiguousarray(dist_labels), dtype=torch.long).contiguous()

# === NEW: 仅计数受体 CA 长度
def count_ca_len(pdb_file, chain_id=None) -> int:
    structure = _load_structure(pdb_file)
    model = next(structure.get_models())
    chain = _pick_chain(model, chain_id)
    n = 0
    for res in chain:
        if getattr(res, "id", None) and res.id[0] != " ":
            continue
        if "CA" in res:
            n += 1
    return n
