#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
DPO 训练（结构头）：读取 dpo_pairs.jsonl (prompt + chosen/rejected PDB)，
把候选 PDB 转为结构标签（距离分箱），用当前模型结构 logits 的 logprob 做 DPO。

假设：你的模型前向
    sequence_logits, struct_logits = model(batch)
其中 struct_logits 形状为 [B, Lq, Lk, NUM_BINS] 或 [B, Lq, Lk*NUM_BINS]
标签为 [B, L, L]（-100 为 ignore）。
"""

# import os, json, math, argparse, copy, warnings
# from pathlib import Path
# from typing import Dict, List, Tuple, Optional
# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# from torch.utils.data import Dataset, DataLoader
# from tqdm import tqdm
# import numpy as np
# from Bio import PDB
# import sys
# sys.path.append("/root/autodl-tmp/Peptide_3D")
# from ranger import Ranger
# from model.esm.tokenization.sequence_tokenizer import EsmSequenceTokenizer
# from model.esm.utils.encoding import tokenize_sequence
# # ====== 复用你现有工程中的定义（务必保持一致） ======
# from models_DPO import ProteinPeptideModel
# from modules.paddingCollate import PaddingCollate
# # —— 3-letter 到 1-letter 的安全转换（兼容非标准残基）——
# from Bio.SeqUtils import seq1 as _seq1
# # 顶部 import 后：
# torch.backends.cuda.matmul.allow_tf32 = True
# torch.backends.cudnn.allow_tf32 = True

# amp_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16

# # 常见别名/修饰残基的自定义映射
# _AA3_CUSTOM = {
#     "MSE": "M",  # Selenomethionine
#     "SEC": "U",  # Selenocysteine
#     "PYL": "O",  # Pyrrolysine
#     "HID": "H", "HIE": "H", "HIP": "H",  # Histidine tautomers
#     "CYX": "C",  # Disulfide-bonded Cys
#     "ASX": "B",  # Asn/Asp ambiguous
#     "GLX": "Z",  # Gln/Glu ambiguous
#     "UNK": "X",
# }

# def resname_to_one(resname: str) -> str:
#     # Bio.SeqUtils.seq1 支持 3-letter -> 1-letter，配合自定义表更稳
#     try:
#         return _seq1(resname.strip(), custom_map=_AA3_CUSTOM, undef_code="X")
#     except Exception:
#         return "X"

# # —— 和你训练脚本保持一致的全局常量（若已在别处统一，请import它们） —— #
# BIN_EDGES = np.linspace(2.0, 22.0, num=65)   # 64个bins
# NUM_BINS  = len(BIN_EDGES) - 1

# # 简单AA词表；如果你有专用tokenizer，请替换 encode_seq()
# AA_VOCAB = {aa:i+1 for i,aa in enumerate("ACDEFGHIKLMNPQRSTVWY")}  # 1..20
# # 创建序列tokenizer
# sequence_tokenizer = EsmSequenceTokenizer()
# PAD_ID = 0
# UNK_ID = 21

# def encode_seq(seq: str) -> torch.LongTensor:
#     idx = [AA_VOCAB.get(a.upper(), UNK_ID) for a in seq]
#     if len(idx) == 0:
#         idx = [PAD_ID]
#     return torch.tensor(idx, dtype=torch.long)

# def load_peptide_seq_from_pdb(peptide_pdb: str) -> str:
#     """如果 jsonl 里没有 peptide_seq 或想双重校验，也可从候选PDB提取一次序列。"""
#     parser = PDB.PDBParser(QUIET=True)
#     structure = parser.get_structure("pep", peptide_pdb)
#     residues = [res for res in structure.get_residues()
#                 if PDB.is_aa(res, standard=False)]
#     seq = "".join(resname_to_one(res.get_resname()) for res in residues)
#     return seq

# def distance_bins_from_pdb(peptide_pdb: str, max_len: Optional[int]=None) -> torch.LongTensor:
#     """
#     从候选 PDB 计算 pairwise CA-CA 距离，并据 BIN_EDGES 量化为 bin_id。
#     返回 [L,L] 的 LongTensor；若 max_len 提供，则裁切为 [:max_len,:max_len]
#     """
#     parser = PDB.PDBParser(QUIET=True)
#     structure = parser.get_structure("pep", peptide_pdb)
#     # 收集 CA 坐标（按残基顺序）
#     cas = []
#     for res in structure.get_residues():
#         if not PDB.is_aa(res, standard=True):
#             continue
#         if "CA" in res:
#             cas.append(res["CA"].get_vector().get_array())
#     if len(cas) == 0:
#         raise RuntimeError(f"No CA atoms in {peptide_pdb}")
#     X = np.stack(cas, axis=0)  # [L,3]
#     if max_len is not None:
#         X = X[:max_len]
#     L = X.shape[0]
#     D = np.sqrt(np.sum((X[:,None,:]-X[None,:,:])**2, axis=-1))  # [L,L]
#     # np.digitize: 返回 bin 索引（1..len(BIN_EDGES)-1），我们减1 -> 0..NUM_BINS-1
#     bins = np.digitize(D, BIN_EDGES) - 1
#     bins = np.clip(bins, 0, NUM_BINS-1).astype(np.int64)
#     # 自对角线可设为 ignore（-100），也可以保留（随你喜好）
#     np.fill_diagonal(bins, -100)
#     return torch.from_numpy(bins)  # [L,L], int64

# def struct_logits_to_logprob(struct_logits: torch.Tensor, labels_2d: torch.Tensor) -> torch.Tensor:
#     """
#     struct_logits: [B,Lq,Lk,NUM_BINS] or [B, Lq, Lk*NUM_BINS]
#     labels_2d:     [B,L,L] （-100 为 ignore）
#     输出：每个样本的总 logprob（标量张量） shape [B]
#     """
#     if struct_logits.dim() == 3:
#         B, Lq, last = struct_logits.shape
#         assert last % NUM_BINS == 0
#         Lk = last // NUM_BINS
#         struct_logits = struct_logits.view(B, Lq, Lk, NUM_BINS)
#     elif struct_logits.dim() == 4:
#         B, Lq, Lk, C = struct_logits.shape
#         assert C == NUM_BINS
#     else:
#         raise RuntimeError(f"Bad shape: {struct_logits.shape}")

#     Ltgt = labels_2d.size(-1)
#     L = min(Lq, Lk, Ltgt)
#     logits = struct_logits[:, :L, :L, :]          # [B,L,L,C]
#     labels = labels_2d[:, :L, :L]                 # [B,L,L]
#     valid = labels != -100

#     # log-softmax over bins
#     logp = F.log_softmax(logits, dim=-1)          # [B,L,L,C]
#     # gather logp at observed bin
#     # 为了 gather，需要 labels 扩一维
#     gather_idx = labels.clamp(min=0).unsqueeze(-1)  # [B,L,L,1], 将 -100 先夹到0但用 valid 掩蔽
#     picked = logp.gather(-1, gather_idx).squeeze(-1)  # [B,L,L]
#     picked = picked * valid  # 抹掉 ignore
#     # sum over pairs；避免出现全无效的NaN，用 0 代替
#     sum_per_sample = picked.view(B, -1).sum(dim=-1)  # [B]
#     return sum_per_sample

# # =============== DPO 数据集 ===============

# class DPOPairsDataset(Dataset):
#     """
#     读取 jsonl：每行包含：
#         {
#           "prompt": {"receptor_pdb": "...", "peptide_seq": "..."},
#           "chosen":  {"pdb_path": "...", "score": {...}},
#           "rejected":{"pdb_path": "...", "score": {...}},
#           "pair_weight": float
#         }
#     若某行有问题（路径不存在/解析失败），会在 __getitem__ 抛出，此行会被 DataLoader 捕获后跳过。
#     """
#     def __init__(self, jsonl_path: str, max_len: Optional[int]=None, max_receptor_len: Optional[int]=None):
#         self.rows = []
#         with open(jsonl_path, "r", encoding="utf-8") as f:
#             for ln in f:
#                 ln = ln.strip()
#                 if not ln:
#                     continue
#                 r = json.loads(ln)
#                 # 过滤无效行
#                 try:
#                     rp = r["prompt"]["receptor_pdb"]
#                     pep = r["prompt"]["peptide_seq"]
#                     cp  = r["chosen"]["pdb_path"]
#                     rpj = r["rejected"]["pdb_path"]
#                     if not (os.path.exists(rp) and os.path.exists(cp) and os.path.exists(rpj)):
#                         continue
#                     r["_pair_weight"] = float(r.get("pair_weight", 1.0))
#                     self.rows.append(r)
#                 except Exception:
#                     continue
#         self.max_len = max_len
#         self.max_receptor_len = max_receptor_len

#     def __len__(self):
#         return len(self.rows)

#     def __getitem__(self, idx: int) -> Dict:
#         r = self.rows[idx]
#         receptor_pdb = r["prompt"]["receptor_pdb"]
#         peptide_seq  = r["prompt"]["peptide_seq"]
#         chosen_pdb   = r["chosen"]["pdb_path"]
#         rejected_pdb = r["rejected"]["pdb_path"]
#         w = float(r.get("_pair_weight", 1.0))

#         # —— 输入张量：简单用 AA_VOCAB 编码；如果你有专用 tokenizer，请替换 encode_seq() —— #
#         rec_seq = self._seq_from_receptor(receptor_pdb)
#         # rec_ids = encode_seq(rec_seq)
#         rec_ids = tokenize_sequence(rec_seq, sequence_tokenizer)
#         if self.max_receptor_len is not None:
#             rec_ids = rec_ids[:self.max_receptor_len]   # ★ 截断 receptor 序列
#         # pep_ids = encode_seq(peptide_seq)
#         pep_ids = tokenize_sequence(peptide_seq, sequence_tokenizer)

#         # —— 结构标签：从候选 PDB 的 CA-距离 矩阵量化 —— #
#         chosen_labels   = distance_bins_from_pdb(chosen_pdb,   max_len=self.max_len)  # [Lc,Lc]
#         rejected_labels = distance_bins_from_pdb(rejected_pdb, max_len=self.max_len)  # [Lr,Lr]

#         return {
#             "receptor_seq_tensor": rec_ids,          # [Lr]
#             "peptide_seq_tensor":  pep_ids,          # [Lp]
#             "chosen_labels":   chosen_labels,        # [L,L]
#             "rejected_labels": rejected_labels,      # [L,L]
#             "pair_weight":     torch.tensor(w, dtype=torch.float32),
#             "meta": {
#                 "receptor_pdb": receptor_pdb,
#                 "chosen_pdb":   chosen_pdb,
#                 "rejected_pdb": rejected_pdb
#             }
#         }

#     @staticmethod
#     def _seq_from_receptor(receptor_pdb: str) -> str:
#         parser = PDB.PDBParser(QUIET=True)
#         structure = parser.get_structure("rec", receptor_pdb)
#         residues = [res for res in structure.get_residues()
#             if PDB.is_aa(res, standard=False)]
#         seq = "".join(resname_to_one(res.get_resname()) for res in residues)
#         if len(seq) == 0:
#             # 兜底：随便给一个A，避免空序列；实际应当过滤这条数据
#             seq = "A"
#         return seq

# # =============== DPO 训练 ===============

# @torch.no_grad()
# def ref_logprob(model_ref: nn.Module, batch: Dict, device: torch.device):
#     model_ref.eval()
#     B = batch["peptide_seq_tensor"].size(0)

#     # 全部转 CPU
#     base_batch = {
#         "receptor_seq_tensor": batch["receptor_seq_tensor"].to(device),
#         "peptide_seq_tensor":  batch["peptide_seq_tensor"].to(device),
#         "attr_cross_attention_mask": None,
#         "cross_attention_mask": None,
#     }
#     # —— 补齐三项连续特征，形状 [B]（你的 forward 里会再 unsqueeze(-1)）——
#     for k in ("stability", "solubility", "vina_affinity"):
#         if k in batch:
#             base_batch[k] = batch[k].to(device)
#         else:
#             base_batch[k] = torch.zeros(B, dtype=torch.float32, device=device)
#     # 不给属性-蛋白的 cross-attn 掩码（让模型内部按长度推断）
#     base_batch["attr_cross_attention_mask"] = None
#     base_batch["cross_attention_mask"] = None
#     # CPU 前向；不需要 autocast
#     with torch.inference_mode():
#         _, struct_logits = model_ref(base_batch)          # [B, Lq, Lk, C] or [B, Lq, Lk*C]

#     # chosen
#     chosen_labels = batch["chosen_labels"].to(device)   # [B, L, L]
#     lp_c = struct_logits_to_logprob(struct_logits, chosen_labels)  # [B]

#     # rejected
#     rejected_labels = batch["rejected_labels"].to(device)
#     lp_r = struct_logits_to_logprob(struct_logits, rejected_labels)

#     return lp_c.to(device), lp_r.to(device)  # 仅返回到 GPU 参与 loss

# def dpo_step(model: nn.Module, model_ref: nn.Module, batch: Dict, beta: float, device: torch.device):
#     B = batch["peptide_seq_tensor"].size(0)
#     base_batch = {
#         "receptor_seq_tensor": batch["receptor_seq_tensor"].to(device),
#         "peptide_seq_tensor":  batch["peptide_seq_tensor"].to(device),
#     }
#     for k in ("stability", "solubility", "vina_affinity"):
#         if k in batch:
#             base_batch[k] = batch[k].to(device)
#         else:
#             base_batch[k] = torch.zeros(B, dtype=torch.float32, device=device)
#     base_batch["attr_cross_attention_mask"] = None
#     base_batch["cross_attention_mask"] = None
#     with torch.cuda.amp.autocast(dtype=amp_dtype):
#         sequence_logits, struct_logits = model(base_batch)

#     # 当前模型 logprob
#     lp_c = struct_logits_to_logprob(struct_logits, batch["chosen_labels"].to(device))   # [B]
#     lp_r = struct_logits_to_logprob(struct_logits, batch["rejected_labels"].to(device)) # [B]
#     del struct_logits
#     torch.cuda.empty_cache()
#     delta_theta = lp_c - lp_r                                                           # [B]

#     # 参考模型 logprob（无梯度）
#     with torch.no_grad():
#         ref_c, ref_r = ref_logprob(model_ref, batch, device)
#         delta_ref = ref_c - ref_r

#     # DPO 目标
#     inside = beta * (delta_theta - delta_ref)
#     loss = -F.logsigmoid(inside)                           # [B]
#     # pair 权重（如果有）
#     if "pair_weight" in batch:
#         w = batch["pair_weight"].to(device).clamp(min=1e-6)
#         loss = loss * w

#     # 指标
#     acc = (delta_theta > delta_ref).float().mean()         # 简单胜率指标
#     return loss.mean(), {
#         "loss_mean": loss.mean().item(),
#         "delta_theta_mean": delta_theta.mean().item(),
#         "delta_ref_mean":   delta_ref.mean().item(),
#         "pair_acc": acc.item(),
#         "lp_c": lp_c.mean().item(),
#         "lp_r": lp_r.mean().item(),
#     }

# def collate_pairs(batch_list: List[Dict]) -> Dict:
#     """
#     简易 collate：pad 两条序列 & 把矩阵打包成 batch。
#     也可以用你自带的 PaddingCollate；这里只做最小实现。
#     """
#     # 序列 pad 到同长
#     def pad_1d(tensors, pad_id=PAD_ID):
#         L = max(t.size(0) for t in tensors)
#         out = torch.full((len(tensors), L), pad_id, dtype=torch.long)
#         for i,t in enumerate(tensors):
#             out[i,:t.size(0)] = t
#         return out

#     # 结构矩阵 pad/crop 到同 L
#     def pad_2d_mats(mats):
#         L = max(m.size(0) for m in mats)
#         outs = []
#         for m in mats:
#             mm = torch.full((L,L), -100, dtype=torch.long)
#             l = m.size(0)
#             mm[:l,:l] = m
#             outs.append(mm)
#         return torch.stack(outs, dim=0)  # [B,L,L]

#     rec = pad_1d([b["receptor_seq_tensor"] for b in batch_list], PAD_ID)
#     pep = pad_1d([b["peptide_seq_tensor"]  for b in batch_list], PAD_ID)
#     ch  = pad_2d_mats([b["chosen_labels"]   for b in batch_list])
#     rj  = pad_2d_mats([b["rejected_labels"] for b in batch_list])
#     w   = torch.stack([b["pair_weight"] for b in batch_list], dim=0)

#     return {
#         "receptor_seq_tensor": rec,   # [B,Lr]
#         "peptide_seq_tensor":  pep,   # [B,Lp]
#         "chosen_labels":   ch,        # [B,L,L]
#         "rejected_labels": rj,        # [B,L,L]
#         "pair_weight":     w,
#         "meta": [b["meta"] for b in batch_list]
#     }

# def main():
#     ap = argparse.ArgumentParser()
#     ap.add_argument("--jsonl", required=False,
#                     default="/root/autodl-tmp/Peptide_3D/utils/dpo/dpo_pairs_cleaned.jsonl")
#     ap.add_argument("--init_ckpt", type=str, default='/root/autodl-tmp/Peptide_3D/logs_Ranger_no_DPO/best_model_epoch_72_loss_2.0048.pth',
#                     help="从监督学习阶段的最佳 checkpoint 加载（policy & reference 初始相同）")
#     ap.add_argument("--save_dir", type=str, default="/root/autodl-tmp/Peptide_3D/logs_Ranger_dpo")
#     ap.add_argument("--epochs", type=int, default=3)
#     ap.add_argument("--batch_size", type=int, default=1)
#     ap.add_argument("--beta", type=float, default=0.1)
#     ap.add_argument("--lr", type=float, default=5e-6)
#     ap.add_argument("--num_workers", type=int, default=1)
#     ap.add_argument("--max_len", type=int, default=None, help="可选：对PDB序列截断长度（加速）")
#     ap.add_argument("--max_receptor_len", type=int, default=512,
#                     help="限制 receptor 序列最大长度（强烈建议 256/384/512）")
#     args = ap.parse_args()

#     os.makedirs(args.save_dir, exist_ok=True)
#     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#     # 数据集（会自动跳过缺失/解析失败的pair）
#     ds = DPOPairsDataset(args.jsonl, max_len=args.max_len, max_receptor_len=args.max_receptor_len)
#     if len(ds) == 0:
#         raise RuntimeError(f"No valid pairs in {args.jsonl}")
#     dl = DataLoader(ds, batch_size=args.batch_size, shuffle=True,
#                     num_workers=args.num_workers, collate_fn=collate_pairs, pin_memory=False)

#     # 模型 & 参考模型
#     policy = ProteinPeptideModel(device).to(device)
#     if args.init_ckpt and os.path.exists(args.init_ckpt):
#         policy.load_state_dict(torch.load(args.init_ckpt, map_location=device))
#     ref_model = copy.deepcopy(policy).to(device).eval()
#     for p in ref_model.parameters():
#         p.requires_grad_(False)
#     ref_model.eval()

#     # 优化器
#     # opt = torch.optim.AdamW(policy.parameters(), lr=args.lr, betas=(0.9,0.95))
#     opt = Ranger(policy.parameters(), lr=args.lr, betas=(0.9,0.95))

#     best = float("inf")
#     for epoch in range(1, args.epochs+1):
#         policy.train()
#         epoch_loss = 0.0
#         pbar = tqdm(dl, desc=f"DPO epoch {epoch}")
#         for batch in pbar:
#             for k in ("pair_weight","chosen_labels","rejected_labels",
#                       "receptor_seq_tensor","peptide_seq_tensor"):
#                 batch[k] = batch[k].to(device, non_blocking=True)

#             loss, logs = dpo_step(policy, ref_model, batch, beta=args.beta, device=device)
#             opt.zero_grad(set_to_none=True)
#             loss.backward()
#             torch.nn.utils.clip_grad_norm_(policy.parameters(), max_norm=1.0)
#             opt.step()

#             epoch_loss += loss.item()
#             pbar.set_postfix({k:f"{v:.3f}" for k,v in logs.items()})

#         avg = epoch_loss / max(1,len(dl))
#         ckpt = os.path.join(args.save_dir, f"policy_dpo_epoch{epoch}_loss_{avg:.4f}.pth")
#         torch.save(policy.state_dict(), ckpt)
#         if avg < best:
#             best = avg
#             torch.save(policy.state_dict(), os.path.join(args.save_dir, "policy_dpo_best.pth"))

#     print(f"[FIN] DPO done. Best={best:.4f}, saved under {args.save_dir}")

# if __name__ == "__main__":
#     main()


#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
DPO+ for protein–peptide structure head (enhanced):
- Soft/Hard/RPO objective
- Length-invariant logprob aggregation
- Confidence weighting & optional margin by Δreward
- Structure KL regularizer (random subsample), GPU-friendly
"""

import os, json, argparse, copy, warnings, math, random
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import numpy as np
from Bio import PDB
from Bio.SeqUtils import seq1 as _seq1

# === project deps ===
import sys
sys.path.append("/root/autodl-tmp/Peptide_3D")
from model.esm.tokenization.sequence_tokenizer import EsmSequenceTokenizer
from model.esm.utils.encoding import tokenize_sequence
from models_DPO import ProteinPeptideModel
try:
    from ranger import Ranger
    _HAS_RANGER = True
except Exception:
    _HAS_RANGER = False

# ===== perf knobs =====
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
try:
    amp_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
except Exception:
    amp_dtype = torch.float16

# ===== chemistry utils =====
_AA3_CUSTOM = {
    "MSE": "M",
    "SEC": "U",
    "PYL": "O",
    "HID": "H", "HIE": "H", "HIP": "H",
    "CYX": "C",
    "ASX": "B",
    "GLX": "Z",
    "UNK": "X",
}
def resname_to_one(resname: str) -> str:
    try:
        return _seq1(resname.strip(), custom_map=_AA3_CUSTOM, undef_code="X")
    except Exception:
        return "X"

# ===== struct labels (CA-CA distances -> bins) =====
BIN_EDGES = np.linspace(2.0, 22.0, num=65)  # 64 bins
NUM_BINS  = len(BIN_EDGES) - 1

sequence_tokenizer = EsmSequenceTokenizer()
PAD_ID = 0

def distance_bins_from_pdb(peptide_pdb: str, max_len: Optional[int]=None) -> torch.LongTensor:
    parser = PDB.PDBParser(QUIET=True)
    structure = parser.get_structure("pep", peptide_pdb)
    cas = []
    for res in structure.get_residues():
        if not PDB.is_aa(res, standard=True):
            continue
        if "CA" in res:
            cas.append(res["CA"].get_vector().get_array())
    if len(cas) == 0:
        raise RuntimeError(f"No CA atoms in {peptide_pdb}")
    X = np.stack(cas, axis=0)  # [L,3]
    if max_len is not None:
        X = X[:max_len]
    L = X.shape[0]
    D = np.sqrt(np.sum((X[:,None,:]-X[None,:,:])**2, axis=-1))  # [L,L]
    bins = np.digitize(D, BIN_EDGES) - 1
    bins = np.clip(bins, 0, NUM_BINS-1).astype(np.int64)
    np.fill_diagonal(bins, -100)  # ignore self-pairs
    return torch.from_numpy(bins)

# ===== struct logits -> sample logprob (length normalized) =====
def struct_logits_to_logprob(struct_logits: torch.Tensor,
                             labels_2d: torch.Tensor,
                             reduce_mode: str = "mean") -> torch.Tensor:
    """
    struct_logits: [B,Lq,Lk,NUM_BINS] or [B,Lq,Lk*NUM_BINS]
    labels_2d:     [B,L,L] with -100 ignored
    returns: [B]
    reduce_mode: {"sum","mean","sqrt"}
    """
    if struct_logits.dim() == 3:
        B, Lq, last = struct_logits.shape
        assert last % NUM_BINS == 0
        Lk = last // NUM_BINS
        struct_logits = struct_logits.view(B, Lq, Lk, NUM_BINS)
    elif struct_logits.dim() == 4:
        B, Lq, Lk, C = struct_logits.shape
        assert C == NUM_BINS
    else:
        raise RuntimeError(f"Bad shape: {struct_logits.shape}")

    Ltgt = labels_2d.size(-1)
    L = min(Lq, Lk, Ltgt)
    logits = struct_logits[:, :L, :L, :]         # [B,L,L,C]
    labels = labels_2d[:, :L, :L]                # [B,L,L]
    valid = labels != -100

    logp = F.log_softmax(logits, dim=-1)         # [B,L,L,C]
    gather_idx = labels.clamp(min=0).unsqueeze(-1)
    picked = logp.gather(-1, gather_idx).squeeze(-1)  # [B,L,L]
    picked = picked * valid

    M = valid.view(B, -1).sum(dim=-1).clamp(min=1)
    s = picked.view(B, -1).sum(dim=-1)           # [B]
    if reduce_mode == "mean":
        return s / M
    elif reduce_mode == "sqrt":
        return s / M.sqrt()
    else:
        return s

# ===== dataset =====
class DPOPairsDataset(Dataset):
    """
    jsonl lines:
      {
        "prompt":  {"receptor_pdb": "...", "peptide_seq": "..."},
        "chosen":  {"pdb_path":"...", "score":{"hdock":..., "R":...}},
        "rejected":{"pdb_path":"...", "score":{"hdock":..., "R":...}},
        "pair_weight": float
      }
    """
    def __init__(self, jsonl_path: str, max_len: Optional[int]=None, max_receptor_len: Optional[int]=None):
        self.rows = []
        with open(jsonl_path, "r", encoding="utf-8") as f:
            for ln in f:
                ln = ln.strip()
                if not ln:
                    continue
                r = json.loads(ln)
                try:
                    rp  = r["prompt"]["receptor_pdb"]
                    pep = r["prompt"]["peptide_seq"]
                    cp  = r["chosen"]["pdb_path"]
                    rpj = r["rejected"]["pdb_path"]
                    if not (os.path.exists(rp) and os.path.exists(cp) and os.path.exists(rpj)):
                        continue
                    # weight
                    r["_pair_weight"] = float(r.get("pair_weight", 1.0))
                    # Δreward: prefer "R" (z-score of affinity) ; fallback to hdock
                    cR = r.get("chosen",{}).get("score",{}).get("R", None)
                    rR = r.get("rejected",{}).get("score",{}).get("R", None)
                    if cR is not None and rR is not None:
                        r["_delta_reward"] = float(cR - rR)
                    else:
                        ch = r.get("chosen",{}).get("score",{}).get("hdock", None)
                        rh = r.get("rejected",{}).get("score",{}).get("hdock", None)
                        if ch is not None and rh is not None:
                            # affinity = -hdock; Δreward = aff_c - aff_r = ( -ch ) - ( -rh ) = rh - ch
                            r["_delta_reward"] = float(rh - ch)
                        else:
                            r["_delta_reward"] = 0.0
                    self.rows.append(r)
                except Exception:
                    continue
        self.max_len = max_len
        self.max_receptor_len = max_receptor_len

    def __len__(self): return len(self.rows)

    def __getitem__(self, idx: int) -> Dict:
        r = self.rows[idx]
        receptor_pdb = r["prompt"]["receptor_pdb"]
        peptide_seq  = r["prompt"]["peptide_seq"]
        chosen_pdb   = r["chosen"]["pdb_path"]
        rejected_pdb = r["rejected"]["pdb_path"]
        w = float(r.get("_pair_weight", 1.0))
        delta_reward = float(r.get("_delta_reward", 0.0))

        rec_seq = self._seq_from_receptor(receptor_pdb)
        rec_ids = tokenize_sequence(rec_seq, sequence_tokenizer)
        if self.max_receptor_len is not None:
            rec_ids = rec_ids[:self.max_receptor_len]
        pep_ids = tokenize_sequence(peptide_seq, sequence_tokenizer)

        chosen_labels   = distance_bins_from_pdb(chosen_pdb,   max_len=self.max_len)
        rejected_labels = distance_bins_from_pdb(rejected_pdb, max_len=self.max_len)

        return {
            "receptor_seq_tensor": rec_ids,   # [Lr]
            "peptide_seq_tensor":  pep_ids,   # [Lp]
            "chosen_labels":   chosen_labels, # [L,L]
            "rejected_labels": rejected_labels,
            "pair_weight":     torch.tensor(w, dtype=torch.float32),
            "delta_reward":    torch.tensor(delta_reward, dtype=torch.float32),
            "meta": {
                "receptor_pdb": receptor_pdb,
                "chosen_pdb":   chosen_pdb,
                "rejected_pdb": rejected_pdb
            }
        }

    @staticmethod
    def _seq_from_receptor(receptor_pdb: str) -> str:
        parser = PDB.PDBParser(QUIET=True)
        structure = parser.get_structure("rec", receptor_pdb)
        residues = [res for res in structure.get_residues() if PDB.is_aa(res, standard=False)]
        seq = "".join(resname_to_one(res.get_resname()) for res in residues)
        if len(seq) == 0:
            seq = "A"
        return seq

def collate_pairs(batch_list: List[Dict]) -> Dict:
    def pad_1d(tensors, pad_id=PAD_ID):
        L = max(t.size(0) for t in tensors)
        out = torch.full((len(tensors), L), pad_id, dtype=torch.long)
        for i,t in enumerate(tensors):
            out[i,:t.size(0)] = t
        return out
    def pad_2d_mats(mats):
        L = max(m.size(0) for m in mats)
        outs = []
        for m in mats:
            mm = torch.full((L,L), -100, dtype=torch.long)
            l = m.size(0)
            mm[:l,:l] = m
            outs.append(mm)
        return torch.stack(outs, dim=0)

    rec = pad_1d([b["receptor_seq_tensor"] for b in batch_list], PAD_ID)
    pep = pad_1d([b["peptide_seq_tensor"]  for b in batch_list], PAD_ID)
    ch  = pad_2d_mats([b["chosen_labels"]   for b in batch_list])
    rj  = pad_2d_mats([b["rejected_labels"] for b in batch_list])
    w   = torch.stack([b["pair_weight"] for b in batch_list], dim=0)
    dr  = torch.stack([b["delta_reward"] for b in batch_list], dim=0)

    return {
        "receptor_seq_tensor": rec,
        "peptide_seq_tensor":  pep,
        "chosen_labels":   ch,
        "rejected_labels": rj,
        "pair_weight":     w,
        "delta_reward":    dr,
        "meta": [b["meta"] for b in batch_list],
    }

# ===== DPO objectives =====
def dpo_objective(delta_theta, delta_ref, delta_reward,
                  beta=0.1, mode="soft",
                  soft_temp=2.0,
                  margin_m=0.0,
                  weight_gamma=2.0):
    """
    delta_theta: [B], policy (lp_c - lp_r)
    delta_ref:   [B], ref    (lp_c - lp_r)
    delta_reward:[B], preference strength (R-chosen - R-rejected) or hdock-derived
    """
    z = beta * (delta_theta - delta_ref)  # [B]
    # confidence weight by Δreward
    w_conf = torch.sigmoid(weight_gamma * (delta_reward - margin_m))  # [B]

    if mode == "hard":
        loss = -F.logsigmoid(z) * w_conf

    elif mode == "soft":
        # soft label in (0,1) from Δreward
        y = torch.sigmoid(delta_reward / soft_temp)  # [B]
        loss = F.binary_cross_entropy_with_logits(z, y, reduction="none") * w_conf

    elif mode == "rpo":
        # regression to reward: (Δθ-Δref) ≈ tanh(Δreward / T)
        target = torch.tanh(delta_reward / soft_temp)
        loss = F.smooth_l1_loss((z / beta), target, reduction="none") * w_conf
    else:
        raise ValueError(f"unknown mode={mode}")

    return loss.mean(), {"w_conf": w_conf.mean().item()}

# ===== KL reg (policy || ref) on structure bins, random subsample =====
def kl_regularizer(policy_logits, ref_logits, max_pairs=2048):
    """
    logits: [B,L,L,C] or [B,L,L*C] (will be reshaped above)
    returns scalar kl
    """
    if policy_logits.dim() == 3:
        B, Lq, last = policy_logits.shape
        Lk = last // NUM_BINS
        policy_logits = policy_logits.view(B, Lq, Lk, NUM_BINS)
        ref_logits    = ref_logits.view(B, Lq, Lk, NUM_BINS)

    B, L, K, C = policy_logits.shape
    N = min(max_pairs, B*L*K)
    idx = torch.randint(0, B*L*K, (N,), device=policy_logits.device)
    p = policy_logits.reshape(B*L*K, C)[idx]
    q = ref_logits.reshape(B*L*K, C)[idx]
    p_log = F.log_softmax(p, dim=-1)
    q_log = F.log_softmax(q, dim=-1)
    p_prob = p_log.exp()
    kl = (p_prob * (p_log - q_log)).sum(dim=-1).mean()
    return kl

# ===== one DPO step =====
def dpo_step(model: nn.Module, model_ref: nn.Module, batch: Dict, device: torch.device,
             beta: float, reduce_mode="mean",
             dpo_mode="soft", soft_temp=2.0,
             margin_m=0.0, weight_gamma=2.0,
             kl_coef=0.01, kl_pairs=2048, do_kl=True):
    B = batch["peptide_seq_tensor"].size(0)
    base_batch = {
        "receptor_seq_tensor": batch["receptor_seq_tensor"].to(device),
        "peptide_seq_tensor":  batch["peptide_seq_tensor"].to(device),
        "attr_cross_attention_mask": None,
        "cross_attention_mask": None,
    }
    for k in ("stability", "solubility", "vina_affinity"):
        base_batch[k] = batch.get(k, torch.zeros(B, dtype=torch.float32, device=device))

    with torch.cuda.amp.autocast(dtype=amp_dtype):
        _, struct_logits_p = model(base_batch)

    lp_c = struct_logits_to_logprob(struct_logits_p, batch["chosen_labels"].to(device), reduce_mode)
    lp_r = struct_logits_to_logprob(struct_logits_p, batch["rejected_labels"].to(device), reduce_mode)
    delta_theta = lp_c - lp_r  # [B]

    with torch.no_grad():
        _, struct_logits_q = model_ref(base_batch)
        ref_c = struct_logits_to_logprob(struct_logits_q, batch["chosen_labels"].to(device), reduce_mode)
        ref_r = struct_logits_to_logprob(struct_logits_q, batch["rejected_labels"].to(device), reduce_mode)
        delta_ref = ref_c - ref_r

    delta_reward = batch.get("delta_reward", torch.zeros(B, device=device))
    dpo_loss, info = dpo_objective(delta_theta, delta_ref, delta_reward,
                                   beta=beta, mode=dpo_mode, soft_temp=soft_temp,
                                   margin_m=margin_m, weight_gamma=weight_gamma)

    if do_kl and kl_coef > 0:
        # ensure 4D
        if struct_logits_p.dim() == 3:
            Bp, Lq, last = struct_logits_p.shape
            Lk = last // NUM_BINS
            struct_logits_p = struct_logits_p.view(Bp, Lq, Lk, NUM_BINS)
            struct_logits_q = struct_logits_q.view(Bp, Lq, Lk, NUM_BINS)
        Lt = min(struct_logits_p.size(1), struct_logits_p.size(2),
                 struct_logits_q.size(1), struct_logits_q.size(2))
        kl = kl_regularizer(struct_logits_p[:, :Lt, :Lt, :],
                            struct_logits_q[:, :Lt, :Lt, :],
                            max_pairs=kl_pairs)
        loss = dpo_loss + kl_coef * kl
    else:
        kl = torch.tensor(0.0, device=device)
        loss = dpo_loss

    logs = {
        "loss": loss.item(),
        "dpo": dpo_loss.item(),
        "kl": kl.item(),
        "delta_theta": delta_theta.mean().item(),
        "delta_ref": delta_ref.mean().item(),
        "w_conf": info["w_conf"],
        "lp_c": lp_c.mean().item(),
        "lp_r": lp_r.mean().item(),
    }
    return loss, logs

# ===== utils =====
def set_seed(seed: int):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

# ===== main =====
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--jsonl", default="/root/autodl-tmp/Peptide_3D/utils/dpo/affinity_pairs_cleaned.jsonl")
    ap.add_argument("--init_ckpt", type=str, default='/root/autodl-tmp/Peptide_3D/logs_Ranger_no_DPO/best_model_epoch_72_loss_2.0048.pth',
                    help="policy/ref init checkpoint from supervised stage")
    ap.add_argument("--save_dir", type=str, default="/root/autodl-tmp/Peptide_3D/logs_Ranger_dpo")

    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--batch_size", type=int, default=1)
    ap.add_argument("--num_workers", type=int, default=1)
    ap.add_argument("--grad_accum", type=int, default=1)

    ap.add_argument("--beta", type=float, default=0.1)
    ap.add_argument("--lr", type=float, default=5e-6)
    ap.add_argument("--optimizer", type=str, default="ranger", choices=["adamw","ranger"])

    ap.add_argument("--max_len", type=int, default=None, help="truncate peptide CA matrix to LxL")
    ap.add_argument("--max_receptor_len", type=int, default=512, help="truncate receptor length")

    ap.add_argument("--reduce_mode", type=str, default="mean", choices=["sum","mean","sqrt"])
    ap.add_argument("--dpo_mode", type=str, default="soft", choices=["hard","soft","rpo"])
    ap.add_argument("--soft_temp", type=float, default=2.0)
    ap.add_argument("--weight_gamma", type=float, default=2.0)
    ap.add_argument("--margin_m", type=float, default=0.0)
    ap.add_argument("--kl_coef", type=float, default=0.01)
    ap.add_argument("--kl_pairs", type=int, default=2048)

    ap.add_argument("--seed", type=int, default=17)
    ap.add_argument("--use_amp", action="store_true", help="enable GradScaler when dtype==fp16")
    ap.add_argument("--save_every_epoch", action="store_true")
    args = ap.parse_args()

    set_seed(args.seed)
    os.makedirs(args.save_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # data
    ds = DPOPairsDataset(args.jsonl, max_len=args.max_len, max_receptor_len=args.max_receptor_len)
    if len(ds) == 0:
        raise RuntimeError(f"No valid pairs in {args.jsonl}")
    dl = DataLoader(ds, batch_size=args.batch_size, shuffle=True,
                    num_workers=args.num_workers, collate_fn=collate_pairs,
                    pin_memory=False, drop_last=False)

    # models
    policy = ProteinPeptideModel(device).to(device)
    if args.init_ckpt and os.path.exists(args.init_ckpt):
        policy.load_state_dict(torch.load(args.init_ckpt, map_location="cpu"), strict=False)
    ref_model = copy.deepcopy(policy).to(device).eval()
    for p in ref_model.parameters():
        p.requires_grad_(False)

    # optimizer
    if args.optimizer == "ranger":
        if not _HAS_RANGER:
            print("[WARN] Ranger not found, falling back to AdamW.")
            opt = torch.optim.AdamW(policy.parameters(), lr=args.lr, betas=(0.9,0.95))
        else:
            opt = Ranger(policy.parameters(), lr=args.lr, betas=(0.9,0.95))
    else:
        opt = torch.optim.AdamW(policy.parameters(), lr=args.lr, betas=(0.9,0.95))

    scaler = torch.cuda.amp.GradScaler(enabled=(args.use_amp and amp_dtype==torch.float16))

    best = float("inf")
    global_step = 0
    for epoch in range(1, args.epochs+1):
        policy.train()
        epoch_loss = 0.0
        pbar = tqdm(dl, desc=f"DPO+ epoch {epoch}")
        opt.zero_grad(set_to_none=True)
        for i, batch in enumerate(pbar):
            for k in ("pair_weight","delta_reward","chosen_labels","rejected_labels",
                      "receptor_seq_tensor","peptide_seq_tensor"):
                batch[k] = batch[k].to(device, non_blocking=True)

            loss, logs = dpo_step(
                policy, ref_model, batch, device=device,
                beta=args.beta, reduce_mode=args.reduce_mode,
                dpo_mode=args.dpo_mode, soft_temp=args.soft_temp,
                margin_m=args.margin_m, weight_gamma=args.weight_gamma,
                kl_coef=args.kl_coef, kl_pairs=args.kl_pairs, do_kl=True
            )

            # grad accumulate
            if scaler.is_enabled():
                scaler.scale(loss / args.grad_accum).backward()
            else:
                (loss / args.grad_accum).backward()

            if ((i + 1) % args.grad_accum == 0) or (i + 1 == len(dl)):
                torch.nn.utils.clip_grad_norm_(policy.parameters(), max_norm=1.0)
                if scaler.is_enabled():
                    scaler.step(opt)
                    scaler.update()
                else:
                    opt.step()
                opt.zero_grad(set_to_none=True)
                global_step += 1

            epoch_loss += loss.item()
            pbar.set_postfix({
                "loss": f"{logs['loss']:.3f}",
                "dpo": f"{logs['dpo']:.3f}",
                "kl": f"{logs['kl']:.3f}",
                "Δθ": f"{logs['delta_theta']:.3f}",
                "Δref": f"{logs['delta_ref']:.3f}",
                "w": f"{logs['w_conf']:.2f}",
            })

        avg = epoch_loss / max(1,len(dl))
        if args.save_every_epoch:
            ckpt = os.path.join(args.save_dir, f"policy_dpo_epoch{epoch}_loss_{avg:.4f}.pth")
            torch.save(policy.state_dict(), ckpt)
        if avg < best:
            best = avg
            torch.save(policy.state_dict(), os.path.join(args.save_dir, "policy_dpo_best.pth"))

    print(f"[FIN] DPO+ done. Best={best:.4f}, saved under {args.save_dir}")

if __name__ == "__main__":
    main()
