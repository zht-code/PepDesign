#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os, json, argparse, random
from typing import Dict, List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import numpy as np
from Bio import PDB
from Bio.SeqUtils import seq1 as _seq1

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

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
try:
    amp_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
except Exception:
    amp_dtype = torch.float16

_AA3_CUSTOM = {
    "MSE": "M", "SEC": "U", "PYL": "O",
    "HID": "H", "HIE": "H", "HIP": "H",
    "CYX": "C", "ASX": "B", "GLX": "Z", "UNK": "X",
}


def resname_to_one(resname: str) -> str:
    try:
        return _seq1(resname.strip(), custom_map=_AA3_CUSTOM, undef_code="X")
    except Exception:
        return "X"


BIN_EDGES = np.linspace(2.0, 22.0, num=65)  # 64 bins
NUM_BINS = len(BIN_EDGES) - 1

sequence_tokenizer = EsmSequenceTokenizer()
PAD_ID = 0


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def distance_bins_from_pdb(peptide_pdb: str, max_len: Optional[int] = None) -> torch.LongTensor:
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
    X = np.stack(cas, axis=0)
    if max_len is not None:
        X = X[:max_len]
    D = np.sqrt(np.sum((X[:, None, :] - X[None, :, :]) ** 2, axis=-1))
    bins = np.digitize(D, BIN_EDGES) - 1
    bins = np.clip(bins, 0, NUM_BINS - 1).astype(np.int64)
    np.fill_diagonal(bins, -100)  # ignore self-pairs
    return torch.from_numpy(bins)


def struct_ce_loss(struct_logits: torch.Tensor, labels_2d: torch.Tensor) -> torch.Tensor:
    """
    struct_logits: [B,Lq,Lk,NUM_BINS] or [B,Lq,Lk*NUM_BINS]
    labels_2d:     [B,L,L] with -100 ignored
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
    L = min(struct_logits.size(1), struct_logits.size(2), Ltgt)
    logits = struct_logits[:, :L, :L, :]  # [B,L,L,C]
    labels = labels_2d[:, :L, :L]         # [B,L,L]
    loss = F.cross_entropy(
        logits.reshape(-1, NUM_BINS),
        labels.reshape(-1),
        ignore_index=-100,
        reduction="mean",
    )
    return loss


class SFTChosenDataset(Dataset):
    """
    复用 DPO jsonl 格式，只取 chosen 做监督学习：
      {
        "prompt":  {"receptor_pdb": "...", "peptide_seq": "..."},
        "chosen":  {"pdb_path":"...", "score":{...}},
        "rejected":{"pdb_path":"...", ...}   # 可忽略
      }
    """

    def __init__(
        self,
        jsonl_path: str,
        max_len: Optional[int] = None,
        max_receptor_len: Optional[int] = None,
        target_field: str = "chosen",
    ):
        self.rows = []
        self.max_len = max_len
        self.max_receptor_len = max_receptor_len
        self.target_field = target_field

        with open(jsonl_path, "r", encoding="utf-8") as f:
            for ln in f:
                ln = ln.strip()
                if not ln:
                    continue
                try:
                    r = json.loads(ln)
                    rp = r["prompt"]["receptor_pdb"]
                    pep = r["prompt"]["peptide_seq"]
                    tp = r.get(target_field, {}).get("pdb_path", None)
                    if not (rp and pep and tp):
                        continue
                    if not (os.path.exists(rp) and os.path.exists(tp)):
                        continue
                    self.rows.append(r)
                except Exception:
                    continue

    def __len__(self) -> int:
        return len(self.rows)

    @staticmethod
    def _seq_from_receptor(receptor_pdb: str) -> str:
        parser = PDB.PDBParser(QUIET=True)
        structure = parser.get_structure("rec", receptor_pdb)
        residues = [res for res in structure.get_residues() if PDB.is_aa(res, standard=False)]
        seq = "".join(resname_to_one(res.get_resname()) for res in residues)
        return seq if len(seq) > 0 else "A"

    def __getitem__(self, idx: int) -> Dict:
        r = self.rows[idx]
        receptor_pdb = r["prompt"]["receptor_pdb"]
        peptide_seq = r["prompt"]["peptide_seq"]
        target_pdb = r[self.target_field]["pdb_path"]

        rec_seq = self._seq_from_receptor(receptor_pdb)
        rec_ids = tokenize_sequence(rec_seq, sequence_tokenizer)
        if self.max_receptor_len is not None:
            rec_ids = rec_ids[: self.max_receptor_len]
        pep_ids = tokenize_sequence(peptide_seq, sequence_tokenizer)
        labels = distance_bins_from_pdb(target_pdb, max_len=self.max_len)

        return {
            "receptor_seq_tensor": rec_ids,
            "peptide_seq_tensor": pep_ids,
            "labels": labels,
            "meta": {"receptor_pdb": receptor_pdb, "target_pdb": target_pdb},
        }


def collate_sft(batch_list: List[Dict]) -> Dict:
    def pad_1d(tensors, pad_id=PAD_ID):
        L = max(t.size(0) for t in tensors)
        out = torch.full((len(tensors), L), pad_id, dtype=torch.long)
        for i, t in enumerate(tensors):
            out[i, : t.size(0)] = t
        return out

    def pad_2d_mats(mats):
        L = max(m.size(0) for m in mats)
        outs = []
        for m in mats:
            mm = torch.full((L, L), -100, dtype=torch.long)
            l = m.size(0)
            mm[:l, :l] = m
            outs.append(mm)
        return torch.stack(outs, dim=0)

    rec = pad_1d([b["receptor_seq_tensor"] for b in batch_list], PAD_ID)
    pep = pad_1d([b["peptide_seq_tensor"] for b in batch_list], PAD_ID)
    lab = pad_2d_mats([b["labels"] for b in batch_list])

    return {
        "receptor_seq_tensor": rec,
        "peptide_seq_tensor": pep,
        "labels": lab,
        "meta": [b["meta"] for b in batch_list],
    }


def ceil_div(a, b):
    return (a + b - 1) // b


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--aff_jsonl", default="/root/autodl-tmp/Peptide_3D/utils/dpo/affinity_pairs_cleaned.jsonl")
    ap.add_argument("--stab_jsonl", default="/root/autodl-tmp/Peptide_3D/utils/dpo/stability_pairs.jsonl")
    ap.add_argument("--sol_jsonl", default="/root/autodl-tmp/Peptide_3D/utils/dpo/solubility_pairs.jsonl")
    ap.add_argument("--jsonl", default=None, help="兼容老参数：只给 --jsonl 时视为 affinity 数据")

    ap.add_argument("--target_field", type=str, default="chosen", choices=["chosen", "rejected"])
    ap.add_argument("--init_ckpt", type=str, default="/root/autodl-tmp/Peptide_3D/logs_data_augmentation/best_model_epoch_59_loss_0.4448.pth")
    ap.add_argument("--save_dir", type=str, default="/root/autodl-tmp/Peptide_3D/logs_SFT")

    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--batch_size", type=int, default=1)
    ap.add_argument("--num_workers", type=int, default=1)
    ap.add_argument("--lr", type=float, default=5e-6)
    ap.add_argument("--optimizer", type=str, default="ranger", choices=["adamw", "ranger"])
    ap.add_argument("--grad_accum", type=int, default=1)
    ap.add_argument("--clip_grad", type=float, default=1.0)
    ap.add_argument("--use_amp", action="store_true")

    ap.add_argument("--max_len", type=int, default=None)
    ap.add_argument("--max_receptor_len", type=int, default=512)

    ap.add_argument("--lambda_aff", type=float, default=1.0)
    ap.add_argument("--lambda_stab", type=float, default=0.35)
    ap.add_argument("--lambda_sol", type=float, default=0.35)
    ap.add_argument("--normalize_lambda", action="store_true")

    ap.add_argument("--save_every_epoch", action="store_true")
    ap.add_argument("--seed", type=int, default=17)
    args = ap.parse_args()

    set_seed(args.seed)
    os.makedirs(args.save_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if args.jsonl and os.path.exists(args.jsonl):
        args.aff_jsonl = args.jsonl

    comps = []

    def _try_add(name, path, lam):
        if path and os.path.exists(path) and lam > 0:
            ds = SFTChosenDataset(
                path,
                max_len=args.max_len,
                max_receptor_len=args.max_receptor_len,
                target_field=args.target_field,
            )
            if len(ds) > 0:
                dl = DataLoader(
                    ds,
                    batch_size=args.batch_size,
                    shuffle=True,
                    num_workers=args.num_workers,
                    collate_fn=collate_sft,
                    pin_memory=False,
                    drop_last=False,
                )
                steps = ceil_div(len(ds), args.batch_size)
                comps.append({"name": name, "lambda": lam, "ds": ds, "dl": dl, "steps": steps})

    _try_add("aff", args.aff_jsonl, args.lambda_aff)
    _try_add("stab", args.stab_jsonl, args.lambda_stab)
    _try_add("sol", args.sol_jsonl, args.lambda_sol)

    if len(comps) == 0:
        raise RuntimeError("没有可用的任务数据（请检查 *jsonl 路径与 λ 是否>0）。")

    if args.normalize_lambda:
        s = sum(c["lambda"] for c in comps)
        if s > 0:
            for c in comps:
                c["lambda"] = c["lambda"] / s

    model = ProteinPeptideModel(device).to(device)
    if args.init_ckpt and os.path.exists(args.init_ckpt):
        model.load_state_dict(torch.load(args.init_ckpt, map_location="cpu"), strict=False)

    if args.optimizer == "ranger" and _HAS_RANGER:
        opt = Ranger(model.parameters(), lr=args.lr, betas=(0.9, 0.95))
    else:
        if args.optimizer == "ranger":
            print("[WARN] Ranger not found, falling back to AdamW.")
        opt = torch.optim.AdamW(model.parameters(), lr=args.lr, betas=(0.9, 0.95))

    scaler = torch.cuda.amp.GradScaler(enabled=(args.use_amp and amp_dtype == torch.float16))

    best = float("inf")
    for epoch in range(1, args.epochs + 1):
        model.train()
        iters = {c["name"]: iter(c["dl"]) for c in comps}
        steps_max = max(c["steps"] for c in comps)

        epoch_loss_total = 0.0
        pbar = tqdm(range(steps_max), desc=f"SFT multi epoch {epoch}", dynamic_ncols=True)
        micro = 0

        opt.zero_grad(set_to_none=True)
        for step in pbar:
            logs_step = {"loss": 0.0}
            for c in comps:
                name, lam, steps_i = c["name"], c["lambda"], c["steps"]
                if step >= steps_i:
                    continue
                try:
                    batch = next(iters[name])
                except StopIteration:
                    iters[name] = iter(c["dl"])
                    batch = next(iters[name])

                base_batch = {
                    "receptor_seq_tensor": batch["receptor_seq_tensor"].to(device, non_blocking=True),
                    "peptide_seq_tensor": batch["peptide_seq_tensor"].to(device, non_blocking=True),
                    "attr_cross_attention_mask": None,
                    "cross_attention_mask": None,
                }
                # ProteinPeptideModel.forward 里会用到这些键；SFT 没有标签时用 0 填充即可
                B = base_batch["peptide_seq_tensor"].size(0)
                for k in ("stability", "solubility", "vina_affinity"):
                    base_batch[k] = torch.zeros(B, dtype=torch.float32, device=device)
                labels = batch["labels"].to(device, non_blocking=True)

                with torch.cuda.amp.autocast(dtype=amp_dtype):
                    _, struct_logits = model(base_batch)
                    loss_i = struct_ce_loss(struct_logits, labels)
                    scaled = lam * loss_i

                if scaler.is_enabled():
                    scaler.scale(scaled / args.grad_accum).backward()
                else:
                    (scaled / args.grad_accum).backward()

                micro += 1
                epoch_loss_total += float(lam) * float(loss_i.item())
                logs_step["loss"] += float(lam) * float(loss_i.item())

                if (micro % args.grad_accum) == 0:
                    if args.clip_grad and args.clip_grad > 0:
                        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=args.clip_grad)
                    if scaler.is_enabled():
                        scaler.step(opt)
                        scaler.update()
                    else:
                        opt.step()
                    opt.zero_grad(set_to_none=True)

            pbar.set_postfix({"loss": f"{logs_step['loss']:.3f}"})

        avg = epoch_loss_total / max(1, steps_max)
        if args.save_every_epoch:
            ckpt = os.path.join(args.save_dir, f"sft_epoch{epoch}_loss_{avg:.4f}.pth")
            torch.save(model.state_dict(), ckpt)
        if avg < best:
            best = avg
            torch.save(model.state_dict(), os.path.join(args.save_dir, "sft_best.pth"))

    print(f"[FIN] SFT done. Best={best:.4f}, saved under {args.save_dir}")


if __name__ == "__main__":
    main()

'''

python /root/autodl-tmp/Peptide_3D/results/3_Pareto_improved/train_SFT_multi_objective.py \
  --use_amp \
  --epochs 3 \
  --batch_size 1 \
  --grad_accum 1

'''