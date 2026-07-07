#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
亲和力单任务 DPO：仅使用亲和力偏好对 jsonl，不做 stability/solubility 多目标混合。
数据格式与 utils/dpo/train_DPO_multi_objective.py 中 aff 任务一致。
权重默认保存到 log_affinity_only/。
"""

import os
# 未在外部设置时，仅对本进程暴露物理 0 号 GPU（必须在 import torch 之前）
if "CUDA_VISIBLE_DEVICES" not in os.environ:
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import json, argparse, copy, random
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


BIN_EDGES = np.linspace(2.0, 22.0, num=65)
NUM_BINS = len(BIN_EDGES) - 1

sequence_tokenizer = EsmSequenceTokenizer()
PAD_ID = 0


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
    np.fill_diagonal(bins, -100)
    return torch.from_numpy(bins)


def struct_logits_to_logprob(
    struct_logits: torch.Tensor,
    labels_2d: torch.Tensor,
    reduce_mode: str = "mean",
) -> torch.Tensor:
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
    logits = struct_logits[:, :L, :L, :]
    labels = labels_2d[:, :L, :L]
    valid = labels != -100

    logp = F.log_softmax(logits, dim=-1)
    gather_idx = labels.clamp(min=0).unsqueeze(-1)
    picked = logp.gather(-1, gather_idx).squeeze(-1)
    picked = picked * valid

    M = valid.view(B, -1).sum(dim=-1).clamp(min=1)
    s = picked.view(B, -1).sum(dim=-1)
    if reduce_mode == "mean":
        return s / M
    elif reduce_mode == "sqrt":
        return s / M.sqrt()
    else:
        return s


class DPOPairsDataset(Dataset):
    """亲和力偏好对 jsonl（chosen/rejected + R 或 hdock）。"""

    def __init__(
        self,
        jsonl_path: str,
        max_len: Optional[int] = None,
        max_receptor_len: Optional[int] = None,
    ):
        self.rows = []
        with open(jsonl_path, "r", encoding="utf-8") as f:
            for ln in f:
                ln = ln.strip()
                if not ln:
                    continue
                r = json.loads(ln)
                try:
                    rp = r["prompt"]["receptor_pdb"]
                    pep = r["prompt"]["peptide_seq"]
                    cp = r["chosen"]["pdb_path"]
                    rpj = r["rejected"]["pdb_path"]
                    if not (os.path.exists(rp) and os.path.exists(cp) and os.path.exists(rpj)):
                        continue
                    r["_pair_weight"] = float(r.get("pair_weight", 1.0))
                    cR = r.get("chosen", {}).get("score", {}).get("R", None)
                    rR = r.get("rejected", {}).get("score", {}).get("R", None)
                    if cR is not None and rR is not None:
                        r["_delta_reward"] = float(cR - rR)
                    else:
                        ch = r.get("chosen", {}).get("score", {}).get("hdock", None)
                        rh = r.get("rejected", {}).get("score", {}).get("hdock", None)
                        if ch is not None and rh is not None:
                            r["_delta_reward"] = float(rh - ch)
                        else:
                            r["_delta_reward"] = 0.0
                    self.rows.append(r)
                except Exception:
                    continue
        self.max_len = max_len
        self.max_receptor_len = max_receptor_len

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, idx: int) -> Dict:
        r = self.rows[idx]
        receptor_pdb = r["prompt"]["receptor_pdb"]
        peptide_seq = r["prompt"]["peptide_seq"]
        chosen_pdb = r["chosen"]["pdb_path"]
        rejected_pdb = r["rejected"]["pdb_path"]
        w = float(r.get("_pair_weight", 1.0))
        delta_reward = float(r.get("_delta_reward", 0.0))

        rec_seq = self._seq_from_receptor(receptor_pdb)
        rec_ids = tokenize_sequence(rec_seq, sequence_tokenizer)
        if self.max_receptor_len is not None:
            rec_ids = rec_ids[: self.max_receptor_len]
        pep_ids = tokenize_sequence(peptide_seq, sequence_tokenizer)

        chosen_labels = distance_bins_from_pdb(chosen_pdb, max_len=self.max_len)
        rejected_labels = distance_bins_from_pdb(rejected_pdb, max_len=self.max_len)

        return {
            "receptor_seq_tensor": rec_ids,
            "peptide_seq_tensor": pep_ids,
            "chosen_labels": chosen_labels,
            "rejected_labels": rejected_labels,
            "pair_weight": torch.tensor(w, dtype=torch.float32),
            "delta_reward": torch.tensor(delta_reward, dtype=torch.float32),
            "meta": {
                "receptor_pdb": receptor_pdb,
                "chosen_pdb": chosen_pdb,
                "rejected_pdb": rejected_pdb,
            },
        }

    @staticmethod
    def _seq_from_receptor(receptor_pdb: str) -> str:
        parser = PDB.PDBParser(QUIET=True)
        structure = parser.get_structure("rec", receptor_pdb)
        residues = [res for res in structure.get_residues() if PDB.is_aa(res, standard=False)]
        seq = "".join(resname_to_one(res.get_resname()) for res in residues)
        return seq if len(seq) > 0 else "A"


def collate_pairs(batch_list: List[Dict]) -> Dict:
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
    ch = pad_2d_mats([b["chosen_labels"] for b in batch_list])
    rj = pad_2d_mats([b["rejected_labels"] for b in batch_list])
    w = torch.stack([b["pair_weight"] for b in batch_list], dim=0)
    dr = torch.stack([b["delta_reward"] for b in batch_list], dim=0)

    return {
        "receptor_seq_tensor": rec,
        "peptide_seq_tensor": pep,
        "chosen_labels": ch,
        "rejected_labels": rj,
        "pair_weight": w,
        "delta_reward": dr,
        "meta": [b["meta"] for b in batch_list],
    }


def dpo_objective(
    delta_theta,
    delta_ref,
    delta_reward,
    beta=0.1,
    mode="soft",
    soft_temp=2.0,
    margin_m=0.0,
    weight_gamma=2.0,
):
    z = beta * (delta_theta - delta_ref)
    w_conf = torch.sigmoid(weight_gamma * (delta_reward - margin_m))

    if mode == "hard":
        loss = -F.logsigmoid(z) * w_conf
    elif mode == "soft":
        y = torch.sigmoid(delta_reward / soft_temp)
        loss = F.binary_cross_entropy_with_logits(z, y, reduction="none") * w_conf
    elif mode == "rpo":
        target = torch.tanh(delta_reward / soft_temp)
        loss = F.smooth_l1_loss((z / beta), target, reduction="none") * w_conf
    else:
        raise ValueError(f"unknown mode={mode}")

    return loss.mean(), {"w_conf": w_conf.mean().item()}


def kl_regularizer(policy_logits, ref_logits, max_pairs=2048):
    if policy_logits.dim() == 3:
        B, Lq, last = policy_logits.shape
        Lk = last // NUM_BINS
        policy_logits = policy_logits.view(B, Lq, Lk, NUM_BINS)
        ref_logits = ref_logits.view(B, Lq, Lk, NUM_BINS)

    B, L, K, C = policy_logits.shape
    N = min(max_pairs, B * L * K)
    idx = torch.randint(0, B * L * K, (N,), device=policy_logits.device)
    p = policy_logits.reshape(B * L * K, C)[idx]
    q = ref_logits.reshape(B * L * K, C)[idx]
    p_log = F.log_softmax(p, dim=-1)
    q_log = F.log_softmax(q, dim=-1)
    p_prob = p_log.exp()
    kl = (p_prob * (p_log - q_log)).sum(dim=-1).mean()
    return kl


def dpo_step(
    model: nn.Module,
    model_ref: nn.Module,
    batch: Dict,
    device: torch.device,
    beta: float,
    reduce_mode="mean",
    dpo_mode="soft",
    soft_temp=2.0,
    margin_m=0.0,
    weight_gamma=2.0,
    kl_coef=0.01,
    kl_pairs=2048,
    do_kl=True,
):
    B = batch["peptide_seq_tensor"].size(0)
    base_batch = {
        "receptor_seq_tensor": batch["receptor_seq_tensor"].to(device),
        "peptide_seq_tensor": batch["peptide_seq_tensor"].to(device),
        "attr_cross_attention_mask": None,
        "cross_attention_mask": None,
    }
    for k in ("stability", "solubility", "vina_affinity"):
        base_batch[k] = batch.get(k, torch.zeros(B, dtype=torch.float32, device=device))

    with torch.cuda.amp.autocast(dtype=amp_dtype):
        _, struct_logits_p = model(base_batch)

    lp_c = struct_logits_to_logprob(
        struct_logits_p, batch["chosen_labels"].to(device), reduce_mode
    )
    lp_r = struct_logits_to_logprob(
        struct_logits_p, batch["rejected_labels"].to(device), reduce_mode
    )
    delta_theta = lp_c - lp_r

    with torch.no_grad():
        _, struct_logits_q = model_ref(base_batch)
        ref_c = struct_logits_to_logprob(
            struct_logits_q, batch["chosen_labels"].to(device), reduce_mode
        )
        ref_r = struct_logits_to_logprob(
            struct_logits_q, batch["rejected_labels"].to(device), reduce_mode
        )
        delta_ref = ref_c - ref_r

    delta_reward = batch.get("delta_reward", torch.zeros(B, device=device))
    dpo_loss, info = dpo_objective(
        delta_theta,
        delta_ref,
        delta_reward,
        beta=beta,
        mode=dpo_mode,
        soft_temp=soft_temp,
        margin_m=margin_m,
        weight_gamma=weight_gamma,
    )

    if do_kl and kl_coef > 0:
        if struct_logits_p.dim() == 3:
            Bp, Lq, last = struct_logits_p.shape
            Lk = last // NUM_BINS
            struct_logits_p = struct_logits_p.view(Bp, Lq, Lk, NUM_BINS)
            struct_logits_q = struct_logits_q.view(Bp, Lq, Lk, NUM_BINS)
        Lt = min(
            struct_logits_p.size(1),
            struct_logits_p.size(2),
            struct_logits_q.size(1),
            struct_logits_q.size(2),
        )
        kl = kl_regularizer(
            struct_logits_p[:, :Lt, :Lt, :],
            struct_logits_q[:, :Lt, :Lt, :],
            max_pairs=kl_pairs,
        )
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


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def ceil_div(a, b):
    return (a + b - 1) // b


def main():
    ap = argparse.ArgumentParser(description="亲和力单任务 DPO")
    ap.add_argument(
        "--jsonl",
        type=str,
        default="/root/autodl-tmp/Peptide_3D/utils/dpo/affinity_pairs_cleaned.jsonl",
        help="亲和力偏好对 jsonl",
    )
    ap.add_argument(
        "--init_ckpt",
        type=str,
        default="/root/autodl-tmp/Peptide_3D/logs_Ranger_no_DPO/best_model_epoch_72_loss_2.0048.pth",
    )
    ap.add_argument(
        "--save_dir",
        type=str,
        default="/root/autodl-tmp/Peptide_3D/log_affinity_only",
    )

    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--batch_size", type=int, default=1)
    ap.add_argument("--num_workers", type=int, default=1)
    ap.add_argument("--beta", type=float, default=0.1)
    ap.add_argument("--lr", type=float, default=5e-6)
    ap.add_argument("--optimizer", type=str, default="ranger", choices=["adamw", "ranger"])
    ap.add_argument("--max_len", type=int, default=None)
    ap.add_argument("--max_receptor_len", type=int, default=512)
    ap.add_argument("--reduce_mode", type=str, default="mean", choices=["sum", "mean", "sqrt"])
    ap.add_argument("--dpo_mode", type=str, default="soft", choices=["hard", "soft", "rpo"])
    ap.add_argument("--soft_temp", type=float, default=2.0)
    ap.add_argument("--weight_gamma", type=float, default=2.0)
    ap.add_argument("--margin_m", type=float, default=0.0)
    ap.add_argument("--kl_coef", type=float, default=0.01)
    ap.add_argument("--kl_pairs", type=int, default=2048)
    ap.add_argument("--seed", type=int, default=17)
    ap.add_argument("--use_amp", action="store_true")
    ap.add_argument("--grad_accum", type=int, default=1)
    ap.add_argument("--save_every_epoch", action="store_true")
    ap.add_argument(
        "--gpu",
        type=int,
        default=0,
        help="本进程使用的 CUDA 序号（默认 0；若已设置 CUDA_VISIBLE_DEVICES，通常为 0）",
    )
    args = ap.parse_args()

    set_seed(args.seed)
    os.makedirs(args.save_dir, exist_ok=True)
    if torch.cuda.is_available():
        torch.cuda.set_device(args.gpu)
        device = torch.device(f"cuda:{args.gpu}")
    else:
        device = torch.device("cpu")

    if not os.path.exists(args.jsonl):
        raise RuntimeError(f"找不到亲和力 jsonl: {args.jsonl}")

    ds = DPOPairsDataset(args.jsonl, max_len=args.max_len, max_receptor_len=args.max_receptor_len)
    if len(ds) == 0:
        raise RuntimeError("亲和力数据集为空，请检查 jsonl 与 pdb 路径。")

    dl = DataLoader(
        ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        collate_fn=collate_pairs,
        pin_memory=False,
        drop_last=False,
    )
    steps_per_epoch = ceil_div(len(ds), args.batch_size)

    policy = ProteinPeptideModel(device).to(device)
    if args.init_ckpt and os.path.exists(args.init_ckpt):
        policy.load_state_dict(torch.load(args.init_ckpt, map_location="cpu"), strict=False)
    ref_model = copy.deepcopy(policy).to(device).eval()
    for p in ref_model.parameters():
        p.requires_grad_(False)

    if args.optimizer == "ranger" and _HAS_RANGER:
        opt = Ranger(policy.parameters(), lr=args.lr, betas=(0.9, 0.95))
    else:
        if args.optimizer == "ranger":
            print("[WARN] Ranger not found, falling back to AdamW.")
        opt = torch.optim.AdamW(policy.parameters(), lr=args.lr, betas=(0.9, 0.95))
    scaler = torch.cuda.amp.GradScaler(enabled=(args.use_amp and amp_dtype == torch.float16))

    best = float("inf")
    for epoch in range(1, args.epochs + 1):
        policy.train()
        epoch_loss_total = 0.0
        pbar = tqdm(range(steps_per_epoch), desc=f"DPO affinity epoch {epoch}", dynamic_ncols=True)
        micro = 0
        opt.zero_grad(set_to_none=True)
        it = iter(dl)

        for step in pbar:
            try:
                batch = next(it)
            except StopIteration:
                it = iter(dl)
                batch = next(it)

            for k in (
                "pair_weight",
                "delta_reward",
                "chosen_labels",
                "rejected_labels",
                "receptor_seq_tensor",
                "peptide_seq_tensor",
            ):
                batch[k] = batch[k].to(device, non_blocking=True)

            loss_i, logs_i = dpo_step(
                policy,
                ref_model,
                batch,
                device=device,
                beta=args.beta,
                reduce_mode=args.reduce_mode,
                dpo_mode=args.dpo_mode,
                soft_temp=args.soft_temp,
                margin_m=args.margin_m,
                weight_gamma=args.weight_gamma,
                kl_coef=args.kl_coef,
                kl_pairs=args.kl_pairs,
                do_kl=True,
            )

            if scaler.is_enabled():
                scaler.scale(loss_i / args.grad_accum).backward()
            else:
                (loss_i / args.grad_accum).backward()

            micro += 1
            epoch_loss_total += float(loss_i.item())

            pbar.set_postfix(
                {
                    "loss": f"{logs_i['loss']:.3f}",
                    "dpo": f"{logs_i['dpo']:.3f}",
                    "kl": f"{logs_i['kl']:.3f}",
                }
            )

            if (micro % args.grad_accum) == 0:
                torch.nn.utils.clip_grad_norm_(policy.parameters(), max_norm=1.0)
                if scaler.is_enabled():
                    scaler.step(opt)
                    scaler.update()
                else:
                    opt.step()
                opt.zero_grad(set_to_none=True)

        avg = epoch_loss_total / max(1, steps_per_epoch)
        if args.save_every_epoch:
            ckpt = os.path.join(
                args.save_dir, f"policy_dpo_affinity_epoch{epoch}_loss_{avg:.4f}.pth"
            )
            torch.save(policy.state_dict(), ckpt)
        if avg < best:
            best = avg
            torch.save(
                policy.state_dict(),
                os.path.join(args.save_dir, "policy_dpo_affinity_best.pth"),
            )

    print(f"[FIN] DPO affinity-only done. Best={best:.4f}, saved under {args.save_dir}")


if __name__ == "__main__":
    main()


'''

nohup python /root/autodl-tmp/Peptide_3D/results/3_Pareto_improved/train_DPO_affinity_only.py \
  --jsonl /root/autodl-tmp/Peptide_3D/utils/dpo/affinity_pairs_cleaned.jsonl \
  --use_amp \
  --epochs 5
  > /root/autodl-tmp/Peptide_3D/results/3_Pareto_improved/affinity_only.log 2>&1 &

'''