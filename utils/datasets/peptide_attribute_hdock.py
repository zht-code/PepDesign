#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Batch scoring for peptide datasets laid out as:
  <root>/<ID>/receptor.pdb
  <root>/<ID>/peptide.pdb

Outputs three JSON files in --outdir:
  stability_scores.json
  solubility_scores.json
  hdock_scores.json

Stability & solubility are fast heuristics that need only peptide PDB.
HDOCK requires hdock-lite installed.
"""

import os, re, json, math, shutil, argparse, subprocess, time
from collections import defaultdict, Counter
from typing import Dict, Tuple, Optional, List

import numpy as np
from Bio.PDB import PDBParser, Polypeptide

# ----------------------------
# Utils
# ----------------------------

def clamp(x, lo=0.0, hi=100.0):
    return float(max(lo, min(hi, x)))

def is_heavy_atom(atom) -> bool:
    # Bio.PDB sometimes lacks element; fallback to name
    elem = getattr(atom, "element", "").strip()
    name = atom.get_name().strip()
    if elem:
        return elem.upper() != "H"
    return not name.upper().startswith("H")

def load_peptide_coords_and_seq(pdb_file: str):
    """Return (coords[N,3], res_idx[N], seq (str), residue_order (list of (chain,id)))"""
    parser = PDBParser(QUIET=True)
    structure = parser.get_structure("pep", pdb_file)
    coords = []
    res_indices = []
    seq_chars = []
    residue_order = []
    res_index_map = {}  # map (chain,id) -> index 0..L-1

    # iterate residues in model 0, all chains (assuming peptide is one chain; robust anyway)
    model = next(structure.get_models())
    idx = 0
    for chain in model:
        for res in chain.get_residues():
            # skip waters/hetero
            hetfield, resseq, icode = res.id
            if hetfield.strip() not in (" ", "A"):  # standard residue only
                continue
            resname = res.get_resname()
            try:
                aa = Polypeptide.three_to_one(resname)
            except KeyError:
                aa = "X"
            seq_chars.append(aa)
            residue_order.append((chain.id, res.id))
            res_index_map[(chain.id, res.id)] = idx

            for atom in res.get_atoms():
                if is_heavy_atom(atom):
                    coords.append(atom.coord)
                    res_indices.append(idx)
            idx += 1

    if len(coords) == 0 or len(seq_chars) == 0:
        raise ValueError(f"No peptide heavy atoms or residues parsed from {pdb_file}")

    return (np.array(coords, dtype=np.float64),
            np.array(res_indices, dtype=np.int32),
            "".join(seq_chars),
            residue_order)

def geometric_center(coords: np.ndarray) -> Tuple[float,float,float]:
    c = coords.mean(axis=0)
    return float(c[0]), float(c[1]), float(c[2])

# ----------------------------
# Stability (heuristic, 0-100)
# ----------------------------

def stability_score_from_structure(coords: np.ndarray, res_idx: np.ndarray) -> float:
    """
    Heuristic stability proxy:
      + more inter-residue contacts (<4.5Å) -> higher
      - more close clashes (<1.8Å) -> lower
    Score ~ [0,100]
    """
    # Pairwise distances (N x N). Peptides are small; OK to do full matrix.
    diff = coords[:, None, :] - coords[None, :, :]
    d2 = np.einsum('ijk,ijk->ij', diff, diff)
    dist = np.sqrt(np.maximum(d2, 0.0))

    different_res = (res_idx[:, None] != res_idx[None, :])

    # Contacts: heavy-atom pairs across different residues < 4.5 Å
    contact_mask = (dist < 4.5) & different_res
    iu = np.triu_indices_from(contact_mask, k=1)
    contact_pairs = np.column_stack([res_idx[iu[0]][contact_mask[iu]],
                                     res_idx[iu[1]][contact_mask[iu]]])
    # unique residue pairs
    if contact_pairs.size > 0:
        contact_pairs = np.unique(np.sort(contact_pairs, axis=1), axis=0)
        L = int(res_idx.max()) + 1
        counts = np.zeros(L, dtype=np.int32)
        for a,b in contact_pairs:
            counts[a] += 1
            counts[b] += 1
        contacts_avg = counts.mean()  # avg #contacting residues per residue
    else:
        contacts_avg = 0.0

    # Clashes: very short distances (<1.8Å) across residues (crude proxy)
    clash_mask = (dist < 1.8) & different_res
    clash_pairs = np.column_stack([res_idx[iu[0]][clash_mask[iu]],
                                   res_idx[iu[1]][clash_mask[iu]]]) if np.any(clash_mask[iu]) else np.empty((0,2), int)
    if clash_pairs.size > 0:
        L = int(res_idx.max()) + 1
        clash_counts = np.zeros(L, dtype=np.int32)
        for a,b in clash_pairs:
            clash_counts[a] += 1
            clash_counts[b] += 1
        clashes_per_res = clash_counts.mean()
    else:
        clashes_per_res = 0.0

    # Map to 0-100. Tuned for typical peptide scales (contacts_avg ~ 0..8)
    base = 20.0
    contact_gain = (min(contacts_avg, 8.0) / 8.0) * 75.0
    clash_penalty = (min(clashes_per_res, 1.0) / 1.0) * 15.0
    score = base + contact_gain - clash_penalty
    return clamp(score, 0.0, 100.0)

# ----------------------------
# Solubility (heuristic, 0-100)
# ----------------------------

KD = {  # Kyte-Doolittle hydropathy
'A':1.8,'R':-4.5,'N':-3.5,'D':-3.5,'C':2.5,'Q':-3.5,'E':-3.5,'G':-0.4,'H':-3.2,
'I':4.5,'L':3.8,'K':-3.9,'M':1.9,'F':2.8,'P':-1.6,'S':-0.8,'T':-0.7,'W':-0.9,'Y':-1.3,'V':4.2,
'X':0.0}

def solubility_score_from_seq(seq: str) -> float:
    seq = seq.upper()
    if len(seq) == 0:
        return 0.0
    gravy = float(np.mean([KD.get(a,0.0) for a in seq]))
    counts = Counter(seq)
    L = len(seq)

    # charged fraction (His半权)
    f_charged = (counts.get('D',0)+counts.get('E',0)+counts.get('K',0)+counts.get('R',0)+0.5*counts.get('H',0))/L
    # polar uncharged fraction (T,S,N,Q) + (Y,C半权)
    f_polar = (counts.get('S',0)+counts.get('T',0)+counts.get('N',0)+counts.get('Q',0)+0.5*(counts.get('Y',0)+counts.get('C',0)))/L

    # Map to 0-100：带电/极性↑、疏水性↑则扣分（仅对正GRAVY扣分）
    score = 60.0 + 40.0*(0.6*f_charged + 0.4*f_polar) - 25.0*max(gravy, 0.0)
    return clamp(score, 0.0, 100.0)

# ----------------------------
# HDOCK
# ----------------------------

def run_hdock(receptor_pdb: str, peptide_pdb: str, workdir: str, hdock_bin: Optional[str]=None, timeout_s: int=900) -> Tuple[Optional[float], Tuple[float,float,float], List[str]]:
    """
    Run HDOCK-Lite if available.
    Returns: (score or None, center_xyz, logs)
      center_xyz: center of docked peptide if model found, else center of input peptide
    """
    logs = []
    os.makedirs(workdir, exist_ok=True)
    r_fn = os.path.join(workdir, "receptor.pdb")
    l_fn = os.path.join(workdir, "peptide.pdb")
    shutil.copy2(receptor_pdb, r_fn)
    shutil.copy2(peptide_pdb, l_fn)

    # Default hdock binary name
    candidates = []
    if hdock_bin:
        candidates.append(hdock_bin)
    candidates += ["hdock", "./hdock"]

    hdock_exec = None
    for c in candidates:
        if shutil.which(c) or os.path.isfile(c):
            hdock_exec = c
            break

    # compute peptide center now; may be used as fallback
    try:
        coords, res_idx, seq, _ = load_peptide_coords_and_seq(l_fn)
        center_fallback = geometric_center(coords)
    except Exception as e:
        logs.append(f"[WARN] Failed to parse peptide for center: {e}")
        center_fallback = (0.0,0.0,0.0)

    if hdock_exec is None:
        logs.append("[WARN] HDOCK executable not found; skip docking.")
        return (None, center_fallback, logs)

    # Try to run hdock: common usage is `hdock receptor.pdb peptide.pdb`
    cmd = [hdock_exec, r_fn, l_fn]
    logs.append(f"[INFO] Running: {' '.join(cmd)}")
    try:
        proc = subprocess.run(cmd, cwd=workdir, stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=timeout_s, text=True)
        logs.append(proc.stdout or "")
        if proc.returncode != 0:
            logs.append(f"[WARN] hdock returned code {proc.returncode}: {proc.stderr}")
    except subprocess.TimeoutExpired:
        logs.append("[WARN] hdock timed out.")
    except Exception as e:
        logs.append(f"[WARN] hdock failed to start: {e}")

    # After run, try to find score and a docked model
    score = None
    docked_pdb = None

    # 1) Search for an .out/.log containing "score"
    text_files = []
    for root,_,files in os.walk(workdir):
        for f in files:
            if f.lower().endswith((".out",".log",".txt")) or f.lower() in ("hdock.out","result.out","output.txt"):
                text_files.append(os.path.join(root,f))

    score_regexes = [
        re.compile(r'(?i)\bhdock\b.*?\bscore\b[^-+\d]*([+-]?\d+(?:\.\d+)?)'),
        re.compile(r'(?i)\bscore\b[^-+\d]*([+-]?\d+(?:\.\d+)?)')
    ]
    for tf in text_files:
        try:
            with open(tf, "r", errors="ignore") as fh:
                for line in fh:
                    for rgx in score_regexes:
                        m = rgx.search(line)
                        if m:
                            val = float(m.group(1))
                            # Choose the "best" (most negative) if multiple
                            score = val if score is None else (val if val < score else score)
        except Exception:
            pass

    # 2) Look for a docked model PDB to compute center (common names)
    pdb_candidates = []
    for root,_,files in os.walk(workdir):
        for f in files:
            if f.lower().endswith(".pdb") and f not in ("receptor.pdb","peptide.pdb"):
                pdb_candidates.append(os.path.join(root,f))
    # Heuristic: prefer files that contain both receptor+ligand (bigger size)
    if pdb_candidates:
        docked_pdb = max(pdb_candidates, key=lambda p: os.path.getsize(p))

    if docked_pdb:
        try:
            coords2, res_idx2, _, _ = load_peptide_coords_and_seq(docked_pdb)  # if model concatenates, this still extracts peptide residues; if not, fallback
            center_xyz = geometric_center(coords2)
        except Exception:
            center_xyz = center_fallback
    else:
        center_xyz = center_fallback

    return (score, center_xyz, logs)

# ----------------------------
# I/O
# ----------------------------

def find_pairs(root: str) -> List[Tuple[str,str,str]]:
    pairs = []
    for entry in os.scandir(root):
        if not entry.is_dir():
            continue
        rid = entry.name  # key in JSON
        r = os.path.join(entry.path, "receptor.pdb")
        p = os.path.join(entry.path, "peptide.pdb")
        if os.path.isfile(r) and os.path.isfile(p):
            pairs.append((rid, r, p))
    return sorted(pairs, key=lambda x: x[0].lower())

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default='/root/autodl-tmp/Peptide_3D/data/train_data', help="Dataset root, e.g., /root/autodl-tmp/Peptide_3D/data/train_data")
    ap.add_argument("--outdir", default='/root/autodl-tmp/Peptide_3D/data', help="Where to write JSONs and hdock work dirs")
    ap.add_argument("--hdock_bin", default='/root/autodl-fs/HDOCKlite/hdock', help="Path/name of HDOCK-Lite executable (if in PATH, omit)")
    ap.add_argument("--run_hdock", action="store_true", help="Actually run HDOCK (default off). If omitted, only stability/solubility are computed.")
    ap.add_argument("--timeout", type=int, default=900, help="HDOCK timeout (seconds) per pair")
    args = ap.parse_args()

    os.makedirs(args.outdir, exist_ok=True)
    hdock_root = os.path.join(args.outdir, "hdock_work")
    os.makedirs(hdock_root, exist_ok=True)

    pairs = find_pairs(args.root)
    if not pairs:
        raise SystemExit(f"No (receptor.pdb, peptide.pdb) pairs found under {args.root}")

    stability_json: Dict[str, Dict[str, float]] = {}
    solubility_json: Dict[str, Dict[str, float]] = {}
    hdock_json: Dict[str, Dict] = {}

    for rid, r_pdb, p_pdb in pairs:
        print(f"==> {rid}")

        # peptide-only features
        try:
            coords, res_idx, seq, _ = load_peptide_coords_and_seq(p_pdb)
        except Exception as e:
            print(f"[WARN] {rid}: failed to parse peptide ({e}); skip scoring.")
            continue

        # Stability
        try:
            stab = float(stability_score_from_structure(coords, res_idx))
            stability_json[rid] = {"stability_score": float(round(stab, 6))}
        except Exception as e:
            print(f"[WARN] {rid}: stability failed: {e}")

        # Solubility
        try:
            sol = float(solubility_score_from_seq(seq))
            solubility_json[rid] = {"solubility_score": float(round(sol, 6))}
        except Exception as e:
            print(f"[WARN] {rid}: solubility failed: {e}")

        # HDOCK
        if args.run_hdock:
            workdir = os.path.join(hdock_root, rid)
            score, center_xyz, logs = run_hdock(r_pdb, p_pdb, workdir, args.hdock_bin, timeout_s=args.timeout)
            for ln in logs:
                print(ln.strip())
            entry = {"center": {"center_x": float(center_xyz[0]),
                                "center_y": float(center_xyz[1]),
                                "center_z": float(center_xyz[2])}}
            if score is not None:
                entry["score"] = float(score)
            hdock_json[rid] = entry

    # Write JSONs
    with open(os.path.join(args.outdir, "stability_scores.json"), "w") as f:
        json.dump(stability_json, f, indent=2)
    with open(os.path.join(args.outdir, "solubility_scores.json"), "w") as f:
        json.dump(solubility_json, f, indent=2)
    if args.run_hdock:
        with open(os.path.join(args.outdir, "hdock_scores.json"), "w") as f:
            json.dump(hdock_json, f, indent=2)

    print("\nDone:")
    print(f"  stability_scores.json -> {os.path.join(args.outdir,'stability_scores.json')}")
    print(f"  solubility_scores.json -> {os.path.join(args.outdir,'solubility_scores.json')}")
    if args.run_hdock:
        print(f"  hdock_scores.json     -> {os.path.join(args.outdir,'hdock_scores.json')}")
        print(f"  HDOCK working dirs    -> {hdock_root}")

if __name__ == "__main__":
    main()
