#!/usr/bin/env python3
"""
从 cands/*.pdb 批量生成 candidate_peptides.csv，并可选计算：

  - docking_score：默认从 7OUN-PD-L1_target_A/dg_separated_cands.json 读取（键为 pep_XX.pdb）；可用 --docking-score-json 指定。
  - distance_to_PD1_interface：多肽 Cα 质心到「PD-L1 上 PD-1 结合面邻域热点残基」各 Cα 的最近距离（Å）。
  - mmgbsa_score：AmberTools MMPBSA.py（GB 模型）单帧结合自由能 ΔG binding（kcal/mol，越负一般越有利）。

依赖：pandas / numpy / scipy；MM-GBSA 需在 PATH（或 conda activate ambertools）中有：
tleap、cpptraj、MMPBSA.py；（可选 sander 短极小化）。受体/肽拓扑切分使用 cpptraj parmstrip，
不再调用 parmed（避免 conda parmed/OpenMM 与 prmtop 识别问题）。

  python build_candidate_peptides_csv.py
  python build_candidate_peptides_csv.py --skip-mmpbsa
  python build_candidate_peptides_csv.py --mmpbsa-minimize 0 --jobs 4
  KEEP_MMPBSA_WORK=1 python ...   # 失败时保留临时目录便于排查
"""

from __future__ import annotations

import argparse
import contextlib
import json
import os
import re
import shutil
import subprocess
import tempfile
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any

try:
    import fcntl
except ImportError:
    fcntl = None  # type: ignore[assignment]


@contextlib.contextmanager
def _process_lock_file(lock_path: Path):
    """多进程并行时 Amber/MMPBSA 共用系统临时目录会引发冲突，对 MM-GBSA 段全局串行化。"""
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    if fcntl is None:
        yield
        return
    with open(lock_path, "a+", encoding="utf-8") as fp:
        fcntl.flock(fp.fileno(), fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(fp.fileno(), fcntl.LOCK_UN)


def _amber_exec_env(scratch: Path) -> dict[str, str]:
    td = scratch / "_amber_tmp_sub"
    td.mkdir(parents=True, exist_ok=True)
    e = dict(os.environ)
    for k in ("TMPDIR", "TMP", "TEMP"):
        e[k] = str(td.resolve())
    return e


PLAUSIBLE_MMPBSA_KCAL_MAG = float(os.environ.get("MMPBSA_MAX_ABS_DE", "2500"))

import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist

SCRIPT_DIR = Path(__file__).resolve().parent

AA3_TO1 = {
    "ALA": "A",
    "ARG": "R",
    "ASN": "N",
    "ASP": "D",
    "CYS": "C",
    "GLN": "Q",
    "GLU": "E",
    "GLY": "G",
    "HIS": "H",
    "ILE": "I",
    "LEU": "L",
    "LYS": "K",
    "MET": "M",
    "PHE": "F",
    "PRO": "P",
    "SER": "S",
    "THR": "T",
    "TRP": "W",
    "TYR": "Y",
    "VAL": "V",
    "SEC": "U",
    "PYL": "O",
    "MSE": "M",
    "HIP": "H",
    "HID": "H",
    "HIE": "H",
}

# PDB 序号（与当前受体文件一致时需自行核对）；可用 --pd1-hotspot-residue-nums 覆盖
DEFAULT_PD1_HOTSPOT_RESIDUES_NUMS = tuple(
    list(range(53, 65))
    + [111, 112, 113, 119, 120, 121, 124, 125, 126, 134, 135, 138, 142, 146]
)


def peptide_index_from_name(stem: str) -> int:
    m = re.fullmatch(r"pep_(\d+)", stem, flags=re.I)
    if not m:
        raise ValueError(f"文件名须为 pep_NNN.pdb，收到: {stem}")
    return int(m.group(1))


def normalize_peptide_id(stem: str) -> str:
    m = re.fullmatch(r"pep_(\d+)", stem, flags=re.I)
    if not m:
        raise ValueError(f"文件名须为 pep_NNN.pdb，收到: {stem}")
    n = int(m.group(1))
    return f"pep_{n:02d}" if n < 1000 else f"pep_{n}"


def _parse_hotspot_nums(spec: str) -> tuple[int, ...]:
    out: list[int] = []
    for part in spec.replace(" ", "").split(","):
        if not part:
            continue
        if "-" in part:
            a, b = part.split("-", 1)
            out.extend(range(int(a), int(b) + 1))
        else:
            out.append(int(part))
    return tuple(sorted(set(out)))


def sequence_from_pdb(path: Path) -> str:
    order: list[tuple[str, int, str, str]] = []
    seen: set[tuple[str, int, str]] = set()
    with open(path, encoding="utf-8", errors="replace") as f:
        for line in f:
            if not line.startswith("ATOM"):
                continue
            if line[12:16].strip() != "CA":
                continue
            chain = line[21]
            try:
                resseq = int(line[22:26])
            except ValueError:
                continue
            icode = line[26].strip() if len(line) > 26 else " "
            icode = icode or " "
            resname = line[17:20].strip()
            key = (chain, resseq, icode)
            if key in seen:
                continue
            seen.add(key)
            aa = AA3_TO1.get(resname, "X")
            order.append((chain, resseq, icode, aa))
    if not order:
        raise ValueError(f"未从 PDB 解析到 CA 残基: {path}")
    order.sort(key=lambda t: (t[0], t[1], t[2]))
    return "".join(t[3] for t in order)


def _format_docking_score(v: float) -> str:
    s = f"{float(v):.6f}".rstrip("0").rstrip(".")
    return s if s else "0"


def load_docking_scores_json(path: Path) -> dict[str, float]:
    """键为 PDB 文件名（如 pep_01.pdb），值为对接分数。"""
    raw = json.loads(path.read_text(encoding="utf-8"))
    out: dict[str, float] = {}
    for k, val in raw.items():
        kk = str(k).strip()
        try:
            out[kk] = float(val)
        except (TypeError, ValueError):
            continue
    return out


def find_executable(candidates: list[str]) -> str | None:
    for name in candidates:
        p = shutil.which(name)
        if p:
            return p
    return None


def _extract_chain_atoms(pdb_path: Path, chain_id: str) -> list[str]:
    lines_out: list[str] = []
    with open(pdb_path, encoding="utf-8", errors="replace") as f:
        for line in f:
            if not line.startswith("ATOM"):
                continue
            if line[21] != chain_id:
                continue
            resnm = line[17:20].strip().upper()
            if resnm in {"HOH", "WAT", "H2O"}:
                continue
            lines_out.append(line)
    return lines_out


def _pdb_atom_element_symbol(line: str) -> str:
    """返回 ATOM/HETATM 行的元素符号（列 77–78）；不可靠时返回空串。"""
    s = line.rstrip("\r\n")
    if len(s) >= 78:
        return s[76:78].strip().upper()
    return ""


def _drop_hydrogen_atom_lines(atom_lines: list[str]) -> list[str]:
    """
    移除氢原子行再交给 tleap。

    OpenMM 等导出常带一整套显式氢，但与 ff14SB 库内残基（如 N 端 NALA）的氢命名不一致，
    会导致 teLeap「Atom ... H does not have a type」直接失败；去氢后由 LEaP 按模板补氢更稳。
    """
    out: list[str] = []
    for ln in atom_lines:
        if not ln.startswith("ATOM"):
            out.append(ln)
            continue
        el = _pdb_atom_element_symbol(ln)
        if el == "H":
            continue
        if el == "" and len(ln.split()) >= 12:
            # 部分短行无标准元素列时，用末列（OpenMM 常在行尾写元素）
            if ln.split()[-1].strip().upper() == "H":
                continue
        out.append(ln)
    return out


def _unique_residue_count(atom_lines: list[str]) -> int:
    """与 tleap 一致：每条链上 (chain, resSeq, insertion) 为一个残基；勿用 CA 行数。"""
    seen: set[tuple[str, int, str]] = set()
    for ln in atom_lines:
        if not ln.startswith("ATOM"):
            continue
        chain = ln[21]
        try:
            resseq = int(ln[22:26])
        except ValueError:
            continue
        icode = (ln[26] if len(ln) > 26 else " ").strip() or " "
        seen.add((chain, resseq, icode))
    return len(seen)


def write_receptor_ligand_pdbs(
    receptor_pdb: Path,
    peptide_pdb: Path,
    outdir: Path,
    receptor_chain: str,
    peptide_chain: str,
) -> tuple[Path, Path, int, int]:
    """写出 receptor_only.pdb / ligand_only.pdb；第三、四项为 **unique 残基数**（可与 tleap combine 后拓扑编号对齐）。"""
    rec_lines = _extract_chain_atoms(receptor_pdb, receptor_chain)
    pep_lines = _extract_chain_atoms(peptide_pdb, peptide_chain)
    if not rec_lines:
        raise ValueError(f"受体链 {receptor_chain} 无 ATOM：{receptor_pdb}")
    if not pep_lines:
        raise ValueError(f"多肽链 {peptide_chain} 无 ATOM：{peptide_pdb}")

    rec_lines = _drop_hydrogen_atom_lines(rec_lines)
    pep_lines = _drop_hydrogen_atom_lines(pep_lines)
    if not rec_lines:
        raise ValueError(f"去氢后受体链 {receptor_chain} 无 ATOM：{receptor_pdb}")
    if not pep_lines:
        raise ValueError(f"去氢后多肽链 {peptide_chain} 无 ATOM：{peptide_pdb}")

    n_rec_res = _unique_residue_count(rec_lines)
    n_lig_res = _unique_residue_count(pep_lines)
    if n_rec_res < 1 or n_lig_res < 1:
        raise ValueError("受体或多肽未能解析出任何残基（检查链 ID）")

    outdir.mkdir(parents=True, exist_ok=True)
    rec_path = outdir / "receptor_only.pdb"
    lig_path = outdir / "ligand_only.pdb"
    rec_path.write_text("".join(rec_lines) + "END\n", encoding="utf-8")
    lig_path.write_text("".join(pep_lines) + "END\n", encoding="utf-8")
    return rec_path, lig_path, n_rec_res, n_lig_res


def hotspot_ca_coordinates(receptor_pdb: Path, chain_id: str, hotspot_res_nums: tuple[int, ...]) -> np.ndarray:
    coords: list[np.ndarray] = []
    want = set(hotspot_res_nums)
    seen: set[tuple[str, int, str]] = set()

    with open(receptor_pdb, encoding="utf-8", errors="replace") as f:
        for line in f:
            if not line.startswith("ATOM"):
                continue
            if line[12:16].strip() != "CA":
                continue
            ch = line[21]
            if ch != chain_id:
                continue
            try:
                resseq = int(line[22:26])
            except ValueError:
                continue
            if resseq not in want:
                continue
            icode = (line[26] if len(line) > 26 else " ").strip() or " "
            key = (ch, resseq, icode)
            if key in seen:
                continue
            seen.add(key)
            x = float(line[30:38])
            y = float(line[38:46])
            z = float(line[46:54])
            coords.append(np.array([x, y, z]))
    return np.asarray(coords, dtype=float) if coords else np.empty((0, 3))


def peptide_ca_coordinates(peptide_pdb: Path, chain_id: str) -> np.ndarray:
    xyz: list[np.ndarray] = []
    seen: set[tuple[int, str]] = set()
    with open(peptide_pdb, encoding="utf-8", errors="replace") as f:
        for line in f:
            if not line.startswith("ATOM"):
                continue
            if line[12:16].strip() != "CA":
                continue
            if line[21] != chain_id:
                continue
            try:
                resseq = int(line[22:26])
            except ValueError:
                continue
            icode = (line[26] if len(line) > 26 else " ").strip() or " "
            key = (resseq, icode)
            if key in seen:
                continue
            seen.add(key)
            xyz.append(np.array([float(line[30:38]), float(line[38:46]), float(line[46:54])]))
    return np.asarray(xyz, dtype=float) if xyz else np.empty((0, 3))


def distance_centroid_nearest_hotspot(pep_xyz: np.ndarray, hot_xyz: np.ndarray) -> float | None:
    if pep_xyz.shape[0] == 0 or hot_xyz.shape[0] == 0:
        return None
    c = pep_xyz.mean(axis=0).reshape(1, 3)
    return float(cdist(c, hot_xyz).min())


def distance_fallback_min_ca(
    peptide_pdb: Path, pep_chain: str, receptor_pdb: Path, rec_chain: str
) -> float | None:
    p = peptide_ca_coordinates(peptide_pdb, pep_chain)
    if p.shape[0] == 0:
        return None
    ctr = p.mean(axis=0).reshape(1, 3)
    coords: list[np.ndarray] = []
    seen: set[tuple[str, int, str]] = set()
    with open(receptor_pdb, encoding="utf-8", errors="replace") as f:
        for line in f:
            if not line.startswith("ATOM"):
                continue
            if line[12:16].strip() != "CA":
                continue
            if line[21] != rec_chain:
                continue
            try:
                resseq = int(line[22:26])
            except ValueError:
                continue
            icode = (line[26] if len(line) > 26 else " ").strip() or " "
            key = (rec_chain, resseq, icode)
            if key in seen:
                continue
            seen.add(key)
            coords.append(np.array([float(line[30:38]), float(line[38:46]), float(line[46:54])]))
    if not coords:
        return None
    return float(cdist(ctr, np.stack(coords)).min())


def _write_tleap_combine(wd: Path, rec_abs: str, lig_abs: str) -> None:
    (wd / "tleap_combine.in").write_text(
        f"""source leaprc.protein.ff14SB
# 与 igb=5/GB 计算常用 mbondi2 半径兼容（见 Amber MMPBSA 文档）
set default PBradii mbondi2
mol1 = loadpdb {rec_abs}
mol2 = loadpdb {lig_abs}
com = combine {{ mol1 mol2 }}
saveamberparm com com.prmtop com.inpcrd
quit
""",
        encoding="utf-8",
    )


def _write_mmpbsa_in(wd: Path, igb: int) -> None:
    (wd / "mmpbsa.in").write_text(
        f"""&general
 startframe=1, endframe=1, interval=1,
 verbose=2, keep_files=0,
/
&gb
 igb={igb}, saltcon=0.0, surften=0.0072,
/
""",
        encoding="utf-8",
    )


def _run_cpptraj_batch(cwd: Path, cpptraj_exe: str, script_name: str, body: str, *, exec_env: dict[str, str] | None) -> None:
    p = cwd / script_name
    p.write_text(body.rstrip() + "\nquit\n", encoding="utf-8")
    subprocess.run(
        [cpptraj_exe, "-i", str(p)],
        cwd=str(cwd),
        check=False,
        stdin=subprocess.DEVNULL,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        env=exec_env,
    )


def _run_logged(cmd: list[str], cwd: Path, log_fn: Path, *, env: dict[str, str] | None = None) -> int:
    with open(log_fn, "wb") as lf:
        p = subprocess.run(cmd, cwd=str(cwd), stdin=subprocess.DEVNULL, stdout=lf, stderr=subprocess.STDOUT, env=env)
    return int(p.returncode)


def parse_mmgbsa_delta_g(results_path: Path) -> float | None:
    """取 FINAL_RESULTS.dat 里「最后一次」DELTA TOTAL（结合能差分项），剔除明显乱值。"""
    if not results_path.is_file():
        return None
    txt = results_path.read_text(encoding="utf-8", errors="replace")
    allm = list(re.finditer(r"(?im)^DELTA TOTAL\s+(-?\d+(?:\.\d+)?)(?:e[+-]\d+)?\s*", txt))
    if not allm:
        return None
    raw = float(allm[-1].group(1))
    if raw != raw or abs(raw) > PLAUSIBLE_MMPBSA_KCAL_MAG:
        return None
    return raw


def compute_mmgbsa_single(
    wd: Path,
    paths: dict[str, str],
    n_rec_res: int,
    n_lig_res: int,
    *,
    max_minimize_cycles: int,
    igb: int,
    log_err: Path,
) -> float | None:
    """
    wd 内需已有 receptor_only.pdb, ligand_only.pdb。
    n_rec_res / n_lig_res 为受体/肽上唯一残基数（与 tleap combine 后 :1-n_rec、肽段 :n_rec+1- 对齐）。
    """
    rec_p = (wd / "receptor_only.pdb").resolve().as_posix()
    lig_p = (wd / "ligand_only.pdb").resolve().as_posix()
    tl = paths["tleap"]
    cpp_exec = paths["cpptraj"]
    mpbs = paths["MMPBSA.py"]

    exec_env = _amber_exec_env(wd)

    _write_tleap_combine(wd, rec_p, lig_p)
    _run_logged([tl, "-f", "tleap_combine.in"], cwd=wd, log_fn=wd / "tleap.log", env=exec_env)
    if not (wd / "com.prmtop").is_file():
        log_err.write_text((wd / "tleap.log").read_text(encoding="utf-8", errors="replace")[-8000:], encoding="utf-8")
        return None
    prm_head = (wd / "com.prmtop").read_text(encoding="utf-8", errors="replace")[:80]
    if "%VERSION" not in prm_head:
        log_err.write_text(f"tleap 未生成合法 Amber ASCII prmtop（文件头）：{prm_head[:200]!s}", encoding="utf-8")
        return None

    if n_lig_res == 0 or n_rec_res == 0:
        log_err.write_text("n_rec_res/n_lig_res 为 0", encoding="utf-8")
        return None

    lig_first = n_rec_res + 1
    lig_last = n_rec_res + n_lig_res

    _run_cpptraj_batch(
        wd,
        cpp_exec,
        "parmstrip_rec.in",
        f"""parm com.prmtop
parmstrip :{lig_first}-{lig_last}
parmwrite out receptor.prmtop
run""",
        exec_env=exec_env,
    )
    _run_cpptraj_batch(
        wd,
        cpp_exec,
        "parmstrip_lig.in",
        f"""parm com.prmtop
parmstrip :1-{n_rec_res}
parmwrite out ligand.prmtop
run""",
        exec_env=exec_env,
    )

    target_traj = wd / "com.inpcrd"
    if max_minimize_cycles > 0 and paths.get("sander"):
        mini = wd / "mini.in"
        mini.write_text(
            f"""Minimisation
 &cntrl
 imin=1, maxcyc={max_minimize_cycles}, ncyc={max(1, max_minimize_cycles - 5)},
 ntmin=2, ntpr=50, cut=999.0, igb=0, ntb=0,
 /
""",
            encoding="utf-8",
        )
        subprocess.run(
            [paths["sander"], "-O", "-i", "mini.in", "-p", "com.prmtop", "-c", "com.inpcrd",
             "-r", "min.rst7", "-o", "mini.mdout"],
            cwd=str(wd), check=False, stdin=subprocess.DEVNULL,
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
            env=exec_env,
        )
        if (wd / "min.rst7").is_file():
            target_traj = wd / "min.rst7"

    traj_name = target_traj.name
    cpp_in = wd / "cpptraj.in"
    cpp_in.write_text(
        f"""parm com.prmtop
trajin {traj_name}
trajout frame.nc netcdf nobox
run
quit
""",
        encoding="utf-8",
    )
    subprocess.run([cpp_exec, "-i", str(cpp_in)], cwd=str(wd), check=False,
                   stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                   env=exec_env)

    traj_for_mmpbsa = "frame.nc" if (wd / "frame.nc").is_file() else traj_name
    if not (wd / "receptor.prmtop").is_file() or not (wd / "ligand.prmtop").is_file():
        log_err.write_text(
            "cpptraj parmstrip 未生成 receptor.prmtop / ligand.prmtop，请在工作目录查看 parmstrip*.in 并重跑 cpptraj。",
            encoding="utf-8",
        )
        return None

    _write_mmpbsa_in(wd, igb)
    fout = wd / "FINAL_RESULTS.dat"
    cmd = [
        mpbs, "-O", "-i", str(wd / "mmpbsa.in"), "-o", str(fout),
        "-prefix", "mmpbsa_out", "-cp", "com.prmtop", "-rp", "receptor.prmtop",
        "-lp", "ligand.prmtop", "-y", traj_for_mmpbsa,
    ]
    _run_logged(cmd, cwd=wd, log_fn=wd / "mmpbsa_run.log", env=exec_env)

    dg = parse_mmgbsa_delta_g(fout)
    if dg is None:
        for alt in sorted(wd.glob("*.dat")):
            dg = parse_mmgbsa_delta_g(alt)
            if dg is not None:
                break
    if dg is None:
        log_err.write_text((wd / "mmpbsa_run.log").read_text(encoding="utf-8", errors="replace")[-12000:], encoding="utf-8")
    return dg


def discover_amber(paths_out: dict[str, str]) -> bool:
    tl = find_executable(["tleap", "tleap.linux"])
    cj = find_executable(["cpptraj"])
    mb = shutil.which("MMPBSA.py")
    sd = find_executable(["sander", "sander.MPI"])
    paths_out.clear()
    if tl:
        paths_out["tleap"] = tl
    if cj:
        paths_out["cpptraj"] = cj
    if mb:
        paths_out["MMPBSA.py"] = mb
    if sd:
        paths_out["sander"] = sd
    return bool(tl and cj and mb)


def _proc_one_peptide(
    pep_file: str,
    receptor: str,
    rec_chain: str,
    pep_chain: str,
    source: str,
    skip_distance: bool,
    skip_mmpbsa: bool,
    hotspot_xyz: np.ndarray,
    amber_ok: bool,
    amber_paths: dict[str, str],
    tmp_root: str,
    mmpbsa_igb: int,
    mmpbsa_minimize: int,
    docking_scores: dict[str, float],
) -> dict[str, Any]:
    """供多进程调用；路径均用 str。"""
    pep_path = Path(pep_file)
    receptor_path = Path(receptor)
    pid = normalize_peptide_id(pep_path.stem)
    seq = sequence_from_pdb(pep_path)
    abs_pdb = str(pep_path.resolve())

    dock_key = pep_path.name
    dock_w = ""
    if dock_key in docking_scores:
        dock_w = _format_docking_score(docking_scores[dock_key])

    if skip_distance:
        db_d = ""
    else:
        hxyz = hotspot_xyz
        pe_xyz = peptide_ca_coordinates(pep_path, pep_chain)
        if hxyz.shape[0]:
            dn = distance_centroid_nearest_hotspot(pe_xyz, hxyz)
        else:
            dn = None
        if dn is None:
            dn = distance_fallback_min_ca(pep_path, pep_chain, receptor_path, rec_chain)
        db_d = "" if dn is None else f"{dn:.3f}"

    mmg_str = ""
    if amber_ok and not skip_mmpbsa:
        lock_p = Path(tmp_root) / ".mmpbsa_serial.lock"
        min_schedule = ([mmpbsa_minimize, 0] if mmpbsa_minimize > 0 else [0])
        with _process_lock_file(lock_p):
            for attempt_min in min_schedule:
                wd = tempfile.mkdtemp(prefix=f"{pid}_", dir=tmp_root)
                wd_path = Path(wd)
                try:
                    _, _, nr, nl = write_receptor_ligand_pdbs(
                        receptor_path, pep_path, wd_path, rec_chain, pep_chain
                    )
                    errf = wd_path / "failed_mmpbsa.txt"
                    dg = compute_mmgbsa_single(
                        wd_path,
                        amber_paths,
                        nr,
                        nl,
                        max_minimize_cycles=attempt_min,
                        igb=mmpbsa_igb,
                        log_err=errf,
                    )
                    if dg is not None:
                        mmg_str = f"{dg:.4f}"
                        break
                finally:
                    if not os.environ.get("KEEP_MMPBSA_WORK"):
                        shutil.rmtree(wd_path, ignore_errors=True)

    return {
        "peptide_id": pid,
        "sequence": seq,
        "source": source,
        "predicted_structure_path": abs_pdb,
        "docking_score": dock_w,
        "mmgbsa_score": mmg_str,
        "distance_to_PD1_interface": db_d,
    }


def main() -> None:
    ap = argparse.ArgumentParser(description="由 cands/*.pdb 生成 candidate_peptides.csv（MMGBSA + 界面距离）")
    ap.add_argument("--cands-dir", type=Path, default=SCRIPT_DIR / "7OUN-PD-L1_target_A" / "cands")
    ap.add_argument("--out", type=Path, default=SCRIPT_DIR / "candidate_peptides.csv")
    ap.add_argument("--receptor-pdb", type=Path, default=SCRIPT_DIR.parent / "data" / "7OUN-PD-L1_target_A.pdb")
    ap.add_argument("--receptor-chain", default="A")
    ap.add_argument("--peptide-chain", default="P")
    ap.add_argument("--source", default="predicted")
    ap.add_argument(
        "--pd1-hotspot-residue-nums",
        default="default",
        help="PD-L1 热点残基 PDB 编号，如 54,113-120；default 用内置",
    )
    ap.add_argument("--skip-distance", action="store_true")
    ap.add_argument("--skip-mmpbsa", action="store_true")
    ap.add_argument(
        "--mmpbsa-igb",
        type=int,
        default=5,
        help="MMPBSA &gb igb（与 mbondi2 常用 5；8 需 mbondi3 等）",
    )
    ap.add_argument("--mmpbsa-minimize", type=int, default=200, help="sander 极小化 maxcyc；0 跳过")
    ap.add_argument("--jobs", type=int, default=1)
    ap.add_argument("--tmpdir", type=Path, default=SCRIPT_DIR / "_mmpbsa_work")
    ap.add_argument(
        "--docking-score-json",
        type=Path,
        default=SCRIPT_DIR / "7OUN-PD-L1_target_A" / "dg_separated_cands.json",
        help="对接分 JSON（键为 pep_XX.pdb，值为分数）；不存在则 docking_score 留空",
    )
    args = ap.parse_args()

    cands = args.cands_dir.expanduser().resolve()
    receptor = args.receptor_pdb.expanduser().resolve()
    if not cands.is_dir():
        raise FileNotFoundError(f"目录不存在: {cands}")
    if not receptor.is_file():
        raise FileNotFoundError(f"受体 PDB 不存在: {receptor}")

    pdb_files = sorted(cands.glob("*.pdb"), key=lambda p: peptide_index_from_name(p.stem))
    if not pdb_files:
        raise FileNotFoundError(f"未找到 PDB: {cands}/*.pdb")

    hotspot_nums = (
        DEFAULT_PD1_HOTSPOT_RESIDUES_NUMS
        if args.pd1_hotspot_residue_nums.strip().lower() == "default"
        else _parse_hotspot_nums(args.pd1_hotspot_residue_nums)
    )
    hot_xyz = hotspot_ca_coordinates(receptor, args.receptor_chain, hotspot_nums)
    print(f"热点 Cα 命中数: {hot_xyz.shape[0]} / 指定 {len(hotspot_nums)} 个残基编号")

    amber_paths: dict[str, str] = {}
    amber_ok = discover_amber(amber_paths) and not args.skip_mmpbsa
    if args.skip_mmpbsa:
        print("跳过 MM-GBSA")
    elif not amber_ok:
        print("WARN: 未找到 AmberTools（tleap / cpptraj / MMPBSA.py），跳过 MM-GBSA（建议: conda activate ambertools）")
    else:
        print("Amber: " + ", ".join(f"{k}={v}" for k, v in sorted(amber_paths.items())))

    tmp_root_base = args.tmpdir.expanduser().resolve()
    tmp_root_base.mkdir(parents=True, exist_ok=True)

    dock_json_path = args.docking_score_json.expanduser().resolve()
    docking_scores: dict[str, float] = {}
    if dock_json_path.is_file():
        docking_scores = load_docking_scores_json(dock_json_path)
        print(f"对接分 JSON: {dock_json_path}（{len(docking_scores)} 条）")

    rows: list[dict[str, Any]] = []

    if args.jobs <= 1:
        for p in pdb_files:
            rows.append(
                _proc_one_peptide(
                    str(p),
                    str(receptor),
                    args.receptor_chain,
                    args.peptide_chain,
                    args.source,
                    args.skip_distance,
                    args.skip_mmpbsa,
                    hot_xyz,
                    amber_ok,
                    amber_paths,
                    str(tmp_root_base),
                    args.mmpbsa_igb,
                    args.mmpbsa_minimize,
                    docking_scores,
                )
            )
    else:
        ex_args = (
            str(receptor),
            args.receptor_chain,
            args.peptide_chain,
            args.source,
            args.skip_distance,
            args.skip_mmpbsa,
            hot_xyz,
            amber_ok,
            amber_paths,
            str(tmp_root_base),
            args.mmpbsa_igb,
            args.mmpbsa_minimize,
            docking_scores,
        )
        with ProcessPoolExecutor(max_workers=args.jobs) as ex:
            futs = {ex.submit(_proc_one_peptide, str(pf), *ex_args): pf for pf in pdb_files}
            tmp: dict[Path, dict[str, Any]] = {}
            for fu in as_completed(futs):
                tmp[futs[fu]] = fu.result()
            rows = [tmp[p] for p in pdb_files]

    df = pd.DataFrame(rows)
    out = args.out.expanduser().resolve()
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)

    nonempty = df["mmgbsa_score"].fillna("").astype(str).str.strip().astype(bool).sum()
    dock_nonempty = df["docking_score"].fillna("").astype(str).str.strip().astype(bool).sum()
    print(
        f"已写入: {out}（{len(df)} 行）；对接分填入: {dock_nonempty}/{len(df)}；MM-GBSA 有效: {nonempty}/{len(df)}",
    )
    if not args.skip_mmpbsa and amber_ok and nonempty < len(df):
        miss = len(df) - int(nonempty)
        print(
            f"提示：有 {miss} 条肽未得到 ΔG — 常为 tleap/MMPBSA 对部分 PDB 报错；可用 "
            "`KEEP_MMPBSA_WORK=1` + `--tmpdir ...` 检查失败目录日志；或减少 `--mmpbsa-minimize`。",
        )


if __name__ == "__main__":
    main()

'''

conda activate ambertools
python /root/autodl-tmp/Peptide_3D/results/PDL1_peptide_virtual_cell/peptides/build_candidate_peptides_csv.py --jobs 1 --mmpbsa-minimize 0 --out candidate_peptides.csv

'''