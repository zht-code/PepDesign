''''
单卡或多卡生成
'''

# generate.py (multi-GPU, per-GPU worker processes)
import os
import glob
import sys
import numpy as np
import torch
import torch.nn.functional as F
import warnings
from multiprocessing import get_context
import math
import random
import tempfile

warnings.filterwarnings("ignore")
sys.path.append("/root/autodl-tmp/Peptide_3D")  # 项目根目录
# ===== 指定统一的临时目录 =====
TEMP_ROOT = "/root/autodl-tmp/tmp"
os.makedirs(TEMP_ROOT, exist_ok=True)

# 让所有库(含 tempfile、部分C库)都用这个目录
os.environ["TMPDIR"] = TEMP_ROOT
os.environ["TEMP"]   = TEMP_ROOT
os.environ["TMP"]    = TEMP_ROOT
import tempfile as _tf
_tf.tempdir = TEMP_ROOT

from models import ProteinPeptideModel

# 进度条
try:
    from tqdm import tqdm
    TQDM_AVAILABLE = True
except ImportError:
    TQDM_AVAILABLE = False
    def tqdm(x, **kwargs): return x

# —— OpenMM（老版本兼容：不调用 findMissingAtoms） ——
OPENMM_OK = True
try:
    from openmm import app, unit, Platform
    import openmm as mm
except Exception:
    OPENMM_OK = False

# —— 全原子构建依赖 ——
from Bio.PDB import PDBIO
try:
    import PeptideBuilder
    from Bio.PDB.Atom import Atom
    PB_OK = True
except Exception:
    PB_OK = False
# 几何初始化策略：'helix'（推荐）或 'mds'
INIT_GEOM_MODE = 'helix'
BIN_EDGES = np.linspace(2.0, 22.0, num=65)  # 64 bins
NUM_BINS  = len(BIN_EDGES) - 1

AA1_TO_AA3 = {
    "A":"ALA","R":"ARG","N":"ASN","D":"ASP","C":"CYS","E":"GLU","Q":"GLN","G":"GLY","H":"HIS",
    "I":"ILE","L":"LEU","K":"LYS","M":"MET","F":"PHE","P":"PRO","S":"SER","T":"THR","W":"TRP",
    "Y":"TYR","V":"VAL","U":"SEC","O":"PYL","B":"ASX","Z":"GLX","X":"UNK"
}

# ---------- 基础工具：读受体 CA、口袋中心、刚体摆放 ----------
def _load_ca_from_pdb(pdb_path: str, chain_filter: str | None = None) -> torch.Tensor:
    coords = []
    with open(pdb_path, "r") as f:
        for line in f:
            if not line.startswith("ATOM"):
                continue
            atom_name = line[12:16].strip()
            if atom_name != "CA":
                continue
            chain_id = line[21].strip() or " "
            if chain_filter is not None and chain_id != chain_filter:
                continue
            try:
                x = float(line[30:38]); y = float(line[38:46]); z = float(line[46:54])
                coords.append((x, y, z))
            except ValueError:
                continue
    if not coords:
        return torch.empty((0, 3), dtype=torch.float32)
    return torch.tensor(coords, dtype=torch.float32)

def _pocket_center_from_receptor(pdb_path: str,
                                 pocket_idx: list[int] | None = None,
                                 chain_id: str | None = None):
    rec_ca = _load_ca_from_pdb(pdb_path, chain_filter=chain_id)  # [Nr,3]
    if rec_ca.numel() == 0:
        raise RuntimeError(f"No CA atoms found in receptor PDB: {pdb_path}")
    if pocket_idx:
        valid_idx = [i for i in pocket_idx if 0 <= i < rec_ca.size(0)]
        center = rec_ca[valid_idx].mean(dim=0) if valid_idx else rec_ca.mean(dim=0)
    else:
        center = rec_ca.mean(dim=0)
    return rec_ca, center

def _rigid_place_near_pocket(peptide_xyz: torch.Tensor,
                             rec_ca: torch.Tensor,
                             pocket_center: torch.Tensor,
                             tries: int = 32,
                             margin: float = 2.5,
                             seed: int = 42) -> torch.Tensor:
    rng = random.Random(seed)
    dev = peptide_xyz.device
    X = peptide_xyz.detach().to(device=dev, dtype=torch.float32).clone()
    rec_ca = rec_ca.detach().to(device=dev, dtype=torch.float32)
    pocket_center = pocket_center.detach().to(device=dev, dtype=torch.float32)

    def rand_unit():
        phi = 2*math.pi*rng.random()
        cost = 2*rng.random()-1
        sint = math.sqrt(1-cost*cost)
        return torch.tensor([sint*math.cos(phi), sint*math.sin(phi), cost],
                            device=dev, dtype=torch.float32)

    offset = 4.0 * rand_unit()
    X = X - X.mean(dim=0, keepdim=True) + (pocket_center + offset)

    best_X = X
    best_penalty = float("inf")

    def random_rot_matrix():
        u1, u2, u3 = rng.random(), rng.random(), rng.random()
        theta = 2*math.pi*u1
        phi   = 2*math.pi*u2
        z     = u3
        r = math.sqrt(z)
        V = torch.tensor([math.cos(phi)*r, math.sin(phi)*r, math.sqrt(1-z)],
                         device=dev, dtype=torch.float32)
        H = torch.eye(3, device=dev, dtype=torch.float32) - 2*V[:, None] @ V[None, :]
        Rz = torch.tensor([[math.cos(theta), -math.sin(theta), 0.0],
                           [math.sin(theta),  math.cos(theta), 0.0],
                           [0.0,              0.0,             1.0]],
                          device=dev, dtype=torch.float32)
        return (Rz @ H)

    for _ in range(int(tries)):
        R = random_rot_matrix()
        Xr = (X - pocket_center) @ R.T + pocket_center
        d = torch.cdist(Xr, rec_ca)
        penalty = float((d < margin).sum()) * 10.0 + d.median().item()
        if penalty < best_penalty:
            best_penalty = penalty
            best_X = Xr

    return best_X

def _distogram_to_coords(struct_logits_llll: torch.Tensor, bin_edges: np.ndarray, target_ca=3.8) -> torch.Tensor:
    probs = F.softmax(struct_logits_llll, dim=-1)[0]   # [L,L,B]
    edges = torch.as_tensor(bin_edges, dtype=probs.dtype, device=probs.device)
    centers = 0.5 * (edges[:-1] + edges[1:])           # [B]
    D = (probs * centers).sum(dim=-1)                  # [L,L]
    D = 0.5 * (D + D.T)
    D.fill_diagonal_(0)

    L = D.size(0)
    J = torch.eye(L, device=D.device) - torch.ones((L, L), device=D.device) / L
    B = -0.5 * J @ (D ** 2) @ J
    w, v = torch.linalg.eigh(B)
    w = torch.clamp(w, min=0)
    idx = torch.argsort(w, descending=True)
    w, v = w[idx], v[:, idx]
    X = v[:, :3] * torch.sqrt(w[:3]).unsqueeze(0)

    if L > 1:
        adj = X[1:] - X[:-1]
        mean_adj = torch.norm(adj, dim=1).mean()
        if mean_adj > 0:
            X = X * (target_ca / mean_adj)

    X = X - X.mean(dim=0, keepdim=True)
    return X  # [L,3]
def _add_alpha_phi_psi_restraints(topology, system, k_kcal_per_mol=0.5):
    """
    给每个内部残基添加弱的 φ/ψ 周期性约束，中心在 α-螺旋(-57°, -47°)。
    这个强度很小，只是“偏好”，不会把构象死锁。
    """
    PHI0 = -57.0 * math.pi/180.0
    PSI0 = -47.0 * math.pi/180.0
    k    = k_kcal_per_mol * unit.kilocalorie_per_mole

    # 周期性势: k * (1 - cos(theta - theta0))
    phi_force = mm.CustomTorsionForce("k*(1 - cos(theta - theta0))")
    phi_force.addPerTorsionParameter("k")
    phi_force.addPerTorsionParameter("theta0")

    psi_force = mm.CustomTorsionForce("k*(1 - cos(theta - theta0))")
    psi_force.addPerTorsionParameter("k")
    psi_force.addPerTorsionParameter("theta0")

    residues = list(topology.residues())
    # 建 per-residue → {atomName: atomIndex} 的索引
    atom_by_resname = []
    for res in residues:
        d = {}
        for atom in res.atoms():
            d[atom.name] = atom.index
        atom_by_resname.append(d)

    # φ(i) = C(i-1) - N(i) - CA(i) - C(i), i = 1..n-1
    # ψ(i) = N(i) - CA(i) - C(i) - N(i+1), i = 0..n-2
    n = len(residues)
    for i in range(1, n):
        # phi for residue i
        prev = atom_by_resname[i-1]
        cur  = atom_by_resname[i]
        if all(name in prev for name in ("C",)) and all(name in cur for name in ("N","CA","C")):
            phi_force.addTorsion(prev["C"], cur["N"], cur["CA"], cur["C"], [k, PHI0])
    for i in range(0, n-1):
        cur  = atom_by_resname[i]
        nxt  = atom_by_resname[i+1]
        if all(name in cur for name in ("N","CA","C")) and all(name in nxt for name in ("N",)):
            psi_force.addTorsion(cur["N"], cur["CA"], cur["C"], nxt["N"], [k, PSI0])

    system.addForce(phi_force)
    system.addForce(psi_force)
def _build_fullatom_peptide_helix(seq: str):
    """
    用 PeptideBuilder 构建理想 α-螺旋。PeptideBuilder 需要角度单位为“度(°)”。
    """
    if not PB_OK:
        raise RuntimeError("需要 peptidebuilder 和 biopython: pip install peptidebuilder biopython")

    # 角度（单位：度）
    PHI_H = -57.0
    PSI_H = -47.0
    OMG   = 180.0  # 反式

    struct = PeptideBuilder.initialize_res(seq[0])
    for aa in seq[1:]:
        # 注意：add_residue 的 phi/psi/omega 是“度”
        struct = PeptideBuilder.add_residue(struct, aa, PHI_H, PSI_H, OMG)

    # 给最后一个残基设置 ψ（有的版本没有 set_psi，就忽略）
    try:
        PeptideBuilder.set_psi(struct, len(seq) - 1, PSI_H)
    except Exception:
        pass

    for model in struct:
        for chain in model:
            chain.id = 'P'
    return struct

# ---------- 最小改造核心：全原子重建 + (兼容老 OpenMM) 最小化 ----------
def _kabsch_align(src_ca: np.ndarray, dst_ca: np.ndarray):
    src = src_ca - src_ca.mean(0, keepdims=True)
    dst = dst_ca - dst_ca.mean(0, keepdims=True)
    H = src.T @ dst
    U, S, Vt = np.linalg.svd(H)
    R = Vt.T @ U.T
    if np.linalg.det(R) < 0:
        Vt[-1, :] *= -1
        R = Vt.T @ U.T
    t = dst.mean(0) - R @ src.mean(0)
    return R, t

# —— 支持的氨基酸与非常规映射 —— 
VALID_AA = set("ACDEFGHIKLMNPQRSTVWY")
AA_REMAP = {
    "B": "N",  # Asx -> Asn
    "Z": "Q",  # Glx -> Gln
    "X": "A",  # Unknown -> Ala
    "U": "C",  # Sec  -> Cys
    "O": "K",  # Pyl  -> Lys
    "J": "L",  # Leu/Ile ambiguous -> Leu
    "*": "A",  # stop -> Ala
    "-": "A",  # gap  -> Ala
}

def sanitize_sequence(seq: str) -> str:
    s = []
    for ch in seq:
        c = ch.upper()
        if c in VALID_AA:
            s.append(c)
        elif c in AA_REMAP:
            s.append(AA_REMAP[c])
        else:
            s.append("A")
    return "".join(s)

def _build_fullatom_peptide(seq: str):
    if not PB_OK:
        raise RuntimeError("PeptideBuilder / Biopython 未安装，无法全原子重建。请 pip install peptidebuilder biopython")
    struct = PeptideBuilder.initialize_res(seq[0])
    for aa in seq[1:]:
        struct = PeptideBuilder.add_residue(struct, aa)
    # 统一链 ID 为 'P'
    for model in struct:
        for chain in model:
            chain.id = 'P'
    return struct  # Bio.PDB Structure

def _set_structure_coords_by_ca(struct, target_ca_xyz: np.ndarray):
    ca_atoms = [a for a in struct.get_atoms() if a.get_name() == "CA"]
    src_ca = np.array([a.get_coord() for a in ca_atoms], dtype=np.float64)
    if len(src_ca) != target_ca_xyz.shape[0]:
        raise RuntimeError(f"CA 数不匹配: ideal={len(src_ca)} vs target={target_ca_xyz.shape[0]}")
    R, t = _kabsch_align(src_ca, target_ca_xyz)
    for atom in struct.get_atoms():
        x = atom.get_coord().astype(np.float64)
        atom.set_coord((R @ x) + t)
    return struct

def _save_structure(struct, out_pdb: str):
    os.makedirs(os.path.dirname(out_pdb), exist_ok=True)
    io = PDBIO()
    io.set_structure(struct)
    io.save(out_pdb)

def _ensure_c_terminus_oxt(in_pdb: str, out_pdb: str, chain_id: str = "P"):
    """
    如果 C 端缺 OXT，则用一个合理的初始几何添加 OXT（C=O 对侧 1.25Å）。
    这是为了兼容老 OpenMM 只用 addHydrogens() 也能通过模板检查。
    """
    from Bio.PDB import PDBParser
    parser = PDBParser(QUIET=True)
    struct = parser.get_structure("pep", in_pdb)
    model = next(struct.get_models())
    chain = None
    for ch in model:
        chain = ch
        if chain.id == chain_id:
            break
    if chain is None:
        chain = next(model.get_chains())

    residues = [r for r in chain.get_residues() if r.id[0] == " "]
    if not residues:
        # nothing to do, just copy
        import shutil; shutil.copyfile(in_pdb, out_pdb); return

    last = residues[-1]
    atom_names = {a.get_name() for a in last.get_atoms()}
    if "OXT" in atom_names:
        import shutil; shutil.copyfile(in_pdb, out_pdb); return

    # 需要 C 和 O 原子
    if ("C" not in atom_names) or ("O" not in atom_names):
        import shutil; shutil.copyfile(in_pdb, out_pdb); return

    C = last["C"].get_vector().get_array()
    O = last["O"].get_vector().get_array()
    # 方向：从 C 指向 O 的反向，长度 ~1.25 Å
    dir_vec = C - O
    norm = np.linalg.norm(dir_vec) + 1e-8
    oxt_pos = C + (1.25 * dir_vec / norm)

    # 构造并加入 OXT 原子
    new_atom = Atom(
        name="OXT",
        coord=oxt_pos,
        bfactor=0.0,
        occupancy=1.0,
        altloc=" ",
        fullname=" OXT",
        serial_number=0,
        element="O"
    )
    last.add(new_atom)

    _save_structure(struct, out_pdb)

def _openmm_minimize_pdb_simple(in_pdb: str, out_pdb: str, max_steps=2000, ph: float = 7.0, add_alpha_restraints=True):
    """
    兼容老 OpenMM：不调用 findMissingAtoms/addMissingAtoms。
    手动补 OXT（若无）-> addHydrogens -> (可选)弱 φ/ψ 约束 -> 最小化。
    """
    tmp_pdb = in_pdb + ".oxt.pdb"
    _ensure_c_terminus_oxt(in_pdb, tmp_pdb)

    pdb = app.PDBFile(tmp_pdb)
    forcefield = app.ForceField('amber14/protein.ff14SB.xml', 'amber14/tip3pfb.xml')

    modeller = app.Modeller(pdb.topology, pdb.positions)
    modeller.addHydrogens(forcefield, pH=ph)

    system = forcefield.createSystem(
        modeller.topology,
        nonbondedMethod=app.NoCutoff,
        constraints=app.HBonds,
    )

    if add_alpha_restraints:
        _add_alpha_phi_psi_restraints(modeller.topology, system, k_kcal_per_mol=0.1)

    integrator = mm.LangevinMiddleIntegrator(
        300*unit.kelvin, 1.0/unit.picosecond, 0.004*unit.picoseconds
    )
    try:
        platform = Platform.getPlatformByName('CUDA')
    except Exception:
        platform = Platform.getPlatformByName('CPU')

    sim = app.Simulation(modeller.topology, system, integrator, platform)
    sim.context.setPositions(modeller.positions)
    sim.minimizeEnergy(maxIterations=int(max_steps))
    state = sim.context.getState(getPositions=True)
    pos = state.getPositions()
    with open(out_pdb, 'w') as f:
        app.PDBFile.writeFile(modeller.topology, pos, f, keepIds=True)

    try:
        os.remove(tmp_pdb)
    except Exception:
        pass
# === NEW: 给 CA 加位置约束，锁住绝对坐标 ===
def _add_positional_restraints_on_CA(topology, system, ca_xyz_nm, k_kcal_per_mol_per_A2=2.0):
    """
    对每个 CA 加一个 0.5*k*((x-x0)^2+(y-y0)^2+(z-z0)^2) 的谐和外势，锁住“刚体摆放后”的位置。
    ca_xyz_nm: list[(res_index, x_nm, y_nm, z_nm)]  —— 注意坐标单位是 nm
    """
    from openmm import CustomExternalForce, unit
    k = k_kcal_per_mol_per_A2 * unit.kilocalorie_per_mole / (unit.angstrom**2)
    k = k * (unit.angstrom**2) / (unit.nanometer**2)  # 转换到 /nm^2

    force = CustomExternalForce("0.5*k*((x-x0)^2 + (y-y0)^2 + (z-z0)^2)")
    force.addPerParticleParameter("k")
    force.addPerParticleParameter("x0")
    force.addPerParticleParameter("y0")
    force.addPerParticleParameter("z0")

    residues = list(topology.residues())
    # 建 per-residue → {atom_name: atom_index} 映射
    res_atoms = []
    for res in residues:
        d = {}
        for a in res.atoms():
            d[a.name] = a.index
        res_atoms.append(d)

    for res_idx, x_nm, y_nm, z_nm in ca_xyz_nm:
        if 0 <= res_idx < len(res_atoms) and "CA" in res_atoms[res_idx]:
            force.addParticle(int(res_atoms[res_idx]["CA"]), [k, x_nm, y_nm, z_nm])

    system.addForce(force)

# === NEW: 用 CA 位置约束 + 弱 α-螺旋约束 的最小化（不会漂走） ===
def _openmm_minimize_with_hard_frame(in_pdb: str, out_pdb: str,
                                     placed_ca_xyz_A: np.ndarray,
                                     ca_k=2.0, max_steps=2000, ph=7.0,
                                     add_alpha_restraints=True):
    # 1) 先确保 C 端有 OXT（很多构建器默认不写 OXT）
    tmp_pdb = in_pdb + ".oxt.pdb"
    _ensure_c_terminus_oxt(in_pdb, tmp_pdb)

    # 2) 再走 OpenMM
    pdb = app.PDBFile(tmp_pdb)
    ff  = app.ForceField('amber14/protein.ff14SB.xml', 'amber14/tip3pfb.xml')
    modeller = app.Modeller(pdb.topology, pdb.positions)
    modeller.addHydrogens(ff, pH=ph)

    system = ff.createSystem(modeller.topology,
                             nonbondedMethod=app.NoCutoff,
                             constraints=app.HBonds)

    if add_alpha_restraints:
        _add_alpha_phi_psi_restraints(modeller.topology, system, k_kcal_per_mol=0.2)

    # 3) 给 CA 加位置约束，锁住“刚体摆放后”的坐标（Å→nm）
    ca_nm = [(i, x*0.1, y*0.1, z*0.1) for i, (x, y, z) in enumerate(placed_ca_xyz_A)]
    _add_positional_restraints_on_CA(modeller.topology, system, ca_nm, k_kcal_per_mol_per_A2=ca_k)

    integrator = mm.LangevinMiddleIntegrator(300*unit.kelvin, 1.0/unit.picosecond, 0.004*unit.picoseconds)
    try:
        platform = Platform.getPlatformByName('CUDA')
    except Exception:
        platform = Platform.getPlatformByName('CPU')

    sim = app.Simulation(modeller.topology, system, integrator, platform)
    sim.context.setPositions(modeller.positions)
    sim.minimizeEnergy(maxIterations=int(max_steps))
    state = sim.context.getState(getPositions=True)
    pos = state.getPositions()
    with open(out_pdb, 'w') as f:
        app.PDBFile.writeFile(modeller.topology, pos, f, keepIds=True)

    # 清理临时文件
    try:
        os.remove(tmp_pdb)
    except Exception:
        pass


# === NEW(可选): 输出复合物，便于肉眼自检 ===
def _write_complex(receptor_pdb: str, peptide_pdb: str, out_pdb: str,
                   rec_chain='A', pep_chain='P'):
    def _lines(p, new_chain=None):
        with open(p) as f:
            for line in f:
                if line.startswith(("ATOM", "HETATM")):
                    if new_chain is not None:
                        line = line[:21] + f"{new_chain:1s}" + line[22:]
                    yield line
    with open(out_pdb, "w") as w:
        for ln in _lines(receptor_pdb, rec_chain): w.write(ln)
        for ln in _lines(peptide_pdb,  pep_chain): w.write(ln)
        w.write("END\n")



# ---------- 其他工具 ----------
def find_protein_pdbs(root: str):
    pairs = []
    for d in sorted([p for p in glob.glob(os.path.join(root, "*")) if os.path.isdir(p)]):
        cands = sorted(glob.glob(os.path.join(d, "*.pdb")))
        if len(cands) > 0:
            pairs.append((d, cands[0]))
    if len(pairs) == 0:
        for pdb in sorted(glob.glob(os.path.join(root, "*.pdb"))):
            pairs.append((os.path.dirname(pdb), pdb))
    return pairs

# ---------- 简单 CA 输出（仅作为 fallback） ----------
def _write_pdb_from_coords(sequence: str, coords: torch.Tensor, out_path: str, chain_id: str = "P") -> None:
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    lines = []
    for i, (aa, xyz) in enumerate(zip(sequence, coords.tolist()), start=1):
        res3 = AA1_TO_AA3.get(aa.upper(), "UNK")
        x, y, z = xyz
        lines.append(f"ATOM  {i:5d}  CA  {res3:>3} {chain_id}{i:4d}    {x:8.3f}{y:8.3f}{z:8.3f}  1.00  0.00           C")
    lines.append("TER")
    lines.append("END")
    with open(out_path, "w") as f:
        f.write("\n".join(lines))


# ---------- 评分函数 ----------
def _score_interface(model, seq: str, enc: torch.Tensor, mask: torch.Tensor) -> float:
    """
    返回一条序列的界面分数（越大越好）。
    做法：forward 一次拿 interface_logits -> sigmoid 概率 ->
         对每个多肽残基取 “与任一受体残基的最大概率”，再对多肽维取平均。
         这个指标 = 期望“有明显接触的位点”覆盖率。
    """
    device = enc.device
    toks = model.esmc._tokenize([seq]).to(device)  # [1, L]
    out  = model.esmc.forward(
        sequence_tokens=toks,
        encoder_embeddings=enc,
        cross_attention_mask=mask,
    )
    if out.interface_logits is None:
        return -1e9  # 防御：不该发生
    prob = torch.sigmoid(out.interface_logits)     # [1, L_pep, L_rec]
    # 每个多肽残基，取对任何受体残基的最大接触概率
    per_pep_max = prob.amax(dim=-1)                # [1, L_pep]
    # 去掉 BOS/EOS（若你的 tokenizer 会在 _tokenize 里加 special，下面两行保护一下）
    # 这里假设首尾各 1 个特殊符号：
    if per_pep_max.size(1) > 2:
        per_pep_max = per_pep_max[:, 1:-1]
    score = per_pep_max.mean().item()
    return float(score)

# ------------------ 每个 GPU 的工作进程 ------------------
def worker(gpu_id: int, prot_shard: list, cfg: dict):
    """
    gpu_id: 当前进程使用的 GPU 序号
    prot_shard: 该进程要处理的 [(prot_dir, pdb_path), ...]
    cfg: {
        ckpt_path, num_per_protein, top_k, max_len, temperature, num_gpus
    }
    依赖的全局/工具：
      - INIT_GEOM_MODE: 'helix' 或 'mds'
      - sanitize_sequence, _build_fullatom_peptide_helix, _build_fullatom_peptide
      - _kabsch_align, _rigid_place_near_pocket, _openmm_minimize_pdb_simple
      - OPENMM_OK, TQDM_AVAILABLE
    """
    # 设备设置
    os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(str(i) for i in range(cfg["num_gpus"]))
    device = torch.device(f"cuda:{gpu_id}" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        torch.cuda.set_device(device)
        torch.backends.cudnn.benchmark = True

    # 模型加载（每卡一份）
    model = ProteinPeptideModel(device).to(device)
    state = torch.load(cfg["ckpt_path"], map_location=device)
    state_dict = state.get("state_dict", state)
    model.load_state_dict(state_dict, strict=False)
    model.eval()

    it = tqdm(
        prot_shard, desc=f"GPU{gpu_id}", unit="protein",
        position=gpu_id, leave=True
    ) if TQDM_AVAILABLE else prot_shard

    with torch.no_grad():
        for prot_dir, pdb_path in it:
            prot_name = os.path.basename(prot_dir)
            out_dir = os.path.join(prot_dir, "multi_cands1")
            os.makedirs(out_dir, exist_ok=True)

            # 1) 先“过采样”一批候选（例如 3 倍），再重排，最后保留前 N
            oversample = cfg.get("oversample_factor", 3)          # ← 可在 main 的 cfg 里改
            num_keep   = cfg["num_per_protein"]
            num_cand   = max(num_keep, num_keep * oversample)

            seqs = model.generate_sequences_from_protein(
                pdb_path,
                num_samples=num_cand,
                top_k=cfg["top_k"],
                max_len=cfg["max_len"],
                temperature=cfg["temperature"],
            )

            # 2) 计算 interface 分数并排序
            enc, mask = model.encode_protein_from_pdb(pdb_path)
            scored = []
            for s in seqs:
                try:
                    sc = _score_interface(model, s, enc, mask)
                except Exception as e:
                    sc = -1e9
                scored.append((sc, s))

            scored.sort(key=lambda x: x[0], reverse=True)
            seqs = [s for (_, s) in scored[:num_keep]]            # 仅保留前 N 条
            # 受体 CA 与口袋中心（一次）
            rec_ca, pocket_center = _pocket_center_from_receptor(pdb_path, pocket_idx=None)

            it_inner = tqdm(
                seqs, desc=f"GPU{gpu_id}:{prot_name}",
                unit="pdb", position=gpu_id, leave=False, total=len(seqs)
            ) if TQDM_AVAILABLE else seqs

            for i, seq in enumerate(it_inner, start=1):
                # —— 统一长度与序列净化（与结构头的 max 长度一致，通常 64）——
                Lcap = 64
                Lwant = min(len(seq), Lcap)
                seq_used = sanitize_sequence(seq[:Lwant])

                # 分支：几何初始化策略
                if INIT_GEOM_MODE == 'helix':
                    # A) 直接构建 α 螺旋的全原子骨架
                    struct = _build_fullatom_peptide_helix(seq_used)

                    # 从 struct 中取 CA，刚体放置到口袋附近（只平移/旋转，不改变二级结构）
                    ca_list = []
                    for atom in struct.get_atoms():
                        if atom.get_name() == "CA":
                            ca_list.append(atom.get_coord())
                    ca_xyz = torch.tensor(np.array(ca_list, dtype=np.float32), device=rec_ca.device)  # [L,3]

                    placed_ca = _rigid_place_near_pocket(
                        ca_xyz, rec_ca, pocket_center, tries=48, margin=2.5
                    )
                    placed_ca_numpy = placed_ca.cpu().numpy()  # <<< NEW: 后面最小化要用

                    # 计算刚体变换并应用到全原子
                    src = ca_xyz.cpu().numpy()
                    dst = placed_ca_numpy
                    R, t = _kabsch_align(src, dst)
                    for atom in struct.get_atoms():
                        x = atom.get_coord().astype(np.float64)
                        atom.set_coord((R @ x) + t)

                else:
                    # B) 维持你的 MDS 路线（便于对照；它会偏直）
                    struct_logits = model.predict_struct_logits_for_sequence(
                        seq_used, encoder_embeddings=enc, cross_attention_mask=mask, distogram_len=64
                    )  # -> [1,L,L,NUM_BINS]
                    coords = _distogram_to_coords(...)
                    placed = _rigid_place_near_pocket(coords, rec_ca, pocket_center, ...)
                    placed_ca_numpy = placed.cpu().numpy()      # <<< NEW

                    struct = _build_fullatom_peptide(seq_used)
                    struct = _set_structure_coords_by_ca(struct, placed_ca_numpy)

                # —— 写出 & 最小化（带弱 α-螺旋 φ/ψ 约束）——
                final_pdb = os.path.join(out_dir, f"pep_{i:02d}.pdb")

                # 用临时文件存 raw，自动清理，不在 cands/ 里留下 raw
                with tempfile.TemporaryDirectory(dir=TEMP_ROOT) as tdir:
                    tmp_raw = os.path.join(tdir, "pep_raw.pdb")
                    _save_structure(struct, tmp_raw)

                    if OPENMM_OK:
                        try:
                            _openmm_minimize_with_hard_frame(
                                in_pdb=tmp_raw,
                                out_pdb=final_pdb,
                                placed_ca_xyz_A=placed_ca_numpy,
                                ca_k=2.0,
                                max_steps=2000,
                                ph=7.0,
                                add_alpha_restraints=True
                            )
                        except Exception as e_mm:
                            warn = f"[WARN] OpenMM 最小化失败：{e_mm}。输出未最小化（直接写 final）。"
                            (tqdm.write(warn) if TQDM_AVAILABLE else print(warn))
                            # 失败兜底：直接把当前 struct 写到 final_pdb
                            _save_structure(struct, final_pdb)
                    else:
                        warn = "[WARN] OpenMM 未安装，输出未最小化构象（直接写 final）。"
                        (tqdm.write(warn) if TQDM_AVAILABLE else print(warn))
                        _save_structure(struct, final_pdb)

            # 每蛋白完成提示
            if TQDM_AVAILABLE:
                tqdm.write(f"[GPU{gpu_id}] {prot_name} -> {len(seqs)} peptides saved to {out_dir}")
            else:
                print(f"[GPU{gpu_id}] {prot_name} -> {len(seqs)} peptides saved to {out_dir}")


# ------------------ 主控 ------------------
def find_protein_pdbs(root: str):
    pairs = []
    for d in sorted([p for p in glob.glob(os.path.join(root, "*")) if os.path.isdir(p)]):
        cands = sorted(glob.glob(os.path.join(d, "*.pdb")))
        if len(cands) > 0:
            pairs.append((d, cands[0]))
    if len(pairs) == 0:
        for pdb in sorted(glob.glob(os.path.join(root, "*.pdb"))):
            pairs.append((os.path.dirname(pdb), pdb))
    return pairs

def main():
    # ======== 配置区域：按需修改 ========
    train_root = "/root/autodl-tmp/PPDbench"
    ckpt_path  = "/root/autodl-tmp/Peptide_3D/logs_Ranger_dpo_multi/policy_dpo_multi_epoch5_loss_0.6073.pth"
    num_per_protein = 10
    top_k = 12            # or 12
    max_len = 30
    temperature = 1.0
    want_gpus = 1           # 想用多少 GPU（实际会被可用数限制）
    # ===================================

    prot_list = find_protein_pdbs(train_root)
    total = len(prot_list)
    print(f"Found {total} proteins to generate.")

    avail = torch.cuda.device_count()
    if avail == 0:
        print("No CUDA device found; running on CPU with a single process.")
        shards = [prot_list]
        world_size = 1
    else:
        world_size = min(want_gpus, avail)
        indices = np.array_split(np.arange(total), world_size)
        shards = [[prot_list[i] for i in idx.tolist()] for idx in indices]

    cfg = dict(
        ckpt_path=ckpt_path,
        num_per_protein=num_per_protein,
        top_k=top_k,
        max_len=max_len,
        temperature=temperature,
        num_gpus=world_size,
        oversample_factor=3,          # ← 过采样倍数（建议 2~5）
    )

    if world_size == 1:
        worker(0, shards[0], cfg)
    else:
        ctx = get_context("spawn")
        procs = []
        for rank in range(world_size):
            p = ctx.Process(target=worker, args=(rank, shards[rank], cfg), daemon=False)
            p.start()
            procs.append(p)
        for p in procs:
            p.join()

if __name__ == "__main__":
    main()


