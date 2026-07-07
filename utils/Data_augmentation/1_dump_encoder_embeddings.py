import os
import glob
import sys
import warnings
from multiprocessing import get_context

import numpy as np
import torch

warnings.filterwarnings("ignore")

# 进度条
try:
    from tqdm import tqdm
    TQDM_AVAILABLE = True
except ImportError:
    TQDM_AVAILABLE = False

    def tqdm(x, **kwargs):
        return x

# ================== 关键新增：对 ESM MultiHeadAttention 做 monkey-patch ==================
import math
import functools
import einops
import torch.nn.functional as F

# 项目根目录
sys.path.append("/root/autodl-tmp/Peptide_3D")

# 先导入 esm 的 MultiHeadAttention，然后给它改 forward
from esm.layers.attention import MultiHeadAttention


def mha_forward_with_save_attn(self, x, seq_id):
    """
    替换原始 MultiHeadAttention.forward：
    - 接口不变：输入 (x, seq_id)，输出仍然是 [B, L, D]
    - 多了两个属性：
        self.save_attn: bool，控制是否保存注意力
        self.last_attn: Tensor | None，保存最近一次的 [B, H, L, L] 注意力
    """
    # 确保属性存在
    if not hasattr(self, "save_attn"):
        self.save_attn = False
    if not hasattr(self, "last_attn"):
        self.last_attn = None

    qkv_BLD3 = self.layernorm_qkv(x)
    query_BLD, key_BLD, value_BLD = torch.chunk(qkv_BLD3, 3, dim=-1)
    query_BLD, key_BLD = (
        self.q_ln(query_BLD).to(query_BLD.dtype),
        self.k_ln(key_BLD).to(query_BLD.dtype),
    )
    query_BLD, key_BLD = self._apply_rotary(query_BLD, key_BLD)

    reshaper = functools.partial(
        einops.rearrange, pattern="b s (h d) -> b h s d", h=self.n_heads
    )
    query_BHLD, key_BHLD, value_BHLD = map(
        reshaper, (query_BLD, key_BLD, value_BLD)
    )

    if self.save_attn:
        # print 一下确认 forward 确实进来了
        # 你要是输出太多，可以注释掉
        # print("[DEBUG] Patched MultiHeadAttention.forward called, save_attn=True, x.shape=",
        #       x.shape, "device=", x.device)

        # seq_id -> mask
        if seq_id is not None:
            # 原实现：同 seq_id 才允许互相注意
            mask_BLL = seq_id.unsqueeze(-1) == seq_id.unsqueeze(-2)  # [B, L, L]
            mask_BHLL = mask_BLL.unsqueeze(1)                        # [B, H, L, L]
        else:
            mask_BHLL = None

        # 手写 scaled dot-product attention，拿到 attn 权重
        d = query_BHLD.size(-1)
        scores = torch.matmul(
            query_BHLD, key_BHLD.transpose(-2, -1)
        ) / math.sqrt(d)            # [B, H, L, L]

        if mask_BHLL is not None:
            scores = scores.masked_fill(~mask_BHLL, float("-inf"))

        attn = torch.softmax(scores, dim=-1)         # [B, H, L, L]
        context_BHLD = torch.matmul(attn, value_BHLD)  # [B, H, L, D]

        # 记录注意力权重
        self.last_attn = attn.detach()

    else:
        # 原来的高性能实现（行为等价，只是看不到权重）
        if seq_id is not None:
            mask_BLL = seq_id.unsqueeze(-1) == seq_id.unsqueeze(-2)
            mask_BHLL = mask_BLL.unsqueeze(1)

            context_BHLD = torch.nn.functional.scaled_dot_product_attention(
                query_BHLD, key_BHLD, value_BHLD, attn_mask=mask_BHLL
            )
        else:
            context_BHLD = torch.nn.functional.scaled_dot_product_attention(
                query_BHLD, key_BHLD, value_BHLD
            )

    context_BLD = einops.rearrange(context_BHLD, "b h s d -> b s (h d)")
    return self.out_proj(context_BLD)


# 真·猴子补丁：把 esm.layers.attention.MultiHeadAttention.forward 换成上面这个
MultiHeadAttention.forward = mha_forward_with_save_attn
# ======================================================================
from models import ProteinPeptideModel


def find_protein_pdbs(root: str):
    """
    在 train_root 下寻找受体 PDB：
    目录结构假设为：
        root/
          1A1M/receptor.pdb
          2ABC/receptor.pdb
          ...
    返回列表: [(prot_dir, pdb_path), ...]
    其中 prot_dir 是类似 /root/autodl-tmp/train_data/1A1M，
    pdb_path 是 /root/autodl-tmp/train_data/1A1M/receptor.pdb
    """
    pairs = []
    root = os.path.abspath(root)

    # 只枚举第一层子目录
    for name in sorted(os.listdir(root)):
        d = os.path.join(root, name)
        if not os.path.isdir(d):
            continue
        rec_pdb = os.path.join(d, "receptor.pdb")
        if os.path.isfile(rec_pdb):
            pairs.append((d, rec_pdb))

    if len(pairs) == 0:
        print(f"[WARN] 在 {root} 下没有找到任何 receptor.pdb，检查目录结构是否正确。")

    return pairs


def _register_last_self_attn_hook(model: torch.nn.Module):
    attn_state = {"tensor": None}

    # 精确锁定 esm3 最后一层 attn 模块
    try:
        esm3 = model.esm3_model
        last_block = esm3.transformer.blocks[-1]
        target_module = last_block.attn
        print("[INFO] hook on esm3_model.transformer.blocks[-1].attn")
    except Exception as e:
        print(f"[ERROR] 找最后一层 attn 失败: {e}")
        return None, attn_state

    # 打开保存注意力的开关
    target_module.save_attn = True

    def hook(m, _input, _output):
        # 直接从模块属性里拿 last_attn
        attn_state["tensor"] = getattr(m, "last_attn", None)

    handle = target_module.register_forward_hook(hook)
    return handle, attn_state




def worker(gpu_id: int, prot_shard, cfg: dict):
    """
    每个 GPU / 进程的工作函数：
    - 加载模型和权重
    - 在 encoder 最后一层 self-attention 上注册 hook
    - 对分配到的蛋白逐个 encode，并把 embedding+注意力保存成 pt 文件
    """
    # 设备
    if torch.cuda.is_available():
        device = torch.device(f"cuda:{gpu_id}")
        torch.cuda.set_device(device)
    else:
        device = torch.device("cpu")
    print(f"[Worker {gpu_id}] using device: {device}")

    save_root = cfg["save_root"]
    os.makedirs(save_root, exist_ok=True)  # 确保保存目录存在

    # 加载模型
    model = ProteinPeptideModel(device).to(device)
    state = torch.load(cfg["ckpt_path"], map_location=device)
    state_dict = state.get("state_dict", state)
    model.load_state_dict(state_dict, strict=False)
    model.eval()


    last_attn_mod = model.esm3_model.transformer.blocks[-1].attn
    last_attn_mod.save_attn = True
    iterator = (
        tqdm(
            prot_shard,
            desc=f"GPU{gpu_id}",
            unit="protein",
            position=gpu_id,
            leave=True,
        )
        if TQDM_AVAILABLE
        else prot_shard
    )

    with torch.no_grad():
        for prot_dir, pdb_path in iterator:
            # 这里的 prot_dir 类似 /root/autodl-tmp/train_data/1A1M
            prot_id = os.path.basename(prot_dir)  # 这里就是 1A1M

            # 统一保存到 save_root 下：/.../train_data_pt/1A1M.pt
            out_path = os.path.join(save_root, f"{prot_id}.pt")

            # 每个蛋白前先清空一次缓存
            last_attn_mod.last_attn = None
        
            # 调用现有的 encode_protein_from_pdb
            out = model.last_attn_from_pdb(pdb_path)


            # 解析返回（默认假设返回 (enc, mask)）
            enc = None
            mask = None
            if isinstance(out, (list, tuple)) and len(out) >= 2:
                enc, mask = out[0], out[1]
            else:
                enc = out

            if enc is None:
                print(f"[WARN][{prot_id}] 无法获取 encoder embedding，跳过。")
                continue

            emb_cpu = enc.detach().cpu()
            attn_cpu = None
            if last_attn_mod.last_attn is not None:
                attn_cpu = last_attn_mod.last_attn.cpu()


            data = {
                "protein_id": prot_id,
                "embedding": emb_cpu,  # 编码器输出（一般是最后 block 的 embedding）
            }
            if mask is not None:
                try:
                    data["mask"] = mask.detach().cpu()
                except Exception:
                    data["mask"] = mask
            if attn_cpu is not None:
                data["last_block_attn"] = attn_cpu

            torch.save(data, out_path)

            if TQDM_AVAILABLE:
                tqdm.write(f"[GPU{gpu_id}] Saved: {out_path}")
            else:
                print(f"[GPU{gpu_id}] Saved: {out_path}")




def main():
    # ======== 配置区域：按需修改 ========
    train_root = "/root/autodl-tmp/train_data"
    ckpt_path = (
        "/root/autodl-tmp/Peptide_3D/"
        "logs_Ranger_no_DPO/best_model_epoch_72_loss_2.0048.pth"
    )

    # 你希望 pt 文件统一保存到的目录
    save_root = "/root/autodl-tmp/Peptide_3D/data/train_data_pt"
    os.makedirs(save_root, exist_ok=True)

    want_gpus = 1
    # ===================================

    prot_list = find_protein_pdbs(train_root)
    total = len(prot_list)
    print(f"Found {total} proteins (with receptor.pdb) under {train_root}")

    avail = torch.cuda.device_count()
    if avail == 0:
        print("No CUDA device found; running worker on CPU.")
        shards = [prot_list]
        world_size = 1
    else:
        world_size = min(want_gpus, avail)
        indices = np.array_split(np.arange(total), world_size)
        shards = [[prot_list[i] for i in idx.tolist()] for idx in indices]

    cfg = dict(
        ckpt_path=ckpt_path,
        save_root=save_root,
    )

    if world_size == 1:
        worker(0, shards[0], cfg)
    else:
        ctx = get_context("spawn")
        procs = []
        for rank in range(world_size):
            p = ctx.Process(
                target=worker, args=(rank, shards[rank], cfg), daemon=False
            )
            p.start()
            procs.append(p)
        for p in procs:
            p.join()


if __name__ == "__main__":
    main()
