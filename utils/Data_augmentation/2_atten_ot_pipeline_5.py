"""
单卡
"""
import torch
import os
from typing import Optional, Tuple
import json
NEIGHBORS_JSON_PATH = "/root/autodl-tmp/Peptide_3D/utils/Data_augmentation/ot_top10_neighbors_5.json"
ROOT_PATH = "/root/autodl-tmp/Peptide_3D/data/train_data_pt" 

def sinkhorn_log(
        cost: torch.Tensor,
        mu: torch.Tensor,
        nu: torch.Tensor,
        eps: float = 0.1,
        max_iters: int = 50,
        thresh: float = 1e-6,
        acc_factor: Optional[float] = None,
) -> torch.Tensor:
    """Sinkhorn algorithm for regularized optimal transport in log space.
    Codes are adapated from https://github.com/gpeyre/SinkhornAutoDiff.

    Args:
        cost: cost matrices, (b, m, n)
        mu, nu: row-wise & column-wise target marginals
        eps: regularization factor; eps -> 0, closer to original ot problem
        thresh: sinkhorn stopping criteria
        max_iters: maximal number of sinkhorn iterations
        accelerate: bool, specify True to accelerate the unbalanced transport
    Return:
        P: optimal transport plan, (b, m, n)
    """

    if acc_factor:
        # To accelerate unbalanced transport
        lam = 0.5 ** 2 / (0.5 ** 2 + eps)
        tau = -acc_factor

    def ave(u: torch.Tensor, u_prev: torch.Tensor) -> torch.Tensor:
        "Barycenter subroutine, used by kinetic acceleration through extrapolation."
        return tau * u + (1 - tau) * u_prev

    def M(u: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        "Modified cost for logarithmic updates"
        "$M_{ij} = (-c_{ij} + u_i + v_j) / \epsilon$"
        return (-cost + u.unsqueeze(-1) + v.unsqueeze(-2)) / eps

    # Sinkhorn iterations
    u, v = torch.zeros_like(mu), torch.zeros_like(nu)
    log_mu, log_nu = torch.log(mu + 1e-8), torch.log(nu + 1e-8)
    for _ in range(max_iters):
        u_prev = u

        # accelerated unbalanced iterations
        if acc_factor:
            u = ave(u, lam * (eps * (log_mu - torch.logsumexp(M(u, v), dim=-1)) + u))
            v = ave(
                v,
                lam
                * (
                        eps * (log_nu - torch.logsumexp(M(u, v).transpose(-2, -1), dim=-1))
                        + v
                ),
            )
        else:
            # Fixed point updates
            u = eps * (log_mu - torch.logsumexp(M(u, v), dim=-1)) + u
            v = eps * (log_nu - torch.logsumexp(M(u, v).transpose(-2, -1), dim=-1)) + v

        # Stopping criteria
        err = (u - u_prev).norm(dim=-1).mean()
        if err < thresh:
            break

    # Transport plan P = diag(a)*K*diag(b)
    P = torch.exp(M(u, v))
    return P


def load_all_files(root: str):
    """
    递归扫描 root 目录下所有 *.pt / *.pth 文件，torch.load 后返回列表。
    每个元素是一个 dict，至少包含键 "embedding" 和 "last_block_attn"。
    """
    suffix = {".pt", ".pth"}
    paths = [os.path.join(dp, f)
             for dp, _, fs in os.walk(root)
             for f in fs if os.path.splitext(f)[1].lower() in suffix]
    if not paths:
        raise RuntimeError(f"在 {root} 下没有找到任何 *.pt 或 *.pth 文件")

    dist_dict={}
    data_list = []
    for p in paths:
        file_name = os.path.basename(p)
        tmp = torch.load(p, map_location="cuda:1")  # 如需 GPU 可换成 map_location="cuda:0"
        # 简单校验 key 是否存在
        if "embedding" not in tmp or "last_block_attn" not in tmp:
            raise KeyError(f"{p} 缺少必要字段")

        attn = process_attn(tmp["last_block_attn"])[1:-1]
        embedding=tmp["embedding"][0,1:-1].unsqueeze(0)

        data_list.append([file_name, embedding, attn])
        dist_dict[file_name]=[]
    return data_list,dist_dict


def process_attn(attn) -> torch.Tensor:
    """
    attn: [1, 24, 280, 280]  ->  [280]
    1. 按头平均 -> [1, 280, 280]
    2.  squeeze 掉 batch 维 -> [280, 280]
    3. 取第 0 个 token 对所有 token 的注意力 -> [280]
    """
    if attn.ndim != 4 or attn.shape[1] != 24:
        raise ValueError("last_block_attn 维度应为 [1, 24, 280, 280]")
    attn = attn.mean(dim=1)  # [1, 280, 280]
    attn = attn.squeeze(0)  # [280, 280]
    attn = attn[0, :]  # [280]
    return attn


def get_ot_distance(a, b, attn_a, attn_b):
    M = torch.cdist(a, b, p=2) ** 2
    Pw = sinkhorn_log(M, attn_a.view(1,-1), attn_b.view(1,-1), eps=0.5, max_iters=100)  # (B, N, N)

    # 7. 计算 OT 损失: sum_{i,j} M[i,j] * Pw[i,j]
    losses = (M * Pw).sum()  # (B,)
    return losses


from tqdm import tqdm

def main(root):
    file_data, dist_dict = load_all_files(root)
    n = len(file_data)
    p = len(file_data)//5

    # 外层：遍历每个蛋白
    outer_bar = tqdm(range(4*p+1, 5*p), desc="蛋白遍历", unit="protein")
    for i in outer_bar:
        name_a = file_data[i][0]
        embeddings_a = file_data[i][1]
        attn_vecs_a = file_data[i][2]

        # 内层：当前蛋白与其他蛋白计算 OT
        inner_bar = tqdm(range(i + 1, n), desc=f"与 {name_a} 计算 OT", leave=False, unit="pair")
        for j in inner_bar:
            name_b = file_data[j][0]
            embeddings_b = file_data[j][1]
            attn_vecs_b = file_data[j][2]

            d = get_ot_distance(embeddings_a, embeddings_b, attn_vecs_a, attn_vecs_b)

            dist_dict[name_a].append([name_b, float(d)])
            dist_dict[name_b].append([name_a, float(d)])

    # 对每个蛋白保留前 10 个最近邻
    for name_b in dist_dict:
        dist_dict[name_b] = sorted(dist_dict[name_b], key=lambda x: x[1])[:10]

    os.makedirs(os.path.dirname(NEIGHBORS_JSON_PATH), exist_ok=True)
    with open(NEIGHBORS_JSON_PATH, "w") as f:
        json.dump(dist_dict, f, indent=2)

    print(f"[ALL DONE] OT top-10 邻居已保存到: {NEIGHBORS_JSON_PATH}")



# --------------- 运行 ---------------
if __name__ == "__main__":
    root_path = ROOT_PATH 
    main(root_path)




