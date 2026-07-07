import os
# os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"  # 避免碎片化
import torch.nn as nn
import torch.distributed as dist
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
import torch
import warnings
import logging
import json
import math
import numpy as np
import psutil
import sys
sys.path.append("/root/autodl-tmp/Peptide_3D")
MEMORY_THRESHOLD = 20 * 1024 ** 3  # 2GB
LOSS_RECORD_FILE = "/root/autodl-tmp/Peptide_3D/model_loss_record1.json"
# 例：64 个距离 bin，从 2Å 到 22Å
BIN_EDGES = np.linspace(2.0, 22.0, num=65)  # 65 个边界 => 64 个区间
NUM_BINS  = len(BIN_EDGES) - 1
LAMBDA_IFACE = 1.0  # 界面损失权重，可在 0.5~2.0 之间调
from ranger import Ranger
from model.esm.utils.structure.protein_chain import ProteinChain
import torch.nn.functional as F
from utils.datasets.dataset_process import DatasetProcess
from models import ProteinPeptideModel
from modules.Reinforce_learning import *
from transformers import get_cosine_schedule_with_warmup  # 如果需要，也可采用 transformers 内的调度器
from modules.paddingCollate import *
from torch.optim.lr_scheduler import CosineAnnealingLR, ReduceLROnPlateau
from typing import Tuple
from torch.optim.lr_scheduler import StepLR
from datetime import timedelta
def get_disk_available_gb(path="/root/autodl-tmp"):
    usage = psutil.disk_usage(path)
    available_gb = usage.free / (1024 ** 3)
    # print(f"可用空间: {available_gb:.2f} GB")
    return available_gb

# 忽略所有警告
warnings.filterwarnings("ignore")

# 设备设置
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.set_float32_matmul_precision("high")

from torch.cuda.amp import autocast, GradScaler
scaler = GradScaler()

# 设置日志
logging.basicConfig(filename='/root/autodl-tmp/Peptide_3D/training.log', level=logging.INFO,
                    format='%(asctime)s:%(levelname)s:%(message)s')

# 加载PDB文件并提取序列
def extract_sequence_from_pdb(pdb_path):
    protein_chain = ProteinChain.from_pdb(pdb_path)
    return protein_chain.sequence

def adaptive_gradient_clipping(parameters, clip_factor=0.01, eps=1e-3):
    """
    自适应梯度裁剪，防止梯度爆炸
    """
    for p in parameters:
        if p.grad is None:
            continue
        param_norm = torch.max(torch.norm(p, dim=-1, keepdim=True), torch.tensor(eps).to(p.device))
        grad_norm = torch.norm(p.grad, dim=-1, keepdim=True)
        max_norm = param_norm * clip_factor
        # 如果梯度范数超过阈值，则对梯度进行缩放
        clip_coef = torch.min(max_norm / (grad_norm + eps), torch.tensor(1.).to(p.device))
        p.grad.data.mul_(clip_coef)
def setup_ddp():
    """Init DDP from torchrun env."""
    dist.init_process_group(
    backend="nccl",
    init_method="env://",
    timeout=timedelta(seconds=7200),)
    local_rank = int(os.environ["LOCAL_RANK"])
    world_size = dist.get_world_size()
    rank = dist.get_rank()
    torch.cuda.set_device(local_rank)
    return local_rank, world_size, rank

def cleanup_ddp():
    if dist.is_initialized():
        dist.destroy_process_group()

# 训练过程示例
def train_model(data_loader, model_save_path, device, rank):
    # 1) 初始化模型并移至设备
    model = ProteinPeptideModel(device).to(device)

    # 2) 先包 DDP（后续需要从 model.module 调用自定义方法）
    model = DDP(model, device_ids=[device.index], output_device=device.index,
                find_unused_parameters=True)

    # ====== 分阶段设置 ======
    base_lr = 1e-5
    stage1_epochs = 10
    stage2_epochs = 20
    STEP_EVERY_EPOCHS = 10     # 每 10 个 epoch 衰减一次
    DECAY_GAMMA = 0.5          # 衰减系数

    # 3) 初始化为 Stage1（从 module 上调用你自定义方法）
    model.module.freeze_strategy(stage="stage1", n_dec_last=0, n_enc_last=0)
    # optimizer = torch.optim.AdamW(model.module.build_param_groups(base_lr=base_lr),
    #                               betas=(0.9, 0.95))
    optimizer = Ranger(model.module.build_param_groups(base_lr=base_lr),
                                  betas=(0.9, 0.95))
    scheduler = StepLR(optimizer, step_size=STEP_EVERY_EPOCHS, gamma=DECAY_GAMMA)

    min_loss = float('inf')
    total_epochs = 1000

    for epoch in range(total_epochs):

        # —— 阶段切换 —— 
        if epoch == stage1_epochs:
            model.module.freeze_strategy(stage="stage2", n_dec_last=4, n_enc_last=0)
            optimizer = Ranger(model.module.build_param_groups(base_lr=base_lr),
                                          betas=(0.9,0.95))
            scheduler = StepLR(optimizer, step_size=STEP_EVERY_EPOCHS, gamma=DECAY_GAMMA)
            if rank == 0:
                logging.info("Switched to Stage2: unfreeze last 4 decoder blocks.")

        if epoch == stage1_epochs + stage2_epochs:
            model.module.freeze_strategy(stage="stage3", n_dec_last=4, n_enc_last=2)
            optimizer = Ranger(model.module.build_param_groups(base_lr=5e-6),
                                          betas=(0.9,0.95))
            scheduler = StepLR(optimizer, step_size=STEP_EVERY_EPOCHS, gamma=DECAY_GAMMA)
            if rank == 0:
                logging.info("Switched to Stage3: unfreeze last 2 encoder blocks and esm3 structure_head.")

        # 分布式 sampler 需要在每个 epoch 设置不同 seed
        if isinstance(data_loader.sampler, DistributedSampler):
            data_loader.sampler.set_epoch(epoch)

        model.train()
        total_loss = 0.0
        count = 0

        for i, batch in enumerate(data_loader):
            optimizer.zero_grad(set_to_none=True)

            # 搬到本进程设备
            for k in ["peptide_seq_tensor", "stability", "solubility",
                      "vina_affinity", "receptor_seq_tensor", "structure_labels", "interface_labels"]:
                if k in batch and torch.is_tensor(batch[k]):
                    batch[k] = batch[k].to(device, non_blocking=True)

            # 前向
            sequence_logits, struct_logits, interface_logits = model(batch)

            # ===== 结构损失（保持不变）=====
            if struct_logits.dim() == 4:
                B, Lq, Lk, C = struct_logits.shape
                assert C == NUM_BINS, f"Expected NUM_BINS={NUM_BINS}, got {C}"
            elif struct_logits.dim() == 3:
                B, Lq, last = struct_logits.shape
                assert last % NUM_BINS == 0, f"last dim {last} not divisible by NUM_BINS={NUM_BINS}"
                Lk = last // NUM_BINS
                struct_logits = struct_logits.contiguous().view(B, Lq, Lk, NUM_BINS)
            else:
                raise RuntimeError(f"Unexpected struct_logits shape: {struct_logits.shape}")

            struct_labels = batch["structure_labels"]
            Ltgt = struct_labels.size(-1)
            L = min(Lq, Lk, Ltgt)
            logits_4d = struct_logits[:, :L, :L, :].contiguous()
            labels_2d = struct_labels[:, :L, :L].contiguous()

            logits_flat = logits_4d.view(-1, NUM_BINS)
            labels_flat = labels_2d.view(-1)

            valid = labels_flat != -100
            if not valid.any():
                continue  # 本 batch 没标签
            struct_loss = F.cross_entropy(logits_flat[valid], labels_flat[valid])

            # ===== NEW: 界面损失（若数据集提供 interface_labels）=====
            iface_loss = torch.tensor(0.0, device=device)
            if (interface_logits is not None) and ("interface_labels" in batch):
                # interface_logits: [B, L_pep, L_rec]
                iface_labels = batch["interface_labels"]  # 期望 shape 同 logits，元素 ∈ {0,1}，缺失处 = -100
                # 对齐长度
                B, Lp, Lr = interface_logits.shape
                Lp_tgt = min(Lp, iface_labels.size(1))
                Lr_tgt = min(Lr, iface_labels.size(2))
                ilog = interface_logits[:, :Lp_tgt, :Lr_tgt].contiguous()
                ilab = iface_labels[:, :Lp_tgt, :Lr_tgt].contiguous()

                # mask ignore
                valid_iface = ilab != -100
                if valid_iface.any():
                    # BCE with logits（把 label 转 float）
                    bce = torch.nn.functional.binary_cross_entropy_with_logits(
                        ilog[valid_iface], ilab[valid_iface].float()
                    )
                    iface_loss = bce
                # else: 仍旧保持 0

            # ===== 总损失 =====
            loss = struct_loss + LAMBDA_IFACE * iface_loss
            loss.backward()

            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            total_loss += struct_loss.item()
            count += 1

        # —— 计算所有进程的全局平均损失（便于稳定日志）——
        total_loss_t = torch.tensor([total_loss], device=device)
        count_t = torch.tensor([count], device=device, dtype=torch.float32)
        dist.all_reduce(total_loss_t, op=dist.ReduceOp.SUM)
        dist.all_reduce(count_t, op=dist.ReduceOp.SUM)
        global_avg_loss = (total_loss_t / torch.clamp(count_t, min=1.)).item()

        scheduler.step()

        if rank == 0:
            logging.info(f"Epoch {epoch+1}, Average Loss: {global_avg_loss}")
            logging.info(f"Learning rate after epoch {epoch+1}: {optimizer.param_groups[0]['lr']}")

            if global_avg_loss < min_loss:
                min_loss = global_avg_loss
                if get_disk_available_gb() < MEMORY_THRESHOLD:
                    logging.warning("Memory low. Attempting to delete worst model before saving new one.")
                model_path = os.path.join(model_save_path, f"best_model_epoch_{epoch+1}_loss_{min_loss:.4f}.pth")
                # DDP 保存要从 module 拿权重
                torch.save(model.module.state_dict(), model_path)
                logging.info(f"Saved new best model with loss {min_loss:.4f}")


if __name__ == "__main__":
    # 1) 初始化 DDP
    local_rank, world_size, rank = setup_ddp()
    device = torch.device(f"cuda:{local_rank}")

    # 仅在 rank0 写日志到文件；其他 rank 简化输出级别
    if rank != 0:
        logging.getLogger().setLevel(logging.ERROR)

    pdb_dir = '/root/autodl-tmp/train_data_augmentation'
    model_save_path = '/root/autodl-tmp/Peptide_3D/logs_data_augmentation'
    json_files = [
        '/root/autodl-tmp/Peptide_3D/data/train_data_augmentation_stability_scores.json',
        '/root/autodl-tmp/Peptide_3D/utils/Data_augmentation/train_data_augmentation_hdock_scores.json',
        '/root/autodl-tmp/Peptide_3D/data/train_data_augmentation_solubility_scores.json'
        # '/root/autodl-tmp/Peptide_3D/data/interaction_windows_results.json'
    ]

    # 用 DatasetProcess 构建 dataset，再手动塞 DistributedSampler
    dp = DatasetProcess(
        pdb_dir,
        json_files,
        batch_size=1,
        collate_fn=PaddingCollate(pad_id=0, dist_ignore_index=-100),
        shuffle=True,                  # 打乱交给 sampler
        num_workers=1
    )
    dataset = dp.create_dataset()
    sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank, shuffle=True, drop_last=False)

    data_loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=dp.batch_size,
        sampler=sampler,               # 关键：用 sampler，不用 shuffle
        num_workers=dp.num_workers,
        collate_fn=dp.collate_fn,
        prefetch_factor=20,
        pin_memory=True,
        persistent_workers=(dp.num_workers > 0),
        drop_last=False
    )

    # 开训
    try:
        train_model(data_loader, model_save_path, device, rank)
    finally:
        cleanup_ddp()
