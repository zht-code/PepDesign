import os
# os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"  # 避免碎片化
import torch.nn as nn
import torch
import warnings
import logging
import json
import numpy as np
import psutil
MEMORY_THRESHOLD = 20 * 1024 ** 3  # 2GB
LOSS_RECORD_FILE = "/root/autodl-tmp/Peptide_3D/model_loss_record.json"
# 例：64 个距离 bin，从 2Å 到 22Å
BIN_EDGES = np.linspace(2.0, 22.0, num=65)  # 65 个边界 => 64 个区间
NUM_BINS  = len(BIN_EDGES) - 1
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
def get_disk_available_gb(path="/root/autodl-tmp"):
    usage = psutil.disk_usage(path)
    available_gb = usage.free / (1024 ** 3)
    # print(f"可用空间: {available_gb:.2f} GB")
    return available_gb

# def update_model_loss_record(model_path, val_loss):
#     record = {}
#     if os.path.exists(LOSS_RECORD_FILE):
#         with open(LOSS_RECORD_FILE, 'r') as f:
#             record = json.load(f)
#     record[model_path] = val_loss
#     with open(LOSS_RECORD_FILE, 'w') as f:
#         json.dump(record, f, indent=2)

# def delete_worst_model():
#     if not os.path.exists(LOSS_RECORD_FILE):
#         return
#     with open(LOSS_RECORD_FILE, 'r') as f:
#         record = json.load(f)
#     if not record:
#         return
#     worst_model = max(record.items(), key=lambda x: x[1])[0]
#     try:
#         os.remove(worst_model)
#         logging.info(f"Deleted worst model due to low memory: {worst_model}")
#         del record[worst_model]
#         with open(LOSS_RECORD_FILE, 'w') as f:
#             json.dump(record, f, indent=2)
#     except Exception as e:
#         logging.warning(f"Failed to delete model: {worst_model}, Error: {e}")

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

# 训练过程示例
def train_model(data_loader, model_save_path):
    # 初始化模型并移至设备
    model = ProteinPeptideModel(device).to(device)
    
    # 使用 Adam 优化器，初始学习率设为 0.001
    # optimizer = torch.optim.SGD(model.parameters(), lr=1e-3)
    # optimizer = torch.optim.Adam(model.parameters(), lr=1e-5)
    # optimizer = Ranger(model.parameters(), lr=1e-5)

    # ====== 分阶段设置 ======
    base_lr = 1e-5
    stage1_epochs = 10      # 先对齐条件：只训 cross-attn + heads + 属性映射
    stage2_epochs = 20     # 解冻 decoder 最后 n 个 block
    STEP_EVERY_EPOCHS = 10     # 每 10 个 epoch 衰减一次
    DECAY_GAMMA = 0.5   
    # 之后进入 stage3：微调 ESM3 尾部
    # 初始化为 Stage1
    model.freeze_strategy(stage="stage1", n_dec_last=0, n_enc_last=0)
    optimizer = torch.optim.AdamW(model.build_param_groups(base_lr=base_lr), betas=(0.9, 0.95))
    scheduler = StepLR(optimizer, step_size=STEP_EVERY_EPOCHS, gamma=DECAY_GAMMA)
    
    
    # 使用余弦退火调度器，设定总迭代周期为100个epoch，最低学习率为1e-6
    # scheduler = CosineAnnealingLR(optimizer, T_max=50, eta_min=1e-10)
    # scheduler = CosineAnnealingLR(optimizer, T_max=26, eta_min=5.9e-6)
    # 如果希望使用基于指标的调度器，也可以选择如下方案：
    # scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3, min_lr=1e-6)
    
    min_loss = float('inf')
    # min_loss = 2.78
    total_epochs = 1000
    for epoch in range(total_epochs):
        # —— 阶段切换 —— 
        if epoch == stage1_epochs:
            # 进入 Stage2：解冻 decoder 尾部，提高表达力
            model.freeze_strategy(stage="stage2", n_dec_last=4, n_enc_last=0)
            optimizer = torch.optim.AdamW(model.build_param_groups(base_lr=base_lr), betas=(0.9,0.95))
            scheduler = StepLR(optimizer, step_size=STEP_EVERY_EPOCHS, gamma=DECAY_GAMMA)
            logging.info("Switched to Stage2: unfreeze last 4 decoder blocks.")
        if epoch == stage1_epochs + stage2_epochs:
            # 进入 Stage3：轻微微调 ESM3 尾部（更小 lr）
            model.freeze_strategy(stage="stage3", n_dec_last=4, n_enc_last=2)
            optimizer = torch.optim.AdamW(model.build_param_groups(base_lr=5e-6), betas=(0.9,0.95))
            scheduler = StepLR(optimizer, step_size=STEP_EVERY_EPOCHS, gamma=DECAY_GAMMA)
            logging.info("Switched to Stage3: unfreeze last 2 encoder blocks and esm3 structure_head.")
        model.train()
        total_loss = 0
        count = 0
        for i, batch in enumerate(data_loader):
            optimizer.zero_grad(set_to_none=True)
            # 将各个batch数据移至设备，同时请确保数据预处理（例如归一化）正确
            batch['peptide_seq_tensor'] = batch['peptide_seq_tensor'].to(device)
            # batch['binding_pocket'] = batch['binding_pocket'].to(device)
            batch['stability'] = batch['stability'].to(device)
            batch['solubility'] = batch['solubility'].to(device)
            batch['vina_affinity'] = batch['vina_affinity'].to(device)
            batch['receptor_seq_tensor'] = batch['receptor_seq_tensor'].to(device)
            struct_labels = batch["structure_labels"].to(device)
            
            # 前向传播
            sequence_logits, struct_logits = model(batch)
            # 如果你提取的是对角以下或 full matrix，需要 reshape，对应 logits 也要一样 reshape
            # 假设你结构 head 只预测每个残基 pairwise 距离分布：
            # struct_logits: [B, L, L, NUM_BINS]
            # struct_labels: [B, L, L]
            # 期望 logits 是 [B, Lq, Lk, NUM_BINS] 或 [B, Lq, Lk*NUM_BINS]
            # NUM_BINS = NUM_BINS  # 已在文件顶部定义为 64
            if struct_logits.dim() == 4:
                B, Lq, Lk, C = struct_logits.shape
                assert C == NUM_BINS, f"Expected NUM_BINS={NUM_BINS}, got {C}"
            elif struct_logits.dim() == 3:
                # 兼容 [B, Lq, Lk*NUM_BINS] 这类输出
                B, Lq, last = struct_logits.shape
                assert last % NUM_BINS == 0, f"last dim {last} not divisible by NUM_BINS={NUM_BINS}"
                Lk = last // NUM_BINS
                struct_logits = struct_logits.contiguous().view(B, Lq, Lk, NUM_BINS)
            else:
                raise RuntimeError(f"Unexpected struct_logits shape: {struct_logits.shape}")

            # 与标签对齐到相同的方阵大小
            # struct_labels: [B, Ltgt, Ltgt]（来自 collate 的 pad 后尺寸）
            Ltgt = struct_labels.size(-1)
            L = min(Lq, Lk, Ltgt)
            logits_4d = struct_logits[:, :L, :L, :].contiguous()
            labels_2d = struct_labels[:, :L, :L].contiguous()

            # 展平计算交叉熵（忽略 padding=-100）
            # logits_flat = logits_4d.view(-1, NUM_BINS)     # [B*L*L, NUM_BINS]
            # labels_flat = labels_2d.view(-1)               # [B*L*L]
            # struct_loss = F.cross_entropy(logits_flat, labels_flat, ignore_index=-100)
            # struct_loss.backward()
            logits_flat = logits_4d.view(-1, NUM_BINS)
            labels_flat = labels_2d.view(-1)

            valid = labels_flat != -100
            if valid.any():
                struct_loss = F.cross_entropy(logits_flat[valid], labels_flat[valid])
                struct_loss.backward()
            else:
                # 本 batch 没有有效监督，直接跳过
                continue

            # loss = seq_loss + 1.0 * struct_loss  # 1.0 可根据需要调整权重
            # adaptive_gradient_clipping(model.parameters(), clip_factor=0.01, eps=1e-3)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            
            total_loss += struct_loss.item()
            count += 1
        avg_loss = total_loss / count
        logging.info(f"Epoch {epoch+1}, Average Loss: {avg_loss}")
        
        # 如果使用 ReduceLROnPlateau，则调用 scheduler.step(avg_loss)
        # scheduler.step(avg_loss)
        
        # 对于余弦退火调度器，每个 epoch 调整一次学习率
        scheduler.step()
        logging.info(f"Learning rate after epoch {epoch+1}: {optimizer.param_groups[0]['lr']}")
            
     
        # 保存表现最佳的模型
        if avg_loss < min_loss:
            min_loss = avg_loss
            # 隔几个epoch保存一次模型
            # if epoch % 4 == 0:
            # torch.save(model.state_dict(), os.path.join(model_save_path, f"best_model_epoch_{epoch+1}_loss_{min_loss:.4f}.pth"))
            # logging.info(f"Saved new best model with loss {min_loss:.4f}")
                    # 先判断内存，如果低于阈值则删除最差模型
            if get_disk_available_gb() < MEMORY_THRESHOLD:
                logging.warning("Memory low. Attempting to delete worst model before saving new one.")
                # delete_worst_model()
            
            model_path = os.path.join(model_save_path, f"best_model_epoch_{epoch+1}_loss_{min_loss:.4f}.pth")
            torch.save(model.state_dict(), model_path)
            # update_model_loss_record(model_path, min_loss)
            logging.info(f"Saved new best model with loss {min_loss:.4f}")

if __name__ == "__main__":
    pdb_dir = '/root/autodl-tmp/train_data'  # PDB文件路径
    # model_save_path = '/root/autodl-fs/pp_esm3'  # 模型保存路径
    model_save_path = '/root/autodl-tmp/Peptide_3D/logs'  # 模型保存路径
    json_files = [
        '/root/autodl-tmp/Peptide_3D/data/stability_scores.json',
        '/root/autodl-tmp/Peptide_3D/data/hdock_scores.json',
        '/root/autodl-tmp/Peptide_3D/data/solubility_scores.json'
        # '/root/autodl-tmp/Peptide_3D/data/interaction_windows_scores.json'
    ]
    # 将 batch_size 从1调整到16，以获得更稳定的梯度更新（根据显存允许情况进行调整）
    dataset_processor = DatasetProcess(
        pdb_dir, 
        json_files, 
        batch_size=1,  
        # collate_fn=PaddingCollate(pad_keys=['peptide_seq_tensor', 'receptor_seq_tensor', 'binding_pocket', 'vina_affinity', 'stability', 'solubility', 'structure_labels']),
        collate_fn=PaddingCollate(pad_id=0, dist_ignore_index=-100),
        shuffle=False,
        num_workers=1
    )
    data_loader = dataset_processor.create_data_loader()  # 获取 DataLoader 实例
    train_model(data_loader, model_save_path)