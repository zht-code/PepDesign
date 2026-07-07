import torch
import torch.nn as nn
import torch
import sys
import os
import math
# 确保正确加载模块路径
sys.path.append("/root/autodl-tmp/Peptide_3D/model/esm/data/weights/")
from tokenizers import Tokenizer
from model.esm.pretrained import load_local_model
from model.esm.utils.encoding import tokenize_sequence
from model.esm.tokenization.sequence_tokenizer import EsmSequenceTokenizer
import numpy as np
from model.esm.utils.structure.protein_chain import ProteinChain
import torch.nn.functional as F
import json


def _ensure_pairwise(struct_logits: torch.Tensor, num_bins: int, Lwant: int) -> torch.Tensor:
    """
    -> [1, Lwant, Lwant, NUM_BINS]
    兼容 [1,L,L,NUM_BINS] 或 [1,L,last] (last=L*NUM_BINS).
    如输出的 L 与 Lwant 不同，则先按较大 L 处理后再裁剪到 Lwant。
    """
    if struct_logits.dim() == 4:
        B, Lq, Lk, C = struct_logits.shape
        assert C == num_bins, f"NUM_BINS mismatch: {C} != {num_bins}"
        L = min(Lq, Lk)
        out = struct_logits[:, :L, :L, :]
    elif struct_logits.dim() == 3:
        B, Lq, last = struct_logits.shape
        assert last % num_bins == 0, f"last ({last}) % NUM_BINS ({num_bins}) != 0"
        Lk = last // num_bins
        out = struct_logits.view(B, Lq, Lk, num_bins)
        L = min(Lq, Lk)
        out = out[:, :L, :L, :]
    else:
        raise RuntimeError(f"Unexpected struct_logits shape: {struct_logits.shape}")
    # 切成目标长度
    Lfinal = min(out.size(1), Lwant)
    return out[:, :Lfinal, :Lfinal, :]
class Config:
    def __init__(self, data):
        for key, value in data.items():
            setattr(self, key, value)

def load_config(config_path):
    """ 加载JSON配置文件 """
    with open(config_path, 'r', encoding='utf-8') as file:
        config = json.load(file)
    return Config(config)
# 创建序列掩码
def create_sequence_mask(seq):
    seq_len = seq.size(0)
    mask = torch.triu(torch.ones((seq_len, seq_len)), diagonal=1)
    return mask # (seq_len, seq_len)

class MLP(nn.Module):
    def __init__(
        self, intermediate_size, config
    ):  # in MLP: intermediate_size= 4 * embed_dim
        super().__init__()
        embed_dim = config.embed_dim
        self.mlp = nn.Sequential(
            # nn.Linear(embed_dim, intermediate_size),
            nn.LeakyReLU(0.01) ,
            nn.Dropout(config.resid_pdrop),
            nn.Linear(intermediate_size, embed_dim),
            nn.LeakyReLU(0.01) ,
            nn.Dropout(config.resid_pdrop),
            nn.LayerNorm(embed_dim, config.layer_norm_epsilon)
        )
    def reparameterize(self, mu, log_sigma):
        """Reparameterization trick to sample from N(mu, sigma^2)."""
        std = torch.exp(0.5 * log_sigma)  # sigma = exp(0.5 * log_sigma)
        eps = torch.randn_like(std)
        return mu + eps * std  # z = mu + sigma * epsilon
    
    def forward(self, hidden_states):
        '''执行编码器前向传递，返回重新参数化的潜变量 z。'''
        mu = self.mlp(hidden_states)
        log_sigma = self.mlp(hidden_states)
        z = self.reparameterize(mu, log_sigma).mean(dim=1, keepdim=True)  # 取平均，维度变为 [1, 1, 1536]
        return z, mu, log_sigma
# 定义整体模型
class ProteinPeptideModel(nn.Module):
    def __init__(self, device):
        super(ProteinPeptideModel, self).__init__()
        # 加载encoder:esm3模型以及tokenizer
        self.esm3_model = load_local_model(model_name="esm3_sm_open_v1")
        self.esm_tokenizer = EsmSequenceTokenizer()
        self.device = device
        # 加载decoder:progen2-medium以及tokenizer
        self.esmc = load_local_model(model_name="esmc_300m", device=device)
        self.decoder_tokenizer = EsmSequenceTokenizer()
        # self.attr = MLP(config.embed_dim, config)
        self.embedding_layer = nn.Embedding(num_embeddings=960, embedding_dim=960)
        self.pocket_embed = nn.Linear(3, 960)
        self.vina_embed = nn.Linear(1, 960)
        self.solubility_embed = nn.Linear(1, 960)
        self.stability_embed = nn.Linear(1, 960)
        self.linear_proj = nn.Linear(4096, 960)
        self.freeze_strategy(stage="stage1", n_dec_last=0, n_enc_last=0)

    def _set_requires_grad(self, module, flag: bool):
        for p in module.parameters():
            p.requires_grad = flag

    def freeze_strategy(self, stage: str = "stage1", n_dec_last: int = 4, n_enc_last: int = 2):
        """
        stage:
        - "stage1": 冻结 ESM3 全部；ESMC 只训练 cross_attn + heads + 你的线性映射/属性嵌入
        - "stage2": 在 stage1 基础上，解冻 ESMC decoder 最近 n_dec_last 个 block（attn+ffn）
        - "stage3": 在 stage2 基础上，微调 ESM3 的末端（最后 n_enc_last 个 block + structure_head）
        """
        # 先全部冻结
        self._set_requires_grad(self.esm3_model, False)
        self._set_requires_grad(self.esmc, False)

        # —— 始终要训练的自有小模块（条件映射/属性嵌入）——
        self._set_requires_grad(self.linear_proj, True)        # 4096->960 的投影
        self._set_requires_grad(self.pocket_embed, True)
        self._set_requires_grad(self.vina_embed, True)
        self._set_requires_grad(self.solubility_embed, True)
        self._set_requires_grad(self.stability_embed, True)

        # —— ESMC: 结构/序列 head 必须训练 —— 
        self._set_requires_grad(self.esmc.sequence_head, True)
        self._set_requires_grad(self.esmc.structure_head, True)

        # —— ESMC: cross-attn 用于引入蛋白条件，建议全层解冻 —— 
        for blk in self.esmc.transformer.blocks:
            if hasattr(blk, "cross_attn"):
                self._set_requires_grad(blk.cross_attn, True)
        # 总体 LayerNorm 也放开，便于适配新分布
        if hasattr(self.esmc.transformer, "norm"):
            self._set_requires_grad(self.esmc.transformer.norm, True)

        if stage in ("stage2", "stage3"):
            # 解冻 ESMC decoder 的最后 n_dec_last 个 block 的自注意力 + FFN
            if n_dec_last > 0:
                for blk in self.esmc.transformer.blocks[-n_dec_last:]:
                    if hasattr(blk, "attn"):
                        self._set_requires_grad(blk.attn, True)
                    if hasattr(blk, "ffn"):
                        self._set_requires_grad(blk.ffn, True)

        if stage == "stage3":
            # 轻微微调 ESM3 的末端：最后 n_enc_last 个 block + 输出的 structure_head + 最终 LayerNorm
            tr = self.esm3_model.transformer
            if n_enc_last > 0:
                for blk in tr.blocks[-n_enc_last:]:
                    self._set_requires_grad(blk, True)
            if hasattr(tr, "norm"):
                self._set_requires_grad(tr.norm, True)
            # 只放开 ESM3 的 structure_head（你实际只用它的结构分布）
            if hasattr(self.esm3_model, "output_heads") and hasattr(self.esm3_model.output_heads, "structure_head"):
                self._set_requires_grad(self.esm3_model.output_heads.structure_head, True)

    def build_param_groups(self, base_lr: float = 1e-5):
        """
        根据 requires_grad=True 划分参数组，并给关键模块更高学习率。
        返回可直接给 optimizer 的 param_groups 列表。
        """
        pg_head = []       # ESMC heads
        pg_xattn = []      # ESMC cross-attn
        pg_dec = []        # ESMC 解冻的 decoder 块
        pg_proj_attr = []  # 线性映射与属性嵌入
        pg_esm3 = []       # ESM3 末端（stage3 才会有）

        # 线性映射/属性嵌入
        for m in [self.linear_proj, self.pocket_embed, self.vina_embed, self.solubility_embed, self.stability_embed]:
            for p in m.parameters():
                if p.requires_grad: pg_proj_attr.append(p)

        # ESMC heads
        for m in [self.esmc.sequence_head, self.esmc.structure_head]:
            for p in m.parameters():
                if p.requires_grad: pg_head.append(p)

        # ESMC cross-attn + transformer.norm
        for blk in self.esmc.transformer.blocks:
            if hasattr(blk, "cross_attn"):
                for p in blk.cross_attn.parameters():
                    if p.requires_grad: pg_xattn.append(p)
        if hasattr(self.esmc.transformer, "norm"):
            for p in self.esmc.transformer.norm.parameters():
                if p.requires_grad: pg_xattn.append(p)

        # ESMC decoder 其他被解冻的层（attn/ffn）
        for blk in self.esmc.transformer.blocks:
            if hasattr(blk, "attn"):
                for p in blk.attn.parameters():
                    if p.requires_grad: pg_dec.append(p)
            if hasattr(blk, "ffn"):
                for p in blk.ffn.parameters():
                    if p.requires_grad: pg_dec.append(p)

        # ESM3 末端
        if hasattr(self.esm3_model, "transformer"):
            tr = self.esm3_model.transformer
            for p in tr.parameters():
                if p.requires_grad: pg_esm3.append(p)
            if hasattr(self.esm3_model, "output_heads") and hasattr(self.esm3_model.output_heads, "structure_head"):
                for p in self.esm3_model.output_heads.structure_head.parameters():
                    if p.requires_grad: pg_esm3.append(p)

        # 组装不同学习率
        return [
            {"params": pg_head,       "lr": base_lr * 5.0,  "weight_decay": 0.01},  # heads 学得快
            {"params": pg_xattn,      "lr": base_lr * 3.0,  "weight_decay": 0.01},  # cross-attn/适配层
            {"params": pg_proj_attr,  "lr": base_lr * 3.0,  "weight_decay": 0.01},  # 条件映射/属性嵌入
            {"params": pg_dec,        "lr": base_lr * 1.0,  "weight_decay": 0.01},  # 解冻的 decoder 块
            {"params": pg_esm3,       "lr": base_lr * 0.5,  "weight_decay": 0.01},  # ESM3 末端（很小 LR）
        ]

    
    def forward(self, batch ,mode='train'):
        # 1. 使用ESM3编码蛋白质序列
        esm_output = self.esm3_model(sequence_tokens=batch['receptor_seq_tensor'])
        esm_structure_features = self.linear_proj(esm_output.structure_logits)          #esm_output.structure_logits    [1, 280, 4096]
        # esm_features = esm_output.sequence_logits
        # 调整ESM特征以适应decoder的输入尺寸w
        # esm_features_resized = F.interpolate(esm_features.permute(0, 2, 1), size=self.config.embed_dim, mode='linear').permute(0, 2, 1)
        # past_key_values = None
        if mode == 'train':
            # 训练模式，处理多肽的属性和序列
            pep_ids = self.embedding_layer(batch['peptide_seq_tensor'])
            # pocket_coords = self.pocket_embed(batch['binding_pocket'])
            affinity = self.vina_embed(batch['vina_affinity'].unsqueeze(-1))
            stability = self.stability_embed(batch['stability'].unsqueeze(-1))
            solubility = self.solubility_embed(batch['solubility'].unsqueeze(-1))
            # 所有属性特征融合
            attributes = torch.cat([affinity, stability, solubility], dim=1)
            # 去掉亲和力属性
            # attributes = torch.cat([pocket_coords, solubility], dim=1)
            # 去掉稳定性属性
            # attributes = torch.cat([pocket_coords, affinity, solubility], dim=1)
            # 去掉溶解性属性
            # attributes = torch.cat([pocket_coords, affinity], dim=1)
            
            # 以平均值扩展属性
            mean_attributes = attributes.mean(dim=1, keepdim=True)  # 取平均，维度变为 [1, 1, 1536]
            # mean_attributes = mean_attributes.expand_as(pep_ids[:,1:-1,:])
            mean_attributes = mean_attributes.view(pep_ids.size(0), 1, 1)\
                                  .expand(-1, pep_ids.size(1)-2, pep_ids.size(2))


            Lq = batch['peptide_seq_tensor'].size(1)             # 多肽序列长度（含 BOS/EOS）
            La = mean_attributes.size(1)                          # 属性序列长度（你这里是 Lq-2）
            Lp = esm_structure_features.size(1)                   # 蛋白结构特征长度

            # True = 屏蔽；False = 允许
            attr_cross_attention_mask = torch.zeros((Lq, La), dtype=torch.bool, device=self.device)
            cross_attention_mask      = torch.zeros((Lq, Lp), dtype=torch.bool, device=self.device)

            # 只屏蔽 BOS/EOS 这两行（即不让它们去 attend 属性/蛋白）
            attr_cross_attention_mask[0,  :] = True
            attr_cross_attention_mask[-1, :] = True
            cross_attention_mask[0,  :] = True
            cross_attention_mask[-1, :] = True
            labels = batch
            generated_peptide = self.esmc(sequence_tokens=batch['peptide_seq_tensor'],
                                             labels = labels,
                                             attributes = mean_attributes,
                                              encoder_embeddings = esm_structure_features,
                                              attr_cross_attention_mask = attr_cross_attention_mask,
                                              cross_attention_mask = cross_attention_mask,
                                            )
            sequence_logits = generated_peptide.sequence_logits    #4*32*64
            structure_logits = generated_peptide.structure_logits   #4*32*4096

            return sequence_logits, structure_logits

    # ====== 1) 仅基于“蛋白信息”得到 cross-attn 条件 ======
    @torch.no_grad()
    def encode_protein_from_pdb(self, pdb_path: str) -> tuple[torch.Tensor, torch.Tensor]:
        """
        返回:
        encoder_embeddings: [1, Lp, D]  —— 供解码器 cross-attn 使用
        cross_attention_mask: [1, Lp]   —— 0/1 掩码(1=有效, 0=padding)
        仅用蛋白信息（从 PDB 提取序列 -> ESM3 结构头 -> 线性投影），不依赖多肽属性/序列。
        """
        device = self.device if hasattr(self, "device") else next(self.parameters()).device

        # 1) 从 PDB 拿到蛋白序列
        chain = ProteinChain.from_pdb(pdb_path)
        prot_seq = chain.sequence

        # 2) 用与你训练时一致的 tokenizer 把“蛋白序列”转 tokens
        #    如果你单独有 self.protein_tokenizer 就用它；没有就用解码器自带的
        if hasattr(self, "protein_tokenizer"):
            tokens = self.esmc._tokenize([prot_seq])  # 若你封装了独立 tokenize，可替换这里
            pad_id = self.protein_tokenizer.pad_token_id
        else:
            tokens = self.esmc._tokenize([prot_seq])   # [1, Lp]
            pad_id = self.esmc.tokenizer.pad_token_id

        tokens = tokens.to(device)

        # 3) cross-attn 用的掩码：0/1（1=有效，0=padding）
        cross_attention_mask = (tokens != pad_id).to(torch.int8)  # [1, Lp]

        # 4) 走 ESM3 模型，拿结构 logits（与你训练时一致）
        #    你训练里写的是：
        #    esm_output = self.esm3_model(sequence_tokens=batch['receptor_seq_tensor'])
        #    esm_structure_features = self.linear_proj(esm_output.structure_logits)
        self.esm3_model.eval()
        esm_structure_features = self.esm3_model(sequence_tokens=tokens)  # 只喂蛋白 tokens

        esm_structure_features = esm_structure_features.structure_logits  # 形状可能是 [1, L, 4096] 或 [1, L, L*NUM_BINS] 或 [1, L, L, NUM_BINS]

        # 5) 统一成 [1, L, C] 再线性投影到 d_model
        if esm_structure_features.dim() == 4:
            # [B, Lq, Lk, Cbins] -> 先对 Lk 做汇聚（mean/max 均可；用 mean 更平滑）
            struct_feats = esm_structure_features.mean(dim=2)        # [B, Lq, Cbins]
        elif esm_structure_features.dim() == 3:
            struct_feats = esm_structure_features                     # [B, L, C]
        else:
            raise RuntimeError(f"Unexpected struct_logits shape: {esm_structure_features.shape}")

        # 6) 线性映射到解码器 d_model（确保 self.linear_proj 的 in_features 与 struct_feats.size(-1) 一致）
        encoder_embeddings = self.linear_proj(struct_feats)  # [1, Lp, D]
        # 与解码器参数 dtype 对齐（很多时候是 bfloat16）
        dtype_dec = next(self.esmc.parameters()).dtype
        encoder_embeddings = encoder_embeddings.to(device=device, dtype=dtype_dec)

        return encoder_embeddings, cross_attention_mask


    # ====== 2) 基于蛋白一次性生成若干多肽序列 ======
    @torch.no_grad()
    def generate_sequences_from_protein(
        self,
        pdb_path: str,
        *,
        num_samples: int = 10,
        top_k: int = 8,
        max_len: int = 30,
        temperature: float = 1.0,
    ) -> list[str]:
        self.esmc.eval()
        enc, mask = self.encode_protein_from_pdb(pdb_path)  #enc[1, 280, 960]

        seq_list = []
        for _ in range(num_samples):
            toks = self.esmc.sample_topk(
                encoder_embeddings=enc,
                cross_attention_mask=mask,
                max_len=max_len,
                top_k=top_k,
                temperature=temperature,
            )  # [1, L]
            # 转字符串（去掉首/尾特殊符号）
            s = self.esmc._detokenize(toks)[0]
            seq_list.append(s)
        return seq_list
    # ====== 3) 把多肽序列写成“标准 PDB”（这里提供 CA-trace，通用且规范） ======



    @torch.no_grad()
    def predict_struct_logits_for_sequence(
        self,
        sequence_str: str,
        encoder_embeddings: torch.Tensor,
        cross_attention_mask: torch.Tensor,
        distogram_len: int = 64,
    ) -> torch.Tensor:
        """
        用解码器再前向一次拿结构 logits（只用蛋白条件，不用多肽属性）。
        为兼容固定头宽度（如 4096=64*64），会 pad/trunc 到 distogram_len，然后切回真实 L。
        返回: [1, L, L, NUM_BINS]
        """
        device = encoder_embeddings.device
        toks = self.esmc._tokenize([sequence_str]).to(device)  # [1,L]
        pad_id = self.esmc.tokenizer.pad_token_id

        L = toks.size(1)
        if L > distogram_len:
            toks_eval = toks[:, :distogram_len]
        elif L < distogram_len:
            pad = torch.full((1, distogram_len - L), pad_id, dtype=toks.dtype, device=device)
            toks_eval = torch.cat([toks, pad], dim=1)
        else:
            toks_eval = toks

        out = self.esmc.forward(
            sequence_tokens=toks_eval,
            encoder_embeddings=encoder_embeddings,
            cross_attention_mask=cross_attention_mask,
        )
        struct_logits = _ensure_pairwise(out.structure_logits, 64, L)  # -> [1,L,L,NUM_BINS]
        return struct_logits

    # @staticmethod
    # def write_ca_trace_pdb(sequence: str, out_path: str, chain_id: str = "P") -> None:
    #     """
    #     生成合规 PDB（CA-trace）：每个残基写一个 CA 原子（坐标按 3.8 Å 等间距放置）。
    #     若你需要全原子/二级结构，可把此函数替换为你的折叠/构建器（也可用你结构头产出的距离图做 distance-geometry 折叠）。
    #     """
    #     os.makedirs(os.path.dirname(out_path), exist_ok=True)
    #     # 简单等距排布：沿 x 轴点间距 3.8Å
    #     CA_DIST = 3.8
    #     x = 0.0
    #     lines = []
    #     atom_idx = 1
    #     for i, aa in enumerate(sequence, start=1):
    #         res3 = AA1_TO_AA3.get(aa.upper(), "UNK")
    #         # 只写 CA（标准 PDB 原子名、字段宽度固定）
    #         # ATOM  {idx:>5}  CA  RES chain {resid:>4}    {x:>8.3f}{y:>8.3f}{z:>8.3f}  1.00  0.00           C
    #         lines.append(
    #             f"ATOM  {atom_idx:>5}  CA  {res3:>3} {chain_id}{i:>4}    {x:8.3f}{0.0:8.3f}{0.0:8.3f}  1.00  0.00           C"
    #         )
    #         atom_idx += 1
    #         x += CA_DIST
    #     lines.append("TER")
    #     lines.append("END")
    #     with open(out_path, "w") as f:
    #         f.write("\n".join(lines))

# class CrossAttentionLayer(nn.Module):
#     def __init__(self, config):
#         super().__init__()
#         self.attention = nn.MultiheadAttention(config.embed_dim, 8)
#         self.linear = nn.Linear(config.embed_dim, config.embed_dim)
#         self.dropout = nn.Dropout(config.resid_pdrop)
#         self.norm = nn.LayerNorm(config.embed_dim)

#     def forward(self, query, key, value, key_padding_mask=None):
#         # MultiheadAttention expects [seq_len, batch_size, embedding_dim]
#         query = query.transpose(0, 1)  
#         key = key.transpose(0, 1)  
#         value = value.transpose(0, 1)

#         attn_output, _ = self.attention(query, key, value, key_padding_mask=key_padding_mask)
#         attn_output = attn_output.transpose(0, 1)  # back to [batch_size, seq_len, embedding_dim]
#         query = query.transpose(0, 1)
        
#         # Apply linear layer and normalization
#         output = self.dropout(self.linear(attn_output))
#         output = self.norm(output + query)  # Add & Norm step

#         return output