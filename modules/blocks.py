import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from esm.layers.attention import (
    FlashMultiHeadAttention,
    MultiHeadAttention,
)
from esm.layers.geom_attention import (
    GeometricReasoningOriginalImpl,
)
from esm.utils.structure.affine3d import Affine3D


def swiglu_correction_fn(expansion_ratio: float, d_model: int) -> int:
    # set hidden dimesion to nearest multiple of 256 after expansion ratio
    return int(((expansion_ratio * d_model) + 255) // 256 * 256)


class SwiGLU(nn.Module):
    """
    SwiGLU activation function as an nn.Module, allowing it to be used within nn.Sequential.
    This module splits the input tensor along the last dimension and applies the SiLU (Swish)
    activation function to the first half, then multiplies it by the second half.
    """

    def __init__(self):
        super(SwiGLU, self).__init__()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x1, x2 = x.chunk(2, dim=-1)
        return F.silu(x1) * x2


def swiglu_ln_ffn(d_model: int, expansion_ratio: float, bias: bool):
    return nn.Sequential(
        nn.LayerNorm(d_model),
        nn.Linear(
            d_model, swiglu_correction_fn(expansion_ratio, d_model) * 2, bias=bias
        ),
        SwiGLU(),
        nn.Linear(swiglu_correction_fn(expansion_ratio, d_model), d_model, bias=bias),
    )


def gelu_ln_ffn(d_model: int, expansion_ratio: float, bias: bool):
    hidden_dim = int(expansion_ratio * d_model)
    return nn.Sequential(
        nn.LayerNorm(d_model),
        nn.Linear(d_model, hidden_dim, bias=bias),
        nn.GELU(),
        nn.Linear(hidden_dim, d_model, bias=bias),
    )

# def scaled_dot_product_attention(q, k, v, mask=None):
#     matmul_qk = torch.matmul(q, k.transpose(-2, -1))  # (..., heads, Lq, Lk)
#     d_k = q.size(-1)
#     scaled_attention_logits = matmul_qk / math.sqrt(d_k)
#     if mask is not None:
#         scaled_attention_logits += mask * -1e9
#     attention_weights = F.softmax(scaled_attention_logits, dim=-1)
#     output = torch.matmul(attention_weights, v)  # (..., heads, Lq, d)
#     return output, attention_weights
# -------------------- 修正后的 SDPA（mask 支持 bool 或 0/1） --------------------
def scaled_dot_product_attention(Q, K, V, attention_mask=None, dropout_p=0.0):
    """
    Q: [B, h, Lq, d]
    K: [B, h, Lk, d]
    V: [B, h, Lk, d]
    attention_mask: 支持 [B,Lk] / [B,Lq,Lk] / [B,1,Lq,Lk] / [1,1,Lq,Lk] / [h,1,Lq,Lk] / [h,Lk] / [Lk]
                    语义：True=屏蔽，False=保留；若是 0/1，自动转成 bool。
    """
    B, h, Lq, d = Q.shape
    Lk = K.size(-2)
    attn_logits = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(d)
    def normalize_mask(mask, B, h, Lq, Lk):
        """
        归一化各种形状的 attention_mask 到 [B, h, Lq, Lk] (bool)。
        兼容输入形状： [Lq,Lk], [B,Lq,Lk], [h,Lq,Lk], [B,1,Lq,Lk], [1,h,Lq,Lk],
                    [B,h,Lq,Lk], 以及偶发的 5D: [B,h,1,Lq,Lk] / [B,h,1,1,Lk] 等。
        True 表示“可见”（允许注意力），False 表示屏蔽。
        """
        if mask is None:
            return None

        m = mask
        # 统一成 bool 掩码
        if m.dtype != torch.bool:
            m = m != 0

        # 先把 >4 维且含有单例维的掩码，反复 squeeze 掉多余的 1 维，直到 <= 4 维
        while m.dim() > 4 and 1 in m.shape:
            # 挤掉第一个为 1 的维度
            for ax, sz in enumerate(m.shape):
                if sz == 1:
                    m = m.squeeze(ax)
                    break

        # 规整到 4 维以内
        if m.dim() == 2:                       # [Lq, Lk]
            m = m.unsqueeze(0).unsqueeze(0)    # [1,1,Lq,Lk]
        elif m.dim() == 3:                     # [B,Lq,Lk] 或 [h,Lq,Lk]
            if m.size(0) == B:
                m = m.unsqueeze(1)             # [B,1,Lq,Lk]
            elif m.size(0) == h:
                m = m.unsqueeze(0)             # [1,h,Lq,Lk]
            else:
                m = m.unsqueeze(0).unsqueeze(1)# [1,1,Lq,Lk]
        elif m.dim() == 4:
            # 可能是 [h,B,Lq,Lk] 这种，把 head 维换到 dim=1
            if m.size(0) == h and m.size(1) in (1, B):
                m = m.permute(1, 0, 2, 3)      # [B,h,Lq,Lk]
            # 也可能是 [1,h,Lk,Lq] 这类顺序错位；确保最后两维是 [Lq,Lk]
            if m.size(-2) != Lq or m.size(-1) != Lk:
                # 只保守地裁剪到目标长度（如果更复杂的顺序，请在上游构造规范的掩码）
                m = m[..., :Lq, :Lk]
        else:
            # 少见：仍是 5 维但没有单例维可挤（比如把某个“分组维”并进来了）
            # 尝试把第 2 维（通常是“通道/分组”）合并掉
            if m.dim() == 5:
                # 优先 squeeze 常见的第三维=1 情形（你的报错就是这种）
                if m.size(2) == 1:
                    m = m.squeeze(2)           # -> [B,h,Lq,Lk] 或 [B,h,1,Lk]
                else:
                    # 否则沿第 2 维做 any 合并为单通道，再视作 [B,1,Lq,Lk]
                    m = m.any(dim=2, keepdim=False)  # [B,h,Lq,Lk] or [B,h,*,*]
            # 再递归一次
            return normalize_mask(m, B, h, Lq, Lk)

        # 对最后两维做裁剪（安全起见）
        m = m[..., :Lq, :Lk]

        # 广播 batch/head 到目标大小
        if m.size(0) == 1 and B > 1:
            m = m.expand(B, m.size(1), Lq, Lk)
        if m.size(1) == 1 and h > 1:
            m = m.expand(m.size(0), h, Lq, Lk)

        # 最终保证形状正确
        assert m.shape == (B, h, Lq, Lk), f"mask normalized to {m.shape}, expect {(B,h,Lq,Lk)}"
        return m

    # def normalize_mask(mask, B, h, Lq, Lk):
    #     # 0/1 -> bool（True=要屏蔽）
    #     if mask.dtype != torch.bool:
    #         mask = (mask == 0)

    #     # 先把最后两维凑成 [*, *, Lq?, Lk?]
    #     if mask.dim() == 1:         # [Lk]（或极少数 [Lq]）
    #         mask = mask.unsqueeze(0).unsqueeze(0).unsqueeze(0)   # [1,1,1,Lk]
    #     elif mask.dim() == 2:       # [B,Lk] / [h,Lk]
    #         mask = mask.unsqueeze(1).unsqueeze(2)                # [*,1,1,Lk]
    #     elif mask.dim() == 3:       # [B,Lq,Lk] / [h,Lq,Lk]
    #         mask = mask.unsqueeze(1)                              # [*,1,Lq,Lk]
    #     elif mask.dim() == 4:
    #         pass
    #     else:
    #         raise RuntimeError(f"Unsupported mask shape: {mask.shape}")

    #     # 对齐 Lk：长裁短补（补 True=全屏蔽）
    #     if mask.size(-1) != Lk:
    #         if mask.size(-1) > Lk:
    #             mask = mask[..., :Lk]
    #         else:
    #             pad_w = Lk - mask.size(-1)
    #             pad = torch.ones(*mask.shape[:-1], pad_w, dtype=torch.bool, device=mask.device)
    #             mask = torch.cat([mask, pad], dim=-1)

    #     # 对齐 Lq：允许 1 或 Lq；短补长裁
    #     if mask.size(-2) not in (1, Lq):
    #         if mask.size(-2) > Lq:
    #             mask = mask[..., :Lq, :]
    #         else:
    #             pad_h = Lq - mask.size(-2)
    #             pad = torch.ones(*mask.shape[:-2], pad_h, mask.size(-1), dtype=torch.bool, device=mask.device)
    #             mask = torch.cat([mask, pad], dim=-2)

    #     # 处理 “把 head 维当成了 batch 维”的情况：
    #     #   1) 如果 mask 的第 0 维恰好是 h（head 数），说明缺 batch 维；补一个 batch 维
    #     if mask.size(0) == h and B != h:
    #         mask = mask.unsqueeze(0)              # [1,h,Lq,Lk]
    #         mask = mask.expand(B, -1, -1, -1)     # [B,h,Lq,Lk]
    #     #   2) 如果第 0 维是 1，按 batch 广播
    #     elif mask.size(0) == 1 and B != 1:
    #         mask = mask.expand(B, -1, -1, -1)
    #     #   3) 若第 0 维既不是 1/ B / h，尝试把第 1 维当 batch（很少见）
    #     elif mask.size(0) not in (B,):
    #         if mask.size(1) == B:
    #             mask = mask.permute(1, 0, 2, 3)   # swap 到 [B,*,*,*]
    #         else:
    #             # 退而求其次：截取或广播到 B
    #             mask = mask[:1].expand(B, -1, -1, -1)

    #     # 头维：允许 1 或 h；若是 1 则广播到 h
    #     if mask.size(1) == 1 and h > 1:
    #         mask = mask.expand(-1, h, -1, -1)
    #     elif mask.size(1) > h:
    #         mask = mask[:, :h, :, :]              # 多了就裁掉
    #     elif mask.size(1) < h:
    #         mask = mask.expand(-1, h, -1, -1)     # 少了就广播

    #     return mask

    if attention_mask is not None:
        mask = normalize_mask(attention_mask, B, h, Lq, Lk)
        # attn_logits = attn_logits.masked_fill(mask, torch.finfo(attn_logits.dtype).min)
        if mask is not None:
            # ★ 关键：把 mask 放到 attn_logits 的设备（CPU 或 GPU 都行）
            mask = mask.to(attn_logits.device)
            attn_logits = attn_logits.masked_fill(mask, torch.finfo(attn_logits.dtype).min)
        # === 兜底：如果某个 query 的整行都被屏蔽，给它一个全 0 的 logits，softmax 后是均匀零，不会 NaN
        all_masked = mask.all(dim=-1, keepdim=True)               # [B,h,Lq,1]
        attn_logits = torch.where(all_masked, torch.zeros_like(attn_logits), attn_logits)
    attn_weights = F.softmax(attn_logits, dim=-1)
    if dropout_p > 0:
        attn_weights = F.dropout(attn_weights, p=dropout_p, training=True)
    out = torch.matmul(attn_weights, V)  # [B,h,Lq,d]
    return out, attn_weights


class CrossAttention(nn.Module):
    def __init__(self, d_model: int, n_heads: int, bias: bool = False):
        super().__init__()
        self.n_heads = n_heads
        self.d_model = d_model
        self.head_dim = d_model // n_heads

        self.query_proj = nn.Linear(d_model, d_model, bias=bias)
        self.key_proj   = nn.Linear(d_model, d_model, bias=bias)
        self.value_proj = nn.Linear(d_model, d_model, bias=bias)
        self.out_proj   = nn.Linear(d_model, d_model, bias=bias)

    def forward(self, query, key, value, attention_mask=None):
        B, Lq, _ = query.shape
        _, Lk, _ = key.shape

        # Linear projections and reshape for multi-head
        Q = self.query_proj(query).view(B, Lq, self.n_heads, self.head_dim).transpose(1, 2)  # [B, h, Lq, d]
        K = self.key_proj(key).view(B, Lk, self.n_heads, self.head_dim).transpose(1, 2)
        V = self.value_proj(value).view(B, Lk, self.n_heads, self.head_dim).transpose(1, 2)

        # Scaled Dot Product Attention
        out, attn_weights = scaled_dot_product_attention(Q, K, V, attention_mask)  # [B, h, Lq, d]

        # Reshape back
        out = out.transpose(1, 2).contiguous().view(B, Lq, self.d_model)  # [B, Lq, D]
        return self.out_proj(out)

class UnifiedTransformerBlock(nn.Module):
    """
    A unified transformer block that can optionally incorporate geometric attention.

    This class defines a transformer block that can be configured to use geometric attention
    alongside the standard multi-head attention mechanism. It is designed to be a flexible
    component of transformer-based models, allowing for the integration of geometric reasoning.

    Parameters
    ----------
    d_model : int
        The dimensionality of the input and output features of the transformer block.
    n_heads : int
        The number of attention heads in the multi-head attention mechanism.
    n_layers : int
        The number of layers in the transformer block.
    use_geom_attn : bool, optional
        Whether to use geometric attention in addition to the standard multi-head attention. Defaults to False.
    v_heads : int, optional
        The number of heads to use for the geometric attention mechanism, if enabled. Must be specified if `use_geom_attn` is True.
    """

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        use_geom_attn: bool = False,
        use_plain_attn: bool = True,
        use_flash_attn: bool = False,
        v_heads: int | None = None,
        use_cross_attn: bool = True, 
        bias: bool = False,
        expansion_ratio: float = 4.0,
        residue_scaling_factor: float = 1,
        mask_and_zero_frameless: bool = False,
        qk_layernorm: bool = True,
        ffn_type: str = "swiglu",  # swiglu | gelu
        use_attr_first: bool = True,       # 先做 属性←蛋白，再 多肽←属性
    ):
        super().__init__()
        self.use_plain_attn = use_plain_attn
        if self.use_plain_attn:
            if use_flash_attn:
                self.attn = FlashMultiHeadAttention(
                    d_model, n_heads, bias, qk_layernorm=qk_layernorm
                )
            else:
                self.attn = MultiHeadAttention(
                    d_model, n_heads, bias, qk_layernorm=qk_layernorm
                )
        self.use_geom_attn = use_geom_attn
        if self.use_geom_attn:
            if v_heads is None:
                raise ValueError("v_heads must be specified when use_geom_attn is True")
            self.geom_attn = GeometricReasoningOriginalImpl(
                c_s=d_model,
                v_heads=v_heads,
                bias=bias,
                mask_and_zero_frameless=mask_and_zero_frameless,
            )
        if ffn_type == "swiglu":
            self.ffn = swiglu_ln_ffn(d_model, expansion_ratio, bias)
        elif ffn_type == "gelu":
            self.ffn = gelu_ln_ffn(d_model, expansion_ratio, bias)
        else:
            raise ValueError(f"Unknown ffn_type: {ffn_type}")
        self.scaling_factor = residue_scaling_factor
        # -------- 新增的两级交叉注意力 --------
        self.use_attr_first = use_attr_first
        # 1) 属性 ← 蛋白
        self.attr_protein_cross_attn = CrossAttention(d_model, n_heads, bias=bias)
        # 2) 多肽 ← (属性←蛋白) 的融合结果
        self.peptide_attr_cross_attn = CrossAttention(d_model, n_heads, bias=bias)

        # （可选）保留原来的“多肽 ← 蛋白”直连 cross-attn
        self.use_cross_attn = use_cross_attn
        if self.use_cross_attn:
            self.cross_attn = CrossAttention(d_model, n_heads, bias=bias)

    def forward(
    #     self,
    #     x: torch.Tensor,
    #     sequence_id: torch.Tensor,
    #     frames: Affine3D,
    #     frames_mask: torch.Tensor,
    #     chain_id: torch.Tensor,
    # ) -> torch.Tensor:
        self,
        x: torch.Tensor,
        sequence_id: torch.Tensor,
        frames: Affine3D,
        frames_mask: torch.Tensor,
        chain_id: torch.Tensor,
        attributes: torch.Tensor | None = None,
        attr_cross_attention_mask: torch.Tensor | None = None,
        encoder_hidden: torch.Tensor | None = None,   # <— 新增参数
        encoder_mask: torch.Tensor  | None = None,   # <— （可选）cross-attn mask
    ) -> torch.Tensor:
        """
        Forward pass for the UnifiedTransformerBlock.

        Parameters
        ----------
        x : torch.Tensor[float]
            Input tensor to the transformer block, typically the output from the previous layer.
        sequence_id : torch.Tensor[int]
            Tensor containing sequence IDs for each element in the batch, used for attention masking.
        frames : Affine3D
            Affine3D containing geometric frame information for geometric attention.
        frames_mask : torch.Tensor[bool]
            Boolean mask tensor indicating valid frames for geometric attention.
        chain_id : torch.Tensor[int]
            Tensor containing chain IDs for each element, used for attention masking in geometric attention.

        Returns
        -------
        torch.Tensor[float]
            The output tensor after applying the transformer block operations.
        """
        # if self.use_plain_attn:
        #     r1 = self.attn(x, sequence_id)
        #     x = x + r1 / self.scaling_factor

        # if self.use_geom_attn:
        #     r2 = self.geom_attn(x, frames, frames_mask, sequence_id, chain_id)
        #     x = x + r2 / self.scaling_factor
        # # ——————— cross-attention 引入 encoder 输出 ———————
        # if self.use_cross_attn and encoder_hidden is not None:
        #     # query=x, key/value=encoder_hidden
        #     # 注意：MultiHeadAttention 的 forward 可能需要调整参数顺序
        #     r_cross = self.cross_attn(query=x, key=encoder_hidden, value=encoder_hidden, attention_mask=encoder_mask)
        #     x = x + r_cross / self.scaling_factor
        # r3 = self.ffn(x) / self.scaling_factor
        # x = x + r3

        # return x
        # 1) 多肽自注意力
        if self.use_plain_attn:
            r1 = self.attn(x, sequence_id)
            x = x + r1 / self.scaling_factor

        # 2) 几何注意力（如果启用）
        if self.use_geom_attn:
            r2 = self.geom_attn(x, frames, frames_mask, sequence_id, chain_id)
            x = x + r2 / self.scaling_factor

        # 3) 先做 “属性 ← 蛋白” 融合（如果都有）
        fused_attr = attributes
        if self.use_attr_first and (attributes is not None) and (encoder_hidden is not None):
            fused_attr = self.attr_protein_cross_attn(
                query=attributes,                 # La
                key=encoder_hidden,               # Lp
                value=encoder_hidden,
                attention_mask=encoder_mask     # [La, Lp] or broadcastable
            )
            # 残差：让属性在自身空间也稳定
            fused_attr = attributes + fused_attr / self.scaling_factor

        # 4) 再做 “多肽 ← 融合后的属性”
        if fused_attr is not None:
            r_attr2pep = self.peptide_attr_cross_attn(
                query=x,                           # Lq (peptide)
                key=fused_attr,                    # La (fused attr)
                value=fused_attr,
                attention_mask=attr_cross_attention_mask  # [Lq, La] or broadcastable
            )
            x = x + r_attr2pep / self.scaling_factor

        # 5) （可选）直接 “多肽 ← 蛋白”
        # if self.use_cross_attn and (encoder_hidden is not None):
        #     r_cross = self.cross_attn(
        #         query=x,
        #         key=encoder_hidden,
        #         value=encoder_hidden,
        #         attention_mask=encoder_mask
        #     )
        #     x = x + r_cross / self.scaling_factor

        # 6) FFN
        r3 = self.ffn(x) / self.scaling_factor
        x = x + r3
        return x