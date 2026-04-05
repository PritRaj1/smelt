import torch
import torch.nn as nn
import torch.nn.functional as F

from .matmul import quantize_activations

MASK_NEG = -(1 << 30)  # large -ve for masked positions in Q16.16


def _int8_qkt(q_i8, k_i8):
    """Int8 QK^T. [bsz, nh, seq, hd] -> [bsz, nh, seq, kv_len] int32."""
    bsz, nh, seq, hd = q_i8.shape
    kv_len = k_i8.size(2)
    return torch.ops.smelt.int8_batched_gemm_t(
        q_i8.reshape(bsz * nh, seq, hd).contiguous(),
        k_i8.reshape(bsz * nh, kv_len, hd).contiguous(),
    ).view(bsz, nh, seq, kv_len)


def _repeat_kv(x, n_rep):
    if n_rep == 1:
        return x

    bsz, nh, seq, hd = x.shape
    return x[:, :, None, :, :].expand(bsz, nh, n_rep, seq, hd).reshape(bsz, nh * n_rep, seq, hd)


def smelt_attention_forward(module, query, key, value, attention_mask, scaling, **kwargs):
    """HF attention dispatch: int8 QK^T, int32 LUT softmax, int8 AV."""
    n_rep = getattr(module, "num_key_value_groups", 1)
    key = _repeat_kv(key, n_rep)
    value = _repeat_kv(value, n_rep)

    bsz, nh, seq, hd = query.shape
    kv_len = key.shape[2]

    # int8 QK^t -> int32
    q_i8, q_s = quantize_activations(query.reshape(-1, hd))
    k_i8, k_s = quantize_activations(key.reshape(-1, hd))
    scores_i32 = _int8_qkt(q_i8.view(bsz, nh, seq, hd), k_i8.view(bsz, nh, kv_len, hd))

    # int32 -> Q16.16 for softmax (one float mul)
    q_s = q_s.view(bsz, nh, seq, 1)
    k_s = k_s.view(bsz, nh, 1, kv_len)
    scores_q16 = (scores_i32.float() * (65536.0 * scaling / (q_s * k_s))).to(torch.int32)

    if attention_mask is not None:
        scores_q16 = scores_q16.masked_fill(attention_mask < -1.0, MASK_NEG)

    # int softmax -> int8 attn weights
    attn_q16 = torch.ops.smelt.softmax(scores_q16)
    attn_i8, attn_mx = torch.ops.smelt.int_quantize(attn_q16.reshape(-1, kv_len))

    # int8 softmax(QK^T) * V (per-tensor: one scale factors out of the sum)
    v_abs = value.abs().amax().clamp(min=1e-5)
    v_i8 = (value * (127.0 / v_abs)).round().clamp(-128, 127).to(torch.int8)
    av_i32 = torch.ops.smelt.int8_batched_gemm_t(
        attn_i8.view(bsz * nh, seq, kv_len).contiguous(),
        v_i8.reshape(bsz * nh, kv_len, hd).transpose(1, 2).contiguous(),
    )

    # final float: real = av_i32 * attn_mx * v_abs / (127^2 * 65536)
    rescale = attn_mx.view(bsz, nh, seq, 1).float() * (v_abs / (127.0 * 127.0 * 65536.0))
    out = (av_i32.view(bsz, nh, seq, hd).float() * rescale).to(value.dtype)
    return out.transpose(1, 2).contiguous(), None


def register_attention():
    from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS
    ALL_ATTENTION_FUNCTIONS["smelt"] = smelt_attention_forward


class KVCache:
    """Pre-allocated KV cache. K as int8 (quantized on insert), V as float."""

    def __init__(self, max_seq_len, n_kv_heads, head_dim, dtype=torch.float32):
        self.k_i8 = torch.zeros(1, max_seq_len, n_kv_heads, head_dim, dtype=torch.int8)
        self.k_scale = torch.zeros(1, max_seq_len, n_kv_heads, 1, dtype=dtype)
        self.v = torch.zeros(1, max_seq_len, n_kv_heads, head_dim, dtype=dtype)
        self.pos = 0

    def update(self, k_float, v_float):
        seq = k_float.size(1)
        k_i8, k_s = quantize_activations(k_float)
        self.k_i8[:, self.pos : self.pos + seq] = k_i8
        self.k_scale[:, self.pos : self.pos + seq] = k_s
        self.v[:, self.pos : self.pos + seq] = v_float
        self.pos += seq
        return (
            self.k_i8[:, : self.pos],
            self.k_scale[:, : self.pos],
            self.v[:, : self.pos],
        )

    def reset(self):
        self.pos = 0


class Attention(nn.Module):
    """Int8 QK^T attention with GQA and KV cache. Used by notebook 06."""

    def __init__(self, q_proj, k_proj, v_proj, o_proj, n_heads, n_kv_heads=None):
        super().__init__()
        self.q_proj = q_proj
        self.k_proj = k_proj
        self.v_proj = v_proj
        self.o_proj = o_proj
        self.n_heads = n_heads
        self.n_kv_heads = n_kv_heads or n_heads
        self.head_dim = q_proj.out_features // n_heads
        self.scale = self.head_dim**-0.5

    def forward(self, x, freqs=None, cache=None):
        from .rope import rope_float

        bsz, seq, _ = x.shape
        hd = self.head_dim

        q = self.q_proj(x).view(bsz, seq, self.n_heads, hd)
        k = self.k_proj(x).view(bsz, seq, self.n_kv_heads, hd)
        v = self.v_proj(x).view(bsz, seq, self.n_kv_heads, hd)

        if freqs is not None:
            cos, sin = freqs
            offset = cache.pos if cache else 0
            q = rope_float(q, cos, sin, offset)
            k = rope_float(k, cos, sin, offset)

        q_i8, q_s = quantize_activations(q)

        if cache is not None:
            k_i8, k_s, v = cache.update(k, v)
        else:
            k_i8, k_s = quantize_activations(k)

        if self.n_kv_heads < self.n_heads:
            rep = self.n_heads // self.n_kv_heads
            k_i8 = k_i8.unsqueeze(3).expand(-1, -1, -1, rep, -1).reshape(bsz, -1, self.n_heads, hd)
            k_s = k_s.unsqueeze(3).expand(-1, -1, -1, rep, -1).reshape(bsz, -1, self.n_heads, 1)
            v = v.unsqueeze(3).expand(-1, -1, -1, rep, -1).reshape(bsz, -1, self.n_heads, hd)

        scores = _int8_qkt(q_i8.transpose(1, 2), k_i8.transpose(1, 2)).float()
        q_s = q_s.transpose(1, 2)
        k_s = k_s.transpose(1, 2)
        scores = scores * (self.scale / (q_s * k_s.transpose(-2, -1)))

        if seq > 1:
            total = k_i8.size(1)
            mask = torch.full((seq, total), float("-inf"), device=x.device)
            mask = torch.triu(mask, diagonal=total - seq + 1)
            scores = scores + mask

        attn = F.softmax(scores, dim=-1)
        v_f = v.transpose(1, 2)
        out = (attn @ v_f).transpose(1, 2).contiguous().view(bsz, seq, -1)
        return self.o_proj(out)
