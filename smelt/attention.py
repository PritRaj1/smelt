import torch
import torch.nn as nn
import torch.nn.functional as F

from ._clib import load_lib
from .rope import rope_float


def _to_i8(x):
    """Per-vector absmax quantize. x: [..., dim] -> (int8, scale[..., 1])."""
    scale = 127.0 / x.abs().amax(dim=-1, keepdim=True).clamp(min=1e-5)
    return (x * scale).round().clamp(-128, 127).to(torch.int8), scale


def _int8_qkt(q_i8, k_i8):
    """Batched QK^T via int8 GEMM."""
    lib = load_lib()
    bsz, nh, seq, hd = q_i8.shape
    kv_len = k_i8.size(2)
    q_flat = q_i8.reshape(bsz * nh, seq, hd).contiguous()
    k_flat = k_i8.reshape(bsz * nh, kv_len, hd).contiguous()
    return lib.int8_batched_gemm_t(q_flat, k_flat).view(bsz, nh, seq, kv_len)


class KVCache:
    """Pre-allocated KV cache. K stored as int8 (quantized once on insert), V as float."""

    def __init__(self, max_seq_len, n_kv_heads, head_dim, dtype=torch.float32):
        self.k_i8 = torch.zeros(1, max_seq_len, n_kv_heads, head_dim, dtype=torch.int8)
        self.k_scale = torch.zeros(1, max_seq_len, n_kv_heads, 1, dtype=dtype)
        self.v = torch.zeros(1, max_seq_len, n_kv_heads, head_dim, dtype=dtype)
        self.pos = 0

    def update(self, k_float, v_float):
        """Quantize K to int8 on insert. Returns (k_i8, k_scale, v) up to current pos."""
        seq = k_float.size(1)
        k_i8, k_s = _to_i8(k_float)
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
    """Multi-head attention with GQA and KV cache.

    QK^T uses int8 GEMM kernel. attn*V and softmax stay float.
    K quantized to int8 once on cache insert, not re-quantized on each decode step.
    """

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

        q_i8, q_s = _to_i8(q)

        if cache is not None:
            k_i8, k_s, v = cache.update(k, v)
        else:
            k_i8, k_s = _to_i8(k)

        # GQA: expand KV heads
        if self.n_kv_heads < self.n_heads:
            rep = self.n_heads // self.n_kv_heads
            k_i8 = k_i8.unsqueeze(3).expand(-1, -1, -1, rep, -1).reshape(bsz, -1, self.n_heads, hd)
            k_s = k_s.unsqueeze(3).expand(-1, -1, -1, rep, -1).reshape(bsz, -1, self.n_heads, 1)
            v = v.unsqueeze(3).expand(-1, -1, -1, rep, -1).reshape(bsz, -1, self.n_heads, hd)

        scores = _int8_qkt(q_i8.transpose(1, 2), k_i8.transpose(1, 2)).float()

        # rescale: int32 / (q_scale * k_scale) * 1/sqrt(hd)
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
