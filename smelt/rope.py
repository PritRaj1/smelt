import torch

from .plac import FRAC_BITS as FRAC


def _angles(dim, max_seq_len, theta=10000.0):
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2, dtype=torch.float64) / dim))
    positions = torch.arange(max_seq_len, dtype=torch.float64)
    return positions.unsqueeze(1) * freqs.unsqueeze(0)


def precompute_freqs(dim, max_seq_len, theta=10000.0):
    """Sin/cos tables in Q16.16. Returns (cos, sin) each [max_seq_len, dim//2] int32."""
    angles = _angles(dim, max_seq_len, theta)
    return (torch.cos(angles) * (1 << FRAC)).round().to(torch.int32), (
        torch.sin(angles) * (1 << FRAC)
    ).round().to(torch.int32)


def precompute_freqs_float(dim, max_seq_len, theta=10000.0):
    """Sin/cos tables in float. Returns (cos, sin) each [max_seq_len, dim//2] float32."""
    angles = _angles(dim, max_seq_len, theta)
    return torch.cos(angles).float(), torch.sin(angles).float()


def rope_int32(x, cos_tab, sin_tab, offset=0):
    """
    RoPE for x in Q16.16. x: [..., dim], tables: [seq, dim//2].

    offset: position offset for KV cache (generation mode).
    """
    *batch, seq, dim = x.shape
    half = dim // 2

    x = x.to(torch.int64)
    x_even = x[..., :half]
    x_odd = x[..., half:]

    cos = cos_tab[offset : offset + seq].to(torch.int64)
    sin = sin_tab[offset : offset + seq].to(torch.int64)

    # broadcast to batch dims
    for _ in batch:
        cos = cos.unsqueeze(0)
        sin = sin.unsqueeze(0)

    # rotation: [even, odd] -> [even*cos - odd*sin, even*sin + odd*cos]
    out_even = (x_even * cos - x_odd * sin) >> FRAC
    out_odd = (x_even * sin + x_odd * cos) >> FRAC

    return torch.cat([out_even, out_odd], dim=-1).to(torch.int32)


def rope_float(x, cos_tab, sin_tab, offset=0):
    """Float RoPE. x: [bsz, seq, n_heads, dim], tables: [max_seq, dim//2]."""
    seq = x.size(1)
    half = x.size(-1) // 2
    c = cos_tab[offset : offset + seq].unsqueeze(0).unsqueeze(2)
    s = sin_tab[offset : offset + seq].unsqueeze(0).unsqueeze(2)
    x0, x1 = x[..., :half], x[..., half:]
    return torch.cat([x0 * c - x1 * s, x0 * s + x1 * c], dim=-1)
