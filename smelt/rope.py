import torch

FRAC = 16


def precompute_freqs(dim, max_seq_len, theta=10000.0):
    """
    Sin/cos tables in Q16.16 for RoPE.

    Returns (cos_table, sin_table) each [max_seq_len, dim//2] int32.
    """
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2, dtype=torch.float64) / dim))
    positions = torch.arange(max_seq_len, dtype=torch.float64)
    angles = positions.unsqueeze(1) * freqs.unsqueeze(0)

    cos_tab = (torch.cos(angles) * (1 << FRAC)).round().to(torch.int32)
    sin_tab = (torch.sin(angles) * (1 << FRAC)).round().to(torch.int32)
    return cos_tab, sin_tab


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
