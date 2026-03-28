import torch

from smelt.plac import from_fixed, to_fixed
from smelt.rope import precompute_freqs, rope_int32


def _float_rope(x, dim, theta=10000.0):
    seq = x.shape[-2]
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2, dtype=torch.float64) / dim))
    pos = torch.arange(seq, dtype=torch.float64)
    angles = pos.unsqueeze(1) * freqs.unsqueeze(0)
    cos, sin = torch.cos(angles), torch.sin(angles)

    for _ in range(x.dim() - 2):
        cos = cos.unsqueeze(0)
        sin = sin.unsqueeze(0)

    half = dim // 2
    x_even, x_odd = x[..., :half], x[..., half:]
    return torch.cat([x_even * cos - x_odd * sin, x_even * sin + x_odd * cos], dim=-1)


def test_correctness():
    """Integer RoPE matches float reference."""
    torch.manual_seed(0)
    dim = 64
    x = torch.randn(2, 8, dim, dtype=torch.float64)

    y_ref = _float_rope(x, dim).numpy()
    cos_tab, sin_tab = precompute_freqs(dim, 8)
    x_fix = torch.from_numpy(to_fixed(x.numpy()))
    y_int = from_fixed(rope_int32(x_fix, cos_tab, sin_tab).numpy())

    assert abs(y_ref - y_int).max() < 0.01


def test_offset():
    """Offset=5 matches position 5 of full sequence."""
    torch.manual_seed(1)
    dim = 32
    cos_tab, sin_tab = precompute_freqs(dim, 16)

    x_full = torch.randn(1, 16, dim, dtype=torch.float64)
    x_full_fix = torch.from_numpy(to_fixed(x_full.numpy()))

    y_full = rope_int32(x_full_fix, cos_tab, sin_tab)
    y_at_5 = rope_int32(x_full_fix[:, 5:6, :], cos_tab, sin_tab, offset=5)

    assert (y_full[:, 5:6, :] == y_at_5).all()
