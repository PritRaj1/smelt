import torch

FRAC = 16
ONE = 1 << FRAC
LOG2E_FIX = 94548  # log2(e) in Q16.16

# 2^(f/256) for f in [0,255], Q16.16
_EXP2_LUT = torch.tensor(
    [round((2 ** (i / 256)) * ONE) for i in range(256)],
    dtype=torch.int64,
)


def _exp2_fixed(x):
    """Vectorized 2^(x/ONE) for negative Q16.16 x. Returns int64."""
    x = x.clamp(min=-(20 << FRAC), max=0).to(torch.int64)
    x_b2 = (x * LOG2E_FIX) >> FRAC  # natural -> base-2

    q = x_b2 >> FRAC
    f_idx = ((x_b2 & (ONE - 1)) >> 8) & 0xFF
    lut_vals = _EXP2_LUT.to(x.device)[f_idx]

    # 2^q as right-shift (q <= 0)
    return lut_vals >> (-q).clamp(min=0, max=31)


def softmax_int32(x):
    """Integer-only softmax. Q16.16 int32 in, Q16.16 int32 out."""
    x_shifted = (x - x.amax(dim=-1, keepdim=True)).to(torch.int64)
    exp_vals = _exp2_fixed(x_shifted)

    s = exp_vals.sum(dim=-1, keepdim=True)
    s = s.clamp(min=1)  # avoid div by zero

    # normalize: out[i] = exp[i] * ONE / sum
    return ((exp_vals << FRAC) // s).to(torch.int32)
