import torch
import torch.nn as nn

from smelt.quantize import TernaryLinear, pack_ternary, quantize_ternary, unpack_ternary


def test_roundtrip_pack_unpack():
    """Pack then unpack recovers ternary weights exactly."""
    torch.manual_seed(0)
    w_t, _ = quantize_ternary(torch.randn(32, 64))
    pos, neg = pack_ternary(w_t)
    assert (w_t == unpack_ternary(pos, neg, 64)).all()


def test_ternary_forward_vs_float():
    """Ternary forward approximates float matmul (NMSE < 0.5)."""
    torch.manual_seed(2)
    linear = nn.Linear(13, 7, bias=False)
    x = torch.randn(8, 13)

    y_ref = linear(x)
    y_t = TernaryLinear(linear)(x)

    nmse = ((y_ref - y_t) ** 2).mean() / (y_ref**2).mean()
    assert nmse < 0.5, f"NMSE {nmse:.3f} too high"


def test_ternary_linear_with_bias():
    """Bias adds correctly to ternary output."""
    torch.manual_seed(3)
    linear = nn.Linear(32, 16, bias=True)
    x = torch.randn(4, 32)

    tl = TernaryLinear(linear)
    tl_nobias = TernaryLinear(nn.Linear(32, 16, bias=False))
    tl_nobias.pos_mask.copy_(tl.pos_mask)
    tl_nobias.neg_mask.copy_(tl.neg_mask)
    tl_nobias.scale.copy_(tl.scale)

    assert torch.allclose(tl(x), tl_nobias(x) + linear.bias, atol=1e-5)
