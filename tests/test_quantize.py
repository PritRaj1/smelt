import torch
import torch.nn as nn

from smelt.quantize import (
    TernaryLinear,
    pack_ternary,
    quantize_activations,
    quantize_ternary,
    unpack_ternary,
)


def test_roundtrip_pack_unpack():
    """Pack then unpack recovers ternary weights exactly."""
    torch.manual_seed(0)
    w_t, _ = quantize_ternary(torch.randn(32, 64))
    val, sign = pack_ternary(w_t)
    assert (w_t == unpack_ternary(val, sign, 64)).all()


def test_activation_reconstruction():
    """Dequantized int8 activations approximate original."""
    torch.manual_seed(1)
    x = torch.randn(4, 32) * 10
    x_q, scale = quantize_activations(x)
    x_recon = x_q.float() / scale
    rel_err = (x - x_recon).abs().max() / x.abs().max()
    assert rel_err < 0.02


def test_ternary_forward_vs_float():
    """Ternary+int8 forward approximates float matmul."""
    torch.manual_seed(2)
    linear = nn.Linear(64, 32, bias=True)
    x = torch.randn(8, 64)

    y_ref = linear(x)
    y_t = TernaryLinear(linear)(x)

    nmse = ((y_ref - y_t) ** 2).mean() / (y_ref**2).mean()
    assert nmse < 1.0, f"NMSE {nmse:.3f} too high"
