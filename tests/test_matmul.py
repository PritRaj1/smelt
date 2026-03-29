import torch
import torch.nn as nn

from smelt.matmul import (
    TernaryLinear,
    pack_tl1,
    quantize_activations,
    quantize_ternary,
    unpack_tl1,
)


def test_roundtrip():
    """Pack then unpack recovers ternary weights exactly."""
    torch.manual_seed(0)
    w_t, _ = quantize_ternary(torch.randn(32, 64))
    packed, n_pairs, n_padded = pack_tl1(w_t)
    assert (w_t == unpack_tl1(packed, n_pairs, n_padded, 32, 64)).all()


def test_reconstruction():
    """Dequantized int8 activations approximate original."""
    torch.manual_seed(1)
    x = torch.randn(4, 32) * 10
    x_q, scale = quantize_activations(x)
    x_recon = x_q.float() / scale
    rel_err = (x - x_recon).abs().max() / x.abs().max()
    assert rel_err < 0.02


def test_forward():
    """Ternary+int8 forward approximates float matmul."""
    torch.manual_seed(2)
    linear = nn.Linear(64, 32, bias=True)
    x = torch.randn(8, 64)

    y_ref = linear(x)
    y_t = TernaryLinear(linear)(x)

    nmse = ((y_ref - y_t) ** 2).mean() / (y_ref**2).mean()
    assert nmse < 0.5, f"NMSE {nmse:.3f} too high"
