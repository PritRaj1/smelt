import torch

from smelt.norm import layernorm_int32, rmsnorm_int32
from smelt.plac import from_fixed, to_fixed


def _fix(x):
    return torch.from_numpy(to_fixed(x.numpy()))


def test_rmsnorm_correctness():
    """Integer RMSNorm approximates float RMSNorm."""
    torch.manual_seed(0)
    x = torch.randn(8, 64, dtype=torch.float64)
    gamma = torch.ones(64, dtype=torch.float64)

    rms = (x**2).mean(dim=-1, keepdim=True).sqrt()
    y_ref = (x / rms.clamp(min=1e-6) * gamma).numpy()
    y_int = from_fixed(rmsnorm_int32(_fix(x), _fix(gamma)).numpy())

    assert abs(y_ref - y_int).max() < 0.1


def test_layernorm_correctness():
    """Integer LayerNorm approximates float LayerNorm (non-trivial gamma and beta)."""
    torch.manual_seed(2)
    x = torch.randn(8, 64, dtype=torch.float64)
    gamma = torch.ones(64, dtype=torch.float64) * 0.8
    beta = torch.ones(64, dtype=torch.float64) * 0.5

    y_ref = torch.nn.functional.layer_norm(x, [64], gamma, beta).numpy()
    y_int = from_fixed(layernorm_int32(_fix(x), _fix(gamma), _fix(beta)).numpy())

    assert abs(y_ref - y_int).max() < 0.15
