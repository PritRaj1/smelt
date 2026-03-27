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


def test_rmsnorm_direction():
    """Output direction matches float RMSNorm."""
    torch.manual_seed(1)
    x = torch.randn(4, 32, dtype=torch.float64)
    gamma = torch.ones(32, dtype=torch.float64) * 0.5

    rms = (x**2).mean(dim=-1, keepdim=True).sqrt()
    y_ref = x / rms.clamp(min=1e-6) * gamma
    y_int = torch.from_numpy(from_fixed(rmsnorm_int32(_fix(x), _fix(gamma)).numpy()))

    for i in range(4):
        assert torch.cosine_similarity(y_ref[i], y_int[i], dim=0) > 0.95


def test_layernorm_correctness():
    """Integer LayerNorm approximates float LayerNorm."""
    torch.manual_seed(2)
    x = torch.randn(8, 64, dtype=torch.float64)
    gamma = torch.ones(64, dtype=torch.float64)
    beta = torch.zeros(64, dtype=torch.float64)

    y_ref = torch.nn.functional.layer_norm(x, [64], gamma, beta).numpy()
    y_int = from_fixed(layernorm_int32(_fix(x), _fix(gamma), _fix(beta)).numpy())

    assert abs(y_ref - y_int).max() < 0.15


def test_layernorm_with_bias():
    """LayerNorm beta shifts output correctly."""
    torch.manual_seed(3)
    x = torch.randn(4, 32, dtype=torch.float64)
    gamma = torch.ones(32, dtype=torch.float64)
    beta = torch.ones(32, dtype=torch.float64) * 0.5

    y = from_fixed(layernorm_int32(_fix(x), _fix(gamma), _fix(beta)).numpy())
    y_no_beta = from_fixed(
        layernorm_int32(_fix(x), _fix(gamma), _fix(torch.zeros(32, dtype=torch.float64))).numpy()
    )

    # beta shifts by ~0.5
    diff = (y - y_no_beta).mean()
    assert abs(diff - 0.5) < 0.05
