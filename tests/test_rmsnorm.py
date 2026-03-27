import torch

from smelt.plac import from_fixed, to_fixed
from smelt.rmsnorm import rmsnorm_int32


def _float_rmsnorm(x, gamma):
    rms = (x**2).mean(dim=-1, keepdim=True).sqrt()
    return x / rms.clamp(min=1e-6) * gamma


def test_correctness():
    """Integer RMSNorm approximates float RMSNorm."""
    torch.manual_seed(0)
    x = torch.randn(8, 64, dtype=torch.float64)
    gamma = torch.ones(64, dtype=torch.float64)

    y_ref = _float_rmsnorm(x, gamma).numpy()
    x_fix = torch.from_numpy(to_fixed(x.numpy()))
    g_fix = torch.from_numpy(to_fixed(gamma.numpy()))
    y_int = from_fixed(rmsnorm_int32(x_fix, g_fix).numpy())

    max_err = abs(y_ref - y_int).max()
    assert max_err < 0.1, f"max error {max_err:.4f}"


def test_direction():
    """Output direction matches float RMSNorm."""
    torch.manual_seed(1)
    x = torch.randn(4, 32, dtype=torch.float64)
    gamma = torch.ones(32, dtype=torch.float64) * 0.5

    y_ref = _float_rmsnorm(x, gamma)
    x_fix = torch.from_numpy(to_fixed(x.numpy()))
    g_fix = torch.from_numpy(to_fixed(gamma.numpy()))
    y_int = torch.from_numpy(from_fixed(rmsnorm_int32(x_fix, g_fix).numpy()))

    # cosine similarity per row
    for i in range(4):
        cos = torch.cosine_similarity(y_ref[i], y_int[i], dim=0)
        assert cos > 0.95, f"row {i} cosine {cos:.4f}"
