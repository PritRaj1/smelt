import torch

from smelt.plac import PLACFunc


def _silu(x):
    return torch.nn.functional.silu(x.float()).to(x.dtype)


def _gelu(x):
    return torch.nn.functional.gelu(x.float()).to(x.dtype)


def test_silu_accuracy():
    """PLAC SiLU within 2x target MAE."""
    plac = PLACFunc(_silu, -8, 8, target_mae=1e-2)
    assert plac.max_error(_silu) < 2e-2


def test_gelu_accuracy():
    """PLAC GELU within 2x target MAE."""
    plac = PLACFunc(_gelu, -8, 8, target_mae=1e-2)
    assert plac.max_error(_gelu) < 2e-2
