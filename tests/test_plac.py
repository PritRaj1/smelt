import pytest
import torch

from smelt.plac import PLACFunc

_ACTS = {
    "silu": lambda x: torch.nn.functional.silu(x.float()).to(x.dtype),
    "gelu": lambda x: torch.nn.functional.gelu(x.float()).to(x.dtype),
}


@pytest.mark.parametrize("name", _ACTS.keys())
def test_accuracy(name):
    fn = _ACTS[name]
    plac = PLACFunc(fn, -8, 8, target_mae=1e-2)
    assert plac.max_error(fn) < 2e-2
