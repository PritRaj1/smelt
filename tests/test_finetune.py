import torch
import torch.nn as nn

from smelt.finetune import _STELinear, freeze, prepare_qat
from smelt.matmul import TernaryLinear


def test_ste_matches_ternary():
    """STE forward approxes TernaryLinear output."""
    torch.manual_seed(0)
    linear = nn.Linear(64, 32, bias=False)
    x = torch.randn(4, 64)

    y_ste = _STELinear(linear)(x)
    y_tl = TernaryLinear(linear)(x)

    nmse = ((y_ste - y_tl) ** 2).mean() / (y_tl**2).mean()
    assert nmse < 0.1, f"NMSE {nmse:.3f} too high"


def test_prep_and_freeze():
    """Prepare -> train step -> freeze -> verify."""
    torch.manual_seed(0)
    model = nn.Sequential(nn.Linear(32, 16), nn.ReLU(), nn.Linear(16, 8))
    x = torch.randn(2, 32)

    prepare_qat(model, skip=[])
    assert isinstance(model[0], _STELinear)

    loss = model(x).sum()
    loss.backward()
    with torch.no_grad():
        for p in model.parameters():
            if p.grad is not None:
                p -= 0.01 * p.grad

    freeze(model, skip=[])
    assert isinstance(model[0], TernaryLinear)
    assert model(x).shape == (2, 8)
