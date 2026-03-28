import torch

from smelt.plac import to_fixed
from smelt.softmax import softmax_int32


def test_sums_to_one():
    """Softmax output sums to ~1.0 in fixed-point."""
    x = torch.tensor(to_fixed([1.0, 2.0, 3.0, 4.0]), dtype=torch.int32)
    y = softmax_int32(x)
    total = y.sum().item() / (1 << 16)
    assert abs(total - 1.0) < 0.005


def test_argmax():
    """Integer softmax preserves argmax of float softmax."""
    torch.manual_seed(0)
    x_float = torch.randn(64, dtype=torch.float64)
    x_fix = torch.tensor(to_fixed(x_float.numpy()), dtype=torch.int32)

    y_int = softmax_int32(x_fix)
    y_float = torch.softmax(x_float, dim=0)

    assert y_int.argmax() == y_float.argmax()


def test_batched():
    """Batched softmax: each row independent."""
    x = torch.tensor(to_fixed([[1.0, 2.0, 3.0], [3.0, 2.0, 1.0]]), dtype=torch.int32)
    y = softmax_int32(x)
    assert y.shape == (2, 3)
    assert y[0].argmax() == 2
    assert y[1].argmax() == 0
