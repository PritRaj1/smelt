import torch

from ._clib import load_lib

FRAC = 16
ONE = 1 << FRAC
LOG2E_FIX = 94548  # log2(e) in Q16.16

_EXP2_LUT = torch.tensor(
    [round((2 ** (i / 256)) * ONE) for i in range(256)],
    dtype=torch.int64,
)


def softmax_int32(x):
    """Integer-only softmax. Q16.16 int32 in, Q16.16 int32 out."""
    lib = load_lib()
    return lib.softmax(x.contiguous())
