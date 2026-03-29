from ._clib import load_lib


def softmax_int32(x):
    """Integer-only softmax. Q16.16 int32 in, Q16.16 int32 out."""
    lib = load_lib()
    return lib.softmax(x.contiguous())
