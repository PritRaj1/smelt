from ._clib import load_lib


def rmsnorm_int32(x, gamma):
    """Int-only RMSNorm. Q16.16 int32 in/out."""
    lib = load_lib()
    return lib.rmsnorm(x.contiguous(), gamma.contiguous())


def layernorm_int32(x, gamma, beta):
    """Int-only LayerNorm. Q16.16 int32 in/out."""
    lib = load_lib()
    return lib.layernorm(x.contiguous(), gamma.contiguous(), beta.contiguous())
