from pathlib import Path

_LIB = None
_CSRC_DIR = Path(__file__).resolve().parent / "csrc"


def load_lib():
    global _LIB
    if _LIB is not None:
        return _LIB

    from torch.utils.cpp_extension import load

    _LIB = load(
        name="smelt_kernels",
        sources=[
            str(_CSRC_DIR / "torch_binding.cpp"),
            str(_CSRC_DIR / "ternary_gemm.c"),
            str(_CSRC_DIR / "plac_eval.c"),
            str(_CSRC_DIR / "softmax_int.c"),
            str(_CSRC_DIR / "norm_int.c"),
            str(_CSRC_DIR / "int8_gemm.c"),
            str(_CSRC_DIR / "int_ops.c"),
        ],
        extra_cflags=["-O3", "-march=native", "-fopenmp"],
        extra_ldflags=["-lgomp"],
        verbose=False,
    )
    return _LIB
