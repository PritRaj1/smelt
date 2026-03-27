import ctypes
import subprocess
from pathlib import Path

_LIB = None
_CSRC_DIR = Path(__file__).resolve().parent / "csrc"


def load_lib():
    global _LIB
    if _LIB is not None:
        return _LIB

    so = _CSRC_DIR / "smelt_kernels.so"
    srcs = [
        _CSRC_DIR / "ternary_gemm.c",
        _CSRC_DIR / "plac_eval.c",
        _CSRC_DIR / "softmax_int.c",
        _CSRC_DIR / "norm_int.c",
    ]
    if not so.exists() and all(s.exists() for s in srcs):
        cmd = [
            "gcc",
            "-O3",
            "-march=native",
            "-flto",
            "-funroll-loops",
            "-fopenmp",
            "-shared",
            "-fPIC",
            "-o",
            str(so),
        ]
        subprocess.run(cmd + [str(s) for s in srcs], check=True)

    if so.exists():
        _LIB = ctypes.CDLL(str(so))
        _LIB.ternary_gemm.restype = None
        _LIB.ternary_gemm.argtypes = [
            ctypes.c_void_p,  # x
            ctypes.c_void_p,  # w_tl1
            ctypes.c_void_p,  # y
            ctypes.c_int,  # m
            ctypes.c_int,  # n_padded
            ctypes.c_int,  # k
            ctypes.c_int,  # n_pairs
        ]
        _LIB.plac_eval_lut.restype = None
        _LIB.plac_eval_lut.argtypes = [
            ctypes.c_void_p,  # x
            ctypes.c_void_p,  # y
            ctypes.c_int,  # n
            ctypes.c_void_p,  # lut
            ctypes.c_int,  # lut_size
            ctypes.c_int,  # x_lo
            ctypes.c_int,  # shift
        ]
        _LIB.softmax_int.restype = None
        _LIB.softmax_int.argtypes = [
            ctypes.c_void_p,  # x
            ctypes.c_void_p,  # y
            ctypes.c_int,  # rows
            ctypes.c_int,  # cols
        ]
        _LIB.rmsnorm_int_batched.restype = None
        _LIB.rmsnorm_int_batched.argtypes = [
            ctypes.c_void_p,  # x
            ctypes.c_void_p,  # gamma
            ctypes.c_void_p,  # y
            ctypes.c_int,  # rows
            ctypes.c_int,  # cols
        ]
        _LIB.layernorm_int_batched.restype = None
        _LIB.layernorm_int_batched.argtypes = [
            ctypes.c_void_p,  # x
            ctypes.c_void_p,  # gamma
            ctypes.c_void_p,  # beta
            ctypes.c_void_p,  # y
            ctypes.c_int,  # rows
            ctypes.c_int,  # cols
        ]
    return _LIB
