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
    srcs = [_CSRC_DIR / "ternary_gemm.c", _CSRC_DIR / "plac_eval.c"]
    if not so.exists() and all(s.exists() for s in srcs):
        cmd = ["gcc", "-O3", "-march=native", "-shared", "-fPIC", "-o", str(so)]
        subprocess.run(cmd + [str(s) for s in srcs], check=True)
    if so.exists():
        _LIB = ctypes.CDLL(str(so))
        _LIB.ternary_gemm.restype = None
        _LIB.ternary_gemm.argtypes = [
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_int,
        ]
        _LIB.plac_eval_int.restype = None
        _LIB.plac_eval_int.argtypes = [
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_void_p,
        ]
    return _LIB
