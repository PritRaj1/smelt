import warnings

import numpy as np
import torch

from ._clib import load_lib


def rmsnorm_int32(x, gamma):
    """Int-only RMSNorm. Q16.16 int32 in/out."""
    lib = load_lib()
    if lib is not None:
        return _norm_c(x, gamma, None, lib, "rmsnorm_int_batched")

    warnings.warn("C kernel unavailable, using torch fallback for RMSNorm", stacklevel=2)
    return _rmsnorm_torch(x, gamma)


def layernorm_int32(x, gamma, beta):
    """Int-only LayerNorm. Q16.16 int32 in/out."""
    lib = load_lib()
    if lib is not None:
        return _norm_c(x, gamma, beta, lib, "layernorm_int_batched")

    warnings.warn("C kernel unavailable, using torch fallback for LayerNorm", stacklevel=2)
    return _layernorm_torch(x, gamma, beta)


def _norm_c(x, gamma, beta, lib, fn_name):
    orig_shape = x.shape
    dim = x.shape[-1]
    x_2d = x.reshape(-1, dim)
    rows = x_2d.shape[0]

    x_np = np.ascontiguousarray(x_2d.numpy())
    g_np = np.ascontiguousarray(gamma.numpy())
    y_np = np.empty_like(x_np)
    fn = getattr(lib, fn_name)

    if beta is not None:
        b_np = np.ascontiguousarray(beta.numpy())
        fn(x_np.ctypes.data, g_np.ctypes.data, b_np.ctypes.data, y_np.ctypes.data, rows, dim)

    else:
        fn(x_np.ctypes.data, g_np.ctypes.data, y_np.ctypes.data, rows, dim)

    return torch.from_numpy(y_np).reshape(orig_shape)


def _rmsnorm_torch(x, gamma):
    frac = 16
    x64 = x.to(torch.int64)
    mean_sq = (x64 * x64 >> frac).sum(dim=-1, keepdim=True) // x.shape[-1]

    # float isqrt (fallback only)
    inv_rms = (1.0 / (mean_sq.float() / (1 << frac)).sqrt().clamp(min=1e-6) * (1 << frac)).to(
        torch.int32
    )
    return ((x64 * inv_rms.to(torch.int64) >> frac) * gamma.to(torch.int64) >> frac).to(torch.int32)


def _layernorm_torch(x, gamma, beta):
    frac = 16
    x64 = x.to(torch.int64)
    mean = x64.sum(dim=-1, keepdim=True) // x.shape[-1]
    centered = x64 - mean
    var = (centered * centered >> frac).sum(dim=-1, keepdim=True) // x.shape[-1]

    # float isqrt (fallback only)
    inv_std = (1.0 / (var.float() / (1 << frac)).sqrt().clamp(min=1e-6) * (1 << frac)).to(
        torch.int32
    )
    out = (centered * inv_std.to(torch.int64) >> frac) * gamma.to(torch.int64) >> frac
    return (out + beta.to(torch.int64)).to(torch.int32)
