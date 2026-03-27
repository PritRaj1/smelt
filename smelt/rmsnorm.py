import warnings

import numpy as np
import torch

from ._clib import load_lib


def rmsnorm_int32(x, gamma):
    """
    Int-only RMSNorm. Q16.16 int32 in/out.

    x: int32 tensor [..., dim]
    gamma: int32 tensor [dim], learned scale in Q16.16
    """
    lib = load_lib()
    if lib is not None:
        return _rmsnorm_c(x, gamma, lib)

    warnings.warn("C kernel unavailable, using torch fallback for RMSNorm", stacklevel=2)
    return _rmsnorm_torch(x, gamma)


def _rmsnorm_c(x, gamma, lib):
    orig_shape = x.shape
    dim = x.shape[-1]
    x_2d = x.reshape(-1, dim)
    rows = x_2d.shape[0]

    x_np = np.ascontiguousarray(x_2d.numpy())
    g_np = np.ascontiguousarray(gamma.numpy())
    y_np = np.empty_like(x_np)

    lib.rmsnorm_int_batched(
        x_np.ctypes.data,
        g_np.ctypes.data,
        y_np.ctypes.data,
        rows,
        dim,
    )
    return torch.from_numpy(y_np).reshape(orig_shape)


def _rmsnorm_torch(x, gamma):
    """Torch fallback using int64 arithmetic."""
    frac = 16
    x64 = x.to(torch.int64)

    mean_sq = (x64 * x64 >> frac).sum(dim=-1, keepdim=True) // x.shape[-1]

    # approx 1/sqrt via float (fallback only)
    inv_rms = (1.0 / (mean_sq.float() / (1 << frac)).sqrt().clamp(min=1e-6) * (1 << frac)).to(
        torch.int32
    )

    return ((x64 * inv_rms.to(torch.int64) >> frac) * gamma.to(torch.int64) >> frac).to(torch.int32)
