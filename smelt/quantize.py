import numpy as np
import torch
import torch.nn as nn

from ._clib import load_lib

_POWERS = torch.tensor([1, 2, 4, 8, 16, 32, 64, 128], dtype=torch.uint8)


def quantize_ternary(w):
    """
    Quantize to {-1, 0, +1} via per-tensor absmean (BitNet b1.58).

    Returns (w_ternary, scale) where w ~= scale * w_ternary.
    """
    scale = w.abs().mean()
    w_t = (w / (scale + 1e-10)).round().clamp(-1, 1).to(torch.int8)
    return w_t, scale


def pack_ternary(w_t):
    """
    Pack into val/sign uint8 bitmasks (8 weights per byte).

    val=1 where weight != 0, sign=1 where weight == -1.
    """
    out_dim, in_dim = w_t.shape
    pad = (8 - in_dim % 8) % 8
    if pad:
        w_t = torch.nn.functional.pad(w_t, (0, pad))

    w_flat = w_t.reshape(out_dim, -1, 8)
    val = ((w_flat != 0).to(torch.uint8) * _POWERS).sum(dim=-1).to(torch.uint8)
    sign = ((w_flat == -1).to(torch.uint8) * _POWERS).sum(dim=-1).to(torch.uint8)
    return val, sign


def unpack_ternary(val_mask, sign_mask, in_dim):
    """Unpack val/sign bitmasks to ternary {-1, 0, +1}."""
    out_dim = val_mask.shape[0]
    val_bits = (val_mask.unsqueeze(-1) & _POWERS).ne(0).to(torch.int8)
    sign_bits = (sign_mask.unsqueeze(-1) & _POWERS).ne(0).to(torch.int8)
    # val=1,sign=0 -> +1; val=1,sign=1 -> -1; val=0 -> 0
    return (val_bits * (1 - 2 * sign_bits)).reshape(out_dim, -1)[:, :in_dim]


def quantize_activations(x):
    """Per-token absmax to int8. Returns (x_int8, scale) where x ~= x_int8 / scale."""
    scale = 127.0 / x.abs().amax(dim=-1, keepdim=True).clamp(min=1e-5)
    x_q = (x * scale).round().clamp(-128, 127).to(torch.int8)
    return x_q, scale


def _ternary_gemm(x_np, wv_np, ws_np, m, n, k):
    """Ternary GEMM: int8 activations, bitmask weights -> int32."""
    y_np = np.empty((m, n), dtype=np.int32)

    lib = load_lib()
    if lib is not None:
        lib.ternary_gemm(
            x_np.ctypes.data,
            wv_np.ctypes.data,
            ws_np.ctypes.data,
            y_np.ctypes.data,
            m,
            n,
            k,
        )
        return torch.from_numpy(y_np)

    # torch fallback
    w_val = torch.from_numpy(wv_np)
    w_sign = torch.from_numpy(ws_np)
    x_q = torch.from_numpy(x_np)
    w_t = unpack_ternary(w_val, w_sign, k)
    return x_q.to(torch.int32) @ w_t.to(torch.int32).T


class TernaryLinear(nn.Module):
    """Drop-in nn.Linear replacement. Ternary weights, int8 activations, int32 accumulator."""

    def __init__(self, linear):
        super().__init__()
        w_t, scale = quantize_ternary(linear.weight.data.float())
        val, sign = pack_ternary(w_t)
        self.register_buffer("w_val", val)
        self.register_buffer("w_sign", sign)
        self.register_buffer("w_scale", scale)
        self.in_features = linear.in_features
        self.out_features = linear.out_features
        self.bias = linear.bias

        # pre-convert to numpy once for C kernel
        self._wv_np = np.ascontiguousarray(val.numpy())
        self._ws_np = np.ascontiguousarray(sign.numpy())

    def forward(self, x):
        orig_shape = x.shape
        x_2d = x.reshape(-1, self.in_features)
        x_q, x_scale = quantize_activations(x_2d)
        x_np = np.ascontiguousarray(x_q.numpy())

        y_int32 = _ternary_gemm(
            x_np,
            self._wv_np,
            self._ws_np,
            x_2d.shape[0],
            self.out_features,
            self.in_features,
        )

        # x ~= x_q / x_scale, w ~= w_t * w_scale
        y = y_int32.float() * self.w_scale / x_scale

        if self.bias is not None:
            y = y + self.bias
        return y.reshape(*orig_shape[:-1], self.out_features)
