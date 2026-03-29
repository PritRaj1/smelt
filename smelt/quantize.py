import warnings

import numpy as np
import torch
import torch.nn as nn

from ._clib import load_lib


def _is_already_ternary(w):
    """Check if w already {-1, 0, +1} or uint8 {0, 1, 255}."""
    unique = w.unique()
    if len(unique) <= 3:
        vals = set(unique.tolist())
        return vals <= {-1, 0, 1} or vals <= {0, 1, 255}

    return False


def _decode_uint8_ternary(w):
    """Convert uint8 {0, 1, 255} to int8 {0, +1, -1}."""
    w = w.to(torch.int16)
    w[w == 255] = -1
    return w.to(torch.int8)


def quantize_ternary(w):
    """
    Quantize to {-1, 0, +1}. Skips already-ternary weights (e.g. BitNet).

    Returns (w_ternary, scale) where w ~= scale * w_ternary.
    """
    if _is_already_ternary(w):
        w_t = _decode_uint8_ternary(w) if w.max() > 1 else w.to(torch.int8)
        scale = torch.tensor(1.0)
        return w_t, scale

    scale = w.abs().mean()
    w_t = (w / (scale + 1e-10)).round().clamp(-1, 1).to(torch.int8)
    return w_t, scale


def pack_tl1(w_t):
    """
    Pack ternary weights into TL1 format, transposed for vpshufb.

    Index = (w0+1)*3 + (w1+1) per pair. Two 4-bit indices per byte.
    Layout: [n_pairs, n_padded/2], k-pairs along rows, columns along cols.
    """
    n, k = w_t.shape

    if k % 2:
        w_t = torch.nn.functional.pad(w_t, (0, 1))
        k = k + 1

    pairs = w_t.reshape(n, k // 2, 2)
    idx = (pairs[:, :, 0].to(torch.int16) + 1) * 3 + (pairs[:, :, 1].to(torch.int16) + 1)
    idx = idx.to(torch.uint8)
    n_pairs = idx.shape[1]

    n_pad = (32 - n % 32) % 32
    if n_pad:
        idx = torch.nn.functional.pad(idx, (0, 0, 0, n_pad), value=4)
    n_padded = idx.shape[0]

    idx_t = idx.T.contiguous()
    even = idx_t[:, 0::2]
    odd = idx_t[:, 1::2]
    packed = (even | (odd << 4)).to(torch.uint8)

    return packed, n_pairs, n_padded


def unpack_tl1(packed, n_pairs, n_padded, n, k):
    """Unpack TL1 transposed format to ternary {-1, 0, +1}."""
    even = (packed & 0x0F).to(torch.int16)
    odd = ((packed >> 4) & 0x0F).to(torch.int16)

    idx_t = torch.zeros(n_pairs, n_padded, dtype=torch.int16)
    idx_t[:, 0::2] = even
    idx_t[:, 1::2] = odd
    idx = idx_t.T[:n, :]

    w0 = (idx // 3 - 1).to(torch.int8)
    w1 = (idx % 3 - 1).to(torch.int8)
    w = torch.stack([w0, w1], dim=2).reshape(n, -1)
    return w[:, :k]


def quantize_activations(x):
    """Per-token absmax to int8. Returns (x_int8, scale) where x ~= x_int8 / scale."""
    scale = 127.0 / x.abs().amax(dim=-1, keepdim=True).clamp(min=1e-5)
    x_q = (x * scale).round().clamp(-128, 127).to(torch.int8)
    return x_q, scale


def _ternary_gemm(x_np, w_np, m, n, n_padded, k, n_pairs):
    y_np = np.empty((m, n_padded), dtype=np.int32)

    lib = load_lib()
    if lib is not None:
        lib.ternary_gemm(
            x_np.ctypes.data,
            w_np.ctypes.data,
            y_np.ctypes.data,
            m,
            n_padded,
            k,
            n_pairs,
        )
        return torch.from_numpy(y_np[:, :n])

    warnings.warn(
        "C kernel unavailable, using slow torch fallback for ternary GEMM",
        stacklevel=2,
    )
    return None


class TernaryLinear(nn.Module):
    """Drop-in nn.Linear replacement. Ternary weights, int8 activations, int32 accumulator."""

    def __init__(self, linear):
        super().__init__()

        w_t, scale = quantize_ternary(linear.weight.data.float())
        packed, n_pairs, n_padded = pack_tl1(w_t)

        self.register_buffer("w_packed", packed)
        self.register_buffer("w_scale", scale)
        self.in_features = linear.in_features
        self.out_features = linear.out_features
        self.n_pairs = n_pairs
        self.n_padded = n_padded
        self.bias = linear.bias

        self._w_np = np.ascontiguousarray(packed.numpy())
        self._w_t = w_t  # for fallback

    def forward(self, x):
        orig_shape = x.shape
        x_2d = x.reshape(-1, self.in_features)
        x_q, x_scale = quantize_activations(x_2d)
        x_np = np.ascontiguousarray(x_q.numpy())

        y_int32 = _ternary_gemm(
            x_np,
            self._w_np,
            x_2d.shape[0],
            self.out_features,
            self.n_padded,
            self.in_features,
            self.n_pairs,
        )

        if y_int32 is None:
            y_int32 = x_q.to(torch.int32) @ self._w_t.to(torch.int32).T

        # x ~= x_q / x_scale, w ~= w_t * w_scale
        y = y_int32.float() * self.w_scale / x_scale

        if self.bias is not None:
            y = y + self.bias

        return y.reshape(*orig_shape[:-1], self.out_features)
