import torch
import torch.nn as nn

from ._clib import load_lib
from .qtensor import SCALE, QTensor

# shared quantize cache: if multiple TernaryLinears see the same input, quantize once
_quant_cache = {"ptr": None, "i8": None, "scale": None}


def _is_already_ternary(w):
    mn, mx = w.min().item(), w.max().item()
    if not torch.equal(w, w.round()):
        return False

    if mn >= -1 and mx <= 1:
        return True

    return mn == 0 and mx in (1, 255)


def _decode_uint8_ternary(w):
    w = w.to(torch.int16)
    w[w == 255] = -1
    return w.to(torch.int8)


def quantize_ternary(w):
    """Quantize to {-1, 0, +1}. Detects already-ternary weights (BitNet)."""
    if _is_already_ternary(w):
        w_t = _decode_uint8_ternary(w) if w.max() > 1 else w.to(torch.int8)
        return w_t, torch.tensor(1.0)

    scale = w.abs().mean()
    w_t = (w / (scale + 1e-10)).round().clamp(-1, 1).to(torch.int8)
    return w_t, scale


def pack_tl1(w_t):
    """Pack ternary into TL1 nibble format, transposed for vpshufb."""
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
    """Per-token absmax to int8."""
    scale = 127.0 / x.abs().amax(dim=-1, keepdim=True).clamp(min=1e-5)
    x_q = (x * scale).round().clamp(-128, 127).to(torch.int8)
    return x_q, scale


def _get_cached_quant(x_2d):
    """Return cached (int8, scale) if same tensor. Scale is scalar for m=1, None for m>1."""
    ptr = x_2d.data_ptr()
    if _quant_cache["ptr"] == ptr:
        return _quant_cache["i8"], _quant_cache["scale"]

    x_i8, x_s = quantize_activations(x_2d)
    x_i8 = x_i8.contiguous()
    _quant_cache["ptr"] = ptr
    _quant_cache["i8"] = x_i8
    _quant_cache["scale"] = x_s.squeeze().item() if x_2d.size(0) == 1 else None
    return x_i8, _quant_cache["scale"]


class TernaryLinear(nn.Module):
    """Drop-in nn.Linear replacement. Caches input quantize for shared-input calls."""

    def __init__(self, linear):
        super().__init__()
        w_t, scale = quantize_ternary(linear.weight.data.float())
        if hasattr(linear, "weight_scale"):
            scale = linear.weight_scale.float().squeeze()

        packed, n_pairs, n_padded = pack_tl1(w_t)
        self.register_buffer("w_packed", packed.contiguous())
        self.register_buffer("w_scale", scale)
        self.in_features = linear.in_features
        self.out_features = linear.out_features
        self.n_pairs = n_pairs
        self.n_padded = n_padded
        self.bias = linear.bias

    def forward(self, x):
        orig = x.shape
        lib = load_lib()
        ws = self.w_scale.item()

        # QTensor input: int quantize (pure int, no float conversion)
        if isinstance(x, QTensor):
            x_2d = x.int_data.reshape(-1, self.in_features).contiguous()
            x_i8, mx = lib.int_quantize(x_2d)
            rf = round(ws * mx.item() / 127.0)
            y_i32 = lib.ternary_gemm(x_i8, self.w_packed, self.n_padded, self.n_pairs)
            y_q16 = lib.int_rescale(y_i32[:, : self.out_features], rf)
            if self.bias is not None:
                y_q16 = y_q16 + (self.bias * SCALE).to(torch.int32)

            return QTensor(y_q16.reshape(*orig[:-1], self.out_features))

        # float input: existing path, return QTensor
        x_2d = x.reshape(-1, self.in_features).contiguous()
        x_i8, act_scale = _get_cached_quant(x_2d)

        # m=1: scalar scale -> int rescale
        if act_scale is not None:
            y_i32 = lib.ternary_gemm(x_i8, self.w_packed, self.n_padded, self.n_pairs)
            rf = round(ws / act_scale * SCALE)
            y_q16 = lib.int_rescale(y_i32[:, : self.out_features], rf)
            if self.bias is not None:
                y_q16 = y_q16 + (self.bias * SCALE).to(torch.int32)
            return QTensor(y_q16.reshape(*orig[:-1], self.out_features))

        # m>1: per-row float scales, use float path then wrap
        y = lib.ternary_linear(
            x_2d,
            self.w_packed,
            self.n_padded,
            self.n_pairs,
            self.out_features,
            ws,
        )
        if self.bias is not None:
            y = y + self.bias
        return QTensor((y * SCALE).to(torch.int32).reshape(*orig[:-1], self.out_features))
