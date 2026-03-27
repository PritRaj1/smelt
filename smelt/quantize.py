import torch
import torch.nn as nn

_POWERS = torch.tensor([128, 64, 32, 16, 8, 4, 2, 1], dtype=torch.uint8)


def quantize_ternary(w):
    """
    Quantize to {-1, 0, +1} via per-tensor absmean (BitNet b1.58).

    Returns (w_ternary, scale) where w ≈ scale * w_ternary.
    """
    scale = w.abs().mean()
    w_t = (w / (scale + 1e-10)).round().clamp(-1, 1).to(torch.int8)
    return w_t, scale


def pack_ternary(w_t):
    """Pack ternary weights into pos/neg uint8 bitmasks (8 weights per byte)."""
    out_dim, in_dim = w_t.shape
    pad = (8 - in_dim % 8) % 8
    if pad:
        w_t = torch.nn.functional.pad(w_t, (0, pad))

    w_flat = w_t.reshape(out_dim, -1, 8)
    pos = ((w_flat == 1).to(torch.uint8) * _POWERS).sum(dim=-1).to(torch.uint8)
    neg = ((w_flat == -1).to(torch.uint8) * _POWERS).sum(dim=-1).to(torch.uint8)
    return pos, neg


def unpack_ternary(pos_mask, neg_mask, in_dim):
    """Unpack bitmasks back to ternary {-1, 0, +1} weight matrix."""
    out_dim = pos_mask.shape[0]
    pos_bits = (pos_mask.unsqueeze(-1) & _POWERS).ne(0).to(torch.int8)
    neg_bits = (neg_mask.unsqueeze(-1) & _POWERS).ne(0).to(torch.int8)
    return (pos_bits - neg_bits).reshape(out_dim, -1)[:, :in_dim]


def ternary_forward(x, pos_mask, neg_mask, scale, in_dim):
    """Multiply-free forward: y = scale * (x @ W_pos.T - x @ W_neg.T)."""
    w_t = unpack_ternary(pos_mask, neg_mask, in_dim).to(x.dtype)
    pos = (w_t == 1).to(x.dtype)
    neg = (w_t == -1).to(x.dtype)
    return scale * (x @ pos.T - x @ neg.T)


class TernaryLinear(nn.Module):
    """Drop-in replacement for nn.Linear with ternary weights."""

    def __init__(self, linear):
        super().__init__()
        w_t, scale = quantize_ternary(linear.weight.data.float())
        pos, neg = pack_ternary(w_t)
        self.register_buffer("pos_mask", pos)
        self.register_buffer("neg_mask", neg)
        self.register_buffer("scale", scale)
        self.in_features = linear.in_features
        self.bias = linear.bias

    def forward(self, x):
        y = ternary_forward(x, self.pos_mask, self.neg_mask, self.scale, self.in_features)
        if self.bias is not None:
            y = y + self.bias

        return y
