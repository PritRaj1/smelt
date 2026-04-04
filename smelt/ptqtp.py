import torch
import torch.nn as nn

from .matmul import pack_tl1

# all 9 (c1, c2) combos for discrete search
_C = torch.tensor([[-1, -1], [-1, 0], [-1, 1], [0, -1], [0, 0], [0, 1], [1, -1], [1, 0], [1, 1]])


def quantize_ptqtp(w, max_iter=30, eps=1e-4, lam_init=1e-8):
    """Decompose W into two trit planes via alternating ridge + discrete search."""
    n, d = w.shape
    w = w.float()
    c = _C.to(w.device)
    c1, c2 = c[:, 0].view(9, 1, 1), c[:, 1].view(9, 1, 1)

    # init T1 from absmean, T2 from residual
    s1 = w.abs().mean(1, keepdim=True).clamp(min=1e-10)
    t1 = (w / s1).round().clamp(-1, 1).to(torch.int8)
    res = w - s1 * t1.float()
    s2 = res.abs().mean(1, keepdim=True).clamp(min=1e-10)
    t2 = (res / s2).round().clamp(-1, 1).to(torch.int8)
    al = torch.stack([s1.squeeze(1), s2.squeeze(1)], dim=1)
    lam = torch.full((n,), lam_init, device=w.device)

    for _ in range(max_iter):
        al_prev = al.clone()
        f1, f2 = t1.float(), t2.float()

        # ridge regression (full row)
        s1s1, s2s2, s1s2 = (f1 * f1).sum(1), (f2 * f2).sum(1), (f1 * f2).sum(1)
        s1w, s2w = (f1 * w).sum(1), (f2 * w).sum(1)
        a00, a01, a11 = s1s1 + lam, s1s2, s2s2 + lam
        det = (a00 * a11 - a01 * a01).clamp(min=1e-12)
        al[:, 0] = (a11 * s1w - a01 * s2w) / det
        al[:, 1] = (a00 * s2w - a01 * s1w) / det

        kappa = (a00 + a11).square() / (4 * det)
        lam = torch.where(kappa > 1e6, (lam * 10).clamp(max=1.0), lam)

        # 9-way discrete search in column chunks to bound memory
        al0 = al[:, 0:1]
        al1 = al[:, 1:2]
        for g0 in range(0, d, 256):
            g1 = min(g0 + 256, d)
            wg = w[:, g0:g1]
            err = (wg.unsqueeze(0) - al0 * c1 - al1 * c2).square()
            best = err.argmin(0)
            t1[:, g0:g1] = c[best, 0].to(torch.int8)
            t2[:, g0:g1] = c[best, 1].to(torch.int8)

        if (al - al_prev).abs().max() < eps:
            break

    return t1, t2, al[:, 0], al[:, 1]


class DualTernaryLinear(nn.Module):
    """Two TL1 ternary planes: y = (x @ T1^T) * a1 + (x @ T2^T) * a2."""

    def __init__(self, linear):
        super().__init__()
        wd = linear.weight.data
        if torch.cuda.is_available():
            wd = wd.cuda()

        t1, t2, a1, a2 = quantize_ptqtp(wd)
        t1, t2, a1, a2 = t1.cpu(), t2.cpu(), a1.cpu(), a2.cpu()

        p1, np1, npad1 = pack_tl1(t1)
        p2, _, _ = pack_tl1(t2)

        self.register_buffer("w1", p1.contiguous())
        self.register_buffer("w2", p2.contiguous())
        self.register_buffer("a1", a1.float())
        self.register_buffer("a2", a2.float())
        self.n_pairs = np1
        self.n_padded = npad1
        self.in_features = linear.in_features
        self.out_features = linear.out_features
        self.bias = linear.bias

    def forward(self, x):
        orig = x.shape
        x_2d = x.reshape(-1, self.in_features).contiguous()
        ops = torch.ops.smelt
        y1 = ops.ternary_linear(x_2d, self.w1, self.n_padded, self.n_pairs, self.out_features, 1.0)
        y2 = ops.ternary_linear(x_2d, self.w2, self.n_padded, self.n_pairs, self.out_features, 1.0)
        y = y1 * self.a1 + y2 * self.a2

        if self.bias is not None:
            y = y + self.bias

        return y.reshape(*orig[:-1], self.out_features)
