import math

import numpy as np
import torch

FRAC_BITS = 16
SCALE = 1 << FRAC_BITS


def to_fixed(x):
    """Float to Q16.16 int32. Clamps to int32 range."""
    if isinstance(x, torch.Tensor):
        x = x.numpy()

    val = np.round(np.asarray(x, dtype=np.float64) * SCALE)
    return np.clip(val, -(1 << 31), (1 << 31) - 1).astype(np.int32)


def from_fixed(x):
    """Q16.16 int32 to float."""
    return x.astype(np.float64) / SCALE


def quantize_slope(s, n_terms=2, min_exp=-16, max_exp=4):
    """
    Approx slope as sum of signed powers of 2.

    Returns (terms, approx_value) where terms = [(sign, exp), ...].
    """
    terms = []
    remaining = s

    for _ in range(n_terms):
        if abs(remaining) < 2.0 ** (min_exp - 1):
            break

        sign = 1 if remaining > 0 else -1
        log_val = math.log2(abs(remaining))
        fl, ce = math.floor(log_val), math.ceil(log_val)
        candidates = [e for e in [fl, ce] if min_exp <= e <= max_exp]
        if not candidates:
            break

        best_exp = min(candidates, key=lambda e: abs(abs(remaining) - 2.0**e))
        terms.append((sign, best_exp))
        remaining -= sign * 2.0**best_exp

    val = sum(sgn * 2.0**exp for sgn, exp in terms) if terms else 0.0
    return terms, val


def _call_f(f, x):
    result = f(torch.tensor(x, dtype=torch.float64))
    return result.item() if isinstance(result, torch.Tensor) else result


def fit_segment(f, x0, x1, n_terms=2, n_samples=1000):
    """Fit one segment: quantized slope + Chebyshev-optimal intercept."""
    s_ideal = (_call_f(f, x1) - _call_f(f, x0)) / (x1 - x0) if x1 != x0 else 0.0
    terms, s_q = quantize_slope(s_ideal, n_terms)
    xs = torch.linspace(x0, x1, n_samples, dtype=torch.float64)
    residuals = f(xs) - s_q * xs
    b = (residuals.max().item() + residuals.min().item()) / 2
    return terms, s_q, b


def fit_pwl(f, breakpoints, n_terms=2):
    """Fit PWL with quantized slopes. Returns (slopes, intercepts, terms)."""
    slopes, intercepts, all_terms = [], [], []
    for i in range(len(breakpoints) - 1):
        terms, s_q, b = fit_segment(f, breakpoints[i], breakpoints[i + 1], n_terms)
        slopes.append(s_q)
        intercepts.append(b)
        all_terms.append(terms)
    return slopes, intercepts, all_terms


def auto_segment(f, x_lo, x_hi, target_mae, n_terms=2, tol=1e-6):
    """Find fewest segments to approximate f within target_mae."""
    breakpoints = [x_lo]
    x = x_lo
    while x < x_hi - tol:
        lo, hi = x, x_hi
        for _ in range(60):
            mid = (lo + hi) / 2
            if mid - x < tol:
                break

            _, s_q, b = fit_segment(f, x, mid, n_terms)
            xs = torch.linspace(x, mid, 500, dtype=torch.float64)
            err = torch.max(torch.abs(f(xs) - (s_q * xs + b))).item()
            if err <= target_mae:
                lo = mid

            else:
                hi = mid

        if lo - x < tol:
            lo = min(x + (x_hi - x_lo) / 1000, x_hi)

        breakpoints.append(lo)
        x = lo

    breakpoints[-1] = x_hi
    return breakpoints


class PLACFunc:
    """Piecewise linear approx with shift-constrained slopes. Direct segment eval."""

    def __init__(self, f, x_lo, x_hi, target_mae=1e-3, n_terms=2):
        bp = auto_segment(f, x_lo, x_hi, target_mae, n_terms)
        slopes, intercepts, terms = fit_pwl(f, bp, n_terms)

        self.breakpoints_f = bp
        self.terms = terms
        self.n_segments = len(slopes)
        self.n_terms = n_terms

        # segment arrays for AVX2 eval
        self._breakpoints = torch.from_numpy(to_fixed(np.array(bp))).contiguous()
        self._intercepts = torch.from_numpy(to_fixed(np.array(intercepts))).contiguous()
        signs = np.zeros(len(slopes) * 2, dtype=np.int32)
        exps = np.zeros(len(slopes) * 2, dtype=np.int32)
        for i, seg_terms in enumerate(terms):
            for t, (sign, exp) in enumerate(seg_terms):
                if t < 2:
                    signs[i * 2 + t] = sign
                    exps[i * 2 + t] = exp

        self._signs = torch.from_numpy(signs).contiguous()
        self._exps = torch.from_numpy(exps).contiguous()

    def __call__(self, x):
        is_np = not isinstance(x, torch.Tensor)
        if is_np:
            x = torch.from_numpy(np.asarray(x, dtype=np.float32))
        x_fix = torch.from_numpy(to_fixed(x.float().numpy())).contiguous()
        y_fix = torch.ops.smelt.plac_int32(
            x_fix,
            self._breakpoints,
            self._intercepts,
            self._signs,
            self._exps,
            self.n_segments,
        )
        y = y_fix.float() / SCALE
        return y.numpy() if is_np else y.to(dtype=x.dtype)

    def max_error(self, f, n_samples=10000):
        x = np.linspace(self.breakpoints_f[0], self.breakpoints_f[-1], n_samples)
        y_ref = f(torch.tensor(x, dtype=torch.float64)).numpy()
        return np.max(np.abs(y_ref - self(x)))


def terms_to_str(terms):
    """String shift terms, e.g. 'x>>1 + x>>3'."""
    parts = []
    for i, (sign, exp) in enumerate(terms):
        shift = f"x>>{-exp}" if exp < 0 else ("x" if exp == 0 else f"x<<{exp}")
        if i == 0:
            parts.append(f"-{shift}" if sign < 0 else shift)

        else:
            parts.append(f" - {shift}" if sign < 0 else f" + {shift}")

    return "".join(parts) if parts else "0"
