import math
import warnings

import numpy as np
import torch

from ._clib import load_lib

FRAC_BITS = 16
SCALE = 1 << FRAC_BITS
LUT_SIZE = 4096


def to_fixed(x):
    """Float to Q16.16 int32."""
    if isinstance(x, torch.Tensor):
        x = x.numpy()
    return np.round(np.asarray(x, dtype=np.float64) * SCALE).astype(np.int32)


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


def _build_lut(f, x_lo, x_hi, n_terms):
    """Build dense LUT from PLAC segments."""
    bp = auto_segment(f, x_lo, x_hi, target_mae=1e-3, n_terms=n_terms)
    slopes, intercepts, terms = fit_pwl(f, bp, n_terms)

    x_lo_fix = to_fixed(np.array([x_lo]))[0]
    x_hi_fix = to_fixed(np.array([x_hi]))[0]
    range_fix = x_hi_fix - x_lo_fix

    # shift so that (x - x_lo) >> shift maps to [0, LUT_SIZE)
    shift = 0
    while (range_fix >> shift) > LUT_SIZE:
        shift += 1

    lut = np.empty(LUT_SIZE, dtype=np.int32)
    bp_inner = np.array(bp[1:])
    intercepts_arr = np.array(intercepts)

    for i in range(LUT_SIZE):
        x_fix = x_lo_fix + (i << shift) + (1 << (shift - 1) if shift > 0 else 0)
        x_float = x_fix / SCALE
        seg = int(np.searchsorted(bp_inner, x_float).clip(0, len(slopes) - 1))

        acc = 0
        for sign, exp in terms[seg]:
            shifted = x_fix << exp if exp >= 0 else x_fix >> (-exp)
            acc += sign * shifted
        lut[i] = np.int32(acc + to_fixed(np.array([intercepts_arr[seg]]))[0])

    return np.ascontiguousarray(lut), x_lo_fix, shift, bp, slopes, intercepts, terms


class PLACFunc:
    """Piecewise linear approx with shift-constrained slopes. Dense LUT eval."""

    def __init__(self, f, x_lo, x_hi, target_mae=1e-3, n_terms=2):
        lut, x_lo_fix, shift, bp, slopes, _intercepts, terms = _build_lut(f, x_lo, x_hi, n_terms)

        self._lut = lut
        self._x_lo_fix = x_lo_fix
        self._shift = shift
        self.breakpoints_f = bp
        self.terms = terms
        self.n_segments = len(slopes)
        self.n_terms = n_terms

    def __call__(self, x):
        """Evaluate. Accepts float numpy/torch."""
        is_torch = isinstance(x, torch.Tensor)
        if is_torch:
            device, dtype = x.device, x.dtype
            x = x.detach().cpu().numpy()

        orig_shape = x.shape
        y_fix = self._eval_lut(to_fixed(x.ravel()))
        y = from_fixed(y_fix).reshape(orig_shape)

        if is_torch:
            return torch.from_numpy(y).to(device=device, dtype=dtype)

        return y

    def eval_int32(self, x_int32):
        """Evaluate directly on int32 array."""
        return self._eval_lut(np.ascontiguousarray(x_int32))

    def _eval_lut(self, x_fix):
        n = x_fix.size
        y = np.empty(n, dtype=np.int32)

        lib = load_lib()
        if lib is not None:
            lib.plac_eval_lut(
                x_fix.ctypes.data,
                y.ctypes.data,
                n,
                self._lut.ctypes.data,
                LUT_SIZE,
                self._x_lo_fix,
                self._shift,
            )
            return y

        warnings.warn("C kernel unavailable, using numpy fallback for PLAC", stacklevel=3)
        idx = ((x_fix - self._x_lo_fix) >> self._shift).clip(0, LUT_SIZE - 1)
        return self._lut[idx]

    def max_error(self, f, n_samples=10000):
        x = np.linspace(self.breakpoints_f[0], self.breakpoints_f[-1], n_samples)
        y_ref = f(torch.tensor(x, dtype=torch.float64)).numpy()
        return np.max(np.abs(y_ref - self(x)))


def relu2_int32(x):
    """Squared ReLU in Q16.16: max(0, x)^2 >> FRAC."""
    x = x.to(torch.int64).clamp(min=0)
    return (x * x >> FRAC_BITS).to(torch.int32)


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
