import logging

import torch
import torch.nn as nn

from ._clib import load_lib
from .matmul import TernaryLinear, _is_already_ternary, quantize_activations
from .plac import PLACFunc
from .ptqtp import DualTernaryLinear
from .qtensor import SCALE, QTensor

log = logging.getLogger(__name__)

_SKIP_ACTS = {nn.ReLU, nn.Identity, nn.Dropout}
_SKIP_ACT_NAMES = {"relusquared"}


def _is_linear(mod):
    if isinstance(mod, nn.Linear):
        return True
    return type(mod).__name__ == "Conv1D" and hasattr(mod, "nf")


def _conv1d_to_linear(mod):
    linear = nn.Linear(mod.weight.shape[0], mod.nf, bias=mod.bias is not None)
    linear.weight.data = mod.weight.data.T
    if mod.bias is not None:
        linear.bias.data = mod.bias.data
    return linear


def _is_activation(mod):
    if type(mod) in _SKIP_ACTS:
        return False
    if any(s in type(mod).__name__.lower() for s in _SKIP_ACT_NAMES):
        return False
    if len(list(mod.children())) > 0:
        return False
    if len(list(mod.parameters())) > 0:
        return False
    name = type(mod).__name__.lower()
    keys = ("activation", "silu", "gelu", "sigmoid", "tanh", "swish", "relu")
    return any(k in name for k in keys)


def _make_plac(mod, target_mae):
    def fn(x):
        with torch.no_grad():
            return mod(x.float()).to(x.dtype)

    return PLACFunc(fn, -8, 8, target_mae)


class _PLACModule(nn.Module):
    def __init__(self, plac):
        super().__init__()
        self.register_buffer("_breakpoints", plac._breakpoints)
        self.register_buffer("_intercepts", plac._intercepts)
        self.register_buffer("_signs", plac._signs)
        self.register_buffer("_exps", plac._exps)
        self.n_segments = plac.n_segments

    def _eval_int(self, x_q16):
        lib = load_lib()
        return lib.plac_int32(
            x_q16.contiguous(),
            self._breakpoints,
            self._intercepts,
            self._signs,
            self._exps,
            self.n_segments,
        )

    def forward(self, x):
        if isinstance(x, QTensor):
            return QTensor(self._eval_int(x.int_data))

        x_q16 = (x.detach().float() * SCALE).to(torch.int32)
        y_q16 = self._eval_int(x_q16)
        return (y_q16.float() / SCALE).to(dtype=x.dtype)


class _Int8Linear(nn.Module):
    """Int8 GEMV for lm_head."""

    def __init__(self, linear):
        super().__init__()
        w_i8, w_s = quantize_activations(linear.weight.data.float())
        self.register_buffer("w_i8", w_i8.contiguous())
        self.register_buffer("w_scale", w_s.squeeze(1))
        self.in_features = linear.in_features
        self.out_features = linear.out_features
        self.bias = linear.bias

    def forward(self, x):
        orig = x.shape
        x_2d = x.reshape(-1, self.in_features)
        x_i8, x_s = quantize_activations(x_2d)
        lib = load_lib()
        y_int = lib.int8_gemm_t(x_i8.contiguous(), self.w_i8)
        y = y_int.float() / (self.w_scale * x_s)
        if self.bias is not None:
            y = y + self.bias
        return y.reshape(*orig[:-1], self.out_features)


def _default_filter(mod, fqn):
    if _is_linear(mod):
        return "linear"
    if _is_activation(mod):
        return "activation"
    return None


def quantize(model, skip=None, target_mae=1e-2, filter_fn=None):
    """Quantize model in-place. Replaces linears and activations."""
    skip = skip or []
    filter_fn = filter_fn or _default_filter

    plac_cache = {}
    replacements = {}
    skipped = []
    float_modules = []

    for name, mod in model.named_modules():
        if any(name.startswith(s) or name == s for s in skip):
            skipped.append(name)
            continue

        if name == "lm_head" and isinstance(mod, nn.Linear):
            replacements[name] = _Int8Linear(mod)
            continue

        kind = filter_fn(mod, name)

        if kind == "linear":
            linear = _conv1d_to_linear(mod) if not isinstance(mod, nn.Linear) else mod
            if _is_already_ternary(linear.weight.data.float()):
                replacements[name] = TernaryLinear(linear)
            else:
                replacements[name] = DualTernaryLinear(linear)

        elif kind == "activation":
            cls = type(mod).__name__
            if cls not in plac_cache:
                plac_cache[cls] = _make_plac(mod, target_mae)
            replacements[name] = _PLACModule(plac_cache[cls])

    for name, new_mod in replacements.items():
        model.set_submodule(name, new_mod)

    # report what was converted and what stayed float
    counts = {}
    for r in replacements.values():
        k = type(r).__name__
        counts[k] = counts.get(k, 0) + 1

    for name, mod in model.named_modules():
        if name in replacements or not name:
            continue

        if isinstance(mod, (nn.LayerNorm, nn.RMSNorm)) or "norm" in type(mod).__name__.lower():
            float_modules.append(f"{name} ({type(mod).__name__}, float -- dequant boundary)")

        elif any(s in type(mod).__name__.lower() for s in _SKIP_ACT_NAMES):
            float_modules.append(f"{name} ({type(mod).__name__}, float -- skipped activation)")

    log.info("replaced: %s", ", ".join(f"{v} {k}" for k, v in counts.items()))
    if float_modules:
        log.info("float boundaries: %s", ", ".join(float_modules))

    if skipped:
        log.info("skipped: %s", ", ".join(skipped))

    return model
