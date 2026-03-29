import logging

import torch
import torch.nn as nn

from .norm import rmsnorm_int32
from .plac import PLACFunc, from_fixed, to_fixed
from .quantize import TernaryLinear

log = logging.getLogger(__name__)

# already int-friendly, no PLAC needed
_SKIP_ACTS = {nn.ReLU, nn.Identity, nn.Dropout}


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


def _is_rmsnorm(mod):
    return hasattr(mod, "weight") and not hasattr(mod, "bias") and "RMSNorm" in type(mod).__name__


def _is_activation(mod):
    """Leaf module, no parameters, likely an activation."""
    if type(mod) in _SKIP_ACTS:
        return False

    if len(list(mod.children())) > 0:
        return False

    if len(list(mod.parameters())) > 0:
        return False

    name = type(mod).__name__.lower()
    keys = ("activation", "silu", "gelu", "sigmoid", "tanh", "swish", "relu")
    return any(k in name for k in keys)


def _make_plac(mod, target_mae):
    """Build PLACFunc by probing the module's forward."""

    def fn(x):
        with torch.no_grad():
            return mod(x.float()).to(x.dtype)

    return PLACFunc(fn, -8, 8, target_mae)


class _PLACModule(nn.Module):
    def __init__(self, plac):
        super().__init__()
        self.plac = plac

    def forward(self, x):
        return self.plac(x)


class _RMSNormInt(nn.Module):
    def __init__(self, norm):
        super().__init__()
        gamma = torch.from_numpy(to_fixed(norm.weight.data.double().numpy()))
        self.register_buffer("gamma_fix", gamma)

    def forward(self, x):
        x_fix = torch.from_numpy(to_fixed(x.detach().float().numpy()))
        y_fix = rmsnorm_int32(x_fix, self.gamma_fix)
        return torch.from_numpy(from_fixed(y_fix.numpy())).to(x.dtype)


def _default_filter(mod, fqn):
    if _is_linear(mod):
        return "linear"

    if _is_activation(mod):
        return "activation"

    if _is_rmsnorm(mod):
        return "rmsnorm"

    return None


def quantize(model, skip=None, target_mae=1e-2, filter_fn=None):
    """
    Quantize model in-place. Replaces linears, activations, norms.

    skip: module name prefixes to leave in float (default: ["lm_head"])
    filter_fn: fn(module, fqn) -> "linear"|"activation"|"rmsnorm"|None
    """
    skip = skip or ["lm_head"]
    filter_fn = filter_fn or _default_filter

    replacements = {}
    skipped = []
    for name, mod in model.named_modules():
        if any(name.startswith(s) or name == s for s in skip):
            skipped.append(name)
            continue

        kind = filter_fn(mod, name)

        if kind == "linear":
            linear = _conv1d_to_linear(mod) if not isinstance(mod, nn.Linear) else mod
            replacements[name] = TernaryLinear(linear)

        elif kind == "activation":
            replacements[name] = _PLACModule(_make_plac(mod, target_mae))

        elif kind == "rmsnorm":
            replacements[name] = _RMSNormInt(mod)

    for name, new_mod in replacements.items():
        model.set_submodule(name, new_mod)

    counts = {}
    for r in replacements.values():
        k = type(r).__name__
        counts[k] = counts.get(k, 0) + 1

    log.info("replaced: %s", ", ".join(f"{v} {k}" for k, v in counts.items()))
    if skipped:
        log.info("skipped: %s", ", ".join(skipped))

    return model
