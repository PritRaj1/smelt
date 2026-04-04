import torch

from ._clib import load_lib

lib = torch.library.Library("smelt", "DEF")

lib.define(
    "ternary_linear(Tensor x, Tensor w, int n_padded, int n_pairs,"
    " int out_features, float w_scale) -> Tensor"
)
lib.define(
    "ternary_linear_i8(Tensor x_i8, float act_scale, Tensor w,"
    " int n_padded, int n_pairs, int out_features, float w_scale) -> Tensor"
)
lib.define(
    "plac_int32(Tensor x, Tensor bp, Tensor ic, Tensor signs, Tensor exps, int n_segs) -> Tensor"
)
lib.define("int8_gemm_t(Tensor a, Tensor b) -> Tensor")
lib.define("int8_batched_gemm_t(Tensor a, Tensor b) -> Tensor")
lib.define("ternary_gemm(Tensor x, Tensor w, int n_padded, int n_pairs) -> Tensor")
lib.define("int_rescale(Tensor x, int rescale) -> Tensor")
lib.define("int_quantize(Tensor x) -> (Tensor, Tensor)")
lib.define("int_mul(Tensor a, Tensor b) -> Tensor")
lib.define("softmax(Tensor x) -> Tensor")
lib.define("rmsnorm(Tensor x, Tensor gamma) -> Tensor")
lib.define("layernorm(Tensor x, Tensor gamma, Tensor beta) -> Tensor")


@torch.library.register_fake("smelt::ternary_linear")
def _(x, w, n_padded, n_pairs, out_features, w_scale):
    return torch.empty(x.shape[0], out_features, dtype=torch.float32, device=x.device)


@torch.library.register_fake("smelt::ternary_linear_i8")
def _(x_i8, act_scale, w, n_padded, n_pairs, out_features, w_scale):
    return torch.empty(x_i8.shape[0], out_features, dtype=torch.float32, device=x_i8.device)


@torch.library.register_fake("smelt::plac_int32")
def _(x, bp, ic, signs, exps, n_segs):
    return torch.empty_like(x)


@torch.library.register_fake("smelt::int8_gemm_t")
def _(a, b):
    return torch.empty(a.shape[0], b.shape[0], dtype=torch.int32, device=a.device)


@torch.library.register_fake("smelt::int8_batched_gemm_t")
def _(a, b):
    return torch.empty(a.shape[0], a.shape[1], b.shape[1], dtype=torch.int32, device=a.device)


@torch.library.register_fake("smelt::ternary_gemm")
def _(x, w, n_padded, n_pairs):
    return torch.empty(x.shape[0], n_padded, dtype=torch.int32, device=x.device)


@torch.library.register_fake("smelt::int_rescale")
def _(x, rescale):
    return torch.empty_like(x)


@torch.library.register_fake("smelt::int_quantize")
def _(x):
    x2 = x.reshape(-1, x.shape[-1])
    return (
        torch.empty(x2.shape, dtype=torch.int8, device=x.device),
        torch.empty(x2.shape[0], dtype=torch.int32, device=x.device),
    )


@torch.library.register_fake("smelt::int_mul")
def _(a, b):
    return torch.empty_like(a)


@torch.library.register_fake("smelt::softmax")
def _(x):
    return torch.empty_like(x)


@torch.library.register_fake("smelt::rmsnorm")
def _(x, gamma):
    return torch.empty_like(x)


@torch.library.register_fake("smelt::layernorm")
def _(x, gamma, beta):
    return torch.empty_like(x)


_ALL_OPS = [
    "ternary_linear",
    "ternary_linear_i8",
    "plac_int32",
    "int8_gemm_t",
    "int8_batched_gemm_t",
    "ternary_gemm",
    "int_rescale",
    "int_quantize",
    "int_mul",
    "softmax",
    "rmsnorm",
    "layernorm",
]


def _register_all():
    clib = load_lib()
    for name in _ALL_OPS:
        torch.library.register_kernel(f"smelt::{name}", "cpu", getattr(clib, name))
