#include <torch/extension.h>

extern "C" {
void ternary_gemm(const int8_t *, const uint8_t *, int32_t *, int, int, int, int);
void plac_eval_lut(const int32_t *, int32_t *, int, const int32_t *, int, int32_t, int);
void softmax_int(const int32_t *, int32_t *, int, int);
void rmsnorm_int_batched(const int32_t *, const int32_t *, int32_t *, int, int);
void layernorm_int_batched(const int32_t *, const int32_t *, const int32_t *, int32_t *, int, int);
}

torch::Tensor smelt_ternary_gemm(torch::Tensor x, torch::Tensor w, int n_padded, int n_pairs) {
    auto y = torch::empty({x.size(0), n_padded}, torch::kInt32);
    ternary_gemm(x.data_ptr<int8_t>(), w.data_ptr<uint8_t>(), y.data_ptr<int32_t>(), x.size(0),
                 n_padded, x.size(1), n_pairs);
    return y;
}

torch::Tensor smelt_plac_lut(torch::Tensor x, torch::Tensor lut, int x_lo, int shift) {
    auto y = torch::empty_like(x);
    plac_eval_lut(x.data_ptr<int32_t>(), y.data_ptr<int32_t>(), x.numel(), lut.data_ptr<int32_t>(),
                  lut.size(0), x_lo, shift);
    return y;
}

torch::Tensor smelt_softmax(torch::Tensor x) {
    auto x2 = x.dim() == 1 ? x.unsqueeze(0) : x.reshape({-1, x.size(-1)});
    auto y = torch::empty_like(x2);
    softmax_int(x2.data_ptr<int32_t>(), y.data_ptr<int32_t>(), x2.size(0), x2.size(1));
    return y.reshape(x.sizes());
}

torch::Tensor smelt_rmsnorm(torch::Tensor x, torch::Tensor gamma) {
    auto x2 = x.reshape({-1, x.size(-1)});
    auto y = torch::empty_like(x2);
    rmsnorm_int_batched(x2.data_ptr<int32_t>(), gamma.data_ptr<int32_t>(), y.data_ptr<int32_t>(),
                        x2.size(0), x2.size(1));
    return y.reshape(x.sizes());
}

torch::Tensor smelt_layernorm(torch::Tensor x, torch::Tensor gamma, torch::Tensor beta) {
    auto x2 = x.reshape({-1, x.size(-1)});
    auto y = torch::empty_like(x2);
    layernorm_int_batched(x2.data_ptr<int32_t>(), gamma.data_ptr<int32_t>(),
                          beta.data_ptr<int32_t>(), y.data_ptr<int32_t>(), x2.size(0), x2.size(1));
    return y.reshape(x.sizes());
}

// float in -> quantize -> ternary GEMM -> rescale -> float out
torch::Tensor smelt_ternary_linear(torch::Tensor x, torch::Tensor w, int n_padded, int n_pairs,
                                   int out_features, float w_scale) {
    int m = x.size(0);
    int k = x.size(1);

    // quantize act to int8 (per-row absmax)
    auto x_abs = x.abs();
    auto x_max = std::get<0>(x_abs.max(/*dim=*/1, /*keepdim=*/true));
    auto x_scale = 127.0f / x_max.clamp_min(1e-5f);
    auto x_q = (x * x_scale).round().clamp(-128, 127).to(torch::kInt8).contiguous();

    // ternary GEMM
    auto y_int = torch::empty({m, n_padded}, torch::kInt32);
    ternary_gemm(x_q.data_ptr<int8_t>(), w.data_ptr<uint8_t>(), y_int.data_ptr<int32_t>(), m,
                 n_padded, k, n_pairs);

    // rescale: y = y_int * w_scale / x_scale
    auto y = y_int.index({"...", torch::indexing::Slice(torch::indexing::None, out_features)})
                 .to(torch::kFloat32) *
             (w_scale / x_scale);

    return y;
}

static void float_to_fixed(const float *in, int32_t *out, int n) {
    for (int i = 0; i < n; i++) {
        double v = (double)in[i] * 65536.0;
        v = v < -(double)(1LL << 31) ? -(double)(1LL << 31) : v;
        v = v > (double)((1LL << 31) - 1) ? (double)((1LL << 31) - 1) : v;
        out[i] = (int32_t)(v + (v >= 0 ? 0.5 : -0.5));
    }
}

static void fixed_to_float(const int32_t *in, float *out, int n) {
    for (int i = 0; i < n; i++)
        out[i] = (float)((double)in[i] / 65536.0);
}

torch::Tensor smelt_rmsnorm_float(torch::Tensor x, torch::Tensor gamma_fix) {
    int dim = x.size(-1);
    auto x2 = x.reshape({-1, dim}).contiguous();
    int rows = x2.size(0);
    int n = rows * dim;

    auto x_fix = torch::empty({rows, dim}, torch::kInt32);
    auto y_fix = torch::empty_like(x_fix);
    auto y_out = torch::empty_like(x2);

    float_to_fixed(x2.data_ptr<float>(), x_fix.data_ptr<int32_t>(), n);
    rmsnorm_int_batched(x_fix.data_ptr<int32_t>(), gamma_fix.data_ptr<int32_t>(),
                        y_fix.data_ptr<int32_t>(), rows, dim);
    fixed_to_float(y_fix.data_ptr<int32_t>(), y_out.data_ptr<float>(), n);

    return y_out.reshape(x.sizes());
}

torch::Tensor smelt_plac_float(torch::Tensor x, torch::Tensor lut, int x_lo, int shift) {
    auto flat = x.reshape({-1}).contiguous();
    int n = flat.numel();

    auto x_fix = torch::empty({n}, torch::kInt32);
    auto y_fix = torch::empty_like(x_fix);
    auto y_out = torch::empty_like(flat);

    float_to_fixed(flat.data_ptr<float>(), x_fix.data_ptr<int32_t>(), n);
    plac_eval_lut(x_fix.data_ptr<int32_t>(), y_fix.data_ptr<int32_t>(), n, lut.data_ptr<int32_t>(),
                  lut.size(0), x_lo, shift);
    fixed_to_float(y_fix.data_ptr<int32_t>(), y_out.data_ptr<float>(), n);

    return y_out.reshape(x.sizes());
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("rmsnorm_float", &smelt_rmsnorm_float);
    m.def("plac_float", &smelt_plac_float);
    m.def("ternary_linear", &smelt_ternary_linear);
    m.def("ternary_gemm", &smelt_ternary_gemm);
    m.def("plac_lut", &smelt_plac_lut);
    m.def("softmax", &smelt_softmax);
    m.def("rmsnorm", &smelt_rmsnorm);
    m.def("layernorm", &smelt_layernorm);
}
