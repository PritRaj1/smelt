#include <math.h>
#include <torch/extension.h>

extern "C" {
void ternary_gemm(const int8_t *, const uint8_t *, int32_t *, int, int, int, int);
void plac_eval_segments(const int32_t *, int32_t *, int, const int32_t *, const int32_t *,
                        const int32_t *, const int32_t *, int);
void softmax_int(const int32_t *, int32_t *, int, int);
void rmsnorm_int_batched(const int32_t *, const int32_t *, int32_t *, int, int);
void layernorm_int_batched(const int32_t *, const int32_t *, const int32_t *, int32_t *, int, int);
void int8_gemm_t(const int8_t *, const int8_t *, int32_t *, int, int, int);
void int8_batched_gemm_t(const int8_t *, const int8_t *, int32_t *, int, int, int, int);
}

torch::Tensor smelt_ternary_gemm(torch::Tensor x, torch::Tensor w, int n_padded, int n_pairs) {
    auto y = torch::empty({x.size(0), n_padded}, torch::kInt32);
    ternary_gemm(x.data_ptr<int8_t>(), w.data_ptr<uint8_t>(), y.data_ptr<int32_t>(), x.size(0),
                 n_padded, x.size(1), n_pairs);
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

// per-row absmax quantize to int8, returns scale per row (127 / max)
static void quantize_act(const float *x, int8_t *out, float *row_scale, int m, int k) {
    for (int i = 0; i < m; i++) {
        const float *row = x + i * k;
        float mx = 0.0f;
        for (int j = 0; j < k; j++)
            mx = fmaxf(mx, fabsf(row[j]));

        float s = 127.0f / fmaxf(mx, 1e-5f);
        row_scale[i] = s;
        for (int j = 0; j < k; j++)
            out[i * k + j] = (int8_t)fminf(fmaxf(roundf(row[j] * s), -128.0f), 127.0f);
    }
}

// rescale int32 GEMM output to float: y_float = y_int * (w_scale / row_scale)
static void rescale_output(const int32_t *y_int, float *y_out, const float *row_scale,
                           float w_scale, int m, int n_padded, int out_features) {
    for (int i = 0; i < m; i++) {
        float s = w_scale / row_scale[i];
        const int32_t *src = y_int + i * n_padded;
        float *dst = y_out + i * out_features;
        for (int j = 0; j < out_features; j++)
            dst[j] = (float)src[j] * s;
    }
}

// float in -> quantize -> ternary GEMM -> rescale -> float out
torch::Tensor smelt_ternary_linear(torch::Tensor x, torch::Tensor w, int n_padded, int n_pairs,
                                   int out_features, float w_scale) {
    int m = x.size(0);
    int k = x.size(1);
    const float *xp = x.data_ptr<float>();

    auto x_q = torch::empty({m, k}, torch::kInt8);
    float *row_scale = (float *)alloca(m * sizeof(float));
    quantize_act(xp, x_q.data_ptr<int8_t>(), row_scale, m, k);

    auto y_int = torch::empty({m, n_padded}, torch::kInt32);
    ternary_gemm(x_q.data_ptr<int8_t>(), w.data_ptr<uint8_t>(), y_int.data_ptr<int32_t>(), m,
                 n_padded, k, n_pairs);

    auto y = torch::empty({m, out_features}, torch::kFloat32);
    rescale_output(y_int.data_ptr<int32_t>(), y.data_ptr<float>(), row_scale, w_scale, m, n_padded,
                   out_features);

    return y;
}

static void float_to_fixed(const float *in, int32_t *out, int n) {
    for (int i = 0; i < n; i++) {
        double v = (double)in[i] * 65536.0;
        out[i] = (int32_t)fmin(fmax(round(v), -(double)(1LL << 31)), (double)((1LL << 31) - 1));
    }
}

static void fixed_to_float(const int32_t *in, float *out, int n) {
    for (int i = 0; i < n; i++)
        out[i] = (float)((double)in[i] / 65536.0);
}

torch::Tensor smelt_plac_segments_float(torch::Tensor x, torch::Tensor breakpoints,
                                        torch::Tensor intercepts, torch::Tensor signs,
                                        torch::Tensor exps, int n_segs) {
    auto flat = x.reshape({-1}).contiguous();
    int n = flat.numel();
    auto x_fix = torch::empty({n}, torch::kInt32);
    auto y_fix = torch::empty_like(x_fix);
    auto y_out = torch::empty_like(flat);
    float_to_fixed(flat.data_ptr<float>(), x_fix.data_ptr<int32_t>(), n);
    plac_eval_segments(x_fix.data_ptr<int32_t>(), y_fix.data_ptr<int32_t>(), n,
                       breakpoints.data_ptr<int32_t>(), intercepts.data_ptr<int32_t>(),
                       signs.data_ptr<int32_t>(), exps.data_ptr<int32_t>(), n_segs);
    fixed_to_float(y_fix.data_ptr<int32_t>(), y_out.data_ptr<float>(), n);
    return y_out.reshape(x.sizes());
}

torch::Tensor smelt_plac_segments_int32(torch::Tensor x, torch::Tensor breakpoints,
                                        torch::Tensor intercepts, torch::Tensor signs,
                                        torch::Tensor exps, int n_segs) {
    auto y = torch::empty_like(x);
    plac_eval_segments(x.data_ptr<int32_t>(), y.data_ptr<int32_t>(), x.numel(),
                       breakpoints.data_ptr<int32_t>(), intercepts.data_ptr<int32_t>(),
                       signs.data_ptr<int32_t>(), exps.data_ptr<int32_t>(), n_segs);
    return y;
}

// int8 in -> ternary GEMM -> rescale -> float out (skip quantize, for shared input)
static torch::Tensor smelt_ternary_linear_i8(torch::Tensor x_i8, float act_scale, torch::Tensor w,
                                             int n_padded, int n_pairs, int out_features,
                                             float w_scale) {
    int m = x_i8.size(0), k = x_i8.size(1);

    auto y_int = torch::empty({m, n_padded}, torch::kInt32);
    ternary_gemm(x_i8.data_ptr<int8_t>(), w.data_ptr<uint8_t>(), y_int.data_ptr<int32_t>(), m,
                 n_padded, k, n_pairs);

    auto y = torch::empty({m, out_features}, torch::kFloat32);
    float s = w_scale / act_scale;
    int32_t *src = y_int.data_ptr<int32_t>();
    float *dst = y.data_ptr<float>();
    for (int i = 0; i < m; i++)
        for (int j = 0; j < out_features; j++)
            dst[i * out_features + j] = (float)src[i * n_padded + j] * s;

    return y;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("plac_segments_float", &smelt_plac_segments_float);
    m.def("plac_segments_int32", &smelt_plac_segments_int32);
    m.def("ternary_linear", &smelt_ternary_linear);
    m.def("ternary_gemm", &smelt_ternary_gemm);
    m.def("softmax", &smelt_softmax);
    m.def("rmsnorm", &smelt_rmsnorm);
    m.def("layernorm", &smelt_layernorm);

    m.def("ternary_linear_i8", &smelt_ternary_linear_i8);
    m.def("int8_gemm_t", [](torch::Tensor a, torch::Tensor b) {
        int m = a.size(0), n = b.size(0), k = a.size(1);
        auto c = torch::empty({m, n}, torch::kInt32);
        int8_gemm_t(a.data_ptr<int8_t>(), b.data_ptr<int8_t>(), c.data_ptr<int32_t>(), m, n, k);
        return c;
    });

    m.def("int8_batched_gemm_t", [](torch::Tensor a, torch::Tensor b) {
        int batch = a.size(0), m = a.size(1), n = b.size(1), k = a.size(2);
        auto c = torch::empty({batch, m, n}, torch::kInt32);
        int8_batched_gemm_t(a.data_ptr<int8_t>(), b.data_ptr<int8_t>(), c.data_ptr<int32_t>(),
                            batch, m, n, k);
        return c;
    });
}
