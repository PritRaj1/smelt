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
    ternary_gemm(x.data_ptr<int8_t>(), w.data_ptr<uint8_t>(), y.data_ptr<int32_t>(),
                 x.size(0), n_padded, x.size(1), n_pairs);
    return y;
}

torch::Tensor smelt_plac_lut(torch::Tensor x, torch::Tensor lut, int x_lo, int shift) {
    auto y = torch::empty_like(x);
    plac_eval_lut(x.data_ptr<int32_t>(), y.data_ptr<int32_t>(), x.numel(),
                  lut.data_ptr<int32_t>(), lut.size(0), x_lo, shift);
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
    rmsnorm_int_batched(x2.data_ptr<int32_t>(), gamma.data_ptr<int32_t>(),
                        y.data_ptr<int32_t>(), x2.size(0), x2.size(1));
    return y.reshape(x.sizes());
}

torch::Tensor smelt_layernorm(torch::Tensor x, torch::Tensor gamma, torch::Tensor beta) {
    auto x2 = x.reshape({-1, x.size(-1)});
    auto y = torch::empty_like(x2);
    layernorm_int_batched(x2.data_ptr<int32_t>(), gamma.data_ptr<int32_t>(),
                          beta.data_ptr<int32_t>(), y.data_ptr<int32_t>(),
                          x2.size(0), x2.size(1));
    return y.reshape(x.sizes());
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("ternary_gemm", &smelt_ternary_gemm);
    m.def("plac_lut", &smelt_plac_lut);
    m.def("softmax", &smelt_softmax);
    m.def("rmsnorm", &smelt_rmsnorm);
    m.def("layernorm", &smelt_layernorm);
}
