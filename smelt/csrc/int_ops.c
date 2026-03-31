#include <immintrin.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

// int32 GEMM output -> Q16.16
// rescale = round(w_scale / act_scale * 65536), precomputed per row
// y[i] = x[i] * rescale (plain multiply, rescale already has Q16.16 factor)
void int_rescale(const int32_t *x, int32_t *y, int n, int32_t rescale) {
    __m256i vr = _mm256_set1_epi32(rescale);
    int j;
    for (j = 0; j + 7 < n; j += 8) {
        __m256i vx = _mm256_loadu_si256((__m256i *)(x + j));
        _mm256_storeu_si256((__m256i *)(y + j), _mm256_mullo_epi32(vx, vr));
    }
    for (; j < n; j++)
        y[j] = (int32_t)((int64_t)x[j] * rescale);
}

// Q16.16 -> int8: pure int absmax quantize
// scale_out = max_abs (dequant: real_q16 = int8_val * max_abs / 127)
void int_quantize(const int32_t *x, int8_t *out, int32_t *scale_out, int n) {
    int32_t mx = 0;
    int j;
    __m256i vmx = _mm256_setzero_si256();
    for (j = 0; j + 7 < n; j += 8) {
        __m256i vx = _mm256_loadu_si256((__m256i *)(x + j));
        vmx = _mm256_max_epi32(vmx, _mm256_abs_epi32(vx));
    }
    __m128i lo = _mm256_castsi256_si128(vmx);
    __m128i hi = _mm256_extracti128_si256(vmx, 1);
    lo = _mm_max_epi32(lo, hi);
    lo = _mm_max_epi32(lo, _mm_shuffle_epi32(lo, 0x4E));
    lo = _mm_max_epi32(lo, _mm_shuffle_epi32(lo, 0xB1));
    mx = _mm_cvtsi128_si32(lo);
    for (; j < n; j++) {
        int32_t a = x[j] < 0 ? -x[j] : x[j];
        if (a > mx)
            mx = a;
    }

    *scale_out = mx;
    if (mx == 0) {
        for (j = 0; j < n; j++)
            out[j] = 0;
        return;
    }

    // out[i] = clamp(x[i] * 127 / mx, -128, 127) with rounding
    int32_t half = mx / 2;
    for (j = 0; j < n; j++) {
        int64_t v = ((int64_t)x[j] * 127 + (x[j] >= 0 ? half : -half)) / mx;
        out[j] = (int8_t)(v < -128 ? -128 : (v > 127 ? 127 : v));
    }
}

// Q16.16 * Q16.16 -> Q16.16: (a * b) >> 16
void int_mul_q16(const int32_t *a, const int32_t *b, int32_t *y, int n) {
    int j;
    for (j = 0; j + 7 < n; j += 8) {
        __m256i va = _mm256_loadu_si256((__m256i *)(a + j));
        __m256i vb = _mm256_loadu_si256((__m256i *)(b + j));
        __m256i va_odd = _mm256_shuffle_epi32(va, 0xF5);
        __m256i vb_odd = _mm256_shuffle_epi32(vb, 0xF5);

        __m256i lo = _mm256_srli_epi64(_mm256_mul_epi32(va, vb), 16);
        __m256i hi = _mm256_srli_epi64(_mm256_mul_epi32(va_odd, vb_odd), 16);

        __m256i mask = _mm256_set1_epi64x(0xFFFFFFFF);
        __m256i result = _mm256_or_si256(_mm256_and_si256(lo, mask),
                                         _mm256_slli_epi64(_mm256_and_si256(hi, mask), 32));
        _mm256_storeu_si256((__m256i *)(y + j), result);
    }
    for (; j < n; j++)
        y[j] = (int32_t)(((int64_t)a[j] * b[j]) >> 16);
}

#ifdef __cplusplus
}
#endif
