#include <immintrin.h>
#include <stdint.h>

#define NR 4 // columns per microkernel: reuse activation loads

// expand 32 packed bits (LSB-first) to 32-lane int8 mask
static inline __m256i expand_mask(uint32_t bits) {
    __m256i vbits = _mm256_set1_epi32((int32_t)bits);
    __m256i shuf = _mm256_setr_epi8(0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2,
                                    2, 2, 2, 3, 3, 3, 3, 3, 3, 3, 3);
    __m256i bit_idx =
        _mm256_setr_epi8(0x01, 0x02, 0x04, 0x08, 0x10, 0x20, 0x40, (char)0x80, 0x01, 0x02, 0x04,
                         0x08, 0x10, 0x20, 0x40, (char)0x80, 0x01, 0x02, 0x04, 0x08, 0x10, 0x20,
                         0x40, (char)0x80, 0x01, 0x02, 0x04, 0x08, 0x10, 0x20, 0x40, (char)0x80);
    __m256i spread = _mm256_shuffle_epi8(vbits, shuf);
    return _mm256_cmpeq_epi8(_mm256_and_si256(spread, bit_idx), bit_idx);
}

// accumulate 32 masked int8 values into int32
static inline __m256i accum_masked(__m256i acc, __m256i xv, __m256i mask, int add) {
    __m256i ones16 = _mm256_set1_epi16(1);
    __m256i v = _mm256_blendv_epi8(_mm256_setzero_si256(), xv, mask);
    __m256i lo = _mm256_madd_epi16(_mm256_cvtepi8_epi16(_mm256_castsi256_si128(v)), ones16);
    __m256i hi = _mm256_madd_epi16(_mm256_cvtepi8_epi16(_mm256_extracti128_si256(v, 1)), ones16);
    __m256i sum = _mm256_add_epi32(lo, hi);
    return add ? _mm256_add_epi32(acc, sum) : _mm256_sub_epi32(acc, sum);
}

static inline int32_t hsum_epi32(__m256i v) {
    __m128i lo = _mm256_castsi256_si128(v);
    __m128i hi = _mm256_extracti128_si256(v, 1);
    lo = _mm_add_epi32(lo, hi);
    lo = _mm_hadd_epi32(lo, lo);
    lo = _mm_hadd_epi32(lo, lo);
    return _mm_cvtsi128_si32(lo);
}

// y[m,n] = x[m,k] @ w[n,k].T
// x: int8 [m,k], w: val/sign bitmasks [n, k/8], LSB-first
void ternary_gemm(const int8_t *x, const uint8_t *w_val, const uint8_t *w_sign, int32_t *y, int m,
                  int n, int k) {
    int k8 = k / 8;
    int k32 = k / 32;

#pragma omp parallel for schedule(dynamic)
    for (int i = 0; i < m; i++) {
        const int8_t *xi = x + i * k;

        int j;
        for (j = 0; j + NR <= n; j += NR) {
            __m256i acc[NR];
            for (int c = 0; c < NR; c++)
                acc[c] = _mm256_setzero_si256();

            for (int b0 = 0; b0 < k32; b0++) {
                __m256i xv = _mm256_loadu_si256((__m256i *)(xi + b0 * 32));

                for (int c = 0; c < NR; c++) {
                    uint32_t val = *(uint32_t *)(w_val + (j + c) * k8 + b0 * 4);
                    uint32_t sgn = *(uint32_t *)(w_sign + (j + c) * k8 + b0 * 4);

                    acc[c] = accum_masked(acc[c], xv, expand_mask(val & ~sgn), 1);
                    acc[c] = accum_masked(acc[c], xv, expand_mask(val & sgn), 0);
                }
            }

            for (int c = 0; c < NR; c++)
                y[i * n + j + c] = hsum_epi32(acc[c]);
        }

        // remainder columns
        for (; j < n; j++) {
            __m256i acc0 = _mm256_setzero_si256();
            for (int b0 = 0; b0 < k32; b0++) {
                __m256i xv = _mm256_loadu_si256((__m256i *)(xi + b0 * 32));
                uint32_t val = *(uint32_t *)(w_val + j * k8 + b0 * 4);
                uint32_t sgn = *(uint32_t *)(w_sign + j * k8 + b0 * 4);

                acc0 = accum_masked(acc0, xv, expand_mask(val & ~sgn), 1);
                acc0 = accum_masked(acc0, xv, expand_mask(val & sgn), 0);
            }
            y[i * n + j] = hsum_epi32(acc0);
        }
    }
}
