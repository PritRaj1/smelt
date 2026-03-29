#include <immintrin.h>
#include <stdint.h>
#include <string.h>

#ifdef __cplusplus
extern "C" {
#endif

// y[m,n] = x[m,k] @ w[n,k].T using TL1 LUT + vpshufb
void ternary_gemm(const int8_t *x, const uint8_t *w_tl1, int32_t *y, int m, int n_padded, int k,
                  int n_pairs) {
    int half_n = n_padded / 2;

#pragma omp parallel for schedule(dynamic)
    for (int i = 0; i < m; i++) {
        const int8_t *xi = x + i * k;
        int32_t *yi = y + i * n_padded;
        memset(yi, 0, n_padded * sizeof(int32_t));

        // int16 accumulators, flush every 64 pairs to avoid overflow
        int16_t *acc16 = (int16_t *)alloca(n_padded * sizeof(int16_t));
        memset(acc16, 0, n_padded * sizeof(int16_t));
        int since_flush = 0;

        for (int p = 0; p < n_pairs; p++) {
            int16_t a0 = xi[p * 2];
            int16_t a1 = xi[p * 2 + 1];

            int16_t lut16[16] = {0};
            lut16[0] = -a0 - a1;
            lut16[1] = -a0;
            lut16[2] = -a0 + a1;
            lut16[3] = -a1;
            lut16[4] = 0;
            lut16[5] = a1;
            lut16[6] = a0 - a1;
            lut16[7] = a0;
            lut16[8] = a0 + a1;

            // split LUT into low/high bytes for vpshufb
            uint8_t lut_lo[16], lut_hi[16];
            for (int t = 0; t < 16; t++) {
                lut_lo[t] = (uint8_t)(lut16[t] & 0xFF);
                lut_hi[t] = (uint8_t)((lut16[t] >> 8) & 0xFF);
            }
            __m128i vlut_lo = _mm_loadu_si128((__m128i *)lut_lo);
            __m128i vlut_hi = _mm_loadu_si128((__m128i *)lut_hi);

            // 32 columns per step
            const uint8_t *wp = w_tl1 + p * half_n;
            for (int jb = 0; jb < half_n; jb += 16) {
                __m128i widx = _mm_loadu_si128((__m128i *)(wp + jb));
                __m128i lo_idx = _mm_and_si128(widx, _mm_set1_epi8(0x0F));
                __m128i hi_idx = _mm_and_si128(_mm_srli_epi16(widx, 4), _mm_set1_epi8(0x0F));

                __m128i even_lo = _mm_shuffle_epi8(vlut_lo, lo_idx);
                __m128i even_hi = _mm_shuffle_epi8(vlut_hi, lo_idx);
                __m128i odd_lo = _mm_shuffle_epi8(vlut_lo, hi_idx);
                __m128i odd_hi = _mm_shuffle_epi8(vlut_hi, hi_idx);

                // reconstruct int16
                __m128i even_a = _mm_unpacklo_epi8(even_lo, even_hi);
                __m128i even_b = _mm_unpackhi_epi8(even_lo, even_hi);
                __m128i odd_a = _mm_unpacklo_epi8(odd_lo, odd_hi);
                __m128i odd_b = _mm_unpackhi_epi8(odd_lo, odd_hi);

                // interleave and acc
                int16_t *dst = acc16 + jb * 2;
                _mm_storeu_si128((__m128i *)(dst),
                                 _mm_add_epi16(_mm_loadu_si128((__m128i *)(dst)),
                                               _mm_unpacklo_epi16(even_a, odd_a)));
                _mm_storeu_si128((__m128i *)(dst + 8),
                                 _mm_add_epi16(_mm_loadu_si128((__m128i *)(dst + 8)),
                                               _mm_unpackhi_epi16(even_a, odd_a)));
                _mm_storeu_si128((__m128i *)(dst + 16),
                                 _mm_add_epi16(_mm_loadu_si128((__m128i *)(dst + 16)),
                                               _mm_unpacklo_epi16(even_b, odd_b)));
                _mm_storeu_si128((__m128i *)(dst + 24),
                                 _mm_add_epi16(_mm_loadu_si128((__m128i *)(dst + 24)),
                                               _mm_unpackhi_epi16(even_b, odd_b)));
            }

            since_flush++;
            if (since_flush >= 64) {
                for (int j = 0; j < n_padded; j++) {
                    yi[j] += acc16[j];
                    acc16[j] = 0;
                }
                since_flush = 0;
            }
        }

        for (int j = 0; j < n_padded; j++)
            yi[j] += acc16[j];
    }
}

#ifdef __cplusplus
}
#endif
