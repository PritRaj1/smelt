#include <immintrin.h>
#include <stdint.h>
#include <string.h>

#ifdef __cplusplus
extern "C" {
#endif

// y[m,n] = x[m,k] @ w[n,k].T using TL1 LUT + AVX2 vpshufb
void ternary_gemm(const int8_t *x, const uint8_t *w_tl1, int32_t *y, int m, int n_padded, int k,
                  int n_pairs) {
    int half_n = n_padded / 2;
    __m256i lo_mask = _mm256_set1_epi8(0x0F);

#pragma omp parallel for schedule(dynamic)
    for (int i = 0; i < m; i++) {
        const int8_t *xi = x + i * k;
        int32_t *yi = y + i * n_padded;
        memset(yi, 0, n_padded * sizeof(int32_t));

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

            uint8_t lo[16], hi[16];
            for (int t = 0; t < 16; t++) {
                lo[t] = (uint8_t)(lut16[t] & 0xFF);
                hi[t] = (uint8_t)((lut16[t] >> 8) & 0xFF);
            }

            // duplicate LUT into both 128-bit lanes of 256-bit register
            __m128i lut_lo_128 = _mm_loadu_si128((__m128i *)lo);
            __m128i lut_hi_128 = _mm_loadu_si128((__m128i *)hi);
            __m256i vlut_lo = _mm256_broadcastsi128_si256(lut_lo_128);
            __m256i vlut_hi = _mm256_broadcastsi128_si256(lut_hi_128);

            // 64 columns per step (32 bytes = 64 nibble indices)
            const uint8_t *wp = w_tl1 + p * half_n;
            int jb;
            for (jb = 0; jb + 31 < half_n; jb += 32) {
                __m256i widx = _mm256_loadu_si256((__m256i *)(wp + jb));
                __m256i lo_idx = _mm256_and_si256(widx, lo_mask);
                __m256i hi_idx = _mm256_and_si256(_mm256_srli_epi16(widx, 4), lo_mask);

                __m256i even_lo = _mm256_shuffle_epi8(vlut_lo, lo_idx);
                __m256i even_hi = _mm256_shuffle_epi8(vlut_hi, lo_idx);
                __m256i odd_lo = _mm256_shuffle_epi8(vlut_lo, hi_idx);
                __m256i odd_hi = _mm256_shuffle_epi8(vlut_hi, hi_idx);

                __m256i even_a = _mm256_unpacklo_epi8(even_lo, even_hi);
                __m256i even_b = _mm256_unpackhi_epi8(even_lo, even_hi);
                __m256i odd_a = _mm256_unpacklo_epi8(odd_lo, odd_hi);
                __m256i odd_b = _mm256_unpackhi_epi8(odd_lo, odd_hi);

                int16_t *dst = acc16 + jb * 2;
                _mm256_storeu_si256((__m256i *)(dst),
                                    _mm256_add_epi16(_mm256_loadu_si256((__m256i *)(dst)),
                                                     _mm256_unpacklo_epi16(even_a, odd_a)));
                _mm256_storeu_si256((__m256i *)(dst + 16),
                                    _mm256_add_epi16(_mm256_loadu_si256((__m256i *)(dst + 16)),
                                                     _mm256_unpackhi_epi16(even_a, odd_a)));
                _mm256_storeu_si256((__m256i *)(dst + 32),
                                    _mm256_add_epi16(_mm256_loadu_si256((__m256i *)(dst + 32)),
                                                     _mm256_unpacklo_epi16(even_b, odd_b)));
                _mm256_storeu_si256((__m256i *)(dst + 48),
                                    _mm256_add_epi16(_mm256_loadu_si256((__m256i *)(dst + 48)),
                                                     _mm256_unpackhi_epi16(even_b, odd_b)));
            }

            // SSE remainder for <32 byte tail
            for (; jb < half_n; jb += 16) {
                __m128i widx = _mm_loadu_si128((__m128i *)(wp + jb));
                __m128i li = _mm_and_si128(widx, _mm256_castsi256_si128(lo_mask));
                __m128i hi2 =
                    _mm_and_si128(_mm_srli_epi16(widx, 4), _mm256_castsi256_si128(lo_mask));

                __m128i elo = _mm_shuffle_epi8(lut_lo_128, li);
                __m128i ehi = _mm_shuffle_epi8(lut_hi_128, li);
                __m128i olo = _mm_shuffle_epi8(lut_lo_128, hi2);
                __m128i ohi = _mm_shuffle_epi8(lut_hi_128, hi2);

                __m128i ea = _mm_unpacklo_epi8(elo, ehi);
                __m128i eb = _mm_unpackhi_epi8(elo, ehi);
                __m128i oa = _mm_unpacklo_epi8(olo, ohi);
                __m128i ob = _mm_unpackhi_epi8(olo, ohi);

                int16_t *dst = acc16 + jb * 2;
                _mm_storeu_si128((__m128i *)(dst), _mm_add_epi16(_mm_loadu_si128((__m128i *)(dst)),
                                                                 _mm_unpacklo_epi16(ea, oa)));
                _mm_storeu_si128((__m128i *)(dst + 8),
                                 _mm_add_epi16(_mm_loadu_si128((__m128i *)(dst + 8)),
                                               _mm_unpackhi_epi16(ea, oa)));
                _mm_storeu_si128((__m128i *)(dst + 16),
                                 _mm_add_epi16(_mm_loadu_si128((__m128i *)(dst + 16)),
                                               _mm_unpacklo_epi16(eb, ob)));
                _mm_storeu_si128((__m128i *)(dst + 24),
                                 _mm_add_epi16(_mm_loadu_si128((__m128i *)(dst + 24)),
                                               _mm_unpackhi_epi16(eb, ob)));
            }

            since_flush++;
            if (since_flush >= 64) {
                __m256i zero = _mm256_setzero_si256();
                for (int j = 0; j < n_padded; j += 16) {
                    __m256i a = _mm256_loadu_si256((__m256i *)(acc16 + j));
                    __m256i lo = _mm256_cvtepi16_epi32(_mm256_castsi256_si128(a));
                    __m256i hi = _mm256_cvtepi16_epi32(_mm256_extracti128_si256(a, 1));
                    _mm256_storeu_si256(
                        (__m256i *)(yi + j),
                        _mm256_add_epi32(_mm256_loadu_si256((__m256i *)(yi + j)), lo));
                    _mm256_storeu_si256(
                        (__m256i *)(yi + j + 8),
                        _mm256_add_epi32(_mm256_loadu_si256((__m256i *)(yi + j + 8)), hi));
                    _mm256_storeu_si256((__m256i *)(acc16 + j), zero);
                }
                since_flush = 0;
            }
        }

        // final flush
        for (int j = 0; j < n_padded; j += 16) {
            __m256i a = _mm256_loadu_si256((__m256i *)(acc16 + j));
            __m256i lo = _mm256_cvtepi16_epi32(_mm256_castsi256_si128(a));
            __m256i hi = _mm256_cvtepi16_epi32(_mm256_extracti128_si256(a, 1));
            _mm256_storeu_si256((__m256i *)(yi + j),
                                _mm256_add_epi32(_mm256_loadu_si256((__m256i *)(yi + j)), lo));
            _mm256_storeu_si256((__m256i *)(yi + j + 8),
                                _mm256_add_epi32(_mm256_loadu_si256((__m256i *)(yi + j + 8)), hi));
        }
    }
}

#ifdef __cplusplus
}
#endif
