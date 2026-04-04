#include <immintrin.h>
#include <omp.h>
#include <stdint.h>
#include <string.h>

#ifdef __cplusplus
extern "C" {
#endif

static inline void acc_avx2(__m256i li, __m256i hi, __m256i vlo, __m256i vhi, int16_t *d) {
    __m256i ea = _mm256_unpacklo_epi8(_mm256_shuffle_epi8(vlo, li), _mm256_shuffle_epi8(vhi, li));
    __m256i eb = _mm256_unpackhi_epi8(_mm256_shuffle_epi8(vlo, li), _mm256_shuffle_epi8(vhi, li));
    __m256i oa = _mm256_unpacklo_epi8(_mm256_shuffle_epi8(vlo, hi), _mm256_shuffle_epi8(vhi, hi));
    __m256i ob = _mm256_unpackhi_epi8(_mm256_shuffle_epi8(vlo, hi), _mm256_shuffle_epi8(vhi, hi));

    __m256i r0 = _mm256_unpacklo_epi16(ea, oa);
    __m256i r1 = _mm256_unpackhi_epi16(ea, oa);
    __m256i r2 = _mm256_unpacklo_epi16(eb, ob);
    __m256i r3 = _mm256_unpackhi_epi16(eb, ob);

    __m256i o0 = _mm256_permute2x128_si256(r0, r1, 0x20);
    __m256i o1 = _mm256_permute2x128_si256(r2, r3, 0x20);
    __m256i o2 = _mm256_permute2x128_si256(r0, r1, 0x31);
    __m256i o3 = _mm256_permute2x128_si256(r2, r3, 0x31);

    _mm256_storeu_si256((__m256i *)d, _mm256_add_epi16(_mm256_loadu_si256((__m256i *)d), o0));
    _mm256_storeu_si256((__m256i *)(d + 16),
                        _mm256_add_epi16(_mm256_loadu_si256((__m256i *)(d + 16)), o1));
    _mm256_storeu_si256((__m256i *)(d + 32),
                        _mm256_add_epi16(_mm256_loadu_si256((__m256i *)(d + 32)), o2));
    _mm256_storeu_si256((__m256i *)(d + 48),
                        _mm256_add_epi16(_mm256_loadu_si256((__m256i *)(d + 48)), o3));
}

static inline void acc_sse(__m128i li, __m128i hi, __m128i vlo, __m128i vhi, int16_t *d) {
    __m128i ea = _mm_unpacklo_epi8(_mm_shuffle_epi8(vlo, li), _mm_shuffle_epi8(vhi, li));
    __m128i eb = _mm_unpackhi_epi8(_mm_shuffle_epi8(vlo, li), _mm_shuffle_epi8(vhi, li));
    __m128i oa = _mm_unpacklo_epi8(_mm_shuffle_epi8(vlo, hi), _mm_shuffle_epi8(vhi, hi));
    __m128i ob = _mm_unpackhi_epi8(_mm_shuffle_epi8(vlo, hi), _mm_shuffle_epi8(vhi, hi));
    _mm_storeu_si128((__m128i *)d,
                     _mm_add_epi16(_mm_loadu_si128((__m128i *)d), _mm_unpacklo_epi16(ea, oa)));
    _mm_storeu_si128((__m128i *)(d + 8), _mm_add_epi16(_mm_loadu_si128((__m128i *)(d + 8)),
                                                       _mm_unpackhi_epi16(ea, oa)));
    _mm_storeu_si128((__m128i *)(d + 16), _mm_add_epi16(_mm_loadu_si128((__m128i *)(d + 16)),
                                                        _mm_unpacklo_epi16(eb, ob)));
    _mm_storeu_si128((__m128i *)(d + 24), _mm_add_epi16(_mm_loadu_si128((__m128i *)(d + 24)),
                                                        _mm_unpackhi_epi16(eb, ob)));
}

static inline void flush(int16_t *acc, int32_t *y, int n) {
    __m256i z = _mm256_setzero_si256();
    for (int j = 0; j < n; j += 16) {
        __m256i a = _mm256_loadu_si256((__m256i *)(acc + j));
        __m256i lo = _mm256_cvtepi16_epi32(_mm256_castsi256_si128(a));
        __m256i hi = _mm256_cvtepi16_epi32(_mm256_extracti128_si256(a, 1));
        _mm256_storeu_si256((__m256i *)(y + j),
                            _mm256_add_epi32(_mm256_loadu_si256((__m256i *)(y + j)), lo));
        _mm256_storeu_si256((__m256i *)(y + j + 8),
                            _mm256_add_epi32(_mm256_loadu_si256((__m256i *)(y + j + 8)), hi));
        _mm256_storeu_si256((__m256i *)(acc + j), z);
    }
}

static void precompute_luts(const int8_t *xi, int n_pairs, uint8_t *lut_buf) {
    for (int p = 0; p < n_pairs; p++) {
        int16_t a0 = xi[p * 2], a1 = xi[p * 2 + 1];
        int16_t t[16] = {0};
        t[0] = -a0 - a1;
        t[1] = -a0;
        t[2] = -a0 + a1;
        t[3] = -a1;
        t[4] = 0;
        t[5] = a1;
        t[6] = a0 - a1;
        t[7] = a0;
        t[8] = a0 + a1;
        uint8_t *lo = lut_buf + p * 32;
        uint8_t *hi = lo + 16;
        for (int i = 0; i < 16; i++) {
            lo[i] = t[i] & 0xFF;
            hi[i] = (t[i] >> 8) & 0xFF;
        }
    }
}

// inner kernel: process pairs [0, n_pairs) for columns [byte0, byte0+nbytes)
static void gemm_col_slice(const uint8_t *lut_buf, const uint8_t *w_tl1, int32_t *yi, int hn,
                           int n_pairs, int byte0, int nbytes) {
    __m256i mask = _mm256_set1_epi8(0x0F);
    __m128i mask128 = _mm256_castsi256_si128(mask);
    int ncols = nbytes * 2;

    int16_t *acc = (int16_t *)alloca(ncols * sizeof(int16_t));
    memset(acc, 0, ncols * sizeof(int16_t));
    memset(yi + byte0 * 2, 0, ncols * sizeof(int32_t));

    int sf = 0;
    for (int p = 0; p < n_pairs; p++) {
        __m128i lut_lo = _mm_loadu_si128((__m128i *)(lut_buf + p * 32));
        __m128i lut_hi = _mm_loadu_si128((__m128i *)(lut_buf + p * 32 + 16));
        __m256i vlo = _mm256_broadcastsi128_si256(lut_lo);
        __m256i vhi = _mm256_broadcastsi128_si256(lut_hi);

        const uint8_t *wp = w_tl1 + p * hn + byte0;
        int jb;
        for (jb = 0; jb + 31 < nbytes; jb += 32) {
            __m256i widx = _mm256_loadu_si256((__m256i *)(wp + jb));
            acc_avx2(_mm256_and_si256(widx, mask),
                     _mm256_and_si256(_mm256_srli_epi16(widx, 4), mask), vlo, vhi, acc + jb * 2);
        }

        for (; jb < nbytes; jb += 16) {
            __m128i widx = _mm_loadu_si128((__m128i *)(wp + jb));
            acc_sse(_mm_and_si128(widx, mask128), _mm_and_si128(_mm_srli_epi16(widx, 4), mask128),
                    lut_lo, lut_hi, acc + jb * 2);
        }

        if (++sf >= 128) {
            flush(acc, yi + byte0 * 2, ncols);
            sf = 0;
        }
    }
    flush(acc, yi + byte0 * 2, ncols);
}

void ternary_gemm(const int8_t *x, const uint8_t *w_tl1, int32_t *y, int m, int n_padded, int k,
                  int n_pairs) {
    int hn = n_padded / 2;

    // decode: parallelize over column tiles within each row
    if (m <= 4) {
        for (int i = 0; i < m; i++) {
            uint8_t *lut_buf = (uint8_t *)alloca(n_pairs * 32);
            precompute_luts(x + i * k, n_pairs, lut_buf);
            int32_t *yi = y + i * n_padded;

            // align chunk to 32 packed bytes (64 output cols)
            int nth = omp_get_max_threads();
            int chunk = ((hn / nth + 31) / 32) * 32;
            if (chunk < 32)
                chunk = 32;

#pragma omp parallel for schedule(static)
            for (int t = 0; t < nth; t++) {
                int b0 = t * chunk;
                int b1 = b0 + chunk;
                if (b0 >= hn)
                    continue;

                if (b1 > hn)
                    b1 = hn;

                gemm_col_slice(lut_buf, w_tl1, yi, hn, n_pairs, b0, b1 - b0);
            }
        }

    } else {

// prefill: parallelize over rows
#pragma omp parallel for schedule(dynamic)
        for (int i = 0; i < m; i++) {
            uint8_t *lut_buf = (uint8_t *)alloca(n_pairs * 32);
            precompute_luts(x + i * k, n_pairs, lut_buf);
            gemm_col_slice(lut_buf, w_tl1, y + i * n_padded, hn, n_pairs, 0, hn);
        }
    }
}

#ifdef __cplusplus
}
#endif
