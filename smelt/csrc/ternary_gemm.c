#include <immintrin.h>
#include <stdint.h>

static inline int32_t hsum_epi32(__m256i v) {
    __m128i lo = _mm256_castsi256_si128(v);
    __m128i hi = _mm256_extracti128_si256(v, 1);
    lo = _mm_add_epi32(lo, hi);
    lo = _mm_hadd_epi32(lo, lo);
    lo = _mm_hadd_epi32(lo, lo);
    return _mm_cvtsi128_si32(lo);
}

/* expand 32 packed bits (LSB-first) to 32-lane int8 mask (0x00 or 0xFF) */
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

/* reverse bits within each byte of a uint32 (MSB-first packing -> LSB-first) */
static inline uint32_t rev_bits_per_byte(uint32_t x) {
    x = ((x & 0xF0F0F0F0u) >> 4) | ((x & 0x0F0F0F0Fu) << 4);
    x = ((x & 0xCCCCCCCCu) >> 2) | ((x & 0x33333333u) << 2);
    x = ((x & 0xAAAAAAAAu) >> 1) | ((x & 0x55555555u) << 1);
    return x;
}

/* y[m,n] = x[m,k] @ w[n,k].T
 * x: int8 [m,k], w packed as val/sign bitmasks [n, k/8] */
void ternary_gemm(const int8_t *x, const uint8_t *w_val, const uint8_t *w_sign, int32_t *y, int m,
                  int n, int k) {
    int k8 = k / 8;
    int k32 = k / 32;

    for (int i = 0; i < m; i++) {
        const int8_t *xi = x + i * k;
        for (int j = 0; j < n; j++) {
            const uint8_t *vj = w_val + j * k8;
            const uint8_t *sj = w_sign + j * k8;
            __m256i acc = _mm256_setzero_si256();

            for (int b = 0; b < k32; b++) {
                __m256i xv = _mm256_loadu_si256((__m256i *)(xi + b * 32));
                uint32_t val = *(uint32_t *)(vj + b * 4);
                uint32_t sgn = *(uint32_t *)(sj + b * 4);

                uint32_t pos = rev_bits_per_byte(val & ~sgn);
                uint32_t neg = rev_bits_per_byte(val & sgn);

                __m256i ones16 = _mm256_set1_epi16(1);
                __m256i pv = _mm256_blendv_epi8(_mm256_setzero_si256(), xv, expand_mask(pos));
                __m256i nv = _mm256_blendv_epi8(_mm256_setzero_si256(), xv, expand_mask(neg));

                /* low 16 lanes */
                acc = _mm256_add_epi32(
                    acc,
                    _mm256_madd_epi16(_mm256_cvtepi8_epi16(_mm256_castsi256_si128(pv)), ones16));
                acc = _mm256_sub_epi32(
                    acc,
                    _mm256_madd_epi16(_mm256_cvtepi8_epi16(_mm256_castsi256_si128(nv)), ones16));
                /* high 16 lanes */
                acc = _mm256_add_epi32(
                    acc, _mm256_madd_epi16(_mm256_cvtepi8_epi16(_mm256_extracti128_si256(pv, 1)),
                                           ones16));
                acc = _mm256_sub_epi32(
                    acc, _mm256_madd_epi16(_mm256_cvtepi8_epi16(_mm256_extracti128_si256(nv, 1)),
                                           ones16));
            }

            y[i * n + j] = hsum_epi32(acc);
        }
    }
}
