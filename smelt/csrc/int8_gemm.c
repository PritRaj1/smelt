#include <immintrin.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

// 8 x int32 -> scalar via shuffle+add tree (faster than hadd on most uarchs)
static inline int32_t hsum_i32(__m256i v) {
    __m128i lo = _mm256_castsi256_si128(v);
    __m128i hi = _mm256_extracti128_si256(v, 1);
    lo = _mm_add_epi32(lo, hi);
    lo = _mm_add_epi32(lo, _mm_shuffle_epi32(lo, 0x4E)); // [0+2, 1+3, ...]
    lo = _mm_add_epi32(lo, _mm_shuffle_epi32(lo, 0xB1)); // [0+1+2+3, ...]
    return _mm_cvtsi128_si32(lo);
}

// dot(ai, bj) via cvtepi8_epi16 + madd_epi16 (safe for full int8 range)
static inline int32_t dot_i8(const int8_t *ai, const int8_t *bj, int k) {
    __m256i acc = _mm256_setzero_si256();
    int d;
    for (d = 0; d + 15 < k; d += 16) {
        __m256i va = _mm256_cvtepi8_epi16(_mm_loadu_si128((__m128i *)(ai + d)));
        __m256i vb = _mm256_cvtepi8_epi16(_mm_loadu_si128((__m128i *)(bj + d)));
        acc = _mm256_add_epi32(acc, _mm256_madd_epi16(va, vb));
    }
    int32_t sum = hsum_i32(acc);
    for (; d < k; d++)
        sum += (int32_t)ai[d] * bj[d];

    return sum;
}

// TILE_N dot products sharing the same A row load
#define TILE_N 12

static inline void dot_tile(const int8_t *ai, const int8_t *b, int32_t *ci, int j, int k) {
    __m256i acc[TILE_N];
    for (int t = 0; t < TILE_N; t++)
        acc[t] = _mm256_setzero_si256();

    for (int d = 0; d + 15 < k; d += 16) {
        __m256i va = _mm256_cvtepi8_epi16(_mm_loadu_si128((__m128i *)(ai + d)));
        for (int t = 0; t < TILE_N; t++) {
            __m256i vb = _mm256_cvtepi8_epi16(_mm_loadu_si128((__m128i *)(b + (j + t) * k + d)));
            acc[t] = _mm256_add_epi32(acc[t], _mm256_madd_epi16(va, vb));
        }
    }

    for (int t = 0; t < TILE_N; t++)
        ci[j + t] = hsum_i32(acc[t]);
}

// c[m,n] = a[m,k] @ b[n,k]^T, int8 x int8 -> int32
void int8_gemm_t(const int8_t *a, const int8_t *b, int32_t *c, int m, int n, int k) {
    int n_tiles = n / TILE_N;

    // parallelize over j-tiles if decode, else over rows for prefill
    if (m <= 4) {
        for (int i = 0; i < m; i++) {
            const int8_t *ai = a + i * k;
            int32_t *ci = c + i * n;
#pragma omp parallel for schedule(static)
            for (int jt = 0; jt < n_tiles; jt++)
                dot_tile(ai, b, ci, jt * TILE_N, k);

            for (int j = n_tiles * TILE_N; j < n; j++)
                ci[j] = dot_i8(ai, b + j * k, k);
        }
    } else {
#pragma omp parallel for schedule(dynamic)
        for (int i = 0; i < m; i++) {
            const int8_t *ai = a + i * k;
            int32_t *ci = c + i * n;
            for (int jt = 0; jt < n_tiles; jt++)
                dot_tile(ai, b, ci, jt * TILE_N, k);

            for (int j = n_tiles * TILE_N; j < n; j++)
                ci[j] = dot_i8(ai, b + j * k, k);
        }
    }
}

void int8_batched_gemm_t(const int8_t *a, const int8_t *b, int32_t *c, int batch, int m, int n,
                         int k) {
    for (int h = 0; h < batch; h++)
        int8_gemm_t(a + h * m * k, b + h * n * k, c + h * m * n, m, n, k);
}

#undef TILE_N

#ifdef __cplusplus
}
#endif
