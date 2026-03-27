#include <immintrin.h>
#include <stdint.h>

#define FRAC 16
#define ONE (1 << FRAC)
#define LOG2E_FIX 94548

// 2^(f/256) LUT, Q16.16
static const int32_t EXP2_LUT[256] = {
    65536,  65714,  65892,  66071,  66250,  66429,  66609,  66790,  66971,  67153,  67335,  67517,
    67700,  67884,  68068,  68252,  68438,  68623,  68809,  68996,  69183,  69370,  69558,  69747,
    69936,  70126,  70316,  70507,  70698,  70889,  71082,  71274,  71468,  71661,  71856,  72050,
    72246,  72442,  72638,  72835,  73032,  73230,  73429,  73628,  73828,  74028,  74229,  74430,
    74632,  74834,  75037,  75240,  75444,  75649,  75854,  76060,  76266,  76473,  76680,  76888,
    77096,  77305,  77515,  77725,  77936,  78147,  78359,  78572,  78785,  78998,  79212,  79427,
    79642,  79858,  80075,  80292,  80510,  80728,  80947,  81166,  81386,  81607,  81828,  82050,
    82273,  82496,  82719,  82944,  83169,  83394,  83620,  83847,  84074,  84302,  84531,  84760,
    84990,  85220,  85451,  85683,  85915,  86148,  86382,  86616,  86851,  87086,  87322,  87559,
    87796,  88034,  88273,  88513,  88752,  88993,  89234,  89476,  89719,  89962,  90206,  90451,
    90696,  90942,  91188,  91436,  91684,  91932,  92181,  92431,  92682,  92933,  93185,  93438,
    93691,  93945,  94200,  94455,  94711,  94968,  95226,  95484,  95743,  96002,  96263,  96524,
    96785,  97048,  97311,  97575,  97839,  98104,  98370,  98637,  98905,  99173,  99442,  99711,
    99982,  100253, 100524, 100797, 101070, 101344, 101619, 101895, 102171, 102448, 102726, 103004,
    103283, 103564, 103844, 104126, 104408, 104691, 104975, 105260, 105545, 105831, 106118, 106406,
    106694, 106984, 107274, 107565, 107856, 108149, 108442, 108736, 109031, 109326, 109623, 109920,
    110218, 110517, 110816, 111117, 111418, 111720, 112023, 112327, 112631, 112937, 113243, 113550,
    113858, 114167, 114476, 114787, 115098, 115410, 115723, 116036, 116351, 116667, 116983, 117300,
    117618, 117937, 118257, 118577, 118899, 119221, 119544, 119869, 120194, 120519, 120846, 121174,
    121502, 121832, 122162, 122493, 122825, 123158, 123492, 123827, 124163, 124500, 124837, 125176,
    125515, 125855, 126197, 126539, 126882, 127226, 127571, 127917, 128263, 128611, 128960, 129310,
    129660, 130012, 130364, 130718,
};

static void softmax_row(const int32_t *x, int32_t *y, int n) {
    // max (AVX2 8-wide)
    int i = 0;
    __m256i vmax = _mm256_set1_epi32(x[0]);
    for (; i + 7 < n; i += 8)
        vmax = _mm256_max_epi32(vmax, _mm256_loadu_si256((__m256i *)(x + i)));
    __m128i lo128 = _mm256_castsi256_si128(vmax);
    __m128i hi128 = _mm256_extracti128_si256(vmax, 1);
    lo128 = _mm_max_epi32(lo128, hi128);
    lo128 = _mm_max_epi32(lo128, _mm_shuffle_epi32(lo128, 0x4E));
    lo128 = _mm_max_epi32(lo128, _mm_shuffle_epi32(lo128, 0xB1));
    int32_t m = _mm_cvtsi128_si32(lo128);
    for (; i < n; i++)
        if (x[i] > m)
            m = x[i];

    // exp + sum (SIMD except int64 widen)
    int32_t exp_buf[n];
    int64_t s = 0;
    __m256i vm = _mm256_set1_epi32(m);
    __m256i vlog2e = _mm256_set1_epi32(LOG2E_FIX);
    __m256i vmask = _mm256_set1_epi32(ONE - 1);
    __m256i vclip = _mm256_set1_epi32(-(20 << FRAC));

    i = 0;
    for (; i + 7 < n; i += 8) {
        __m256i vshifted =
            _mm256_max_epi32(_mm256_sub_epi32(_mm256_loadu_si256((__m256i *)(x + i)), vm), vclip);

        // (val>>8) * log2e >> 8 to avoid int32 overflow
        __m256i vb2 =
            _mm256_srai_epi32(_mm256_mullo_epi32(_mm256_srai_epi32(vshifted, 8), vlog2e), 8);
        __m256i vq = _mm256_srai_epi32(vb2, FRAC);
        __m256i vfidx = _mm256_and_si256(_mm256_srli_epi32(_mm256_and_si256(vb2, vmask), 8),
                                         _mm256_set1_epi32(0xFF));

        __m256i vlut = _mm256_i32gather_epi32(EXP2_LUT, vfidx, 4);

        // LUT values always positive, so unsigned srlv is correct
        __m256i vnegq =
            _mm256_min_epi32(_mm256_sub_epi32(_mm256_setzero_si256(), vq), _mm256_set1_epi32(31));
        __m256i vexp = _mm256_srlv_epi32(vlut, vnegq);
        _mm256_storeu_si256((__m256i *)(exp_buf + i), vexp);

        // sum from store (already in L1)
        for (int j = 0; j < 8; j++)
            s += (uint32_t)exp_buf[i + j];
    }

    for (; i < n; i++) {
        int64_t val = (int64_t)(x[i] - m);
        if (val < -(20LL << FRAC)) {
            exp_buf[i] = 0;
            continue;
        }
        int64_t vb2 = (val * LOG2E_FIX) >> FRAC;
        int q = (int)(vb2 >> FRAC);
        int fidx = (int)((vb2 & (ONE - 1)) >> 8) & 0xFF;
        int nq = -q;
        exp_buf[i] = (nq < 32) ? (EXP2_LUT[fidx] >> nq) : 0;
        s += exp_buf[i];
    }

    if (s == 0) {
        for (i = 0; i < n; i++)
            y[i] = 0;
        return;
    }

    // reciprocal multiply: div per row
    uint64_t inv_s = (1ULL << 48) / (uint64_t)s;
    for (i = 0; i < n; i++)
        y[i] = (int32_t)(((int64_t)exp_buf[i] * inv_s) >> 32);
}

void softmax_int(const int32_t *x, int32_t *y, int rows, int cols) {
#pragma omp parallel for schedule(dynamic)
    for (int r = 0; r < rows; r++)
        softmax_row(x + r * cols, y + r * cols, cols);
}
