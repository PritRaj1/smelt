#include <immintrin.h>
#include <stdint.h>

#define FRAC 16
#define ONE (1 << FRAC)
#define LOG2E_FIX 94548

// 2^(f/256) LUT, Q16.16
static const int32_t EXP2_LUT[256] = {
    65536,  65713,  65890,  66068,  66246,  66425,  66604,  66784,  66964,  67145,  67327,  67509,
    67692,  67875,  68059,  68244,  68429,  68615,  68801,  68988,  69176,  69364,  69553,  69742,
    69932,  70123,  70314,  70506,  70699,  70892,  71086,  71281,  71476,  71672,  71869,  72066,
    72264,  72463,  72662,  72862,  73063,  73264,  73466,  73669,  73873,  74077,  74282,  74488,
    74694,  74901,  75109,  75318,  75527,  75737,  75948,  76159,  76372,  76585,  76799,  77013,
    77229,  77445,  77662,  77879,  78098,  78317,  78537,  78758,  78979,  79202,  79425,  79649,
    79874,  80099,  80326,  80553,  80781,  81010,  81240,  81471,  81703,  81935,  82168,  82402,
    82637,  82873,  83110,  83347,  83586,  83825,  84065,  84306,  84548,  84791,  85035,  85280,
    85525,  85772,  86019,  86268,  86517,  86767,  87018,  87270,  87523,  87777,  88032,  88288,
    88545,  88803,  89062,  89321,  89582,  89844,  90107,  90371,  90636,  90902,  91169,  91437,
    91706,  91976,  92247,  92519,  92792,  93066,  93341,  93618,  93895,  94173,  94453,  94733,
    95015,  95297,  95581,  95866,  96152,  96439,  96727,  97016,  97307,  97598,  97891,  98185,
    98480,  98776,  99073,  99372,  99671,  99972,  100274, 100577, 100882, 101187, 101494, 101802,
    102111, 102422, 102733, 103046, 103360, 103676, 103993, 104311, 104630, 104951, 105273, 105596,
    105921, 106247, 106574, 106903, 107233, 107564, 107896, 108230, 108566, 108903, 109241, 109580,
    109921, 110264, 110607, 110953, 111299, 111648, 111997, 112348, 112701, 113055, 113410, 113767,
    114126, 114486, 114848, 115211, 115576, 115942, 116310, 116679, 117050, 117423, 117797, 118173,
    118550, 118929, 119310, 119692, 120076, 120462, 120849, 121238, 121629, 122021, 122415, 122811,
    123209, 123608, 124009, 124412, 124816, 125223, 125631, 126041, 126453, 126866, 127282, 127699,
    128118, 128539, 128962, 129387, 129814, 130242, 130673, 131105, 131540, 131976, 132415, 132855,
    133297, 133742, 134188, 134637, 135087, 135540, 135995, 136452, 136911, 137372, 137835, 138300,
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
