#include <immintrin.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

// segment layout (all int32):
//   breakpoints[n_segs], intercepts[n_segs]
//   signs[n_segs * 2], exps[n_segs * 2]  (term0 at [seg*2], term1 at [seg*2+1])
void plac_eval_segments(const int32_t *x, int32_t *y, int n, const int32_t *breakpoints,
                        const int32_t *intercepts, const int32_t *signs, const int32_t *exps,
                        int n_segs) {
    int jb;
    for (jb = 0; jb + 7 < n; jb += 8) {
        __m256i vx = _mm256_loadu_si256((__m256i *)(x + jb));

        // segment lookup: linear scan with vectorized comparison
        __m256i seg = _mm256_setzero_si256();
        for (int s = 1; s < n_segs; s++) {
            __m256i mask = _mm256_cmpgt_epi32(vx, _mm256_set1_epi32(breakpoints[s]));
            seg = _mm256_blendv_epi8(seg, _mm256_set1_epi32(s), mask);
        }

        // gather per-segment data (seg*2 for term indexing)
        __m256i seg2 = _mm256_slli_epi32(seg, 1);
        __m256i seg2p1 = _mm256_add_epi32(seg2, _mm256_set1_epi32(1));

        __m256i vintercept = _mm256_i32gather_epi32(intercepts, seg, 4);
        __m256i vs0 = _mm256_i32gather_epi32(signs, seg2, 4);
        __m256i ve0 = _mm256_i32gather_epi32(exps, seg2, 4);
        __m256i vs1 = _mm256_i32gather_epi32(signs, seg2p1, 4);
        __m256i ve1 = _mm256_i32gather_epi32(exps, seg2p1, 4);

        // evaluate: sign * (x shift exp) per term. negative exp = right shift.
        __m256i zero = _mm256_setzero_si256();
        __m256i ae0 = _mm256_abs_epi32(ve0);
        __m256i ae1 = _mm256_abs_epi32(ve1);
        __m256i pos0 = _mm256_cmpgt_epi32(ve0, zero);
        __m256i pos1 = _mm256_cmpgt_epi32(ve1, zero);

        __m256i sh0 =
            _mm256_blendv_epi8(_mm256_srav_epi32(vx, ae0), _mm256_sllv_epi32(vx, ae0), pos0);
        __m256i sh1 =
            _mm256_blendv_epi8(_mm256_srav_epi32(vx, ae1), _mm256_sllv_epi32(vx, ae1), pos1);

        __m256i t0 = _mm256_sign_epi32(sh0, vs0);
        __m256i t1 = _mm256_sign_epi32(sh1, vs1);

        _mm256_storeu_si256((__m256i *)(y + jb),
                            _mm256_add_epi32(_mm256_add_epi32(t0, t1), vintercept));
    }

    for (; jb < n; jb++) {
        int32_t val = x[jb];
        int seg = 0;
        for (int s = 1; s < n_segs; s++)
            if (val > breakpoints[s])
                seg = s;
        int32_t result = intercepts[seg];
        for (int t = 0; t < 2; t++) {
            int e = exps[seg * 2 + t];
            int32_t shifted = (e >= 0) ? (val << e) : (val >> -e);
            result += signs[seg * 2 + t] * shifted;
        }
        y[jb] = result;
    }
}

#ifdef __cplusplus
}
#endif
