#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

// Dense LUT: y = lut[clamp((x - x_lo) >> shift)]
void plac_eval_lut(const int32_t *x, int32_t *y, int n, const int32_t *lut, int lut_size,
                   int32_t x_lo, int shift) {
    int max_idx = lut_size - 1;

    for (int k = 0; k < n; k++) {
        int idx = (x[k] - x_lo) >> shift;

        if (idx < 0)
            idx = 0;
        if (idx > max_idx)
            idx = max_idx;

        y[k] = lut[idx];
    }
}

#ifdef __cplusplus
}
#endif
