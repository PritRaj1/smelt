#include <stdint.h>

void plac_eval_int(const int32_t *x, int32_t *y, int n, int n_seg, int n_terms, const int32_t *bp,
                   const int32_t *signs, const int32_t *exps, const int32_t *intercepts) {
    int32_t x_lo = bp[0];
    int32_t range = bp[n_seg] - x_lo;

    for (int k = 0; k < n; k++) {
        int32_t xk = x[k];

        int seg = (int)(((int64_t)(xk - x_lo) * n_seg) / range);
        if (seg < 0)
            seg = 0;
        if (seg >= n_seg)
            seg = n_seg - 1;

        while (seg < n_seg - 1 && xk >= bp[seg + 1])
            seg++;
        while (seg > 0 && xk < bp[seg])
            seg--;

        int32_t acc = 0;
        const int32_t *s = signs + seg * n_terms;
        const int32_t *e = exps + seg * n_terms;
        for (int j = 0; j < n_terms; j++) {
            if (s[j] == 0)
                break;
            int32_t shifted = e[j] >= 0 ? (xk << e[j]) : (xk >> (-e[j]));
            acc += s[j] * shifted;
        }

        y[k] = acc + intercepts[seg];
    }
}
