#include "../include/ggml.h"

// ggml_compute_forward_win_part

static void ggml_compute_forward_win_part_f32(
    const struct ggml_tensor* src0,
    struct ggml_tensor* dst, int32_t* params) {

    GGML_TENSOR_LOCALS(int64_t, ne0, src0, ne)
        GGML_TENSOR_LOCALS(int64_t, ne, dst, ne)

    const int32_t nep0 = params[0];
    const int32_t nep1 = params[1];
    const int32_t w = params[2];

    // TODO: optimize / multi-thread
    for (int py = 0; py < nep1; ++py) {
        for (int px = 0; px < nep0; ++px) {
            const int64_t i3 = py * nep0 + px;
            for (int64_t i2 = 0; i2 < ne2; ++i2) {
                for (int64_t i1 = 0; i1 < ne1; ++i1) {
                    for (int64_t i0 = 0; i0 < ne0; ++i0) {
                        const int64_t i02 = py * w + i2;
                        const int64_t i01 = px * w + i1;
                        const int64_t i00 = i0;

                        const int64_t i = i3 * ne2 * ne1 * ne0 + i2 * ne1 * ne0 + i1 * ne0 + i0;
                        const int64_t j = i02 * ne01 * ne00 + i01 * ne00 + i00;

                        if (py * w + i2 >= ne02 || px * w + i1 >= ne01) {
                            ((float*)dst->data)[i] = 0.0f;
                        }
                        else {
                            ((float*)dst->data)[i] = ((float*)src0->data)[j];
                        }
                    }
                }
            }
        }
    }
}

static void ggml_compute_forward_win_part(
    const struct ggml_tensor* src0,
    struct ggml_tensor* dst, int32_t *params) {

    switch (src0->type) {
    case GGML_TYPE_F32:
    {
        ggml_compute_forward_win_part_f32(src0, dst, params);
    } break;
    default:
    {
        // GGML_ABORT("fatal error");
    }
    }
}

// ggml_win_part

struct ggml_tensor* ggml_win_part(
    struct ggml_tensor* a,
    int                   w) {

    // padding
    const int px = (w - a->ne[1] % w) % w;
    const int py = (w - a->ne[2] % w) % w;

    const int npx = (px + a->ne[1]) / w;
    const int npy = (py + a->ne[2]) / w;
    const int np = npx * npy;

    const int64_t ne[4] = { a->ne[0], w, w, np, };
    struct ggml_tensor* result = ggml_new_tensor(GGML_TYPE_F32, 4, ne);

    int32_t params[] = { npx, npy, w };

    ggml_compute_forward_win_part(a, result, params);

    return result;
}