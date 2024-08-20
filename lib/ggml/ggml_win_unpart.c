#include "../include/ggml.h"

// ggml_compute_forward_win_unpart

static void ggml_compute_forward_win_unpart_f32(
    const struct ggml_tensor* src0,
    struct ggml_tensor* dst, const int32_t w) {

    GGML_TENSOR_LOCALS(int64_t, ne0, src0, ne)
    GGML_TENSOR_LOCALS(int64_t, ne, dst, ne)

    // padding
    const int px = (w - ne1 % w) % w;
    //const int py = (w - ne2%w)%w;

    const int npx = (px + ne1) / w;
    //const int npy = (py + ne2)/w;

    // TODO: optimize / multi-thread
    for (int64_t i2 = 0; i2 < ne2; ++i2) {
        for (int64_t i1 = 0; i1 < ne1; ++i1) {
            for (int64_t i0 = 0; i0 < ne0; ++i0) {
                const int ip2 = i2 / w;
                const int ip1 = i1 / w;

                const int64_t i02 = i2 % w;
                const int64_t i01 = i1 % w;
                const int64_t i00 = i0;

                const int64_t i = (ip2 * npx + ip1) * ne02 * ne01 * ne00 + i02 * ne01 * ne00 + i01 * ne00 + i00;
                const int64_t j = i2 * ne1 * ne0 + i1 * ne0 + i0;

                ((float*)dst->data)[j] = ((float*)src0->data)[i];
            }
        }
    }
}

static void ggml_compute_forward_win_unpart(
    const struct ggml_tensor* src0,
    struct ggml_tensor* dst, int w) {

    switch (src0->type) {
    case GGML_TYPE_F32:
    {
        ggml_compute_forward_win_unpart_f32(src0, dst, w);
    } break;
    default:
    {
        // GGML_ABORT("fatal error");
    }
    }
}

// ggml_win_unpart

struct ggml_tensor * ggml_win_unpart(
        struct ggml_tensor  * a,
        int                   w0,
        int                   h0,
        int                   w) {

    const int64_t ne[4] = { a->ne[0], w0, h0, 1, };
    struct ggml_tensor * result = ggml_new_tensor(GGML_TYPE_F32, 3, ne);

    ggml_compute_forward_win_unpart(a, result, w);

    return result;
}
