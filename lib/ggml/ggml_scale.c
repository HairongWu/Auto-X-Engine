#include "../include/ggml.h"

// ggml_compute_forward_scale

static void ggml_compute_forward_scale_f32(
    const struct ggml_tensor* src0,
    struct ggml_tensor* dst, float v) {

    const int ith = 0;
    const int nth = 1;

    const int nc = src0->ne[0];
    const int nr = ggml_nrows(src0);

    // rows per thread
    const int dr = (nr + nth - 1) / nth;

    // row range for this thread
    const int ir0 = dr * ith;
    const int ir1 = min(ir0 + dr, nr);

    const size_t nb01 = src0->nb[1];

    const size_t nb1 = dst->nb[1];

    for (int i1 = ir0; i1 < ir1; i1++) {
        if (dst->data != src0->data) {
            // src0 is same shape as dst => same indices
            memcpy((char*)dst->data + i1 * nb1, (char*)src0->data + i1 * nb01, nc * sizeof(float));
        }
        ggml_vec_scale_f32(nc, (float*)((char*)dst->data + i1 * nb1), v);
    }
}

static void ggml_compute_forward_scale(
    const struct ggml_tensor* src0,
    struct ggml_tensor* dst, float s) {

    switch (src0->type) {
    case GGML_TYPE_F32:
    {
        ggml_compute_forward_scale_f32(src0, dst, s);
    } break;
    default:
    {
        // GGML_ABORT("fatal error");
    }
    }
}

// ggml_scale

static struct ggml_tensor* ggml_scale_impl(
    struct ggml_tensor* a,
    float                 s,
    bool inplace) {

    struct ggml_tensor* result = inplace ? ggml_view_tensor(a) : ggml_dup_tensor(a);

    ggml_compute_forward_scale(a, result, s);

    return result;
}

struct ggml_tensor* ggml_scale(
    struct ggml_tensor* a,
    float                s) {
    return ggml_scale_impl(a, s, false);
}

struct ggml_tensor* ggml_scale_inplace(
    struct ggml_tensor* a,
    float                s) {
    return ggml_scale_impl(a, s, true);
}
