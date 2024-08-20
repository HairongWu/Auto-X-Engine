#include "../include/ggml.h"


// ggml_compute_forward_get_rel_pos

static void ggml_compute_forward_get_rel_pos_f16(
    const struct ggml_tensor* src0,
    struct ggml_tensor* dst) {

    // ref: https://github.com/facebookresearch/segment-anything/blob/main/segment_anything/modeling/image_encoder.py#L292-L322

    GGML_TENSOR_UNARY_OP_LOCALS

        const int64_t w = ne1;

    ggml_fp16_t* src0_data = (ggml_fp16_t*)src0->data;
    ggml_fp16_t* dst_data = (ggml_fp16_t*)dst->data;

    for (int64_t i2 = 0; i2 < ne2; ++i2) {
        for (int64_t i1 = 0; i1 < ne1; ++i1) {
            const int64_t pos = (w - i1 - 1) + i2;
            for (int64_t i0 = 0; i0 < ne0; ++i0) {
                dst_data[i2 * ne1 * ne0 + i1 * ne0 + i0] = src0_data[pos * ne00 + i0];
            }
        }
    }
}

static void ggml_compute_forward_get_rel_pos(
    const struct ggml_tensor* src0,
    struct ggml_tensor* dst) {

    switch (src0->type) {
    case GGML_TYPE_F16:
    case GGML_TYPE_BF16:
    {
        ggml_compute_forward_get_rel_pos_f16(src0, dst);
    } break;
    default:
    {
        // GGML_ABORT("fatal error");
    }
    }
}

// ggml_get_rel_pos

struct ggml_tensor* ggml_get_rel_pos(
    struct ggml_tensor* a,
    int                   qh,
    int                   kh) {

    const int64_t ne[4] = { a->ne[0], kh, qh, 1, };
    struct ggml_tensor* result = ggml_new_tensor(GGML_TYPE_F16, 3, ne);

    ggml_compute_forward_get_rel_pos(a, result);

    return result;
}