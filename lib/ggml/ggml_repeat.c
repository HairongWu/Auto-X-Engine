#include "../include/ggml.h"

inline static void ggml_vec_cpy_f32(const int n, float* y, const float* x) { for (int i = 0; i < n; ++i) y[i] = x[i]; }
// ggml_compute_forward_repeat

static void ggml_compute_forward_repeat_f32(
    const struct ggml_tensor* src0,
    struct ggml_tensor* dst) {

    GGML_TENSOR_UNARY_OP_LOCALS

    // guaranteed to be an integer due to the check in ggml_can_repeat
    const int nr0 = (int)(ne0 / ne00);
    const int nr1 = (int)(ne1 / ne01);
    const int nr2 = (int)(ne2 / ne02);
    const int nr3 = (int)(ne3 / ne03);

    // TODO: maybe this is not optimal?
    for (int i3 = 0; i3 < nr3; i3++) {
        for (int k3 = 0; k3 < ne03; k3++) {
            for (int i2 = 0; i2 < nr2; i2++) {
                for (int k2 = 0; k2 < ne02; k2++) {
                    for (int i1 = 0; i1 < nr1; i1++) {
                        for (int k1 = 0; k1 < ne01; k1++) {
                            for (int i0 = 0; i0 < nr0; i0++) {
                                ggml_vec_cpy_f32(ne00,
                                    (float*)((char*)dst->data + (i3 * ne03 + k3) * nb3 + (i2 * ne02 + k2) * nb2 + (i1 * ne01 + k1) * nb1 + (i0 * ne00) * nb0),
                                    (float*)((char*)src0->data + (k3)*nb03 + (k2)*nb02 + (k1)*nb01));
                            }
                        }
                    }
                }
            }
        }
    }
}

static void ggml_compute_forward_repeat_f16(
    const struct ggml_tensor* src0,
    struct ggml_tensor* dst) {

    GGML_TENSOR_UNARY_OP_LOCALS

    // guaranteed to be an integer due to the check in ggml_can_repeat
    const int nr0 = (int)(ne0 / ne00);
    const int nr1 = (int)(ne1 / ne01);
    const int nr2 = (int)(ne2 / ne02);
    const int nr3 = (int)(ne3 / ne03);

    // TODO: maybe this is not optimal?
    for (int i3 = 0; i3 < nr3; i3++) {
        for (int k3 = 0; k3 < ne03; k3++) {
            for (int i2 = 0; i2 < nr2; i2++) {
                for (int k2 = 0; k2 < ne02; k2++) {
                    for (int i1 = 0; i1 < nr1; i1++) {
                        for (int k1 = 0; k1 < ne01; k1++) {
                            for (int i0 = 0; i0 < nr0; i0++) {
                                ggml_fp16_t* y = (ggml_fp16_t*)((char*)dst->data + (i3 * ne03 + k3) * nb3 + (i2 * ne02 + k2) * nb2 + (i1 * ne01 + k1) * nb1 + (i0 * ne00) * nb0);
                                ggml_fp16_t* x = (ggml_fp16_t*)((char*)src0->data + (k3)*nb03 + (k2)*nb02 + (k1)*nb01);
                                // ggml_vec_cpy_f16(ne00, y, x)
                                for (int i = 0; i < ne00; ++i) {
                                    y[i] = x[i];
                                }
                            }
                        }
                    }
                }
            }
        }
    }
}

static void ggml_compute_forward_repeat(
    const struct ggml_tensor* src0,
    struct ggml_tensor* dst) {

    switch (src0->type) {
    case GGML_TYPE_F16:
    case GGML_TYPE_BF16:
    case GGML_TYPE_I16:
    {
        ggml_compute_forward_repeat_f16(src0, dst);
    } break;
    case GGML_TYPE_F32:
    case GGML_TYPE_I32:
    {
        ggml_compute_forward_repeat_f32(src0, dst);
    } break;
    default:
    {
        // GGML_ABORT("fatal error");
    }
    }
}
// ggml_repeat

struct ggml_tensor* ggml_repeat(
    struct ggml_tensor* a,
    struct ggml_tensor* b) {

    struct ggml_tensor* result = ggml_new_tensor(a->type, GGML_MAX_DIMS, b->ne);


    return result;
}