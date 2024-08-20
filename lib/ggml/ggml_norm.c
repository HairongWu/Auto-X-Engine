#include "../include/ggml.h"

// ggml_compute_forward_norm

static void ggml_compute_forward_norm_f32(
    const struct ggml_tensor* src0,
    struct ggml_tensor* dst, float eps) {

    const int ith = 0;
    const int nth = 1;

    GGML_TENSOR_UNARY_OP_LOCALS

    // TODO: optimize
    for (int64_t i03 = 0; i03 < ne03; i03++) {
        for (int64_t i02 = 0; i02 < ne02; i02++) {
            for (int64_t i01 = ith; i01 < ne01; i01 += nth) {
                const float* x = (float*)((char*)src0->data + i01 * nb01 + i02 * nb02 + i03 * nb03);

                float sum = 0.0;
                for (int64_t i00 = 0; i00 < ne00; i00++) {
                    sum += (float)x[i00];
                }

                float mean = sum / ne00;

                float* y = (float*)((char*)dst->data + i01 * nb1 + i02 * nb2 + i03 * nb3);

                float sum2 = 0.0;
                for (int64_t i00 = 0; i00 < ne00; i00++) {
                    float v = x[i00] - mean;
                    y[i00] = v;
                    sum2 += (float)(v * v);
                }

                float variance = sum2 / ne00;
                const float scale = 1.0f / sqrtf(variance + eps);

                ggml_vec_scale_f32(ne00, y, scale);
            }
        }
    }
}

static void ggml_compute_forward_norm(
    const struct ggml_tensor* src0,
    struct ggml_tensor* dst, float eps) {

    switch (src0->type) {
    case GGML_TYPE_F32:
    {
        ggml_compute_forward_norm_f32(src0, dst, eps);
    } break;
    default:
    {
        // GGML_ABORT("fatal error");
    }
    }
}

static struct ggml_tensor* ggml_norm_impl(
    struct ggml_tensor* a,
    float eps,
    bool inplace) {

    struct ggml_tensor* result = inplace ? ggml_view_tensor(a) : ggml_dup_tensor(a);

    ggml_compute_forward_norm(a, result, eps);

    return result;
}

struct ggml_tensor* ggml_norm(
    struct ggml_tensor* a,
    float eps) {
    return ggml_norm_impl(a, eps, false);
}