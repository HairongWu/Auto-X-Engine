#include "../include/ggml.h"

inline static void ggml_vec_mul_f32(const int n, float* z, const float* x, const float* y) { for (int i = 0; i < n; ++i) z[i] = x[i] * y[i]; }
// ggml_compute_forward_mul

static void ggml_compute_forward_mul_f32(
    const struct ggml_tensor* src0,
    const struct ggml_tensor* src1,
    struct ggml_tensor* dst) {

    const int ith = 0;
    const int nth = 1;

    const int64_t nr = ggml_nrows(src0);

    GGML_TENSOR_BINARY_OP_LOCALS

    if (nb10 == sizeof(float)) {
        for (int64_t ir = ith; ir < nr; ir += nth) {
            // src0 and dst are same shape => same indices
            const int64_t i03 = ir / (ne02 * ne01);
            const int64_t i02 = (ir - i03 * ne02 * ne01) / ne01;
            const int64_t i01 = (ir - i03 * ne02 * ne01 - i02 * ne01);

            const int64_t i13 = i03 % ne13;
            const int64_t i12 = i02 % ne12;
            const int64_t i11 = i01 % ne11;
            const int64_t nr0 = ne00 / ne10;

            float* dst_ptr = (float*)((char*)dst->data + i03 * nb3 + i02 * nb2 + i01 * nb1);
            float* src0_ptr = (float*)((char*)src0->data + i03 * nb03 + i02 * nb02 + i01 * nb01);
            float* src1_ptr = (float*)((char*)src1->data + i13 * nb13 + i12 * nb12 + i11 * nb11);

            for (int64_t r = 0; r < nr0; ++r) {

                ggml_vec_mul_f32(ne10, dst_ptr + r * ne10, src0_ptr + r * ne10, src1_ptr);
            }
        }
    }
    else {
        // src1 is not contiguous
        for (int64_t ir = ith; ir < nr; ir += nth) {
            // src0 and dst are same shape => same indices
            // src1 is broadcastable across src0 and dst in i1, i2, i3
            const int64_t i03 = ir / (ne02 * ne01);
            const int64_t i02 = (ir - i03 * ne02 * ne01) / ne01;
            const int64_t i01 = (ir - i03 * ne02 * ne01 - i02 * ne01);

            const int64_t i13 = i03 % ne13;
            const int64_t i12 = i02 % ne12;
            const int64_t i11 = i01 % ne11;

            float* dst_ptr = (float*)((char*)dst->data + i03 * nb3 + i02 * nb2 + i01 * nb1);
            float* src0_ptr = (float*)((char*)src0->data + i03 * nb03 + i02 * nb02 + i01 * nb01);

            for (int64_t i0 = 0; i0 < ne00; ++i0) {
                const int64_t i10 = i0 % ne10;
                float* src1_ptr = (float*)((char*)src1->data + i13 * nb13 + i12 * nb12 + i11 * nb11 + i10 * nb10);

                dst_ptr[i0] = src0_ptr[i0] * (*src1_ptr);
            }
        }
    }
}

static void ggml_compute_forward_mul(
    const struct ggml_tensor* src0,
    const struct ggml_tensor* src1,
    struct ggml_tensor* dst) {

    switch (src0->type) {
    case GGML_TYPE_F32:
    {
        ggml_compute_forward_mul_f32(src0, src1, dst);
    } break;
    default:
    {
        // GGML_ABORT("fatal error");
    }
    }
}

// ggml_mul

static struct ggml_tensor* ggml_mul_impl(
    struct ggml_tensor* a,
    struct ggml_tensor* b,
    bool inplace) {

    struct ggml_tensor* result = inplace ? ggml_view_tensor(a) : ggml_dup_tensor(a);

    ggml_compute_forward_mul(a, b, result);

    return result;
}

struct ggml_tensor* ggml_mul(
    struct ggml_tensor* a,
    struct ggml_tensor* b) {
    return ggml_mul_impl(a, b, false);
}