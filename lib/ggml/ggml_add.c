#include "../include/ggml.h"

inline static void ggml_vec_add_f32(const int n, float* z, const float* x, const float* y) { for (int i = 0; i < n; ++i) z[i] = x[i] + y[i]; }

size_t ggml_nbytes(const struct ggml_tensor* tensor) {
    size_t nbytes;
    size_t blck_size = ggml_blck_size(tensor->type);
    if (blck_size == 1) {
        nbytes = ggml_type_size(tensor->type);
        for (int i = 0; i < GGML_MAX_DIMS; ++i) {
            nbytes += (tensor->ne[i] - 1) * tensor->nb[i];
        }
    }
    else {
        nbytes = tensor->ne[0] * tensor->nb[0] / blck_size;
        for (int i = 1; i < GGML_MAX_DIMS; ++i) {
            nbytes += (tensor->ne[i] - 1) * tensor->nb[i];
        }
    }

    return nbytes;
}
// ggml_compute_forward_add

static void ggml_compute_forward_add_f32(
    const struct ggml_tensor* src0,
    const struct ggml_tensor* src1,
    struct ggml_tensor* dst) {

    const int ith = 0;
    const int nth = 1;

    const int nr = ggml_nrows(src0);

    GGML_TENSOR_BINARY_OP_LOCALS

    // rows per thread
    const int dr = (nr + nth - 1) / nth;

    // row range for this thread
    const int ir0 = dr * ith;
    const int ir1 = min(ir0 + dr, nr);

    if (nb10 == sizeof(float)) {
        for (int ir = ir0; ir < ir1; ++ir) {
            // src1 is broadcastable across src0 and dst in i1, i2, i3
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
                ggml_vec_add_f32(ne10, dst_ptr + r * ne10, src0_ptr + r * ne10, src1_ptr);
            }
        }
    }
    else {
        // src1 is not contiguous
        for (int ir = ir0; ir < ir1; ++ir) {
            // src1 is broadcastable across src0 and dst in i1, i2, i3
            const int64_t i03 = ir / (ne02 * ne01);
            const int64_t i02 = (ir - i03 * ne02 * ne01) / ne01;
            const int64_t i01 = (ir - i03 * ne02 * ne01 - i02 * ne01);

            const int64_t i13 = i03 % ne13;
            const int64_t i12 = i02 % ne12;
            const int64_t i11 = i01 % ne11;

            float* dst_ptr = (float*)((char*)dst->data + i03 * nb3 + i02 * nb2 + i01 * nb1);
            float* src0_ptr = (float*)((char*)src0->data + i03 * nb03 + i02 * nb02 + i01 * nb01);

            for (int64_t i0 = 0; i0 < ne0; ++i0) {
                const int64_t i10 = i0 % ne10;
                float* src1_ptr = (float*)((char*)src1->data + i13 * nb13 + i12 * nb12 + i11 * nb11 + i10 * nb10);

                dst_ptr[i0] = src0_ptr[i0] + *src1_ptr;
            }
        }
    }
}

static void ggml_compute_forward_add_f16_f32(
    const struct ggml_tensor* src0,
    const struct ggml_tensor* src1,
    struct ggml_tensor* dst) {

    const int ith = 0;
    const int nth = 1;

    const int nr = ggml_nrows(src0);

    GGML_TENSOR_BINARY_OP_LOCALS

    // rows per thread
    const int dr = (nr + nth - 1) / nth;

    // row range for this thread
    const int ir0 = dr * ith;
    const int ir1 = min(ir0 + dr, nr);

    if (nb10 == sizeof(float)) {
        if (dst->type == GGML_TYPE_F16) {
            for (int ir = ir0; ir < ir1; ++ir) {
                // src0, src1 and dst are same shape => same indices
                const int i3 = ir / (ne2 * ne1);
                const int i2 = (ir - i3 * ne2 * ne1) / ne1;
                const int i1 = (ir - i3 * ne2 * ne1 - i2 * ne1);

                ggml_fp16_t* dst_ptr = (ggml_fp16_t*)((char*)dst->data + i3 * nb3 + i2 * nb2 + i1 * nb1);
                ggml_fp16_t* src0_ptr = (ggml_fp16_t*)((char*)src0->data + i3 * nb03 + i2 * nb02 + i1 * nb01);
                float* src1_ptr = (float*)((char*)src1->data + i3 * nb13 + i2 * nb12 + i1 * nb11);

                for (int i = 0; i < ne0; i++) {
                    dst_ptr[i] = GGML_FP32_TO_FP16(GGML_FP16_TO_FP32(src0_ptr[i]) + src1_ptr[i]);
                }
            }
        }
        else {
            for (int ir = ir0; ir < ir1; ++ir) {
                // src0, src1 and dst are same shape => same indices
                const int i3 = ir / (ne2 * ne1);
                const int i2 = (ir - i3 * ne2 * ne1) / ne1;
                const int i1 = (ir - i3 * ne2 * ne1 - i2 * ne1);

                float* dst_ptr = (float*)((char*)dst->data + i3 * nb3 + i2 * nb2 + i1 * nb1);
                ggml_fp16_t* src0_ptr = (ggml_fp16_t*)((char*)src0->data + i3 * nb03 + i2 * nb02 + i1 * nb01);
                float* src1_ptr = (float*)((char*)src1->data + i3 * nb13 + i2 * nb12 + i1 * nb11);

                for (int i = 0; i < ne0; i++) {
                    dst_ptr[i] = GGML_FP16_TO_FP32(src0_ptr[i]) + src1_ptr[i];
                }
            }
        }
    }
    //else {
    //    // src1 is not contiguous
    //    GGML_ABORT("fatal error");
    //}
}

static void ggml_compute_forward_add_bf16_f32(
    const struct ggml_tensor* src0,
    const struct ggml_tensor* src1,
    struct ggml_tensor* dst) {

    const int ith = 0;
    const int nth = 1;

    const int nr = ggml_nrows(src0);

    GGML_TENSOR_BINARY_OP_LOCALS

    // rows per thread
    const int dr = (nr + nth - 1) / nth;

    // row range for this thread
    const int ir0 = dr * ith;
    const int ir1 = min(ir0 + dr, nr);

    if (nb10 == sizeof(float)) {
        if (dst->type == GGML_TYPE_BF16) {
            for (int ir = ir0; ir < ir1; ++ir) {
                // src0, src1 and dst are same shape => same indices
                const int i3 = ir / (ne2 * ne1);
                const int i2 = (ir - i3 * ne2 * ne1) / ne1;
                const int i1 = (ir - i3 * ne2 * ne1 - i2 * ne1);

                ggml_bf16_t* dst_ptr = (ggml_bf16_t*)((char*)dst->data + i3 * nb3 + i2 * nb2 + i1 * nb1);
                ggml_bf16_t* src0_ptr = (ggml_bf16_t*)((char*)src0->data + i3 * nb03 + i2 * nb02 + i1 * nb01);
                float* src1_ptr = (float*)((char*)src1->data + i3 * nb13 + i2 * nb12 + i1 * nb11);

                for (int i = 0; i < ne0; i++) {
                    dst_ptr[i] = GGML_FP32_TO_BF16(GGML_BF16_TO_FP32(src0_ptr[i]) + src1_ptr[i]);
                }
            }
        }
        else {
            for (int ir = ir0; ir < ir1; ++ir) {
                // src0, src1 and dst are same shape => same indices
                const int i3 = ir / (ne2 * ne1);
                const int i2 = (ir - i3 * ne2 * ne1) / ne1;
                const int i1 = (ir - i3 * ne2 * ne1 - i2 * ne1);

                float* dst_ptr = (float*)((char*)dst->data + i3 * nb3 + i2 * nb2 + i1 * nb1);
                ggml_bf16_t* src0_ptr = (ggml_bf16_t*)((char*)src0->data + i3 * nb03 + i2 * nb02 + i1 * nb01);
                float* src1_ptr = (float*)((char*)src1->data + i3 * nb13 + i2 * nb12 + i1 * nb11);

                for (int i = 0; i < ne0; i++) {
                    dst_ptr[i] = GGML_BF16_TO_FP32(src0_ptr[i]) + src1_ptr[i];
                }
            }
        }
    }
    //else {
    //    // src1 is not contiguous
    //    GGML_ABORT("fatal error");
    //}
}

static void ggml_compute_forward_add_f16_f16(
    const struct ggml_tensor* src0,
    const struct ggml_tensor* src1,
    struct ggml_tensor* dst) {

    const int ith = 0;
    const int nth = 1;

    const int nr = ggml_nrows(src0);

    GGML_TENSOR_BINARY_OP_LOCALS

    // rows per thread
    const int dr = (nr + nth - 1) / nth;

    // row range for this thread
    const int ir0 = dr * ith;
    const int ir1 = min(ir0 + dr, nr);

    if (nb10 == sizeof(ggml_fp16_t)) {
        for (int ir = ir0; ir < ir1; ++ir) {
            // src0, src1 and dst are same shape => same indices
            const int i3 = ir / (ne2 * ne1);
            const int i2 = (ir - i3 * ne2 * ne1) / ne1;
            const int i1 = (ir - i3 * ne2 * ne1 - i2 * ne1);

            ggml_fp16_t* dst_ptr = (ggml_fp16_t*)((char*)dst->data + i3 * nb3 + i2 * nb2 + i1 * nb1);
            ggml_fp16_t* src0_ptr = (ggml_fp16_t*)((char*)src0->data + i3 * nb03 + i2 * nb02 + i1 * nb01);
            ggml_fp16_t* src1_ptr = (ggml_fp16_t*)((char*)src1->data + i3 * nb13 + i2 * nb12 + i1 * nb11);

            for (int i = 0; i < ne0; i++) {
                dst_ptr[i] = GGML_FP32_TO_FP16(GGML_FP16_TO_FP32(src0_ptr[i]) + GGML_FP16_TO_FP32(src1_ptr[i]));
            }
        }
    }
    //else {
    //    // src1 is not contiguous
    //    GGML_ABORT("fatal error");
    //}
}

static void ggml_compute_forward_add_bf16_bf16(
    const struct ggml_tensor* src0,
    const struct ggml_tensor* src1,
    struct ggml_tensor* dst) {

    const int ith = 0;
    const int nth = 1;

    const int nr = ggml_nrows(src0);

    GGML_TENSOR_BINARY_OP_LOCALS

    // rows per thread
    const int dr = (nr + nth - 1) / nth;

    // row range for this thread
    const int ir0 = dr * ith;
    const int ir1 = min(ir0 + dr, nr);

    if (nb10 == sizeof(ggml_bf16_t)) {
        for (int ir = ir0; ir < ir1; ++ir) {
            // src0, src1 and dst are same shape => same indices
            const int i3 = ir / (ne2 * ne1);
            const int i2 = (ir - i3 * ne2 * ne1) / ne1;
            const int i1 = (ir - i3 * ne2 * ne1 - i2 * ne1);

            ggml_bf16_t* dst_ptr = (ggml_bf16_t*)((char*)dst->data + i3 * nb3 + i2 * nb2 + i1 * nb1);
            ggml_bf16_t* src0_ptr = (ggml_bf16_t*)((char*)src0->data + i3 * nb03 + i2 * nb02 + i1 * nb01);
            ggml_bf16_t* src1_ptr = (ggml_bf16_t*)((char*)src1->data + i3 * nb13 + i2 * nb12 + i1 * nb11);

            for (int i = 0; i < ne0; i++) {
                dst_ptr[i] = GGML_FP32_TO_BF16(GGML_BF16_TO_FP32(src0_ptr[i]) + GGML_BF16_TO_FP32(src1_ptr[i]));
            }
        }
    }
    //else {
    //    // src1 is not contiguous
    //    GGML_ABORT("fatal error");
    //}
}

//static void ggml_compute_forward_add_q_f32(
//    const struct ggml_tensor* src0,
//    const struct ggml_tensor* src1,
//    struct ggml_tensor* dst) {
//
//    const int nr = ggml_nrows(src0);
//
//    GGML_TENSOR_BINARY_OP_LOCALS
//
//    const int ith = 0;
//    const int nth = 1;
//
//    const enum ggml_type type = src0->type;
//    const enum ggml_type dtype = dst->type;
//    ggml_to_float_t const dequantize_row_q = type_traits[type].to_float;
//    ggml_from_float_t const quantize_row_q = type_traits[dtype].from_float;
//
//    // rows per thread
//    const int dr = (nr + nth - 1) / nth;
//
//    // row range for this thread
//    const int ir0 = dr * ith;
//    const int ir1 = MIN(ir0 + dr, nr);
//
//    float* wdata = (float*)params->wdata + (ne00 + CACHE_LINE_SIZE_F32) * ith;
//
//    for (int ir = ir0; ir < ir1; ++ir) {
//        // src0 indices
//        const int i03 = ir / (ne02 * ne01);
//        const int i02 = (ir - i03 * ne02 * ne01) / ne01;
//        const int i01 = (ir - i03 * ne02 * ne01 - i02 * ne01);
//
//        // src1 and dst are same shape as src0 => same indices
//        const int i13 = i03;
//        const int i12 = i02;
//        const int i11 = i01;
//
//        const int i3 = i03;
//        const int i2 = i02;
//        const int i1 = i01;
//
//        void* src0_row = (void*)((char*)src0->data + (i01 * nb01 + i02 * nb02 + i03 * nb03));
//        float* src1_row = (float*)((char*)src1->data + (i11 * nb11 + i12 * nb12 + i13 * nb13));
//        void* dst_row = (void*)((char*)dst->data + (i1 * nb1 + i2 * nb2 + i3 * nb3));
//
//        assert(ne00 % 32 == 0);
//
//        // unquantize row from src0 to temp buffer
//        dequantize_row_q(src0_row, wdata, ne00);
//        // add src1
//        ggml_vec_acc_f32(ne00, wdata, src1_row);
//        // quantize row to dst
//        if (quantize_row_q != NULL) {
//            quantize_row_q(wdata, dst_row, ne00);
//        }
//        else {
//            memcpy(dst_row, wdata, ne0 * nb0);
//        }
//    }
//}

static void ggml_compute_forward_add(
    const struct ggml_tensor* src0,
    const struct ggml_tensor* src1,
    struct ggml_tensor* dst) {

    switch (src0->type) {
    case GGML_TYPE_F32:
    {
        if (src1->type == GGML_TYPE_F32) {
            ggml_compute_forward_add_f32(src0, src1, dst);
        }
        /*else {
            GGML_ABORT("fatal error");
        }*/
    } break;
    case GGML_TYPE_F16:
    {
        if (src1->type == GGML_TYPE_F16) {
            ggml_compute_forward_add_f16_f16(src0, src1, dst);
        }
        else if (src1->type == GGML_TYPE_F32) {
            ggml_compute_forward_add_f16_f32(src0, src1, dst);
        }
        /*else {
            GGML_ABORT("fatal error");
        }*/
    } break;
    case GGML_TYPE_BF16:
    {
        if (src1->type == GGML_TYPE_BF16) {
            ggml_compute_forward_add_bf16_bf16(src0, src1, dst);
        }
        else if (src1->type == GGML_TYPE_F32) {
            ggml_compute_forward_add_bf16_f32(src0, src1, dst);
        }
        /*else {
            GGML_ABORT("fatal error");
        }*/
    } break;
    case GGML_TYPE_Q4_0:
    case GGML_TYPE_Q4_1:
    case GGML_TYPE_Q5_0:
    case GGML_TYPE_Q5_1:
    case GGML_TYPE_Q8_0:
    case GGML_TYPE_Q2_K:
    case GGML_TYPE_Q3_K:
    case GGML_TYPE_Q4_K:
    case GGML_TYPE_Q5_K:
    case GGML_TYPE_Q6_K:
    case GGML_TYPE_IQ2_XXS:
    case GGML_TYPE_IQ2_XS:
    case GGML_TYPE_IQ3_XXS:
    case GGML_TYPE_IQ1_S:
    case GGML_TYPE_IQ1_M:
    case GGML_TYPE_IQ4_NL:
    case GGML_TYPE_IQ4_XS:
    case GGML_TYPE_IQ3_S:
    case GGML_TYPE_IQ2_S:
    case GGML_TYPE_Q4_0_4_4:
    case GGML_TYPE_Q4_0_4_8:
    case GGML_TYPE_Q4_0_8_8:
    {
        // ggml_compute_forward_add_q_f32(src0, src1, dst);
    } break;
    default:
    {
        // GGML_ABORT("fatal error");
    }
    }
}

static struct ggml_tensor* ggml_add_impl(
    struct ggml_tensor* a,
    struct ggml_tensor* b,
    bool inplace) {

    struct ggml_tensor* result = inplace ? ggml_view_tensor(a) : ggml_dup_tensor(a);

    ggml_compute_forward_add(a, b, result);

    return result;
}

struct ggml_tensor* ggml_add(
    struct ggml_tensor* a,
    struct ggml_tensor* b) {
    return ggml_add_impl(a, b, false);
}

struct ggml_tensor* ggml_add_inplace(
    struct ggml_tensor* a,
    struct ggml_tensor* b) {
    return ggml_add_impl(a, b, true);
}

/////////////////////////////////////////////////////////////////////
// ggml_compute_forward_add_rel_pos

static void ggml_compute_forward_add_rel_pos_f32(
    const struct ggml_tensor* src0,
    const struct ggml_tensor* src1,
    const struct ggml_tensor* src2,
    struct ggml_tensor* dst, const bool inplace) {

    if (!inplace) {
       memcpy((char*)dst->data, (char*)src0->data, ggml_nbytes(dst));
    }
    // ref: https://github.com/facebookresearch/segment-anything/blob/main/segment_anything/modeling/image_encoder.py#L357-L359

    float* src1_data = (float*)src1->data;
    float* src2_data = (float*)src2->data;
    float* dst_data = (float*)dst->data;

    const int64_t ne10 = src1->ne[0];
    const int64_t ne11 = src1->ne[1];
    const int64_t ne12 = src1->ne[2];
    const int64_t ne13 = src1->ne[3];

    const int ith = 0;
    const int nth = 1;

    // total patches in dst
    const int np = ne13;

    // patches per thread
    const int dp = (np + nth - 1) / nth;

    // patch range for this thread
    const int ip0 = dp * ith;
    const int ip1 = min(ip0 + dp, np);

    for (int64_t i13 = ip0; i13 < ip1; ++i13) {
        for (int64_t i12 = 0; i12 < ne12; ++i12) {
            for (int64_t i11 = 0; i11 < ne11; ++i11) {
                const int64_t jp1 = i13 * ne12 * ne11 * ne10 + i12 * ne11 * ne10 + i11 * ne10;
                for (int64_t i10 = 0; i10 < ne10; ++i10) {
                    const int64_t jp0 = jp1 + i10;
                    const float src1_e = src1_data[jp0];
                    const float src2_e = src2_data[jp0];

                    const int64_t jdh = jp0 * ne10;
                    const int64_t jdw = jdh - (ne10 - 1) * i10;

                    for (int64_t j = 0; j < ne10; ++j) {
                        dst_data[jdh + j] += src2_e;
                        dst_data[jdw + j * ne10] += src1_e;
                    }
                }
            }
        }
    }
}

static void ggml_compute_forward_add_rel_pos(
    const struct ggml_tensor* src0,
    struct ggml_tensor* pw,
    struct ggml_tensor* ph,
    struct ggml_tensor* dst, const bool inplace) {

    switch (src0->type) {
    case GGML_TYPE_F32:
    {
        ggml_compute_forward_add_rel_pos_f32(src0, pw, ph, dst, inplace);
    } break;
    default:
    {
        // GGML_ABORT("fatal error");
    }
    }
}
// ggml_add_rel_pos

static struct ggml_tensor* ggml_add_rel_pos_impl(
    struct ggml_tensor* a,
    struct ggml_tensor* pw,
    struct ggml_tensor* ph,
    bool                  inplace) {

    struct ggml_tensor* result = inplace ? ggml_view_tensor(a) : ggml_dup_tensor(a);
    
    ggml_compute_forward_add_rel_pos(a, pw, ph, result, inplace);

    return result;
}

struct ggml_tensor* ggml_add_rel_pos(
    struct ggml_tensor* a,
    struct ggml_tensor* pw,
    struct ggml_tensor* ph) {
    return ggml_add_rel_pos_impl(a, pw, ph, false);
}

struct ggml_tensor* ggml_add_rel_pos_inplace(
    struct ggml_tensor* a,
    struct ggml_tensor* pw,
    struct ggml_tensor* ph) {
    return ggml_add_rel_pos_impl(a, pw, ph, true);
}
