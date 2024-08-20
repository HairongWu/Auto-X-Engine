#include "../include/ggml.h"

static bool ggml_is_contiguous_n(const struct ggml_tensor* tensor, int n) {
    size_t next_nb = ggml_type_size(tensor->type);
    if (tensor->ne[0] != ggml_blck_size(tensor->type) && tensor->nb[0] != next_nb) {
        return false;
    }
    next_nb *= tensor->ne[0] / ggml_blck_size(tensor->type);
    for (int i = 1; i < GGML_MAX_DIMS; i++) {
        if (tensor->ne[i] != 1) {
            if (i > n) {
                if (tensor->nb[i] != next_nb) {
                    return false;
                }
                next_nb *= tensor->ne[i];
            }
            else {
                // this dimension does not need to be contiguous
                next_nb = tensor->ne[i] * tensor->nb[i];
            }
        }
    }
    return true;
}


bool ggml_is_contiguous_0(const struct ggml_tensor* tensor) {
    return ggml_is_contiguous_n(tensor, 0);
}

bool ggml_is_contiguous(const struct ggml_tensor* tensor) {
    return ggml_is_contiguous_0(tensor);
}


// ggml_compute_forward_dup

static void ggml_compute_forward_dup_same_cont(
    const struct ggml_tensor* src0,
    struct ggml_tensor* dst) {

    const size_t nb00 = src0->nb[0];
    const size_t nb0 = dst->nb[0];

    const int ith = 0; // thread index
    const int nth = 1; // number of threads

    // parallelize by elements
    const int ne = ggml_nelements(dst);
    const int dr = (ne + nth - 1) / nth;
    const int ie0 = dr * ith;
    const int ie1 = min(ie0 + dr, ne);

    if (ie0 < ie1) {
        memcpy(
            ((char*)dst->data + ie0 * nb0),
            ((char*)src0->data + ie0 * nb00),
            (ie1 - ie0) * ggml_type_size(src0->type));
    }
}

static void ggml_compute_forward_dup_f16(
    const struct ggml_tensor* src0,
    struct ggml_tensor* dst) {

    GGML_TENSOR_UNARY_OP_LOCALS

    const int ith = 0; // thread index
    const int nth = 1; // number of threads

    if (ggml_is_contiguous(src0) && ggml_is_contiguous(dst) && src0->type == dst->type) {
        ggml_compute_forward_dup_same_cont(src0, dst);
        return;
    }

    // parallelize by rows
    const int nr = ne01;
    // number of rows per thread
    const int dr = (nr + nth - 1) / nth;
    // row range for this thread
    const int ir0 = dr * ith;
    const int ir1 = min(ir0 + dr, nr);

    if (src0->type == dst->type &&
        ne00 == ne0 &&
        nb00 == ggml_type_size(src0->type) && nb0 == ggml_type_size(dst->type)) {
        // copy by rows
        const size_t rs = ne00 * nb00;
        for (int64_t i03 = 0; i03 < ne03; i03++) {
            for (int64_t i02 = 0; i02 < ne02; i02++) {
                for (int64_t i01 = ir0; i01 < ir1; i01++) {
                    memcpy(
                        ((char*)dst->data + i01 * nb1 + i02 * nb2 + i03 * nb3),
                        ((char*)src0->data + i01 * nb01 + i02 * nb02 + i03 * nb03),
                        rs);
                }
            }
        }
        return;
    }

    // TODO: add more special-case implementations for tensor shapes/strides that can benefit from memcpy

    if (ggml_is_contiguous(dst)) {
        if (nb00 == sizeof(ggml_fp16_t)) {
            if (dst->type == GGML_TYPE_F16) {
                size_t id = 0;
                const size_t rs = ne00 * nb00;
                char* dst_ptr = (char*)dst->data;

                for (int i03 = 0; i03 < ne03; i03++) {
                    for (int i02 = 0; i02 < ne02; i02++) {
                        id += rs * ir0;
                        for (int i01 = ir0; i01 < ir1; i01++) {
                            const char* src0_ptr = (char*)src0->data + i01 * nb01 + i02 * nb02 + i03 * nb03;
                            memcpy(dst_ptr + id, src0_ptr, rs);
                            id += rs;
                        }
                        id += rs * (ne01 - ir1);
                    }
                }
            }
            else if (dst->type == GGML_TYPE_F32) {
                size_t id = 0;
                float* dst_ptr = (float*)dst->data;

                for (int i03 = 0; i03 < ne03; i03++) {
                    for (int i02 = 0; i02 < ne02; i02++) {
                        id += ne00 * ir0;
                        for (int i01 = ir0; i01 < ir1; i01++) {
                            const ggml_fp16_t* src0_ptr = (ggml_fp16_t*)((char*)src0->data + i01 * nb01 + i02 * nb02 + i03 * nb03);
                            for (int i00 = 0; i00 < ne00; i00++) {
                                dst_ptr[id] = GGML_FP16_TO_FP32(src0_ptr[i00]);
                                id++;
                            }
                        }
                        id += ne00 * (ne01 - ir1);
                    }
                }
            }
            /*else if (type_traits[dst->type].from_float) {
                ggml_from_float_t const quantize_row_q = type_traits[dst->type].from_float;
                float* src0_f32 = (float*)params->wdata + (ne00 + CACHE_LINE_SIZE_F32) * ith;

                size_t id = 0;
                size_t rs = nb0 * (ne00 / ggml_blck_size(dst->type));
                char* dst_ptr = (char*)dst->data;

                for (int i03 = 0; i03 < ne03; i03++) {
                    for (int i02 = 0; i02 < ne02; i02++) {
                        id += rs * ir0;
                        for (int i01 = ir0; i01 < ir1; i01++) {
                            const ggml_fp16_t* src0_ptr = (ggml_fp16_t*)((char*)src0->data + i01 * nb01 + i02 * nb02 + i03 * nb03);

                            for (int i00 = 0; i00 < ne00; i00++) {
                                src0_f32[i00] = GGML_FP16_TO_FP32(src0_ptr[i00]);
                            }

                            quantize_row_q(src0_f32, dst_ptr + id, ne00);
                            id += rs;
                        }
                        id += rs * (ne01 - ir1);
                    }
                }
            }*/
            else {
                // GGML_ABORT("fatal error"); // TODO: implement
            }
        }
        else {
            //printf("%s: this is not optimal - fix me\n", __func__);

            if (dst->type == GGML_TYPE_F32) {
                size_t id = 0;
                float* dst_ptr = (float*)dst->data;

                for (int i03 = 0; i03 < ne03; i03++) {
                    for (int i02 = 0; i02 < ne02; i02++) {
                        id += ne00 * ir0;
                        for (int i01 = ir0; i01 < ir1; i01++) {
                            for (int i00 = 0; i00 < ne00; i00++) {
                                const ggml_fp16_t* src0_ptr = (ggml_fp16_t*)((char*)src0->data + i00 * nb00 + i01 * nb01 + i02 * nb02 + i03 * nb03);

                                dst_ptr[id] = GGML_FP16_TO_FP32(*src0_ptr);
                                id++;
                            }
                        }
                        id += ne00 * (ne01 - ir1);
                    }
                }
            }
            else if (dst->type == GGML_TYPE_F16) {
                size_t id = 0;
                ggml_fp16_t* dst_ptr = (ggml_fp16_t*)dst->data;

                for (int i03 = 0; i03 < ne03; i03++) {
                    for (int i02 = 0; i02 < ne02; i02++) {
                        id += ne00 * ir0;
                        for (int i01 = ir0; i01 < ir1; i01++) {
                            for (int i00 = 0; i00 < ne00; i00++) {
                                const ggml_fp16_t* src0_ptr = (ggml_fp16_t*)((char*)src0->data + i00 * nb00 + i01 * nb01 + i02 * nb02 + i03 * nb03);

                                dst_ptr[id] = *src0_ptr;
                                id++;
                            }
                        }
                        id += ne00 * (ne01 - ir1);
                    }
                }
            }
            else {
                // GGML_ABORT("fatal error"); // TODO: implement
            }
        }
        return;
    }

    // dst counters
    int64_t i10 = 0;
    int64_t i11 = 0;
    int64_t i12 = 0;
    int64_t i13 = 0;

    if (dst->type == GGML_TYPE_F16) {
        for (int64_t i03 = 0; i03 < ne03; i03++) {
            for (int64_t i02 = 0; i02 < ne02; i02++) {
                i10 += ne00 * ir0;
                while (i10 >= ne0) {
                    i10 -= ne0;
                    if (++i11 == ne1) {
                        i11 = 0;
                        if (++i12 == ne2) {
                            i12 = 0;
                            if (++i13 == ne3) {
                                i13 = 0;
                            }
                        }
                    }
                }
                for (int64_t i01 = ir0; i01 < ir1; i01++) {
                    for (int64_t i00 = 0; i00 < ne00; i00++) {
                        const char* src0_ptr = ((char*)src0->data + i00 * nb00 + i01 * nb01 + i02 * nb02 + i03 * nb03);
                        char* dst_ptr = ((char*)dst->data + i10 * nb0 + i11 * nb1 + i12 * nb2 + i13 * nb3);

                        memcpy(dst_ptr, src0_ptr, sizeof(ggml_fp16_t));

                        if (++i10 == ne00) {
                            i10 = 0;
                            if (++i11 == ne01) {
                                i11 = 0;
                                if (++i12 == ne02) {
                                    i12 = 0;
                                    if (++i13 == ne03) {
                                        i13 = 0;
                                    }
                                }
                            }
                        }
                    }
                }
                i10 += ne00 * (ne01 - ir1);
                while (i10 >= ne0) {
                    i10 -= ne0;
                    if (++i11 == ne1) {
                        i11 = 0;
                        if (++i12 == ne2) {
                            i12 = 0;
                            if (++i13 == ne3) {
                                i13 = 0;
                            }
                        }
                    }
                }
            }
        }
    }
    else if (dst->type == GGML_TYPE_F32) {
        for (int64_t i03 = 0; i03 < ne03; i03++) {
            for (int64_t i02 = 0; i02 < ne02; i02++) {
                i10 += ne00 * ir0;
                while (i10 >= ne0) {
                    i10 -= ne0;
                    if (++i11 == ne1) {
                        i11 = 0;
                        if (++i12 == ne2) {
                            i12 = 0;
                            if (++i13 == ne3) {
                                i13 = 0;
                            }
                        }
                    }
                }
                for (int64_t i01 = ir0; i01 < ir1; i01++) {
                    for (int64_t i00 = 0; i00 < ne00; i00++) {
                        const char* src0_ptr = ((char*)src0->data + i00 * nb00 + i01 * nb01 + i02 * nb02 + i03 * nb03);
                        char* dst_ptr = ((char*)dst->data + i10 * nb0 + i11 * nb1 + i12 * nb2 + i13 * nb3);

                        *(float*)dst_ptr = GGML_FP16_TO_FP32(*(const ggml_fp16_t*)src0_ptr);

                        if (++i10 == ne0) {
                            i10 = 0;
                            if (++i11 == ne1) {
                                i11 = 0;
                                if (++i12 == ne2) {
                                    i12 = 0;
                                    if (++i13 == ne3) {
                                        i13 = 0;
                                    }
                                }
                            }
                        }
                    }
                }
                i10 += ne00 * (ne01 - ir1);
                while (i10 >= ne0) {
                    i10 -= ne0;
                    if (++i11 == ne1) {
                        i11 = 0;
                        if (++i12 == ne2) {
                            i12 = 0;
                            if (++i13 == ne3) {
                                i13 = 0;
                            }
                        }
                    }
                }
            }
        }
    }
    else {
        // GGML_ABORT("fatal error"); // TODO: implement
    }
}

static void ggml_compute_forward_dup_bf16(
    const struct ggml_tensor* src0,
    struct ggml_tensor* dst) {

    GGML_TENSOR_UNARY_OP_LOCALS

    const int ith = 0; // thread index
    const int nth = 1; // number of threads

    if (ggml_is_contiguous(src0) && ggml_is_contiguous(dst) && src0->type == dst->type) {
        ggml_compute_forward_dup_same_cont(src0, dst);
        return;
    }

    // parallelize by rows
    const int nr = ne01;
    // number of rows per thread
    const int dr = (nr + nth - 1) / nth;
    // row range for this thread
    const int ir0 = dr * ith;
    const int ir1 = min(ir0 + dr, nr);

    if (src0->type == dst->type &&
        ne00 == ne0 &&
        nb00 == ggml_type_size(src0->type) && nb0 == ggml_type_size(dst->type)) {
        // copy by rows
        const size_t rs = ne00 * nb00;
        for (int64_t i03 = 0; i03 < ne03; i03++) {
            for (int64_t i02 = 0; i02 < ne02; i02++) {
                for (int64_t i01 = ir0; i01 < ir1; i01++) {
                    memcpy(
                        ((char*)dst->data + i01 * nb1 + i02 * nb2 + i03 * nb3),
                        ((char*)src0->data + i01 * nb01 + i02 * nb02 + i03 * nb03),
                        rs);
                }
            }
        }
        return;
    }

    // TODO: add more special-case implementations for tensor shapes/strides that can benefit from memcpy

    if (ggml_is_contiguous(dst)) {
        if (nb00 == sizeof(ggml_bf16_t)) {
            if (dst->type == GGML_TYPE_BF16) {
                size_t id = 0;
                const size_t rs = ne00 * nb00;
                char* dst_ptr = (char*)dst->data;

                for (int i03 = 0; i03 < ne03; i03++) {
                    for (int i02 = 0; i02 < ne02; i02++) {
                        id += rs * ir0;
                        for (int i01 = ir0; i01 < ir1; i01++) {
                            const char* src0_ptr = (char*)src0->data + i01 * nb01 + i02 * nb02 + i03 * nb03;
                            memcpy(dst_ptr + id, src0_ptr, rs);
                            id += rs;
                        }
                        id += rs * (ne01 - ir1);
                    }
                }
            }
            else if (dst->type == GGML_TYPE_F16) {
                size_t id = 0;
                ggml_fp16_t* dst_ptr = (ggml_fp16_t*)dst->data;

                for (int i03 = 0; i03 < ne03; i03++) {
                    for (int i02 = 0; i02 < ne02; i02++) {
                        id += ne00 * ir0;
                        for (int i01 = ir0; i01 < ir1; i01++) {
                            const ggml_bf16_t* src0_ptr = (ggml_bf16_t*)((char*)src0->data + i01 * nb01 + i02 * nb02 + i03 * nb03);
                            for (int i00 = 0; i00 < ne00; i00++) {
                                dst_ptr[id] = GGML_FP32_TO_FP16(GGML_BF16_TO_FP32(src0_ptr[i00]));
                                id++;
                            }
                        }
                        id += ne00 * (ne01 - ir1);
                    }
                }
            }
            else if (dst->type == GGML_TYPE_F32) {
                size_t id = 0;
                float* dst_ptr = (float*)dst->data;

                for (int i03 = 0; i03 < ne03; i03++) {
                    for (int i02 = 0; i02 < ne02; i02++) {
                        id += ne00 * ir0;
                        for (int i01 = ir0; i01 < ir1; i01++) {
                            const ggml_bf16_t* src0_ptr = (ggml_bf16_t*)((char*)src0->data + i01 * nb01 + i02 * nb02 + i03 * nb03);
                            for (int i00 = 0; i00 < ne00; i00++) {
                                dst_ptr[id] = GGML_BF16_TO_FP32(src0_ptr[i00]);
                                id++;
                            }
                        }
                        id += ne00 * (ne01 - ir1);
                    }
                }
            }
            /*else if (type_traits[dst->type].from_float) {
                ggml_from_float_t const quantize_row_q = type_traits[dst->type].from_float;
                float* src0_f32 = (float*)params->wdata;

                size_t id = 0;
                size_t rs = nb0 * (ne00 / ggml_blck_size(dst->type));
                char* dst_ptr = (char*)dst->data;

                for (int i03 = 0; i03 < ne03; i03++) {
                    for (int i02 = 0; i02 < ne02; i02++) {
                        id += rs * ir0;
                        for (int i01 = ir0; i01 < ir1; i01++) {
                            const ggml_bf16_t* src0_ptr = (ggml_bf16_t*)((char*)src0->data + i01 * nb01 + i02 * nb02 + i03 * nb03);

                            for (int i00 = 0; i00 < ne00; i00++) {
                                src0_f32[i00] = GGML_BF16_TO_FP32(src0_ptr[i00]);
                            }

                            quantize_row_q(src0_f32, dst_ptr + id, ne00);
                            id += rs;
                        }
                        id += rs * (ne01 - ir1);
                    }
                }
            }*/
            else {
                // GGML_ABORT("fatal error"); // TODO: implement
            }
        }
        else {
            //printf("%s: this is not optimal - fix me\n", __func__);

            if (dst->type == GGML_TYPE_F32) {
                size_t id = 0;
                float* dst_ptr = (float*)dst->data;

                for (int i03 = 0; i03 < ne03; i03++) {
                    for (int i02 = 0; i02 < ne02; i02++) {
                        id += ne00 * ir0;
                        for (int i01 = ir0; i01 < ir1; i01++) {
                            for (int i00 = 0; i00 < ne00; i00++) {
                                const ggml_bf16_t* src0_ptr = (ggml_bf16_t*)((char*)src0->data + i00 * nb00 + i01 * nb01 + i02 * nb02 + i03 * nb03);

                                dst_ptr[id] = GGML_BF16_TO_FP32(*src0_ptr);
                                id++;
                            }
                        }
                        id += ne00 * (ne01 - ir1);
                    }
                }
            }
            else if (dst->type == GGML_TYPE_BF16) {
                size_t id = 0;
                ggml_bf16_t* dst_ptr = (ggml_bf16_t*)dst->data;

                for (int i03 = 0; i03 < ne03; i03++) {
                    for (int i02 = 0; i02 < ne02; i02++) {
                        id += ne00 * ir0;
                        for (int i01 = ir0; i01 < ir1; i01++) {
                            for (int i00 = 0; i00 < ne00; i00++) {
                                const ggml_bf16_t* src0_ptr = (ggml_bf16_t*)((char*)src0->data + i00 * nb00 + i01 * nb01 + i02 * nb02 + i03 * nb03);

                                dst_ptr[id] = *src0_ptr;
                                id++;
                            }
                        }
                        id += ne00 * (ne01 - ir1);
                    }
                }
            }
            else if (dst->type == GGML_TYPE_F16) {
                size_t id = 0;
                ggml_fp16_t* dst_ptr = (ggml_fp16_t*)dst->data;

                for (int i03 = 0; i03 < ne03; i03++) {
                    for (int i02 = 0; i02 < ne02; i02++) {
                        id += ne00 * ir0;
                        for (int i01 = ir0; i01 < ir1; i01++) {
                            for (int i00 = 0; i00 < ne00; i00++) {
                                const ggml_bf16_t* src0_ptr = (ggml_bf16_t*)((char*)src0->data + i00 * nb00 + i01 * nb01 + i02 * nb02 + i03 * nb03);

                                dst_ptr[id] = GGML_FP32_TO_FP16(GGML_BF16_TO_FP32(*src0_ptr));
                                id++;
                            }
                        }
                        id += ne00 * (ne01 - ir1);
                    }
                }
            }
            else {
                // GGML_ABORT("fatal error"); // TODO: implement
            }
        }
        return;
    }

    // dst counters
    int64_t i10 = 0;
    int64_t i11 = 0;
    int64_t i12 = 0;
    int64_t i13 = 0;

    if (dst->type == GGML_TYPE_BF16) {
        for (int64_t i03 = 0; i03 < ne03; i03++) {
            for (int64_t i02 = 0; i02 < ne02; i02++) {
                i10 += ne00 * ir0;
                while (i10 >= ne0) {
                    i10 -= ne0;
                    if (++i11 == ne1) {
                        i11 = 0;
                        if (++i12 == ne2) {
                            i12 = 0;
                            if (++i13 == ne3) {
                                i13 = 0;
                            }
                        }
                    }
                }
                for (int64_t i01 = ir0; i01 < ir1; i01++) {
                    for (int64_t i00 = 0; i00 < ne00; i00++) {
                        const char* src0_ptr = ((char*)src0->data + i00 * nb00 + i01 * nb01 + i02 * nb02 + i03 * nb03);
                        char* dst_ptr = ((char*)dst->data + i10 * nb0 + i11 * nb1 + i12 * nb2 + i13 * nb3);

                        memcpy(dst_ptr, src0_ptr, sizeof(ggml_bf16_t));

                        if (++i10 == ne00) {
                            i10 = 0;
                            if (++i11 == ne01) {
                                i11 = 0;
                                if (++i12 == ne02) {
                                    i12 = 0;
                                    if (++i13 == ne03) {
                                        i13 = 0;
                                    }
                                }
                            }
                        }
                    }
                }
                i10 += ne00 * (ne01 - ir1);
                while (i10 >= ne0) {
                    i10 -= ne0;
                    if (++i11 == ne1) {
                        i11 = 0;
                        if (++i12 == ne2) {
                            i12 = 0;
                            if (++i13 == ne3) {
                                i13 = 0;
                            }
                        }
                    }
                }
            }
        }
    }
    else if (dst->type == GGML_TYPE_F16) {
        for (int64_t i03 = 0; i03 < ne03; i03++) {
            for (int64_t i02 = 0; i02 < ne02; i02++) {
                i10 += ne00 * ir0;
                while (i10 >= ne0) {
                    i10 -= ne0;
                    if (++i11 == ne1) {
                        i11 = 0;
                        if (++i12 == ne2) {
                            i12 = 0;
                            if (++i13 == ne3) {
                                i13 = 0;
                            }
                        }
                    }
                }
                for (int64_t i01 = ir0; i01 < ir1; i01++) {
                    for (int64_t i00 = 0; i00 < ne00; i00++) {
                        const char* src0_ptr = ((char*)src0->data + i00 * nb00 + i01 * nb01 + i02 * nb02 + i03 * nb03);
                        char* dst_ptr = ((char*)dst->data + i10 * nb0 + i11 * nb1 + i12 * nb2 + i13 * nb3);

                        *(ggml_fp16_t*)dst_ptr = GGML_FP32_TO_FP16(GGML_BF16_TO_FP32(*(const ggml_bf16_t*)src0_ptr));

                        if (++i10 == ne0) {
                            i10 = 0;
                            if (++i11 == ne1) {
                                i11 = 0;
                                if (++i12 == ne2) {
                                    i12 = 0;
                                    if (++i13 == ne3) {
                                        i13 = 0;
                                    }
                                }
                            }
                        }
                    }
                }
                i10 += ne00 * (ne01 - ir1);
                while (i10 >= ne0) {
                    i10 -= ne0;
                    if (++i11 == ne1) {
                        i11 = 0;
                        if (++i12 == ne2) {
                            i12 = 0;
                            if (++i13 == ne3) {
                                i13 = 0;
                            }
                        }
                    }
                }
            }
        }
    }
    else if (dst->type == GGML_TYPE_F32) {
        for (int64_t i03 = 0; i03 < ne03; i03++) {
            for (int64_t i02 = 0; i02 < ne02; i02++) {
                i10 += ne00 * ir0;
                while (i10 >= ne0) {
                    i10 -= ne0;
                    if (++i11 == ne1) {
                        i11 = 0;
                        if (++i12 == ne2) {
                            i12 = 0;
                            if (++i13 == ne3) {
                                i13 = 0;
                            }
                        }
                    }
                }
                for (int64_t i01 = ir0; i01 < ir1; i01++) {
                    for (int64_t i00 = 0; i00 < ne00; i00++) {
                        const char* src0_ptr = ((char*)src0->data + i00 * nb00 + i01 * nb01 + i02 * nb02 + i03 * nb03);
                        char* dst_ptr = ((char*)dst->data + i10 * nb0 + i11 * nb1 + i12 * nb2 + i13 * nb3);

                        *(float*)dst_ptr = GGML_BF16_TO_FP32(*(const ggml_bf16_t*)src0_ptr);

                        if (++i10 == ne0) {
                            i10 = 0;
                            if (++i11 == ne1) {
                                i11 = 0;
                                if (++i12 == ne2) {
                                    i12 = 0;
                                    if (++i13 == ne3) {
                                        i13 = 0;
                                    }
                                }
                            }
                        }
                    }
                }
                i10 += ne00 * (ne01 - ir1);
                while (i10 >= ne0) {
                    i10 -= ne0;
                    if (++i11 == ne1) {
                        i11 = 0;
                        if (++i12 == ne2) {
                            i12 = 0;
                            if (++i13 == ne3) {
                                i13 = 0;
                            }
                        }
                    }
                }
            }
        }
    }
    else {
        // GGML_ABORT("fatal error"); // TODO: implement
    }
}

static void ggml_compute_forward_dup_f32(
    const struct ggml_tensor* src0,
    struct ggml_tensor* dst) {

    GGML_TENSOR_UNARY_OP_LOCALS

    const int ith = 0; // thread index
    const int nth = 1; // number of threads

    if (ggml_is_contiguous(src0) && ggml_is_contiguous(dst) && src0->type == dst->type) {
        ggml_compute_forward_dup_same_cont(src0, dst);
        return;
    }

    // parallelize by rows
    const int nr = ne01;
    // number of rows per thread
    const int dr = (nr + nth - 1) / nth;
    // row range for this thread
    const int ir0 = dr * ith;
    const int ir1 = min(ir0 + dr, nr);

    if (src0->type == dst->type &&
        ne00 == ne0 &&
        nb00 == ggml_type_size(src0->type) && nb0 == ggml_type_size(dst->type)) {
        // copy by rows
        const size_t rs = ne00 * nb00;
        for (int64_t i03 = 0; i03 < ne03; i03++) {
            for (int64_t i02 = 0; i02 < ne02; i02++) {
                for (int64_t i01 = ir0; i01 < ir1; i01++) {
                    memcpy(
                        ((char*)dst->data + i01 * nb1 + i02 * nb2 + i03 * nb3),
                        ((char*)src0->data + i01 * nb01 + i02 * nb02 + i03 * nb03),
                        rs);
                }
            }
        }
        return;
    }

    if (ggml_is_contiguous(dst)) {
        // TODO: simplify
        if (nb00 == sizeof(float)) {
            if (dst->type == GGML_TYPE_F32) {
                size_t id = 0;
                const size_t rs = ne00 * nb00;
                char* dst_ptr = (char*)dst->data;

                for (int i03 = 0; i03 < ne03; i03++) {
                    for (int i02 = 0; i02 < ne02; i02++) {
                        id += rs * ir0;
                        for (int i01 = ir0; i01 < ir1; i01++) {
                            const char* src0_ptr = (char*)src0->data + i01 * nb01 + i02 * nb02 + i03 * nb03;
                            memcpy(dst_ptr + id, src0_ptr, rs);
                            id += rs;
                        }
                        id += rs * (ne01 - ir1);
                    }
                }
            }
            //else if (type_traits[dst->type].from_float) {
            //    ggml_from_float_t const quantize_row_q = type_traits[dst->type].from_float;

            //    size_t id = 0;
            //    size_t rs = nb0 * (ne00 / ggml_blck_size(dst->type));
            //    char* dst_ptr = (char*)dst->data;

            //    for (int i03 = 0; i03 < ne03; i03++) {
            //        for (int i02 = 0; i02 < ne02; i02++) {
            //            id += rs * ir0;
            //            for (int i01 = ir0; i01 < ir1; i01++) {
            //                const float* src0_ptr = (float*)((char*)src0->data + i01 * nb01 + i02 * nb02 + i03 * nb03);
            //                quantize_row_q(src0_ptr, dst_ptr + id, ne00);
            //                id += rs;
            //            }
            //            id += rs * (ne01 - ir1);
            //        }
            //    }
            //}
            else {
                // GGML_ABORT("fatal error"); // TODO: implement
            }
        }
        else {
            //printf("%s: this is not optimal - fix me\n", __func__);

            if (dst->type == GGML_TYPE_F32) {
                size_t id = 0;
                float* dst_ptr = (float*)dst->data;

                for (int i03 = 0; i03 < ne03; i03++) {
                    for (int i02 = 0; i02 < ne02; i02++) {
                        id += ne00 * ir0;
                        for (int i01 = ir0; i01 < ir1; i01++) {
                            for (int i00 = 0; i00 < ne00; i00++) {
                                const float* src0_ptr = (float*)((char*)src0->data + i00 * nb00 + i01 * nb01 + i02 * nb02 + i03 * nb03);

                                dst_ptr[id] = *src0_ptr;
                                id++;
                            }
                        }
                        id += ne00 * (ne01 - ir1);
                    }
                }
            }
            else if (dst->type == GGML_TYPE_F16) {
                size_t id = 0;
                ggml_fp16_t* dst_ptr = (ggml_fp16_t*)dst->data;

                for (int i03 = 0; i03 < ne03; i03++) {
                    for (int i02 = 0; i02 < ne02; i02++) {
                        id += ne00 * ir0;
                        for (int i01 = ir0; i01 < ir1; i01++) {
                            for (int i00 = 0; i00 < ne00; i00++) {
                                const float* src0_ptr = (float*)((char*)src0->data + i00 * nb00 + i01 * nb01 + i02 * nb02 + i03 * nb03);

                                dst_ptr[id] = GGML_FP32_TO_FP16(*src0_ptr);
                                id++;
                            }
                        }
                        id += ne00 * (ne01 - ir1);
                    }
                }
            }
            else if (dst->type == GGML_TYPE_BF16) {
                size_t id = 0;
                ggml_bf16_t* dst_ptr = (ggml_bf16_t*)dst->data;

                for (int i03 = 0; i03 < ne03; i03++) {
                    for (int i02 = 0; i02 < ne02; i02++) {
                        id += ne00 * ir0;
                        for (int i01 = ir0; i01 < ir1; i01++) {
                            for (int i00 = 0; i00 < ne00; i00++) {
                                const float* src0_ptr = (float*)((char*)src0->data + i00 * nb00 + i01 * nb01 + i02 * nb02 + i03 * nb03);

                                dst_ptr[id] = GGML_FP32_TO_BF16(*src0_ptr);
                                id++;
                            }
                        }
                        id += ne00 * (ne01 - ir1);
                    }
                }
            }
            else {
                // GGML_ABORT("fatal error"); // TODO: implement
            }
        }

        return;
    }

    // dst counters

    int64_t i10 = 0;
    int64_t i11 = 0;
    int64_t i12 = 0;
    int64_t i13 = 0;

    if (dst->type == GGML_TYPE_F32) {
        for (int64_t i03 = 0; i03 < ne03; i03++) {
            for (int64_t i02 = 0; i02 < ne02; i02++) {
                i10 += ne00 * ir0;
                while (i10 >= ne0) {
                    i10 -= ne0;
                    if (++i11 == ne1) {
                        i11 = 0;
                        if (++i12 == ne2) {
                            i12 = 0;
                            if (++i13 == ne3) {
                                i13 = 0;
                            }
                        }
                    }
                }
                for (int64_t i01 = ir0; i01 < ir1; i01++) {
                    for (int64_t i00 = 0; i00 < ne00; i00++) {
                        const char* src0_ptr = ((char*)src0->data + i00 * nb00 + i01 * nb01 + i02 * nb02 + i03 * nb03);
                        char* dst_ptr = ((char*)dst->data + i10 * nb0 + i11 * nb1 + i12 * nb2 + i13 * nb3);

                        memcpy(dst_ptr, src0_ptr, sizeof(float));

                        if (++i10 == ne0) {
                            i10 = 0;
                            if (++i11 == ne1) {
                                i11 = 0;
                                if (++i12 == ne2) {
                                    i12 = 0;
                                    if (++i13 == ne3) {
                                        i13 = 0;
                                    }
                                }
                            }
                        }
                    }
                }
                i10 += ne00 * (ne01 - ir1);
                while (i10 >= ne0) {
                    i10 -= ne0;
                    if (++i11 == ne1) {
                        i11 = 0;
                        if (++i12 == ne2) {
                            i12 = 0;
                            if (++i13 == ne3) {
                                i13 = 0;
                            }
                        }
                    }
                }
            }
        }
    }
    else if (dst->type == GGML_TYPE_F16) {
        for (int64_t i03 = 0; i03 < ne03; i03++) {
            for (int64_t i02 = 0; i02 < ne02; i02++) {
                i10 += ne00 * ir0;
                while (i10 >= ne0) {
                    i10 -= ne0;
                    if (++i11 == ne1) {
                        i11 = 0;
                        if (++i12 == ne2) {
                            i12 = 0;
                            if (++i13 == ne3) {
                                i13 = 0;
                            }
                        }
                    }
                }
                for (int64_t i01 = ir0; i01 < ir1; i01++) {
                    for (int64_t i00 = 0; i00 < ne00; i00++) {
                        const char* src0_ptr = ((char*)src0->data + i00 * nb00 + i01 * nb01 + i02 * nb02 + i03 * nb03);
                        char* dst_ptr = ((char*)dst->data + i10 * nb0 + i11 * nb1 + i12 * nb2 + i13 * nb3);

                        *(ggml_fp16_t*)dst_ptr = GGML_FP32_TO_FP16(*(const float*)src0_ptr);

                        if (++i10 == ne0) {
                            i10 = 0;
                            if (++i11 == ne1) {
                                i11 = 0;
                                if (++i12 == ne2) {
                                    i12 = 0;
                                    if (++i13 == ne3) {
                                        i13 = 0;
                                    }
                                }
                            }
                        }
                    }
                }
                i10 += ne00 * (ne01 - ir1);
                while (i10 >= ne0) {
                    i10 -= ne0;
                    if (++i11 == ne1) {
                        i11 = 0;
                        if (++i12 == ne2) {
                            i12 = 0;
                            if (++i13 == ne3) {
                                i13 = 0;
                            }
                        }
                    }
                }
            }
        }
    }
    else if (dst->type == GGML_TYPE_BF16) {
        for (int64_t i03 = 0; i03 < ne03; i03++) {
            for (int64_t i02 = 0; i02 < ne02; i02++) {
                i10 += ne00 * ir0;
                while (i10 >= ne0) {
                    i10 -= ne0;
                    if (++i11 == ne1) {
                        i11 = 0;
                        if (++i12 == ne2) {
                            i12 = 0;
                            if (++i13 == ne3) {
                                i13 = 0;
                            }
                        }
                    }
                }
                for (int64_t i01 = ir0; i01 < ir1; i01++) {
                    for (int64_t i00 = 0; i00 < ne00; i00++) {
                        const char* src0_ptr = ((char*)src0->data + i00 * nb00 + i01 * nb01 + i02 * nb02 + i03 * nb03);
                        char* dst_ptr = ((char*)dst->data + i10 * nb0 + i11 * nb1 + i12 * nb2 + i13 * nb3);

                        *(ggml_bf16_t*)dst_ptr = GGML_FP32_TO_BF16(*(const float*)src0_ptr);

                        if (++i10 == ne0) {
                            i10 = 0;
                            if (++i11 == ne1) {
                                i11 = 0;
                                if (++i12 == ne2) {
                                    i12 = 0;
                                    if (++i13 == ne3) {
                                        i13 = 0;
                                    }
                                }
                            }
                        }
                    }
                }
                i10 += ne00 * (ne01 - ir1);
                while (i10 >= ne0) {
                    i10 -= ne0;
                    if (++i11 == ne1) {
                        i11 = 0;
                        if (++i12 == ne2) {
                            i12 = 0;
                            if (++i13 == ne3) {
                                i13 = 0;
                            }
                        }
                    }
                }
            }
        }
    }
    else {
        // GGML_ABORT("fatal error"); // TODO: implement
    }
}

// A simplified version of ggml_compute_forward_dup that doesn't do float upcasting, and just plain old memcpy.
static void ggml_compute_forward_dup_bytes(
    const struct ggml_tensor* src0,
    struct ggml_tensor* dst) {

    if (ggml_is_contiguous(src0) && ggml_is_contiguous(dst)) {
        ggml_compute_forward_dup_same_cont(src0, dst);
        return;
    }

    GGML_TENSOR_UNARY_OP_LOCALS;

    const size_t type_size = ggml_type_size(src0->type);
    const int ith = 0; // thread index
    const int nth = 1; // number of threads


    // parallelize by rows
    const int nr = ne01;
    // number of rows per thread
    const int dr = (nr + nth - 1) / nth;
    // row range for this thread
    const int ir0 = dr * ith;
    const int ir1 = min(ir0 + dr, nr);

    if (src0->type == dst->type &&
        ne00 == ne0 &&
        nb00 == type_size && nb0 == type_size) {
        // copy by rows
        const size_t rs = ne00 * type_size;
        for (int64_t i03 = 0; i03 < ne03; i03++) {
            for (int64_t i02 = 0; i02 < ne02; i02++) {
                for (int64_t i01 = ir0; i01 < ir1; i01++) {
                    memcpy(
                        ((char*)dst->data + i01 * nb1 + i02 * nb2 + i03 * nb3),
                        ((char*)src0->data + i01 * nb01 + i02 * nb02 + i03 * nb03),
                        rs);
                }
            }
        }
        return;
    }

    if (ggml_is_contiguous(dst)) {
        size_t id = 0;
        char* dst_ptr = (char*)dst->data;
        const size_t rs = ne00 * type_size;

        if (nb00 == type_size) {
            // src0 is contigous on first dimension, copy by rows
            for (int64_t i03 = 0; i03 < ne03; i03++) {
                for (int64_t i02 = 0; i02 < ne02; i02++) {
                    id += rs * ir0;
                    for (int64_t i01 = ir0; i01 < ir1; i01++) {
                        const char* src0_ptr = (char*)src0->data + i01 * nb01 + i02 * nb02 + i03 * nb03;
                        memcpy(dst_ptr + id, src0_ptr, rs);
                        id += rs;
                    }
                    id += rs * (ne01 - ir1);
                }
            }
        }
        else {
            //printf("%s: this is not optimal - fix me\n", __func__);

            for (int64_t i03 = 0; i03 < ne03; i03++) {
                for (int64_t i02 = 0; i02 < ne02; i02++) {
                    id += rs * ir0;
                    for (int64_t i01 = ir0; i01 < ir1; i01++) {
                        for (int64_t i00 = 0; i00 < ne00; i00++) {
                            const char* src0_ptr = (char*)src0->data + i00 * nb00 + i01 * nb01 + i02 * nb02 + i03 * nb03;
                            memcpy(dst_ptr + id, src0_ptr, type_size);

                            id += type_size;
                        }
                    }
                    id += rs * (ne01 - ir1);
                }
            }
        }

        return;
    }

    // dst counters

    int64_t i10 = 0;
    int64_t i11 = 0;
    int64_t i12 = 0;
    int64_t i13 = 0;

    for (int64_t i03 = 0; i03 < ne03; i03++) {
        for (int64_t i02 = 0; i02 < ne02; i02++) {
            i10 += ne00 * ir0;
            while (i10 >= ne0) {
                i10 -= ne0;
                if (++i11 == ne1) {
                    i11 = 0;
                    if (++i12 == ne2) {
                        i12 = 0;
                        if (++i13 == ne3) {
                            i13 = 0;
                        }
                    }
                }
            }
            for (int64_t i01 = ir0; i01 < ir1; i01++) {
                for (int64_t i00 = 0; i00 < ne00; i00++) {
                    const char* src0_ptr = ((char*)src0->data + i00 * nb00 + i01 * nb01 + i02 * nb02 + i03 * nb03);
                    char* dst_ptr = ((char*)dst->data + i10 * nb0 + i11 * nb1 + i12 * nb2 + i13 * nb3);

                    memcpy(dst_ptr, src0_ptr, type_size);

                    if (++i10 == ne0) {
                        i10 = 0;
                        if (++i11 == ne1) {
                            i11 = 0;
                            if (++i12 == ne2) {
                                i12 = 0;
                                if (++i13 == ne3) {
                                    i13 = 0;
                                }
                            }
                        }
                    }
                }
            }
            i10 += ne00 * (ne01 - ir1);
            while (i10 >= ne0) {
                i10 -= ne0;
                if (++i11 == ne1) {
                    i11 = 0;
                    if (++i12 == ne2) {
                        i12 = 0;
                        if (++i13 == ne3) {
                            i13 = 0;
                        }
                    }
                }
            }
        }
    }
}

void ggml_compute_forward_dup(
    const struct ggml_tensor* src0,
    struct ggml_tensor* dst) {

    if (src0->type == dst->type) {
        ggml_compute_forward_dup_bytes(src0, dst);
        return;
    }

    switch (src0->type) {
    case GGML_TYPE_F16:
    {
        ggml_compute_forward_dup_f16(src0, dst);
    } break;
    case GGML_TYPE_BF16:
    {
        ggml_compute_forward_dup_bf16(src0, dst);
    } break;
    case GGML_TYPE_F32:
    {
        ggml_compute_forward_dup_f32(src0, dst);
    } break;
    default:
    {
        // GGML_ABORT("fatal error");
    }
    }
}

static struct ggml_tensor* ggml_cont_impl(
    struct ggml_tensor* a) {

    struct ggml_tensor* result = ggml_dup_tensor(a);

    ggml_compute_forward_dup(a, result);

    return result;
}

struct ggml_tensor* ggml_cont(
    struct ggml_tensor* a) {
    return ggml_cont_impl(a);
}
