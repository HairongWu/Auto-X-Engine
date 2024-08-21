#include "../include/ggml.h"

static void ggml_vec_dot_f16(int n, float* s, size_t bs, ggml_fp16_t* x, size_t bx, ggml_fp16_t* y, size_t by, int nrc) {
    float sumf = 0.0;

#if defined(GGML_SIMD)
    const int np = (n & ~(GGML_F16_STEP - 1));

    GGML_F16_VEC sum[GGML_F16_ARR] = { GGML_F16_VEC_ZERO };

    GGML_F16_VEC ax[GGML_F16_ARR];
    GGML_F16_VEC ay[GGML_F16_ARR];

    for (int i = 0; i < np; i += GGML_F16_STEP) {
        for (int j = 0; j < GGML_F16_ARR; j++) {
            ax[j] = GGML_F16_VEC_LOAD(x + i + j * GGML_F16_EPR, j);
            ay[j] = GGML_F16_VEC_LOAD(y + i + j * GGML_F16_EPR, j);

            sum[j] = GGML_F16_VEC_FMA(sum[j], ax[j], ay[j]);
        }
    }

    // reduce sum0..sum3 to sum0
    GGML_F16_VEC_REDUCE(sumf, sum);

    // leftovers
    for (int i = np; i < n; ++i) {
        sumf += (ggml_float)(GGML_FP16_TO_FP32(x[i]) * GGML_FP16_TO_FP32(y[i]));
    }
#else
    for (int i = 0; i < n; ++i) {
        sumf += (float)(GGML_FP16_TO_FP32(x[i]) * GGML_FP16_TO_FP32(y[i]));
    }
#endif

    * s = sumf;
}

// ggml_transpose

struct ggml_tensor* ggml_transpose(
    struct ggml_tensor* a) {

    struct ggml_tensor* result = ggml_view_tensor(a);

    result->ne[0] = a->ne[1];
    result->ne[1] = a->ne[0];

    result->nb[0] = a->nb[1];
    result->nb[1] = a->nb[0];

    return result;
}

// ggml_compute_forward_conv_transpose_2d

static void ggml_compute_forward_conv_transpose_2d(
    const struct ggml_tensor* src0,
    const struct ggml_tensor* src1,
    struct ggml_tensor* dst, const int32_t stride) {

    GGML_TENSOR_BINARY_OP_LOCALS

        const int ith = 0;
    const int nth = 1;

    const int nk = ne00 * ne01 * ne02 * ne03;
    int params_wsize = 1;
    ggml_fp16_t* params_wdata = (ggml_fp16_t*)calloc(1, sizeof(ggml_fp16_t));
    if (ith == 0) {
        memset(params_wdata, 0, params_wsize);

        // permute kernel data (src0) from (Kw x Kh x Cout x Cin) to (Cin x Kw x Kh x Cout)
        {
            ggml_fp16_t* const wdata = (ggml_fp16_t*)params_wdata + 0;

            for (int64_t i03 = 0; i03 < ne03; i03++) {
                for (int64_t i02 = 0; i02 < ne02; i02++) {
                    const ggml_fp16_t* const src = (ggml_fp16_t*)((char*)src0->data + i03 * nb03 + i02 * nb02);
                    ggml_fp16_t* dst_data = wdata + i02 * ne01 * ne00 * ne03;
                    for (int64_t i01 = 0; i01 < ne01; i01++) {
                        for (int64_t i00 = 0; i00 < ne00; i00++) {
                            dst_data[i01 * ne00 * ne03 + i00 * ne03 + i03] = src[i01 * ne00 + i00];
                        }
                    }
                }
            }
        }

        // permute source data (src1) from (Sw x Sh x Cin) to (Cin x Sw x Sh)
        {
            ggml_fp16_t* const wdata = (ggml_fp16_t*)params_wdata + nk;
            for (int i12 = 0; i12 < ne12; i12++) {
                for (int i11 = 0; i11 < ne11; i11++) {
                    const float* const src = (float*)((char*)src1->data + i12 * nb12 + i11 * nb11);
                    ggml_fp16_t* dst_data = wdata + i11 * ne10 * ne12;
                    for (int i10 = 0; i10 < ne10; i10++) {
                        dst_data[i10 * ne12 + i12] = GGML_FP32_TO_FP16(src[i10]);
                    }
                }
            }
        }

        memset(dst->data, 0, ggml_nbytes(dst));
    }

    // total patches in dst
    const int np = ne2;

    // patches per thread
    const int dp = (np + nth - 1) / nth;

    // patch range for this thread
    const int ip0 = dp * ith;
    const int ip1 = min(ip0 + dp, np);

    ggml_fp16_t* const wdata = (ggml_fp16_t*)params_wdata + 0;
    ggml_fp16_t* const wdata_src = wdata + nk;

    for (int i2 = ip0; i2 < ip1; i2++) { // Cout
        float* dst_data = (float*)((char*)dst->data + i2 * nb2);
        ggml_fp16_t* wdata_kernel = wdata + i2 * ne01 * ne00 * ne03;
        for (int i11 = 0; i11 < ne11; i11++) {
            for (int i10 = 0; i10 < ne10; i10++) {
                const int i1n = i11 * ne10 * ne12 + i10 * ne12;
                for (int i01 = 0; i01 < ne01; i01++) {
                    for (int i00 = 0; i00 < ne00; i00++) {
                        float v = 0;
                        ggml_vec_dot_f16(ne03, &v, 0,
                            wdata_src + i1n, 0,
                            wdata_kernel + i01 * ne00 * ne03 + i00 * ne03, 0, 1);
                        dst_data[(i11 * stride + i01) * ne0 + i10 * stride + i00] += v;
                    }
                }
            }
        }
    }
}

// ggml_conv_transpose_2d_p0

static int64_t ggml_calc_conv_transpose_output_size(int64_t ins, int64_t ks, int s, int p) {
    return (ins - 1) * s - 2 * p + ks;
}

struct ggml_tensor* ggml_conv_transpose_2d_p0(
    struct ggml_tensor* a,
    struct ggml_tensor* b,
    int                   stride) {

    const int64_t ne[4] = {
        ggml_calc_conv_transpose_output_size(b->ne[0], a->ne[0], stride, 0 /*p0*/),
        ggml_calc_conv_transpose_output_size(b->ne[1], a->ne[1], stride, 0 /*p1*/),
        a->ne[2], b->ne[3],
    };

    struct ggml_tensor* result = ggml_new_tensor(GGML_TYPE_F32, 4, ne);

    ggml_compute_forward_conv_transpose_2d(a, b, result, stride);

    return result;
}
