#include "../include/ggml.h"


static int64_t ggml_calc_conv_output_size(int64_t ins, int64_t ks, int s, int p, int d) {
    return (ins + 2 * p - d * (ks - 1) - 1) / s + 1;
}

// src0: kernel [OC, IC, KH, KW]
// src1: image [N, IC, IH, IW]
// dst:  result [N, OH, OW, IC*KH*KW]
static void ggml_compute_forward_im2col_f32(
    const int32_t* params,
    const struct ggml_tensor* src0,
    const struct ggml_tensor* src1,
    struct ggml_tensor* dst) {

    GGML_TENSOR_BINARY_OP_LOCALS;

    const int32_t s0 = params[0];
    const int32_t s1 = params[1];
    const int32_t p0 = params[2];
    const int32_t p1 = params[3];
    const int32_t d0 = params[4];
    const int32_t d1 = params[5];
    const bool is_2D = params[6] == 1;

    const int64_t N = is_2D ? ne13 : ne12;
    const int64_t IC = is_2D ? ne12 : ne11;
    const int64_t IH = is_2D ? ne11 : 1;
    const int64_t IW = ne10;

    const int64_t KH = is_2D ? ne01 : 1;
    const int64_t KW = ne00;

    const int64_t OH = is_2D ? ne2 : 1;
    const int64_t OW = ne1;

    int ofs0 = is_2D ? nb13 : nb12;
    int ofs1 = is_2D ? nb12 : nb11;

    // im2col: [N, IC, IH, IW] => [N, OH, OW, IC*KH*KW]
    {
        float* const wdata = (float*)dst->data;

        for (int64_t in = 0; in < N; in++) {
            for (int64_t ioh = 0; ioh < OH; ioh++) { // 1
                for (int64_t iow = 0; iow < OW; iow++) {
                    for (int64_t iic = 0; iic < IC; iic += 1) {

                        // micro kernel
                        float* dst_data = wdata + (in * OH * OW + ioh * OW + iow) * (IC * KH * KW); // [IC, KH, KW]
                        const float* const src_data = (float*)((char*)src1->data + in * ofs0 + iic * ofs1); // [IH, IW]

                        for (int64_t ikh = 0; ikh < KH; ikh++) {  // 1
                            for (int64_t ikw = 0; ikw < KW; ikw++) {
                                const int64_t iiw = iow * s0 + ikw * d0 - p0;
                                const int64_t iih = ioh * s1 + ikh * d1 - p1;

                                if (iih < 0 || iih >= IH || iiw < 0 || iiw >= IW) {
                                    dst_data[iic * (KH * KW) + ikh * KW + ikw] = 0;
                                }
                                else {
                                    dst_data[iic * (KH * KW) + ikh * KW + ikw] = (src_data[iih * IW + iiw]);
                                }
                            }
                        }
                    }
                }
            }
        }
    }
}

// src0: kernel [OC, IC, KH, KW]
// src1: image [N, IC, IH, IW]
// dst:  result [N, OH, OW, IC*KH*KW]
static void ggml_compute_forward_im2col_f16(
    const int32_t* params,
    const struct ggml_tensor* src0,
    const struct ggml_tensor* src1,
    struct ggml_tensor* dst) {

    GGML_TENSOR_BINARY_OP_LOCALS;

    const int32_t s0 = params[0];
    const int32_t s1 = params[1];
    const int32_t p0 = params[2];
    const int32_t p1 = params[3];
    const int32_t d0 = params[4];
    const int32_t d1 = params[5];
    const bool is_2D = params[6] == 1;

    const int64_t N = is_2D ? ne13 : ne12;
    const int64_t IC = is_2D ? ne12 : ne11;
    const int64_t IH = is_2D ? ne11 : 1;
    const int64_t IW = ne10;

    const int64_t KH = is_2D ? ne01 : 1;
    const int64_t KW = ne00;

    const int64_t OH = is_2D ? ne2 : 1;
    const int64_t OW = ne1;

    int ofs0 = is_2D ? nb13 : nb12;
    int ofs1 = is_2D ? nb12 : nb11;

    // im2col: [N, IC, IH, IW] => [N, OH, OW, IC*KH*KW]
    {
        ggml_fp16_t* const wdata = (ggml_fp16_t*)dst->data;

        for (int64_t in = 0; in < N; in++) {
            for (int64_t ioh = 0; ioh < OH; ioh++) { // 1
                for (int64_t iow = 0; iow < OW; iow++) {
                    for (int64_t iic = 0; iic < IC; iic += 1) {

                        // micro kernel
                        ggml_fp16_t* dst_data = wdata + (in * OH * OW + ioh * OW + iow) * (IC * KH * KW); // [IC, KH, KW]
                        const float* const src_data = (float*)((char*)src1->data + in * ofs0 + iic * ofs1); // [IH, IW]

                        for (int64_t ikh = 0; ikh < KH; ikh++) {  // 1
                            for (int64_t ikw = 0; ikw < KW; ikw++) {
                                const int64_t iiw = iow * s0 + ikw * d0 - p0;
                                const int64_t iih = ioh * s1 + ikh * d1 - p1;

                                if (iih < 0 || iih >= IH || iiw < 0 || iiw >= IW) {
                                    dst_data[iic * (KH * KW) + ikh * KW + ikw] = 0;
                                }
                                else {
                                    dst_data[iic * (KH * KW) + ikh * KW + ikw] = GGML_FP32_TO_FP16(src_data[iih * IW + iiw]);
                                }
                            }
                        }
                    }
                }
            }
        }
    }
}

// im2col: [N, IC, IH, IW] => [N, OH, OW, IC*KH*KW]
// a: [OC£¬IC, KH, KW]
// b: [N, IC, IH, IW]
// result: [N, OH, OW, IC*KH*KW]
struct ggml_tensor* ggml_im2col(
    struct ggml_tensor* a,
    struct ggml_tensor* b,
    int                  s0,
    int                  s1,
    int                  p0,
    int                  p1,
    int                  d0,
    int                  d1,
    bool                 is_2D,
    enum ggml_type       dst_type) {

    const int64_t OH = is_2D ? ggml_calc_conv_output_size(b->ne[1], a->ne[1], s1, p1, d1) : 0;
    const int64_t OW = ggml_calc_conv_output_size(b->ne[0], a->ne[0], s0, p0, d0);

    const int64_t ne[4] = {
        is_2D ? (a->ne[2] * a->ne[1] * a->ne[0]) : a->ne[1] * a->ne[0],
        OW,
        is_2D ? OH : b->ne[2],
        is_2D ? b->ne[3] : 1,
    };

    struct ggml_tensor* result = ggml_new_tensor(dst_type, 4, ne);
    int32_t params[] = { s0, s1, p0, p1, d0, d1, (is_2D ? 1 : 0) };

    switch (result->type) {
    case GGML_TYPE_F16:
    {
        ggml_compute_forward_im2col_f16(params, a, b, result);
    } break;
    case GGML_TYPE_F32:
    {
        ggml_compute_forward_im2col_f32(params, a, b, result);
    } break;
    default:
    {

    }
    }

    return result;
}
