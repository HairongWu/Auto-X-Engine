#include "../include/ggml.h"

// a: [OC£¬IC, KH, KW]
// b: [N, IC, IH, IW]
// result: [N, OC, OH, OW]
struct ggml_tensor* ggml_conv_2d(
    struct ggml_tensor* a,
    struct ggml_tensor* b,
    int                  s0,
    int                  s1,
    int                  p0,
    int                  p1,
    int                  d0,
    int                  d1) {
    struct ggml_tensor* im2col = ggml_im2col(a, b, s0, s1, p0, p1, d0, d1, true, GGML_TYPE_F16); // [N, OH, OW, IC * KH * KW]

    struct ggml_tensor* result =
        ggml_mul_mat(
            im2col, // [N, OH, OW, IC * KH * KW] => [N*OH*OW, IC * KH * KW]
            a);                       // [OC£¬IC, KH, KW] => [OC, IC * KH * KW]

    result = ggml_cont(result); // [N, OC, OH, OW]


    return result;
}

// ggml_conv_2d_sk_p0
struct ggml_tensor* ggml_conv_2d_sk_p0(
    struct ggml_tensor* a,
    struct ggml_tensor* b) {
    return ggml_conv_2d(a, b, a->ne[0], a->ne[1], 0, 0, 1, 1);
}

// ggml_conv_2d_s1_ph

struct ggml_tensor* ggml_conv_2d_s1_ph(
    struct ggml_tensor* a,
    struct ggml_tensor* b) {
    return ggml_conv_2d(a, b, 1, 1, a->ne[0] / 2, a->ne[1] / 2, 1, 1);
}