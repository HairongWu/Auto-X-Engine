#include "../include/ggml.h"

enum ggml_unary_op {
    GGML_UNARY_OP_ABS,
    GGML_UNARY_OP_SGN,
    GGML_UNARY_OP_NEG,
    GGML_UNARY_OP_STEP,
    GGML_UNARY_OP_TANH,
    GGML_UNARY_OP_ELU,
    GGML_UNARY_OP_RELU,
    GGML_UNARY_OP_SIGMOID,
    GGML_UNARY_OP_GELU,
    GGML_UNARY_OP_GELU_QUICK,
    GGML_UNARY_OP_SILU,
    GGML_UNARY_OP_HARDSWISH,
    GGML_UNARY_OP_HARDSIGMOID,

    GGML_UNARY_OP_COUNT,
};

inline static void ggml_vec_relu_f32(const int n, float* y, const float* x) { for (int i = 0; i < n; ++i) y[i] = (x[i] > 0.f) ? x[i] : 0.f; }
// ggml_compute_forward_relu

static void ggml_compute_forward_relu_f32(
    const struct ggml_tensor* src0,
    struct ggml_tensor* dst) {

    const int n = ggml_nrows(src0);
    const int nc = src0->ne[0];

    for (int i = 0; i < n; i++) {
        ggml_vec_relu_f32(nc,
            (float*)((char*)dst->data + i * (dst->nb[1])),
            (float*)((char*)src0->data + i * (src0->nb[1])));
    }
}

static void ggml_compute_forward_relu(
    const struct ggml_tensor* src0,
    struct ggml_tensor* dst) {

    switch (src0->type) {
    case GGML_TYPE_F32:
    {
        ggml_compute_forward_relu_f32(src0, dst);
    } break;
    default:
    {
        // GGML_ABORT("fatal error");
    }
    }
}

// precomputed gelu table for f16 (128 KB)
static ggml_fp16_t ggml_table_gelu_f16[1 << 16];

inline static void ggml_vec_gelu_f32(const int n, float* y, const float* x) {
    uint16_t t;
    for (int i = 0; i < n; ++i) {
        if (x[i] <= -10.0f) {
            y[i] = 0.0f;
        }
        else if (x[i] >= 10.0f) {
            y[i] = x[i];
        }
        else {
            ggml_fp16_t fp16 = GGML_FP32_TO_FP16(x[i]);
            memcpy(&t, &fp16, sizeof(uint16_t));
            y[i] = GGML_FP16_TO_FP32(ggml_table_gelu_f16[t]);
        }
    }
}

// ggml_compute_forward_gelu

static void ggml_compute_forward_gelu_f32(
    const struct ggml_tensor* src0,
    struct ggml_tensor* dst) {

    const int ith = 0;
    const int nth = 1;

    const int nc = src0->ne[0];
    const int nr = ggml_nrows(src0);

    // rows per thread
    const int dr = (nr + nth - 1) / nth;

    // row range for this thread
    const int ir0 = dr * ith;
    const int ir1 = min(ir0 + dr, nr);

    for (int i1 = ir0; i1 < ir1; i1++) {
        ggml_vec_gelu_f32(nc,
            (float*)((char*)dst->data + i1 * (dst->nb[1])),
            (float*)((char*)src0->data + i1 * (src0->nb[1])));
    }
}

static void ggml_compute_forward_gelu(
    const struct ggml_tensor* src0,
    struct ggml_tensor* dst) {

    switch (src0->type) {
    case GGML_TYPE_F32:
    {
        ggml_compute_forward_gelu_f32(src0, dst);
    } break;
    default:
    {
        // GGML_ABORT("fatal error");
    }
    }
}

//gmml_compute_forward_unary

static void ggml_compute_forward_unary(
    const struct ggml_tensor* src0,
    struct ggml_tensor* dst, const enum ggml_unary_op op) {

    switch (op) {
    //case GGML_UNARY_OP_ABS:
    //{
    //    ggml_compute_forward_abs(src0, dst);
    //} break;
    //case GGML_UNARY_OP_SGN:
    //{
    //    ggml_compute_forward_sgn(src0, dst);
    //} break;
    //case GGML_UNARY_OP_NEG:
    //{
    //    ggml_compute_forward_neg(src0, dst);
    //} break;
    //case GGML_UNARY_OP_STEP:
    //{
    //    ggml_compute_forward_step(src0, dst);
    //} break;
    //case GGML_UNARY_OP_TANH:
    //{
    //    ggml_compute_forward_tanh(src0, dst);
    //} break;
    //case GGML_UNARY_OP_ELU:
    //{
    //    ggml_compute_forward_elu(src0, dst);
    //} break;
    case GGML_UNARY_OP_RELU:
    {
        ggml_compute_forward_relu(src0, dst);
    } break;
    //case GGML_UNARY_OP_SIGMOID:
    //{
    //    ggml_compute_forward_sigmoid(src0, dst);
    //} break;
    case GGML_UNARY_OP_GELU:
    {
        ggml_compute_forward_gelu(src0, dst);
    } break;
    //case GGML_UNARY_OP_GELU_QUICK:
    //{
    //    ggml_compute_forward_gelu_quick(src0, dst);
    //} break;
    //case GGML_UNARY_OP_SILU:
    //{
    //    ggml_compute_forward_silu(src0, dst);
    //} break;
    //case GGML_UNARY_OP_HARDSWISH:
    //{
    //    ggml_compute_forward_hardswish(src0, dst);
    //} break;
    //case GGML_UNARY_OP_HARDSIGMOID:
    //{
    //    ggml_compute_forward_hardsigmoid(src0, dst);
    //} break;
    default:
    {
        // GGML_ABORT("fatal error");
    }
    }
}

// ggml_unary

static struct ggml_tensor* ggml_unary_impl(
    struct ggml_tensor* a,
    enum ggml_unary_op op,
    bool inplace) {

    struct ggml_tensor* result = inplace ? ggml_view_tensor(a) : ggml_dup_tensor(a);

    ggml_compute_forward_unary(a, result, op);

    return result;
}

struct ggml_tensor* ggml_unary(
    struct ggml_tensor* a,
    enum ggml_unary_op op) {
    return ggml_unary_impl(a, op, false);
}

struct ggml_tensor* ggml_unary_inplace(
    struct ggml_tensor* a,
    enum ggml_unary_op op) {
    return ggml_unary_impl(a, op, true);
}

// ggml_relu

struct ggml_tensor* ggml_relu(
    struct ggml_tensor* a) {
    return ggml_unary(a, GGML_UNARY_OP_RELU);
}

struct ggml_tensor* ggml_relu_inplace(
    struct ggml_tensor* a) {
    return ggml_unary_inplace(a, GGML_UNARY_OP_RELU);
}

// ggml_gelu

struct ggml_tensor* ggml_gelu(
    struct ggml_tensor* a) {
    return ggml_unary(a, GGML_UNARY_OP_GELU);
}