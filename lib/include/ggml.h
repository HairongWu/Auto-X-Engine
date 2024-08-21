
#pragma once

#include "autox_nn.h"
#include <math.h>

#define GGML_MAX_DIMS           4



#define GGML_UNUSED(x) (void)(x)
// used to copy the number of elements and stride in bytes of tensors into local variables.
// main purpose is to reduce code duplication and improve readability.
//
// example:
//
//    GGML_TENSOR_LOCALS(int64_t, ne1, src1, ne);
//    GGML_TENSOR_LOCALS(size_t,  nb1, src1, nb);
//
#define GGML_TENSOR_LOCALS_1(type, prefix, pointer, array) \
    const type prefix##0 = (pointer)->array[0]; \
    GGML_UNUSED(prefix##0);
#define GGML_TENSOR_LOCALS_2(type, prefix, pointer, array) \
    GGML_TENSOR_LOCALS_1    (type, prefix, pointer, array) \
    const type prefix##1 = (pointer)->array[1]; \
    GGML_UNUSED(prefix##1);
#define GGML_TENSOR_LOCALS_3(type, prefix, pointer, array) \
    GGML_TENSOR_LOCALS_2    (type, prefix, pointer, array) \
    const type prefix##2 = (pointer)->array[2]; \
    GGML_UNUSED(prefix##2);
#define GGML_TENSOR_LOCALS(type, prefix, pointer, array) \
    GGML_TENSOR_LOCALS_3  (type, prefix, pointer, array) \
    const type prefix##3 = (pointer)->array[3]; \
    GGML_UNUSED(prefix##3);
#define GGML_TENSOR_BINARY_OP_LOCALS \
    GGML_TENSOR_LOCALS(int64_t, ne0, src0, ne) \
    GGML_TENSOR_LOCALS(size_t,  nb0, src0, nb) \
    GGML_TENSOR_LOCALS(int64_t, ne1, src1, ne) \
    GGML_TENSOR_LOCALS(size_t,  nb1, src1, nb) \
    GGML_TENSOR_LOCALS(int64_t, ne,  dst,  ne) \
    GGML_TENSOR_LOCALS(size_t,  nb,  dst,  nb)
#define GGML_TENSOR_UNARY_OP_LOCALS \
    GGML_TENSOR_LOCALS(int64_t, ne0, src0, ne) \
    GGML_TENSOR_LOCALS(size_t,  nb0, src0, nb) \
    GGML_TENSOR_LOCALS(int64_t, ne,  dst,  ne) \
    GGML_TENSOR_LOCALS(size_t,  nb,  dst,  nb)

typedef uint16_t ggml_fp16_t;
typedef struct { uint16_t bits; } ggml_bf16_t;

enum ggml_type {
    GGML_TYPE_F32 = 0,
    GGML_TYPE_F16 = 1,
    GGML_TYPE_Q4_0 = 2,
    GGML_TYPE_Q4_1 = 3,
    // GGML_TYPE_Q4_2 = 4, support has been removed
    // GGML_TYPE_Q4_3 = 5, support has been removed
    GGML_TYPE_Q5_0 = 6,
    GGML_TYPE_Q5_1 = 7,
    GGML_TYPE_Q8_0 = 8,
    GGML_TYPE_Q8_1 = 9,
    GGML_TYPE_Q2_K = 10,
    GGML_TYPE_Q3_K = 11,
    GGML_TYPE_Q4_K = 12,
    GGML_TYPE_Q5_K = 13,
    GGML_TYPE_Q6_K = 14,
    GGML_TYPE_Q8_K = 15,
    GGML_TYPE_IQ2_XXS = 16,
    GGML_TYPE_IQ2_XS = 17,
    GGML_TYPE_IQ3_XXS = 18,
    GGML_TYPE_IQ1_S = 19,
    GGML_TYPE_IQ4_NL = 20,
    GGML_TYPE_IQ3_S = 21,
    GGML_TYPE_IQ2_S = 22,
    GGML_TYPE_IQ4_XS = 23,
    GGML_TYPE_I8 = 24,
    GGML_TYPE_I16 = 25,
    GGML_TYPE_I32 = 26,
    GGML_TYPE_I64 = 27,
    GGML_TYPE_F64 = 28,
    GGML_TYPE_IQ1_M = 29,
    GGML_TYPE_BF16 = 30,
    GGML_TYPE_Q4_0_4_4 = 31,
    GGML_TYPE_Q4_0_4_8 = 32,
    GGML_TYPE_Q4_0_8_8 = 33,
    GGML_TYPE_COUNT,
};


// n-dimensional tensor
struct ggml_tensor {
    enum ggml_type         type;

    int64_t ne[GGML_MAX_DIMS]; // number of elements
    size_t  nb[GGML_MAX_DIMS]; // stride in bytes:

    void* data;
};

// FP16 <-> FP32
// ref: https://github.com/Maratyszcza/FP16

static inline float fp32_from_bits(uint32_t w) {
    union {
        uint32_t as_bits;
        float as_value;
    } fp32;
    fp32.as_bits = w;
    return fp32.as_value;
}

static inline uint32_t fp32_to_bits(float f) {
    union {
        float as_value;
        uint32_t as_bits;
    } fp32;
    fp32.as_value = f;
    return fp32.as_bits;
}

static inline ggml_fp16_t ggml_compute_fp32_to_fp16(float f) {
#if defined(__STDC_VERSION__) && (__STDC_VERSION__ >= 199901L) || defined(__GNUC__) && !defined(__STRICT_ANSI__)
    const float scale_to_inf = 0x1.0p+112f;
    const float scale_to_zero = 0x1.0p-110f;
#else
    const float scale_to_inf = fp32_from_bits(UINT32_C(0x77800000));
    const float scale_to_zero = fp32_from_bits(UINT32_C(0x08800000));
#endif
    float base = (fabsf(f) * scale_to_inf) * scale_to_zero;

    const uint32_t w = fp32_to_bits(f);
    const uint32_t shl1_w = w + w;
    const uint32_t sign = w & UINT32_C(0x80000000);
    uint32_t bias = shl1_w & UINT32_C(0xFF000000);
    if (bias < UINT32_C(0x71000000)) {
        bias = UINT32_C(0x71000000);
    }

    base = fp32_from_bits((bias >> 1) + UINT32_C(0x07800000)) + base;
    const uint32_t bits = fp32_to_bits(base);
    const uint32_t exp_bits = (bits >> 13) & UINT32_C(0x00007C00);
    const uint32_t mantissa_bits = bits & UINT32_C(0x00000FFF);
    const uint32_t nonsign = exp_bits + mantissa_bits;
    return (sign >> 16) | (shl1_w > UINT32_C(0xFF000000) ? UINT16_C(0x7E00) : nonsign);
}

#define GGML_COMPUTE_FP32_TO_FP16(x) ggml_compute_fp32_to_fp16(x)

// precomputed f32 table for f16 (256 KB)
// defined in ggml.c, initialized in ggml_init()
float ggml_table_f32_f16[1 << 16];

// On ARM NEON, it's quicker to directly convert x -> x instead of calling into ggml_lookup_fp16_to_fp32,
// so we define GGML_FP16_TO_FP32 and GGML_FP32_TO_FP16 elsewhere for NEON.
// This is also true for POWER9.
#if !defined(GGML_FP16_TO_FP32)
inline static float ggml_lookup_fp16_to_fp32(ggml_fp16_t f) {
    uint16_t s;
    memcpy(&s, &f, sizeof(uint16_t));
    return ggml_table_f32_f16[s];
}

#define GGML_FP16_TO_FP32(x) ggml_lookup_fp16_to_fp32(x)
#endif

#if !defined(GGML_FP32_TO_FP16)
#define GGML_FP32_TO_FP16(x) GGML_COMPUTE_FP32_TO_FP16(x)
#endif

inline int64_t ggml_nelements(const struct ggml_tensor* tensor) {
    return tensor->ne[0] * tensor->ne[1] * tensor->ne[2] * tensor->ne[3];
}

inline int64_t ggml_nrows(const struct ggml_tensor* tensor) {
    return tensor->ne[1] * tensor->ne[2] * tensor->ne[3];
}

inline int64_t ggml_blck_size(enum ggml_type type) {
    return 1;
}

inline size_t ggml_type_size(enum ggml_type type) {
    switch (type) {
    case  GGML_TYPE_F32:
        return sizeof(float);
    case GGML_TYPE_F16:
        return sizeof(ggml_fp16_t);
    default:
    {
        return 0;
    } break;
    };
}

//inline static void ggml_vec_scale_f32(const int n, float * y, const float   v) { for (int i = 0; i < n; ++i) y[i] *= v;          }
inline static void ggml_vec_scale_f32(const int n, float* y, const float   v) {
#if defined(GGML_USE_ACCELERATE)
    vDSP_vsmul(y, 1, &v, y, 1, n);
#elif defined(GGML_SIMD)
    const int np = (n & ~(GGML_F32_STEP - 1));

    GGML_F32_VEC vx = GGML_F32_VEC_SET1(v);

    GGML_F32_VEC ay[GGML_F32_ARR];

    for (int i = 0; i < np; i += GGML_F32_STEP) {
        for (int j = 0; j < GGML_F32_ARR; j++) {
            ay[j] = GGML_F32_VEC_LOAD(y + i + j * GGML_F32_EPR);
            ay[j] = GGML_F32_VEC_MUL(ay[j], vx);

            GGML_F32_VEC_STORE(y + i + j * GGML_F32_EPR, ay[j]);
        }
    }

    // leftovers
    for (int i = np; i < n; ++i) {
        y[i] *= v;
    }
#else
    // scalar
    for (int i = 0; i < n; ++i) {
        y[i] *= v;
    }
#endif
}
/**
 * Converts brain16 to float32.
 *
 * The bfloat16 floating point format has the following structure:
 *
 *       ©°sign
 *       ©¦
 *       ©¦   ©°exponent
 *       ©¦   ©¦
 *       ©¦   ©¦      ©°mantissa
 *       ©¦   ©¦      ©¦
 *       ©¦©°©¤©¤©Ø©¤©¤©¤©´©°©¤©Ø©¤©¤©¤©´
 *     0b0000000000000000 brain16
 *
 * Since bf16 has the same number of exponent bits as a 32bit float,
 * encoding and decoding numbers becomes relatively straightforward.
 *
 *       ©°sign
 *       ©¦
 *       ©¦   ©°exponent
 *       ©¦   ©¦
 *       ©¦   ©¦      ©°mantissa
 *       ©¦   ©¦      ©¦
 *       ©¦©°©¤©¤©Ø©¤©¤©¤©´©°©¤©Ø©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©´
 *     0b00000000000000000000000000000000 IEEE binary32
 *
 * For comparison, the standard fp16 format has fewer exponent bits.
 *
 *       ©°sign
 *       ©¦
 *       ©¦  ©°exponent
 *       ©¦  ©¦
 *       ©¦  ©¦    ©°mantissa
 *       ©¦  ©¦    ©¦
 *       ©¦©°©¤©Ø©¤©´©°©¤©Ø©¤©¤©¤©¤©¤©¤©´
 *     0b0000000000000000 IEEE binary16
 *
 * @see IEEE 754-2008
 */
static inline float ggml_compute_bf16_to_fp32(ggml_bf16_t h) {
    union {
        float f;
        uint32_t i;
    } u;
    u.i = (uint32_t)h.bits << 16;
    return u.f;
}

/**
 * Converts float32 to brain16.
 *
 * This function is binary identical to AMD Zen4 VCVTNEPS2BF16.
 * Subnormals shall be flushed to zero, and NANs will be quiet.
 * This code should vectorize nicely if using modern compilers.
 */
static inline ggml_bf16_t ggml_compute_fp32_to_bf16(float s) {
    ggml_bf16_t h;
    union {
        float f;
        uint32_t i;
    } u;
    u.f = s;
    if ((u.i & 0x7fffffff) > 0x7f800000) { /* nan */
        h.bits = (u.i >> 16) | 64; /* force to quiet */
        return h;
    }
    if (!(u.i & 0x7f800000)) { /* subnormal */
        h.bits = (u.i & 0x80000000) >> 16; /* flush to zero */
        return h;
    }
    h.bits = (u.i + (0x7fff + ((u.i >> 16) & 1))) >> 16;
    return h;
}

#define GGML_FP32_TO_BF16(x) ggml_compute_fp32_to_bf16(x)
#define GGML_BF16_TO_FP32(x) ggml_compute_bf16_to_fp32(x)

#ifdef  __cplusplus
extern "C" {
#endif
    //mem
    struct ggml_tensor* ggml_new_tensor_impl(
        enum   ggml_type      type,
        int                   n_dims,
        const int64_t* ne,
        struct ggml_tensor* view_src,
        size_t                view_offs);
    struct ggml_tensor* ggml_new_tensor(
        enum   ggml_type      type,
        int                   n_dims,
        const int64_t* ne);
    struct ggml_tensor* ggml_new_tensor_1d(
        enum   ggml_type      type,
        int64_t ne0);
    struct ggml_tensor* ggml_new_tensor_2d(
        enum   ggml_type      type,
        int64_t ne0,
        int64_t ne1);
    struct ggml_tensor* ggml_new_tensor_3d(
        enum   ggml_type      type,
        int64_t ne0,
        int64_t ne1,
        int64_t ne2);
    struct ggml_tensor* ggml_new_tensor_4d(
        enum   ggml_type type,
        int64_t ne0,
        int64_t ne1,
        int64_t ne2,
        int64_t ne3);
    struct ggml_tensor* ggml_dup_tensor(const struct ggml_tensor* src);
    struct ggml_tensor* ggml_view_tensor(
        struct ggml_tensor* src);

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
        enum ggml_type       dst_type);
    struct ggml_tensor* ggml_mul_mat(
        struct ggml_tensor* a,
        struct ggml_tensor* b);
    struct ggml_tensor* ggml_conv_2d(
        struct ggml_tensor* a,
        struct ggml_tensor* b,
        int                  s0,
        int                  s1,
        int                  p0,
        int                  p1,
        int                  d0,
        int                  d1);
    struct ggml_tensor* ggml_conv_2d_sk_p0(
        struct ggml_tensor* a,
        struct ggml_tensor* b);
    struct ggml_tensor* ggml_conv_2d_s1_ph(
        struct ggml_tensor* a,
        struct ggml_tensor* b);
    struct ggml_tensor* ggml_cont(
        struct ggml_tensor* a);
    void ggml_compute_forward_dup(
        const struct ggml_tensor* src0,
        struct ggml_tensor* dst);
    struct ggml_tensor* ggml_add(
        struct ggml_tensor* a,
        struct ggml_tensor* b);
    struct ggml_tensor* ggml_add_inplace(
        struct ggml_tensor* a,
        struct ggml_tensor* b);
    struct ggml_tensor* ggml_repeat(
        struct ggml_tensor* a,
        struct ggml_tensor* b);
    struct ggml_tensor* ggml_norm(
        struct ggml_tensor* a,
        float eps);
    struct ggml_tensor* ggml_norm_inplace(
        struct ggml_tensor* a,
        float eps);
    struct ggml_tensor* ggml_mul(
        struct ggml_tensor* a,
        struct ggml_tensor* b);
    struct ggml_tensor* ggml_scale_inplace(
        struct ggml_tensor* a,
        float                s);
    struct ggml_tensor* ggml_scale(
        struct ggml_tensor* a,
        float                s);
    struct ggml_tensor* ggml_get_rel_pos(
        struct ggml_tensor* a,
        int                   qh,
        int                   kh);
    struct ggml_tensor* ggml_add_rel_pos(
        struct ggml_tensor* a,
        struct ggml_tensor* pw,
        struct ggml_tensor* ph);
    struct ggml_tensor* ggml_add_rel_pos_inplace(
        struct ggml_tensor* a,
        struct ggml_tensor* pw,
        struct ggml_tensor* ph);
    struct ggml_tensor* ggml_soft_max(
        struct ggml_tensor* a);
    struct ggml_tensor* ggml_soft_max_inplace(
        struct ggml_tensor* a);
    struct ggml_tensor* ggml_win_part(
        struct ggml_tensor* a,
        int                   w);
    struct ggml_tensor* ggml_win_unpart(
        struct ggml_tensor* a,
        int                   w0,
        int                   h0,
        int                   w);
    struct ggml_tensor* ggml_gelu(
        struct ggml_tensor* a);
    struct ggml_tensor* ggml_gelu_inplace(
        struct ggml_tensor* a);
    struct ggml_tensor* ggml_layer_norm_2d(
        struct ggml_tensor* layer,
        int                   n_channels,
        struct ggml_tensor* w,
        struct ggml_tensor* b,
        float                 eps);
    struct ggml_tensor* ggml_sam_cos(
        struct ggml_tensor* a);
    struct ggml_tensor* ggml_sam_sin(
        struct ggml_tensor* a);
    struct ggml_tensor* ggml_transpose(
        struct ggml_tensor* a);
    struct ggml_tensor* ggml_conv_transpose_2d_p0(
        struct ggml_tensor* a,
        struct ggml_tensor* b,
        int                   stride);
    struct ggml_tensor* ggml_relu_inplace(
        struct ggml_tensor* a);
    struct ggml_tensor* ggml_relu(
        struct ggml_tensor* a);
    struct ggml_tensor* ggml_cpy(
        struct ggml_tensor* a,
        struct ggml_tensor* b);
    struct ggml_tensor* ggml_layer_norm_2d(
        struct ggml_tensor* layer,
        int                   n_channels,
        struct ggml_tensor* w,
        struct ggml_tensor* b,
        float                 eps);
    struct ggml_tensor* ggml_permute(
        struct ggml_tensor* a,
        int                   axis0,
        int                   axis1,
        int                   axis2,
        int                   axis3);
    struct ggml_tensor* ggml_reshape_3d(
        struct ggml_tensor* a,
        int64_t               ne0,
        int64_t               ne1,
        int64_t               ne2);
    struct ggml_tensor* ggml_reshape_4d(
        struct ggml_tensor* a,
        int64_t               ne0,
        int64_t               ne1,
        int64_t               ne2,
        int64_t               ne3);
    struct ggml_tensor* ggml_view_1d(
        struct ggml_tensor* a,
        int64_t               ne0,
        size_t                offset);
    struct ggml_tensor* ggml_view_2d(
        struct ggml_tensor* a,
        int64_t               ne0,
        int64_t               ne1,
        size_t                nb1,
        size_t                offset);
    struct ggml_tensor* ggml_view_3d(
        struct ggml_tensor* a,
        int64_t               ne0,
        int64_t               ne1,
        int64_t               ne2,
        size_t                nb1,
        size_t                nb2,
        size_t                offset);
    struct ggml_tensor* ggml_view_4d(
        struct ggml_tensor* a,
        int64_t               ne0,
        int64_t               ne1,
        int64_t               ne2,
        int64_t               ne3,
        size_t                nb1,
        size_t                nb2,
        size_t                nb3,
        size_t                offset);
#ifdef  __cplusplus
}
#endif
