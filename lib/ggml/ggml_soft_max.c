#include "../include/ggml.h"

inline static void ggml_vec_cpy_f32(const int n, float* y, const float* x) { for (int i = 0; i < n; ++i) y[i] = x[i]; }

inline static void ggml_vec_max_f32(const int n, float* s, const float* x) {
#ifndef GGML_USE_ACCELERATE
    float max1 = -INFINITY;
    for (int i = 0; i < n; ++i) {
        max1 = max(max1, x[i]);
    }
    *s = max1;
#else
    vDSP_maxv(x, 1, s, n);
#endif
}
static float ggml_vec_soft_max_f32(const int n, float* y, const float* x, float max) {
    int i = 0;
    float sum = 0;
#if defined(__AVX512F__) && defined(__AVX512DQ__)
    for (; i + 15 < n; i += 16) {
        __m512 val = ggml_v_expf(_mm512_sub_ps(_mm512_loadu_ps(x + i),
            _mm512_set1_ps(max)));
        _mm512_storeu_ps(y + i, val);
        sum += (ggml_float)_mm512_reduce_add_ps(val);
    }
#elif defined(__AVX2__) && defined(__FMA__)
    for (; i + 7 < n; i += 8) {
        __m256 val = ggml_v_expf(_mm256_sub_ps(_mm256_loadu_ps(x + i),
            _mm256_set1_ps(max)));
        _mm256_storeu_ps(y + i, val);
        __m128 val2 = _mm_add_ps(_mm256_extractf128_ps(val, 1),
            _mm256_castps256_ps128(val));
        val2 = _mm_add_ps(val2, _mm_movehl_ps(val2, val2));
        val2 = _mm_add_ss(val2, _mm_movehdup_ps(val2));
        sum += (ggml_float)_mm_cvtss_f32(val2);
    }
#elif defined(__SSE2__)
    for (; i + 3 < n; i += 4) {
        __m128 val = ggml_v_expf(_mm_sub_ps(_mm_loadu_ps(x + i),
            _mm_set1_ps(max)));
        _mm_storeu_ps(y + i, val);
#if defined(__AVX__) || defined(__AVX2__) || defined(__AVX512F__)
        val = _mm_add_ps(val, _mm_movehl_ps(val, val));
        val = _mm_add_ss(val, _mm_movehdup_ps(val));
#else
        __m128 tmp = _mm_shuffle_ps(val, val, _MM_SHUFFLE(2, 3, 0, 1));
        val = _mm_add_ps(val, tmp);
        tmp = _mm_movehl_ps(tmp, val);
        val = _mm_add_ss(val, tmp);
#endif
        sum += (ggml_float)_mm_cvtss_f32(val);
    }
#elif defined(__ARM_NEON) && defined(__aarch64__)
    for (; i + 3 < n; i += 4) {
        float32x4_t val = ggml_v_expf(vsubq_f32(vld1q_f32(x + i),
            vdupq_n_f32(max)));
        vst1q_f32(y + i, val);
        sum += (ggml_float)vaddvq_f32(val);
    }
#endif
    for (; i < n; ++i) {
        float val = expf(x[i] - max);
        sum += (float)val;
        y[i] = val;
    }
    return sum;
}

// ggml_compute_forward_soft_max

static void ggml_compute_forward_soft_max_f32(
    const struct ggml_tensor* src0,
    const struct ggml_tensor* src1,
    struct ggml_tensor* dst,
    float                 scale,
    float                 max_bias) {

    // TODO: handle transposed/permuted matrices

    const int ith = 0;
    const int nth = 1;

    GGML_TENSOR_UNARY_OP_LOCALS

        //const int64_t ne11 = src1 ? src1->ne[1] : 1;

        // TODO: is this supposed to be ceil instead of floor?
        //       https://huggingface.co/mosaicml/mpt-7b/blob/main/attention.py#L370
        const uint32_t n_head = ne02;
    const uint32_t n_head_log2 = 1u << (uint32_t)floor(log2(n_head));

    const float m0 = powf(2.0f, -(max_bias) / n_head_log2);
    const float m1 = powf(2.0f, -(max_bias / 2.0f) / n_head_log2);

    const int nc = src0->ne[0];
    const int nr = ggml_nrows(src0);

    // rows per thread
    const int dr = (nr + nth - 1) / nth;

    // row range for this thread
    const int ir0 = dr * ith;
    const int ir1 = min(ir0 + dr, nr);

    float* wp = (float*)calloc(1,1);

    const bool use_f16 = (src1 && src1->type == GGML_TYPE_F16);

    for (int i1 = ir0; i1 < ir1; i1++) {
        // ALiBi
        const uint32_t h = (i1 / ne01) % ne02; // head
        const float slope = (max_bias > 0.0f) ? h < n_head_log2 ? powf(m0, h + 1) : powf(m1, 2 * (h - n_head_log2) + 1) : 1.0f;

        float* sp = (float*)((char*)src0->data + i1 * src0->nb[1]);
        float* dp = (float*)((char*)dst->data + i1 * dst->nb[1]);

        // broadcast the mask across rows
        ggml_fp16_t* mp_f16 = src1 ? (ggml_fp16_t*)((char*)src1->data) + (i1 % ne01) * ne00 : NULL;
        float* mp_f32 = src1 ? (float*)((char*)src1->data) + (i1 % ne01) * ne00 : NULL;

        ggml_vec_cpy_f32(nc, wp, sp);
        ggml_vec_scale_f32(nc, wp, scale);
        if (mp_f32) {
            if (use_f16) {
                for (int i = 0; i < nc; ++i) {
                    wp[i] += slope * GGML_FP16_TO_FP32(mp_f16[i]);
                }
            }
            else {
                for (int i = 0; i < nc; ++i) {
                    wp[i] += slope * mp_f32[i];
                }
            }
        }

        float max = -INFINITY;
        ggml_vec_max_f32(nc, &max, wp);

        float sum = ggml_vec_soft_max_f32(nc, dp, wp, max);

        sum = 1.0 / sum;
        ggml_vec_scale_f32(nc, dp, sum);

    }
}

static void ggml_compute_forward_soft_max(
    const struct ggml_tensor* src0,
    struct ggml_tensor* mask,
    struct ggml_tensor* dst,
    float                 scale,
    float                 max_bias) {

    switch (src0->type) {
    case GGML_TYPE_F32:
    {
        ggml_compute_forward_soft_max_f32(src0, mask, dst, scale, max_bias);
    } break;
    default:
    {
        // GGML_ABORT("fatal error");
    }
    }
}

// ggml_soft_max

static struct ggml_tensor* ggml_soft_max_impl(
    struct ggml_tensor* a,
    struct ggml_tensor* mask,
    float                 scale,
    float                 max_bias,
    bool                  inplace) {

    struct ggml_tensor* result = inplace ? ggml_view_tensor(a) : ggml_dup_tensor(a);

    ggml_compute_forward_soft_max(a, mask, result, scale, max_bias);

    return result;
}

struct ggml_tensor* ggml_soft_max(
    struct ggml_tensor* a) {
    return ggml_soft_max_impl(a, NULL, 1.0f, 0.0f, false);
}

struct ggml_tensor* ggml_soft_max_inplace(
    struct ggml_tensor* a) {
    return ggml_soft_max_impl(a, NULL, 1.0f, 0.0f, true);
}