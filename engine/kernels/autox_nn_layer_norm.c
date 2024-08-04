#include "../include/autox_nn.h"

// The overall computation can be split into two stages. 
// The first stage is standardization, which makes the normalized elements have zero mean and unit variances. 
// The second stage then scales and shifts the outcome of the first stage
#define YMM_FLOAT_BLOCK 8

void autox_layer_norm(float* x,
    const float* bias,
    const float* scale,
    float* out,
    int height,
    const float epsilon,
    int right) {
    __m256 sum;
    __m256 mean_vec, var_vec;
    __m128 hi, lo;
    __m256 tmp;
    size_t offset;
    size_t j;
    int block = YMM_FLOAT_BLOCK;
    const int rest = right % block;
    const int end = right - rest;

    __m256 reverse_num_vec =
        _mm256_div_ps(_mm256_set1_ps(1.0), _mm256_set1_ps(right));
    __m256 epsilon_vec = _mm256_set1_ps(epsilon);
    int rest_mask =
        ((-1) & (~((~0U) >> (sizeof(int) * 8 - (block - rest))))) & 0x0ff;
    __m256i mask_vec = _mm256_set_epi32(rest_mask & 0x80 ? 0xffffffff : 0,
        rest_mask & 0x40 ? 0xffffffff : 0,
        rest_mask & 0x20 ? 0xffffffff : 0,
        rest_mask & 0x10 ? 0xffffffff : 0,
        rest_mask & 0x8 ? 0xffffffff : 0,
        rest_mask & 0x4 ? 0xffffffff : 0,
        rest_mask & 0x2 ? 0xffffffff : 0,
        rest_mask & 0x1 ? 0xffffffff : 0);

    for (int i = 0; i < height; ++i) {
        offset = i * right;

        /* get mean */
        sum = _mm256_setzero_ps();
        for (j = offset; j < end + offset; j += block) {
            sum = _mm256_add_ps(sum, _mm256_loadu_ps((const float*)x + j));
        }
        if (rest != 0) {
            j = offset + right - block;
            tmp = _mm256_loadu_ps((const float*)x + j);
            tmp = _mm256_blendv_ps(_mm256_setzero_ps(),
                tmp,
                *(__m256*) & mask_vec);  // NOLINT
            sum = _mm256_add_ps(sum, tmp);
        }
        hi = _mm256_extractf128_ps(sum, 1);
        lo = _mm256_extractf128_ps(sum, 0);
        sum = _mm256_add_ps(
            sum,
            _mm256_insertf128_ps(
                _mm256_insertf128_ps(_mm256_setzero_ps(), hi, 0), lo, 1));
        sum = _mm256_hadd_ps(sum, sum);
        sum = _mm256_hadd_ps(sum, sum);
        mean_vec = _mm256_mul_ps(sum, reverse_num_vec);
        //mean[i] = *(float*)(&mean_vec);

        /* get variance */
        sum = _mm256_setzero_ps();
        for (j = offset; j < end + offset; j += block) {
            tmp = _mm256_sub_ps(_mm256_loadu_ps((const float*)x + j), mean_vec);
            tmp = _mm256_mul_ps(tmp, tmp);
            sum = _mm256_add_ps(sum, tmp);
        }
        if (rest != 0) {
            j = offset + right - block;
            tmp = _mm256_sub_ps(_mm256_loadu_ps((const float*)x + j), mean_vec);
            tmp = _mm256_mul_ps(tmp, tmp);
            tmp = _mm256_blendv_ps(_mm256_setzero_ps(),
                tmp,
                *(__m256*) & mask_vec);  // NOLINT
            sum = _mm256_add_ps(sum, tmp);
        }
        hi = _mm256_extractf128_ps(sum, 1);
        lo = _mm256_extractf128_ps(sum, 0);
        sum = _mm256_add_ps(
            sum,
            _mm256_insertf128_ps(
                _mm256_insertf128_ps(_mm256_setzero_ps(), hi, 0), lo, 1));
        sum = _mm256_hadd_ps(sum, sum);
        sum = _mm256_hadd_ps(sum, sum);
        var_vec = _mm256_mul_ps(sum, reverse_num_vec);
        //var[i] = *(float*)(&var_vec);

        /* get x_norm and calculate output*/
        for (j = offset; j < end + offset; j += block) {
            tmp = _mm256_sub_ps(_mm256_loadu_ps((const float*)x + j), mean_vec);
            tmp = _mm256_div_ps(tmp,
                _mm256_sqrt_ps(_mm256_add_ps(var_vec, epsilon_vec)));
            _mm256_storeu_ps((float*)(out) + j, tmp);
        }
        if (rest != 0) {
            j = offset + right - block;
            tmp = _mm256_sub_ps(_mm256_loadu_ps((const float*)x + j), mean_vec);
            tmp = _mm256_div_ps(tmp,
                _mm256_sqrt_ps(_mm256_add_ps(var_vec, epsilon_vec)));
            _mm256_storeu_ps((float*)(out) + j, tmp);
        }

        if (scale) {
            if (rest != 0) {
                j = offset + right - block;
                tmp = _mm256_loadu_ps((const float*)out + j);
            }
            for (j = offset; j < end + offset; j += block) {
                _mm256_storeu_ps(
                    (float*)(out) + j,
                    _mm256_mul_ps(_mm256_loadu_ps((const float*)out + j),
                        _mm256_loadu_ps((const float*)scale + j - offset)));
            }
            if (rest != 0) {
                j = offset + right - block;
                _mm256_storeu_ps(
                    (float*)(out) + j,
                    _mm256_mul_ps(tmp,
                        _mm256_loadu_ps((const float*)scale + j - offset)));
            }
        }

        if (bias) {
            if (rest != 0) {
                j = offset + right - block;
                tmp = _mm256_loadu_ps((const float*)out + j);
            }
            for (j = offset; j < end + offset; j += block) {
                _mm256_storeu_ps(
                    (float*)(out) + j,
                    _mm256_add_ps(_mm256_loadu_ps((const float*)out + j),
                        _mm256_loadu_ps((const float*)bias + j - offset)));
            }
            if (rest != 0) {
                j = offset + right - block;
                _mm256_storeu_ps(
                    (float*)(out) + j,
                    _mm256_add_ps(tmp,
                        _mm256_loadu_ps((const float*)bias + j - offset)));
            }
        }
    }
}
