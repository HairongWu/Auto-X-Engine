#include "autox_nn_ansi.h"

void autox_hardswish_x86(const float* din,
                float* dout,
                int size,
                float scale,
                float offset,
                float threshold) {
#ifdef __AVX__
  int cnt = size >> 5;
  int remain = size & 31;
  __m256 vec_zero = _mm256_set1_ps(0.f);
  __m256 vec_scale = _mm256_set1_ps(1.0 / scale);
  __m256 vec_threshold = _mm256_set1_ps(threshold);
  __m256 vec_offset = _mm256_set1_ps(offset);
#else
  int cnt = size >> 4;
  int remain = size & 15;
#endif
  __m128 vec_zero_128 = _mm_set1_ps(0.f);
  __m128 vec_scale_128 = _mm_set1_ps(1.0 / scale);
  __m128 vec_threshold_128 = _mm_set1_ps(threshold);
  __m128 vec_offset_128 = _mm_set1_ps(offset);
  int cnt_4 = remain >> 2;
  int rem_4 = remain & 3;
  for (int i = 0; i < cnt; i++) {
#ifdef __AVX__
    __m256 vin0 = _mm256_loadu_ps(din);
    __m256 vin1 = _mm256_loadu_ps(din + 8);
    __m256 vin2 = _mm256_loadu_ps(din + 16);
    __m256 vin3 = _mm256_loadu_ps(din + 24);
    __m256 vadd0 = _mm256_add_ps(vin0, vec_offset);
    __m256 vadd1 = _mm256_add_ps(vin1, vec_offset);
    __m256 vadd2 = _mm256_add_ps(vin2, vec_offset);
    __m256 vadd3 = _mm256_add_ps(vin3, vec_offset);
    __m256 vsum0 = _mm256_mul_ps(vin0, vec_scale);
    __m256 vsum1 = _mm256_mul_ps(vin1, vec_scale);
    __m256 vsum2 = _mm256_mul_ps(vin2, vec_scale);
    __m256 vsum3 = _mm256_mul_ps(vin3, vec_scale);
    __m256 vres0 = _mm256_min_ps(_mm256_max_ps(vadd0, vec_zero), vec_threshold);
    __m256 vres1 = _mm256_min_ps(_mm256_max_ps(vadd1, vec_zero), vec_threshold);
    __m256 vres2 = _mm256_min_ps(_mm256_max_ps(vadd2, vec_zero), vec_threshold);
    __m256 vres3 = _mm256_min_ps(_mm256_max_ps(vadd3, vec_zero), vec_threshold);
    _mm256_storeu_ps(dout, _mm256_mul_ps(vres0, vsum0));
    _mm256_storeu_ps(dout + 8, _mm256_mul_ps(vres1, vsum1));
    _mm256_storeu_ps(dout + 16, _mm256_mul_ps(vres2, vsum2));
    _mm256_storeu_ps(dout + 24, _mm256_mul_ps(vres3, vsum3));
    din += 32;
    dout += 32;
#else
    __m128 vin0 = _mm_loadu_ps(din);
    __m128 vin1 = _mm_loadu_ps(din + 4);
    __m128 vin2 = _mm_loadu_ps(din + 8);
    __m128 vin3 = _mm_loadu_ps(din + 12);
    __m128 vadd0 = _mm_add_ps(vin0, vec_offset_128);
    __m128 vadd1 = _mm_add_ps(vin1, vec_offset_128);
    __m128 vadd2 = _mm_add_ps(vin2, vec_offset_128);
    __m128 vadd3 = _mm_add_ps(vin3, vec_offset_128);
    __m128 vsum0 = _mm_mul_ps(vin0, vec_scale_128);
    __m128 vsum1 = _mm_mul_ps(vin1, vec_scale_128);
    __m128 vsum2 = _mm_mul_ps(vin2, vec_scale_128);
    __m128 vsum3 = _mm_mul_ps(vin3, vec_scale_128);
    __m128 vres0 =
        _mm_min_ps(_mm_max_ps(vadd0, vec_zero_128), vec_threshold_128);
    __m128 vres1 =
        _mm_min_ps(_mm_max_ps(vadd1, vec_zero_128), vec_threshold_128);
    __m128 vres2 =
        _mm_min_ps(_mm_max_ps(vadd2, vec_zero_128), vec_threshold_128);
    __m128 vres3 =
        _mm_min_ps(_mm_max_ps(vadd3, vec_zero_128), vec_threshold_128);
    _mm_storeu_ps(dout, _mm_mul_ps(vres0, vsum0));
    _mm_storeu_ps(dout + 4, _mm_mul_ps(vres1, vsum1));
    _mm_storeu_ps(dout + 8, _mm_mul_ps(vres2, vsum2));
    _mm_storeu_ps(dout + 12, _mm_mul_ps(vres3, vsum3));
    din += 16;
    dout += 16;
#endif
  }
  for (int i = 0; i < cnt_4; i++) {
    __m128 vin0 = _mm_loadu_ps(din);
    __m128 vadd0 = _mm_add_ps(vin0, vec_offset_128);
    __m128 vsum0 = _mm_mul_ps(vin0, vec_scale_128);
    __m128 vres0 =
        _mm_min_ps(_mm_max_ps(vadd0, vec_zero_128), vec_threshold_128);
    _mm_storeu_ps(dout, _mm_mul_ps(vres0, vsum0));
    din += 4;
    dout += 4;
  }
  for (int i = 0; i < rem_4; i++) {
    dout[0] =
        min(max(0.f, din[0] + offset), threshold) * din[0] / scale;
    dout++;
    din++;
  }
}