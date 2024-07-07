#pragma once

#include <stdint.h>
#include <stdlib.h>

typedef __fp16 float16_t;

#define LITE_PARALLEL_BEGIN(index, tid, work_size) \
  for (int index = 0; index < (work_size); ++index) {
#define LITE_PARALLEL_END() }


void autox_relu_fp16_arm(const float16_t* din,
                         float16_t* dout,
                         int size,
                         int threads);
void autox_relu_fp32_arm(const float* din, float* dout, int size, int threads);

void autox_hardswish_fp16_arm(const float16_t* din,
                               float16_t* dout,
                               const int size,
                               const float threshold,
                               const float scale,
                               const float offset,
                               int threads);
void autox_hardswish_f32_arm(const float* din,
                           float* dout,
                           int size,
                           float threshold,
                           float scale,
                           float offset,
                           int threads);