#pragma once

#include "autox_nn.h"

#ifdef  __cplusplus
extern "C" {
#endif

void shufflenetv2_x_0_25(const float* image, void* weights, float* Out);
void picodet_xs_320(const float* x, void* weights, float* dets, float* boxes);
void tinypose_128x96(const float* image, void* weights, float* Out);
void japan_PP_OCRv3_rec(const float* x, void* weights, float* Out);
void multilingual_det(const float* x, void* weights, float* Out);

#ifdef  __cplusplus
}
#endif