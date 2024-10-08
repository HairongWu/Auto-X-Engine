#pragma once

#include "autox_nn.h"

#ifdef  __cplusplus
extern "C" {
#endif

void shufflenetv2_x_0_25(const float* image, const void* weights, float* Out);
void picodet_xs_320(const float* x, const void* weights, float* dets, float* boxes);
// void tinypose_128x96(const float* image, void* weights, float* Out);
// void japan_PP_OCRv3_rec(const float* x, void* weights, float* Out);
// autox_err_t run_llama3(char* checkpoint_path, char* tokenizer_path, char* prompt);

#ifdef  __cplusplus
}
#endif