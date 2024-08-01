#pragma once

#include "autox_nn.h"

#ifdef  __cplusplus
extern "C" {
#endif

void shufflenetv2_x_0_25(const uint8_t* image, const uint16_t ssize_h, const uint16_t ssize_w, float* weights, float* Out);
void picodet_xs_320(const uint8_t* image, const uint16_t ssize_h, const uint16_t ssize_w, float* weights, float* Out);
void tinypose_128x96(const uint8_t* image, const uint16_t ssize_h, const uint16_t ssize_w, float* weights, float* Out);
void japan_PP_OCRv3_rec(const uint8_t* image, const uint16_t ssize_h, const uint16_t ssize_w, float* weights, float* Out);
autox_err_t run_llama3(char* checkpoint_path, char* tokenizer_path, char* prompt);

#ifdef  __cplusplus
}
#endif