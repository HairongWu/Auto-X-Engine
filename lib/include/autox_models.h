#pragma once

#include "autox_nn.h"
#include "ggml.h"

#ifdef  __cplusplus
extern "C" {
#endif

    void shufflenetv2_x_0_25(const float* image, void* weights, float* Out);
    void picodet_xs_320(const float* x, void* weights, float* dets, float* boxes);
    void tinypose_128x96(const float* image, void* weights, float* Out);
    void japan_PP_OCRv3_rec(const float* x, void* weights, float* Out);
    void multilingual_det(const float* x, void* weights, float* Out);
    autox_err_t run_llama3(char* checkpoint_path, char* tokenizer_path, char* prompt, int step);
    bool sam_image_preprocess(const uint8_t* img, float* res, const int nx, const int ny,
        const int nx2, const int ny2);
    autox_err_t autox_sam(float* img1, void* weights, int nx, int ny, float x, float y);
#ifdef  __cplusplus
}
#endif