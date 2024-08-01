#include "../include/autox_nn.h"

void autox_rmsnorm(float* o, float* x, float* weight, uint32_t size) {
    // calculate sum of squares
    float ss = 0.0f;
    for (uint32_t j = 0; j < size; j++) {
        ss += x[j] * x[j];
    }
    ss /= size;
    ss += 1e-5f;
    ss = 1.0f / sqrtf(ss);
    // normalize and scale
    for (uint32_t j = 0; j < size; j++) {
        o[j] = weight[j] * (ss * x[j]);
    }
}