#include "autox_nn_ansi.h"

// Calculates the cosine of the given input tensor, element-wise.
void autox_cos_ansi(float *x_data, uint32_t *x_dims) {
    for (uint32_t i = 0; i < x_dims.production(); i++) {
        x_data[i] = cosf(x_data[i]);
    }
}