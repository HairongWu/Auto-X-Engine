#include "autox_nn_ansi.h"

void autox_sin_ansi(float *x_data, uint32_t *x_dims) {
    for (uint32_t i = 0; i < x_dims.production(); i++) {
        x_data[i] = sinf(x_data[i]);
    }
}
