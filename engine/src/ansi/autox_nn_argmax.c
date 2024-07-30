#include "autox_nn_ansi.h"

#ifndef _HUGE_ENUF
    #define _HUGE_ENUF  1e+300  // _HUGE_ENUF*_HUGE_ENUF must overflow
#endif

#define INFINITY   ((float)(_HUGE_ENUF * _HUGE_ENUF))

// Computes the indices of the max elements of the input tensor's element along the provided axis. 
void autox_argmax_ansi(const float *input,
                 float *output,
                 const uint32_t *input_ddim,
                 const uint32_t *output_ddim,
                 const uint8_t input_ddim_size,
                 const uint8_t output_ddim_size,
                 const int8_t axis) {
    if (axis < 0) {
        axis += input_ddim_size;
    }
    const uint32_t size = input_ddim[axis];
    const uint32_t in_channel = count(input_ddim, axis, input_ddim_size);
    const uint32_t out_channel = count(output_ddim, axis, output_ddim_size);
    const uint32_t in_stride = count(input_ddim, axis + 1, input_ddim_size);
    const uint32_t out_stride = count(input_ddim, 0, axis);

    for (uint32_t n = 0; n < out_stride; n++) {
        for (uint32_t k = 0; k < in_stride; k++) {
            const float *in_ptr = input + n * in_channel + k;
            float first = in_ptr[0];
            float second = 0;
            for (uint32_t i = 1; i < size; i++) {
                if (in_ptr[i * in_stride] > first) {
                    first = in_ptr[i * in_stride];
                    second = i;
                }
            }
            // out
            float *out_ptr = output + n * out_channel + k;
            *out_ptr = second;
        }
    }
}
