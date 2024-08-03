#include "../include/autox_nn.h"


void autox_slice(float *x_data, float *o_data, uint16_t *in_dims, uint16_t *out_dims, 
        int8_t axis, int8_t decrease_axis, int8_t end, int8_t start) {

    uint16_t extent = out_dims[0];

    if (start < 0) {
      start = (start + in_dims[axis]);
    }
    start = max(start, 0);
    uint16_t offset = start;

    o_data = slice(x_data, offset, extent);

}
