#include "../include/autox_nn.h"


void autox_slice(float *x_data, float *o_data[], uint16_t* in_dims, uint16_t* out_dims, uint8_t in_dim_size, uint8_t out_dim_size) 
{

    uint16_t slices = in_dims[0];
    uint32_t size = count(out_dims, 0, out_dim_size);
    for (uint16_t i = 0; i < slices; i++)
    {
        o_data[i] = x_data + i*size;
    }
}
