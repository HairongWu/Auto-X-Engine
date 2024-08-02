#include "../include/autox_nn.h"

void autox_swish(float* x_data, uint32_t* dims, uint8_t dim_size, float beta)
{
	uint32_t size = count(dims, 0, dim_size);
	for (uint32_t i = 0; i < size; i++) {
		x_data[i] = x_data[i] / (1 + expf(-x_data[i] * beta));
	}
}
