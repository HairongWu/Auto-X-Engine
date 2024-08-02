#include "../include/autox_nn.h"

void autox_relu(float* x_data, uint32_t* dims, uint8_t dim_size)
{
	uint32_t size = count(dims, 0, dim_size);
	for (int i = 0; i < size; i++) {
		x_data[i] = x_data[i] > 0 ? x_data[i] : 0;
	}
}