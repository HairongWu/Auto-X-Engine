#include "../include/autox_nn.h"

void autox_relu(float* x_data, uint16_t* dims, uint8_t dim_size)
{
	uint32_t size = count(dims, 0, dim_size);
	for (uint32_t i = 0; i < size; i++) {
		x_data[i] = x_data[i] > 0 ? x_data[i] : 0;
	}
}

void autox_relu_noreplace(float* x_data, float* y_data, uint16_t* dims, uint8_t dim_size)
{
	uint32_t size = count(dims, 0, dim_size);
	for (uint32_t i = 0; i < size; i++) {
		y_data[i] = x_data[i] > 0 ? x_data[i] : 0;
	}
}