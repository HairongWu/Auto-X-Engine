#include "../include/autox_nn.h"

void autox_hard_swish(float* x_data, uint16_t* dims, uint8_t dim_size, float threshold, float scale, float offset)
{
	uint32_t size = count(dims, 0, dim_size);
	for (uint32_t i = 0; i < size; i++) {
		x_data[i] =
			min(max(0.f, x_data[i] + offset), threshold) * x_data[i] /
			scale;
	}
}
