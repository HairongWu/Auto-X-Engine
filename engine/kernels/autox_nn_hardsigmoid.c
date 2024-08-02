#include "../include/autox_nn.h"

void autox_hard_sigmoid(float *data, uint32_t* dims, uint8_t dim_size, float offset, float slope)
{
	uint32_t size = count(dims, 0, dim_size);
	for (uint32_t i = 0; i < size; i++) {
		float tmp = data[i] * slope + offset;
		tmp = tmp < 1.0f ? tmp : 1.0f;
		tmp = tmp > 0.0f ? tmp : 0.0f;
		data[i] = tmp;
	}
}
