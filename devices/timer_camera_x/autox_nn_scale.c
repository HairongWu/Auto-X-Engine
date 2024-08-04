#include "autox_nn.h"

void autox_scale(float *data, uint16_t* dims, uint8_t dim_size, float bias, int8_t bias_before, float scale)
{
	uint32_t size = count(dims, 0, dim_size);
	if (bias_before) bias *= scale;
	for (uint32_t i = 0; i < size; i++) {
		data[i] = data[i] * scale + bias;
	}
}