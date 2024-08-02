#include "../include/autox_nn.h"

void autox_sigmoid(float *data, uint32_t* dims, uint8_t dim_size)
{
	uint32_t size = count(dims, 0, dim_size);
	for (uint32_t i = 0; i < size; i++)
		data[i] = 1.0 / (1.0 + expf(-data[i]));
}
