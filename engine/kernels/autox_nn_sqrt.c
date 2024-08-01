#include "../include/autox_nn.h"

void autox_sqrt(float *data, uint32_t size)
{
	for (uint32_t i = 0; i < size; i++)
		data[i] = sqrtf(data[i]);
}
