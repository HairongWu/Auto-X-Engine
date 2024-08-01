#include "../include/autox_nn.h"

void autox_sigmoid(float *data, uint32_t size)
{
	for (uint32_t i = 0; i < size; i++)
		data[i] = 1.0 / (1.0 + expf(-data[i]));
}
