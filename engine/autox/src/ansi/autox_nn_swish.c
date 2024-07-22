#include "autox_nn_ansi.h"
#include <float.h>

void autox_swish_ansi(float* x_data, uint32_t size, float beta)
{
	for (uint32_t i = 0; i < size; i++) {
		x_data[i] = x_data[i] / (1 + expf(-x_data[i] * beta));
	}
}
