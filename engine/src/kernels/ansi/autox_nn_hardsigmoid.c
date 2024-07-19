#include "autox_nn_ansi.h"
#include <float.h>

void autox_hardsigmoid_ansi(float *data, uint32_t size, float offset, float slope)
{
	for (uint32_t i = 0; i < size; i++) {
		float tmp = data[i] * slope + offset;
		tmp = tmp < 1.0f ? tmp : 1.0f;
		tmp = tmp > 0.0f ? tmp : 0.0f;
		data[i] = tmp;
	}
}
