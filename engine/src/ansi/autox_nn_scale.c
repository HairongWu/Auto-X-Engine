#include "autox_nn_ansi.h"
#include <float.h>

void autox_scale_ansi(float *data, uint32_t size, float scale, float bias, int8_t bias_before)
{
	if (bias_before) bias *= scale;
	for (uint32_t i = 0; i < size; i++) {
		data[i] = data[i] * scale + bias;
	}
}