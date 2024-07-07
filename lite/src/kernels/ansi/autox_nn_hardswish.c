#include "autox_nn_ansi.h"

void autox_hardswish_ansi(float* x_data,
	const uint32_t size, float threshold, float scale, float offset)
{
	for (uint32_t i = 0; i < size; i++) {
		x_data[i] =
			min(max(0.f, x_data[i] + offset), threshold) * x_data[i] /
			scale;
	}
}
