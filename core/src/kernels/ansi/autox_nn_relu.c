#include "autox_nn_ansi.h"

void autox_relu_ansi(float* x_data, uint32_t size, float beta)
{
	for (int i = 0; i < size; i++) {
		x_data[i] = x_data[i] > 0 ? x_data[i] : 0;
	}
}