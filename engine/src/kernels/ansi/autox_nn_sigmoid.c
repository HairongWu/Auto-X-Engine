#include "autox_nn_ansi.h"
#include <float.h>

void autox_sigmoid_ansi(float *data, uint32_t size)
{
	for (uint32_t i = 0; i < size; i++)
		data[i] = 1.0 / (1.0 + expf(-data[i]));
}
