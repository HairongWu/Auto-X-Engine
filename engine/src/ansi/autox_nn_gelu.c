#include "autox_nn_ansi.h"

static const float GELU_COEF_A     = 0.044715f;
static const float GELU_QUICK_COEF = -1.702f;
static const float SQRT_2_OVER_PI  = 0.79788456080286535587989211986876f;

void autox_gelu_ansi(float *x, uint32_t size) {
	for (uint32_t i = 0; i < size; i++) {
		x[i] = 0.5f*x[i]*(1.0f + tanhf(SQRT_2_OVER_PI*x[i]*(1.0f + GELU_COEF_A*x[i]*x[i])));
	}
}
