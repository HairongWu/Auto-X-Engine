#include "autox_nn_ansi.h"

void autox_normalize_image(uint8_t* p, float* out, uint16_t p_h, uint16_t p_w, uint8_t p_c, float scale)
{
	const float means[3] = { 123.675f, 116.280f, 103.530f };
    const float stds[3] = {  58.395f,  57.120f,  57.375f };

	float temp[3];
	for (int i = 0; i < p_h; i++) {
		for (int j = 0; j < p_w*p_c; j += p_c) {
			int index = j + i * p_w * p_c;
			temp[0] = (float)p[index];
			temp[1] = (float)p[index + 1];
			temp[2] = (float)p[index + 2];

			temp[0] = temp[0] * scale - mean[0];
			temp[1] = temp[1] * scale - mean[1];
			temp[2] = temp[2] * scale - mean[2];

			temp[0] /= stds[0];
			temp[1] /= stds[1];
			temp[2] /= stds[2];

			memcpy(out + index, temp, 3 * sizeof(float));
		}
	}
}