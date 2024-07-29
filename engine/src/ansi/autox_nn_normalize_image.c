#include "autox_nn_ansi.h"

void autox_normalize_image(uint8_t* p, float* out, uint16_t p_h, uint16_t p_w, uint8_t p_c, float scale, float* means, float* stds)
{
	float mean0 = means[0];
	float std0 = stds[0];
	float mean1 = means[1];
	float std1 = stds[1];
	float mean2 = means[2];
	float std2 = stds[2];

	float temp[3];
	for (int i = 0; i < p_h; i++) {
		for (int j = 0; j < p_w*p_c; j += p_c) {
			int index = j + i * p_w * p_c;
			temp[0] = (float)p[index];
			temp[1] = (float)p[index + 1];
			temp[2] = (float)p[index + 2];

			temp[0] = temp[0] * scale - mean0;
			temp[1] = temp[1] * scale - mean1;
			temp[2] = temp[2] * scale - mean2;

			temp[0] /= std0;
			temp[1] /= std1;
			temp[2] /= std2;

			memcpy(out + index, temp, 3 * sizeof(float));
		}
	}
}