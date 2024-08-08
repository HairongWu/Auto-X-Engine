#include "../include/autox_nn.h"

void autox_normalize_image(uint8_t* p, float* out, uint32_t p_h, uint32_t p_w, uint8_t p_c)
{
  float scale = 1.0/255.0;
	const float means[3] = { 0.485, 0.456, 0.406 };
    const float stds[3] = { 0.229, 0.224, 0.225 };

	float temp[3];
	for (uint32_t i = 0; i < p_h; i++) {
		for (uint32_t j = 0; j < p_w*p_c; j += p_c) {
			uint32_t index = j + i * p_w * p_c;
			temp[0] = (float)p[index];
			temp[1] = (float)p[index + 1];
			temp[2] = (float)p[index + 2];

			temp[0] = temp[0] * scale - means[0];
			temp[1] = temp[1] * scale - means[1];
			temp[2] = temp[2] * scale - means[2];

			temp[0] /= stds[0];
			temp[1] /= stds[1];
			temp[2] /= stds[2];

			memcpy(out + index, temp, 3 * sizeof(float));
		}
	}
}