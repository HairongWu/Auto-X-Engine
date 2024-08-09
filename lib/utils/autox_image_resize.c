#include "../include/autox_nn.h"

void autox_resize_nearest(const uint8_t* src_im, uint8_t* dst_im, uint16_t src_width, uint16_t src_height, uint8_t channels,
	uint16_t dst_width, uint16_t dst_height) {
	if (src_width == dst_width && src_height == dst_height) {
		memcpy(dst_im, src_im, src_width * src_height * channels * sizeof(uint8_t));
		// memcpy(dst_im, MemoryDevice(CPU), src_width * src_height * channels * sizeof(T),
		//        src_im, MemoryDevice(CPU), src_width * src_height * channels * sizeof(T));
		return;
	}

	float lfx_scl = (float)(src_width) / dst_width;
	float lfy_scl = (float)(src_height) / dst_height;
	float bias_x = lfx_scl / 2 - 0.5;
	float bias_y = lfy_scl / 2 - 0.5;

	for (int n_y_d = 0; n_y_d < dst_height; n_y_d++) {
		for (int n_x_d = 0; n_x_d < dst_width; n_x_d++) {
			float lf_x_s = lfx_scl * n_x_d + bias_x;
			float lf_y_s = lfy_scl * n_y_d + bias_y;

			int n_x_s = (int)(round(lf_x_s));
			int n_y_s = (int)(round(lf_y_s));

			n_x_s = n_x_s >= 0 ? n_x_s : 0;
			n_x_s = n_x_s < src_width - 1 ? n_x_s : src_width - 1;
			n_y_s = n_y_s >= 0 ? n_y_s : 0;
			n_y_s = n_y_s < src_height - 1 ? n_y_s : src_height - 1;

			for (int c = 0; c < channels; c++) {
				dst_im[(n_y_d * dst_width + n_x_d) * channels + c] = src_im[(n_y_s * src_width + n_x_s) * channels + c];
			}//end for c
		}
	}
}

void autox_resize_linear(const uint8_t* src_im, uint8_t* dst_im, uint16_t src_width, uint16_t src_height, uint8_t channels,
	uint16_t dst_width, uint16_t dst_height) {
	if (src_width == dst_width && src_height == dst_height) {
		memcpy(dst_im, src_im, src_width * src_height * channels * sizeof(uint8_t));
		// memcpy(dst_im, MemoryDevice(CPU), src_width * src_height * channels * sizeof(T),
		//        src_im, MemoryDevice(CPU), src_width * src_height * channels * sizeof(T));
		return;
	}

	float lfx_scl = (float)(src_width) / dst_width;
	float lfy_scl = (float)(src_height) / dst_height;
	float bias_x = lfx_scl / 2 - 0.5;
	float bias_y = lfy_scl / 2 - 0.5;

	for (int n_y_d = 0; n_y_d < dst_height; n_y_d++) {
		for (int n_x_d = 0; n_x_d < dst_width; n_x_d++) {
			float lf_x_s = lfx_scl * n_x_d + bias_x;
			float lf_y_s = lfy_scl * n_y_d + bias_y;

			lf_x_s = lf_x_s >= 0 ? lf_x_s : 0;
			lf_x_s = lf_x_s < src_width - 1 ? lf_x_s : src_width - 1 - 1e-5;
			lf_y_s = lf_y_s >= 0 ? lf_y_s : 0;
			lf_y_s = lf_y_s < src_height - 1 ? lf_y_s : src_height - 1 - 1e-5;

			int n_x_s = (int)(lf_x_s);
			int n_y_s = (int)(lf_y_s);

			double lf_weight_x = lf_x_s - n_x_s;
			double lf_weight_y = lf_y_s - n_y_s;

			for (int c = 0; c < channels; c++) {
				dst_im[(n_y_d * dst_width + n_x_d) * channels + c] =
					(uint8_t)((1 - lf_weight_y) * (1 - lf_weight_x) *
						src_im[(n_y_s * src_width + n_x_s) * channels + c] +
						(1 - lf_weight_y) * lf_weight_x *
						src_im[(n_y_s * src_width + n_x_s + 1) * channels + c] +
						lf_weight_y * (1 - lf_weight_x) *
						src_im[((n_y_s + 1) * src_width + n_x_s) * channels + c] +
						lf_weight_y * lf_weight_x *
						src_im[((n_y_s + 1) * src_width + n_x_s + 1) * channels + c]);

			}//end for c
		}
	}
}

void autox_resize_cubic(const uint8_t* src_im, uint8_t* dst_im, uint16_t src_width, uint16_t src_height, uint8_t channels,
	uint16_t dst_width, uint16_t dst_height) {
	float scale_x = (float)src_width / dst_width;
	float scale_y = (float)src_height / dst_height;

	int srcrows = src_width * channels;
	int dstrows = dst_width * channels;

	for (int j = 0; j < dst_height; ++j) {
		float fy = (float)((j + 0.5) * scale_y - 0.5);
		int sy = (int)(floor(fy));
		fy -= sy;
		//sy = std::min(sy, src_height - 3);
		//sy = std::max(1, sy);
		if (sy < 1) {
			fy = 0;
			sy = 1;
		}

		if (sy >= src_height - 3) {
			fy = 0, sy = src_height - 3;
		}

		const float A = -0.75f;

		float coeffsY[4];
		coeffsY[0] = ((A * (fy + 1) - 5 * A) * (fy + 1) + 8 * A) * (fy + 1) - 4 * A;
		coeffsY[1] = ((A + 2) * fy - (A + 3)) * fy * fy + 1;
		coeffsY[2] = ((A + 2) * (1 - fy) - (A + 3)) * (1 - fy) * (1 - fy) + 1;
		coeffsY[3] = 1.f - coeffsY[0] - coeffsY[1] - coeffsY[2];

		for (int i = 0; i < dst_width; ++i) {
			float fx = (float)((i + 0.5) * scale_x - 0.5);
			int sx = (int)(floor(fx));
			fx -= sx;

			if (sx < 1) {
				fx = 0, sx = 1;
			}
			if (sx >= src_width - 3) {
				fx = 0, sx = src_width - 3;
			}

			float coeffsX[4];
			coeffsX[0] = ((A * (fx + 1) - 5 * A) * (fx + 1) + 8 * A) * (fx + 1) - 4 * A;
			coeffsX[1] = ((A + 2) * fx - (A + 3)) * fx * fx + 1;
			coeffsX[2] = ((A + 2) * (1 - fx) - (A + 3)) * (1 - fx) * (1 - fx) + 1;
			coeffsX[3] = 1.f - coeffsX[0] - coeffsX[1] - coeffsX[2];

			for (int k = 0; k < channels; ++k) {
				dst_im[j * dstrows + i * channels + k] = (uint8_t)((
					src_im[(sy - 1) * srcrows + (sx - 1) * channels + k] * coeffsX[0] * coeffsY[0] +
					src_im[(sy)*srcrows + (sx - 1) * channels + k] * coeffsX[0] * coeffsY[1] +
					src_im[(sy + 1) * srcrows + (sx - 1) * channels + k] * coeffsX[0] * coeffsY[2] +
					src_im[(sy + 2) * srcrows + (sx - 1) * channels + k] * coeffsX[0] * coeffsY[3] +

					src_im[(sy - 1) * srcrows + (sx)*channels + k] * coeffsX[1] * coeffsY[0] +
					src_im[(sy)*srcrows + (sx)*channels + k] * coeffsX[1] * coeffsY[1] +
					src_im[(sy + 1) * srcrows + (sx)*channels + k] * coeffsX[1] * coeffsY[2] +
					src_im[(sy + 2) * srcrows + (sx)*channels + k] * coeffsX[1] * coeffsY[3] +

					src_im[(sy - 1) * srcrows + (sx + 1) * channels + k] * coeffsX[2] * coeffsY[0] +
					src_im[(sy)*srcrows + (sx + 1) * channels + k] * coeffsX[2] * coeffsY[1] +
					src_im[(sy + 1) * srcrows + (sx + 1) * channels + k] * coeffsX[2] * coeffsY[2] +
					src_im[(sy + 2) * srcrows + (sx + 1) * channels + k] * coeffsX[2] * coeffsY[3] +

					src_im[(sy - 1) * srcrows + (sx + 2) * channels + k] * coeffsX[3] * coeffsY[0] +
					src_im[(sy)*srcrows + (sx + 2) * channels + k] * coeffsX[3] * coeffsY[1] +
					src_im[(sy + 1) * srcrows + (sx + 2) * channels + k] * coeffsX[3] * coeffsY[2] +
					src_im[(sy + 2) * srcrows + (sx + 2) * channels + k] * coeffsX[3] * coeffsY[3]));

			}//end k
		}
	}
}

void autox_resize_type0(const uint8_t* src_im, uint8_t* dst_im, uint16_t src_width, uint16_t src_height, uint8_t channels,
	uint16_t limit_side_len, const char* limit_type) {
	float ratio;
	if (limit_type == "max")
		if (max(src_height, src_width) > limit_side_len)
			if (src_height > src_width)
				ratio = (float)(limit_side_len) / src_height;
			else
				ratio = (float)(limit_side_len) / src_width;
		else
			ratio = 1.;
	else if (limit_type == "min")
		if (min(src_height, src_width) < limit_side_len)
			if (src_height < src_width)
				ratio = (float)(limit_side_len) / src_height;
			else
				ratio = (float)(limit_side_len) / src_width;
		else
			ratio = 1.;
	else if (limit_type == "resize_long")
		ratio = (float)(limit_side_len) / max(src_height, src_width);

	int resize_h = (int)(src_height * ratio);
	int resize_w = (int)(src_width * ratio);

	resize_h = max((int)(round(resize_h / 32) * 32), 32);
	resize_w = max((int)(round(resize_w / 32) * 32), 32);

	autox_resize_linear(src_im, dst_im, src_width, src_height, channels, resize_w, resize_h);

	//ratio_h = resize_h / float(h);
	//ratio_w = resize_w / float(w);
}