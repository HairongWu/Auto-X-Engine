#include "autox_nn_ansi.h"

static int8_t is_a_ge_zero_and_a_lt_b(int a, int b) {
	return (unsigned)(a) < (unsigned)(b);
}

void autox_im2col_ansi(const float* data_im,
	int channels,
	int height,
	int width,
	int kernel_h,
	int kernel_w,
	int pad_top,
	int pad_bottom,
	int pad_left,
	int pad_right,
	int stride_h,
	int stride_w,
	int dilation_h,
	int dilation_w,
	float* data_col) {
	const int output_h =
		(height + pad_top + pad_bottom - (dilation_h * (kernel_h - 1) + 1)) /
		stride_h +
		1;
	const int output_w =
		(width + pad_left + pad_right - (dilation_w * (kernel_w - 1) + 1)) /
		stride_w +
		1;
	const int channel_size = height * width;
	for (int channel = channels; channel--; data_im += channel_size) {
		for (int kernel_row = 0; kernel_row < kernel_h; kernel_row++) {
			for (int kernel_col = 0; kernel_col < kernel_w; kernel_col++) {
				int input_row = -pad_top + kernel_row * dilation_h;
				for (int output_rows = output_h; output_rows; output_rows--) {
					if (!is_a_ge_zero_and_a_lt_b(input_row, height)) {
						for (int output_cols = output_w; output_cols; output_cols--) {
							*(data_col++) = 0;
						}
					}
					else {
						int input_col = -pad_left + kernel_col * dilation_w;
						for (int output_col = output_w; output_col; output_col--) {
							if (is_a_ge_zero_and_a_lt_b(input_col, width)) {
								*(data_col++) = data_im[input_row * width + input_col];
							}
							else {
								*(data_col++) = 0;
							}
							input_col += stride_w;
						}
					}
					input_row += stride_h;
				}
			}
		}
	}
}