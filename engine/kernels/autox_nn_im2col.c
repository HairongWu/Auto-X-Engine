#include "../include/autox_nn.h"

static int8_t is_a_ge_zero_and_a_lt_b(int a, int b) {
	return (unsigned)(a) < (unsigned)(b);
}

static void im2col_common(const float* data_im,
	int channels,
	int height,
	int width,
	int kernel,
	int pad,
	int stride,
	int dilation,
	float* data_col) {
	const int output_h =
		(height + pad + pad - (dilation * (kernel - 1) + 1)) /
		stride +
		1;
	const int output_w =
		(width + pad + pad - (dilation * (kernel - 1) + 1)) /
		stride +
		1;
	const int channel_size = height * width;
	for (int channel = channels; channel--; data_im += channel_size) {
		for (int kernel_row = 0; kernel_row < kernel; kernel_row++) {
			for (int kernel_col = 0; kernel_col < kernel; kernel_col++) {
				int input_row = -pad + kernel_row * dilation;
				for (int output_rows = output_h; output_rows; output_rows--) {
					if (!is_a_ge_zero_and_a_lt_b(input_row, height)) {
						for (int output_cols = output_w; output_cols; output_cols--) {
							*(data_col++) = 0;
						}
					}
					else {
						int input_col = -pad + kernel_col * dilation;
						for (int output_col = output_w; output_col; output_col--) {
							if (is_a_ge_zero_and_a_lt_b(input_col, width)) {
								*(data_col++) = data_im[input_row * width + input_col];
							}
							else {
								*(data_col++) = 0;
							}
							input_col += stride;
						}
					}
					input_row += stride;
				}
			}
		}
	}
}

static void im2col_s1(const float* data_im,
	int channels,
	int height,
	int width,
	int kernel,
	int pad,
	int dilation,
	float* data_col) {
	const int output_h =
		(height + pad + pad - (dilation * (kernel - 1) + 1)) + 1;
	const int output_w =
		(width + pad + pad - (dilation * (kernel - 1) + 1)) + 1;
	const int in_channel_size = height * width;
	const int out_channel_size = output_h * output_w;
	const unsigned int output_plane_size =
		output_h * output_w * kernel * kernel;
	size_t tmp_size = (size_t)(output_plane_size);
	size_t mem_size = tmp_size * channels * sizeof(float);
	memset(data_col, 0, mem_size);

	for (int c = 0; c < channels; c++) {
		unsigned int data_im_z = (unsigned int)(c * in_channel_size);
		int data_col_z1 = c * output_plane_size;
		for (int ky = 0, h_offset = 0; ky < kernel;
			ky++, h_offset += dilation) {
			int data_col_z2 = ky * out_channel_size * kernel;
			for (int kx = 0, w_offset = 0; kx < kernel;
				kx++, w_offset += dilation) {
				int data_col_z3 = kx * out_channel_size;
				unsigned int data_col_z =
					(unsigned int)(data_col_z1 + data_col_z2 + data_col_z3);
				int oh_begin = max(((pad - h_offset)), 0);  // always >= 0
				int oh_end = min(((height + pad - h_offset)), output_h);
				oh_end = max(oh_begin, oh_end);
				int ow_begin = max(((pad - w_offset)), 0);
				int ow_end = min(((width + pad - w_offset)), output_w);
				ow_end = max(ow_begin, ow_end);
				int ih = oh_begin - pad + h_offset;
				for (int oh = oh_begin; oh < oh_end; ++oh, ++ih) {
					int iw = ow_begin - pad + w_offset;
					int ow = ow_begin;
					unsigned int data_im_offset = data_im_z + ih * width;
					unsigned int data_col_offset = data_col_z + oh * output_w;
					const float* data_im_ptr = data_im + data_im_offset;
					float* data_col_ptr = data_col + data_col_offset;

					for (; ow + 7 < ow_end; ow += 8, iw += 8) {
						__m256 vtmp = _mm256_loadu_ps(data_im_ptr + iw);
						_mm256_storeu_ps(data_col_ptr + ow, vtmp);
					}

					for (; ow < ow_end; ++ow, ++iw) {
						data_col[data_col_offset + ow] = data_im[data_im_offset + iw];
					}
				}
			}
		}
	}
}

static void im2col_s2(const float* data_im,
	int channels,
	int height,
	int width,
	int kernel,
	int pad,
	int dilation,
	float* data_col) {
	const int output_h =
		(height + pad + pad - (dilation * (kernel - 1) + 1)) / 2 +
		1;
	const int output_w =
		(width + pad + pad - (dilation * (kernel - 1) + 1)) / 2 +
		1;
	const int in_channel_size = height * width;
	const unsigned int output_plane_size =
		output_h * output_w * kernel * kernel;
	size_t tmp_size = (size_t)(output_plane_size);
	size_t mem_size = tmp_size * channels * sizeof(float);
	memset(data_col, 0, mem_size);

	for (int c = 0; c < channels; c++) {
		unsigned int data_im_z = (unsigned int)(c * in_channel_size);
		int data_col_z1 = c * output_plane_size;
		for (int ky = 0, h_offset = 0; ky < kernel;
			ky++, h_offset += dilation) {
			int data_col_z2 = ky * output_h * output_w * kernel;
			for (int kx = 0, w_offset = 0; kx < kernel;
				kx++, w_offset += dilation) {
				int data_col_z3 = kx * output_h * output_w;
				unsigned int data_col_z =
					(unsigned int)(data_col_z1 + data_col_z2 + data_col_z3);
				int oh_begin = max(((pad - h_offset + 1) / 2), 0);
				int oh_end =
					min(((height + pad - h_offset + 1) / 2), output_h);
				oh_end = max(oh_begin, oh_end);
				int ow_begin = max(((pad - w_offset + 1) / 2), 0);
				int ow_end =
					min(((width + pad - w_offset + 1) / 2), output_w);
				ow_end = max(ow_begin, ow_end);
				int ih = oh_begin * 2 - pad + h_offset;
				for (int oh = oh_begin; oh < oh_end; ++oh, ih += 2) {
					int iw = ow_begin * 2 - pad + w_offset;
					int ow = ow_begin;
					unsigned int data_im_offset = data_im_z + ih * width;
					unsigned int data_col_offset = data_col_z + oh * output_w;
					const float* data_im_ptr = data_im + data_im_offset;
					float* data_col_ptr = data_col + data_col_offset;
					for (; ow + 3 < ow_end; ow += 4, iw += 8) {
						// a0a1a2a3
						__m128 vtmp0 = _mm_loadu_ps(data_im_ptr + iw);
						// a4a5a6a7
						__m128 vtmp1 = _mm_loadu_ps(data_im_ptr + iw + 4);
						// a0a2a4a6
						_mm_storeu_ps(data_col_ptr + ow,
							_mm_shuffle_ps(vtmp0, vtmp1, 0x88));
					}
					for (; ow < ow_end; ++ow, iw += 2) {
						data_col[data_col_offset + ow] = data_im[data_im_offset + iw];
					}
				}
			}
		}
	}
}

void autox_im2col(const float* data_im,
	int channels,
	int height,
	int width,
	int kernel,
	int pad,
	int stride,
	int dilation,
	float* data_col) {

	if (dilation == 1 && stride == 1) {
		im2col_s1(data_im,
			channels,
			height,
			width,
			kernel,
			pad,
			dilation,
			data_col);
	}
	else if (dilation == 1 && stride == 2) {
		im2col_s2(data_im,
			channels,
			height,
			width,
			kernel,
			pad,
			dilation,
			data_col);
	}
	else {
		im2col_common(data_im,
			channels,
			height,
			width,
			kernel,
			pad,
			stride,
			dilation,
			data_col);
	}
}