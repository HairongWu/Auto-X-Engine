#include "../include/autox_models.h"

void japan_PP_OCRv3_rec(const float* x, void* weights, float* Out)
{
	uint16_t x_dim[] = { 1, 3, 48, 1 };
	uint16_t batch_norm_0_b_0_dim[] = { 16 };
	uint16_t batch_norm_0_tmp_4_dim[] = { 1, 16, 24, 1 };
	uint16_t conv2d_0_w_0_dim[] = { 16, 3, 3, 3 };
	uint16_t batch_norm_1_b_0_dim[] = { 16 };
	uint16_t batch_norm_1_tmp_4_dim[] = { 1, 16, 24, 1 };
	uint16_t conv2d_1_w_0_dim[] = { 16, 1, 3, 3 };
	uint16_t batch_norm_2_b_0_dim[] = { 32 };
	uint16_t batch_norm_2_tmp_4_dim[] = { 1, 32, 24, 1 };
	uint16_t conv2d_2_w_0_dim[] = { 32, 16, 1, 1 };
	uint16_t batch_norm_3_b_0_dim[] = { 32 };
	uint16_t batch_norm_3_tmp_4_dim[] = { 1, 32, 24, 1 };
	uint16_t conv2d_3_w_0_dim[] = { 32, 1, 3, 3 };
	uint16_t batch_norm_4_b_0_dim[] = { 64 };
	uint16_t batch_norm_4_tmp_4_dim[] = { 1, 64, 24, 1 };
	uint16_t conv2d_4_w_0_dim[] = { 64, 32, 1, 1 };
	uint16_t batch_norm_5_b_0_dim[] = { 64 };
	uint16_t batch_norm_5_tmp_4_dim[] = { 1, 64, 24, 1 };
	uint16_t conv2d_5_w_0_dim[] = { 64, 1, 3, 3 };
	uint16_t batch_norm_6_b_0_dim[] = { 64 };
	uint16_t batch_norm_6_tmp_4_dim[] = { 1, 64, 24, 1 };
	uint16_t conv2d_6_w_0_dim[] = { 64, 64, 1, 1 };
	uint16_t batch_norm_7_b_0_dim[] = { 64 };
	uint16_t batch_norm_7_tmp_4_dim[] = { 1, 64, 12, 1 };
	uint16_t conv2d_7_w_0_dim[] = { 64, 1, 3, 3 };
	uint16_t batch_norm_8_b_0_dim[] = { 128 };
	uint16_t batch_norm_8_tmp_4_dim[] = { 1, 128, 12, 1 };
	uint16_t conv2d_8_w_0_dim[] = { 128, 64, 1, 1 };
	uint16_t batch_norm_9_b_0_dim[] = { 128 };
	uint16_t batch_norm_9_tmp_4_dim[] = { 1, 128, 12, 1 };
	uint16_t conv2d_9_w_0_dim[] = { 128, 1, 3, 3 };
	uint16_t batch_norm_10_b_0_dim[] = { 128 };
	uint16_t batch_norm_10_tmp_4_dim[] = { 1, 128, 12, 1 };
	uint16_t conv2d_10_w_0_dim[] = { 128, 128, 1, 1 };
	uint16_t batch_norm_11_b_0_dim[] = { 128 };
	uint16_t batch_norm_11_tmp_4_dim[] = { 1, 128, 6, 1 };
	uint16_t conv2d_11_w_0_dim[] = { 128, 1, 3, 3 };
	uint16_t batch_norm_12_b_0_dim[] = { 256 };
	uint16_t batch_norm_12_tmp_4_dim[] = { 1, 256, 6, 1 };
	uint16_t conv2d_12_w_0_dim[] = { 256, 128, 1, 1 };
	uint16_t batch_norm_13_b_0_dim[] = { 256 };
	uint16_t batch_norm_13_tmp_4_dim[] = { 1, 256, 6, 1 };
	uint16_t conv2d_13_w_0_dim[] = { 256, 1, 5, 5 };
	uint16_t batch_norm_14_b_0_dim[] = { 256 };
	uint16_t batch_norm_14_tmp_4_dim[] = { 1, 256, 6, 1 };
	uint16_t conv2d_14_w_0_dim[] = { 256, 256, 1, 1 };
	uint16_t batch_norm_15_b_0_dim[] = { 256 };
	uint16_t batch_norm_15_tmp_4_dim[] = { 1, 256, 6, 1 };
	uint16_t conv2d_15_w_0_dim[] = { 256, 1, 5, 5 };
	uint16_t batch_norm_16_b_0_dim[] = { 256 };
	uint16_t batch_norm_16_tmp_4_dim[] = { 1, 256, 6, 1 };
	uint16_t conv2d_16_w_0_dim[] = { 256, 256, 1, 1 };
	uint16_t batch_norm_17_b_0_dim[] = { 256 };
	uint16_t batch_norm_17_tmp_4_dim[] = { 1, 256, 6, 1 };
	uint16_t conv2d_17_w_0_dim[] = { 256, 1, 5, 5 };
	uint16_t batch_norm_18_b_0_dim[] = { 256 };
	uint16_t batch_norm_18_tmp_4_dim[] = { 1, 256, 6, 1 };
	uint16_t conv2d_18_w_0_dim[] = { 256, 256, 1, 1 };
	uint16_t batch_norm_19_b_0_dim[] = { 256 };
	uint16_t batch_norm_19_tmp_4_dim[] = { 1, 256, 6, 1 };
	uint16_t conv2d_19_w_0_dim[] = { 256, 1, 5, 5 };
	uint16_t batch_norm_20_b_0_dim[] = { 256 };
	uint16_t batch_norm_20_tmp_4_dim[] = { 1, 256, 6, 1 };
	uint16_t conv2d_20_w_0_dim[] = { 256, 256, 1, 1 };
	uint16_t batch_norm_21_b_0_dim[] = { 256 };
	uint16_t batch_norm_21_tmp_4_dim[] = { 1, 256, 6, 1 };
	uint16_t conv2d_21_w_0_dim[] = { 256, 1, 5, 5 };
	uint16_t batch_norm_22_b_0_dim[] = { 256 };
	uint16_t batch_norm_22_tmp_4_dim[] = { 1, 256, 6, 1 };
	uint16_t conv2d_22_w_0_dim[] = { 256, 256, 1, 1 };
	uint16_t batch_norm_23_b_0_dim[] = { 256 };
	uint16_t batch_norm_23_tmp_4_dim[] = { 1, 256, 3, 1 };
	uint16_t conv2d_23_w_0_dim[] = { 256, 1, 5, 5 };
	uint16_t pool2d_0_tmp_0_dim[] = { 1, 256, 1, 1 };
	uint16_t conv2d_24_b_0_dim[] = { 64 };
	uint16_t conv2d_24_w_0_dim[] = { 64, 256, 1, 1 };
	uint16_t relu_0_tmp_0_dim[] = { 1, 64, 1, 1 };
	uint16_t conv2d_25_b_0_dim[] = { 256 };
	uint16_t conv2d_25_w_0_dim[] = { 256, 64, 1, 1 };
	uint16_t conv2d_50_tmp_1_dim[] = { 1, 256, 1, 1 };
	uint16_t hardsigmoid_0_tmp_0_dim[] = { 1, 256, 1, 1 };
	uint16_t elementwise_mul_0_dim[] = { 1, 256, 3, 1 };
	uint16_t batch_norm_24_b_0_dim[] = { 512 };
	uint16_t batch_norm_24_tmp_4_dim[] = { 1, 512, 3, 1 };
	uint16_t conv2d_26_w_0_dim[] = { 512, 256, 1, 1 };
	uint16_t batch_norm_25_b_0_dim[] = { 512 };
	uint16_t batch_norm_25_tmp_4_dim[] = { 1, 512, 3, 1 };
	uint16_t conv2d_27_w_0_dim[] = { 512, 1, 5, 5 };
	uint16_t pool2d_1_tmp_0_dim[] = { 1, 512, 1, 1 };
	uint16_t conv2d_28_b_0_dim[] = { 128 };
	uint16_t conv2d_28_w_0_dim[] = { 128, 512, 1, 1 };
	uint16_t relu_1_tmp_0_dim[] = { 1, 128, 1, 1 };
	uint16_t conv2d_29_b_0_dim[] = { 512 };
	uint16_t conv2d_29_w_0_dim[] = { 512, 128, 1, 1 };
	uint16_t conv2d_53_tmp_1_dim[] = { 1, 512, 1, 1 };
	uint16_t hardsigmoid_1_tmp_0_dim[] = { 1, 512, 1, 1 };
	uint16_t elementwise_mul_1_dim[] = { 1, 512, 3, 1 };
	uint16_t batch_norm_26_b_0_dim[] = { 512 };
	uint16_t batch_norm_26_tmp_4_dim[] = { 1, 512, 3, 1 };
	uint16_t conv2d_30_w_0_dim[] = { 512, 512, 1, 1 };
	uint16_t pool2d_2_tmp_0_dim[] = { 1, 512, 1, 1 };
	uint16_t pool2d_2_tmp_0_clone_0_dim[] = { 1, 512, 1, 1 };
	uint16_t batch_norm2d_0_b_0_dim[] = { 64 };
	uint16_t batch_norm_27_tmp_2_dim[] = { 1, 64, 1, 1 };
	uint16_t conv2d_31_w_0_dim[] = { 64, 512, 3, 3 };
	uint16_t swish_7_tmp_0_dim[] = { 1, 64, 1, 1 };
	uint16_t batch_norm2d_1_b_0_dim[] = { 120 };
	uint16_t batch_norm_28_tmp_2_dim[] = { 1, 120, 1, 1 };
	uint16_t conv2d_32_w_0_dim[] = { 120, 64, 1, 1 };
	uint16_t swish_8_tmp_0_dim[] = { 1, 120, 1, 1 };
	uint16_t shape_0_tmp_0_dim[] = { 4 };
	uint16_t shape_0_tmp_0_slice_1_dim[] = { 1 };
	uint16_t flatten_0_tmp_0_dim[] = { 1, 120, 1 };
	uint16_t flatten_0_tmp_1_dim[] = { 0, 1, 120, 1, 1 };
	uint16_t transpose_0_tmp_0_dim[] = { 1, 1, 120 };
	uint16_t transpose_0_tmp_1_dim[] = { 0, 1, 120, 1 };
	uint16_t layer_norm_0_b_0_dim[] = { 120 };
	uint16_t layer_norm_0_w_0_dim[] = { 120 };
	uint16_t layer_norm_5_tmp_0_dim[] = { 1 };
	uint16_t layer_norm_5_tmp_1_dim[] = { 1 };
	uint16_t layer_norm_5_tmp_2_dim[] = { 1, 1, 120 };
	uint16_t linear_0_w_0_dim[] = { 120, 360 };
	uint16_t linear_13_tmp_0_dim[] = { 1, 1, 360 };
	uint16_t linear_0_b_0_dim[] = { 360 };
	uint16_t linear_13_tmp_1_dim[] = { 1, 1, 360 };
	uint16_t reshape2_0_tmp_0_dim[] = { 1, 1, 3, 8, 15 };
	uint16_t reshape2_0_tmp_1_dim[] = { 0, 1, 1, 360 };
	uint16_t transpose_1_tmp_0_dim[] = { 3, 1, 8, 1, 15 };
	uint16_t transpose_1_tmp_1_dim[] = { 0, 1, 1, 3, 8, 15 };
	uint16_t transpose_1_tmp_0_slice_0_dim[] = { 1, 8, 1, 15 };
	uint16_t tmp_0_dim[] = { 1, 8, 1, 15 };
	uint16_t transpose_1_tmp_0_slice_1_dim[] = { 1, 8, 1, 15 };
	uint16_t transpose_1_tmp_0_slice_2_dim[] = { 1, 8, 1, 15 };
	uint16_t transpose_2_tmp_0_dim[] = { 1, 8, 15, 1 };
	uint16_t transpose_2_tmp_1_dim[] = { 0, 1, 8, 1, 15 };
	uint16_t matmul_v2_0_tmp_0_dim[] = { 1, 8, 1, 1 };
	uint16_t dropout_7_tmp_0_dim[] = { 1, 8, 1, 1 };
	uint16_t matmul_v2_1_tmp_0_dim[] = { 1, 8, 1, 15 };
	uint16_t transpose_3_tmp_0_dim[] = { 1, 1, 8, 15 };
	uint16_t transpose_3_tmp_1_dim[] = { 0, 1, 8, 1, 15 };
	uint16_t reshape2_1_tmp_0_dim[] = { 1, 1, 120 };
	uint16_t reshape2_1_tmp_1_dim[] = { 0, 1, 1, 8, 15 };
	uint16_t linear_1_w_0_dim[] = { 120, 120 };
	uint16_t linear_14_tmp_0_dim[] = { 1, 1, 120 };
	uint16_t dropout_8_tmp_0_dim[] = { 1, 1, 120 };
	uint16_t linear_1_b_0_dim[] = { 120 };
	uint16_t tmp_1_dim[] = { 1, 1, 120 };
	uint16_t layer_norm_1_b_0_dim[] = { 120 };
	uint16_t layer_norm_1_w_0_dim[] = { 120 };
	uint16_t layer_norm_6_tmp_0_dim[] = { 1 };
	uint16_t layer_norm_6_tmp_1_dim[] = { 1 };
	uint16_t layer_norm_6_tmp_2_dim[] = { 1, 1, 120 };
	uint16_t linear_15_tmp_0_dim[] = { 1, 1, 240 };
	uint16_t linear_2_w_0_dim[] = { 120, 240 };
	uint16_t linear_15_tmp_1_dim[] = { 1, 1, 240 };
	uint16_t linear_2_b_0_dim[] = { 240 };
	uint16_t dropout_9_tmp_0_dim[] = { 1, 1, 240 };
	uint16_t linear_16_tmp_0_dim[] = { 1, 1, 120 };
	uint16_t linear_3_w_0_dim[] = { 240, 120 };
	uint16_t dropout_10_tmp_0_dim[] = { 1, 1, 120 };
	uint16_t linear_3_b_0_dim[] = { 120 };
	uint16_t tmp_2_dim[] = { 1, 1, 120 };
	uint16_t layer_norm_2_b_0_dim[] = { 120 };
	uint16_t layer_norm_2_w_0_dim[] = { 120 };
	uint16_t layer_norm_7_tmp_0_dim[] = { 1 };
	uint16_t layer_norm_7_tmp_1_dim[] = { 1 };
	uint16_t layer_norm_7_tmp_2_dim[] = { 1, 1, 120 };
	uint16_t linear_17_tmp_0_dim[] = { 1, 1, 360 };
	uint16_t linear_4_w_0_dim[] = { 120, 360 };
	uint16_t linear_17_tmp_1_dim[] = { 1, 1, 360 };
	uint16_t linear_4_b_0_dim[] = { 360 };
	uint16_t reshape2_2_tmp_0_dim[] = { 1, 1, 3, 8, 15 };
	uint16_t reshape2_2_tmp_1_dim[] = { 0, 1, 1, 360 };
	uint16_t transpose_4_tmp_0_dim[] = { 3, 1, 8, 1, 15 };
	uint16_t transpose_4_tmp_1_dim[] = { 0, 1, 1, 3, 8, 15 };
	uint16_t transpose_4_tmp_0_slice_0_dim[] = { 1, 8, 1, 15 };
	uint16_t tmp_3_dim[] = { 1, 8, 1, 15 };
	uint16_t transpose_4_tmp_0_slice_1_dim[] = { 1, 8, 1, 15 };
	uint16_t transpose_4_tmp_0_slice_2_dim[] = { 1, 8, 1, 15 };
	uint16_t transpose_5_tmp_0_dim[] = { 1, 8, 15, 1 };
	uint16_t transpose_5_tmp_1_dim[] = { 0, 1, 8, 1, 15 };
	uint16_t matmul_v2_2_tmp_0_dim[] = { 1, 8, 1, 1 };
	uint16_t dropout_11_tmp_0_dim[] = { 1, 8, 1, 1 };
	uint16_t matmul_v2_3_tmp_0_dim[] = { 1, 8, 1, 15 };
	uint16_t transpose_6_tmp_0_dim[] = { 1, 1, 8, 15 };
	uint16_t transpose_6_tmp_1_dim[] = { 0, 1, 8, 1, 15 };
	uint16_t reshape2_3_tmp_0_dim[] = { 1, 1, 120 };
	uint16_t reshape2_3_tmp_1_dim[] = { 0, 1, 1, 8, 15 };
	uint16_t linear_18_tmp_0_dim[] = { 1, 1, 120 };
	uint16_t linear_5_w_0_dim[] = { 120, 120 };
	uint16_t dropout_12_tmp_0_dim[] = { 1, 1, 120 };
	uint16_t linear_5_b_0_dim[] = { 120 };
	uint16_t tmp_4_dim[] = { 1, 1, 120 };
	uint16_t layer_norm_3_b_0_dim[] = { 120 };
	uint16_t layer_norm_3_w_0_dim[] = { 120 };
	uint16_t layer_norm_8_tmp_0_dim[] = { 1 };
	uint16_t layer_norm_8_tmp_1_dim[] = { 1 };
	uint16_t layer_norm_8_tmp_2_dim[] = { 1, 1, 120 };
	uint16_t linear_19_tmp_0_dim[] = { 1, 1, 240 };
	uint16_t linear_6_w_0_dim[] = { 120, 240 };
	uint16_t linear_19_tmp_1_dim[] = { 1, 1, 240 };
	uint16_t linear_6_b_0_dim[] = { 240 };
	uint16_t dropout_13_tmp_0_dim[] = { 1, 1, 240 };
	uint16_t linear_20_tmp_0_dim[] = { 1, 1, 120 };
	uint16_t linear_7_w_0_dim[] = { 240, 120 };
	uint16_t dropout_14_tmp_0_dim[] = { 1, 1, 120 };
	uint16_t linear_7_b_0_dim[] = { 120 };
	uint16_t tmp_5_dim[] = { 1, 1, 120 };
	uint16_t layer_norm_4_b_0_dim[] = { 120 };
	uint16_t layer_norm_4_w_0_dim[] = { 120 };
	uint16_t layer_norm_9_tmp_0_dim[] = { 1 };
	uint16_t layer_norm_9_tmp_1_dim[] = { 1 };
	uint16_t layer_norm_9_tmp_2_dim[] = { 1, 1, 120 };
	uint16_t reshape2_4_tmp_0_dim[] = { 1, 1, 1, 120 };
	uint16_t reshape2_4_tmp_1_dim[] = { 0, 1, 1, 120 };
	uint16_t transpose_7_tmp_0_dim[] = { 1, 120, 1, 1 };
	uint16_t transpose_7_tmp_1_dim[] = { 0, 1, 1, 1, 120 };
	uint16_t batch_norm2d_2_b_0_dim[] = { 512 };
	uint16_t batch_norm_29_tmp_2_dim[] = { 1, 512, 1, 1 };
	uint16_t conv2d_33_w_0_dim[] = { 512, 120, 1, 1 };
	uint16_t swish_11_tmp_0_dim[] = { 1, 512, 1, 1 };
	uint16_t concat_0_tmp_0_dim[] = { 1, 1024, 1, 1 };
	uint16_t batch_norm2d_3_b_0_dim[] = { 64 };
	uint16_t batch_norm_30_tmp_2_dim[] = { 1, 64, 1, 1 };
	uint16_t conv2d_34_w_0_dim[] = { 64, 1024, 3, 3 };
	uint16_t swish_12_tmp_0_dim[] = { 1, 64, 1, 1 };
	uint16_t batch_norm2d_4_b_0_dim[] = { 64 };
	uint16_t batch_norm_31_tmp_2_dim[] = { 1, 64, 1, 1 };
	uint16_t conv2d_35_w_0_dim[] = { 64, 64, 1, 1 };
	uint16_t swish_13_tmp_0_dim[] = { 1, 64, 1, 1 };
	uint16_t squeeze_0_tmp_0_dim[] = { 1, 64, 1 };
	uint16_t squeeze_0_tmp_1_dim[] = { 0, 1, 64, 1, 1 };
	uint16_t transpose_8_tmp_0_dim[] = { 1, 1, 64 };
	uint16_t transpose_8_tmp_1_dim[] = { 0, 1, 64, 1 };
	uint16_t linear_21_tmp_0_dim[] = { 1, 1, 4401 };
	uint16_t linear_8_w_0_dim[] = { 64, 4401 };
	uint16_t linear_21_tmp_1_dim[] = { 1, 1, 4401 };
	uint16_t linear_8_b_0_dim[] = { 4401 };
	uint16_t softmax_2_tmp_0_dim[] = { 1, 1, 4401 };

	float* batch_norm_0_tmp_4 = (float*)calloc(384, sizeof(float));
	autox_conv2d(x, (float*)((int8_t*)weights) + 0, (float*)((int8_t*)weights) + 16, batch_norm_0_tmp_4, x_dim, conv2d_0_w_0_dim, batch_norm_0_tmp_4_dim, 1, 1, 2, 1, 10);
	free(x);

	float* batch_norm_1_tmp_4 = (float*)calloc(384, sizeof(float));
	autox_conv2d(batch_norm_0_tmp_4, (float*)((int8_t*)weights) + 448, (float*)((int8_t*)weights) + 464, batch_norm_1_tmp_4, batch_norm_0_tmp_4_dim, conv2d_1_w_0_dim, batch_norm_1_tmp_4_dim, 16, 1, 1, 1, 10);
	free(batch_norm_0_tmp_4);

	float* batch_norm_2_tmp_4 = (float*)calloc(768, sizeof(float));
	autox_conv2d(batch_norm_1_tmp_4, (float*)((int8_t*)weights) + 608, (float*)((int8_t*)weights) + 640, batch_norm_2_tmp_4, batch_norm_1_tmp_4_dim, conv2d_2_w_0_dim, batch_norm_2_tmp_4_dim, 1, 0, 1, 1, 10);
	free(batch_norm_1_tmp_4);

	float* batch_norm_3_tmp_4 = (float*)calloc(768, sizeof(float));
	autox_conv2d(batch_norm_2_tmp_4, (float*)((int8_t*)weights) + 1152, (float*)((int8_t*)weights) + 1184, batch_norm_3_tmp_4, batch_norm_2_tmp_4_dim, conv2d_3_w_0_dim, batch_norm_3_tmp_4_dim, 32, 1, 1, 1, 10);
	free(batch_norm_2_tmp_4);

	float* batch_norm_4_tmp_4 = (float*)calloc(1536, sizeof(float));
	autox_conv2d(batch_norm_3_tmp_4, (float*)((int8_t*)weights) + 1472, (float*)((int8_t*)weights) + 1536, batch_norm_4_tmp_4, batch_norm_3_tmp_4_dim, conv2d_4_w_0_dim, batch_norm_4_tmp_4_dim, 1, 0, 1, 1, 10);
	free(batch_norm_3_tmp_4);

	float* batch_norm_5_tmp_4 = (float*)calloc(1536, sizeof(float));
	autox_conv2d(batch_norm_4_tmp_4, (float*)((int8_t*)weights) + 3584, (float*)((int8_t*)weights) + 3648, batch_norm_5_tmp_4, batch_norm_4_tmp_4_dim, conv2d_5_w_0_dim, batch_norm_5_tmp_4_dim, 64, 1, 1, 1, 10);
	free(batch_norm_4_tmp_4);

	float* batch_norm_6_tmp_4 = (float*)calloc(1536, sizeof(float));
	autox_conv2d(batch_norm_5_tmp_4, (float*)((int8_t*)weights) + 4224, (float*)((int8_t*)weights) + 4288, batch_norm_6_tmp_4, batch_norm_5_tmp_4_dim, conv2d_6_w_0_dim, batch_norm_6_tmp_4_dim, 1, 0, 1, 1, 10);
	free(batch_norm_5_tmp_4);

	float* batch_norm_7_tmp_4 = (float*)calloc(768, sizeof(float));
	autox_conv2d(batch_norm_6_tmp_4, (float*)((int8_t*)weights) + 8384, (float*)((int8_t*)weights) + 8448, batch_norm_7_tmp_4, batch_norm_6_tmp_4_dim, conv2d_7_w_0_dim, batch_norm_7_tmp_4_dim, 64, 1, 2, 1, 10);
	free(batch_norm_6_tmp_4);

	float* batch_norm_8_tmp_4 = (float*)calloc(1536, sizeof(float));
	autox_conv2d(batch_norm_7_tmp_4, (float*)((int8_t*)weights) + 9024, (float*)((int8_t*)weights) + 9152, batch_norm_8_tmp_4, batch_norm_7_tmp_4_dim, conv2d_8_w_0_dim, batch_norm_8_tmp_4_dim, 1, 0, 1, 1, 10);
	free(batch_norm_7_tmp_4);

	float* batch_norm_9_tmp_4 = (float*)calloc(1536, sizeof(float));
	autox_conv2d(batch_norm_8_tmp_4, (float*)((int8_t*)weights) + 17344, (float*)((int8_t*)weights) + 17472, batch_norm_9_tmp_4, batch_norm_8_tmp_4_dim, conv2d_9_w_0_dim, batch_norm_9_tmp_4_dim, 128, 1, 1, 1, 10);
	free(batch_norm_8_tmp_4);

	float* batch_norm_10_tmp_4 = (float*)calloc(1536, sizeof(float));
	autox_conv2d(batch_norm_9_tmp_4, (float*)((int8_t*)weights) + 18624, (float*)((int8_t*)weights) + 18752, batch_norm_10_tmp_4, batch_norm_9_tmp_4_dim, conv2d_10_w_0_dim, batch_norm_10_tmp_4_dim, 1, 0, 1, 1, 10);
	free(batch_norm_9_tmp_4);

	float* batch_norm_11_tmp_4 = (float*)calloc(768, sizeof(float));
	autox_conv2d(batch_norm_10_tmp_4, (float*)((int8_t*)weights) + 35136, (float*)((int8_t*)weights) + 35264, batch_norm_11_tmp_4, batch_norm_10_tmp_4_dim, conv2d_11_w_0_dim, batch_norm_11_tmp_4_dim, 128, 1, 2, 1, 10);
	free(batch_norm_10_tmp_4);

	float* batch_norm_12_tmp_4 = (float*)calloc(1536, sizeof(float));
	autox_conv2d(batch_norm_11_tmp_4, (float*)((int8_t*)weights) + 36416, (float*)((int8_t*)weights) + 36672, batch_norm_12_tmp_4, batch_norm_11_tmp_4_dim, conv2d_12_w_0_dim, batch_norm_12_tmp_4_dim, 1, 0, 1, 1, 10);
	free(batch_norm_11_tmp_4);

	float* batch_norm_13_tmp_4 = (float*)calloc(1536, sizeof(float));
	autox_conv2d(batch_norm_12_tmp_4, (float*)((int8_t*)weights) + 69440, (float*)((int8_t*)weights) + 69696, batch_norm_13_tmp_4, batch_norm_12_tmp_4_dim, conv2d_13_w_0_dim, batch_norm_13_tmp_4_dim, 256, 2, 1, 1, 10);
	free(batch_norm_12_tmp_4);

	float* batch_norm_14_tmp_4 = (float*)calloc(1536, sizeof(float));
	autox_conv2d(batch_norm_13_tmp_4, (float*)((int8_t*)weights) + 76096, (float*)((int8_t*)weights) + 76352, batch_norm_14_tmp_4, batch_norm_13_tmp_4_dim, conv2d_14_w_0_dim, batch_norm_14_tmp_4_dim, 1, 0, 1, 1, 10);
	free(batch_norm_13_tmp_4);

	float* batch_norm_15_tmp_4 = (float*)calloc(1536, sizeof(float));
	autox_conv2d(batch_norm_14_tmp_4, (float*)((int8_t*)weights) + 141888, (float*)((int8_t*)weights) + 142144, batch_norm_15_tmp_4, batch_norm_14_tmp_4_dim, conv2d_15_w_0_dim, batch_norm_15_tmp_4_dim, 256, 2, 1, 1, 10);
	free(batch_norm_14_tmp_4);

	float* batch_norm_16_tmp_4 = (float*)calloc(1536, sizeof(float));
	autox_conv2d(batch_norm_15_tmp_4, (float*)((int8_t*)weights) + 148544, (float*)((int8_t*)weights) + 148800, batch_norm_16_tmp_4, batch_norm_15_tmp_4_dim, conv2d_16_w_0_dim, batch_norm_16_tmp_4_dim, 1, 0, 1, 1, 10);
	free(batch_norm_15_tmp_4);

	float* batch_norm_17_tmp_4 = (float*)calloc(1536, sizeof(float));
	autox_conv2d(batch_norm_16_tmp_4, (float*)((int8_t*)weights) + 214336, (float*)((int8_t*)weights) + 214592, batch_norm_17_tmp_4, batch_norm_16_tmp_4_dim, conv2d_17_w_0_dim, batch_norm_17_tmp_4_dim, 256, 2, 1, 1, 10);
	free(batch_norm_16_tmp_4);

	float* batch_norm_18_tmp_4 = (float*)calloc(1536, sizeof(float));
	autox_conv2d(batch_norm_17_tmp_4, (float*)((int8_t*)weights) + 220992, (float*)((int8_t*)weights) + 221248, batch_norm_18_tmp_4, batch_norm_17_tmp_4_dim, conv2d_18_w_0_dim, batch_norm_18_tmp_4_dim, 1, 0, 1, 1, 10);
	free(batch_norm_17_tmp_4);

	float* batch_norm_19_tmp_4 = (float*)calloc(1536, sizeof(float));
	autox_conv2d(batch_norm_18_tmp_4, (float*)((int8_t*)weights) + 286784, (float*)((int8_t*)weights) + 287040, batch_norm_19_tmp_4, batch_norm_18_tmp_4_dim, conv2d_19_w_0_dim, batch_norm_19_tmp_4_dim, 256, 2, 1, 1, 10);
	free(batch_norm_18_tmp_4);

	float* batch_norm_20_tmp_4 = (float*)calloc(1536, sizeof(float));
	autox_conv2d(batch_norm_19_tmp_4, (float*)((int8_t*)weights) + 293440, (float*)((int8_t*)weights) + 293696, batch_norm_20_tmp_4, batch_norm_19_tmp_4_dim, conv2d_20_w_0_dim, batch_norm_20_tmp_4_dim, 1, 0, 1, 1, 10);
	free(batch_norm_19_tmp_4);

	float* batch_norm_21_tmp_4 = (float*)calloc(1536, sizeof(float));
	autox_conv2d(batch_norm_20_tmp_4, (float*)((int8_t*)weights) + 359232, (float*)((int8_t*)weights) + 359488, batch_norm_21_tmp_4, batch_norm_20_tmp_4_dim, conv2d_21_w_0_dim, batch_norm_21_tmp_4_dim, 256, 2, 1, 1, 10);
	free(batch_norm_20_tmp_4);

	float* batch_norm_22_tmp_4 = (float*)calloc(1536, sizeof(float));
	autox_conv2d(batch_norm_21_tmp_4, (float*)((int8_t*)weights) + 365888, (float*)((int8_t*)weights) + 366144, batch_norm_22_tmp_4, batch_norm_21_tmp_4_dim, conv2d_22_w_0_dim, batch_norm_22_tmp_4_dim, 1, 0, 1, 1, 10);
	free(batch_norm_21_tmp_4);

	float* batch_norm_23_tmp_4 = (float*)calloc(768, sizeof(float));
	autox_conv2d(batch_norm_22_tmp_4, (float*)((int8_t*)weights) + 431680, (float*)((int8_t*)weights) + 431936, batch_norm_23_tmp_4, batch_norm_22_tmp_4_dim, conv2d_23_w_0_dim, batch_norm_23_tmp_4_dim, 256, 2, 2, 1, 10);
	free(batch_norm_22_tmp_4);

	float* pool2d_0_tmp_0 = (float*)calloc(256, sizeof(float));
	autox_pool2d(batch_norm_23_tmp_4, pool2d_0_tmp_0, batch_norm_23_tmp_4_dim, pool2d_0_tmp_0_dim, 1, 1, 0, 1, 1);
	//free(batch_norm_23_tmp_4);

	float* relu_0_tmp_0 = (float*)calloc(64, sizeof(float));
	autox_conv2d(pool2d_0_tmp_0, (float*)((int8_t*)weights) + 438336, (float*)((int8_t*)weights) + 438400, relu_0_tmp_0, pool2d_0_tmp_0_dim, conv2d_24_w_0_dim, relu_0_tmp_0_dim, 1, 0, 1, 1, 1);
	free(pool2d_0_tmp_0);

	float* conv2d_50_tmp_1 = (float*)calloc(256, sizeof(float));
	autox_conv2d(relu_0_tmp_0, (float*)((int8_t*)weights) + 454784, (float*)((int8_t*)weights) + 455040, conv2d_50_tmp_1, relu_0_tmp_0_dim, conv2d_25_w_0_dim, conv2d_50_tmp_1_dim, 1, 0, 1, 1, 0);
	free(relu_0_tmp_0);

	autox_hard_sigmoid(conv2d_50_tmp_1, conv2d_50_tmp_1_dim, 4, 0.5, 0.1666667);

	float* elementwise_mul_0 = (float*)calloc(768, sizeof(float));
	autox_elementwise_mul(batch_norm_23_tmp_4, conv2d_50_tmp_1, elementwise_mul_0, batch_norm_23_tmp_4_dim, hardsigmoid_0_tmp_0_dim, elementwise_mul_0_dim, -1, 4, 4, 4);
	free(batch_norm_23_tmp_4);
	free(conv2d_50_tmp_1);

	float* batch_norm_24_tmp_4 = (float*)calloc(1536, sizeof(float));
	autox_conv2d(elementwise_mul_0, (float*)((int8_t*)weights) + 471424, (float*)((int8_t*)weights) + 471936, batch_norm_24_tmp_4, elementwise_mul_0_dim, conv2d_26_w_0_dim, batch_norm_24_tmp_4_dim, 1, 0, 1, 1, 10);
	free(elementwise_mul_0);

	float* batch_norm_25_tmp_4 = (float*)calloc(1536, sizeof(float));
	autox_conv2d(batch_norm_24_tmp_4, (float*)((int8_t*)weights) + 603008, (float*)((int8_t*)weights) + 603520, batch_norm_25_tmp_4, batch_norm_24_tmp_4_dim, conv2d_27_w_0_dim, batch_norm_25_tmp_4_dim, 512, 2, 1, 1, 10);
	free(batch_norm_24_tmp_4);

	float* pool2d_1_tmp_0 = (float*)calloc(512, sizeof(float));
	autox_pool2d(batch_norm_25_tmp_4, pool2d_1_tmp_0, batch_norm_25_tmp_4_dim, pool2d_1_tmp_0_dim, 1, 1, 0, 1, 1);
	//free(batch_norm_25_tmp_4);

	float* relu_1_tmp_0 = (float*)calloc(128, sizeof(float));
	autox_conv2d(pool2d_1_tmp_0, (float*)((int8_t*)weights) + 616320, (float*)((int8_t*)weights) + 616448, relu_1_tmp_0, pool2d_1_tmp_0_dim, conv2d_28_w_0_dim, relu_1_tmp_0_dim, 1, 0, 1, 1, 1);
	free(pool2d_1_tmp_0);

	float* conv2d_53_tmp_1 = (float*)calloc(512, sizeof(float));
	autox_conv2d(relu_1_tmp_0, (float*)((int8_t*)weights) + 681984, (float*)((int8_t*)weights) + 682496, conv2d_53_tmp_1, relu_1_tmp_0_dim, conv2d_29_w_0_dim, conv2d_53_tmp_1_dim, 1, 0, 1, 1, 0);
	free(relu_1_tmp_0);

	autox_hard_sigmoid(conv2d_53_tmp_1, conv2d_53_tmp_1_dim, 4, 0.5, 0.1666667);

	float* elementwise_mul_1 = (float*)calloc(1536, sizeof(float));
	autox_elementwise_mul(batch_norm_25_tmp_4, conv2d_53_tmp_1, elementwise_mul_1, batch_norm_25_tmp_4_dim, hardsigmoid_1_tmp_0_dim, elementwise_mul_1_dim, -1, 4, 4, 4);
	free(batch_norm_25_tmp_4);
	free(conv2d_53_tmp_1);

	float* batch_norm_26_tmp_4 = (float*)calloc(1536, sizeof(float));
	autox_conv2d(elementwise_mul_1, (float*)((int8_t*)weights) + 748032, (float*)((int8_t*)weights) + 748544, batch_norm_26_tmp_4, elementwise_mul_1_dim, conv2d_30_w_0_dim, batch_norm_26_tmp_4_dim, 1, 0, 1, 1, 10);
	free(elementwise_mul_1);

	float* pool2d_2_tmp_0 = (float*)calloc(512, sizeof(float));
	autox_pool2d(batch_norm_26_tmp_4, pool2d_2_tmp_0, batch_norm_26_tmp_4_dim, pool2d_2_tmp_0_dim, 2, 2, 0, 0, 1);
	free(batch_norm_26_tmp_4);

	float* batch_norm_27_tmp_2 = (float*)calloc(64, sizeof(float));
	autox_conv2d(pool2d_2_tmp_0, (float*)((int8_t*)weights) + 1010688, (float*)((int8_t*)weights) + 1010752, batch_norm_27_tmp_2, pool2d_2_tmp_0_clone_0_dim, conv2d_31_w_0_dim, batch_norm_27_tmp_2_dim, 1, 1, 1, 1, 0);
	//free(pool2d_2_tmp_0);

	autox_swish(batch_norm_27_tmp_2, batch_norm_27_tmp_2_dim, 4, 1.0);

	float* batch_norm_28_tmp_2 = (float*)calloc(120, sizeof(float));
	autox_conv2d(batch_norm_27_tmp_2, (float*)((int8_t*)weights) + 1305664, (float*)((int8_t*)weights) + 1305784, batch_norm_28_tmp_2, swish_7_tmp_0_dim, conv2d_32_w_0_dim, batch_norm_28_tmp_2_dim, 1, 0, 1, 1, 0);
	free(batch_norm_27_tmp_2);

	autox_swish(batch_norm_28_tmp_2, batch_norm_28_tmp_2_dim, 4, 1.0);

	float* transpose_0_tmp_0 = (float*)calloc(120, sizeof(float));
	uint16_t axis_42[] = { 0, 2, 1 };
	autox_transpose(batch_norm_28_tmp_2, transpose_0_tmp_0, flatten_0_tmp_0_dim, transpose_0_tmp_0_dim, axis_42, 3);
	//free(batch_norm_28_tmp_2);

	float* layer_norm_5_tmp_2 = (float*)calloc(120, sizeof(float));
	autox_layer_norm(transpose_0_tmp_0, (float*)((int8_t*)weights) + 1313464, (float*)((int8_t*)weights) + 1313584, layer_norm_5_tmp_2, 1, 1e-05, 120);
	//free(transpose_0_tmp_0);

	float* linear_13_tmp_0 = (float*)calloc(360, sizeof(float));
	autox_matmul(layer_norm_5_tmp_2, (float*)((int8_t*)weights) + 1313704, linear_13_tmp_0, layer_norm_5_tmp_2_dim, linear_0_w_0_dim, linear_13_tmp_0_dim, 0, 0, 3, 2, 3);
	free(layer_norm_5_tmp_2);

	float* linear_13_tmp_1 = (float*)calloc(360, sizeof(float));
	autox_elementwise_add(linear_13_tmp_0, (float*)((int8_t*)weights) + 1356904, linear_13_tmp_1, linear_13_tmp_0_dim, linear_0_b_0_dim, linear_13_tmp_1_dim, 2, 3, 1, 3);
	free(linear_13_tmp_0);

	float* transpose_1_tmp_0 = (float*)calloc(360, sizeof(float));
	uint16_t axis_46[] = { 2, 0, 3, 1, 4 };
	autox_transpose2(linear_13_tmp_1, transpose_1_tmp_0, reshape2_0_tmp_0_dim, transpose_1_tmp_0_dim, axis_46, 5);
	free(linear_13_tmp_1);

	float* o_data[] = { NULL ,NULL ,NULL };
	autox_slice(transpose_1_tmp_0, o_data, transpose_1_tmp_0_dim, transpose_1_tmp_0_slice_0_dim, 5, 4);
	float* transpose_1_tmp_0_slice_0 = o_data[0];
	float* transpose_1_tmp_0_slice_1 = o_data[1];
	float* transpose_1_tmp_0_slice_2 = o_data[2];
	
	autox_scale(transpose_1_tmp_0_slice_0, transpose_1_tmp_0_slice_0_dim, 4, 0, 0, 0.2581989);

	float* transpose_2_tmp_0 = (float*)calloc(120, sizeof(float));
	uint16_t axis_48[] = { 0, 1, 3, 2 };
	autox_transpose(transpose_1_tmp_0_slice_1, transpose_2_tmp_0, transpose_1_tmp_0_slice_1_dim, transpose_2_tmp_0_dim, axis_48, 4);
	//free(transpose_1_tmp_0_slice_1);

	float* matmul_v2_0_tmp_0 = (float*)calloc(8, sizeof(float));
	autox_matmul(transpose_1_tmp_0_slice_0, transpose_2_tmp_0, matmul_v2_0_tmp_0, tmp_0_dim, transpose_2_tmp_0_dim, matmul_v2_0_tmp_0_dim, 0, 0, 4, 4, 4);
	//free(transpose_1_tmp_0_slice_0);
	free(transpose_2_tmp_0);

	autox_softmax(matmul_v2_0_tmp_0, matmul_v2_0_tmp_0_dim[1], matmul_v2_0_tmp_0_dim[2]);

	float* matmul_v2_1_tmp_0 = (float*)calloc(120, sizeof(float));
	autox_matmul(matmul_v2_0_tmp_0, transpose_1_tmp_0_slice_2, matmul_v2_1_tmp_0, dropout_7_tmp_0_dim, transpose_1_tmp_0_slice_2_dim, matmul_v2_1_tmp_0_dim, 0, 0, 4, 4, 4);
	free(matmul_v2_0_tmp_0);
	//free(transpose_1_tmp_0_slice_2);

	free(transpose_1_tmp_0);

	float* transpose_3_tmp_0 = (float*)calloc(120, sizeof(float));
	uint16_t axis_52[] = { 0, 2, 1, 3 };
	autox_transpose(matmul_v2_1_tmp_0, transpose_3_tmp_0, matmul_v2_1_tmp_0_dim, transpose_3_tmp_0_dim, axis_52, 4);
	free(matmul_v2_1_tmp_0);

	float* linear_14_tmp_0 = (float*)calloc(120, sizeof(float));
	autox_matmul(transpose_3_tmp_0, (float*)((int8_t*)weights) + 1357264, linear_14_tmp_0, reshape2_1_tmp_0_dim, linear_1_w_0_dim, linear_14_tmp_0_dim, 0, 0, 3, 2, 3);
	free(transpose_3_tmp_0);

	float* dropout_8_tmp_0 = (float*)calloc(120, sizeof(float));
	autox_elementwise_add(linear_14_tmp_0, (float*)((int8_t*)weights) + 1371664, dropout_8_tmp_0, linear_14_tmp_0_dim, linear_1_b_0_dim, dropout_8_tmp_0_dim, 2, 3, 1, 3);
	free(linear_14_tmp_0);

	float* tmp_1 = (float*)calloc(120, sizeof(float));
	autox_elementwise_add(transpose_0_tmp_0, dropout_8_tmp_0, tmp_1, transpose_0_tmp_0_dim, dropout_8_tmp_0_dim, tmp_1_dim, -1, 3, 3, 3);
	free(transpose_0_tmp_0);
	free(dropout_8_tmp_0);

	float* layer_norm_6_tmp_2 = (float*)calloc(120, sizeof(float));
	autox_layer_norm(tmp_1, (float*)((int8_t*)weights) + 1371784, (float*)((int8_t*)weights) + 1371904, layer_norm_6_tmp_2, 1, 1e-05, 120);
	//free(tmp_1);

	float* linear_15_tmp_0 = (float*)calloc(240, sizeof(float));
	autox_matmul(layer_norm_6_tmp_2, (float*)((int8_t*)weights) + 1372024, linear_15_tmp_0, layer_norm_6_tmp_2_dim, linear_2_w_0_dim, linear_15_tmp_0_dim, 0, 0, 3, 2, 3);
	free(layer_norm_6_tmp_2);

	float* linear_15_tmp_1 = (float*)calloc(240, sizeof(float));
	autox_elementwise_add(linear_15_tmp_0, (float*)((int8_t*)weights) + 1400824, linear_15_tmp_1, linear_15_tmp_0_dim, linear_2_b_0_dim, linear_15_tmp_1_dim, 2, 3, 1, 3);
	free(linear_15_tmp_0);

	autox_swish(linear_15_tmp_1, linear_15_tmp_1_dim, 3, 1.0);

	float* linear_16_tmp_0 = (float*)calloc(120, sizeof(float));
	autox_matmul(linear_15_tmp_1, (float*)((int8_t*)weights) + 1401064, linear_16_tmp_0, dropout_9_tmp_0_dim, linear_3_w_0_dim, linear_16_tmp_0_dim, 0, 0, 3, 2, 3);
	free(linear_15_tmp_1);

	float* dropout_10_tmp_0 = (float*)calloc(120, sizeof(float));
	autox_elementwise_add(linear_16_tmp_0, (float*)((int8_t*)weights) + 1429864, dropout_10_tmp_0, linear_16_tmp_0_dim, linear_3_b_0_dim, dropout_10_tmp_0_dim, 2, 3, 1, 3);
	free(linear_16_tmp_0);

	float* tmp_2 = (float*)calloc(120, sizeof(float));
	autox_elementwise_add(tmp_1, dropout_10_tmp_0, tmp_2, tmp_1_dim, dropout_10_tmp_0_dim, tmp_2_dim, -1, 3, 3, 3);
	free(tmp_1);
	free(dropout_10_tmp_0);

	float* layer_norm_7_tmp_2 = (float*)calloc(120, sizeof(float));
	autox_layer_norm(tmp_2, (float*)((int8_t*)weights) + 1429984, (float*)((int8_t*)weights) + 1430104, layer_norm_7_tmp_2, 1, 1e-05, 120);
	//free(tmp_2);

	float* linear_17_tmp_0 = (float*)calloc(360, sizeof(float));
	autox_matmul(layer_norm_7_tmp_2, (float*)((int8_t*)weights) + 1430224, linear_17_tmp_0, layer_norm_7_tmp_2_dim, linear_4_w_0_dim, linear_17_tmp_0_dim, 0, 0, 3, 2, 3);
	free(layer_norm_7_tmp_2);

	float* linear_17_tmp_1 = (float*)calloc(360, sizeof(float));
	autox_elementwise_add(linear_17_tmp_0, (float*)((int8_t*)weights) + 1473424, linear_17_tmp_1, linear_17_tmp_0_dim, linear_4_b_0_dim, linear_17_tmp_1_dim, 2, 3, 1, 3);
	free(linear_17_tmp_0);

	float* transpose_4_tmp_0 = (float*)calloc(360, sizeof(float));
	uint16_t axis_66[] = { 2, 0, 3, 1, 4 };
	autox_transpose2(linear_17_tmp_1, transpose_4_tmp_0, reshape2_2_tmp_0_dim, transpose_4_tmp_0_dim, axis_66, 5);
	free(linear_17_tmp_1);

	autox_slice(transpose_4_tmp_0, o_data, transpose_4_tmp_0_dim, transpose_4_tmp_0_slice_0_dim, 5, 4);
	float* transpose_4_tmp_0_slice_0 = o_data[0];
	float* transpose_4_tmp_0_slice_1 = o_data[1];
	float* transpose_4_tmp_0_slice_2 = o_data[2];

	autox_scale(transpose_4_tmp_0_slice_0, transpose_4_tmp_0_slice_0_dim, 4, 0, 1, 0.2581989);

	float* transpose_5_tmp_0 = (float*)calloc(120, sizeof(float));
	uint16_t axis_68[] = { 0, 1, 3, 2 };
	autox_transpose(transpose_4_tmp_0_slice_1, transpose_5_tmp_0, transpose_4_tmp_0_slice_1_dim, transpose_5_tmp_0_dim, axis_68, 4);
	//free(transpose_4_tmp_0_slice_1);

	float* matmul_v2_2_tmp_0 = (float*)calloc(8, sizeof(float));
	autox_matmul(transpose_4_tmp_0_slice_0, transpose_5_tmp_0, matmul_v2_2_tmp_0, tmp_3_dim, transpose_5_tmp_0_dim, matmul_v2_2_tmp_0_dim, 0, 0, 4, 4, 4);
	//free(transpose_4_tmp_0_slice_0);
	free(transpose_5_tmp_0);

	autox_softmax(matmul_v2_2_tmp_0, matmul_v2_2_tmp_0_dim[1], matmul_v2_2_tmp_0_dim[2]);

	float* matmul_v2_3_tmp_0 = (float*)calloc(120, sizeof(float));
	autox_matmul(matmul_v2_2_tmp_0, transpose_4_tmp_0_slice_2, matmul_v2_3_tmp_0, dropout_11_tmp_0_dim, transpose_4_tmp_0_slice_2_dim, matmul_v2_3_tmp_0_dim, 0, 0, 4, 4, 4);
	free(matmul_v2_2_tmp_0);
	//free(transpose_4_tmp_0_slice_2);

	free(transpose_4_tmp_0);

	float* transpose_6_tmp_0 = (float*)calloc(120, sizeof(float));
	uint16_t axis_72[] = { 0, 2, 1, 3 };
	autox_transpose(matmul_v2_3_tmp_0, transpose_6_tmp_0, matmul_v2_3_tmp_0_dim, transpose_6_tmp_0_dim, axis_72, 4);
	free(matmul_v2_3_tmp_0);

	float* linear_18_tmp_0 = (float*)calloc(120, sizeof(float));
	autox_matmul(transpose_6_tmp_0, (float*)((int8_t*)weights) + 1473784, linear_18_tmp_0, reshape2_3_tmp_0_dim, linear_5_w_0_dim, linear_18_tmp_0_dim, 0, 0, 3, 2, 3);
	free(transpose_6_tmp_0);

	float* dropout_12_tmp_0 = (float*)calloc(120, sizeof(float));
	autox_elementwise_add(linear_18_tmp_0, (float*)((int8_t*)weights) + 1488184, dropout_12_tmp_0, linear_18_tmp_0_dim, linear_5_b_0_dim, dropout_12_tmp_0_dim, 2, 3, 1, 3);
	free(linear_18_tmp_0);

	float* tmp_4 = (float*)calloc(120, sizeof(float));
	autox_elementwise_add(tmp_2, dropout_12_tmp_0, tmp_4, tmp_2_dim, dropout_12_tmp_0_dim, tmp_4_dim, -1, 3, 3, 3);
	free(tmp_2);
	free(dropout_12_tmp_0);

	float* layer_norm_8_tmp_2 = (float*)calloc(120, sizeof(float));
	autox_layer_norm(tmp_4, (float*)((int8_t*)weights) + 1488304, (float*)((int8_t*)weights) + 1488424, layer_norm_8_tmp_2, 1, 1e-05, 120);
	//free(tmp_4);

	float* linear_19_tmp_0 = (float*)calloc(240, sizeof(float));
	autox_matmul(layer_norm_8_tmp_2, (float*)((int8_t*)weights) + 1488544, linear_19_tmp_0, layer_norm_8_tmp_2_dim, linear_6_w_0_dim, linear_19_tmp_0_dim, 0, 0, 3, 2, 3);
	free(layer_norm_8_tmp_2);

	float* linear_19_tmp_1 = (float*)calloc(240, sizeof(float));
	autox_elementwise_add(linear_19_tmp_0, (float*)((int8_t*)weights) + 1517344, linear_19_tmp_1, linear_19_tmp_0_dim, linear_6_b_0_dim, linear_19_tmp_1_dim, 2, 3, 1, 3);
	free(linear_19_tmp_0);

	autox_swish(linear_19_tmp_1, linear_19_tmp_1_dim, 3, 1.0);

	float* linear_20_tmp_0 = (float*)calloc(120, sizeof(float));
	autox_matmul(linear_19_tmp_1, (float*)((int8_t*)weights) + 1517584, linear_20_tmp_0, dropout_13_tmp_0_dim, linear_7_w_0_dim, linear_20_tmp_0_dim, 0, 0, 3, 2, 3);
	free(linear_19_tmp_1);

	float* dropout_14_tmp_0 = (float*)calloc(120, sizeof(float));
	autox_elementwise_add(linear_20_tmp_0, (float*)((int8_t*)weights) + 1546384, dropout_14_tmp_0, linear_20_tmp_0_dim, linear_7_b_0_dim, dropout_14_tmp_0_dim, 2, 3, 1, 3);
	free(linear_20_tmp_0);

	float* tmp_5 = (float*)calloc(120, sizeof(float));
	autox_elementwise_add(tmp_4, dropout_14_tmp_0, tmp_5, tmp_4_dim, dropout_14_tmp_0_dim, tmp_5_dim, -1, 3, 3, 3);
	free(tmp_4);
	free(dropout_14_tmp_0);

	float* layer_norm_9_tmp_2 = (float*)calloc(120, sizeof(float));
	autox_layer_norm(tmp_5, (float*)((int8_t*)weights) + 1546504, (float*)((int8_t*)weights) + 1546624, layer_norm_9_tmp_2, 1, 1e-06, 120);
	free(tmp_5);

	float* transpose_7_tmp_0 = (float*)calloc(120, sizeof(float));
	uint16_t axis_84[] = { 0, 3, 1, 2 };
	autox_transpose(layer_norm_9_tmp_2, transpose_7_tmp_0, reshape2_4_tmp_0_dim, transpose_7_tmp_0_dim, axis_84, 4);
	free(layer_norm_9_tmp_2);

	float* batch_norm_29_tmp_2 = (float*)calloc(512, sizeof(float));
	autox_conv2d(transpose_7_tmp_0, (float*)((int8_t*)weights) + 1546744, (float*)((int8_t*)weights) + 1547256, batch_norm_29_tmp_2, transpose_7_tmp_0_dim, conv2d_33_w_0_dim, batch_norm_29_tmp_2_dim, 1, 0, 1, 1, 0);
	free(transpose_7_tmp_0);

	autox_swish(batch_norm_29_tmp_2, batch_norm_29_tmp_2_dim, 4, 1.0);

	float* p_87[] = { pool2d_2_tmp_0, batch_norm_29_tmp_2, };
	uint16_t* p_87_dim[] = { pool2d_2_tmp_0_clone_0_dim, swish_11_tmp_0_dim, };
	float* concat_0_tmp_0 = (float*)calloc(1024, sizeof(float));
	autox_concat(p_87, concat_0_tmp_0, p_87_dim, concat_0_tmp_0_dim, 1, 2, 4);
	free(pool2d_2_tmp_0);
	free(batch_norm_29_tmp_2);

	float* batch_norm_30_tmp_2 = (float*)calloc(64, sizeof(float));
	autox_conv2d(concat_0_tmp_0, (float*)((int8_t*)weights) + 1608696, (float*)((int8_t*)weights) + 1608760, batch_norm_30_tmp_2, concat_0_tmp_0_dim, conv2d_34_w_0_dim, batch_norm_30_tmp_2_dim, 1, 1, 1, 1, 0);
	free(concat_0_tmp_0);

	autox_swish(batch_norm_30_tmp_2, batch_norm_30_tmp_2_dim, 4, 1.0);

	float* batch_norm_31_tmp_2 = (float*)calloc(64, sizeof(float));
	autox_conv2d(batch_norm_30_tmp_2, (float*)((int8_t*)weights) + 2198584, (float*)((int8_t*)weights) + 2198648, batch_norm_31_tmp_2, swish_12_tmp_0_dim, conv2d_35_w_0_dim, batch_norm_31_tmp_2_dim, 1, 0, 1, 1, 0);
	free(batch_norm_30_tmp_2);

	autox_swish(batch_norm_31_tmp_2, batch_norm_31_tmp_2_dim, 4, 1.0);

	float* transpose_8_tmp_0 = (float*)calloc(64, sizeof(float));
	uint16_t axis_92[] = { 0, 2, 1 };
	autox_transpose(batch_norm_31_tmp_2, transpose_8_tmp_0, squeeze_0_tmp_0_dim, transpose_8_tmp_0_dim, axis_92, 3);
	free(batch_norm_31_tmp_2);

	float* linear_21_tmp_0 = (float*)calloc(4401, sizeof(float));
	autox_matmul(transpose_8_tmp_0, (float*)((int8_t*)weights) + 2202744, linear_21_tmp_0, transpose_8_tmp_0_dim, linear_8_w_0_dim, linear_21_tmp_0_dim, 0, 0, 3, 2, 3);
	free(transpose_8_tmp_0);

	autox_elementwise_add(linear_21_tmp_0, (float*)((int8_t*)weights) + 2484408, Out, linear_21_tmp_0_dim, linear_8_b_0_dim, linear_21_tmp_1_dim, 2, 3, 1, 3);

	// autox_softmax(Out, linear_21_tmp_1_dim[1], linear_21_tmp_1_dim[2]);
}
