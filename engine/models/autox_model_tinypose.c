#include "../include/autox_models.h"

void tinypose_128x96(const float* x, void* weights, float* Out)
{
	uint16_t image_dim[] = { 1, 3, 128, 96 };
	uint16_t batch_norm2d_0_b_0_dim[] = { 32 };
	uint16_t conv2d_0_w_0_dim[] = { 32, 3, 3, 3 };
	uint16_t relu_0_tmp_0_dim[] = { 1, 32, 64, 48 };
	uint16_t split_0_tmp_0_dim[] = { 1, 16, 64, 48 };
	uint16_t split_0_tmp_1_dim[] = { 1, 16, 64, 48 };
	uint16_t batch_norm2d_1_b_0_dim[] = { 16 };
	uint16_t batch_norm_1_tmp_2_dim[] = { 1, 16, 32, 24 };
	uint16_t conv2d_1_w_0_dim[] = { 16, 1, 3, 3 };
	uint16_t batch_norm2d_3_b_0_dim[] = { 32 };
	uint16_t conv2d_3_w_0_dim[] = { 32, 16, 1, 1 };
	uint16_t relu_2_tmp_0_dim[] = { 1, 32, 64, 48 };
	uint16_t batch_norm2d_4_b_0_dim[] = { 32 };
	uint16_t batch_norm_4_tmp_2_dim[] = { 1, 32, 32, 24 };
	uint16_t conv2d_4_w_0_dim[] = { 32, 1, 3, 3 };
	uint16_t batch_norm2d_2_b_0_dim[] = { 16 };
	uint16_t conv2d_2_w_0_dim[] = { 16, 16, 1, 1 };
	uint16_t relu_1_tmp_0_dim[] = { 1, 16, 32, 24 };
	uint16_t batch_norm2d_5_b_0_dim[] = { 16 };
	uint16_t conv2d_5_w_0_dim[] = { 16, 32, 1, 1 };
	uint16_t relu_3_tmp_0_dim[] = { 1, 16, 32, 24 };
	uint16_t concat_0_tmp_0_dim[] = { 1, 32, 32, 24 };
	uint16_t shape_1_tmp_0_dim[] = { 4 };
	uint16_t shape_1_tmp_0_slice_0_dim[] = { 1 };
	uint16_t reshape2_0_tmp_0_dim[] = { 1, 2, 16, 32, 24 };
	uint16_t reshape2_0_tmp_1_dim[] = { 0, 1, 32, 32, 24 };
	uint16_t transpose_0_tmp_0_dim[] = { 1, 16, 2, 32, 24 };
	uint16_t transpose_0_tmp_1_dim[] = { 0, 1, 2, 16, 32, 24 };
	uint16_t reshape2_1_tmp_0_dim[] = { 1, 32, 32, 24 };
	uint16_t reshape2_1_tmp_1_dim[] = { 0, 1, 16, 2, 32, 24 };
	uint16_t batch_norm2d_6_b_0_dim[] = { 32 };
	uint16_t batch_norm_6_tmp_2_dim[] = { 1, 32, 32, 24 };
	uint16_t conv2d_6_w_0_dim[] = { 32, 1, 3, 3 };
	uint16_t batch_norm2d_8_b_0_dim[] = { 32 };
	uint16_t batch_norm_8_tmp_2_dim[] = { 1, 32, 16, 12 };
	uint16_t conv2d_8_w_0_dim[] = { 32, 1, 3, 3 };
	uint16_t batch_norm2d_7_b_0_dim[] = { 40 };
	uint16_t conv2d_7_w_0_dim[] = { 40, 32, 1, 1 };
	uint16_t relu_4_tmp_0_dim[] = { 1, 40, 32, 24 };
	uint16_t split_1_tmp_0_dim[] = { 1, 20, 32, 24 };
	uint16_t split_1_tmp_1_dim[] = { 1, 20, 32, 24 };
	uint16_t batch_norm2d_10_b_0_dim[] = { 20 };
	uint16_t conv2d_10_w_0_dim[] = { 20, 20, 1, 1 };
	uint16_t relu_6_tmp_0_dim[] = { 1, 20, 32, 24 };
	uint16_t batch_norm2d_11_b_0_dim[] = { 20 };
	uint16_t batch_norm_11_tmp_2_dim[] = { 1, 20, 32, 24 };
	uint16_t conv2d_11_w_0_dim[] = { 20, 1, 3, 3 };
	uint16_t batch_norm2d_12_b_0_dim[] = { 20 };
	uint16_t conv2d_12_w_0_dim[] = { 20, 20, 1, 1 };
	uint16_t relu_7_tmp_0_dim[] = { 1, 20, 32, 24 };
	uint16_t concat_1_tmp_0_dim[] = { 1, 40, 32, 24 };
	uint16_t shape_2_tmp_0_dim[] = { 4 };
	uint16_t shape_2_tmp_0_slice_0_dim[] = { 1 };
	uint16_t reshape2_2_tmp_0_dim[] = { 1, 2, 20, 32, 24 };
	uint16_t reshape2_2_tmp_1_dim[] = { 0, 1, 40, 32, 24 };
	uint16_t transpose_1_tmp_0_dim[] = { 1, 20, 2, 32, 24 };
	uint16_t transpose_1_tmp_1_dim[] = { 0, 1, 2, 20, 32, 24 };
	uint16_t reshape2_3_tmp_0_dim[] = { 1, 40, 32, 24 };
	uint16_t reshape2_3_tmp_1_dim[] = { 0, 1, 20, 2, 32, 24 };
	uint16_t split_2_tmp_0_dim[] = { 1, 20, 32, 24 };
	uint16_t split_2_tmp_1_dim[] = { 1, 20, 32, 24 };
	uint16_t batch_norm2d_13_b_0_dim[] = { 20 };
	uint16_t conv2d_13_w_0_dim[] = { 20, 20, 1, 1 };
	uint16_t relu_8_tmp_0_dim[] = { 1, 20, 32, 24 };
	uint16_t batch_norm2d_14_b_0_dim[] = { 20 };
	uint16_t batch_norm_14_tmp_2_dim[] = { 1, 20, 32, 24 };
	uint16_t conv2d_14_w_0_dim[] = { 20, 1, 3, 3 };
	uint16_t batch_norm2d_15_b_0_dim[] = { 20 };
	uint16_t conv2d_15_w_0_dim[] = { 20, 20, 1, 1 };
	uint16_t relu_9_tmp_0_dim[] = { 1, 20, 32, 24 };
	uint16_t concat_2_tmp_0_dim[] = { 1, 40, 32, 24 };
	uint16_t shape_3_tmp_0_dim[] = { 4 };
	uint16_t shape_3_tmp_0_slice_0_dim[] = { 1 };
	uint16_t reshape2_4_tmp_0_dim[] = { 1, 2, 20, 32, 24 };
	uint16_t reshape2_4_tmp_1_dim[] = { 0, 1, 40, 32, 24 };
	uint16_t transpose_2_tmp_0_dim[] = { 1, 20, 2, 32, 24 };
	uint16_t transpose_2_tmp_1_dim[] = { 0, 1, 2, 20, 32, 24 };
	uint16_t reshape2_5_tmp_0_dim[] = { 1, 40, 32, 24 };
	uint16_t reshape2_5_tmp_1_dim[] = { 0, 1, 20, 2, 32, 24 };
	uint16_t batch_norm2d_9_b_0_dim[] = { 80 };
	uint16_t conv2d_9_w_0_dim[] = { 80, 32, 1, 1 };
	uint16_t relu_5_tmp_0_dim[] = { 1, 80, 16, 12 };
	uint16_t split_3_tmp_0_dim[] = { 1, 40, 16, 12 };
	uint16_t split_3_tmp_1_dim[] = { 1, 40, 16, 12 };
	uint16_t batch_norm2d_16_b_0_dim[] = { 40 };
	uint16_t conv2d_16_w_0_dim[] = { 40, 40, 1, 1 };
	uint16_t relu_10_tmp_0_dim[] = { 1, 40, 16, 12 };
	uint16_t batch_norm2d_17_b_0_dim[] = { 40 };
	uint16_t batch_norm_17_tmp_2_dim[] = { 1, 40, 16, 12 };
	uint16_t conv2d_17_w_0_dim[] = { 40, 1, 3, 3 };
	uint16_t batch_norm2d_18_b_0_dim[] = { 40 };
	uint16_t conv2d_18_w_0_dim[] = { 40, 40, 1, 1 };
	uint16_t relu_11_tmp_0_dim[] = { 1, 40, 16, 12 };
	uint16_t concat_3_tmp_0_dim[] = { 1, 80, 16, 12 };
	uint16_t shape_4_tmp_0_dim[] = { 4 };
	uint16_t shape_4_tmp_0_slice_0_dim[] = { 1 };
	uint16_t reshape2_6_tmp_0_dim[] = { 1, 2, 40, 16, 12 };
	uint16_t reshape2_6_tmp_1_dim[] = { 0, 1, 80, 16, 12 };
	uint16_t transpose_3_tmp_0_dim[] = { 1, 40, 2, 16, 12 };
	uint16_t transpose_3_tmp_1_dim[] = { 0, 1, 2, 40, 16, 12 };
	uint16_t reshape2_7_tmp_0_dim[] = { 1, 80, 16, 12 };
	uint16_t reshape2_7_tmp_1_dim[] = { 0, 1, 40, 2, 16, 12 };
	uint16_t split_4_tmp_0_dim[] = { 1, 40, 16, 12 };
	uint16_t split_4_tmp_1_dim[] = { 1, 40, 16, 12 };
	uint16_t batch_norm2d_19_b_0_dim[] = { 40 };
	uint16_t conv2d_19_w_0_dim[] = { 40, 40, 1, 1 };
	uint16_t relu_12_tmp_0_dim[] = { 1, 40, 16, 12 };
	uint16_t batch_norm2d_20_b_0_dim[] = { 40 };
	uint16_t batch_norm_20_tmp_2_dim[] = { 1, 40, 16, 12 };
	uint16_t conv2d_20_w_0_dim[] = { 40, 1, 3, 3 };
	uint16_t batch_norm2d_21_b_0_dim[] = { 40 };
	uint16_t conv2d_21_w_0_dim[] = { 40, 40, 1, 1 };
	uint16_t relu_13_tmp_0_dim[] = { 1, 40, 16, 12 };
	uint16_t concat_4_tmp_0_dim[] = { 1, 80, 16, 12 };
	uint16_t shape_5_tmp_0_dim[] = { 4 };
	uint16_t shape_5_tmp_0_slice_0_dim[] = { 1 };
	uint16_t reshape2_8_tmp_0_dim[] = { 1, 2, 40, 16, 12 };
	uint16_t reshape2_8_tmp_1_dim[] = { 0, 1, 80, 16, 12 };
	uint16_t transpose_4_tmp_0_dim[] = { 1, 40, 2, 16, 12 };
	uint16_t transpose_4_tmp_1_dim[] = { 0, 1, 2, 40, 16, 12 };
	uint16_t reshape2_9_tmp_0_dim[] = { 1, 80, 16, 12 };
	uint16_t reshape2_9_tmp_1_dim[] = { 0, 1, 40, 2, 16, 12 };
	uint16_t tmp_0_dim[] = { 1, 40, 32, 24 };
	uint16_t batch_norm2d_22_b_0_dim[] = { 40 };
	uint16_t batch_norm_22_tmp_2_dim[] = { 1, 40, 16, 12 };
	uint16_t conv2d_22_w_0_dim[] = { 40, 80, 1, 1 };
	uint16_t nearest_interp_v2_0_tmp_0_dim[] = { 1, 40, 32, 24 };
	uint16_t tmp_1_dim[] = { 1, 40, 32, 24 };
	uint16_t relu_14_tmp_0_dim[] = { 1, 40, 32, 24 };
	uint16_t batch_norm2d_23_b_0_dim[] = { 40 };
	uint16_t batch_norm_23_tmp_2_dim[] = { 1, 40, 16, 12 };
	uint16_t conv2d_23_w_0_dim[] = { 40, 1, 3, 3 };
	uint16_t batch_norm2d_24_b_0_dim[] = { 80 };
	uint16_t batch_norm_24_tmp_2_dim[] = { 1, 80, 16, 12 };
	uint16_t conv2d_24_w_0_dim[] = { 80, 40, 1, 1 };
	uint16_t tmp_2_dim[] = { 1, 80, 16, 12 };
	uint16_t split_5_tmp_0_dim[] = { 1, 20, 32, 24 };
	uint16_t split_5_tmp_1_dim[] = { 1, 20, 32, 24 };
	uint16_t batch_norm2d_25_b_0_dim[] = { 20 };
	uint16_t conv2d_25_w_0_dim[] = { 20, 20, 1, 1 };
	uint16_t relu_16_tmp_0_dim[] = { 1, 20, 32, 24 };
	uint16_t batch_norm2d_26_b_0_dim[] = { 20 };
	uint16_t batch_norm_26_tmp_2_dim[] = { 1, 20, 32, 24 };
	uint16_t conv2d_26_w_0_dim[] = { 20, 1, 3, 3 };
	uint16_t batch_norm2d_27_b_0_dim[] = { 20 };
	uint16_t conv2d_27_w_0_dim[] = { 20, 20, 1, 1 };
	uint16_t relu_17_tmp_0_dim[] = { 1, 20, 32, 24 };
	uint16_t concat_5_tmp_0_dim[] = { 1, 40, 32, 24 };
	uint16_t shape_6_tmp_0_dim[] = { 4 };
	uint16_t shape_6_tmp_0_slice_0_dim[] = { 1 };
	uint16_t reshape2_10_tmp_0_dim[] = { 1, 2, 20, 32, 24 };
	uint16_t reshape2_10_tmp_1_dim[] = { 0, 1, 40, 32, 24 };
	uint16_t transpose_5_tmp_0_dim[] = { 1, 20, 2, 32, 24 };
	uint16_t transpose_5_tmp_1_dim[] = { 0, 1, 2, 20, 32, 24 };
	uint16_t reshape2_11_tmp_0_dim[] = { 1, 40, 32, 24 };
	uint16_t reshape2_11_tmp_1_dim[] = { 0, 1, 20, 2, 32, 24 };
	uint16_t split_6_tmp_0_dim[] = { 1, 20, 32, 24 };
	uint16_t split_6_tmp_1_dim[] = { 1, 20, 32, 24 };
	uint16_t batch_norm2d_28_b_0_dim[] = { 20 };
	uint16_t conv2d_28_w_0_dim[] = { 20, 20, 1, 1 };
	uint16_t relu_18_tmp_0_dim[] = { 1, 20, 32, 24 };
	uint16_t batch_norm2d_29_b_0_dim[] = { 20 };
	uint16_t batch_norm_29_tmp_2_dim[] = { 1, 20, 32, 24 };
	uint16_t conv2d_29_w_0_dim[] = { 20, 1, 3, 3 };
	uint16_t batch_norm2d_30_b_0_dim[] = { 20 };
	uint16_t conv2d_30_w_0_dim[] = { 20, 20, 1, 1 };
	uint16_t relu_19_tmp_0_dim[] = { 1, 20, 32, 24 };
	uint16_t concat_6_tmp_0_dim[] = { 1, 40, 32, 24 };
	uint16_t shape_7_tmp_0_dim[] = { 4 };
	uint16_t shape_7_tmp_0_slice_0_dim[] = { 1 };
	uint16_t reshape2_12_tmp_0_dim[] = { 1, 2, 20, 32, 24 };
	uint16_t reshape2_12_tmp_1_dim[] = { 0, 1, 40, 32, 24 };
	uint16_t transpose_6_tmp_0_dim[] = { 1, 20, 2, 32, 24 };
	uint16_t transpose_6_tmp_1_dim[] = { 0, 1, 2, 20, 32, 24 };
	uint16_t reshape2_13_tmp_0_dim[] = { 1, 40, 32, 24 };
	uint16_t reshape2_13_tmp_1_dim[] = { 0, 1, 20, 2, 32, 24 };
	uint16_t relu_15_tmp_0_dim[] = { 1, 80, 16, 12 };
	uint16_t split_7_tmp_0_dim[] = { 1, 40, 16, 12 };
	uint16_t split_7_tmp_1_dim[] = { 1, 40, 16, 12 };
	uint16_t batch_norm2d_31_b_0_dim[] = { 40 };
	uint16_t conv2d_31_w_0_dim[] = { 40, 40, 1, 1 };
	uint16_t relu_20_tmp_0_dim[] = { 1, 40, 16, 12 };
	uint16_t batch_norm2d_32_b_0_dim[] = { 40 };
	uint16_t batch_norm_32_tmp_2_dim[] = { 1, 40, 16, 12 };
	uint16_t conv2d_32_w_0_dim[] = { 40, 1, 3, 3 };
	uint16_t batch_norm2d_33_b_0_dim[] = { 40 };
	uint16_t conv2d_33_w_0_dim[] = { 40, 40, 1, 1 };
	uint16_t relu_21_tmp_0_dim[] = { 1, 40, 16, 12 };
	uint16_t concat_7_tmp_0_dim[] = { 1, 80, 16, 12 };
	uint16_t shape_8_tmp_0_dim[] = { 4 };
	uint16_t shape_8_tmp_0_slice_0_dim[] = { 1 };
	uint16_t reshape2_14_tmp_0_dim[] = { 1, 2, 40, 16, 12 };
	uint16_t reshape2_14_tmp_1_dim[] = { 0, 1, 80, 16, 12 };
	uint16_t transpose_7_tmp_0_dim[] = { 1, 40, 2, 16, 12 };
	uint16_t transpose_7_tmp_1_dim[] = { 0, 1, 2, 40, 16, 12 };
	uint16_t reshape2_15_tmp_0_dim[] = { 1, 80, 16, 12 };
	uint16_t reshape2_15_tmp_1_dim[] = { 0, 1, 40, 2, 16, 12 };
	uint16_t split_8_tmp_0_dim[] = { 1, 40, 16, 12 };
	uint16_t split_8_tmp_1_dim[] = { 1, 40, 16, 12 };
	uint16_t batch_norm2d_34_b_0_dim[] = { 40 };
	uint16_t conv2d_34_w_0_dim[] = { 40, 40, 1, 1 };
	uint16_t relu_22_tmp_0_dim[] = { 1, 40, 16, 12 };
	uint16_t batch_norm2d_35_b_0_dim[] = { 40 };
	uint16_t batch_norm_35_tmp_2_dim[] = { 1, 40, 16, 12 };
	uint16_t conv2d_35_w_0_dim[] = { 40, 1, 3, 3 };
	uint16_t batch_norm2d_36_b_0_dim[] = { 40 };
	uint16_t conv2d_36_w_0_dim[] = { 40, 40, 1, 1 };
	uint16_t relu_23_tmp_0_dim[] = { 1, 40, 16, 12 };
	uint16_t concat_8_tmp_0_dim[] = { 1, 80, 16, 12 };
	uint16_t shape_9_tmp_0_dim[] = { 4 };
	uint16_t shape_9_tmp_0_slice_0_dim[] = { 1 };
	uint16_t reshape2_16_tmp_0_dim[] = { 1, 2, 40, 16, 12 };
	uint16_t reshape2_16_tmp_1_dim[] = { 0, 1, 80, 16, 12 };
	uint16_t transpose_8_tmp_0_dim[] = { 1, 40, 2, 16, 12 };
	uint16_t transpose_8_tmp_1_dim[] = { 0, 1, 2, 40, 16, 12 };
	uint16_t reshape2_17_tmp_0_dim[] = { 1, 80, 16, 12 };
	uint16_t reshape2_17_tmp_1_dim[] = { 0, 1, 40, 2, 16, 12 };
	uint16_t tmp_4_dim[] = { 1, 40, 32, 24 };
	uint16_t batch_norm2d_37_b_0_dim[] = { 40 };
	uint16_t batch_norm_37_tmp_2_dim[] = { 1, 40, 16, 12 };
	uint16_t conv2d_37_w_0_dim[] = { 40, 80, 1, 1 };
	uint16_t nearest_interp_v2_1_tmp_0_dim[] = { 1, 40, 32, 24 };
	uint16_t tmp_5_dim[] = { 1, 40, 32, 24 };
	uint16_t relu_24_tmp_0_dim[] = { 1, 40, 32, 24 };
	uint16_t batch_norm2d_38_b_0_dim[] = { 40 };
	uint16_t batch_norm_38_tmp_2_dim[] = { 1, 40, 16, 12 };
	uint16_t conv2d_38_w_0_dim[] = { 40, 1, 3, 3 };
	uint16_t batch_norm2d_39_b_0_dim[] = { 80 };
	uint16_t batch_norm_39_tmp_2_dim[] = { 1, 80, 16, 12 };
	uint16_t conv2d_39_w_0_dim[] = { 80, 40, 1, 1 };
	uint16_t tmp_6_dim[] = { 1, 80, 16, 12 };
	uint16_t relu_25_tmp_0_dim[] = { 1, 80, 16, 12 };
	uint16_t batch_norm2d_40_b_0_dim[] = { 80 };
	uint16_t batch_norm_40_tmp_2_dim[] = { 1, 80, 8, 6 };
	uint16_t conv2d_40_w_0_dim[] = { 80, 1, 3, 3 };
	uint16_t split_9_tmp_0_dim[] = { 1, 20, 32, 24 };
	uint16_t split_9_tmp_1_dim[] = { 1, 20, 32, 24 };
	uint16_t batch_norm2d_42_b_0_dim[] = { 20 };
	uint16_t conv2d_42_w_0_dim[] = { 20, 20, 1, 1 };
	uint16_t relu_27_tmp_0_dim[] = { 1, 20, 32, 24 };
	uint16_t batch_norm2d_43_b_0_dim[] = { 20 };
	uint16_t batch_norm_43_tmp_2_dim[] = { 1, 20, 32, 24 };
	uint16_t conv2d_43_w_0_dim[] = { 20, 1, 3, 3 };
	uint16_t batch_norm2d_44_b_0_dim[] = { 20 };
	uint16_t conv2d_44_w_0_dim[] = { 20, 20, 1, 1 };
	uint16_t relu_28_tmp_0_dim[] = { 1, 20, 32, 24 };
	uint16_t concat_9_tmp_0_dim[] = { 1, 40, 32, 24 };
	uint16_t shape_10_tmp_0_dim[] = { 4 };
	uint16_t shape_10_tmp_0_slice_0_dim[] = { 1 };
	uint16_t reshape2_18_tmp_0_dim[] = { 1, 2, 20, 32, 24 };
	uint16_t reshape2_18_tmp_1_dim[] = { 0, 1, 40, 32, 24 };
	uint16_t transpose_9_tmp_0_dim[] = { 1, 20, 2, 32, 24 };
	uint16_t transpose_9_tmp_1_dim[] = { 0, 1, 2, 20, 32, 24 };
	uint16_t reshape2_19_tmp_0_dim[] = { 1, 40, 32, 24 };
	uint16_t reshape2_19_tmp_1_dim[] = { 0, 1, 20, 2, 32, 24 };
	uint16_t split_10_tmp_0_dim[] = { 1, 20, 32, 24 };
	uint16_t split_10_tmp_1_dim[] = { 1, 20, 32, 24 };
	uint16_t batch_norm2d_45_b_0_dim[] = { 20 };
	uint16_t conv2d_45_w_0_dim[] = { 20, 20, 1, 1 };
	uint16_t relu_29_tmp_0_dim[] = { 1, 20, 32, 24 };
	uint16_t batch_norm2d_46_b_0_dim[] = { 20 };
	uint16_t batch_norm_46_tmp_2_dim[] = { 1, 20, 32, 24 };
	uint16_t conv2d_46_w_0_dim[] = { 20, 1, 3, 3 };
	uint16_t batch_norm2d_47_b_0_dim[] = { 20 };
	uint16_t conv2d_47_w_0_dim[] = { 20, 20, 1, 1 };
	uint16_t relu_30_tmp_0_dim[] = { 1, 20, 32, 24 };
	uint16_t concat_10_tmp_0_dim[] = { 1, 40, 32, 24 };
	uint16_t shape_11_tmp_0_dim[] = { 4 };
	uint16_t shape_11_tmp_0_slice_0_dim[] = { 1 };
	uint16_t reshape2_20_tmp_0_dim[] = { 1, 2, 20, 32, 24 };
	uint16_t reshape2_20_tmp_1_dim[] = { 0, 1, 40, 32, 24 };
	uint16_t transpose_10_tmp_0_dim[] = { 1, 20, 2, 32, 24 };
	uint16_t transpose_10_tmp_1_dim[] = { 0, 1, 2, 20, 32, 24 };
	uint16_t reshape2_21_tmp_0_dim[] = { 1, 40, 32, 24 };
	uint16_t reshape2_21_tmp_1_dim[] = { 0, 1, 20, 2, 32, 24 };
	uint16_t split_11_tmp_0_dim[] = { 1, 40, 16, 12 };
	uint16_t split_11_tmp_1_dim[] = { 1, 40, 16, 12 };
	uint16_t batch_norm2d_48_b_0_dim[] = { 40 };
	uint16_t conv2d_48_w_0_dim[] = { 40, 40, 1, 1 };
	uint16_t relu_31_tmp_0_dim[] = { 1, 40, 16, 12 };
	uint16_t batch_norm2d_49_b_0_dim[] = { 40 };
	uint16_t batch_norm_49_tmp_2_dim[] = { 1, 40, 16, 12 };
	uint16_t conv2d_49_w_0_dim[] = { 40, 1, 3, 3 };
	uint16_t batch_norm2d_50_b_0_dim[] = { 40 };
	uint16_t conv2d_50_w_0_dim[] = { 40, 40, 1, 1 };
	uint16_t relu_32_tmp_0_dim[] = { 1, 40, 16, 12 };
	uint16_t concat_11_tmp_0_dim[] = { 1, 80, 16, 12 };
	uint16_t shape_12_tmp_0_dim[] = { 4 };
	uint16_t shape_12_tmp_0_slice_0_dim[] = { 1 };
	uint16_t reshape2_22_tmp_0_dim[] = { 1, 2, 40, 16, 12 };
	uint16_t reshape2_22_tmp_1_dim[] = { 0, 1, 80, 16, 12 };
	uint16_t transpose_11_tmp_0_dim[] = { 1, 40, 2, 16, 12 };
	uint16_t transpose_11_tmp_1_dim[] = { 0, 1, 2, 40, 16, 12 };
	uint16_t reshape2_23_tmp_0_dim[] = { 1, 80, 16, 12 };
	uint16_t reshape2_23_tmp_1_dim[] = { 0, 1, 40, 2, 16, 12 };
	uint16_t split_12_tmp_0_dim[] = { 1, 40, 16, 12 };
	uint16_t split_12_tmp_1_dim[] = { 1, 40, 16, 12 };
	uint16_t batch_norm2d_51_b_0_dim[] = { 40 };
	uint16_t conv2d_51_w_0_dim[] = { 40, 40, 1, 1 };
	uint16_t relu_33_tmp_0_dim[] = { 1, 40, 16, 12 };
	uint16_t batch_norm2d_52_b_0_dim[] = { 40 };
	uint16_t batch_norm_52_tmp_2_dim[] = { 1, 40, 16, 12 };
	uint16_t conv2d_52_w_0_dim[] = { 40, 1, 3, 3 };
	uint16_t batch_norm2d_53_b_0_dim[] = { 40 };
	uint16_t conv2d_53_w_0_dim[] = { 40, 40, 1, 1 };
	uint16_t relu_34_tmp_0_dim[] = { 1, 40, 16, 12 };
	uint16_t concat_12_tmp_0_dim[] = { 1, 80, 16, 12 };
	uint16_t shape_13_tmp_0_dim[] = { 4 };
	uint16_t shape_13_tmp_0_slice_0_dim[] = { 1 };
	uint16_t reshape2_24_tmp_0_dim[] = { 1, 2, 40, 16, 12 };
	uint16_t reshape2_24_tmp_1_dim[] = { 0, 1, 80, 16, 12 };
	uint16_t transpose_12_tmp_0_dim[] = { 1, 40, 2, 16, 12 };
	uint16_t transpose_12_tmp_1_dim[] = { 0, 1, 2, 40, 16, 12 };
	uint16_t reshape2_25_tmp_0_dim[] = { 1, 80, 16, 12 };
	uint16_t reshape2_25_tmp_1_dim[] = { 0, 1, 40, 2, 16, 12 };
	uint16_t batch_norm2d_41_b_0_dim[] = { 160 };
	uint16_t conv2d_41_w_0_dim[] = { 160, 80, 1, 1 };
	uint16_t relu_26_tmp_0_dim[] = { 1, 160, 8, 6 };
	uint16_t split_13_tmp_0_dim[] = { 1, 80, 8, 6 };
	uint16_t split_13_tmp_1_dim[] = { 1, 80, 8, 6 };
	uint16_t batch_norm2d_54_b_0_dim[] = { 80 };
	uint16_t conv2d_54_w_0_dim[] = { 80, 80, 1, 1 };
	uint16_t relu_35_tmp_0_dim[] = { 1, 80, 8, 6 };
	uint16_t batch_norm2d_55_b_0_dim[] = { 80 };
	uint16_t batch_norm_55_tmp_2_dim[] = { 1, 80, 8, 6 };
	uint16_t conv2d_55_w_0_dim[] = { 80, 1, 3, 3 };
	uint16_t batch_norm2d_56_b_0_dim[] = { 80 };
	uint16_t conv2d_56_w_0_dim[] = { 80, 80, 1, 1 };
	uint16_t relu_36_tmp_0_dim[] = { 1, 80, 8, 6 };
	uint16_t concat_13_tmp_0_dim[] = { 1, 160, 8, 6 };
	uint16_t shape_14_tmp_0_dim[] = { 4 };
	uint16_t shape_14_tmp_0_slice_0_dim[] = { 1 };
	uint16_t reshape2_26_tmp_0_dim[] = { 1, 2, 80, 8, 6 };
	uint16_t reshape2_26_tmp_1_dim[] = { 0, 1, 160, 8, 6 };
	uint16_t transpose_13_tmp_0_dim[] = { 1, 80, 2, 8, 6 };
	uint16_t transpose_13_tmp_1_dim[] = { 0, 1, 2, 80, 8, 6 };
	uint16_t reshape2_27_tmp_0_dim[] = { 1, 160, 8, 6 };
	uint16_t reshape2_27_tmp_1_dim[] = { 0, 1, 80, 2, 8, 6 };
	uint16_t split_14_tmp_0_dim[] = { 1, 80, 8, 6 };
	uint16_t split_14_tmp_1_dim[] = { 1, 80, 8, 6 };
	uint16_t batch_norm2d_57_b_0_dim[] = { 80 };
	uint16_t conv2d_57_w_0_dim[] = { 80, 80, 1, 1 };
	uint16_t relu_37_tmp_0_dim[] = { 1, 80, 8, 6 };
	uint16_t batch_norm2d_58_b_0_dim[] = { 80 };
	uint16_t batch_norm_58_tmp_2_dim[] = { 1, 80, 8, 6 };
	uint16_t conv2d_58_w_0_dim[] = { 80, 1, 3, 3 };
	uint16_t batch_norm2d_59_b_0_dim[] = { 80 };
	uint16_t conv2d_59_w_0_dim[] = { 80, 80, 1, 1 };
	uint16_t relu_38_tmp_0_dim[] = { 1, 80, 8, 6 };
	uint16_t concat_14_tmp_0_dim[] = { 1, 160, 8, 6 };
	uint16_t shape_15_tmp_0_dim[] = { 4 };
	uint16_t shape_15_tmp_0_slice_0_dim[] = { 1 };
	uint16_t reshape2_28_tmp_0_dim[] = { 1, 2, 80, 8, 6 };
	uint16_t reshape2_28_tmp_1_dim[] = { 0, 1, 160, 8, 6 };
	uint16_t transpose_14_tmp_0_dim[] = { 1, 80, 2, 8, 6 };
	uint16_t transpose_14_tmp_1_dim[] = { 0, 1, 2, 80, 8, 6 };
	uint16_t reshape2_29_tmp_0_dim[] = { 1, 160, 8, 6 };
	uint16_t reshape2_29_tmp_1_dim[] = { 0, 1, 80, 2, 8, 6 };
	uint16_t tmp_8_dim[] = { 1, 40, 32, 24 };
	uint16_t batch_norm2d_60_b_0_dim[] = { 40 };
	uint16_t batch_norm_60_tmp_2_dim[] = { 1, 40, 16, 12 };
	uint16_t conv2d_60_w_0_dim[] = { 40, 80, 1, 1 };
	uint16_t nearest_interp_v2_2_tmp_0_dim[] = { 1, 40, 32, 24 };
	uint16_t tmp_9_dim[] = { 1, 40, 32, 24 };
	uint16_t batch_norm2d_61_b_0_dim[] = { 40 };
	uint16_t batch_norm_61_tmp_2_dim[] = { 1, 40, 8, 6 };
	uint16_t conv2d_61_w_0_dim[] = { 40, 160, 1, 1 };
	uint16_t nearest_interp_v2_3_tmp_0_dim[] = { 1, 40, 32, 24 };
	uint16_t tmp_10_dim[] = { 1, 40, 32, 24 };
	uint16_t relu_39_tmp_0_dim[] = { 1, 40, 32, 24 };
	uint16_t batch_norm2d_62_b_0_dim[] = { 40 };
	uint16_t batch_norm_62_tmp_2_dim[] = { 1, 40, 16, 12 };
	uint16_t conv2d_62_w_0_dim[] = { 40, 1, 3, 3 };
	uint16_t batch_norm2d_63_b_0_dim[] = { 80 };
	uint16_t batch_norm_63_tmp_2_dim[] = { 1, 80, 16, 12 };
	uint16_t conv2d_63_w_0_dim[] = { 80, 40, 1, 1 };
	uint16_t tmp_11_dim[] = { 1, 80, 16, 12 };
	uint16_t tmp_12_dim[] = { 1, 80, 16, 12 };
	uint16_t batch_norm2d_64_b_0_dim[] = { 80 };
	uint16_t batch_norm_64_tmp_2_dim[] = { 1, 80, 8, 6 };
	uint16_t conv2d_64_w_0_dim[] = { 80, 160, 1, 1 };
	uint16_t nearest_interp_v2_4_tmp_0_dim[] = { 1, 80, 16, 12 };
	uint16_t batch_norm2d_65_b_0_dim[] = { 40 };
	uint16_t batch_norm_65_tmp_2_dim[] = { 1, 40, 16, 12 };
	uint16_t conv2d_65_w_0_dim[] = { 40, 1, 3, 3 };
	uint16_t batch_norm2d_66_b_0_dim[] = { 40 };
	uint16_t conv2d_66_w_0_dim[] = { 40, 40, 1, 1 };
	uint16_t relu_41_tmp_0_dim[] = { 1, 40, 16, 12 };
	uint16_t batch_norm2d_67_b_0_dim[] = { 40 };
	uint16_t batch_norm_67_tmp_2_dim[] = { 1, 40, 8, 6 };
	uint16_t conv2d_67_w_0_dim[] = { 40, 1, 3, 3 };
	uint16_t batch_norm2d_68_b_0_dim[] = { 160 };
	uint16_t batch_norm_68_tmp_2_dim[] = { 1, 160, 8, 6 };
	uint16_t conv2d_68_w_0_dim[] = { 160, 40, 1, 1 };
	uint16_t tmp_14_dim[] = { 1, 160, 8, 6 };
	uint16_t batch_norm2d_69_b_0_dim[] = { 80 };
	uint16_t batch_norm_69_tmp_2_dim[] = { 1, 80, 8, 6 };
	uint16_t conv2d_69_w_0_dim[] = { 80, 1, 3, 3 };
	uint16_t batch_norm2d_70_b_0_dim[] = { 160 };
	uint16_t batch_norm_70_tmp_2_dim[] = { 1, 160, 8, 6 };
	uint16_t conv2d_70_w_0_dim[] = { 160, 80, 1, 1 };
	uint16_t tmp_15_dim[] = { 1, 160, 8, 6 };
	uint16_t split_15_tmp_0_dim[] = { 1, 20, 32, 24 };
	uint16_t split_15_tmp_1_dim[] = { 1, 20, 32, 24 };
	uint16_t batch_norm2d_71_b_0_dim[] = { 20 };
	uint16_t conv2d_71_w_0_dim[] = { 20, 20, 1, 1 };
	uint16_t relu_43_tmp_0_dim[] = { 1, 20, 32, 24 };
	uint16_t batch_norm2d_72_b_0_dim[] = { 20 };
	uint16_t batch_norm_72_tmp_2_dim[] = { 1, 20, 32, 24 };
	uint16_t conv2d_72_w_0_dim[] = { 20, 1, 3, 3 };
	uint16_t batch_norm2d_73_b_0_dim[] = { 20 };
	uint16_t conv2d_73_w_0_dim[] = { 20, 20, 1, 1 };
	uint16_t relu_44_tmp_0_dim[] = { 1, 20, 32, 24 };
	uint16_t concat_15_tmp_0_dim[] = { 1, 40, 32, 24 };
	uint16_t shape_16_tmp_0_dim[] = { 4 };
	uint16_t shape_16_tmp_0_slice_0_dim[] = { 1 };
	uint16_t reshape2_30_tmp_0_dim[] = { 1, 2, 20, 32, 24 };
	uint16_t reshape2_30_tmp_1_dim[] = { 0, 1, 40, 32, 24 };
	uint16_t transpose_15_tmp_0_dim[] = { 1, 20, 2, 32, 24 };
	uint16_t transpose_15_tmp_1_dim[] = { 0, 1, 2, 20, 32, 24 };
	uint16_t reshape2_31_tmp_0_dim[] = { 1, 40, 32, 24 };
	uint16_t reshape2_31_tmp_1_dim[] = { 0, 1, 20, 2, 32, 24 };
	uint16_t split_16_tmp_0_dim[] = { 1, 20, 32, 24 };
	uint16_t split_16_tmp_1_dim[] = { 1, 20, 32, 24 };
	uint16_t batch_norm2d_74_b_0_dim[] = { 20 };
	uint16_t conv2d_74_w_0_dim[] = { 20, 20, 1, 1 };
	uint16_t relu_45_tmp_0_dim[] = { 1, 20, 32, 24 };
	uint16_t batch_norm2d_75_b_0_dim[] = { 20 };
	uint16_t batch_norm_75_tmp_2_dim[] = { 1, 20, 32, 24 };
	uint16_t conv2d_75_w_0_dim[] = { 20, 1, 3, 3 };
	uint16_t batch_norm2d_76_b_0_dim[] = { 20 };
	uint16_t conv2d_76_w_0_dim[] = { 20, 20, 1, 1 };
	uint16_t relu_46_tmp_0_dim[] = { 1, 20, 32, 24 };
	uint16_t concat_16_tmp_0_dim[] = { 1, 40, 32, 24 };
	uint16_t shape_17_tmp_0_dim[] = { 4 };
	uint16_t shape_17_tmp_0_slice_0_dim[] = { 1 };
	uint16_t reshape2_32_tmp_0_dim[] = { 1, 2, 20, 32, 24 };
	uint16_t reshape2_32_tmp_1_dim[] = { 0, 1, 40, 32, 24 };
	uint16_t transpose_16_tmp_0_dim[] = { 1, 20, 2, 32, 24 };
	uint16_t transpose_16_tmp_1_dim[] = { 0, 1, 2, 20, 32, 24 };
	uint16_t reshape2_33_tmp_0_dim[] = { 1, 40, 32, 24 };
	uint16_t reshape2_33_tmp_1_dim[] = { 0, 1, 20, 2, 32, 24 };
	uint16_t relu_40_tmp_0_dim[] = { 1, 80, 16, 12 };
	uint16_t split_17_tmp_0_dim[] = { 1, 40, 16, 12 };
	uint16_t split_17_tmp_1_dim[] = { 1, 40, 16, 12 };
	uint16_t batch_norm2d_77_b_0_dim[] = { 40 };
	uint16_t conv2d_77_w_0_dim[] = { 40, 40, 1, 1 };
	uint16_t relu_47_tmp_0_dim[] = { 1, 40, 16, 12 };
	uint16_t batch_norm2d_78_b_0_dim[] = { 40 };
	uint16_t batch_norm_78_tmp_2_dim[] = { 1, 40, 16, 12 };
	uint16_t conv2d_78_w_0_dim[] = { 40, 1, 3, 3 };
	uint16_t batch_norm2d_79_b_0_dim[] = { 40 };
	uint16_t conv2d_79_w_0_dim[] = { 40, 40, 1, 1 };
	uint16_t relu_48_tmp_0_dim[] = { 1, 40, 16, 12 };
	uint16_t concat_17_tmp_0_dim[] = { 1, 80, 16, 12 };
	uint16_t shape_18_tmp_0_dim[] = { 4 };
	uint16_t shape_18_tmp_0_slice_0_dim[] = { 1 };
	uint16_t reshape2_34_tmp_0_dim[] = { 1, 2, 40, 16, 12 };
	uint16_t reshape2_34_tmp_1_dim[] = { 0, 1, 80, 16, 12 };
	uint16_t transpose_17_tmp_0_dim[] = { 1, 40, 2, 16, 12 };
	uint16_t transpose_17_tmp_1_dim[] = { 0, 1, 2, 40, 16, 12 };
	uint16_t reshape2_35_tmp_0_dim[] = { 1, 80, 16, 12 };
	uint16_t reshape2_35_tmp_1_dim[] = { 0, 1, 40, 2, 16, 12 };
	uint16_t split_18_tmp_0_dim[] = { 1, 40, 16, 12 };
	uint16_t split_18_tmp_1_dim[] = { 1, 40, 16, 12 };
	uint16_t batch_norm2d_80_b_0_dim[] = { 40 };
	uint16_t conv2d_80_w_0_dim[] = { 40, 40, 1, 1 };
	uint16_t relu_49_tmp_0_dim[] = { 1, 40, 16, 12 };
	uint16_t batch_norm2d_81_b_0_dim[] = { 40 };
	uint16_t batch_norm_81_tmp_2_dim[] = { 1, 40, 16, 12 };
	uint16_t conv2d_81_w_0_dim[] = { 40, 1, 3, 3 };
	uint16_t batch_norm2d_82_b_0_dim[] = { 40 };
	uint16_t conv2d_82_w_0_dim[] = { 40, 40, 1, 1 };
	uint16_t relu_50_tmp_0_dim[] = { 1, 40, 16, 12 };
	uint16_t concat_18_tmp_0_dim[] = { 1, 80, 16, 12 };
	uint16_t shape_19_tmp_0_dim[] = { 4 };
	uint16_t shape_19_tmp_0_slice_0_dim[] = { 1 };
	uint16_t reshape2_36_tmp_0_dim[] = { 1, 2, 40, 16, 12 };
	uint16_t reshape2_36_tmp_1_dim[] = { 0, 1, 80, 16, 12 };
	uint16_t transpose_18_tmp_0_dim[] = { 1, 40, 2, 16, 12 };
	uint16_t transpose_18_tmp_1_dim[] = { 0, 1, 2, 40, 16, 12 };
	uint16_t reshape2_37_tmp_0_dim[] = { 1, 80, 16, 12 };
	uint16_t reshape2_37_tmp_1_dim[] = { 0, 1, 40, 2, 16, 12 };
	uint16_t relu_42_tmp_0_dim[] = { 1, 160, 8, 6 };
	uint16_t split_19_tmp_0_dim[] = { 1, 80, 8, 6 };
	uint16_t split_19_tmp_1_dim[] = { 1, 80, 8, 6 };
	uint16_t batch_norm2d_83_b_0_dim[] = { 80 };
	uint16_t conv2d_83_w_0_dim[] = { 80, 80, 1, 1 };
	uint16_t relu_51_tmp_0_dim[] = { 1, 80, 8, 6 };
	uint16_t batch_norm2d_84_b_0_dim[] = { 80 };
	uint16_t batch_norm_84_tmp_2_dim[] = { 1, 80, 8, 6 };
	uint16_t conv2d_84_w_0_dim[] = { 80, 1, 3, 3 };
	uint16_t batch_norm2d_85_b_0_dim[] = { 80 };
	uint16_t conv2d_85_w_0_dim[] = { 80, 80, 1, 1 };
	uint16_t relu_52_tmp_0_dim[] = { 1, 80, 8, 6 };
	uint16_t concat_19_tmp_0_dim[] = { 1, 160, 8, 6 };
	uint16_t shape_20_tmp_0_dim[] = { 4 };
	uint16_t shape_20_tmp_0_slice_0_dim[] = { 1 };
	uint16_t reshape2_38_tmp_0_dim[] = { 1, 2, 80, 8, 6 };
	uint16_t reshape2_38_tmp_1_dim[] = { 0, 1, 160, 8, 6 };
	uint16_t transpose_19_tmp_0_dim[] = { 1, 80, 2, 8, 6 };
	uint16_t transpose_19_tmp_1_dim[] = { 0, 1, 2, 80, 8, 6 };
	uint16_t reshape2_39_tmp_0_dim[] = { 1, 160, 8, 6 };
	uint16_t reshape2_39_tmp_1_dim[] = { 0, 1, 80, 2, 8, 6 };
	uint16_t split_20_tmp_0_dim[] = { 1, 80, 8, 6 };
	uint16_t split_20_tmp_1_dim[] = { 1, 80, 8, 6 };
	uint16_t batch_norm2d_86_b_0_dim[] = { 80 };
	uint16_t conv2d_86_w_0_dim[] = { 80, 80, 1, 1 };
	uint16_t relu_53_tmp_0_dim[] = { 1, 80, 8, 6 };
	uint16_t batch_norm2d_87_b_0_dim[] = { 80 };
	uint16_t batch_norm_87_tmp_2_dim[] = { 1, 80, 8, 6 };
	uint16_t conv2d_87_w_0_dim[] = { 80, 1, 3, 3 };
	uint16_t batch_norm2d_88_b_0_dim[] = { 80 };
	uint16_t conv2d_88_w_0_dim[] = { 80, 80, 1, 1 };
	uint16_t relu_54_tmp_0_dim[] = { 1, 80, 8, 6 };
	uint16_t concat_20_tmp_0_dim[] = { 1, 160, 8, 6 };
	uint16_t shape_21_tmp_0_dim[] = { 4 };
	uint16_t shape_21_tmp_0_slice_0_dim[] = { 1 };
	uint16_t reshape2_40_tmp_0_dim[] = { 1, 2, 80, 8, 6 };
	uint16_t reshape2_40_tmp_1_dim[] = { 0, 1, 160, 8, 6 };
	uint16_t transpose_20_tmp_0_dim[] = { 1, 80, 2, 8, 6 };
	uint16_t transpose_20_tmp_1_dim[] = { 0, 1, 2, 80, 8, 6 };
	uint16_t reshape2_41_tmp_0_dim[] = { 1, 160, 8, 6 };
	uint16_t reshape2_41_tmp_1_dim[] = { 0, 1, 80, 2, 8, 6 };
	uint16_t tmp_17_dim[] = { 1, 40, 32, 24 };
	uint16_t batch_norm2d_89_b_0_dim[] = { 40 };
	uint16_t batch_norm_89_tmp_2_dim[] = { 1, 40, 16, 12 };
	uint16_t conv2d_89_w_0_dim[] = { 40, 80, 1, 1 };
	uint16_t nearest_interp_v2_5_tmp_0_dim[] = { 1, 40, 32, 24 };
	uint16_t tmp_18_dim[] = { 1, 40, 32, 24 };
	uint16_t batch_norm2d_90_b_0_dim[] = { 40 };
	uint16_t batch_norm_90_tmp_2_dim[] = { 1, 40, 8, 6 };
	uint16_t conv2d_90_w_0_dim[] = { 40, 160, 1, 1 };
	uint16_t nearest_interp_v2_6_tmp_0_dim[] = { 1, 40, 32, 24 };
	uint16_t tmp_19_dim[] = { 1, 40, 32, 24 };
	uint16_t relu_55_tmp_0_dim[] = { 1, 40, 32, 24 };
	uint16_t batch_norm2d_91_b_0_dim[] = { 40 };
	uint16_t batch_norm_91_tmp_2_dim[] = { 1, 40, 16, 12 };
	uint16_t conv2d_91_w_0_dim[] = { 40, 1, 3, 3 };
	uint16_t batch_norm2d_92_b_0_dim[] = { 80 };
	uint16_t batch_norm_92_tmp_2_dim[] = { 1, 80, 16, 12 };
	uint16_t conv2d_92_w_0_dim[] = { 80, 40, 1, 1 };
	uint16_t tmp_20_dim[] = { 1, 80, 16, 12 };
	uint16_t tmp_21_dim[] = { 1, 80, 16, 12 };
	uint16_t batch_norm2d_93_b_0_dim[] = { 80 };
	uint16_t batch_norm_93_tmp_2_dim[] = { 1, 80, 8, 6 };
	uint16_t conv2d_93_w_0_dim[] = { 80, 160, 1, 1 };
	uint16_t nearest_interp_v2_7_tmp_0_dim[] = { 1, 80, 16, 12 };
	uint16_t batch_norm2d_94_b_0_dim[] = { 40 };
	uint16_t batch_norm_94_tmp_2_dim[] = { 1, 40, 16, 12 };
	uint16_t conv2d_94_w_0_dim[] = { 40, 1, 3, 3 };
	uint16_t batch_norm2d_95_b_0_dim[] = { 40 };
	uint16_t conv2d_95_w_0_dim[] = { 40, 40, 1, 1 };
	uint16_t relu_57_tmp_0_dim[] = { 1, 40, 16, 12 };
	uint16_t batch_norm2d_96_b_0_dim[] = { 40 };
	uint16_t batch_norm_96_tmp_2_dim[] = { 1, 40, 8, 6 };
	uint16_t conv2d_96_w_0_dim[] = { 40, 1, 3, 3 };
	uint16_t batch_norm2d_97_b_0_dim[] = { 160 };
	uint16_t batch_norm_97_tmp_2_dim[] = { 1, 160, 8, 6 };
	uint16_t conv2d_97_w_0_dim[] = { 160, 40, 1, 1 };
	uint16_t tmp_23_dim[] = { 1, 160, 8, 6 };
	uint16_t batch_norm2d_98_b_0_dim[] = { 80 };
	uint16_t batch_norm_98_tmp_2_dim[] = { 1, 80, 8, 6 };
	uint16_t conv2d_98_w_0_dim[] = { 80, 1, 3, 3 };
	uint16_t batch_norm2d_99_b_0_dim[] = { 160 };
	uint16_t batch_norm_99_tmp_2_dim[] = { 1, 160, 8, 6 };
	uint16_t conv2d_99_w_0_dim[] = { 160, 80, 1, 1 };
	uint16_t tmp_24_dim[] = { 1, 160, 8, 6 };
	uint16_t split_21_tmp_0_dim[] = { 1, 20, 32, 24 };
	uint16_t split_21_tmp_1_dim[] = { 1, 20, 32, 24 };
	uint16_t batch_norm2d_100_b_0_dim[] = { 20 };
	uint16_t conv2d_100_w_0_dim[] = { 20, 20, 1, 1 };
	uint16_t relu_59_tmp_0_dim[] = { 1, 20, 32, 24 };
	uint16_t batch_norm2d_101_b_0_dim[] = { 20 };
	uint16_t batch_norm_101_tmp_2_dim[] = { 1, 20, 32, 24 };
	uint16_t conv2d_101_w_0_dim[] = { 20, 1, 3, 3 };
	uint16_t batch_norm2d_102_b_0_dim[] = { 20 };
	uint16_t conv2d_102_w_0_dim[] = { 20, 20, 1, 1 };
	uint16_t relu_60_tmp_0_dim[] = { 1, 20, 32, 24 };
	uint16_t concat_21_tmp_0_dim[] = { 1, 40, 32, 24 };
	uint16_t shape_22_tmp_0_dim[] = { 4 };
	uint16_t shape_22_tmp_0_slice_0_dim[] = { 1 };
	uint16_t reshape2_42_tmp_0_dim[] = { 1, 2, 20, 32, 24 };
	uint16_t reshape2_42_tmp_1_dim[] = { 0, 1, 40, 32, 24 };
	uint16_t transpose_21_tmp_0_dim[] = { 1, 20, 2, 32, 24 };
	uint16_t transpose_21_tmp_1_dim[] = { 0, 1, 2, 20, 32, 24 };
	uint16_t reshape2_43_tmp_0_dim[] = { 1, 40, 32, 24 };
	uint16_t reshape2_43_tmp_1_dim[] = { 0, 1, 20, 2, 32, 24 };
	uint16_t split_22_tmp_0_dim[] = { 1, 20, 32, 24 };
	uint16_t split_22_tmp_1_dim[] = { 1, 20, 32, 24 };
	uint16_t batch_norm2d_103_b_0_dim[] = { 20 };
	uint16_t conv2d_103_w_0_dim[] = { 20, 20, 1, 1 };
	uint16_t relu_61_tmp_0_dim[] = { 1, 20, 32, 24 };
	uint16_t batch_norm2d_104_b_0_dim[] = { 20 };
	uint16_t batch_norm_104_tmp_2_dim[] = { 1, 20, 32, 24 };
	uint16_t conv2d_104_w_0_dim[] = { 20, 1, 3, 3 };
	uint16_t batch_norm2d_105_b_0_dim[] = { 20 };
	uint16_t conv2d_105_w_0_dim[] = { 20, 20, 1, 1 };
	uint16_t relu_62_tmp_0_dim[] = { 1, 20, 32, 24 };
	uint16_t concat_22_tmp_0_dim[] = { 1, 40, 32, 24 };
	uint16_t shape_23_tmp_0_dim[] = { 4 };
	uint16_t shape_23_tmp_0_slice_0_dim[] = { 1 };
	uint16_t reshape2_44_tmp_0_dim[] = { 1, 2, 20, 32, 24 };
	uint16_t reshape2_44_tmp_1_dim[] = { 0, 1, 40, 32, 24 };
	uint16_t transpose_22_tmp_0_dim[] = { 1, 20, 2, 32, 24 };
	uint16_t transpose_22_tmp_1_dim[] = { 0, 1, 2, 20, 32, 24 };
	uint16_t reshape2_45_tmp_0_dim[] = { 1, 40, 32, 24 };
	uint16_t reshape2_45_tmp_1_dim[] = { 0, 1, 20, 2, 32, 24 };
	uint16_t relu_56_tmp_0_dim[] = { 1, 80, 16, 12 };
	uint16_t split_23_tmp_0_dim[] = { 1, 40, 16, 12 };
	uint16_t split_23_tmp_1_dim[] = { 1, 40, 16, 12 };
	uint16_t batch_norm2d_106_b_0_dim[] = { 40 };
	uint16_t conv2d_106_w_0_dim[] = { 40, 40, 1, 1 };
	uint16_t relu_63_tmp_0_dim[] = { 1, 40, 16, 12 };
	uint16_t batch_norm2d_107_b_0_dim[] = { 40 };
	uint16_t batch_norm_107_tmp_2_dim[] = { 1, 40, 16, 12 };
	uint16_t conv2d_107_w_0_dim[] = { 40, 1, 3, 3 };
	uint16_t batch_norm2d_108_b_0_dim[] = { 40 };
	uint16_t conv2d_108_w_0_dim[] = { 40, 40, 1, 1 };
	uint16_t relu_64_tmp_0_dim[] = { 1, 40, 16, 12 };
	uint16_t concat_23_tmp_0_dim[] = { 1, 80, 16, 12 };
	uint16_t shape_24_tmp_0_dim[] = { 4 };
	uint16_t shape_24_tmp_0_slice_0_dim[] = { 1 };
	uint16_t reshape2_46_tmp_0_dim[] = { 1, 2, 40, 16, 12 };
	uint16_t reshape2_46_tmp_1_dim[] = { 0, 1, 80, 16, 12 };
	uint16_t transpose_23_tmp_0_dim[] = { 1, 40, 2, 16, 12 };
	uint16_t transpose_23_tmp_1_dim[] = { 0, 1, 2, 40, 16, 12 };
	uint16_t reshape2_47_tmp_0_dim[] = { 1, 80, 16, 12 };
	uint16_t reshape2_47_tmp_1_dim[] = { 0, 1, 40, 2, 16, 12 };
	uint16_t split_24_tmp_0_dim[] = { 1, 40, 16, 12 };
	uint16_t split_24_tmp_1_dim[] = { 1, 40, 16, 12 };
	uint16_t batch_norm2d_109_b_0_dim[] = { 40 };
	uint16_t conv2d_109_w_0_dim[] = { 40, 40, 1, 1 };
	uint16_t relu_65_tmp_0_dim[] = { 1, 40, 16, 12 };
	uint16_t batch_norm2d_110_b_0_dim[] = { 40 };
	uint16_t batch_norm_110_tmp_2_dim[] = { 1, 40, 16, 12 };
	uint16_t conv2d_110_w_0_dim[] = { 40, 1, 3, 3 };
	uint16_t batch_norm2d_111_b_0_dim[] = { 40 };
	uint16_t conv2d_111_w_0_dim[] = { 40, 40, 1, 1 };
	uint16_t relu_66_tmp_0_dim[] = { 1, 40, 16, 12 };
	uint16_t concat_24_tmp_0_dim[] = { 1, 80, 16, 12 };
	uint16_t shape_25_tmp_0_dim[] = { 4 };
	uint16_t shape_25_tmp_0_slice_0_dim[] = { 1 };
	uint16_t reshape2_48_tmp_0_dim[] = { 1, 2, 40, 16, 12 };
	uint16_t reshape2_48_tmp_1_dim[] = { 0, 1, 80, 16, 12 };
	uint16_t transpose_24_tmp_0_dim[] = { 1, 40, 2, 16, 12 };
	uint16_t transpose_24_tmp_1_dim[] = { 0, 1, 2, 40, 16, 12 };
	uint16_t reshape2_49_tmp_0_dim[] = { 1, 80, 16, 12 };
	uint16_t reshape2_49_tmp_1_dim[] = { 0, 1, 40, 2, 16, 12 };
	uint16_t relu_58_tmp_0_dim[] = { 1, 160, 8, 6 };
	uint16_t split_25_tmp_0_dim[] = { 1, 80, 8, 6 };
	uint16_t split_25_tmp_1_dim[] = { 1, 80, 8, 6 };
	uint16_t batch_norm2d_112_b_0_dim[] = { 80 };
	uint16_t conv2d_112_w_0_dim[] = { 80, 80, 1, 1 };
	uint16_t relu_67_tmp_0_dim[] = { 1, 80, 8, 6 };
	uint16_t batch_norm2d_113_b_0_dim[] = { 80 };
	uint16_t batch_norm_113_tmp_2_dim[] = { 1, 80, 8, 6 };
	uint16_t conv2d_113_w_0_dim[] = { 80, 1, 3, 3 };
	uint16_t batch_norm2d_114_b_0_dim[] = { 80 };
	uint16_t conv2d_114_w_0_dim[] = { 80, 80, 1, 1 };
	uint16_t relu_68_tmp_0_dim[] = { 1, 80, 8, 6 };
	uint16_t concat_25_tmp_0_dim[] = { 1, 160, 8, 6 };
	uint16_t shape_26_tmp_0_dim[] = { 4 };
	uint16_t shape_26_tmp_0_slice_0_dim[] = { 1 };
	uint16_t reshape2_50_tmp_0_dim[] = { 1, 2, 80, 8, 6 };
	uint16_t reshape2_50_tmp_1_dim[] = { 0, 1, 160, 8, 6 };
	uint16_t transpose_25_tmp_0_dim[] = { 1, 80, 2, 8, 6 };
	uint16_t transpose_25_tmp_1_dim[] = { 0, 1, 2, 80, 8, 6 };
	uint16_t reshape2_51_tmp_0_dim[] = { 1, 160, 8, 6 };
	uint16_t reshape2_51_tmp_1_dim[] = { 0, 1, 80, 2, 8, 6 };
	uint16_t split_26_tmp_0_dim[] = { 1, 80, 8, 6 };
	uint16_t split_26_tmp_1_dim[] = { 1, 80, 8, 6 };
	uint16_t batch_norm2d_115_b_0_dim[] = { 80 };
	uint16_t conv2d_115_w_0_dim[] = { 80, 80, 1, 1 };
	uint16_t relu_69_tmp_0_dim[] = { 1, 80, 8, 6 };
	uint16_t batch_norm2d_116_b_0_dim[] = { 80 };
	uint16_t batch_norm_116_tmp_2_dim[] = { 1, 80, 8, 6 };
	uint16_t conv2d_116_w_0_dim[] = { 80, 1, 3, 3 };
	uint16_t batch_norm2d_117_b_0_dim[] = { 80 };
	uint16_t conv2d_117_w_0_dim[] = { 80, 80, 1, 1 };
	uint16_t relu_70_tmp_0_dim[] = { 1, 80, 8, 6 };
	uint16_t concat_26_tmp_0_dim[] = { 1, 160, 8, 6 };
	uint16_t shape_27_tmp_0_dim[] = { 4 };
	uint16_t shape_27_tmp_0_slice_0_dim[] = { 1 };
	uint16_t reshape2_52_tmp_0_dim[] = { 1, 2, 80, 8, 6 };
	uint16_t reshape2_52_tmp_1_dim[] = { 0, 1, 160, 8, 6 };
	uint16_t transpose_26_tmp_0_dim[] = { 1, 80, 2, 8, 6 };
	uint16_t transpose_26_tmp_1_dim[] = { 0, 1, 2, 80, 8, 6 };
	uint16_t reshape2_53_tmp_0_dim[] = { 1, 160, 8, 6 };
	uint16_t reshape2_53_tmp_1_dim[] = { 0, 1, 80, 2, 8, 6 };
	uint16_t tmp_26_dim[] = { 1, 40, 32, 24 };
	uint16_t batch_norm2d_118_b_0_dim[] = { 40 };
	uint16_t batch_norm_118_tmp_2_dim[] = { 1, 40, 16, 12 };
	uint16_t conv2d_118_w_0_dim[] = { 40, 80, 1, 1 };
	uint16_t nearest_interp_v2_8_tmp_0_dim[] = { 1, 40, 32, 24 };
	uint16_t tmp_27_dim[] = { 1, 40, 32, 24 };
	uint16_t batch_norm2d_119_b_0_dim[] = { 40 };
	uint16_t batch_norm_119_tmp_2_dim[] = { 1, 40, 8, 6 };
	uint16_t conv2d_119_w_0_dim[] = { 40, 160, 1, 1 };
	uint16_t nearest_interp_v2_9_tmp_0_dim[] = { 1, 40, 32, 24 };
	uint16_t tmp_28_dim[] = { 1, 40, 32, 24 };
	uint16_t relu_71_tmp_0_dim[] = { 1, 40, 32, 24 };
	uint16_t batch_norm2d_120_b_0_dim[] = { 40 };
	uint16_t batch_norm_120_tmp_2_dim[] = { 1, 40, 16, 12 };
	uint16_t conv2d_120_w_0_dim[] = { 40, 1, 3, 3 };
	uint16_t batch_norm2d_121_b_0_dim[] = { 80 };
	uint16_t batch_norm_121_tmp_2_dim[] = { 1, 80, 16, 12 };
	uint16_t conv2d_121_w_0_dim[] = { 80, 40, 1, 1 };
	uint16_t tmp_29_dim[] = { 1, 80, 16, 12 };
	uint16_t tmp_30_dim[] = { 1, 80, 16, 12 };
	uint16_t batch_norm2d_122_b_0_dim[] = { 80 };
	uint16_t batch_norm_122_tmp_2_dim[] = { 1, 80, 8, 6 };
	uint16_t conv2d_122_w_0_dim[] = { 80, 160, 1, 1 };
	uint16_t nearest_interp_v2_10_tmp_0_dim[] = { 1, 80, 16, 12 };
	uint16_t batch_norm2d_123_b_0_dim[] = { 40 };
	uint16_t batch_norm_123_tmp_2_dim[] = { 1, 40, 16, 12 };
	uint16_t conv2d_123_w_0_dim[] = { 40, 1, 3, 3 };
	uint16_t batch_norm2d_124_b_0_dim[] = { 40 };
	uint16_t conv2d_124_w_0_dim[] = { 40, 40, 1, 1 };
	uint16_t relu_73_tmp_0_dim[] = { 1, 40, 16, 12 };
	uint16_t batch_norm2d_125_b_0_dim[] = { 40 };
	uint16_t batch_norm_125_tmp_2_dim[] = { 1, 40, 8, 6 };
	uint16_t conv2d_125_w_0_dim[] = { 40, 1, 3, 3 };
	uint16_t batch_norm2d_126_b_0_dim[] = { 160 };
	uint16_t batch_norm_126_tmp_2_dim[] = { 1, 160, 8, 6 };
	uint16_t conv2d_126_w_0_dim[] = { 160, 40, 1, 1 };
	uint16_t tmp_32_dim[] = { 1, 160, 8, 6 };
	uint16_t batch_norm2d_127_b_0_dim[] = { 80 };
	uint16_t batch_norm_127_tmp_2_dim[] = { 1, 80, 8, 6 };
	uint16_t conv2d_127_w_0_dim[] = { 80, 1, 3, 3 };
	uint16_t batch_norm2d_128_b_0_dim[] = { 160 };
	uint16_t batch_norm_128_tmp_2_dim[] = { 1, 160, 8, 6 };
	uint16_t conv2d_128_w_0_dim[] = { 160, 80, 1, 1 };
	uint16_t tmp_33_dim[] = { 1, 160, 8, 6 };
	uint16_t split_27_tmp_0_dim[] = { 1, 20, 32, 24 };
	uint16_t split_27_tmp_1_dim[] = { 1, 20, 32, 24 };
	uint16_t batch_norm2d_129_b_0_dim[] = { 20 };
	uint16_t conv2d_129_w_0_dim[] = { 20, 20, 1, 1 };
	uint16_t relu_75_tmp_0_dim[] = { 1, 20, 32, 24 };
	uint16_t batch_norm2d_130_b_0_dim[] = { 20 };
	uint16_t batch_norm_130_tmp_2_dim[] = { 1, 20, 32, 24 };
	uint16_t conv2d_130_w_0_dim[] = { 20, 1, 3, 3 };
	uint16_t batch_norm2d_131_b_0_dim[] = { 20 };
	uint16_t conv2d_131_w_0_dim[] = { 20, 20, 1, 1 };
	uint16_t relu_76_tmp_0_dim[] = { 1, 20, 32, 24 };
	uint16_t concat_27_tmp_0_dim[] = { 1, 40, 32, 24 };
	uint16_t shape_28_tmp_0_dim[] = { 4 };
	uint16_t shape_28_tmp_0_slice_0_dim[] = { 1 };
	uint16_t reshape2_54_tmp_0_dim[] = { 1, 2, 20, 32, 24 };
	uint16_t reshape2_54_tmp_1_dim[] = { 0, 1, 40, 32, 24 };
	uint16_t transpose_27_tmp_0_dim[] = { 1, 20, 2, 32, 24 };
	uint16_t transpose_27_tmp_1_dim[] = { 0, 1, 2, 20, 32, 24 };
	uint16_t reshape2_55_tmp_0_dim[] = { 1, 40, 32, 24 };
	uint16_t reshape2_55_tmp_1_dim[] = { 0, 1, 20, 2, 32, 24 };
	uint16_t split_28_tmp_0_dim[] = { 1, 20, 32, 24 };
	uint16_t split_28_tmp_1_dim[] = { 1, 20, 32, 24 };
	uint16_t batch_norm2d_132_b_0_dim[] = { 20 };
	uint16_t conv2d_132_w_0_dim[] = { 20, 20, 1, 1 };
	uint16_t relu_77_tmp_0_dim[] = { 1, 20, 32, 24 };
	uint16_t batch_norm2d_133_b_0_dim[] = { 20 };
	uint16_t batch_norm_133_tmp_2_dim[] = { 1, 20, 32, 24 };
	uint16_t conv2d_133_w_0_dim[] = { 20, 1, 3, 3 };
	uint16_t batch_norm2d_134_b_0_dim[] = { 20 };
	uint16_t conv2d_134_w_0_dim[] = { 20, 20, 1, 1 };
	uint16_t relu_78_tmp_0_dim[] = { 1, 20, 32, 24 };
	uint16_t concat_28_tmp_0_dim[] = { 1, 40, 32, 24 };
	uint16_t shape_29_tmp_0_dim[] = { 4 };
	uint16_t shape_29_tmp_0_slice_0_dim[] = { 1 };
	uint16_t reshape2_56_tmp_0_dim[] = { 1, 2, 20, 32, 24 };
	uint16_t reshape2_56_tmp_1_dim[] = { 0, 1, 40, 32, 24 };
	uint16_t transpose_28_tmp_0_dim[] = { 1, 20, 2, 32, 24 };
	uint16_t transpose_28_tmp_1_dim[] = { 0, 1, 2, 20, 32, 24 };
	uint16_t reshape2_57_tmp_0_dim[] = { 1, 40, 32, 24 };
	uint16_t reshape2_57_tmp_1_dim[] = { 0, 1, 20, 2, 32, 24 };
	uint16_t relu_72_tmp_0_dim[] = { 1, 80, 16, 12 };
	uint16_t split_29_tmp_0_dim[] = { 1, 40, 16, 12 };
	uint16_t split_29_tmp_1_dim[] = { 1, 40, 16, 12 };
	uint16_t batch_norm2d_135_b_0_dim[] = { 40 };
	uint16_t conv2d_135_w_0_dim[] = { 40, 40, 1, 1 };
	uint16_t relu_79_tmp_0_dim[] = { 1, 40, 16, 12 };
	uint16_t batch_norm2d_136_b_0_dim[] = { 40 };
	uint16_t batch_norm_136_tmp_2_dim[] = { 1, 40, 16, 12 };
	uint16_t conv2d_136_w_0_dim[] = { 40, 1, 3, 3 };
	uint16_t batch_norm2d_137_b_0_dim[] = { 40 };
	uint16_t conv2d_137_w_0_dim[] = { 40, 40, 1, 1 };
	uint16_t relu_80_tmp_0_dim[] = { 1, 40, 16, 12 };
	uint16_t concat_29_tmp_0_dim[] = { 1, 80, 16, 12 };
	uint16_t shape_30_tmp_0_dim[] = { 4 };
	uint16_t shape_30_tmp_0_slice_0_dim[] = { 1 };
	uint16_t reshape2_58_tmp_0_dim[] = { 1, 2, 40, 16, 12 };
	uint16_t reshape2_58_tmp_1_dim[] = { 0, 1, 80, 16, 12 };
	uint16_t transpose_29_tmp_0_dim[] = { 1, 40, 2, 16, 12 };
	uint16_t transpose_29_tmp_1_dim[] = { 0, 1, 2, 40, 16, 12 };
	uint16_t reshape2_59_tmp_0_dim[] = { 1, 80, 16, 12 };
	uint16_t reshape2_59_tmp_1_dim[] = { 0, 1, 40, 2, 16, 12 };
	uint16_t split_30_tmp_0_dim[] = { 1, 40, 16, 12 };
	uint16_t split_30_tmp_1_dim[] = { 1, 40, 16, 12 };
	uint16_t batch_norm2d_138_b_0_dim[] = { 40 };
	uint16_t conv2d_138_w_0_dim[] = { 40, 40, 1, 1 };
	uint16_t relu_81_tmp_0_dim[] = { 1, 40, 16, 12 };
	uint16_t batch_norm2d_139_b_0_dim[] = { 40 };
	uint16_t batch_norm_139_tmp_2_dim[] = { 1, 40, 16, 12 };
	uint16_t conv2d_139_w_0_dim[] = { 40, 1, 3, 3 };
	uint16_t batch_norm2d_140_b_0_dim[] = { 40 };
	uint16_t conv2d_140_w_0_dim[] = { 40, 40, 1, 1 };
	uint16_t relu_82_tmp_0_dim[] = { 1, 40, 16, 12 };
	uint16_t concat_30_tmp_0_dim[] = { 1, 80, 16, 12 };
	uint16_t shape_31_tmp_0_dim[] = { 4 };
	uint16_t shape_31_tmp_0_slice_0_dim[] = { 1 };
	uint16_t reshape2_60_tmp_0_dim[] = { 1, 2, 40, 16, 12 };
	uint16_t reshape2_60_tmp_1_dim[] = { 0, 1, 80, 16, 12 };
	uint16_t transpose_30_tmp_0_dim[] = { 1, 40, 2, 16, 12 };
	uint16_t transpose_30_tmp_1_dim[] = { 0, 1, 2, 40, 16, 12 };
	uint16_t reshape2_61_tmp_0_dim[] = { 1, 80, 16, 12 };
	uint16_t reshape2_61_tmp_1_dim[] = { 0, 1, 40, 2, 16, 12 };
	uint16_t relu_74_tmp_0_dim[] = { 1, 160, 8, 6 };
	uint16_t split_31_tmp_0_dim[] = { 1, 80, 8, 6 };
	uint16_t split_31_tmp_1_dim[] = { 1, 80, 8, 6 };
	uint16_t batch_norm2d_141_b_0_dim[] = { 80 };
	uint16_t conv2d_141_w_0_dim[] = { 80, 80, 1, 1 };
	uint16_t relu_83_tmp_0_dim[] = { 1, 80, 8, 6 };
	uint16_t batch_norm2d_142_b_0_dim[] = { 80 };
	uint16_t batch_norm_142_tmp_2_dim[] = { 1, 80, 8, 6 };
	uint16_t conv2d_142_w_0_dim[] = { 80, 1, 3, 3 };
	uint16_t batch_norm2d_143_b_0_dim[] = { 80 };
	uint16_t conv2d_143_w_0_dim[] = { 80, 80, 1, 1 };
	uint16_t relu_84_tmp_0_dim[] = { 1, 80, 8, 6 };
	uint16_t concat_31_tmp_0_dim[] = { 1, 160, 8, 6 };
	uint16_t shape_32_tmp_0_dim[] = { 4 };
	uint16_t shape_32_tmp_0_slice_0_dim[] = { 1 };
	uint16_t reshape2_62_tmp_0_dim[] = { 1, 2, 80, 8, 6 };
	uint16_t reshape2_62_tmp_1_dim[] = { 0, 1, 160, 8, 6 };
	uint16_t transpose_31_tmp_0_dim[] = { 1, 80, 2, 8, 6 };
	uint16_t transpose_31_tmp_1_dim[] = { 0, 1, 2, 80, 8, 6 };
	uint16_t reshape2_63_tmp_0_dim[] = { 1, 160, 8, 6 };
	uint16_t reshape2_63_tmp_1_dim[] = { 0, 1, 80, 2, 8, 6 };
	uint16_t split_32_tmp_0_dim[] = { 1, 80, 8, 6 };
	uint16_t split_32_tmp_1_dim[] = { 1, 80, 8, 6 };
	uint16_t batch_norm2d_144_b_0_dim[] = { 80 };
	uint16_t conv2d_144_w_0_dim[] = { 80, 80, 1, 1 };
	uint16_t relu_85_tmp_0_dim[] = { 1, 80, 8, 6 };
	uint16_t batch_norm2d_145_b_0_dim[] = { 80 };
	uint16_t batch_norm_145_tmp_2_dim[] = { 1, 80, 8, 6 };
	uint16_t conv2d_145_w_0_dim[] = { 80, 1, 3, 3 };
	uint16_t batch_norm2d_146_b_0_dim[] = { 80 };
	uint16_t conv2d_146_w_0_dim[] = { 80, 80, 1, 1 };
	uint16_t relu_86_tmp_0_dim[] = { 1, 80, 8, 6 };
	uint16_t concat_32_tmp_0_dim[] = { 1, 160, 8, 6 };
	uint16_t shape_33_tmp_0_dim[] = { 4 };
	uint16_t shape_33_tmp_0_slice_0_dim[] = { 1 };
	uint16_t reshape2_64_tmp_0_dim[] = { 1, 2, 80, 8, 6 };
	uint16_t reshape2_64_tmp_1_dim[] = { 0, 1, 160, 8, 6 };
	uint16_t transpose_32_tmp_0_dim[] = { 1, 80, 2, 8, 6 };
	uint16_t transpose_32_tmp_1_dim[] = { 0, 1, 2, 80, 8, 6 };
	uint16_t reshape2_65_tmp_0_dim[] = { 1, 160, 8, 6 };
	uint16_t reshape2_65_tmp_1_dim[] = { 0, 1, 80, 2, 8, 6 };
	uint16_t tmp_35_dim[] = { 1, 40, 32, 24 };
	uint16_t batch_norm2d_147_b_0_dim[] = { 40 };
	uint16_t batch_norm_147_tmp_2_dim[] = { 1, 40, 16, 12 };
	uint16_t conv2d_147_w_0_dim[] = { 40, 80, 1, 1 };
	uint16_t nearest_interp_v2_11_tmp_0_dim[] = { 1, 40, 32, 24 };
	uint16_t tmp_36_dim[] = { 1, 40, 32, 24 };
	uint16_t batch_norm2d_148_b_0_dim[] = { 40 };
	uint16_t batch_norm_148_tmp_2_dim[] = { 1, 40, 8, 6 };
	uint16_t conv2d_148_w_0_dim[] = { 40, 160, 1, 1 };
	uint16_t nearest_interp_v2_12_tmp_0_dim[] = { 1, 40, 32, 24 };
	uint16_t tmp_37_dim[] = { 1, 40, 32, 24 };
	uint16_t relu_87_tmp_0_dim[] = { 1, 40, 32, 24 };
	uint16_t batch_norm2d_149_b_0_dim[] = { 40 };
	uint16_t batch_norm_149_tmp_2_dim[] = { 1, 40, 16, 12 };
	uint16_t conv2d_149_w_0_dim[] = { 40, 1, 3, 3 };
	uint16_t batch_norm2d_150_b_0_dim[] = { 80 };
	uint16_t batch_norm_150_tmp_2_dim[] = { 1, 80, 16, 12 };
	uint16_t conv2d_150_w_0_dim[] = { 80, 40, 1, 1 };
	uint16_t tmp_38_dim[] = { 1, 80, 16, 12 };
	uint16_t tmp_39_dim[] = { 1, 80, 16, 12 };
	uint16_t batch_norm2d_151_b_0_dim[] = { 80 };
	uint16_t batch_norm_151_tmp_2_dim[] = { 1, 80, 8, 6 };
	uint16_t conv2d_151_w_0_dim[] = { 80, 160, 1, 1 };
	uint16_t nearest_interp_v2_13_tmp_0_dim[] = { 1, 80, 16, 12 };
	uint16_t batch_norm2d_152_b_0_dim[] = { 40 };
	uint16_t batch_norm_152_tmp_2_dim[] = { 1, 40, 16, 12 };
	uint16_t conv2d_152_w_0_dim[] = { 40, 1, 3, 3 };
	uint16_t batch_norm2d_153_b_0_dim[] = { 40 };
	uint16_t conv2d_153_w_0_dim[] = { 40, 40, 1, 1 };
	uint16_t relu_89_tmp_0_dim[] = { 1, 40, 16, 12 };
	uint16_t batch_norm2d_154_b_0_dim[] = { 40 };
	uint16_t batch_norm_154_tmp_2_dim[] = { 1, 40, 8, 6 };
	uint16_t conv2d_154_w_0_dim[] = { 40, 1, 3, 3 };
	uint16_t batch_norm2d_155_b_0_dim[] = { 160 };
	uint16_t batch_norm_155_tmp_2_dim[] = { 1, 160, 8, 6 };
	uint16_t conv2d_155_w_0_dim[] = { 160, 40, 1, 1 };
	uint16_t tmp_41_dim[] = { 1, 160, 8, 6 };
	uint16_t batch_norm2d_156_b_0_dim[] = { 80 };
	uint16_t batch_norm_156_tmp_2_dim[] = { 1, 80, 8, 6 };
	uint16_t conv2d_156_w_0_dim[] = { 80, 1, 3, 3 };
	uint16_t batch_norm2d_157_b_0_dim[] = { 160 };
	uint16_t batch_norm_157_tmp_2_dim[] = { 1, 160, 8, 6 };
	uint16_t conv2d_157_w_0_dim[] = { 160, 80, 1, 1 };
	uint16_t tmp_42_dim[] = { 1, 160, 8, 6 };
	uint16_t relu_90_tmp_0_dim[] = { 1, 160, 8, 6 };
	uint16_t batch_norm2d_158_b_0_dim[] = { 160 };
	uint16_t batch_norm_158_tmp_2_dim[] = { 1, 160, 4, 3 };
	uint16_t conv2d_158_w_0_dim[] = { 160, 1, 3, 3 };
	uint16_t split_33_tmp_0_dim[] = { 1, 20, 32, 24 };
	uint16_t split_33_tmp_1_dim[] = { 1, 20, 32, 24 };
	uint16_t batch_norm2d_160_b_0_dim[] = { 20 };
	uint16_t conv2d_160_w_0_dim[] = { 20, 20, 1, 1 };
	uint16_t relu_92_tmp_0_dim[] = { 1, 20, 32, 24 };
	uint16_t batch_norm2d_161_b_0_dim[] = { 20 };
	uint16_t batch_norm_161_tmp_2_dim[] = { 1, 20, 32, 24 };
	uint16_t conv2d_161_w_0_dim[] = { 20, 1, 3, 3 };
	uint16_t batch_norm2d_162_b_0_dim[] = { 20 };
	uint16_t conv2d_162_w_0_dim[] = { 20, 20, 1, 1 };
	uint16_t relu_93_tmp_0_dim[] = { 1, 20, 32, 24 };
	uint16_t concat_33_tmp_0_dim[] = { 1, 40, 32, 24 };
	uint16_t shape_34_tmp_0_dim[] = { 4 };
	uint16_t shape_34_tmp_0_slice_0_dim[] = { 1 };
	uint16_t reshape2_66_tmp_0_dim[] = { 1, 2, 20, 32, 24 };
	uint16_t reshape2_66_tmp_1_dim[] = { 0, 1, 40, 32, 24 };
	uint16_t transpose_33_tmp_0_dim[] = { 1, 20, 2, 32, 24 };
	uint16_t transpose_33_tmp_1_dim[] = { 0, 1, 2, 20, 32, 24 };
	uint16_t reshape2_67_tmp_0_dim[] = { 1, 40, 32, 24 };
	uint16_t reshape2_67_tmp_1_dim[] = { 0, 1, 20, 2, 32, 24 };
	uint16_t split_34_tmp_0_dim[] = { 1, 20, 32, 24 };
	uint16_t split_34_tmp_1_dim[] = { 1, 20, 32, 24 };
	uint16_t batch_norm2d_163_b_0_dim[] = { 20 };
	uint16_t conv2d_163_w_0_dim[] = { 20, 20, 1, 1 };
	uint16_t relu_94_tmp_0_dim[] = { 1, 20, 32, 24 };
	uint16_t batch_norm2d_164_b_0_dim[] = { 20 };
	uint16_t batch_norm_164_tmp_2_dim[] = { 1, 20, 32, 24 };
	uint16_t conv2d_164_w_0_dim[] = { 20, 1, 3, 3 };
	uint16_t batch_norm2d_165_b_0_dim[] = { 20 };
	uint16_t conv2d_165_w_0_dim[] = { 20, 20, 1, 1 };
	uint16_t relu_95_tmp_0_dim[] = { 1, 20, 32, 24 };
	uint16_t concat_34_tmp_0_dim[] = { 1, 40, 32, 24 };
	uint16_t shape_35_tmp_0_dim[] = { 4 };
	uint16_t shape_35_tmp_0_slice_0_dim[] = { 1 };
	uint16_t reshape2_68_tmp_0_dim[] = { 1, 2, 20, 32, 24 };
	uint16_t reshape2_68_tmp_1_dim[] = { 0, 1, 40, 32, 24 };
	uint16_t transpose_34_tmp_0_dim[] = { 1, 20, 2, 32, 24 };
	uint16_t transpose_34_tmp_1_dim[] = { 0, 1, 2, 20, 32, 24 };
	uint16_t reshape2_69_tmp_0_dim[] = { 1, 40, 32, 24 };
	uint16_t reshape2_69_tmp_1_dim[] = { 0, 1, 20, 2, 32, 24 };
	uint16_t relu_88_tmp_0_dim[] = { 1, 80, 16, 12 };
	uint16_t split_35_tmp_0_dim[] = { 1, 40, 16, 12 };
	uint16_t split_35_tmp_1_dim[] = { 1, 40, 16, 12 };
	uint16_t batch_norm2d_166_b_0_dim[] = { 40 };
	uint16_t conv2d_166_w_0_dim[] = { 40, 40, 1, 1 };
	uint16_t relu_96_tmp_0_dim[] = { 1, 40, 16, 12 };
	uint16_t batch_norm2d_167_b_0_dim[] = { 40 };
	uint16_t batch_norm_167_tmp_2_dim[] = { 1, 40, 16, 12 };
	uint16_t conv2d_167_w_0_dim[] = { 40, 1, 3, 3 };
	uint16_t batch_norm2d_168_b_0_dim[] = { 40 };
	uint16_t conv2d_168_w_0_dim[] = { 40, 40, 1, 1 };
	uint16_t relu_97_tmp_0_dim[] = { 1, 40, 16, 12 };
	uint16_t concat_35_tmp_0_dim[] = { 1, 80, 16, 12 };
	uint16_t shape_36_tmp_0_dim[] = { 4 };
	uint16_t shape_36_tmp_0_slice_0_dim[] = { 1 };
	uint16_t reshape2_70_tmp_0_dim[] = { 1, 2, 40, 16, 12 };
	uint16_t reshape2_70_tmp_1_dim[] = { 0, 1, 80, 16, 12 };
	uint16_t transpose_35_tmp_0_dim[] = { 1, 40, 2, 16, 12 };
	uint16_t transpose_35_tmp_1_dim[] = { 0, 1, 2, 40, 16, 12 };
	uint16_t reshape2_71_tmp_0_dim[] = { 1, 80, 16, 12 };
	uint16_t reshape2_71_tmp_1_dim[] = { 0, 1, 40, 2, 16, 12 };
	uint16_t split_36_tmp_0_dim[] = { 1, 40, 16, 12 };
	uint16_t split_36_tmp_1_dim[] = { 1, 40, 16, 12 };
	uint16_t batch_norm2d_169_b_0_dim[] = { 40 };
	uint16_t conv2d_169_w_0_dim[] = { 40, 40, 1, 1 };
	uint16_t relu_98_tmp_0_dim[] = { 1, 40, 16, 12 };
	uint16_t batch_norm2d_170_b_0_dim[] = { 40 };
	uint16_t batch_norm_170_tmp_2_dim[] = { 1, 40, 16, 12 };
	uint16_t conv2d_170_w_0_dim[] = { 40, 1, 3, 3 };
	uint16_t batch_norm2d_171_b_0_dim[] = { 40 };
	uint16_t conv2d_171_w_0_dim[] = { 40, 40, 1, 1 };
	uint16_t relu_99_tmp_0_dim[] = { 1, 40, 16, 12 };
	uint16_t concat_36_tmp_0_dim[] = { 1, 80, 16, 12 };
	uint16_t shape_37_tmp_0_dim[] = { 4 };
	uint16_t shape_37_tmp_0_slice_0_dim[] = { 1 };
	uint16_t reshape2_72_tmp_0_dim[] = { 1, 2, 40, 16, 12 };
	uint16_t reshape2_72_tmp_1_dim[] = { 0, 1, 80, 16, 12 };
	uint16_t transpose_36_tmp_0_dim[] = { 1, 40, 2, 16, 12 };
	uint16_t transpose_36_tmp_1_dim[] = { 0, 1, 2, 40, 16, 12 };
	uint16_t reshape2_73_tmp_0_dim[] = { 1, 80, 16, 12 };
	uint16_t reshape2_73_tmp_1_dim[] = { 0, 1, 40, 2, 16, 12 };
	uint16_t split_37_tmp_0_dim[] = { 1, 80, 8, 6 };
	uint16_t split_37_tmp_1_dim[] = { 1, 80, 8, 6 };
	uint16_t batch_norm2d_172_b_0_dim[] = { 80 };
	uint16_t conv2d_172_w_0_dim[] = { 80, 80, 1, 1 };
	uint16_t relu_100_tmp_0_dim[] = { 1, 80, 8, 6 };
	uint16_t batch_norm2d_173_b_0_dim[] = { 80 };
	uint16_t batch_norm_173_tmp_2_dim[] = { 1, 80, 8, 6 };
	uint16_t conv2d_173_w_0_dim[] = { 80, 1, 3, 3 };
	uint16_t batch_norm2d_174_b_0_dim[] = { 80 };
	uint16_t conv2d_174_w_0_dim[] = { 80, 80, 1, 1 };
	uint16_t relu_101_tmp_0_dim[] = { 1, 80, 8, 6 };
	uint16_t concat_37_tmp_0_dim[] = { 1, 160, 8, 6 };
	uint16_t shape_38_tmp_0_dim[] = { 4 };
	uint16_t shape_38_tmp_0_slice_0_dim[] = { 1 };
	uint16_t reshape2_74_tmp_0_dim[] = { 1, 2, 80, 8, 6 };
	uint16_t reshape2_74_tmp_1_dim[] = { 0, 1, 160, 8, 6 };
	uint16_t transpose_37_tmp_0_dim[] = { 1, 80, 2, 8, 6 };
	uint16_t transpose_37_tmp_1_dim[] = { 0, 1, 2, 80, 8, 6 };
	uint16_t reshape2_75_tmp_0_dim[] = { 1, 160, 8, 6 };
	uint16_t reshape2_75_tmp_1_dim[] = { 0, 1, 80, 2, 8, 6 };
	uint16_t split_38_tmp_0_dim[] = { 1, 80, 8, 6 };
	uint16_t split_38_tmp_1_dim[] = { 1, 80, 8, 6 };
	uint16_t batch_norm2d_175_b_0_dim[] = { 80 };
	uint16_t conv2d_175_w_0_dim[] = { 80, 80, 1, 1 };
	uint16_t relu_102_tmp_0_dim[] = { 1, 80, 8, 6 };
	uint16_t batch_norm2d_176_b_0_dim[] = { 80 };
	uint16_t batch_norm_176_tmp_2_dim[] = { 1, 80, 8, 6 };
	uint16_t conv2d_176_w_0_dim[] = { 80, 1, 3, 3 };
	uint16_t batch_norm2d_177_b_0_dim[] = { 80 };
	uint16_t conv2d_177_w_0_dim[] = { 80, 80, 1, 1 };
	uint16_t relu_103_tmp_0_dim[] = { 1, 80, 8, 6 };
	uint16_t concat_38_tmp_0_dim[] = { 1, 160, 8, 6 };
	uint16_t shape_39_tmp_0_dim[] = { 4 };
	uint16_t shape_39_tmp_0_slice_0_dim[] = { 1 };
	uint16_t reshape2_76_tmp_0_dim[] = { 1, 2, 80, 8, 6 };
	uint16_t reshape2_76_tmp_1_dim[] = { 0, 1, 160, 8, 6 };
	uint16_t transpose_38_tmp_0_dim[] = { 1, 80, 2, 8, 6 };
	uint16_t transpose_38_tmp_1_dim[] = { 0, 1, 2, 80, 8, 6 };
	uint16_t reshape2_77_tmp_0_dim[] = { 1, 160, 8, 6 };
	uint16_t reshape2_77_tmp_1_dim[] = { 0, 1, 80, 2, 8, 6 };
	uint16_t batch_norm2d_159_b_0_dim[] = { 320 };
	uint16_t conv2d_159_w_0_dim[] = { 320, 160, 1, 1 };
	uint16_t relu_91_tmp_0_dim[] = { 1, 320, 4, 3 };
	uint16_t split_39_tmp_0_dim[] = { 1, 160, 4, 3 };
	uint16_t split_39_tmp_1_dim[] = { 1, 160, 4, 3 };
	uint16_t batch_norm2d_178_b_0_dim[] = { 160 };
	uint16_t conv2d_178_w_0_dim[] = { 160, 160, 1, 1 };
	uint16_t relu_104_tmp_0_dim[] = { 1, 160, 4, 3 };
	uint16_t batch_norm2d_179_b_0_dim[] = { 160 };
	uint16_t batch_norm_179_tmp_2_dim[] = { 1, 160, 4, 3 };
	uint16_t conv2d_179_w_0_dim[] = { 160, 1, 3, 3 };
	uint16_t batch_norm2d_180_b_0_dim[] = { 160 };
	uint16_t conv2d_180_w_0_dim[] = { 160, 160, 1, 1 };
	uint16_t relu_105_tmp_0_dim[] = { 1, 160, 4, 3 };
	uint16_t concat_39_tmp_0_dim[] = { 1, 320, 4, 3 };
	uint16_t shape_40_tmp_0_dim[] = { 4 };
	uint16_t shape_40_tmp_0_slice_0_dim[] = { 1 };
	uint16_t reshape2_78_tmp_0_dim[] = { 1, 2, 160, 4, 3 };
	uint16_t reshape2_78_tmp_1_dim[] = { 0, 1, 320, 4, 3 };
	uint16_t transpose_39_tmp_0_dim[] = { 1, 160, 2, 4, 3 };
	uint16_t transpose_39_tmp_1_dim[] = { 0, 1, 2, 160, 4, 3 };
	uint16_t reshape2_79_tmp_0_dim[] = { 1, 320, 4, 3 };
	uint16_t reshape2_79_tmp_1_dim[] = { 0, 1, 160, 2, 4, 3 };
	uint16_t split_40_tmp_0_dim[] = { 1, 160, 4, 3 };
	uint16_t split_40_tmp_1_dim[] = { 1, 160, 4, 3 };
	uint16_t batch_norm2d_181_b_0_dim[] = { 160 };
	uint16_t conv2d_181_w_0_dim[] = { 160, 160, 1, 1 };
	uint16_t relu_106_tmp_0_dim[] = { 1, 160, 4, 3 };
	uint16_t batch_norm2d_182_b_0_dim[] = { 160 };
	uint16_t batch_norm_182_tmp_2_dim[] = { 1, 160, 4, 3 };
	uint16_t conv2d_182_w_0_dim[] = { 160, 1, 3, 3 };
	uint16_t batch_norm2d_183_b_0_dim[] = { 160 };
	uint16_t conv2d_183_w_0_dim[] = { 160, 160, 1, 1 };
	uint16_t relu_107_tmp_0_dim[] = { 1, 160, 4, 3 };
	uint16_t concat_40_tmp_0_dim[] = { 1, 320, 4, 3 };
	uint16_t shape_41_tmp_0_dim[] = { 4 };
	uint16_t shape_41_tmp_0_slice_0_dim[] = { 1 };
	uint16_t reshape2_80_tmp_0_dim[] = { 1, 2, 160, 4, 3 };
	uint16_t reshape2_80_tmp_1_dim[] = { 0, 1, 320, 4, 3 };
	uint16_t transpose_40_tmp_0_dim[] = { 1, 160, 2, 4, 3 };
	uint16_t transpose_40_tmp_1_dim[] = { 0, 1, 2, 160, 4, 3 };
	uint16_t reshape2_81_tmp_0_dim[] = { 1, 320, 4, 3 };
	uint16_t reshape2_81_tmp_1_dim[] = { 0, 1, 160, 2, 4, 3 };
	uint16_t tmp_44_dim[] = { 1, 40, 32, 24 };
	uint16_t batch_norm2d_184_b_0_dim[] = { 40 };
	uint16_t batch_norm_184_tmp_2_dim[] = { 1, 40, 16, 12 };
	uint16_t conv2d_184_w_0_dim[] = { 40, 80, 1, 1 };
	uint16_t nearest_interp_v2_14_tmp_0_dim[] = { 1, 40, 32, 24 };
	uint16_t tmp_45_dim[] = { 1, 40, 32, 24 };
	uint16_t batch_norm2d_185_b_0_dim[] = { 40 };
	uint16_t batch_norm_185_tmp_2_dim[] = { 1, 40, 8, 6 };
	uint16_t conv2d_185_w_0_dim[] = { 40, 160, 1, 1 };
	uint16_t nearest_interp_v2_15_tmp_0_dim[] = { 1, 40, 32, 24 };
	uint16_t tmp_46_dim[] = { 1, 40, 32, 24 };
	uint16_t batch_norm2d_186_b_0_dim[] = { 40 };
	uint16_t batch_norm_186_tmp_2_dim[] = { 1, 40, 4, 3 };
	uint16_t conv2d_186_w_0_dim[] = { 40, 320, 1, 1 };
	uint16_t nearest_interp_v2_16_tmp_0_dim[] = { 1, 40, 32, 24 };
	uint16_t tmp_47_dim[] = { 1, 40, 32, 24 };
	uint16_t relu_108_tmp_0_dim[] = { 1, 40, 32, 24 };
	uint16_t batch_norm2d_187_b_0_dim[] = { 40 };
	uint16_t batch_norm_187_tmp_2_dim[] = { 1, 40, 16, 12 };
	uint16_t conv2d_187_w_0_dim[] = { 40, 1, 3, 3 };
	uint16_t batch_norm2d_188_b_0_dim[] = { 80 };
	uint16_t batch_norm_188_tmp_2_dim[] = { 1, 80, 16, 12 };
	uint16_t conv2d_188_w_0_dim[] = { 80, 40, 1, 1 };
	uint16_t tmp_48_dim[] = { 1, 80, 16, 12 };
	uint16_t tmp_49_dim[] = { 1, 80, 16, 12 };
	uint16_t batch_norm2d_189_b_0_dim[] = { 80 };
	uint16_t batch_norm_189_tmp_2_dim[] = { 1, 80, 8, 6 };
	uint16_t conv2d_189_w_0_dim[] = { 80, 160, 1, 1 };
	uint16_t nearest_interp_v2_17_tmp_0_dim[] = { 1, 80, 16, 12 };
	uint16_t tmp_50_dim[] = { 1, 80, 16, 12 };
	uint16_t batch_norm2d_190_b_0_dim[] = { 80 };
	uint16_t batch_norm_190_tmp_2_dim[] = { 1, 80, 4, 3 };
	uint16_t conv2d_190_w_0_dim[] = { 80, 320, 1, 1 };
	uint16_t nearest_interp_v2_18_tmp_0_dim[] = { 1, 80, 16, 12 };
	uint16_t batch_norm2d_191_b_0_dim[] = { 40 };
	uint16_t batch_norm_191_tmp_2_dim[] = { 1, 40, 16, 12 };
	uint16_t conv2d_191_w_0_dim[] = { 40, 1, 3, 3 };
	uint16_t batch_norm2d_192_b_0_dim[] = { 40 };
	uint16_t conv2d_192_w_0_dim[] = { 40, 40, 1, 1 };
	uint16_t relu_110_tmp_0_dim[] = { 1, 40, 16, 12 };
	uint16_t batch_norm2d_193_b_0_dim[] = { 40 };
	uint16_t batch_norm_193_tmp_2_dim[] = { 1, 40, 8, 6 };
	uint16_t conv2d_193_w_0_dim[] = { 40, 1, 3, 3 };
	uint16_t batch_norm2d_194_b_0_dim[] = { 160 };
	uint16_t batch_norm_194_tmp_2_dim[] = { 1, 160, 8, 6 };
	uint16_t conv2d_194_w_0_dim[] = { 160, 40, 1, 1 };
	uint16_t tmp_52_dim[] = { 1, 160, 8, 6 };
	uint16_t batch_norm2d_195_b_0_dim[] = { 80 };
	uint16_t batch_norm_195_tmp_2_dim[] = { 1, 80, 8, 6 };
	uint16_t conv2d_195_w_0_dim[] = { 80, 1, 3, 3 };
	uint16_t batch_norm2d_196_b_0_dim[] = { 160 };
	uint16_t batch_norm_196_tmp_2_dim[] = { 1, 160, 8, 6 };
	uint16_t conv2d_196_w_0_dim[] = { 160, 80, 1, 1 };
	uint16_t tmp_53_dim[] = { 1, 160, 8, 6 };
	uint16_t tmp_54_dim[] = { 1, 160, 8, 6 };
	uint16_t batch_norm2d_197_b_0_dim[] = { 160 };
	uint16_t batch_norm_197_tmp_2_dim[] = { 1, 160, 4, 3 };
	uint16_t conv2d_197_w_0_dim[] = { 160, 320, 1, 1 };
	uint16_t nearest_interp_v2_19_tmp_0_dim[] = { 1, 160, 8, 6 };
	uint16_t batch_norm2d_198_b_0_dim[] = { 40 };
	uint16_t batch_norm_198_tmp_2_dim[] = { 1, 40, 16, 12 };
	uint16_t conv2d_198_w_0_dim[] = { 40, 1, 3, 3 };
	uint16_t batch_norm2d_199_b_0_dim[] = { 40 };
	uint16_t conv2d_199_w_0_dim[] = { 40, 40, 1, 1 };
	uint16_t relu_112_tmp_0_dim[] = { 1, 40, 16, 12 };
	uint16_t batch_norm2d_200_b_0_dim[] = { 40 };
	uint16_t batch_norm_200_tmp_2_dim[] = { 1, 40, 8, 6 };
	uint16_t conv2d_200_w_0_dim[] = { 40, 1, 3, 3 };
	uint16_t batch_norm2d_201_b_0_dim[] = { 40 };
	uint16_t conv2d_201_w_0_dim[] = { 40, 40, 1, 1 };
	uint16_t relu_113_tmp_0_dim[] = { 1, 40, 8, 6 };
	uint16_t batch_norm2d_202_b_0_dim[] = { 40 };
	uint16_t batch_norm_202_tmp_2_dim[] = { 1, 40, 4, 3 };
	uint16_t conv2d_202_w_0_dim[] = { 40, 1, 3, 3 };
	uint16_t batch_norm2d_203_b_0_dim[] = { 320 };
	uint16_t batch_norm_203_tmp_2_dim[] = { 1, 320, 4, 3 };
	uint16_t conv2d_203_w_0_dim[] = { 320, 40, 1, 1 };
	uint16_t tmp_56_dim[] = { 1, 320, 4, 3 };
	uint16_t batch_norm2d_204_b_0_dim[] = { 80 };
	uint16_t batch_norm_204_tmp_2_dim[] = { 1, 80, 8, 6 };
	uint16_t conv2d_204_w_0_dim[] = { 80, 1, 3, 3 };
	uint16_t batch_norm2d_205_b_0_dim[] = { 80 };
	uint16_t conv2d_205_w_0_dim[] = { 80, 80, 1, 1 };
	uint16_t relu_114_tmp_0_dim[] = { 1, 80, 8, 6 };
	uint16_t batch_norm2d_206_b_0_dim[] = { 80 };
	uint16_t batch_norm_206_tmp_2_dim[] = { 1, 80, 4, 3 };
	uint16_t conv2d_206_w_0_dim[] = { 80, 1, 3, 3 };
	uint16_t batch_norm2d_207_b_0_dim[] = { 320 };
	uint16_t batch_norm_207_tmp_2_dim[] = { 1, 320, 4, 3 };
	uint16_t conv2d_207_w_0_dim[] = { 320, 80, 1, 1 };
	uint16_t tmp_57_dim[] = { 1, 320, 4, 3 };
	uint16_t batch_norm2d_208_b_0_dim[] = { 160 };
	uint16_t batch_norm_208_tmp_2_dim[] = { 1, 160, 4, 3 };
	uint16_t conv2d_208_w_0_dim[] = { 160, 1, 3, 3 };
	uint16_t batch_norm2d_209_b_0_dim[] = { 320 };
	uint16_t batch_norm_209_tmp_2_dim[] = { 1, 320, 4, 3 };
	uint16_t conv2d_209_w_0_dim[] = { 320, 160, 1, 1 };
	uint16_t tmp_58_dim[] = { 1, 320, 4, 3 };
	uint16_t split_41_tmp_0_dim[] = { 1, 20, 32, 24 };
	uint16_t split_41_tmp_1_dim[] = { 1, 20, 32, 24 };
	uint16_t batch_norm2d_210_b_0_dim[] = { 20 };
	uint16_t conv2d_210_w_0_dim[] = { 20, 20, 1, 1 };
	uint16_t relu_116_tmp_0_dim[] = { 1, 20, 32, 24 };
	uint16_t batch_norm2d_211_b_0_dim[] = { 20 };
	uint16_t batch_norm_211_tmp_2_dim[] = { 1, 20, 32, 24 };
	uint16_t conv2d_211_w_0_dim[] = { 20, 1, 3, 3 };
	uint16_t batch_norm2d_212_b_0_dim[] = { 20 };
	uint16_t conv2d_212_w_0_dim[] = { 20, 20, 1, 1 };
	uint16_t relu_117_tmp_0_dim[] = { 1, 20, 32, 24 };
	uint16_t concat_41_tmp_0_dim[] = { 1, 40, 32, 24 };
	uint16_t shape_42_tmp_0_dim[] = { 4 };
	uint16_t shape_42_tmp_0_slice_0_dim[] = { 1 };
	uint16_t reshape2_82_tmp_0_dim[] = { 1, 2, 20, 32, 24 };
	uint16_t reshape2_82_tmp_1_dim[] = { 0, 1, 40, 32, 24 };
	uint16_t transpose_41_tmp_0_dim[] = { 1, 20, 2, 32, 24 };
	uint16_t transpose_41_tmp_1_dim[] = { 0, 1, 2, 20, 32, 24 };
	uint16_t reshape2_83_tmp_0_dim[] = { 1, 40, 32, 24 };
	uint16_t reshape2_83_tmp_1_dim[] = { 0, 1, 20, 2, 32, 24 };
	uint16_t split_42_tmp_0_dim[] = { 1, 20, 32, 24 };
	uint16_t split_42_tmp_1_dim[] = { 1, 20, 32, 24 };
	uint16_t batch_norm2d_213_b_0_dim[] = { 20 };
	uint16_t conv2d_213_w_0_dim[] = { 20, 20, 1, 1 };
	uint16_t relu_118_tmp_0_dim[] = { 1, 20, 32, 24 };
	uint16_t batch_norm2d_214_b_0_dim[] = { 20 };
	uint16_t batch_norm_214_tmp_2_dim[] = { 1, 20, 32, 24 };
	uint16_t conv2d_214_w_0_dim[] = { 20, 1, 3, 3 };
	uint16_t batch_norm2d_215_b_0_dim[] = { 20 };
	uint16_t conv2d_215_w_0_dim[] = { 20, 20, 1, 1 };
	uint16_t relu_119_tmp_0_dim[] = { 1, 20, 32, 24 };
	uint16_t concat_42_tmp_0_dim[] = { 1, 40, 32, 24 };
	uint16_t shape_43_tmp_0_dim[] = { 4 };
	uint16_t shape_43_tmp_0_slice_0_dim[] = { 1 };
	uint16_t reshape2_84_tmp_0_dim[] = { 1, 2, 20, 32, 24 };
	uint16_t reshape2_84_tmp_1_dim[] = { 0, 1, 40, 32, 24 };
	uint16_t transpose_42_tmp_0_dim[] = { 1, 20, 2, 32, 24 };
	uint16_t transpose_42_tmp_1_dim[] = { 0, 1, 2, 20, 32, 24 };
	uint16_t reshape2_85_tmp_0_dim[] = { 1, 40, 32, 24 };
	uint16_t reshape2_85_tmp_1_dim[] = { 0, 1, 20, 2, 32, 24 };
	uint16_t relu_109_tmp_0_dim[] = { 1, 80, 16, 12 };
	uint16_t split_43_tmp_0_dim[] = { 1, 40, 16, 12 };
	uint16_t split_43_tmp_1_dim[] = { 1, 40, 16, 12 };
	uint16_t batch_norm2d_216_b_0_dim[] = { 40 };
	uint16_t conv2d_216_w_0_dim[] = { 40, 40, 1, 1 };
	uint16_t relu_120_tmp_0_dim[] = { 1, 40, 16, 12 };
	uint16_t batch_norm2d_217_b_0_dim[] = { 40 };
	uint16_t batch_norm_217_tmp_2_dim[] = { 1, 40, 16, 12 };
	uint16_t conv2d_217_w_0_dim[] = { 40, 1, 3, 3 };
	uint16_t batch_norm2d_218_b_0_dim[] = { 40 };
	uint16_t conv2d_218_w_0_dim[] = { 40, 40, 1, 1 };
	uint16_t relu_121_tmp_0_dim[] = { 1, 40, 16, 12 };
	uint16_t concat_43_tmp_0_dim[] = { 1, 80, 16, 12 };
	uint16_t shape_44_tmp_0_dim[] = { 4 };
	uint16_t shape_44_tmp_0_slice_0_dim[] = { 1 };
	uint16_t reshape2_86_tmp_0_dim[] = { 1, 2, 40, 16, 12 };
	uint16_t reshape2_86_tmp_1_dim[] = { 0, 1, 80, 16, 12 };
	uint16_t transpose_43_tmp_0_dim[] = { 1, 40, 2, 16, 12 };
	uint16_t transpose_43_tmp_1_dim[] = { 0, 1, 2, 40, 16, 12 };
	uint16_t reshape2_87_tmp_0_dim[] = { 1, 80, 16, 12 };
	uint16_t reshape2_87_tmp_1_dim[] = { 0, 1, 40, 2, 16, 12 };
	uint16_t split_44_tmp_0_dim[] = { 1, 40, 16, 12 };
	uint16_t split_44_tmp_1_dim[] = { 1, 40, 16, 12 };
	uint16_t batch_norm2d_219_b_0_dim[] = { 40 };
	uint16_t conv2d_219_w_0_dim[] = { 40, 40, 1, 1 };
	uint16_t relu_122_tmp_0_dim[] = { 1, 40, 16, 12 };
	uint16_t batch_norm2d_220_b_0_dim[] = { 40 };
	uint16_t batch_norm_220_tmp_2_dim[] = { 1, 40, 16, 12 };
	uint16_t conv2d_220_w_0_dim[] = { 40, 1, 3, 3 };
	uint16_t batch_norm2d_221_b_0_dim[] = { 40 };
	uint16_t conv2d_221_w_0_dim[] = { 40, 40, 1, 1 };
	uint16_t relu_123_tmp_0_dim[] = { 1, 40, 16, 12 };
	uint16_t concat_44_tmp_0_dim[] = { 1, 80, 16, 12 };
	uint16_t shape_45_tmp_0_dim[] = { 4 };
	uint16_t shape_45_tmp_0_slice_0_dim[] = { 1 };
	uint16_t reshape2_88_tmp_0_dim[] = { 1, 2, 40, 16, 12 };
	uint16_t reshape2_88_tmp_1_dim[] = { 0, 1, 80, 16, 12 };
	uint16_t transpose_44_tmp_0_dim[] = { 1, 40, 2, 16, 12 };
	uint16_t transpose_44_tmp_1_dim[] = { 0, 1, 2, 40, 16, 12 };
	uint16_t reshape2_89_tmp_0_dim[] = { 1, 80, 16, 12 };
	uint16_t reshape2_89_tmp_1_dim[] = { 0, 1, 40, 2, 16, 12 };
	uint16_t relu_111_tmp_0_dim[] = { 1, 160, 8, 6 };
	uint16_t split_45_tmp_0_dim[] = { 1, 80, 8, 6 };
	uint16_t split_45_tmp_1_dim[] = { 1, 80, 8, 6 };
	uint16_t batch_norm2d_222_b_0_dim[] = { 80 };
	uint16_t conv2d_222_w_0_dim[] = { 80, 80, 1, 1 };
	uint16_t relu_124_tmp_0_dim[] = { 1, 80, 8, 6 };
	uint16_t batch_norm2d_223_b_0_dim[] = { 80 };
	uint16_t batch_norm_223_tmp_2_dim[] = { 1, 80, 8, 6 };
	uint16_t conv2d_223_w_0_dim[] = { 80, 1, 3, 3 };
	uint16_t batch_norm2d_224_b_0_dim[] = { 80 };
	uint16_t conv2d_224_w_0_dim[] = { 80, 80, 1, 1 };
	uint16_t relu_125_tmp_0_dim[] = { 1, 80, 8, 6 };
	uint16_t concat_45_tmp_0_dim[] = { 1, 160, 8, 6 };
	uint16_t shape_46_tmp_0_dim[] = { 4 };
	uint16_t shape_46_tmp_0_slice_0_dim[] = { 1 };
	uint16_t reshape2_90_tmp_0_dim[] = { 1, 2, 80, 8, 6 };
	uint16_t reshape2_90_tmp_1_dim[] = { 0, 1, 160, 8, 6 };
	uint16_t transpose_45_tmp_0_dim[] = { 1, 80, 2, 8, 6 };
	uint16_t transpose_45_tmp_1_dim[] = { 0, 1, 2, 80, 8, 6 };
	uint16_t reshape2_91_tmp_0_dim[] = { 1, 160, 8, 6 };
	uint16_t reshape2_91_tmp_1_dim[] = { 0, 1, 80, 2, 8, 6 };
	uint16_t split_46_tmp_0_dim[] = { 1, 80, 8, 6 };
	uint16_t split_46_tmp_1_dim[] = { 1, 80, 8, 6 };
	uint16_t batch_norm2d_225_b_0_dim[] = { 80 };
	uint16_t conv2d_225_w_0_dim[] = { 80, 80, 1, 1 };
	uint16_t relu_126_tmp_0_dim[] = { 1, 80, 8, 6 };
	uint16_t batch_norm2d_226_b_0_dim[] = { 80 };
	uint16_t batch_norm_226_tmp_2_dim[] = { 1, 80, 8, 6 };
	uint16_t conv2d_226_w_0_dim[] = { 80, 1, 3, 3 };
	uint16_t batch_norm2d_227_b_0_dim[] = { 80 };
	uint16_t conv2d_227_w_0_dim[] = { 80, 80, 1, 1 };
	uint16_t relu_127_tmp_0_dim[] = { 1, 80, 8, 6 };
	uint16_t concat_46_tmp_0_dim[] = { 1, 160, 8, 6 };
	uint16_t shape_47_tmp_0_dim[] = { 4 };
	uint16_t shape_47_tmp_0_slice_0_dim[] = { 1 };
	uint16_t reshape2_92_tmp_0_dim[] = { 1, 2, 80, 8, 6 };
	uint16_t reshape2_92_tmp_1_dim[] = { 0, 1, 160, 8, 6 };
	uint16_t transpose_46_tmp_0_dim[] = { 1, 80, 2, 8, 6 };
	uint16_t transpose_46_tmp_1_dim[] = { 0, 1, 2, 80, 8, 6 };
	uint16_t reshape2_93_tmp_0_dim[] = { 1, 160, 8, 6 };
	uint16_t reshape2_93_tmp_1_dim[] = { 0, 1, 80, 2, 8, 6 };
	uint16_t relu_115_tmp_0_dim[] = { 1, 320, 4, 3 };
	uint16_t split_47_tmp_0_dim[] = { 1, 160, 4, 3 };
	uint16_t split_47_tmp_1_dim[] = { 1, 160, 4, 3 };
	uint16_t batch_norm2d_228_b_0_dim[] = { 160 };
	uint16_t conv2d_228_w_0_dim[] = { 160, 160, 1, 1 };
	uint16_t relu_128_tmp_0_dim[] = { 1, 160, 4, 3 };
	uint16_t batch_norm2d_229_b_0_dim[] = { 160 };
	uint16_t batch_norm_229_tmp_2_dim[] = { 1, 160, 4, 3 };
	uint16_t conv2d_229_w_0_dim[] = { 160, 1, 3, 3 };
	uint16_t batch_norm2d_230_b_0_dim[] = { 160 };
	uint16_t conv2d_230_w_0_dim[] = { 160, 160, 1, 1 };
	uint16_t relu_129_tmp_0_dim[] = { 1, 160, 4, 3 };
	uint16_t concat_47_tmp_0_dim[] = { 1, 320, 4, 3 };
	uint16_t shape_48_tmp_0_dim[] = { 4 };
	uint16_t shape_48_tmp_0_slice_0_dim[] = { 1 };
	uint16_t reshape2_94_tmp_0_dim[] = { 1, 2, 160, 4, 3 };
	uint16_t reshape2_94_tmp_1_dim[] = { 0, 1, 320, 4, 3 };
	uint16_t transpose_47_tmp_0_dim[] = { 1, 160, 2, 4, 3 };
	uint16_t transpose_47_tmp_1_dim[] = { 0, 1, 2, 160, 4, 3 };
	uint16_t reshape2_95_tmp_0_dim[] = { 1, 320, 4, 3 };
	uint16_t reshape2_95_tmp_1_dim[] = { 0, 1, 160, 2, 4, 3 };
	uint16_t split_48_tmp_0_dim[] = { 1, 160, 4, 3 };
	uint16_t split_48_tmp_1_dim[] = { 1, 160, 4, 3 };
	uint16_t batch_norm2d_231_b_0_dim[] = { 160 };
	uint16_t conv2d_231_w_0_dim[] = { 160, 160, 1, 1 };
	uint16_t relu_130_tmp_0_dim[] = { 1, 160, 4, 3 };
	uint16_t batch_norm2d_232_b_0_dim[] = { 160 };
	uint16_t batch_norm_232_tmp_2_dim[] = { 1, 160, 4, 3 };
	uint16_t conv2d_232_w_0_dim[] = { 160, 1, 3, 3 };
	uint16_t batch_norm2d_233_b_0_dim[] = { 160 };
	uint16_t conv2d_233_w_0_dim[] = { 160, 160, 1, 1 };
	uint16_t relu_131_tmp_0_dim[] = { 1, 160, 4, 3 };
	uint16_t concat_48_tmp_0_dim[] = { 1, 320, 4, 3 };
	uint16_t shape_49_tmp_0_dim[] = { 4 };
	uint16_t shape_49_tmp_0_slice_0_dim[] = { 1 };
	uint16_t reshape2_96_tmp_0_dim[] = { 1, 2, 160, 4, 3 };
	uint16_t reshape2_96_tmp_1_dim[] = { 0, 1, 320, 4, 3 };
	uint16_t transpose_48_tmp_0_dim[] = { 1, 160, 2, 4, 3 };
	uint16_t transpose_48_tmp_1_dim[] = { 0, 1, 2, 160, 4, 3 };
	uint16_t reshape2_97_tmp_0_dim[] = { 1, 320, 4, 3 };
	uint16_t reshape2_97_tmp_1_dim[] = { 0, 1, 160, 2, 4, 3 };
	uint16_t tmp_60_dim[] = { 1, 40, 32, 24 };
	uint16_t batch_norm2d_234_b_0_dim[] = { 40 };
	uint16_t batch_norm_234_tmp_2_dim[] = { 1, 40, 16, 12 };
	uint16_t conv2d_234_w_0_dim[] = { 40, 80, 1, 1 };
	uint16_t nearest_interp_v2_20_tmp_0_dim[] = { 1, 40, 32, 24 };
	uint16_t tmp_61_dim[] = { 1, 40, 32, 24 };
	uint16_t batch_norm2d_235_b_0_dim[] = { 40 };
	uint16_t batch_norm_235_tmp_2_dim[] = { 1, 40, 8, 6 };
	uint16_t conv2d_235_w_0_dim[] = { 40, 160, 1, 1 };
	uint16_t nearest_interp_v2_21_tmp_0_dim[] = { 1, 40, 32, 24 };
	uint16_t tmp_62_dim[] = { 1, 40, 32, 24 };
	uint16_t batch_norm2d_236_b_0_dim[] = { 40 };
	uint16_t batch_norm_236_tmp_2_dim[] = { 1, 40, 4, 3 };
	uint16_t conv2d_236_w_0_dim[] = { 40, 320, 1, 1 };
	uint16_t nearest_interp_v2_22_tmp_0_dim[] = { 1, 40, 32, 24 };
	uint16_t tmp_63_dim[] = { 1, 40, 32, 24 };
	uint16_t relu_132_tmp_0_dim[] = { 1, 40, 32, 24 };
	uint16_t batch_norm2d_237_b_0_dim[] = { 40 };
	uint16_t batch_norm_237_tmp_2_dim[] = { 1, 40, 16, 12 };
	uint16_t conv2d_237_w_0_dim[] = { 40, 1, 3, 3 };
	uint16_t batch_norm2d_238_b_0_dim[] = { 80 };
	uint16_t batch_norm_238_tmp_2_dim[] = { 1, 80, 16, 12 };
	uint16_t conv2d_238_w_0_dim[] = { 80, 40, 1, 1 };
	uint16_t tmp_64_dim[] = { 1, 80, 16, 12 };
	uint16_t tmp_65_dim[] = { 1, 80, 16, 12 };
	uint16_t batch_norm2d_239_b_0_dim[] = { 80 };
	uint16_t batch_norm_239_tmp_2_dim[] = { 1, 80, 8, 6 };
	uint16_t conv2d_239_w_0_dim[] = { 80, 160, 1, 1 };
	uint16_t nearest_interp_v2_23_tmp_0_dim[] = { 1, 80, 16, 12 };
	uint16_t tmp_66_dim[] = { 1, 80, 16, 12 };
	uint16_t batch_norm2d_240_b_0_dim[] = { 80 };
	uint16_t batch_norm_240_tmp_2_dim[] = { 1, 80, 4, 3 };
	uint16_t conv2d_240_w_0_dim[] = { 80, 320, 1, 1 };
	uint16_t nearest_interp_v2_24_tmp_0_dim[] = { 1, 80, 16, 12 };
	uint16_t batch_norm2d_241_b_0_dim[] = { 40 };
	uint16_t batch_norm_241_tmp_2_dim[] = { 1, 40, 16, 12 };
	uint16_t conv2d_241_w_0_dim[] = { 40, 1, 3, 3 };
	uint16_t batch_norm2d_242_b_0_dim[] = { 40 };
	uint16_t conv2d_242_w_0_dim[] = { 40, 40, 1, 1 };
	uint16_t relu_134_tmp_0_dim[] = { 1, 40, 16, 12 };
	uint16_t batch_norm2d_243_b_0_dim[] = { 40 };
	uint16_t batch_norm_243_tmp_2_dim[] = { 1, 40, 8, 6 };
	uint16_t conv2d_243_w_0_dim[] = { 40, 1, 3, 3 };
	uint16_t batch_norm2d_244_b_0_dim[] = { 160 };
	uint16_t batch_norm_244_tmp_2_dim[] = { 1, 160, 8, 6 };
	uint16_t conv2d_244_w_0_dim[] = { 160, 40, 1, 1 };
	uint16_t tmp_68_dim[] = { 1, 160, 8, 6 };
	uint16_t batch_norm2d_245_b_0_dim[] = { 80 };
	uint16_t batch_norm_245_tmp_2_dim[] = { 1, 80, 8, 6 };
	uint16_t conv2d_245_w_0_dim[] = { 80, 1, 3, 3 };
	uint16_t batch_norm2d_246_b_0_dim[] = { 160 };
	uint16_t batch_norm_246_tmp_2_dim[] = { 1, 160, 8, 6 };
	uint16_t conv2d_246_w_0_dim[] = { 160, 80, 1, 1 };
	uint16_t tmp_69_dim[] = { 1, 160, 8, 6 };
	uint16_t tmp_70_dim[] = { 1, 160, 8, 6 };
	uint16_t batch_norm2d_247_b_0_dim[] = { 160 };
	uint16_t batch_norm_247_tmp_2_dim[] = { 1, 160, 4, 3 };
	uint16_t conv2d_247_w_0_dim[] = { 160, 320, 1, 1 };
	uint16_t nearest_interp_v2_25_tmp_0_dim[] = { 1, 160, 8, 6 };
	uint16_t batch_norm2d_248_b_0_dim[] = { 40 };
	uint16_t batch_norm_248_tmp_2_dim[] = { 1, 40, 16, 12 };
	uint16_t conv2d_248_w_0_dim[] = { 40, 1, 3, 3 };
	uint16_t batch_norm2d_249_b_0_dim[] = { 40 };
	uint16_t conv2d_249_w_0_dim[] = { 40, 40, 1, 1 };
	uint16_t relu_136_tmp_0_dim[] = { 1, 40, 16, 12 };
	uint16_t batch_norm2d_250_b_0_dim[] = { 40 };
	uint16_t batch_norm_250_tmp_2_dim[] = { 1, 40, 8, 6 };
	uint16_t conv2d_250_w_0_dim[] = { 40, 1, 3, 3 };
	uint16_t batch_norm2d_251_b_0_dim[] = { 40 };
	uint16_t conv2d_251_w_0_dim[] = { 40, 40, 1, 1 };
	uint16_t relu_137_tmp_0_dim[] = { 1, 40, 8, 6 };
	uint16_t batch_norm2d_252_b_0_dim[] = { 40 };
	uint16_t batch_norm_252_tmp_2_dim[] = { 1, 40, 4, 3 };
	uint16_t conv2d_252_w_0_dim[] = { 40, 1, 3, 3 };
	uint16_t batch_norm2d_253_b_0_dim[] = { 320 };
	uint16_t batch_norm_253_tmp_2_dim[] = { 1, 320, 4, 3 };
	uint16_t conv2d_253_w_0_dim[] = { 320, 40, 1, 1 };
	uint16_t tmp_72_dim[] = { 1, 320, 4, 3 };
	uint16_t batch_norm2d_254_b_0_dim[] = { 80 };
	uint16_t batch_norm_254_tmp_2_dim[] = { 1, 80, 8, 6 };
	uint16_t conv2d_254_w_0_dim[] = { 80, 1, 3, 3 };
	uint16_t batch_norm2d_255_b_0_dim[] = { 80 };
	uint16_t conv2d_255_w_0_dim[] = { 80, 80, 1, 1 };
	uint16_t relu_138_tmp_0_dim[] = { 1, 80, 8, 6 };
	uint16_t batch_norm2d_256_b_0_dim[] = { 80 };
	uint16_t batch_norm_256_tmp_2_dim[] = { 1, 80, 4, 3 };
	uint16_t conv2d_256_w_0_dim[] = { 80, 1, 3, 3 };
	uint16_t batch_norm2d_257_b_0_dim[] = { 320 };
	uint16_t batch_norm_257_tmp_2_dim[] = { 1, 320, 4, 3 };
	uint16_t conv2d_257_w_0_dim[] = { 320, 80, 1, 1 };
	uint16_t tmp_73_dim[] = { 1, 320, 4, 3 };
	uint16_t batch_norm2d_258_b_0_dim[] = { 160 };
	uint16_t batch_norm_258_tmp_2_dim[] = { 1, 160, 4, 3 };
	uint16_t conv2d_258_w_0_dim[] = { 160, 1, 3, 3 };
	uint16_t batch_norm2d_259_b_0_dim[] = { 320 };
	uint16_t batch_norm_259_tmp_2_dim[] = { 1, 320, 4, 3 };
	uint16_t conv2d_259_w_0_dim[] = { 320, 160, 1, 1 };
	uint16_t tmp_74_dim[] = { 1, 320, 4, 3 };
	uint16_t relu_139_tmp_0_dim[] = { 1, 320, 4, 3 };
	uint16_t batch_norm2d_260_b_0_dim[] = { 320 };
	uint16_t batch_norm_260_tmp_2_dim[] = { 1, 320, 4, 3 };
	uint16_t conv2d_260_w_0_dim[] = { 320, 1, 3, 3 };
	uint16_t batch_norm2d_261_b_0_dim[] = { 160 };
	uint16_t conv2d_261_w_0_dim[] = { 160, 320, 1, 1 };
	uint16_t relu_140_tmp_0_dim[] = { 1, 160, 4, 3 };
	uint16_t bilinear_interp_v2_0_tmp_0_dim[] = { 1, 160, 8, 6 };
	uint16_t relu_135_tmp_0_dim[] = { 1, 160, 8, 6 };
	uint16_t tmp_76_dim[] = { 1, 160, 8, 6 };
	uint16_t batch_norm2d_262_b_0_dim[] = { 160 };
	uint16_t batch_norm_262_tmp_2_dim[] = { 1, 160, 8, 6 };
	uint16_t conv2d_262_w_0_dim[] = { 160, 1, 3, 3 };
	uint16_t batch_norm2d_263_b_0_dim[] = { 80 };
	uint16_t conv2d_263_w_0_dim[] = { 80, 160, 1, 1 };
	uint16_t relu_141_tmp_0_dim[] = { 1, 80, 8, 6 };
	uint16_t bilinear_interp_v2_1_tmp_0_dim[] = { 1, 80, 16, 12 };
	uint16_t relu_133_tmp_0_dim[] = { 1, 80, 16, 12 };
	uint16_t tmp_77_dim[] = { 1, 80, 16, 12 };
	uint16_t batch_norm2d_264_b_0_dim[] = { 80 };
	uint16_t batch_norm_264_tmp_2_dim[] = { 1, 80, 16, 12 };
	uint16_t conv2d_264_w_0_dim[] = { 80, 1, 3, 3 };
	uint16_t batch_norm2d_265_b_0_dim[] = { 40 };
	uint16_t conv2d_265_w_0_dim[] = { 40, 80, 1, 1 };
	uint16_t relu_142_tmp_0_dim[] = { 1, 40, 16, 12 };
	uint16_t bilinear_interp_v2_2_tmp_0_dim[] = { 1, 40, 32, 24 };
	uint16_t tmp_78_dim[] = { 1, 40, 32, 24 };
	uint16_t batch_norm2d_266_b_0_dim[] = { 40 };
	uint16_t batch_norm_266_tmp_2_dim[] = { 1, 40, 32, 24 };
	uint16_t conv2d_266_w_0_dim[] = { 40, 1, 3, 3 };
	uint16_t batch_norm2d_267_b_0_dim[] = { 40 };
	uint16_t conv2d_267_w_0_dim[] = { 40, 40, 1, 1 };
	uint16_t relu_143_tmp_0_dim[] = { 1, 40, 32, 24 };
	uint16_t conv2d_268_b_0_dim[] = { 17 };
	uint16_t conv2d_268_w_0_dim[] = { 17, 40, 1, 1 };
	uint16_t conv2d_441_tmp_1_dim[] = { 1, 17, 32, 24 };
	uint16_t shape_53_tmp_0_dim[] = { 4 };
	uint16_t shape_53_tmp_0_slice_0_dim[] = { 1 };
	uint16_t reshape2_98_tmp_0_dim[] = { 1, 17, 768 };
	uint16_t reshape2_98_tmp_1_dim[] = { 0, 1, 17, 32, 24 };
	uint16_t argmax_0_tmp_0_dim[] = { 1, 17 };

	float* relu_0_tmp_0 = (float*)calloc(98304, sizeof(float));
	autox_conv2d(x, (float*)((int8_t*)weights) + 0, (float*)((int8_t*)weights) + 32, relu_0_tmp_0, image_dim, conv2d_0_w_0_dim, relu_0_tmp_0_dim, 1, 1, 2, 1, 1);
	free(x);

	float* split_0_tmp_0 = (float*)calloc(49152, sizeof(float));
	float* split_0_tmp_1 = (float*)calloc(49152, sizeof(float));
	float* p_1[] = { split_0_tmp_0, split_0_tmp_1, };
	uint16_t* p_1_dim[] = { split_0_tmp_0_dim, split_0_tmp_1_dim, };
	autox_split(relu_0_tmp_0, p_1, relu_0_tmp_0_dim, p_1_dim, 1, 4, 2, 4);
	free(relu_0_tmp_0);

	float* batch_norm_1_tmp_2 = (float*)calloc(12288, sizeof(float));
	autox_conv2d(split_0_tmp_0, (float*)((int8_t*)weights) + 896, (float*)((int8_t*)weights) + 912, batch_norm_1_tmp_2, split_0_tmp_0_dim, conv2d_1_w_0_dim, batch_norm_1_tmp_2_dim, 16, 1, 2, 1, 0);
	free(split_0_tmp_0);

	float* relu_2_tmp_0 = (float*)calloc(98304, sizeof(float));
	autox_conv2d(split_0_tmp_1, (float*)((int8_t*)weights) + 1056, (float*)((int8_t*)weights) + 1088, relu_2_tmp_0, split_0_tmp_1_dim, conv2d_3_w_0_dim, relu_2_tmp_0_dim, 1, 0, 1, 1, 1);
	free(split_0_tmp_1);

	float* batch_norm_4_tmp_2 = (float*)calloc(24576, sizeof(float));
	autox_conv2d(relu_2_tmp_0, (float*)((int8_t*)weights) + 1600, (float*)((int8_t*)weights) + 1632, batch_norm_4_tmp_2, relu_2_tmp_0_dim, conv2d_4_w_0_dim, batch_norm_4_tmp_2_dim, 32, 1, 2, 1, 0);
	free(relu_2_tmp_0);

	float* relu_1_tmp_0 = (float*)calloc(12288, sizeof(float));
	autox_conv2d(batch_norm_1_tmp_2, (float*)((int8_t*)weights) + 1920, (float*)((int8_t*)weights) + 1936, relu_1_tmp_0, batch_norm_1_tmp_2_dim, conv2d_2_w_0_dim, relu_1_tmp_0_dim, 1, 0, 1, 1, 1);
	free(batch_norm_1_tmp_2);

	float* relu_3_tmp_0 = (float*)calloc(12288, sizeof(float));
	autox_conv2d(batch_norm_4_tmp_2, (float*)((int8_t*)weights) + 2192, (float*)((int8_t*)weights) + 2208, relu_3_tmp_0, batch_norm_4_tmp_2_dim, conv2d_5_w_0_dim, relu_3_tmp_0_dim, 1, 0, 1, 1, 1);
	free(batch_norm_4_tmp_2);

	float* p_7[] = { relu_1_tmp_0, relu_3_tmp_0, };
	uint16_t* p_7_dim[] = { relu_1_tmp_0_dim, relu_3_tmp_0_dim, };
	float* concat_0_tmp_0 = (float*)calloc(24576, sizeof(float));
	autox_concat(p_7, concat_0_tmp_0, p_7_dim, concat_0_tmp_0_dim, 1, 2, 4);
	free(relu_1_tmp_0);
	free(relu_3_tmp_0);

	float* transpose_0_tmp_0 = (float*)calloc(24576, sizeof(float));
	uint16_t axis_8[] = { 0, 2, 1, 3, 4 };
	autox_transpose(concat_0_tmp_0, transpose_0_tmp_0, reshape2_0_tmp_0_dim, transpose_0_tmp_0_dim, axis_8, 5);
	free(concat_0_tmp_0);

	float* batch_norm_6_tmp_2 = (float*)calloc(24576, sizeof(float));
	autox_conv2d(transpose_0_tmp_0, (float*)((int8_t*)weights) + 2720, (float*)((int8_t*)weights) + 2752, batch_norm_6_tmp_2, reshape2_1_tmp_0_dim, conv2d_6_w_0_dim, batch_norm_6_tmp_2_dim, 32, 1, 1, 1, 0);
	// free(transpose_0_tmp_0);

	float* batch_norm_8_tmp_2 = (float*)calloc(6144, sizeof(float));
	autox_conv2d(transpose_0_tmp_0, (float*)((int8_t*)weights) + 3040, (float*)((int8_t*)weights) + 3072, batch_norm_8_tmp_2, reshape2_1_tmp_0_dim, conv2d_8_w_0_dim, batch_norm_8_tmp_2_dim, 32, 1, 2, 1, 0);
	free(transpose_0_tmp_0);

	float* relu_4_tmp_0 = (float*)calloc(30720, sizeof(float));
	autox_conv2d(batch_norm_6_tmp_2, (float*)((int8_t*)weights) + 3360, (float*)((int8_t*)weights) + 3400, relu_4_tmp_0, batch_norm_6_tmp_2_dim, conv2d_7_w_0_dim, relu_4_tmp_0_dim, 1, 0, 1, 1, 1);
	free(batch_norm_6_tmp_2);

	float* split_1_tmp_0 = (float*)calloc(15360, sizeof(float));
	float* split_1_tmp_1 = (float*)calloc(15360, sizeof(float));
	float* p_12[] = { split_1_tmp_0, split_1_tmp_1, };
	uint16_t* p_12_dim[] = { split_1_tmp_0_dim, split_1_tmp_1_dim, };
	autox_split(relu_4_tmp_0, p_12, relu_4_tmp_0_dim, p_12_dim, 1, 4, 2, 4);
	free(relu_4_tmp_0);

	float* relu_6_tmp_0 = (float*)calloc(15360, sizeof(float));
	autox_conv2d(split_1_tmp_1, (float*)((int8_t*)weights) + 4680, (float*)((int8_t*)weights) + 4700, relu_6_tmp_0, split_1_tmp_1_dim, conv2d_10_w_0_dim, relu_6_tmp_0_dim, 1, 0, 1, 1, 1);
	free(split_1_tmp_1);

	float* batch_norm_11_tmp_2 = (float*)calloc(15360, sizeof(float));
	autox_conv2d(relu_6_tmp_0, (float*)((int8_t*)weights) + 5100, (float*)((int8_t*)weights) + 5120, batch_norm_11_tmp_2, relu_6_tmp_0_dim, conv2d_11_w_0_dim, batch_norm_11_tmp_2_dim, 20, 1, 1, 1, 0);
	free(relu_6_tmp_0);

	float* relu_7_tmp_0 = (float*)calloc(15360, sizeof(float));
	autox_conv2d(batch_norm_11_tmp_2, (float*)((int8_t*)weights) + 5300, (float*)((int8_t*)weights) + 5320, relu_7_tmp_0, batch_norm_11_tmp_2_dim, conv2d_12_w_0_dim, relu_7_tmp_0_dim, 1, 0, 1, 1, 1);
	free(batch_norm_11_tmp_2);

	float* p_16[] = { split_1_tmp_0, relu_7_tmp_0, };
	uint16_t* p_16_dim[] = { split_1_tmp_0_dim, relu_7_tmp_0_dim, };
	float* concat_1_tmp_0 = (float*)calloc(30720, sizeof(float));
	autox_concat(p_16, concat_1_tmp_0, p_16_dim, concat_1_tmp_0_dim, 1, 2, 4);
	free(split_1_tmp_0);
	free(relu_7_tmp_0);

	float* transpose_1_tmp_0 = (float*)calloc(30720, sizeof(float));
	uint16_t axis_17[] = { 0, 2, 1, 3, 4 };
	autox_transpose(concat_1_tmp_0, transpose_1_tmp_0, reshape2_2_tmp_0_dim, transpose_1_tmp_0_dim, axis_17, 5);
	free(concat_1_tmp_0);

	float* split_2_tmp_0 = (float*)calloc(15360, sizeof(float));
	float* split_2_tmp_1 = (float*)calloc(15360, sizeof(float));
	float* p_18[] = { split_2_tmp_0, split_2_tmp_1, };
	uint16_t* p_18_dim[] = { split_2_tmp_0_dim, split_2_tmp_1_dim, };
	autox_split(transpose_1_tmp_0, p_18, reshape2_3_tmp_0_dim, p_18_dim, 1, 4, 2, 4);
	free(transpose_1_tmp_0);

	float* relu_8_tmp_0 = (float*)calloc(15360, sizeof(float));
	autox_conv2d(split_2_tmp_1, (float*)((int8_t*)weights) + 5720, (float*)((int8_t*)weights) + 5740, relu_8_tmp_0, split_2_tmp_1_dim, conv2d_13_w_0_dim, relu_8_tmp_0_dim, 1, 0, 1, 1, 1);
	free(split_2_tmp_1);

	float* batch_norm_14_tmp_2 = (float*)calloc(15360, sizeof(float));
	autox_conv2d(relu_8_tmp_0, (float*)((int8_t*)weights) + 6140, (float*)((int8_t*)weights) + 6160, batch_norm_14_tmp_2, relu_8_tmp_0_dim, conv2d_14_w_0_dim, batch_norm_14_tmp_2_dim, 20, 1, 1, 1, 0);
	free(relu_8_tmp_0);

	float* relu_9_tmp_0 = (float*)calloc(15360, sizeof(float));
	autox_conv2d(batch_norm_14_tmp_2, (float*)((int8_t*)weights) + 6340, (float*)((int8_t*)weights) + 6360, relu_9_tmp_0, batch_norm_14_tmp_2_dim, conv2d_15_w_0_dim, relu_9_tmp_0_dim, 1, 0, 1, 1, 1);
	free(batch_norm_14_tmp_2);

	float* p_22[] = { split_2_tmp_0, relu_9_tmp_0, };
	uint16_t* p_22_dim[] = { split_2_tmp_0_dim, relu_9_tmp_0_dim, };
	float* concat_2_tmp_0 = (float*)calloc(30720, sizeof(float));
	autox_concat(p_22, concat_2_tmp_0, p_22_dim, concat_2_tmp_0_dim, 1, 2, 4);
	free(split_2_tmp_0);
	free(relu_9_tmp_0);

	float* transpose_2_tmp_0 = (float*)calloc(30720, sizeof(float));
	uint16_t axis_23[] = { 0, 2, 1, 3, 4 };
	autox_transpose(concat_2_tmp_0, transpose_2_tmp_0, reshape2_4_tmp_0_dim, transpose_2_tmp_0_dim, axis_23, 5);
	free(concat_2_tmp_0);

	float* relu_5_tmp_0 = (float*)calloc(15360, sizeof(float));
	autox_conv2d(batch_norm_8_tmp_2, (float*)((int8_t*)weights) + 6760, (float*)((int8_t*)weights) + 6840, relu_5_tmp_0, batch_norm_8_tmp_2_dim, conv2d_9_w_0_dim, relu_5_tmp_0_dim, 1, 0, 1, 1, 1);
	free(batch_norm_8_tmp_2);

	float* split_3_tmp_0 = (float*)calloc(7680, sizeof(float));
	float* split_3_tmp_1 = (float*)calloc(7680, sizeof(float));
	float* p_25[] = { split_3_tmp_0, split_3_tmp_1, };
	uint16_t* p_25_dim[] = { split_3_tmp_0_dim, split_3_tmp_1_dim, };
	autox_split(relu_5_tmp_0, p_25, relu_5_tmp_0_dim, p_25_dim, 1, 4, 2, 4);
	free(relu_5_tmp_0);

	float* relu_10_tmp_0 = (float*)calloc(7680, sizeof(float));
	autox_conv2d(split_3_tmp_1, (float*)((int8_t*)weights) + 9400, (float*)((int8_t*)weights) + 9440, relu_10_tmp_0, split_3_tmp_1_dim, conv2d_16_w_0_dim, relu_10_tmp_0_dim, 1, 0, 1, 1, 1);
	free(split_3_tmp_1);

	float* batch_norm_17_tmp_2 = (float*)calloc(7680, sizeof(float));
	autox_conv2d(relu_10_tmp_0, (float*)((int8_t*)weights) + 11040, (float*)((int8_t*)weights) + 11080, batch_norm_17_tmp_2, relu_10_tmp_0_dim, conv2d_17_w_0_dim, batch_norm_17_tmp_2_dim, 40, 1, 1, 1, 0);
	free(relu_10_tmp_0);

	float* relu_11_tmp_0 = (float*)calloc(7680, sizeof(float));
	autox_conv2d(batch_norm_17_tmp_2, (float*)((int8_t*)weights) + 11440, (float*)((int8_t*)weights) + 11480, relu_11_tmp_0, batch_norm_17_tmp_2_dim, conv2d_18_w_0_dim, relu_11_tmp_0_dim, 1, 0, 1, 1, 1);
	free(batch_norm_17_tmp_2);

	float* p_29[] = { split_3_tmp_0, relu_11_tmp_0, };
	uint16_t* p_29_dim[] = { split_3_tmp_0_dim, relu_11_tmp_0_dim, };
	float* concat_3_tmp_0 = (float*)calloc(15360, sizeof(float));
	autox_concat(p_29, concat_3_tmp_0, p_29_dim, concat_3_tmp_0_dim, 1, 2, 4);
	free(split_3_tmp_0);
	free(relu_11_tmp_0);

	float* transpose_3_tmp_0 = (float*)calloc(15360, sizeof(float));
	uint16_t axis_30[] = { 0, 2, 1, 3, 4 };
	autox_transpose(concat_3_tmp_0, transpose_3_tmp_0, reshape2_6_tmp_0_dim, transpose_3_tmp_0_dim, axis_30, 5);
	free(concat_3_tmp_0);

	float* split_4_tmp_0 = (float*)calloc(7680, sizeof(float));
	float* split_4_tmp_1 = (float*)calloc(7680, sizeof(float));
	float* p_31[] = { split_4_tmp_0, split_4_tmp_1, };
	uint16_t* p_31_dim[] = { split_4_tmp_0_dim, split_4_tmp_1_dim, };
	autox_split(transpose_3_tmp_0, p_31, reshape2_7_tmp_0_dim, p_31_dim, 1, 4, 2, 4);
	free(transpose_3_tmp_0);

	float* relu_12_tmp_0 = (float*)calloc(7680, sizeof(float));
	autox_conv2d(split_4_tmp_1, (float*)((int8_t*)weights) + 13080, (float*)((int8_t*)weights) + 13120, relu_12_tmp_0, split_4_tmp_1_dim, conv2d_19_w_0_dim, relu_12_tmp_0_dim, 1, 0, 1, 1, 1);
	free(split_4_tmp_1);

	float* batch_norm_20_tmp_2 = (float*)calloc(7680, sizeof(float));
	autox_conv2d(relu_12_tmp_0, (float*)((int8_t*)weights) + 14720, (float*)((int8_t*)weights) + 14760, batch_norm_20_tmp_2, relu_12_tmp_0_dim, conv2d_20_w_0_dim, batch_norm_20_tmp_2_dim, 40, 1, 1, 1, 0);
	free(relu_12_tmp_0);

	float* relu_13_tmp_0 = (float*)calloc(7680, sizeof(float));
	autox_conv2d(batch_norm_20_tmp_2, (float*)((int8_t*)weights) + 15120, (float*)((int8_t*)weights) + 15160, relu_13_tmp_0, batch_norm_20_tmp_2_dim, conv2d_21_w_0_dim, relu_13_tmp_0_dim, 1, 0, 1, 1, 1);
	free(batch_norm_20_tmp_2);

	float* p_35[] = { split_4_tmp_0, relu_13_tmp_0, };
	uint16_t* p_35_dim[] = { split_4_tmp_0_dim, relu_13_tmp_0_dim, };
	float* concat_4_tmp_0 = (float*)calloc(15360, sizeof(float));
	autox_concat(p_35, concat_4_tmp_0, p_35_dim, concat_4_tmp_0_dim, 1, 2, 4);
	free(split_4_tmp_0);
	free(relu_13_tmp_0);

	float* transpose_4_tmp_0 = (float*)calloc(15360, sizeof(float));
	uint16_t axis_36[] = { 0, 2, 1, 3, 4 };
	autox_transpose(concat_4_tmp_0, transpose_4_tmp_0, reshape2_8_tmp_0_dim, transpose_4_tmp_0_dim, axis_36, 5);
	free(concat_4_tmp_0);

	float* tmp_0 = (float*)calloc(30720, sizeof(float));
	autox_elementwise_add(transpose_2_tmp_0, transpose_2_tmp_0, tmp_0, reshape2_5_tmp_0_dim, reshape2_5_tmp_0_dim, tmp_0_dim, -1, 4, 4, 4);
	free(transpose_2_tmp_0);
	// free(transpose_2.tmp_0);

	float* batch_norm_22_tmp_2 = (float*)calloc(7680, sizeof(float));
	autox_conv2d(transpose_4_tmp_0, (float*)((int8_t*)weights) + 16760, (float*)((int8_t*)weights) + 16800, batch_norm_22_tmp_2, reshape2_9_tmp_0_dim, conv2d_22_w_0_dim, batch_norm_22_tmp_2_dim, 1, 0, 1, 1, 0);
	//free(transpose_4_tmp_0);

	float* nearest_interp_v2_0_tmp_0 = (float*)calloc(30720, sizeof(float));
	autox_nearest_interp(batch_norm_22_tmp_2, nearest_interp_v2_0_tmp_0, batch_norm_22_tmp_2_dim, nearest_interp_v2_0_tmp_0_dim, 2, 0);
	free(batch_norm_22_tmp_2);

	float* tmp_1 = (float*)calloc(30720, sizeof(float));
	autox_elementwise_add(tmp_0, nearest_interp_v2_0_tmp_0, tmp_1, tmp_0_dim, nearest_interp_v2_0_tmp_0_dim, tmp_1_dim, -1, 4, 4, 4);
	free(tmp_0);
	free(nearest_interp_v2_0_tmp_0);

	float*	relu_14_tmp_0 = (float*)calloc(30720, sizeof(float));
	autox_relu_noreplace(tmp_1, relu_14_tmp_0, tmp_1_dim, 4);

	float* batch_norm_23_tmp_2 = (float*)calloc(7680, sizeof(float));
	autox_conv2d(tmp_1, (float*)((int8_t*)weights) + 20000, (float*)((int8_t*)weights) + 20040, batch_norm_23_tmp_2, tmp_1_dim, conv2d_23_w_0_dim, batch_norm_23_tmp_2_dim, 40, 1, 2, 1, 0);
	// free(tmp_1);

	float* batch_norm_24_tmp_2 = (float*)calloc(15360, sizeof(float));
	autox_conv2d(batch_norm_23_tmp_2, (float*)((int8_t*)weights) + 20400, (float*)((int8_t*)weights) + 20480, batch_norm_24_tmp_2, batch_norm_23_tmp_2_dim, conv2d_24_w_0_dim, batch_norm_24_tmp_2_dim, 1, 0, 1, 1, 0);
	free(batch_norm_23_tmp_2);

	float* tmp_2 = (float*)calloc(15360, sizeof(float));
	autox_elementwise_add(batch_norm_24_tmp_2, batch_norm_24_tmp_2, tmp_2, batch_norm_24_tmp_2_dim, batch_norm_24_tmp_2_dim, tmp_2_dim, -1, 4, 4, 4);
	free(batch_norm_24_tmp_2);
	//free(batch_norm_24_tmp_2);

	float* split_5_tmp_0 = (float*)calloc(15360, sizeof(float));
	float* split_5_tmp_1 = (float*)calloc(15360, sizeof(float));
	float* p_45[] = { split_5_tmp_0, split_5_tmp_1, };
	uint16_t* p_45_dim[] = { split_5_tmp_0_dim, split_5_tmp_1_dim, };
	autox_split(relu_14_tmp_0, p_45, relu_14_tmp_0_dim, p_45_dim, 1, 4, 2, 4);
	free(relu_14_tmp_0);

	float* relu_16_tmp_0 = (float*)calloc(15360, sizeof(float));
	autox_conv2d(split_5_tmp_1, (float*)((int8_t*)weights) + 23680, (float*)((int8_t*)weights) + 23700, relu_16_tmp_0, split_5_tmp_1_dim, conv2d_25_w_0_dim, relu_16_tmp_0_dim, 1, 0, 1, 1, 1);
	free(split_5_tmp_1);

	float* batch_norm_26_tmp_2 = (float*)calloc(15360, sizeof(float));
	autox_conv2d(relu_16_tmp_0, (float*)((int8_t*)weights) + 24100, (float*)((int8_t*)weights) + 24120, batch_norm_26_tmp_2, relu_16_tmp_0_dim, conv2d_26_w_0_dim, batch_norm_26_tmp_2_dim, 20, 1, 1, 1, 0);
	free(relu_16_tmp_0);

	float* relu_17_tmp_0 = (float*)calloc(15360, sizeof(float));
	autox_conv2d(batch_norm_26_tmp_2, (float*)((int8_t*)weights) + 24300, (float*)((int8_t*)weights) + 24320, relu_17_tmp_0, batch_norm_26_tmp_2_dim, conv2d_27_w_0_dim, relu_17_tmp_0_dim, 1, 0, 1, 1, 1);
	free(batch_norm_26_tmp_2);

	float* p_49[] = { split_5_tmp_0, relu_17_tmp_0, };
	uint16_t* p_49_dim[] = { split_5_tmp_0_dim, relu_17_tmp_0_dim, };
	float* concat_5_tmp_0 = (float*)calloc(30720, sizeof(float));
	autox_concat(p_49, concat_5_tmp_0, p_49_dim, concat_5_tmp_0_dim, 1, 2, 4);
	free(split_5_tmp_0);
	free(relu_17_tmp_0);

	float* transpose_5_tmp_0 = (float*)calloc(30720, sizeof(float));
	uint16_t axis_50[] = { 0, 2, 1, 3, 4 };
	autox_transpose(concat_5_tmp_0, transpose_5_tmp_0, reshape2_10_tmp_0_dim, transpose_5_tmp_0_dim, axis_50, 5);
	free(concat_5_tmp_0);

	float* split_6_tmp_0 = (float*)calloc(15360, sizeof(float));
	float* split_6_tmp_1 = (float*)calloc(15360, sizeof(float));
	float* p_51[] = { split_6_tmp_0, split_6_tmp_1, };
	uint16_t* p_51_dim[] = { split_6_tmp_0_dim, split_6_tmp_1_dim, };
	autox_split(transpose_5_tmp_0, p_51, reshape2_11_tmp_0_dim, p_51_dim, 1, 4, 2, 4);
	free(transpose_5_tmp_0);

	float* relu_18_tmp_0 = (float*)calloc(15360, sizeof(float));
	autox_conv2d(split_6_tmp_1, (float*)((int8_t*)weights) + 24720, (float*)((int8_t*)weights) + 24740, relu_18_tmp_0, split_6_tmp_1_dim, conv2d_28_w_0_dim, relu_18_tmp_0_dim, 1, 0, 1, 1, 1);
	free(split_6_tmp_1);

	float* batch_norm_29_tmp_2 = (float*)calloc(15360, sizeof(float));
	autox_conv2d(relu_18_tmp_0, (float*)((int8_t*)weights) + 25140, (float*)((int8_t*)weights) + 25160, batch_norm_29_tmp_2, relu_18_tmp_0_dim, conv2d_29_w_0_dim, batch_norm_29_tmp_2_dim, 20, 1, 1, 1, 0);
	free(relu_18_tmp_0);

	float* relu_19_tmp_0 = (float*)calloc(15360, sizeof(float));
	autox_conv2d(batch_norm_29_tmp_2, (float*)((int8_t*)weights) + 25340, (float*)((int8_t*)weights) + 25360, relu_19_tmp_0, batch_norm_29_tmp_2_dim, conv2d_30_w_0_dim, relu_19_tmp_0_dim, 1, 0, 1, 1, 1);
	free(batch_norm_29_tmp_2);

	float* p_55[] = { split_6_tmp_0, relu_19_tmp_0, };
	uint16_t* p_55_dim[] = { split_6_tmp_0_dim, relu_19_tmp_0_dim, };
	float* concat_6_tmp_0 = (float*)calloc(30720, sizeof(float));
	autox_concat(p_55, concat_6_tmp_0, p_55_dim, concat_6_tmp_0_dim, 1, 2, 4);
	free(split_6_tmp_0);
	free(relu_19_tmp_0);

	float* transpose_6_tmp_0 = (float*)calloc(30720, sizeof(float));
	uint16_t axis_56[] = { 0, 2, 1, 3, 4 };
	autox_transpose(concat_6_tmp_0, transpose_6_tmp_0, reshape2_12_tmp_0_dim, transpose_6_tmp_0_dim, axis_56, 5);
	free(concat_6_tmp_0);

	float* relu_15_tmp_0 = (float*)calloc(15360, sizeof(float));
	autox_fusion_elementwise_add_activation(tmp_2, transpose_4_tmp_0, relu_15_tmp_0, tmp_2_dim, reshape2_9_tmp_0_dim, relu_15_tmp_0_dim, -1, 4, 4, 4);
	free(tmp_2);
	free(transpose_4_tmp_0);

	float* split_7_tmp_0 = (float*)calloc(7680, sizeof(float));
	float* split_7_tmp_1 = (float*)calloc(7680, sizeof(float));
	float* p_58[] = { split_7_tmp_0, split_7_tmp_1, };
	uint16_t* p_58_dim[] = { split_7_tmp_0_dim, split_7_tmp_1_dim, };
	autox_split(relu_15_tmp_0, p_58, relu_15_tmp_0_dim, p_58_dim, 1, 4, 2, 4);
	free(relu_15_tmp_0);

	float* relu_20_tmp_0 = (float*)calloc(7680, sizeof(float));
	autox_conv2d(split_7_tmp_1, (float*)((int8_t*)weights) + 25760, (float*)((int8_t*)weights) + 25800, relu_20_tmp_0, split_7_tmp_1_dim, conv2d_31_w_0_dim, relu_20_tmp_0_dim, 1, 0, 1, 1, 1);
	free(split_7_tmp_1);

	float* batch_norm_32_tmp_2 = (float*)calloc(7680, sizeof(float));
	autox_conv2d(relu_20_tmp_0, (float*)((int8_t*)weights) + 27400, (float*)((int8_t*)weights) + 27440, batch_norm_32_tmp_2, relu_20_tmp_0_dim, conv2d_32_w_0_dim, batch_norm_32_tmp_2_dim, 40, 1, 1, 1, 0);
	free(relu_20_tmp_0);

	float* relu_21_tmp_0 = (float*)calloc(7680, sizeof(float));
	autox_conv2d(batch_norm_32_tmp_2, (float*)((int8_t*)weights) + 27800, (float*)((int8_t*)weights) + 27840, relu_21_tmp_0, batch_norm_32_tmp_2_dim, conv2d_33_w_0_dim, relu_21_tmp_0_dim, 1, 0, 1, 1, 1);
	free(batch_norm_32_tmp_2);

	float* p_62[] = { split_7_tmp_0, relu_21_tmp_0, };
	uint16_t* p_62_dim[] = { split_7_tmp_0_dim, relu_21_tmp_0_dim, };
	float* concat_7_tmp_0 = (float*)calloc(15360, sizeof(float));
	autox_concat(p_62, concat_7_tmp_0, p_62_dim, concat_7_tmp_0_dim, 1, 2, 4);
	free(split_7_tmp_0);
	free(relu_21_tmp_0);

	float* transpose_7_tmp_0 = (float*)calloc(15360, sizeof(float));
	uint16_t axis_63[] = { 0, 2, 1, 3, 4 };
	autox_transpose(concat_7_tmp_0, transpose_7_tmp_0, reshape2_14_tmp_0_dim, transpose_7_tmp_0_dim, axis_63, 5);
	free(concat_7_tmp_0);

	float* split_8_tmp_0 = (float*)calloc(7680, sizeof(float));
	float* split_8_tmp_1 = (float*)calloc(7680, sizeof(float));
	float* p_64[] = { split_8_tmp_0, split_8_tmp_1, };
	uint16_t* p_64_dim[] = { split_8_tmp_0_dim, split_8_tmp_1_dim, };
	autox_split(transpose_7_tmp_0, p_64, reshape2_15_tmp_0_dim, p_64_dim, 1, 4, 2, 4);
	free(transpose_7_tmp_0);

	float* relu_22_tmp_0 = (float*)calloc(7680, sizeof(float));
	autox_conv2d(split_8_tmp_1, (float*)((int8_t*)weights) + 29440, (float*)((int8_t*)weights) + 29480, relu_22_tmp_0, split_8_tmp_1_dim, conv2d_34_w_0_dim, relu_22_tmp_0_dim, 1, 0, 1, 1, 1);
	free(split_8_tmp_1);

	float* batch_norm_35_tmp_2 = (float*)calloc(7680, sizeof(float));
	autox_conv2d(relu_22_tmp_0, (float*)((int8_t*)weights) + 31080, (float*)((int8_t*)weights) + 31120, batch_norm_35_tmp_2, relu_22_tmp_0_dim, conv2d_35_w_0_dim, batch_norm_35_tmp_2_dim, 40, 1, 1, 1, 0);
	free(relu_22_tmp_0);

	float* relu_23_tmp_0 = (float*)calloc(7680, sizeof(float));
	autox_conv2d(batch_norm_35_tmp_2, (float*)((int8_t*)weights) + 31480, (float*)((int8_t*)weights) + 31520, relu_23_tmp_0, batch_norm_35_tmp_2_dim, conv2d_36_w_0_dim, relu_23_tmp_0_dim, 1, 0, 1, 1, 1);
	free(batch_norm_35_tmp_2);

	float* p_68[] = { split_8_tmp_0, relu_23_tmp_0, };
	uint16_t* p_68_dim[] = { split_8_tmp_0_dim, relu_23_tmp_0_dim, };
	float* concat_8_tmp_0 = (float*)calloc(15360, sizeof(float));
	autox_concat(p_68, concat_8_tmp_0, p_68_dim, concat_8_tmp_0_dim, 1, 2, 4);
	free(split_8_tmp_0);
	free(relu_23_tmp_0);

	float* transpose_8_tmp_0 = (float*)calloc(15360, sizeof(float));
	uint16_t axis_69[] = { 0, 2, 1, 3, 4 };
	autox_transpose(concat_8_tmp_0, transpose_8_tmp_0, reshape2_16_tmp_0_dim, transpose_8_tmp_0_dim, axis_69, 5);
	free(concat_8_tmp_0);

	float* tmp_4 = (float*)calloc(30720, sizeof(float));
	autox_elementwise_add(transpose_6_tmp_0, transpose_6_tmp_0, tmp_4, reshape2_13_tmp_0_dim, reshape2_13_tmp_0_dim, tmp_4_dim, -1, 4, 4, 4);
	free(transpose_6_tmp_0);
	//free(transpose_6.tmp_0);

	float* batch_norm_37_tmp_2 = (float*)calloc(7680, sizeof(float));
	autox_conv2d(transpose_8_tmp_0, (float*)((int8_t*)weights) + 33120, (float*)((int8_t*)weights) + 33160, batch_norm_37_tmp_2, reshape2_17_tmp_0_dim, conv2d_37_w_0_dim, batch_norm_37_tmp_2_dim, 1, 0, 1, 1, 0);
	// free(transpose_8_tmp_0);

	float* nearest_interp_v2_1_tmp_0 = (float*)calloc(30720, sizeof(float));
	autox_nearest_interp(batch_norm_37_tmp_2, nearest_interp_v2_1_tmp_0, batch_norm_37_tmp_2_dim, nearest_interp_v2_1_tmp_0_dim, 2, 0);
	free(batch_norm_37_tmp_2);

	float* tmp_5 = (float*)calloc(30720, sizeof(float));
	autox_elementwise_add(tmp_4, nearest_interp_v2_1_tmp_0, tmp_5, tmp_4_dim, nearest_interp_v2_1_tmp_0_dim, tmp_5_dim, -1, 4, 4, 4);
	free(tmp_4);
	free(nearest_interp_v2_1_tmp_0);

	float*	relu_24_tmp_0 = (float*)calloc(30720, sizeof(float));
	autox_relu_noreplace(tmp_5, relu_24_tmp_0, tmp_5_dim, 4);

	float* batch_norm_38_tmp_2 = (float*)calloc(7680, sizeof(float));
	autox_conv2d(tmp_5, (float*)((int8_t*)weights) + 36360, (float*)((int8_t*)weights) + 36400, batch_norm_38_tmp_2, tmp_5_dim, conv2d_38_w_0_dim, batch_norm_38_tmp_2_dim, 40, 1, 2, 1, 0);
	//free(tmp_5);

	float* batch_norm_39_tmp_2 = (float*)calloc(15360, sizeof(float));
	autox_conv2d(batch_norm_38_tmp_2, (float*)((int8_t*)weights) + 36760, (float*)((int8_t*)weights) + 36840, batch_norm_39_tmp_2, batch_norm_38_tmp_2_dim, conv2d_39_w_0_dim, batch_norm_39_tmp_2_dim, 1, 0, 1, 1, 0);
	free(batch_norm_38_tmp_2);

	float* tmp_6 = (float*)calloc(15360, sizeof(float));
	autox_elementwise_add(batch_norm_39_tmp_2, batch_norm_39_tmp_2, tmp_6, batch_norm_39_tmp_2_dim, batch_norm_39_tmp_2_dim, tmp_6_dim, -1, 4, 4, 4);
	free(batch_norm_39_tmp_2);
	// free(batch_norm_39_tmp_2);

	float* relu_25_tmp_0 = (float*)calloc(15360, sizeof(float));
	autox_fusion_elementwise_add_activation(tmp_6,transpose_8_tmp_0, relu_25_tmp_0, tmp_6_dim, reshape2_17_tmp_0_dim, relu_25_tmp_0_dim, -1, 4, 4, 4);
	free(tmp_6);
	free(transpose_8_tmp_0);

	float* batch_norm_40_tmp_2 = (float*)calloc(3840, sizeof(float));
	autox_conv2d(relu_25_tmp_0, (float*)((int8_t*)weights) + 40040, (float*)((int8_t*)weights) + 40120, batch_norm_40_tmp_2, relu_25_tmp_0_dim, conv2d_40_w_0_dim, batch_norm_40_tmp_2_dim, 80, 1, 2, 1, 0);
	//free(relu_25_tmp_0);

	float* split_9_tmp_0 = (float*)calloc(15360, sizeof(float));
	float* split_9_tmp_1 = (float*)calloc(15360, sizeof(float));
	float* p_80[] = { split_9_tmp_0, split_9_tmp_1, };
	uint16_t* p_80_dim[] = { split_9_tmp_0_dim, split_9_tmp_1_dim, };
	autox_split(relu_24_tmp_0, p_80, relu_24_tmp_0_dim, p_80_dim, 1, 4, 2, 4);
	free(relu_24_tmp_0);

	float* relu_27_tmp_0 = (float*)calloc(15360, sizeof(float));
	autox_conv2d(split_9_tmp_1, (float*)((int8_t*)weights) + 40840, (float*)((int8_t*)weights) + 40860, relu_27_tmp_0, split_9_tmp_1_dim, conv2d_42_w_0_dim, relu_27_tmp_0_dim, 1, 0, 1, 1, 1);
	free(split_9_tmp_1);

	float* batch_norm_43_tmp_2 = (float*)calloc(15360, sizeof(float));
	autox_conv2d(relu_27_tmp_0, (float*)((int8_t*)weights) + 41260, (float*)((int8_t*)weights) + 41280, batch_norm_43_tmp_2, relu_27_tmp_0_dim, conv2d_43_w_0_dim, batch_norm_43_tmp_2_dim, 20, 1, 1, 1, 0);
	free(relu_27_tmp_0);

	float* relu_28_tmp_0 = (float*)calloc(15360, sizeof(float));
	autox_conv2d(batch_norm_43_tmp_2, (float*)((int8_t*)weights) + 41460, (float*)((int8_t*)weights) + 41480, relu_28_tmp_0, batch_norm_43_tmp_2_dim, conv2d_44_w_0_dim, relu_28_tmp_0_dim, 1, 0, 1, 1, 1);
	free(batch_norm_43_tmp_2);

	float* p_84[] = { split_9_tmp_0, relu_28_tmp_0, };
	uint16_t* p_84_dim[] = { split_9_tmp_0_dim, relu_28_tmp_0_dim, };
	float* concat_9_tmp_0 = (float*)calloc(30720, sizeof(float));
	autox_concat(p_84, concat_9_tmp_0, p_84_dim, concat_9_tmp_0_dim, 1, 2, 4);
	free(split_9_tmp_0);
	free(relu_28_tmp_0);

	float* transpose_9_tmp_0 = (float*)calloc(30720, sizeof(float));
	uint16_t axis_85[] = { 0, 2, 1, 3, 4 };
	autox_transpose(concat_9_tmp_0, transpose_9_tmp_0, reshape2_18_tmp_0_dim, transpose_9_tmp_0_dim, axis_85, 5);
	free(concat_9_tmp_0);

	float* split_10_tmp_0 = (float*)calloc(15360, sizeof(float));
	float* split_10_tmp_1 = (float*)calloc(15360, sizeof(float));
	float* p_86[] = { split_10_tmp_0, split_10_tmp_1, };
	uint16_t* p_86_dim[] = { split_10_tmp_0_dim, split_10_tmp_1_dim, };
	autox_split(transpose_9_tmp_0, p_86, reshape2_19_tmp_0_dim, p_86_dim, 1, 4, 2, 4);
	free(transpose_9_tmp_0);

	float* relu_29_tmp_0 = (float*)calloc(15360, sizeof(float));
	autox_conv2d(split_10_tmp_1, (float*)((int8_t*)weights) + 41880, (float*)((int8_t*)weights) + 41900, relu_29_tmp_0, split_10_tmp_1_dim, conv2d_45_w_0_dim, relu_29_tmp_0_dim, 1, 0, 1, 1, 1);
	free(split_10_tmp_1);

	float* batch_norm_46_tmp_2 = (float*)calloc(15360, sizeof(float));
	autox_conv2d(relu_29_tmp_0, (float*)((int8_t*)weights) + 42300, (float*)((int8_t*)weights) + 42320, batch_norm_46_tmp_2, relu_29_tmp_0_dim, conv2d_46_w_0_dim, batch_norm_46_tmp_2_dim, 20, 1, 1, 1, 0);
	free(relu_29_tmp_0);

	float* relu_30_tmp_0 = (float*)calloc(15360, sizeof(float));
	autox_conv2d(batch_norm_46_tmp_2, (float*)((int8_t*)weights) + 42500, (float*)((int8_t*)weights) + 42520, relu_30_tmp_0, batch_norm_46_tmp_2_dim, conv2d_47_w_0_dim, relu_30_tmp_0_dim, 1, 0, 1, 1, 1);
	free(batch_norm_46_tmp_2);

	float* p_90[] = { split_10_tmp_0, relu_30_tmp_0, };
	uint16_t* p_90_dim[] = { split_10_tmp_0_dim, relu_30_tmp_0_dim, };
	float* concat_10_tmp_0 = (float*)calloc(30720, sizeof(float));
	autox_concat(p_90, concat_10_tmp_0, p_90_dim, concat_10_tmp_0_dim, 1, 2, 4);
	free(split_10_tmp_0);
	free(relu_30_tmp_0);

	float* transpose_10_tmp_0 = (float*)calloc(30720, sizeof(float));
	uint16_t axis_91[] = { 0, 2, 1, 3, 4 };
	autox_transpose(concat_10_tmp_0, transpose_10_tmp_0, reshape2_20_tmp_0_dim, transpose_10_tmp_0_dim, axis_91, 5);
	free(concat_10_tmp_0);

	float* split_11_tmp_0 = (float*)calloc(7680, sizeof(float));
	float* split_11_tmp_1 = (float*)calloc(7680, sizeof(float));
	float* p_92[] = { split_11_tmp_0, split_11_tmp_1, };
	uint16_t* p_92_dim[] = { split_11_tmp_0_dim, split_11_tmp_1_dim, };
	autox_split(relu_25_tmp_0, p_92, relu_25_tmp_0_dim, p_92_dim, 1, 4, 2, 4);
	free(relu_25_tmp_0);

	float* relu_31_tmp_0 = (float*)calloc(7680, sizeof(float));
	autox_conv2d(split_11_tmp_1, (float*)((int8_t*)weights) + 42920, (float*)((int8_t*)weights) + 42960, relu_31_tmp_0, split_11_tmp_1_dim, conv2d_48_w_0_dim, relu_31_tmp_0_dim, 1, 0, 1, 1, 1);
	free(split_11_tmp_1);

	float* batch_norm_49_tmp_2 = (float*)calloc(7680, sizeof(float));
	autox_conv2d(relu_31_tmp_0, (float*)((int8_t*)weights) + 44560, (float*)((int8_t*)weights) + 44600, batch_norm_49_tmp_2, relu_31_tmp_0_dim, conv2d_49_w_0_dim, batch_norm_49_tmp_2_dim, 40, 1, 1, 1, 0);
	free(relu_31_tmp_0);

	float* relu_32_tmp_0 = (float*)calloc(7680, sizeof(float));
	autox_conv2d(batch_norm_49_tmp_2, (float*)((int8_t*)weights) + 44960, (float*)((int8_t*)weights) + 45000, relu_32_tmp_0, batch_norm_49_tmp_2_dim, conv2d_50_w_0_dim, relu_32_tmp_0_dim, 1, 0, 1, 1, 1);
	free(batch_norm_49_tmp_2);

	float* p_96[] = { split_11_tmp_0, relu_32_tmp_0, };
	uint16_t* p_96_dim[] = { split_11_tmp_0_dim, relu_32_tmp_0_dim, };
	float* concat_11_tmp_0 = (float*)calloc(15360, sizeof(float));
	autox_concat(p_96, concat_11_tmp_0, p_96_dim, concat_11_tmp_0_dim, 1, 2, 4);
	free(split_11_tmp_0);
	free(relu_32_tmp_0);

	float* transpose_11_tmp_0 = (float*)calloc(15360, sizeof(float));
	uint16_t axis_97[] = { 0, 2, 1, 3, 4 };
	autox_transpose(concat_11_tmp_0, transpose_11_tmp_0, reshape2_22_tmp_0_dim, transpose_11_tmp_0_dim, axis_97, 5);
	free(concat_11_tmp_0);

	float* split_12_tmp_0 = (float*)calloc(7680, sizeof(float));
	float* split_12_tmp_1 = (float*)calloc(7680, sizeof(float));
	float* p_98[] = { split_12_tmp_0, split_12_tmp_1, };
	uint16_t* p_98_dim[] = { split_12_tmp_0_dim, split_12_tmp_1_dim, };
	autox_split(transpose_11_tmp_0, p_98, reshape2_23_tmp_0_dim, p_98_dim, 1, 4, 2, 4);
	free(transpose_11_tmp_0);

	float* relu_33_tmp_0 = (float*)calloc(7680, sizeof(float));
	autox_conv2d(split_12_tmp_1, (float*)((int8_t*)weights) + 46600, (float*)((int8_t*)weights) + 46640, relu_33_tmp_0, split_12_tmp_1_dim, conv2d_51_w_0_dim, relu_33_tmp_0_dim, 1, 0, 1, 1, 1);
	free(split_12_tmp_1);

	float* batch_norm_52_tmp_2 = (float*)calloc(7680, sizeof(float));
	autox_conv2d(relu_33_tmp_0, (float*)((int8_t*)weights) + 48240, (float*)((int8_t*)weights) + 48280, batch_norm_52_tmp_2, relu_33_tmp_0_dim, conv2d_52_w_0_dim, batch_norm_52_tmp_2_dim, 40, 1, 1, 1, 0);
	free(relu_33_tmp_0);

	float* relu_34_tmp_0 = (float*)calloc(7680, sizeof(float));
	autox_conv2d(batch_norm_52_tmp_2, (float*)((int8_t*)weights) + 48640, (float*)((int8_t*)weights) + 48680, relu_34_tmp_0, batch_norm_52_tmp_2_dim, conv2d_53_w_0_dim, relu_34_tmp_0_dim, 1, 0, 1, 1, 1);
	free(batch_norm_52_tmp_2);

	float* p_102[] = { split_12_tmp_0, relu_34_tmp_0, };
	uint16_t* p_102_dim[] = { split_12_tmp_0_dim, relu_34_tmp_0_dim, };
	float* concat_12_tmp_0 = (float*)calloc(15360, sizeof(float));
	autox_concat(p_102, concat_12_tmp_0, p_102_dim, concat_12_tmp_0_dim, 1, 2, 4);
	free(split_12_tmp_0);
	free(relu_34_tmp_0);

	float* transpose_12_tmp_0 = (float*)calloc(15360, sizeof(float));
	uint16_t axis_103[] = { 0, 2, 1, 3, 4 };
	autox_transpose(concat_12_tmp_0, transpose_12_tmp_0, reshape2_24_tmp_0_dim, transpose_12_tmp_0_dim, axis_103, 5);
	free(concat_12_tmp_0);

	float* relu_26_tmp_0 = (float*)calloc(7680, sizeof(float));
	autox_conv2d(batch_norm_40_tmp_2, (float*)((int8_t*)weights) + 50280, (float*)((int8_t*)weights) + 50440, relu_26_tmp_0, batch_norm_40_tmp_2_dim, conv2d_41_w_0_dim, relu_26_tmp_0_dim, 1, 0, 1, 1, 1);
	free(batch_norm_40_tmp_2);

	float* split_13_tmp_0 = (float*)calloc(3840, sizeof(float));
	float* split_13_tmp_1 = (float*)calloc(3840, sizeof(float));
	float* p_105[] = { split_13_tmp_0, split_13_tmp_1, };
	uint16_t* p_105_dim[] = { split_13_tmp_0_dim, split_13_tmp_1_dim, };
	autox_split(relu_26_tmp_0, p_105, relu_26_tmp_0_dim, p_105_dim, 1, 4, 2, 4);
	free(relu_26_tmp_0);

	float* relu_35_tmp_0 = (float*)calloc(3840, sizeof(float));
	autox_conv2d(split_13_tmp_1, (float*)((int8_t*)weights) + 63240, (float*)((int8_t*)weights) + 63320, relu_35_tmp_0, split_13_tmp_1_dim, conv2d_54_w_0_dim, relu_35_tmp_0_dim, 1, 0, 1, 1, 1);
	free(split_13_tmp_1);

	float* batch_norm_55_tmp_2 = (float*)calloc(3840, sizeof(float));
	autox_conv2d(relu_35_tmp_0, (float*)((int8_t*)weights) + 69720, (float*)((int8_t*)weights) + 69800, batch_norm_55_tmp_2, relu_35_tmp_0_dim, conv2d_55_w_0_dim, batch_norm_55_tmp_2_dim, 80, 1, 1, 1, 0);
	free(relu_35_tmp_0);

	float* relu_36_tmp_0 = (float*)calloc(3840, sizeof(float));
	autox_conv2d(batch_norm_55_tmp_2, (float*)((int8_t*)weights) + 70520, (float*)((int8_t*)weights) + 70600, relu_36_tmp_0, batch_norm_55_tmp_2_dim, conv2d_56_w_0_dim, relu_36_tmp_0_dim, 1, 0, 1, 1, 1);
	free(batch_norm_55_tmp_2);

	float* p_109[] = { split_13_tmp_0, relu_36_tmp_0, };
	uint16_t* p_109_dim[] = { split_13_tmp_0_dim, relu_36_tmp_0_dim, };
	float* concat_13_tmp_0 = (float*)calloc(7680, sizeof(float));
	autox_concat(p_109, concat_13_tmp_0, p_109_dim, concat_13_tmp_0_dim, 1, 2, 4);
	free(split_13_tmp_0);
	free(relu_36_tmp_0);

	float* transpose_13_tmp_0 = (float*)calloc(7680, sizeof(float));
	uint16_t axis_110[] = { 0, 2, 1, 3, 4 };
	autox_transpose(concat_13_tmp_0, transpose_13_tmp_0, reshape2_26_tmp_0_dim, transpose_13_tmp_0_dim, axis_110, 5);
	free(concat_13_tmp_0);

	float* split_14_tmp_0 = (float*)calloc(3840, sizeof(float));
	float* split_14_tmp_1 = (float*)calloc(3840, sizeof(float));
	float* p_111[] = { split_14_tmp_0, split_14_tmp_1, };
	uint16_t* p_111_dim[] = { split_14_tmp_0_dim, split_14_tmp_1_dim, };
	autox_split(transpose_13_tmp_0, p_111, reshape2_27_tmp_0_dim, p_111_dim, 1, 4, 2, 4);
	free(transpose_13_tmp_0);

	float* relu_37_tmp_0 = (float*)calloc(3840, sizeof(float));
	autox_conv2d(split_14_tmp_1, (float*)((int8_t*)weights) + 77000, (float*)((int8_t*)weights) + 77080, relu_37_tmp_0, split_14_tmp_1_dim, conv2d_57_w_0_dim, relu_37_tmp_0_dim, 1, 0, 1, 1, 1);
	free(split_14_tmp_1);

	float* batch_norm_58_tmp_2 = (float*)calloc(3840, sizeof(float));
	autox_conv2d(relu_37_tmp_0, (float*)((int8_t*)weights) + 83480, (float*)((int8_t*)weights) + 83560, batch_norm_58_tmp_2, relu_37_tmp_0_dim, conv2d_58_w_0_dim, batch_norm_58_tmp_2_dim, 80, 1, 1, 1, 0);
	free(relu_37_tmp_0);

	float* relu_38_tmp_0 = (float*)calloc(3840, sizeof(float));
	autox_conv2d(batch_norm_58_tmp_2, (float*)((int8_t*)weights) + 84280, (float*)((int8_t*)weights) + 84360, relu_38_tmp_0, batch_norm_58_tmp_2_dim, conv2d_59_w_0_dim, relu_38_tmp_0_dim, 1, 0, 1, 1, 1);
	free(batch_norm_58_tmp_2);

	float* p_115[] = { split_14_tmp_0, relu_38_tmp_0, };
	uint16_t* p_115_dim[] = { split_14_tmp_0_dim, relu_38_tmp_0_dim, };
	float* concat_14_tmp_0 = (float*)calloc(7680, sizeof(float));
	autox_concat(p_115, concat_14_tmp_0, p_115_dim, concat_14_tmp_0_dim, 1, 2, 4);
	free(split_14_tmp_0);
	free(relu_38_tmp_0);

	float* transpose_14_tmp_0 = (float*)calloc(7680, sizeof(float));
	uint16_t axis_116[] = { 0, 2, 1, 3, 4 };
	autox_transpose(concat_14_tmp_0, transpose_14_tmp_0, reshape2_28_tmp_0_dim, transpose_14_tmp_0_dim, axis_116, 5);
	free(concat_14_tmp_0);

	float* tmp_8 = (float*)calloc(30720, sizeof(float));
	autox_elementwise_add(transpose_10_tmp_0, transpose_10_tmp_0, tmp_8, reshape2_21_tmp_0_dim, reshape2_21_tmp_0_dim, tmp_8_dim, -1, 4, 4, 4);
	free(transpose_10_tmp_0);
	//free(transpose_10.tmp_0);

	float* batch_norm_60_tmp_2 = (float*)calloc(7680, sizeof(float));
	autox_conv2d(transpose_12_tmp_0, (float*)((int8_t*)weights) + 90760, (float*)((int8_t*)weights) + 90800, batch_norm_60_tmp_2, reshape2_25_tmp_0_dim, conv2d_60_w_0_dim, batch_norm_60_tmp_2_dim, 1, 0, 1, 1, 0);
	//free(transpose_12_tmp_0);

	float* nearest_interp_v2_2_tmp_0 = (float*)calloc(30720, sizeof(float));
	autox_nearest_interp(batch_norm_60_tmp_2, nearest_interp_v2_2_tmp_0, batch_norm_60_tmp_2_dim, nearest_interp_v2_2_tmp_0_dim, 2, 0);
	free(batch_norm_60_tmp_2);

	float* tmp_9 = (float*)calloc(30720, sizeof(float));
	autox_elementwise_add(tmp_8, nearest_interp_v2_2_tmp_0, tmp_9, tmp_8_dim, nearest_interp_v2_2_tmp_0_dim, tmp_9_dim, -1, 4, 4, 4);
	free(tmp_8);
	free(nearest_interp_v2_2_tmp_0);

	float* batch_norm_61_tmp_2 = (float*)calloc(1920, sizeof(float));
	autox_conv2d(transpose_14_tmp_0, (float*)((int8_t*)weights) + 94000, (float*)((int8_t*)weights) + 94040, batch_norm_61_tmp_2, reshape2_29_tmp_0_dim, conv2d_61_w_0_dim, batch_norm_61_tmp_2_dim, 1, 0, 1, 1, 0);
	//free(transpose_14_tmp_0);

	float* nearest_interp_v2_3_tmp_0 = (float*)calloc(30720, sizeof(float));
	autox_nearest_interp(batch_norm_61_tmp_2, nearest_interp_v2_3_tmp_0, batch_norm_61_tmp_2_dim, nearest_interp_v2_3_tmp_0_dim, 4, 0);
	free(batch_norm_61_tmp_2);

	float* tmp_10 = (float*)calloc(30720, sizeof(float));
	autox_elementwise_add(tmp_9, nearest_interp_v2_3_tmp_0, tmp_10, tmp_9_dim, nearest_interp_v2_3_tmp_0_dim, tmp_10_dim, -1, 4, 4, 4);
	free(tmp_9);
	free(nearest_interp_v2_3_tmp_0);

	float*	relu_39_tmp_0 = (float*)calloc(30720, sizeof(float));
	autox_relu_noreplace(tmp_10, relu_39_tmp_0, tmp_10_dim, 4);

	float* batch_norm_62_tmp_2 = (float*)calloc(7680, sizeof(float));
	autox_conv2d(tmp_10, (float*)((int8_t*)weights) + 100440, (float*)((int8_t*)weights) + 100480, batch_norm_62_tmp_2, tmp_10_dim, conv2d_62_w_0_dim, batch_norm_62_tmp_2_dim, 40, 1, 2, 1, 0);
	// free(tmp_10);

	float* batch_norm_63_tmp_2 = (float*)calloc(15360, sizeof(float));
	autox_conv2d(batch_norm_62_tmp_2, (float*)((int8_t*)weights) + 100840, (float*)((int8_t*)weights) + 100920, batch_norm_63_tmp_2, batch_norm_62_tmp_2_dim, conv2d_63_w_0_dim, batch_norm_63_tmp_2_dim, 1, 0, 1, 1, 0);
	free(batch_norm_62_tmp_2);

	float* tmp_11 = (float*)calloc(15360, sizeof(float));
	autox_elementwise_add(batch_norm_63_tmp_2, batch_norm_63_tmp_2, tmp_11, batch_norm_63_tmp_2_dim, batch_norm_63_tmp_2_dim, tmp_11_dim, -1, 4, 4, 4);
	free(batch_norm_63_tmp_2);
	// free(batch_norm_63_tmp_2);

	float* tmp_12 = (float*)calloc(15360, sizeof(float));
	autox_elementwise_add(tmp_11, transpose_12_tmp_0, tmp_12, tmp_11_dim, reshape2_25_tmp_0_dim, tmp_12_dim, -1, 4, 4, 4);
	free(tmp_11);
	//free(transpose_14_tmp_0);

	float* batch_norm_64_tmp_2 = (float*)calloc(3840, sizeof(float));
	autox_conv2d(transpose_14_tmp_0, (float*)((int8_t*)weights) + 104120, (float*)((int8_t*)weights) + 104200, batch_norm_64_tmp_2, reshape2_29_tmp_0_dim, conv2d_64_w_0_dim, batch_norm_64_tmp_2_dim, 1, 0, 1, 1, 0);
	//free(transpose_14_tmp_0);

	float* nearest_interp_v2_4_tmp_0 = (float*)calloc(15360, sizeof(float));
	autox_nearest_interp(batch_norm_64_tmp_2, nearest_interp_v2_4_tmp_0, batch_norm_64_tmp_2_dim, nearest_interp_v2_4_tmp_0_dim, 2, 0);
	free(batch_norm_64_tmp_2);

	float* batch_norm_65_tmp_2 = (float*)calloc(7680, sizeof(float));
	autox_conv2d(tmp_10, (float*)((int8_t*)weights) + 117000, (float*)((int8_t*)weights) + 117040, batch_norm_65_tmp_2, tmp_10_dim, conv2d_65_w_0_dim, batch_norm_65_tmp_2_dim, 40, 1, 2, 1, 0);
	// free(tmp_10);

	float* relu_41_tmp_0 = (float*)calloc(7680, sizeof(float));
	autox_conv2d(batch_norm_65_tmp_2, (float*)((int8_t*)weights) + 117400, (float*)((int8_t*)weights) + 117440, relu_41_tmp_0, batch_norm_65_tmp_2_dim, conv2d_66_w_0_dim, relu_41_tmp_0_dim, 1, 0, 1, 1, 1);
	free(batch_norm_65_tmp_2);

	float* batch_norm_67_tmp_2 = (float*)calloc(1920, sizeof(float));
	autox_conv2d(relu_41_tmp_0, (float*)((int8_t*)weights) + 119040, (float*)((int8_t*)weights) + 119080, batch_norm_67_tmp_2, relu_41_tmp_0_dim, conv2d_67_w_0_dim, batch_norm_67_tmp_2_dim, 40, 1, 2, 1, 0);
	free(relu_41_tmp_0);

	float* batch_norm_68_tmp_2 = (float*)calloc(7680, sizeof(float));
	autox_conv2d(batch_norm_67_tmp_2, (float*)((int8_t*)weights) + 119440, (float*)((int8_t*)weights) + 119600, batch_norm_68_tmp_2, batch_norm_67_tmp_2_dim, conv2d_68_w_0_dim, batch_norm_68_tmp_2_dim, 1, 0, 1, 1, 0);
	free(batch_norm_67_tmp_2);

	float* tmp_14 = (float*)calloc(7680, sizeof(float));
	autox_elementwise_add(batch_norm_68_tmp_2, batch_norm_68_tmp_2, tmp_14, batch_norm_68_tmp_2_dim, batch_norm_68_tmp_2_dim, tmp_14_dim, -1, 4, 4, 4);
	free(batch_norm_68_tmp_2);
	// free(batch_norm_68_tmp_2);

	float* batch_norm_69_tmp_2 = (float*)calloc(3840, sizeof(float));
	autox_conv2d(transpose_12_tmp_0, (float*)((int8_t*)weights) + 126000, (float*)((int8_t*)weights) + 126080, batch_norm_69_tmp_2, reshape2_25_tmp_0_dim, conv2d_69_w_0_dim, batch_norm_69_tmp_2_dim, 80, 1, 2, 1, 0);
	free(transpose_12_tmp_0);

	float* batch_norm_70_tmp_2 = (float*)calloc(7680, sizeof(float));
	autox_conv2d(batch_norm_69_tmp_2, (float*)((int8_t*)weights) + 126800, (float*)((int8_t*)weights) + 126960, batch_norm_70_tmp_2, batch_norm_69_tmp_2_dim, conv2d_70_w_0_dim, batch_norm_70_tmp_2_dim, 1, 0, 1, 1, 0);
	free(batch_norm_69_tmp_2);

	float* tmp_15 = (float*)calloc(7680, sizeof(float));
	autox_elementwise_add(tmp_14, batch_norm_70_tmp_2, tmp_15, tmp_14_dim, batch_norm_70_tmp_2_dim, tmp_15_dim, -1, 4, 4, 4);
	free(tmp_14);
	free(batch_norm_70_tmp_2);

	float* split_15_tmp_0 = (float*)calloc(15360, sizeof(float));
	float* split_15_tmp_1 = (float*)calloc(15360, sizeof(float));
	float* p_139[] = { split_15_tmp_0, split_15_tmp_1, };
	uint16_t* p_139_dim[] = { split_15_tmp_0_dim, split_15_tmp_1_dim, };
	autox_split(relu_39_tmp_0, p_139, relu_39_tmp_0_dim, p_139_dim, 1, 4, 2, 4);
	free(relu_39_tmp_0);

	float* relu_43_tmp_0 = (float*)calloc(15360, sizeof(float));
	autox_conv2d(split_15_tmp_1, (float*)((int8_t*)weights) + 139760, (float*)((int8_t*)weights) + 139780, relu_43_tmp_0, split_15_tmp_1_dim, conv2d_71_w_0_dim, relu_43_tmp_0_dim, 1, 0, 1, 1, 1);
	free(split_15_tmp_1);

	float* batch_norm_72_tmp_2 = (float*)calloc(15360, sizeof(float));
	autox_conv2d(relu_43_tmp_0, (float*)((int8_t*)weights) + 140180, (float*)((int8_t*)weights) + 140200, batch_norm_72_tmp_2, relu_43_tmp_0_dim, conv2d_72_w_0_dim, batch_norm_72_tmp_2_dim, 20, 1, 1, 1, 0);
	free(relu_43_tmp_0);

	float* relu_44_tmp_0 = (float*)calloc(15360, sizeof(float));
	autox_conv2d(batch_norm_72_tmp_2, (float*)((int8_t*)weights) + 140380, (float*)((int8_t*)weights) + 140400, relu_44_tmp_0, batch_norm_72_tmp_2_dim, conv2d_73_w_0_dim, relu_44_tmp_0_dim, 1, 0, 1, 1, 1);
	free(batch_norm_72_tmp_2);

	float* p_143[] = { split_15_tmp_0, relu_44_tmp_0, };
	uint16_t* p_143_dim[] = { split_15_tmp_0_dim, relu_44_tmp_0_dim, };
	float* concat_15_tmp_0 = (float*)calloc(30720, sizeof(float));
	autox_concat(p_143, concat_15_tmp_0, p_143_dim, concat_15_tmp_0_dim, 1, 2, 4);
	free(split_15_tmp_0);
	free(relu_44_tmp_0);

	float* transpose_15_tmp_0 = (float*)calloc(30720, sizeof(float));
	uint16_t axis_144[] = { 0, 2, 1, 3, 4 };
	autox_transpose(concat_15_tmp_0, transpose_15_tmp_0, reshape2_30_tmp_0_dim, transpose_15_tmp_0_dim, axis_144, 5);
	free(concat_15_tmp_0);

	float* split_16_tmp_0 = (float*)calloc(15360, sizeof(float));
	float* split_16_tmp_1 = (float*)calloc(15360, sizeof(float));
	float* p_145[] = { split_16_tmp_0, split_16_tmp_1, };
	uint16_t* p_145_dim[] = { split_16_tmp_0_dim, split_16_tmp_1_dim, };
	autox_split(transpose_15_tmp_0, p_145, reshape2_31_tmp_0_dim, p_145_dim, 1, 4, 2, 4);
	free(transpose_15_tmp_0);

	float* relu_45_tmp_0 = (float*)calloc(15360, sizeof(float));
	autox_conv2d(split_16_tmp_1, (float*)((int8_t*)weights) + 140800, (float*)((int8_t*)weights) + 140820, relu_45_tmp_0, split_16_tmp_1_dim, conv2d_74_w_0_dim, relu_45_tmp_0_dim, 1, 0, 1, 1, 1);
	free(split_16_tmp_1);

	float* batch_norm_75_tmp_2 = (float*)calloc(15360, sizeof(float));
	autox_conv2d(relu_45_tmp_0, (float*)((int8_t*)weights) + 141220, (float*)((int8_t*)weights) + 141240, batch_norm_75_tmp_2, relu_45_tmp_0_dim, conv2d_75_w_0_dim, batch_norm_75_tmp_2_dim, 20, 1, 1, 1, 0);
	free(relu_45_tmp_0);

	float* relu_46_tmp_0 = (float*)calloc(15360, sizeof(float));
	autox_conv2d(batch_norm_75_tmp_2, (float*)((int8_t*)weights) + 141420, (float*)((int8_t*)weights) + 141440, relu_46_tmp_0, batch_norm_75_tmp_2_dim, conv2d_76_w_0_dim, relu_46_tmp_0_dim, 1, 0, 1, 1, 1);
	free(batch_norm_75_tmp_2);

	float* p_149[] = { split_16_tmp_0, relu_46_tmp_0, };
	uint16_t* p_149_dim[] = { split_16_tmp_0_dim, relu_46_tmp_0_dim, };
	float* concat_16_tmp_0 = (float*)calloc(30720, sizeof(float));
	autox_concat(p_149, concat_16_tmp_0, p_149_dim, concat_16_tmp_0_dim, 1, 2, 4);
	free(split_16_tmp_0);
	free(relu_46_tmp_0);

	float* transpose_16_tmp_0 = (float*)calloc(30720, sizeof(float));
	uint16_t axis_150[] = { 0, 2, 1, 3, 4 };
	autox_transpose(concat_16_tmp_0, transpose_16_tmp_0, reshape2_32_tmp_0_dim, transpose_16_tmp_0_dim, axis_150, 5);
	free(concat_16_tmp_0);

	float* relu_40_tmp_0 = (float*)calloc(15360, sizeof(float));
	autox_fusion_elementwise_add_activation(tmp_12, nearest_interp_v2_4_tmp_0, relu_40_tmp_0, tmp_12_dim, nearest_interp_v2_4_tmp_0_dim, relu_40_tmp_0_dim, -1, 4, 4, 4);
	free(tmp_12);
	free(nearest_interp_v2_4_tmp_0);

	float* split_17_tmp_0 = (float*)calloc(7680, sizeof(float));
	float* split_17_tmp_1 = (float*)calloc(7680, sizeof(float));
	float* p_152[] = { split_17_tmp_0, split_17_tmp_1, };
	uint16_t* p_152_dim[] = { split_17_tmp_0_dim, split_17_tmp_1_dim, };
	autox_split(relu_40_tmp_0, p_152, relu_40_tmp_0_dim, p_152_dim, 1, 4, 2, 4);
	free(relu_40_tmp_0);

	float* relu_47_tmp_0 = (float*)calloc(7680, sizeof(float));
	autox_conv2d(split_17_tmp_1, (float*)((int8_t*)weights) + 141840, (float*)((int8_t*)weights) + 141880, relu_47_tmp_0, split_17_tmp_1_dim, conv2d_77_w_0_dim, relu_47_tmp_0_dim, 1, 0, 1, 1, 1);
	free(split_17_tmp_1);

	float* batch_norm_78_tmp_2 = (float*)calloc(7680, sizeof(float));
	autox_conv2d(relu_47_tmp_0, (float*)((int8_t*)weights) + 143480, (float*)((int8_t*)weights) + 143520, batch_norm_78_tmp_2, relu_47_tmp_0_dim, conv2d_78_w_0_dim, batch_norm_78_tmp_2_dim, 40, 1, 1, 1, 0);
	free(relu_47_tmp_0);

	float* relu_48_tmp_0 = (float*)calloc(7680, sizeof(float));
	autox_conv2d(batch_norm_78_tmp_2, (float*)((int8_t*)weights) + 143880, (float*)((int8_t*)weights) + 143920, relu_48_tmp_0, batch_norm_78_tmp_2_dim, conv2d_79_w_0_dim, relu_48_tmp_0_dim, 1, 0, 1, 1, 1);
	free(batch_norm_78_tmp_2);

	float* p_156[] = { split_17_tmp_0, relu_48_tmp_0, };
	uint16_t* p_156_dim[] = { split_17_tmp_0_dim, relu_48_tmp_0_dim, };
	float* concat_17_tmp_0 = (float*)calloc(15360, sizeof(float));
	autox_concat(p_156, concat_17_tmp_0, p_156_dim, concat_17_tmp_0_dim, 1, 2, 4);
	free(split_17_tmp_0);
	free(relu_48_tmp_0);

	float* transpose_17_tmp_0 = (float*)calloc(15360, sizeof(float));
	uint16_t axis_157[] = { 0, 2, 1, 3, 4 };
	autox_transpose(concat_17_tmp_0, transpose_17_tmp_0, reshape2_34_tmp_0_dim, transpose_17_tmp_0_dim, axis_157, 5);
	free(concat_17_tmp_0);

	float* split_18_tmp_0 = (float*)calloc(7680, sizeof(float));
	float* split_18_tmp_1 = (float*)calloc(7680, sizeof(float));
	float* p_158[] = { split_18_tmp_0, split_18_tmp_1, };
	uint16_t* p_158_dim[] = { split_18_tmp_0_dim, split_18_tmp_1_dim, };
	autox_split(transpose_17_tmp_0, p_158, reshape2_35_tmp_0_dim, p_158_dim, 1, 4, 2, 4);
	free(transpose_17_tmp_0);

	float* relu_49_tmp_0 = (float*)calloc(7680, sizeof(float));
	autox_conv2d(split_18_tmp_1, (float*)((int8_t*)weights) + 145520, (float*)((int8_t*)weights) + 145560, relu_49_tmp_0, split_18_tmp_1_dim, conv2d_80_w_0_dim, relu_49_tmp_0_dim, 1, 0, 1, 1, 1);
	free(split_18_tmp_1);

	float* batch_norm_81_tmp_2 = (float*)calloc(7680, sizeof(float));
	autox_conv2d(relu_49_tmp_0, (float*)((int8_t*)weights) + 147160, (float*)((int8_t*)weights) + 147200, batch_norm_81_tmp_2, relu_49_tmp_0_dim, conv2d_81_w_0_dim, batch_norm_81_tmp_2_dim, 40, 1, 1, 1, 0);
	free(relu_49_tmp_0);

	float* relu_50_tmp_0 = (float*)calloc(7680, sizeof(float));
	autox_conv2d(batch_norm_81_tmp_2, (float*)((int8_t*)weights) + 147560, (float*)((int8_t*)weights) + 147600, relu_50_tmp_0, batch_norm_81_tmp_2_dim, conv2d_82_w_0_dim, relu_50_tmp_0_dim, 1, 0, 1, 1, 1);
	free(batch_norm_81_tmp_2);

	float* p_162[] = { split_18_tmp_0, relu_50_tmp_0, };
	uint16_t* p_162_dim[] = { split_18_tmp_0_dim, relu_50_tmp_0_dim, };
	float* concat_18_tmp_0 = (float*)calloc(15360, sizeof(float));
	autox_concat(p_162, concat_18_tmp_0, p_162_dim, concat_18_tmp_0_dim, 1, 2, 4);
	free(split_18_tmp_0);
	free(relu_50_tmp_0);

	float* transpose_18_tmp_0 = (float*)calloc(15360, sizeof(float));
	uint16_t axis_163[] = { 0, 2, 1, 3, 4 };
	autox_transpose(concat_18_tmp_0, transpose_18_tmp_0, reshape2_36_tmp_0_dim, transpose_18_tmp_0_dim, axis_163, 5);
	free(concat_18_tmp_0);

	float* relu_42_tmp_0 = (float*)calloc(7680, sizeof(float));
	autox_fusion_elementwise_add_activation(tmp_15, transpose_14_tmp_0, relu_42_tmp_0, tmp_15_dim, reshape2_29_tmp_0_dim, relu_42_tmp_0_dim, -1, 4, 4, 4);
	free(tmp_15);
	free(transpose_14_tmp_0);

	float* split_19_tmp_0 = (float*)calloc(3840, sizeof(float));
	float* split_19_tmp_1 = (float*)calloc(3840, sizeof(float));
	float* p_165[] = { split_19_tmp_0, split_19_tmp_1, };
	uint16_t* p_165_dim[] = { split_19_tmp_0_dim, split_19_tmp_1_dim, };
	autox_split(relu_42_tmp_0, p_165, relu_42_tmp_0_dim, p_165_dim, 1, 4, 2, 4);
	free(relu_42_tmp_0);

	float* relu_51_tmp_0 = (float*)calloc(3840, sizeof(float));
	autox_conv2d(split_19_tmp_1, (float*)((int8_t*)weights) + 149200, (float*)((int8_t*)weights) + 149280, relu_51_tmp_0, split_19_tmp_1_dim, conv2d_83_w_0_dim, relu_51_tmp_0_dim, 1, 0, 1, 1, 1);
	free(split_19_tmp_1);

	float* batch_norm_84_tmp_2 = (float*)calloc(3840, sizeof(float));
	autox_conv2d(relu_51_tmp_0, (float*)((int8_t*)weights) + 155680, (float*)((int8_t*)weights) + 155760, batch_norm_84_tmp_2, relu_51_tmp_0_dim, conv2d_84_w_0_dim, batch_norm_84_tmp_2_dim, 80, 1, 1, 1, 0);
	free(relu_51_tmp_0);

	float* relu_52_tmp_0 = (float*)calloc(3840, sizeof(float));
	autox_conv2d(batch_norm_84_tmp_2, (float*)((int8_t*)weights) + 156480, (float*)((int8_t*)weights) + 156560, relu_52_tmp_0, batch_norm_84_tmp_2_dim, conv2d_85_w_0_dim, relu_52_tmp_0_dim, 1, 0, 1, 1, 1);
	free(batch_norm_84_tmp_2);

	float* p_169[] = { split_19_tmp_0, relu_52_tmp_0, };
	uint16_t* p_169_dim[] = { split_19_tmp_0_dim, relu_52_tmp_0_dim, };
	float* concat_19_tmp_0 = (float*)calloc(7680, sizeof(float));
	autox_concat(p_169, concat_19_tmp_0, p_169_dim, concat_19_tmp_0_dim, 1, 2, 4);
	free(split_19_tmp_0);
	free(relu_52_tmp_0);

	float* transpose_19_tmp_0 = (float*)calloc(7680, sizeof(float));
	uint16_t axis_170[] = { 0, 2, 1, 3, 4 };
	autox_transpose(concat_19_tmp_0, transpose_19_tmp_0, reshape2_38_tmp_0_dim, transpose_19_tmp_0_dim, axis_170, 5);
	free(concat_19_tmp_0);

	float* split_20_tmp_0 = (float*)calloc(3840, sizeof(float));
	float* split_20_tmp_1 = (float*)calloc(3840, sizeof(float));
	float* p_171[] = { split_20_tmp_0, split_20_tmp_1, };
	uint16_t* p_171_dim[] = { split_20_tmp_0_dim, split_20_tmp_1_dim, };
	autox_split(transpose_19_tmp_0, p_171, reshape2_39_tmp_0_dim, p_171_dim, 1, 4, 2, 4);
	free(transpose_19_tmp_0);

	float* relu_53_tmp_0 = (float*)calloc(3840, sizeof(float));
	autox_conv2d(split_20_tmp_1, (float*)((int8_t*)weights) + 162960, (float*)((int8_t*)weights) + 163040, relu_53_tmp_0, split_20_tmp_1_dim, conv2d_86_w_0_dim, relu_53_tmp_0_dim, 1, 0, 1, 1, 1);
	free(split_20_tmp_1);

	float* batch_norm_87_tmp_2 = (float*)calloc(3840, sizeof(float));
	autox_conv2d(relu_53_tmp_0, (float*)((int8_t*)weights) + 169440, (float*)((int8_t*)weights) + 169520, batch_norm_87_tmp_2, relu_53_tmp_0_dim, conv2d_87_w_0_dim, batch_norm_87_tmp_2_dim, 80, 1, 1, 1, 0);
	free(relu_53_tmp_0);

	float* relu_54_tmp_0 = (float*)calloc(3840, sizeof(float));
	autox_conv2d(batch_norm_87_tmp_2, (float*)((int8_t*)weights) + 170240, (float*)((int8_t*)weights) + 170320, relu_54_tmp_0, batch_norm_87_tmp_2_dim, conv2d_88_w_0_dim, relu_54_tmp_0_dim, 1, 0, 1, 1, 1);
	free(batch_norm_87_tmp_2);

	float* p_175[] = { split_20_tmp_0, relu_54_tmp_0, };
	uint16_t* p_175_dim[] = { split_20_tmp_0_dim, relu_54_tmp_0_dim, };
	float* concat_20_tmp_0 = (float*)calloc(7680, sizeof(float));
	autox_concat(p_175, concat_20_tmp_0, p_175_dim, concat_20_tmp_0_dim, 1, 2, 4);
	free(split_20_tmp_0);
	free(relu_54_tmp_0);

	float* transpose_20_tmp_0 = (float*)calloc(7680, sizeof(float));
	uint16_t axis_176[] = { 0, 2, 1, 3, 4 };
	autox_transpose(concat_20_tmp_0, transpose_20_tmp_0, reshape2_40_tmp_0_dim, transpose_20_tmp_0_dim, axis_176, 5);
	free(concat_20_tmp_0);

	float* tmp_17 = (float*)calloc(30720, sizeof(float));
	autox_elementwise_add(transpose_16_tmp_0, transpose_16_tmp_0, tmp_17, reshape2_33_tmp_0_dim, reshape2_33_tmp_0_dim, tmp_17_dim, -1, 4, 4, 4);
	free(transpose_16_tmp_0);
	// free(transpose_16_tmp_0);

	float* batch_norm_89_tmp_2 = (float*)calloc(7680, sizeof(float));
	autox_conv2d(transpose_18_tmp_0, (float*)((int8_t*)weights) + 176720, (float*)((int8_t*)weights) + 176760, batch_norm_89_tmp_2, reshape2_37_tmp_0_dim, conv2d_89_w_0_dim, batch_norm_89_tmp_2_dim, 1, 0, 1, 1, 0);
	// free(transpose_18_tmp_0);

	float* nearest_interp_v2_5_tmp_0 = (float*)calloc(30720, sizeof(float));
	autox_nearest_interp(batch_norm_89_tmp_2, nearest_interp_v2_5_tmp_0, batch_norm_89_tmp_2_dim, nearest_interp_v2_5_tmp_0_dim, 2, 0);
	free(batch_norm_89_tmp_2);

	float* tmp_18 = (float*)calloc(30720, sizeof(float));
	autox_elementwise_add(tmp_17, nearest_interp_v2_5_tmp_0, tmp_18, tmp_17_dim, nearest_interp_v2_5_tmp_0_dim, tmp_18_dim, -1, 4, 4, 4);
	free(tmp_17);
	free(nearest_interp_v2_5_tmp_0);

	float* batch_norm_90_tmp_2 = (float*)calloc(1920, sizeof(float));
	autox_conv2d(transpose_20_tmp_0, (float*)((int8_t*)weights) + 179960, (float*)((int8_t*)weights) + 180000, batch_norm_90_tmp_2, reshape2_41_tmp_0_dim, conv2d_90_w_0_dim, batch_norm_90_tmp_2_dim, 1, 0, 1, 1, 0);
	// free(transpose_20_tmp_0);

	float* nearest_interp_v2_6_tmp_0 = (float*)calloc(30720, sizeof(float));
	autox_nearest_interp(batch_norm_90_tmp_2, nearest_interp_v2_6_tmp_0, batch_norm_90_tmp_2_dim, nearest_interp_v2_6_tmp_0_dim, 4, 0);
	free(batch_norm_90_tmp_2);

	float* tmp_19 = (float*)calloc(30720, sizeof(float));
	autox_elementwise_add(tmp_18, nearest_interp_v2_6_tmp_0, tmp_19, tmp_18_dim, nearest_interp_v2_6_tmp_0_dim, tmp_19_dim, -1, 4, 4, 4);
	free(tmp_18);
	free(nearest_interp_v2_6_tmp_0);

	float*	relu_55_tmp_0 = (float*)calloc(30720, sizeof(float));
	autox_relu_noreplace(tmp_19, relu_55_tmp_0, tmp_19_dim, 4);

	float* batch_norm_91_tmp_2 = (float*)calloc(7680, sizeof(float));
	autox_conv2d(tmp_19, (float*)((int8_t*)weights) + 186400, (float*)((int8_t*)weights) + 186440, batch_norm_91_tmp_2, tmp_19_dim, conv2d_91_w_0_dim, batch_norm_91_tmp_2_dim, 40, 1, 2, 1, 0);
	//free(tmp_19);

	float* batch_norm_92_tmp_2 = (float*)calloc(15360, sizeof(float));
	autox_conv2d(batch_norm_91_tmp_2, (float*)((int8_t*)weights) + 186800, (float*)((int8_t*)weights) + 186880, batch_norm_92_tmp_2, batch_norm_91_tmp_2_dim, conv2d_92_w_0_dim, batch_norm_92_tmp_2_dim, 1, 0, 1, 1, 0);
	free(batch_norm_91_tmp_2);

	float* tmp_20 = (float*)calloc(15360, sizeof(float));
	autox_elementwise_add(batch_norm_92_tmp_2, batch_norm_92_tmp_2, tmp_20, batch_norm_92_tmp_2_dim, batch_norm_92_tmp_2_dim, tmp_20_dim, -1, 4, 4, 4);
	free(batch_norm_92_tmp_2);
	//free(batch_norm_92_tmp_2);

	float* tmp_21 = (float*)calloc(15360, sizeof(float));
	autox_elementwise_add(tmp_20,transpose_18_tmp_0, tmp_21, tmp_20_dim, reshape2_37_tmp_0_dim, tmp_21_dim, -1, 4, 4, 4);
	free(tmp_20);
	// free(transpose_18_tmp_0);

	float* batch_norm_93_tmp_2 = (float*)calloc(3840, sizeof(float));
	autox_conv2d(transpose_20_tmp_0, (float*)((int8_t*)weights) + 190080, (float*)((int8_t*)weights) + 190160, batch_norm_93_tmp_2, reshape2_41_tmp_0_dim, conv2d_93_w_0_dim, batch_norm_93_tmp_2_dim, 1, 0, 1, 1, 0);
	//free(transpose_20_tmp_0);

	float* nearest_interp_v2_7_tmp_0 = (float*)calloc(15360, sizeof(float));
	autox_nearest_interp(batch_norm_93_tmp_2, nearest_interp_v2_7_tmp_0, batch_norm_93_tmp_2_dim, nearest_interp_v2_7_tmp_0_dim, 2, 0);
	free(batch_norm_93_tmp_2);

	float* batch_norm_94_tmp_2 = (float*)calloc(7680, sizeof(float));
	autox_conv2d(tmp_19, (float*)((int8_t*)weights) + 202960, (float*)((int8_t*)weights) + 203000, batch_norm_94_tmp_2, tmp_19_dim, conv2d_94_w_0_dim, batch_norm_94_tmp_2_dim, 40, 1, 2, 1, 0);
	//free(tmp_19);

	float* relu_57_tmp_0 = (float*)calloc(7680, sizeof(float));
	autox_conv2d(batch_norm_94_tmp_2, (float*)((int8_t*)weights) + 203360, (float*)((int8_t*)weights) + 203400, relu_57_tmp_0, batch_norm_94_tmp_2_dim, conv2d_95_w_0_dim, relu_57_tmp_0_dim, 1, 0, 1, 1, 1);
	free(batch_norm_94_tmp_2);

	float* batch_norm_96_tmp_2 = (float*)calloc(1920, sizeof(float));
	autox_conv2d(relu_57_tmp_0, (float*)((int8_t*)weights) + 205000, (float*)((int8_t*)weights) + 205040, batch_norm_96_tmp_2, relu_57_tmp_0_dim, conv2d_96_w_0_dim, batch_norm_96_tmp_2_dim, 40, 1, 2, 1, 0);
	free(relu_57_tmp_0);

	float* batch_norm_97_tmp_2 = (float*)calloc(7680, sizeof(float));
	autox_conv2d(batch_norm_96_tmp_2, (float*)((int8_t*)weights) + 205400, (float*)((int8_t*)weights) + 205560, batch_norm_97_tmp_2, batch_norm_96_tmp_2_dim, conv2d_97_w_0_dim, batch_norm_97_tmp_2_dim, 1, 0, 1, 1, 0);
	free(batch_norm_96_tmp_2);

	float* tmp_23 = (float*)calloc(7680, sizeof(float));
	autox_elementwise_add(batch_norm_97_tmp_2, batch_norm_97_tmp_2, tmp_23, batch_norm_97_tmp_2_dim, batch_norm_97_tmp_2_dim, tmp_23_dim, -1, 4, 4, 4);
	free(batch_norm_97_tmp_2);
	// free(batch_norm_97_tmp_2);

	float* batch_norm_98_tmp_2 = (float*)calloc(3840, sizeof(float));
	autox_conv2d(transpose_18_tmp_0, (float*)((int8_t*)weights) + 211960, (float*)((int8_t*)weights) + 212040, batch_norm_98_tmp_2, reshape2_37_tmp_0_dim, conv2d_98_w_0_dim, batch_norm_98_tmp_2_dim, 80, 1, 2, 1, 0);
	free(transpose_18_tmp_0);

	float* batch_norm_99_tmp_2 = (float*)calloc(7680, sizeof(float));
	autox_conv2d(batch_norm_98_tmp_2, (float*)((int8_t*)weights) + 212760, (float*)((int8_t*)weights) + 212920, batch_norm_99_tmp_2, batch_norm_98_tmp_2_dim, conv2d_99_w_0_dim, batch_norm_99_tmp_2_dim, 1, 0, 1, 1, 0);
	free(batch_norm_98_tmp_2);

	float* tmp_24 = (float*)calloc(7680, sizeof(float));
	autox_elementwise_add(tmp_23, batch_norm_99_tmp_2, tmp_24, tmp_23_dim, batch_norm_99_tmp_2_dim, tmp_24_dim, -1, 4, 4, 4);
	free(tmp_23);
	free(batch_norm_99_tmp_2);

	float* split_21_tmp_0 = (float*)calloc(15360, sizeof(float));
	float* split_21_tmp_1 = (float*)calloc(15360, sizeof(float));
	float* p_199[] = { split_21_tmp_0, split_21_tmp_1, };
	uint16_t* p_199_dim[] = { split_21_tmp_0_dim, split_21_tmp_1_dim, };
	autox_split(relu_55_tmp_0, p_199, relu_55_tmp_0_dim, p_199_dim, 1, 4, 2, 4);
	free(relu_55_tmp_0);

	float* relu_59_tmp_0 = (float*)calloc(15360, sizeof(float));
	autox_conv2d(split_21_tmp_1, (float*)((int8_t*)weights) + 225720, (float*)((int8_t*)weights) + 225740, relu_59_tmp_0, split_21_tmp_1_dim, conv2d_100_w_0_dim, relu_59_tmp_0_dim, 1, 0, 1, 1, 1);
	free(split_21_tmp_1);

	float* batch_norm_101_tmp_2 = (float*)calloc(15360, sizeof(float));
	autox_conv2d(relu_59_tmp_0, (float*)((int8_t*)weights) + 226140, (float*)((int8_t*)weights) + 226160, batch_norm_101_tmp_2, relu_59_tmp_0_dim, conv2d_101_w_0_dim, batch_norm_101_tmp_2_dim, 20, 1, 1, 1, 0);
	free(relu_59_tmp_0);

	float* relu_60_tmp_0 = (float*)calloc(15360, sizeof(float));
	autox_conv2d(batch_norm_101_tmp_2, (float*)((int8_t*)weights) + 226340, (float*)((int8_t*)weights) + 226360, relu_60_tmp_0, batch_norm_101_tmp_2_dim, conv2d_102_w_0_dim, relu_60_tmp_0_dim, 1, 0, 1, 1, 1);
	free(batch_norm_101_tmp_2);

	float* p_203[] = { split_21_tmp_0, relu_60_tmp_0, };
	uint16_t* p_203_dim[] = { split_21_tmp_0_dim, relu_60_tmp_0_dim, };
	float* concat_21_tmp_0 = (float*)calloc(30720, sizeof(float));
	autox_concat(p_203, concat_21_tmp_0, p_203_dim, concat_21_tmp_0_dim, 1, 2, 4);
	free(split_21_tmp_0);
	free(relu_60_tmp_0);

	float* transpose_21_tmp_0 = (float*)calloc(30720, sizeof(float));
	uint16_t axis_204[] = { 0, 2, 1, 3, 4 };
	autox_transpose(concat_21_tmp_0, transpose_21_tmp_0, reshape2_42_tmp_0_dim, transpose_21_tmp_0_dim, axis_204, 5);
	free(concat_21_tmp_0);

	float* split_22_tmp_0 = (float*)calloc(15360, sizeof(float));
	float* split_22_tmp_1 = (float*)calloc(15360, sizeof(float));
	float* p_205[] = { split_22_tmp_0, split_22_tmp_1, };
	uint16_t* p_205_dim[] = { split_22_tmp_0_dim, split_22_tmp_1_dim, };
	autox_split(transpose_21_tmp_0, p_205, reshape2_43_tmp_0_dim, p_205_dim, 1, 4, 2, 4);
	free(transpose_21_tmp_0);

	float* relu_61_tmp_0 = (float*)calloc(15360, sizeof(float));
	autox_conv2d(split_22_tmp_1, (float*)((int8_t*)weights) + 226760, (float*)((int8_t*)weights) + 226780, relu_61_tmp_0, split_22_tmp_1_dim, conv2d_103_w_0_dim, relu_61_tmp_0_dim, 1, 0, 1, 1, 1);
	free(split_22_tmp_1);

	float* batch_norm_104_tmp_2 = (float*)calloc(15360, sizeof(float));
	autox_conv2d(relu_61_tmp_0, (float*)((int8_t*)weights) + 227180, (float*)((int8_t*)weights) + 227200, batch_norm_104_tmp_2, relu_61_tmp_0_dim, conv2d_104_w_0_dim, batch_norm_104_tmp_2_dim, 20, 1, 1, 1, 0);
	free(relu_61_tmp_0);

	float* relu_62_tmp_0 = (float*)calloc(15360, sizeof(float));
	autox_conv2d(batch_norm_104_tmp_2, (float*)((int8_t*)weights) + 227380, (float*)((int8_t*)weights) + 227400, relu_62_tmp_0, batch_norm_104_tmp_2_dim, conv2d_105_w_0_dim, relu_62_tmp_0_dim, 1, 0, 1, 1, 1);
	free(batch_norm_104_tmp_2);

	float* p_209[] = { split_22_tmp_0, relu_62_tmp_0, };
	uint16_t* p_209_dim[] = { split_22_tmp_0_dim, relu_62_tmp_0_dim, };
	float* concat_22_tmp_0 = (float*)calloc(30720, sizeof(float));
	autox_concat(p_209, concat_22_tmp_0, p_209_dim, concat_22_tmp_0_dim, 1, 2, 4);
	free(split_22_tmp_0);
	free(relu_62_tmp_0);

	float* transpose_22_tmp_0 = (float*)calloc(30720, sizeof(float));
	uint16_t axis_210[] = { 0, 2, 1, 3, 4 };
	autox_transpose(concat_22_tmp_0, transpose_22_tmp_0, reshape2_44_tmp_0_dim, transpose_22_tmp_0_dim, axis_210, 5);
	free(concat_22_tmp_0);

	float* relu_56_tmp_0 = (float*)calloc(15360, sizeof(float));
	autox_fusion_elementwise_add_activation(tmp_21, nearest_interp_v2_7_tmp_0, relu_56_tmp_0, tmp_21_dim, nearest_interp_v2_7_tmp_0_dim, relu_56_tmp_0_dim, -1, 4, 4, 4);
	free(tmp_21);
	free(nearest_interp_v2_7_tmp_0);

	float* split_23_tmp_0 = (float*)calloc(7680, sizeof(float));
	float* split_23_tmp_1 = (float*)calloc(7680, sizeof(float));
	float* p_212[] = { split_23_tmp_0, split_23_tmp_1, };
	uint16_t* p_212_dim[] = { split_23_tmp_0_dim, split_23_tmp_1_dim, };
	autox_split(relu_56_tmp_0, p_212, relu_56_tmp_0_dim, p_212_dim, 1, 4, 2, 4);
	free(relu_56_tmp_0);

	float* relu_63_tmp_0 = (float*)calloc(7680, sizeof(float));
	autox_conv2d(split_23_tmp_1, (float*)((int8_t*)weights) + 227800, (float*)((int8_t*)weights) + 227840, relu_63_tmp_0, split_23_tmp_1_dim, conv2d_106_w_0_dim, relu_63_tmp_0_dim, 1, 0, 1, 1, 1);
	free(split_23_tmp_1);

	float* batch_norm_107_tmp_2 = (float*)calloc(7680, sizeof(float));
	autox_conv2d(relu_63_tmp_0, (float*)((int8_t*)weights) + 229440, (float*)((int8_t*)weights) + 229480, batch_norm_107_tmp_2, relu_63_tmp_0_dim, conv2d_107_w_0_dim, batch_norm_107_tmp_2_dim, 40, 1, 1, 1, 0);
	free(relu_63_tmp_0);

	float* relu_64_tmp_0 = (float*)calloc(7680, sizeof(float));
	autox_conv2d(batch_norm_107_tmp_2, (float*)((int8_t*)weights) + 229840, (float*)((int8_t*)weights) + 229880, relu_64_tmp_0, batch_norm_107_tmp_2_dim, conv2d_108_w_0_dim, relu_64_tmp_0_dim, 1, 0, 1, 1, 1);
	free(batch_norm_107_tmp_2);

	float* p_216[] = { split_23_tmp_0, relu_64_tmp_0, };
	uint16_t* p_216_dim[] = { split_23_tmp_0_dim, relu_64_tmp_0_dim, };
	float* concat_23_tmp_0 = (float*)calloc(15360, sizeof(float));
	autox_concat(p_216, concat_23_tmp_0, p_216_dim, concat_23_tmp_0_dim, 1, 2, 4);
	free(split_23_tmp_0);
	free(relu_64_tmp_0);

	float* transpose_23_tmp_0 = (float*)calloc(15360, sizeof(float));
	uint16_t axis_217[] = { 0, 2, 1, 3, 4 };
	autox_transpose(concat_23_tmp_0, transpose_23_tmp_0, reshape2_46_tmp_0_dim, transpose_23_tmp_0_dim, axis_217, 5);
	free(concat_23_tmp_0);

	float* split_24_tmp_0 = (float*)calloc(7680, sizeof(float));
	float* split_24_tmp_1 = (float*)calloc(7680, sizeof(float));
	float* p_218[] = { split_24_tmp_0, split_24_tmp_1, };
	uint16_t* p_218_dim[] = { split_24_tmp_0_dim, split_24_tmp_1_dim, };
	autox_split(transpose_23_tmp_0, p_218, reshape2_47_tmp_0_dim, p_218_dim, 1, 4, 2, 4);
	free(transpose_23_tmp_0);

	float* relu_65_tmp_0 = (float*)calloc(7680, sizeof(float));
	autox_conv2d(split_24_tmp_1, (float*)((int8_t*)weights) + 231480, (float*)((int8_t*)weights) + 231520, relu_65_tmp_0, split_24_tmp_1_dim, conv2d_109_w_0_dim, relu_65_tmp_0_dim, 1, 0, 1, 1, 1);
	free(split_24_tmp_1);

	float* batch_norm_110_tmp_2 = (float*)calloc(7680, sizeof(float));
	autox_conv2d(relu_65_tmp_0, (float*)((int8_t*)weights) + 233120, (float*)((int8_t*)weights) + 233160, batch_norm_110_tmp_2, relu_65_tmp_0_dim, conv2d_110_w_0_dim, batch_norm_110_tmp_2_dim, 40, 1, 1, 1, 0);
	free(relu_65_tmp_0);

	float* relu_66_tmp_0 = (float*)calloc(7680, sizeof(float));
	autox_conv2d(batch_norm_110_tmp_2, (float*)((int8_t*)weights) + 233520, (float*)((int8_t*)weights) + 233560, relu_66_tmp_0, batch_norm_110_tmp_2_dim, conv2d_111_w_0_dim, relu_66_tmp_0_dim, 1, 0, 1, 1, 1);
	free(batch_norm_110_tmp_2);

	float* p_222[] = { split_24_tmp_0, relu_66_tmp_0, };
	uint16_t* p_222_dim[] = { split_24_tmp_0_dim, relu_66_tmp_0_dim, };
	float* concat_24_tmp_0 = (float*)calloc(15360, sizeof(float));
	autox_concat(p_222, concat_24_tmp_0, p_222_dim, concat_24_tmp_0_dim, 1, 2, 4);
	free(split_24_tmp_0);
	free(relu_66_tmp_0);

	float* transpose_24_tmp_0 = (float*)calloc(15360, sizeof(float));
	uint16_t axis_223[] = { 0, 2, 1, 3, 4 };
	autox_transpose(concat_24_tmp_0, transpose_24_tmp_0, reshape2_48_tmp_0_dim, transpose_24_tmp_0_dim, axis_223, 5);
	free(concat_24_tmp_0);

	float* relu_58_tmp_0 = (float*)calloc(7680, sizeof(float));
	autox_fusion_elementwise_add_activation(tmp_24, transpose_20_tmp_0, relu_58_tmp_0, tmp_24_dim, reshape2_41_tmp_0_dim, relu_58_tmp_0_dim, -1, 4, 4, 4);
	free(tmp_24);
	free(transpose_20_tmp_0);

	float* split_25_tmp_0 = (float*)calloc(3840, sizeof(float));
	float* split_25_tmp_1 = (float*)calloc(3840, sizeof(float));
	float* p_225[] = { split_25_tmp_0, split_25_tmp_1, };
	uint16_t* p_225_dim[] = { split_25_tmp_0_dim, split_25_tmp_1_dim, };
	autox_split(relu_58_tmp_0, p_225, relu_58_tmp_0_dim, p_225_dim, 1, 4, 2, 4);
	free(relu_58_tmp_0);

	float* relu_67_tmp_0 = (float*)calloc(3840, sizeof(float));
	autox_conv2d(split_25_tmp_1, (float*)((int8_t*)weights) + 235160, (float*)((int8_t*)weights) + 235240, relu_67_tmp_0, split_25_tmp_1_dim, conv2d_112_w_0_dim, relu_67_tmp_0_dim, 1, 0, 1, 1, 1);
	free(split_25_tmp_1);

	float* batch_norm_113_tmp_2 = (float*)calloc(3840, sizeof(float));
	autox_conv2d(relu_67_tmp_0, (float*)((int8_t*)weights) + 241640, (float*)((int8_t*)weights) + 241720, batch_norm_113_tmp_2, relu_67_tmp_0_dim, conv2d_113_w_0_dim, batch_norm_113_tmp_2_dim, 80, 1, 1, 1, 0);
	free(relu_67_tmp_0);

	float* relu_68_tmp_0 = (float*)calloc(3840, sizeof(float));
	autox_conv2d(batch_norm_113_tmp_2, (float*)((int8_t*)weights) + 242440, (float*)((int8_t*)weights) + 242520, relu_68_tmp_0, batch_norm_113_tmp_2_dim, conv2d_114_w_0_dim, relu_68_tmp_0_dim, 1, 0, 1, 1, 1);
	free(batch_norm_113_tmp_2);

	float* p_229[] = { split_25_tmp_0, relu_68_tmp_0, };
	uint16_t* p_229_dim[] = { split_25_tmp_0_dim, relu_68_tmp_0_dim, };
	float* concat_25_tmp_0 = (float*)calloc(7680, sizeof(float));
	autox_concat(p_229, concat_25_tmp_0, p_229_dim, concat_25_tmp_0_dim, 1, 2, 4);
	free(split_25_tmp_0);
	free(relu_68_tmp_0);

	float* transpose_25_tmp_0 = (float*)calloc(7680, sizeof(float));
	uint16_t axis_230[] = { 0, 2, 1, 3, 4 };
	autox_transpose(concat_25_tmp_0, transpose_25_tmp_0, reshape2_50_tmp_0_dim, transpose_25_tmp_0_dim, axis_230, 5);
	free(concat_25_tmp_0);

	float* split_26_tmp_0 = (float*)calloc(3840, sizeof(float));
	float* split_26_tmp_1 = (float*)calloc(3840, sizeof(float));
	float* p_231[] = { split_26_tmp_0, split_26_tmp_1, };
	uint16_t* p_231_dim[] = { split_26_tmp_0_dim, split_26_tmp_1_dim, };
	autox_split(transpose_25_tmp_0, p_231, reshape2_51_tmp_0_dim, p_231_dim, 1, 4, 2, 4);
	free(transpose_25_tmp_0);

	float* relu_69_tmp_0 = (float*)calloc(3840, sizeof(float));
	autox_conv2d(split_26_tmp_1, (float*)((int8_t*)weights) + 248920, (float*)((int8_t*)weights) + 249000, relu_69_tmp_0, split_26_tmp_1_dim, conv2d_115_w_0_dim, relu_69_tmp_0_dim, 1, 0, 1, 1, 1);
	free(split_26_tmp_1);

	float* batch_norm_116_tmp_2 = (float*)calloc(3840, sizeof(float));
	autox_conv2d(relu_69_tmp_0, (float*)((int8_t*)weights) + 255400, (float*)((int8_t*)weights) + 255480, batch_norm_116_tmp_2, relu_69_tmp_0_dim, conv2d_116_w_0_dim, batch_norm_116_tmp_2_dim, 80, 1, 1, 1, 0);
	free(relu_69_tmp_0);

	float* relu_70_tmp_0 = (float*)calloc(3840, sizeof(float));
	autox_conv2d(batch_norm_116_tmp_2, (float*)((int8_t*)weights) + 256200, (float*)((int8_t*)weights) + 256280, relu_70_tmp_0, batch_norm_116_tmp_2_dim, conv2d_117_w_0_dim, relu_70_tmp_0_dim, 1, 0, 1, 1, 1);
	free(batch_norm_116_tmp_2);

	float* p_235[] = { split_26_tmp_0, relu_70_tmp_0, };
	uint16_t* p_235_dim[] = { split_26_tmp_0_dim, relu_70_tmp_0_dim, };
	float* concat_26_tmp_0 = (float*)calloc(7680, sizeof(float));
	autox_concat(p_235, concat_26_tmp_0, p_235_dim, concat_26_tmp_0_dim, 1, 2, 4);
	free(split_26_tmp_0);
	free(relu_70_tmp_0);

	float* transpose_26_tmp_0 = (float*)calloc(7680, sizeof(float));
	uint16_t axis_236[] = { 0, 2, 1, 3, 4 };
	autox_transpose(concat_26_tmp_0, transpose_26_tmp_0, reshape2_52_tmp_0_dim, transpose_26_tmp_0_dim, axis_236, 5);
	free(concat_26_tmp_0);

	float* tmp_26 = (float*)calloc(30720, sizeof(float));
	autox_elementwise_add(transpose_22_tmp_0, transpose_22_tmp_0, tmp_26, reshape2_45_tmp_0_dim, reshape2_45_tmp_0_dim, tmp_26_dim, -1, 4, 4, 4);
	free(transpose_22_tmp_0);
	// free(transpose_22_tmp_0);

	float* batch_norm_118_tmp_2 = (float*)calloc(7680, sizeof(float));
	autox_conv2d(transpose_24_tmp_0, (float*)((int8_t*)weights) + 262680, (float*)((int8_t*)weights) + 262720, batch_norm_118_tmp_2, reshape2_49_tmp_0_dim, conv2d_118_w_0_dim, batch_norm_118_tmp_2_dim, 1, 0, 1, 1, 0);
	//free(transpose_24_tmp_0);

	float* nearest_interp_v2_8_tmp_0 = (float*)calloc(30720, sizeof(float));
	autox_nearest_interp(batch_norm_118_tmp_2, nearest_interp_v2_8_tmp_0, batch_norm_118_tmp_2_dim, nearest_interp_v2_8_tmp_0_dim, 2, 0);
	free(batch_norm_118_tmp_2);

	float* tmp_27 = (float*)calloc(30720, sizeof(float));
	autox_elementwise_add(tmp_26, nearest_interp_v2_8_tmp_0, tmp_27, tmp_26_dim, nearest_interp_v2_8_tmp_0_dim, tmp_27_dim, -1, 4, 4, 4);
	free(tmp_26);
	free(nearest_interp_v2_8_tmp_0);

	float* batch_norm_119_tmp_2 = (float*)calloc(1920, sizeof(float));
	autox_conv2d(transpose_26_tmp_0, (float*)((int8_t*)weights) + 265920, (float*)((int8_t*)weights) + 265960, batch_norm_119_tmp_2, reshape2_53_tmp_0_dim, conv2d_119_w_0_dim, batch_norm_119_tmp_2_dim, 1, 0, 1, 1, 0);
	//free(transpose_26_tmp_0);

	float* nearest_interp_v2_9_tmp_0 = (float*)calloc(30720, sizeof(float));
	autox_nearest_interp(batch_norm_119_tmp_2, nearest_interp_v2_9_tmp_0, batch_norm_119_tmp_2_dim, nearest_interp_v2_9_tmp_0_dim, 4, 0);
	free(batch_norm_119_tmp_2);

	float* tmp_28 = (float*)calloc(30720, sizeof(float));
	autox_elementwise_add(tmp_27, nearest_interp_v2_9_tmp_0, tmp_28, tmp_27_dim, nearest_interp_v2_9_tmp_0_dim, tmp_28_dim, -1, 4, 4, 4);
	free(tmp_27);
	free(nearest_interp_v2_9_tmp_0);

	float*	relu_71_tmp_0 = (float*)calloc(30720, sizeof(float));
	autox_relu_noreplace(tmp_28, relu_71_tmp_0, tmp_28_dim, 4);

	float* batch_norm_120_tmp_2 = (float*)calloc(7680, sizeof(float));
	autox_conv2d(tmp_28, (float*)((int8_t*)weights) + 272360, (float*)((int8_t*)weights) + 272400, batch_norm_120_tmp_2, tmp_28_dim, conv2d_120_w_0_dim, batch_norm_120_tmp_2_dim, 40, 1, 2, 1, 0);
	// free(tmp_28);

	float* batch_norm_121_tmp_2 = (float*)calloc(15360, sizeof(float));
	autox_conv2d(batch_norm_120_tmp_2, (float*)((int8_t*)weights) + 272760, (float*)((int8_t*)weights) + 272840, batch_norm_121_tmp_2, batch_norm_120_tmp_2_dim, conv2d_121_w_0_dim, batch_norm_121_tmp_2_dim, 1, 0, 1, 1, 0);
	free(batch_norm_120_tmp_2);

	float* tmp_29 = (float*)calloc(15360, sizeof(float));
	autox_elementwise_add(batch_norm_121_tmp_2, batch_norm_121_tmp_2, tmp_29, batch_norm_121_tmp_2_dim, batch_norm_121_tmp_2_dim, tmp_29_dim, -1, 4, 4, 4);
	free(batch_norm_121_tmp_2);
	//free(batch_norm_121_tmp_2);

	float* tmp_30 = (float*)calloc(15360, sizeof(float));
	autox_elementwise_add(tmp_29, transpose_24_tmp_0, tmp_30, tmp_29_dim, reshape2_49_tmp_0_dim, tmp_30_dim, -1, 4, 4, 4);
	free(tmp_29);
	//free(transpose_24_tmp_0);

	float* batch_norm_122_tmp_2 = (float*)calloc(3840, sizeof(float));
	autox_conv2d(transpose_26_tmp_0, (float*)((int8_t*)weights) + 276040, (float*)((int8_t*)weights) + 276120, batch_norm_122_tmp_2, reshape2_53_tmp_0_dim, conv2d_122_w_0_dim, batch_norm_122_tmp_2_dim, 1, 0, 1, 1, 0);
	//free(transpose_26_tmp_0);

	float* nearest_interp_v2_10_tmp_0 = (float*)calloc(15360, sizeof(float));
	autox_nearest_interp(batch_norm_122_tmp_2, nearest_interp_v2_10_tmp_0, batch_norm_122_tmp_2_dim, nearest_interp_v2_10_tmp_0_dim, 2, 0);
	free(batch_norm_122_tmp_2);

	float* batch_norm_123_tmp_2 = (float*)calloc(7680, sizeof(float));
	autox_conv2d(tmp_28, (float*)((int8_t*)weights) + 288920, (float*)((int8_t*)weights) + 288960, batch_norm_123_tmp_2, tmp_28_dim, conv2d_123_w_0_dim, batch_norm_123_tmp_2_dim, 40, 1, 2, 1, 0);
	// free(tmp_28);

	float* relu_73_tmp_0 = (float*)calloc(7680, sizeof(float));
	autox_conv2d(batch_norm_123_tmp_2, (float*)((int8_t*)weights) + 289320, (float*)((int8_t*)weights) + 289360, relu_73_tmp_0, batch_norm_123_tmp_2_dim, conv2d_124_w_0_dim, relu_73_tmp_0_dim, 1, 0, 1, 1, 1);
	free(batch_norm_123_tmp_2);

	float* batch_norm_125_tmp_2 = (float*)calloc(1920, sizeof(float));
	autox_conv2d(relu_73_tmp_0, (float*)((int8_t*)weights) + 290960, (float*)((int8_t*)weights) + 291000, batch_norm_125_tmp_2, relu_73_tmp_0_dim, conv2d_125_w_0_dim, batch_norm_125_tmp_2_dim, 40, 1, 2, 1, 0);
	free(relu_73_tmp_0);

	float* batch_norm_126_tmp_2 = (float*)calloc(7680, sizeof(float));
	autox_conv2d(batch_norm_125_tmp_2, (float*)((int8_t*)weights) + 291360, (float*)((int8_t*)weights) + 291520, batch_norm_126_tmp_2, batch_norm_125_tmp_2_dim, conv2d_126_w_0_dim, batch_norm_126_tmp_2_dim, 1, 0, 1, 1, 0);
	free(batch_norm_125_tmp_2);

	float* tmp_32 = (float*)calloc(7680, sizeof(float));
	autox_elementwise_add(batch_norm_126_tmp_2, batch_norm_126_tmp_2, tmp_32, batch_norm_126_tmp_2_dim, batch_norm_126_tmp_2_dim, tmp_32_dim, -1, 4, 4, 4);
	free(batch_norm_126_tmp_2);
	//free(batch_norm_126_tmp_2);

	float* batch_norm_127_tmp_2 = (float*)calloc(3840, sizeof(float));
	autox_conv2d(transpose_24_tmp_0, (float*)((int8_t*)weights) + 297920, (float*)((int8_t*)weights) + 298000, batch_norm_127_tmp_2, reshape2_49_tmp_0_dim, conv2d_127_w_0_dim, batch_norm_127_tmp_2_dim, 80, 1, 2, 1, 0);
	free(transpose_24_tmp_0);

	float* batch_norm_128_tmp_2 = (float*)calloc(7680, sizeof(float));
	autox_conv2d(batch_norm_127_tmp_2, (float*)((int8_t*)weights) + 298720, (float*)((int8_t*)weights) + 298880, batch_norm_128_tmp_2, batch_norm_127_tmp_2_dim, conv2d_128_w_0_dim, batch_norm_128_tmp_2_dim, 1, 0, 1, 1, 0);
	free(batch_norm_127_tmp_2);

	float* tmp_33 = (float*)calloc(7680, sizeof(float));
	autox_elementwise_add(tmp_32, batch_norm_128_tmp_2, tmp_33, tmp_32_dim, batch_norm_128_tmp_2_dim, tmp_33_dim, -1, 4, 4, 4);
	free(tmp_32);
	free(batch_norm_128_tmp_2);

	float* split_27_tmp_0 = (float*)calloc(15360, sizeof(float));
	float* split_27_tmp_1 = (float*)calloc(15360, sizeof(float));
	float* p_259[] = { split_27_tmp_0, split_27_tmp_1, };
	uint16_t* p_259_dim[] = { split_27_tmp_0_dim, split_27_tmp_1_dim, };
	autox_split(relu_71_tmp_0, p_259, relu_71_tmp_0_dim, p_259_dim, 1, 4, 2, 4);
	free(relu_71_tmp_0);

	float* relu_75_tmp_0 = (float*)calloc(15360, sizeof(float));
	autox_conv2d(split_27_tmp_1, (float*)((int8_t*)weights) + 311680, (float*)((int8_t*)weights) + 311700, relu_75_tmp_0, split_27_tmp_1_dim, conv2d_129_w_0_dim, relu_75_tmp_0_dim, 1, 0, 1, 1, 1);
	free(split_27_tmp_1);

	float* batch_norm_130_tmp_2 = (float*)calloc(15360, sizeof(float));
	autox_conv2d(relu_75_tmp_0, (float*)((int8_t*)weights) + 312100, (float*)((int8_t*)weights) + 312120, batch_norm_130_tmp_2, relu_75_tmp_0_dim, conv2d_130_w_0_dim, batch_norm_130_tmp_2_dim, 20, 1, 1, 1, 0);
	free(relu_75_tmp_0);

	float* relu_76_tmp_0 = (float*)calloc(15360, sizeof(float));
	autox_conv2d(batch_norm_130_tmp_2, (float*)((int8_t*)weights) + 312300, (float*)((int8_t*)weights) + 312320, relu_76_tmp_0, batch_norm_130_tmp_2_dim, conv2d_131_w_0_dim, relu_76_tmp_0_dim, 1, 0, 1, 1, 1);
	free(batch_norm_130_tmp_2);

	float* p_263[] = { split_27_tmp_0, relu_76_tmp_0, };
	uint16_t* p_263_dim[] = { split_27_tmp_0_dim, relu_76_tmp_0_dim, };
	float* concat_27_tmp_0 = (float*)calloc(30720, sizeof(float));
	autox_concat(p_263, concat_27_tmp_0, p_263_dim, concat_27_tmp_0_dim, 1, 2, 4);
	free(split_27_tmp_0);
	free(relu_76_tmp_0);

	float* transpose_27_tmp_0 = (float*)calloc(30720, sizeof(float));
	uint16_t axis_264[] = { 0, 2, 1, 3, 4 };
	autox_transpose(concat_27_tmp_0, transpose_27_tmp_0, reshape2_54_tmp_0_dim, transpose_27_tmp_0_dim, axis_264, 5);
	free(concat_27_tmp_0);

	float* split_28_tmp_0 = (float*)calloc(15360, sizeof(float));
	float* split_28_tmp_1 = (float*)calloc(15360, sizeof(float));
	float* p_265[] = { split_28_tmp_0, split_28_tmp_1, };
	uint16_t* p_265_dim[] = { split_28_tmp_0_dim, split_28_tmp_1_dim, };
	autox_split(transpose_27_tmp_0, p_265, reshape2_55_tmp_0_dim, p_265_dim, 1, 4, 2, 4);
	free(transpose_27_tmp_0);

	float* relu_77_tmp_0 = (float*)calloc(15360, sizeof(float));
	autox_conv2d(split_28_tmp_1, (float*)((int8_t*)weights) + 312720, (float*)((int8_t*)weights) + 312740, relu_77_tmp_0, split_28_tmp_1_dim, conv2d_132_w_0_dim, relu_77_tmp_0_dim, 1, 0, 1, 1, 1);
	free(split_28_tmp_1);

	float* batch_norm_133_tmp_2 = (float*)calloc(15360, sizeof(float));
	autox_conv2d(relu_77_tmp_0, (float*)((int8_t*)weights) + 313140, (float*)((int8_t*)weights) + 313160, batch_norm_133_tmp_2, relu_77_tmp_0_dim, conv2d_133_w_0_dim, batch_norm_133_tmp_2_dim, 20, 1, 1, 1, 0);
	free(relu_77_tmp_0);

	float* relu_78_tmp_0 = (float*)calloc(15360, sizeof(float));
	autox_conv2d(batch_norm_133_tmp_2, (float*)((int8_t*)weights) + 313340, (float*)((int8_t*)weights) + 313360, relu_78_tmp_0, batch_norm_133_tmp_2_dim, conv2d_134_w_0_dim, relu_78_tmp_0_dim, 1, 0, 1, 1, 1);
	free(batch_norm_133_tmp_2);

	float* p_269[] = { split_28_tmp_0, relu_78_tmp_0, };
	uint16_t* p_269_dim[] = { split_28_tmp_0_dim, relu_78_tmp_0_dim, };
	float* concat_28_tmp_0 = (float*)calloc(30720, sizeof(float));
	autox_concat(p_269, concat_28_tmp_0, p_269_dim, concat_28_tmp_0_dim, 1, 2, 4);
	free(split_28_tmp_0);
	free(relu_78_tmp_0);

	float* transpose_28_tmp_0 = (float*)calloc(30720, sizeof(float));
	uint16_t axis_270[] = { 0, 2, 1, 3, 4 };
	autox_transpose(concat_28_tmp_0, transpose_28_tmp_0, reshape2_56_tmp_0_dim, transpose_28_tmp_0_dim, axis_270, 5);
	free(concat_28_tmp_0);

	float* relu_72_tmp_0 = (float*)calloc(15360, sizeof(float));
	autox_fusion_elementwise_add_activation(tmp_30, nearest_interp_v2_10_tmp_0, relu_72_tmp_0, tmp_30_dim, nearest_interp_v2_10_tmp_0_dim, relu_72_tmp_0_dim, -1, 4, 4, 4);
	free(tmp_30);
	free(nearest_interp_v2_10_tmp_0);

	float* split_29_tmp_0 = (float*)calloc(7680, sizeof(float));
	float* split_29_tmp_1 = (float*)calloc(7680, sizeof(float));
	float* p_272[] = { split_29_tmp_0, split_29_tmp_1, };
	uint16_t* p_272_dim[] = { split_29_tmp_0_dim, split_29_tmp_1_dim, };
	autox_split(relu_72_tmp_0, p_272, relu_72_tmp_0_dim, p_272_dim, 1, 4, 2, 4);
	free(relu_72_tmp_0);

	float* relu_79_tmp_0 = (float*)calloc(7680, sizeof(float));
	autox_conv2d(split_29_tmp_1, (float*)((int8_t*)weights) + 313760, (float*)((int8_t*)weights) + 313800, relu_79_tmp_0, split_29_tmp_1_dim, conv2d_135_w_0_dim, relu_79_tmp_0_dim, 1, 0, 1, 1, 1);
	free(split_29_tmp_1);

	float* batch_norm_136_tmp_2 = (float*)calloc(7680, sizeof(float));
	autox_conv2d(relu_79_tmp_0, (float*)((int8_t*)weights) + 315400, (float*)((int8_t*)weights) + 315440, batch_norm_136_tmp_2, relu_79_tmp_0_dim, conv2d_136_w_0_dim, batch_norm_136_tmp_2_dim, 40, 1, 1, 1, 0);
	free(relu_79_tmp_0);

	float* relu_80_tmp_0 = (float*)calloc(7680, sizeof(float));
	autox_conv2d(batch_norm_136_tmp_2, (float*)((int8_t*)weights) + 315800, (float*)((int8_t*)weights) + 315840, relu_80_tmp_0, batch_norm_136_tmp_2_dim, conv2d_137_w_0_dim, relu_80_tmp_0_dim, 1, 0, 1, 1, 1);
	free(batch_norm_136_tmp_2);

	float* p_276[] = { split_29_tmp_0, relu_80_tmp_0, };
	uint16_t* p_276_dim[] = { split_29_tmp_0_dim, relu_80_tmp_0_dim, };
	float* concat_29_tmp_0 = (float*)calloc(15360, sizeof(float));
	autox_concat(p_276, concat_29_tmp_0, p_276_dim, concat_29_tmp_0_dim, 1, 2, 4);
	free(split_29_tmp_0);
	free(relu_80_tmp_0);

	float* transpose_29_tmp_0 = (float*)calloc(15360, sizeof(float));
	uint16_t axis_277[] = { 0, 2, 1, 3, 4 };
	autox_transpose(concat_29_tmp_0, transpose_29_tmp_0, reshape2_58_tmp_0_dim, transpose_29_tmp_0_dim, axis_277, 5);
	free(concat_29_tmp_0);

	float* split_30_tmp_0 = (float*)calloc(7680, sizeof(float));
	float* split_30_tmp_1 = (float*)calloc(7680, sizeof(float));
	float* p_278[] = { split_30_tmp_0, split_30_tmp_1, };
	uint16_t* p_278_dim[] = { split_30_tmp_0_dim, split_30_tmp_1_dim, };
	autox_split(transpose_29_tmp_0, p_278, reshape2_59_tmp_0_dim, p_278_dim, 1, 4, 2, 4);
	free(transpose_29_tmp_0);

	float* relu_81_tmp_0 = (float*)calloc(7680, sizeof(float));
	autox_conv2d(split_30_tmp_1, (float*)((int8_t*)weights) + 317440, (float*)((int8_t*)weights) + 317480, relu_81_tmp_0, split_30_tmp_1_dim, conv2d_138_w_0_dim, relu_81_tmp_0_dim, 1, 0, 1, 1, 1);
	free(split_30_tmp_1);

	float* batch_norm_139_tmp_2 = (float*)calloc(7680, sizeof(float));
	autox_conv2d(relu_81_tmp_0, (float*)((int8_t*)weights) + 319080, (float*)((int8_t*)weights) + 319120, batch_norm_139_tmp_2, relu_81_tmp_0_dim, conv2d_139_w_0_dim, batch_norm_139_tmp_2_dim, 40, 1, 1, 1, 0);
	free(relu_81_tmp_0);

	float* relu_82_tmp_0 = (float*)calloc(7680, sizeof(float));
	autox_conv2d(batch_norm_139_tmp_2, (float*)((int8_t*)weights) + 319480, (float*)((int8_t*)weights) + 319520, relu_82_tmp_0, batch_norm_139_tmp_2_dim, conv2d_140_w_0_dim, relu_82_tmp_0_dim, 1, 0, 1, 1, 1);
	free(batch_norm_139_tmp_2);

	float* p_282[] = { split_30_tmp_0, relu_82_tmp_0, };
	uint16_t* p_282_dim[] = { split_30_tmp_0_dim, relu_82_tmp_0_dim, };
	float* concat_30_tmp_0 = (float*)calloc(15360, sizeof(float));
	autox_concat(p_282, concat_30_tmp_0, p_282_dim, concat_30_tmp_0_dim, 1, 2, 4);
	free(split_30_tmp_0);
	free(relu_82_tmp_0);

	float* transpose_30_tmp_0 = (float*)calloc(15360, sizeof(float));
	uint16_t axis_283[] = { 0, 2, 1, 3, 4 };
	autox_transpose(concat_30_tmp_0, transpose_30_tmp_0, reshape2_60_tmp_0_dim, transpose_30_tmp_0_dim, axis_283, 5);
	free(concat_30_tmp_0);

	float* relu_74_tmp_0 = (float*)calloc(7680, sizeof(float));
	autox_fusion_elementwise_add_activation(tmp_33, transpose_26_tmp_0, relu_74_tmp_0, tmp_33_dim, reshape2_53_tmp_0_dim, relu_74_tmp_0_dim, -1, 4, 4, 4);
	free(tmp_33);
	//free(transpose_30_tmp_0);

	float* split_31_tmp_0 = (float*)calloc(3840, sizeof(float));
	float* split_31_tmp_1 = (float*)calloc(3840, sizeof(float));
	float* p_285[] = { split_31_tmp_0, split_31_tmp_1, };
	uint16_t* p_285_dim[] = { split_31_tmp_0_dim, split_31_tmp_1_dim, };
	autox_split(relu_74_tmp_0, p_285, relu_74_tmp_0_dim, p_285_dim, 1, 4, 2, 4);
	free(relu_74_tmp_0);

	float* relu_83_tmp_0 = (float*)calloc(3840, sizeof(float));
	autox_conv2d(split_31_tmp_1, (float*)((int8_t*)weights) + 321120, (float*)((int8_t*)weights) + 321200, relu_83_tmp_0, split_31_tmp_1_dim, conv2d_141_w_0_dim, relu_83_tmp_0_dim, 1, 0, 1, 1, 1);
	free(split_31_tmp_1);

	float* batch_norm_142_tmp_2 = (float*)calloc(3840, sizeof(float));
	autox_conv2d(relu_83_tmp_0, (float*)((int8_t*)weights) + 327600, (float*)((int8_t*)weights) + 327680, batch_norm_142_tmp_2, relu_83_tmp_0_dim, conv2d_142_w_0_dim, batch_norm_142_tmp_2_dim, 80, 1, 1, 1, 0);
	free(relu_83_tmp_0);

	float* relu_84_tmp_0 = (float*)calloc(3840, sizeof(float));
	autox_conv2d(batch_norm_142_tmp_2, (float*)((int8_t*)weights) + 328400, (float*)((int8_t*)weights) + 328480, relu_84_tmp_0, batch_norm_142_tmp_2_dim, conv2d_143_w_0_dim, relu_84_tmp_0_dim, 1, 0, 1, 1, 1);
	free(batch_norm_142_tmp_2);

	float* p_289[] = { split_31_tmp_0, relu_84_tmp_0, };
	uint16_t* p_289_dim[] = { split_31_tmp_0_dim, relu_84_tmp_0_dim, };
	float* concat_31_tmp_0 = (float*)calloc(7680, sizeof(float));
	autox_concat(p_289, concat_31_tmp_0, p_289_dim, concat_31_tmp_0_dim, 1, 2, 4);
	free(split_31_tmp_0);
	free(relu_84_tmp_0);

	float* transpose_31_tmp_0 = (float*)calloc(7680, sizeof(float));
	uint16_t axis_290[] = { 0, 2, 1, 3, 4 };
	autox_transpose(concat_31_tmp_0, transpose_31_tmp_0, reshape2_62_tmp_0_dim, transpose_31_tmp_0_dim, axis_290, 5);
	free(concat_31_tmp_0);

	float* split_32_tmp_0 = (float*)calloc(3840, sizeof(float));
	float* split_32_tmp_1 = (float*)calloc(3840, sizeof(float));
	float* p_291[] = { split_32_tmp_0, split_32_tmp_1, };
	uint16_t* p_291_dim[] = { split_32_tmp_0_dim, split_32_tmp_1_dim, };
	autox_split(transpose_31_tmp_0, p_291, reshape2_63_tmp_0_dim, p_291_dim, 1, 4, 2, 4);
	free(transpose_31_tmp_0);

	float* relu_85_tmp_0 = (float*)calloc(3840, sizeof(float));
	autox_conv2d(split_32_tmp_1, (float*)((int8_t*)weights) + 334880, (float*)((int8_t*)weights) + 334960, relu_85_tmp_0, split_32_tmp_1_dim, conv2d_144_w_0_dim, relu_85_tmp_0_dim, 1, 0, 1, 1, 1);
	free(split_32_tmp_1);

	float* batch_norm_145_tmp_2 = (float*)calloc(3840, sizeof(float));
	autox_conv2d(relu_85_tmp_0, (float*)((int8_t*)weights) + 341360, (float*)((int8_t*)weights) + 341440, batch_norm_145_tmp_2, relu_85_tmp_0_dim, conv2d_145_w_0_dim, batch_norm_145_tmp_2_dim, 80, 1, 1, 1, 0);
	free(relu_85_tmp_0);

	float* relu_86_tmp_0 = (float*)calloc(3840, sizeof(float));
	autox_conv2d(batch_norm_145_tmp_2, (float*)((int8_t*)weights) + 342160, (float*)((int8_t*)weights) + 342240, relu_86_tmp_0, batch_norm_145_tmp_2_dim, conv2d_146_w_0_dim, relu_86_tmp_0_dim, 1, 0, 1, 1, 1);
	free(batch_norm_145_tmp_2);

	float* p_295[] = { split_32_tmp_0, relu_86_tmp_0, };
	uint16_t* p_295_dim[] = { split_32_tmp_0_dim, relu_86_tmp_0_dim, };
	float* concat_32_tmp_0 = (float*)calloc(7680, sizeof(float));
	autox_concat(p_295, concat_32_tmp_0, p_295_dim, concat_32_tmp_0_dim, 1, 2, 4);
	free(split_32_tmp_0);
	free(relu_86_tmp_0);

	float* transpose_32_tmp_0 = (float*)calloc(7680, sizeof(float));
	uint16_t axis_296[] = { 0, 2, 1, 3, 4 };
	autox_transpose(concat_32_tmp_0, transpose_32_tmp_0, reshape2_64_tmp_0_dim, transpose_32_tmp_0_dim, axis_296, 5);
	free(concat_32_tmp_0);

	float* tmp_35 = (float*)calloc(30720, sizeof(float));
	autox_elementwise_add(transpose_28_tmp_0, transpose_28_tmp_0, tmp_35, reshape2_57_tmp_0_dim, reshape2_57_tmp_0_dim, tmp_35_dim, -1, 4, 4, 4);
	free(transpose_28_tmp_0);
	// free(transpose_28_tmp_0);

	float* batch_norm_147_tmp_2 = (float*)calloc(7680, sizeof(float));
	autox_conv2d(transpose_30_tmp_0, (float*)((int8_t*)weights) + 348640, (float*)((int8_t*)weights) + 348680, batch_norm_147_tmp_2, reshape2_61_tmp_0_dim, conv2d_147_w_0_dim, batch_norm_147_tmp_2_dim, 1, 0, 1, 1, 0);
	//free(transpose_30_tmp_0);

	float* nearest_interp_v2_11_tmp_0 = (float*)calloc(30720, sizeof(float));
	autox_nearest_interp(batch_norm_147_tmp_2, nearest_interp_v2_11_tmp_0, batch_norm_147_tmp_2_dim, nearest_interp_v2_11_tmp_0_dim, 2, 0);
	free(batch_norm_147_tmp_2);

	float* tmp_36 = (float*)calloc(30720, sizeof(float));
	autox_elementwise_add(tmp_35, nearest_interp_v2_11_tmp_0, tmp_36, tmp_35_dim, nearest_interp_v2_11_tmp_0_dim, tmp_36_dim, -1, 4, 4, 4);
	free(tmp_35);
	free(nearest_interp_v2_11_tmp_0);

	float* batch_norm_148_tmp_2 = (float*)calloc(1920, sizeof(float));
	autox_conv2d(transpose_32_tmp_0, (float*)((int8_t*)weights) + 351880, (float*)((int8_t*)weights) + 351920, batch_norm_148_tmp_2, reshape2_65_tmp_0_dim, conv2d_148_w_0_dim, batch_norm_148_tmp_2_dim, 1, 0, 1, 1, 0);
	// free(transpose_32_tmp_0);

	float* nearest_interp_v2_12_tmp_0 = (float*)calloc(30720, sizeof(float));
	autox_nearest_interp(batch_norm_148_tmp_2, nearest_interp_v2_12_tmp_0, batch_norm_148_tmp_2_dim, nearest_interp_v2_12_tmp_0_dim, 4, 0);
	free(batch_norm_148_tmp_2);

	float* tmp_37 = (float*)calloc(30720, sizeof(float));
	autox_elementwise_add(tmp_36, nearest_interp_v2_12_tmp_0, tmp_37, tmp_36_dim, nearest_interp_v2_12_tmp_0_dim, tmp_37_dim, -1, 4, 4, 4);
	free(tmp_36);
	free(nearest_interp_v2_12_tmp_0);

	float*	relu_87_tmp_0 = (float*)calloc(30720, sizeof(float));
	autox_relu_noreplace(tmp_37, relu_87_tmp_0, tmp_37_dim, 4);

	float* batch_norm_149_tmp_2 = (float*)calloc(7680, sizeof(float));
	autox_conv2d(tmp_37, (float*)((int8_t*)weights) + 358320, (float*)((int8_t*)weights) + 358360, batch_norm_149_tmp_2, tmp_37_dim, conv2d_149_w_0_dim, batch_norm_149_tmp_2_dim, 40, 1, 2, 1, 0);
	//free(tmp_37);

	float* batch_norm_150_tmp_2 = (float*)calloc(15360, sizeof(float));
	autox_conv2d(batch_norm_149_tmp_2, (float*)((int8_t*)weights) + 358720, (float*)((int8_t*)weights) + 358800, batch_norm_150_tmp_2, batch_norm_149_tmp_2_dim, conv2d_150_w_0_dim, batch_norm_150_tmp_2_dim, 1, 0, 1, 1, 0);
	free(batch_norm_149_tmp_2);

	float* tmp_38 = (float*)calloc(15360, sizeof(float));
	autox_elementwise_add(batch_norm_150_tmp_2, batch_norm_150_tmp_2, tmp_38, batch_norm_150_tmp_2_dim, batch_norm_150_tmp_2_dim, tmp_38_dim, -1, 4, 4, 4);
	free(batch_norm_150_tmp_2);
	// free(batch_norm_150_tmp_2);

	float* tmp_39 = (float*)calloc(15360, sizeof(float));
	autox_elementwise_add(tmp_38, transpose_30_tmp_0, tmp_39, tmp_38_dim, reshape2_61_tmp_0_dim, tmp_39_dim, -1, 4, 4, 4);
	free(tmp_38);
	// free(transpose_30_tmp_0);

	float* batch_norm_151_tmp_2 = (float*)calloc(3840, sizeof(float));
	autox_conv2d(transpose_32_tmp_0, (float*)((int8_t*)weights) + 362000, (float*)((int8_t*)weights) + 362080, batch_norm_151_tmp_2, reshape2_65_tmp_0_dim, conv2d_151_w_0_dim, batch_norm_151_tmp_2_dim, 1, 0, 1, 1, 0);
	//free(transpose_32_tmp_0);

	float* nearest_interp_v2_13_tmp_0 = (float*)calloc(15360, sizeof(float));
	autox_nearest_interp(batch_norm_151_tmp_2, nearest_interp_v2_13_tmp_0, batch_norm_151_tmp_2_dim, nearest_interp_v2_13_tmp_0_dim, 2, 0);
	free(batch_norm_151_tmp_2);

	float* batch_norm_152_tmp_2 = (float*)calloc(7680, sizeof(float));
	autox_conv2d(tmp_37, (float*)((int8_t*)weights) + 374880, (float*)((int8_t*)weights) + 374920, batch_norm_152_tmp_2, tmp_37_dim, conv2d_152_w_0_dim, batch_norm_152_tmp_2_dim, 40, 1, 2, 1, 0);
	// free(tmp_37);

	float* relu_89_tmp_0 = (float*)calloc(7680, sizeof(float));
	autox_conv2d(batch_norm_152_tmp_2, (float*)((int8_t*)weights) + 375280, (float*)((int8_t*)weights) + 375320, relu_89_tmp_0, batch_norm_152_tmp_2_dim, conv2d_153_w_0_dim, relu_89_tmp_0_dim, 1, 0, 1, 1, 1);
	free(batch_norm_152_tmp_2);

	float* batch_norm_154_tmp_2 = (float*)calloc(1920, sizeof(float));
	autox_conv2d(relu_89_tmp_0, (float*)((int8_t*)weights) + 376920, (float*)((int8_t*)weights) + 376960, batch_norm_154_tmp_2, relu_89_tmp_0_dim, conv2d_154_w_0_dim, batch_norm_154_tmp_2_dim, 40, 1, 2, 1, 0);
	free(relu_89_tmp_0);

	float* batch_norm_155_tmp_2 = (float*)calloc(7680, sizeof(float));
	autox_conv2d(batch_norm_154_tmp_2, (float*)((int8_t*)weights) + 377320, (float*)((int8_t*)weights) + 377480, batch_norm_155_tmp_2, batch_norm_154_tmp_2_dim, conv2d_155_w_0_dim, batch_norm_155_tmp_2_dim, 1, 0, 1, 1, 0);
	free(batch_norm_154_tmp_2);

	float* tmp_41 = (float*)calloc(7680, sizeof(float));
	autox_elementwise_add(batch_norm_155_tmp_2, batch_norm_155_tmp_2, tmp_41, batch_norm_155_tmp_2_dim, batch_norm_155_tmp_2_dim, tmp_41_dim, -1, 4, 4, 4);
	free(batch_norm_155_tmp_2);
	//free(batch_norm_155_tmp_2);

	float* batch_norm_156_tmp_2 = (float*)calloc(3840, sizeof(float));
	autox_conv2d(transpose_30_tmp_0, (float*)((int8_t*)weights) + 383880, (float*)((int8_t*)weights) + 383960, batch_norm_156_tmp_2, reshape2_61_tmp_0_dim, conv2d_156_w_0_dim, batch_norm_156_tmp_2_dim, 80, 1, 2, 1, 0);
	free(transpose_30_tmp_0);

	float* batch_norm_157_tmp_2 = (float*)calloc(7680, sizeof(float));
	autox_conv2d(batch_norm_156_tmp_2, (float*)((int8_t*)weights) + 384680, (float*)((int8_t*)weights) + 384840, batch_norm_157_tmp_2, batch_norm_156_tmp_2_dim, conv2d_157_w_0_dim, batch_norm_157_tmp_2_dim, 1, 0, 1, 1, 0);
	free(batch_norm_156_tmp_2);

	float* tmp_42 = (float*)calloc(7680, sizeof(float));
	autox_elementwise_add(tmp_41, batch_norm_157_tmp_2, tmp_42, tmp_41_dim, batch_norm_157_tmp_2_dim, tmp_42_dim, -1, 4, 4, 4);
	free(tmp_41);
	free(batch_norm_157_tmp_2);

	float* relu_90_tmp_0 = (float*)calloc(7680, sizeof(float));
	autox_fusion_elementwise_add_activation(tmp_42, transpose_32_tmp_0, relu_90_tmp_0, tmp_42_dim, reshape2_65_tmp_0_dim, relu_90_tmp_0_dim, -1, 4, 4, 4);
	free(tmp_42);
	free(transpose_32_tmp_0);

	float* batch_norm_158_tmp_2 = (float*)calloc(1920, sizeof(float));
	autox_conv2d(relu_90_tmp_0, (float*)((int8_t*)weights) + 397640, (float*)((int8_t*)weights) + 397800, batch_norm_158_tmp_2, relu_90_tmp_0_dim, conv2d_158_w_0_dim, batch_norm_158_tmp_2_dim, 160, 1, 2, 1, 0);
	//free(relu_90_tmp_0);

	float* split_33_tmp_0 = (float*)calloc(15360, sizeof(float));
	float* split_33_tmp_1 = (float*)calloc(15360, sizeof(float));
	float* p_321[] = { split_33_tmp_0, split_33_tmp_1, };
	uint16_t* p_321_dim[] = { split_33_tmp_0_dim, split_33_tmp_1_dim, };
	autox_split(relu_87_tmp_0, p_321, relu_87_tmp_0_dim, p_321_dim, 1, 4, 2, 4);
	free(relu_87_tmp_0);

	float* relu_92_tmp_0 = (float*)calloc(15360, sizeof(float));
	autox_conv2d(split_33_tmp_1, (float*)((int8_t*)weights) + 399240, (float*)((int8_t*)weights) + 399260, relu_92_tmp_0, split_33_tmp_1_dim, conv2d_160_w_0_dim, relu_92_tmp_0_dim, 1, 0, 1, 1, 1);
	free(split_33_tmp_1);

	float* batch_norm_161_tmp_2 = (float*)calloc(15360, sizeof(float));
	autox_conv2d(relu_92_tmp_0, (float*)((int8_t*)weights) + 399660, (float*)((int8_t*)weights) + 399680, batch_norm_161_tmp_2, relu_92_tmp_0_dim, conv2d_161_w_0_dim, batch_norm_161_tmp_2_dim, 20, 1, 1, 1, 0);
	free(relu_92_tmp_0);

	float* relu_93_tmp_0 = (float*)calloc(15360, sizeof(float));
	autox_conv2d(batch_norm_161_tmp_2, (float*)((int8_t*)weights) + 399860, (float*)((int8_t*)weights) + 399880, relu_93_tmp_0, batch_norm_161_tmp_2_dim, conv2d_162_w_0_dim, relu_93_tmp_0_dim, 1, 0, 1, 1, 1);
	free(batch_norm_161_tmp_2);

	float* p_325[] = { split_33_tmp_0, relu_93_tmp_0, };
	uint16_t* p_325_dim[] = { split_33_tmp_0_dim, relu_93_tmp_0_dim, };
	float* concat_33_tmp_0 = (float*)calloc(30720, sizeof(float));
	autox_concat(p_325, concat_33_tmp_0, p_325_dim, concat_33_tmp_0_dim, 1, 2, 4);
	free(split_33_tmp_0);
	free(relu_93_tmp_0);

	float* transpose_33_tmp_0 = (float*)calloc(30720, sizeof(float));
	uint16_t axis_326[] = { 0, 2, 1, 3, 4 };
	autox_transpose(concat_33_tmp_0, transpose_33_tmp_0, reshape2_66_tmp_0_dim, transpose_33_tmp_0_dim, axis_326, 5);
	free(concat_33_tmp_0);

	float* split_34_tmp_0 = (float*)calloc(15360, sizeof(float));
	float* split_34_tmp_1 = (float*)calloc(15360, sizeof(float));
	float* p_327[] = { split_34_tmp_0, split_34_tmp_1, };
	uint16_t* p_327_dim[] = { split_34_tmp_0_dim, split_34_tmp_1_dim, };
	autox_split(transpose_33_tmp_0, p_327, reshape2_67_tmp_0_dim, p_327_dim, 1, 4, 2, 4);
	free(transpose_33_tmp_0);

	float* relu_94_tmp_0 = (float*)calloc(15360, sizeof(float));
	autox_conv2d(split_34_tmp_1, (float*)((int8_t*)weights) + 400280, (float*)((int8_t*)weights) + 400300, relu_94_tmp_0, split_34_tmp_1_dim, conv2d_163_w_0_dim, relu_94_tmp_0_dim, 1, 0, 1, 1, 1);
	free(split_34_tmp_1);

	float* batch_norm_164_tmp_2 = (float*)calloc(15360, sizeof(float));
	autox_conv2d(relu_94_tmp_0, (float*)((int8_t*)weights) + 400700, (float*)((int8_t*)weights) + 400720, batch_norm_164_tmp_2, relu_94_tmp_0_dim, conv2d_164_w_0_dim, batch_norm_164_tmp_2_dim, 20, 1, 1, 1, 0);
	free(relu_94_tmp_0);

	float* relu_95_tmp_0 = (float*)calloc(15360, sizeof(float));
	autox_conv2d(batch_norm_164_tmp_2, (float*)((int8_t*)weights) + 400900, (float*)((int8_t*)weights) + 400920, relu_95_tmp_0, batch_norm_164_tmp_2_dim, conv2d_165_w_0_dim, relu_95_tmp_0_dim, 1, 0, 1, 1, 1);
	free(batch_norm_164_tmp_2);

	float* p_331[] = { split_34_tmp_0, relu_95_tmp_0, };
	uint16_t* p_331_dim[] = { split_34_tmp_0_dim, relu_95_tmp_0_dim, };
	float* concat_34_tmp_0 = (float*)calloc(30720, sizeof(float));
	autox_concat(p_331, concat_34_tmp_0, p_331_dim, concat_34_tmp_0_dim, 1, 2, 4);
	free(split_34_tmp_0);
	free(relu_95_tmp_0);

	float* transpose_34_tmp_0 = (float*)calloc(30720, sizeof(float));
	uint16_t axis_332[] = { 0, 2, 1, 3, 4 };
	autox_transpose(concat_34_tmp_0, transpose_34_tmp_0, reshape2_68_tmp_0_dim, transpose_34_tmp_0_dim, axis_332, 5);
	free(concat_34_tmp_0);

	float* relu_88_tmp_0 = (float*)calloc(15360, sizeof(float));
	autox_fusion_elementwise_add_activation(tmp_39, nearest_interp_v2_13_tmp_0, relu_88_tmp_0, tmp_39_dim, nearest_interp_v2_13_tmp_0_dim, relu_88_tmp_0_dim, -1, 4, 4, 4);
	free(tmp_39);
	free(nearest_interp_v2_13_tmp_0);

	float* split_35_tmp_0 = (float*)calloc(7680, sizeof(float));
	float* split_35_tmp_1 = (float*)calloc(7680, sizeof(float));
	float* p_334[] = { split_35_tmp_0, split_35_tmp_1, };
	uint16_t* p_334_dim[] = { split_35_tmp_0_dim, split_35_tmp_1_dim, };
	autox_split(relu_88_tmp_0, p_334, relu_88_tmp_0_dim, p_334_dim, 1, 4, 2, 4);
	free(relu_88_tmp_0);

	float* relu_96_tmp_0 = (float*)calloc(7680, sizeof(float));
	autox_conv2d(split_35_tmp_1, (float*)((int8_t*)weights) + 401320, (float*)((int8_t*)weights) + 401360, relu_96_tmp_0, split_35_tmp_1_dim, conv2d_166_w_0_dim, relu_96_tmp_0_dim, 1, 0, 1, 1, 1);
	free(split_35_tmp_1);

	float* batch_norm_167_tmp_2 = (float*)calloc(7680, sizeof(float));
	autox_conv2d(relu_96_tmp_0, (float*)((int8_t*)weights) + 402960, (float*)((int8_t*)weights) + 403000, batch_norm_167_tmp_2, relu_96_tmp_0_dim, conv2d_167_w_0_dim, batch_norm_167_tmp_2_dim, 40, 1, 1, 1, 0);
	free(relu_96_tmp_0);

	float* relu_97_tmp_0 = (float*)calloc(7680, sizeof(float));
	autox_conv2d(batch_norm_167_tmp_2, (float*)((int8_t*)weights) + 403360, (float*)((int8_t*)weights) + 403400, relu_97_tmp_0, batch_norm_167_tmp_2_dim, conv2d_168_w_0_dim, relu_97_tmp_0_dim, 1, 0, 1, 1, 1);
	free(batch_norm_167_tmp_2);

	float* p_338[] = { split_35_tmp_0, relu_97_tmp_0, };
	uint16_t* p_338_dim[] = { split_35_tmp_0_dim, relu_97_tmp_0_dim, };
	float* concat_35_tmp_0 = (float*)calloc(15360, sizeof(float));
	autox_concat(p_338, concat_35_tmp_0, p_338_dim, concat_35_tmp_0_dim, 1, 2, 4);
	free(split_35_tmp_0);
	free(relu_97_tmp_0);

	float* transpose_35_tmp_0 = (float*)calloc(15360, sizeof(float));
	uint16_t axis_339[] = { 0, 2, 1, 3, 4 };
	autox_transpose(concat_35_tmp_0, transpose_35_tmp_0, reshape2_70_tmp_0_dim, transpose_35_tmp_0_dim, axis_339, 5);
	free(concat_35_tmp_0);

	float* split_36_tmp_0 = (float*)calloc(7680, sizeof(float));
	float* split_36_tmp_1 = (float*)calloc(7680, sizeof(float));
	float* p_340[] = { split_36_tmp_0, split_36_tmp_1, };
	uint16_t* p_340_dim[] = { split_36_tmp_0_dim, split_36_tmp_1_dim, };
	autox_split(transpose_35_tmp_0, p_340, reshape2_71_tmp_0_dim, p_340_dim, 1, 4, 2, 4);
	free(transpose_35_tmp_0);

	float* relu_98_tmp_0 = (float*)calloc(7680, sizeof(float));
	autox_conv2d(split_36_tmp_1, (float*)((int8_t*)weights) + 405000, (float*)((int8_t*)weights) + 405040, relu_98_tmp_0, split_36_tmp_1_dim, conv2d_169_w_0_dim, relu_98_tmp_0_dim, 1, 0, 1, 1, 1);
	free(split_36_tmp_1);

	float* batch_norm_170_tmp_2 = (float*)calloc(7680, sizeof(float));
	autox_conv2d(relu_98_tmp_0, (float*)((int8_t*)weights) + 406640, (float*)((int8_t*)weights) + 406680, batch_norm_170_tmp_2, relu_98_tmp_0_dim, conv2d_170_w_0_dim, batch_norm_170_tmp_2_dim, 40, 1, 1, 1, 0);
	free(relu_98_tmp_0);

	float* relu_99_tmp_0 = (float*)calloc(7680, sizeof(float));
	autox_conv2d(batch_norm_170_tmp_2, (float*)((int8_t*)weights) + 407040, (float*)((int8_t*)weights) + 407080, relu_99_tmp_0, batch_norm_170_tmp_2_dim, conv2d_171_w_0_dim, relu_99_tmp_0_dim, 1, 0, 1, 1, 1);
	free(batch_norm_170_tmp_2);

	float* p_344[] = { split_36_tmp_0, relu_99_tmp_0, };
	uint16_t* p_344_dim[] = { split_36_tmp_0_dim, relu_99_tmp_0_dim, };
	float* concat_36_tmp_0 = (float*)calloc(15360, sizeof(float));
	autox_concat(p_344, concat_36_tmp_0, p_344_dim, concat_36_tmp_0_dim, 1, 2, 4);
	free(split_36_tmp_0);
	free(relu_99_tmp_0);

	float* transpose_36_tmp_0 = (float*)calloc(15360, sizeof(float));
	uint16_t axis_345[] = { 0, 2, 1, 3, 4 };
	autox_transpose(concat_36_tmp_0, transpose_36_tmp_0, reshape2_72_tmp_0_dim, transpose_36_tmp_0_dim, axis_345, 5);
	free(concat_36_tmp_0);

	float* split_37_tmp_0 = (float*)calloc(3840, sizeof(float));
	float* split_37_tmp_1 = (float*)calloc(3840, sizeof(float));
	float* p_346[] = { split_37_tmp_0, split_37_tmp_1, };
	uint16_t* p_346_dim[] = { split_37_tmp_0_dim, split_37_tmp_1_dim, };
	autox_split(relu_90_tmp_0, p_346, relu_90_tmp_0_dim, p_346_dim, 1, 4, 2, 4);
	free(relu_90_tmp_0);

	float* relu_100_tmp_0 = (float*)calloc(3840, sizeof(float));
	autox_conv2d(split_37_tmp_1, (float*)((int8_t*)weights) + 408680, (float*)((int8_t*)weights) + 408760, relu_100_tmp_0, split_37_tmp_1_dim, conv2d_172_w_0_dim, relu_100_tmp_0_dim, 1, 0, 1, 1, 1);
	free(split_37_tmp_1);

	float* batch_norm_173_tmp_2 = (float*)calloc(3840, sizeof(float));
	autox_conv2d(relu_100_tmp_0, (float*)((int8_t*)weights) + 415160, (float*)((int8_t*)weights) + 415240, batch_norm_173_tmp_2, relu_100_tmp_0_dim, conv2d_173_w_0_dim, batch_norm_173_tmp_2_dim, 80, 1, 1, 1, 0);
	free(relu_100_tmp_0);

	float* relu_101_tmp_0 = (float*)calloc(3840, sizeof(float));
	autox_conv2d(batch_norm_173_tmp_2, (float*)((int8_t*)weights) + 415960, (float*)((int8_t*)weights) + 416040, relu_101_tmp_0, batch_norm_173_tmp_2_dim, conv2d_174_w_0_dim, relu_101_tmp_0_dim, 1, 0, 1, 1, 1);
	free(batch_norm_173_tmp_2);

	float* p_350[] = { split_37_tmp_0, relu_101_tmp_0, };
	uint16_t* p_350_dim[] = { split_37_tmp_0_dim, relu_101_tmp_0_dim, };
	float* concat_37_tmp_0 = (float*)calloc(7680, sizeof(float));
	autox_concat(p_350, concat_37_tmp_0, p_350_dim, concat_37_tmp_0_dim, 1, 2, 4);
	free(split_37_tmp_0);
	free(relu_101_tmp_0);

	float* transpose_37_tmp_0 = (float*)calloc(7680, sizeof(float));
	uint16_t axis_351[] = { 0, 2, 1, 3, 4 };
	autox_transpose(concat_37_tmp_0, transpose_37_tmp_0, reshape2_74_tmp_0_dim, transpose_37_tmp_0_dim, axis_351, 5);
	free(concat_37_tmp_0);

	float* split_38_tmp_0 = (float*)calloc(3840, sizeof(float));
	float* split_38_tmp_1 = (float*)calloc(3840, sizeof(float));
	float* p_352[] = { split_38_tmp_0, split_38_tmp_1, };
	uint16_t* p_352_dim[] = { split_38_tmp_0_dim, split_38_tmp_1_dim, };
	autox_split(transpose_37_tmp_0, p_352, reshape2_75_tmp_0_dim, p_352_dim, 1, 4, 2, 4);
	free(transpose_37_tmp_0);

	float* relu_102_tmp_0 = (float*)calloc(3840, sizeof(float));
	autox_conv2d(split_38_tmp_1, (float*)((int8_t*)weights) + 422440, (float*)((int8_t*)weights) + 422520, relu_102_tmp_0, split_38_tmp_1_dim, conv2d_175_w_0_dim, relu_102_tmp_0_dim, 1, 0, 1, 1, 1);
	free(split_38_tmp_1);

	float* batch_norm_176_tmp_2 = (float*)calloc(3840, sizeof(float));
	autox_conv2d(relu_102_tmp_0, (float*)((int8_t*)weights) + 428920, (float*)((int8_t*)weights) + 429000, batch_norm_176_tmp_2, relu_102_tmp_0_dim, conv2d_176_w_0_dim, batch_norm_176_tmp_2_dim, 80, 1, 1, 1, 0);
	free(relu_102_tmp_0);

	float* relu_103_tmp_0 = (float*)calloc(3840, sizeof(float));
	autox_conv2d(batch_norm_176_tmp_2, (float*)((int8_t*)weights) + 429720, (float*)((int8_t*)weights) + 429800, relu_103_tmp_0, batch_norm_176_tmp_2_dim, conv2d_177_w_0_dim, relu_103_tmp_0_dim, 1, 0, 1, 1, 1);
	free(batch_norm_176_tmp_2);

	float* p_356[] = { split_38_tmp_0, relu_103_tmp_0, };
	uint16_t* p_356_dim[] = { split_38_tmp_0_dim, relu_103_tmp_0_dim, };
	float* concat_38_tmp_0 = (float*)calloc(7680, sizeof(float));
	autox_concat(p_356, concat_38_tmp_0, p_356_dim, concat_38_tmp_0_dim, 1, 2, 4);
	free(split_38_tmp_0);
	free(relu_103_tmp_0);

	float* transpose_38_tmp_0 = (float*)calloc(7680, sizeof(float));
	uint16_t axis_357[] = { 0, 2, 1, 3, 4 };
	autox_transpose(concat_38_tmp_0, transpose_38_tmp_0, reshape2_76_tmp_0_dim, transpose_38_tmp_0_dim, axis_357, 5);
	free(concat_38_tmp_0);

	float* relu_91_tmp_0 = (float*)calloc(3840, sizeof(float));
	autox_conv2d(batch_norm_158_tmp_2, (float*)((int8_t*)weights) + 436200, (float*)((int8_t*)weights) + 436520, relu_91_tmp_0, batch_norm_158_tmp_2_dim, conv2d_159_w_0_dim, relu_91_tmp_0_dim, 1, 0, 1, 1, 1);
	free(batch_norm_158_tmp_2);

	float* split_39_tmp_0 = (float*)calloc(1920, sizeof(float));
	float* split_39_tmp_1 = (float*)calloc(1920, sizeof(float));
	float* p_359[] = { split_39_tmp_0, split_39_tmp_1, };
	uint16_t* p_359_dim[] = { split_39_tmp_0_dim, split_39_tmp_1_dim, };
	autox_split(relu_91_tmp_0, p_359, relu_91_tmp_0_dim, p_359_dim, 1, 4, 2, 4);
	free(relu_91_tmp_0);

	float* relu_104_tmp_0 = (float*)calloc(1920, sizeof(float));
	autox_conv2d(split_39_tmp_1, (float*)((int8_t*)weights) + 487720, (float*)((int8_t*)weights) + 487880, relu_104_tmp_0, split_39_tmp_1_dim, conv2d_178_w_0_dim, relu_104_tmp_0_dim, 1, 0, 1, 1, 1);
	free(split_39_tmp_1);

	float* batch_norm_179_tmp_2 = (float*)calloc(1920, sizeof(float));
	autox_conv2d(relu_104_tmp_0, (float*)((int8_t*)weights) + 513480, (float*)((int8_t*)weights) + 513640, batch_norm_179_tmp_2, relu_104_tmp_0_dim, conv2d_179_w_0_dim, batch_norm_179_tmp_2_dim, 160, 1, 1, 1, 0);
	free(relu_104_tmp_0);

	float* relu_105_tmp_0 = (float*)calloc(1920, sizeof(float));
	autox_conv2d(batch_norm_179_tmp_2, (float*)((int8_t*)weights) + 515080, (float*)((int8_t*)weights) + 515240, relu_105_tmp_0, batch_norm_179_tmp_2_dim, conv2d_180_w_0_dim, relu_105_tmp_0_dim, 1, 0, 1, 1, 1);
	free(batch_norm_179_tmp_2);

	float* p_363[] = { split_39_tmp_0, relu_105_tmp_0, };
	uint16_t* p_363_dim[] = { split_39_tmp_0_dim, relu_105_tmp_0_dim, };
	float* concat_39_tmp_0 = (float*)calloc(3840, sizeof(float));
	autox_concat(p_363, concat_39_tmp_0, p_363_dim, concat_39_tmp_0_dim, 1, 2, 4);
	free(split_39_tmp_0);
	free(relu_105_tmp_0);

	float* transpose_39_tmp_0 = (float*)calloc(3840, sizeof(float));
	uint16_t axis_364[] = { 0, 2, 1, 3, 4 };
	autox_transpose(concat_39_tmp_0, transpose_39_tmp_0, reshape2_78_tmp_0_dim, transpose_39_tmp_0_dim, axis_364, 5);
	free(concat_39_tmp_0);

	float* split_40_tmp_0 = (float*)calloc(1920, sizeof(float));
	float* split_40_tmp_1 = (float*)calloc(1920, sizeof(float));
	float* p_365[] = { split_40_tmp_0, split_40_tmp_1, };
	uint16_t* p_365_dim[] = { split_40_tmp_0_dim, split_40_tmp_1_dim, };
	autox_split(transpose_39_tmp_0, p_365, reshape2_79_tmp_0_dim, p_365_dim, 1, 4, 2, 4);
	free(transpose_39_tmp_0);

	float* relu_106_tmp_0 = (float*)calloc(1920, sizeof(float));
	autox_conv2d(split_40_tmp_1, (float*)((int8_t*)weights) + 540840, (float*)((int8_t*)weights) + 541000, relu_106_tmp_0, split_40_tmp_1_dim, conv2d_181_w_0_dim, relu_106_tmp_0_dim, 1, 0, 1, 1, 1);
	free(split_40_tmp_1);

	float* batch_norm_182_tmp_2 = (float*)calloc(1920, sizeof(float));
	autox_conv2d(relu_106_tmp_0, (float*)((int8_t*)weights) + 566600, (float*)((int8_t*)weights) + 566760, batch_norm_182_tmp_2, relu_106_tmp_0_dim, conv2d_182_w_0_dim, batch_norm_182_tmp_2_dim, 160, 1, 1, 1, 0);
	free(relu_106_tmp_0);

	float* relu_107_tmp_0 = (float*)calloc(1920, sizeof(float));
	autox_conv2d(batch_norm_182_tmp_2, (float*)((int8_t*)weights) + 568200, (float*)((int8_t*)weights) + 568360, relu_107_tmp_0, batch_norm_182_tmp_2_dim, conv2d_183_w_0_dim, relu_107_tmp_0_dim, 1, 0, 1, 1, 1);
	free(batch_norm_182_tmp_2);

	float* p_369[] = { split_40_tmp_0, relu_107_tmp_0, };
	uint16_t* p_369_dim[] = { split_40_tmp_0_dim, relu_107_tmp_0_dim, };
	float* concat_40_tmp_0 = (float*)calloc(3840, sizeof(float));
	autox_concat(p_369, concat_40_tmp_0, p_369_dim, concat_40_tmp_0_dim, 1, 2, 4);
	free(split_40_tmp_0);
	free(relu_107_tmp_0);

	float* transpose_40_tmp_0 = (float*)calloc(3840, sizeof(float));
	uint16_t axis_370[] = { 0, 2, 1, 3, 4 };
	autox_transpose(concat_40_tmp_0, transpose_40_tmp_0, reshape2_80_tmp_0_dim, transpose_40_tmp_0_dim, axis_370, 5);
	free(concat_40_tmp_0);

	float* tmp_44 = (float*)calloc(30720, sizeof(float));
	autox_elementwise_add(transpose_34_tmp_0, transpose_34_tmp_0, tmp_44, reshape2_69_tmp_0_dim, reshape2_69_tmp_0_dim, tmp_44_dim, -1, 4, 4, 4);
	free(transpose_34_tmp_0);
	//free(transpose_34_tmp_0);

	float* batch_norm_184_tmp_2 = (float*)calloc(7680, sizeof(float));
	autox_conv2d(transpose_36_tmp_0, (float*)((int8_t*)weights) + 593960, (float*)((int8_t*)weights) + 594000, batch_norm_184_tmp_2, reshape2_73_tmp_0_dim, conv2d_184_w_0_dim, batch_norm_184_tmp_2_dim, 1, 0, 1, 1, 0);
	// free(transpose_36_tmp_0);

	float* nearest_interp_v2_14_tmp_0 = (float*)calloc(30720, sizeof(float));
	autox_nearest_interp(batch_norm_184_tmp_2, nearest_interp_v2_14_tmp_0, batch_norm_184_tmp_2_dim, nearest_interp_v2_14_tmp_0_dim, 2, 0);
	free(batch_norm_184_tmp_2);

	float* tmp_45 = (float*)calloc(30720, sizeof(float));
	autox_elementwise_add(tmp_44, nearest_interp_v2_14_tmp_0, tmp_45, tmp_44_dim, nearest_interp_v2_14_tmp_0_dim, tmp_45_dim, -1, 4, 4, 4);
	free(tmp_44);
	free(nearest_interp_v2_14_tmp_0);

	float* batch_norm_185_tmp_2 = (float*)calloc(1920, sizeof(float));
	autox_conv2d(transpose_38_tmp_0, (float*)((int8_t*)weights) + 597200, (float*)((int8_t*)weights) + 597240, batch_norm_185_tmp_2, reshape2_77_tmp_0_dim, conv2d_185_w_0_dim, batch_norm_185_tmp_2_dim, 1, 0, 1, 1, 0);
	//free(transpose_38_tmp_0);

	float* nearest_interp_v2_15_tmp_0 = (float*)calloc(30720, sizeof(float));
	autox_nearest_interp(batch_norm_185_tmp_2, nearest_interp_v2_15_tmp_0, batch_norm_185_tmp_2_dim, nearest_interp_v2_15_tmp_0_dim, 4, 0);
	free(batch_norm_185_tmp_2);

	float* tmp_46 = (float*)calloc(30720, sizeof(float));
	autox_elementwise_add(tmp_45, nearest_interp_v2_15_tmp_0, tmp_46, tmp_45_dim, nearest_interp_v2_15_tmp_0_dim, tmp_46_dim, -1, 4, 4, 4);
	free(tmp_45);
	free(nearest_interp_v2_15_tmp_0);

	float* batch_norm_186_tmp_2 = (float*)calloc(480, sizeof(float));
	autox_conv2d(transpose_40_tmp_0, (float*)((int8_t*)weights) + 603640, (float*)((int8_t*)weights) + 603680, batch_norm_186_tmp_2, reshape2_81_tmp_0_dim, conv2d_186_w_0_dim, batch_norm_186_tmp_2_dim, 1, 0, 1, 1, 0);
	//free(transpose_40_tmp_0);

	float* nearest_interp_v2_16_tmp_0 = (float*)calloc(30720, sizeof(float));
	autox_nearest_interp(batch_norm_186_tmp_2, nearest_interp_v2_16_tmp_0, batch_norm_186_tmp_2_dim, nearest_interp_v2_16_tmp_0_dim, 8, 0);
	free(batch_norm_186_tmp_2);

	float* tmp_47 = (float*)calloc(30720, sizeof(float));
	autox_elementwise_add(tmp_46, nearest_interp_v2_16_tmp_0, tmp_47, tmp_46_dim, nearest_interp_v2_16_tmp_0_dim, tmp_47_dim, -1, 4, 4, 4);
	free(tmp_46);
	free(nearest_interp_v2_16_tmp_0);

	float*	relu_108_tmp_0 = (float*)calloc(30720, sizeof(float));
	autox_relu_noreplace(tmp_47, relu_108_tmp_0, tmp_47_dim, 4);

	float* batch_norm_187_tmp_2 = (float*)calloc(7680, sizeof(float));
	autox_conv2d(tmp_47, (float*)((int8_t*)weights) + 616480, (float*)((int8_t*)weights) + 616520, batch_norm_187_tmp_2, tmp_47_dim, conv2d_187_w_0_dim, batch_norm_187_tmp_2_dim, 40, 1, 2, 1, 0);
	//free(tmp_47);

	float* batch_norm_188_tmp_2 = (float*)calloc(15360, sizeof(float));
	autox_conv2d(batch_norm_187_tmp_2, (float*)((int8_t*)weights) + 616880, (float*)((int8_t*)weights) + 616960, batch_norm_188_tmp_2, batch_norm_187_tmp_2_dim, conv2d_188_w_0_dim, batch_norm_188_tmp_2_dim, 1, 0, 1, 1, 0);
	free(batch_norm_187_tmp_2);

	float* tmp_48 = (float*)calloc(15360, sizeof(float));
	autox_elementwise_add(batch_norm_188_tmp_2, batch_norm_188_tmp_2, tmp_48, batch_norm_188_tmp_2_dim, batch_norm_188_tmp_2_dim, tmp_48_dim, -1, 4, 4, 4);
	free(batch_norm_188_tmp_2);
	// free(batch_norm_188_tmp_2);

	float* tmp_49 = (float*)calloc(15360, sizeof(float));
	autox_elementwise_add(tmp_48, transpose_36_tmp_0, tmp_49, tmp_48_dim, reshape2_73_tmp_0_dim, tmp_49_dim, -1, 4, 4, 4);
	free(tmp_48);
	//free(transpose_36_tmp_0);

	float* batch_norm_189_tmp_2 = (float*)calloc(3840, sizeof(float));
	autox_conv2d(transpose_38_tmp_0, (float*)((int8_t*)weights) + 620160, (float*)((int8_t*)weights) + 620240, batch_norm_189_tmp_2, reshape2_77_tmp_0_dim, conv2d_189_w_0_dim, batch_norm_189_tmp_2_dim, 1, 0, 1, 1, 0);
	//free(transpose_38_tmp_0);

	float* nearest_interp_v2_17_tmp_0 = (float*)calloc(15360, sizeof(float));
	autox_nearest_interp(batch_norm_189_tmp_2, nearest_interp_v2_17_tmp_0, batch_norm_189_tmp_2_dim, nearest_interp_v2_17_tmp_0_dim, 2, 0);
	free(batch_norm_189_tmp_2);

	float* tmp_50 = (float*)calloc(15360, sizeof(float));
	autox_elementwise_add(tmp_49, nearest_interp_v2_17_tmp_0, tmp_50, tmp_49_dim, nearest_interp_v2_17_tmp_0_dim, tmp_50_dim, -1, 4, 4, 4);
	free(tmp_49);
	free(nearest_interp_v2_17_tmp_0);

	float* batch_norm_190_tmp_2 = (float*)calloc(960, sizeof(float));
	autox_conv2d(transpose_40_tmp_0, (float*)((int8_t*)weights) + 633040, (float*)((int8_t*)weights) + 633120, batch_norm_190_tmp_2, reshape2_81_tmp_0_dim, conv2d_190_w_0_dim, batch_norm_190_tmp_2_dim, 1, 0, 1, 1, 0);
	//free(transpose_40_tmp_0);

	float* nearest_interp_v2_18_tmp_0 = (float*)calloc(15360, sizeof(float));
	autox_nearest_interp(batch_norm_190_tmp_2, nearest_interp_v2_18_tmp_0, batch_norm_190_tmp_2_dim, nearest_interp_v2_18_tmp_0_dim, 4, 0);
	free(batch_norm_190_tmp_2);

	float* batch_norm_191_tmp_2 = (float*)calloc(7680, sizeof(float));
	autox_conv2d(tmp_47, (float*)((int8_t*)weights) + 658720, (float*)((int8_t*)weights) + 658760, batch_norm_191_tmp_2, tmp_47_dim, conv2d_191_w_0_dim, batch_norm_191_tmp_2_dim, 40, 1, 2, 1, 0);
	//free(tmp_47);

	float* relu_110_tmp_0 = (float*)calloc(7680, sizeof(float));
	autox_conv2d(batch_norm_191_tmp_2, (float*)((int8_t*)weights) + 659120, (float*)((int8_t*)weights) + 659160, relu_110_tmp_0, batch_norm_191_tmp_2_dim, conv2d_192_w_0_dim, relu_110_tmp_0_dim, 1, 0, 1, 1, 1);
	free(batch_norm_191_tmp_2);

	float* batch_norm_193_tmp_2 = (float*)calloc(1920, sizeof(float));
	autox_conv2d(relu_110_tmp_0, (float*)((int8_t*)weights) + 660760, (float*)((int8_t*)weights) + 660800, batch_norm_193_tmp_2, relu_110_tmp_0_dim, conv2d_193_w_0_dim, batch_norm_193_tmp_2_dim, 40, 1, 2, 1, 0);
	free(relu_110_tmp_0);

	float* batch_norm_194_tmp_2 = (float*)calloc(7680, sizeof(float));
	autox_conv2d(batch_norm_193_tmp_2, (float*)((int8_t*)weights) + 661160, (float*)((int8_t*)weights) + 661320, batch_norm_194_tmp_2, batch_norm_193_tmp_2_dim, conv2d_194_w_0_dim, batch_norm_194_tmp_2_dim, 1, 0, 1, 1, 0);
	free(batch_norm_193_tmp_2);

	float* tmp_52 = (float*)calloc(7680, sizeof(float));
	autox_elementwise_add(batch_norm_194_tmp_2, batch_norm_194_tmp_2, tmp_52, batch_norm_194_tmp_2_dim, batch_norm_194_tmp_2_dim, tmp_52_dim, -1, 4, 4, 4);
	free(batch_norm_194_tmp_2);
	// free(batch_norm_194_tmp_2);

	float* batch_norm_195_tmp_2 = (float*)calloc(3840, sizeof(float));
	autox_conv2d(transpose_36_tmp_0, (float*)((int8_t*)weights) + 667720, (float*)((int8_t*)weights) + 667800, batch_norm_195_tmp_2, reshape2_73_tmp_0_dim, conv2d_195_w_0_dim, batch_norm_195_tmp_2_dim, 80, 1, 2, 1, 0);
	// free(transpose_36_tmp_0);

	float* batch_norm_196_tmp_2 = (float*)calloc(7680, sizeof(float));
	autox_conv2d(batch_norm_195_tmp_2, (float*)((int8_t*)weights) + 668520, (float*)((int8_t*)weights) + 668680, batch_norm_196_tmp_2, batch_norm_195_tmp_2_dim, conv2d_196_w_0_dim, batch_norm_196_tmp_2_dim, 1, 0, 1, 1, 0);
	free(batch_norm_195_tmp_2);

	float* tmp_53 = (float*)calloc(7680, sizeof(float));
	autox_elementwise_add(tmp_52, batch_norm_196_tmp_2, tmp_53, tmp_52_dim, batch_norm_196_tmp_2_dim, tmp_53_dim, -1, 4, 4, 4);
	free(tmp_52);
	free(batch_norm_196_tmp_2);

	float* tmp_54 = (float*)calloc(7680, sizeof(float));
	autox_elementwise_add(tmp_53, transpose_38_tmp_0, tmp_54, tmp_53_dim, reshape2_77_tmp_0_dim, tmp_54_dim, -1, 4, 4, 4);
	free(tmp_53);
	//free(transpose_38_tmp_0);

	float* batch_norm_197_tmp_2 = (float*)calloc(1920, sizeof(float));
	autox_conv2d(transpose_40_tmp_0, (float*)((int8_t*)weights) + 681480, (float*)((int8_t*)weights) + 681640, batch_norm_197_tmp_2, reshape2_81_tmp_0_dim, conv2d_197_w_0_dim, batch_norm_197_tmp_2_dim, 1, 0, 1, 1, 0);
	//free(transpose_40_tmp_0);

	float* nearest_interp_v2_19_tmp_0 = (float*)calloc(7680, sizeof(float));
	autox_nearest_interp(batch_norm_197_tmp_2, nearest_interp_v2_19_tmp_0, batch_norm_197_tmp_2_dim, nearest_interp_v2_19_tmp_0_dim, 2, 0);
	free(batch_norm_197_tmp_2);

	float* batch_norm_198_tmp_2 = (float*)calloc(7680, sizeof(float));
	autox_conv2d(tmp_47, (float*)((int8_t*)weights) + 732840, (float*)((int8_t*)weights) + 732880, batch_norm_198_tmp_2, tmp_47_dim, conv2d_198_w_0_dim, batch_norm_198_tmp_2_dim, 40, 1, 2, 1, 0);
	//free(tmp_47);

	float* relu_112_tmp_0 = (float*)calloc(7680, sizeof(float));
	autox_conv2d(batch_norm_198_tmp_2, (float*)((int8_t*)weights) + 733240, (float*)((int8_t*)weights) + 733280, relu_112_tmp_0, batch_norm_198_tmp_2_dim, conv2d_199_w_0_dim, relu_112_tmp_0_dim, 1, 0, 1, 1, 1);
	free(batch_norm_198_tmp_2);

	float* batch_norm_200_tmp_2 = (float*)calloc(1920, sizeof(float));
	autox_conv2d(relu_112_tmp_0, (float*)((int8_t*)weights) + 734880, (float*)((int8_t*)weights) + 734920, batch_norm_200_tmp_2, relu_112_tmp_0_dim, conv2d_200_w_0_dim, batch_norm_200_tmp_2_dim, 40, 1, 2, 1, 0);
	free(relu_112_tmp_0);

	float* relu_113_tmp_0 = (float*)calloc(1920, sizeof(float));
	autox_conv2d(batch_norm_200_tmp_2, (float*)((int8_t*)weights) + 735280, (float*)((int8_t*)weights) + 735320, relu_113_tmp_0, batch_norm_200_tmp_2_dim, conv2d_201_w_0_dim, relu_113_tmp_0_dim, 1, 0, 1, 1, 1);
	free(batch_norm_200_tmp_2);

	float* batch_norm_202_tmp_2 = (float*)calloc(480, sizeof(float));
	autox_conv2d(relu_113_tmp_0, (float*)((int8_t*)weights) + 736920, (float*)((int8_t*)weights) + 736960, batch_norm_202_tmp_2, relu_113_tmp_0_dim, conv2d_202_w_0_dim, batch_norm_202_tmp_2_dim, 40, 1, 2, 1, 0);
	free(relu_113_tmp_0);

	float* batch_norm_203_tmp_2 = (float*)calloc(3840, sizeof(float));
	autox_conv2d(batch_norm_202_tmp_2, (float*)((int8_t*)weights) + 737320, (float*)((int8_t*)weights) + 737640, batch_norm_203_tmp_2, batch_norm_202_tmp_2_dim, conv2d_203_w_0_dim, batch_norm_203_tmp_2_dim, 1, 0, 1, 1, 0);
	free(batch_norm_202_tmp_2);

	float* tmp_56 = (float*)calloc(3840, sizeof(float));
	autox_elementwise_add(batch_norm_203_tmp_2, batch_norm_203_tmp_2, tmp_56, batch_norm_203_tmp_2_dim, batch_norm_203_tmp_2_dim, tmp_56_dim, -1, 4, 4, 4);
	free(batch_norm_203_tmp_2);
	// free(batch_norm_203_tmp_2);

	float* batch_norm_204_tmp_2 = (float*)calloc(3840, sizeof(float));
	autox_conv2d(transpose_36_tmp_0, (float*)((int8_t*)weights) + 750440, (float*)((int8_t*)weights) + 750520, batch_norm_204_tmp_2, reshape2_73_tmp_0_dim, conv2d_204_w_0_dim, batch_norm_204_tmp_2_dim, 80, 1, 2, 1, 0);
	free(transpose_36_tmp_0);

	float* relu_114_tmp_0 = (float*)calloc(3840, sizeof(float));
	autox_conv2d(batch_norm_204_tmp_2, (float*)((int8_t*)weights) + 751240, (float*)((int8_t*)weights) + 751320, relu_114_tmp_0, batch_norm_204_tmp_2_dim, conv2d_205_w_0_dim, relu_114_tmp_0_dim, 1, 0, 1, 1, 1);
	free(batch_norm_204_tmp_2);

	float* batch_norm_206_tmp_2 = (float*)calloc(960, sizeof(float));
	autox_conv2d(relu_114_tmp_0, (float*)((int8_t*)weights) + 757720, (float*)((int8_t*)weights) + 757800, batch_norm_206_tmp_2, relu_114_tmp_0_dim, conv2d_206_w_0_dim, batch_norm_206_tmp_2_dim, 80, 1, 2, 1, 0);
	free(relu_114_tmp_0);

	float* batch_norm_207_tmp_2 = (float*)calloc(3840, sizeof(float));
	autox_conv2d(batch_norm_206_tmp_2, (float*)((int8_t*)weights) + 758520, (float*)((int8_t*)weights) + 758840, batch_norm_207_tmp_2, batch_norm_206_tmp_2_dim, conv2d_207_w_0_dim, batch_norm_207_tmp_2_dim, 1, 0, 1, 1, 0);
	free(batch_norm_206_tmp_2);

	float* tmp_57 = (float*)calloc(3840, sizeof(float));
	autox_elementwise_add(tmp_56, batch_norm_207_tmp_2, tmp_57, tmp_56_dim, batch_norm_207_tmp_2_dim, tmp_57_dim, -1, 4, 4, 4);
	free(tmp_56);
	free(batch_norm_207_tmp_2);

	float* batch_norm_208_tmp_2 = (float*)calloc(1920, sizeof(float));
	autox_conv2d(transpose_38_tmp_0, (float*)((int8_t*)weights) + 784440, (float*)((int8_t*)weights) + 784600, batch_norm_208_tmp_2, reshape2_77_tmp_0_dim, conv2d_208_w_0_dim, batch_norm_208_tmp_2_dim, 160, 1, 2, 1, 0);
	free(transpose_38_tmp_0);

	float* batch_norm_209_tmp_2 = (float*)calloc(3840, sizeof(float));
	autox_conv2d(batch_norm_208_tmp_2, (float*)((int8_t*)weights) + 786040, (float*)((int8_t*)weights) + 786360, batch_norm_209_tmp_2, batch_norm_208_tmp_2_dim, conv2d_209_w_0_dim, batch_norm_209_tmp_2_dim, 1, 0, 1, 1, 0);
	free(batch_norm_208_tmp_2);

	float* tmp_58 = (float*)calloc(3840, sizeof(float));
	autox_elementwise_add(tmp_57, batch_norm_209_tmp_2, tmp_58, tmp_57_dim, batch_norm_209_tmp_2_dim, tmp_58_dim, -1, 4, 4, 4);
	free(tmp_57);
	free(batch_norm_209_tmp_2);

	float* split_41_tmp_0 = (float*)calloc(15360, sizeof(float));
	float* split_41_tmp_1 = (float*)calloc(15360, sizeof(float));
	float* p_417[] = { split_41_tmp_0, split_41_tmp_1, };
	uint16_t* p_417_dim[] = { split_41_tmp_0_dim, split_41_tmp_1_dim, };
	autox_split(relu_108_tmp_0, p_417, relu_108_tmp_0_dim, p_417_dim, 1, 4, 2, 4);
	free(relu_108_tmp_0);

	float* relu_116_tmp_0 = (float*)calloc(15360, sizeof(float));
	autox_conv2d(split_41_tmp_1, (float*)((int8_t*)weights) + 837560, (float*)((int8_t*)weights) + 837580, relu_116_tmp_0, split_41_tmp_1_dim, conv2d_210_w_0_dim, relu_116_tmp_0_dim, 1, 0, 1, 1, 1);
	free(split_41_tmp_1);

	float* batch_norm_211_tmp_2 = (float*)calloc(15360, sizeof(float));
	autox_conv2d(relu_116_tmp_0, (float*)((int8_t*)weights) + 837980, (float*)((int8_t*)weights) + 838000, batch_norm_211_tmp_2, relu_116_tmp_0_dim, conv2d_211_w_0_dim, batch_norm_211_tmp_2_dim, 20, 1, 1, 1, 0);
	free(relu_116_tmp_0);

	float* relu_117_tmp_0 = (float*)calloc(15360, sizeof(float));
	autox_conv2d(batch_norm_211_tmp_2, (float*)((int8_t*)weights) + 838180, (float*)((int8_t*)weights) + 838200, relu_117_tmp_0, batch_norm_211_tmp_2_dim, conv2d_212_w_0_dim, relu_117_tmp_0_dim, 1, 0, 1, 1, 1);
	free(batch_norm_211_tmp_2);

	float* p_421[] = { split_41_tmp_0, relu_117_tmp_0, };
	uint16_t* p_421_dim[] = { split_41_tmp_0_dim, relu_117_tmp_0_dim, };
	float* concat_41_tmp_0 = (float*)calloc(30720, sizeof(float));
	autox_concat(p_421, concat_41_tmp_0, p_421_dim, concat_41_tmp_0_dim, 1, 2, 4);
	free(split_41_tmp_0);
	free(relu_117_tmp_0);

	float* transpose_41_tmp_0 = (float*)calloc(30720, sizeof(float));
	uint16_t axis_422[] = { 0, 2, 1, 3, 4 };
	autox_transpose(concat_41_tmp_0, transpose_41_tmp_0, reshape2_82_tmp_0_dim, transpose_41_tmp_0_dim, axis_422, 5);
	free(concat_41_tmp_0);

	float* split_42_tmp_0 = (float*)calloc(15360, sizeof(float));
	float* split_42_tmp_1 = (float*)calloc(15360, sizeof(float));
	float* p_423[] = { split_42_tmp_0, split_42_tmp_1, };
	uint16_t* p_423_dim[] = { split_42_tmp_0_dim, split_42_tmp_1_dim, };
	autox_split(transpose_41_tmp_0, p_423, reshape2_83_tmp_0_dim, p_423_dim, 1, 4, 2, 4);
	free(transpose_41_tmp_0);

	float* relu_118_tmp_0 = (float*)calloc(15360, sizeof(float));
	autox_conv2d(split_42_tmp_1, (float*)((int8_t*)weights) + 838600, (float*)((int8_t*)weights) + 838620, relu_118_tmp_0, split_42_tmp_1_dim, conv2d_213_w_0_dim, relu_118_tmp_0_dim, 1, 0, 1, 1, 1);
	free(split_42_tmp_1);

	float* batch_norm_214_tmp_2 = (float*)calloc(15360, sizeof(float));
	autox_conv2d(relu_118_tmp_0, (float*)((int8_t*)weights) + 839020, (float*)((int8_t*)weights) + 839040, batch_norm_214_tmp_2, relu_118_tmp_0_dim, conv2d_214_w_0_dim, batch_norm_214_tmp_2_dim, 20, 1, 1, 1, 0);
	free(relu_118_tmp_0);

	float* relu_119_tmp_0 = (float*)calloc(15360, sizeof(float));
	autox_conv2d(batch_norm_214_tmp_2, (float*)((int8_t*)weights) + 839220, (float*)((int8_t*)weights) + 839240, relu_119_tmp_0, batch_norm_214_tmp_2_dim, conv2d_215_w_0_dim, relu_119_tmp_0_dim, 1, 0, 1, 1, 1);
	free(batch_norm_214_tmp_2);

	float* p_427[] = { split_42_tmp_0, relu_119_tmp_0, };
	uint16_t* p_427_dim[] = { split_42_tmp_0_dim, relu_119_tmp_0_dim, };
	float* concat_42_tmp_0 = (float*)calloc(30720, sizeof(float));
	autox_concat(p_427, concat_42_tmp_0, p_427_dim, concat_42_tmp_0_dim, 1, 2, 4);
	free(split_42_tmp_0);
	free(relu_119_tmp_0);

	float* transpose_42_tmp_0 = (float*)calloc(30720, sizeof(float));
	uint16_t axis_428[] = { 0, 2, 1, 3, 4 };
	autox_transpose(concat_42_tmp_0, transpose_42_tmp_0, reshape2_84_tmp_0_dim, transpose_42_tmp_0_dim, axis_428, 5);
	free(concat_42_tmp_0);

	float* relu_109_tmp_0 = (float*)calloc(15360, sizeof(float));
	autox_fusion_elementwise_add_activation(tmp_50, nearest_interp_v2_18_tmp_0, relu_109_tmp_0, tmp_50_dim, nearest_interp_v2_18_tmp_0_dim, relu_109_tmp_0_dim, -1, 4, 4, 4);
	free(tmp_50);
	free(nearest_interp_v2_18_tmp_0);

	float* split_43_tmp_0 = (float*)calloc(7680, sizeof(float));
	float* split_43_tmp_1 = (float*)calloc(7680, sizeof(float));
	float* p_430[] = { split_43_tmp_0, split_43_tmp_1, };
	uint16_t* p_430_dim[] = { split_43_tmp_0_dim, split_43_tmp_1_dim, };
	autox_split(relu_109_tmp_0, p_430, relu_109_tmp_0_dim, p_430_dim, 1, 4, 2, 4);
	free(relu_109_tmp_0);

	float* relu_120_tmp_0 = (float*)calloc(7680, sizeof(float));
	autox_conv2d(split_43_tmp_1, (float*)((int8_t*)weights) + 839640, (float*)((int8_t*)weights) + 839680, relu_120_tmp_0, split_43_tmp_1_dim, conv2d_216_w_0_dim, relu_120_tmp_0_dim, 1, 0, 1, 1, 1);
	free(split_43_tmp_1);

	float* batch_norm_217_tmp_2 = (float*)calloc(7680, sizeof(float));
	autox_conv2d(relu_120_tmp_0, (float*)((int8_t*)weights) + 841280, (float*)((int8_t*)weights) + 841320, batch_norm_217_tmp_2, relu_120_tmp_0_dim, conv2d_217_w_0_dim, batch_norm_217_tmp_2_dim, 40, 1, 1, 1, 0);
	free(relu_120_tmp_0);

	float* relu_121_tmp_0 = (float*)calloc(7680, sizeof(float));
	autox_conv2d(batch_norm_217_tmp_2, (float*)((int8_t*)weights) + 841680, (float*)((int8_t*)weights) + 841720, relu_121_tmp_0, batch_norm_217_tmp_2_dim, conv2d_218_w_0_dim, relu_121_tmp_0_dim, 1, 0, 1, 1, 1);
	free(batch_norm_217_tmp_2);

	float* p_434[] = { split_43_tmp_0, relu_121_tmp_0, };
	uint16_t* p_434_dim[] = { split_43_tmp_0_dim, relu_121_tmp_0_dim, };
	float* concat_43_tmp_0 = (float*)calloc(15360, sizeof(float));
	autox_concat(p_434, concat_43_tmp_0, p_434_dim, concat_43_tmp_0_dim, 1, 2, 4);
	free(split_43_tmp_0);
	free(relu_121_tmp_0);

	float* transpose_43_tmp_0 = (float*)calloc(15360, sizeof(float));
	uint16_t axis_435[] = { 0, 2, 1, 3, 4 };
	autox_transpose(concat_43_tmp_0, transpose_43_tmp_0, reshape2_86_tmp_0_dim, transpose_43_tmp_0_dim, axis_435, 5);
	free(concat_43_tmp_0);

	float* split_44_tmp_0 = (float*)calloc(7680, sizeof(float));
	float* split_44_tmp_1 = (float*)calloc(7680, sizeof(float));
	float* p_436[] = { split_44_tmp_0, split_44_tmp_1, };
	uint16_t* p_436_dim[] = { split_44_tmp_0_dim, split_44_tmp_1_dim, };
	autox_split(transpose_43_tmp_0, p_436, reshape2_87_tmp_0_dim, p_436_dim, 1, 4, 2, 4);
	free(transpose_43_tmp_0);

	float* relu_122_tmp_0 = (float*)calloc(7680, sizeof(float));
	autox_conv2d(split_44_tmp_1, (float*)((int8_t*)weights) + 843320, (float*)((int8_t*)weights) + 843360, relu_122_tmp_0, split_44_tmp_1_dim, conv2d_219_w_0_dim, relu_122_tmp_0_dim, 1, 0, 1, 1, 1);
	free(split_44_tmp_1);

	float* batch_norm_220_tmp_2 = (float*)calloc(7680, sizeof(float));
	autox_conv2d(relu_122_tmp_0, (float*)((int8_t*)weights) + 844960, (float*)((int8_t*)weights) + 845000, batch_norm_220_tmp_2, relu_122_tmp_0_dim, conv2d_220_w_0_dim, batch_norm_220_tmp_2_dim, 40, 1, 1, 1, 0);
	free(relu_122_tmp_0);

	float* relu_123_tmp_0 = (float*)calloc(7680, sizeof(float));
	autox_conv2d(batch_norm_220_tmp_2, (float*)((int8_t*)weights) + 845360, (float*)((int8_t*)weights) + 845400, relu_123_tmp_0, batch_norm_220_tmp_2_dim, conv2d_221_w_0_dim, relu_123_tmp_0_dim, 1, 0, 1, 1, 1);
	free(batch_norm_220_tmp_2);

	float* p_440[] = { split_44_tmp_0, relu_123_tmp_0, };
	uint16_t* p_440_dim[] = { split_44_tmp_0_dim, relu_123_tmp_0_dim, };
	float* concat_44_tmp_0 = (float*)calloc(15360, sizeof(float));
	autox_concat(p_440, concat_44_tmp_0, p_440_dim, concat_44_tmp_0_dim, 1, 2, 4);
	free(split_44_tmp_0);
	free(relu_123_tmp_0);

	float* transpose_44_tmp_0 = (float*)calloc(15360, sizeof(float));
	uint16_t axis_441[] = { 0, 2, 1, 3, 4 };
	autox_transpose(concat_44_tmp_0, transpose_44_tmp_0, reshape2_88_tmp_0_dim, transpose_44_tmp_0_dim, axis_441, 5);
	free(concat_44_tmp_0);

	float* relu_111_tmp_0 = (float*)calloc(7680, sizeof(float));
	autox_fusion_elementwise_add_activation(tmp_54, nearest_interp_v2_19_tmp_0, relu_111_tmp_0, tmp_54_dim, nearest_interp_v2_19_tmp_0_dim, relu_111_tmp_0_dim, -1, 4, 4, 4);
	free(tmp_54);
	free(nearest_interp_v2_19_tmp_0);

	float* split_45_tmp_0 = (float*)calloc(3840, sizeof(float));
	float* split_45_tmp_1 = (float*)calloc(3840, sizeof(float));
	float* p_443[] = { split_45_tmp_0, split_45_tmp_1, };
	uint16_t* p_443_dim[] = { split_45_tmp_0_dim, split_45_tmp_1_dim, };
	autox_split(relu_111_tmp_0, p_443, relu_111_tmp_0_dim, p_443_dim, 1, 4, 2, 4);
	free(relu_111_tmp_0);

	float* relu_124_tmp_0 = (float*)calloc(3840, sizeof(float));
	autox_conv2d(split_45_tmp_1, (float*)((int8_t*)weights) + 847000, (float*)((int8_t*)weights) + 847080, relu_124_tmp_0, split_45_tmp_1_dim, conv2d_222_w_0_dim, relu_124_tmp_0_dim, 1, 0, 1, 1, 1);
	free(split_45_tmp_1);

	float* batch_norm_223_tmp_2 = (float*)calloc(3840, sizeof(float));
	autox_conv2d(relu_124_tmp_0, (float*)((int8_t*)weights) + 853480, (float*)((int8_t*)weights) + 853560, batch_norm_223_tmp_2, relu_124_tmp_0_dim, conv2d_223_w_0_dim, batch_norm_223_tmp_2_dim, 80, 1, 1, 1, 0);
	free(relu_124_tmp_0);

	float* relu_125_tmp_0 = (float*)calloc(3840, sizeof(float));
	autox_conv2d(batch_norm_223_tmp_2, (float*)((int8_t*)weights) + 854280, (float*)((int8_t*)weights) + 854360, relu_125_tmp_0, batch_norm_223_tmp_2_dim, conv2d_224_w_0_dim, relu_125_tmp_0_dim, 1, 0, 1, 1, 1);
	free(batch_norm_223_tmp_2);

	float* p_447[] = { split_45_tmp_0, relu_125_tmp_0, };
	uint16_t* p_447_dim[] = { split_45_tmp_0_dim, relu_125_tmp_0_dim, };
	float* concat_45_tmp_0 = (float*)calloc(7680, sizeof(float));
	autox_concat(p_447, concat_45_tmp_0, p_447_dim, concat_45_tmp_0_dim, 1, 2, 4);
	free(split_45_tmp_0);
	free(relu_125_tmp_0);

	float* transpose_45_tmp_0 = (float*)calloc(7680, sizeof(float));
	uint16_t axis_448[] = { 0, 2, 1, 3, 4 };
	autox_transpose(concat_45_tmp_0, transpose_45_tmp_0, reshape2_90_tmp_0_dim, transpose_45_tmp_0_dim, axis_448, 5);
	free(concat_45_tmp_0);

	float* split_46_tmp_0 = (float*)calloc(3840, sizeof(float));
	float* split_46_tmp_1 = (float*)calloc(3840, sizeof(float));
	float* p_449[] = { split_46_tmp_0, split_46_tmp_1, };
	uint16_t* p_449_dim[] = { split_46_tmp_0_dim, split_46_tmp_1_dim, };
	autox_split(transpose_45_tmp_0, p_449, reshape2_91_tmp_0_dim, p_449_dim, 1, 4, 2, 4);
	free(transpose_45_tmp_0);

	float* relu_126_tmp_0 = (float*)calloc(3840, sizeof(float));
	autox_conv2d(split_46_tmp_1, (float*)((int8_t*)weights) + 860760, (float*)((int8_t*)weights) + 860840, relu_126_tmp_0, split_46_tmp_1_dim, conv2d_225_w_0_dim, relu_126_tmp_0_dim, 1, 0, 1, 1, 1);
	free(split_46_tmp_1);

	float* batch_norm_226_tmp_2 = (float*)calloc(3840, sizeof(float));
	autox_conv2d(relu_126_tmp_0, (float*)((int8_t*)weights) + 867240, (float*)((int8_t*)weights) + 867320, batch_norm_226_tmp_2, relu_126_tmp_0_dim, conv2d_226_w_0_dim, batch_norm_226_tmp_2_dim, 80, 1, 1, 1, 0);
	free(relu_126_tmp_0);

	float* relu_127_tmp_0 = (float*)calloc(3840, sizeof(float));
	autox_conv2d(batch_norm_226_tmp_2, (float*)((int8_t*)weights) + 868040, (float*)((int8_t*)weights) + 868120, relu_127_tmp_0, batch_norm_226_tmp_2_dim, conv2d_227_w_0_dim, relu_127_tmp_0_dim, 1, 0, 1, 1, 1);
	free(batch_norm_226_tmp_2);

	float* p_453[] = { split_46_tmp_0, relu_127_tmp_0, };
	uint16_t* p_453_dim[] = { split_46_tmp_0_dim, relu_127_tmp_0_dim, };
	float* concat_46_tmp_0 = (float*)calloc(7680, sizeof(float));
	autox_concat(p_453, concat_46_tmp_0, p_453_dim, concat_46_tmp_0_dim, 1, 2, 4);
	free(split_46_tmp_0);
	free(relu_127_tmp_0);

	float* transpose_46_tmp_0 = (float*)calloc(7680, sizeof(float));
	uint16_t axis_454[] = { 0, 2, 1, 3, 4 };
	autox_transpose(concat_46_tmp_0, transpose_46_tmp_0, reshape2_92_tmp_0_dim, transpose_46_tmp_0_dim, axis_454, 5);
	free(concat_46_tmp_0);

	float* relu_115_tmp_0 = (float*)calloc(3840, sizeof(float));
	autox_fusion_elementwise_add_activation(tmp_58, transpose_40_tmp_0, relu_115_tmp_0, tmp_58_dim, reshape2_81_tmp_0_dim, relu_115_tmp_0_dim, -1, 4, 4, 4);
	free(tmp_58);
	free(transpose_40_tmp_0);

	float* split_47_tmp_0 = (float*)calloc(1920, sizeof(float));
	float* split_47_tmp_1 = (float*)calloc(1920, sizeof(float));
	float* p_456[] = { split_47_tmp_0, split_47_tmp_1, };
	uint16_t* p_456_dim[] = { split_47_tmp_0_dim, split_47_tmp_1_dim, };
	autox_split(relu_115_tmp_0, p_456, relu_115_tmp_0_dim, p_456_dim, 1, 4, 2, 4);
	free(relu_115_tmp_0);

	float* relu_128_tmp_0 = (float*)calloc(1920, sizeof(float));
	autox_conv2d(split_47_tmp_1, (float*)((int8_t*)weights) + 874520, (float*)((int8_t*)weights) + 874680, relu_128_tmp_0, split_47_tmp_1_dim, conv2d_228_w_0_dim, relu_128_tmp_0_dim, 1, 0, 1, 1, 1);
	free(split_47_tmp_1);

	float* batch_norm_229_tmp_2 = (float*)calloc(1920, sizeof(float));
	autox_conv2d(relu_128_tmp_0, (float*)((int8_t*)weights) + 900280, (float*)((int8_t*)weights) + 900440, batch_norm_229_tmp_2, relu_128_tmp_0_dim, conv2d_229_w_0_dim, batch_norm_229_tmp_2_dim, 160, 1, 1, 1, 0);
	free(relu_128_tmp_0);

	float* relu_129_tmp_0 = (float*)calloc(1920, sizeof(float));
	autox_conv2d(batch_norm_229_tmp_2, (float*)((int8_t*)weights) + 901880, (float*)((int8_t*)weights) + 902040, relu_129_tmp_0, batch_norm_229_tmp_2_dim, conv2d_230_w_0_dim, relu_129_tmp_0_dim, 1, 0, 1, 1, 1);
	free(batch_norm_229_tmp_2);

	float* p_460[] = { split_47_tmp_0, relu_129_tmp_0, };
	uint16_t* p_460_dim[] = { split_47_tmp_0_dim, relu_129_tmp_0_dim, };
	float* concat_47_tmp_0 = (float*)calloc(3840, sizeof(float));
	autox_concat(p_460, concat_47_tmp_0, p_460_dim, concat_47_tmp_0_dim, 1, 2, 4);
	free(split_47_tmp_0);
	free(relu_129_tmp_0);

	float* transpose_47_tmp_0 = (float*)calloc(3840, sizeof(float));
	uint16_t axis_461[] = { 0, 2, 1, 3, 4 };
	autox_transpose(concat_47_tmp_0, transpose_47_tmp_0, reshape2_94_tmp_0_dim, transpose_47_tmp_0_dim, axis_461, 5);
	free(concat_47_tmp_0);

	float* split_48_tmp_0 = (float*)calloc(1920, sizeof(float));
	float* split_48_tmp_1 = (float*)calloc(1920, sizeof(float));
	float* p_462[] = { split_48_tmp_0, split_48_tmp_1, };
	uint16_t* p_462_dim[] = { split_48_tmp_0_dim, split_48_tmp_1_dim, };
	autox_split(transpose_47_tmp_0, p_462, reshape2_95_tmp_0_dim, p_462_dim, 1, 4, 2, 4);
	free(transpose_47_tmp_0);

	float* relu_130_tmp_0 = (float*)calloc(1920, sizeof(float));
	autox_conv2d(split_48_tmp_1, (float*)((int8_t*)weights) + 927640, (float*)((int8_t*)weights) + 927800, relu_130_tmp_0, split_48_tmp_1_dim, conv2d_231_w_0_dim, relu_130_tmp_0_dim, 1, 0, 1, 1, 1);
	free(split_48_tmp_1);

	float* batch_norm_232_tmp_2 = (float*)calloc(1920, sizeof(float));
	autox_conv2d(relu_130_tmp_0, (float*)((int8_t*)weights) + 953400, (float*)((int8_t*)weights) + 953560, batch_norm_232_tmp_2, relu_130_tmp_0_dim, conv2d_232_w_0_dim, batch_norm_232_tmp_2_dim, 160, 1, 1, 1, 0);
	free(relu_130_tmp_0);

	float* relu_131_tmp_0 = (float*)calloc(1920, sizeof(float));
	autox_conv2d(batch_norm_232_tmp_2, (float*)((int8_t*)weights) + 955000, (float*)((int8_t*)weights) + 955160, relu_131_tmp_0, batch_norm_232_tmp_2_dim, conv2d_233_w_0_dim, relu_131_tmp_0_dim, 1, 0, 1, 1, 1);
	free(batch_norm_232_tmp_2);

	float* p_466[] = { split_48_tmp_0, relu_131_tmp_0, };
	uint16_t* p_466_dim[] = { split_48_tmp_0_dim, relu_131_tmp_0_dim, };
	float* concat_48_tmp_0 = (float*)calloc(3840, sizeof(float));
	autox_concat(p_466, concat_48_tmp_0, p_466_dim, concat_48_tmp_0_dim, 1, 2, 4);
	free(split_48_tmp_0);
	free(relu_131_tmp_0);

	float* transpose_48_tmp_0 = (float*)calloc(3840, sizeof(float));
	uint16_t axis_467[] = { 0, 2, 1, 3, 4 };
	autox_transpose(concat_48_tmp_0, transpose_48_tmp_0, reshape2_96_tmp_0_dim, transpose_48_tmp_0_dim, axis_467, 5);
	free(concat_48_tmp_0);

	float* tmp_60 = (float*)calloc(30720, sizeof(float));
	autox_elementwise_add(transpose_42_tmp_0, transpose_42_tmp_0, tmp_60, reshape2_85_tmp_0_dim, reshape2_85_tmp_0_dim, tmp_60_dim, -1, 4, 4, 4);
	free(transpose_42_tmp_0);
	// free(transpose_42_tmp_0);

	float* batch_norm_234_tmp_2 = (float*)calloc(7680, sizeof(float));
	autox_conv2d(transpose_44_tmp_0, (float*)((int8_t*)weights) + 980760, (float*)((int8_t*)weights) + 980800, batch_norm_234_tmp_2, reshape2_89_tmp_0_dim, conv2d_234_w_0_dim, batch_norm_234_tmp_2_dim, 1, 0, 1, 1, 0);
	//free(transpose_44_tmp_0);

	float* nearest_interp_v2_20_tmp_0 = (float*)calloc(30720, sizeof(float));
	autox_nearest_interp(batch_norm_234_tmp_2, nearest_interp_v2_20_tmp_0, batch_norm_234_tmp_2_dim, nearest_interp_v2_20_tmp_0_dim, 2, 0);
	free(batch_norm_234_tmp_2);

	float* tmp_61 = (float*)calloc(30720, sizeof(float));
	autox_elementwise_add(tmp_60, nearest_interp_v2_20_tmp_0, tmp_61, tmp_60_dim, nearest_interp_v2_20_tmp_0_dim, tmp_61_dim, -1, 4, 4, 4);
	free(tmp_60);
	free(nearest_interp_v2_20_tmp_0);

	float* batch_norm_235_tmp_2 = (float*)calloc(1920, sizeof(float));
	autox_conv2d(transpose_46_tmp_0, (float*)((int8_t*)weights) + 984000, (float*)((int8_t*)weights) + 984040, batch_norm_235_tmp_2, reshape2_93_tmp_0_dim, conv2d_235_w_0_dim, batch_norm_235_tmp_2_dim, 1, 0, 1, 1, 0);
	//free(transpose_46_tmp_0);

	float* nearest_interp_v2_21_tmp_0 = (float*)calloc(30720, sizeof(float));
	autox_nearest_interp(batch_norm_235_tmp_2, nearest_interp_v2_21_tmp_0, batch_norm_235_tmp_2_dim, nearest_interp_v2_21_tmp_0_dim, 4, 0);
	free(batch_norm_235_tmp_2);

	float* tmp_62 = (float*)calloc(30720, sizeof(float));
	autox_elementwise_add(tmp_61, nearest_interp_v2_21_tmp_0, tmp_62, tmp_61_dim, nearest_interp_v2_21_tmp_0_dim, tmp_62_dim, -1, 4, 4, 4);
	free(tmp_61);
	free(nearest_interp_v2_21_tmp_0);

	float* batch_norm_236_tmp_2 = (float*)calloc(480, sizeof(float));
	autox_conv2d(transpose_48_tmp_0, (float*)((int8_t*)weights) + 990440, (float*)((int8_t*)weights) + 990480, batch_norm_236_tmp_2, reshape2_97_tmp_0_dim, conv2d_236_w_0_dim, batch_norm_236_tmp_2_dim, 1, 0, 1, 1, 0);
	//free(transpose_48_tmp_0);

	float* nearest_interp_v2_22_tmp_0 = (float*)calloc(30720, sizeof(float));
	autox_nearest_interp(batch_norm_236_tmp_2, nearest_interp_v2_22_tmp_0, batch_norm_236_tmp_2_dim, nearest_interp_v2_22_tmp_0_dim, 8, 0);
	free(batch_norm_236_tmp_2);

	float* tmp_63 = (float*)calloc(30720, sizeof(float));
	autox_elementwise_add(tmp_62, nearest_interp_v2_22_tmp_0, tmp_63, tmp_62_dim, nearest_interp_v2_22_tmp_0_dim, tmp_63_dim, -1, 4, 4, 4);
	free(tmp_62);
	free(nearest_interp_v2_22_tmp_0);

	float* relu_132_tmp_0 = (float*)calloc(30720, sizeof(float));
	autox_relu_noreplace(tmp_63, relu_132_tmp_0, tmp_63_dim, 4);

	float* batch_norm_237_tmp_2 = (float*)calloc(7680, sizeof(float));
	autox_conv2d(tmp_63, (float*)((int8_t*)weights) + 1003280, (float*)((int8_t*)weights) + 1003320, batch_norm_237_tmp_2, tmp_63_dim, conv2d_237_w_0_dim, batch_norm_237_tmp_2_dim, 40, 1, 2, 1, 0);
	//free(tmp_63);

	float* batch_norm_238_tmp_2 = (float*)calloc(15360, sizeof(float));
	autox_conv2d(batch_norm_237_tmp_2, (float*)((int8_t*)weights) + 1003680, (float*)((int8_t*)weights) + 1003760, batch_norm_238_tmp_2, batch_norm_237_tmp_2_dim, conv2d_238_w_0_dim, batch_norm_238_tmp_2_dim, 1, 0, 1, 1, 0);
	free(batch_norm_237_tmp_2);

	float* tmp_64 = (float*)calloc(15360, sizeof(float));
	autox_elementwise_add(batch_norm_238_tmp_2, batch_norm_238_tmp_2, tmp_64, batch_norm_238_tmp_2_dim, batch_norm_238_tmp_2_dim, tmp_64_dim, -1, 4, 4, 4);
	free(batch_norm_238_tmp_2);
	// free(batch_norm_238_tmp_2);

	float* tmp_65 = (float*)calloc(15360, sizeof(float));
	autox_elementwise_add(tmp_64, transpose_44_tmp_0, tmp_65, tmp_64_dim, reshape2_89_tmp_0_dim, tmp_65_dim, -1, 4, 4, 4);
	free(tmp_64);
	//free(transpose_44_tmp_0);

	float* batch_norm_239_tmp_2 = (float*)calloc(3840, sizeof(float));
	autox_conv2d(transpose_46_tmp_0, (float*)((int8_t*)weights) + 1006960, (float*)((int8_t*)weights) + 1007040, batch_norm_239_tmp_2, reshape2_93_tmp_0_dim, conv2d_239_w_0_dim, batch_norm_239_tmp_2_dim, 1, 0, 1, 1, 0);
	// free(transpose_46_tmp_0);

	float* nearest_interp_v2_23_tmp_0 = (float*)calloc(15360, sizeof(float));
	autox_nearest_interp(batch_norm_239_tmp_2, nearest_interp_v2_23_tmp_0, batch_norm_239_tmp_2_dim, nearest_interp_v2_23_tmp_0_dim, 2, 0);
	free(batch_norm_239_tmp_2);

	float* tmp_66 = (float*)calloc(15360, sizeof(float));
	autox_elementwise_add(tmp_65, nearest_interp_v2_23_tmp_0, tmp_66, tmp_65_dim, nearest_interp_v2_23_tmp_0_dim, tmp_66_dim, -1, 4, 4, 4);
	free(tmp_65);
	free(nearest_interp_v2_23_tmp_0);

	float* batch_norm_240_tmp_2 = (float*)calloc(960, sizeof(float));
	autox_conv2d(transpose_48_tmp_0, (float*)((int8_t*)weights) + 1019840, (float*)((int8_t*)weights) + 1019920, batch_norm_240_tmp_2, reshape2_97_tmp_0_dim, conv2d_240_w_0_dim, batch_norm_240_tmp_2_dim, 1, 0, 1, 1, 0);
	// free(transpose_48_tmp_0);

	float* nearest_interp_v2_24_tmp_0 = (float*)calloc(15360, sizeof(float));
	autox_nearest_interp(batch_norm_240_tmp_2, nearest_interp_v2_24_tmp_0, batch_norm_240_tmp_2_dim, nearest_interp_v2_24_tmp_0_dim, 4, 0);
	free(batch_norm_240_tmp_2);

	float* batch_norm_241_tmp_2 = (float*)calloc(7680, sizeof(float));
	autox_conv2d(tmp_63, (float*)((int8_t*)weights) + 1045520, (float*)((int8_t*)weights) + 1045560, batch_norm_241_tmp_2, tmp_63_dim, conv2d_241_w_0_dim, batch_norm_241_tmp_2_dim, 40, 1, 2, 1, 0);
	//free(tmp_63);

	float* relu_134_tmp_0 = (float*)calloc(7680, sizeof(float));
	autox_conv2d(batch_norm_241_tmp_2, (float*)((int8_t*)weights) + 1045920, (float*)((int8_t*)weights) + 1045960, relu_134_tmp_0, batch_norm_241_tmp_2_dim, conv2d_242_w_0_dim, relu_134_tmp_0_dim, 1, 0, 1, 1, 1);
	free(batch_norm_241_tmp_2);

	float* batch_norm_243_tmp_2 = (float*)calloc(1920, sizeof(float));
	autox_conv2d(relu_134_tmp_0, (float*)((int8_t*)weights) + 1047560, (float*)((int8_t*)weights) + 1047600, batch_norm_243_tmp_2, relu_134_tmp_0_dim, conv2d_243_w_0_dim, batch_norm_243_tmp_2_dim, 40, 1, 2, 1, 0);
	free(relu_134_tmp_0);

	float* batch_norm_244_tmp_2 = (float*)calloc(7680, sizeof(float));
	autox_conv2d(batch_norm_243_tmp_2, (float*)((int8_t*)weights) + 1047960, (float*)((int8_t*)weights) + 1048120, batch_norm_244_tmp_2, batch_norm_243_tmp_2_dim, conv2d_244_w_0_dim, batch_norm_244_tmp_2_dim, 1, 0, 1, 1, 0);
	free(batch_norm_243_tmp_2);

	float* tmp_68 = (float*)calloc(7680, sizeof(float));
	autox_elementwise_add(batch_norm_244_tmp_2, batch_norm_244_tmp_2, tmp_68, batch_norm_244_tmp_2_dim, batch_norm_244_tmp_2_dim, tmp_68_dim, -1, 4, 4, 4);
	free(batch_norm_244_tmp_2);
	//free(batch_norm_244_tmp_2);

	float* batch_norm_245_tmp_2 = (float*)calloc(3840, sizeof(float));
	autox_conv2d(transpose_44_tmp_0, (float*)((int8_t*)weights) + 1054520, (float*)((int8_t*)weights) + 1054600, batch_norm_245_tmp_2, reshape2_89_tmp_0_dim, conv2d_245_w_0_dim, batch_norm_245_tmp_2_dim, 80, 1, 2, 1, 0);
	//free(transpose_44_tmp_0);

	float* batch_norm_246_tmp_2 = (float*)calloc(7680, sizeof(float));
	autox_conv2d(batch_norm_245_tmp_2, (float*)((int8_t*)weights) + 1055320, (float*)((int8_t*)weights) + 1055480, batch_norm_246_tmp_2, batch_norm_245_tmp_2_dim, conv2d_246_w_0_dim, batch_norm_246_tmp_2_dim, 1, 0, 1, 1, 0);
	free(batch_norm_245_tmp_2);

	float* tmp_69 = (float*)calloc(7680, sizeof(float));
	autox_elementwise_add(tmp_68, batch_norm_246_tmp_2, tmp_69, tmp_68_dim, batch_norm_246_tmp_2_dim, tmp_69_dim, -1, 4, 4, 4);
	free(tmp_68);
	free(batch_norm_246_tmp_2);

	float* tmp_70 = (float*)calloc(7680, sizeof(float));
	autox_elementwise_add(tmp_69, transpose_46_tmp_0, tmp_70, tmp_69_dim, reshape2_93_tmp_0_dim, tmp_70_dim, -1, 4, 4, 4);
	free(tmp_69);
	//free(transpose_46_tmp_0);

	float* batch_norm_247_tmp_2 = (float*)calloc(1920, sizeof(float));
	autox_conv2d(transpose_48_tmp_0, (float*)((int8_t*)weights) + 1068280, (float*)((int8_t*)weights) + 1068440, batch_norm_247_tmp_2, reshape2_97_tmp_0_dim, conv2d_247_w_0_dim, batch_norm_247_tmp_2_dim, 1, 0, 1, 1, 0);
	//free(transpose_48_tmp_0);

	float* nearest_interp_v2_25_tmp_0 = (float*)calloc(7680, sizeof(float));
	autox_nearest_interp(batch_norm_247_tmp_2, nearest_interp_v2_25_tmp_0, batch_norm_247_tmp_2_dim, nearest_interp_v2_25_tmp_0_dim, 2, 0);
	free(batch_norm_247_tmp_2);

	float* batch_norm_248_tmp_2 = (float*)calloc(7680, sizeof(float));
	autox_conv2d(tmp_63, (float*)((int8_t*)weights) + 1119640, (float*)((int8_t*)weights) + 1119680, batch_norm_248_tmp_2, tmp_63_dim, conv2d_248_w_0_dim, batch_norm_248_tmp_2_dim, 40, 1, 2, 1, 0);
	//free(tmp_63);

	float* relu_136_tmp_0 = (float*)calloc(7680, sizeof(float));
	autox_conv2d(batch_norm_248_tmp_2, (float*)((int8_t*)weights) + 1120040, (float*)((int8_t*)weights) + 1120080, relu_136_tmp_0, batch_norm_248_tmp_2_dim, conv2d_249_w_0_dim, relu_136_tmp_0_dim, 1, 0, 1, 1, 1);
	free(batch_norm_248_tmp_2);

	float* batch_norm_250_tmp_2 = (float*)calloc(1920, sizeof(float));
	autox_conv2d(relu_136_tmp_0, (float*)((int8_t*)weights) + 1121680, (float*)((int8_t*)weights) + 1121720, batch_norm_250_tmp_2, relu_136_tmp_0_dim, conv2d_250_w_0_dim, batch_norm_250_tmp_2_dim, 40, 1, 2, 1, 0);
	free(relu_136_tmp_0);

	float* relu_137_tmp_0 = (float*)calloc(1920, sizeof(float));
	autox_conv2d(batch_norm_250_tmp_2, (float*)((int8_t*)weights) + 1122080, (float*)((int8_t*)weights) + 1122120, relu_137_tmp_0, batch_norm_250_tmp_2_dim, conv2d_251_w_0_dim, relu_137_tmp_0_dim, 1, 0, 1, 1, 1);
	free(batch_norm_250_tmp_2);

	float* batch_norm_252_tmp_2 = (float*)calloc(480, sizeof(float));
	autox_conv2d(relu_137_tmp_0, (float*)((int8_t*)weights) + 1123720, (float*)((int8_t*)weights) + 1123760, batch_norm_252_tmp_2, relu_137_tmp_0_dim, conv2d_252_w_0_dim, batch_norm_252_tmp_2_dim, 40, 1, 2, 1, 0);
	free(relu_137_tmp_0);

	float* batch_norm_253_tmp_2 = (float*)calloc(3840, sizeof(float));
	autox_conv2d(batch_norm_252_tmp_2, (float*)((int8_t*)weights) + 1124120, (float*)((int8_t*)weights) + 1124440, batch_norm_253_tmp_2, batch_norm_252_tmp_2_dim, conv2d_253_w_0_dim, batch_norm_253_tmp_2_dim, 1, 0, 1, 1, 0);
	free(batch_norm_252_tmp_2);

	float* tmp_72 = (float*)calloc(3840, sizeof(float));
	autox_elementwise_add(batch_norm_253_tmp_2, batch_norm_253_tmp_2, tmp_72, batch_norm_253_tmp_2_dim, batch_norm_253_tmp_2_dim, tmp_72_dim, -1, 4, 4, 4);
	free(batch_norm_253_tmp_2);
	//free(batch_norm_253_tmp_2);

	float* batch_norm_254_tmp_2 = (float*)calloc(3840, sizeof(float));
	autox_conv2d(transpose_44_tmp_0, (float*)((int8_t*)weights) + 1137240, (float*)((int8_t*)weights) + 1137320, batch_norm_254_tmp_2, reshape2_89_tmp_0_dim, conv2d_254_w_0_dim, batch_norm_254_tmp_2_dim, 80, 1, 2, 1, 0);
	free(transpose_44_tmp_0);

	float* relu_138_tmp_0 = (float*)calloc(3840, sizeof(float));
	autox_conv2d(batch_norm_254_tmp_2, (float*)((int8_t*)weights) + 1138040, (float*)((int8_t*)weights) + 1138120, relu_138_tmp_0, batch_norm_254_tmp_2_dim, conv2d_255_w_0_dim, relu_138_tmp_0_dim, 1, 0, 1, 1, 1);
	free(batch_norm_254_tmp_2);

	float* batch_norm_256_tmp_2 = (float*)calloc(960, sizeof(float));
	autox_conv2d(relu_138_tmp_0, (float*)((int8_t*)weights) + 1144520, (float*)((int8_t*)weights) + 1144600, batch_norm_256_tmp_2, relu_138_tmp_0_dim, conv2d_256_w_0_dim, batch_norm_256_tmp_2_dim, 80, 1, 2, 1, 0);
	free(relu_138_tmp_0);

	float* batch_norm_257_tmp_2 = (float*)calloc(3840, sizeof(float));
	autox_conv2d(batch_norm_256_tmp_2, (float*)((int8_t*)weights) + 1145320, (float*)((int8_t*)weights) + 1145640, batch_norm_257_tmp_2, batch_norm_256_tmp_2_dim, conv2d_257_w_0_dim, batch_norm_257_tmp_2_dim, 1, 0, 1, 1, 0);
	free(batch_norm_256_tmp_2);

	float* tmp_73 = (float*)calloc(3840, sizeof(float));
	autox_elementwise_add(tmp_72, batch_norm_257_tmp_2, tmp_73, tmp_72_dim, batch_norm_257_tmp_2_dim, tmp_73_dim, -1, 4, 4, 4);
	free(tmp_72);
	free(batch_norm_257_tmp_2);

	float* batch_norm_258_tmp_2 = (float*)calloc(1920, sizeof(float));
	autox_conv2d(transpose_46_tmp_0, (float*)((int8_t*)weights) + 1171240, (float*)((int8_t*)weights) + 1171400, batch_norm_258_tmp_2, reshape2_93_tmp_0_dim, conv2d_258_w_0_dim, batch_norm_258_tmp_2_dim, 160, 1, 2, 1, 0);
	free(transpose_46_tmp_0);

	float* batch_norm_259_tmp_2 = (float*)calloc(3840, sizeof(float));
	autox_conv2d(batch_norm_258_tmp_2, (float*)((int8_t*)weights) + 1172840, (float*)((int8_t*)weights) + 1173160, batch_norm_259_tmp_2, batch_norm_258_tmp_2_dim, conv2d_259_w_0_dim, batch_norm_259_tmp_2_dim, 1, 0, 1, 1, 0);
	free(batch_norm_258_tmp_2);

	float* tmp_74 = (float*)calloc(3840, sizeof(float));
	autox_elementwise_add(tmp_73, batch_norm_259_tmp_2, tmp_74, tmp_73_dim, batch_norm_259_tmp_2_dim, tmp_74_dim, -1, 4, 4, 4);
	free(tmp_73);
	free(batch_norm_259_tmp_2);

	float* relu_139_tmp_0 = (float*)calloc(3840, sizeof(float));
	autox_fusion_elementwise_add_activation(tmp_74, transpose_48_tmp_0, relu_139_tmp_0, tmp_74_dim, reshape2_97_tmp_0_dim, relu_139_tmp_0_dim, -1, 4, 4, 4);
	free(tmp_74);
	free(transpose_48_tmp_0);

	float* batch_norm_260_tmp_2 = (float*)calloc(3840, sizeof(float));
	autox_conv2d(relu_139_tmp_0, (float*)((int8_t*)weights) + 1224360, (float*)((int8_t*)weights) + 1224680, batch_norm_260_tmp_2, relu_139_tmp_0_dim, conv2d_260_w_0_dim, batch_norm_260_tmp_2_dim, 320, 1, 1, 1, 0);
	free(relu_139_tmp_0);

	float* relu_140_tmp_0 = (float*)calloc(1920, sizeof(float));
	autox_conv2d(batch_norm_260_tmp_2, (float*)((int8_t*)weights) + 1227560, (float*)((int8_t*)weights) + 1227720, relu_140_tmp_0, batch_norm_260_tmp_2_dim, conv2d_261_w_0_dim, relu_140_tmp_0_dim, 1, 0, 1, 1, 1);
	free(batch_norm_260_tmp_2);

	float* bilinear_interp_v2_0_tmp_0 = (float*)calloc(7680, sizeof(float));
	autox_bilinear_interp(relu_140_tmp_0, bilinear_interp_v2_0_tmp_0, relu_140_tmp_0_dim, bilinear_interp_v2_0_tmp_0_dim, 0, 1, 0);
	free(relu_140_tmp_0);

	float* relu_135_tmp_0 = (float*)calloc(7680, sizeof(float));
	autox_fusion_elementwise_add_activation(tmp_70, nearest_interp_v2_25_tmp_0, relu_135_tmp_0, tmp_70_dim, nearest_interp_v2_25_tmp_0_dim, relu_135_tmp_0_dim, -1, 4, 4, 4);
	free(tmp_70);
	free(nearest_interp_v2_25_tmp_0);

	float* tmp_76 = (float*)calloc(7680, sizeof(float));
	autox_elementwise_add(relu_135_tmp_0, bilinear_interp_v2_0_tmp_0, tmp_76, relu_135_tmp_0_dim, bilinear_interp_v2_0_tmp_0_dim, tmp_76_dim, -1, 4, 4, 4);
	free(relu_135_tmp_0);
	free(bilinear_interp_v2_0_tmp_0);

	float* batch_norm_262_tmp_2 = (float*)calloc(7680, sizeof(float));
	autox_conv2d(tmp_76, (float*)((int8_t*)weights) + 1278920, (float*)((int8_t*)weights) + 1279080, batch_norm_262_tmp_2, tmp_76_dim, conv2d_262_w_0_dim, batch_norm_262_tmp_2_dim, 160, 1, 1, 1, 0);
	free(tmp_76);

	float* relu_141_tmp_0 = (float*)calloc(3840, sizeof(float));
	autox_conv2d(batch_norm_262_tmp_2, (float*)((int8_t*)weights) + 1280520, (float*)((int8_t*)weights) + 1280600, relu_141_tmp_0, batch_norm_262_tmp_2_dim, conv2d_263_w_0_dim, relu_141_tmp_0_dim, 1, 0, 1, 1, 1);
	free(batch_norm_262_tmp_2);

	float* bilinear_interp_v2_1_tmp_0 = (float*)calloc(15360, sizeof(float));
	autox_bilinear_interp(relu_141_tmp_0, bilinear_interp_v2_1_tmp_0, relu_141_tmp_0_dim, bilinear_interp_v2_1_tmp_0_dim, 0, 1,0);
	free(relu_141_tmp_0);

	float* relu_133_tmp_0 = (float*)calloc(15360, sizeof(float));
	autox_fusion_elementwise_add_activation(tmp_66, nearest_interp_v2_24_tmp_0, relu_133_tmp_0, tmp_66_dim, nearest_interp_v2_24_tmp_0_dim, relu_133_tmp_0_dim, -1, 4, 4, 4);
	free(tmp_66);
	free(nearest_interp_v2_24_tmp_0);

	float* tmp_77 = (float*)calloc(15360, sizeof(float));
	autox_elementwise_add(relu_133_tmp_0, bilinear_interp_v2_1_tmp_0, tmp_77, relu_133_tmp_0_dim, bilinear_interp_v2_1_tmp_0_dim, tmp_77_dim, -1, 4, 4, 4);
	free(relu_133_tmp_0);
	free(bilinear_interp_v2_1_tmp_0);

	float* batch_norm_264_tmp_2 = (float*)calloc(15360, sizeof(float));
	autox_conv2d(tmp_77, (float*)((int8_t*)weights) + 1293400, (float*)((int8_t*)weights) + 1293480, batch_norm_264_tmp_2, tmp_77_dim, conv2d_264_w_0_dim, batch_norm_264_tmp_2_dim, 80, 1, 1, 1, 0);
	free(tmp_77);

	float* relu_142_tmp_0 = (float*)calloc(7680, sizeof(float));
	autox_conv2d(batch_norm_264_tmp_2, (float*)((int8_t*)weights) + 1294200, (float*)((int8_t*)weights) + 1294240, relu_142_tmp_0, batch_norm_264_tmp_2_dim, conv2d_265_w_0_dim, relu_142_tmp_0_dim, 1, 0, 1, 1, 1);
	free(batch_norm_264_tmp_2);

	float* bilinear_interp_v2_2_tmp_0 = (float*)calloc(30720, sizeof(float));
	autox_bilinear_interp(relu_142_tmp_0, bilinear_interp_v2_2_tmp_0, relu_142_tmp_0_dim, bilinear_interp_v2_2_tmp_0_dim, 0, 1, 0);
	free(relu_142_tmp_0);

	float* tmp_78 = (float*)calloc(30720, sizeof(float));
	autox_elementwise_add(relu_132_tmp_0, bilinear_interp_v2_2_tmp_0, tmp_78, relu_132_tmp_0_dim, bilinear_interp_v2_2_tmp_0_dim, tmp_78_dim, -1, 4, 4, 4);
	free(relu_132_tmp_0);
	free(bilinear_interp_v2_2_tmp_0);

	float* batch_norm_266_tmp_2 = (float*)calloc(30720, sizeof(float));
	autox_conv2d(tmp_78, (float*)((int8_t*)weights) + 1297440, (float*)((int8_t*)weights) + 1297480, batch_norm_266_tmp_2, tmp_78_dim, conv2d_266_w_0_dim, batch_norm_266_tmp_2_dim, 40, 1, 1, 1, 0);
	free(tmp_78);

	float* relu_143_tmp_0 = (float*)calloc(30720, sizeof(float));
	autox_conv2d(batch_norm_266_tmp_2, (float*)((int8_t*)weights) + 1297840, (float*)((int8_t*)weights) + 1297880, relu_143_tmp_0, batch_norm_266_tmp_2_dim, conv2d_267_w_0_dim, relu_143_tmp_0_dim, 1, 0, 1, 1, 1);
	free(batch_norm_266_tmp_2);

	autox_conv2d(relu_143_tmp_0, (float*)((int8_t*)weights) + 1299480, (float*)((int8_t*)weights) + 1299497, Out, relu_143_tmp_0_dim, conv2d_268_w_0_dim, conv2d_441_tmp_1_dim, 1, 0, 1, 1, 0);
	free(relu_143_tmp_0);

}
