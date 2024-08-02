#include "../include/autox_models.h"

void shufflenetv2_x_0_25(const float* x, const uint16_t ssize_h, const uint16_t ssize_w, void* weights, float* Out)
{
	uint16_t x_dim[] = { 1, 3, 224, 224 };
	uint16_t batch_norm_0_tmp_4_dim[] = { 1, 24, 112, 112 };
	uint16_t stage1_conv_bn_offset_dim[] = { 24 };
	uint16_t stage1_conv_weights_dim[] = { 24, 3, 3, 3 };
	uint16_t pool2d_0_tmp_0_dim[] = { 1, 24, 56, 56 };
	uint16_t batch_norm_1_tmp_3_dim[] = { 1, 24, 28, 28 };
	uint16_t stage_2_1_conv4_bn_offset_dim[] = { 24 };
	uint16_t stage_2_1_conv4_weights_dim[] = { 24, 1, 3, 3 };
	uint16_t batch_norm_3_tmp_4_dim[] = { 1, 12, 56, 56 };
	uint16_t stage_2_1_conv1_bn_offset_dim[] = { 12 };
	uint16_t stage_2_1_conv1_weights_dim[] = { 12, 24, 1, 1 };
	uint16_t batch_norm_4_tmp_3_dim[] = { 1, 12, 28, 28 };
	uint16_t stage_2_1_conv2_bn_offset_dim[] = { 12 };
	uint16_t stage_2_1_conv2_weights_dim[] = { 12, 1, 3, 3 };
	uint16_t batch_norm_2_tmp_4_dim[] = { 1, 12, 28, 28 };
	uint16_t stage_2_1_conv5_bn_offset_dim[] = { 12 };
	uint16_t stage_2_1_conv5_weights_dim[] = { 12, 24, 1, 1 };
	uint16_t batch_norm_5_tmp_4_dim[] = { 1, 12, 28, 28 };
	uint16_t stage_2_1_conv3_bn_offset_dim[] = { 12 };
	uint16_t stage_2_1_conv3_weights_dim[] = { 12, 12, 1, 1 };
	uint16_t concat_0_tmp_0_dim[] = { 1, 24, 28, 28 };
	uint16_t shape_0_tmp_0_dim[] = { 4 };
	uint16_t shape_0_tmp_0_slice_0_dim[] = { 1 };
	uint16_t reshape2_0_tmp_0_dim[] = { 1, 2, 12, 28, 28 };
	uint16_t reshape2_0_tmp_1_dim[] = { 0, 1, 24, 28, 28 };
	uint16_t transpose_0_tmp_0_dim[] = { 1, 12, 2, 28, 28 };
	uint16_t transpose_0_tmp_1_dim[] = { 0, 1, 2, 12, 28, 28 };
	uint16_t reshape2_1_tmp_0_dim[] = { 1, 24, 28, 28 };
	uint16_t reshape2_1_tmp_1_dim[] = { 0, 1, 12, 2, 28, 28 };
	uint16_t split_0_tmp_0_dim[] = { 1, 12, 28, 28 };
	uint16_t split_0_tmp_1_dim[] = { 1, 12, 28, 28 };
	uint16_t batch_norm_6_tmp_4_dim[] = { 1, 12, 28, 28 };
	uint16_t stage_2_2_conv1_bn_offset_dim[] = { 12 };
	uint16_t stage_2_2_conv1_weights_dim[] = { 12, 12, 1, 1 };
	uint16_t batch_norm_7_tmp_3_dim[] = { 1, 12, 28, 28 };
	uint16_t stage_2_2_conv2_bn_offset_dim[] = { 12 };
	uint16_t stage_2_2_conv2_weights_dim[] = { 12, 1, 3, 3 };
	uint16_t batch_norm_8_tmp_4_dim[] = { 1, 12, 28, 28 };
	uint16_t stage_2_2_conv3_bn_offset_dim[] = { 12 };
	uint16_t stage_2_2_conv3_weights_dim[] = { 12, 12, 1, 1 };
	uint16_t concat_1_tmp_0_dim[] = { 1, 24, 28, 28 };
	uint16_t shape_3_tmp_0_dim[] = { 4 };
	uint16_t shape_3_tmp_0_slice_0_dim[] = { 1 };
	uint16_t reshape2_2_tmp_0_dim[] = { 1, 2, 12, 28, 28 };
	uint16_t reshape2_2_tmp_1_dim[] = { 0, 1, 24, 28, 28 };
	uint16_t transpose_1_tmp_0_dim[] = { 1, 12, 2, 28, 28 };
	uint16_t transpose_1_tmp_1_dim[] = { 0, 1, 2, 12, 28, 28 };
	uint16_t reshape2_3_tmp_0_dim[] = { 1, 24, 28, 28 };
	uint16_t reshape2_3_tmp_1_dim[] = { 0, 1, 12, 2, 28, 28 };
	uint16_t split_1_tmp_0_dim[] = { 1, 12, 28, 28 };
	uint16_t split_1_tmp_1_dim[] = { 1, 12, 28, 28 };
	uint16_t batch_norm_9_tmp_4_dim[] = { 1, 12, 28, 28 };
	uint16_t stage_2_3_conv1_bn_offset_dim[] = { 12 };
	uint16_t stage_2_3_conv1_weights_dim[] = { 12, 12, 1, 1 };
	uint16_t batch_norm_10_tmp_3_dim[] = { 1, 12, 28, 28 };
	uint16_t stage_2_3_conv2_bn_offset_dim[] = { 12 };
	uint16_t stage_2_3_conv2_weights_dim[] = { 12, 1, 3, 3 };
	uint16_t batch_norm_11_tmp_4_dim[] = { 1, 12, 28, 28 };
	uint16_t stage_2_3_conv3_bn_offset_dim[] = { 12 };
	uint16_t stage_2_3_conv3_weights_dim[] = { 12, 12, 1, 1 };
	uint16_t concat_2_tmp_0_dim[] = { 1, 24, 28, 28 };
	uint16_t shape_6_tmp_0_dim[] = { 4 };
	uint16_t shape_6_tmp_0_slice_0_dim[] = { 1 };
	uint16_t reshape2_4_tmp_0_dim[] = { 1, 2, 12, 28, 28 };
	uint16_t reshape2_4_tmp_1_dim[] = { 0, 1, 24, 28, 28 };
	uint16_t transpose_2_tmp_0_dim[] = { 1, 12, 2, 28, 28 };
	uint16_t transpose_2_tmp_1_dim[] = { 0, 1, 2, 12, 28, 28 };
	uint16_t reshape2_5_tmp_0_dim[] = { 1, 24, 28, 28 };
	uint16_t reshape2_5_tmp_1_dim[] = { 0, 1, 12, 2, 28, 28 };
	uint16_t split_2_tmp_0_dim[] = { 1, 12, 28, 28 };
	uint16_t split_2_tmp_1_dim[] = { 1, 12, 28, 28 };
	uint16_t batch_norm_12_tmp_4_dim[] = { 1, 12, 28, 28 };
	uint16_t stage_2_4_conv1_bn_offset_dim[] = { 12 };
	uint16_t stage_2_4_conv1_weights_dim[] = { 12, 12, 1, 1 };
	uint16_t batch_norm_13_tmp_3_dim[] = { 1, 12, 28, 28 };
	uint16_t stage_2_4_conv2_bn_offset_dim[] = { 12 };
	uint16_t stage_2_4_conv2_weights_dim[] = { 12, 1, 3, 3 };
	uint16_t batch_norm_14_tmp_4_dim[] = { 1, 12, 28, 28 };
	uint16_t stage_2_4_conv3_bn_offset_dim[] = { 12 };
	uint16_t stage_2_4_conv3_weights_dim[] = { 12, 12, 1, 1 };
	uint16_t concat_3_tmp_0_dim[] = { 1, 24, 28, 28 };
	uint16_t shape_9_tmp_0_dim[] = { 4 };
	uint16_t shape_9_tmp_0_slice_0_dim[] = { 1 };
	uint16_t reshape2_6_tmp_0_dim[] = { 1, 2, 12, 28, 28 };
	uint16_t reshape2_6_tmp_1_dim[] = { 0, 1, 24, 28, 28 };
	uint16_t transpose_3_tmp_0_dim[] = { 1, 12, 2, 28, 28 };
	uint16_t transpose_3_tmp_1_dim[] = { 0, 1, 2, 12, 28, 28 };
	uint16_t reshape2_7_tmp_0_dim[] = { 1, 24, 28, 28 };
	uint16_t reshape2_7_tmp_1_dim[] = { 0, 1, 12, 2, 28, 28 };
	uint16_t batch_norm_15_tmp_3_dim[] = { 1, 24, 14, 14 };
	uint16_t stage_3_1_conv4_bn_offset_dim[] = { 24 };
	uint16_t stage_3_1_conv4_weights_dim[] = { 24, 1, 3, 3 };
	uint16_t batch_norm_17_tmp_4_dim[] = { 1, 24, 28, 28 };
	uint16_t stage_3_1_conv1_bn_offset_dim[] = { 24 };
	uint16_t stage_3_1_conv1_weights_dim[] = { 24, 24, 1, 1 };
	uint16_t batch_norm_18_tmp_3_dim[] = { 1, 24, 14, 14 };
	uint16_t stage_3_1_conv2_bn_offset_dim[] = { 24 };
	uint16_t stage_3_1_conv2_weights_dim[] = { 24, 1, 3, 3 };
	uint16_t batch_norm_16_tmp_4_dim[] = { 1, 24, 14, 14 };
	uint16_t stage_3_1_conv5_bn_offset_dim[] = { 24 };
	uint16_t stage_3_1_conv5_weights_dim[] = { 24, 24, 1, 1 };
	uint16_t batch_norm_19_tmp_4_dim[] = { 1, 24, 14, 14 };
	uint16_t stage_3_1_conv3_bn_offset_dim[] = { 24 };
	uint16_t stage_3_1_conv3_weights_dim[] = { 24, 24, 1, 1 };
	uint16_t concat_4_tmp_0_dim[] = { 1, 48, 14, 14 };
	uint16_t shape_10_tmp_0_dim[] = { 4 };
	uint16_t shape_10_tmp_0_slice_0_dim[] = { 1 };
	uint16_t reshape2_8_tmp_0_dim[] = { 1, 2, 24, 14, 14 };
	uint16_t reshape2_8_tmp_1_dim[] = { 0, 1, 48, 14, 14 };
	uint16_t transpose_4_tmp_0_dim[] = { 1, 24, 2, 14, 14 };
	uint16_t transpose_4_tmp_1_dim[] = { 0, 1, 2, 24, 14, 14 };
	uint16_t reshape2_9_tmp_0_dim[] = { 1, 48, 14, 14 };
	uint16_t reshape2_9_tmp_1_dim[] = { 0, 1, 24, 2, 14, 14 };
	uint16_t split_3_tmp_0_dim[] = { 1, 24, 14, 14 };
	uint16_t split_3_tmp_1_dim[] = { 1, 24, 14, 14 };
	uint16_t batch_norm_20_tmp_4_dim[] = { 1, 24, 14, 14 };
	uint16_t stage_3_2_conv1_bn_offset_dim[] = { 24 };
	uint16_t stage_3_2_conv1_weights_dim[] = { 24, 24, 1, 1 };
	uint16_t batch_norm_21_tmp_3_dim[] = { 1, 24, 14, 14 };
	uint16_t stage_3_2_conv2_bn_offset_dim[] = { 24 };
	uint16_t stage_3_2_conv2_weights_dim[] = { 24, 1, 3, 3 };
	uint16_t batch_norm_22_tmp_4_dim[] = { 1, 24, 14, 14 };
	uint16_t stage_3_2_conv3_bn_offset_dim[] = { 24 };
	uint16_t stage_3_2_conv3_weights_dim[] = { 24, 24, 1, 1 };
	uint16_t concat_5_tmp_0_dim[] = { 1, 48, 14, 14 };
	uint16_t shape_13_tmp_0_dim[] = { 4 };
	uint16_t shape_13_tmp_0_slice_0_dim[] = { 1 };
	uint16_t reshape2_10_tmp_0_dim[] = { 1, 2, 24, 14, 14 };
	uint16_t reshape2_10_tmp_1_dim[] = { 0, 1, 48, 14, 14 };
	uint16_t transpose_5_tmp_0_dim[] = { 1, 24, 2, 14, 14 };
	uint16_t transpose_5_tmp_1_dim[] = { 0, 1, 2, 24, 14, 14 };
	uint16_t reshape2_11_tmp_0_dim[] = { 1, 48, 14, 14 };
	uint16_t reshape2_11_tmp_1_dim[] = { 0, 1, 24, 2, 14, 14 };
	uint16_t split_4_tmp_0_dim[] = { 1, 24, 14, 14 };
	uint16_t split_4_tmp_1_dim[] = { 1, 24, 14, 14 };
	uint16_t batch_norm_23_tmp_4_dim[] = { 1, 24, 14, 14 };
	uint16_t stage_3_3_conv1_bn_offset_dim[] = { 24 };
	uint16_t stage_3_3_conv1_weights_dim[] = { 24, 24, 1, 1 };
	uint16_t batch_norm_24_tmp_3_dim[] = { 1, 24, 14, 14 };
	uint16_t stage_3_3_conv2_bn_offset_dim[] = { 24 };
	uint16_t stage_3_3_conv2_weights_dim[] = { 24, 1, 3, 3 };
	uint16_t batch_norm_25_tmp_4_dim[] = { 1, 24, 14, 14 };
	uint16_t stage_3_3_conv3_bn_offset_dim[] = { 24 };
	uint16_t stage_3_3_conv3_weights_dim[] = { 24, 24, 1, 1 };
	uint16_t concat_6_tmp_0_dim[] = { 1, 48, 14, 14 };
	uint16_t shape_16_tmp_0_dim[] = { 4 };
	uint16_t shape_16_tmp_0_slice_0_dim[] = { 1 };
	uint16_t reshape2_12_tmp_0_dim[] = { 1, 2, 24, 14, 14 };
	uint16_t reshape2_12_tmp_1_dim[] = { 0, 1, 48, 14, 14 };
	uint16_t transpose_6_tmp_0_dim[] = { 1, 24, 2, 14, 14 };
	uint16_t transpose_6_tmp_1_dim[] = { 0, 1, 2, 24, 14, 14 };
	uint16_t reshape2_13_tmp_0_dim[] = { 1, 48, 14, 14 };
	uint16_t reshape2_13_tmp_1_dim[] = { 0, 1, 24, 2, 14, 14 };
	uint16_t split_5_tmp_0_dim[] = { 1, 24, 14, 14 };
	uint16_t split_5_tmp_1_dim[] = { 1, 24, 14, 14 };
	uint16_t batch_norm_26_tmp_4_dim[] = { 1, 24, 14, 14 };
	uint16_t stage_3_4_conv1_bn_offset_dim[] = { 24 };
	uint16_t stage_3_4_conv1_weights_dim[] = { 24, 24, 1, 1 };
	uint16_t batch_norm_27_tmp_3_dim[] = { 1, 24, 14, 14 };
	uint16_t stage_3_4_conv2_bn_offset_dim[] = { 24 };
	uint16_t stage_3_4_conv2_weights_dim[] = { 24, 1, 3, 3 };
	uint16_t batch_norm_28_tmp_4_dim[] = { 1, 24, 14, 14 };
	uint16_t stage_3_4_conv3_bn_offset_dim[] = { 24 };
	uint16_t stage_3_4_conv3_weights_dim[] = { 24, 24, 1, 1 };
	uint16_t concat_7_tmp_0_dim[] = { 1, 48, 14, 14 };
	uint16_t shape_19_tmp_0_dim[] = { 4 };
	uint16_t shape_19_tmp_0_slice_0_dim[] = { 1 };
	uint16_t reshape2_14_tmp_0_dim[] = { 1, 2, 24, 14, 14 };
	uint16_t reshape2_14_tmp_1_dim[] = { 0, 1, 48, 14, 14 };
	uint16_t transpose_7_tmp_0_dim[] = { 1, 24, 2, 14, 14 };
	uint16_t transpose_7_tmp_1_dim[] = { 0, 1, 2, 24, 14, 14 };
	uint16_t reshape2_15_tmp_0_dim[] = { 1, 48, 14, 14 };
	uint16_t reshape2_15_tmp_1_dim[] = { 0, 1, 24, 2, 14, 14 };
	uint16_t split_6_tmp_0_dim[] = { 1, 24, 14, 14 };
	uint16_t split_6_tmp_1_dim[] = { 1, 24, 14, 14 };
	uint16_t batch_norm_29_tmp_4_dim[] = { 1, 24, 14, 14 };
	uint16_t stage_3_5_conv1_bn_offset_dim[] = { 24 };
	uint16_t stage_3_5_conv1_weights_dim[] = { 24, 24, 1, 1 };
	uint16_t batch_norm_30_tmp_3_dim[] = { 1, 24, 14, 14 };
	uint16_t stage_3_5_conv2_bn_offset_dim[] = { 24 };
	uint16_t stage_3_5_conv2_weights_dim[] = { 24, 1, 3, 3 };
	uint16_t batch_norm_31_tmp_4_dim[] = { 1, 24, 14, 14 };
	uint16_t stage_3_5_conv3_bn_offset_dim[] = { 24 };
	uint16_t stage_3_5_conv3_weights_dim[] = { 24, 24, 1, 1 };
	uint16_t concat_8_tmp_0_dim[] = { 1, 48, 14, 14 };
	uint16_t shape_22_tmp_0_dim[] = { 4 };
	uint16_t shape_22_tmp_0_slice_0_dim[] = { 1 };
	uint16_t reshape2_16_tmp_0_dim[] = { 1, 2, 24, 14, 14 };
	uint16_t reshape2_16_tmp_1_dim[] = { 0, 1, 48, 14, 14 };
	uint16_t transpose_8_tmp_0_dim[] = { 1, 24, 2, 14, 14 };
	uint16_t transpose_8_tmp_1_dim[] = { 0, 1, 2, 24, 14, 14 };
	uint16_t reshape2_17_tmp_0_dim[] = { 1, 48, 14, 14 };
	uint16_t reshape2_17_tmp_1_dim[] = { 0, 1, 24, 2, 14, 14 };
	uint16_t split_7_tmp_0_dim[] = { 1, 24, 14, 14 };
	uint16_t split_7_tmp_1_dim[] = { 1, 24, 14, 14 };
	uint16_t batch_norm_32_tmp_4_dim[] = { 1, 24, 14, 14 };
	uint16_t stage_3_6_conv1_bn_offset_dim[] = { 24 };
	uint16_t stage_3_6_conv1_weights_dim[] = { 24, 24, 1, 1 };
	uint16_t batch_norm_33_tmp_3_dim[] = { 1, 24, 14, 14 };
	uint16_t stage_3_6_conv2_bn_offset_dim[] = { 24 };
	uint16_t stage_3_6_conv2_weights_dim[] = { 24, 1, 3, 3 };
	uint16_t batch_norm_34_tmp_4_dim[] = { 1, 24, 14, 14 };
	uint16_t stage_3_6_conv3_bn_offset_dim[] = { 24 };
	uint16_t stage_3_6_conv3_weights_dim[] = { 24, 24, 1, 1 };
	uint16_t concat_9_tmp_0_dim[] = { 1, 48, 14, 14 };
	uint16_t shape_25_tmp_0_dim[] = { 4 };
	uint16_t shape_25_tmp_0_slice_0_dim[] = { 1 };
	uint16_t reshape2_18_tmp_0_dim[] = { 1, 2, 24, 14, 14 };
	uint16_t reshape2_18_tmp_1_dim[] = { 0, 1, 48, 14, 14 };
	uint16_t transpose_9_tmp_0_dim[] = { 1, 24, 2, 14, 14 };
	uint16_t transpose_9_tmp_1_dim[] = { 0, 1, 2, 24, 14, 14 };
	uint16_t reshape2_19_tmp_0_dim[] = { 1, 48, 14, 14 };
	uint16_t reshape2_19_tmp_1_dim[] = { 0, 1, 24, 2, 14, 14 };
	uint16_t split_8_tmp_0_dim[] = { 1, 24, 14, 14 };
	uint16_t split_8_tmp_1_dim[] = { 1, 24, 14, 14 };
	uint16_t batch_norm_35_tmp_4_dim[] = { 1, 24, 14, 14 };
	uint16_t stage_3_7_conv1_bn_offset_dim[] = { 24 };
	uint16_t stage_3_7_conv1_weights_dim[] = { 24, 24, 1, 1 };
	uint16_t batch_norm_36_tmp_3_dim[] = { 1, 24, 14, 14 };
	uint16_t stage_3_7_conv2_bn_offset_dim[] = { 24 };
	uint16_t stage_3_7_conv2_weights_dim[] = { 24, 1, 3, 3 };
	uint16_t batch_norm_37_tmp_4_dim[] = { 1, 24, 14, 14 };
	uint16_t stage_3_7_conv3_bn_offset_dim[] = { 24 };
	uint16_t stage_3_7_conv3_weights_dim[] = { 24, 24, 1, 1 };
	uint16_t concat_10_tmp_0_dim[] = { 1, 48, 14, 14 };
	uint16_t shape_28_tmp_0_dim[] = { 4 };
	uint16_t shape_28_tmp_0_slice_0_dim[] = { 1 };
	uint16_t reshape2_20_tmp_0_dim[] = { 1, 2, 24, 14, 14 };
	uint16_t reshape2_20_tmp_1_dim[] = { 0, 1, 48, 14, 14 };
	uint16_t transpose_10_tmp_0_dim[] = { 1, 24, 2, 14, 14 };
	uint16_t transpose_10_tmp_1_dim[] = { 0, 1, 2, 24, 14, 14 };
	uint16_t reshape2_21_tmp_0_dim[] = { 1, 48, 14, 14 };
	uint16_t reshape2_21_tmp_1_dim[] = { 0, 1, 24, 2, 14, 14 };
	uint16_t split_9_tmp_0_dim[] = { 1, 24, 14, 14 };
	uint16_t split_9_tmp_1_dim[] = { 1, 24, 14, 14 };
	uint16_t batch_norm_38_tmp_4_dim[] = { 1, 24, 14, 14 };
	uint16_t stage_3_8_conv1_bn_offset_dim[] = { 24 };
	uint16_t stage_3_8_conv1_weights_dim[] = { 24, 24, 1, 1 };
	uint16_t batch_norm_39_tmp_3_dim[] = { 1, 24, 14, 14 };
	uint16_t stage_3_8_conv2_bn_offset_dim[] = { 24 };
	uint16_t stage_3_8_conv2_weights_dim[] = { 24, 1, 3, 3 };
	uint16_t batch_norm_40_tmp_4_dim[] = { 1, 24, 14, 14 };
	uint16_t stage_3_8_conv3_bn_offset_dim[] = { 24 };
	uint16_t stage_3_8_conv3_weights_dim[] = { 24, 24, 1, 1 };
	uint16_t concat_11_tmp_0_dim[] = { 1, 48, 14, 14 };
	uint16_t shape_31_tmp_0_dim[] = { 4 };
	uint16_t shape_31_tmp_0_slice_0_dim[] = { 1 };
	uint16_t reshape2_22_tmp_0_dim[] = { 1, 2, 24, 14, 14 };
	uint16_t reshape2_22_tmp_1_dim[] = { 0, 1, 48, 14, 14 };
	uint16_t transpose_11_tmp_0_dim[] = { 1, 24, 2, 14, 14 };
	uint16_t transpose_11_tmp_1_dim[] = { 0, 1, 2, 24, 14, 14 };
	uint16_t reshape2_23_tmp_0_dim[] = { 1, 48, 14, 14 };
	uint16_t reshape2_23_tmp_1_dim[] = { 0, 1, 24, 2, 14, 14 };
	uint16_t batch_norm_41_tmp_3_dim[] = { 1, 48, 7, 7 };
	uint16_t stage_4_1_conv4_bn_offset_dim[] = { 48 };
	uint16_t stage_4_1_conv4_weights_dim[] = { 48, 1, 3, 3 };
	uint16_t batch_norm_43_tmp_4_dim[] = { 1, 48, 14, 14 };
	uint16_t stage_4_1_conv1_bn_offset_dim[] = { 48 };
	uint16_t stage_4_1_conv1_weights_dim[] = { 48, 48, 1, 1 };
	uint16_t batch_norm_44_tmp_3_dim[] = { 1, 48, 7, 7 };
	uint16_t stage_4_1_conv2_bn_offset_dim[] = { 48 };
	uint16_t stage_4_1_conv2_weights_dim[] = { 48, 1, 3, 3 };
	uint16_t batch_norm_42_tmp_4_dim[] = { 1, 48, 7, 7 };
	uint16_t stage_4_1_conv5_bn_offset_dim[] = { 48 };
	uint16_t stage_4_1_conv5_weights_dim[] = { 48, 48, 1, 1 };
	uint16_t batch_norm_45_tmp_4_dim[] = { 1, 48, 7, 7 };
	uint16_t stage_4_1_conv3_bn_offset_dim[] = { 48 };
	uint16_t stage_4_1_conv3_weights_dim[] = { 48, 48, 1, 1 };
	uint16_t concat_12_tmp_0_dim[] = { 1, 96, 7, 7 };
	uint16_t shape_32_tmp_0_dim[] = { 4 };
	uint16_t shape_32_tmp_0_slice_0_dim[] = { 1 };
	uint16_t reshape2_24_tmp_0_dim[] = { 1, 2, 48, 7, 7 };
	uint16_t reshape2_24_tmp_1_dim[] = { 0, 1, 96, 7, 7 };
	uint16_t transpose_12_tmp_0_dim[] = { 1, 48, 2, 7, 7 };
	uint16_t transpose_12_tmp_1_dim[] = { 0, 1, 2, 48, 7, 7 };
	uint16_t reshape2_25_tmp_0_dim[] = { 1, 96, 7, 7 };
	uint16_t reshape2_25_tmp_1_dim[] = { 0, 1, 48, 2, 7, 7 };
	uint16_t split_10_tmp_0_dim[] = { 1, 48, 7, 7 };
	uint16_t split_10_tmp_1_dim[] = { 1, 48, 7, 7 };
	uint16_t batch_norm_46_tmp_4_dim[] = { 1, 48, 7, 7 };
	uint16_t stage_4_2_conv1_bn_offset_dim[] = { 48 };
	uint16_t stage_4_2_conv1_weights_dim[] = { 48, 48, 1, 1 };
	uint16_t batch_norm_47_tmp_3_dim[] = { 1, 48, 7, 7 };
	uint16_t stage_4_2_conv2_bn_offset_dim[] = { 48 };
	uint16_t stage_4_2_conv2_weights_dim[] = { 48, 1, 3, 3 };
	uint16_t batch_norm_48_tmp_4_dim[] = { 1, 48, 7, 7 };
	uint16_t stage_4_2_conv3_bn_offset_dim[] = { 48 };
	uint16_t stage_4_2_conv3_weights_dim[] = { 48, 48, 1, 1 };
	uint16_t concat_13_tmp_0_dim[] = { 1, 96, 7, 7 };
	uint16_t shape_35_tmp_0_dim[] = { 4 };
	uint16_t shape_35_tmp_0_slice_0_dim[] = { 1 };
	uint16_t reshape2_26_tmp_0_dim[] = { 1, 2, 48, 7, 7 };
	uint16_t reshape2_26_tmp_1_dim[] = { 0, 1, 96, 7, 7 };
	uint16_t transpose_13_tmp_0_dim[] = { 1, 48, 2, 7, 7 };
	uint16_t transpose_13_tmp_1_dim[] = { 0, 1, 2, 48, 7, 7 };
	uint16_t reshape2_27_tmp_0_dim[] = { 1, 96, 7, 7 };
	uint16_t reshape2_27_tmp_1_dim[] = { 0, 1, 48, 2, 7, 7 };
	uint16_t split_11_tmp_0_dim[] = { 1, 48, 7, 7 };
	uint16_t split_11_tmp_1_dim[] = { 1, 48, 7, 7 };
	uint16_t batch_norm_49_tmp_4_dim[] = { 1, 48, 7, 7 };
	uint16_t stage_4_3_conv1_bn_offset_dim[] = { 48 };
	uint16_t stage_4_3_conv1_weights_dim[] = { 48, 48, 1, 1 };
	uint16_t batch_norm_50_tmp_3_dim[] = { 1, 48, 7, 7 };
	uint16_t stage_4_3_conv2_bn_offset_dim[] = { 48 };
	uint16_t stage_4_3_conv2_weights_dim[] = { 48, 1, 3, 3 };
	uint16_t batch_norm_51_tmp_4_dim[] = { 1, 48, 7, 7 };
	uint16_t stage_4_3_conv3_bn_offset_dim[] = { 48 };
	uint16_t stage_4_3_conv3_weights_dim[] = { 48, 48, 1, 1 };
	uint16_t concat_14_tmp_0_dim[] = { 1, 96, 7, 7 };
	uint16_t shape_38_tmp_0_dim[] = { 4 };
	uint16_t shape_38_tmp_0_slice_0_dim[] = { 1 };
	uint16_t reshape2_28_tmp_0_dim[] = { 1, 2, 48, 7, 7 };
	uint16_t reshape2_28_tmp_1_dim[] = { 0, 1, 96, 7, 7 };
	uint16_t transpose_14_tmp_0_dim[] = { 1, 48, 2, 7, 7 };
	uint16_t transpose_14_tmp_1_dim[] = { 0, 1, 2, 48, 7, 7 };
	uint16_t reshape2_29_tmp_0_dim[] = { 1, 96, 7, 7 };
	uint16_t reshape2_29_tmp_1_dim[] = { 0, 1, 48, 2, 7, 7 };
	uint16_t split_12_tmp_0_dim[] = { 1, 48, 7, 7 };
	uint16_t split_12_tmp_1_dim[] = { 1, 48, 7, 7 };
	uint16_t batch_norm_52_tmp_4_dim[] = { 1, 48, 7, 7 };
	uint16_t stage_4_4_conv1_bn_offset_dim[] = { 48 };
	uint16_t stage_4_4_conv1_weights_dim[] = { 48, 48, 1, 1 };
	uint16_t batch_norm_53_tmp_3_dim[] = { 1, 48, 7, 7 };
	uint16_t stage_4_4_conv2_bn_offset_dim[] = { 48 };
	uint16_t stage_4_4_conv2_weights_dim[] = { 48, 1, 3, 3 };
	uint16_t batch_norm_54_tmp_4_dim[] = { 1, 48, 7, 7 };
	uint16_t stage_4_4_conv3_bn_offset_dim[] = { 48 };
	uint16_t stage_4_4_conv3_weights_dim[] = { 48, 48, 1, 1 };
	uint16_t concat_15_tmp_0_dim[] = { 1, 96, 7, 7 };
	uint16_t shape_41_tmp_0_dim[] = { 4 };
	uint16_t shape_41_tmp_0_slice_0_dim[] = { 1 };
	uint16_t reshape2_30_tmp_0_dim[] = { 1, 2, 48, 7, 7 };
	uint16_t reshape2_30_tmp_1_dim[] = { 0, 1, 96, 7, 7 };
	uint16_t transpose_15_tmp_0_dim[] = { 1, 48, 2, 7, 7 };
	uint16_t transpose_15_tmp_1_dim[] = { 0, 1, 2, 48, 7, 7 };
	uint16_t reshape2_31_tmp_0_dim[] = { 1, 96, 7, 7 };
	uint16_t reshape2_31_tmp_1_dim[] = { 0, 1, 48, 2, 7, 7 };
	uint16_t batch_norm_55_tmp_4_dim[] = { 1, 512, 7, 7 };
	uint16_t conv5_bn_offset_dim[] = { 512 };
	uint16_t conv5_weights_dim[] = { 512, 96, 1, 1 };
	uint16_t pool2d_1_tmp_0_dim[] = { 1, 512, 1, 1 };
	uint16_t flatten_0_tmp_0_dim[] = { 1, 512 };
	uint16_t flatten_0_tmp_1_dim[] = { 0, 1, 512, 1, 1 };
	uint16_t fc6_weights_dim[] = { 512, 1000 };
	uint16_t linear_1_tmp_0_dim[] = { 1, 1000 };
	uint16_t fc6_offset_dim[] = { 1000 };
	uint16_t linear_1_tmp_1_dim[] = { 1, 1000 };
	uint16_t softmax_1_tmp_0_dim[] = { 1, 1000 };

	float* batch_norm_0_tmp_4 = (float*)calloc(301056, sizeof(float));
	autox_conv2d(x, (float*)((int8_t*)weights), (float*)((int8_t*)weights) + 24, batch_norm_0_tmp_4, x_dim, stage1_conv_weights_dim, batch_norm_0_tmp_4_dim, 1, 1, 2, 1, 1);
	free(x);

	float* pool2d_0_tmp_0 = (float*)calloc(75264, sizeof(float));
	autox_pool2d(batch_norm_0_tmp_4, pool2d_0_tmp_0, batch_norm_0_tmp_4_dim, pool2d_0_tmp_0_dim, 3, 2, 1, 0, 0);
	free(batch_norm_0_tmp_4);

	float* batch_norm_1_tmp_3 = (float*)calloc(18816, sizeof(float));
	autox_conv2d(pool2d_0_tmp_0, (float*)((int8_t*)weights) + 672, (float*)((int8_t*)weights) + 696, batch_norm_1_tmp_3, pool2d_0_tmp_0_dim, stage_2_1_conv4_weights_dim, batch_norm_1_tmp_3_dim, 24, 1, 2, 1, 0);
	// free(pool2d_0_tmp_0);

	float* batch_norm_3_tmp_4 = (float*)calloc(37632, sizeof(float));
	autox_conv2d(pool2d_0_tmp_0, (float*)((int8_t*)weights) + 912, (float*)((int8_t*)weights) + 924, batch_norm_3_tmp_4, pool2d_0_tmp_0_dim, stage_2_1_conv1_weights_dim, batch_norm_3_tmp_4_dim, 1, 0, 1, 1, 1);
	free(pool2d_0_tmp_0);

	float* batch_norm_4_tmp_3 = (float*)calloc(9408, sizeof(float));
	autox_conv2d(batch_norm_3_tmp_4, (float*)((int8_t*)weights) + 1212, (float*)((int8_t*)weights) + 1224, batch_norm_4_tmp_3, batch_norm_3_tmp_4_dim, stage_2_1_conv2_weights_dim, batch_norm_4_tmp_3_dim, 12, 1, 2, 1, 0);
	free(batch_norm_3_tmp_4);

	float* batch_norm_2_tmp_4 = (float*)calloc(9408, sizeof(float));
	autox_conv2d(batch_norm_1_tmp_3, (float*)((int8_t*)weights) + 1332, (float*)((int8_t*)weights) + 1344, batch_norm_2_tmp_4, batch_norm_1_tmp_3_dim, stage_2_1_conv5_weights_dim, batch_norm_2_tmp_4_dim, 1, 0, 1, 1, 1);
	free(batch_norm_1_tmp_3);

	float* batch_norm_5_tmp_4 = (float*)calloc(9408, sizeof(float));
	autox_conv2d(batch_norm_4_tmp_3, (float*)((int8_t*)weights) + 1632, (float*)((int8_t*)weights) + 1644, batch_norm_5_tmp_4, batch_norm_4_tmp_3_dim, stage_2_1_conv3_weights_dim, batch_norm_5_tmp_4_dim, 1, 0, 1, 1, 1);
	free(batch_norm_4_tmp_3);

	float* p_7[] = { batch_norm_2_tmp_4, batch_norm_5_tmp_4, };
	uint16_t* p_7_dim[] = { batch_norm_2_tmp_4_dim, batch_norm_5_tmp_4_dim, };
	float* concat_0_tmp_0 = (float*)calloc(18816, sizeof(float));
	autox_concat(p_7, concat_0_tmp_0, p_7_dim, concat_0_tmp_0_dim, 1, 2, 4);
	free(batch_norm_2_tmp_4);
	free(batch_norm_5_tmp_4);

	float* transpose_0_tmp_0 = (float*)calloc(18816, sizeof(float));
	uint16_t axis_8[] = { 0, 2, 1, 3, 4 };
	autox_transpose(concat_0_tmp_0, transpose_0_tmp_0, reshape2_0_tmp_0_dim, transpose_0_tmp_0_dim, axis_8, 5);
	free(concat_0_tmp_0);

	float* split_0_tmp_0 = (float*)calloc(9408, sizeof(float));
	float* split_0_tmp_1 = (float*)calloc(9408, sizeof(float));
	float* p_9[] = { split_0_tmp_0, split_0_tmp_1, };
	uint16_t* p_9_dim[] = { split_0_tmp_0_dim, split_0_tmp_1_dim, };
	autox_split(transpose_0_tmp_0, p_9, reshape2_1_tmp_0_dim, p_9_dim, 1, 4, 2, 4);
	free(transpose_0_tmp_0);

	float* batch_norm_6_tmp_4 = (float*)calloc(9408, sizeof(float));
	autox_conv2d(split_0_tmp_1, (float*)((int8_t*)weights) + 1788, (float*)((int8_t*)weights) + 1800, batch_norm_6_tmp_4, split_0_tmp_1_dim, stage_2_2_conv1_weights_dim, batch_norm_6_tmp_4_dim, 1, 0, 1, 1, 1);
	free(split_0_tmp_1);

	float* batch_norm_7_tmp_3 = (float*)calloc(9408, sizeof(float));
	autox_conv2d(batch_norm_6_tmp_4, (float*)((int8_t*)weights) + 1944, (float*)((int8_t*)weights) + 1956, batch_norm_7_tmp_3, batch_norm_6_tmp_4_dim, stage_2_2_conv2_weights_dim, batch_norm_7_tmp_3_dim, 12, 1, 1, 1, 0);
	free(batch_norm_6_tmp_4);

	float* batch_norm_8_tmp_4 = (float*)calloc(9408, sizeof(float));
	autox_conv2d(batch_norm_7_tmp_3, (float*)((int8_t*)weights) + 2064, (float*)((int8_t*)weights) + 2076, batch_norm_8_tmp_4, batch_norm_7_tmp_3_dim, stage_2_2_conv3_weights_dim, batch_norm_8_tmp_4_dim, 1, 0, 1, 1, 1);
	free(batch_norm_7_tmp_3);

	float* p_13[] = { split_0_tmp_0, batch_norm_8_tmp_4, };
	uint16_t* p_13_dim[] = { split_0_tmp_0_dim, batch_norm_8_tmp_4_dim, };
	float* concat_1_tmp_0 = (float*)calloc(18816, sizeof(float));
	autox_concat(p_13, concat_1_tmp_0, p_13_dim, concat_1_tmp_0_dim, 1, 2, 4);
	free(split_0_tmp_0);
	free(batch_norm_8_tmp_4);

	float* transpose_1_tmp_0 = (float*)calloc(18816, sizeof(float));
	uint16_t axis_14[] = { 0, 2, 1, 3, 4 };
	autox_transpose(concat_1_tmp_0, transpose_1_tmp_0, reshape2_2_tmp_0_dim, transpose_1_tmp_0_dim, axis_14, 5);
	free(concat_1_tmp_0);

	float* split_1_tmp_0 = (float*)calloc(9408, sizeof(float));
	float* split_1_tmp_1 = (float*)calloc(9408, sizeof(float));
	float* p_15[] = { split_1_tmp_0, split_1_tmp_1, };
	uint16_t* p_15_dim[] = { split_1_tmp_0_dim, split_1_tmp_1_dim, };
	autox_split(transpose_1_tmp_0, p_15, reshape2_3_tmp_0_dim, p_15_dim, 1, 4, 2, 4);
	free(transpose_1_tmp_0);

	float* batch_norm_9_tmp_4 = (float*)calloc(9408, sizeof(float));
	autox_conv2d(split_1_tmp_1, (float*)((int8_t*)weights) + 2220, (float*)((int8_t*)weights) + 2232, batch_norm_9_tmp_4, split_1_tmp_1_dim, stage_2_3_conv1_weights_dim, batch_norm_9_tmp_4_dim, 1, 0, 1, 1, 1);
	free(split_1_tmp_1);

	float* batch_norm_10_tmp_3 = (float*)calloc(9408, sizeof(float));
	autox_conv2d(batch_norm_9_tmp_4, (float*)((int8_t*)weights) + 2376, (float*)((int8_t*)weights) + 2388, batch_norm_10_tmp_3, batch_norm_9_tmp_4_dim, stage_2_3_conv2_weights_dim, batch_norm_10_tmp_3_dim, 12, 1, 1, 1, 0);
	free(batch_norm_9_tmp_4);

	float* batch_norm_11_tmp_4 = (float*)calloc(9408, sizeof(float));
	autox_conv2d(batch_norm_10_tmp_3, (float*)((int8_t*)weights) + 2496, (float*)((int8_t*)weights) + 2508, batch_norm_11_tmp_4, batch_norm_10_tmp_3_dim, stage_2_3_conv3_weights_dim, batch_norm_11_tmp_4_dim, 1, 0, 1, 1, 1);
	free(batch_norm_10_tmp_3);

	float* p_19[] = { split_1_tmp_0, batch_norm_11_tmp_4, };
	uint16_t* p_19_dim[] = { split_1_tmp_0_dim, batch_norm_11_tmp_4_dim, };
	float* concat_2_tmp_0 = (float*)calloc(18816, sizeof(float));
	autox_concat(p_19, concat_2_tmp_0, p_19_dim, concat_2_tmp_0_dim, 1, 2, 4);
	free(split_1_tmp_0);
	free(batch_norm_11_tmp_4);

	float* transpose_2_tmp_0 = (float*)calloc(18816, sizeof(float));
	uint16_t axis_20[] = { 0, 2, 1, 3, 4 };
	autox_transpose(concat_2_tmp_0, transpose_2_tmp_0, reshape2_4_tmp_0_dim, transpose_2_tmp_0_dim, axis_20, 5);
	free(concat_2_tmp_0);

	float* split_2_tmp_0 = (float*)calloc(9408, sizeof(float));
	float* split_2_tmp_1 = (float*)calloc(9408, sizeof(float));
	float* p_21[] = { split_2_tmp_0, split_2_tmp_1, };
	uint16_t* p_21_dim[] = { split_2_tmp_0_dim, split_2_tmp_1_dim, };
	autox_split(transpose_2_tmp_0, p_21, reshape2_5_tmp_0_dim, p_21_dim, 1, 4, 2, 4);
	free(transpose_2_tmp_0);

	float* batch_norm_12_tmp_4 = (float*)calloc(9408, sizeof(float));
	autox_conv2d(split_2_tmp_1, (float*)((int8_t*)weights) + 2652, (float*)((int8_t*)weights) + 2664, batch_norm_12_tmp_4, split_2_tmp_1_dim, stage_2_4_conv1_weights_dim, batch_norm_12_tmp_4_dim, 1, 0, 1, 1, 1);
	free(split_2_tmp_1);

	float* batch_norm_13_tmp_3 = (float*)calloc(9408, sizeof(float));
	autox_conv2d(batch_norm_12_tmp_4, (float*)((int8_t*)weights) + 2808, (float*)((int8_t*)weights) + 2820, batch_norm_13_tmp_3, batch_norm_12_tmp_4_dim, stage_2_4_conv2_weights_dim, batch_norm_13_tmp_3_dim, 12, 1, 1, 1, 0);
	free(batch_norm_12_tmp_4);

	float* batch_norm_14_tmp_4 = (float*)calloc(9408, sizeof(float));
	autox_conv2d(batch_norm_13_tmp_3, (float*)((int8_t*)weights) + 2928, (float*)((int8_t*)weights) + 2940, batch_norm_14_tmp_4, batch_norm_13_tmp_3_dim, stage_2_4_conv3_weights_dim, batch_norm_14_tmp_4_dim, 1, 0, 1, 1, 1);
	free(batch_norm_13_tmp_3);

	float* p_25[] = { split_2_tmp_0, batch_norm_14_tmp_4, };
	uint16_t* p_25_dim[] = { split_2_tmp_0_dim, batch_norm_14_tmp_4_dim, };
	float* concat_3_tmp_0 = (float*)calloc(18816, sizeof(float));
	autox_concat(p_25, concat_3_tmp_0, p_25_dim, concat_3_tmp_0_dim, 1, 2, 4);
	free(split_2_tmp_0);
	free(batch_norm_14_tmp_4);

	float* transpose_3_tmp_0 = (float*)calloc(18816, sizeof(float));
	uint16_t axis_26[] = { 0, 2, 1, 3, 4 };
	autox_transpose(concat_3_tmp_0, transpose_3_tmp_0, reshape2_6_tmp_0_dim, transpose_3_tmp_0_dim, axis_26, 5);
	free(concat_3_tmp_0);

	float* batch_norm_15_tmp_3 = (float*)calloc(4704, sizeof(float));
	autox_conv2d(transpose_3_tmp_0, (float*)((int8_t*)weights) + 3084, (float*)((int8_t*)weights) + 3108, batch_norm_15_tmp_3, reshape2_7_tmp_0_dim, stage_3_1_conv4_weights_dim, batch_norm_15_tmp_3_dim, 24, 1, 2, 1, 0);
	// free(transpose_3_tmp_0);

	float* batch_norm_17_tmp_4 = (float*)calloc(18816, sizeof(float));
	autox_conv2d(transpose_3_tmp_0, (float*)((int8_t*)weights) + 3324, (float*)((int8_t*)weights) + 3348, batch_norm_17_tmp_4, reshape2_7_tmp_0_dim, stage_3_1_conv1_weights_dim, batch_norm_17_tmp_4_dim, 1, 0, 1, 1, 1);
	free(transpose_3_tmp_0);

	float* batch_norm_18_tmp_3 = (float*)calloc(4704, sizeof(float));
	autox_conv2d(batch_norm_17_tmp_4, (float*)((int8_t*)weights) + 3924, (float*)((int8_t*)weights) + 3948, batch_norm_18_tmp_3, batch_norm_17_tmp_4_dim, stage_3_1_conv2_weights_dim, batch_norm_18_tmp_3_dim, 24, 1, 2, 1, 0);
	free(batch_norm_17_tmp_4);

	float* batch_norm_16_tmp_4 = (float*)calloc(4704, sizeof(float));
	autox_conv2d(batch_norm_15_tmp_3, (float*)((int8_t*)weights) + 4164, (float*)((int8_t*)weights) + 4188, batch_norm_16_tmp_4, batch_norm_15_tmp_3_dim, stage_3_1_conv5_weights_dim, batch_norm_16_tmp_4_dim, 1, 0, 1, 1, 1);
	free(batch_norm_15_tmp_3);

	float* batch_norm_19_tmp_4 = (float*)calloc(4704, sizeof(float));
	autox_conv2d(batch_norm_18_tmp_3, (float*)((int8_t*)weights) + 4764, (float*)((int8_t*)weights) + 4788, batch_norm_19_tmp_4, batch_norm_18_tmp_3_dim, stage_3_1_conv3_weights_dim, batch_norm_19_tmp_4_dim, 1, 0, 1, 1, 1);
	free(batch_norm_18_tmp_3);

	float* p_32[] = { batch_norm_16_tmp_4, batch_norm_19_tmp_4, };
	uint16_t* p_32_dim[] = { batch_norm_16_tmp_4_dim, batch_norm_19_tmp_4_dim, };
	float* concat_4_tmp_0 = (float*)calloc(9408, sizeof(float));
	autox_concat(p_32, concat_4_tmp_0, p_32_dim, concat_4_tmp_0_dim, 1, 2, 4);
	free(batch_norm_16_tmp_4);
	free(batch_norm_19_tmp_4);

	float* transpose_4_tmp_0 = (float*)calloc(9408, sizeof(float));
	uint16_t axis_33[] = { 0, 2, 1, 3, 4 };
	autox_transpose(concat_4_tmp_0, transpose_4_tmp_0, reshape2_8_tmp_0_dim, transpose_4_tmp_0_dim, axis_33, 5);
	free(concat_4_tmp_0);

	float* split_3_tmp_0 = (float*)calloc(4704, sizeof(float));
	float* split_3_tmp_1 = (float*)calloc(4704, sizeof(float));
	float* p_34[] = { split_3_tmp_0, split_3_tmp_1, };
	uint16_t* p_34_dim[] = { split_3_tmp_0_dim, split_3_tmp_1_dim, };
	autox_split(transpose_4_tmp_0, p_34, reshape2_9_tmp_0_dim, p_34_dim, 1, 4, 2, 4);
	free(transpose_4_tmp_0);

	float* batch_norm_20_tmp_4 = (float*)calloc(4704, sizeof(float));
	autox_conv2d(split_3_tmp_1, (float*)((int8_t*)weights) + 5364, (float*)((int8_t*)weights) + 5388, batch_norm_20_tmp_4, split_3_tmp_1_dim, stage_3_2_conv1_weights_dim, batch_norm_20_tmp_4_dim, 1, 0, 1, 1, 1);
	free(split_3_tmp_1);

	float* batch_norm_21_tmp_3 = (float*)calloc(4704, sizeof(float));
	autox_conv2d(batch_norm_20_tmp_4, (float*)((int8_t*)weights) + 5964, (float*)((int8_t*)weights) + 5988, batch_norm_21_tmp_3, batch_norm_20_tmp_4_dim, stage_3_2_conv2_weights_dim, batch_norm_21_tmp_3_dim, 24, 1, 1, 1, 0);
	free(batch_norm_20_tmp_4);

	float* batch_norm_22_tmp_4 = (float*)calloc(4704, sizeof(float));
	autox_conv2d(batch_norm_21_tmp_3, (float*)((int8_t*)weights) + 6204, (float*)((int8_t*)weights) + 6228, batch_norm_22_tmp_4, batch_norm_21_tmp_3_dim, stage_3_2_conv3_weights_dim, batch_norm_22_tmp_4_dim, 1, 0, 1, 1, 1);
	free(batch_norm_21_tmp_3);

	float* p_38[] = { split_3_tmp_0, batch_norm_22_tmp_4, };
	uint16_t* p_38_dim[] = { split_3_tmp_0_dim, batch_norm_22_tmp_4_dim, };
	float* concat_5_tmp_0 = (float*)calloc(9408, sizeof(float));
	autox_concat(p_38, concat_5_tmp_0, p_38_dim, concat_5_tmp_0_dim, 1, 2, 4);
	free(split_3_tmp_0);
	free(batch_norm_22_tmp_4);

	float* transpose_5_tmp_0 = (float*)calloc(9408, sizeof(float));
	uint16_t axis_39[] = { 0, 2, 1, 3, 4 };
	autox_transpose(concat_5_tmp_0, transpose_5_tmp_0, reshape2_10_tmp_0_dim, transpose_5_tmp_0_dim, axis_39, 5);
	free(concat_5_tmp_0);

	float* split_4_tmp_0 = (float*)calloc(4704, sizeof(float));
	float* split_4_tmp_1 = (float*)calloc(4704, sizeof(float));
	float* p_40[] = { split_4_tmp_0, split_4_tmp_1, };
	uint16_t* p_40_dim[] = { split_4_tmp_0_dim, split_4_tmp_1_dim, };
	autox_split(transpose_5_tmp_0, p_40, reshape2_11_tmp_0_dim, p_40_dim, 1, 4, 2, 4);
	free(transpose_5_tmp_0);

	float* batch_norm_23_tmp_4 = (float*)calloc(4704, sizeof(float));
	autox_conv2d(split_4_tmp_1, (float*)((int8_t*)weights) + 6804, (float*)((int8_t*)weights) + 6828, batch_norm_23_tmp_4, split_4_tmp_1_dim, stage_3_3_conv1_weights_dim, batch_norm_23_tmp_4_dim, 1, 0, 1, 1, 1);
	free(split_4_tmp_1);

	float* batch_norm_24_tmp_3 = (float*)calloc(4704, sizeof(float));
	autox_conv2d(batch_norm_23_tmp_4, (float*)((int8_t*)weights) + 7404, (float*)((int8_t*)weights) + 7428, batch_norm_24_tmp_3, batch_norm_23_tmp_4_dim, stage_3_3_conv2_weights_dim, batch_norm_24_tmp_3_dim, 24, 1, 1, 1, 0);
	free(batch_norm_23_tmp_4);

	float* batch_norm_25_tmp_4 = (float*)calloc(4704, sizeof(float));
	autox_conv2d(batch_norm_24_tmp_3, (float*)((int8_t*)weights) + 7644, (float*)((int8_t*)weights) + 7668, batch_norm_25_tmp_4, batch_norm_24_tmp_3_dim, stage_3_3_conv3_weights_dim, batch_norm_25_tmp_4_dim, 1, 0, 1, 1, 1);
	free(batch_norm_24_tmp_3);

	float* p_44[] = { split_4_tmp_0, batch_norm_25_tmp_4, };
	uint16_t* p_44_dim[] = { split_4_tmp_0_dim, batch_norm_25_tmp_4_dim, };
	float* concat_6_tmp_0 = (float*)calloc(9408, sizeof(float));
	autox_concat(p_44, concat_6_tmp_0, p_44_dim, concat_6_tmp_0_dim, 1, 2, 4);
	free(split_4_tmp_0);
	free(batch_norm_25_tmp_4);

	float* transpose_6_tmp_0 = (float*)calloc(9408, sizeof(float));
	uint16_t axis_45[] = { 0, 2, 1, 3, 4 };
	autox_transpose(concat_6_tmp_0, transpose_6_tmp_0, reshape2_12_tmp_0_dim, transpose_6_tmp_0_dim, axis_45, 5);
	free(concat_6_tmp_0);

	float* split_5_tmp_0 = (float*)calloc(4704, sizeof(float));
	float* split_5_tmp_1 = (float*)calloc(4704, sizeof(float));
	float* p_46[] = { split_5_tmp_0, split_5_tmp_1, };
	uint16_t* p_46_dim[] = { split_5_tmp_0_dim, split_5_tmp_1_dim, };
	autox_split(transpose_6_tmp_0, p_46, reshape2_13_tmp_0_dim, p_46_dim, 1, 4, 2, 4);
	free(transpose_6_tmp_0);

	float* batch_norm_26_tmp_4 = (float*)calloc(4704, sizeof(float));
	autox_conv2d(split_5_tmp_1, (float*)((int8_t*)weights) + 8244, (float*)((int8_t*)weights) + 8268, batch_norm_26_tmp_4, split_5_tmp_1_dim, stage_3_4_conv1_weights_dim, batch_norm_26_tmp_4_dim, 1, 0, 1, 1, 1);
	free(split_5_tmp_1);

	float* batch_norm_27_tmp_3 = (float*)calloc(4704, sizeof(float));
	autox_conv2d(batch_norm_26_tmp_4, (float*)((int8_t*)weights) + 8844, (float*)((int8_t*)weights) + 8868, batch_norm_27_tmp_3, batch_norm_26_tmp_4_dim, stage_3_4_conv2_weights_dim, batch_norm_27_tmp_3_dim, 24, 1, 1, 1, 0);
	free(batch_norm_26_tmp_4);

	float* batch_norm_28_tmp_4 = (float*)calloc(4704, sizeof(float));
	autox_conv2d(batch_norm_27_tmp_3, (float*)((int8_t*)weights) + 9084, (float*)((int8_t*)weights) + 9108, batch_norm_28_tmp_4, batch_norm_27_tmp_3_dim, stage_3_4_conv3_weights_dim, batch_norm_28_tmp_4_dim, 1, 0, 1, 1, 1);
	free(batch_norm_27_tmp_3);

	float* p_50[] = { split_5_tmp_0, batch_norm_28_tmp_4, };
	uint16_t* p_50_dim[] = { split_5_tmp_0_dim, batch_norm_28_tmp_4_dim, };
	float* concat_7_tmp_0 = (float*)calloc(9408, sizeof(float));
	autox_concat(p_50, concat_7_tmp_0, p_50_dim, concat_7_tmp_0_dim, 1, 2, 4);
	free(split_5_tmp_0);
	free(batch_norm_28_tmp_4);

	float* transpose_7_tmp_0 = (float*)calloc(9408, sizeof(float));
	uint16_t axis_51[] = { 0, 2, 1, 3, 4 };
	autox_transpose(concat_7_tmp_0, transpose_7_tmp_0, reshape2_14_tmp_0_dim, transpose_7_tmp_0_dim, axis_51, 5);
	free(concat_7_tmp_0);

	float* split_6_tmp_0 = (float*)calloc(4704, sizeof(float));
	float* split_6_tmp_1 = (float*)calloc(4704, sizeof(float));
	float* p_52[] = { split_6_tmp_0, split_6_tmp_1, };
	uint16_t* p_52_dim[] = { split_6_tmp_0_dim, split_6_tmp_1_dim, };
	autox_split(transpose_7_tmp_0, p_52, reshape2_15_tmp_0_dim, p_52_dim, 1, 4, 2, 4);
	free(transpose_7_tmp_0);

	float* batch_norm_29_tmp_4 = (float*)calloc(4704, sizeof(float));
	autox_conv2d(split_6_tmp_1, (float*)((int8_t*)weights) + 9684, (float*)((int8_t*)weights) + 9708, batch_norm_29_tmp_4, split_6_tmp_1_dim, stage_3_5_conv1_weights_dim, batch_norm_29_tmp_4_dim, 1, 0, 1, 1, 1);
	free(split_6_tmp_1);

	float* batch_norm_30_tmp_3 = (float*)calloc(4704, sizeof(float));
	autox_conv2d(batch_norm_29_tmp_4, (float*)((int8_t*)weights) + 10284, (float*)((int8_t*)weights) + 10308, batch_norm_30_tmp_3, batch_norm_29_tmp_4_dim, stage_3_5_conv2_weights_dim, batch_norm_30_tmp_3_dim, 24, 1, 1, 1, 0);
	free(batch_norm_29_tmp_4);

	float* batch_norm_31_tmp_4 = (float*)calloc(4704, sizeof(float));
	autox_conv2d(batch_norm_30_tmp_3, (float*)((int8_t*)weights) + 10524, (float*)((int8_t*)weights) + 10548, batch_norm_31_tmp_4, batch_norm_30_tmp_3_dim, stage_3_5_conv3_weights_dim, batch_norm_31_tmp_4_dim, 1, 0, 1, 1, 1);
	free(batch_norm_30_tmp_3);

	float* p_56[] = { split_6_tmp_0, batch_norm_31_tmp_4, };
	uint16_t* p_56_dim[] = { split_6_tmp_0_dim, batch_norm_31_tmp_4_dim, };
	float* concat_8_tmp_0 = (float*)calloc(9408, sizeof(float));
	autox_concat(p_56, concat_8_tmp_0, p_56_dim, concat_8_tmp_0_dim, 1, 2, 4);
	free(split_6_tmp_0);
	free(batch_norm_31_tmp_4);

	float* transpose_8_tmp_0 = (float*)calloc(9408, sizeof(float));
	uint16_t axis_57[] = { 0, 2, 1, 3, 4 };
	autox_transpose(concat_8_tmp_0, transpose_8_tmp_0, reshape2_16_tmp_0_dim, transpose_8_tmp_0_dim, axis_57, 5);
	free(concat_8_tmp_0);

	float* split_7_tmp_0 = (float*)calloc(4704, sizeof(float));
	float* split_7_tmp_1 = (float*)calloc(4704, sizeof(float));
	float* p_58[] = { split_7_tmp_0, split_7_tmp_1, };
	uint16_t* p_58_dim[] = { split_7_tmp_0_dim, split_7_tmp_1_dim, };
	autox_split(transpose_8_tmp_0, p_58, reshape2_17_tmp_0_dim, p_58_dim, 1, 4, 2, 4);
	free(transpose_8_tmp_0);

	float* batch_norm_32_tmp_4 = (float*)calloc(4704, sizeof(float));
	autox_conv2d(split_7_tmp_1, (float*)((int8_t*)weights) + 11124, (float*)((int8_t*)weights) + 11148, batch_norm_32_tmp_4, split_7_tmp_1_dim, stage_3_6_conv1_weights_dim, batch_norm_32_tmp_4_dim, 1, 0, 1, 1, 1);
	free(split_7_tmp_1);

	float* batch_norm_33_tmp_3 = (float*)calloc(4704, sizeof(float));
	autox_conv2d(batch_norm_32_tmp_4, (float*)((int8_t*)weights) + 11724, (float*)((int8_t*)weights) + 11748, batch_norm_33_tmp_3, batch_norm_32_tmp_4_dim, stage_3_6_conv2_weights_dim, batch_norm_33_tmp_3_dim, 24, 1, 1, 1, 0);
	free(batch_norm_32_tmp_4);

	float* batch_norm_34_tmp_4 = (float*)calloc(4704, sizeof(float));
	autox_conv2d(batch_norm_33_tmp_3, (float*)((int8_t*)weights) + 11964, (float*)((int8_t*)weights) + 11988, batch_norm_34_tmp_4, batch_norm_33_tmp_3_dim, stage_3_6_conv3_weights_dim, batch_norm_34_tmp_4_dim, 1, 0, 1, 1, 1);
	free(batch_norm_33_tmp_3);

	float* p_62[] = { split_7_tmp_0, batch_norm_34_tmp_4, };
	uint16_t* p_62_dim[] = { split_7_tmp_0_dim, batch_norm_34_tmp_4_dim, };
	float* concat_9_tmp_0 = (float*)calloc(9408, sizeof(float));
	autox_concat(p_62, concat_9_tmp_0, p_62_dim, concat_9_tmp_0_dim, 1, 2, 4);
	free(split_7_tmp_0);
	free(batch_norm_34_tmp_4);

	float* transpose_9_tmp_0 = (float*)calloc(9408, sizeof(float));
	uint16_t axis_63[] = { 0, 2, 1, 3, 4 };
	autox_transpose(concat_9_tmp_0, transpose_9_tmp_0, reshape2_18_tmp_0_dim, transpose_9_tmp_0_dim, axis_63, 5);
	free(concat_9_tmp_0);

	float* split_8_tmp_0 = (float*)calloc(4704, sizeof(float));
	float* split_8_tmp_1 = (float*)calloc(4704, sizeof(float));
	float* p_64[] = { split_8_tmp_0, split_8_tmp_1, };
	uint16_t* p_64_dim[] = { split_8_tmp_0_dim, split_8_tmp_1_dim, };
	autox_split(transpose_9_tmp_0, p_64, reshape2_19_tmp_0_dim, p_64_dim, 1, 4, 2, 4);
	free(transpose_9_tmp_0);

	float* batch_norm_35_tmp_4 = (float*)calloc(4704, sizeof(float));
	autox_conv2d(split_8_tmp_1, (float*)((int8_t*)weights) + 12564, (float*)((int8_t*)weights) + 12588, batch_norm_35_tmp_4, split_8_tmp_1_dim, stage_3_7_conv1_weights_dim, batch_norm_35_tmp_4_dim, 1, 0, 1, 1, 1);
	free(split_8_tmp_1);

	float* batch_norm_36_tmp_3 = (float*)calloc(4704, sizeof(float));
	autox_conv2d(batch_norm_35_tmp_4, (float*)((int8_t*)weights) + 13164, (float*)((int8_t*)weights) + 13188, batch_norm_36_tmp_3, batch_norm_35_tmp_4_dim, stage_3_7_conv2_weights_dim, batch_norm_36_tmp_3_dim, 24, 1, 1, 1, 0);
	free(batch_norm_35_tmp_4);

	float* batch_norm_37_tmp_4 = (float*)calloc(4704, sizeof(float));
	autox_conv2d(batch_norm_36_tmp_3, (float*)((int8_t*)weights) + 13404, (float*)((int8_t*)weights) + 13428, batch_norm_37_tmp_4, batch_norm_36_tmp_3_dim, stage_3_7_conv3_weights_dim, batch_norm_37_tmp_4_dim, 1, 0, 1, 1, 1);
	free(batch_norm_36_tmp_3);

	float* p_68[] = { split_8_tmp_0, batch_norm_37_tmp_4, };
	uint16_t* p_68_dim[] = { split_8_tmp_0_dim, batch_norm_37_tmp_4_dim, };
	float* concat_10_tmp_0 = (float*)calloc(9408, sizeof(float));
	autox_concat(p_68, concat_10_tmp_0, p_68_dim, concat_10_tmp_0_dim, 1, 2, 4);
	free(split_8_tmp_0);
	free(batch_norm_37_tmp_4);

	float* transpose_10_tmp_0 = (float*)calloc(9408, sizeof(float));
	uint16_t axis_69[] = { 0, 2, 1, 3, 4 };
	autox_transpose(concat_10_tmp_0, transpose_10_tmp_0, reshape2_20_tmp_0_dim, transpose_10_tmp_0_dim, axis_69, 5);
	free(concat_10_tmp_0);

	float* split_9_tmp_0 = (float*)calloc(4704, sizeof(float));
	float* split_9_tmp_1 = (float*)calloc(4704, sizeof(float));
	float* p_70[] = { split_9_tmp_0, split_9_tmp_1, };
	uint16_t* p_70_dim[] = { split_9_tmp_0_dim, split_9_tmp_1_dim, };
	autox_split(transpose_10_tmp_0, p_70, reshape2_21_tmp_0_dim, p_70_dim, 1, 4, 2, 4);
	free(transpose_10_tmp_0);

	float* batch_norm_38_tmp_4 = (float*)calloc(4704, sizeof(float));
	autox_conv2d(split_9_tmp_1, (float*)((int8_t*)weights) + 14004, (float*)((int8_t*)weights) + 14028, batch_norm_38_tmp_4, split_9_tmp_1_dim, stage_3_8_conv1_weights_dim, batch_norm_38_tmp_4_dim, 1, 0, 1, 1, 1);
	free(split_9_tmp_1);

	float* batch_norm_39_tmp_3 = (float*)calloc(4704, sizeof(float));
	autox_conv2d(batch_norm_38_tmp_4, (float*)((int8_t*)weights) + 14604, (float*)((int8_t*)weights) + 14628, batch_norm_39_tmp_3, batch_norm_38_tmp_4_dim, stage_3_8_conv2_weights_dim, batch_norm_39_tmp_3_dim, 24, 1, 1, 1, 0);
	free(batch_norm_38_tmp_4);

	float* batch_norm_40_tmp_4 = (float*)calloc(4704, sizeof(float));
	autox_conv2d(batch_norm_39_tmp_3, (float*)((int8_t*)weights) + 14844, (float*)((int8_t*)weights) + 14868, batch_norm_40_tmp_4, batch_norm_39_tmp_3_dim, stage_3_8_conv3_weights_dim, batch_norm_40_tmp_4_dim, 1, 0, 1, 1, 1);
	free(batch_norm_39_tmp_3);

	float* p_74[] = { split_9_tmp_0, batch_norm_40_tmp_4, };
	uint16_t* p_74_dim[] = { split_9_tmp_0_dim, batch_norm_40_tmp_4_dim, };
	float* concat_11_tmp_0 = (float*)calloc(9408, sizeof(float));
	autox_concat(p_74, concat_11_tmp_0, p_74_dim, concat_11_tmp_0_dim, 1, 2, 4);
	free(split_9_tmp_0);
	free(batch_norm_40_tmp_4);

	float* transpose_11_tmp_0 = (float*)calloc(9408, sizeof(float));
	uint16_t axis_75[] = { 0, 2, 1, 3, 4 };
	autox_transpose(concat_11_tmp_0, transpose_11_tmp_0, reshape2_22_tmp_0_dim, transpose_11_tmp_0_dim, axis_75, 5);
	free(concat_11_tmp_0);

	float* batch_norm_41_tmp_3 = (float*)calloc(2352, sizeof(float));
	autox_conv2d(transpose_11_tmp_0, (float*)((int8_t*)weights) + 15444, (float*)((int8_t*)weights) + 15492, batch_norm_41_tmp_3, reshape2_23_tmp_0_dim, stage_4_1_conv4_weights_dim, batch_norm_41_tmp_3_dim, 48, 1, 2, 1, 0);
	// free(transpose_11_tmp_0);

	float* batch_norm_43_tmp_4 = (float*)calloc(9408, sizeof(float));
	autox_conv2d(transpose_11_tmp_0, (float*)((int8_t*)weights) + 15924, (float*)((int8_t*)weights) + 15972, batch_norm_43_tmp_4, reshape2_23_tmp_0_dim, stage_4_1_conv1_weights_dim, batch_norm_43_tmp_4_dim, 1, 0, 1, 1, 1);
	free(transpose_11_tmp_0);

	float* batch_norm_44_tmp_3 = (float*)calloc(2352, sizeof(float));
	autox_conv2d(batch_norm_43_tmp_4, (float*)((int8_t*)weights) + 18276, (float*)((int8_t*)weights) + 18324, batch_norm_44_tmp_3, batch_norm_43_tmp_4_dim, stage_4_1_conv2_weights_dim, batch_norm_44_tmp_3_dim, 48, 1, 2, 1, 0);
	free(batch_norm_43_tmp_4);

	float* batch_norm_42_tmp_4 = (float*)calloc(2352, sizeof(float));
	autox_conv2d(batch_norm_41_tmp_3, (float*)((int8_t*)weights) + 18756, (float*)((int8_t*)weights) + 18804, batch_norm_42_tmp_4, batch_norm_41_tmp_3_dim, stage_4_1_conv5_weights_dim, batch_norm_42_tmp_4_dim, 1, 0, 1, 1, 1);
	free(batch_norm_41_tmp_3);

	float* batch_norm_45_tmp_4 = (float*)calloc(2352, sizeof(float));
	autox_conv2d(batch_norm_44_tmp_3, (float*)((int8_t*)weights) + 21108, (float*)((int8_t*)weights) + 21156, batch_norm_45_tmp_4, batch_norm_44_tmp_3_dim, stage_4_1_conv3_weights_dim, batch_norm_45_tmp_4_dim, 1, 0, 1, 1, 1);
	free(batch_norm_44_tmp_3);

	float* p_81[] = { batch_norm_42_tmp_4, batch_norm_45_tmp_4, };
	uint16_t* p_81_dim[] = { batch_norm_42_tmp_4_dim, batch_norm_45_tmp_4_dim, };
	float* concat_12_tmp_0 = (float*)calloc(4704, sizeof(float));
	autox_concat(p_81, concat_12_tmp_0, p_81_dim, concat_12_tmp_0_dim, 1, 2, 4);
	free(batch_norm_42_tmp_4);
	free(batch_norm_45_tmp_4);

	float* transpose_12_tmp_0 = (float*)calloc(4704, sizeof(float));
	uint16_t axis_82[] = { 0, 2, 1, 3, 4 };
	autox_transpose(concat_12_tmp_0, transpose_12_tmp_0, reshape2_24_tmp_0_dim, transpose_12_tmp_0_dim, axis_82, 5);
	free(concat_12_tmp_0);

	float* split_10_tmp_0 = (float*)calloc(2352, sizeof(float));
	float* split_10_tmp_1 = (float*)calloc(2352, sizeof(float));
	float* p_83[] = { split_10_tmp_0, split_10_tmp_1, };
	uint16_t* p_83_dim[] = { split_10_tmp_0_dim, split_10_tmp_1_dim, };
	autox_split(transpose_12_tmp_0, p_83, reshape2_25_tmp_0_dim, p_83_dim, 1, 4, 2, 4);
	free(transpose_12_tmp_0);

	float* batch_norm_46_tmp_4 = (float*)calloc(2352, sizeof(float));
	autox_conv2d(split_10_tmp_1, (float*)((int8_t*)weights) + 23460, (float*)((int8_t*)weights) + 23508, batch_norm_46_tmp_4, split_10_tmp_1_dim, stage_4_2_conv1_weights_dim, batch_norm_46_tmp_4_dim, 1, 0, 1, 1, 1);
	free(split_10_tmp_1);

	float* batch_norm_47_tmp_3 = (float*)calloc(2352, sizeof(float));
	autox_conv2d(batch_norm_46_tmp_4, (float*)((int8_t*)weights) + 25812, (float*)((int8_t*)weights) + 25860, batch_norm_47_tmp_3, batch_norm_46_tmp_4_dim, stage_4_2_conv2_weights_dim, batch_norm_47_tmp_3_dim, 48, 1, 1, 1, 0);
	free(batch_norm_46_tmp_4);

	float* batch_norm_48_tmp_4 = (float*)calloc(2352, sizeof(float));
	autox_conv2d(batch_norm_47_tmp_3, (float*)((int8_t*)weights) + 26292, (float*)((int8_t*)weights) + 26340, batch_norm_48_tmp_4, batch_norm_47_tmp_3_dim, stage_4_2_conv3_weights_dim, batch_norm_48_tmp_4_dim, 1, 0, 1, 1, 1);
	free(batch_norm_47_tmp_3);

	float* p_87[] = { split_10_tmp_0, batch_norm_48_tmp_4, };
	uint16_t* p_87_dim[] = { split_10_tmp_0_dim, batch_norm_48_tmp_4_dim, };
	float* concat_13_tmp_0 = (float*)calloc(4704, sizeof(float));
	autox_concat(p_87, concat_13_tmp_0, p_87_dim, concat_13_tmp_0_dim, 1, 2, 4);
	free(split_10_tmp_0);
	free(batch_norm_48_tmp_4);

	float* transpose_13_tmp_0 = (float*)calloc(4704, sizeof(float));
	uint16_t axis_88[] = { 0, 2, 1, 3, 4 };
	autox_transpose(concat_13_tmp_0, transpose_13_tmp_0, reshape2_26_tmp_0_dim, transpose_13_tmp_0_dim, axis_88, 5);
	free(concat_13_tmp_0);

	float* split_11_tmp_0 = (float*)calloc(2352, sizeof(float));
	float* split_11_tmp_1 = (float*)calloc(2352, sizeof(float));
	float* p_89[] = { split_11_tmp_0, split_11_tmp_1, };
	uint16_t* p_89_dim[] = { split_11_tmp_0_dim, split_11_tmp_1_dim, };
	autox_split(transpose_13_tmp_0, p_89, reshape2_27_tmp_0_dim, p_89_dim, 1, 4, 2, 4);
	free(transpose_13_tmp_0);

	float* batch_norm_49_tmp_4 = (float*)calloc(2352, sizeof(float));
	autox_conv2d(split_11_tmp_1, (float*)((int8_t*)weights) + 28644, (float*)((int8_t*)weights) + 28692, batch_norm_49_tmp_4, split_11_tmp_1_dim, stage_4_3_conv1_weights_dim, batch_norm_49_tmp_4_dim, 1, 0, 1, 1, 1);
	free(split_11_tmp_1);

	float* batch_norm_50_tmp_3 = (float*)calloc(2352, sizeof(float));
	autox_conv2d(batch_norm_49_tmp_4, (float*)((int8_t*)weights) + 30996, (float*)((int8_t*)weights) + 31044, batch_norm_50_tmp_3, batch_norm_49_tmp_4_dim, stage_4_3_conv2_weights_dim, batch_norm_50_tmp_3_dim, 48, 1, 1, 1, 0);
	free(batch_norm_49_tmp_4);

	float* batch_norm_51_tmp_4 = (float*)calloc(2352, sizeof(float));
	autox_conv2d(batch_norm_50_tmp_3, (float*)((int8_t*)weights) + 31476, (float*)((int8_t*)weights) + 31524, batch_norm_51_tmp_4, batch_norm_50_tmp_3_dim, stage_4_3_conv3_weights_dim, batch_norm_51_tmp_4_dim, 1, 0, 1, 1, 1);
	free(batch_norm_50_tmp_3);

	float* p_93[] = { split_11_tmp_0, batch_norm_51_tmp_4, };
	uint16_t* p_93_dim[] = { split_11_tmp_0_dim, batch_norm_51_tmp_4_dim, };
	float* concat_14_tmp_0 = (float*)calloc(4704, sizeof(float));
	autox_concat(p_93, concat_14_tmp_0, p_93_dim, concat_14_tmp_0_dim, 1, 2, 4);
	free(split_11_tmp_0);
	free(batch_norm_51_tmp_4);

	float* transpose_14_tmp_0 = (float*)calloc(4704, sizeof(float));
	uint16_t axis_94[] = { 0, 2, 1, 3, 4 };
	autox_transpose(concat_14_tmp_0, transpose_14_tmp_0, reshape2_28_tmp_0_dim, transpose_14_tmp_0_dim, axis_94, 5);
	free(concat_14_tmp_0);

	float* split_12_tmp_0 = (float*)calloc(2352, sizeof(float));
	float* split_12_tmp_1 = (float*)calloc(2352, sizeof(float));
	float* p_95[] = { split_12_tmp_0, split_12_tmp_1, };
	uint16_t* p_95_dim[] = { split_12_tmp_0_dim, split_12_tmp_1_dim, };
	autox_split(transpose_14_tmp_0, p_95, reshape2_29_tmp_0_dim, p_95_dim, 1, 4, 2, 4);
	free(transpose_14_tmp_0);

	float* batch_norm_52_tmp_4 = (float*)calloc(2352, sizeof(float));
	autox_conv2d(split_12_tmp_1, (float*)((int8_t*)weights) + 33828, (float*)((int8_t*)weights) + 33876, batch_norm_52_tmp_4, split_12_tmp_1_dim, stage_4_4_conv1_weights_dim, batch_norm_52_tmp_4_dim, 1, 0, 1, 1, 1);
	free(split_12_tmp_1);

	float* batch_norm_53_tmp_3 = (float*)calloc(2352, sizeof(float));
	autox_conv2d(batch_norm_52_tmp_4, (float*)((int8_t*)weights) + 36180, (float*)((int8_t*)weights) + 36228, batch_norm_53_tmp_3, batch_norm_52_tmp_4_dim, stage_4_4_conv2_weights_dim, batch_norm_53_tmp_3_dim, 48, 1, 1, 1, 0);
	free(batch_norm_52_tmp_4);

	float* batch_norm_54_tmp_4 = (float*)calloc(2352, sizeof(float));
	autox_conv2d(batch_norm_53_tmp_3, (float*)((int8_t*)weights) + 36660, (float*)((int8_t*)weights) + 36708, batch_norm_54_tmp_4, batch_norm_53_tmp_3_dim, stage_4_4_conv3_weights_dim, batch_norm_54_tmp_4_dim, 1, 0, 1, 1, 1);
	free(batch_norm_53_tmp_3);

	float* p_99[] = { split_12_tmp_0, batch_norm_54_tmp_4, };
	uint16_t* p_99_dim[] = { split_12_tmp_0_dim, batch_norm_54_tmp_4_dim, };
	float* concat_15_tmp_0 = (float*)calloc(4704, sizeof(float));
	autox_concat(p_99, concat_15_tmp_0, p_99_dim, concat_15_tmp_0_dim, 1, 2, 4);
	free(split_12_tmp_0);
	free(batch_norm_54_tmp_4);

	float* transpose_15_tmp_0 = (float*)calloc(4704, sizeof(float));
	uint16_t axis_100[] = { 0, 2, 1, 3, 4 };
	autox_transpose(concat_15_tmp_0, transpose_15_tmp_0, reshape2_30_tmp_0_dim, transpose_15_tmp_0_dim, axis_100, 5);
	free(concat_15_tmp_0);

	float* batch_norm_55_tmp_4 = (float*)calloc(25088, sizeof(float));
	autox_conv2d(transpose_15_tmp_0, (float*)((int8_t*)weights) + 39012, (float*)((int8_t*)weights) + 39524, batch_norm_55_tmp_4, reshape2_31_tmp_0_dim, conv5_weights_dim, batch_norm_55_tmp_4_dim, 1, 0, 1, 1, 1);
	free(transpose_15_tmp_0);

	float* pool2d_1_tmp_0 = (float*)calloc(512, sizeof(float));
	autox_pool2d(batch_norm_55_tmp_4, pool2d_1_tmp_0, batch_norm_55_tmp_4_dim, pool2d_1_tmp_0_dim, 1, 1, 0, 1, 1);
	free(batch_norm_55_tmp_4);

	float* linear_1_tmp_0 = (float*)calloc(1000, sizeof(float));
	autox_matmul(pool2d_1_tmp_0, (float*)((int8_t*)weights) + 88676, linear_1_tmp_0, flatten_0_tmp_0_dim, fc6_weights_dim, linear_1_tmp_0_dim, 0, 0, 2, 2, 2);
	free(pool2d_1_tmp_0);

	float* linear_1_tmp_1 = (float*)calloc(1000, sizeof(float));
	autox_elementwise_add(linear_1_tmp_0, (float*)((int8_t*)weights) + 600676, linear_1_tmp_1, linear_1_tmp_0_dim, fc6_offset_dim, linear_1_tmp_1_dim, 1, 2, 1, 2);
	free(linear_1_tmp_0);

	uint32_t input_ddim[] = {1000};
	uint32_t output_ddim[] = {1};

	autox_argmax(linear_1_tmp_1,Out,input_ddim,output_ddim,1,1,0);
	free(linear_1_tmp_1);

}
