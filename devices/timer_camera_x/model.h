void model(const uint8_t *image, const uint16_t ssize_h, const uint16_t ssize_w, const float *weights, uint32_t *Out)
{
	float *batch_norm_0_tmp_4 = (float *)calloc(301056*sizeof(float));
	autox_conv2d(x, batch_norm_0_tmp_4, weights + 0, weights + 24, {1, 3, 224, 224}, {1, 24, 112, 112}, {24, 3, 3, 3}, 1, 1, 2, 1, 1);
	free(x);
	float *pool2d_0_tmp_0 = (float *)calloc(75264*sizeof(float));
	autox_pool2d(batch_norm_0_tmp_4, pool2d_0_tmp_0, {1, 24, 112, 112}, {1, 24, 56, 56}, 3, 2, 1, 0, 0);
	free(batch_norm_0_tmp_4);
	float *batch_norm_1_tmp_3 = (float *)calloc(18816*sizeof(float));
	autox_conv2d(pool2d_0_tmp_0, batch_norm_1_tmp_3, weights + 672, weights + 696, {1, 24, 56, 56}, {1, 24, 28, 28}, {24, 1, 3, 3}, 24, 1, 2, 1, 0);
	free(pool2d_0_tmp_0);
	float *batch_norm_3_tmp_4 = (float *)calloc(37632*sizeof(float));
	autox_conv2d(pool2d_0_tmp_0, batch_norm_3_tmp_4, weights + 912, weights + 924, {1, 24, 56, 56}, {1, 12, 56, 56}, {12, 24, 1, 1}, 1, 0, 1, 1, 1);
	free(pool2d_0_tmp_0);
	float *batch_norm_4_tmp_3 = (float *)calloc(9408*sizeof(float));
	autox_conv2d(batch_norm_3_tmp_4, batch_norm_4_tmp_3, weights + 1212, weights + 1224, {1, 12, 56, 56}, {1, 12, 28, 28}, {12, 1, 3, 3}, 12, 1, 2, 1, 0);
	free(batch_norm_3_tmp_4);
	float *batch_norm_5_tmp_4 = (float *)calloc(9408*sizeof(float));
	autox_conv2d(batch_norm_4_tmp_3, batch_norm_5_tmp_4, weights + 1332, weights + 1344, {1, 12, 28, 28}, {1, 12, 28, 28}, {12, 12, 1, 1}, 1, 0, 1, 1, 1);
	free(batch_norm_4_tmp_3);
	float *batch_norm_2_tmp_4 = (float *)calloc(9408*sizeof(float));
	autox_conv2d(batch_norm_1_tmp_3, batch_norm_2_tmp_4, weights + 1488, weights + 1500, {1, 24, 28, 28}, {1, 12, 28, 28}, {12, 24, 1, 1}, 1, 0, 1, 1, 1);
	free(batch_norm_1_tmp_3);
	float *concat_0_tmp_0 = (float *)calloc(18816*sizeof(float));
	autox_concat(batch_norm_2_tmp_4, batch_norm_5_tmp_4, concat_0_tmp_0, {1, 12, 28, 28}, {1, 12, 28, 28}, {1, 24, 28, 28}, 1);
	free(batch_norm_2_tmp_4);
	free(batch_norm_5_tmp_4);
	float *transpose_0_tmp_0 = (float *)calloc(18816*sizeof(float));
	autox_transpose2(reshape2_0_tmp_0, transpose_0_tmp_0, {1, 2, 12, 28, 28}, {1, 12, 2, 28, 28}, {0, 2, 1, 3, 4}, 4);
	free(reshape2_0_tmp_0);
	float *split_0_tmp_0 = (float *)calloc(9408*sizeof(float));
	float *split_0_tmp_1 = (float *)calloc(9408*sizeof(float));
	autox_split(reshape2_1_tmp_0, split_0_tmp_0, split_0_tmp_1, {1, 24, 28, 28}, {1, 12, 28, 28}, {1, 12, 28, 28}, 1);
	free(reshape2_1_tmp_0);
	float *batch_norm_6_tmp_4 = (float *)calloc(9408*sizeof(float));
	autox_conv2d(split_0_tmp_1, batch_norm_6_tmp_4, weights + 1788, weights + 1800, {1, 12, 28, 28}, {1, 12, 28, 28}, {12, 12, 1, 1}, 1, 0, 1, 1, 1);
	free(split_0_tmp_1);
	float *batch_norm_7_tmp_3 = (float *)calloc(9408*sizeof(float));
	autox_conv2d(batch_norm_6_tmp_4, batch_norm_7_tmp_3, weights + 1944, weights + 1956, {1, 12, 28, 28}, {1, 12, 28, 28}, {12, 1, 3, 3}, 12, 1, 1, 1, 0);
	free(batch_norm_6_tmp_4);
	float *batch_norm_8_tmp_4 = (float *)calloc(9408*sizeof(float));
	autox_conv2d(batch_norm_7_tmp_3, batch_norm_8_tmp_4, weights + 2064, weights + 2076, {1, 12, 28, 28}, {1, 12, 28, 28}, {12, 12, 1, 1}, 1, 0, 1, 1, 1);
	free(batch_norm_7_tmp_3);
	float *concat_1_tmp_0 = (float *)calloc(18816*sizeof(float));
	autox_concat(split_0_tmp_0, batch_norm_8_tmp_4, concat_1_tmp_0, {1, 12, 28, 28}, {1, 12, 28, 28}, {1, 24, 28, 28}, 1);
	free(split_0_tmp_0);
	free(batch_norm_8_tmp_4);
	float *transpose_1_tmp_0 = (float *)calloc(18816*sizeof(float));
	autox_transpose2(reshape2_2_tmp_0, transpose_1_tmp_0, {1, 2, 12, 28, 28}, {1, 12, 2, 28, 28}, {0, 2, 1, 3, 4}, 4);
	free(reshape2_2_tmp_0);
	float *split_1_tmp_0 = (float *)calloc(9408*sizeof(float));
	float *split_1_tmp_1 = (float *)calloc(9408*sizeof(float));
	autox_split(reshape2_3_tmp_0, split_1_tmp_0, split_1_tmp_1, {1, 24, 28, 28}, {1, 12, 28, 28}, {1, 12, 28, 28}, 1);
	free(reshape2_3_tmp_0);
	float *batch_norm_9_tmp_4 = (float *)calloc(9408*sizeof(float));
	autox_conv2d(split_1_tmp_1, batch_norm_9_tmp_4, weights + 2220, weights + 2232, {1, 12, 28, 28}, {1, 12, 28, 28}, {12, 12, 1, 1}, 1, 0, 1, 1, 1);
	free(split_1_tmp_1);
	float *batch_norm_10_tmp_3 = (float *)calloc(9408*sizeof(float));
	autox_conv2d(batch_norm_9_tmp_4, batch_norm_10_tmp_3, weights + 2376, weights + 2388, {1, 12, 28, 28}, {1, 12, 28, 28}, {12, 1, 3, 3}, 12, 1, 1, 1, 0);
	free(batch_norm_9_tmp_4);
	float *batch_norm_11_tmp_4 = (float *)calloc(9408*sizeof(float));
	autox_conv2d(batch_norm_10_tmp_3, batch_norm_11_tmp_4, weights + 2496, weights + 2508, {1, 12, 28, 28}, {1, 12, 28, 28}, {12, 12, 1, 1}, 1, 0, 1, 1, 1);
	free(batch_norm_10_tmp_3);
	float *concat_2_tmp_0 = (float *)calloc(18816*sizeof(float));
	autox_concat(split_1_tmp_0, batch_norm_11_tmp_4, concat_2_tmp_0, {1, 12, 28, 28}, {1, 12, 28, 28}, {1, 24, 28, 28}, 1);
	free(split_1_tmp_0);
	free(batch_norm_11_tmp_4);
	float *transpose_2_tmp_0 = (float *)calloc(18816*sizeof(float));
	autox_transpose2(reshape2_4_tmp_0, transpose_2_tmp_0, {1, 2, 12, 28, 28}, {1, 12, 2, 28, 28}, {0, 2, 1, 3, 4}, 4);
	free(reshape2_4_tmp_0);
	float *split_2_tmp_0 = (float *)calloc(9408*sizeof(float));
	float *split_2_tmp_1 = (float *)calloc(9408*sizeof(float));
	autox_split(reshape2_5_tmp_0, split_2_tmp_0, split_2_tmp_1, {1, 24, 28, 28}, {1, 12, 28, 28}, {1, 12, 28, 28}, 1);
	free(reshape2_5_tmp_0);
	float *batch_norm_12_tmp_4 = (float *)calloc(9408*sizeof(float));
	autox_conv2d(split_2_tmp_1, batch_norm_12_tmp_4, weights + 2652, weights + 2664, {1, 12, 28, 28}, {1, 12, 28, 28}, {12, 12, 1, 1}, 1, 0, 1, 1, 1);
	free(split_2_tmp_1);
	float *batch_norm_13_tmp_3 = (float *)calloc(9408*sizeof(float));
	autox_conv2d(batch_norm_12_tmp_4, batch_norm_13_tmp_3, weights + 2808, weights + 2820, {1, 12, 28, 28}, {1, 12, 28, 28}, {12, 1, 3, 3}, 12, 1, 1, 1, 0);
	free(batch_norm_12_tmp_4);
	float *batch_norm_14_tmp_4 = (float *)calloc(9408*sizeof(float));
	autox_conv2d(batch_norm_13_tmp_3, batch_norm_14_tmp_4, weights + 2928, weights + 2940, {1, 12, 28, 28}, {1, 12, 28, 28}, {12, 12, 1, 1}, 1, 0, 1, 1, 1);
	free(batch_norm_13_tmp_3);
	float *concat_3_tmp_0 = (float *)calloc(18816*sizeof(float));
	autox_concat(split_2_tmp_0, batch_norm_14_tmp_4, concat_3_tmp_0, {1, 12, 28, 28}, {1, 12, 28, 28}, {1, 24, 28, 28}, 1);
	free(split_2_tmp_0);
	free(batch_norm_14_tmp_4);
	float *transpose_3_tmp_0 = (float *)calloc(18816*sizeof(float));
	autox_transpose2(reshape2_6_tmp_0, transpose_3_tmp_0, {1, 2, 12, 28, 28}, {1, 12, 2, 28, 28}, {0, 2, 1, 3, 4}, 4);
	free(reshape2_6_tmp_0);
	float *batch_norm_15_tmp_3 = (float *)calloc(4704*sizeof(float));
	autox_conv2d(reshape2_7_tmp_0, batch_norm_15_tmp_3, weights + 3084, weights + 3108, {1, 24, 28, 28}, {1, 24, 14, 14}, {24, 1, 3, 3}, 24, 1, 2, 1, 0);
	free(reshape2_7_tmp_0);
	float *batch_norm_17_tmp_4 = (float *)calloc(18816*sizeof(float));
	autox_conv2d(reshape2_7_tmp_0, batch_norm_17_tmp_4, weights + 3324, weights + 3348, {1, 24, 28, 28}, {1, 24, 28, 28}, {24, 24, 1, 1}, 1, 0, 1, 1, 1);
	free(reshape2_7_tmp_0);
	float *batch_norm_18_tmp_3 = (float *)calloc(4704*sizeof(float));
	autox_conv2d(batch_norm_17_tmp_4, batch_norm_18_tmp_3, weights + 3924, weights + 3948, {1, 24, 28, 28}, {1, 24, 14, 14}, {24, 1, 3, 3}, 24, 1, 2, 1, 0);
	free(batch_norm_17_tmp_4);
	float *batch_norm_16_tmp_4 = (float *)calloc(4704*sizeof(float));
	autox_conv2d(batch_norm_15_tmp_3, batch_norm_16_tmp_4, weights + 4164, weights + 4188, {1, 24, 14, 14}, {1, 24, 14, 14}, {24, 24, 1, 1}, 1, 0, 1, 1, 1);
	free(batch_norm_15_tmp_3);
	float *batch_norm_19_tmp_4 = (float *)calloc(4704*sizeof(float));
	autox_conv2d(batch_norm_18_tmp_3, batch_norm_19_tmp_4, weights + 4764, weights + 4788, {1, 24, 14, 14}, {1, 24, 14, 14}, {24, 24, 1, 1}, 1, 0, 1, 1, 1);
	free(batch_norm_18_tmp_3);
	float *concat_4_tmp_0 = (float *)calloc(9408*sizeof(float));
	autox_concat(batch_norm_16_tmp_4, batch_norm_19_tmp_4, concat_4_tmp_0, {1, 24, 14, 14}, {1, 24, 14, 14}, {1, 48, 14, 14}, 1);
	free(batch_norm_16_tmp_4);
	free(batch_norm_19_tmp_4);
	float *transpose_4_tmp_0 = (float *)calloc(9408*sizeof(float));
	autox_transpose2(reshape2_8_tmp_0, transpose_4_tmp_0, {1, 2, 24, 14, 14}, {1, 24, 2, 14, 14}, {0, 2, 1, 3, 4}, 4);
	free(reshape2_8_tmp_0);
	float *split_3_tmp_0 = (float *)calloc(4704*sizeof(float));
	float *split_3_tmp_1 = (float *)calloc(4704*sizeof(float));
	autox_split(reshape2_9_tmp_0, split_3_tmp_0, split_3_tmp_1, {1, 48, 14, 14}, {1, 24, 14, 14}, {1, 24, 14, 14}, 1);
	free(reshape2_9_tmp_0);
	float *batch_norm_20_tmp_4 = (float *)calloc(4704*sizeof(float));
	autox_conv2d(split_3_tmp_1, batch_norm_20_tmp_4, weights + 5364, weights + 5388, {1, 24, 14, 14}, {1, 24, 14, 14}, {24, 24, 1, 1}, 1, 0, 1, 1, 1);
	free(split_3_tmp_1);
	float *batch_norm_21_tmp_3 = (float *)calloc(4704*sizeof(float));
	autox_conv2d(batch_norm_20_tmp_4, batch_norm_21_tmp_3, weights + 5964, weights + 5988, {1, 24, 14, 14}, {1, 24, 14, 14}, {24, 1, 3, 3}, 24, 1, 1, 1, 0);
	free(batch_norm_20_tmp_4);
	float *batch_norm_22_tmp_4 = (float *)calloc(4704*sizeof(float));
	autox_conv2d(batch_norm_21_tmp_3, batch_norm_22_tmp_4, weights + 6204, weights + 6228, {1, 24, 14, 14}, {1, 24, 14, 14}, {24, 24, 1, 1}, 1, 0, 1, 1, 1);
	free(batch_norm_21_tmp_3);
	float *concat_5_tmp_0 = (float *)calloc(9408*sizeof(float));
	autox_concat(split_3_tmp_0, batch_norm_22_tmp_4, concat_5_tmp_0, {1, 24, 14, 14}, {1, 24, 14, 14}, {1, 48, 14, 14}, 1);
	free(split_3_tmp_0);
	free(batch_norm_22_tmp_4);
	float *transpose_5_tmp_0 = (float *)calloc(9408*sizeof(float));
	autox_transpose2(reshape2_10_tmp_0, transpose_5_tmp_0, {1, 2, 24, 14, 14}, {1, 24, 2, 14, 14}, {0, 2, 1, 3, 4}, 4);
	free(reshape2_10_tmp_0);
	float *split_4_tmp_0 = (float *)calloc(4704*sizeof(float));
	float *split_4_tmp_1 = (float *)calloc(4704*sizeof(float));
	autox_split(reshape2_11_tmp_0, split_4_tmp_0, split_4_tmp_1, {1, 48, 14, 14}, {1, 24, 14, 14}, {1, 24, 14, 14}, 1);
	free(reshape2_11_tmp_0);
	float *batch_norm_23_tmp_4 = (float *)calloc(4704*sizeof(float));
	autox_conv2d(split_4_tmp_1, batch_norm_23_tmp_4, weights + 6804, weights + 6828, {1, 24, 14, 14}, {1, 24, 14, 14}, {24, 24, 1, 1}, 1, 0, 1, 1, 1);
	free(split_4_tmp_1);
	float *batch_norm_24_tmp_3 = (float *)calloc(4704*sizeof(float));
	autox_conv2d(batch_norm_23_tmp_4, batch_norm_24_tmp_3, weights + 7404, weights + 7428, {1, 24, 14, 14}, {1, 24, 14, 14}, {24, 1, 3, 3}, 24, 1, 1, 1, 0);
	free(batch_norm_23_tmp_4);
	float *batch_norm_25_tmp_4 = (float *)calloc(4704*sizeof(float));
	autox_conv2d(batch_norm_24_tmp_3, batch_norm_25_tmp_4, weights + 7644, weights + 7668, {1, 24, 14, 14}, {1, 24, 14, 14}, {24, 24, 1, 1}, 1, 0, 1, 1, 1);
	free(batch_norm_24_tmp_3);
	float *concat_6_tmp_0 = (float *)calloc(9408*sizeof(float));
	autox_concat(split_4_tmp_0, batch_norm_25_tmp_4, concat_6_tmp_0, {1, 24, 14, 14}, {1, 24, 14, 14}, {1, 48, 14, 14}, 1);
	free(split_4_tmp_0);
	free(batch_norm_25_tmp_4);
	float *transpose_6_tmp_0 = (float *)calloc(9408*sizeof(float));
	autox_transpose2(reshape2_12_tmp_0, transpose_6_tmp_0, {1, 2, 24, 14, 14}, {1, 24, 2, 14, 14}, {0, 2, 1, 3, 4}, 4);
	free(reshape2_12_tmp_0);
	float *split_5_tmp_0 = (float *)calloc(4704*sizeof(float));
	float *split_5_tmp_1 = (float *)calloc(4704*sizeof(float));
	autox_split(reshape2_13_tmp_0, split_5_tmp_0, split_5_tmp_1, {1, 48, 14, 14}, {1, 24, 14, 14}, {1, 24, 14, 14}, 1);
	free(reshape2_13_tmp_0);
	float *batch_norm_26_tmp_4 = (float *)calloc(4704*sizeof(float));
	autox_conv2d(split_5_tmp_1, batch_norm_26_tmp_4, weights + 8244, weights + 8268, {1, 24, 14, 14}, {1, 24, 14, 14}, {24, 24, 1, 1}, 1, 0, 1, 1, 1);
	free(split_5_tmp_1);
	float *batch_norm_27_tmp_3 = (float *)calloc(4704*sizeof(float));
	autox_conv2d(batch_norm_26_tmp_4, batch_norm_27_tmp_3, weights + 8844, weights + 8868, {1, 24, 14, 14}, {1, 24, 14, 14}, {24, 1, 3, 3}, 24, 1, 1, 1, 0);
	free(batch_norm_26_tmp_4);
	float *batch_norm_28_tmp_4 = (float *)calloc(4704*sizeof(float));
	autox_conv2d(batch_norm_27_tmp_3, batch_norm_28_tmp_4, weights + 9084, weights + 9108, {1, 24, 14, 14}, {1, 24, 14, 14}, {24, 24, 1, 1}, 1, 0, 1, 1, 1);
	free(batch_norm_27_tmp_3);
	float *concat_7_tmp_0 = (float *)calloc(9408*sizeof(float));
	autox_concat(split_5_tmp_0, batch_norm_28_tmp_4, concat_7_tmp_0, {1, 24, 14, 14}, {1, 24, 14, 14}, {1, 48, 14, 14}, 1);
	free(split_5_tmp_0);
	free(batch_norm_28_tmp_4);
	float *transpose_7_tmp_0 = (float *)calloc(9408*sizeof(float));
	autox_transpose2(reshape2_14_tmp_0, transpose_7_tmp_0, {1, 2, 24, 14, 14}, {1, 24, 2, 14, 14}, {0, 2, 1, 3, 4}, 4);
	free(reshape2_14_tmp_0);
	float *split_6_tmp_0 = (float *)calloc(4704*sizeof(float));
	float *split_6_tmp_1 = (float *)calloc(4704*sizeof(float));
	autox_split(reshape2_15_tmp_0, split_6_tmp_0, split_6_tmp_1, {1, 48, 14, 14}, {1, 24, 14, 14}, {1, 24, 14, 14}, 1);
	free(reshape2_15_tmp_0);
	float *batch_norm_29_tmp_4 = (float *)calloc(4704*sizeof(float));
	autox_conv2d(split_6_tmp_1, batch_norm_29_tmp_4, weights + 9684, weights + 9708, {1, 24, 14, 14}, {1, 24, 14, 14}, {24, 24, 1, 1}, 1, 0, 1, 1, 1);
	free(split_6_tmp_1);
	float *batch_norm_30_tmp_3 = (float *)calloc(4704*sizeof(float));
	autox_conv2d(batch_norm_29_tmp_4, batch_norm_30_tmp_3, weights + 10284, weights + 10308, {1, 24, 14, 14}, {1, 24, 14, 14}, {24, 1, 3, 3}, 24, 1, 1, 1, 0);
	free(batch_norm_29_tmp_4);
	float *batch_norm_31_tmp_4 = (float *)calloc(4704*sizeof(float));
	autox_conv2d(batch_norm_30_tmp_3, batch_norm_31_tmp_4, weights + 10524, weights + 10548, {1, 24, 14, 14}, {1, 24, 14, 14}, {24, 24, 1, 1}, 1, 0, 1, 1, 1);
	free(batch_norm_30_tmp_3);
	float *concat_8_tmp_0 = (float *)calloc(9408*sizeof(float));
	autox_concat(split_6_tmp_0, batch_norm_31_tmp_4, concat_8_tmp_0, {1, 24, 14, 14}, {1, 24, 14, 14}, {1, 48, 14, 14}, 1);
	free(split_6_tmp_0);
	free(batch_norm_31_tmp_4);
	float *transpose_8_tmp_0 = (float *)calloc(9408*sizeof(float));
	autox_transpose2(reshape2_16_tmp_0, transpose_8_tmp_0, {1, 2, 24, 14, 14}, {1, 24, 2, 14, 14}, {0, 2, 1, 3, 4}, 4);
	free(reshape2_16_tmp_0);
	float *split_7_tmp_0 = (float *)calloc(4704*sizeof(float));
	float *split_7_tmp_1 = (float *)calloc(4704*sizeof(float));
	autox_split(reshape2_17_tmp_0, split_7_tmp_0, split_7_tmp_1, {1, 48, 14, 14}, {1, 24, 14, 14}, {1, 24, 14, 14}, 1);
	free(reshape2_17_tmp_0);
	float *batch_norm_32_tmp_4 = (float *)calloc(4704*sizeof(float));
	autox_conv2d(split_7_tmp_1, batch_norm_32_tmp_4, weights + 11124, weights + 11148, {1, 24, 14, 14}, {1, 24, 14, 14}, {24, 24, 1, 1}, 1, 0, 1, 1, 1);
	free(split_7_tmp_1);
	float *batch_norm_33_tmp_3 = (float *)calloc(4704*sizeof(float));
	autox_conv2d(batch_norm_32_tmp_4, batch_norm_33_tmp_3, weights + 11724, weights + 11748, {1, 24, 14, 14}, {1, 24, 14, 14}, {24, 1, 3, 3}, 24, 1, 1, 1, 0);
	free(batch_norm_32_tmp_4);
	float *batch_norm_34_tmp_4 = (float *)calloc(4704*sizeof(float));
	autox_conv2d(batch_norm_33_tmp_3, batch_norm_34_tmp_4, weights + 11964, weights + 11988, {1, 24, 14, 14}, {1, 24, 14, 14}, {24, 24, 1, 1}, 1, 0, 1, 1, 1);
	free(batch_norm_33_tmp_3);
	float *concat_9_tmp_0 = (float *)calloc(9408*sizeof(float));
	autox_concat(split_7_tmp_0, batch_norm_34_tmp_4, concat_9_tmp_0, {1, 24, 14, 14}, {1, 24, 14, 14}, {1, 48, 14, 14}, 1);
	free(split_7_tmp_0);
	free(batch_norm_34_tmp_4);
	float *transpose_9_tmp_0 = (float *)calloc(9408*sizeof(float));
	autox_transpose2(reshape2_18_tmp_0, transpose_9_tmp_0, {1, 2, 24, 14, 14}, {1, 24, 2, 14, 14}, {0, 2, 1, 3, 4}, 4);
	free(reshape2_18_tmp_0);
	float *split_8_tmp_0 = (float *)calloc(4704*sizeof(float));
	float *split_8_tmp_1 = (float *)calloc(4704*sizeof(float));
	autox_split(reshape2_19_tmp_0, split_8_tmp_0, split_8_tmp_1, {1, 48, 14, 14}, {1, 24, 14, 14}, {1, 24, 14, 14}, 1);
	free(reshape2_19_tmp_0);
	float *batch_norm_35_tmp_4 = (float *)calloc(4704*sizeof(float));
	autox_conv2d(split_8_tmp_1, batch_norm_35_tmp_4, weights + 12564, weights + 12588, {1, 24, 14, 14}, {1, 24, 14, 14}, {24, 24, 1, 1}, 1, 0, 1, 1, 1);
	free(split_8_tmp_1);
	float *batch_norm_36_tmp_3 = (float *)calloc(4704*sizeof(float));
	autox_conv2d(batch_norm_35_tmp_4, batch_norm_36_tmp_3, weights + 13164, weights + 13188, {1, 24, 14, 14}, {1, 24, 14, 14}, {24, 1, 3, 3}, 24, 1, 1, 1, 0);
	free(batch_norm_35_tmp_4);
	float *batch_norm_37_tmp_4 = (float *)calloc(4704*sizeof(float));
	autox_conv2d(batch_norm_36_tmp_3, batch_norm_37_tmp_4, weights + 13404, weights + 13428, {1, 24, 14, 14}, {1, 24, 14, 14}, {24, 24, 1, 1}, 1, 0, 1, 1, 1);
	free(batch_norm_36_tmp_3);
	float *concat_10_tmp_0 = (float *)calloc(9408*sizeof(float));
	autox_concat(split_8_tmp_0, batch_norm_37_tmp_4, concat_10_tmp_0, {1, 24, 14, 14}, {1, 24, 14, 14}, {1, 48, 14, 14}, 1);
	free(split_8_tmp_0);
	free(batch_norm_37_tmp_4);
	float *transpose_10_tmp_0 = (float *)calloc(9408*sizeof(float));
	autox_transpose2(reshape2_20_tmp_0, transpose_10_tmp_0, {1, 2, 24, 14, 14}, {1, 24, 2, 14, 14}, {0, 2, 1, 3, 4}, 4);
	free(reshape2_20_tmp_0);
	float *split_9_tmp_0 = (float *)calloc(4704*sizeof(float));
	float *split_9_tmp_1 = (float *)calloc(4704*sizeof(float));
	autox_split(reshape2_21_tmp_0, split_9_tmp_0, split_9_tmp_1, {1, 48, 14, 14}, {1, 24, 14, 14}, {1, 24, 14, 14}, 1);
	free(reshape2_21_tmp_0);
	float *batch_norm_38_tmp_4 = (float *)calloc(4704*sizeof(float));
	autox_conv2d(split_9_tmp_1, batch_norm_38_tmp_4, weights + 14004, weights + 14028, {1, 24, 14, 14}, {1, 24, 14, 14}, {24, 24, 1, 1}, 1, 0, 1, 1, 1);
	free(split_9_tmp_1);
	float *batch_norm_39_tmp_3 = (float *)calloc(4704*sizeof(float));
	autox_conv2d(batch_norm_38_tmp_4, batch_norm_39_tmp_3, weights + 14604, weights + 14628, {1, 24, 14, 14}, {1, 24, 14, 14}, {24, 1, 3, 3}, 24, 1, 1, 1, 0);
	free(batch_norm_38_tmp_4);
	float *batch_norm_40_tmp_4 = (float *)calloc(4704*sizeof(float));
	autox_conv2d(batch_norm_39_tmp_3, batch_norm_40_tmp_4, weights + 14844, weights + 14868, {1, 24, 14, 14}, {1, 24, 14, 14}, {24, 24, 1, 1}, 1, 0, 1, 1, 1);
	free(batch_norm_39_tmp_3);
	float *concat_11_tmp_0 = (float *)calloc(9408*sizeof(float));
	autox_concat(split_9_tmp_0, batch_norm_40_tmp_4, concat_11_tmp_0, {1, 24, 14, 14}, {1, 24, 14, 14}, {1, 48, 14, 14}, 1);
	free(split_9_tmp_0);
	free(batch_norm_40_tmp_4);
	float *transpose_11_tmp_0 = (float *)calloc(9408*sizeof(float));
	autox_transpose2(reshape2_22_tmp_0, transpose_11_tmp_0, {1, 2, 24, 14, 14}, {1, 24, 2, 14, 14}, {0, 2, 1, 3, 4}, 4);
	free(reshape2_22_tmp_0);
	float *batch_norm_41_tmp_3 = (float *)calloc(2352*sizeof(float));
	autox_conv2d(reshape2_23_tmp_0, batch_norm_41_tmp_3, weights + 15444, weights + 15492, {1, 48, 14, 14}, {1, 48, 7, 7}, {48, 1, 3, 3}, 48, 1, 2, 1, 0);
	free(reshape2_23_tmp_0);
	float *batch_norm_43_tmp_4 = (float *)calloc(9408*sizeof(float));
	autox_conv2d(reshape2_23_tmp_0, batch_norm_43_tmp_4, weights + 15924, weights + 15972, {1, 48, 14, 14}, {1, 48, 14, 14}, {48, 48, 1, 1}, 1, 0, 1, 1, 1);
	free(reshape2_23_tmp_0);
	float *batch_norm_44_tmp_3 = (float *)calloc(2352*sizeof(float));
	autox_conv2d(batch_norm_43_tmp_4, batch_norm_44_tmp_3, weights + 18276, weights + 18324, {1, 48, 14, 14}, {1, 48, 7, 7}, {48, 1, 3, 3}, 48, 1, 2, 1, 0);
	free(batch_norm_43_tmp_4);
	float *batch_norm_42_tmp_4 = (float *)calloc(2352*sizeof(float));
	autox_conv2d(batch_norm_41_tmp_3, batch_norm_42_tmp_4, weights + 18756, weights + 18804, {1, 48, 7, 7}, {1, 48, 7, 7}, {48, 48, 1, 1}, 1, 0, 1, 1, 1);
	free(batch_norm_41_tmp_3);
	float *batch_norm_45_tmp_4 = (float *)calloc(2352*sizeof(float));
	autox_conv2d(batch_norm_44_tmp_3, batch_norm_45_tmp_4, weights + 21108, weights + 21156, {1, 48, 7, 7}, {1, 48, 7, 7}, {48, 48, 1, 1}, 1, 0, 1, 1, 1);
	free(batch_norm_44_tmp_3);
	float *concat_12_tmp_0 = (float *)calloc(4704*sizeof(float));
	autox_concat(batch_norm_42_tmp_4, batch_norm_45_tmp_4, concat_12_tmp_0, {1, 48, 7, 7}, {1, 48, 7, 7}, {1, 96, 7, 7}, 1);
	free(batch_norm_42_tmp_4);
	free(batch_norm_45_tmp_4);
	float *transpose_12_tmp_0 = (float *)calloc(4704*sizeof(float));
	autox_transpose2(reshape2_24_tmp_0, transpose_12_tmp_0, {1, 2, 48, 7, 7}, {1, 48, 2, 7, 7}, {0, 2, 1, 3, 4}, 4);
	free(reshape2_24_tmp_0);
	float *split_10_tmp_0 = (float *)calloc(2352*sizeof(float));
	float *split_10_tmp_1 = (float *)calloc(2352*sizeof(float));
	autox_split(reshape2_25_tmp_0, split_10_tmp_0, split_10_tmp_1, {1, 96, 7, 7}, {1, 48, 7, 7}, {1, 48, 7, 7}, 1);
	free(reshape2_25_tmp_0);
	float *batch_norm_46_tmp_4 = (float *)calloc(2352*sizeof(float));
	autox_conv2d(split_10_tmp_1, batch_norm_46_tmp_4, weights + 23460, weights + 23508, {1, 48, 7, 7}, {1, 48, 7, 7}, {48, 48, 1, 1}, 1, 0, 1, 1, 1);
	free(split_10_tmp_1);
	float *batch_norm_47_tmp_3 = (float *)calloc(2352*sizeof(float));
	autox_conv2d(batch_norm_46_tmp_4, batch_norm_47_tmp_3, weights + 25812, weights + 25860, {1, 48, 7, 7}, {1, 48, 7, 7}, {48, 1, 3, 3}, 48, 1, 1, 1, 0);
	free(batch_norm_46_tmp_4);
	float *batch_norm_48_tmp_4 = (float *)calloc(2352*sizeof(float));
	autox_conv2d(batch_norm_47_tmp_3, batch_norm_48_tmp_4, weights + 26292, weights + 26340, {1, 48, 7, 7}, {1, 48, 7, 7}, {48, 48, 1, 1}, 1, 0, 1, 1, 1);
	free(batch_norm_47_tmp_3);
	float *concat_13_tmp_0 = (float *)calloc(4704*sizeof(float));
	autox_concat(split_10_tmp_0, batch_norm_48_tmp_4, concat_13_tmp_0, {1, 48, 7, 7}, {1, 48, 7, 7}, {1, 96, 7, 7}, 1);
	free(split_10_tmp_0);
	free(batch_norm_48_tmp_4);
	float *transpose_13_tmp_0 = (float *)calloc(4704*sizeof(float));
	autox_transpose2(reshape2_26_tmp_0, transpose_13_tmp_0, {1, 2, 48, 7, 7}, {1, 48, 2, 7, 7}, {0, 2, 1, 3, 4}, 4);
	free(reshape2_26_tmp_0);
	float *split_11_tmp_0 = (float *)calloc(2352*sizeof(float));
	float *split_11_tmp_1 = (float *)calloc(2352*sizeof(float));
	autox_split(reshape2_27_tmp_0, split_11_tmp_0, split_11_tmp_1, {1, 96, 7, 7}, {1, 48, 7, 7}, {1, 48, 7, 7}, 1);
	free(reshape2_27_tmp_0);
	float *batch_norm_49_tmp_4 = (float *)calloc(2352*sizeof(float));
	autox_conv2d(split_11_tmp_1, batch_norm_49_tmp_4, weights + 28644, weights + 28692, {1, 48, 7, 7}, {1, 48, 7, 7}, {48, 48, 1, 1}, 1, 0, 1, 1, 1);
	free(split_11_tmp_1);
	float *batch_norm_50_tmp_3 = (float *)calloc(2352*sizeof(float));
	autox_conv2d(batch_norm_49_tmp_4, batch_norm_50_tmp_3, weights + 30996, weights + 31044, {1, 48, 7, 7}, {1, 48, 7, 7}, {48, 1, 3, 3}, 48, 1, 1, 1, 0);
	free(batch_norm_49_tmp_4);
	float *batch_norm_51_tmp_4 = (float *)calloc(2352*sizeof(float));
	autox_conv2d(batch_norm_50_tmp_3, batch_norm_51_tmp_4, weights + 31476, weights + 31524, {1, 48, 7, 7}, {1, 48, 7, 7}, {48, 48, 1, 1}, 1, 0, 1, 1, 1);
	free(batch_norm_50_tmp_3);
	float *concat_14_tmp_0 = (float *)calloc(4704*sizeof(float));
	autox_concat(split_11_tmp_0, batch_norm_51_tmp_4, concat_14_tmp_0, {1, 48, 7, 7}, {1, 48, 7, 7}, {1, 96, 7, 7}, 1);
	free(split_11_tmp_0);
	free(batch_norm_51_tmp_4);
	float *transpose_14_tmp_0 = (float *)calloc(4704*sizeof(float));
	autox_transpose2(reshape2_28_tmp_0, transpose_14_tmp_0, {1, 2, 48, 7, 7}, {1, 48, 2, 7, 7}, {0, 2, 1, 3, 4}, 4);
	free(reshape2_28_tmp_0);
	float *split_12_tmp_0 = (float *)calloc(2352*sizeof(float));
	float *split_12_tmp_1 = (float *)calloc(2352*sizeof(float));
	autox_split(reshape2_29_tmp_0, split_12_tmp_0, split_12_tmp_1, {1, 96, 7, 7}, {1, 48, 7, 7}, {1, 48, 7, 7}, 1);
	free(reshape2_29_tmp_0);
	float *batch_norm_52_tmp_4 = (float *)calloc(2352*sizeof(float));
	autox_conv2d(split_12_tmp_1, batch_norm_52_tmp_4, weights + 33828, weights + 33876, {1, 48, 7, 7}, {1, 48, 7, 7}, {48, 48, 1, 1}, 1, 0, 1, 1, 1);
	free(split_12_tmp_1);
	float *batch_norm_53_tmp_3 = (float *)calloc(2352*sizeof(float));
	autox_conv2d(batch_norm_52_tmp_4, batch_norm_53_tmp_3, weights + 36180, weights + 36228, {1, 48, 7, 7}, {1, 48, 7, 7}, {48, 1, 3, 3}, 48, 1, 1, 1, 0);
	free(batch_norm_52_tmp_4);
	float *batch_norm_54_tmp_4 = (float *)calloc(2352*sizeof(float));
	autox_conv2d(batch_norm_53_tmp_3, batch_norm_54_tmp_4, weights + 36660, weights + 36708, {1, 48, 7, 7}, {1, 48, 7, 7}, {48, 48, 1, 1}, 1, 0, 1, 1, 1);
	free(batch_norm_53_tmp_3);
	float *concat_15_tmp_0 = (float *)calloc(4704*sizeof(float));
	autox_concat(split_12_tmp_0, batch_norm_54_tmp_4, concat_15_tmp_0, {1, 48, 7, 7}, {1, 48, 7, 7}, {1, 96, 7, 7}, 1);
	free(split_12_tmp_0);
	free(batch_norm_54_tmp_4);
	float *transpose_15_tmp_0 = (float *)calloc(4704*sizeof(float));
	autox_transpose2(reshape2_30_tmp_0, transpose_15_tmp_0, {1, 2, 48, 7, 7}, {1, 48, 2, 7, 7}, {0, 2, 1, 3, 4}, 4);
	free(reshape2_30_tmp_0);
	float *batch_norm_55_tmp_4 = (float *)calloc(25088*sizeof(float));
	autox_conv2d(reshape2_31_tmp_0, batch_norm_55_tmp_4, weights + 39012, weights + 39524, {1, 96, 7, 7}, {1, 512, 7, 7}, {512, 96, 1, 1}, 1, 0, 1, 1, 1);
	free(reshape2_31_tmp_0);
	float *pool2d_1_tmp_0 = (float *)calloc(512*sizeof(float));
	autox_pool2d(batch_norm_55_tmp_4, pool2d_1_tmp_0, {1, 512, 7, 7}, {1, 512, 1, 1}, 1, 1, 0, 1, 0);
	free(batch_norm_55_tmp_4);
	float *linear_1_tmp_0 = (float *)calloc(1000*sizeof(float));
	autox_matmul_v2(flatten_0_tmp_0, linear_1_tmp_0, weights + 88676, {1, 512}, {1, 1000}, {512, 1000}, 0, 0);
	free(flatten_0_tmp_0);
	float *linear_1_tmp_1 = (float *)calloc(1000*sizeof(float));
	autox_elementwise_add(linear_1_tmp_0, linear_1_tmp_1, weights + 600676, {1, 1000}, {1, 1000}, {1000}, 1);
	free(linear_1_tmp_0);
	autox_softmax(linear_1_tmp_1, {1, 1000}, -1);
}