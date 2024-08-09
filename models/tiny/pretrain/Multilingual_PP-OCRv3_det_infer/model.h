void model(const uint8_t *image, const float *weights, uint32_t *Out)
{
	uint16_t x_dim[] = {1, 3, 960, 960};
	uint16_t batch_norm_0_b_0_dim[] = {8};
	uint16_t conv2d_0_w_0_dim[] = {8, 3, 3, 3};
	uint16_t hardswish_0_tmp_0_dim[] = {1, 8, 480, 480};
	uint16_t batch_norm_1_b_0_dim[] = {8};
	uint16_t conv2d_1_w_0_dim[] = {8, 8, 1, 1};
	uint16_t relu_0_tmp_0_dim[] = {1, 8, 480, 480};
	uint16_t batch_norm_2_b_0_dim[] = {8};
	uint16_t conv2d_2_w_0_dim[] = {8, 1, 3, 3};
	uint16_t relu_1_tmp_0_dim[] = {1, 8, 480, 480};
	uint16_t batch_norm_3_b_0_dim[] = {8};
	uint16_t batch_norm_3_tmp_3_dim[] = {1, 8, 480, 480};
	uint16_t conv2d_3_w_0_dim[] = {8, 8, 1, 1};
	uint16_t elementwise_add_0_dim[] = {1, 8, 480, 480};
	uint16_t batch_norm_4_b_0_dim[] = {32};
	uint16_t conv2d_4_w_0_dim[] = {32, 8, 1, 1};
	uint16_t relu_2_tmp_0_dim[] = {1, 32, 480, 480};
	uint16_t batch_norm_5_b_0_dim[] = {32};
	uint16_t conv2d_5_w_0_dim[] = {32, 1, 3, 3};
	uint16_t relu_3_tmp_0_dim[] = {1, 32, 240, 240};
	uint16_t batch_norm_6_b_0_dim[] = {16};
	uint16_t batch_norm_6_tmp_3_dim[] = {1, 16, 240, 240};
	uint16_t conv2d_6_w_0_dim[] = {16, 32, 1, 1};
	uint16_t batch_norm_7_b_0_dim[] = {40};
	uint16_t conv2d_7_w_0_dim[] = {40, 16, 1, 1};
	uint16_t relu_4_tmp_0_dim[] = {1, 40, 240, 240};
	uint16_t batch_norm_8_b_0_dim[] = {40};
	uint16_t conv2d_8_w_0_dim[] = {40, 1, 3, 3};
	uint16_t relu_5_tmp_0_dim[] = {1, 40, 240, 240};
	uint16_t batch_norm_9_b_0_dim[] = {16};
	uint16_t batch_norm_9_tmp_3_dim[] = {1, 16, 240, 240};
	uint16_t conv2d_9_w_0_dim[] = {16, 40, 1, 1};
	uint16_t elementwise_add_1_dim[] = {1, 16, 240, 240};
	uint16_t batch_norm_10_b_0_dim[] = {40};
	uint16_t conv2d_10_w_0_dim[] = {40, 16, 1, 1};
	uint16_t relu_6_tmp_0_dim[] = {1, 40, 240, 240};
	uint16_t batch_norm_11_b_0_dim[] = {40};
	uint16_t conv2d_11_w_0_dim[] = {40, 1, 5, 5};
	uint16_t relu_7_tmp_0_dim[] = {1, 40, 120, 120};
	uint16_t batch_norm_12_b_0_dim[] = {24};
	uint16_t batch_norm_12_tmp_3_dim[] = {1, 24, 120, 120};
	uint16_t conv2d_12_w_0_dim[] = {24, 40, 1, 1};
	uint16_t batch_norm_13_b_0_dim[] = {64};
	uint16_t conv2d_13_w_0_dim[] = {64, 24, 1, 1};
	uint16_t relu_8_tmp_0_dim[] = {1, 64, 120, 120};
	uint16_t batch_norm_14_b_0_dim[] = {64};
	uint16_t conv2d_14_w_0_dim[] = {64, 1, 5, 5};
	uint16_t relu_9_tmp_0_dim[] = {1, 64, 120, 120};
	uint16_t batch_norm_15_b_0_dim[] = {24};
	uint16_t batch_norm_15_tmp_3_dim[] = {1, 24, 120, 120};
	uint16_t conv2d_15_w_0_dim[] = {24, 64, 1, 1};
	uint16_t elementwise_add_2_dim[] = {1, 24, 120, 120};
	uint16_t batch_norm_16_b_0_dim[] = {64};
	uint16_t conv2d_16_w_0_dim[] = {64, 24, 1, 1};
	uint16_t relu_10_tmp_0_dim[] = {1, 64, 120, 120};
	uint16_t batch_norm_17_b_0_dim[] = {64};
	uint16_t conv2d_17_w_0_dim[] = {64, 1, 5, 5};
	uint16_t relu_11_tmp_0_dim[] = {1, 64, 120, 120};
	uint16_t batch_norm_18_b_0_dim[] = {24};
	uint16_t batch_norm_18_tmp_3_dim[] = {1, 24, 120, 120};
	uint16_t conv2d_18_w_0_dim[] = {24, 64, 1, 1};
	uint16_t elementwise_add_3_dim[] = {1, 24, 120, 120};
	uint16_t batch_norm_19_b_0_dim[] = {120};
	uint16_t conv2d_19_w_0_dim[] = {120, 24, 1, 1};
	uint16_t hardswish_1_tmp_0_dim[] = {1, 120, 120, 120};
	uint16_t batch_norm_20_b_0_dim[] = {120};
	uint16_t conv2d_20_w_0_dim[] = {120, 1, 3, 3};
	uint16_t hardswish_2_tmp_0_dim[] = {1, 120, 60, 60};
	uint16_t batch_norm_21_b_0_dim[] = {40};
	uint16_t batch_norm_21_tmp_3_dim[] = {1, 40, 60, 60};
	uint16_t conv2d_21_w_0_dim[] = {40, 120, 1, 1};
	uint16_t batch_norm_22_b_0_dim[] = {104};
	uint16_t conv2d_22_w_0_dim[] = {104, 40, 1, 1};
	uint16_t hardswish_3_tmp_0_dim[] = {1, 104, 60, 60};
	uint16_t batch_norm_23_b_0_dim[] = {104};
	uint16_t conv2d_23_w_0_dim[] = {104, 1, 3, 3};
	uint16_t hardswish_4_tmp_0_dim[] = {1, 104, 60, 60};
	uint16_t batch_norm_24_b_0_dim[] = {40};
	uint16_t batch_norm_24_tmp_3_dim[] = {1, 40, 60, 60};
	uint16_t conv2d_24_w_0_dim[] = {40, 104, 1, 1};
	uint16_t elementwise_add_4_dim[] = {1, 40, 60, 60};
	uint16_t batch_norm_25_b_0_dim[] = {96};
	uint16_t conv2d_25_w_0_dim[] = {96, 40, 1, 1};
	uint16_t hardswish_5_tmp_0_dim[] = {1, 96, 60, 60};
	uint16_t batch_norm_26_b_0_dim[] = {96};
	uint16_t conv2d_26_w_0_dim[] = {96, 1, 3, 3};
	uint16_t hardswish_6_tmp_0_dim[] = {1, 96, 60, 60};
	uint16_t batch_norm_27_b_0_dim[] = {40};
	uint16_t batch_norm_27_tmp_3_dim[] = {1, 40, 60, 60};
	uint16_t conv2d_27_w_0_dim[] = {40, 96, 1, 1};
	uint16_t elementwise_add_5_dim[] = {1, 40, 60, 60};
	uint16_t batch_norm_28_b_0_dim[] = {96};
	uint16_t conv2d_28_w_0_dim[] = {96, 40, 1, 1};
	uint16_t hardswish_7_tmp_0_dim[] = {1, 96, 60, 60};
	uint16_t batch_norm_29_b_0_dim[] = {96};
	uint16_t conv2d_29_w_0_dim[] = {96, 1, 3, 3};
	uint16_t hardswish_8_tmp_0_dim[] = {1, 96, 60, 60};
	uint16_t batch_norm_30_b_0_dim[] = {40};
	uint16_t batch_norm_30_tmp_3_dim[] = {1, 40, 60, 60};
	uint16_t conv2d_30_w_0_dim[] = {40, 96, 1, 1};
	uint16_t elementwise_add_6_dim[] = {1, 40, 60, 60};
	uint16_t batch_norm_31_b_0_dim[] = {240};
	uint16_t conv2d_31_w_0_dim[] = {240, 40, 1, 1};
	uint16_t hardswish_9_tmp_0_dim[] = {1, 240, 60, 60};
	uint16_t batch_norm_32_b_0_dim[] = {240};
	uint16_t conv2d_32_w_0_dim[] = {240, 1, 3, 3};
	uint16_t hardswish_10_tmp_0_dim[] = {1, 240, 60, 60};
	uint16_t batch_norm_33_b_0_dim[] = {56};
	uint16_t batch_norm_33_tmp_3_dim[] = {1, 56, 60, 60};
	uint16_t conv2d_33_w_0_dim[] = {56, 240, 1, 1};
	uint16_t batch_norm_34_b_0_dim[] = {336};
	uint16_t conv2d_34_w_0_dim[] = {336, 56, 1, 1};
	uint16_t hardswish_11_tmp_0_dim[] = {1, 336, 60, 60};
	uint16_t batch_norm_35_b_0_dim[] = {336};
	uint16_t conv2d_35_w_0_dim[] = {336, 1, 3, 3};
	uint16_t hardswish_12_tmp_0_dim[] = {1, 336, 60, 60};
	uint16_t batch_norm_36_b_0_dim[] = {56};
	uint16_t batch_norm_36_tmp_3_dim[] = {1, 56, 60, 60};
	uint16_t conv2d_36_w_0_dim[] = {56, 336, 1, 1};
	uint16_t elementwise_add_7_dim[] = {1, 56, 60, 60};
	uint16_t batch_norm_37_b_0_dim[] = {336};
	uint16_t conv2d_37_w_0_dim[] = {336, 56, 1, 1};
	uint16_t hardswish_13_tmp_0_dim[] = {1, 336, 60, 60};
	uint16_t batch_norm_38_b_0_dim[] = {336};
	uint16_t conv2d_38_w_0_dim[] = {336, 1, 5, 5};
	uint16_t hardswish_14_tmp_0_dim[] = {1, 336, 30, 30};
	uint16_t batch_norm_39_b_0_dim[] = {80};
	uint16_t batch_norm_39_tmp_3_dim[] = {1, 80, 30, 30};
	uint16_t conv2d_39_w_0_dim[] = {80, 336, 1, 1};
	uint16_t batch_norm_40_b_0_dim[] = {480};
	uint16_t conv2d_40_w_0_dim[] = {480, 80, 1, 1};
	uint16_t hardswish_15_tmp_0_dim[] = {1, 480, 30, 30};
	uint16_t batch_norm_41_b_0_dim[] = {480};
	uint16_t conv2d_41_w_0_dim[] = {480, 1, 5, 5};
	uint16_t hardswish_16_tmp_0_dim[] = {1, 480, 30, 30};
	uint16_t batch_norm_42_b_0_dim[] = {80};
	uint16_t batch_norm_42_tmp_3_dim[] = {1, 80, 30, 30};
	uint16_t conv2d_42_w_0_dim[] = {80, 480, 1, 1};
	uint16_t elementwise_add_8_dim[] = {1, 80, 30, 30};
	uint16_t batch_norm_43_b_0_dim[] = {480};
	uint16_t conv2d_43_w_0_dim[] = {480, 80, 1, 1};
	uint16_t hardswish_17_tmp_0_dim[] = {1, 480, 30, 30};
	uint16_t batch_norm_44_b_0_dim[] = {480};
	uint16_t conv2d_44_w_0_dim[] = {480, 1, 5, 5};
	uint16_t hardswish_18_tmp_0_dim[] = {1, 480, 30, 30};
	uint16_t batch_norm_45_b_0_dim[] = {80};
	uint16_t batch_norm_45_tmp_3_dim[] = {1, 80, 30, 30};
	uint16_t conv2d_45_w_0_dim[] = {80, 480, 1, 1};
	uint16_t elementwise_add_9_dim[] = {1, 80, 30, 30};
	uint16_t batch_norm_46_b_0_dim[] = {480};
	uint16_t conv2d_46_w_0_dim[] = {480, 80, 1, 1};
	uint16_t hardswish_19_tmp_0_dim[] = {1, 480, 30, 30};
	uint16_t conv2d_250_tmp_0_dim[] = {1, 96, 30, 30};
	uint16_t conv2d_65_w_0_dim[] = {96, 480, 1, 1};
	uint16_t pool2d_0_tmp_0_dim[] = {1, 96, 1, 1};
	uint16_t conv2d_66_b_0_dim[] = {24};
	uint16_t conv2d_66_w_0_dim[] = {24, 96, 1, 1};
	uint16_t relu_12_tmp_0_dim[] = {1, 24, 1, 1};
	uint16_t conv2d_252_tmp_1_dim[] = {1, 96, 1, 1};
	uint16_t conv2d_67_b_0_dim[] = {96};
	uint16_t conv2d_67_w_0_dim[] = {96, 24, 1, 1};
	uint16_t hardsigmoid_0_tmp_0_dim[] = {1, 96, 1, 1};
	uint16_t tmp_0_dim[] = {1, 96, 30, 30};
	uint16_t tmp_1_dim[] = {1, 96, 30, 30};
	uint16_t conv2d_253_tmp_0_dim[] = {1, 96, 60, 60};
	uint16_t conv2d_59_w_0_dim[] = {96, 56, 1, 1};
	uint16_t pool2d_1_tmp_0_dim[] = {1, 96, 1, 1};
	uint16_t conv2d_60_b_0_dim[] = {24};
	uint16_t conv2d_60_w_0_dim[] = {24, 96, 1, 1};
	uint16_t relu_13_tmp_0_dim[] = {1, 24, 1, 1};
	uint16_t conv2d_255_tmp_1_dim[] = {1, 96, 1, 1};
	uint16_t conv2d_61_b_0_dim[] = {96};
	uint16_t conv2d_61_w_0_dim[] = {96, 24, 1, 1};
	uint16_t hardsigmoid_1_tmp_0_dim[] = {1, 96, 1, 1};
	uint16_t tmp_2_dim[] = {1, 96, 60, 60};
	uint16_t tmp_3_dim[] = {1, 96, 60, 60};
	uint16_t conv2d_256_tmp_0_dim[] = {1, 96, 120, 120};
	uint16_t conv2d_53_w_0_dim[] = {96, 24, 1, 1};
	uint16_t pool2d_2_tmp_0_dim[] = {1, 96, 1, 1};
	uint16_t conv2d_54_b_0_dim[] = {24};
	uint16_t conv2d_54_w_0_dim[] = {24, 96, 1, 1};
	uint16_t relu_14_tmp_0_dim[] = {1, 24, 1, 1};
	uint16_t conv2d_258_tmp_1_dim[] = {1, 96, 1, 1};
	uint16_t conv2d_55_b_0_dim[] = {96};
	uint16_t conv2d_55_w_0_dim[] = {96, 24, 1, 1};
	uint16_t hardsigmoid_2_tmp_0_dim[] = {1, 96, 1, 1};
	uint16_t tmp_4_dim[] = {1, 96, 120, 120};
	uint16_t tmp_5_dim[] = {1, 96, 120, 120};
	uint16_t conv2d_259_tmp_0_dim[] = {1, 96, 240, 240};
	uint16_t conv2d_47_w_0_dim[] = {96, 16, 1, 1};
	uint16_t pool2d_3_tmp_0_dim[] = {1, 96, 1, 1};
	uint16_t conv2d_48_b_0_dim[] = {24};
	uint16_t conv2d_48_w_0_dim[] = {24, 96, 1, 1};
	uint16_t relu_15_tmp_0_dim[] = {1, 24, 1, 1};
	uint16_t conv2d_261_tmp_1_dim[] = {1, 96, 1, 1};
	uint16_t conv2d_49_b_0_dim[] = {96};
	uint16_t conv2d_49_w_0_dim[] = {96, 24, 1, 1};
	uint16_t hardsigmoid_3_tmp_0_dim[] = {1, 96, 1, 1};
	uint16_t tmp_6_dim[] = {1, 96, 240, 240};
	uint16_t tmp_7_dim[] = {1, 96, 240, 240};
	uint16_t nearest_interp_v2_0_tmp_0_dim[] = {1, 96, 60, 60};
	uint16_t tmp_8_dim[] = {1, 96, 60, 60};
	uint16_t nearest_interp_v2_1_tmp_0_dim[] = {1, 96, 120, 120};
	uint16_t tmp_9_dim[] = {1, 96, 120, 120};
	uint16_t nearest_interp_v2_2_tmp_0_dim[] = {1, 96, 240, 240};
	uint16_t tmp_10_dim[] = {1, 96, 240, 240};
	uint16_t conv2d_262_tmp_0_dim[] = {1, 24, 30, 30};
	uint16_t conv2d_68_w_0_dim[] = {24, 96, 3, 3};
	uint16_t pool2d_4_tmp_0_dim[] = {1, 24, 1, 1};
	uint16_t conv2d_69_b_0_dim[] = {6};
	uint16_t conv2d_69_w_0_dim[] = {6, 24, 1, 1};
	uint16_t relu_16_tmp_0_dim[] = {1, 6, 1, 1};
	uint16_t conv2d_264_tmp_1_dim[] = {1, 24, 1, 1};
	uint16_t conv2d_70_b_0_dim[] = {24};
	uint16_t conv2d_70_w_0_dim[] = {24, 6, 1, 1};
	uint16_t hardsigmoid_4_tmp_0_dim[] = {1, 24, 1, 1};
	uint16_t tmp_11_dim[] = {1, 24, 30, 30};
	uint16_t tmp_12_dim[] = {1, 24, 30, 30};
	uint16_t conv2d_265_tmp_0_dim[] = {1, 24, 60, 60};
	uint16_t conv2d_62_w_0_dim[] = {24, 96, 3, 3};
	uint16_t pool2d_5_tmp_0_dim[] = {1, 24, 1, 1};
	uint16_t conv2d_63_b_0_dim[] = {6};
	uint16_t conv2d_63_w_0_dim[] = {6, 24, 1, 1};
	uint16_t relu_17_tmp_0_dim[] = {1, 6, 1, 1};
	uint16_t conv2d_267_tmp_1_dim[] = {1, 24, 1, 1};
	uint16_t conv2d_64_b_0_dim[] = {24};
	uint16_t conv2d_64_w_0_dim[] = {24, 6, 1, 1};
	uint16_t hardsigmoid_5_tmp_0_dim[] = {1, 24, 1, 1};
	uint16_t tmp_13_dim[] = {1, 24, 60, 60};
	uint16_t tmp_14_dim[] = {1, 24, 60, 60};
	uint16_t conv2d_268_tmp_0_dim[] = {1, 24, 120, 120};
	uint16_t conv2d_56_w_0_dim[] = {24, 96, 3, 3};
	uint16_t pool2d_6_tmp_0_dim[] = {1, 24, 1, 1};
	uint16_t conv2d_57_b_0_dim[] = {6};
	uint16_t conv2d_57_w_0_dim[] = {6, 24, 1, 1};
	uint16_t relu_18_tmp_0_dim[] = {1, 6, 1, 1};
	uint16_t conv2d_270_tmp_1_dim[] = {1, 24, 1, 1};
	uint16_t conv2d_58_b_0_dim[] = {24};
	uint16_t conv2d_58_w_0_dim[] = {24, 6, 1, 1};
	uint16_t hardsigmoid_6_tmp_0_dim[] = {1, 24, 1, 1};
	uint16_t tmp_15_dim[] = {1, 24, 120, 120};
	uint16_t tmp_16_dim[] = {1, 24, 120, 120};
	uint16_t conv2d_271_tmp_0_dim[] = {1, 24, 240, 240};
	uint16_t conv2d_50_w_0_dim[] = {24, 96, 3, 3};
	uint16_t pool2d_7_tmp_0_dim[] = {1, 24, 1, 1};
	uint16_t conv2d_51_b_0_dim[] = {6};
	uint16_t conv2d_51_w_0_dim[] = {6, 24, 1, 1};
	uint16_t relu_19_tmp_0_dim[] = {1, 6, 1, 1};
	uint16_t conv2d_273_tmp_1_dim[] = {1, 24, 1, 1};
	uint16_t conv2d_52_b_0_dim[] = {24};
	uint16_t conv2d_52_w_0_dim[] = {24, 6, 1, 1};
	uint16_t hardsigmoid_7_tmp_0_dim[] = {1, 24, 1, 1};
	uint16_t tmp_17_dim[] = {1, 24, 240, 240};
	uint16_t tmp_18_dim[] = {1, 24, 240, 240};
	uint16_t nearest_interp_v2_3_tmp_0_dim[] = {1, 24, 240, 240};
	uint16_t nearest_interp_v2_4_tmp_0_dim[] = {1, 24, 240, 240};
	uint16_t nearest_interp_v2_5_tmp_0_dim[] = {1, 24, 240, 240};
	uint16_t concat_0_tmp_0_dim[] = {1, 96, 240, 240};
	uint16_t batch_norm_47_b_0_dim[] = {24};
	uint16_t batch_norm_47_tmp_4_dim[] = {1, 24, 240, 240};
	uint16_t conv2d_71_w_0_dim[] = {24, 96, 3, 3};
	uint16_t batch_norm_48_b_0_dim[] = {24};
	uint16_t batch_norm_48_tmp_4_dim[] = {1, 24, 480, 480};
	uint16_t conv2d_transpose_0_w_0_dim[] = {24, 24, 2, 2};
	uint16_t conv2d_transpose_1_b_0_dim[] = {1};
	uint16_t conv2d_transpose_1_w_0_dim[] = {24, 1, 2, 2};
	uint16_t elementwise_add_11_tmp_0_dim[] = {1, 1, 960, 960};
	uint16_t sigmoid_0_tmp_0_dim[] = {1, 1, 960, 960};

	float *hardswish_0_tmp_0 = (float *)calloc(1843200, sizeof(float));
	autox_conv2d(x, (float*)((int8_t*)weights) + 0, (float*)((int8_t*)weights) + 8, hardswish_0_tmp_0, x_dim, conv2d_0_w_0_dim, hardswish_0_tmp_0_dim, 1, 1, 2, 1, 10);
	free(x);

	float *relu_0_tmp_0 = (float *)calloc(1843200, sizeof(float));
	autox_conv2d(hardswish_0_tmp_0, (float*)((int8_t*)weights) + 224, (float*)((int8_t*)weights) + 232, relu_0_tmp_0, hardswish_0_tmp_0_dim, conv2d_1_w_0_dim, relu_0_tmp_0_dim, 1, 0, 1, 1, 1);
	free(hardswish_0_tmp_0);

	float *relu_1_tmp_0 = (float *)calloc(1843200, sizeof(float));
	autox_conv2d(relu_0_tmp_0, (float*)((int8_t*)weights) + 296, (float*)((int8_t*)weights) + 304, relu_1_tmp_0, relu_0_tmp_0_dim, conv2d_2_w_0_dim, relu_1_tmp_0_dim, 8, 1, 1, 1, 1);
	free(relu_0_tmp_0);

	float *batch_norm_3_tmp_3 = (float *)calloc(1843200, sizeof(float));
	autox_conv2d(relu_1_tmp_0, (float*)((int8_t*)weights) + 376, (float*)((int8_t*)weights) + 384, batch_norm_3_tmp_3, relu_1_tmp_0_dim, conv2d_3_w_0_dim, batch_norm_3_tmp_3_dim, 1, 0, 1, 1, 0);
	free(relu_1_tmp_0);

	float *elementwise_add_0 = (float *)calloc(1843200, sizeof(float));
	autox_elementwise_add(hardswish_0_tmp_0, batch_norm_3_tmp_3, elementwise_add_0, hardswish_0_tmp_0_dim, batch_norm_3_tmp_3_dim, elementwise_add_0_dim, -1, 4, 4, 4);
	free(hardswish_0_tmp_0);
	free(batch_norm_3_tmp_3);

	float *relu_2_tmp_0 = (float *)calloc(7372800, sizeof(float));
	autox_conv2d(elementwise_add_0, (float*)((int8_t*)weights) + 448, (float*)((int8_t*)weights) + 480, relu_2_tmp_0, elementwise_add_0_dim, conv2d_4_w_0_dim, relu_2_tmp_0_dim, 1, 0, 1, 1, 1);
	free(elementwise_add_0);

	float *relu_3_tmp_0 = (float *)calloc(1843200, sizeof(float));
	autox_conv2d(relu_2_tmp_0, (float*)((int8_t*)weights) + 736, (float*)((int8_t*)weights) + 768, relu_3_tmp_0, relu_2_tmp_0_dim, conv2d_5_w_0_dim, relu_3_tmp_0_dim, 32, 1, 2, 1, 1);
	free(relu_2_tmp_0);

	float *batch_norm_6_tmp_3 = (float *)calloc(921600, sizeof(float));
	autox_conv2d(relu_3_tmp_0, (float*)((int8_t*)weights) + 1056, (float*)((int8_t*)weights) + 1072, batch_norm_6_tmp_3, relu_3_tmp_0_dim, conv2d_6_w_0_dim, batch_norm_6_tmp_3_dim, 1, 0, 1, 1, 0);
	free(relu_3_tmp_0);

	float *relu_4_tmp_0 = (float *)calloc(2304000, sizeof(float));
	autox_conv2d(batch_norm_6_tmp_3, (float*)((int8_t*)weights) + 1584, (float*)((int8_t*)weights) + 1624, relu_4_tmp_0, batch_norm_6_tmp_3_dim, conv2d_7_w_0_dim, relu_4_tmp_0_dim, 1, 0, 1, 1, 1);
	free(batch_norm_6_tmp_3);

	float *relu_5_tmp_0 = (float *)calloc(2304000, sizeof(float));
	autox_conv2d(relu_4_tmp_0, (float*)((int8_t*)weights) + 2264, (float*)((int8_t*)weights) + 2304, relu_5_tmp_0, relu_4_tmp_0_dim, conv2d_8_w_0_dim, relu_5_tmp_0_dim, 40, 1, 1, 1, 1);
	free(relu_4_tmp_0);

	float *batch_norm_9_tmp_3 = (float *)calloc(921600, sizeof(float));
	autox_conv2d(relu_5_tmp_0, (float*)((int8_t*)weights) + 2664, (float*)((int8_t*)weights) + 2680, batch_norm_9_tmp_3, relu_5_tmp_0_dim, conv2d_9_w_0_dim, batch_norm_9_tmp_3_dim, 1, 0, 1, 1, 0);
	free(relu_5_tmp_0);

	float *elementwise_add_1 = (float *)calloc(921600, sizeof(float));
	autox_elementwise_add(batch_norm_6_tmp_3, batch_norm_9_tmp_3, elementwise_add_1, batch_norm_6_tmp_3_dim, batch_norm_9_tmp_3_dim, elementwise_add_1_dim, -1, 4, 4, 4);
	free(batch_norm_6_tmp_3);
	free(batch_norm_9_tmp_3);

	float *relu_6_tmp_0 = (float *)calloc(2304000, sizeof(float));
	autox_conv2d(elementwise_add_1, (float*)((int8_t*)weights) + 3320, (float*)((int8_t*)weights) + 3360, relu_6_tmp_0, elementwise_add_1_dim, conv2d_10_w_0_dim, relu_6_tmp_0_dim, 1, 0, 1, 1, 1);
	free(elementwise_add_1);

	float *relu_7_tmp_0 = (float *)calloc(576000, sizeof(float));
	autox_conv2d(relu_6_tmp_0, (float*)((int8_t*)weights) + 4000, (float*)((int8_t*)weights) + 4040, relu_7_tmp_0, relu_6_tmp_0_dim, conv2d_11_w_0_dim, relu_7_tmp_0_dim, 40, 2, 2, 1, 1);
	free(relu_6_tmp_0);

	float *batch_norm_12_tmp_3 = (float *)calloc(345600, sizeof(float));
	autox_conv2d(relu_7_tmp_0, (float*)((int8_t*)weights) + 5040, (float*)((int8_t*)weights) + 5064, batch_norm_12_tmp_3, relu_7_tmp_0_dim, conv2d_12_w_0_dim, batch_norm_12_tmp_3_dim, 1, 0, 1, 1, 0);
	free(relu_7_tmp_0);

	float *relu_8_tmp_0 = (float *)calloc(921600, sizeof(float));
	autox_conv2d(batch_norm_12_tmp_3, (float*)((int8_t*)weights) + 6024, (float*)((int8_t*)weights) + 6088, relu_8_tmp_0, batch_norm_12_tmp_3_dim, conv2d_13_w_0_dim, relu_8_tmp_0_dim, 1, 0, 1, 1, 1);
	free(batch_norm_12_tmp_3);

	float *relu_9_tmp_0 = (float *)calloc(921600, sizeof(float));
	autox_conv2d(relu_8_tmp_0, (float*)((int8_t*)weights) + 7624, (float*)((int8_t*)weights) + 7688, relu_9_tmp_0, relu_8_tmp_0_dim, conv2d_14_w_0_dim, relu_9_tmp_0_dim, 64, 2, 1, 1, 1);
	free(relu_8_tmp_0);

	float *batch_norm_15_tmp_3 = (float *)calloc(345600, sizeof(float));
	autox_conv2d(relu_9_tmp_0, (float*)((int8_t*)weights) + 9288, (float*)((int8_t*)weights) + 9312, batch_norm_15_tmp_3, relu_9_tmp_0_dim, conv2d_15_w_0_dim, batch_norm_15_tmp_3_dim, 1, 0, 1, 1, 0);
	free(relu_9_tmp_0);

	float *elementwise_add_2 = (float *)calloc(345600, sizeof(float));
	autox_elementwise_add(batch_norm_12_tmp_3, batch_norm_15_tmp_3, elementwise_add_2, batch_norm_12_tmp_3_dim, batch_norm_15_tmp_3_dim, elementwise_add_2_dim, -1, 4, 4, 4);
	free(batch_norm_12_tmp_3);
	free(batch_norm_15_tmp_3);

	float *relu_10_tmp_0 = (float *)calloc(921600, sizeof(float));
	autox_conv2d(elementwise_add_2, (float*)((int8_t*)weights) + 10848, (float*)((int8_t*)weights) + 10912, relu_10_tmp_0, elementwise_add_2_dim, conv2d_16_w_0_dim, relu_10_tmp_0_dim, 1, 0, 1, 1, 1);
	free(elementwise_add_2);

	float *relu_11_tmp_0 = (float *)calloc(921600, sizeof(float));
	autox_conv2d(relu_10_tmp_0, (float*)((int8_t*)weights) + 12448, (float*)((int8_t*)weights) + 12512, relu_11_tmp_0, relu_10_tmp_0_dim, conv2d_17_w_0_dim, relu_11_tmp_0_dim, 64, 2, 1, 1, 1);
	free(relu_10_tmp_0);

	float *batch_norm_18_tmp_3 = (float *)calloc(345600, sizeof(float));
	autox_conv2d(relu_11_tmp_0, (float*)((int8_t*)weights) + 14112, (float*)((int8_t*)weights) + 14136, batch_norm_18_tmp_3, relu_11_tmp_0_dim, conv2d_18_w_0_dim, batch_norm_18_tmp_3_dim, 1, 0, 1, 1, 0);
	free(relu_11_tmp_0);

	float *elementwise_add_3 = (float *)calloc(345600, sizeof(float));
	autox_elementwise_add(elementwise_add_2, batch_norm_18_tmp_3, elementwise_add_3, elementwise_add_2_dim, batch_norm_18_tmp_3_dim, elementwise_add_3_dim, -1, 4, 4, 4);
	free(elementwise_add_2);
	free(batch_norm_18_tmp_3);

	float *hardswish_1_tmp_0 = (float *)calloc(1728000, sizeof(float));
	autox_conv2d(elementwise_add_3, (float*)((int8_t*)weights) + 15672, (float*)((int8_t*)weights) + 15792, hardswish_1_tmp_0, elementwise_add_3_dim, conv2d_19_w_0_dim, hardswish_1_tmp_0_dim, 1, 0, 1, 1, 10);
	free(elementwise_add_3);

	float *hardswish_2_tmp_0 = (float *)calloc(432000, sizeof(float));
	autox_conv2d(hardswish_1_tmp_0, (float*)((int8_t*)weights) + 18672, (float*)((int8_t*)weights) + 18792, hardswish_2_tmp_0, hardswish_1_tmp_0_dim, conv2d_20_w_0_dim, hardswish_2_tmp_0_dim, 120, 1, 2, 1, 10);
	free(hardswish_1_tmp_0);

	float *batch_norm_21_tmp_3 = (float *)calloc(144000, sizeof(float));
	autox_conv2d(hardswish_2_tmp_0, (float*)((int8_t*)weights) + 19872, (float*)((int8_t*)weights) + 19912, batch_norm_21_tmp_3, hardswish_2_tmp_0_dim, conv2d_21_w_0_dim, batch_norm_21_tmp_3_dim, 1, 0, 1, 1, 0);
	free(hardswish_2_tmp_0);

	float *hardswish_3_tmp_0 = (float *)calloc(374400, sizeof(float));
	autox_conv2d(batch_norm_21_tmp_3, (float*)((int8_t*)weights) + 24712, (float*)((int8_t*)weights) + 24816, hardswish_3_tmp_0, batch_norm_21_tmp_3_dim, conv2d_22_w_0_dim, hardswish_3_tmp_0_dim, 1, 0, 1, 1, 10);
	free(batch_norm_21_tmp_3);

	float *hardswish_4_tmp_0 = (float *)calloc(374400, sizeof(float));
	autox_conv2d(hardswish_3_tmp_0, (float*)((int8_t*)weights) + 28976, (float*)((int8_t*)weights) + 29080, hardswish_4_tmp_0, hardswish_3_tmp_0_dim, conv2d_23_w_0_dim, hardswish_4_tmp_0_dim, 104, 1, 1, 1, 10);
	free(hardswish_3_tmp_0);

	float *batch_norm_24_tmp_3 = (float *)calloc(144000, sizeof(float));
	autox_conv2d(hardswish_4_tmp_0, (float*)((int8_t*)weights) + 30016, (float*)((int8_t*)weights) + 30056, batch_norm_24_tmp_3, hardswish_4_tmp_0_dim, conv2d_24_w_0_dim, batch_norm_24_tmp_3_dim, 1, 0, 1, 1, 0);
	free(hardswish_4_tmp_0);

	float *elementwise_add_4 = (float *)calloc(144000, sizeof(float));
	autox_elementwise_add(batch_norm_21_tmp_3, batch_norm_24_tmp_3, elementwise_add_4, batch_norm_21_tmp_3_dim, batch_norm_24_tmp_3_dim, elementwise_add_4_dim, -1, 4, 4, 4);
	free(batch_norm_21_tmp_3);
	free(batch_norm_24_tmp_3);

	float *hardswish_5_tmp_0 = (float *)calloc(345600, sizeof(float));
	autox_conv2d(elementwise_add_4, (float*)((int8_t*)weights) + 34216, (float*)((int8_t*)weights) + 34312, hardswish_5_tmp_0, elementwise_add_4_dim, conv2d_25_w_0_dim, hardswish_5_tmp_0_dim, 1, 0, 1, 1, 10);
	free(elementwise_add_4);

	float *hardswish_6_tmp_0 = (float *)calloc(345600, sizeof(float));
	autox_conv2d(hardswish_5_tmp_0, (float*)((int8_t*)weights) + 38152, (float*)((int8_t*)weights) + 38248, hardswish_6_tmp_0, hardswish_5_tmp_0_dim, conv2d_26_w_0_dim, hardswish_6_tmp_0_dim, 96, 1, 1, 1, 10);
	free(hardswish_5_tmp_0);

	float *batch_norm_27_tmp_3 = (float *)calloc(144000, sizeof(float));
	autox_conv2d(hardswish_6_tmp_0, (float*)((int8_t*)weights) + 39112, (float*)((int8_t*)weights) + 39152, batch_norm_27_tmp_3, hardswish_6_tmp_0_dim, conv2d_27_w_0_dim, batch_norm_27_tmp_3_dim, 1, 0, 1, 1, 0);
	free(hardswish_6_tmp_0);

	float *elementwise_add_5 = (float *)calloc(144000, sizeof(float));
	autox_elementwise_add(elementwise_add_4, batch_norm_27_tmp_3, elementwise_add_5, elementwise_add_4_dim, batch_norm_27_tmp_3_dim, elementwise_add_5_dim, -1, 4, 4, 4);
	free(elementwise_add_4);
	free(batch_norm_27_tmp_3);

	float *hardswish_7_tmp_0 = (float *)calloc(345600, sizeof(float));
	autox_conv2d(elementwise_add_5, (float*)((int8_t*)weights) + 42992, (float*)((int8_t*)weights) + 43088, hardswish_7_tmp_0, elementwise_add_5_dim, conv2d_28_w_0_dim, hardswish_7_tmp_0_dim, 1, 0, 1, 1, 10);
	free(elementwise_add_5);

	float *hardswish_8_tmp_0 = (float *)calloc(345600, sizeof(float));
	autox_conv2d(hardswish_7_tmp_0, (float*)((int8_t*)weights) + 46928, (float*)((int8_t*)weights) + 47024, hardswish_8_tmp_0, hardswish_7_tmp_0_dim, conv2d_29_w_0_dim, hardswish_8_tmp_0_dim, 96, 1, 1, 1, 10);
	free(hardswish_7_tmp_0);

	float *batch_norm_30_tmp_3 = (float *)calloc(144000, sizeof(float));
	autox_conv2d(hardswish_8_tmp_0, (float*)((int8_t*)weights) + 47888, (float*)((int8_t*)weights) + 47928, batch_norm_30_tmp_3, hardswish_8_tmp_0_dim, conv2d_30_w_0_dim, batch_norm_30_tmp_3_dim, 1, 0, 1, 1, 0);
	free(hardswish_8_tmp_0);

	float *elementwise_add_6 = (float *)calloc(144000, sizeof(float));
	autox_elementwise_add(elementwise_add_5, batch_norm_30_tmp_3, elementwise_add_6, elementwise_add_5_dim, batch_norm_30_tmp_3_dim, elementwise_add_6_dim, -1, 4, 4, 4);
	free(elementwise_add_5);
	free(batch_norm_30_tmp_3);

	float *hardswish_9_tmp_0 = (float *)calloc(864000, sizeof(float));
	autox_conv2d(elementwise_add_6, (float*)((int8_t*)weights) + 51768, (float*)((int8_t*)weights) + 52008, hardswish_9_tmp_0, elementwise_add_6_dim, conv2d_31_w_0_dim, hardswish_9_tmp_0_dim, 1, 0, 1, 1, 10);
	free(elementwise_add_6);

	float *hardswish_10_tmp_0 = (float *)calloc(864000, sizeof(float));
	autox_conv2d(hardswish_9_tmp_0, (float*)((int8_t*)weights) + 61608, (float*)((int8_t*)weights) + 61848, hardswish_10_tmp_0, hardswish_9_tmp_0_dim, conv2d_32_w_0_dim, hardswish_10_tmp_0_dim, 240, 1, 1, 1, 10);
	free(hardswish_9_tmp_0);

	float *batch_norm_33_tmp_3 = (float *)calloc(201600, sizeof(float));
	autox_conv2d(hardswish_10_tmp_0, (float*)((int8_t*)weights) + 64008, (float*)((int8_t*)weights) + 64064, batch_norm_33_tmp_3, hardswish_10_tmp_0_dim, conv2d_33_w_0_dim, batch_norm_33_tmp_3_dim, 1, 0, 1, 1, 0);
	free(hardswish_10_tmp_0);

	float *hardswish_11_tmp_0 = (float *)calloc(1209600, sizeof(float));
	autox_conv2d(batch_norm_33_tmp_3, (float*)((int8_t*)weights) + 77504, (float*)((int8_t*)weights) + 77840, hardswish_11_tmp_0, batch_norm_33_tmp_3_dim, conv2d_34_w_0_dim, hardswish_11_tmp_0_dim, 1, 0, 1, 1, 10);
	free(batch_norm_33_tmp_3);

	float *hardswish_12_tmp_0 = (float *)calloc(1209600, sizeof(float));
	autox_conv2d(hardswish_11_tmp_0, (float*)((int8_t*)weights) + 96656, (float*)((int8_t*)weights) + 96992, hardswish_12_tmp_0, hardswish_11_tmp_0_dim, conv2d_35_w_0_dim, hardswish_12_tmp_0_dim, 336, 1, 1, 1, 10);
	free(hardswish_11_tmp_0);

	float *batch_norm_36_tmp_3 = (float *)calloc(201600, sizeof(float));
	autox_conv2d(hardswish_12_tmp_0, (float*)((int8_t*)weights) + 100016, (float*)((int8_t*)weights) + 100072, batch_norm_36_tmp_3, hardswish_12_tmp_0_dim, conv2d_36_w_0_dim, batch_norm_36_tmp_3_dim, 1, 0, 1, 1, 0);
	free(hardswish_12_tmp_0);

	float *elementwise_add_7 = (float *)calloc(201600, sizeof(float));
	autox_elementwise_add(batch_norm_33_tmp_3, batch_norm_36_tmp_3, elementwise_add_7, batch_norm_33_tmp_3_dim, batch_norm_36_tmp_3_dim, elementwise_add_7_dim, -1, 4, 4, 4);
	free(batch_norm_33_tmp_3);
	free(batch_norm_36_tmp_3);

	float *hardswish_13_tmp_0 = (float *)calloc(1209600, sizeof(float));
	autox_conv2d(elementwise_add_7, (float*)((int8_t*)weights) + 118888, (float*)((int8_t*)weights) + 119224, hardswish_13_tmp_0, elementwise_add_7_dim, conv2d_37_w_0_dim, hardswish_13_tmp_0_dim, 1, 0, 1, 1, 10);
	free(elementwise_add_7);

	float *hardswish_14_tmp_0 = (float *)calloc(302400, sizeof(float));
	autox_conv2d(hardswish_13_tmp_0, (float*)((int8_t*)weights) + 138040, (float*)((int8_t*)weights) + 138376, hardswish_14_tmp_0, hardswish_13_tmp_0_dim, conv2d_38_w_0_dim, hardswish_14_tmp_0_dim, 336, 2, 2, 1, 10);
	free(hardswish_13_tmp_0);

	float *batch_norm_39_tmp_3 = (float *)calloc(72000, sizeof(float));
	autox_conv2d(hardswish_14_tmp_0, (float*)((int8_t*)weights) + 146776, (float*)((int8_t*)weights) + 146856, batch_norm_39_tmp_3, hardswish_14_tmp_0_dim, conv2d_39_w_0_dim, batch_norm_39_tmp_3_dim, 1, 0, 1, 1, 0);
	free(hardswish_14_tmp_0);

	float *hardswish_15_tmp_0 = (float *)calloc(432000, sizeof(float));
	autox_conv2d(batch_norm_39_tmp_3, (float*)((int8_t*)weights) + 173736, (float*)((int8_t*)weights) + 174216, hardswish_15_tmp_0, batch_norm_39_tmp_3_dim, conv2d_40_w_0_dim, hardswish_15_tmp_0_dim, 1, 0, 1, 1, 10);
	free(batch_norm_39_tmp_3);

	float *hardswish_16_tmp_0 = (float *)calloc(432000, sizeof(float));
	autox_conv2d(hardswish_15_tmp_0, (float*)((int8_t*)weights) + 212616, (float*)((int8_t*)weights) + 213096, hardswish_16_tmp_0, hardswish_15_tmp_0_dim, conv2d_41_w_0_dim, hardswish_16_tmp_0_dim, 480, 2, 1, 1, 10);
	free(hardswish_15_tmp_0);

	float *batch_norm_42_tmp_3 = (float *)calloc(72000, sizeof(float));
	autox_conv2d(hardswish_16_tmp_0, (float*)((int8_t*)weights) + 225096, (float*)((int8_t*)weights) + 225176, batch_norm_42_tmp_3, hardswish_16_tmp_0_dim, conv2d_42_w_0_dim, batch_norm_42_tmp_3_dim, 1, 0, 1, 1, 0);
	free(hardswish_16_tmp_0);

	float *elementwise_add_8 = (float *)calloc(72000, sizeof(float));
	autox_elementwise_add(batch_norm_39_tmp_3, batch_norm_42_tmp_3, elementwise_add_8, batch_norm_39_tmp_3_dim, batch_norm_42_tmp_3_dim, elementwise_add_8_dim, -1, 4, 4, 4);
	free(batch_norm_39_tmp_3);
	free(batch_norm_42_tmp_3);

	float *hardswish_17_tmp_0 = (float *)calloc(432000, sizeof(float));
	autox_conv2d(elementwise_add_8, (float*)((int8_t*)weights) + 263576, (float*)((int8_t*)weights) + 264056, hardswish_17_tmp_0, elementwise_add_8_dim, conv2d_43_w_0_dim, hardswish_17_tmp_0_dim, 1, 0, 1, 1, 10);
	free(elementwise_add_8);

	float *hardswish_18_tmp_0 = (float *)calloc(432000, sizeof(float));
	autox_conv2d(hardswish_17_tmp_0, (float*)((int8_t*)weights) + 302456, (float*)((int8_t*)weights) + 302936, hardswish_18_tmp_0, hardswish_17_tmp_0_dim, conv2d_44_w_0_dim, hardswish_18_tmp_0_dim, 480, 2, 1, 1, 10);
	free(hardswish_17_tmp_0);

	float *batch_norm_45_tmp_3 = (float *)calloc(72000, sizeof(float));
	autox_conv2d(hardswish_18_tmp_0, (float*)((int8_t*)weights) + 314936, (float*)((int8_t*)weights) + 315016, batch_norm_45_tmp_3, hardswish_18_tmp_0_dim, conv2d_45_w_0_dim, batch_norm_45_tmp_3_dim, 1, 0, 1, 1, 0);
	free(hardswish_18_tmp_0);

	float *elementwise_add_9 = (float *)calloc(72000, sizeof(float));
	autox_elementwise_add(elementwise_add_8, batch_norm_45_tmp_3, elementwise_add_9, elementwise_add_8_dim, batch_norm_45_tmp_3_dim, elementwise_add_9_dim, -1, 4, 4, 4);
	free(elementwise_add_8);
	free(batch_norm_45_tmp_3);

	float *hardswish_19_tmp_0 = (float *)calloc(432000, sizeof(float));
	autox_conv2d(elementwise_add_9, (float*)((int8_t*)weights) + 353416, (float*)((int8_t*)weights) + 353896, hardswish_19_tmp_0, elementwise_add_9_dim, conv2d_46_w_0_dim, hardswish_19_tmp_0_dim, 1, 0, 1, 1, 10);
	free(elementwise_add_9);

	float *conv2d_250_tmp_0 = (float *)calloc(86400, sizeof(float));
	autox_conv2d(hardswish_19_tmp_0, (float*)((int8_t*)weights) + 392296, conv2d_250_tmp_0, hardswish_19_tmp_0_dim, conv2d_65_w_0_dim, conv2d_250_tmp_0_dim, 1, 0, 1, 1, 0);
	free(hardswish_19_tmp_0);

	float *pool2d_0_tmp_0 = (float *)calloc(96, sizeof(float));
	autox_pool2d(conv2d_250_tmp_0, pool2d_0_tmp_0, conv2d_250_tmp_0_dim, pool2d_0_tmp_0_dim, 1, 1, 0, 1, 1);
	free(conv2d_250_tmp_0);

	float *relu_12_tmp_0 = (float *)calloc(24, sizeof(float));
	autox_conv2d(pool2d_0_tmp_0, (float*)((int8_t*)weights) + 438376, (float*)((int8_t*)weights) + 438400, relu_12_tmp_0, pool2d_0_tmp_0_dim, conv2d_66_w_0_dim, relu_12_tmp_0_dim, 1, 0, 1, 1, 1);
	free(pool2d_0_tmp_0);

	float *conv2d_252_tmp_1 = (float *)calloc(96, sizeof(float));
	autox_conv2d(relu_12_tmp_0, (float*)((int8_t*)weights) + 440704, (float*)((int8_t*)weights) + 440800, conv2d_252_tmp_1, relu_12_tmp_0_dim, conv2d_67_w_0_dim, conv2d_252_tmp_1_dim, 1, 0, 1, 1, 0);
	free(relu_12_tmp_0);

	autox_hard_sigmoid(conv2d_252_tmp_1, conv2d_252_tmp_1_dim, 4, 0.5, 0.2);

	float *tmp_0 = (float *)calloc(86400, sizeof(float));
	autox_elementwise_mul(conv2d_250_tmp_0, hardsigmoid_0_tmp_0, tmp_0, conv2d_250_tmp_0_dim, hardsigmoid_0_tmp_0_dim, tmp_0_dim, -1, 4, 4, 4);
	free(conv2d_250_tmp_0);
	free(hardsigmoid_0_tmp_0);

	float *tmp_1 = (float *)calloc(86400, sizeof(float));
	autox_elementwise_add(conv2d_250_tmp_0, tmp_0, tmp_1, conv2d_250_tmp_0_dim, tmp_0_dim, tmp_1_dim, -1, 4, 4, 4);
	free(conv2d_250_tmp_0);
	free(tmp_0);

	float *conv2d_253_tmp_0 = (float *)calloc(345600, sizeof(float));
	autox_conv2d(elementwise_add_7, (float*)((int8_t*)weights) + 443104, conv2d_253_tmp_0, elementwise_add_7_dim, conv2d_59_w_0_dim, conv2d_253_tmp_0_dim, 1, 0, 1, 1, 0);
	free(elementwise_add_7);

	float *pool2d_1_tmp_0 = (float *)calloc(96, sizeof(float));
	autox_pool2d(conv2d_253_tmp_0, pool2d_1_tmp_0, conv2d_253_tmp_0_dim, pool2d_1_tmp_0_dim, 1, 1, 0, 1, 1);
	free(conv2d_253_tmp_0);

	float *relu_13_tmp_0 = (float *)calloc(24, sizeof(float));
	autox_conv2d(pool2d_1_tmp_0, (float*)((int8_t*)weights) + 448480, (float*)((int8_t*)weights) + 448504, relu_13_tmp_0, pool2d_1_tmp_0_dim, conv2d_60_w_0_dim, relu_13_tmp_0_dim, 1, 0, 1, 1, 1);
	free(pool2d_1_tmp_0);

	float *conv2d_255_tmp_1 = (float *)calloc(96, sizeof(float));
	autox_conv2d(relu_13_tmp_0, (float*)((int8_t*)weights) + 450808, (float*)((int8_t*)weights) + 450904, conv2d_255_tmp_1, relu_13_tmp_0_dim, conv2d_61_w_0_dim, conv2d_255_tmp_1_dim, 1, 0, 1, 1, 0);
	free(relu_13_tmp_0);

	autox_hard_sigmoid(conv2d_255_tmp_1, conv2d_255_tmp_1_dim, 4, 0.5, 0.2);

	float *tmp_2 = (float *)calloc(345600, sizeof(float));
	autox_elementwise_mul(conv2d_253_tmp_0, hardsigmoid_1_tmp_0, tmp_2, conv2d_253_tmp_0_dim, hardsigmoid_1_tmp_0_dim, tmp_2_dim, -1, 4, 4, 4);
	free(conv2d_253_tmp_0);
	free(hardsigmoid_1_tmp_0);

	float *tmp_3 = (float *)calloc(345600, sizeof(float));
	autox_elementwise_add(conv2d_253_tmp_0, tmp_2, tmp_3, conv2d_253_tmp_0_dim, tmp_2_dim, tmp_3_dim, -1, 4, 4, 4);
	free(conv2d_253_tmp_0);
	free(tmp_2);

	float *conv2d_256_tmp_0 = (float *)calloc(1382400, sizeof(float));
	autox_conv2d(elementwise_add_3, (float*)((int8_t*)weights) + 453208, conv2d_256_tmp_0, elementwise_add_3_dim, conv2d_53_w_0_dim, conv2d_256_tmp_0_dim, 1, 0, 1, 1, 0);
	free(elementwise_add_3);

	float *pool2d_2_tmp_0 = (float *)calloc(96, sizeof(float));
	autox_pool2d(conv2d_256_tmp_0, pool2d_2_tmp_0, conv2d_256_tmp_0_dim, pool2d_2_tmp_0_dim, 1, 1, 0, 1, 1);
	free(conv2d_256_tmp_0);

	float *relu_14_tmp_0 = (float *)calloc(24, sizeof(float));
	autox_conv2d(pool2d_2_tmp_0, (float*)((int8_t*)weights) + 455512, (float*)((int8_t*)weights) + 455536, relu_14_tmp_0, pool2d_2_tmp_0_dim, conv2d_54_w_0_dim, relu_14_tmp_0_dim, 1, 0, 1, 1, 1);
	free(pool2d_2_tmp_0);

	float *conv2d_258_tmp_1 = (float *)calloc(96, sizeof(float));
	autox_conv2d(relu_14_tmp_0, (float*)((int8_t*)weights) + 457840, (float*)((int8_t*)weights) + 457936, conv2d_258_tmp_1, relu_14_tmp_0_dim, conv2d_55_w_0_dim, conv2d_258_tmp_1_dim, 1, 0, 1, 1, 0);
	free(relu_14_tmp_0);

	autox_hard_sigmoid(conv2d_258_tmp_1, conv2d_258_tmp_1_dim, 4, 0.5, 0.2);

	float *tmp_4 = (float *)calloc(1382400, sizeof(float));
	autox_elementwise_mul(conv2d_256_tmp_0, hardsigmoid_2_tmp_0, tmp_4, conv2d_256_tmp_0_dim, hardsigmoid_2_tmp_0_dim, tmp_4_dim, -1, 4, 4, 4);
	free(conv2d_256_tmp_0);
	free(hardsigmoid_2_tmp_0);

	float *tmp_5 = (float *)calloc(1382400, sizeof(float));
	autox_elementwise_add(conv2d_256_tmp_0, tmp_4, tmp_5, conv2d_256_tmp_0_dim, tmp_4_dim, tmp_5_dim, -1, 4, 4, 4);
	free(conv2d_256_tmp_0);
	free(tmp_4);

	float *conv2d_259_tmp_0 = (float *)calloc(5529600, sizeof(float));
	autox_conv2d(elementwise_add_1, (float*)((int8_t*)weights) + 460240, conv2d_259_tmp_0, elementwise_add_1_dim, conv2d_47_w_0_dim, conv2d_259_tmp_0_dim, 1, 0, 1, 1, 0);
	free(elementwise_add_1);

	float *pool2d_3_tmp_0 = (float *)calloc(96, sizeof(float));
	autox_pool2d(conv2d_259_tmp_0, pool2d_3_tmp_0, conv2d_259_tmp_0_dim, pool2d_3_tmp_0_dim, 1, 1, 0, 1, 1);
	free(conv2d_259_tmp_0);

	float *relu_15_tmp_0 = (float *)calloc(24, sizeof(float));
	autox_conv2d(pool2d_3_tmp_0, (float*)((int8_t*)weights) + 461776, (float*)((int8_t*)weights) + 461800, relu_15_tmp_0, pool2d_3_tmp_0_dim, conv2d_48_w_0_dim, relu_15_tmp_0_dim, 1, 0, 1, 1, 1);
	free(pool2d_3_tmp_0);

	float *conv2d_261_tmp_1 = (float *)calloc(96, sizeof(float));
	autox_conv2d(relu_15_tmp_0, (float*)((int8_t*)weights) + 464104, (float*)((int8_t*)weights) + 464200, conv2d_261_tmp_1, relu_15_tmp_0_dim, conv2d_49_w_0_dim, conv2d_261_tmp_1_dim, 1, 0, 1, 1, 0);
	free(relu_15_tmp_0);

	autox_hard_sigmoid(conv2d_261_tmp_1, conv2d_261_tmp_1_dim, 4, 0.5, 0.2);

	float *tmp_6 = (float *)calloc(5529600, sizeof(float));
	autox_elementwise_mul(conv2d_259_tmp_0, hardsigmoid_3_tmp_0, tmp_6, conv2d_259_tmp_0_dim, hardsigmoid_3_tmp_0_dim, tmp_6_dim, -1, 4, 4, 4);
	free(conv2d_259_tmp_0);
	free(hardsigmoid_3_tmp_0);

	float *tmp_7 = (float *)calloc(5529600, sizeof(float));
	autox_elementwise_add(conv2d_259_tmp_0, tmp_6, tmp_7, conv2d_259_tmp_0_dim, tmp_6_dim, tmp_7_dim, -1, 4, 4, 4);
	free(conv2d_259_tmp_0);
	free(tmp_6);

	float *nearest_interp_v2_0_tmp_0 = (float *)calloc(345600, sizeof(float));
	autox_nearest_interp(tmp_1, nearest_interp_v2_0_tmp_0, tmp_1_dim, nearest_interp_v2_0_tmp_0_dim, 2, 0);
	free(tmp_1);

	float *tmp_8 = (float *)calloc(345600, sizeof(float));
	autox_elementwise_add(tmp_3, nearest_interp_v2_0_tmp_0, tmp_8, tmp_3_dim, nearest_interp_v2_0_tmp_0_dim, tmp_8_dim, -1, 4, 4, 4);
	free(tmp_3);
	free(nearest_interp_v2_0_tmp_0);

	float *nearest_interp_v2_1_tmp_0 = (float *)calloc(1382400, sizeof(float));
	autox_nearest_interp(tmp_8, nearest_interp_v2_1_tmp_0, tmp_8_dim, nearest_interp_v2_1_tmp_0_dim, 2, 0);
	free(tmp_8);

	float *tmp_9 = (float *)calloc(1382400, sizeof(float));
	autox_elementwise_add(tmp_5, nearest_interp_v2_1_tmp_0, tmp_9, tmp_5_dim, nearest_interp_v2_1_tmp_0_dim, tmp_9_dim, -1, 4, 4, 4);
	free(tmp_5);
	free(nearest_interp_v2_1_tmp_0);

	float *nearest_interp_v2_2_tmp_0 = (float *)calloc(5529600, sizeof(float));
	autox_nearest_interp(tmp_9, nearest_interp_v2_2_tmp_0, tmp_9_dim, nearest_interp_v2_2_tmp_0_dim, 2, 0);
	free(tmp_9);

	float *tmp_10 = (float *)calloc(5529600, sizeof(float));
	autox_elementwise_add(tmp_7, nearest_interp_v2_2_tmp_0, tmp_10, tmp_7_dim, nearest_interp_v2_2_tmp_0_dim, tmp_10_dim, -1, 4, 4, 4);
	free(tmp_7);
	free(nearest_interp_v2_2_tmp_0);

	float *conv2d_262_tmp_0 = (float *)calloc(21600, sizeof(float));
	autox_conv2d(tmp_1, (float*)((int8_t*)weights) + 466504, conv2d_262_tmp_0, tmp_1_dim, conv2d_68_w_0_dim, conv2d_262_tmp_0_dim, 1, 1, 1, 1, 0);
	free(tmp_1);

	float *pool2d_4_tmp_0 = (float *)calloc(24, sizeof(float));
	autox_pool2d(conv2d_262_tmp_0, pool2d_4_tmp_0, conv2d_262_tmp_0_dim, pool2d_4_tmp_0_dim, 1, 1, 0, 1, 1);
	free(conv2d_262_tmp_0);

	float *relu_16_tmp_0 = (float *)calloc(6, sizeof(float));
	autox_conv2d(pool2d_4_tmp_0, (float*)((int8_t*)weights) + 487240, (float*)((int8_t*)weights) + 487246, relu_16_tmp_0, pool2d_4_tmp_0_dim, conv2d_69_w_0_dim, relu_16_tmp_0_dim, 1, 0, 1, 1, 1);
	free(pool2d_4_tmp_0);

	float *conv2d_264_tmp_1 = (float *)calloc(24, sizeof(float));
	autox_conv2d(relu_16_tmp_0, (float*)((int8_t*)weights) + 487390, (float*)((int8_t*)weights) + 487414, conv2d_264_tmp_1, relu_16_tmp_0_dim, conv2d_70_w_0_dim, conv2d_264_tmp_1_dim, 1, 0, 1, 1, 0);
	free(relu_16_tmp_0);

	autox_hard_sigmoid(conv2d_264_tmp_1, conv2d_264_tmp_1_dim, 4, 0.5, 0.2);

	float *tmp_11 = (float *)calloc(21600, sizeof(float));
	autox_elementwise_mul(conv2d_262_tmp_0, hardsigmoid_4_tmp_0, tmp_11, conv2d_262_tmp_0_dim, hardsigmoid_4_tmp_0_dim, tmp_11_dim, -1, 4, 4, 4);
	free(conv2d_262_tmp_0);
	free(hardsigmoid_4_tmp_0);

	float *tmp_12 = (float *)calloc(21600, sizeof(float));
	autox_elementwise_add(conv2d_262_tmp_0, tmp_11, tmp_12, conv2d_262_tmp_0_dim, tmp_11_dim, tmp_12_dim, -1, 4, 4, 4);
	free(conv2d_262_tmp_0);
	free(tmp_11);

	float *conv2d_265_tmp_0 = (float *)calloc(86400, sizeof(float));
	autox_conv2d(tmp_8, (float*)((int8_t*)weights) + 487558, conv2d_265_tmp_0, tmp_8_dim, conv2d_62_w_0_dim, conv2d_265_tmp_0_dim, 1, 1, 1, 1, 0);
	free(tmp_8);

	float *pool2d_5_tmp_0 = (float *)calloc(24, sizeof(float));
	autox_pool2d(conv2d_265_tmp_0, pool2d_5_tmp_0, conv2d_265_tmp_0_dim, pool2d_5_tmp_0_dim, 1, 1, 0, 1, 1);
	free(conv2d_265_tmp_0);

	float *relu_17_tmp_0 = (float *)calloc(6, sizeof(float));
	autox_conv2d(pool2d_5_tmp_0, (float*)((int8_t*)weights) + 508294, (float*)((int8_t*)weights) + 508300, relu_17_tmp_0, pool2d_5_tmp_0_dim, conv2d_63_w_0_dim, relu_17_tmp_0_dim, 1, 0, 1, 1, 1);
	free(pool2d_5_tmp_0);

	float *conv2d_267_tmp_1 = (float *)calloc(24, sizeof(float));
	autox_conv2d(relu_17_tmp_0, (float*)((int8_t*)weights) + 508444, (float*)((int8_t*)weights) + 508468, conv2d_267_tmp_1, relu_17_tmp_0_dim, conv2d_64_w_0_dim, conv2d_267_tmp_1_dim, 1, 0, 1, 1, 0);
	free(relu_17_tmp_0);

	autox_hard_sigmoid(conv2d_267_tmp_1, conv2d_267_tmp_1_dim, 4, 0.5, 0.2);

	float *tmp_13 = (float *)calloc(86400, sizeof(float));
	autox_elementwise_mul(conv2d_265_tmp_0, hardsigmoid_5_tmp_0, tmp_13, conv2d_265_tmp_0_dim, hardsigmoid_5_tmp_0_dim, tmp_13_dim, -1, 4, 4, 4);
	free(conv2d_265_tmp_0);
	free(hardsigmoid_5_tmp_0);

	float *tmp_14 = (float *)calloc(86400, sizeof(float));
	autox_elementwise_add(conv2d_265_tmp_0, tmp_13, tmp_14, conv2d_265_tmp_0_dim, tmp_13_dim, tmp_14_dim, -1, 4, 4, 4);
	free(conv2d_265_tmp_0);
	free(tmp_13);

	float *conv2d_268_tmp_0 = (float *)calloc(345600, sizeof(float));
	autox_conv2d(tmp_9, (float*)((int8_t*)weights) + 508612, conv2d_268_tmp_0, tmp_9_dim, conv2d_56_w_0_dim, conv2d_268_tmp_0_dim, 1, 1, 1, 1, 0);
	free(tmp_9);

	float *pool2d_6_tmp_0 = (float *)calloc(24, sizeof(float));
	autox_pool2d(conv2d_268_tmp_0, pool2d_6_tmp_0, conv2d_268_tmp_0_dim, pool2d_6_tmp_0_dim, 1, 1, 0, 1, 1);
	free(conv2d_268_tmp_0);

	float *relu_18_tmp_0 = (float *)calloc(6, sizeof(float));
	autox_conv2d(pool2d_6_tmp_0, (float*)((int8_t*)weights) + 529348, (float*)((int8_t*)weights) + 529354, relu_18_tmp_0, pool2d_6_tmp_0_dim, conv2d_57_w_0_dim, relu_18_tmp_0_dim, 1, 0, 1, 1, 1);
	free(pool2d_6_tmp_0);

	float *conv2d_270_tmp_1 = (float *)calloc(24, sizeof(float));
	autox_conv2d(relu_18_tmp_0, (float*)((int8_t*)weights) + 529498, (float*)((int8_t*)weights) + 529522, conv2d_270_tmp_1, relu_18_tmp_0_dim, conv2d_58_w_0_dim, conv2d_270_tmp_1_dim, 1, 0, 1, 1, 0);
	free(relu_18_tmp_0);

	autox_hard_sigmoid(conv2d_270_tmp_1, conv2d_270_tmp_1_dim, 4, 0.5, 0.2);

	float *tmp_15 = (float *)calloc(345600, sizeof(float));
	autox_elementwise_mul(conv2d_268_tmp_0, hardsigmoid_6_tmp_0, tmp_15, conv2d_268_tmp_0_dim, hardsigmoid_6_tmp_0_dim, tmp_15_dim, -1, 4, 4, 4);
	free(conv2d_268_tmp_0);
	free(hardsigmoid_6_tmp_0);

	float *tmp_16 = (float *)calloc(345600, sizeof(float));
	autox_elementwise_add(conv2d_268_tmp_0, tmp_15, tmp_16, conv2d_268_tmp_0_dim, tmp_15_dim, tmp_16_dim, -1, 4, 4, 4);
	free(conv2d_268_tmp_0);
	free(tmp_15);

	float *conv2d_271_tmp_0 = (float *)calloc(1382400, sizeof(float));
	autox_conv2d(tmp_10, (float*)((int8_t*)weights) + 529666, conv2d_271_tmp_0, tmp_10_dim, conv2d_50_w_0_dim, conv2d_271_tmp_0_dim, 1, 1, 1, 1, 0);
	free(tmp_10);

	float *pool2d_7_tmp_0 = (float *)calloc(24, sizeof(float));
	autox_pool2d(conv2d_271_tmp_0, pool2d_7_tmp_0, conv2d_271_tmp_0_dim, pool2d_7_tmp_0_dim, 1, 1, 0, 1, 1);
	free(conv2d_271_tmp_0);

	float *relu_19_tmp_0 = (float *)calloc(6, sizeof(float));
	autox_conv2d(pool2d_7_tmp_0, (float*)((int8_t*)weights) + 550402, (float*)((int8_t*)weights) + 550408, relu_19_tmp_0, pool2d_7_tmp_0_dim, conv2d_51_w_0_dim, relu_19_tmp_0_dim, 1, 0, 1, 1, 1);
	free(pool2d_7_tmp_0);

	float *conv2d_273_tmp_1 = (float *)calloc(24, sizeof(float));
	autox_conv2d(relu_19_tmp_0, (float*)((int8_t*)weights) + 550552, (float*)((int8_t*)weights) + 550576, conv2d_273_tmp_1, relu_19_tmp_0_dim, conv2d_52_w_0_dim, conv2d_273_tmp_1_dim, 1, 0, 1, 1, 0);
	free(relu_19_tmp_0);

	autox_hard_sigmoid(conv2d_273_tmp_1, conv2d_273_tmp_1_dim, 4, 0.5, 0.2);

	float *tmp_17 = (float *)calloc(1382400, sizeof(float));
	autox_elementwise_mul(conv2d_271_tmp_0, hardsigmoid_7_tmp_0, tmp_17, conv2d_271_tmp_0_dim, hardsigmoid_7_tmp_0_dim, tmp_17_dim, -1, 4, 4, 4);
	free(conv2d_271_tmp_0);
	free(hardsigmoid_7_tmp_0);

	float *tmp_18 = (float *)calloc(1382400, sizeof(float));
	autox_elementwise_add(conv2d_271_tmp_0, tmp_17, tmp_18, conv2d_271_tmp_0_dim, tmp_17_dim, tmp_18_dim, -1, 4, 4, 4);
	free(conv2d_271_tmp_0);
	free(tmp_17);

	float *nearest_interp_v2_3_tmp_0 = (float *)calloc(1382400, sizeof(float));
	autox_nearest_interp(tmp_12, nearest_interp_v2_3_tmp_0, tmp_12_dim, nearest_interp_v2_3_tmp_0_dim, 8, 0);
	free(tmp_12);

	float *nearest_interp_v2_4_tmp_0 = (float *)calloc(1382400, sizeof(float));
	autox_nearest_interp(tmp_14, nearest_interp_v2_4_tmp_0, tmp_14_dim, nearest_interp_v2_4_tmp_0_dim, 4, 0);
	free(tmp_14);

	float *nearest_interp_v2_5_tmp_0 = (float *)calloc(1382400, sizeof(float));
	autox_nearest_interp(tmp_16, nearest_interp_v2_5_tmp_0, tmp_16_dim, nearest_interp_v2_5_tmp_0_dim, 2, 0);
	free(tmp_16);

	float* p_122[] = {nearest_interp_v2_3_tmp_0, nearest_interp_v2_4_tmp_0, nearest_interp_v2_5_tmp_0, tmp_18, };
	uint16_t* p_122_dim[] = {nearest_interp_v2_3_tmp_0_dim, nearest_interp_v2_4_tmp_0_dim, nearest_interp_v2_5_tmp_0_dim, tmp_18_dim, };
	float *concat_0_tmp_0 = (float *)calloc(5529600, sizeof(float));
	autox_concat(p_122, concat_0_tmp_0, p_122_dim, concat_0_tmp_0_dim, 1, 4, 4);
	free(nearest_interp_v2_3_tmp_0);
	free(nearest_interp_v2_4_tmp_0);
	free(nearest_interp_v2_5_tmp_0);
	free(tmp_18);

	float *batch_norm_47_tmp_4 = (float *)calloc(1382400, sizeof(float));
	autox_conv2d(concat_0_tmp_0, (float*)((int8_t*)weights) + 550720, (float*)((int8_t*)weights) + 550744, batch_norm_47_tmp_4, concat_0_tmp_0_dim, conv2d_71_w_0_dim, batch_norm_47_tmp_4_dim, 1, 1, 1, 1, 1);
	free(concat_0_tmp_0);

	float *batch_norm_48_tmp_4 = (float *)calloc(5529600, sizeof(float));
	autox_conv2d_transpose(batch_norm_47_tmp_4, (float*)((int8_t*)weights) + 571480, (float*)((int8_t*)weights) + 571504, batch_norm_48_tmp_4, batch_norm_47_tmp_4_dim, conv2d_transpose_0_w_0_dim, batch_norm_48_tmp_4_dim, 1, 0, 2, 1, 1);
	free(batch_norm_47_tmp_4);

	float *elementwise_add_11_tmp_0 = (float *)calloc(921600, sizeof(float));
	autox_conv2d_transpose(batch_norm_48_tmp_4, (float*)((int8_t*)weights) + 573808, (float*)((int8_t*)weights) + 573809, elementwise_add_11_tmp_0, batch_norm_48_tmp_4_dim, conv2d_transpose_1_w_0_dim, elementwise_add_11_tmp_0_dim, 1, 0, 2, 1, 0);
	free(batch_norm_48_tmp_4);

	autox_sigmoid(elementwise_add_11_tmp_0, elementwise_add_11_tmp_0_dim, 4);

}