#include "../include/autox_models.h"

void picodet_xs_320(const uint8_t* image, const uint16_t ssize_h, const uint16_t ssize_w, float* weights, float* Out)
{
	uint16_t scale_factor_dim[] = { 1, 2 };
	uint16_t image_dim[] = { 1, 3, 320, 320 };
	uint16_t batch_norm2d_0_b_0_dim[] = { 8 };
	uint16_t conv2d_0_w_0_dim[] = { 8, 3, 3, 3 };
	uint16_t relu6_0_tmp_0_dim[] = { 1, 8, 160, 160 };
	uint16_t batch_norm2d_1_b_0_dim[] = { 8 };
	uint16_t conv2d_1_w_0_dim[] = { 8, 1, 3, 3 };
	uint16_t relu6_1_tmp_0_dim[] = { 1, 8, 160, 160 };
	uint16_t batch_norm2d_2_b_0_dim[] = { 16 };
	uint16_t conv2d_2_w_0_dim[] = { 16, 8, 1, 1 };
	uint16_t relu6_2_tmp_0_dim[] = { 1, 16, 160, 160 };
	uint16_t batch_norm2d_3_b_0_dim[] = { 16 };
	uint16_t conv2d_3_w_0_dim[] = { 16, 1, 3, 3 };
	uint16_t relu6_3_tmp_0_dim[] = { 1, 16, 80, 80 };
	uint16_t batch_norm2d_4_b_0_dim[] = { 24 };
	uint16_t conv2d_4_w_0_dim[] = { 24, 16, 1, 1 };
	uint16_t relu6_4_tmp_0_dim[] = { 1, 24, 80, 80 };
	uint16_t batch_norm2d_5_b_0_dim[] = { 24 };
	uint16_t conv2d_5_w_0_dim[] = { 24, 1, 3, 3 };
	uint16_t relu6_5_tmp_0_dim[] = { 1, 24, 80, 80 };
	uint16_t batch_norm2d_6_b_0_dim[] = { 24 };
	uint16_t conv2d_6_w_0_dim[] = { 24, 24, 1, 1 };
	uint16_t relu6_6_tmp_0_dim[] = { 1, 24, 80, 80 };
	uint16_t batch_norm2d_7_b_0_dim[] = { 24 };
	uint16_t conv2d_7_w_0_dim[] = { 24, 1, 3, 3 };
	uint16_t relu6_7_tmp_0_dim[] = { 1, 24, 40, 40 };
	uint16_t batch_norm2d_8_b_0_dim[] = { 48 };
	uint16_t conv2d_8_w_0_dim[] = { 48, 24, 1, 1 };
	uint16_t relu6_8_tmp_0_dim[] = { 1, 48, 40, 40 };
	uint16_t batch_norm2d_9_b_0_dim[] = { 48 };
	uint16_t conv2d_9_w_0_dim[] = { 48, 1, 3, 3 };
	uint16_t relu6_9_tmp_0_dim[] = { 1, 48, 40, 40 };
	uint16_t batch_norm2d_10_b_0_dim[] = { 48 };
	uint16_t conv2d_10_w_0_dim[] = { 48, 48, 1, 1 };
	uint16_t relu6_10_tmp_0_dim[] = { 1, 48, 40, 40 };
	uint16_t batch_norm2d_11_b_0_dim[] = { 48 };
	uint16_t conv2d_11_w_0_dim[] = { 48, 1, 3, 3 };
	uint16_t relu6_11_tmp_0_dim[] = { 1, 48, 20, 20 };
	uint16_t batch_norm2d_12_b_0_dim[] = { 88 };
	uint16_t conv2d_12_w_0_dim[] = { 88, 48, 1, 1 };
	uint16_t relu6_12_tmp_0_dim[] = { 1, 88, 20, 20 };
	uint16_t batch_norm2d_13_b_0_dim[] = { 88 };
	uint16_t conv2d_13_w_0_dim[] = { 88, 1, 5, 5 };
	uint16_t relu6_13_tmp_0_dim[] = { 1, 88, 20, 20 };
	uint16_t batch_norm2d_14_b_0_dim[] = { 88 };
	uint16_t conv2d_14_w_0_dim[] = { 88, 88, 1, 1 };
	uint16_t relu6_14_tmp_0_dim[] = { 1, 88, 20, 20 };
	uint16_t batch_norm2d_15_b_0_dim[] = { 88 };
	uint16_t conv2d_15_w_0_dim[] = { 88, 1, 5, 5 };
	uint16_t relu6_15_tmp_0_dim[] = { 1, 88, 20, 20 };
	uint16_t batch_norm2d_16_b_0_dim[] = { 88 };
	uint16_t conv2d_16_w_0_dim[] = { 88, 88, 1, 1 };
	uint16_t relu6_16_tmp_0_dim[] = { 1, 88, 20, 20 };
	uint16_t batch_norm2d_17_b_0_dim[] = { 88 };
	uint16_t conv2d_17_w_0_dim[] = { 88, 1, 5, 5 };
	uint16_t relu6_17_tmp_0_dim[] = { 1, 88, 20, 20 };
	uint16_t batch_norm2d_18_b_0_dim[] = { 88 };
	uint16_t conv2d_18_w_0_dim[] = { 88, 88, 1, 1 };
	uint16_t relu6_18_tmp_0_dim[] = { 1, 88, 20, 20 };
	uint16_t batch_norm2d_19_b_0_dim[] = { 88 };
	uint16_t conv2d_19_w_0_dim[] = { 88, 1, 5, 5 };
	uint16_t relu6_19_tmp_0_dim[] = { 1, 88, 20, 20 };
	uint16_t batch_norm2d_20_b_0_dim[] = { 88 };
	uint16_t conv2d_20_w_0_dim[] = { 88, 88, 1, 1 };
	uint16_t relu6_20_tmp_0_dim[] = { 1, 88, 20, 20 };
	uint16_t batch_norm2d_21_b_0_dim[] = { 88 };
	uint16_t conv2d_21_w_0_dim[] = { 88, 1, 5, 5 };
	uint16_t relu6_21_tmp_0_dim[] = { 1, 88, 20, 20 };
	uint16_t batch_norm2d_22_b_0_dim[] = { 88 };
	uint16_t conv2d_22_w_0_dim[] = { 88, 88, 1, 1 };
	uint16_t relu6_22_tmp_0_dim[] = { 1, 88, 20, 20 };
	uint16_t batch_norm2d_23_b_0_dim[] = { 88 };
	uint16_t conv2d_23_w_0_dim[] = { 88, 1, 5, 5 };
	uint16_t relu6_23_tmp_0_dim[] = { 1, 88, 10, 10 };
	uint16_t pool2d_0_tmp_0_dim[] = { 1, 88, 1, 1 };
	uint16_t conv2d_24_b_0_dim[] = { 22 };
	uint16_t conv2d_24_w_0_dim[] = { 22, 88, 1, 1 };
	uint16_t relu_0_tmp_0_dim[] = { 1, 22, 1, 1 };
	uint16_t conv2d_113_tmp_1_dim[] = { 1, 88, 1, 1 };
	uint16_t conv2d_25_b_0_dim[] = { 88 };
	uint16_t conv2d_25_w_0_dim[] = { 88, 22, 1, 1 };
	uint16_t hardsigmoid_2_tmp_0_dim[] = { 1, 88, 1, 1 };
	uint16_t elementwise_mul_0_dim[] = { 1, 88, 10, 10 };
	uint16_t batch_norm2d_24_b_0_dim[] = { 176 };
	uint16_t conv2d_26_w_0_dim[] = { 176, 88, 1, 1 };
	uint16_t relu6_24_tmp_0_dim[] = { 1, 176, 10, 10 };
	uint16_t batch_norm2d_25_b_0_dim[] = { 176 };
	uint16_t conv2d_27_w_0_dim[] = { 176, 1, 5, 5 };
	uint16_t relu6_25_tmp_0_dim[] = { 1, 176, 10, 10 };
	uint16_t pool2d_1_tmp_0_dim[] = { 1, 176, 1, 1 };
	uint16_t conv2d_28_b_0_dim[] = { 44 };
	uint16_t conv2d_28_w_0_dim[] = { 44, 176, 1, 1 };
	uint16_t relu_1_tmp_0_dim[] = { 1, 44, 1, 1 };
	uint16_t conv2d_116_tmp_1_dim[] = { 1, 176, 1, 1 };
	uint16_t conv2d_29_b_0_dim[] = { 176 };
	uint16_t conv2d_29_w_0_dim[] = { 176, 44, 1, 1 };
	uint16_t hardsigmoid_3_tmp_0_dim[] = { 1, 176, 1, 1 };
	uint16_t elementwise_mul_1_dim[] = { 1, 176, 10, 10 };
	uint16_t batch_norm2d_26_b_0_dim[] = { 176 };
	uint16_t conv2d_30_w_0_dim[] = { 176, 176, 1, 1 };
	uint16_t relu6_26_tmp_0_dim[] = { 1, 176, 10, 10 };
	uint16_t batch_norm2d_29_b_0_dim[] = { 96 };
	uint16_t conv2d_33_w_0_dim[] = { 96, 176, 1, 1 };
	uint16_t relu6_29_tmp_0_dim[] = { 1, 96, 10, 10 };
	uint16_t nearest_interp_v2_0_tmp_0_dim[] = { 1, 96, 20, 20 };
	uint16_t batch_norm2d_28_b_0_dim[] = { 96 };
	uint16_t conv2d_32_w_0_dim[] = { 96, 88, 1, 1 };
	uint16_t relu6_28_tmp_0_dim[] = { 1, 96, 20, 20 };
	uint16_t concat_0_tmp_0_dim[] = { 1, 192, 20, 20 };
	uint16_t batch_norm2d_34_b_0_dim[] = { 192 };
	uint16_t conv2d_38_w_0_dim[] = { 192, 1, 5, 5 };
	uint16_t relu6_30_tmp_0_dim[] = { 1, 192, 20, 20 };
	uint16_t batch_norm2d_35_b_0_dim[] = { 192 };
	uint16_t conv2d_39_w_0_dim[] = { 192, 192, 1, 1 };
	uint16_t relu6_31_tmp_0_dim[] = { 1, 192, 20, 20 };
	uint16_t batch_norm2d_36_b_0_dim[] = { 192 };
	uint16_t conv2d_40_w_0_dim[] = { 192, 1, 5, 5 };
	uint16_t relu6_32_tmp_0_dim[] = { 1, 192, 20, 20 };
	uint16_t batch_norm2d_37_b_0_dim[] = { 96 };
	uint16_t conv2d_41_w_0_dim[] = { 96, 192, 1, 1 };
	uint16_t relu6_33_tmp_0_dim[] = { 1, 96, 20, 20 };
	uint16_t nearest_interp_v2_1_tmp_0_dim[] = { 1, 96, 40, 40 };
	uint16_t batch_norm2d_27_b_0_dim[] = { 96 };
	uint16_t conv2d_31_w_0_dim[] = { 96, 48, 1, 1 };
	uint16_t relu6_27_tmp_0_dim[] = { 1, 96, 40, 40 };
	uint16_t concat_1_tmp_0_dim[] = { 1, 192, 40, 40 };
	uint16_t batch_norm2d_38_b_0_dim[] = { 192 };
	uint16_t conv2d_42_w_0_dim[] = { 192, 1, 5, 5 };
	uint16_t relu6_34_tmp_0_dim[] = { 1, 192, 40, 40 };
	uint16_t batch_norm2d_39_b_0_dim[] = { 192 };
	uint16_t conv2d_43_w_0_dim[] = { 192, 192, 1, 1 };
	uint16_t relu6_35_tmp_0_dim[] = { 1, 192, 40, 40 };
	uint16_t batch_norm2d_40_b_0_dim[] = { 192 };
	uint16_t conv2d_44_w_0_dim[] = { 192, 1, 5, 5 };
	uint16_t relu6_36_tmp_0_dim[] = { 1, 192, 40, 40 };
	uint16_t batch_norm2d_41_b_0_dim[] = { 96 };
	uint16_t conv2d_45_w_0_dim[] = { 96, 192, 1, 1 };
	uint16_t relu6_37_tmp_0_dim[] = { 1, 96, 40, 40 };
	uint16_t batch_norm2d_42_b_0_dim[] = { 96 };
	uint16_t conv2d_46_w_0_dim[] = { 96, 1, 5, 5 };
	uint16_t relu6_38_tmp_0_dim[] = { 1, 96, 20, 20 };
	uint16_t batch_norm2d_43_b_0_dim[] = { 96 };
	uint16_t conv2d_47_w_0_dim[] = { 96, 96, 1, 1 };
	uint16_t relu6_39_tmp_0_dim[] = { 1, 96, 20, 20 };
	uint16_t concat_2_tmp_0_dim[] = { 1, 192, 20, 20 };
	uint16_t batch_norm2d_44_b_0_dim[] = { 192 };
	uint16_t conv2d_48_w_0_dim[] = { 192, 1, 5, 5 };
	uint16_t relu6_40_tmp_0_dim[] = { 1, 192, 20, 20 };
	uint16_t batch_norm2d_45_b_0_dim[] = { 192 };
	uint16_t conv2d_49_w_0_dim[] = { 192, 192, 1, 1 };
	uint16_t relu6_41_tmp_0_dim[] = { 1, 192, 20, 20 };
	uint16_t batch_norm2d_46_b_0_dim[] = { 192 };
	uint16_t conv2d_50_w_0_dim[] = { 192, 1, 5, 5 };
	uint16_t relu6_42_tmp_0_dim[] = { 1, 192, 20, 20 };
	uint16_t batch_norm2d_47_b_0_dim[] = { 96 };
	uint16_t conv2d_51_w_0_dim[] = { 96, 192, 1, 1 };
	uint16_t relu6_43_tmp_0_dim[] = { 1, 96, 20, 20 };
	uint16_t batch_norm2d_48_b_0_dim[] = { 96 };
	uint16_t conv2d_52_w_0_dim[] = { 96, 1, 5, 5 };
	uint16_t relu6_44_tmp_0_dim[] = { 1, 96, 10, 10 };
	uint16_t batch_norm2d_49_b_0_dim[] = { 96 };
	uint16_t conv2d_53_w_0_dim[] = { 96, 96, 1, 1 };
	uint16_t relu6_45_tmp_0_dim[] = { 1, 96, 10, 10 };
	uint16_t concat_3_tmp_0_dim[] = { 1, 192, 10, 10 };
	uint16_t batch_norm2d_50_b_0_dim[] = { 192 };
	uint16_t conv2d_54_w_0_dim[] = { 192, 1, 5, 5 };
	uint16_t relu6_46_tmp_0_dim[] = { 1, 192, 10, 10 };
	uint16_t batch_norm2d_51_b_0_dim[] = { 192 };
	uint16_t conv2d_55_w_0_dim[] = { 192, 192, 1, 1 };
	uint16_t relu6_47_tmp_0_dim[] = { 1, 192, 10, 10 };
	uint16_t batch_norm2d_52_b_0_dim[] = { 192 };
	uint16_t conv2d_56_w_0_dim[] = { 192, 1, 5, 5 };
	uint16_t relu6_48_tmp_0_dim[] = { 1, 192, 10, 10 };
	uint16_t batch_norm2d_53_b_0_dim[] = { 96 };
	uint16_t conv2d_57_w_0_dim[] = { 96, 192, 1, 1 };
	uint16_t relu6_49_tmp_0_dim[] = { 1, 96, 10, 10 };
	uint16_t batch_norm2d_32_b_0_dim[] = { 96 };
	uint16_t conv2d_36_w_0_dim[] = { 96, 1, 5, 5 };
	uint16_t relu6_52_tmp_0_dim[] = { 1, 96, 5, 5 };
	uint16_t batch_norm2d_33_b_0_dim[] = { 96 };
	uint16_t conv2d_37_w_0_dim[] = { 96, 96, 1, 1 };
	uint16_t relu6_53_tmp_0_dim[] = { 1, 96, 5, 5 };
	uint16_t batch_norm2d_30_b_0_dim[] = { 96 };
	uint16_t conv2d_34_w_0_dim[] = { 96, 1, 5, 5 };
	uint16_t relu6_50_tmp_0_dim[] = { 1, 96, 5, 5 };
	uint16_t batch_norm2d_31_b_0_dim[] = { 96 };
	uint16_t conv2d_35_w_0_dim[] = { 96, 96, 1, 1 };
	uint16_t relu6_51_tmp_0_dim[] = { 1, 96, 5, 5 };
	uint16_t tmp_0_dim[] = { 1, 96, 5, 5 };
	uint16_t batch_norm2d_54_b_0_dim[] = { 96 };
	uint16_t conv2d_58_w_0_dim[] = { 96, 1, 5, 5 };
	uint16_t relu6_54_tmp_0_dim[] = { 1, 96, 40, 40 };
	uint16_t batch_norm2d_55_b_0_dim[] = { 96 };
	uint16_t conv2d_59_w_0_dim[] = { 96, 96, 1, 1 };
	uint16_t relu6_55_tmp_0_dim[] = { 1, 96, 40, 40 };
	uint16_t batch_norm2d_56_b_0_dim[] = { 96 };
	uint16_t conv2d_60_w_0_dim[] = { 96, 1, 5, 5 };
	uint16_t relu6_56_tmp_0_dim[] = { 1, 96, 40, 40 };
	uint16_t batch_norm2d_57_b_0_dim[] = { 96 };
	uint16_t conv2d_61_w_0_dim[] = { 96, 96, 1, 1 };
	uint16_t relu6_57_tmp_0_dim[] = { 1, 96, 40, 40 };
	uint16_t pool2d_2_tmp_0_dim[] = { 1, 96, 1, 1 };
	uint16_t conv2d_135_tmp_1_dim[] = { 1, 96, 1, 1 };
	uint16_t conv2d_62_b_0_dim[] = { 96 };
	uint16_t conv2d_62_w_0_dim[] = { 96, 96, 1, 1 };
	uint16_t sigmoid_0_tmp_0_dim[] = { 1, 96, 1, 1 };
	uint16_t tmp_1_dim[] = { 1, 96, 40, 40 };
	uint16_t batch_norm2d_58_b_0_dim[] = { 96 };
	uint16_t conv2d_63_w_0_dim[] = { 96, 96, 1, 1 };
	uint16_t relu6_58_tmp_0_dim[] = { 1, 96, 40, 40 };
	uint16_t conv2d_137_tmp_1_dim[] = { 1, 25, 40, 40 };
	uint16_t conv2d_84_b_0_dim[] = { 25 };
	uint16_t conv2d_84_w_0_dim[] = { 25, 96, 1, 1 };
	uint16_t conv2d_138_tmp_1_dim[] = { 1, 32, 40, 40 };
	uint16_t conv2d_85_b_0_dim[] = { 32 };
	uint16_t conv2d_85_w_0_dim[] = { 32, 96, 1, 1 };
	uint16_t batch_norm2d_74_b_0_dim[] = { 1 };
	uint16_t conv2d_86_w_0_dim[] = { 1, 96, 5, 5 };
	uint16_t relu6_59_tmp_0_dim[] = { 1, 1, 40, 40 };
	uint16_t batch_norm2d_75_b_0_dim[] = { 1 };
	uint16_t batch_norm_60_tmp_2_dim[] = { 1, 1, 40, 40 };
	uint16_t conv2d_87_w_0_dim[] = { 1, 1, 1, 1 };
	uint16_t sigmoid_1_tmp_0_dim[] = { 1, 1, 40, 40 };
	uint16_t sigmoid_2_tmp_0_dim[] = { 1, 25, 40, 40 };
	uint16_t tmp_2_dim[] = { 1, 25, 40, 40 };
	uint16_t tmp_3_dim[] = { 1, 25, 40, 40 };
	uint16_t sqrt_0_tmp_0_dim[] = { 1, 25, 40, 40 };
	uint16_t reshape2_0_tmp_0_dim[] = { 1, 25, 1600 };
	uint16_t reshape2_0_tmp_1_dim[] = { 0, 1, 25, 40, 40 };
	uint16_t transpose_0_tmp_0_dim[] = { 1, 40, 40, 32 };
	uint16_t transpose_0_tmp_1_dim[] = { 0, 1, 32, 40, 40 };
	uint16_t reshape2_1_tmp_0_dim[] = { 1, 8 };
	uint16_t reshape2_1_tmp_1_dim[] = { 0, 1, 40, 40, 32 };
	uint16_t softmax_0_tmp_0_dim[] = { 1, 8 };
	uint16_t eager_tmp_4_dim[] = { 8 };
	uint16_t linear_0_tmp_0_dim[] = { 1 };
	uint16_t reshape2_2_tmp_0_dim[] = { 1, 1600, 4 };
	uint16_t reshape2_2_tmp_1_dim[] = { 0, 1 };
	uint16_t batch_norm2d_59_b_0_dim[] = { 96 };
	uint16_t conv2d_64_w_0_dim[] = { 96, 1, 5, 5 };
	uint16_t relu6_60_tmp_0_dim[] = { 1, 96, 20, 20 };
	uint16_t batch_norm2d_60_b_0_dim[] = { 96 };
	uint16_t conv2d_65_w_0_dim[] = { 96, 96, 1, 1 };
	uint16_t relu6_61_tmp_0_dim[] = { 1, 96, 20, 20 };
	uint16_t batch_norm2d_61_b_0_dim[] = { 96 };
	uint16_t conv2d_66_w_0_dim[] = { 96, 1, 5, 5 };
	uint16_t relu6_62_tmp_0_dim[] = { 1, 96, 20, 20 };
	uint16_t batch_norm2d_62_b_0_dim[] = { 96 };
	uint16_t conv2d_67_w_0_dim[] = { 96, 96, 1, 1 };
	uint16_t relu6_63_tmp_0_dim[] = { 1, 96, 20, 20 };
	uint16_t pool2d_3_tmp_0_dim[] = { 1, 96, 1, 1 };
	uint16_t conv2d_143_tmp_1_dim[] = { 1, 96, 1, 1 };
	uint16_t conv2d_68_b_0_dim[] = { 96 };
	uint16_t conv2d_68_w_0_dim[] = { 96, 96, 1, 1 };
	uint16_t sigmoid_3_tmp_0_dim[] = { 1, 96, 1, 1 };
	uint16_t tmp_4_dim[] = { 1, 96, 20, 20 };
	uint16_t batch_norm2d_63_b_0_dim[] = { 96 };
	uint16_t conv2d_69_w_0_dim[] = { 96, 96, 1, 1 };
	uint16_t relu6_64_tmp_0_dim[] = { 1, 96, 20, 20 };
	uint16_t conv2d_145_tmp_1_dim[] = { 1, 25, 20, 20 };
	uint16_t conv2d_88_b_0_dim[] = { 25 };
	uint16_t conv2d_88_w_0_dim[] = { 25, 96, 1, 1 };
	uint16_t conv2d_146_tmp_1_dim[] = { 1, 32, 20, 20 };
	uint16_t conv2d_89_b_0_dim[] = { 32 };
	uint16_t conv2d_89_w_0_dim[] = { 32, 96, 1, 1 };
	uint16_t batch_norm2d_76_b_0_dim[] = { 1 };
	uint16_t conv2d_90_w_0_dim[] = { 1, 96, 5, 5 };
	uint16_t relu6_65_tmp_0_dim[] = { 1, 1, 20, 20 };
	uint16_t batch_norm2d_77_b_0_dim[] = { 1 };
	uint16_t batch_norm_67_tmp_2_dim[] = { 1, 1, 20, 20 };
	uint16_t conv2d_91_w_0_dim[] = { 1, 1, 1, 1 };
	uint16_t sigmoid_4_tmp_0_dim[] = { 1, 1, 20, 20 };
	uint16_t sigmoid_5_tmp_0_dim[] = { 1, 25, 20, 20 };
	uint16_t tmp_5_dim[] = { 1, 25, 20, 20 };
	uint16_t tmp_6_dim[] = { 1, 25, 20, 20 };
	uint16_t sqrt_1_tmp_0_dim[] = { 1, 25, 20, 20 };
	uint16_t reshape2_3_tmp_0_dim[] = { 1, 25, 400 };
	uint16_t reshape2_3_tmp_1_dim[] = { 0, 1, 25, 20, 20 };
	uint16_t transpose_1_tmp_0_dim[] = { 1, 20, 20, 32 };
	uint16_t transpose_1_tmp_1_dim[] = { 0, 1, 32, 20, 20 };
	uint16_t reshape2_4_tmp_0_dim[] = { 1, 8 };
	uint16_t reshape2_4_tmp_1_dim[] = { 0, 1, 20, 20, 32 };
	uint16_t softmax_1_tmp_0_dim[] = { 1, 8 };
	uint16_t linear_1_tmp_0_dim[] = { 1 };
	uint16_t reshape2_5_tmp_0_dim[] = { 1, 400, 4 };
	uint16_t reshape2_5_tmp_1_dim[] = { 0, 1 };
	uint16_t batch_norm2d_64_b_0_dim[] = { 96 };
	uint16_t conv2d_70_w_0_dim[] = { 96, 1, 5, 5 };
	uint16_t relu6_66_tmp_0_dim[] = { 1, 96, 10, 10 };
	uint16_t batch_norm2d_65_b_0_dim[] = { 96 };
	uint16_t conv2d_71_w_0_dim[] = { 96, 96, 1, 1 };
	uint16_t relu6_67_tmp_0_dim[] = { 1, 96, 10, 10 };
	uint16_t batch_norm2d_66_b_0_dim[] = { 96 };
	uint16_t conv2d_72_w_0_dim[] = { 96, 1, 5, 5 };
	uint16_t relu6_68_tmp_0_dim[] = { 1, 96, 10, 10 };
	uint16_t batch_norm2d_67_b_0_dim[] = { 96 };
	uint16_t conv2d_73_w_0_dim[] = { 96, 96, 1, 1 };
	uint16_t relu6_69_tmp_0_dim[] = { 1, 96, 10, 10 };
	uint16_t pool2d_4_tmp_0_dim[] = { 1, 96, 1, 1 };
	uint16_t conv2d_151_tmp_1_dim[] = { 1, 96, 1, 1 };
	uint16_t conv2d_74_b_0_dim[] = { 96 };
	uint16_t conv2d_74_w_0_dim[] = { 96, 96, 1, 1 };
	uint16_t sigmoid_6_tmp_0_dim[] = { 1, 96, 1, 1 };
	uint16_t tmp_7_dim[] = { 1, 96, 10, 10 };
	uint16_t batch_norm2d_68_b_0_dim[] = { 96 };
	uint16_t conv2d_75_w_0_dim[] = { 96, 96, 1, 1 };
	uint16_t relu6_70_tmp_0_dim[] = { 1, 96, 10, 10 };
	uint16_t conv2d_153_tmp_1_dim[] = { 1, 25, 10, 10 };
	uint16_t conv2d_92_b_0_dim[] = { 25 };
	uint16_t conv2d_92_w_0_dim[] = { 25, 96, 1, 1 };
	uint16_t conv2d_154_tmp_1_dim[] = { 1, 32, 10, 10 };
	uint16_t conv2d_93_b_0_dim[] = { 32 };
	uint16_t conv2d_93_w_0_dim[] = { 32, 96, 1, 1 };
	uint16_t batch_norm2d_78_b_0_dim[] = { 1 };
	uint16_t conv2d_94_w_0_dim[] = { 1, 96, 5, 5 };
	uint16_t relu6_71_tmp_0_dim[] = { 1, 1, 10, 10 };
	uint16_t batch_norm2d_79_b_0_dim[] = { 1 };
	uint16_t batch_norm_74_tmp_2_dim[] = { 1, 1, 10, 10 };
	uint16_t conv2d_95_w_0_dim[] = { 1, 1, 1, 1 };
	uint16_t sigmoid_7_tmp_0_dim[] = { 1, 1, 10, 10 };
	uint16_t sigmoid_8_tmp_0_dim[] = { 1, 25, 10, 10 };
	uint16_t tmp_8_dim[] = { 1, 25, 10, 10 };
	uint16_t tmp_9_dim[] = { 1, 25, 10, 10 };
	uint16_t sqrt_2_tmp_0_dim[] = { 1, 25, 10, 10 };
	uint16_t reshape2_6_tmp_0_dim[] = { 1, 25, 100 };
	uint16_t reshape2_6_tmp_1_dim[] = { 0, 1, 25, 10, 10 };
	uint16_t transpose_2_tmp_0_dim[] = { 1, 10, 10, 32 };
	uint16_t transpose_2_tmp_1_dim[] = { 0, 1, 32, 10, 10 };
	uint16_t reshape2_7_tmp_0_dim[] = { 1, 8 };
	uint16_t reshape2_7_tmp_1_dim[] = { 0, 1, 10, 10, 32 };
	uint16_t softmax_2_tmp_0_dim[] = { 1, 8 };
	uint16_t linear_2_tmp_0_dim[] = { 1 };
	uint16_t reshape2_8_tmp_0_dim[] = { 1, 100, 4 };
	uint16_t reshape2_8_tmp_1_dim[] = { 0, 1 };
	uint16_t batch_norm2d_69_b_0_dim[] = { 96 };
	uint16_t conv2d_76_w_0_dim[] = { 96, 1, 5, 5 };
	uint16_t relu6_72_tmp_0_dim[] = { 1, 96, 5, 5 };
	uint16_t batch_norm2d_70_b_0_dim[] = { 96 };
	uint16_t conv2d_77_w_0_dim[] = { 96, 96, 1, 1 };
	uint16_t relu6_73_tmp_0_dim[] = { 1, 96, 5, 5 };
	uint16_t batch_norm2d_71_b_0_dim[] = { 96 };
	uint16_t conv2d_78_w_0_dim[] = { 96, 1, 5, 5 };
	uint16_t relu6_74_tmp_0_dim[] = { 1, 96, 5, 5 };
	uint16_t batch_norm2d_72_b_0_dim[] = { 96 };
	uint16_t conv2d_79_w_0_dim[] = { 96, 96, 1, 1 };
	uint16_t relu6_75_tmp_0_dim[] = { 1, 96, 5, 5 };
	uint16_t pool2d_5_tmp_0_dim[] = { 1, 96, 1, 1 };
	uint16_t conv2d_159_tmp_1_dim[] = { 1, 96, 1, 1 };
	uint16_t conv2d_80_b_0_dim[] = { 96 };
	uint16_t conv2d_80_w_0_dim[] = { 96, 96, 1, 1 };
	uint16_t sigmoid_9_tmp_0_dim[] = { 1, 96, 1, 1 };
	uint16_t tmp_10_dim[] = { 1, 96, 5, 5 };
	uint16_t batch_norm2d_73_b_0_dim[] = { 96 };
	uint16_t conv2d_81_w_0_dim[] = { 96, 96, 1, 1 };
	uint16_t relu6_76_tmp_0_dim[] = { 1, 96, 5, 5 };
	uint16_t conv2d_161_tmp_1_dim[] = { 1, 25, 5, 5 };
	uint16_t conv2d_96_b_0_dim[] = { 25 };
	uint16_t conv2d_96_w_0_dim[] = { 25, 96, 1, 1 };
	uint16_t conv2d_162_tmp_1_dim[] = { 1, 32, 5, 5 };
	uint16_t conv2d_97_b_0_dim[] = { 32 };
	uint16_t conv2d_97_w_0_dim[] = { 32, 96, 1, 1 };
	uint16_t batch_norm2d_80_b_0_dim[] = { 1 };
	uint16_t conv2d_98_w_0_dim[] = { 1, 96, 5, 5 };
	uint16_t relu6_77_tmp_0_dim[] = { 1, 1, 5, 5 };
	uint16_t batch_norm2d_81_b_0_dim[] = { 1 };
	uint16_t batch_norm_81_tmp_2_dim[] = { 1, 1, 5, 5 };
	uint16_t conv2d_99_w_0_dim[] = { 1, 1, 1, 1 };
	uint16_t sigmoid_10_tmp_0_dim[] = { 1, 1, 5, 5 };
	uint16_t sigmoid_11_tmp_0_dim[] = { 1, 25, 5, 5 };
	uint16_t tmp_11_dim[] = { 1, 25, 5, 5 };
	uint16_t tmp_12_dim[] = { 1, 25, 5, 5 };
	uint16_t sqrt_3_tmp_0_dim[] = { 1, 25, 5, 5 };
	uint16_t reshape2_9_tmp_0_dim[] = { 1, 25, 25 };
	uint16_t reshape2_9_tmp_1_dim[] = { 0, 1, 25, 5, 5 };
	uint16_t transpose_3_tmp_0_dim[] = { 1, 5, 5, 32 };
	uint16_t transpose_3_tmp_1_dim[] = { 0, 1, 32, 5, 5 };
	uint16_t reshape2_10_tmp_0_dim[] = { 1, 8 };
	uint16_t reshape2_10_tmp_1_dim[] = { 0, 1, 5, 5, 32 };
	uint16_t softmax_3_tmp_0_dim[] = { 1, 8 };
	uint16_t linear_3_tmp_0_dim[] = { 1 };
	uint16_t reshape2_11_tmp_0_dim[] = { 1, 25, 4 };
	uint16_t reshape2_11_tmp_1_dim[] = { 0, 1 };
	uint16_t concat_4_tmp_0_dim[] = { 1, 25, 2125 };
	uint16_t concat_5_tmp_0_dim[] = { 1, 2125, 4 };
	uint16_t split_0_tmp_0_dim[] = { 1, 2125, 2 };
	uint16_t split_0_tmp_1_dim[] = { 1, 2125, 2 };
	uint16_t tmp_13_dim[] = { 1, 2125, 2 };
	uint16_t eager_tmp_0_dim[] = { 2125, 2 };
	uint16_t tmp_14_dim[] = { 1, 2125, 2 };
	uint16_t tmp_15_dim[] = { 1, 2125, 2 };
	uint16_t concat_6_tmp_0_dim[] = { 1, 2125, 4 };
	uint16_t eager_tmp_1_dim[] = { 2125, 1 };
	uint16_t tmp_16_dim[] = { 1, 2125, 4 };
	uint16_t split_1_tmp_0_dim[] = { 1, 1 };
	uint16_t split_1_tmp_1_dim[] = { 1, 1 };
	uint16_t concat_7_tmp_0_dim[] = { 1, 4 };
	uint16_t reshape2_12_tmp_0_dim[] = { 1, 1, 4 };
	uint16_t reshape2_12_tmp_1_dim[] = { 0, 1, 4 };
	uint16_t tmp_17_dim[] = { 1, 2125, 4 };
	uint16_t multiclass_nms3_0_tmp_0_dim[] = { 1, 6 };
	uint16_t multiclass_nms3_0_tmp_1_dim[] = { 1, 1 };
	uint16_t multiclass_nms3_0_tmp_2_dim[] = { 1 };

	float* relu6_0_tmp_0 = (float*)calloc(204800, sizeof(float));
	autox_conv2d(image, weights + 0, weights + 8, relu6_0_tmp_0, image_dim, conv2d_0_w_0_dim, relu6_0_tmp_0_dim, 1, 1, 2, 1, 2);
	free(image);

	float* relu6_1_tmp_0 = (float*)calloc(204800, sizeof(float));
	autox_conv2d(relu6_0_tmp_0, weights + 224, weights + 232, relu6_1_tmp_0, relu6_0_tmp_0_dim, conv2d_1_w_0_dim, relu6_1_tmp_0_dim, 8, 1, 1, 1, 2);
	free(relu6_0_tmp_0);

	float* relu6_2_tmp_0 = (float*)calloc(409600, sizeof(float));
	autox_conv2d(relu6_1_tmp_0, weights + 304, weights + 320, relu6_2_tmp_0, relu6_1_tmp_0_dim, conv2d_2_w_0_dim, relu6_2_tmp_0_dim, 1, 0, 1, 1, 2);
	free(relu6_1_tmp_0);

	float* relu6_3_tmp_0 = (float*)calloc(102400, sizeof(float));
	autox_conv2d(relu6_2_tmp_0, weights + 448, weights + 464, relu6_3_tmp_0, relu6_2_tmp_0_dim, conv2d_3_w_0_dim, relu6_3_tmp_0_dim, 16, 1, 2, 1, 2);
	free(relu6_2_tmp_0);

	float* relu6_4_tmp_0 = (float*)calloc(153600, sizeof(float));
	autox_conv2d(relu6_3_tmp_0, weights + 608, weights + 632, relu6_4_tmp_0, relu6_3_tmp_0_dim, conv2d_4_w_0_dim, relu6_4_tmp_0_dim, 1, 0, 1, 1, 2);
	free(relu6_3_tmp_0);

	float* relu6_5_tmp_0 = (float*)calloc(153600, sizeof(float));
	autox_conv2d(relu6_4_tmp_0, weights + 1016, weights + 1040, relu6_5_tmp_0, relu6_4_tmp_0_dim, conv2d_5_w_0_dim, relu6_5_tmp_0_dim, 24, 1, 1, 1, 2);
	free(relu6_4_tmp_0);

	float* relu6_6_tmp_0 = (float*)calloc(153600, sizeof(float));
	autox_conv2d(relu6_5_tmp_0, weights + 1256, weights + 1280, relu6_6_tmp_0, relu6_5_tmp_0_dim, conv2d_6_w_0_dim, relu6_6_tmp_0_dim, 1, 0, 1, 1, 2);
	free(relu6_5_tmp_0);

	float* relu6_7_tmp_0 = (float*)calloc(38400, sizeof(float));
	autox_conv2d(relu6_6_tmp_0, weights + 1856, weights + 1880, relu6_7_tmp_0, relu6_6_tmp_0_dim, conv2d_7_w_0_dim, relu6_7_tmp_0_dim, 24, 1, 2, 1, 2);
	free(relu6_6_tmp_0);

	float* relu6_8_tmp_0 = (float*)calloc(76800, sizeof(float));
	autox_conv2d(relu6_7_tmp_0, weights + 2096, weights + 2144, relu6_8_tmp_0, relu6_7_tmp_0_dim, conv2d_8_w_0_dim, relu6_8_tmp_0_dim, 1, 0, 1, 1, 2);
	free(relu6_7_tmp_0);

	float* relu6_9_tmp_0 = (float*)calloc(76800, sizeof(float));
	autox_conv2d(relu6_8_tmp_0, weights + 3296, weights + 3344, relu6_9_tmp_0, relu6_8_tmp_0_dim, conv2d_9_w_0_dim, relu6_9_tmp_0_dim, 48, 1, 1, 1, 2);
	free(relu6_8_tmp_0);

	float* relu6_10_tmp_0 = (float*)calloc(76800, sizeof(float));
	autox_conv2d(relu6_9_tmp_0, weights + 3776, weights + 3824, relu6_10_tmp_0, relu6_9_tmp_0_dim, conv2d_10_w_0_dim, relu6_10_tmp_0_dim, 1, 0, 1, 1, 2);
	free(relu6_9_tmp_0);

	float* relu6_11_tmp_0 = (float*)calloc(19200, sizeof(float));
	autox_conv2d(relu6_10_tmp_0, weights + 6128, weights + 6176, relu6_11_tmp_0, relu6_10_tmp_0_dim, conv2d_11_w_0_dim, relu6_11_tmp_0_dim, 48, 1, 2, 1, 2);
	free(relu6_10_tmp_0);

	float* relu6_12_tmp_0 = (float*)calloc(35200, sizeof(float));
	autox_conv2d(relu6_11_tmp_0, weights + 6608, weights + 6696, relu6_12_tmp_0, relu6_11_tmp_0_dim, conv2d_12_w_0_dim, relu6_12_tmp_0_dim, 1, 0, 1, 1, 2);
	free(relu6_11_tmp_0);

	float* relu6_13_tmp_0 = (float*)calloc(35200, sizeof(float));
	autox_conv2d(relu6_12_tmp_0, weights + 10920, weights + 11008, relu6_13_tmp_0, relu6_12_tmp_0_dim, conv2d_13_w_0_dim, relu6_13_tmp_0_dim, 88, 2, 1, 1, 2);
	free(relu6_12_tmp_0);

	float* relu6_14_tmp_0 = (float*)calloc(35200, sizeof(float));
	autox_conv2d(relu6_13_tmp_0, weights + 13208, weights + 13296, relu6_14_tmp_0, relu6_13_tmp_0_dim, conv2d_14_w_0_dim, relu6_14_tmp_0_dim, 1, 0, 1, 1, 2);
	free(relu6_13_tmp_0);

	float* relu6_15_tmp_0 = (float*)calloc(35200, sizeof(float));
	autox_conv2d(relu6_14_tmp_0, weights + 21040, weights + 21128, relu6_15_tmp_0, relu6_14_tmp_0_dim, conv2d_15_w_0_dim, relu6_15_tmp_0_dim, 88, 2, 1, 1, 2);
	free(relu6_14_tmp_0);

	float* relu6_16_tmp_0 = (float*)calloc(35200, sizeof(float));
	autox_conv2d(relu6_15_tmp_0, weights + 23328, weights + 23416, relu6_16_tmp_0, relu6_15_tmp_0_dim, conv2d_16_w_0_dim, relu6_16_tmp_0_dim, 1, 0, 1, 1, 2);
	free(relu6_15_tmp_0);

	float* relu6_17_tmp_0 = (float*)calloc(35200, sizeof(float));
	autox_conv2d(relu6_16_tmp_0, weights + 31160, weights + 31248, relu6_17_tmp_0, relu6_16_tmp_0_dim, conv2d_17_w_0_dim, relu6_17_tmp_0_dim, 88, 2, 1, 1, 2);
	free(relu6_16_tmp_0);

	float* relu6_18_tmp_0 = (float*)calloc(35200, sizeof(float));
	autox_conv2d(relu6_17_tmp_0, weights + 33448, weights + 33536, relu6_18_tmp_0, relu6_17_tmp_0_dim, conv2d_18_w_0_dim, relu6_18_tmp_0_dim, 1, 0, 1, 1, 2);
	free(relu6_17_tmp_0);

	float* relu6_19_tmp_0 = (float*)calloc(35200, sizeof(float));
	autox_conv2d(relu6_18_tmp_0, weights + 41280, weights + 41368, relu6_19_tmp_0, relu6_18_tmp_0_dim, conv2d_19_w_0_dim, relu6_19_tmp_0_dim, 88, 2, 1, 1, 2);
	free(relu6_18_tmp_0);

	float* relu6_20_tmp_0 = (float*)calloc(35200, sizeof(float));
	autox_conv2d(relu6_19_tmp_0, weights + 43568, weights + 43656, relu6_20_tmp_0, relu6_19_tmp_0_dim, conv2d_20_w_0_dim, relu6_20_tmp_0_dim, 1, 0, 1, 1, 2);
	free(relu6_19_tmp_0);

	float* relu6_21_tmp_0 = (float*)calloc(35200, sizeof(float));
	autox_conv2d(relu6_20_tmp_0, weights + 51400, weights + 51488, relu6_21_tmp_0, relu6_20_tmp_0_dim, conv2d_21_w_0_dim, relu6_21_tmp_0_dim, 88, 2, 1, 1, 2);
	free(relu6_20_tmp_0);

	float* relu6_22_tmp_0 = (float*)calloc(35200, sizeof(float));
	autox_conv2d(relu6_21_tmp_0, weights + 53688, weights + 53776, relu6_22_tmp_0, relu6_21_tmp_0_dim, conv2d_22_w_0_dim, relu6_22_tmp_0_dim, 1, 0, 1, 1, 2);
	free(relu6_21_tmp_0);

	float* relu6_23_tmp_0 = (float*)calloc(8800, sizeof(float));
	autox_conv2d(relu6_22_tmp_0, weights + 61520, weights + 61608, relu6_23_tmp_0, relu6_22_tmp_0_dim, conv2d_23_w_0_dim, relu6_23_tmp_0_dim, 88, 2, 2, 1, 2);
	free(relu6_22_tmp_0);

	float* pool2d_0_tmp_0 = (float*)calloc(88, sizeof(float));
	autox_pool2d(relu6_23_tmp_0, pool2d_0_tmp_0, relu6_23_tmp_0_dim, pool2d_0_tmp_0_dim, 1, 1, 0, 1, 0);
	free(relu6_23_tmp_0);

	float* relu_0_tmp_0 = (float*)calloc(22, sizeof(float));
	autox_conv2d(pool2d_0_tmp_0, weights + 63808, weights + 63830, relu_0_tmp_0, pool2d_0_tmp_0_dim, conv2d_24_w_0_dim, relu_0_tmp_0_dim, 1, 0, 1, 1, 1);
	free(pool2d_0_tmp_0);

	float* conv2d_113_tmp_1 = (float*)calloc(88, sizeof(float));
	autox_conv2d(relu_0_tmp_0, weights + 65766, weights + 65854, conv2d_113_tmp_1, relu_0_tmp_0_dim, conv2d_25_w_0_dim, conv2d_113_tmp_1_dim, 1, 0, 1, 1, 0);
	free(relu_0_tmp_0);

	autox_hard_sigmoid(conv2d_113_tmp_1, conv2d_113_tmp_1_dim);

	float* elementwise_mul_0 = (float*)calloc(8800, sizeof(float));
	autox_elementwise_mul(relu6_23_tmp_0, hardsigmoid_2_tmp_0, elementwise_mul_0, relu6_23_tmp_0_dim, hardsigmoid_2_tmp_0_dim, elementwise_mul_0_dim, -1, 4, 4);
	free(relu6_23_tmp_0);
	free(hardsigmoid_2_tmp_0);

	float* relu6_24_tmp_0 = (float*)calloc(17600, sizeof(float));
	autox_conv2d(elementwise_mul_0, weights + 67790, weights + 67966, relu6_24_tmp_0, elementwise_mul_0_dim, conv2d_26_w_0_dim, relu6_24_tmp_0_dim, 1, 0, 1, 1, 2);
	free(elementwise_mul_0);

	float* relu6_25_tmp_0 = (float*)calloc(17600, sizeof(float));
	autox_conv2d(relu6_24_tmp_0, weights + 83454, weights + 83630, relu6_25_tmp_0, relu6_24_tmp_0_dim, conv2d_27_w_0_dim, relu6_25_tmp_0_dim, 176, 2, 1, 1, 2);
	free(relu6_24_tmp_0);

	float* pool2d_1_tmp_0 = (float*)calloc(176, sizeof(float));
	autox_pool2d(relu6_25_tmp_0, pool2d_1_tmp_0, relu6_25_tmp_0_dim, pool2d_1_tmp_0_dim, 1, 1, 0, 1, 0);
	free(relu6_25_tmp_0);

	float* relu_1_tmp_0 = (float*)calloc(44, sizeof(float));
	autox_conv2d(pool2d_1_tmp_0, weights + 88030, weights + 88074, relu_1_tmp_0, pool2d_1_tmp_0_dim, conv2d_28_w_0_dim, relu_1_tmp_0_dim, 1, 0, 1, 1, 1);
	free(pool2d_1_tmp_0);

	float* conv2d_116_tmp_1 = (float*)calloc(176, sizeof(float));
	autox_conv2d(relu_1_tmp_0, weights + 95818, weights + 95994, conv2d_116_tmp_1, relu_1_tmp_0_dim, conv2d_29_w_0_dim, conv2d_116_tmp_1_dim, 1, 0, 1, 1, 0);
	free(relu_1_tmp_0);

	autox_hard_sigmoid(conv2d_116_tmp_1, conv2d_116_tmp_1_dim);

	float* elementwise_mul_1 = (float*)calloc(17600, sizeof(float));
	autox_elementwise_mul(relu6_25_tmp_0, hardsigmoid_3_tmp_0, elementwise_mul_1, relu6_25_tmp_0_dim, hardsigmoid_3_tmp_0_dim, elementwise_mul_1_dim, -1, 4, 4);
	free(relu6_25_tmp_0);
	free(hardsigmoid_3_tmp_0);

	float* relu6_26_tmp_0 = (float*)calloc(17600, sizeof(float));
	autox_conv2d(elementwise_mul_1, weights + 103738, weights + 103914, relu6_26_tmp_0, elementwise_mul_1_dim, conv2d_30_w_0_dim, relu6_26_tmp_0_dim, 1, 0, 1, 1, 2);
	free(elementwise_mul_1);

	float* relu6_29_tmp_0 = (float*)calloc(9600, sizeof(float));
	autox_conv2d(relu6_26_tmp_0, weights + 134890, weights + 134986, relu6_29_tmp_0, relu6_26_tmp_0_dim, conv2d_33_w_0_dim, relu6_29_tmp_0_dim, 1, 0, 1, 1, 2);
	free(relu6_26_tmp_0);

	float* nearest_interp_v2_0_tmp_0 = (float*)calloc(38400, sizeof(float));
	autox_nearest_interp_v2(relu6_29_tmp_0, nearest_interp_v2_0_tmp_0, relu6_29_tmp_0_dim, nearest_interp_v2_0_tmp_0_dim);
	free(relu6_29_tmp_0);

	float* relu6_28_tmp_0 = (float*)calloc(38400, sizeof(float));
	autox_conv2d(relu6_22_tmp_0, weights + 151882, weights + 151978, relu6_28_tmp_0, relu6_22_tmp_0_dim, conv2d_32_w_0_dim, relu6_28_tmp_0_dim, 1, 0, 1, 1, 2);
	free(relu6_22_tmp_0);

	float* p_40[] = { nearest_interp_v2_0_tmp_0, relu6_28_tmp_0, };
	uint16_t* p_40_dim[] = { nearest_interp_v2_0_tmp_0_dim, relu6_28_tmp_0_dim, };
	float* concat_0_tmp_0 = (float*)calloc(76800, sizeof(float));
	autox_concat(p_40, concat_0_tmp_0, p_40_dim, concat_0_tmp_0_dim, 1, 2, 4);
	free(nearest_interp_v2_0_tmp_0);
	free(relu6_28_tmp_0);

	float* relu6_30_tmp_0 = (float*)calloc(76800, sizeof(float));
	autox_conv2d(concat_0_tmp_0, weights + 160426, weights + 160618, relu6_30_tmp_0, concat_0_tmp_0_dim, conv2d_38_w_0_dim, relu6_30_tmp_0_dim, 192, 2, 1, 1, 2);
	free(concat_0_tmp_0);

	float* relu6_31_tmp_0 = (float*)calloc(76800, sizeof(float));
	autox_conv2d(relu6_30_tmp_0, weights + 165418, weights + 165610, relu6_31_tmp_0, relu6_30_tmp_0_dim, conv2d_39_w_0_dim, relu6_31_tmp_0_dim, 1, 0, 1, 1, 2);
	free(relu6_30_tmp_0);

	float* relu6_32_tmp_0 = (float*)calloc(76800, sizeof(float));
	autox_conv2d(relu6_31_tmp_0, weights + 202474, weights + 202666, relu6_32_tmp_0, relu6_31_tmp_0_dim, conv2d_40_w_0_dim, relu6_32_tmp_0_dim, 192, 2, 1, 1, 2);
	free(relu6_31_tmp_0);

	float* relu6_33_tmp_0 = (float*)calloc(38400, sizeof(float));
	autox_conv2d(relu6_32_tmp_0, weights + 207466, weights + 207562, relu6_33_tmp_0, relu6_32_tmp_0_dim, conv2d_41_w_0_dim, relu6_33_tmp_0_dim, 1, 0, 1, 1, 2);
	free(relu6_32_tmp_0);

	float* nearest_interp_v2_1_tmp_0 = (float*)calloc(153600, sizeof(float));
	autox_nearest_interp_v2(relu6_33_tmp_0, nearest_interp_v2_1_tmp_0, relu6_33_tmp_0_dim, nearest_interp_v2_1_tmp_0_dim);
	free(relu6_33_tmp_0);

	float* relu6_27_tmp_0 = (float*)calloc(153600, sizeof(float));
	autox_conv2d(relu6_10_tmp_0, weights + 225994, weights + 226090, relu6_27_tmp_0, relu6_10_tmp_0_dim, conv2d_31_w_0_dim, relu6_27_tmp_0_dim, 1, 0, 1, 1, 2);
	free(relu6_10_tmp_0);

	float* p_47[] = { nearest_interp_v2_1_tmp_0, relu6_27_tmp_0, };
	uint16_t* p_47_dim[] = { nearest_interp_v2_1_tmp_0_dim, relu6_27_tmp_0_dim, };
	float* concat_1_tmp_0 = (float*)calloc(307200, sizeof(float));
	autox_concat(p_47, concat_1_tmp_0, p_47_dim, concat_1_tmp_0_dim, 1, 2, 4);
	free(nearest_interp_v2_1_tmp_0);
	free(relu6_27_tmp_0);

	float* relu6_34_tmp_0 = (float*)calloc(307200, sizeof(float));
	autox_conv2d(concat_1_tmp_0, weights + 230698, weights + 230890, relu6_34_tmp_0, concat_1_tmp_0_dim, conv2d_42_w_0_dim, relu6_34_tmp_0_dim, 192, 2, 1, 1, 2);
	free(concat_1_tmp_0);

	float* relu6_35_tmp_0 = (float*)calloc(307200, sizeof(float));
	autox_conv2d(relu6_34_tmp_0, weights + 235690, weights + 235882, relu6_35_tmp_0, relu6_34_tmp_0_dim, conv2d_43_w_0_dim, relu6_35_tmp_0_dim, 1, 0, 1, 1, 2);
	free(relu6_34_tmp_0);

	float* relu6_36_tmp_0 = (float*)calloc(307200, sizeof(float));
	autox_conv2d(relu6_35_tmp_0, weights + 272746, weights + 272938, relu6_36_tmp_0, relu6_35_tmp_0_dim, conv2d_44_w_0_dim, relu6_36_tmp_0_dim, 192, 2, 1, 1, 2);
	free(relu6_35_tmp_0);

	float* relu6_37_tmp_0 = (float*)calloc(153600, sizeof(float));
	autox_conv2d(relu6_36_tmp_0, weights + 277738, weights + 277834, relu6_37_tmp_0, relu6_36_tmp_0_dim, conv2d_45_w_0_dim, relu6_37_tmp_0_dim, 1, 0, 1, 1, 2);
	free(relu6_36_tmp_0);

	float* relu6_38_tmp_0 = (float*)calloc(38400, sizeof(float));
	autox_conv2d(relu6_37_tmp_0, weights + 296266, weights + 296362, relu6_38_tmp_0, relu6_37_tmp_0_dim, conv2d_46_w_0_dim, relu6_38_tmp_0_dim, 96, 2, 2, 1, 2);
	free(relu6_37_tmp_0);

	float* relu6_39_tmp_0 = (float*)calloc(38400, sizeof(float));
	autox_conv2d(relu6_38_tmp_0, weights + 298762, weights + 298858, relu6_39_tmp_0, relu6_38_tmp_0_dim, conv2d_47_w_0_dim, relu6_39_tmp_0_dim, 1, 0, 1, 1, 2);
	free(relu6_38_tmp_0);

	float* p_54[] = { relu6_39_tmp_0, relu6_33_tmp_0, };
	uint16_t* p_54_dim[] = { relu6_39_tmp_0_dim, relu6_33_tmp_0_dim, };
	float* concat_2_tmp_0 = (float*)calloc(76800, sizeof(float));
	autox_concat(p_54, concat_2_tmp_0, p_54_dim, concat_2_tmp_0_dim, 1, 2, 4);
	free(relu6_39_tmp_0);
	free(relu6_33_tmp_0);

	float* relu6_40_tmp_0 = (float*)calloc(76800, sizeof(float));
	autox_conv2d(concat_2_tmp_0, weights + 308074, weights + 308266, relu6_40_tmp_0, concat_2_tmp_0_dim, conv2d_48_w_0_dim, relu6_40_tmp_0_dim, 192, 2, 1, 1, 2);
	free(concat_2_tmp_0);

	float* relu6_41_tmp_0 = (float*)calloc(76800, sizeof(float));
	autox_conv2d(relu6_40_tmp_0, weights + 313066, weights + 313258, relu6_41_tmp_0, relu6_40_tmp_0_dim, conv2d_49_w_0_dim, relu6_41_tmp_0_dim, 1, 0, 1, 1, 2);
	free(relu6_40_tmp_0);

	float* relu6_42_tmp_0 = (float*)calloc(76800, sizeof(float));
	autox_conv2d(relu6_41_tmp_0, weights + 350122, weights + 350314, relu6_42_tmp_0, relu6_41_tmp_0_dim, conv2d_50_w_0_dim, relu6_42_tmp_0_dim, 192, 2, 1, 1, 2);
	free(relu6_41_tmp_0);

	float* relu6_43_tmp_0 = (float*)calloc(38400, sizeof(float));
	autox_conv2d(relu6_42_tmp_0, weights + 355114, weights + 355210, relu6_43_tmp_0, relu6_42_tmp_0_dim, conv2d_51_w_0_dim, relu6_43_tmp_0_dim, 1, 0, 1, 1, 2);
	free(relu6_42_tmp_0);

	float* relu6_44_tmp_0 = (float*)calloc(9600, sizeof(float));
	autox_conv2d(relu6_43_tmp_0, weights + 373642, weights + 373738, relu6_44_tmp_0, relu6_43_tmp_0_dim, conv2d_52_w_0_dim, relu6_44_tmp_0_dim, 96, 2, 2, 1, 2);
	free(relu6_43_tmp_0);

	float* relu6_45_tmp_0 = (float*)calloc(9600, sizeof(float));
	autox_conv2d(relu6_44_tmp_0, weights + 376138, weights + 376234, relu6_45_tmp_0, relu6_44_tmp_0_dim, conv2d_53_w_0_dim, relu6_45_tmp_0_dim, 1, 0, 1, 1, 2);
	free(relu6_44_tmp_0);

	float* p_61[] = { relu6_45_tmp_0, relu6_29_tmp_0, };
	uint16_t* p_61_dim[] = { relu6_45_tmp_0_dim, relu6_29_tmp_0_dim, };
	float* concat_3_tmp_0 = (float*)calloc(19200, sizeof(float));
	autox_concat(p_61, concat_3_tmp_0, p_61_dim, concat_3_tmp_0_dim, 1, 2, 4);
	free(relu6_45_tmp_0);
	free(relu6_29_tmp_0);

	float* relu6_46_tmp_0 = (float*)calloc(19200, sizeof(float));
	autox_conv2d(concat_3_tmp_0, weights + 385450, weights + 385642, relu6_46_tmp_0, concat_3_tmp_0_dim, conv2d_54_w_0_dim, relu6_46_tmp_0_dim, 192, 2, 1, 1, 2);
	free(concat_3_tmp_0);

	float* relu6_47_tmp_0 = (float*)calloc(19200, sizeof(float));
	autox_conv2d(relu6_46_tmp_0, weights + 390442, weights + 390634, relu6_47_tmp_0, relu6_46_tmp_0_dim, conv2d_55_w_0_dim, relu6_47_tmp_0_dim, 1, 0, 1, 1, 2);
	free(relu6_46_tmp_0);

	float* relu6_48_tmp_0 = (float*)calloc(19200, sizeof(float));
	autox_conv2d(relu6_47_tmp_0, weights + 427498, weights + 427690, relu6_48_tmp_0, relu6_47_tmp_0_dim, conv2d_56_w_0_dim, relu6_48_tmp_0_dim, 192, 2, 1, 1, 2);
	free(relu6_47_tmp_0);

	float* relu6_49_tmp_0 = (float*)calloc(9600, sizeof(float));
	autox_conv2d(relu6_48_tmp_0, weights + 432490, weights + 432586, relu6_49_tmp_0, relu6_48_tmp_0_dim, conv2d_57_w_0_dim, relu6_49_tmp_0_dim, 1, 0, 1, 1, 2);
	free(relu6_48_tmp_0);

	float* relu6_52_tmp_0 = (float*)calloc(2400, sizeof(float));
	autox_conv2d(relu6_49_tmp_0, weights + 451018, weights + 451114, relu6_52_tmp_0, relu6_49_tmp_0_dim, conv2d_36_w_0_dim, relu6_52_tmp_0_dim, 96, 2, 2, 1, 2);
	free(relu6_49_tmp_0);

	float* relu6_53_tmp_0 = (float*)calloc(2400, sizeof(float));
	autox_conv2d(relu6_52_tmp_0, weights + 453514, weights + 453610, relu6_53_tmp_0, relu6_52_tmp_0_dim, conv2d_37_w_0_dim, relu6_53_tmp_0_dim, 1, 0, 1, 1, 2);
	free(relu6_52_tmp_0);

	float* relu6_50_tmp_0 = (float*)calloc(2400, sizeof(float));
	autox_conv2d(relu6_29_tmp_0, weights + 462826, weights + 462922, relu6_50_tmp_0, relu6_29_tmp_0_dim, conv2d_34_w_0_dim, relu6_50_tmp_0_dim, 96, 2, 2, 1, 2);
	free(relu6_29_tmp_0);

	float* relu6_51_tmp_0 = (float*)calloc(2400, sizeof(float));
	autox_conv2d(relu6_50_tmp_0, weights + 465322, weights + 465418, relu6_51_tmp_0, relu6_50_tmp_0_dim, conv2d_35_w_0_dim, relu6_51_tmp_0_dim, 1, 0, 1, 1, 2);
	free(relu6_50_tmp_0);

	float* tmp_0 = (float*)calloc(2400, sizeof(float));
	autox_elementwise_add(relu6_51_tmp_0, relu6_53_tmp_0, tmp_0, relu6_51_tmp_0_dim, relu6_53_tmp_0_dim, tmp_0_dim, -1, 4, 4);
	free(relu6_51_tmp_0);
	free(relu6_53_tmp_0);

	float* relu6_54_tmp_0 = (float*)calloc(153600, sizeof(float));
	autox_conv2d(relu6_37_tmp_0, weights + 474634, weights + 474730, relu6_54_tmp_0, relu6_37_tmp_0_dim, conv2d_58_w_0_dim, relu6_54_tmp_0_dim, 96, 2, 1, 1, 2);
	free(relu6_37_tmp_0);

	float* relu6_55_tmp_0 = (float*)calloc(153600, sizeof(float));
	autox_conv2d(relu6_54_tmp_0, weights + 477130, weights + 477226, relu6_55_tmp_0, relu6_54_tmp_0_dim, conv2d_59_w_0_dim, relu6_55_tmp_0_dim, 1, 0, 1, 1, 2);
	free(relu6_54_tmp_0);

	float* relu6_56_tmp_0 = (float*)calloc(153600, sizeof(float));
	autox_conv2d(relu6_55_tmp_0, weights + 486442, weights + 486538, relu6_56_tmp_0, relu6_55_tmp_0_dim, conv2d_60_w_0_dim, relu6_56_tmp_0_dim, 96, 2, 1, 1, 2);
	free(relu6_55_tmp_0);

	float* relu6_57_tmp_0 = (float*)calloc(153600, sizeof(float));
	autox_conv2d(relu6_56_tmp_0, weights + 488938, weights + 489034, relu6_57_tmp_0, relu6_56_tmp_0_dim, conv2d_61_w_0_dim, relu6_57_tmp_0_dim, 1, 0, 1, 1, 2);
	free(relu6_56_tmp_0);

	float* pool2d_2_tmp_0 = (float*)calloc(96, sizeof(float));
	autox_pool2d(relu6_57_tmp_0, pool2d_2_tmp_0, relu6_57_tmp_0_dim, pool2d_2_tmp_0_dim, 1, 1, 0, 1, 0);
	free(relu6_57_tmp_0);

	float* conv2d_135_tmp_1 = (float*)calloc(96, sizeof(float));
	autox_conv2d(pool2d_2_tmp_0, weights + 498250, weights + 498346, conv2d_135_tmp_1, pool2d_2_tmp_0_dim, conv2d_62_w_0_dim, conv2d_135_tmp_1_dim, 1, 0, 1, 1, 0);
	free(pool2d_2_tmp_0);

	float* sigmoid_0_tmp_0 = (float*)calloc(96, sizeof(float));
	autox_sigmoid(conv2d_135_tmp_1, conv2d_135_tmp_1_dim);

	float* tmp_1 = (float*)calloc(153600, sizeof(float));
	autox_elementwise_mul(relu6_57_tmp_0, sigmoid_0_tmp_0, tmp_1, relu6_57_tmp_0_dim, sigmoid_0_tmp_0_dim, tmp_1_dim, -1, 4, 4);
	free(relu6_57_tmp_0);
	free(sigmoid_0_tmp_0);

	float* relu6_58_tmp_0 = (float*)calloc(153600, sizeof(float));
	autox_conv2d(tmp_1, weights + 507562, weights + 507658, relu6_58_tmp_0, tmp_1_dim, conv2d_63_w_0_dim, relu6_58_tmp_0_dim, 1, 0, 1, 1, 2);
	free(tmp_1);

	float* conv2d_137_tmp_1 = (float*)calloc(40000, sizeof(float));
	autox_conv2d(relu6_58_tmp_0, weights + 516874, weights + 516899, conv2d_137_tmp_1, relu6_58_tmp_0_dim, conv2d_84_w_0_dim, conv2d_137_tmp_1_dim, 1, 0, 1, 1, 0);
	free(relu6_58_tmp_0);

	float* conv2d_138_tmp_1 = (float*)calloc(51200, sizeof(float));
	autox_conv2d(relu6_58_tmp_0, weights + 519299, weights + 519331, conv2d_138_tmp_1, relu6_58_tmp_0_dim, conv2d_85_w_0_dim, conv2d_138_tmp_1_dim, 1, 0, 1, 1, 0);
	free(relu6_58_tmp_0);

	float* relu6_59_tmp_0 = (float*)calloc(1600, sizeof(float));
	autox_conv2d(relu6_57_tmp_0, weights + 522403, weights + 522404, relu6_59_tmp_0, relu6_57_tmp_0_dim, conv2d_86_w_0_dim, relu6_59_tmp_0_dim, 1, 2, 1, 1, 2);
	free(relu6_57_tmp_0);

	float* batch_norm_60_tmp_2 = (float*)calloc(1600, sizeof(float));
	autox_conv2d(relu6_59_tmp_0, weights + 524804, weights + 524805, batch_norm_60_tmp_2, relu6_59_tmp_0_dim, conv2d_87_w_0_dim, batch_norm_60_tmp_2_dim, 1, 0, 1, 1, 0);
	free(relu6_59_tmp_0);

	float* sigmoid_1_tmp_0 = (float*)calloc(1600, sizeof(float));
	autox_sigmoid(batch_norm_60_tmp_2, batch_norm_60_tmp_2_dim);

	float* sigmoid_2_tmp_0 = (float*)calloc(40000, sizeof(float));
	autox_sigmoid(conv2d_137_tmp_1, conv2d_137_tmp_1_dim);

	float* tmp_2 = (float*)calloc(40000, sizeof(float));
	autox_elementwise_mul(sigmoid_2_tmp_0, sigmoid_1_tmp_0, tmp_2, sigmoid_2_tmp_0_dim, sigmoid_1_tmp_0_dim, tmp_2_dim, -1, 4, 4);
	free(sigmoid_2_tmp_0);
	free(sigmoid_1_tmp_0);

	float* tmp_3 = (float*)calloc(40000, sizeof(float));
	autox_scale(tmp_2, tmp_2_dim);

	float* sqrt_0_tmp_0 = (float*)calloc(40000, sizeof(float));
	autox_sqrt(tmp_3, tmp_3_dim);

	float* transpose_0_tmp_0 = (float*)calloc(51200, sizeof(float));
	uint16_t axis_89[] = { 0, 2, 3, 1 };
	autox_transpose(conv2d_138_tmp_1, transpose_0_tmp_0, conv2d_138_tmp_1_dim, transpose_0_tmp_0_dim, axis_89, 4);
	free(conv2d_138_tmp_1);

	autox_softmax(reshape2_1_tmp_0, reshape2_1_tmp_0_dim, 1);

	float* linear_0_tmp_0 = (float*)calloc(1, sizeof(float));
	autox_matmul(softmax_0_tmp_0, weights + 524806, linear_0_tmp_0, softmax_0_tmp_0_dim, eager_tmp_4_dim, linear_0_tmp_0_dim, 0, 0, 2, 1, 1);
	free(softmax_0_tmp_0);

	float* relu6_60_tmp_0 = (float*)calloc(38400, sizeof(float));
	autox_conv2d(relu6_43_tmp_0, weights + 524814, weights + 524910, relu6_60_tmp_0, relu6_43_tmp_0_dim, conv2d_64_w_0_dim, relu6_60_tmp_0_dim, 96, 2, 1, 1, 2);
	free(relu6_43_tmp_0);

	float* relu6_61_tmp_0 = (float*)calloc(38400, sizeof(float));
	autox_conv2d(relu6_60_tmp_0, weights + 527310, weights + 527406, relu6_61_tmp_0, relu6_60_tmp_0_dim, conv2d_65_w_0_dim, relu6_61_tmp_0_dim, 1, 0, 1, 1, 2);
	free(relu6_60_tmp_0);

	float* relu6_62_tmp_0 = (float*)calloc(38400, sizeof(float));
	autox_conv2d(relu6_61_tmp_0, weights + 536622, weights + 536718, relu6_62_tmp_0, relu6_61_tmp_0_dim, conv2d_66_w_0_dim, relu6_62_tmp_0_dim, 96, 2, 1, 1, 2);
	free(relu6_61_tmp_0);

	float* relu6_63_tmp_0 = (float*)calloc(38400, sizeof(float));
	autox_conv2d(relu6_62_tmp_0, weights + 539118, weights + 539214, relu6_63_tmp_0, relu6_62_tmp_0_dim, conv2d_67_w_0_dim, relu6_63_tmp_0_dim, 1, 0, 1, 1, 2);
	free(relu6_62_tmp_0);

	float* pool2d_3_tmp_0 = (float*)calloc(96, sizeof(float));
	autox_pool2d(relu6_63_tmp_0, pool2d_3_tmp_0, relu6_63_tmp_0_dim, pool2d_3_tmp_0_dim, 1, 1, 0, 1, 0);
	free(relu6_63_tmp_0);

	float* conv2d_143_tmp_1 = (float*)calloc(96, sizeof(float));
	autox_conv2d(pool2d_3_tmp_0, weights + 548430, weights + 548526, conv2d_143_tmp_1, pool2d_3_tmp_0_dim, conv2d_68_w_0_dim, conv2d_143_tmp_1_dim, 1, 0, 1, 1, 0);
	free(pool2d_3_tmp_0);

	float* sigmoid_3_tmp_0 = (float*)calloc(96, sizeof(float));
	autox_sigmoid(conv2d_143_tmp_1, conv2d_143_tmp_1_dim);

	float* tmp_4 = (float*)calloc(38400, sizeof(float));
	autox_elementwise_mul(relu6_63_tmp_0, sigmoid_3_tmp_0, tmp_4, relu6_63_tmp_0_dim, sigmoid_3_tmp_0_dim, tmp_4_dim, -1, 4, 4);
	free(relu6_63_tmp_0);
	free(sigmoid_3_tmp_0);

	float* relu6_64_tmp_0 = (float*)calloc(38400, sizeof(float));
	autox_conv2d(tmp_4, weights + 557742, weights + 557838, relu6_64_tmp_0, tmp_4_dim, conv2d_69_w_0_dim, relu6_64_tmp_0_dim, 1, 0, 1, 1, 2);
	free(tmp_4);

	float* conv2d_145_tmp_1 = (float*)calloc(10000, sizeof(float));
	autox_conv2d(relu6_64_tmp_0, weights + 567054, weights + 567079, conv2d_145_tmp_1, relu6_64_tmp_0_dim, conv2d_88_w_0_dim, conv2d_145_tmp_1_dim, 1, 0, 1, 1, 0);
	free(relu6_64_tmp_0);

	float* conv2d_146_tmp_1 = (float*)calloc(12800, sizeof(float));
	autox_conv2d(relu6_64_tmp_0, weights + 569479, weights + 569511, conv2d_146_tmp_1, relu6_64_tmp_0_dim, conv2d_89_w_0_dim, conv2d_146_tmp_1_dim, 1, 0, 1, 1, 0);
	free(relu6_64_tmp_0);

	float* relu6_65_tmp_0 = (float*)calloc(400, sizeof(float));
	autox_conv2d(relu6_63_tmp_0, weights + 572583, weights + 572584, relu6_65_tmp_0, relu6_63_tmp_0_dim, conv2d_90_w_0_dim, relu6_65_tmp_0_dim, 1, 2, 1, 1, 2);
	free(relu6_63_tmp_0);

	float* batch_norm_67_tmp_2 = (float*)calloc(400, sizeof(float));
	autox_conv2d(relu6_65_tmp_0, weights + 574984, weights + 574985, batch_norm_67_tmp_2, relu6_65_tmp_0_dim, conv2d_91_w_0_dim, batch_norm_67_tmp_2_dim, 1, 0, 1, 1, 0);
	free(relu6_65_tmp_0);

	float* sigmoid_4_tmp_0 = (float*)calloc(400, sizeof(float));
	autox_sigmoid(batch_norm_67_tmp_2, batch_norm_67_tmp_2_dim);

	float* sigmoid_5_tmp_0 = (float*)calloc(10000, sizeof(float));
	autox_sigmoid(conv2d_145_tmp_1, conv2d_145_tmp_1_dim);

	float* tmp_5 = (float*)calloc(10000, sizeof(float));
	autox_elementwise_mul(sigmoid_5_tmp_0, sigmoid_4_tmp_0, tmp_5, sigmoid_5_tmp_0_dim, sigmoid_4_tmp_0_dim, tmp_5_dim, -1, 4, 4);
	free(sigmoid_5_tmp_0);
	free(sigmoid_4_tmp_0);

	float* tmp_6 = (float*)calloc(10000, sizeof(float));
	autox_scale(tmp_5, tmp_5_dim);

	float* sqrt_1_tmp_0 = (float*)calloc(10000, sizeof(float));
	autox_sqrt(tmp_6, tmp_6_dim);

	float* transpose_1_tmp_0 = (float*)calloc(12800, sizeof(float));
	uint16_t axis_110[] = { 0, 2, 3, 1 };
	autox_transpose(conv2d_146_tmp_1, transpose_1_tmp_0, conv2d_146_tmp_1_dim, transpose_1_tmp_0_dim, axis_110, 4);
	free(conv2d_146_tmp_1);

	autox_softmax(reshape2_4_tmp_0, reshape2_4_tmp_0_dim, 1);

	float* linear_1_tmp_0 = (float*)calloc(1, sizeof(float));
	autox_matmul(softmax_1_tmp_0, weights + 574986, linear_1_tmp_0, softmax_1_tmp_0_dim, eager_tmp_4_dim, linear_1_tmp_0_dim, 0, 0, 2, 1, 1);
	free(softmax_1_tmp_0);

	float* relu6_66_tmp_0 = (float*)calloc(9600, sizeof(float));
	autox_conv2d(relu6_49_tmp_0, weights + 574994, weights + 575090, relu6_66_tmp_0, relu6_49_tmp_0_dim, conv2d_70_w_0_dim, relu6_66_tmp_0_dim, 96, 2, 1, 1, 2);
	free(relu6_49_tmp_0);

	float* relu6_67_tmp_0 = (float*)calloc(9600, sizeof(float));
	autox_conv2d(relu6_66_tmp_0, weights + 577490, weights + 577586, relu6_67_tmp_0, relu6_66_tmp_0_dim, conv2d_71_w_0_dim, relu6_67_tmp_0_dim, 1, 0, 1, 1, 2);
	free(relu6_66_tmp_0);

	float* relu6_68_tmp_0 = (float*)calloc(9600, sizeof(float));
	autox_conv2d(relu6_67_tmp_0, weights + 586802, weights + 586898, relu6_68_tmp_0, relu6_67_tmp_0_dim, conv2d_72_w_0_dim, relu6_68_tmp_0_dim, 96, 2, 1, 1, 2);
	free(relu6_67_tmp_0);

	float* relu6_69_tmp_0 = (float*)calloc(9600, sizeof(float));
	autox_conv2d(relu6_68_tmp_0, weights + 589298, weights + 589394, relu6_69_tmp_0, relu6_68_tmp_0_dim, conv2d_73_w_0_dim, relu6_69_tmp_0_dim, 1, 0, 1, 1, 2);
	free(relu6_68_tmp_0);

	float* pool2d_4_tmp_0 = (float*)calloc(96, sizeof(float));
	autox_pool2d(relu6_69_tmp_0, pool2d_4_tmp_0, relu6_69_tmp_0_dim, pool2d_4_tmp_0_dim, 1, 1, 0, 1, 0);
	free(relu6_69_tmp_0);

	float* conv2d_151_tmp_1 = (float*)calloc(96, sizeof(float));
	autox_conv2d(pool2d_4_tmp_0, weights + 598610, weights + 598706, conv2d_151_tmp_1, pool2d_4_tmp_0_dim, conv2d_74_w_0_dim, conv2d_151_tmp_1_dim, 1, 0, 1, 1, 0);
	free(pool2d_4_tmp_0);

	float* sigmoid_6_tmp_0 = (float*)calloc(96, sizeof(float));
	autox_sigmoid(conv2d_151_tmp_1, conv2d_151_tmp_1_dim);

	float* tmp_7 = (float*)calloc(9600, sizeof(float));
	autox_elementwise_mul(relu6_69_tmp_0, sigmoid_6_tmp_0, tmp_7, relu6_69_tmp_0_dim, sigmoid_6_tmp_0_dim, tmp_7_dim, -1, 4, 4);
	free(relu6_69_tmp_0);
	free(sigmoid_6_tmp_0);

	float* relu6_70_tmp_0 = (float*)calloc(9600, sizeof(float));
	autox_conv2d(tmp_7, weights + 607922, weights + 608018, relu6_70_tmp_0, tmp_7_dim, conv2d_75_w_0_dim, relu6_70_tmp_0_dim, 1, 0, 1, 1, 2);
	free(tmp_7);

	float* conv2d_153_tmp_1 = (float*)calloc(2500, sizeof(float));
	autox_conv2d(relu6_70_tmp_0, weights + 617234, weights + 617259, conv2d_153_tmp_1, relu6_70_tmp_0_dim, conv2d_92_w_0_dim, conv2d_153_tmp_1_dim, 1, 0, 1, 1, 0);
	free(relu6_70_tmp_0);

	float* conv2d_154_tmp_1 = (float*)calloc(3200, sizeof(float));
	autox_conv2d(relu6_70_tmp_0, weights + 619659, weights + 619691, conv2d_154_tmp_1, relu6_70_tmp_0_dim, conv2d_93_w_0_dim, conv2d_154_tmp_1_dim, 1, 0, 1, 1, 0);
	free(relu6_70_tmp_0);

	float* relu6_71_tmp_0 = (float*)calloc(100, sizeof(float));
	autox_conv2d(relu6_69_tmp_0, weights + 622763, weights + 622764, relu6_71_tmp_0, relu6_69_tmp_0_dim, conv2d_94_w_0_dim, relu6_71_tmp_0_dim, 1, 2, 1, 1, 2);
	free(relu6_69_tmp_0);

	float* batch_norm_74_tmp_2 = (float*)calloc(100, sizeof(float));
	autox_conv2d(relu6_71_tmp_0, weights + 625164, weights + 625165, batch_norm_74_tmp_2, relu6_71_tmp_0_dim, conv2d_95_w_0_dim, batch_norm_74_tmp_2_dim, 1, 0, 1, 1, 0);
	free(relu6_71_tmp_0);

	float* sigmoid_7_tmp_0 = (float*)calloc(100, sizeof(float));
	autox_sigmoid(batch_norm_74_tmp_2, batch_norm_74_tmp_2_dim);

	float* sigmoid_8_tmp_0 = (float*)calloc(2500, sizeof(float));
	autox_sigmoid(conv2d_153_tmp_1, conv2d_153_tmp_1_dim);

	float* tmp_8 = (float*)calloc(2500, sizeof(float));
	autox_elementwise_mul(sigmoid_8_tmp_0, sigmoid_7_tmp_0, tmp_8, sigmoid_8_tmp_0_dim, sigmoid_7_tmp_0_dim, tmp_8_dim, -1, 4, 4);
	free(sigmoid_8_tmp_0);
	free(sigmoid_7_tmp_0);

	float* tmp_9 = (float*)calloc(2500, sizeof(float));
	autox_scale(tmp_8, tmp_8_dim);

	float* sqrt_2_tmp_0 = (float*)calloc(2500, sizeof(float));
	autox_sqrt(tmp_9, tmp_9_dim);

	float* transpose_2_tmp_0 = (float*)calloc(3200, sizeof(float));
	uint16_t axis_131[] = { 0, 2, 3, 1 };
	autox_transpose(conv2d_154_tmp_1, transpose_2_tmp_0, conv2d_154_tmp_1_dim, transpose_2_tmp_0_dim, axis_131, 4);
	free(conv2d_154_tmp_1);

	autox_softmax(reshape2_7_tmp_0, reshape2_7_tmp_0_dim, 1);

	float* linear_2_tmp_0 = (float*)calloc(1, sizeof(float));
	autox_matmul(softmax_2_tmp_0, weights + 625166, linear_2_tmp_0, softmax_2_tmp_0_dim, eager_tmp_4_dim, linear_2_tmp_0_dim, 0, 0, 2, 1, 1);
	free(softmax_2_tmp_0);

	float* relu6_72_tmp_0 = (float*)calloc(2400, sizeof(float));
	autox_conv2d(tmp_0, weights + 625174, weights + 625270, relu6_72_tmp_0, tmp_0_dim, conv2d_76_w_0_dim, relu6_72_tmp_0_dim, 96, 2, 1, 1, 2);
	free(tmp_0);

	float* relu6_73_tmp_0 = (float*)calloc(2400, sizeof(float));
	autox_conv2d(relu6_72_tmp_0, weights + 627670, weights + 627766, relu6_73_tmp_0, relu6_72_tmp_0_dim, conv2d_77_w_0_dim, relu6_73_tmp_0_dim, 1, 0, 1, 1, 2);
	free(relu6_72_tmp_0);

	float* relu6_74_tmp_0 = (float*)calloc(2400, sizeof(float));
	autox_conv2d(relu6_73_tmp_0, weights + 636982, weights + 637078, relu6_74_tmp_0, relu6_73_tmp_0_dim, conv2d_78_w_0_dim, relu6_74_tmp_0_dim, 96, 2, 1, 1, 2);
	free(relu6_73_tmp_0);

	float* relu6_75_tmp_0 = (float*)calloc(2400, sizeof(float));
	autox_conv2d(relu6_74_tmp_0, weights + 639478, weights + 639574, relu6_75_tmp_0, relu6_74_tmp_0_dim, conv2d_79_w_0_dim, relu6_75_tmp_0_dim, 1, 0, 1, 1, 2);
	free(relu6_74_tmp_0);

	float* pool2d_5_tmp_0 = (float*)calloc(96, sizeof(float));
	autox_pool2d(relu6_75_tmp_0, pool2d_5_tmp_0, relu6_75_tmp_0_dim, pool2d_5_tmp_0_dim, 1, 1, 0, 1, 0);
	free(relu6_75_tmp_0);

	float* conv2d_159_tmp_1 = (float*)calloc(96, sizeof(float));
	autox_conv2d(pool2d_5_tmp_0, weights + 648790, weights + 648886, conv2d_159_tmp_1, pool2d_5_tmp_0_dim, conv2d_80_w_0_dim, conv2d_159_tmp_1_dim, 1, 0, 1, 1, 0);
	free(pool2d_5_tmp_0);

	float* sigmoid_9_tmp_0 = (float*)calloc(96, sizeof(float));
	autox_sigmoid(conv2d_159_tmp_1, conv2d_159_tmp_1_dim);

	float* tmp_10 = (float*)calloc(2400, sizeof(float));
	autox_elementwise_mul(relu6_75_tmp_0, sigmoid_9_tmp_0, tmp_10, relu6_75_tmp_0_dim, sigmoid_9_tmp_0_dim, tmp_10_dim, -1, 4, 4);
	free(relu6_75_tmp_0);
	free(sigmoid_9_tmp_0);

	float* relu6_76_tmp_0 = (float*)calloc(2400, sizeof(float));
	autox_conv2d(tmp_10, weights + 658102, weights + 658198, relu6_76_tmp_0, tmp_10_dim, conv2d_81_w_0_dim, relu6_76_tmp_0_dim, 1, 0, 1, 1, 2);
	free(tmp_10);

	float* conv2d_161_tmp_1 = (float*)calloc(625, sizeof(float));
	autox_conv2d(relu6_76_tmp_0, weights + 667414, weights + 667439, conv2d_161_tmp_1, relu6_76_tmp_0_dim, conv2d_96_w_0_dim, conv2d_161_tmp_1_dim, 1, 0, 1, 1, 0);
	free(relu6_76_tmp_0);

	float* conv2d_162_tmp_1 = (float*)calloc(800, sizeof(float));
	autox_conv2d(relu6_76_tmp_0, weights + 669839, weights + 669871, conv2d_162_tmp_1, relu6_76_tmp_0_dim, conv2d_97_w_0_dim, conv2d_162_tmp_1_dim, 1, 0, 1, 1, 0);
	free(relu6_76_tmp_0);

	float* relu6_77_tmp_0 = (float*)calloc(25, sizeof(float));
	autox_conv2d(relu6_75_tmp_0, weights + 672943, weights + 672944, relu6_77_tmp_0, relu6_75_tmp_0_dim, conv2d_98_w_0_dim, relu6_77_tmp_0_dim, 1, 2, 1, 1, 2);
	free(relu6_75_tmp_0);

	float* batch_norm_81_tmp_2 = (float*)calloc(25, sizeof(float));
	autox_conv2d(relu6_77_tmp_0, weights + 675344, weights + 675345, batch_norm_81_tmp_2, relu6_77_tmp_0_dim, conv2d_99_w_0_dim, batch_norm_81_tmp_2_dim, 1, 0, 1, 1, 0);
	free(relu6_77_tmp_0);

	float* sigmoid_10_tmp_0 = (float*)calloc(25, sizeof(float));
	autox_sigmoid(batch_norm_81_tmp_2, batch_norm_81_tmp_2_dim);

	float* sigmoid_11_tmp_0 = (float*)calloc(625, sizeof(float));
	autox_sigmoid(conv2d_161_tmp_1, conv2d_161_tmp_1_dim);

	float* tmp_11 = (float*)calloc(625, sizeof(float));
	autox_elementwise_mul(sigmoid_11_tmp_0, sigmoid_10_tmp_0, tmp_11, sigmoid_11_tmp_0_dim, sigmoid_10_tmp_0_dim, tmp_11_dim, -1, 4, 4);
	free(sigmoid_11_tmp_0);
	free(sigmoid_10_tmp_0);

	float* tmp_12 = (float*)calloc(625, sizeof(float));
	autox_scale(tmp_11, tmp_11_dim);

	float* sqrt_3_tmp_0 = (float*)calloc(625, sizeof(float));
	autox_sqrt(tmp_12, tmp_12_dim);

	float* transpose_3_tmp_0 = (float*)calloc(800, sizeof(float));
	uint16_t axis_152[] = { 0, 2, 3, 1 };
	autox_transpose(conv2d_162_tmp_1, transpose_3_tmp_0, conv2d_162_tmp_1_dim, transpose_3_tmp_0_dim, axis_152, 4);
	free(conv2d_162_tmp_1);

	autox_softmax(reshape2_10_tmp_0, reshape2_10_tmp_0_dim, 1);

	float* linear_3_tmp_0 = (float*)calloc(1, sizeof(float));
	autox_matmul(softmax_3_tmp_0, weights + 675346, linear_3_tmp_0, softmax_3_tmp_0_dim, eager_tmp_4_dim, linear_3_tmp_0_dim, 0, 0, 2, 1, 1);
	free(softmax_3_tmp_0);

	float* p_155[] = { reshape2_0_tmp_0, reshape2_3_tmp_0, reshape2_6_tmp_0, reshape2_9_tmp_0, };
	uint16_t* p_155_dim[] = { reshape2_0_tmp_0_dim, reshape2_3_tmp_0_dim, reshape2_6_tmp_0_dim, reshape2_9_tmp_0_dim, };
	float* concat_4_tmp_0 = (float*)calloc(53125, sizeof(float));
	autox_concat(p_155, concat_4_tmp_0, p_155_dim, concat_4_tmp_0_dim, -1, 4, 3);
	free(reshape2_0_tmp_0);
	free(reshape2_3_tmp_0);
	free(reshape2_6_tmp_0);
	free(reshape2_9_tmp_0);

	float* p_156[] = { reshape2_2_tmp_0, reshape2_5_tmp_0, reshape2_8_tmp_0, reshape2_11_tmp_0, };
	uint16_t* p_156_dim[] = { reshape2_2_tmp_0_dim, reshape2_5_tmp_0_dim, reshape2_8_tmp_0_dim, reshape2_11_tmp_0_dim, };
	float* concat_5_tmp_0 = (float*)calloc(8500, sizeof(float));
	autox_concat(p_156, concat_5_tmp_0, p_156_dim, concat_5_tmp_0_dim, 1, 4, 3);
	free(reshape2_2_tmp_0);
	free(reshape2_5_tmp_0);
	free(reshape2_8_tmp_0);
	free(reshape2_11_tmp_0);

	float* split_0_tmp_0 = (float*)calloc(4250, sizeof(float));
	float* split_0_tmp_1 = (float*)calloc(4250, sizeof(float));
	float* p_157[] = { split_0_tmp_0, split_0_tmp_1, };
	uint16_t* p_157_dim[] = { split_0_tmp_0_dim, split_0_tmp_1_dim, };
	autox_split(concat_5_tmp_0, p_157, concat_5_tmp_0_dim, p_157_dim, 2, 3, 2, 3);
	free(concat_5_tmp_0);

	float* tmp_13 = (float*)calloc(4250, sizeof(float));
	autox_scale(split_0_tmp_0, split_0_tmp_0_dim);

	float* tmp_14 = (float*)calloc(4250, sizeof(float));
	autox_elementwise_add(tmp_13, weights + 675354, tmp_14, tmp_13_dim, eager_tmp_0_dim, tmp_14_dim, -1, 3, 2, 3);
	free(tmp_13);

	float* tmp_15 = (float*)calloc(4250, sizeof(float));
	autox_elementwise_add(split_0_tmp_1, weights + 679604, tmp_15, split_0_tmp_1_dim, eager_tmp_0_dim, tmp_15_dim, -1, 3, 2, 3);
	free(split_0_tmp_1);

	float* p_161[] = { tmp_14, tmp_15, };
	uint16_t* p_161_dim[] = { tmp_14_dim, tmp_15_dim, };
	float* concat_6_tmp_0 = (float*)calloc(8500, sizeof(float));
	autox_concat(p_161, concat_6_tmp_0, p_161_dim, concat_6_tmp_0_dim, -1, 2, 3);
	free(tmp_14);
	free(tmp_15);

	float* tmp_16 = (float*)calloc(8500, sizeof(float));
	autox_elementwise_mul(concat_6_tmp_0, weights + 683854, tmp_16, concat_6_tmp_0_dim, eager_tmp_1_dim, tmp_16_dim, -1, 3, 2, 3);
	free(concat_6_tmp_0);

	float* split_1_tmp_0 = (float*)calloc(1, sizeof(float));
	float* split_1_tmp_1 = (float*)calloc(1, sizeof(float));
	float* p_163[] = { split_1_tmp_0, split_1_tmp_1, };
	uint16_t* p_163_dim[] = { split_1_tmp_0_dim, split_1_tmp_1_dim, };
	autox_split(scale_factor, p_163, scale_factor_dim, p_163_dim, 1, 2, 2, 2);
	free(scale_factor);

	float* p_164[] = { split_1_tmp_1, split_1_tmp_0, split_1_tmp_1, split_1_tmp_0, };
	uint16_t* p_164_dim[] = { split_1_tmp_1_dim, split_1_tmp_0_dim, split_1_tmp_1_dim, split_1_tmp_0_dim, };
	float* concat_7_tmp_0 = (float*)calloc(4, sizeof(float));
	autox_concat(p_164, concat_7_tmp_0, p_164_dim, concat_7_tmp_0_dim, -1, 4, 2);
	free(split_1_tmp_1);
	free(split_1_tmp_0);
	free(split_1_tmp_1);
	free(split_1_tmp_0);

	float* tmp_17 = (float*)calloc(8500, sizeof(float));
	autox_elementwise_div(tmp_16, reshape2_12_tmp_0, tmp_17, tmp_16_dim, reshape2_12_tmp_0_dim, tmp_17_dim, -1, 3, 3);
	free(tmp_16);
	free(reshape2_12_tmp_0);

	float* multiclass_nms3_0_tmp_1 = (float*)calloc(1, sizeof(float));
	float* multiclass_nms3_0_tmp_2 = (float*)calloc(1, sizeof(float));
	float* multiclass_nms3_0_tmp_0 = (float*)calloc(6, sizeof(float));
	autox_multiclass_nms3(tmp_17, concat_4_tmp_0, multiclass_nms3_0_tmp_1, multiclass_nms3_0_tmp_2, multiclass_nms3_0_tmp_0, tmp_17_dim, concat_4_tmp_0_dim, multiclass_nms3_0_tmp_1_dim, multiclass_nms3_0_tmp_2_dim, multiclass_nms3_0_tmp_0_dim);
	free(tmp_17);
	free(concat_4_tmp_0);

}
