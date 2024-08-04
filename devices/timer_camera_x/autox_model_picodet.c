#include "autox_models.h"

void picodet_xs_320(const float* x, const void* weights, float* dets, float* boxes)
{
	uint16_t image_dim[] = { 1, 3, 320, 320 };
	uint16_t batch_norm2d_0_b_0_dim[] = { 8 };
	uint16_t conv2d_0_w_0_dim[] = { 8, 3, 3, 3 };
	uint16_t hardswish_43_tmp_0_dim[] = { 1, 8, 160, 160 };
	uint16_t batch_norm2d_1_b_0_dim[] = { 8 };
	uint16_t conv2d_1_w_0_dim[] = { 8, 1, 3, 3 };
	uint16_t hardswish_44_tmp_0_dim[] = { 1, 8, 160, 160 };
	uint16_t batch_norm2d_2_b_0_dim[] = { 16 };
	uint16_t conv2d_2_w_0_dim[] = { 16, 8, 1, 1 };
	uint16_t hardswish_45_tmp_0_dim[] = { 1, 16, 160, 160 };
	uint16_t batch_norm2d_3_b_0_dim[] = { 16 };
	uint16_t conv2d_3_w_0_dim[] = { 16, 1, 3, 3 };
	uint16_t hardswish_46_tmp_0_dim[] = { 1, 16, 80, 80 };
	uint16_t batch_norm2d_4_b_0_dim[] = { 24 };
	uint16_t conv2d_4_w_0_dim[] = { 24, 16, 1, 1 };
	uint16_t hardswish_47_tmp_0_dim[] = { 1, 24, 80, 80 };
	uint16_t batch_norm2d_5_b_0_dim[] = { 24 };
	uint16_t conv2d_5_w_0_dim[] = { 24, 1, 3, 3 };
	uint16_t hardswish_48_tmp_0_dim[] = { 1, 24, 80, 80 };
	uint16_t batch_norm2d_6_b_0_dim[] = { 24 };
	uint16_t conv2d_6_w_0_dim[] = { 24, 24, 1, 1 };
	uint16_t hardswish_49_tmp_0_dim[] = { 1, 24, 80, 80 };
	uint16_t batch_norm2d_7_b_0_dim[] = { 24 };
	uint16_t conv2d_7_w_0_dim[] = { 24, 1, 3, 3 };
	uint16_t hardswish_50_tmp_0_dim[] = { 1, 24, 40, 40 };
	uint16_t batch_norm2d_8_b_0_dim[] = { 48 };
	uint16_t conv2d_8_w_0_dim[] = { 48, 24, 1, 1 };
	uint16_t hardswish_51_tmp_0_dim[] = { 1, 48, 40, 40 };
	uint16_t batch_norm2d_9_b_0_dim[] = { 48 };
	uint16_t conv2d_9_w_0_dim[] = { 48, 1, 3, 3 };
	uint16_t hardswish_52_tmp_0_dim[] = { 1, 48, 40, 40 };
	uint16_t batch_norm2d_10_b_0_dim[] = { 48 };
	uint16_t conv2d_10_w_0_dim[] = { 48, 48, 1, 1 };
	uint16_t hardswish_53_tmp_0_dim[] = { 1, 48, 40, 40 };
	uint16_t batch_norm2d_11_b_0_dim[] = { 48 };
	uint16_t conv2d_11_w_0_dim[] = { 48, 1, 3, 3 };
	uint16_t hardswish_54_tmp_0_dim[] = { 1, 48, 20, 20 };
	uint16_t batch_norm2d_12_b_0_dim[] = { 88 };
	uint16_t conv2d_12_w_0_dim[] = { 88, 48, 1, 1 };
	uint16_t hardswish_55_tmp_0_dim[] = { 1, 88, 20, 20 };
	uint16_t batch_norm2d_13_b_0_dim[] = { 88 };
	uint16_t conv2d_13_w_0_dim[] = { 88, 1, 5, 5 };
	uint16_t hardswish_56_tmp_0_dim[] = { 1, 88, 20, 20 };
	uint16_t batch_norm2d_14_b_0_dim[] = { 88 };
	uint16_t conv2d_14_w_0_dim[] = { 88, 88, 1, 1 };
	uint16_t hardswish_57_tmp_0_dim[] = { 1, 88, 20, 20 };
	uint16_t batch_norm2d_15_b_0_dim[] = { 88 };
	uint16_t conv2d_15_w_0_dim[] = { 88, 1, 5, 5 };
	uint16_t hardswish_58_tmp_0_dim[] = { 1, 88, 20, 20 };
	uint16_t batch_norm2d_16_b_0_dim[] = { 88 };
	uint16_t conv2d_16_w_0_dim[] = { 88, 88, 1, 1 };
	uint16_t hardswish_59_tmp_0_dim[] = { 1, 88, 20, 20 };
	uint16_t batch_norm2d_17_b_0_dim[] = { 88 };
	uint16_t conv2d_17_w_0_dim[] = { 88, 1, 5, 5 };
	uint16_t hardswish_60_tmp_0_dim[] = { 1, 88, 20, 20 };
	uint16_t batch_norm2d_18_b_0_dim[] = { 88 };
	uint16_t conv2d_18_w_0_dim[] = { 88, 88, 1, 1 };
	uint16_t hardswish_61_tmp_0_dim[] = { 1, 88, 20, 20 };
	uint16_t batch_norm2d_19_b_0_dim[] = { 88 };
	uint16_t conv2d_19_w_0_dim[] = { 88, 1, 5, 5 };
	uint16_t hardswish_62_tmp_0_dim[] = { 1, 88, 20, 20 };
	uint16_t batch_norm2d_20_b_0_dim[] = { 88 };
	uint16_t conv2d_20_w_0_dim[] = { 88, 88, 1, 1 };
	uint16_t hardswish_63_tmp_0_dim[] = { 1, 88, 20, 20 };
	uint16_t batch_norm2d_21_b_0_dim[] = { 88 };
	uint16_t conv2d_21_w_0_dim[] = { 88, 1, 5, 5 };
	uint16_t hardswish_64_tmp_0_dim[] = { 1, 88, 20, 20 };
	uint16_t batch_norm2d_22_b_0_dim[] = { 88 };
	uint16_t conv2d_22_w_0_dim[] = { 88, 88, 1, 1 };
	uint16_t hardswish_65_tmp_0_dim[] = { 1, 88, 20, 20 };
	uint16_t batch_norm2d_23_b_0_dim[] = { 88 };
	uint16_t conv2d_23_w_0_dim[] = { 88, 1, 5, 5 };
	uint16_t hardswish_66_tmp_0_dim[] = { 1, 88, 10, 10 };
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
	uint16_t hardswish_67_tmp_0_dim[] = { 1, 176, 10, 10 };
	uint16_t batch_norm2d_25_b_0_dim[] = { 176 };
	uint16_t conv2d_27_w_0_dim[] = { 176, 1, 5, 5 };
	uint16_t hardswish_68_tmp_0_dim[] = { 1, 176, 10, 10 };
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
	uint16_t hardswish_69_tmp_0_dim[] = { 1, 176, 10, 10 };
	uint16_t batch_norm2d_29_b_0_dim[] = { 96 };
	uint16_t conv2d_33_w_0_dim[] = { 96, 176, 1, 1 };
	uint16_t hardswish_72_tmp_0_dim[] = { 1, 96, 10, 10 };
	uint16_t nearest_interp_v2_0_tmp_0_dim[] = { 1, 96, 20, 20 };
	uint16_t batch_norm2d_28_b_0_dim[] = { 96 };
	uint16_t conv2d_32_w_0_dim[] = { 96, 88, 1, 1 };
	uint16_t hardswish_71_tmp_0_dim[] = { 1, 96, 20, 20 };
	uint16_t concat_0_tmp_0_dim[] = { 1, 192, 20, 20 };
	uint16_t batch_norm2d_34_b_0_dim[] = { 192 };
	uint16_t conv2d_38_w_0_dim[] = { 192, 1, 5, 5 };
	uint16_t hardswish_73_tmp_0_dim[] = { 1, 192, 20, 20 };
	uint16_t batch_norm2d_35_b_0_dim[] = { 192 };
	uint16_t conv2d_39_w_0_dim[] = { 192, 192, 1, 1 };
	uint16_t hardswish_74_tmp_0_dim[] = { 1, 192, 20, 20 };
	uint16_t batch_norm2d_36_b_0_dim[] = { 192 };
	uint16_t conv2d_40_w_0_dim[] = { 192, 1, 5, 5 };
	uint16_t hardswish_75_tmp_0_dim[] = { 1, 192, 20, 20 };
	uint16_t batch_norm2d_37_b_0_dim[] = { 96 };
	uint16_t conv2d_41_w_0_dim[] = { 96, 192, 1, 1 };
	uint16_t hardswish_76_tmp_0_dim[] = { 1, 96, 20, 20 };
	uint16_t nearest_interp_v2_1_tmp_0_dim[] = { 1, 96, 40, 40 };
	uint16_t batch_norm2d_27_b_0_dim[] = { 96 };
	uint16_t conv2d_31_w_0_dim[] = { 96, 48, 1, 1 };
	uint16_t hardswish_70_tmp_0_dim[] = { 1, 96, 40, 40 };
	uint16_t concat_1_tmp_0_dim[] = { 1, 192, 40, 40 };
	uint16_t batch_norm2d_38_b_0_dim[] = { 192 };
	uint16_t conv2d_42_w_0_dim[] = { 192, 1, 5, 5 };
	uint16_t hardswish_77_tmp_0_dim[] = { 1, 192, 40, 40 };
	uint16_t batch_norm2d_39_b_0_dim[] = { 192 };
	uint16_t conv2d_43_w_0_dim[] = { 192, 192, 1, 1 };
	uint16_t hardswish_78_tmp_0_dim[] = { 1, 192, 40, 40 };
	uint16_t batch_norm2d_40_b_0_dim[] = { 192 };
	uint16_t conv2d_44_w_0_dim[] = { 192, 1, 5, 5 };
	uint16_t hardswish_79_tmp_0_dim[] = { 1, 192, 40, 40 };
	uint16_t batch_norm2d_41_b_0_dim[] = { 96 };
	uint16_t conv2d_45_w_0_dim[] = { 96, 192, 1, 1 };
	uint16_t hardswish_80_tmp_0_dim[] = { 1, 96, 40, 40 };
	uint16_t batch_norm2d_42_b_0_dim[] = { 96 };
	uint16_t conv2d_46_w_0_dim[] = { 96, 1, 5, 5 };
	uint16_t hardswish_81_tmp_0_dim[] = { 1, 96, 20, 20 };
	uint16_t batch_norm2d_43_b_0_dim[] = { 96 };
	uint16_t conv2d_47_w_0_dim[] = { 96, 96, 1, 1 };
	uint16_t hardswish_82_tmp_0_dim[] = { 1, 96, 20, 20 };
	uint16_t concat_2_tmp_0_dim[] = { 1, 192, 20, 20 };
	uint16_t batch_norm2d_44_b_0_dim[] = { 192 };
	uint16_t conv2d_48_w_0_dim[] = { 192, 1, 5, 5 };
	uint16_t hardswish_83_tmp_0_dim[] = { 1, 192, 20, 20 };
	uint16_t batch_norm2d_45_b_0_dim[] = { 192 };
	uint16_t conv2d_49_w_0_dim[] = { 192, 192, 1, 1 };
	uint16_t hardswish_84_tmp_0_dim[] = { 1, 192, 20, 20 };
	uint16_t batch_norm2d_46_b_0_dim[] = { 192 };
	uint16_t conv2d_50_w_0_dim[] = { 192, 1, 5, 5 };
	uint16_t hardswish_85_tmp_0_dim[] = { 1, 192, 20, 20 };
	uint16_t batch_norm2d_47_b_0_dim[] = { 96 };
	uint16_t conv2d_51_w_0_dim[] = { 96, 192, 1, 1 };
	uint16_t hardswish_86_tmp_0_dim[] = { 1, 96, 20, 20 };
	uint16_t batch_norm2d_48_b_0_dim[] = { 96 };
	uint16_t conv2d_52_w_0_dim[] = { 96, 1, 5, 5 };
	uint16_t hardswish_87_tmp_0_dim[] = { 1, 96, 10, 10 };
	uint16_t batch_norm2d_49_b_0_dim[] = { 96 };
	uint16_t conv2d_53_w_0_dim[] = { 96, 96, 1, 1 };
	uint16_t hardswish_88_tmp_0_dim[] = { 1, 96, 10, 10 };
	uint16_t concat_3_tmp_0_dim[] = { 1, 192, 10, 10 };
	uint16_t batch_norm2d_30_b_0_dim[] = { 96 };
	uint16_t conv2d_34_w_0_dim[] = { 96, 1, 5, 5 };
	uint16_t hardswish_93_tmp_0_dim[] = { 1, 96, 5, 5 };
	uint16_t batch_norm2d_31_b_0_dim[] = { 96 };
	uint16_t conv2d_35_w_0_dim[] = { 96, 96, 1, 1 };
	uint16_t hardswish_94_tmp_0_dim[] = { 1, 96, 5, 5 };
	uint16_t batch_norm2d_50_b_0_dim[] = { 192 };
	uint16_t conv2d_54_w_0_dim[] = { 192, 1, 5, 5 };
	uint16_t hardswish_89_tmp_0_dim[] = { 1, 192, 10, 10 };
	uint16_t batch_norm2d_51_b_0_dim[] = { 192 };
	uint16_t conv2d_55_w_0_dim[] = { 192, 192, 1, 1 };
	uint16_t hardswish_90_tmp_0_dim[] = { 1, 192, 10, 10 };
	uint16_t batch_norm2d_52_b_0_dim[] = { 192 };
	uint16_t conv2d_56_w_0_dim[] = { 192, 1, 5, 5 };
	uint16_t hardswish_91_tmp_0_dim[] = { 1, 192, 10, 10 };
	uint16_t batch_norm2d_53_b_0_dim[] = { 96 };
	uint16_t conv2d_57_w_0_dim[] = { 96, 192, 1, 1 };
	uint16_t hardswish_92_tmp_0_dim[] = { 1, 96, 10, 10 };
	uint16_t batch_norm2d_32_b_0_dim[] = { 96 };
	uint16_t conv2d_36_w_0_dim[] = { 96, 1, 5, 5 };
	uint16_t hardswish_95_tmp_0_dim[] = { 1, 96, 5, 5 };
	uint16_t batch_norm2d_33_b_0_dim[] = { 96 };
	uint16_t conv2d_37_w_0_dim[] = { 96, 96, 1, 1 };
	uint16_t hardswish_96_tmp_0_dim[] = { 1, 96, 5, 5 };
	uint16_t tmp_0_dim[] = { 1, 96, 5, 5 };
	uint16_t batch_norm2d_54_b_0_dim[] = { 96 };
	uint16_t conv2d_58_w_0_dim[] = { 96, 1, 5, 5 };
	uint16_t hardswish_97_tmp_0_dim[] = { 1, 96, 40, 40 };
	uint16_t batch_norm2d_55_b_0_dim[] = { 96 };
	uint16_t conv2d_59_w_0_dim[] = { 96, 96, 1, 1 };
	uint16_t hardswish_98_tmp_0_dim[] = { 1, 96, 40, 40 };
	uint16_t batch_norm2d_56_b_0_dim[] = { 96 };
	uint16_t conv2d_60_w_0_dim[] = { 96, 1, 5, 5 };
	uint16_t hardswish_99_tmp_0_dim[] = { 1, 96, 40, 40 };
	uint16_t batch_norm2d_57_b_0_dim[] = { 96 };
	uint16_t conv2d_61_w_0_dim[] = { 96, 96, 1, 1 };
	uint16_t hardswish_100_tmp_0_dim[] = { 1, 96, 40, 40 };
	uint16_t pool2d_2_tmp_0_dim[] = { 1, 96, 1, 1 };
	uint16_t conv2d_135_tmp_1_dim[] = { 1, 96, 1, 1 };
	uint16_t conv2d_62_b_0_dim[] = { 96 };
	uint16_t conv2d_62_w_0_dim[] = { 96, 96, 1, 1 };
	uint16_t sigmoid_0_tmp_0_dim[] = { 1, 96, 1, 1 };
	uint16_t tmp_1_dim[] = { 1, 96, 40, 40 };
	uint16_t batch_norm2d_58_b_0_dim[] = { 96 };
	uint16_t conv2d_63_w_0_dim[] = { 96, 96, 1, 1 };
	uint16_t hardswish_101_tmp_0_dim[] = { 1, 96, 40, 40 };
	uint16_t conv2d_137_tmp_1_dim[] = { 1, 80, 40, 40 };
	uint16_t conv2d_84_b_0_dim[] = { 80 };
	uint16_t conv2d_84_w_0_dim[] = { 80, 96, 1, 1 };
	uint16_t conv2d_138_tmp_1_dim[] = { 1, 32, 40, 40 };
	uint16_t conv2d_85_b_0_dim[] = { 32 };
	uint16_t conv2d_85_w_0_dim[] = { 32, 96, 1, 1 };
	uint16_t batch_norm2d_74_b_0_dim[] = { 1 };
	uint16_t conv2d_86_w_0_dim[] = { 1, 96, 5, 5 };
	uint16_t hardswish_102_tmp_0_dim[] = { 1, 1, 40, 40 };
	uint16_t batch_norm2d_75_b_0_dim[] = { 1 };
	uint16_t batch_norm_60_tmp_2_dim[] = { 1, 1, 40, 40 };
	uint16_t conv2d_87_w_0_dim[] = { 1, 1, 1, 1 };
	uint16_t sigmoid_1_tmp_0_dim[] = { 1, 1, 40, 40 };
	uint16_t sigmoid_2_tmp_0_dim[] = { 1, 80, 40, 40 };
	uint16_t tmp_2_dim[] = { 1, 80, 40, 40 };
	uint16_t tmp_3_dim[] = { 1, 80, 40, 40 };
	uint16_t sqrt_0_tmp_0_dim[] = { 1, 80, 40, 40 };
	uint16_t reshape2_0_tmp_0_dim[] = { 1, 80, 1600 };
	uint16_t reshape2_0_tmp_1_dim[] = { 0, 1, 80, 40, 40 };
	uint16_t transpose_0_tmp_0_dim[] = { 1, 40, 40, 32 };
	uint16_t transpose_0_tmp_1_dim[] = { 0, 1, 32, 40, 40 };
	uint16_t reshape2_1_tmp_0_dim[] = { 6400, 8 };

	uint16_t reshape2_1_tmp_1_dim[] = { 0, 1, 40, 40, 32 };
	uint16_t softmax_0_tmp_0_dim[] = { 6400, 8 };
	uint16_t eager_tmp_4_dim[] = { 8 };
	uint16_t linear_0_tmp_0_dim[] = { 1 };
	uint16_t reshape2_2_tmp_0_dim[] = { 1, 1600, 4 };
	uint16_t reshape2_2_tmp_1_dim[] = { 0, 1 };
	uint16_t batch_norm2d_59_b_0_dim[] = { 96 };
	uint16_t conv2d_64_w_0_dim[] = { 96, 1, 5, 5 };
	uint16_t hardswish_103_tmp_0_dim[] = { 1, 96, 20, 20 };
	uint16_t batch_norm2d_60_b_0_dim[] = { 96 };
	uint16_t conv2d_65_w_0_dim[] = { 96, 96, 1, 1 };
	uint16_t hardswish_104_tmp_0_dim[] = { 1, 96, 20, 20 };
	uint16_t batch_norm2d_61_b_0_dim[] = { 96 };
	uint16_t conv2d_66_w_0_dim[] = { 96, 1, 5, 5 };
	uint16_t hardswish_105_tmp_0_dim[] = { 1, 96, 20, 20 };
	uint16_t batch_norm2d_62_b_0_dim[] = { 96 };
	uint16_t conv2d_67_w_0_dim[] = { 96, 96, 1, 1 };
	uint16_t hardswish_106_tmp_0_dim[] = { 1, 96, 20, 20 };
	uint16_t pool2d_3_tmp_0_dim[] = { 1, 96, 1, 1 };
	uint16_t conv2d_143_tmp_1_dim[] = { 1, 96, 1, 1 };
	uint16_t conv2d_68_b_0_dim[] = { 96 };
	uint16_t conv2d_68_w_0_dim[] = { 96, 96, 1, 1 };
	uint16_t sigmoid_3_tmp_0_dim[] = { 1, 96, 1, 1 };
	uint16_t tmp_4_dim[] = { 1, 96, 20, 20 };
	uint16_t batch_norm2d_63_b_0_dim[] = { 96 };
	uint16_t conv2d_69_w_0_dim[] = { 96, 96, 1, 1 };
	uint16_t hardswish_107_tmp_0_dim[] = { 1, 96, 20, 20 };
	uint16_t conv2d_145_tmp_1_dim[] = { 1, 80, 20, 20 };
	uint16_t conv2d_88_b_0_dim[] = { 80 };
	uint16_t conv2d_88_w_0_dim[] = { 80, 96, 1, 1 };
	uint16_t conv2d_146_tmp_1_dim[] = { 1, 32, 20, 20 };
	uint16_t conv2d_89_b_0_dim[] = { 32 };
	uint16_t conv2d_89_w_0_dim[] = { 32, 96, 1, 1 };
	uint16_t batch_norm2d_76_b_0_dim[] = { 1 };
	uint16_t conv2d_90_w_0_dim[] = { 1, 96, 5, 5 };
	uint16_t hardswish_108_tmp_0_dim[] = { 1, 1, 20, 20 };
	uint16_t batch_norm2d_77_b_0_dim[] = { 1 };
	uint16_t batch_norm_67_tmp_2_dim[] = { 1, 1, 20, 20 };
	uint16_t conv2d_91_w_0_dim[] = { 1, 1, 1, 1 };
	uint16_t sigmoid_4_tmp_0_dim[] = { 1, 1, 20, 20 };
	uint16_t sigmoid_5_tmp_0_dim[] = { 1, 80, 20, 20 };
	uint16_t tmp_5_dim[] = { 1, 80, 20, 20 };
	uint16_t tmp_6_dim[] = { 1, 80, 20, 20 };
	uint16_t sqrt_1_tmp_0_dim[] = { 1, 80, 20, 20 };
	uint16_t reshape2_3_tmp_0_dim[] = { 1, 80, 400 };
	uint16_t reshape2_3_tmp_1_dim[] = { 0, 1, 80, 20, 20 };
	uint16_t transpose_1_tmp_0_dim[] = { 1, 20, 20, 32 };
	uint16_t transpose_1_tmp_1_dim[] = { 0, 1, 32, 20, 20 };
	uint16_t reshape2_4_tmp_0_dim[] = { 1600, 8 };

	uint16_t reshape2_4_tmp_1_dim[] = { 0, 1, 20, 20, 32 };
	uint16_t softmax_1_tmp_0_dim[] = { 1600, 8 };
	uint16_t linear_1_tmp_0_dim[] = { 1 };
	uint16_t reshape2_5_tmp_0_dim[] = { 1, 400, 4 };
	uint16_t reshape2_5_tmp_1_dim[] = { 0, 1 };
	uint16_t batch_norm2d_64_b_0_dim[] = { 96 };
	uint16_t conv2d_70_w_0_dim[] = { 96, 1, 5, 5 };
	uint16_t hardswish_109_tmp_0_dim[] = { 1, 96, 10, 10 };
	uint16_t batch_norm2d_65_b_0_dim[] = { 96 };
	uint16_t conv2d_71_w_0_dim[] = { 96, 96, 1, 1 };
	uint16_t hardswish_110_tmp_0_dim[] = { 1, 96, 10, 10 };
	uint16_t batch_norm2d_66_b_0_dim[] = { 96 };
	uint16_t conv2d_72_w_0_dim[] = { 96, 1, 5, 5 };
	uint16_t hardswish_111_tmp_0_dim[] = { 1, 96, 10, 10 };
	uint16_t batch_norm2d_67_b_0_dim[] = { 96 };
	uint16_t conv2d_73_w_0_dim[] = { 96, 96, 1, 1 };
	uint16_t hardswish_112_tmp_0_dim[] = { 1, 96, 10, 10 };
	uint16_t pool2d_4_tmp_0_dim[] = { 1, 96, 1, 1 };
	uint16_t conv2d_151_tmp_1_dim[] = { 1, 96, 1, 1 };
	uint16_t conv2d_74_b_0_dim[] = { 96 };
	uint16_t conv2d_74_w_0_dim[] = { 96, 96, 1, 1 };
	uint16_t sigmoid_6_tmp_0_dim[] = { 1, 96, 1, 1 };
	uint16_t tmp_7_dim[] = { 1, 96, 10, 10 };
	uint16_t batch_norm2d_68_b_0_dim[] = { 96 };
	uint16_t conv2d_75_w_0_dim[] = { 96, 96, 1, 1 };
	uint16_t hardswish_113_tmp_0_dim[] = { 1, 96, 10, 10 };
	uint16_t conv2d_153_tmp_1_dim[] = { 1, 80, 10, 10 };
	uint16_t conv2d_92_b_0_dim[] = { 80 };
	uint16_t conv2d_92_w_0_dim[] = { 80, 96, 1, 1 };
	uint16_t conv2d_154_tmp_1_dim[] = { 1, 32, 10, 10 };
	uint16_t conv2d_93_b_0_dim[] = { 32 };
	uint16_t conv2d_93_w_0_dim[] = { 32, 96, 1, 1 };
	uint16_t batch_norm2d_78_b_0_dim[] = { 1 };
	uint16_t conv2d_94_w_0_dim[] = { 1, 96, 5, 5 };
	uint16_t hardswish_114_tmp_0_dim[] = { 1, 1, 10, 10 };
	uint16_t batch_norm2d_79_b_0_dim[] = { 1 };
	uint16_t batch_norm_74_tmp_2_dim[] = { 1, 1, 10, 10 };
	uint16_t conv2d_95_w_0_dim[] = { 1, 1, 1, 1 };
	uint16_t sigmoid_7_tmp_0_dim[] = { 1, 1, 10, 10 };
	uint16_t sigmoid_8_tmp_0_dim[] = { 1, 80, 10, 10 };
	uint16_t tmp_8_dim[] = { 1, 80, 10, 10 };
	uint16_t tmp_9_dim[] = { 1, 80, 10, 10 };
	uint16_t sqrt_2_tmp_0_dim[] = { 1, 80, 10, 10 };
	uint16_t reshape2_6_tmp_0_dim[] = { 1, 80, 100 };
	uint16_t reshape2_6_tmp_1_dim[] = { 0, 1, 80, 10, 10 };
	uint16_t transpose_2_tmp_0_dim[] = { 1, 10, 10, 32 };
	uint16_t transpose_2_tmp_1_dim[] = { 0, 1, 32, 10, 10 };
	uint16_t reshape2_7_tmp_0_dim[] = { 400, 8 };

	uint16_t reshape2_7_tmp_1_dim[] = { 0, 1, 10, 10, 32 };
	uint16_t softmax_2_tmp_0_dim[] = { 400, 8 };
	uint16_t linear_2_tmp_0_dim[] = { 1 };
	uint16_t reshape2_8_tmp_0_dim[] = { 1, 100, 4 };
	uint16_t reshape2_8_tmp_1_dim[] = { 0, 1 };
	uint16_t batch_norm2d_69_b_0_dim[] = { 96 };
	uint16_t conv2d_76_w_0_dim[] = { 96, 1, 5, 5 };
	uint16_t hardswish_115_tmp_0_dim[] = { 1, 96, 5, 5 };
	uint16_t batch_norm2d_70_b_0_dim[] = { 96 };
	uint16_t conv2d_77_w_0_dim[] = { 96, 96, 1, 1 };
	uint16_t hardswish_116_tmp_0_dim[] = { 1, 96, 5, 5 };
	uint16_t batch_norm2d_71_b_0_dim[] = { 96 };
	uint16_t conv2d_78_w_0_dim[] = { 96, 1, 5, 5 };
	uint16_t hardswish_117_tmp_0_dim[] = { 1, 96, 5, 5 };
	uint16_t batch_norm2d_72_b_0_dim[] = { 96 };
	uint16_t conv2d_79_w_0_dim[] = { 96, 96, 1, 1 };
	uint16_t hardswish_118_tmp_0_dim[] = { 1, 96, 5, 5 };
	uint16_t pool2d_5_tmp_0_dim[] = { 1, 96, 1, 1 };
	uint16_t conv2d_159_tmp_1_dim[] = { 1, 96, 1, 1 };
	uint16_t conv2d_80_b_0_dim[] = { 96 };
	uint16_t conv2d_80_w_0_dim[] = { 96, 96, 1, 1 };
	uint16_t sigmoid_9_tmp_0_dim[] = { 1, 96, 1, 1 };
	uint16_t tmp_10_dim[] = { 1, 96, 5, 5 };
	uint16_t batch_norm2d_73_b_0_dim[] = { 96 };
	uint16_t conv2d_81_w_0_dim[] = { 96, 96, 1, 1 };
	uint16_t hardswish_119_tmp_0_dim[] = { 1, 96, 5, 5 };
	uint16_t conv2d_161_tmp_1_dim[] = { 1, 80, 5, 5 };
	uint16_t conv2d_96_b_0_dim[] = { 80 };
	uint16_t conv2d_96_w_0_dim[] = { 80, 96, 1, 1 };
	uint16_t conv2d_162_tmp_1_dim[] = { 1, 32, 5, 5 };
	uint16_t conv2d_97_b_0_dim[] = { 32 };
	uint16_t conv2d_97_w_0_dim[] = { 32, 96, 1, 1 };
	uint16_t batch_norm2d_80_b_0_dim[] = { 1 };
	uint16_t conv2d_98_w_0_dim[] = { 1, 96, 5, 5 };
	uint16_t hardswish_120_tmp_0_dim[] = { 1, 1, 5, 5 };
	uint16_t batch_norm2d_81_b_0_dim[] = { 1 };
	uint16_t batch_norm_81_tmp_2_dim[] = { 1, 1, 5, 5 };
	uint16_t conv2d_99_w_0_dim[] = { 1, 1, 1, 1 };
	uint16_t sigmoid_10_tmp_0_dim[] = { 1, 1, 5, 5 };
	uint16_t sigmoid_11_tmp_0_dim[] = { 1, 80, 5, 5 };
	uint16_t tmp_11_dim[] = { 1, 80, 5, 5 };
	uint16_t tmp_12_dim[] = { 1, 80, 5, 5 };
	uint16_t sqrt_3_tmp_0_dim[] = { 1, 80, 5, 5 };
	uint16_t reshape2_9_tmp_0_dim[] = { 1, 80, 25 };
	uint16_t reshape2_9_tmp_1_dim[] = { 0, 1, 80, 5, 5 };
	uint16_t transpose_3_tmp_0_dim[] = { 1, 5, 5, 32 };
	uint16_t transpose_3_tmp_1_dim[] = { 0, 1, 32, 5, 5 };
	uint16_t reshape2_10_tmp_0_dim[] = { 100, 8 };

	uint16_t reshape2_10_tmp_1_dim[] = { 0, 1, 5, 5, 32 };
	uint16_t softmax_3_tmp_0_dim[] = { 100, 8 };
	uint16_t linear_3_tmp_0_dim[] = { 1 };
	uint16_t reshape2_11_tmp_0_dim[] = { 1, 25, 4 };
	uint16_t reshape2_11_tmp_1_dim[] = { 0, 1 };
	uint16_t concat_4_tmp_0_dim[] = { 1, 80, 2125 };
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

	float* hardswish_43_tmp_0 = (float*)calloc(204800, sizeof(float));
	autox_conv2d(x, (float*)((int8_t*)weights) + 0, (float*)((int8_t*)weights) + 8, hardswish_43_tmp_0, image_dim, conv2d_0_w_0_dim, hardswish_43_tmp_0_dim, 1, 1, 2, 1, 10);
	free(x);

	float* hardswish_44_tmp_0 = (float*)calloc(204800, sizeof(float));
	autox_conv2d(hardswish_43_tmp_0, (float*)((int8_t*)weights) + 224, (float*)((int8_t*)weights) + 232, hardswish_44_tmp_0, hardswish_43_tmp_0_dim, conv2d_1_w_0_dim, hardswish_44_tmp_0_dim, 8, 1, 1, 1, 10);
	free(hardswish_43_tmp_0);

	float* hardswish_45_tmp_0 = (float*)calloc(409600, sizeof(float));
	autox_conv2d(hardswish_44_tmp_0, (float*)((int8_t*)weights) + 304, (float*)((int8_t*)weights) + 320, hardswish_45_tmp_0, hardswish_44_tmp_0_dim, conv2d_2_w_0_dim, hardswish_45_tmp_0_dim, 1, 0, 1, 1, 10);
	free(hardswish_44_tmp_0);

	float* hardswish_46_tmp_0 = (float*)calloc(102400, sizeof(float));
	autox_conv2d(hardswish_45_tmp_0, (float*)((int8_t*)weights) + 448, (float*)((int8_t*)weights) + 464, hardswish_46_tmp_0, hardswish_45_tmp_0_dim, conv2d_3_w_0_dim, hardswish_46_tmp_0_dim, 16, 1, 2, 1, 10);
	free(hardswish_45_tmp_0);

	float* hardswish_47_tmp_0 = (float*)calloc(153600, sizeof(float));
	autox_conv2d(hardswish_46_tmp_0, (float*)((int8_t*)weights) + 608, (float*)((int8_t*)weights) + 632, hardswish_47_tmp_0, hardswish_46_tmp_0_dim, conv2d_4_w_0_dim, hardswish_47_tmp_0_dim, 1, 0, 1, 1, 10);
	free(hardswish_46_tmp_0);

	float* hardswish_48_tmp_0 = (float*)calloc(153600, sizeof(float));
	autox_conv2d(hardswish_47_tmp_0, (float*)((int8_t*)weights) + 1016, (float*)((int8_t*)weights) + 1040, hardswish_48_tmp_0, hardswish_47_tmp_0_dim, conv2d_5_w_0_dim, hardswish_48_tmp_0_dim, 24, 1, 1, 1, 10);
	free(hardswish_47_tmp_0);

	float* hardswish_49_tmp_0 = (float*)calloc(153600, sizeof(float));
	autox_conv2d(hardswish_48_tmp_0, (float*)((int8_t*)weights) + 1256, (float*)((int8_t*)weights) + 1280, hardswish_49_tmp_0, hardswish_48_tmp_0_dim, conv2d_6_w_0_dim, hardswish_49_tmp_0_dim, 1, 0, 1, 1, 10);
	free(hardswish_48_tmp_0);

	float* hardswish_50_tmp_0 = (float*)calloc(38400, sizeof(float));
	autox_conv2d(hardswish_49_tmp_0, (float*)((int8_t*)weights) + 1856, (float*)((int8_t*)weights) + 1880, hardswish_50_tmp_0, hardswish_49_tmp_0_dim, conv2d_7_w_0_dim, hardswish_50_tmp_0_dim, 24, 1, 2, 1, 10);
	free(hardswish_49_tmp_0);

	float* hardswish_51_tmp_0 = (float*)calloc(76800, sizeof(float));
	autox_conv2d(hardswish_50_tmp_0, (float*)((int8_t*)weights) + 2096, (float*)((int8_t*)weights) + 2144, hardswish_51_tmp_0, hardswish_50_tmp_0_dim, conv2d_8_w_0_dim, hardswish_51_tmp_0_dim, 1, 0, 1, 1, 10);
	free(hardswish_50_tmp_0);

	float* hardswish_52_tmp_0 = (float*)calloc(76800, sizeof(float));
	autox_conv2d(hardswish_51_tmp_0, (float*)((int8_t*)weights) + 3296, (float*)((int8_t*)weights) + 3344, hardswish_52_tmp_0, hardswish_51_tmp_0_dim, conv2d_9_w_0_dim, hardswish_52_tmp_0_dim, 48, 1, 1, 1, 10);
	free(hardswish_51_tmp_0);

	float* hardswish_53_tmp_0 = (float*)calloc(76800, sizeof(float));
	autox_conv2d(hardswish_52_tmp_0, (float*)((int8_t*)weights) + 3776, (float*)((int8_t*)weights) + 3824, hardswish_53_tmp_0, hardswish_52_tmp_0_dim, conv2d_10_w_0_dim, hardswish_53_tmp_0_dim, 1, 0, 1, 1, 10);
	free(hardswish_52_tmp_0);

	float* hardswish_54_tmp_0 = (float*)calloc(19200, sizeof(float));
	autox_conv2d(hardswish_53_tmp_0, (float*)((int8_t*)weights) + 6128, (float*)((int8_t*)weights) + 6176, hardswish_54_tmp_0, hardswish_53_tmp_0_dim, conv2d_11_w_0_dim, hardswish_54_tmp_0_dim, 48, 1, 2, 1, 10);
	// free(hardswish_53_tmp_0);

	float* hardswish_55_tmp_0 = (float*)calloc(35200, sizeof(float));
	autox_conv2d(hardswish_54_tmp_0, (float*)((int8_t*)weights) + 6608, (float*)((int8_t*)weights) + 6696, hardswish_55_tmp_0, hardswish_54_tmp_0_dim, conv2d_12_w_0_dim, hardswish_55_tmp_0_dim, 1, 0, 1, 1, 10);
	free(hardswish_54_tmp_0);

	float* hardswish_56_tmp_0 = (float*)calloc(35200, sizeof(float));
	autox_conv2d(hardswish_55_tmp_0, (float*)((int8_t*)weights) + 10920, (float*)((int8_t*)weights) + 11008, hardswish_56_tmp_0, hardswish_55_tmp_0_dim, conv2d_13_w_0_dim, hardswish_56_tmp_0_dim, 88, 2, 1, 1, 10);
	free(hardswish_55_tmp_0);

	float* hardswish_57_tmp_0 = (float*)calloc(35200, sizeof(float));
	autox_conv2d(hardswish_56_tmp_0, (float*)((int8_t*)weights) + 13208, (float*)((int8_t*)weights) + 13296, hardswish_57_tmp_0, hardswish_56_tmp_0_dim, conv2d_14_w_0_dim, hardswish_57_tmp_0_dim, 1, 0, 1, 1, 10);
	free(hardswish_56_tmp_0);

	float* hardswish_58_tmp_0 = (float*)calloc(35200, sizeof(float));
	autox_conv2d(hardswish_57_tmp_0, (float*)((int8_t*)weights) + 21040, (float*)((int8_t*)weights) + 21128, hardswish_58_tmp_0, hardswish_57_tmp_0_dim, conv2d_15_w_0_dim, hardswish_58_tmp_0_dim, 88, 2, 1, 1, 10);
	free(hardswish_57_tmp_0);

	float* hardswish_59_tmp_0 = (float*)calloc(35200, sizeof(float));
	autox_conv2d(hardswish_58_tmp_0, (float*)((int8_t*)weights) + 23328, (float*)((int8_t*)weights) + 23416, hardswish_59_tmp_0, hardswish_58_tmp_0_dim, conv2d_16_w_0_dim, hardswish_59_tmp_0_dim, 1, 0, 1, 1, 10);
	free(hardswish_58_tmp_0);

	float* hardswish_60_tmp_0 = (float*)calloc(35200, sizeof(float));
	autox_conv2d(hardswish_59_tmp_0, (float*)((int8_t*)weights) + 31160, (float*)((int8_t*)weights) + 31248, hardswish_60_tmp_0, hardswish_59_tmp_0_dim, conv2d_17_w_0_dim, hardswish_60_tmp_0_dim, 88, 2, 1, 1, 10);
	free(hardswish_59_tmp_0);

	float* hardswish_61_tmp_0 = (float*)calloc(35200, sizeof(float));
	autox_conv2d(hardswish_60_tmp_0, (float*)((int8_t*)weights) + 33448, (float*)((int8_t*)weights) + 33536, hardswish_61_tmp_0, hardswish_60_tmp_0_dim, conv2d_18_w_0_dim, hardswish_61_tmp_0_dim, 1, 0, 1, 1, 10);
	free(hardswish_60_tmp_0);

	float* hardswish_62_tmp_0 = (float*)calloc(35200, sizeof(float));
	autox_conv2d(hardswish_61_tmp_0, (float*)((int8_t*)weights) + 41280, (float*)((int8_t*)weights) + 41368, hardswish_62_tmp_0, hardswish_61_tmp_0_dim, conv2d_19_w_0_dim, hardswish_62_tmp_0_dim, 88, 2, 1, 1, 10);
	free(hardswish_61_tmp_0);

	float* hardswish_63_tmp_0 = (float*)calloc(35200, sizeof(float));
	autox_conv2d(hardswish_62_tmp_0, (float*)((int8_t*)weights) + 43568, (float*)((int8_t*)weights) + 43656, hardswish_63_tmp_0, hardswish_62_tmp_0_dim, conv2d_20_w_0_dim, hardswish_63_tmp_0_dim, 1, 0, 1, 1, 10);
	free(hardswish_62_tmp_0);

	float* hardswish_64_tmp_0 = (float*)calloc(35200, sizeof(float));
	autox_conv2d(hardswish_63_tmp_0, (float*)((int8_t*)weights) + 51400, (float*)((int8_t*)weights) + 51488, hardswish_64_tmp_0, hardswish_63_tmp_0_dim, conv2d_21_w_0_dim, hardswish_64_tmp_0_dim, 88, 2, 1, 1, 10);
	free(hardswish_63_tmp_0);

	float* hardswish_65_tmp_0 = (float*)calloc(35200, sizeof(float));
	autox_conv2d(hardswish_64_tmp_0, (float*)((int8_t*)weights) + 53688, (float*)((int8_t*)weights) + 53776, hardswish_65_tmp_0, hardswish_64_tmp_0_dim, conv2d_22_w_0_dim, hardswish_65_tmp_0_dim, 1, 0, 1, 1, 10);
	free(hardswish_64_tmp_0);

	float* hardswish_66_tmp_0 = (float*)calloc(8800, sizeof(float));
	autox_conv2d(hardswish_65_tmp_0, (float*)((int8_t*)weights) + 61520, (float*)((int8_t*)weights) + 61608, hardswish_66_tmp_0, hardswish_65_tmp_0_dim, conv2d_23_w_0_dim, hardswish_66_tmp_0_dim, 88, 2, 2, 1, 10);
	//free(hardswish_65_tmp_0);

	float* pool2d_0_tmp_0 = (float*)calloc(88, sizeof(float));
	autox_pool2d(hardswish_66_tmp_0, pool2d_0_tmp_0, hardswish_66_tmp_0_dim, pool2d_0_tmp_0_dim, 1, 1, 0, 1, 1);
	// free(hardswish_66_tmp_0);

	float* relu_0_tmp_0 = (float*)calloc(22, sizeof(float));
	autox_conv2d(pool2d_0_tmp_0, (float*)((int8_t*)weights) + 63808, (float*)((int8_t*)weights) + 63830, relu_0_tmp_0, pool2d_0_tmp_0_dim, conv2d_24_w_0_dim, relu_0_tmp_0_dim, 1, 0, 1, 1, 1);
	free(pool2d_0_tmp_0);

	float* conv2d_113_tmp_1 = (float*)calloc(88, sizeof(float));
	autox_conv2d(relu_0_tmp_0, (float*)((int8_t*)weights) + 65766, (float*)((int8_t*)weights) + 65854, conv2d_113_tmp_1, relu_0_tmp_0_dim, conv2d_25_w_0_dim, conv2d_113_tmp_1_dim, 1, 0, 1, 1, 0);
	free(relu_0_tmp_0);

	autox_hard_sigmoid(conv2d_113_tmp_1, conv2d_113_tmp_1_dim, 4, 0.5, 0.1666667);

	float* elementwise_mul_0 = (float*)calloc(8800, sizeof(float));
	autox_elementwise_mul(hardswish_66_tmp_0, conv2d_113_tmp_1, elementwise_mul_0, hardswish_66_tmp_0_dim, hardsigmoid_2_tmp_0_dim, elementwise_mul_0_dim, -1, 4, 4, 4);
	free(hardswish_66_tmp_0);
	free(conv2d_113_tmp_1);

	float* hardswish_67_tmp_0 = (float*)calloc(17600, sizeof(float));
	autox_conv2d(elementwise_mul_0, (float*)((int8_t*)weights) + 67790, (float*)((int8_t*)weights) + 67966, hardswish_67_tmp_0, elementwise_mul_0_dim, conv2d_26_w_0_dim, hardswish_67_tmp_0_dim, 1, 0, 1, 1, 10);
	free(elementwise_mul_0);

	float* hardswish_68_tmp_0 = (float*)calloc(17600, sizeof(float));
	autox_conv2d(hardswish_67_tmp_0, (float*)((int8_t*)weights) + 83454, (float*)((int8_t*)weights) + 83630, hardswish_68_tmp_0, hardswish_67_tmp_0_dim, conv2d_27_w_0_dim, hardswish_68_tmp_0_dim, 176, 2, 1, 1, 10);
	free(hardswish_67_tmp_0);

	float* pool2d_1_tmp_0 = (float*)calloc(176, sizeof(float));
	autox_pool2d(hardswish_68_tmp_0, pool2d_1_tmp_0, hardswish_68_tmp_0_dim, pool2d_1_tmp_0_dim, 1, 1, 0, 1, 1);
	//free(hardswish_68_tmp_0);

	float* relu_1_tmp_0 = (float*)calloc(44, sizeof(float));
	autox_conv2d(pool2d_1_tmp_0, (float*)((int8_t*)weights) + 88030, (float*)((int8_t*)weights) + 88074, relu_1_tmp_0, pool2d_1_tmp_0_dim, conv2d_28_w_0_dim, relu_1_tmp_0_dim, 1, 0, 1, 1, 1);
	free(pool2d_1_tmp_0);

	float* conv2d_116_tmp_1 = (float*)calloc(176, sizeof(float));
	autox_conv2d(relu_1_tmp_0, (float*)((int8_t*)weights) + 95818, (float*)((int8_t*)weights) + 95994, conv2d_116_tmp_1, relu_1_tmp_0_dim, conv2d_29_w_0_dim, conv2d_116_tmp_1_dim, 1, 0, 1, 1, 0);
	free(relu_1_tmp_0);

	autox_hard_sigmoid(conv2d_116_tmp_1, conv2d_116_tmp_1_dim, 4, 0.5, 0.1666667);

	float* elementwise_mul_1 = (float*)calloc(17600, sizeof(float));
	autox_elementwise_mul(hardswish_68_tmp_0, conv2d_116_tmp_1, elementwise_mul_1, hardswish_68_tmp_0_dim, hardsigmoid_3_tmp_0_dim, elementwise_mul_1_dim, -1, 4, 4, 4);
	free(hardswish_68_tmp_0);
	free(conv2d_116_tmp_1);

	float* hardswish_69_tmp_0 = (float*)calloc(17600, sizeof(float));
	autox_conv2d(elementwise_mul_1, (float*)((int8_t*)weights) + 103738, (float*)((int8_t*)weights) + 103914, hardswish_69_tmp_0, elementwise_mul_1_dim, conv2d_30_w_0_dim, hardswish_69_tmp_0_dim, 1, 0, 1, 1, 10);
	free(elementwise_mul_1);

	float* hardswish_72_tmp_0 = (float*)calloc(9600, sizeof(float));
	autox_conv2d(hardswish_69_tmp_0, (float*)((int8_t*)weights) + 134890, (float*)((int8_t*)weights) + 134986, hardswish_72_tmp_0, hardswish_69_tmp_0_dim, conv2d_33_w_0_dim, hardswish_72_tmp_0_dim, 1, 0, 1, 1, 10);
	free(hardswish_69_tmp_0);

	float* nearest_interp_v2_0_tmp_0 = (float*)calloc(38400, sizeof(float));
	autox_nearest_interp(hardswish_72_tmp_0, nearest_interp_v2_0_tmp_0, hardswish_72_tmp_0_dim, nearest_interp_v2_0_tmp_0_dim, 2, 0);
	//free(hardswish_72_tmp_0);

	float* hardswish_71_tmp_0 = (float*)calloc(38400, sizeof(float));
	autox_conv2d(hardswish_65_tmp_0, (float*)((int8_t*)weights) + 151882, (float*)((int8_t*)weights) + 151978, hardswish_71_tmp_0, hardswish_65_tmp_0_dim, conv2d_32_w_0_dim, hardswish_71_tmp_0_dim, 1, 0, 1, 1, 10);
	free(hardswish_65_tmp_0);

	float* p_40[] = { nearest_interp_v2_0_tmp_0, hardswish_71_tmp_0, };
	uint16_t* p_40_dim[] = { nearest_interp_v2_0_tmp_0_dim, hardswish_71_tmp_0_dim, };
	float* concat_0_tmp_0 = (float*)calloc(76800, sizeof(float));
	autox_concat(p_40, concat_0_tmp_0, p_40_dim, concat_0_tmp_0_dim, 1, 2, 4);
	free(nearest_interp_v2_0_tmp_0);
	free(hardswish_71_tmp_0);

	float* hardswish_73_tmp_0 = (float*)calloc(76800, sizeof(float));
	autox_conv2d(concat_0_tmp_0, (float*)((int8_t*)weights) + 160426, (float*)((int8_t*)weights) + 160618, hardswish_73_tmp_0, concat_0_tmp_0_dim, conv2d_38_w_0_dim, hardswish_73_tmp_0_dim, 192, 2, 1, 1, 10);
	free(concat_0_tmp_0);

	float* hardswish_74_tmp_0 = (float*)calloc(76800, sizeof(float));
	autox_conv2d(hardswish_73_tmp_0, (float*)((int8_t*)weights) + 165418, (float*)((int8_t*)weights) + 165610, hardswish_74_tmp_0, hardswish_73_tmp_0_dim, conv2d_39_w_0_dim, hardswish_74_tmp_0_dim, 1, 0, 1, 1, 10);
	free(hardswish_73_tmp_0);

	float* hardswish_75_tmp_0 = (float*)calloc(76800, sizeof(float));
	autox_conv2d(hardswish_74_tmp_0, (float*)((int8_t*)weights) + 202474, (float*)((int8_t*)weights) + 202666, hardswish_75_tmp_0, hardswish_74_tmp_0_dim, conv2d_40_w_0_dim, hardswish_75_tmp_0_dim, 192, 2, 1, 1, 10);
	free(hardswish_74_tmp_0);

	float* hardswish_76_tmp_0 = (float*)calloc(38400, sizeof(float));
	autox_conv2d(hardswish_75_tmp_0, (float*)((int8_t*)weights) + 207466, (float*)((int8_t*)weights) + 207562, hardswish_76_tmp_0, hardswish_75_tmp_0_dim, conv2d_41_w_0_dim, hardswish_76_tmp_0_dim, 1, 0, 1, 1, 10);
	free(hardswish_75_tmp_0);

	float* nearest_interp_v2_1_tmp_0 = (float*)calloc(153600, sizeof(float));
	autox_nearest_interp(hardswish_76_tmp_0, nearest_interp_v2_1_tmp_0, hardswish_76_tmp_0_dim, nearest_interp_v2_1_tmp_0_dim, 2, 0);
	//free(hardswish_76_tmp_0);

	float* hardswish_70_tmp_0 = (float*)calloc(153600, sizeof(float));
	autox_conv2d(hardswish_53_tmp_0, (float*)((int8_t*)weights) + 225994, (float*)((int8_t*)weights) + 226090, hardswish_70_tmp_0, hardswish_53_tmp_0_dim, conv2d_31_w_0_dim, hardswish_70_tmp_0_dim, 1, 0, 1, 1, 10);
	free(hardswish_53_tmp_0);

	float* p_47[] = { nearest_interp_v2_1_tmp_0, hardswish_70_tmp_0, };
	uint16_t* p_47_dim[] = { nearest_interp_v2_1_tmp_0_dim, hardswish_70_tmp_0_dim, };
	float* concat_1_tmp_0 = (float*)calloc(307200, sizeof(float));
	autox_concat(p_47, concat_1_tmp_0, p_47_dim, concat_1_tmp_0_dim, 1, 2, 4);
	free(nearest_interp_v2_1_tmp_0);
	free(hardswish_70_tmp_0);

	float* hardswish_77_tmp_0 = (float*)calloc(307200, sizeof(float));
	autox_conv2d(concat_1_tmp_0, (float*)((int8_t*)weights) + 230698, (float*)((int8_t*)weights) + 230890, hardswish_77_tmp_0, concat_1_tmp_0_dim, conv2d_42_w_0_dim, hardswish_77_tmp_0_dim, 192, 2, 1, 1, 10);
	free(concat_1_tmp_0);

	float* hardswish_78_tmp_0 = (float*)calloc(307200, sizeof(float));
	autox_conv2d(hardswish_77_tmp_0, (float*)((int8_t*)weights) + 235690, (float*)((int8_t*)weights) + 235882, hardswish_78_tmp_0, hardswish_77_tmp_0_dim, conv2d_43_w_0_dim, hardswish_78_tmp_0_dim, 1, 0, 1, 1, 10);
	free(hardswish_77_tmp_0);

	float* hardswish_79_tmp_0 = (float*)calloc(307200, sizeof(float));
	autox_conv2d(hardswish_78_tmp_0, (float*)((int8_t*)weights) + 272746, (float*)((int8_t*)weights) + 272938, hardswish_79_tmp_0, hardswish_78_tmp_0_dim, conv2d_44_w_0_dim, hardswish_79_tmp_0_dim, 192, 2, 1, 1, 10);
	free(hardswish_78_tmp_0);

	float* hardswish_80_tmp_0 = (float*)calloc(153600, sizeof(float));
	autox_conv2d(hardswish_79_tmp_0, (float*)((int8_t*)weights) + 277738, (float*)((int8_t*)weights) + 277834, hardswish_80_tmp_0, hardswish_79_tmp_0_dim, conv2d_45_w_0_dim, hardswish_80_tmp_0_dim, 1, 0, 1, 1, 10);
	free(hardswish_79_tmp_0);

	float* hardswish_81_tmp_0 = (float*)calloc(38400, sizeof(float));
	autox_conv2d(hardswish_80_tmp_0, (float*)((int8_t*)weights) + 296266, (float*)((int8_t*)weights) + 296362, hardswish_81_tmp_0, hardswish_80_tmp_0_dim, conv2d_46_w_0_dim, hardswish_81_tmp_0_dim, 96, 2, 2, 1, 10);
	//free(hardswish_80_tmp_0);

	float* hardswish_82_tmp_0 = (float*)calloc(38400, sizeof(float));
	autox_conv2d(hardswish_81_tmp_0, (float*)((int8_t*)weights) + 298762, (float*)((int8_t*)weights) + 298858, hardswish_82_tmp_0, hardswish_81_tmp_0_dim, conv2d_47_w_0_dim, hardswish_82_tmp_0_dim, 1, 0, 1, 1, 10);
	free(hardswish_81_tmp_0);

	float* p_54[] = { hardswish_82_tmp_0, hardswish_76_tmp_0, };
	uint16_t* p_54_dim[] = { hardswish_82_tmp_0_dim, hardswish_76_tmp_0_dim, };
	float* concat_2_tmp_0 = (float*)calloc(76800, sizeof(float));
	autox_concat(p_54, concat_2_tmp_0, p_54_dim, concat_2_tmp_0_dim, 1, 2, 4);
	free(hardswish_82_tmp_0);
	free(hardswish_76_tmp_0);

	float* hardswish_83_tmp_0 = (float*)calloc(76800, sizeof(float));
	autox_conv2d(concat_2_tmp_0, (float*)((int8_t*)weights) + 308074, (float*)((int8_t*)weights) + 308266, hardswish_83_tmp_0, concat_2_tmp_0_dim, conv2d_48_w_0_dim, hardswish_83_tmp_0_dim, 192, 2, 1, 1, 10);
	free(concat_2_tmp_0);

	float* hardswish_84_tmp_0 = (float*)calloc(76800, sizeof(float));
	autox_conv2d(hardswish_83_tmp_0, (float*)((int8_t*)weights) + 313066, (float*)((int8_t*)weights) + 313258, hardswish_84_tmp_0, hardswish_83_tmp_0_dim, conv2d_49_w_0_dim, hardswish_84_tmp_0_dim, 1, 0, 1, 1, 10);
	free(hardswish_83_tmp_0);

	float* hardswish_85_tmp_0 = (float*)calloc(76800, sizeof(float));
	autox_conv2d(hardswish_84_tmp_0, (float*)((int8_t*)weights) + 350122, (float*)((int8_t*)weights) + 350314, hardswish_85_tmp_0, hardswish_84_tmp_0_dim, conv2d_50_w_0_dim, hardswish_85_tmp_0_dim, 192, 2, 1, 1, 10);
	free(hardswish_84_tmp_0);

	float* hardswish_86_tmp_0 = (float*)calloc(38400, sizeof(float));
	autox_conv2d(hardswish_85_tmp_0, (float*)((int8_t*)weights) + 355114, (float*)((int8_t*)weights) + 355210, hardswish_86_tmp_0, hardswish_85_tmp_0_dim, conv2d_51_w_0_dim, hardswish_86_tmp_0_dim, 1, 0, 1, 1, 10);
	free(hardswish_85_tmp_0);

	float* hardswish_87_tmp_0 = (float*)calloc(9600, sizeof(float));
	autox_conv2d(hardswish_86_tmp_0, (float*)((int8_t*)weights) + 373642, (float*)((int8_t*)weights) + 373738, hardswish_87_tmp_0, hardswish_86_tmp_0_dim, conv2d_52_w_0_dim, hardswish_87_tmp_0_dim, 96, 2, 2, 1, 10);
	//free(hardswish_86_tmp_0);

	float* hardswish_88_tmp_0 = (float*)calloc(9600, sizeof(float));
	autox_conv2d(hardswish_87_tmp_0, (float*)((int8_t*)weights) + 376138, (float*)((int8_t*)weights) + 376234, hardswish_88_tmp_0, hardswish_87_tmp_0_dim, conv2d_53_w_0_dim, hardswish_88_tmp_0_dim, 1, 0, 1, 1, 10);
	free(hardswish_87_tmp_0);

	float* p_61[] = { hardswish_88_tmp_0, hardswish_72_tmp_0, };
	uint16_t* p_61_dim[] = { hardswish_88_tmp_0_dim, hardswish_72_tmp_0_dim, };
	float* concat_3_tmp_0 = (float*)calloc(19200, sizeof(float));
	autox_concat(p_61, concat_3_tmp_0, p_61_dim, concat_3_tmp_0_dim, 1, 2, 4);
	free(hardswish_88_tmp_0);
	//free(hardswish_72_tmp_0);

	float* hardswish_93_tmp_0 = (float*)calloc(2400, sizeof(float));
	autox_conv2d(hardswish_72_tmp_0, (float*)((int8_t*)weights) + 385450, (float*)((int8_t*)weights) + 385546, hardswish_93_tmp_0, hardswish_72_tmp_0_dim, conv2d_34_w_0_dim, hardswish_93_tmp_0_dim, 96, 2, 2, 1, 10);
	free(hardswish_72_tmp_0);

	float* hardswish_94_tmp_0 = (float*)calloc(2400, sizeof(float));
	autox_conv2d(hardswish_93_tmp_0, (float*)((int8_t*)weights) + 387946, (float*)((int8_t*)weights) + 388042, hardswish_94_tmp_0, hardswish_93_tmp_0_dim, conv2d_35_w_0_dim, hardswish_94_tmp_0_dim, 1, 0, 1, 1, 10);
	free(hardswish_93_tmp_0);

	float* hardswish_89_tmp_0 = (float*)calloc(19200, sizeof(float));
	autox_conv2d(concat_3_tmp_0, (float*)((int8_t*)weights) + 397258, (float*)((int8_t*)weights) + 397450, hardswish_89_tmp_0, concat_3_tmp_0_dim, conv2d_54_w_0_dim, hardswish_89_tmp_0_dim, 192, 2, 1, 1, 10);
	free(concat_3_tmp_0);

	float* hardswish_90_tmp_0 = (float*)calloc(19200, sizeof(float));
	autox_conv2d(hardswish_89_tmp_0, (float*)((int8_t*)weights) + 402250, (float*)((int8_t*)weights) + 402442, hardswish_90_tmp_0, hardswish_89_tmp_0_dim, conv2d_55_w_0_dim, hardswish_90_tmp_0_dim, 1, 0, 1, 1, 10);
	free(hardswish_89_tmp_0);

	float* hardswish_91_tmp_0 = (float*)calloc(19200, sizeof(float));
	autox_conv2d(hardswish_90_tmp_0, (float*)((int8_t*)weights) + 439306, (float*)((int8_t*)weights) + 439498, hardswish_91_tmp_0, hardswish_90_tmp_0_dim, conv2d_56_w_0_dim, hardswish_91_tmp_0_dim, 192, 2, 1, 1, 10);
	free(hardswish_90_tmp_0);

	float* hardswish_92_tmp_0 = (float*)calloc(9600, sizeof(float));
	autox_conv2d(hardswish_91_tmp_0, (float*)((int8_t*)weights) + 444298, (float*)((int8_t*)weights) + 444394, hardswish_92_tmp_0, hardswish_91_tmp_0_dim, conv2d_57_w_0_dim, hardswish_92_tmp_0_dim, 1, 0, 1, 1, 10);
	free(hardswish_91_tmp_0);

	float* hardswish_95_tmp_0 = (float*)calloc(2400, sizeof(float));
	autox_conv2d(hardswish_92_tmp_0, (float*)((int8_t*)weights) + 462826, (float*)((int8_t*)weights) + 462922, hardswish_95_tmp_0, hardswish_92_tmp_0_dim, conv2d_36_w_0_dim, hardswish_95_tmp_0_dim, 96, 2, 2, 1, 10);
	//free(hardswish_92_tmp_0);

	float* hardswish_96_tmp_0 = (float*)calloc(2400, sizeof(float));
	autox_conv2d(hardswish_95_tmp_0, (float*)((int8_t*)weights) + 465322, (float*)((int8_t*)weights) + 465418, hardswish_96_tmp_0, hardswish_95_tmp_0_dim, conv2d_37_w_0_dim, hardswish_96_tmp_0_dim, 1, 0, 1, 1, 10);
	free(hardswish_95_tmp_0);

	float* tmp_0 = (float*)calloc(2400, sizeof(float));
	autox_elementwise_add(hardswish_94_tmp_0, hardswish_96_tmp_0, tmp_0, hardswish_94_tmp_0_dim, hardswish_96_tmp_0_dim, tmp_0_dim, -1, 4, 4, 4);
	free(hardswish_94_tmp_0);
	free(hardswish_96_tmp_0);

	float* hardswish_97_tmp_0 = (float*)calloc(153600, sizeof(float));
	autox_conv2d(hardswish_80_tmp_0, (float*)((int8_t*)weights) + 474634, (float*)((int8_t*)weights) + 474730, hardswish_97_tmp_0, hardswish_80_tmp_0_dim, conv2d_58_w_0_dim, hardswish_97_tmp_0_dim, 96, 2, 1, 1, 10);
	free(hardswish_80_tmp_0);

	float* hardswish_98_tmp_0 = (float*)calloc(153600, sizeof(float));
	autox_conv2d(hardswish_97_tmp_0, (float*)((int8_t*)weights) + 477130, (float*)((int8_t*)weights) + 477226, hardswish_98_tmp_0, hardswish_97_tmp_0_dim, conv2d_59_w_0_dim, hardswish_98_tmp_0_dim, 1, 0, 1, 1, 10);
	free(hardswish_97_tmp_0);

	float* hardswish_99_tmp_0 = (float*)calloc(153600, sizeof(float));
	autox_conv2d(hardswish_98_tmp_0, (float*)((int8_t*)weights) + 486442, (float*)((int8_t*)weights) + 486538, hardswish_99_tmp_0, hardswish_98_tmp_0_dim, conv2d_60_w_0_dim, hardswish_99_tmp_0_dim, 96, 2, 1, 1, 10);
	free(hardswish_98_tmp_0);

	float* hardswish_100_tmp_0 = (float*)calloc(153600, sizeof(float));
	autox_conv2d(hardswish_99_tmp_0, (float*)((int8_t*)weights) + 488938, (float*)((int8_t*)weights) + 489034, hardswish_100_tmp_0, hardswish_99_tmp_0_dim, conv2d_61_w_0_dim, hardswish_100_tmp_0_dim, 1, 0, 1, 1, 10);
	free(hardswish_99_tmp_0);

	float* pool2d_2_tmp_0 = (float*)calloc(96, sizeof(float));
	autox_pool2d(hardswish_100_tmp_0, pool2d_2_tmp_0, hardswish_100_tmp_0_dim, pool2d_2_tmp_0_dim, 1, 1, 0, 1, 1);
	//free(hardswish_100_tmp_0);

	float* conv2d_135_tmp_1 = (float*)calloc(96, sizeof(float));
	autox_conv2d(pool2d_2_tmp_0, (float*)((int8_t*)weights) + 498250, (float*)((int8_t*)weights) + 498346, conv2d_135_tmp_1, pool2d_2_tmp_0_dim, conv2d_62_w_0_dim, conv2d_135_tmp_1_dim, 1, 0, 1, 1, 0);
	free(pool2d_2_tmp_0);

	autox_sigmoid(conv2d_135_tmp_1, conv2d_135_tmp_1_dim, 4);

	float* tmp_1 = (float*)calloc(153600, sizeof(float));
	autox_elementwise_mul(hardswish_100_tmp_0, conv2d_135_tmp_1, tmp_1, hardswish_100_tmp_0_dim, sigmoid_0_tmp_0_dim, tmp_1_dim, -1, 4, 4, 4);
	//free(hardswish_100_tmp_0);
	free(conv2d_135_tmp_1);

	float* hardswish_101_tmp_0 = (float*)calloc(153600, sizeof(float));
	autox_conv2d(tmp_1, (float*)((int8_t*)weights) + 507562, (float*)((int8_t*)weights) + 507658, hardswish_101_tmp_0, tmp_1_dim, conv2d_63_w_0_dim, hardswish_101_tmp_0_dim, 1, 0, 1, 1, 10);
	free(tmp_1);

	float* conv2d_137_tmp_1 = (float*)calloc(128000, sizeof(float));
	autox_conv2d(hardswish_101_tmp_0, (float*)((int8_t*)weights) + 516874, (float*)((int8_t*)weights) + 516954, conv2d_137_tmp_1, hardswish_101_tmp_0_dim, conv2d_84_w_0_dim, conv2d_137_tmp_1_dim, 1, 0, 1, 1, 0);
	//free(hardswish_101_tmp_0);

	float* conv2d_138_tmp_1 = (float*)calloc(51200, sizeof(float));
	autox_conv2d(hardswish_101_tmp_0, (float*)((int8_t*)weights) + 524634, (float*)((int8_t*)weights) + 524666, conv2d_138_tmp_1, hardswish_101_tmp_0_dim, conv2d_85_w_0_dim, conv2d_138_tmp_1_dim, 1, 0, 1, 1, 0);
	free(hardswish_101_tmp_0);

	float* hardswish_102_tmp_0 = (float*)calloc(1600, sizeof(float));
	autox_conv2d(hardswish_100_tmp_0, (float*)((int8_t*)weights) + 527738, (float*)((int8_t*)weights) + 527739, hardswish_102_tmp_0, hardswish_100_tmp_0_dim, conv2d_86_w_0_dim, hardswish_102_tmp_0_dim, 1, 2, 1, 1, 10);
	free(hardswish_100_tmp_0);

	float* batch_norm_60_tmp_2 = (float*)calloc(1600, sizeof(float));
	autox_conv2d(hardswish_102_tmp_0, (float*)((int8_t*)weights) + 530139, (float*)((int8_t*)weights) + 530140, batch_norm_60_tmp_2, hardswish_102_tmp_0_dim, conv2d_87_w_0_dim, batch_norm_60_tmp_2_dim, 1, 0, 1, 1, 0);
	free(hardswish_102_tmp_0);

	autox_sigmoid(batch_norm_60_tmp_2, batch_norm_60_tmp_2_dim, 4);

	autox_sigmoid(conv2d_137_tmp_1, conv2d_137_tmp_1_dim, 4);

	float* tmp_2 = (float*)calloc(128000, sizeof(float));
	autox_elementwise_mul(conv2d_137_tmp_1, batch_norm_60_tmp_2, tmp_2, sigmoid_2_tmp_0_dim, sigmoid_1_tmp_0_dim, tmp_2_dim, -1, 4, 4, 4);
	free(conv2d_137_tmp_1);
	free(batch_norm_60_tmp_2);

	autox_scale(tmp_2, tmp_2_dim, 4, 1e-09, 1, 1.0);

	autox_sqrt(tmp_2, tmp_3_dim, 4);

	float* transpose_0_tmp_0 = (float*)calloc(51200, sizeof(float));
	uint16_t axis_89[] = { 0, 2, 3, 1 };
	autox_transpose(conv2d_138_tmp_1, transpose_0_tmp_0, conv2d_138_tmp_1_dim, transpose_0_tmp_0_dim, axis_89, 4);
	free(conv2d_138_tmp_1);

	autox_softmax(transpose_0_tmp_0, reshape2_1_tmp_0_dim[0], reshape2_1_tmp_0_dim[1]);

	float* linear_0_tmp_0 = (float*)calloc(6400, sizeof(float));
	autox_matmul(transpose_0_tmp_0, (float*)((int8_t*)weights) + 530141, linear_0_tmp_0, softmax_0_tmp_0_dim, eager_tmp_4_dim, linear_0_tmp_0_dim, 0, 0, 2, 1, 1);
	free(transpose_0_tmp_0);

	float* hardswish_103_tmp_0 = (float*)calloc(38400, sizeof(float));
	autox_conv2d(hardswish_86_tmp_0, (float*)((int8_t*)weights) + 530149, (float*)((int8_t*)weights) + 530245, hardswish_103_tmp_0, hardswish_86_tmp_0_dim, conv2d_64_w_0_dim, hardswish_103_tmp_0_dim, 96, 2, 1, 1, 10);
	free(hardswish_86_tmp_0);

	float* hardswish_104_tmp_0 = (float*)calloc(38400, sizeof(float));
	autox_conv2d(hardswish_103_tmp_0, (float*)((int8_t*)weights) + 532645, (float*)((int8_t*)weights) + 532741, hardswish_104_tmp_0, hardswish_103_tmp_0_dim, conv2d_65_w_0_dim, hardswish_104_tmp_0_dim, 1, 0, 1, 1, 10);
	free(hardswish_103_tmp_0);

	float* hardswish_105_tmp_0 = (float*)calloc(38400, sizeof(float));
	autox_conv2d(hardswish_104_tmp_0, (float*)((int8_t*)weights) + 541957, (float*)((int8_t*)weights) + 542053, hardswish_105_tmp_0, hardswish_104_tmp_0_dim, conv2d_66_w_0_dim, hardswish_105_tmp_0_dim, 96, 2, 1, 1, 10);
	free(hardswish_104_tmp_0);

	float* hardswish_106_tmp_0 = (float*)calloc(38400, sizeof(float));
	autox_conv2d(hardswish_105_tmp_0, (float*)((int8_t*)weights) + 544453, (float*)((int8_t*)weights) + 544549, hardswish_106_tmp_0, hardswish_105_tmp_0_dim, conv2d_67_w_0_dim, hardswish_106_tmp_0_dim, 1, 0, 1, 1, 10);
	free(hardswish_105_tmp_0);

	float* pool2d_3_tmp_0 = (float*)calloc(96, sizeof(float));
	autox_pool2d(hardswish_106_tmp_0, pool2d_3_tmp_0, hardswish_106_tmp_0_dim, pool2d_3_tmp_0_dim, 1, 1, 0, 1, 1);
	//free(hardswish_106_tmp_0);

	float* conv2d_143_tmp_1 = (float*)calloc(96, sizeof(float));
	autox_conv2d(pool2d_3_tmp_0, (float*)((int8_t*)weights) + 553765, (float*)((int8_t*)weights) + 553861, conv2d_143_tmp_1, pool2d_3_tmp_0_dim, conv2d_68_w_0_dim, conv2d_143_tmp_1_dim, 1, 0, 1, 1, 0);
	free(pool2d_3_tmp_0);

	autox_sigmoid(conv2d_143_tmp_1, conv2d_143_tmp_1_dim, 4);

	float* tmp_4 = (float*)calloc(38400, sizeof(float));
	autox_elementwise_mul(hardswish_106_tmp_0, conv2d_143_tmp_1, tmp_4, hardswish_106_tmp_0_dim, sigmoid_3_tmp_0_dim, tmp_4_dim, -1, 4, 4, 4);
	//free(hardswish_106_tmp_0);
	free(conv2d_143_tmp_1);

	float* hardswish_107_tmp_0 = (float*)calloc(38400, sizeof(float));
	autox_conv2d(tmp_4, (float*)((int8_t*)weights) + 563077, (float*)((int8_t*)weights) + 563173, hardswish_107_tmp_0, tmp_4_dim, conv2d_69_w_0_dim, hardswish_107_tmp_0_dim, 1, 0, 1, 1, 10);
	free(tmp_4);

	float* conv2d_145_tmp_1 = (float*)calloc(32000, sizeof(float));
	autox_conv2d(hardswish_107_tmp_0, (float*)((int8_t*)weights) + 572389, (float*)((int8_t*)weights) + 572469, conv2d_145_tmp_1, hardswish_107_tmp_0_dim, conv2d_88_w_0_dim, conv2d_145_tmp_1_dim, 1, 0, 1, 1, 0);
	//free(hardswish_107_tmp_0);

	float* conv2d_146_tmp_1 = (float*)calloc(12800, sizeof(float));
	autox_conv2d(hardswish_107_tmp_0, (float*)((int8_t*)weights) + 580149, (float*)((int8_t*)weights) + 580181, conv2d_146_tmp_1, hardswish_107_tmp_0_dim, conv2d_89_w_0_dim, conv2d_146_tmp_1_dim, 1, 0, 1, 1, 0);
	free(hardswish_107_tmp_0);

	float* hardswish_108_tmp_0 = (float*)calloc(400, sizeof(float));
	autox_conv2d(hardswish_106_tmp_0, (float*)((int8_t*)weights) + 583253, (float*)((int8_t*)weights) + 583254, hardswish_108_tmp_0, hardswish_106_tmp_0_dim, conv2d_90_w_0_dim, hardswish_108_tmp_0_dim, 1, 2, 1, 1, 10);
	free(hardswish_106_tmp_0);

	float* batch_norm_67_tmp_2 = (float*)calloc(400, sizeof(float));
	autox_conv2d(hardswish_108_tmp_0, (float*)((int8_t*)weights) + 585654, (float*)((int8_t*)weights) + 585655, batch_norm_67_tmp_2, hardswish_108_tmp_0_dim, conv2d_91_w_0_dim, batch_norm_67_tmp_2_dim, 1, 0, 1, 1, 0);
	free(hardswish_108_tmp_0);

	autox_sigmoid(batch_norm_67_tmp_2, batch_norm_67_tmp_2_dim, 4);

	autox_sigmoid(conv2d_145_tmp_1, conv2d_145_tmp_1_dim, 4);

	float* tmp_5 = (float*)calloc(32000, sizeof(float));
	autox_elementwise_mul(conv2d_145_tmp_1, batch_norm_67_tmp_2, tmp_5, sigmoid_5_tmp_0_dim, sigmoid_4_tmp_0_dim, tmp_5_dim, -1, 4, 4, 4);
	free(conv2d_145_tmp_1);
	free(batch_norm_67_tmp_2);

	autox_scale(tmp_5, tmp_5_dim, 4, 1e-09, 1, 1.0);

	autox_sqrt(tmp_5, tmp_6_dim, 4);

	float* transpose_1_tmp_0 = (float*)calloc(12800, sizeof(float));
	uint16_t axis_110[] = { 0, 2, 3, 1 };
	autox_transpose(conv2d_146_tmp_1, transpose_1_tmp_0, conv2d_146_tmp_1_dim, transpose_1_tmp_0_dim, axis_110, 4);
	free(conv2d_146_tmp_1);

	autox_softmax(transpose_1_tmp_0, reshape2_4_tmp_0_dim[0], reshape2_4_tmp_0_dim[1]);

	float* linear_1_tmp_0 = (float*)calloc(1600, sizeof(float));
	autox_matmul(transpose_1_tmp_0, (float*)((int8_t*)weights) + 585656, linear_1_tmp_0, softmax_1_tmp_0_dim, eager_tmp_4_dim, linear_1_tmp_0_dim, 0, 0, 2, 1, 1);
	free(transpose_1_tmp_0);

	float* hardswish_109_tmp_0 = (float*)calloc(9600, sizeof(float));
	autox_conv2d(hardswish_92_tmp_0, (float*)((int8_t*)weights) + 585664, (float*)((int8_t*)weights) + 585760, hardswish_109_tmp_0, hardswish_92_tmp_0_dim, conv2d_70_w_0_dim, hardswish_109_tmp_0_dim, 96, 2, 1, 1, 10);
	free(hardswish_92_tmp_0);

	float* hardswish_110_tmp_0 = (float*)calloc(9600, sizeof(float));
	autox_conv2d(hardswish_109_tmp_0, (float*)((int8_t*)weights) + 588160, (float*)((int8_t*)weights) + 588256, hardswish_110_tmp_0, hardswish_109_tmp_0_dim, conv2d_71_w_0_dim, hardswish_110_tmp_0_dim, 1, 0, 1, 1, 10);
	free(hardswish_109_tmp_0);

	float* hardswish_111_tmp_0 = (float*)calloc(9600, sizeof(float));
	autox_conv2d(hardswish_110_tmp_0, (float*)((int8_t*)weights) + 597472, (float*)((int8_t*)weights) + 597568, hardswish_111_tmp_0, hardswish_110_tmp_0_dim, conv2d_72_w_0_dim, hardswish_111_tmp_0_dim, 96, 2, 1, 1, 10);
	free(hardswish_110_tmp_0);

	float* hardswish_112_tmp_0 = (float*)calloc(9600, sizeof(float));
	autox_conv2d(hardswish_111_tmp_0, (float*)((int8_t*)weights) + 599968, (float*)((int8_t*)weights) + 600064, hardswish_112_tmp_0, hardswish_111_tmp_0_dim, conv2d_73_w_0_dim, hardswish_112_tmp_0_dim, 1, 0, 1, 1, 10);
	free(hardswish_111_tmp_0);

	float* pool2d_4_tmp_0 = (float*)calloc(96, sizeof(float));
	autox_pool2d(hardswish_112_tmp_0, pool2d_4_tmp_0, hardswish_112_tmp_0_dim, pool2d_4_tmp_0_dim, 1, 1, 0, 1, 1);
	//free(hardswish_112_tmp_0);

	float* conv2d_151_tmp_1 = (float*)calloc(96, sizeof(float));
	autox_conv2d(pool2d_4_tmp_0, (float*)((int8_t*)weights) + 609280, (float*)((int8_t*)weights) + 609376, conv2d_151_tmp_1, pool2d_4_tmp_0_dim, conv2d_74_w_0_dim, conv2d_151_tmp_1_dim, 1, 0, 1, 1, 0);
	free(pool2d_4_tmp_0);

	autox_sigmoid(conv2d_151_tmp_1, conv2d_151_tmp_1_dim, 4);

	float* tmp_7 = (float*)calloc(9600, sizeof(float));
	autox_elementwise_mul(hardswish_112_tmp_0, conv2d_151_tmp_1, tmp_7, hardswish_112_tmp_0_dim, sigmoid_6_tmp_0_dim, tmp_7_dim, -1, 4, 4, 4);
	//free(hardswish_112_tmp_0);
	free(conv2d_151_tmp_1);

	float* hardswish_113_tmp_0 = (float*)calloc(9600, sizeof(float));
	autox_conv2d(tmp_7, (float*)((int8_t*)weights) + 618592, (float*)((int8_t*)weights) + 618688, hardswish_113_tmp_0, tmp_7_dim, conv2d_75_w_0_dim, hardswish_113_tmp_0_dim, 1, 0, 1, 1, 10);
	free(tmp_7);

	float* conv2d_153_tmp_1 = (float*)calloc(8000, sizeof(float));
	autox_conv2d(hardswish_113_tmp_0, (float*)((int8_t*)weights) + 627904, (float*)((int8_t*)weights) + 627984, conv2d_153_tmp_1, hardswish_113_tmp_0_dim, conv2d_92_w_0_dim, conv2d_153_tmp_1_dim, 1, 0, 1, 1, 0);
	//free(hardswish_113_tmp_0);

	float* conv2d_154_tmp_1 = (float*)calloc(3200, sizeof(float));
	autox_conv2d(hardswish_113_tmp_0, (float*)((int8_t*)weights) + 635664, (float*)((int8_t*)weights) + 635696, conv2d_154_tmp_1, hardswish_113_tmp_0_dim, conv2d_93_w_0_dim, conv2d_154_tmp_1_dim, 1, 0, 1, 1, 0);
	free(hardswish_113_tmp_0);

	float* hardswish_114_tmp_0 = (float*)calloc(100, sizeof(float));
	autox_conv2d(hardswish_112_tmp_0, (float*)((int8_t*)weights) + 638768, (float*)((int8_t*)weights) + 638769, hardswish_114_tmp_0, hardswish_112_tmp_0_dim, conv2d_94_w_0_dim, hardswish_114_tmp_0_dim, 1, 2, 1, 1, 10);
	free(hardswish_112_tmp_0);

	float* batch_norm_74_tmp_2 = (float*)calloc(100, sizeof(float));
	autox_conv2d(hardswish_114_tmp_0, (float*)((int8_t*)weights) + 641169, (float*)((int8_t*)weights) + 641170, batch_norm_74_tmp_2, hardswish_114_tmp_0_dim, conv2d_95_w_0_dim, batch_norm_74_tmp_2_dim, 1, 0, 1, 1, 0);
	free(hardswish_114_tmp_0);

	autox_sigmoid(batch_norm_74_tmp_2, batch_norm_74_tmp_2_dim, 4);

	autox_sigmoid(conv2d_153_tmp_1, conv2d_153_tmp_1_dim, 4);

	float* tmp_8 = (float*)calloc(8000, sizeof(float));
	autox_elementwise_mul(conv2d_153_tmp_1, batch_norm_74_tmp_2, tmp_8, sigmoid_8_tmp_0_dim, sigmoid_7_tmp_0_dim, tmp_8_dim, -1, 4, 4, 4);
	free(conv2d_153_tmp_1);
	free(batch_norm_74_tmp_2);

	autox_scale(tmp_8, tmp_8_dim, 4, 1e-09, 1, 1.0);

	autox_sqrt(tmp_8, tmp_9_dim, 4);

	float* transpose_2_tmp_0 = (float*)calloc(3200, sizeof(float));
	uint16_t axis_131[] = { 0, 2, 3, 1 };
	autox_transpose(conv2d_154_tmp_1, transpose_2_tmp_0, conv2d_154_tmp_1_dim, transpose_2_tmp_0_dim, axis_131, 4);
	free(conv2d_154_tmp_1);

	autox_softmax(transpose_2_tmp_0, reshape2_7_tmp_0_dim[0], reshape2_7_tmp_0_dim[1]);

	float* linear_2_tmp_0 = (float*)calloc(400, sizeof(float));
	autox_matmul(transpose_2_tmp_0, (float*)((int8_t*)weights) + 641171, linear_2_tmp_0, softmax_2_tmp_0_dim, eager_tmp_4_dim, linear_2_tmp_0_dim, 0, 0, 2, 1, 1);
	free(transpose_2_tmp_0);

	float* hardswish_115_tmp_0 = (float*)calloc(2400, sizeof(float));
	autox_conv2d(tmp_0, (float*)((int8_t*)weights) + 641179, (float*)((int8_t*)weights) + 641275, hardswish_115_tmp_0, tmp_0_dim, conv2d_76_w_0_dim, hardswish_115_tmp_0_dim, 96, 2, 1, 1, 10);
	free(tmp_0);

	float* hardswish_116_tmp_0 = (float*)calloc(2400, sizeof(float));
	autox_conv2d(hardswish_115_tmp_0, (float*)((int8_t*)weights) + 643675, (float*)((int8_t*)weights) + 643771, hardswish_116_tmp_0, hardswish_115_tmp_0_dim, conv2d_77_w_0_dim, hardswish_116_tmp_0_dim, 1, 0, 1, 1, 10);
	free(hardswish_115_tmp_0);

	float* hardswish_117_tmp_0 = (float*)calloc(2400, sizeof(float));
	autox_conv2d(hardswish_116_tmp_0, (float*)((int8_t*)weights) + 652987, (float*)((int8_t*)weights) + 653083, hardswish_117_tmp_0, hardswish_116_tmp_0_dim, conv2d_78_w_0_dim, hardswish_117_tmp_0_dim, 96, 2, 1, 1, 10);
	free(hardswish_116_tmp_0);

	float* hardswish_118_tmp_0 = (float*)calloc(2400, sizeof(float));
	autox_conv2d(hardswish_117_tmp_0, (float*)((int8_t*)weights) + 655483, (float*)((int8_t*)weights) + 655579, hardswish_118_tmp_0, hardswish_117_tmp_0_dim, conv2d_79_w_0_dim, hardswish_118_tmp_0_dim, 1, 0, 1, 1, 10);
	free(hardswish_117_tmp_0);

	float* pool2d_5_tmp_0 = (float*)calloc(96, sizeof(float));
	autox_pool2d(hardswish_118_tmp_0, pool2d_5_tmp_0, hardswish_118_tmp_0_dim, pool2d_5_tmp_0_dim, 1, 1, 0, 1, 1);
	//free(hardswish_118_tmp_0);

	float* conv2d_159_tmp_1 = (float*)calloc(96, sizeof(float));
	autox_conv2d(pool2d_5_tmp_0, (float*)((int8_t*)weights) + 664795, (float*)((int8_t*)weights) + 664891, conv2d_159_tmp_1, pool2d_5_tmp_0_dim, conv2d_80_w_0_dim, conv2d_159_tmp_1_dim, 1, 0, 1, 1, 0);
	free(pool2d_5_tmp_0);

	autox_sigmoid(conv2d_159_tmp_1, conv2d_159_tmp_1_dim, 4);

	float* tmp_10 = (float*)calloc(2400, sizeof(float));
	autox_elementwise_mul(hardswish_118_tmp_0, conv2d_159_tmp_1, tmp_10, hardswish_118_tmp_0_dim, sigmoid_9_tmp_0_dim, tmp_10_dim, -1, 4, 4, 4);
	//free(hardswish_118_tmp_0);
	free(conv2d_159_tmp_1);

	float* hardswish_119_tmp_0 = (float*)calloc(2400, sizeof(float));
	autox_conv2d(tmp_10, (float*)((int8_t*)weights) + 674107, (float*)((int8_t*)weights) + 674203, hardswish_119_tmp_0, tmp_10_dim, conv2d_81_w_0_dim, hardswish_119_tmp_0_dim, 1, 0, 1, 1, 10);
	free(tmp_10);

	float* conv2d_161_tmp_1 = (float*)calloc(2000, sizeof(float));
	autox_conv2d(hardswish_119_tmp_0, (float*)((int8_t*)weights) + 683419, (float*)((int8_t*)weights) + 683499, conv2d_161_tmp_1, hardswish_119_tmp_0_dim, conv2d_96_w_0_dim, conv2d_161_tmp_1_dim, 1, 0, 1, 1, 0);
	//free(hardswish_119_tmp_0);

	float* conv2d_162_tmp_1 = (float*)calloc(800, sizeof(float));
	autox_conv2d(hardswish_119_tmp_0, (float*)((int8_t*)weights) + 691179, (float*)((int8_t*)weights) + 691211, conv2d_162_tmp_1, hardswish_119_tmp_0_dim, conv2d_97_w_0_dim, conv2d_162_tmp_1_dim, 1, 0, 1, 1, 0);
	free(hardswish_119_tmp_0);

	float* hardswish_120_tmp_0 = (float*)calloc(25, sizeof(float));
	autox_conv2d(hardswish_118_tmp_0, (float*)((int8_t*)weights) + 694283, (float*)((int8_t*)weights) + 694284, hardswish_120_tmp_0, hardswish_118_tmp_0_dim, conv2d_98_w_0_dim, hardswish_120_tmp_0_dim, 1, 2, 1, 1, 10);
	free(hardswish_118_tmp_0);

	float* batch_norm_81_tmp_2 = (float*)calloc(25, sizeof(float));
	autox_conv2d(hardswish_120_tmp_0, (float*)((int8_t*)weights) + 696684, (float*)((int8_t*)weights) + 696685, batch_norm_81_tmp_2, hardswish_120_tmp_0_dim, conv2d_99_w_0_dim, batch_norm_81_tmp_2_dim, 1, 0, 1, 1, 0);
	free(hardswish_120_tmp_0);

	autox_sigmoid(batch_norm_81_tmp_2, batch_norm_81_tmp_2_dim, 4);

	autox_sigmoid(conv2d_161_tmp_1, conv2d_161_tmp_1_dim, 4);

	float* tmp_11 = (float*)calloc(2000, sizeof(float));
	autox_elementwise_mul(conv2d_161_tmp_1, batch_norm_81_tmp_2, tmp_11, sigmoid_11_tmp_0_dim, sigmoid_10_tmp_0_dim, tmp_11_dim, -1, 4, 4, 4);
	free(conv2d_161_tmp_1);
	free(batch_norm_81_tmp_2);

	autox_scale(tmp_11, tmp_11_dim, 4, 1e-09, 1, 1.0);

	autox_sqrt(tmp_11, tmp_12_dim, 4);

	float* transpose_3_tmp_0 = (float*)calloc(800, sizeof(float));
	uint16_t axis_152[] = { 0, 2, 3, 1 };
	autox_transpose(conv2d_162_tmp_1, transpose_3_tmp_0, conv2d_162_tmp_1_dim, transpose_3_tmp_0_dim, axis_152, 4);
	free(conv2d_162_tmp_1);

	autox_softmax(transpose_3_tmp_0, reshape2_10_tmp_0_dim[0], reshape2_10_tmp_0_dim[1]);

	float* linear_3_tmp_0 = (float*)calloc(100, sizeof(float));
	autox_matmul(transpose_3_tmp_0, (float*)((int8_t*)weights) + 696686, linear_3_tmp_0, softmax_3_tmp_0_dim, eager_tmp_4_dim, linear_3_tmp_0_dim, 0, 0, 2, 1, 1);
	free(transpose_3_tmp_0);

	float* p_155[] = {tmp_2, tmp_5, tmp_8, tmp_11 };
	uint16_t* p_155_dim[] = { reshape2_0_tmp_0_dim, reshape2_3_tmp_0_dim, reshape2_6_tmp_0_dim, reshape2_9_tmp_0_dim };
	autox_concat(p_155, dets, p_155_dim, concat_4_tmp_0_dim, -1, 4, 3);
	free(tmp_2);
	free(tmp_5);
	free(tmp_8);
	free(tmp_11);

	float* p_156[] = { linear_0_tmp_0, linear_1_tmp_0, linear_2_tmp_0,linear_3_tmp_0 };
	uint16_t* p_156_dim[] = { reshape2_2_tmp_0_dim, reshape2_5_tmp_0_dim, reshape2_8_tmp_0_dim, reshape2_11_tmp_0_dim };
	float* concat_5_tmp_0 = (float*)calloc(8500, sizeof(float));
	autox_concat(p_156, concat_5_tmp_0, p_156_dim, concat_5_tmp_0_dim, 1, 4, 3);
	free(linear_0_tmp_0);
	free(linear_1_tmp_0);
	free(linear_2_tmp_0);
	free(linear_3_tmp_0);

	float* split_0_tmp_0 = (float*)calloc(4250, sizeof(float));
	float* split_0_tmp_1 = (float*)calloc(4250, sizeof(float));
	float* p_157[] = { split_0_tmp_0, split_0_tmp_1, };
	uint16_t* p_157_dim[] = { split_0_tmp_0_dim, split_0_tmp_1_dim, };
	autox_split(concat_5_tmp_0, p_157, concat_5_tmp_0_dim, p_157_dim, 2, 3, 2, 3);
	free(concat_5_tmp_0);

	autox_scale(split_0_tmp_0, split_0_tmp_0_dim, 3, 0, 1, -1.0);

	float* tmp_14 = (float*)calloc(4250, sizeof(float));
	autox_elementwise_add(split_0_tmp_0, (float*)((int8_t*)weights) + 696694, tmp_14, tmp_13_dim, eager_tmp_0_dim, tmp_14_dim, -1, 3, 2, 3);
	free(split_0_tmp_0);

	float* tmp_15 = (float*)calloc(4250, sizeof(float));
	autox_elementwise_add(split_0_tmp_1, (float*)((int8_t*)weights) + 700944, tmp_15, split_0_tmp_1_dim, eager_tmp_0_dim, tmp_15_dim, -1, 3, 2, 3);
	free(split_0_tmp_1);

	float* p_161[] = { tmp_14, tmp_15, };
	uint16_t* p_161_dim[] = { tmp_14_dim, tmp_15_dim, };
	float* concat_6_tmp_0 = (float*)calloc(8500, sizeof(float));
	autox_concat(p_161, concat_6_tmp_0, p_161_dim, concat_6_tmp_0_dim, -1, 2, 3);
	free(tmp_14);
	free(tmp_15);

	autox_elementwise_mul(concat_6_tmp_0, (float*)((int8_t*)weights) + 705194, boxes, concat_6_tmp_0_dim, eager_tmp_1_dim, tmp_16_dim, -1, 3, 2, 3);
	free(concat_6_tmp_0);

}
