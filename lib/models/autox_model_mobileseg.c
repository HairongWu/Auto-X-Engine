#include "../include/autox_models.h"

void mobileseg_tiny(const float* x, void* weights, float* Out)
{
	uint16_t x_dim[] = { 1, 3, 512, 512 };
	uint16_t batch_norm_0_b_0_dim[] = { 16 };
	uint16_t batch_norm_31_tmp_2_dim[] = { 1, 128, 16, 16 };
	uint16_t conv2d_0_w_0_dim[] = { 16, 3, 3, 3 };
	uint16_t batch_norm_1_b_0_dim[] = { 16 };
	uint16_t batch_norm_11_tmp_3_dim[] = { 1, 96, 64, 64 };
	uint16_t conv2d_1_w_0_dim[] = { 16, 16, 1, 1 };
	uint16_t batch_norm_2_b_0_dim[] = { 16 };
	uint16_t batch_norm_34_tmp_2_dim[] = { 1, 128, 16, 1 };
	uint16_t conv2d_2_w_0_dim[] = { 16, 1, 3, 3 };
	uint16_t batch_norm_14_tmp_3_dim[] = { 1, 96, 64, 64 };
	uint16_t conv2d_3_b_0_dim[] = { 4 };
	uint16_t conv2d_3_w_0_dim[] = { 4, 16, 1, 1 };
	uint16_t conv2d_4_b_0_dim[] = { 16 };
	uint16_t conv2d_4_w_0_dim[] = { 16, 4, 1, 1 };
	uint16_t batch_norm_30_tmp_2_dim[] = { 1, 64, 16, 16 };
	uint16_t batch_norm_3_b_0_dim[] = { 16 };
	uint16_t conv2d_5_w_0_dim[] = { 16, 16, 1, 1 };
	uint16_t batch_norm_4_b_0_dim[] = { 64 };
	uint16_t conv2d_6_w_0_dim[] = { 64, 16, 1, 1 };
	uint16_t batch_norm_5_b_0_dim[] = { 64 };
	uint16_t conv2d_7_w_0_dim[] = { 64, 1, 3, 3 };
	uint16_t batch_norm_7_b_0_dim[] = { 48 };
	uint16_t conv2d_8_w_0_dim[] = { 32, 64, 1, 1 };
	uint16_t batch_norm_8_b_0_dim[] = { 48 };
	uint16_t conv2d_10_w_0_dim[] = { 48, 1, 3, 3 };
	uint16_t batch_norm_10_b_0_dim[] = { 96 };
	uint16_t conv2d_11_w_0_dim[] = { 24, 48, 1, 1 };
	uint16_t batch_norm_11_b_0_dim[] = { 96 };
	uint16_t conv2d_13_w_0_dim[] = { 96, 1, 5, 5 };
	uint16_t conv2d_14_b_0_dim[] = { 24 };
	uint16_t conv2d_14_w_0_dim[] = { 24, 96, 1, 1 };
	uint16_t conv2d_15_b_0_dim[] = { 96 };
	uint16_t conv2d_15_w_0_dim[] = { 96, 24, 1, 1 };
	uint16_t batch_norm_12_b_0_dim[] = { 32 };
	uint16_t conv2d_16_w_0_dim[] = { 32, 96, 1, 1 };
	uint16_t batch_norm_13_b_0_dim[] = { 96 };
	uint16_t conv2d_17_w_0_dim[] = { 96, 32, 1, 1 };
	uint16_t batch_norm_14_b_0_dim[] = { 96 };
	uint16_t conv2d_18_w_0_dim[] = { 96, 1, 5, 5 };
	uint16_t conv2d_19_b_0_dim[] = { 24 };
	uint16_t conv2d_19_w_0_dim[] = { 24, 96, 1, 1 };
	uint16_t conv2d_20_b_0_dim[] = { 96 };
	uint16_t conv2d_20_w_0_dim[] = { 96, 24, 1, 1 };
	uint16_t batch_norm_15_b_0_dim[] = { 32 };
	uint16_t conv2d_21_w_0_dim[] = { 32, 96, 1, 1 };
	uint16_t elementwise_add_1_dim[] = { 1, 32, 64, 64 };
	uint16_t batch_norm_16_b_0_dim[] = { 160 };
	uint16_t conv2d_22_w_0_dim[] = { 160, 32, 1, 1 };
	uint16_t batch_norm_17_b_0_dim[] = { 160 };
	uint16_t conv2d_23_w_0_dim[] = { 160, 1, 5, 5 };
	uint16_t conv2d_24_b_0_dim[] = { 40 };
	uint16_t conv2d_24_w_0_dim[] = { 40, 160, 1, 1 };
	uint16_t conv2d_25_b_0_dim[] = { 160 };
	uint16_t conv2d_25_w_0_dim[] = { 160, 40, 1, 1 };
	uint16_t batch_norm_18_b_0_dim[] = { 64 };
	uint16_t conv2d_26_w_0_dim[] = { 64, 160, 1, 1 };
	uint16_t batch_norm_19_b_0_dim[] = { 160 };
	uint16_t conv2d_27_w_0_dim[] = { 160, 64, 1, 1 };
	uint16_t batch_norm_20_b_0_dim[] = { 160 };
	uint16_t conv2d_28_w_0_dim[] = { 160, 1, 5, 5 };
	uint16_t conv2d_29_b_0_dim[] = { 40 };
	uint16_t conv2d_29_w_0_dim[] = { 40, 160, 1, 1 };
	uint16_t conv2d_30_b_0_dim[] = { 160 };
	uint16_t conv2d_30_w_0_dim[] = { 160, 40, 1, 1 };
	uint16_t batch_norm_21_b_0_dim[] = { 64 };
	uint16_t conv2d_31_w_0_dim[] = { 64, 160, 1, 1 };
	uint16_t elementwise_add_2_dim[] = { 1, 64, 32, 32 };
	uint16_t batch_norm2d_3_b_0_dim[] = { 64 };
	uint16_t conv2d_45_w_0_dim[] = { 64, 1, 3, 3 };
	uint16_t batch_norm2d_0_b_0_dim[] = { 64 };
	uint16_t conv2d_42_w_0_dim[] = { 64, 64, 1, 1 };
	uint16_t batch_norm2d_1_b_0_dim[] = { 64 };
	uint16_t conv2d_43_w_0_dim[] = { 64, 64, 1, 1 };
	uint16_t batch_norm2d_2_b_0_dim[] = { 128 };
	uint16_t conv2d_44_w_0_dim[] = { 128, 64, 1, 1 };
	uint16_t batch_norm2d_7_b_0_dim[] = { 256 };
	uint16_t conv2d_49_w_0_dim[] = { 256, 1, 3, 3 };
	uint16_t batch_norm2d_8_b_0_dim[] = { 64 };
	uint16_t conv2d_50_w_0_dim[] = { 64, 256, 1, 1 };
	uint16_t mean_0_tmp_0_dim[] = { 1, 64, 16 };
	uint16_t create_parameter_0_w_0_dim[] = { 1, 64, 16 };
	uint16_t tmp_0_dim[] = { 1, 64, 16 };
	uint16_t reshape2_0_tmp_0_dim[] = { 1, 4, 16, 16 };
	uint16_t tmp_13_dim[] = { 1, 4, 16, 16 };
	uint16_t create_parameter_1_w_0_dim[] = { 1, 64, 16 };
	uint16_t tmp_1_dim[] = { 1, 64, 16 };
	uint16_t reshape2_1_tmp_0_dim[] = { 1, 4, 16, 16 };
	uint16_t mean_2_tmp_0_dim[] = { 1, 128, 16 };
	uint16_t reshape2_2_tmp_0_dim[] = { 1, 4, 32, 16 };
	uint16_t tmp_24_dim[] = { 1, 4, 8, 8 };
	uint16_t transpose_2_tmp_0_dim[] = { 1, 4, 32, 16 };
	uint16_t reshape2_3_tmp_0_dim[] = { 1, 128, 16, 1 };
	uint16_t batch_norm2d_5_b_0_dim[] = { 128 };
	uint16_t conv2d_47_w_0_dim[] = { 128, 128, 1, 1 };
	uint16_t create_parameter_2_w_0_dim[] = { 1, 64, 16 };
	uint16_t tmp_3_dim[] = { 1, 64, 16 };
	uint16_t reshape2_4_tmp_0_dim[] = { 1, 4, 16, 16 };
	uint16_t create_parameter_3_w_0_dim[] = { 1, 64, 16 };
	uint16_t tmp_4_dim[] = { 1, 64, 16 };
	uint16_t reshape2_5_tmp_0_dim[] = { 1, 4, 16, 16 };
	uint16_t mean_5_tmp_0_dim[] = { 1, 128, 16 };
	uint16_t reshape2_6_tmp_0_dim[] = { 1, 4, 32, 16 };
	uint16_t transpose_5_tmp_0_dim[] = { 1, 4, 32, 16 };
	uint16_t reshape2_7_tmp_0_dim[] = { 1, 128, 1, 16 };
	uint16_t batch_norm2d_6_b_0_dim[] = { 128 };
	uint16_t conv2d_48_w_0_dim[] = { 128, 128, 1, 1 };
	uint16_t batch_norm2d_4_b_0_dim[] = { 64 };
	uint16_t conv2d_46_w_0_dim[] = { 64, 128, 1, 1 };
	uint16_t batch_norm2d_9_b_0_dim[] = { 128 };
	uint16_t conv2d_51_w_0_dim[] = { 128, 64, 1, 1 };
	uint16_t conv2d_52_b_0_dim[] = { 128 };
	uint16_t conv2d_52_w_0_dim[] = { 128, 1, 3, 3 };
	uint16_t batch_norm2d_10_b_0_dim[] = { 64 };
	uint16_t conv2d_53_w_0_dim[] = { 64, 128, 1, 1 };
	uint16_t batch_norm2d_14_b_0_dim[] = { 64 };
	uint16_t conv2d_57_w_0_dim[] = { 64, 1, 3, 3 };
	uint16_t batch_norm2d_11_b_0_dim[] = { 64 };
	uint16_t conv2d_54_w_0_dim[] = { 64, 64, 1, 1 };
	uint16_t batch_norm2d_12_b_0_dim[] = { 64 };
	uint16_t conv2d_55_w_0_dim[] = { 64, 64, 1, 1 };
	uint16_t batch_norm2d_13_b_0_dim[] = { 128 };
	uint16_t conv2d_56_w_0_dim[] = { 128, 64, 1, 1 };
	uint16_t batch_norm2d_18_b_0_dim[] = { 256 };
	uint16_t conv2d_61_w_0_dim[] = { 256, 1, 3, 3 };
	uint16_t batch_norm2d_19_b_0_dim[] = { 64 };
	uint16_t conv2d_62_w_0_dim[] = { 64, 256, 1, 1 };
	uint16_t create_parameter_4_w_0_dim[] = { 1, 64, 16 };
	uint16_t tmp_11_dim[] = { 1, 64, 16 };
	uint16_t reshape2_8_tmp_0_dim[] = { 1, 4, 16, 16 };
	uint16_t create_parameter_5_w_0_dim[] = { 1, 64, 16 };
	uint16_t tmp_12_dim[] = { 1, 64, 16 };
	uint16_t reshape2_9_tmp_0_dim[] = { 1, 4, 16, 16 };
	uint16_t mean_8_tmp_0_dim[] = { 1, 128, 16 };
	uint16_t reshape2_10_tmp_0_dim[] = { 1, 4, 32, 16 };
	uint16_t transpose_8_tmp_0_dim[] = { 1, 4, 32, 16 };
	uint16_t reshape2_11_tmp_0_dim[] = { 1, 128, 16, 1 };
	uint16_t batch_norm2d_16_b_0_dim[] = { 128 };
	uint16_t conv2d_59_w_0_dim[] = { 128, 128, 1, 1 };
	uint16_t create_parameter_6_w_0_dim[] = { 1, 64, 16 };
	uint16_t tmp_14_dim[] = { 1, 64, 16 };
	uint16_t reshape2_12_tmp_0_dim[] = { 1, 4, 16, 16 };
	uint16_t create_parameter_7_w_0_dim[] = { 1, 64, 16 };
	uint16_t tmp_15_dim[] = { 1, 64, 16 };
	uint16_t reshape2_13_tmp_0_dim[] = { 1, 4, 16, 16 };
	uint16_t mean_11_tmp_0_dim[] = { 1, 128, 16 };
	uint16_t reshape2_14_tmp_0_dim[] = { 1, 4, 32, 16 };
	uint16_t transpose_11_tmp_0_dim[] = { 1, 4, 32, 16 };
	uint16_t reshape2_15_tmp_0_dim[] = { 1, 128, 1, 16 };
	uint16_t batch_norm2d_17_b_0_dim[] = { 128 };
	uint16_t conv2d_60_w_0_dim[] = { 128, 128, 1, 1 };
	uint16_t batch_norm2d_15_b_0_dim[] = { 64 };
	uint16_t conv2d_58_w_0_dim[] = { 64, 128, 1, 1 };
	uint16_t batch_norm2d_20_b_0_dim[] = { 128 };
	uint16_t conv2d_63_w_0_dim[] = { 128, 64, 1, 1 };
	uint16_t conv2d_64_b_0_dim[] = { 128 };
	uint16_t conv2d_64_w_0_dim[] = { 128, 1, 3, 3 };
	uint16_t batch_norm2d_21_b_0_dim[] = { 64 };
	uint16_t conv2d_65_w_0_dim[] = { 64, 128, 1, 1 };
	uint16_t batch_norm_22_b_0_dim[] = { 384 };
	uint16_t conv2d_32_w_0_dim[] = { 384, 64, 1, 1 };
	uint16_t batch_norm_23_b_0_dim[] = { 384 };
	uint16_t conv2d_33_w_0_dim[] = { 384, 1, 3, 3 };
	uint16_t conv2d_34_b_0_dim[] = { 96 };
	uint16_t conv2d_34_w_0_dim[] = { 96, 384, 1, 1 };
	uint16_t conv2d_35_b_0_dim[] = { 384 };
	uint16_t conv2d_35_w_0_dim[] = { 384, 96, 1, 1 };
	uint16_t batch_norm_24_b_0_dim[] = { 128 };
	uint16_t conv2d_36_w_0_dim[] = { 128, 384, 1, 1 };
	uint16_t batch_norm_25_b_0_dim[] = { 384 };
	uint16_t conv2d_37_w_0_dim[] = { 384, 128, 1, 1 };
	uint16_t batch_norm_26_b_0_dim[] = { 384 };
	uint16_t conv2d_38_w_0_dim[] = { 384, 1, 3, 3 };
	uint16_t conv2d_39_b_0_dim[] = { 96 };
	uint16_t conv2d_39_w_0_dim[] = { 96, 384, 1, 1 };
	uint16_t conv2d_40_b_0_dim[] = { 384 };
	uint16_t conv2d_40_w_0_dim[] = { 384, 96, 1, 1 };
	uint16_t batch_norm_27_b_0_dim[] = { 128 };
	uint16_t conv2d_41_w_0_dim[] = { 128, 384, 1, 1 };
	uint16_t batch_norm2d_25_b_0_dim[] = { 128 };
	uint16_t conv2d_69_w_0_dim[] = { 128, 1, 3, 3 };
	uint16_t batch_norm2d_22_b_0_dim[] = { 96 };
	uint16_t conv2d_66_w_0_dim[] = { 96, 128, 1, 1 };
	uint16_t batch_norm2d_23_b_0_dim[] = { 96 };
	uint16_t conv2d_67_w_0_dim[] = { 96, 128, 1, 1 };
	uint16_t batch_norm2d_24_b_0_dim[] = { 192 };
	uint16_t conv2d_68_w_0_dim[] = { 192, 128, 1, 1 };
	uint16_t batch_norm2d_29_b_0_dim[] = { 384 };
	uint16_t conv2d_73_w_0_dim[] = { 384, 1, 3, 3 };
	uint16_t batch_norm2d_30_b_0_dim[] = { 128 };
	uint16_t conv2d_74_w_0_dim[] = { 128, 384, 1, 1 };
	uint16_t create_parameter_8_w_0_dim[] = { 1, 96, 16 };
	uint16_t tmp_22_dim[] = { 1, 96, 8 };
	uint16_t reshape2_16_tmp_0_dim[] = { 1, 4, 24, 8 };
	uint16_t create_parameter_9_w_0_dim[] = { 1, 96, 16 };
	uint16_t tmp_23_dim[] = { 1, 96, 8 };
	uint16_t reshape2_17_tmp_0_dim[] = { 1, 4, 24, 8 };
	uint16_t mean_14_tmp_0_dim[] = { 1, 192, 8 };
	uint16_t reshape2_18_tmp_0_dim[] = { 1, 4, 48, 8 };
	uint16_t transpose_13_tmp_0_dim[] = { 1, 4, 8, 48 };
	uint16_t transpose_14_tmp_0_dim[] = { 1, 4, 48, 8 };
	uint16_t reshape2_19_tmp_0_dim[] = { 1, 192, 8, 1 };
	uint16_t batch_norm2d_27_b_0_dim[] = { 192 };
	uint16_t conv2d_71_w_0_dim[] = { 192, 192, 1, 1 };
	uint16_t create_parameter_10_w_0_dim[] = { 1, 96, 16 };
	uint16_t tmp_25_dim[] = { 1, 96, 8 };
	uint16_t reshape2_20_tmp_0_dim[] = { 1, 4, 24, 8 };
	uint16_t create_parameter_11_w_0_dim[] = { 1, 96, 16 };
	uint16_t tmp_26_dim[] = { 1, 96, 8 };
	uint16_t reshape2_21_tmp_0_dim[] = { 1, 4, 24, 8 };
	uint16_t mean_17_tmp_0_dim[] = { 1, 192, 8 };
	uint16_t reshape2_22_tmp_0_dim[] = { 1, 4, 48, 8 };
	uint16_t transpose_17_tmp_0_dim[] = { 1, 4, 48, 8 };
	uint16_t reshape2_23_tmp_0_dim[] = { 1, 192, 1, 8 };
	uint16_t batch_norm2d_28_b_0_dim[] = { 192 };
	uint16_t conv2d_72_w_0_dim[] = { 192, 192, 1, 1 };
	uint16_t batch_norm2d_26_b_0_dim[] = { 128 };
	uint16_t conv2d_70_w_0_dim[] = { 128, 192, 1, 1 };
	uint16_t batch_norm2d_31_b_0_dim[] = { 512 };
	uint16_t conv2d_75_w_0_dim[] = { 512, 128, 1, 1 };
	uint16_t conv2d_76_b_0_dim[] = { 512 };
	uint16_t conv2d_76_w_0_dim[] = { 512, 1, 3, 3 };
	uint16_t batch_norm2d_32_b_0_dim[] = { 128 };
	uint16_t conv2d_77_w_0_dim[] = { 128, 512, 1, 1 };
	uint16_t batch_norm2d_36_b_0_dim[] = { 128 };
	uint16_t conv2d_81_w_0_dim[] = { 128, 1, 3, 3 };
	uint16_t batch_norm2d_33_b_0_dim[] = { 96 };
	uint16_t conv2d_78_w_0_dim[] = { 96, 128, 1, 1 };
	uint16_t batch_norm2d_34_b_0_dim[] = { 96 };
	uint16_t conv2d_79_w_0_dim[] = { 96, 128, 1, 1 };
	uint16_t batch_norm2d_35_b_0_dim[] = { 192 };
	uint16_t conv2d_80_w_0_dim[] = { 192, 128, 1, 1 };
	uint16_t batch_norm2d_40_b_0_dim[] = { 384 };
	uint16_t conv2d_85_w_0_dim[] = { 384, 1, 3, 3 };
	uint16_t batch_norm2d_41_b_0_dim[] = { 128 };
	uint16_t conv2d_86_w_0_dim[] = { 128, 384, 1, 1 };
	uint16_t create_parameter_12_w_0_dim[] = { 1, 96, 16 };
	uint16_t tmp_33_dim[] = { 1, 96, 8 };
	uint16_t reshape2_24_tmp_0_dim[] = { 1, 4, 24, 8 };
	uint16_t create_parameter_13_w_0_dim[] = { 1, 96, 16 };
	uint16_t tmp_34_dim[] = { 1, 96, 8 };
	uint16_t reshape2_25_tmp_0_dim[] = { 1, 4, 24, 8 };
	uint16_t mean_20_tmp_0_dim[] = { 1, 192, 8 };
	uint16_t reshape2_26_tmp_0_dim[] = { 1, 4, 48, 8 };
	uint16_t transpose_20_tmp_0_dim[] = { 1, 4, 48, 8 };
	uint16_t reshape2_27_tmp_0_dim[] = { 1, 192, 8, 1 };
	uint16_t batch_norm2d_38_b_0_dim[] = { 192 };
	uint16_t conv2d_83_w_0_dim[] = { 192, 192, 1, 1 };
	uint16_t create_parameter_14_w_0_dim[] = { 1, 96, 16 };
	uint16_t tmp_36_dim[] = { 1, 96, 8 };
	uint16_t reshape2_28_tmp_0_dim[] = { 1, 4, 24, 8 };
	uint16_t create_parameter_15_w_0_dim[] = { 1, 96, 16 };
	uint16_t tmp_37_dim[] = { 1, 96, 8 };
	uint16_t reshape2_29_tmp_0_dim[] = { 1, 4, 24, 8 };
	uint16_t mean_23_tmp_0_dim[] = { 1, 192, 8 };
	uint16_t reshape2_30_tmp_0_dim[] = { 1, 4, 48, 8 };
	uint16_t transpose_23_tmp_0_dim[] = { 1, 4, 48, 8 };
	uint16_t reshape2_31_tmp_0_dim[] = { 1, 192, 1, 8 };
	uint16_t batch_norm2d_39_b_0_dim[] = { 192 };
	uint16_t conv2d_84_w_0_dim[] = { 192, 192, 1, 1 };
	uint16_t batch_norm2d_37_b_0_dim[] = { 128 };
	uint16_t conv2d_82_w_0_dim[] = { 128, 192, 1, 1 };
	uint16_t batch_norm2d_42_b_0_dim[] = { 512 };
	uint16_t conv2d_87_w_0_dim[] = { 512, 128, 1, 1 };
	uint16_t conv2d_88_b_0_dim[] = { 512 };
	uint16_t conv2d_88_w_0_dim[] = { 512, 1, 3, 3 };
	uint16_t batch_norm2d_43_b_0_dim[] = { 128 };
	uint16_t conv2d_89_w_0_dim[] = { 128, 512, 1, 1 };
	uint16_t batch_norm2d_45_b_0_dim[] = { 256 };
	uint16_t conv2d_91_w_0_dim[] = { 256, 32, 1, 1 };
	uint16_t batch_norm2d_44_b_0_dim[] = { 256 };
	uint16_t conv2d_90_w_0_dim[] = { 256, 32, 1, 1 };
	uint16_t batch_norm2d_47_b_0_dim[] = { 256 };
	uint16_t conv2d_93_w_0_dim[] = { 256, 64, 1, 1 };
	uint16_t batch_norm2d_46_b_0_dim[] = { 256 };
	uint16_t conv2d_92_w_0_dim[] = { 256, 64, 1, 1 };
	uint16_t batch_norm2d_49_b_0_dim[] = { 256 };
	uint16_t conv2d_95_w_0_dim[] = { 256, 128, 1, 1 };
	uint16_t batch_norm2d_48_b_0_dim[] = { 256 };
	uint16_t conv2d_94_w_0_dim[] = { 256, 128, 1, 1 };
	uint16_t batch_norm2d_50_b_0_dim[] = { 256 };
	uint16_t conv2d_96_w_0_dim[] = { 256, 1, 1, 1 };
	uint16_t conv2d_97_b_0_dim[] = { 150 };
	uint16_t conv2d_97_w_0_dim[] = { 150, 256, 1, 1 };
	uint16_t argmax_0_tmp_0_dim[] = { 1, 512, 512 };

	float *batch_norm_31_tmp_2 = (float*)calloc(32768, sizeof(float));
	autox_conv2d(x, (float*)((int8_t*)weights) + 0, (float*)((int8_t*)weights) + 16, batch_norm_31_tmp_2, x_dim, conv2d_0_w_0_dim, batch_norm_31_tmp_2_dim, 1, 1, 2, 1, 10);
	free(x);

	float *batch_norm_11_tmp_3 = (float*)calloc(393216, sizeof(float));
	autox_conv2d(batch_norm_31_tmp_2, (float*)((int8_t*)weights) + 448, (float*)((int8_t*)weights) + 464, batch_norm_11_tmp_3, batch_norm_31_tmp_2_dim, conv2d_1_w_0_dim, batch_norm_11_tmp_3_dim, 1, 0, 1, 1, 1);
	free(batch_norm_31_tmp_2);

	float *batch_norm_34_tmp_2 = (float*)calloc(2048, sizeof(float));
	autox_conv2d(batch_norm_11_tmp_3, (float*)((int8_t*)weights) + 720, (float*)((int8_t*)weights) + 736, batch_norm_34_tmp_2, batch_norm_11_tmp_3_dim, conv2d_2_w_0_dim, batch_norm_34_tmp_2_dim, 16, 1, 1, 1, 1);
	free(batch_norm_11_tmp_3);

	batch_norm_11_tmp_3 = (float*)calloc(393216, sizeof(float));
	autox_pool2d(batch_norm_34_tmp_2, batch_norm_11_tmp_3, batch_norm_34_tmp_2_dim, batch_norm_11_tmp_3_dim, 1, 1, 0, 0, 1);
	free(batch_norm_34_tmp_2);

	float *batch_norm_14_tmp_3 = (float*)calloc(393216, sizeof(float));
	autox_conv2d(batch_norm_11_tmp_3, (float*)((int8_t*)weights) + 880, (float*)((int8_t*)weights) + 884, batch_norm_14_tmp_3, batch_norm_11_tmp_3_dim, conv2d_3_w_0_dim, batch_norm_14_tmp_3_dim, 1, 0, 1, 1, 1);
	free(batch_norm_11_tmp_3);

	batch_norm_11_tmp_3 = (float*)calloc(393216, sizeof(float));
	autox_conv2d(batch_norm_14_tmp_3, (float*)((int8_t*)weights) + 948, (float*)((int8_t*)weights) + 964, batch_norm_11_tmp_3, batch_norm_14_tmp_3_dim, conv2d_4_w_0_dim, batch_norm_11_tmp_3_dim, 1, 0, 1, 1, 0);
	free(batch_norm_14_tmp_3);

	autox_hard_sigmoid(batch_norm_11_tmp_3, batch_norm_11_tmp_3_dim, 4, 0.5, 0.2);

	batch_norm_14_tmp_3 = (float*)calloc(393216, sizeof(float));
	autox_elementwise_mul(batch_norm_34_tmp_2, batch_norm_11_tmp_3, batch_norm_14_tmp_3, batch_norm_34_tmp_2_dim, batch_norm_30_tmp_2_dim, batch_norm_14_tmp_3_dim, -1, 4, 4, 4);
	free(batch_norm_34_tmp_2);
	free(batch_norm_11_tmp_3);

	batch_norm_11_tmp_3 = (float*)calloc(393216, sizeof(float));
	autox_conv2d(batch_norm_14_tmp_3, (float*)((int8_t*)weights) + 1028, (float*)((int8_t*)weights) + 1044, batch_norm_11_tmp_3, batch_norm_14_tmp_3_dim, conv2d_5_w_0_dim, batch_norm_11_tmp_3_dim, 1, 0, 1, 1, 0);
	free(batch_norm_14_tmp_3);

	batch_norm_14_tmp_3 = (float*)calloc(393216, sizeof(float));
	autox_elementwise_add(batch_norm_31_tmp_2, batch_norm_11_tmp_3, batch_norm_14_tmp_3, batch_norm_31_tmp_2_dim, batch_norm_11_tmp_3_dim, batch_norm_14_tmp_3_dim, -1, 4, 4, 4);
	free(batch_norm_31_tmp_2);
	free(batch_norm_11_tmp_3);

	batch_norm_11_tmp_3 = (float*)calloc(393216, sizeof(float));
	autox_conv2d(batch_norm_14_tmp_3, (float*)((int8_t*)weights) + 1300, (float*)((int8_t*)weights) + 1364, batch_norm_11_tmp_3, batch_norm_14_tmp_3_dim, conv2d_6_w_0_dim, batch_norm_11_tmp_3_dim, 1, 0, 1, 1, 1);
	free(batch_norm_14_tmp_3);

	batch_norm_14_tmp_3 = (float*)calloc(393216, sizeof(float));
	autox_conv2d(batch_norm_11_tmp_3, (float*)((int8_t*)weights) + 2388, (float*)((int8_t*)weights) + 2452, batch_norm_14_tmp_3, batch_norm_11_tmp_3_dim, conv2d_7_w_0_dim, batch_norm_14_tmp_3_dim, 64, 1, 2, 1, 1);
	free(batch_norm_11_tmp_3);

	batch_norm_11_tmp_3 = (float*)calloc(393216, sizeof(float));
	autox_conv2d(batch_norm_14_tmp_3, (float*)((int8_t*)weights) + 3028, (float*)((int8_t*)weights) + 3076, batch_norm_11_tmp_3, batch_norm_14_tmp_3_dim, conv2d_8_w_0_dim, batch_norm_11_tmp_3_dim, 1, 0, 1, 1, 1);
	free(batch_norm_14_tmp_3);

	float* batch_norm_30_tmp_2 = (float*)calloc(16384, sizeof(float));
	autox_conv2d(batch_norm_11_tmp_3, (float*)((int8_t*)weights) + 5124, (float*)((int8_t*)weights) + 5172, batch_norm_30_tmp_2, batch_norm_11_tmp_3_dim, conv2d_10_w_0_dim, batch_norm_30_tmp_2_dim, 48, 1, 1, 1, 1);
	free(batch_norm_11_tmp_3);

	batch_norm_14_tmp_3 = (float*)calloc(393216, sizeof(float));
	autox_conv2d(batch_norm_30_tmp_2, (float*)((int8_t*)weights) + 5604, (float*)((int8_t*)weights) + 5700, batch_norm_14_tmp_3, batch_norm_30_tmp_2_dim, conv2d_11_w_0_dim, batch_norm_14_tmp_3_dim, 1, 0, 1, 1, 10);
	free(batch_norm_30_tmp_2);

	batch_norm_11_tmp_3 = (float*)calloc(393216, sizeof(float));
	autox_conv2d(batch_norm_14_tmp_3, (float*)((int8_t*)weights) + 6852, (float*)((int8_t*)weights) + 6948, batch_norm_11_tmp_3, batch_norm_14_tmp_3_dim, conv2d_13_w_0_dim, batch_norm_11_tmp_3_dim, 96, 2, 2, 1, 0);
	free(batch_norm_14_tmp_3);

	batch_norm_31_tmp_2 = (float*)calloc(32768, sizeof(float));
	autox_hard_swish(batch_norm_11_tmp_3, batch_norm_11_tmp_3_dim, 4, 3.0, 6.0, 6.0);

	batch_norm_11_tmp_3 = (float*)calloc(393216, sizeof(float));
	autox_pool2d(batch_norm_31_tmp_2, batch_norm_11_tmp_3, batch_norm_31_tmp_2_dim, batch_norm_11_tmp_3_dim, 1, 1, 0, 0, 1);
	free(batch_norm_31_tmp_2);

	batch_norm_14_tmp_3 = (float*)calloc(393216, sizeof(float));
	autox_conv2d(batch_norm_11_tmp_3, (float*)((int8_t*)weights) + 9348, (float*)((int8_t*)weights) + 9372, batch_norm_14_tmp_3, batch_norm_11_tmp_3_dim, conv2d_14_w_0_dim, batch_norm_14_tmp_3_dim, 1, 0, 1, 1, 1);
	free(batch_norm_11_tmp_3);

	batch_norm_11_tmp_3 = (float*)calloc(393216, sizeof(float));
	autox_conv2d(batch_norm_14_tmp_3, (float*)((int8_t*)weights) + 11676, (float*)((int8_t*)weights) + 11772, batch_norm_11_tmp_3, batch_norm_14_tmp_3_dim, conv2d_15_w_0_dim, batch_norm_11_tmp_3_dim, 1, 0, 1, 1, 0);
	free(batch_norm_14_tmp_3);

	autox_hard_sigmoid(batch_norm_11_tmp_3, batch_norm_11_tmp_3_dim, 4, 0.5, 0.2);

	batch_norm_14_tmp_3 = (float*)calloc(393216, sizeof(float));
	autox_elementwise_mul(batch_norm_31_tmp_2, batch_norm_30_tmp_2, batch_norm_14_tmp_3, batch_norm_31_tmp_2_dim, batch_norm_30_tmp_2_dim, batch_norm_14_tmp_3_dim, -1, 4, 4, 4);
	free(batch_norm_31_tmp_2);
	free(batch_norm_30_tmp_2);

	batch_norm_11_tmp_3 = (float*)calloc(393216, sizeof(float));
	autox_conv2d(batch_norm_14_tmp_3, (float*)((int8_t*)weights) + 14076, (float*)((int8_t*)weights) + 14108, batch_norm_11_tmp_3, batch_norm_14_tmp_3_dim, conv2d_16_w_0_dim, batch_norm_11_tmp_3_dim, 1, 0, 1, 1, 0);
	free(batch_norm_14_tmp_3);

	batch_norm_30_tmp_2 = (float*)calloc(16384, sizeof(float));
	autox_conv2d(batch_norm_11_tmp_3, (float*)((int8_t*)weights) + 17180, (float*)((int8_t*)weights) + 17276, batch_norm_30_tmp_2, batch_norm_11_tmp_3_dim, conv2d_17_w_0_dim, batch_norm_30_tmp_2_dim, 1, 0, 1, 1, 10);
	free(batch_norm_11_tmp_3);

	batch_norm_14_tmp_3 = (float*)calloc(393216, sizeof(float));
	autox_conv2d(batch_norm_30_tmp_2, (float*)((int8_t*)weights) + 20348, (float*)((int8_t*)weights) + 20444, batch_norm_14_tmp_3, batch_norm_30_tmp_2_dim, conv2d_18_w_0_dim, batch_norm_14_tmp_3_dim, 96, 2, 1, 1, 0);
	free(batch_norm_30_tmp_2);

	batch_norm_34_tmp_2 = (float*)calloc(2048, sizeof(float));
	autox_hard_swish(batch_norm_14_tmp_3, batch_norm_14_tmp_3_dim, 4, 3.0, 6.0, 6.0);

	batch_norm_14_tmp_3 = (float*)calloc(393216, sizeof(float));
	autox_pool2d(batch_norm_34_tmp_2, batch_norm_14_tmp_3, batch_norm_34_tmp_2_dim, batch_norm_14_tmp_3_dim, 1, 1, 0, 0, 1);
	free(batch_norm_34_tmp_2);

	batch_norm_30_tmp_2 = (float*)calloc(16384, sizeof(float));
	autox_conv2d(batch_norm_14_tmp_3, (float*)((int8_t*)weights) + 22844, (float*)((int8_t*)weights) + 22868, batch_norm_30_tmp_2, batch_norm_14_tmp_3_dim, conv2d_19_w_0_dim, batch_norm_30_tmp_2_dim, 1, 0, 1, 1, 1);
	free(batch_norm_14_tmp_3);

	batch_norm_14_tmp_3 = (float*)calloc(393216, sizeof(float));
	autox_conv2d(batch_norm_30_tmp_2, (float*)((int8_t*)weights) + 25172, (float*)((int8_t*)weights) + 25268, batch_norm_14_tmp_3, batch_norm_30_tmp_2_dim, conv2d_20_w_0_dim, batch_norm_14_tmp_3_dim, 1, 0, 1, 1, 0);
	free(batch_norm_30_tmp_2);

	autox_hard_sigmoid(batch_norm_14_tmp_3, batch_norm_14_tmp_3_dim, 4, 0.5, 0.2);

	batch_norm_30_tmp_2 = (float*)calloc(16384, sizeof(float));
	autox_elementwise_mul(batch_norm_34_tmp_2, batch_norm_31_tmp_2, batch_norm_30_tmp_2, batch_norm_34_tmp_2_dim, batch_norm_31_tmp_2_dim, batch_norm_30_tmp_2_dim, -1, 4, 4, 4);
	free(batch_norm_34_tmp_2);
	free(batch_norm_31_tmp_2);

	batch_norm_14_tmp_3 = (float*)calloc(393216, sizeof(float));
	autox_conv2d(batch_norm_30_tmp_2, (float*)((int8_t*)weights) + 27572, (float*)((int8_t*)weights) + 27604, batch_norm_14_tmp_3, batch_norm_30_tmp_2_dim, conv2d_21_w_0_dim, batch_norm_14_tmp_3_dim, 1, 0, 1, 1, 0);
	free(batch_norm_30_tmp_2);

	float* elementwise_add_1 = (float*)calloc(131072, sizeof(float));
	autox_elementwise_add(batch_norm_11_tmp_3, batch_norm_14_tmp_3, elementwise_add_1, batch_norm_11_tmp_3_dim, batch_norm_14_tmp_3_dim, elementwise_add_1_dim, -1, 4, 4, 4);
	free(batch_norm_11_tmp_3);
	free(batch_norm_14_tmp_3);

	batch_norm_14_tmp_3 = (float*)calloc(393216, sizeof(float));
	autox_conv2d(elementwise_add_1, (float*)((int8_t*)weights) + 30676, (float*)((int8_t*)weights) + 30836, batch_norm_14_tmp_3, elementwise_add_1_dim, conv2d_22_w_0_dim, batch_norm_14_tmp_3_dim, 1, 0, 1, 1, 10);
	free(elementwise_add_1);

	batch_norm_11_tmp_3 = (float*)calloc(393216, sizeof(float));
	autox_conv2d(batch_norm_14_tmp_3, (float*)((int8_t*)weights) + 35956, (float*)((int8_t*)weights) + 36116, batch_norm_11_tmp_3, batch_norm_14_tmp_3_dim, conv2d_23_w_0_dim, batch_norm_11_tmp_3_dim, 160, 2, 2, 1, 0);
	free(batch_norm_14_tmp_3);

	batch_norm_31_tmp_2 = (float*)calloc(32768, sizeof(float));
	autox_hard_swish(batch_norm_11_tmp_3, batch_norm_11_tmp_3_dim, 4, 3.0, 6.0, 6.0);

	batch_norm_11_tmp_3 = (float*)calloc(393216, sizeof(float));
	autox_pool2d(batch_norm_31_tmp_2, batch_norm_11_tmp_3, batch_norm_31_tmp_2_dim, batch_norm_11_tmp_3_dim, 1, 1, 0, 0, 1);
	free(batch_norm_31_tmp_2);

	batch_norm_14_tmp_3 = (float*)calloc(393216, sizeof(float));
	autox_conv2d(batch_norm_11_tmp_3, (float*)((int8_t*)weights) + 40116, (float*)((int8_t*)weights) + 40156, batch_norm_14_tmp_3, batch_norm_11_tmp_3_dim, conv2d_24_w_0_dim, batch_norm_14_tmp_3_dim, 1, 0, 1, 1, 1);
	free(batch_norm_11_tmp_3);

	batch_norm_11_tmp_3 = (float*)calloc(393216, sizeof(float));
	autox_conv2d(batch_norm_14_tmp_3, (float*)((int8_t*)weights) + 46556, (float*)((int8_t*)weights) + 46716, batch_norm_11_tmp_3, batch_norm_14_tmp_3_dim, conv2d_25_w_0_dim, batch_norm_11_tmp_3_dim, 1, 0, 1, 1, 0);
	free(batch_norm_14_tmp_3);

	autox_hard_sigmoid(batch_norm_11_tmp_3, batch_norm_11_tmp_3_dim, 4, 0.5, 0.2);

	batch_norm_14_tmp_3 = (float*)calloc(393216, sizeof(float));
	autox_elementwise_mul(batch_norm_31_tmp_2, batch_norm_30_tmp_2, batch_norm_14_tmp_3, batch_norm_31_tmp_2_dim, batch_norm_30_tmp_2_dim, batch_norm_14_tmp_3_dim, -1, 4, 4, 4);
	free(batch_norm_31_tmp_2);
	free(batch_norm_30_tmp_2);

	batch_norm_11_tmp_3 = (float*)calloc(393216, sizeof(float));
	autox_conv2d(batch_norm_14_tmp_3, (float*)((int8_t*)weights) + 53116, (float*)((int8_t*)weights) + 53180, batch_norm_11_tmp_3, batch_norm_14_tmp_3_dim, conv2d_26_w_0_dim, batch_norm_11_tmp_3_dim, 1, 0, 1, 1, 0);
	free(batch_norm_14_tmp_3);

	batch_norm_30_tmp_2 = (float*)calloc(16384, sizeof(float));
	autox_conv2d(batch_norm_11_tmp_3, (float*)((int8_t*)weights) + 63420, (float*)((int8_t*)weights) + 63580, batch_norm_30_tmp_2, batch_norm_11_tmp_3_dim, conv2d_27_w_0_dim, batch_norm_30_tmp_2_dim, 1, 0, 1, 1, 10);
	free(batch_norm_11_tmp_3);

	batch_norm_14_tmp_3 = (float*)calloc(393216, sizeof(float));
	autox_conv2d(batch_norm_30_tmp_2, (float*)((int8_t*)weights) + 73820, (float*)((int8_t*)weights) + 73980, batch_norm_14_tmp_3, batch_norm_30_tmp_2_dim, conv2d_28_w_0_dim, batch_norm_14_tmp_3_dim, 160, 2, 1, 1, 0);
	free(batch_norm_30_tmp_2);

	batch_norm_34_tmp_2 = (float*)calloc(2048, sizeof(float));
	autox_hard_swish(batch_norm_14_tmp_3, batch_norm_14_tmp_3_dim, 4, 3.0, 6.0, 6.0);

	batch_norm_14_tmp_3 = (float*)calloc(393216, sizeof(float));
	autox_pool2d(batch_norm_34_tmp_2, batch_norm_14_tmp_3, batch_norm_34_tmp_2_dim, batch_norm_14_tmp_3_dim, 1, 1, 0, 0, 1);
	free(batch_norm_34_tmp_2);

	batch_norm_30_tmp_2 = (float*)calloc(16384, sizeof(float));
	autox_conv2d(batch_norm_14_tmp_3, (float*)((int8_t*)weights) + 77980, (float*)((int8_t*)weights) + 78020, batch_norm_30_tmp_2, batch_norm_14_tmp_3_dim, conv2d_29_w_0_dim, batch_norm_30_tmp_2_dim, 1, 0, 1, 1, 1);
	free(batch_norm_14_tmp_3);

	batch_norm_14_tmp_3 = (float*)calloc(393216, sizeof(float));
	autox_conv2d(batch_norm_30_tmp_2, (float*)((int8_t*)weights) + 84420, (float*)((int8_t*)weights) + 84580, batch_norm_14_tmp_3, batch_norm_30_tmp_2_dim, conv2d_30_w_0_dim, batch_norm_14_tmp_3_dim, 1, 0, 1, 1, 0);
	free(batch_norm_30_tmp_2);

	autox_hard_sigmoid(batch_norm_14_tmp_3, batch_norm_14_tmp_3_dim, 4, 0.5, 0.2);

	batch_norm_30_tmp_2 = (float*)calloc(16384, sizeof(float));
	autox_elementwise_mul(batch_norm_34_tmp_2, batch_norm_31_tmp_2, batch_norm_30_tmp_2, batch_norm_34_tmp_2_dim, batch_norm_31_tmp_2_dim, batch_norm_30_tmp_2_dim, -1, 4, 4, 4);
	free(batch_norm_34_tmp_2);
	free(batch_norm_31_tmp_2);

	batch_norm_14_tmp_3 = (float*)calloc(393216, sizeof(float));
	autox_conv2d(batch_norm_30_tmp_2, (float*)((int8_t*)weights) + 90980, (float*)((int8_t*)weights) + 91044, batch_norm_14_tmp_3, batch_norm_30_tmp_2_dim, conv2d_31_w_0_dim, batch_norm_14_tmp_3_dim, 1, 0, 1, 1, 0);
	free(batch_norm_30_tmp_2);

	float *elementwise_add_2 = (float*)calloc(65536, sizeof(float));
	autox_elementwise_add(batch_norm_11_tmp_3, batch_norm_14_tmp_3, elementwise_add_2, batch_norm_11_tmp_3_dim, batch_norm_14_tmp_3_dim, elementwise_add_2_dim, -1, 4, 4, 4);
	free(batch_norm_11_tmp_3);
	free(batch_norm_14_tmp_3);

	batch_norm_11_tmp_3 = (float*)calloc(393216, sizeof(float));
	autox_conv2d(elementwise_add_2, (float*)((int8_t*)weights) + 101284, (float*)((int8_t*)weights) + 101348, batch_norm_11_tmp_3, elementwise_add_2_dim, conv2d_45_w_0_dim, batch_norm_11_tmp_3_dim, 64, 1, 2, 1, 0);
	free(elementwise_add_2);

	batch_norm_14_tmp_3 = (float*)calloc(393216, sizeof(float));
	autox_conv2d(batch_norm_11_tmp_3, (float*)((int8_t*)weights) + 101924, (float*)((int8_t*)weights) + 101988, batch_norm_14_tmp_3, batch_norm_11_tmp_3_dim, conv2d_42_w_0_dim, batch_norm_14_tmp_3_dim, 1, 0, 1, 1, 0);
	free(batch_norm_11_tmp_3);

	batch_norm_30_tmp_2 = (float*)calloc(16384, sizeof(float));
	autox_conv2d(batch_norm_11_tmp_3, (float*)((int8_t*)weights) + 106084, (float*)((int8_t*)weights) + 106148, batch_norm_30_tmp_2, batch_norm_11_tmp_3_dim, conv2d_43_w_0_dim, batch_norm_30_tmp_2_dim, 1, 0, 1, 1, 0);
	free(batch_norm_11_tmp_3);

	batch_norm_31_tmp_2 = (float*)calloc(32768, sizeof(float));
	autox_conv2d(batch_norm_11_tmp_3, (float*)((int8_t*)weights) + 110244, (float*)((int8_t*)weights) + 110372, batch_norm_31_tmp_2, batch_norm_11_tmp_3_dim, conv2d_44_w_0_dim, batch_norm_31_tmp_2_dim, 1, 0, 1, 1, 0);
	free(batch_norm_11_tmp_3);

	float* p_56[] = { batch_norm_14_tmp_3, batch_norm_30_tmp_2, batch_norm_31_tmp_2, };
	uint16_t* p_56_dim[] = { batch_norm_14_tmp_3_dim, batch_norm_30_tmp_2_dim, batch_norm_31_tmp_2_dim, };
	batch_norm_11_tmp_3 = (float*)calloc(393216, sizeof(float));
	autox_concat(p_56, batch_norm_11_tmp_3, p_56_dim, batch_norm_11_tmp_3_dim, 1, 3, 4);
	free(batch_norm_14_tmp_3);
	free(batch_norm_30_tmp_2);
	free(batch_norm_31_tmp_2);

	batch_norm_34_tmp_2 = (float*)calloc(2048, sizeof(float));
	autox_conv2d(batch_norm_11_tmp_3, (float*)((int8_t*)weights) + 118564, (float*)((int8_t*)weights) + 118820, batch_norm_34_tmp_2, batch_norm_11_tmp_3_dim, conv2d_49_w_0_dim, batch_norm_34_tmp_2_dim, 256, 1, 1, 1, 2);
	free(batch_norm_11_tmp_3);

	batch_norm_11_tmp_3 = (float*)calloc(393216, sizeof(float));
	autox_conv2d(batch_norm_34_tmp_2, (float*)((int8_t*)weights) + 121124, (float*)((int8_t*)weights) + 121188, batch_norm_11_tmp_3, batch_norm_34_tmp_2_dim, conv2d_50_w_0_dim, batch_norm_11_tmp_3_dim, 1, 0, 1, 1, 0);
	free(batch_norm_34_tmp_2);

	float* mean_0_tmp_0 = (float*)calloc(1024, sizeof(float));
	autox_reduce_mean(batch_norm_14_tmp_3, mean_0_tmp_0, batch_norm_14_tmp_3_dim, mean_0_tmp_0_dim);
	free(batch_norm_14_tmp_3);

	batch_norm_34_tmp_2 = (float*)calloc(2048, sizeof(float));
	// autox_linear_interp_v2((float*)((int8_t*)weights) + 137572, batch_norm_34_tmp_2, (float*)((int8_t*)weights) + 137572_dim, batch_norm_34_tmp_2_dim, 0);

	float* tmp_0 = (float*)calloc(1024, sizeof(float));
	autox_elementwise_add(mean_0_tmp_0, batch_norm_34_tmp_2, tmp_0, mean_0_tmp_0_dim, batch_norm_34_tmp_2_dim, tmp_0_dim, -1, 3, 4, 3);
	free(mean_0_tmp_0);
	free(batch_norm_34_tmp_2);

	float* tmp_13 = (float*)calloc(1024, sizeof(float));
	uint16_t axis_62[] = { 0, 1, 3, 2 };
	autox_transpose(tmp_0, tmp_13, reshape2_0_tmp_0_dim, tmp_13_dim, axis_62, 4);
	free(tmp_0);

	mean_0_tmp_0 = (float*)calloc(1024, sizeof(float));
	//autox_reduce_mean(batch_norm_30_tmp_2, mean_0_tmp_0, batch_norm_30_tmp_2_dim, mean_0_tmp_0_dim);
	free(batch_norm_30_tmp_2);

	batch_norm_34_tmp_2 = (float*)calloc(2048, sizeof(float));
	// autox_linear_interp_v2((float*)((int8_t*)weights) + 138596, batch_norm_34_tmp_2, (float*)((int8_t*)weights) + 138596_dim, batch_norm_34_tmp_2_dim, 0);

	float* tmp_1 = (float*)calloc(1024, sizeof(float));
	autox_elementwise_add(mean_0_tmp_0, batch_norm_34_tmp_2, tmp_1, mean_0_tmp_0_dim, batch_norm_34_tmp_2_dim, tmp_1_dim, -1, 3, 4, 3);
	free(mean_0_tmp_0);
	free(batch_norm_34_tmp_2);

	float* mean_2_tmp_0 = (float*)calloc(2048, sizeof(float));
	autox_reduce_mean(batch_norm_31_tmp_2, mean_2_tmp_0, batch_norm_31_tmp_2_dim, mean_2_tmp_0_dim);
	free(batch_norm_31_tmp_2);

	float* tmp_24 = (float*)calloc(256, sizeof(float));
	uint16_t axis_67[] = { 0, 1, 3, 2 };
	autox_transpose(mean_2_tmp_0, tmp_24, reshape2_2_tmp_0_dim, tmp_24_dim, axis_67, 4);
	free(mean_2_tmp_0);

	batch_norm_34_tmp_2 = (float*)calloc(2048, sizeof(float));
	autox_matmul(tmp_13, tmp_1, batch_norm_34_tmp_2, tmp_13_dim, reshape2_1_tmp_0_dim, batch_norm_34_tmp_2_dim, 0, 0, 4, 4, 4);
	free(tmp_13);
	free(tmp_1);

	autox_scale(batch_norm_34_tmp_2, batch_norm_34_tmp_2_dim, 4, 0, 1, 0.25);

	autox_softmax(tmp_13, tmp_13_dim[0], tmp_13_dim[1]);

	batch_norm_34_tmp_2 = (float*)calloc(2048, sizeof(float));
	autox_matmul(mean_0_tmp_0, tmp_24, batch_norm_34_tmp_2, mean_0_tmp_0_dim, tmp_24_dim, batch_norm_34_tmp_2_dim, 0, 0, 3, 4, 4);
	free(mean_0_tmp_0);
	free(tmp_24);

	float* transpose_2_tmp_0 = (float*)calloc(2048, sizeof(float));
	uint16_t axis_72[] = { 0, 1, 3, 2 };
	autox_transpose(batch_norm_34_tmp_2, transpose_2_tmp_0, batch_norm_34_tmp_2_dim, transpose_2_tmp_0_dim, axis_72, 4);
	free(batch_norm_34_tmp_2);

	mean_0_tmp_0 = (float*)calloc(1024, sizeof(float));
	autox_relu6(transpose_2_tmp_0, reshape2_3_tmp_0_dim, 4, 6.0);

	batch_norm_34_tmp_2 = (float*)calloc(2048, sizeof(float));
	autox_conv2d(mean_0_tmp_0, (float*)((int8_t*)weights) + 139620, (float*)((int8_t*)weights) + 139748, batch_norm_34_tmp_2, mean_0_tmp_0_dim, conv2d_47_w_0_dim, batch_norm_34_tmp_2_dim, 1, 0, 1, 1, 0);
	free(mean_0_tmp_0);

	mean_0_tmp_0 = (float*)calloc(1024, sizeof(float));
	autox_reduce_mean(batch_norm_14_tmp_3, mean_0_tmp_0, batch_norm_14_tmp_3_dim, mean_0_tmp_0_dim);
	free(batch_norm_14_tmp_3);

	batch_norm_14_tmp_3 = (float*)calloc(393216, sizeof(float));
	// autox_linear_interp_v2((float*)((int8_t*)weights) + 156132, batch_norm_14_tmp_3, (float*)((int8_t*)weights) + 156132_dim, batch_norm_14_tmp_3_dim, 0);

	float* tmp_3 = (float*)calloc(1024, sizeof(float));
	autox_elementwise_add(mean_0_tmp_0, batch_norm_14_tmp_3, tmp_3, mean_0_tmp_0_dim, batch_norm_14_tmp_3_dim, tmp_3_dim, -1, 3, 4, 3);
	free(mean_0_tmp_0);
	free(batch_norm_14_tmp_3);

	tmp_13 = (float*)calloc(1024, sizeof(float));
	uint16_t axis_78[] = { 0, 1, 3, 2 };
	autox_transpose(tmp_3, tmp_13, reshape2_4_tmp_0_dim, tmp_13_dim, axis_78, 4);
	free(tmp_3);

	mean_0_tmp_0 = (float*)calloc(1024, sizeof(float));
	autox_reduce_mean(batch_norm_30_tmp_2, mean_0_tmp_0, batch_norm_30_tmp_2_dim, mean_0_tmp_0_dim);
	free(batch_norm_30_tmp_2);

	batch_norm_14_tmp_3 = (float*)calloc(393216, sizeof(float));
	// autox_linear_interp_v2((float*)((int8_t*)weights) + 157156, batch_norm_14_tmp_3, (float*)((int8_t*)weights) + 157156_dim, batch_norm_14_tmp_3_dim, 0);

	float* tmp_4 = (float*)calloc(1024, sizeof(float));
	autox_elementwise_add(mean_0_tmp_0, batch_norm_14_tmp_3, tmp_4, mean_0_tmp_0_dim, batch_norm_14_tmp_3_dim, tmp_4_dim, -1, 3, 4, 3);
	free(mean_0_tmp_0);
	free(batch_norm_14_tmp_3);

	float* mean_5_tmp_0 = (float*)calloc(2048, sizeof(float));
	autox_reduce_mean(batch_norm_31_tmp_2, mean_5_tmp_0, batch_norm_31_tmp_2_dim, mean_5_tmp_0_dim);
	free(batch_norm_31_tmp_2);

	tmp_24 = (float*)calloc(256, sizeof(float));
	uint16_t axis_83[] = { 0, 1, 3, 2 };
	autox_transpose(mean_5_tmp_0, tmp_24, reshape2_6_tmp_0_dim, tmp_24_dim, axis_83, 4);
	free(mean_5_tmp_0);

	batch_norm_14_tmp_3 = (float*)calloc(393216, sizeof(float));
	autox_matmul(tmp_13, tmp_4, batch_norm_14_tmp_3, tmp_13_dim, reshape2_5_tmp_0_dim, batch_norm_14_tmp_3_dim, 0, 0, 4, 4, 4);
	free(tmp_13);
	free(tmp_4);

	autox_scale(batch_norm_14_tmp_3, batch_norm_14_tmp_3_dim, 4, 0, 1, 0.25);

	autox_softmax(mean_0_tmp_0, mean_0_tmp_0_dim[0], mean_0_tmp_0_dim[1]);

	batch_norm_14_tmp_3 = (float*)calloc(393216, sizeof(float));
	autox_matmul(batch_norm_30_tmp_2, tmp_24, batch_norm_14_tmp_3, batch_norm_30_tmp_2_dim, tmp_24_dim, batch_norm_14_tmp_3_dim, 0, 0, 4, 4, 4);
	free(batch_norm_30_tmp_2);
	free(tmp_24);

	float* transpose_5_tmp_0 = (float*)calloc(2048, sizeof(float));
	uint16_t axis_88[] = { 0, 1, 3, 2 };
	autox_transpose(batch_norm_14_tmp_3, transpose_5_tmp_0, batch_norm_14_tmp_3_dim, transpose_5_tmp_0_dim, axis_88, 4);
	free(batch_norm_14_tmp_3);

	batch_norm_30_tmp_2 = (float*)calloc(16384, sizeof(float));
	autox_relu6(transpose_5_tmp_0, reshape2_7_tmp_0_dim, 4, 6.0);

	batch_norm_14_tmp_3 = (float*)calloc(393216, sizeof(float));
	autox_conv2d(batch_norm_30_tmp_2, (float*)((int8_t*)weights) + 158180, (float*)((int8_t*)weights) + 158308, batch_norm_14_tmp_3, batch_norm_30_tmp_2_dim, conv2d_48_w_0_dim, batch_norm_14_tmp_3_dim, 1, 0, 1, 1, 0);
	free(batch_norm_30_tmp_2);

	batch_norm_30_tmp_2 = (float*)calloc(16384, sizeof(float));
	autox_elementwise_add(batch_norm_34_tmp_2, batch_norm_14_tmp_3, batch_norm_30_tmp_2, batch_norm_34_tmp_2_dim, batch_norm_14_tmp_3_dim, batch_norm_30_tmp_2_dim, -1, 4, 4, 4);
	free(batch_norm_34_tmp_2);
	free(batch_norm_14_tmp_3);

	batch_norm_14_tmp_3 = (float*)calloc(393216, sizeof(float));
	autox_elementwise_add(batch_norm_31_tmp_2, batch_norm_30_tmp_2, batch_norm_14_tmp_3, batch_norm_31_tmp_2_dim, batch_norm_30_tmp_2_dim, batch_norm_14_tmp_3_dim, -1, 4, 4, 4);
	free(batch_norm_31_tmp_2);
	free(batch_norm_30_tmp_2);

	batch_norm_30_tmp_2 = (float*)calloc(16384, sizeof(float));
	autox_relu6(batch_norm_14_tmp_3, batch_norm_14_tmp_3_dim, 4, 6.0);

	batch_norm_14_tmp_3 = (float*)calloc(393216, sizeof(float));
	autox_conv2d(batch_norm_30_tmp_2, (float*)((int8_t*)weights) + 174692, (float*)((int8_t*)weights) + 174756, batch_norm_14_tmp_3, batch_norm_30_tmp_2_dim, conv2d_46_w_0_dim, batch_norm_14_tmp_3_dim, 1, 0, 1, 1, 2);
	free(batch_norm_30_tmp_2);

	autox_scale(batch_norm_14_tmp_3, batch_norm_14_tmp_3_dim, 4, 0, 1, 0.16666667);

	batch_norm_14_tmp_3 = (float*)calloc(393216, sizeof(float));
	autox_elementwise_mul(batch_norm_30_tmp_2, batch_norm_11_tmp_3, batch_norm_14_tmp_3, batch_norm_30_tmp_2_dim, batch_norm_11_tmp_3_dim, batch_norm_14_tmp_3_dim, -1, 4, 4, 4);
	free(batch_norm_30_tmp_2);
	free(batch_norm_11_tmp_3);

	batch_norm_11_tmp_3 = (float*)calloc(393216, sizeof(float));
	//// autox_bilinear_interp(batch_norm_14_tmp_3, batch_norm_11_tmp_3, batch_norm_14_tmp_3_dim, batch_norm_11_tmp_3_dim, , 0);
	free(batch_norm_14_tmp_3);

	batch_norm_30_tmp_2 = (float*)calloc(16384, sizeof(float));
	autox_elementwise_add(elementwise_add_2, batch_norm_11_tmp_3, batch_norm_30_tmp_2, elementwise_add_2_dim, batch_norm_11_tmp_3_dim, batch_norm_30_tmp_2_dim, -1, 4, 4, 4);
	free(elementwise_add_2);
	free(batch_norm_11_tmp_3);

	batch_norm_11_tmp_3 = (float*)calloc(393216, sizeof(float));
	autox_conv2d(batch_norm_30_tmp_2, (float*)((int8_t*)weights) + 182948, (float*)((int8_t*)weights) + 183076, batch_norm_11_tmp_3, batch_norm_30_tmp_2_dim, conv2d_51_w_0_dim, batch_norm_11_tmp_3_dim, 1, 0, 1, 1, 0);
	free(batch_norm_30_tmp_2);

	batch_norm_14_tmp_3 = (float*)calloc(393216, sizeof(float));
	autox_conv2d(batch_norm_11_tmp_3, (float*)((int8_t*)weights) + 191268, (float*)((int8_t*)weights) + 191396, batch_norm_14_tmp_3, batch_norm_11_tmp_3_dim, conv2d_52_w_0_dim, batch_norm_14_tmp_3_dim, 128, 1, 1, 1, 2);
	free(batch_norm_11_tmp_3);

	batch_norm_11_tmp_3 = (float*)calloc(393216, sizeof(float));
	autox_conv2d(batch_norm_14_tmp_3, (float*)((int8_t*)weights) + 192548, (float*)((int8_t*)weights) + 192612, batch_norm_11_tmp_3, batch_norm_14_tmp_3_dim, conv2d_53_w_0_dim, batch_norm_11_tmp_3_dim, 1, 0, 1, 1, 0);
	free(batch_norm_14_tmp_3);

	mean_0_tmp_0 = (float*)calloc(1024, sizeof(float));
	autox_elementwise_add(batch_norm_30_tmp_2, batch_norm_11_tmp_3, mean_0_tmp_0, batch_norm_30_tmp_2_dim, batch_norm_11_tmp_3_dim, mean_0_tmp_0_dim, -1, 4, 4, 3);
	free(batch_norm_30_tmp_2);
	free(batch_norm_11_tmp_3);

	batch_norm_11_tmp_3 = (float*)calloc(393216, sizeof(float));
	autox_conv2d(mean_0_tmp_0, (float*)((int8_t*)weights) + 200804, (float*)((int8_t*)weights) + 200868, batch_norm_11_tmp_3, mean_0_tmp_0_dim, conv2d_57_w_0_dim, batch_norm_11_tmp_3_dim, 64, 1, 2, 1, 0);
	free(mean_0_tmp_0);

	batch_norm_14_tmp_3 = (float*)calloc(393216, sizeof(float));
	autox_conv2d(batch_norm_11_tmp_3, (float*)((int8_t*)weights) + 201444, (float*)((int8_t*)weights) + 201508, batch_norm_14_tmp_3, batch_norm_11_tmp_3_dim, conv2d_54_w_0_dim, batch_norm_14_tmp_3_dim, 1, 0, 1, 1, 0);
	free(batch_norm_11_tmp_3);

	batch_norm_30_tmp_2 = (float*)calloc(16384, sizeof(float));
	autox_conv2d(batch_norm_11_tmp_3, (float*)((int8_t*)weights) + 205604, (float*)((int8_t*)weights) + 205668, batch_norm_30_tmp_2, batch_norm_11_tmp_3_dim, conv2d_55_w_0_dim, batch_norm_30_tmp_2_dim, 1, 0, 1, 1, 0);
	free(batch_norm_11_tmp_3);

	batch_norm_31_tmp_2 = (float*)calloc(32768, sizeof(float));
	autox_conv2d(batch_norm_11_tmp_3, (float*)((int8_t*)weights) + 209764, (float*)((int8_t*)weights) + 209892, batch_norm_31_tmp_2, batch_norm_11_tmp_3_dim, conv2d_56_w_0_dim, batch_norm_31_tmp_2_dim, 1, 0, 1, 1, 0);
	free(batch_norm_11_tmp_3);

	float* p_107[] = { batch_norm_14_tmp_3, batch_norm_30_tmp_2, batch_norm_31_tmp_2, };
	uint16_t* p_107_dim[] = { batch_norm_14_tmp_3_dim, batch_norm_30_tmp_2_dim, batch_norm_31_tmp_2_dim, };
	batch_norm_11_tmp_3 = (float*)calloc(393216, sizeof(float));
	autox_concat(p_107, batch_norm_11_tmp_3, p_107_dim, batch_norm_11_tmp_3_dim, 1, 3, 4);
	free(batch_norm_14_tmp_3);
	free(batch_norm_30_tmp_2);
	free(batch_norm_31_tmp_2);

	batch_norm_34_tmp_2 = (float*)calloc(2048, sizeof(float));
	autox_conv2d(batch_norm_11_tmp_3, (float*)((int8_t*)weights) + 218084, (float*)((int8_t*)weights) + 218340, batch_norm_34_tmp_2, batch_norm_11_tmp_3_dim, conv2d_61_w_0_dim, batch_norm_34_tmp_2_dim, 256, 1, 1, 1, 2);
	free(batch_norm_11_tmp_3);

	batch_norm_11_tmp_3 = (float*)calloc(393216, sizeof(float));
	autox_conv2d(batch_norm_34_tmp_2, (float*)((int8_t*)weights) + 220644, (float*)((int8_t*)weights) + 220708, batch_norm_11_tmp_3, batch_norm_34_tmp_2_dim, conv2d_62_w_0_dim, batch_norm_11_tmp_3_dim, 1, 0, 1, 1, 0);
	free(batch_norm_34_tmp_2);

	elementwise_add_2 = (float*)calloc(65536, sizeof(float));
	autox_reduce_mean(batch_norm_14_tmp_3, elementwise_add_2, batch_norm_14_tmp_3_dim, elementwise_add_2_dim);
	free(batch_norm_14_tmp_3);

	batch_norm_34_tmp_2 = (float*)calloc(2048, sizeof(float));
	// autox_linear_interp_v2((float*)((int8_t*)weights) + 237092, batch_norm_34_tmp_2, (float*)((int8_t*)weights) + 237092_dim, batch_norm_34_tmp_2_dim, 0);

	float* tmp_11 = (float*)calloc(1024, sizeof(float));
	autox_elementwise_add(elementwise_add_2, batch_norm_34_tmp_2, tmp_11, elementwise_add_2_dim, batch_norm_34_tmp_2_dim, tmp_11_dim, -1, 4, 4, 3);
	free(elementwise_add_2);
	free(batch_norm_34_tmp_2);

	tmp_13 = (float*)calloc(1024, sizeof(float));
	uint16_t axis_113[] = { 0, 1, 3, 2 };
	autox_transpose(tmp_11, tmp_13, reshape2_8_tmp_0_dim, tmp_13_dim, axis_113, 4);
	free(tmp_11);

	elementwise_add_2 = (float*)calloc(65536, sizeof(float));
	autox_reduce_mean(batch_norm_30_tmp_2, elementwise_add_2, batch_norm_30_tmp_2_dim, elementwise_add_2_dim);
	free(batch_norm_30_tmp_2);

	batch_norm_34_tmp_2 = (float*)calloc(2048, sizeof(float));
	// autox_linear_interp_v2((float*)((int8_t*)weights) + 238116, batch_norm_34_tmp_2, (float*)((int8_t*)weights) + 238116_dim, batch_norm_34_tmp_2_dim, 0);

	float* tmp_12 = (float*)calloc(1024, sizeof(float));
	autox_elementwise_add(elementwise_add_2, batch_norm_34_tmp_2, tmp_12, elementwise_add_2_dim, batch_norm_34_tmp_2_dim, tmp_12_dim, -1, 4, 4, 3);
	free(elementwise_add_2);
	free(batch_norm_34_tmp_2);

	float* mean_8_tmp_0 = (float*)calloc(2048, sizeof(float));
	autox_reduce_mean(batch_norm_31_tmp_2, mean_8_tmp_0, batch_norm_31_tmp_2_dim, mean_8_tmp_0_dim);
	free(batch_norm_31_tmp_2);

	tmp_24 = (float*)calloc(256, sizeof(float));
	uint16_t axis_118[] = { 0, 1, 3, 2 };
	autox_transpose(mean_8_tmp_0, tmp_24, reshape2_10_tmp_0_dim, tmp_24_dim, axis_118, 4);
	free(mean_8_tmp_0);

	batch_norm_34_tmp_2 = (float*)calloc(2048, sizeof(float));
	autox_matmul(tmp_13, tmp_12, batch_norm_34_tmp_2, tmp_13_dim, reshape2_9_tmp_0_dim, batch_norm_34_tmp_2_dim, 0, 0, 4, 4, 4);
	free(tmp_13);
	free(tmp_12);

	autox_scale(batch_norm_34_tmp_2, batch_norm_34_tmp_2_dim, 4, 0, 1, 0.25);

	autox_softmax(tmp_13, tmp_13_dim[0], tmp_13_dim[1]);

	batch_norm_34_tmp_2 = (float*)calloc(2048, sizeof(float));
	autox_matmul(elementwise_add_2, tmp_24, batch_norm_34_tmp_2, elementwise_add_2_dim, tmp_24_dim, batch_norm_34_tmp_2_dim, 0, 0, 4, 4, 4);
	free(elementwise_add_2);
	free(tmp_24);

	float* transpose_8_tmp_0 = (float*)calloc(2048, sizeof(float));
	uint16_t axis_123[] = { 0, 1, 3, 2 };
	autox_transpose(batch_norm_34_tmp_2, transpose_8_tmp_0, batch_norm_34_tmp_2_dim, transpose_8_tmp_0_dim, axis_123, 4);
	free(batch_norm_34_tmp_2);

	elementwise_add_2 = (float*)calloc(65536, sizeof(float));
	autox_relu6(transpose_8_tmp_0, reshape2_11_tmp_0_dim, 4, 6.0);

	batch_norm_34_tmp_2 = (float*)calloc(2048, sizeof(float));
	autox_conv2d(elementwise_add_2, (float*)((int8_t*)weights) + 239140, (float*)((int8_t*)weights) + 239268, batch_norm_34_tmp_2, elementwise_add_2_dim, conv2d_59_w_0_dim, batch_norm_34_tmp_2_dim, 1, 0, 1, 1, 0);
	free(elementwise_add_2);

	elementwise_add_2 = (float*)calloc(65536, sizeof(float));
	autox_reduce_mean(batch_norm_14_tmp_3, elementwise_add_2, batch_norm_14_tmp_3_dim, elementwise_add_2_dim);
	free(batch_norm_14_tmp_3);

	batch_norm_14_tmp_3 = (float*)calloc(393216, sizeof(float));
	// autox_linear_interp_v2((float*)((int8_t*)weights) + 255652, batch_norm_14_tmp_3, (float*)((int8_t*)weights) + 255652_dim, batch_norm_14_tmp_3_dim, 0);

	float* tmp_14 = (float*)calloc(1024, sizeof(float));
	autox_elementwise_add(elementwise_add_2, batch_norm_14_tmp_3, tmp_14, elementwise_add_2_dim, batch_norm_14_tmp_3_dim, tmp_14_dim, -1, 4, 4, 3);
	free(elementwise_add_2);
	free(batch_norm_14_tmp_3);

	tmp_24 = (float*)calloc(256, sizeof(float));
	uint16_t axis_129[] = { 0, 1, 3, 2 };
	autox_transpose(tmp_14, tmp_24, reshape2_12_tmp_0_dim, tmp_24_dim, axis_129, 4);
	free(tmp_14);

	elementwise_add_2 = (float*)calloc(65536, sizeof(float));
	autox_reduce_mean(batch_norm_30_tmp_2, elementwise_add_2, batch_norm_30_tmp_2_dim, elementwise_add_2_dim);
	free(batch_norm_30_tmp_2);

	batch_norm_14_tmp_3 = (float*)calloc(393216, sizeof(float));
	// autox_linear_interp_v2((float*)((int8_t*)weights) + 256676, batch_norm_14_tmp_3, (float*)((int8_t*)weights) + 256676_dim, batch_norm_14_tmp_3_dim, 0);

	float* tmp_15 = (float*)calloc(1024, sizeof(float));
	autox_elementwise_add(elementwise_add_2, batch_norm_14_tmp_3, tmp_15, elementwise_add_2_dim, batch_norm_14_tmp_3_dim, tmp_15_dim, -1, 4, 4, 3);
	free(elementwise_add_2);
	free(batch_norm_14_tmp_3);

	float* mean_11_tmp_0 = (float*)calloc(2048, sizeof(float));
	autox_reduce_mean(batch_norm_31_tmp_2, mean_11_tmp_0, batch_norm_31_tmp_2_dim, mean_11_tmp_0_dim);
	free(batch_norm_31_tmp_2);

	tmp_13 = (float*)calloc(1024, sizeof(float));
	uint16_t axis_134[] = { 0, 1, 3, 2 };
	autox_transpose(mean_11_tmp_0, tmp_13, reshape2_14_tmp_0_dim, tmp_13_dim, axis_134, 4);
	free(mean_11_tmp_0);

	batch_norm_14_tmp_3 = (float*)calloc(393216, sizeof(float));
	autox_matmul(tmp_24, tmp_15, batch_norm_14_tmp_3, tmp_24_dim, reshape2_13_tmp_0_dim, batch_norm_14_tmp_3_dim, 0, 0, 4, 4, 4);
	free(tmp_24);
	free(tmp_15);

	autox_scale(batch_norm_14_tmp_3, batch_norm_14_tmp_3_dim, 4, 0, 1, 0.25);

	autox_softmax(elementwise_add_2, elementwise_add_2_dim[0], elementwise_add_2_dim[1]);

	batch_norm_14_tmp_3 = (float*)calloc(393216, sizeof(float));
	autox_matmul(batch_norm_30_tmp_2, tmp_13, batch_norm_14_tmp_3, batch_norm_30_tmp_2_dim, tmp_13_dim, batch_norm_14_tmp_3_dim, 0, 0, 4, 4, 4);
	free(batch_norm_30_tmp_2);
	free(tmp_13);

	float* transpose_11_tmp_0 = (float*)calloc(2048, sizeof(float));
	uint16_t axis_139[] = { 0, 1, 3, 2 };
	autox_transpose(batch_norm_14_tmp_3, transpose_11_tmp_0, batch_norm_14_tmp_3_dim, transpose_11_tmp_0_dim, axis_139, 4);
	free(batch_norm_14_tmp_3);

	batch_norm_30_tmp_2 = (float*)calloc(16384, sizeof(float));
	autox_relu6(transpose_11_tmp_0, reshape2_15_tmp_0_dim, 4, 6.0);

	batch_norm_14_tmp_3 = (float*)calloc(393216, sizeof(float));
	autox_conv2d(batch_norm_30_tmp_2, (float*)((int8_t*)weights) + 257700, (float*)((int8_t*)weights) + 257828, batch_norm_14_tmp_3, batch_norm_30_tmp_2_dim, conv2d_60_w_0_dim, batch_norm_14_tmp_3_dim, 1, 0, 1, 1, 0);
	free(batch_norm_30_tmp_2);

	batch_norm_30_tmp_2 = (float*)calloc(16384, sizeof(float));
	autox_elementwise_add(batch_norm_34_tmp_2, batch_norm_14_tmp_3, batch_norm_30_tmp_2, batch_norm_34_tmp_2_dim, batch_norm_14_tmp_3_dim, batch_norm_30_tmp_2_dim, -1, 4, 4, 4);
	free(batch_norm_34_tmp_2);
	free(batch_norm_14_tmp_3);

	batch_norm_14_tmp_3 = (float*)calloc(393216, sizeof(float));
	autox_elementwise_add(batch_norm_31_tmp_2, batch_norm_30_tmp_2, batch_norm_14_tmp_3, batch_norm_31_tmp_2_dim, batch_norm_30_tmp_2_dim, batch_norm_14_tmp_3_dim, -1, 4, 4, 4);
	free(batch_norm_31_tmp_2);
	free(batch_norm_30_tmp_2);

	batch_norm_30_tmp_2 = (float*)calloc(16384, sizeof(float));
	autox_relu6(batch_norm_14_tmp_3, batch_norm_14_tmp_3_dim, 4, 6.0);

	batch_norm_14_tmp_3 = (float*)calloc(393216, sizeof(float));
	autox_conv2d(batch_norm_30_tmp_2, (float*)((int8_t*)weights) + 274212, (float*)((int8_t*)weights) + 274276, batch_norm_14_tmp_3, batch_norm_30_tmp_2_dim, conv2d_58_w_0_dim, batch_norm_14_tmp_3_dim, 1, 0, 1, 1, 2);
	free(batch_norm_30_tmp_2);

	autox_scale(batch_norm_14_tmp_3, batch_norm_14_tmp_3_dim, 4, 0, 1, 0.16666667);

	batch_norm_14_tmp_3 = (float*)calloc(393216, sizeof(float));
	autox_elementwise_mul(batch_norm_30_tmp_2, batch_norm_11_tmp_3, batch_norm_14_tmp_3, batch_norm_30_tmp_2_dim, batch_norm_11_tmp_3_dim, batch_norm_14_tmp_3_dim, -1, 4, 4, 4);
	free(batch_norm_30_tmp_2);
	free(batch_norm_11_tmp_3);

	batch_norm_11_tmp_3 = (float*)calloc(393216, sizeof(float));
	// autox_bilinear_interp(batch_norm_14_tmp_3, batch_norm_11_tmp_3, batch_norm_14_tmp_3_dim, batch_norm_11_tmp_3_dim, , 0);
	free(batch_norm_14_tmp_3);

	batch_norm_30_tmp_2 = (float*)calloc(16384, sizeof(float));
	autox_elementwise_add(mean_0_tmp_0, batch_norm_11_tmp_3, batch_norm_30_tmp_2, mean_0_tmp_0_dim, batch_norm_11_tmp_3_dim, batch_norm_30_tmp_2_dim, -1, 3, 4, 4);
	free(mean_0_tmp_0);
	free(batch_norm_11_tmp_3);

	batch_norm_11_tmp_3 = (float*)calloc(393216, sizeof(float));
	autox_conv2d(batch_norm_30_tmp_2, (float*)((int8_t*)weights) + 282468, (float*)((int8_t*)weights) + 282596, batch_norm_11_tmp_3, batch_norm_30_tmp_2_dim, conv2d_63_w_0_dim, batch_norm_11_tmp_3_dim, 1, 0, 1, 1, 0);
	free(batch_norm_30_tmp_2);

	batch_norm_14_tmp_3 = (float*)calloc(393216, sizeof(float));
	autox_conv2d(batch_norm_11_tmp_3, (float*)((int8_t*)weights) + 290788, (float*)((int8_t*)weights) + 290916, batch_norm_14_tmp_3, batch_norm_11_tmp_3_dim, conv2d_64_w_0_dim, batch_norm_14_tmp_3_dim, 128, 1, 1, 1, 2);
	free(batch_norm_11_tmp_3);

	batch_norm_11_tmp_3 = (float*)calloc(393216, sizeof(float));
	autox_conv2d(batch_norm_14_tmp_3, (float*)((int8_t*)weights) + 292068, (float*)((int8_t*)weights) + 292132, batch_norm_11_tmp_3, batch_norm_14_tmp_3_dim, conv2d_65_w_0_dim, batch_norm_11_tmp_3_dim, 1, 0, 1, 1, 0);
	free(batch_norm_14_tmp_3);

	tmp_13 = (float*)calloc(1024, sizeof(float));
	autox_elementwise_add(batch_norm_30_tmp_2, batch_norm_11_tmp_3, tmp_13, batch_norm_30_tmp_2_dim, batch_norm_11_tmp_3_dim, tmp_13_dim, -1, 4, 4, 4);
	free(batch_norm_30_tmp_2);
	free(batch_norm_11_tmp_3);

	batch_norm_14_tmp_3 = (float*)calloc(393216, sizeof(float));
	autox_conv2d(tmp_13, (float*)((int8_t*)weights) + 300324, (float*)((int8_t*)weights) + 300708, batch_norm_14_tmp_3, tmp_13_dim, conv2d_32_w_0_dim, batch_norm_14_tmp_3_dim, 1, 0, 1, 1, 10);
	free(tmp_13);

	batch_norm_11_tmp_3 = (float*)calloc(393216, sizeof(float));
	autox_conv2d(batch_norm_14_tmp_3, (float*)((int8_t*)weights) + 325284, (float*)((int8_t*)weights) + 325668, batch_norm_11_tmp_3, batch_norm_14_tmp_3_dim, conv2d_33_w_0_dim, batch_norm_11_tmp_3_dim, 384, 1, 2, 1, 0);
	free(batch_norm_14_tmp_3);

	batch_norm_31_tmp_2 = (float*)calloc(32768, sizeof(float));
	autox_hard_swish(batch_norm_11_tmp_3, batch_norm_11_tmp_3_dim, 4, 3.0, 6.0, 6.0);

	batch_norm_11_tmp_3 = (float*)calloc(393216, sizeof(float));
	autox_pool2d(batch_norm_31_tmp_2, batch_norm_11_tmp_3, batch_norm_31_tmp_2_dim, batch_norm_11_tmp_3_dim, 1, 1, 0, 0, 1);
	free(batch_norm_31_tmp_2);

	batch_norm_14_tmp_3 = (float*)calloc(393216, sizeof(float));
	autox_conv2d(batch_norm_11_tmp_3, (float*)((int8_t*)weights) + 329124, (float*)((int8_t*)weights) + 329220, batch_norm_14_tmp_3, batch_norm_11_tmp_3_dim, conv2d_34_w_0_dim, batch_norm_14_tmp_3_dim, 1, 0, 1, 1, 1);
	free(batch_norm_11_tmp_3);

	batch_norm_11_tmp_3 = (float*)calloc(393216, sizeof(float));
	autox_conv2d(batch_norm_14_tmp_3, (float*)((int8_t*)weights) + 366084, (float*)((int8_t*)weights) + 366468, batch_norm_11_tmp_3, batch_norm_14_tmp_3_dim, conv2d_35_w_0_dim, batch_norm_11_tmp_3_dim, 1, 0, 1, 1, 0);
	free(batch_norm_14_tmp_3);

	autox_hard_sigmoid(batch_norm_11_tmp_3, batch_norm_11_tmp_3_dim, 4, 0.5, 0.2);

	batch_norm_14_tmp_3 = (float*)calloc(393216, sizeof(float));
	autox_elementwise_mul(batch_norm_31_tmp_2, batch_norm_30_tmp_2, batch_norm_14_tmp_3, batch_norm_31_tmp_2_dim, batch_norm_30_tmp_2_dim, batch_norm_14_tmp_3_dim, -1, 4, 4, 4);
	free(batch_norm_31_tmp_2);
	free(batch_norm_30_tmp_2);

	batch_norm_11_tmp_3 = (float*)calloc(393216, sizeof(float));
	autox_conv2d(batch_norm_14_tmp_3, (float*)((int8_t*)weights) + 403332, (float*)((int8_t*)weights) + 403460, batch_norm_11_tmp_3, batch_norm_14_tmp_3_dim, conv2d_36_w_0_dim, batch_norm_11_tmp_3_dim, 1, 0, 1, 1, 0);
	free(batch_norm_14_tmp_3);

	batch_norm_30_tmp_2 = (float*)calloc(16384, sizeof(float));
	autox_conv2d(batch_norm_11_tmp_3, (float*)((int8_t*)weights) + 452612, (float*)((int8_t*)weights) + 452996, batch_norm_30_tmp_2, batch_norm_11_tmp_3_dim, conv2d_37_w_0_dim, batch_norm_30_tmp_2_dim, 1, 0, 1, 1, 10);
	free(batch_norm_11_tmp_3);

	batch_norm_14_tmp_3 = (float*)calloc(393216, sizeof(float));
	autox_conv2d(batch_norm_30_tmp_2, (float*)((int8_t*)weights) + 502148, (float*)((int8_t*)weights) + 502532, batch_norm_14_tmp_3, batch_norm_30_tmp_2_dim, conv2d_38_w_0_dim, batch_norm_14_tmp_3_dim, 384, 1, 1, 1, 0);
	free(batch_norm_30_tmp_2);

	batch_norm_34_tmp_2 = (float*)calloc(2048, sizeof(float));
	autox_hard_swish(batch_norm_14_tmp_3, batch_norm_14_tmp_3_dim, 4, 3.0, 6.0, 6.0);

	batch_norm_14_tmp_3 = (float*)calloc(393216, sizeof(float));
	autox_pool2d(batch_norm_34_tmp_2, batch_norm_14_tmp_3, batch_norm_34_tmp_2_dim, batch_norm_14_tmp_3_dim, 1, 1, 0, 0, 1);
	free(batch_norm_34_tmp_2);

	batch_norm_30_tmp_2 = (float*)calloc(16384, sizeof(float));
	autox_conv2d(batch_norm_14_tmp_3, (float*)((int8_t*)weights) + 505988, (float*)((int8_t*)weights) + 506084, batch_norm_30_tmp_2, batch_norm_14_tmp_3_dim, conv2d_39_w_0_dim, batch_norm_30_tmp_2_dim, 1, 0, 1, 1, 1);
	free(batch_norm_14_tmp_3);

	batch_norm_14_tmp_3 = (float*)calloc(393216, sizeof(float));
	autox_conv2d(batch_norm_30_tmp_2, (float*)((int8_t*)weights) + 542948, (float*)((int8_t*)weights) + 543332, batch_norm_14_tmp_3, batch_norm_30_tmp_2_dim, conv2d_40_w_0_dim, batch_norm_14_tmp_3_dim, 1, 0, 1, 1, 0);
	free(batch_norm_30_tmp_2);

	autox_hard_sigmoid(batch_norm_14_tmp_3, batch_norm_14_tmp_3_dim, 4, 0.5, 0.2);

	batch_norm_30_tmp_2 = (float*)calloc(16384, sizeof(float));
	autox_elementwise_mul(batch_norm_34_tmp_2, batch_norm_31_tmp_2, batch_norm_30_tmp_2, batch_norm_34_tmp_2_dim, batch_norm_31_tmp_2_dim, batch_norm_30_tmp_2_dim, -1, 4, 4, 4);
	free(batch_norm_34_tmp_2);
	free(batch_norm_31_tmp_2);

	batch_norm_14_tmp_3 = (float*)calloc(393216, sizeof(float));
	autox_conv2d(batch_norm_30_tmp_2, (float*)((int8_t*)weights) + 580196, (float*)((int8_t*)weights) + 580324, batch_norm_14_tmp_3, batch_norm_30_tmp_2_dim, conv2d_41_w_0_dim, batch_norm_14_tmp_3_dim, 1, 0, 1, 1, 0);
	free(batch_norm_30_tmp_2);

	elementwise_add_2 = (float*)calloc(65536, sizeof(float));
	autox_elementwise_add(batch_norm_11_tmp_3, batch_norm_14_tmp_3, elementwise_add_2, batch_norm_11_tmp_3_dim, batch_norm_14_tmp_3_dim, elementwise_add_2_dim, -1, 4, 4, 4);
	free(batch_norm_11_tmp_3);
	free(batch_norm_14_tmp_3);

	batch_norm_11_tmp_3 = (float*)calloc(393216, sizeof(float));
	autox_conv2d(elementwise_add_2, (float*)((int8_t*)weights) + 629476, (float*)((int8_t*)weights) + 629604, batch_norm_11_tmp_3, elementwise_add_2_dim, conv2d_69_w_0_dim, batch_norm_11_tmp_3_dim, 128, 1, 2, 1, 0);
	free(elementwise_add_2);

	batch_norm_14_tmp_3 = (float*)calloc(393216, sizeof(float));
	autox_conv2d(batch_norm_11_tmp_3, (float*)((int8_t*)weights) + 630756, (float*)((int8_t*)weights) + 630852, batch_norm_14_tmp_3, batch_norm_11_tmp_3_dim, conv2d_66_w_0_dim, batch_norm_14_tmp_3_dim, 1, 0, 1, 1, 0);
	free(batch_norm_11_tmp_3);

	batch_norm_30_tmp_2 = (float*)calloc(16384, sizeof(float));
	autox_conv2d(batch_norm_11_tmp_3, (float*)((int8_t*)weights) + 643140, (float*)((int8_t*)weights) + 643236, batch_norm_30_tmp_2, batch_norm_11_tmp_3_dim, conv2d_67_w_0_dim, batch_norm_30_tmp_2_dim, 1, 0, 1, 1, 0);
	free(batch_norm_11_tmp_3);

	batch_norm_31_tmp_2 = (float*)calloc(32768, sizeof(float));
	autox_conv2d(batch_norm_11_tmp_3, (float*)((int8_t*)weights) + 655524, (float*)((int8_t*)weights) + 655716, batch_norm_31_tmp_2, batch_norm_11_tmp_3_dim, conv2d_68_w_0_dim, batch_norm_31_tmp_2_dim, 1, 0, 1, 1, 0);
	free(batch_norm_11_tmp_3);

	float* p_177[] = { batch_norm_14_tmp_3, batch_norm_30_tmp_2, batch_norm_31_tmp_2, };
	uint16_t* p_177_dim[] = { batch_norm_14_tmp_3_dim, batch_norm_30_tmp_2_dim, batch_norm_31_tmp_2_dim, };
	batch_norm_11_tmp_3 = (float*)calloc(393216, sizeof(float));
	autox_concat(p_177, batch_norm_11_tmp_3, p_177_dim, batch_norm_11_tmp_3_dim, 1, 3, 4);
	free(batch_norm_14_tmp_3);
	free(batch_norm_30_tmp_2);
	free(batch_norm_31_tmp_2);

	batch_norm_34_tmp_2 = (float*)calloc(2048, sizeof(float));
	autox_conv2d(batch_norm_11_tmp_3, (float*)((int8_t*)weights) + 680292, (float*)((int8_t*)weights) + 680676, batch_norm_34_tmp_2, batch_norm_11_tmp_3_dim, conv2d_73_w_0_dim, batch_norm_34_tmp_2_dim, 384, 1, 1, 1, 2);
	free(batch_norm_11_tmp_3);

	batch_norm_11_tmp_3 = (float*)calloc(393216, sizeof(float));
	autox_conv2d(batch_norm_34_tmp_2, (float*)((int8_t*)weights) + 684132, (float*)((int8_t*)weights) + 684260, batch_norm_11_tmp_3, batch_norm_34_tmp_2_dim, conv2d_74_w_0_dim, batch_norm_11_tmp_3_dim, 1, 0, 1, 1, 0);
	free(batch_norm_34_tmp_2);

	mean_0_tmp_0 = (float*)calloc(1024, sizeof(float));
	autox_reduce_mean(batch_norm_14_tmp_3, mean_0_tmp_0, batch_norm_14_tmp_3_dim, mean_0_tmp_0_dim);
	free(batch_norm_14_tmp_3);

	batch_norm_34_tmp_2 = (float*)calloc(2048, sizeof(float));
	// autox_linear_interp_v2((float*)((int8_t*)weights) + 733412, batch_norm_34_tmp_2, (float*)((int8_t*)weights) + 733412_dim, batch_norm_34_tmp_2_dim, 0);

	float* tmp_22 = (float*)calloc(768, sizeof(float));
	autox_elementwise_add(mean_0_tmp_0, batch_norm_34_tmp_2, tmp_22, mean_0_tmp_0_dim, batch_norm_34_tmp_2_dim, tmp_22_dim, -1, 3, 4, 3);
	free(mean_0_tmp_0);
	free(batch_norm_34_tmp_2);

	tmp_24 = (float*)calloc(256, sizeof(float));
	uint16_t axis_183[] = { 0, 1, 3, 2 };
	autox_transpose(tmp_22, tmp_24, reshape2_16_tmp_0_dim, tmp_24_dim, axis_183, 4);
	free(tmp_22);

	mean_0_tmp_0 = (float*)calloc(1024, sizeof(float));
	autox_reduce_mean(batch_norm_30_tmp_2, mean_0_tmp_0, batch_norm_30_tmp_2_dim, mean_0_tmp_0_dim);
	free(batch_norm_30_tmp_2);

	batch_norm_34_tmp_2 = (float*)calloc(2048, sizeof(float));
	// autox_linear_interp_v2((float*)((int8_t*)weights) + 734948, batch_norm_34_tmp_2, (float*)((int8_t*)weights) + 734948_dim, batch_norm_34_tmp_2_dim, 0);

	float* tmp_23 = (float*)calloc(768, sizeof(float));
	autox_elementwise_add(mean_0_tmp_0, batch_norm_34_tmp_2, tmp_23, mean_0_tmp_0_dim, batch_norm_34_tmp_2_dim, tmp_23_dim, -1, 3, 4, 3);
	free(mean_0_tmp_0);
	free(batch_norm_34_tmp_2);

	float* mean_14_tmp_0 = (float*)calloc(1536, sizeof(float));
	autox_reduce_mean(batch_norm_31_tmp_2, mean_14_tmp_0, batch_norm_31_tmp_2_dim, mean_14_tmp_0_dim);
	free(batch_norm_31_tmp_2);

	float* transpose_13_tmp_0 = (float*)calloc(1536, sizeof(float));
	uint16_t axis_188[] = { 0, 1, 3, 2 };
	autox_transpose(mean_14_tmp_0, transpose_13_tmp_0, reshape2_18_tmp_0_dim, transpose_13_tmp_0_dim, axis_188, 4);
	free(mean_14_tmp_0);

	batch_norm_34_tmp_2 = (float*)calloc(2048, sizeof(float));
	autox_matmul(tmp_24, tmp_23, batch_norm_34_tmp_2, tmp_24_dim, reshape2_17_tmp_0_dim, batch_norm_34_tmp_2_dim, 0, 0, 4, 4, 4);
	free(tmp_24);
	free(tmp_23);

	autox_scale(batch_norm_34_tmp_2, batch_norm_34_tmp_2_dim, 4, 0, 1, 0.20412415);

	autox_softmax(tmp_24, tmp_24_dim[0], tmp_24_dim[1]);

	batch_norm_34_tmp_2 = (float*)calloc(2048, sizeof(float));
	autox_matmul(mean_0_tmp_0, transpose_13_tmp_0, batch_norm_34_tmp_2, mean_0_tmp_0_dim, transpose_13_tmp_0_dim, batch_norm_34_tmp_2_dim, 0, 0, 3, 4, 4);
	free(mean_0_tmp_0);
	free(transpose_13_tmp_0);

	float* transpose_14_tmp_0 = (float*)calloc(1536, sizeof(float));
	uint16_t axis_193[] = { 0, 1, 3, 2 };
	autox_transpose(batch_norm_34_tmp_2, transpose_14_tmp_0, batch_norm_34_tmp_2_dim, transpose_14_tmp_0_dim, axis_193, 4);
	free(batch_norm_34_tmp_2);

	mean_0_tmp_0 = (float*)calloc(1024, sizeof(float));
	autox_relu6(transpose_14_tmp_0, reshape2_19_tmp_0_dim, 4, 6.0);

	batch_norm_34_tmp_2 = (float*)calloc(2048, sizeof(float));
	autox_conv2d(mean_0_tmp_0, (float*)((int8_t*)weights) + 736484, (float*)((int8_t*)weights) + 736676, batch_norm_34_tmp_2, mean_0_tmp_0_dim, conv2d_71_w_0_dim, batch_norm_34_tmp_2_dim, 1, 0, 1, 1, 0);
	free(mean_0_tmp_0);

	mean_0_tmp_0 = (float*)calloc(1024, sizeof(float));
	autox_reduce_mean(batch_norm_14_tmp_3, mean_0_tmp_0, batch_norm_14_tmp_3_dim, mean_0_tmp_0_dim);
	free(batch_norm_14_tmp_3);

	batch_norm_14_tmp_3 = (float*)calloc(393216, sizeof(float));
	// autox_linear_interp_v2((float*)((int8_t*)weights) + 773540, batch_norm_14_tmp_3, (float*)((int8_t*)weights) + 773540_dim, batch_norm_14_tmp_3_dim, 0);

	float* tmp_25 = (float*)calloc(768, sizeof(float));
	autox_elementwise_add(mean_0_tmp_0, batch_norm_14_tmp_3, tmp_25, mean_0_tmp_0_dim, batch_norm_14_tmp_3_dim, tmp_25_dim, -1, 3, 4, 3);
	free(mean_0_tmp_0);
	free(batch_norm_14_tmp_3);

	tmp_24 = (float*)calloc(256, sizeof(float));
	uint16_t axis_199[] = { 0, 1, 3, 2 };
	autox_transpose(tmp_25, tmp_24, reshape2_20_tmp_0_dim, tmp_24_dim, axis_199, 4);
	free(tmp_25);

	mean_0_tmp_0 = (float*)calloc(1024, sizeof(float));
	autox_reduce_mean(batch_norm_30_tmp_2, mean_0_tmp_0, batch_norm_30_tmp_2_dim, mean_0_tmp_0_dim);
	free(batch_norm_30_tmp_2);

	batch_norm_14_tmp_3 = (float*)calloc(393216, sizeof(float));
	// autox_linear_interp_v2((float*)((int8_t*)weights) + 775076, batch_norm_14_tmp_3, (float*)((int8_t*)weights) + 775076_dim, batch_norm_14_tmp_3_dim, 0);

	float* tmp_26 = (float*)calloc(768, sizeof(float));
	autox_elementwise_add(mean_0_tmp_0, batch_norm_14_tmp_3, tmp_26, mean_0_tmp_0_dim, batch_norm_14_tmp_3_dim, tmp_26_dim, -1, 3, 4, 3);
	free(mean_0_tmp_0);
	free(batch_norm_14_tmp_3);

	float* mean_17_tmp_0 = (float*)calloc(1536, sizeof(float));
	autox_reduce_mean(batch_norm_31_tmp_2, mean_17_tmp_0, batch_norm_31_tmp_2_dim, mean_17_tmp_0_dim);
	free(batch_norm_31_tmp_2);

	transpose_13_tmp_0 = (float*)calloc(1536, sizeof(float));
	uint16_t axis_204[] = { 0, 1, 3, 2 };
	autox_transpose(mean_17_tmp_0, transpose_13_tmp_0, reshape2_22_tmp_0_dim, transpose_13_tmp_0_dim, axis_204, 4);
	free(mean_17_tmp_0);

	batch_norm_14_tmp_3 = (float*)calloc(393216, sizeof(float));
	autox_matmul(tmp_24, tmp_26, batch_norm_14_tmp_3, tmp_24_dim, reshape2_21_tmp_0_dim, batch_norm_14_tmp_3_dim, 0, 0, 4, 4, 4);
	free(tmp_24);
	free(tmp_26);

	autox_scale(batch_norm_14_tmp_3, batch_norm_14_tmp_3_dim, 4, 0, 1, 0.20412415);

	autox_softmax(mean_0_tmp_0, mean_0_tmp_0_dim[0], mean_0_tmp_0_dim[1]);

	batch_norm_14_tmp_3 = (float*)calloc(393216, sizeof(float));
	autox_matmul(batch_norm_30_tmp_2, transpose_13_tmp_0, batch_norm_14_tmp_3, batch_norm_30_tmp_2_dim, transpose_13_tmp_0_dim, batch_norm_14_tmp_3_dim, 0, 0, 4, 4, 4);
	free(batch_norm_30_tmp_2);
	free(transpose_13_tmp_0);

	float* transpose_17_tmp_0 = (float*)calloc(1536, sizeof(float));
	uint16_t axis_209[] = { 0, 1, 3, 2 };
	autox_transpose(batch_norm_14_tmp_3, transpose_17_tmp_0, batch_norm_14_tmp_3_dim, transpose_17_tmp_0_dim, axis_209, 4);
	free(batch_norm_14_tmp_3);

	batch_norm_30_tmp_2 = (float*)calloc(16384, sizeof(float));
	autox_relu6(transpose_17_tmp_0, reshape2_23_tmp_0_dim, 4, 6.0);

	batch_norm_14_tmp_3 = (float*)calloc(393216, sizeof(float));
	autox_conv2d(batch_norm_30_tmp_2, (float*)((int8_t*)weights) + 776612, (float*)((int8_t*)weights) + 776804, batch_norm_14_tmp_3, batch_norm_30_tmp_2_dim, conv2d_72_w_0_dim, batch_norm_14_tmp_3_dim, 1, 0, 1, 1, 0);
	free(batch_norm_30_tmp_2);

	batch_norm_30_tmp_2 = (float*)calloc(16384, sizeof(float));
	autox_elementwise_add(batch_norm_34_tmp_2, batch_norm_14_tmp_3, batch_norm_30_tmp_2, batch_norm_34_tmp_2_dim, batch_norm_14_tmp_3_dim, batch_norm_30_tmp_2_dim, -1, 4, 4, 4);
	free(batch_norm_34_tmp_2);
	free(batch_norm_14_tmp_3);

	batch_norm_14_tmp_3 = (float*)calloc(393216, sizeof(float));
	autox_elementwise_add(batch_norm_31_tmp_2, batch_norm_30_tmp_2, batch_norm_14_tmp_3, batch_norm_31_tmp_2_dim, batch_norm_30_tmp_2_dim, batch_norm_14_tmp_3_dim, -1, 4, 4, 4);
	free(batch_norm_31_tmp_2);
	free(batch_norm_30_tmp_2);

	batch_norm_30_tmp_2 = (float*)calloc(16384, sizeof(float));
	autox_relu6(batch_norm_14_tmp_3, batch_norm_14_tmp_3_dim, 4, 6.0);

	batch_norm_14_tmp_3 = (float*)calloc(393216, sizeof(float));
	autox_conv2d(batch_norm_30_tmp_2, (float*)((int8_t*)weights) + 813668, (float*)((int8_t*)weights) + 813796, batch_norm_14_tmp_3, batch_norm_30_tmp_2_dim, conv2d_70_w_0_dim, batch_norm_14_tmp_3_dim, 1, 0, 1, 1, 2);
	free(batch_norm_30_tmp_2);

	autox_scale(batch_norm_14_tmp_3, batch_norm_14_tmp_3_dim, 4, 0, 1, 0.16666667);

	batch_norm_14_tmp_3 = (float*)calloc(393216, sizeof(float));
	autox_elementwise_mul(batch_norm_30_tmp_2, batch_norm_11_tmp_3, batch_norm_14_tmp_3, batch_norm_30_tmp_2_dim, batch_norm_11_tmp_3_dim, batch_norm_14_tmp_3_dim, -1, 4, 4, 4);
	free(batch_norm_30_tmp_2);
	free(batch_norm_11_tmp_3);

	batch_norm_11_tmp_3 = (float*)calloc(393216, sizeof(float));
	// autox_bilinear_interp(batch_norm_14_tmp_3, batch_norm_11_tmp_3, batch_norm_14_tmp_3_dim, batch_norm_11_tmp_3_dim, , 0);
	free(batch_norm_14_tmp_3);

	batch_norm_30_tmp_2 = (float*)calloc(16384, sizeof(float));
	autox_elementwise_add(elementwise_add_2, batch_norm_11_tmp_3, batch_norm_30_tmp_2, elementwise_add_2_dim, batch_norm_11_tmp_3_dim, batch_norm_30_tmp_2_dim, -1, 4, 4, 4);
	free(elementwise_add_2);
	free(batch_norm_11_tmp_3);

	batch_norm_11_tmp_3 = (float*)calloc(393216, sizeof(float));
	autox_conv2d(batch_norm_30_tmp_2, (float*)((int8_t*)weights) + 838372, (float*)((int8_t*)weights) + 838884, batch_norm_11_tmp_3, batch_norm_30_tmp_2_dim, conv2d_75_w_0_dim, batch_norm_11_tmp_3_dim, 1, 0, 1, 1, 0);
	free(batch_norm_30_tmp_2);

	batch_norm_14_tmp_3 = (float*)calloc(393216, sizeof(float));
	autox_conv2d(batch_norm_11_tmp_3, (float*)((int8_t*)weights) + 904420, (float*)((int8_t*)weights) + 904932, batch_norm_14_tmp_3, batch_norm_11_tmp_3_dim, conv2d_76_w_0_dim, batch_norm_14_tmp_3_dim, 512, 1, 1, 1, 2);
	free(batch_norm_11_tmp_3);

	batch_norm_11_tmp_3 = (float*)calloc(393216, sizeof(float));
	autox_conv2d(batch_norm_14_tmp_3, (float*)((int8_t*)weights) + 909540, (float*)((int8_t*)weights) + 909668, batch_norm_11_tmp_3, batch_norm_14_tmp_3_dim, conv2d_77_w_0_dim, batch_norm_11_tmp_3_dim, 1, 0, 1, 1, 0);
	free(batch_norm_14_tmp_3);

	mean_0_tmp_0 = (float*)calloc(1024, sizeof(float));
	autox_elementwise_add(batch_norm_30_tmp_2, batch_norm_11_tmp_3, mean_0_tmp_0, batch_norm_30_tmp_2_dim, batch_norm_11_tmp_3_dim, mean_0_tmp_0_dim, -1, 4, 4, 3);
	free(batch_norm_30_tmp_2);
	free(batch_norm_11_tmp_3);

	batch_norm_11_tmp_3 = (float*)calloc(393216, sizeof(float));
	autox_conv2d(mean_0_tmp_0, (float*)((int8_t*)weights) + 975204, (float*)((int8_t*)weights) + 975332, batch_norm_11_tmp_3, mean_0_tmp_0_dim, conv2d_81_w_0_dim, batch_norm_11_tmp_3_dim, 128, 1, 2, 1, 0);
	free(mean_0_tmp_0);

	batch_norm_14_tmp_3 = (float*)calloc(393216, sizeof(float));
	autox_conv2d(batch_norm_11_tmp_3, (float*)((int8_t*)weights) + 976484, (float*)((int8_t*)weights) + 976580, batch_norm_14_tmp_3, batch_norm_11_tmp_3_dim, conv2d_78_w_0_dim, batch_norm_14_tmp_3_dim, 1, 0, 1, 1, 0);
	free(batch_norm_11_tmp_3);

	batch_norm_30_tmp_2 = (float*)calloc(16384, sizeof(float));
	autox_conv2d(batch_norm_11_tmp_3, (float*)((int8_t*)weights) + 988868, (float*)((int8_t*)weights) + 988964, batch_norm_30_tmp_2, batch_norm_11_tmp_3_dim, conv2d_79_w_0_dim, batch_norm_30_tmp_2_dim, 1, 0, 1, 1, 0);
	free(batch_norm_11_tmp_3);

	batch_norm_31_tmp_2 = (float*)calloc(32768, sizeof(float));
	autox_conv2d(batch_norm_11_tmp_3, (float*)((int8_t*)weights) + 1001252, (float*)((int8_t*)weights) + 1001444, batch_norm_31_tmp_2, batch_norm_11_tmp_3_dim, conv2d_80_w_0_dim, batch_norm_31_tmp_2_dim, 1, 0, 1, 1, 0);
	free(batch_norm_11_tmp_3);

	float* p_228[] = { batch_norm_14_tmp_3, batch_norm_30_tmp_2, batch_norm_31_tmp_2, };
	uint16_t* p_228_dim[] = { batch_norm_14_tmp_3_dim, batch_norm_30_tmp_2_dim, batch_norm_31_tmp_2_dim, };
	batch_norm_11_tmp_3 = (float*)calloc(393216, sizeof(float));
	autox_concat(p_228, batch_norm_11_tmp_3, p_228_dim, batch_norm_11_tmp_3_dim, 1, 3, 4);
	free(batch_norm_14_tmp_3);
	free(batch_norm_30_tmp_2);
	free(batch_norm_31_tmp_2);

	batch_norm_34_tmp_2 = (float*)calloc(2048, sizeof(float));
	autox_conv2d(batch_norm_11_tmp_3, (float*)((int8_t*)weights) + 1026020, (float*)((int8_t*)weights) + 1026404, batch_norm_34_tmp_2, batch_norm_11_tmp_3_dim, conv2d_85_w_0_dim, batch_norm_34_tmp_2_dim, 384, 1, 1, 1, 2);
	free(batch_norm_11_tmp_3);

	batch_norm_11_tmp_3 = (float*)calloc(393216, sizeof(float));
	autox_conv2d(batch_norm_34_tmp_2, (float*)((int8_t*)weights) + 1029860, (float*)((int8_t*)weights) + 1029988, batch_norm_11_tmp_3, batch_norm_34_tmp_2_dim, conv2d_86_w_0_dim, batch_norm_11_tmp_3_dim, 1, 0, 1, 1, 0);
	free(batch_norm_34_tmp_2);

	elementwise_add_2 = (float*)calloc(65536, sizeof(float));
	autox_reduce_mean(batch_norm_14_tmp_3, elementwise_add_2, batch_norm_14_tmp_3_dim, elementwise_add_2_dim);
	free(batch_norm_14_tmp_3);

	batch_norm_34_tmp_2 = (float*)calloc(2048, sizeof(float));
	// autox_linear_interp_v2((float*)((int8_t*)weights) + 1079140, batch_norm_34_tmp_2, (float*)((int8_t*)weights) + 1079140_dim, batch_norm_34_tmp_2_dim, 0);

	float* tmp_33 = (float*)calloc(768, sizeof(float));
	autox_elementwise_add(elementwise_add_2, batch_norm_34_tmp_2, tmp_33, elementwise_add_2_dim, batch_norm_34_tmp_2_dim, tmp_33_dim, -1, 4, 4, 3);
	free(elementwise_add_2);
	free(batch_norm_34_tmp_2);

	tmp_24 = (float*)calloc(256, sizeof(float));
	uint16_t axis_234[] = { 0, 1, 3, 2 };
	autox_transpose(tmp_33, tmp_24, reshape2_24_tmp_0_dim, tmp_24_dim, axis_234, 4);
	free(tmp_33);

	elementwise_add_2 = (float*)calloc(65536, sizeof(float));
	autox_reduce_mean(batch_norm_30_tmp_2, elementwise_add_2, batch_norm_30_tmp_2_dim, elementwise_add_2_dim);
	free(batch_norm_30_tmp_2);

	batch_norm_34_tmp_2 = (float*)calloc(2048, sizeof(float));
	// autox_linear_interp_v2((float*)((int8_t*)weights) + 1080676, batch_norm_34_tmp_2, (float*)((int8_t*)weights) + 1080676_dim, batch_norm_34_tmp_2_dim, 0);

	float* tmp_34 = (float*)calloc(768, sizeof(float));
	autox_elementwise_add(elementwise_add_2, batch_norm_34_tmp_2, tmp_34, elementwise_add_2_dim, batch_norm_34_tmp_2_dim, tmp_34_dim, -1, 4, 4, 3);
	free(elementwise_add_2);
	free(batch_norm_34_tmp_2);

	float* mean_20_tmp_0 = (float*)calloc(1536, sizeof(float));
	autox_reduce_mean(batch_norm_31_tmp_2, mean_20_tmp_0, batch_norm_31_tmp_2_dim, mean_20_tmp_0_dim);
	free(batch_norm_31_tmp_2);

	transpose_13_tmp_0 = (float*)calloc(1536, sizeof(float));
	uint16_t axis_239[] = { 0, 1, 3, 2 };
	autox_transpose(mean_20_tmp_0, transpose_13_tmp_0, reshape2_26_tmp_0_dim, transpose_13_tmp_0_dim, axis_239, 4);
	free(mean_20_tmp_0);

	batch_norm_34_tmp_2 = (float*)calloc(2048, sizeof(float));
	autox_matmul(tmp_24, tmp_34, batch_norm_34_tmp_2, tmp_24_dim, reshape2_25_tmp_0_dim, batch_norm_34_tmp_2_dim, 0, 0, 4, 4, 4);
	free(tmp_24);
	free(tmp_34);

	autox_scale(batch_norm_34_tmp_2, batch_norm_34_tmp_2_dim, 4, 0, 1, 0.20412415);

	autox_softmax(tmp_24, tmp_24_dim[0], tmp_24_dim[1]);

	batch_norm_34_tmp_2 = (float*)calloc(2048, sizeof(float));
	autox_matmul(elementwise_add_2, transpose_13_tmp_0, batch_norm_34_tmp_2, elementwise_add_2_dim, transpose_13_tmp_0_dim, batch_norm_34_tmp_2_dim, 0, 0, 4, 4, 4);
	free(elementwise_add_2);
	free(transpose_13_tmp_0);

	float* transpose_20_tmp_0 = (float*)calloc(1536, sizeof(float));
	uint16_t axis_244[] = { 0, 1, 3, 2 };
	autox_transpose(batch_norm_34_tmp_2, transpose_20_tmp_0, batch_norm_34_tmp_2_dim, transpose_20_tmp_0_dim, axis_244, 4);
	free(batch_norm_34_tmp_2);

	elementwise_add_2 = (float*)calloc(65536, sizeof(float));
	autox_relu6(transpose_20_tmp_0, reshape2_27_tmp_0_dim, 4, 6.0);

	batch_norm_34_tmp_2 = (float*)calloc(2048, sizeof(float));
	autox_conv2d(elementwise_add_2, (float*)((int8_t*)weights) + 1082212, (float*)((int8_t*)weights) + 1082404, batch_norm_34_tmp_2, elementwise_add_2_dim, conv2d_83_w_0_dim, batch_norm_34_tmp_2_dim, 1, 0, 1, 1, 0);
	free(elementwise_add_2);

	elementwise_add_2 = (float*)calloc(65536, sizeof(float));
	autox_reduce_mean(batch_norm_14_tmp_3, elementwise_add_2, batch_norm_14_tmp_3_dim, elementwise_add_2_dim);
	free(batch_norm_14_tmp_3);

	batch_norm_14_tmp_3 = (float*)calloc(393216, sizeof(float));
	// autox_linear_interp_v2((float*)((int8_t*)weights) + 1119268, batch_norm_14_tmp_3, (float*)((int8_t*)weights) + 1119268_dim, batch_norm_14_tmp_3_dim, 0);

	float* tmp_36 = (float*)calloc(768, sizeof(float));
	autox_elementwise_add(elementwise_add_2, batch_norm_14_tmp_3, tmp_36, elementwise_add_2_dim, batch_norm_14_tmp_3_dim, tmp_36_dim, -1, 4, 4, 3);
	free(elementwise_add_2);
	free(batch_norm_14_tmp_3);

	tmp_24 = (float*)calloc(256, sizeof(float));
	uint16_t axis_250[] = { 0, 1, 3, 2 };
	autox_transpose(tmp_36, tmp_24, reshape2_28_tmp_0_dim, tmp_24_dim, axis_250, 4);
	free(tmp_36);

	elementwise_add_2 = (float*)calloc(65536, sizeof(float));
	autox_reduce_mean(batch_norm_30_tmp_2, elementwise_add_2, batch_norm_30_tmp_2_dim, elementwise_add_2_dim);
	free(batch_norm_30_tmp_2);

	batch_norm_14_tmp_3 = (float*)calloc(393216, sizeof(float));
	// autox_linear_interp_v2((float*)((int8_t*)weights) + 1120804, batch_norm_14_tmp_3, (float*)((int8_t*)weights) + 1120804_dim, batch_norm_14_tmp_3_dim, 0);

	float* tmp_37 = (float*)calloc(768, sizeof(float));
	autox_elementwise_add(elementwise_add_2, batch_norm_14_tmp_3, tmp_37, elementwise_add_2_dim, batch_norm_14_tmp_3_dim, tmp_37_dim, -1, 4, 4, 3);
	free(elementwise_add_2);
	free(batch_norm_14_tmp_3);

	float* mean_23_tmp_0 = (float*)calloc(1536, sizeof(float));
	autox_reduce_mean(batch_norm_31_tmp_2, mean_23_tmp_0, batch_norm_31_tmp_2_dim, mean_23_tmp_0_dim);
	free(batch_norm_31_tmp_2);

	transpose_13_tmp_0 = (float*)calloc(1536, sizeof(float));
	uint16_t axis_255[] = { 0, 1, 3, 2 };
	autox_transpose(mean_23_tmp_0, transpose_13_tmp_0, reshape2_30_tmp_0_dim, transpose_13_tmp_0_dim, axis_255, 4);
	free(mean_23_tmp_0);

	batch_norm_14_tmp_3 = (float*)calloc(393216, sizeof(float));
	autox_matmul(tmp_24, tmp_37, batch_norm_14_tmp_3, tmp_24_dim, reshape2_29_tmp_0_dim, batch_norm_14_tmp_3_dim, 0, 0, 4, 4, 4);
	free(tmp_24);
	free(tmp_37);

	autox_scale(batch_norm_14_tmp_3, batch_norm_14_tmp_3_dim, 4, 0, 1, 0.20412415);

	autox_softmax(elementwise_add_2, elementwise_add_2_dim[0], elementwise_add_2_dim[1]);

	batch_norm_14_tmp_3 = (float*)calloc(393216, sizeof(float));
	autox_matmul(batch_norm_30_tmp_2, transpose_13_tmp_0, batch_norm_14_tmp_3, batch_norm_30_tmp_2_dim, transpose_13_tmp_0_dim, batch_norm_14_tmp_3_dim, 0, 0, 4, 4, 4);
	free(batch_norm_30_tmp_2);
	free(transpose_13_tmp_0);

	float* transpose_23_tmp_0 = (float*)calloc(1536, sizeof(float));
	uint16_t axis_260[] = { 0, 1, 3, 2 };
	autox_transpose(batch_norm_14_tmp_3, transpose_23_tmp_0, batch_norm_14_tmp_3_dim, transpose_23_tmp_0_dim, axis_260, 4);
	free(batch_norm_14_tmp_3);

	batch_norm_30_tmp_2 = (float*)calloc(16384, sizeof(float));
	autox_relu6(transpose_23_tmp_0, reshape2_31_tmp_0_dim, 4, 6.0);

	batch_norm_14_tmp_3 = (float*)calloc(393216, sizeof(float));
	autox_conv2d(batch_norm_30_tmp_2, (float*)((int8_t*)weights) + 1122340, (float*)((int8_t*)weights) + 1122532, batch_norm_14_tmp_3, batch_norm_30_tmp_2_dim, conv2d_84_w_0_dim, batch_norm_14_tmp_3_dim, 1, 0, 1, 1, 0);
	free(batch_norm_30_tmp_2);

	batch_norm_30_tmp_2 = (float*)calloc(16384, sizeof(float));
	autox_elementwise_add(batch_norm_34_tmp_2, batch_norm_14_tmp_3, batch_norm_30_tmp_2, batch_norm_34_tmp_2_dim, batch_norm_14_tmp_3_dim, batch_norm_30_tmp_2_dim, -1, 4, 4, 4);
	free(batch_norm_34_tmp_2);
	free(batch_norm_14_tmp_3);

	batch_norm_14_tmp_3 = (float*)calloc(393216, sizeof(float));
	autox_elementwise_add(batch_norm_31_tmp_2, batch_norm_30_tmp_2, batch_norm_14_tmp_3, batch_norm_31_tmp_2_dim, batch_norm_30_tmp_2_dim, batch_norm_14_tmp_3_dim, -1, 4, 4, 4);
	free(batch_norm_31_tmp_2);
	free(batch_norm_30_tmp_2);

	batch_norm_30_tmp_2 = (float*)calloc(16384, sizeof(float));
	autox_relu6(batch_norm_14_tmp_3, batch_norm_14_tmp_3_dim, 4, 6.0);

	batch_norm_14_tmp_3 = (float*)calloc(393216, sizeof(float));
	autox_conv2d(batch_norm_30_tmp_2, (float*)((int8_t*)weights) + 1159396, (float*)((int8_t*)weights) + 1159524, batch_norm_14_tmp_3, batch_norm_30_tmp_2_dim, conv2d_82_w_0_dim, batch_norm_14_tmp_3_dim, 1, 0, 1, 1, 2);
	free(batch_norm_30_tmp_2);

	autox_scale(batch_norm_14_tmp_3, batch_norm_14_tmp_3_dim, 4, 0, 1, 0.16666667);

	batch_norm_14_tmp_3 = (float*)calloc(393216, sizeof(float));
	autox_elementwise_mul(batch_norm_30_tmp_2, batch_norm_11_tmp_3, batch_norm_14_tmp_3, batch_norm_30_tmp_2_dim, batch_norm_11_tmp_3_dim, batch_norm_14_tmp_3_dim, -1, 4, 4, 4);
	free(batch_norm_30_tmp_2);
	free(batch_norm_11_tmp_3);

	batch_norm_11_tmp_3 = (float*)calloc(393216, sizeof(float));
	autox_bilinear_interp(batch_norm_14_tmp_3, batch_norm_11_tmp_3, batch_norm_14_tmp_3_dim, batch_norm_11_tmp_3_dim, 0, 0, 0);
	free(batch_norm_14_tmp_3);

	batch_norm_30_tmp_2 = (float*)calloc(16384, sizeof(float));
	autox_elementwise_add(mean_0_tmp_0, batch_norm_11_tmp_3, batch_norm_30_tmp_2, mean_0_tmp_0_dim, batch_norm_11_tmp_3_dim, batch_norm_30_tmp_2_dim, -1, 3, 4, 4);
	free(mean_0_tmp_0);
	free(batch_norm_11_tmp_3);

	batch_norm_11_tmp_3 = (float*)calloc(393216, sizeof(float));
	autox_conv2d(batch_norm_30_tmp_2, (float*)((int8_t*)weights) + 1184100, (float*)((int8_t*)weights) + 1184612, batch_norm_11_tmp_3, batch_norm_30_tmp_2_dim, conv2d_87_w_0_dim, batch_norm_11_tmp_3_dim, 1, 0, 1, 1, 0);
	free(batch_norm_30_tmp_2);

	batch_norm_14_tmp_3 = (float*)calloc(393216, sizeof(float));
	autox_conv2d(batch_norm_11_tmp_3, (float*)((int8_t*)weights) + 1250148, (float*)((int8_t*)weights) + 1250660, batch_norm_14_tmp_3, batch_norm_11_tmp_3_dim, conv2d_88_w_0_dim, batch_norm_14_tmp_3_dim, 512, 1, 1, 1, 2);
	free(batch_norm_11_tmp_3);

	batch_norm_11_tmp_3 = (float*)calloc(393216, sizeof(float));
	autox_conv2d(batch_norm_14_tmp_3, (float*)((int8_t*)weights) + 1255268, (float*)((int8_t*)weights) + 1255396, batch_norm_11_tmp_3, batch_norm_14_tmp_3_dim, conv2d_89_w_0_dim, batch_norm_11_tmp_3_dim, 1, 0, 1, 1, 0);
	free(batch_norm_14_tmp_3);

	mean_0_tmp_0 = (float*)calloc(1024, sizeof(float));
	autox_elementwise_add(batch_norm_30_tmp_2, batch_norm_11_tmp_3, mean_0_tmp_0, batch_norm_30_tmp_2_dim, batch_norm_11_tmp_3_dim, mean_0_tmp_0_dim, -1, 4, 4, 3);
	free(batch_norm_30_tmp_2);
	free(batch_norm_11_tmp_3);

	batch_norm_14_tmp_3 = (float*)calloc(393216, sizeof(float));
	autox_bilinear_interp(elementwise_add_1, batch_norm_14_tmp_3, elementwise_add_1_dim, batch_norm_14_tmp_3_dim, 0, 0,0);
	free(elementwise_add_1);

	batch_norm_11_tmp_3 = (float*)calloc(393216, sizeof(float));
	autox_conv2d(batch_norm_14_tmp_3, (float*)((int8_t*)weights) + 1320932, (float*)((int8_t*)weights) + 1321188, batch_norm_11_tmp_3, batch_norm_14_tmp_3_dim, conv2d_91_w_0_dim, batch_norm_11_tmp_3_dim, 1, 0, 1, 1, 0);
	free(batch_norm_14_tmp_3);

	autox_sigmoid(batch_norm_11_tmp_3, batch_norm_11_tmp_3_dim, 4);

	batch_norm_11_tmp_3 = (float*)calloc(393216, sizeof(float));
	autox_conv2d(batch_norm_14_tmp_3, (float*)((int8_t*)weights) + 1329380, (float*)((int8_t*)weights) + 1329636, batch_norm_11_tmp_3, batch_norm_14_tmp_3_dim, conv2d_90_w_0_dim, batch_norm_11_tmp_3_dim, 1, 0, 1, 1, 0);
	free(batch_norm_14_tmp_3);

	batch_norm_30_tmp_2 = (float*)calloc(16384, sizeof(float));
	autox_bilinear_interp(tmp_13, batch_norm_30_tmp_2, tmp_13_dim, batch_norm_30_tmp_2_dim, 0, 0,0);
	free(tmp_13);

	batch_norm_14_tmp_3 = (float*)calloc(393216, sizeof(float));
	autox_conv2d(batch_norm_30_tmp_2, (float*)((int8_t*)weights) + 1337828, (float*)((int8_t*)weights) + 1338084, batch_norm_14_tmp_3, batch_norm_30_tmp_2_dim, conv2d_93_w_0_dim, batch_norm_14_tmp_3_dim, 1, 0, 1, 1, 0);
	free(batch_norm_30_tmp_2);

	autox_sigmoid(batch_norm_14_tmp_3, batch_norm_14_tmp_3_dim, 4);

	batch_norm_14_tmp_3 = (float*)calloc(393216, sizeof(float));
	autox_conv2d(batch_norm_30_tmp_2, (float*)((int8_t*)weights) + 1354468, (float*)((int8_t*)weights) + 1354724, batch_norm_14_tmp_3, batch_norm_30_tmp_2_dim, conv2d_92_w_0_dim, batch_norm_14_tmp_3_dim, 1, 0, 1, 1, 0);
	free(batch_norm_30_tmp_2);

	batch_norm_30_tmp_2 = (float*)calloc(16384, sizeof(float));
	autox_conv2d(mean_0_tmp_0, (float*)((int8_t*)weights) + 1371108, (float*)((int8_t*)weights) + 1371364, batch_norm_30_tmp_2, mean_0_tmp_0_dim, conv2d_95_w_0_dim, batch_norm_30_tmp_2_dim, 1, 0, 1, 1, 0);
	free(mean_0_tmp_0);

	autox_sigmoid(batch_norm_30_tmp_2, batch_norm_30_tmp_2_dim, 4);

	batch_norm_31_tmp_2 = (float*)calloc(32768, sizeof(float));
	autox_bilinear_interp(batch_norm_34_tmp_2, batch_norm_31_tmp_2, batch_norm_34_tmp_2_dim, batch_norm_31_tmp_2_dim, 0, 0,0);
	free(batch_norm_34_tmp_2);

	batch_norm_30_tmp_2 = (float*)calloc(16384, sizeof(float));
	autox_conv2d(mean_0_tmp_0, (float*)((int8_t*)weights) + 1404132, (float*)((int8_t*)weights) + 1404388, batch_norm_30_tmp_2, mean_0_tmp_0_dim, conv2d_94_w_0_dim, batch_norm_30_tmp_2_dim, 1, 0, 1, 1, 0);
	free(mean_0_tmp_0);

	batch_norm_34_tmp_2 = (float*)calloc(2048, sizeof(float));
	autox_bilinear_interp(batch_norm_30_tmp_2, batch_norm_34_tmp_2, batch_norm_30_tmp_2_dim, batch_norm_34_tmp_2_dim, 0, 0,0);
	free(batch_norm_30_tmp_2);

	batch_norm_30_tmp_2 = (float*)calloc(16384, sizeof(float));
	autox_elementwise_mul(elementwise_add_1, elementwise_add_2, batch_norm_30_tmp_2, elementwise_add_1_dim, elementwise_add_2_dim, batch_norm_30_tmp_2_dim, -1, 4, 4, 4);
	free(elementwise_add_1);
	free(elementwise_add_2);

	elementwise_add_1 = (float*)calloc(131072, sizeof(float));
	autox_elementwise_mul(batch_norm_30_tmp_2, batch_norm_31_tmp_2, elementwise_add_1, batch_norm_30_tmp_2_dim, batch_norm_31_tmp_2_dim, elementwise_add_1_dim, -1, 4, 4, 4);
	free(batch_norm_30_tmp_2);
	free(batch_norm_31_tmp_2);

	batch_norm_30_tmp_2 = (float*)calloc(16384, sizeof(float));
	autox_elementwise_add(batch_norm_11_tmp_3, batch_norm_14_tmp_3, batch_norm_30_tmp_2, batch_norm_11_tmp_3_dim, batch_norm_14_tmp_3_dim, batch_norm_30_tmp_2_dim, -1, 4, 4, 4);
	free(batch_norm_11_tmp_3);
	free(batch_norm_14_tmp_3);

	batch_norm_11_tmp_3 = (float*)calloc(393216, sizeof(float));
	autox_elementwise_mul(elementwise_add_1, batch_norm_30_tmp_2, batch_norm_11_tmp_3, elementwise_add_1_dim, batch_norm_30_tmp_2_dim, batch_norm_11_tmp_3_dim, -1, 4, 4, 4);
	free(elementwise_add_1);
	free(batch_norm_30_tmp_2);

	batch_norm_14_tmp_3 = (float*)calloc(393216, sizeof(float));
	autox_elementwise_add(batch_norm_11_tmp_3, batch_norm_34_tmp_2, batch_norm_14_tmp_3, batch_norm_11_tmp_3_dim, batch_norm_34_tmp_2_dim, batch_norm_14_tmp_3_dim, -1, 4, 4, 4);
	free(batch_norm_11_tmp_3);
	free(batch_norm_34_tmp_2);

	batch_norm_11_tmp_3 = (float*)calloc(393216, sizeof(float));
	autox_conv2d(batch_norm_14_tmp_3, (float*)((int8_t*)weights) + 1437156, (float*)((int8_t*)weights) + 1437412, batch_norm_11_tmp_3, batch_norm_14_tmp_3_dim, conv2d_96_w_0_dim, batch_norm_11_tmp_3_dim, 256, 0, 1, 1, 1);
	free(batch_norm_14_tmp_3);

	batch_norm_14_tmp_3 = (float*)calloc(393216, sizeof(float));
	autox_conv2d(batch_norm_11_tmp_3, (float*)((int8_t*)weights) + 1437668, (float*)((int8_t*)weights) + 1437818, batch_norm_14_tmp_3, batch_norm_11_tmp_3_dim, conv2d_97_w_0_dim, batch_norm_14_tmp_3_dim, 1, 0, 1, 1, 0);
	free(batch_norm_11_tmp_3);

	batch_norm_11_tmp_3 = (float*)calloc(393216, sizeof(float));
	autox_bilinear_interp(batch_norm_14_tmp_3, batch_norm_11_tmp_3, batch_norm_14_tmp_3_dim, batch_norm_11_tmp_3_dim, 0, 0,0);
	free(batch_norm_14_tmp_3);

	autox_argmax(batch_norm_11_tmp_3, Out, batch_norm_11_tmp_3_dim, argmax_0_tmp_0_dim, 4, 3,0);
	free(batch_norm_11_tmp_3);
}
