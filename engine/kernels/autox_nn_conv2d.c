
#include "../include/autox_nn.h"

#define INIT_PARAM                      \
  int win = x_dims[3];                  \
  int hin = x_dims[2];                  \
  int chin = x_dims[1];                 \
  int num = x_dims[0];                  \
  int wout = o_dims[3];                 \
  int hout = o_dims[2];                 \
  int chout = o_dims[1];                \
  int kw = w_dims[3];                   \
  int kh = w_dims[2];                   \
  int m = chout / group;                \
  int n = hout * wout;                  \
  int k = chin * kw * kh / group;\
  const int output_channel = w_dims[0];                         \
  bool dw_kernel = (chin == group && output_channel == group);     \
  bool no_dilation = dilations == 1;              \
  bool flag_dw_3x3 =                                                          \
      (kh == 3) && (strides == 1 || strides == 2); \
  bool flag_dw_5x5 =                                                          \
      (kh == 5) && (strides == 1 || strides == 2);

static void activate_relu_inplace(float* data, int len, float alpha, int mode) {
	int i = 0;

	if (0 == mode) {  // relu
#ifdef __AVX__
		__m256 vec_zero = _mm256_set1_ps(0.f);
		for (; i + 7 < len; i += 8) {
			__m256 vec_data = _mm256_loadu_ps(data + i);
			_mm256_storeu_ps(data + i, _mm256_max_ps(vec_data, vec_zero));
		}
#endif
#ifdef __SSE__
		__m128 vec_zero_128 = _mm_set1_ps(0.f);
		for (; i + 3 < len; i += 4) {
			__m128 vec_data_128 = _mm_loadu_ps(data + i);
			_mm_storeu_ps(data + i, _mm_max_ps(vec_data_128, vec_zero_128));
		}
#endif
		for (; i < len; i++) {
			data[i] = data[i] > 0.f ? data[i] : 0.f;
		}
	}
	else {  // relu6
#ifdef __AVX__
		__m256 vec_zero = _mm256_set1_ps(0.f);
		__m256 vec_alph = _mm256_set1_ps(alpha);
		for (; i + 7 < len; i += 8) {
			__m256 vec_data = _mm256_loadu_ps(data + i);
			_mm256_storeu_ps(
				data + i, _mm256_min_ps(_mm256_max_ps(vec_data, vec_zero), vec_alph));
		}
#endif
#ifdef __SSE__
		__m128 vec_zero_128 = _mm_set1_ps(0.f);
		__m128 vec_alph_128 = _mm_set1_ps(alpha);
		for (; i + 3 < len; i += 4) {
			__m128 vec_data_128 = _mm_loadu_ps(data + i);
			_mm_storeu_ps(
				data + i,
				_mm_min_ps(_mm_max_ps(vec_data_128, vec_zero_128), vec_alph_128));
		}
#endif
		for (; i < len; i++) {
			data[i] = data[i] > 0.f ? data[i] : 0.f;
			data[i] = data[i] < alpha ? data[i] : alpha;
		}
	}
}
static void activate_relu_inplace_bias(float* data,
	const float* bias,
	int channel,
	int channel_size,
	float alpha,
	int mode) {
	int i = 0;
	int j = 0;
	float* tmp_data = data;

#ifdef __AVX__
	__m256 vec_zero = { 0.f };
	__m256 vec_bias = { 0.f };
	__m256 vec_data = { 0.f };
	__m256 vec_alph = _mm256_set1_ps(alpha);
#endif
#ifdef __SSE__
	__m128 vec_zero_128 = { 0.f };
	__m128 vec_bias_128 = { 0.f };
	__m128 vec_data_128 = { 0.f };
	__m128 vec_alph_128 = _mm_set1_ps(alpha);
#endif

	if (0 == mode) {  // relu
		for (j = 0; j < channel; j++) {
			i = 0;
			tmp_data = data + j * channel_size;
#ifdef __AVX__
			vec_bias = _mm256_set1_ps(bias[j]);
			for (; i + 7 < channel_size; i += 8) {
				vec_data = _mm256_loadu_ps(tmp_data + i);
				vec_data = _mm256_add_ps(vec_bias, vec_data);
				_mm256_storeu_ps(tmp_data + i, _mm256_max_ps(vec_data, vec_zero));
			}
#endif
#ifdef __SSE__
			vec_bias_128 = _mm_set1_ps(bias[j]);
			for (; i + 3 < channel_size; i += 4) {
				vec_data_128 = _mm_loadu_ps(tmp_data + i);
				vec_data_128 = _mm_add_ps(vec_data_128, vec_bias_128);
				_mm_storeu_ps(tmp_data + i, _mm_max_ps(vec_data_128, vec_zero_128));
			}
#endif
			for (; i < channel_size; i++) {
				tmp_data[i] += bias[j];
				tmp_data[i] = tmp_data[i] > 0.f ? tmp_data[i] : 0.f;
			}
		}
	}
	else {  // relu6
		for (j = 0; j < channel; j++) {
			i = 0;
			tmp_data = data + j * channel_size;
#ifdef __AVX__
			vec_bias = _mm256_set1_ps(bias[j]);
			for (; i + 7 < channel_size; i += 8) {
				vec_data = _mm256_loadu_ps(tmp_data + i);
				vec_data = _mm256_add_ps(vec_bias, vec_data);
				_mm256_storeu_ps(
					tmp_data + i,
					_mm256_min_ps(_mm256_max_ps(vec_data, vec_zero), vec_alph));
			}
#endif
#ifdef __SSE__
			vec_bias_128 = _mm_set1_ps(bias[j]);
			for (; i + 3 < channel_size; i += 4) {
				vec_data_128 = _mm_loadu_ps(tmp_data + i);
				vec_data_128 = _mm_add_ps(vec_data_128, vec_bias_128);
				_mm_storeu_ps(
					tmp_data + i,
					_mm_min_ps(_mm_max_ps(vec_data_128, vec_zero_128), vec_alph_128));
			}
#endif
			for (; i < channel_size; i++) {
				tmp_data[i] += bias[j];
				tmp_data[i] = tmp_data[i] > 0.f ? tmp_data[i] : 0.f;
				tmp_data[i] = tmp_data[i] < alpha ? tmp_data[i] : alpha;
			}
		}
	}
}
static void activate_hardswish_inplace_bias(float* data,
	const float* bias,
	int channel,
	int channel_size,
	float scale,
	float threshold,
	float offset) {

	int cnt = channel_size >> 5;
	int remain = channel_size & 31;
	__m256 vec_zero = _mm256_set1_ps(0.f);
	__m256 vec_scale = _mm256_set1_ps(1.0 / scale);
	__m256 vec_threshold = _mm256_set1_ps(threshold);
	__m256 vec_offset = _mm256_set1_ps(offset);

	__m128 vec_zero_128 = _mm_set1_ps(0.f);
	__m128 vec_scale_128 = _mm_set1_ps(1.0 / scale);
	__m128 vec_threshold_128 = _mm_set1_ps(threshold);
	__m128 vec_offset_128 = _mm_set1_ps(offset);
	int cnt_4 = remain >> 2;
	int rem_4 = remain & 3;
	for (int i = 0; i < channel; i++) {

		__m256 vec_bias = _mm256_set1_ps(bias[i]);

		__m128 vec_bias_128 = _mm_set1_ps(bias[i]);
		float* tmp_data = data + i * channel_size;

		for (int j = 0; j < cnt; j++) {

			__m256 vin0 = _mm256_add_ps(_mm256_loadu_ps(tmp_data), vec_bias);
			__m256 vin1 = _mm256_add_ps(_mm256_loadu_ps(tmp_data + 8), vec_bias);
			__m256 vin2 = _mm256_add_ps(_mm256_loadu_ps(tmp_data + 16), vec_bias);
			__m256 vin3 = _mm256_add_ps(_mm256_loadu_ps(tmp_data + 24), vec_bias);
			__m256 vadd0 = _mm256_add_ps(vin0, vec_offset);
			__m256 vadd1 = _mm256_add_ps(vin1, vec_offset);
			__m256 vadd2 = _mm256_add_ps(vin2, vec_offset);
			__m256 vadd3 = _mm256_add_ps(vin3, vec_offset);
			__m256 vsum0 = _mm256_mul_ps(vin0, vec_scale);
			__m256 vsum1 = _mm256_mul_ps(vin1, vec_scale);
			__m256 vsum2 = _mm256_mul_ps(vin2, vec_scale);
			__m256 vsum3 = _mm256_mul_ps(vin3, vec_scale);
			__m256 vres0 =
				_mm256_min_ps(_mm256_max_ps(vadd0, vec_zero), vec_threshold);
			__m256 vres1 =
				_mm256_min_ps(_mm256_max_ps(vadd1, vec_zero), vec_threshold);
			__m256 vres2 =
				_mm256_min_ps(_mm256_max_ps(vadd2, vec_zero), vec_threshold);
			__m256 vres3 =
				_mm256_min_ps(_mm256_max_ps(vadd3, vec_zero), vec_threshold);
			_mm256_storeu_ps(tmp_data, _mm256_mul_ps(vres0, vsum0));
			_mm256_storeu_ps(tmp_data + 8, _mm256_mul_ps(vres1, vsum1));
			_mm256_storeu_ps(tmp_data + 16, _mm256_mul_ps(vres2, vsum2));
			_mm256_storeu_ps(tmp_data + 24, _mm256_mul_ps(vres3, vsum3));
			tmp_data += 32;

		}
		for (int j = 0; j < cnt_4; j++) {
			__m128 vin0 = _mm_add_ps(_mm_loadu_ps(tmp_data), vec_bias_128);
			__m128 vadd0 = _mm_add_ps(vin0, vec_offset_128);
			__m128 vsum0 = _mm_mul_ps(vin0, vec_scale_128);
			__m128 vres0 =
				_mm_min_ps(_mm_max_ps(vadd0, vec_zero_128), vec_threshold_128);
			_mm_storeu_ps(tmp_data, _mm_mul_ps(vres0, vsum0));
			tmp_data += 4;
		}
		for (int j = 0; j < rem_4; j++) {
			tmp_data[0] = tmp_data[0] + bias[i];
			tmp_data[0] = min(max(0.f, tmp_data[0] + offset), threshold) *
				tmp_data[0] / scale;
			tmp_data++;
		}
	}
}

static void activate_hardswish_inplace(
	float* data, int len, float scale, float threshold, float offset) {
#ifdef __AVX__
	int cnt = len >> 5;
	int remain = len & 31;
	__m256 vec_zero = _mm256_set1_ps(0.f);
	__m256 vec_scale = _mm256_set1_ps(1.0 / scale);
	__m256 vec_threshold = _mm256_set1_ps(threshold);
	__m256 vec_offset = _mm256_set1_ps(offset);
#else
	int cnt = len >> 4;
	int remain = len & 15;
#endif
	__m128 vec_zero_128 = _mm_set1_ps(0.f);
	__m128 vec_scale_128 = _mm_set1_ps(1.0 / scale);
	__m128 vec_threshold_128 = _mm_set1_ps(threshold);
	__m128 vec_offset_128 = _mm_set1_ps(offset);
	int cnt_4 = remain >> 2;
	int rem_4 = remain & 3;
	float* tmp_data = data;
	for (int i = 0; i < cnt; i++) {
#ifdef __AVX__
		__m256 vin0 = _mm256_loadu_ps(tmp_data);
		__m256 vin1 = _mm256_loadu_ps(tmp_data + 8);
		__m256 vin2 = _mm256_loadu_ps(tmp_data + 16);
		__m256 vin3 = _mm256_loadu_ps(tmp_data + 24);
		__m256 vadd0 = _mm256_add_ps(vin0, vec_offset);
		__m256 vadd1 = _mm256_add_ps(vin1, vec_offset);
		__m256 vadd2 = _mm256_add_ps(vin2, vec_offset);
		__m256 vadd3 = _mm256_add_ps(vin3, vec_offset);
		__m256 vsum0 = _mm256_mul_ps(vin0, vec_scale);
		__m256 vsum1 = _mm256_mul_ps(vin1, vec_scale);
		__m256 vsum2 = _mm256_mul_ps(vin2, vec_scale);
		__m256 vsum3 = _mm256_mul_ps(vin3, vec_scale);
		__m256 vres0 = _mm256_min_ps(_mm256_max_ps(vadd0, vec_zero), vec_threshold);
		__m256 vres1 = _mm256_min_ps(_mm256_max_ps(vadd1, vec_zero), vec_threshold);
		__m256 vres2 = _mm256_min_ps(_mm256_max_ps(vadd2, vec_zero), vec_threshold);
		__m256 vres3 = _mm256_min_ps(_mm256_max_ps(vadd3, vec_zero), vec_threshold);
		_mm256_storeu_ps(tmp_data, _mm256_mul_ps(vres0, vsum0));
		_mm256_storeu_ps(tmp_data + 8, _mm256_mul_ps(vres1, vsum1));
		_mm256_storeu_ps(tmp_data + 16, _mm256_mul_ps(vres2, vsum2));
		_mm256_storeu_ps(tmp_data + 24, _mm256_mul_ps(vres3, vsum3));
		tmp_data += 32;
#else
		__m128 vin0 = _mm_loadu_ps(tmp_data);
		__m128 vin1 = _mm_loadu_ps(tmp_data + 4);
		__m128 vin2 = _mm_loadu_ps(tmp_data + 8);
		__m128 vin3 = _mm_loadu_ps(tmp_data + 12);
		__m128 vadd0 = _mm_add_ps(vin0, vec_offset_128);
		__m128 vadd1 = _mm_add_ps(vin1, vec_offset_128);
		__m128 vadd2 = _mm_add_ps(vin2, vec_offset_128);
		__m128 vadd3 = _mm_add_ps(vin3, vec_offset_128);
		__m128 vsum0 = _mm_mul_ps(vin0, vec_scale_128);
		__m128 vsum1 = _mm_mul_ps(vin1, vec_scale_128);
		__m128 vsum2 = _mm_mul_ps(vin2, vec_scale_128);
		__m128 vsum3 = _mm_mul_ps(vin3, vec_scale_128);
		__m128 vres0 =
			_mm_min_ps(_mm_max_ps(vadd0, vec_zero_128), vec_threshold_128);
		__m128 vres1 =
			_mm_min_ps(_mm_max_ps(vadd1, vec_zero_128), vec_threshold_128);
		__m128 vres2 =
			_mm_min_ps(_mm_max_ps(vadd2, vec_zero_128), vec_threshold_128);
		__m128 vres3 =
			_mm_min_ps(_mm_max_ps(vadd3, vec_zero_128), vec_threshold_128);
		_mm_storeu_ps(tmp_data, _mm_mul_ps(vres0, vsum0));
		_mm_storeu_ps(tmp_data + 4, _mm_mul_ps(vres1, vsum1));
		_mm_storeu_ps(tmp_data + 8, _mm_mul_ps(vres2, vsum2));
		_mm_storeu_ps(tmp_data + 12, _mm_mul_ps(vres3, vsum3));
		tmp_data += 16;
#endif
	}
	for (int i = 0; i < cnt_4; i++) {
		__m128 vin0 = _mm_loadu_ps(tmp_data);
		__m128 vadd0 = _mm_add_ps(vin0, vec_offset_128);
		__m128 vsum0 = _mm_mul_ps(vin0, vec_scale_128);
		__m128 vres0 =
			_mm_min_ps(_mm_max_ps(vadd0, vec_zero_128), vec_threshold_128);
		_mm_storeu_ps(tmp_data, _mm_mul_ps(vres0, vsum0));
		tmp_data += 4;
	}
	for (int i = 0; i < rem_4; i++) {
		tmp_data[0] = min(max(0.f, tmp_data[0] + offset), threshold) *
			tmp_data[0] / scale;
		tmp_data++;
	}
}

static void activate_none_inplace_bias(float* data,
	const float* bias,
	int channel,
	int channel_size) {
	int i = 0;
	int j = 0;
	float* tmp_data = data;

#ifdef __AVX__
	__m256 vec_bias = { 0.f };
	__m256 vec_data = { 0.f };
#endif

	for (j = 0; j < channel; j++) {
		i = 0;
		tmp_data = data + j * channel_size;
#ifdef __AVX__
		vec_bias = _mm256_set1_ps(bias[j]);
		for (; i + 7 < channel_size; i += 8) {
			vec_data = _mm256_loadu_ps(tmp_data + i);
			vec_data = _mm256_add_ps(vec_bias, vec_data);
			_mm256_storeu_ps(tmp_data + i, vec_data);
		}
#endif
		for (; i < channel_size; i++) {
			tmp_data[i] += bias[j];
		}
	}
}

void fill_bias_act(float* tensor,
	const float* bias,
	int channel,
	int channel_size,
	int act_type,
	float relu_alpha) {
	int len = channel * channel_size;

	if (act_type) {
		if (bias != NULL) {
			// activate and bias
			if (act_type == kRelu) {
				activate_relu_inplace_bias(
					tensor, bias, channel, channel_size, relu_alpha, 0);
			}
			else if (act_type == kRelu6) {
				activate_relu_inplace_bias(
					tensor, bias, channel, channel_size, relu_alpha, 1);
			}
			/*else if (act_type == lite_api::ActivationType::kLeakyRelu) {
				local_alpha = act_param->Leaky_relu_alpha;
				activate_lrelu_inplace_bias(
					tensor, bias, channel, channel_size, local_alpha);
			}*/
			else if (act_type == kHardSwish) {
				activate_hardswish_inplace_bias(tensor,
					bias,
					channel,
					channel_size,
					6,
					6,
					3);
			}
		}
		else {
			// activate
			if (act_type == kRelu) {
				activate_relu_inplace(tensor, len, relu_alpha, 0);
			}
			else if (act_type == kRelu6) {
				activate_relu_inplace(tensor, len, relu_alpha, 1);
			}
			/*else if (act_type == lite_api::ActivationType::kLeakyRelu) {
				local_alpha = act_param->Leaky_relu_alpha;
				activate_lrelu_inplace(tensor, len, local_alpha);
			}*/
			else if (act_type == kHardSwish) {
				activate_hardswish_inplace(tensor,
					len,
					6,
					6,
					3);
			}
		}
	}
	else {
		// only add bias
		if (bias != NULL)
			activate_none_inplace_bias(tensor, bias, channel, channel_size);
	}
}

// The convolution operator consumes an input tensor and a filter, and computes the output.
void autox_conv2d(float* din, const float* bias, const float* weights, float* dout, uint16_t* x_dims, uint16_t* w_dims, uint16_t* o_dims,
	uint16_t group, uint8_t paddings, uint8_t strides, uint8_t dilations, int8_t act_type)
{
	INIT_PARAM

	bool flag_1x1gemm_ = false;
	bool flag_dw = flag_dw_3x3 || flag_dw_5x5;
	if (kw == 1 && strides == 1 && paddings == 0) {
		flag_1x1gemm_ = true;
	}
	else {
		flag_1x1gemm_ = false;
	}

	//! select conv impl
	if (dw_kernel && flag_dw &&
		((flag_dw_5x5 && no_dilation) || (flag_dw_3x3 && (group & 3) == 0))) {
		autox_conv2d_depthwise(din, dout, weights, bias,
			x_dims, w_dims, o_dims, strides, dilations, paddings, bias != NULL, act_type);
		return;
	}

	float relu_alpha = 1.f;
	if (act_type == 2) {
		relu_alpha = 6.0;
	}

	unsigned int group_size_out = m * n;
	unsigned int group_size_weights = m * k;
	unsigned int group_size_coldata = n * k;
	unsigned int channel_in_size = chin * hin * win;
	unsigned int channel_out_size = chout * hout * wout;

	float* col_data = NULL;

	if (!flag_1x1gemm_) {
		int col_size = group * group_size_coldata;
		col_data = (float*)malloc(col_size * sizeof(float));
	}
	for (int i = 0; i < num; i++) {
		const float* din_batch = din + i * channel_in_size;
		float* dout_batch = dout + i * channel_out_size;
		const float* din_data = din_batch;
		if (!flag_1x1gemm_) {
			autox_im2col(din_batch,
				chin,
				hin,
				win,
				kh,
				paddings,
				strides,
				dilations,
				col_data);
			din_data = col_data;
		}

		for (int g = 0; g < group; g++) {
			const float* col_data_group = din_data + g * group_size_coldata;
			const float* weights_group = weights + g * group_size_weights;
			float* dout_group = dout_batch + g * group_size_out;
			/*if (n == 1) {
				gemm_cpu_f32(
					false, m, k, 1.f, weights_group, col_data_group, 0.f, dout_group);
			}
			else {*/
			gemm_cpu(false,
				false,
				m,
				n,
				k,
				1.f,
				weights_group,
				k,
				col_data_group,
				n,
				0.f,
				dout_group,
				n);
			//}
		}
		//! bias and activate
		fill_bias_act(
			dout_batch, bias, chout, wout * hout, act_type, relu_alpha);
	}
	if (!flag_1x1gemm_) free(col_data);
}

