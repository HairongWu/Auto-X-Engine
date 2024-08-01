#include "autox_nn_ansi.h"

#define INIT_PARAM                                   \
  auto& param = this->Param<param_t>();              \
  auto x_dims = param.x->dims();                     \
  auto w_dims = param.filter->dims();                \
  auto o_dims = param.output->dims();                \
  int win = x_dims[3];                               \
  int hin = x_dims[2];                               \
  int chin = x_dims[1];                              \
  int num = x_dims[0];                               \
  int wout = o_dims[3];                              \
  int hout = o_dims[2];                              \
  int chout = o_dims[1];                             \
  int kw = w_dims[3];                                \
  int kh = w_dims[2];                                \
  int group = param.groups;                          \
  /* deconv weights layout: chin * chout * kh * kw*/ \
  int m = chout * kw * kh / group;                   \
  int n = hin * win;                                 \
  int k = chin / group;

static bool is_a_ge_zero_and_a_lt_b(int a, int b) {
  return static_cast<unsigned>(a) < static_cast<unsigned>(b);
}

void col2im(const float* data_col,
            const int channels,
            const int height,
            const int width,
            const int kernel_h,
            const int kernel_w,
            const int pad_h0,
            const int pad_h1,
            const int pad_w0,
            const int pad_w1,
            const int stride_h,
            const int stride_w,
            const int dilation_h,
            const int dilation_w,
            float* data_im) {
  memset(data_im, 0, height * width * channels * sizeof(float));
  const int output_h =
      (height + pad_h0 + pad_h1 - (dilation_h * (kernel_h - 1) + 1)) /
          stride_h +
      1;
  const int output_w =
      (width + pad_w0 + pad_w1 - (dilation_w * (kernel_w - 1) + 1)) / stride_w +
      1;
  const int channel_size = height * width;
  for (int channel = channels; channel--; data_im += channel_size) {
    for (int kernel_row = 0; kernel_row < kernel_h; kernel_row++) {
      for (int kernel_col = 0; kernel_col < kernel_w; kernel_col++) {
        int input_row = -pad_h0 + kernel_row * dilation_h;
        for (int output_rows = output_h; output_rows; output_rows--) {
          if (!is_a_ge_zero_and_a_lt_b(input_row, height)) {
            data_col += output_w;
          } else {
            int input_col = -pad_w0 + kernel_col * dilation_w;
            for (int output_col = output_w; output_col; output_col--) {
              if (is_a_ge_zero_and_a_lt_b(input_col, width)) {
                data_im[input_row * width + input_col] += *data_col;
              }
              data_col++;
              input_col += stride_w;
            }
          }
          input_row += stride_h;
        }
      }
    }
  }
}


void conv_transpose_depthwise_s1(const float* dst,
                                 const float* weights,
                                 const int channels,
                                 const int height,
                                 const int width,
                                 const int kernel_h,
                                 const int kernel_w,
                                 const int pad_h0,
                                 const int pad_h1,
                                 const int pad_w0,
                                 const int pad_w1,
                                 const int dilation_h,
                                 const int dilation_w,
                                 float* src,
                                 X86Context* ctx) {
  memset(src, 0, height * width * channels * sizeof(float));
  const int output_h =
      (height + pad_h0 + pad_h1 - (dilation_h * (kernel_h - 1) + 1)) + 1;
  const int output_w =
      (width + pad_w0 + pad_w1 - (dilation_w * (kernel_w - 1) + 1)) + 1;
  float* zero_ptr =
      static_cast<float*>(TargetMalloc(TARGET(kX86), width * sizeof(float)));
  memset(zero_ptr, 0, width * sizeof(float));
  const int ic_plane_size = height * width;
  const int oc_plane_size = output_h * output_w;
  const int rr_plane_size = kernel_h * kernel_w;

  __m256 vec_zero = _mm256_set1_ps(0.f);
  __m256 vec_width = _mm256_set1_ps(width * 1.0f);

  for (int c = 0; c < channels; c++) {
    int dst_z = c * oc_plane_size;
    int weight_z = c * rr_plane_size;
    int src_z = c * ic_plane_size;
    for (int ky = 0; ky < kernel_h; ky++) {
      int weight_y = ky * kernel_w;
      for (int kx = 0; kx < kernel_w; kx++) {
        int weight_offset = weight_z + weight_y + kx;
        const float* weight_addr = weights + weight_offset;
        for (int ih = -pad_h0 + ky * dilation_h, oh = 0; oh < output_h;
             ih += 4, oh += 4) {
          int src_y = ih * width;
          int dst_y = oh * output_w;
          bool boundary_y0 = ((ih >= 0) && (ih < height)) && (oh < output_h);
          bool boundary_y1 =
              ((ih + 1) >= 0) && ((ih + 1) < height) && ((oh + 1) < output_h);
          bool boundary_y2 =
              ((ih + 2) >= 0) && ((ih + 2) < height) && ((oh + 2) < output_h);
          bool boundary_y3 =
              ((ih + 3) >= 0) && ((ih + 3) < height) && ((oh + 3) < output_h);
          float* src_addr_h0 = boundary_y0 ? (src + src_z + src_y) : zero_ptr;
          float* src_addr_h1 =
              boundary_y1 ? (src + src_z + width + src_y) : zero_ptr;
          float* src_addr_h2 =
              boundary_y2 ? (src + src_z + width * 2 + src_y) : zero_ptr;
          float* src_addr_h3 =
              boundary_y3 ? (src + src_z + width * 3 + src_y) : zero_ptr;
          int iw = -pad_w0 + kx * dilation_w;
          int i = 0;

          for (; i + 7 < output_w; i += 8, iw += 8) {
            int dst_offset = dst_z + dst_y + i;
            const float* dst_addr = dst + dst_offset;
            const float iw_data[8] = {iw + 0.f,
                                      iw + 1.f,
                                      iw + 2.f,
                                      iw + 3.f,
                                      iw + 4.f,
                                      iw + 5.f,
                                      iw + 6.f,
                                      iw + 7.f};
            // select weight
            __m256 vec_iw = _mm256_loadu_ps(&iw_data[0]);
            __m256 vec_mask = _mm256_and_ps(
                _mm256_cmp_ps(vec_iw, vec_zero, 13),
                _mm256_cmp_ps(vec_iw, vec_width, 1));  // GE:13  LT:1
            __m256 vec_weight = _mm256_set1_ps(weight_addr[0]);
            vec_weight = _mm256_blendv_ps(vec_zero, vec_weight, vec_mask);

            // compute 4 lines
            __m256 vec_dst = _mm256_fmadd_ps(_mm256_loadu_ps(dst_addr),
                                             vec_weight,
                                             _mm256_loadu_ps(src_addr_h0 + iw));
            _mm256_storeu_ps(src_addr_h0 + iw, vec_dst);

            vec_dst = _mm256_fmadd_ps(_mm256_loadu_ps(dst_addr + output_w),
                                      vec_weight,
                                      _mm256_loadu_ps(src_addr_h1 + iw));
            _mm256_storeu_ps(src_addr_h1 + iw, vec_dst);

            vec_dst = _mm256_fmadd_ps(_mm256_loadu_ps(dst_addr + 2 * output_w),
                                      vec_weight,
                                      _mm256_loadu_ps(src_addr_h2 + iw));
            _mm256_storeu_ps(src_addr_h2 + iw, vec_dst);

            vec_dst = _mm256_fmadd_ps(_mm256_loadu_ps(dst_addr + 3 * output_w),
                                      vec_weight,
                                      _mm256_loadu_ps(src_addr_h3 + iw));
            _mm256_storeu_ps(src_addr_h3 + iw, vec_dst);
          }

          for (; i < output_w; i++, iw++) {
            bool boundary_x = ((iw >= 0) && (iw < width));
            int src_offset = src_z + src_y + iw;
            int dst_offset = dst_z + dst_y + i;
            src[src_offset] += (boundary_x) * (boundary_y0)*dst[dst_offset] *
                               weights[weight_offset];
            src[src_offset + width] +=
                (boundary_x) * (boundary_y1)*dst[dst_offset + output_w] *
                weights[weight_offset];
            src[src_offset + width * 2] +=
                (boundary_x) * (boundary_y2)*dst[dst_offset + output_w * 2] *
                weights[weight_offset];
            src[src_offset + width * 3] +=
                (boundary_x) * (boundary_y3)*dst[dst_offset + output_w * 3] *
                weights[weight_offset];
          }
        }
      }
    }
  }
  free(zero_ptr);
}

void conv_transpose_depthwise_s2(const float* dst,
                                 const float* weights,
                                 const int channels,
                                 const int height,
                                 const int width,
                                 const int kernel_h,
                                 const int kernel_w,
                                 const int pad_h0,
                                 const int pad_h1,
                                 const int pad_w0,
                                 const int pad_w1,
                                 const int dilation_h,
                                 const int dilation_w,
                                 float* src,
                                 X86Context* ctx) {
  memset(src, 0, height * width * channels * sizeof(float));
  const int output_h =
      (height + pad_h0 + pad_h1 - (dilation_h * (kernel_h - 1) + 1)) / 2 + 1;
  const int output_w =
      (width + pad_w0 + pad_w1 - (dilation_w * (kernel_w - 1) + 1)) / 2 + 1;
  float* zero_ptr =
      static_cast<float*>(TargetMalloc(TARGET(kX86), width * sizeof(float)));
  memset(zero_ptr, 0, width * sizeof(float));
  const int ic_plane_size = height * width;
  const int oc_plane_size = output_h * output_w;
  const int rr_plane_size = kernel_h * kernel_w;

  __m256 vec_zero = _mm256_set1_ps(0.f);
  __m256 vec_width = _mm256_set1_ps(width * 1.0f);
  const int mask_store[8] = {-1, 0, -1, 0, -1, 0, -1, 0};
  __m256i vec_store_mask = _mm256_loadu_si256((const __m256i*)&mask_store[0]);

  for (int c = 0; c < channels; c++) {
    int dst_z = c * oc_plane_size;
    int weight_z = c * rr_plane_size;
    int src_z = c * ic_plane_size;
    for (int ky = 0; ky < kernel_h; ky++) {
      int weight_y = ky * kernel_w;
      for (int kx = 0; kx < kernel_w; kx++) {
        int weight_offset = weight_z + weight_y + kx;
        const float* weight_addr = weights + weight_offset;
        for (int ih = -pad_h0 + ky * dilation_h, oh = 0; oh < output_h;
             ih += 8, oh += 4) {
          int src_y = ih * width;
          int dst_y = oh * output_w;
          bool boundary_y0 = ((ih >= 0) && (ih < height)) && (oh < output_h);
          bool boundary_y1 =
              ((ih + 2) >= 0) && ((ih + 2) < height) && ((oh + 1) < output_h);
          bool boundary_y2 =
              ((ih + 4) >= 0) && ((ih + 4) < height) && ((oh + 2) < output_h);
          bool boundary_y3 =
              ((ih + 6) >= 0) && ((ih + 6) < height) && ((oh + 3) < output_h);
          float* src_addr_h0 = boundary_y0 ? (src + src_z + src_y) : zero_ptr;
          float* src_addr_h1 =
              boundary_y1 ? (src + src_z + width * 2 + src_y) : zero_ptr;
          float* src_addr_h2 =
              boundary_y2 ? (src + src_z + width * 4 + src_y) : zero_ptr;
          float* src_addr_h3 =
              boundary_y3 ? (src + src_z + width * 6 + src_y) : zero_ptr;
          int iw = -pad_w0 + kx * dilation_w;
          int i = 0;

          for (; i + 7 < output_w; i += 8, iw += 16) {
            int dst_offset = dst_z + dst_y + i;
            const float* dst_addr = dst + dst_offset;
            const float iw_data[8] = {iw + 0.f,
                                      iw + 2.f,
                                      iw + 4.f,
                                      iw + 6.f,
                                      iw + 8.f,
                                      iw + 10.f,
                                      iw + 12.f,
                                      iw + 14.f};

            // select weight
            __m256 vec_iw = _mm256_loadu_ps(&iw_data[0]);
            __m256 vec_mask = _mm256_and_ps(
                _mm256_cmp_ps(vec_iw, vec_zero, 13),
                _mm256_cmp_ps(vec_iw, vec_width, 1));  // GE:13  LT:1
            __m256 vec_weight = _mm256_set1_ps(weight_addr[0]);
            vec_weight = _mm256_blendv_ps(vec_zero, vec_weight, vec_mask);

            // compute 4 lines
            __m256 vec_data_lo = _mm256_loadu_ps(src_addr_h0 + iw);
            __m256 vec_data_hi = _mm256_loadu_ps(src_addr_h0 + iw + 8);
            __m256 vec_data =
                _mm256_shuffle_ps(vec_data_lo, vec_data_hi, 136);  // 0x88
            __m256i vec_tmp_data =
                _mm256_permute4x64_epi64(_mm256_castps_si256(vec_data),
                                         216);  // 11011000b
            vec_data = _mm256_castsi256_ps(vec_tmp_data);
            __m256 vec_dst = _mm256_fmadd_ps(
                _mm256_loadu_ps(dst_addr), vec_weight, vec_data);
            __m256 vec_dst_lo = _mm256_unpacklo_ps(vec_dst, vec_zero);
            __m256 vec_dst_hi = _mm256_unpackhi_ps(vec_dst, vec_zero);
            _mm256_maskstore_ps(
                src_addr_h0 + iw,
                vec_store_mask,
                _mm256_permute2f128_ps(vec_dst_lo, vec_dst_hi, 0x20));
            _mm256_maskstore_ps(
                src_addr_h0 + iw + 8,
                vec_store_mask,
                _mm256_permute2f128_ps(vec_dst_lo, vec_dst_hi, 0x31));

            vec_data_lo = _mm256_loadu_ps(src_addr_h1 + iw);
            vec_data_hi = _mm256_loadu_ps(src_addr_h1 + iw + 8);
            vec_data =
                _mm256_shuffle_ps(vec_data_lo, vec_data_hi, 136);  // 0x88
            vec_tmp_data =
                _mm256_permute4x64_epi64(_mm256_castps_si256(vec_data),
                                         216);  // 11011000b
            vec_data = _mm256_castsi256_ps(vec_tmp_data);

            vec_dst = _mm256_fmadd_ps(
                _mm256_loadu_ps(dst_addr + output_w), vec_weight, vec_data);
            vec_dst_lo = _mm256_unpacklo_ps(vec_dst, vec_zero);
            vec_dst_hi = _mm256_unpackhi_ps(vec_dst, vec_zero);
            _mm256_maskstore_ps(
                src_addr_h1 + iw,
                vec_store_mask,
                _mm256_permute2f128_ps(vec_dst_lo, vec_dst_hi, 0x20));
            _mm256_maskstore_ps(
                src_addr_h1 + iw + 8,
                vec_store_mask,
                _mm256_permute2f128_ps(vec_dst_lo, vec_dst_hi, 0x31));

            vec_data_lo = _mm256_loadu_ps(src_addr_h2 + iw);
            vec_data_hi = _mm256_loadu_ps(src_addr_h2 + iw + 8);
            vec_data =
                _mm256_shuffle_ps(vec_data_lo, vec_data_hi, 136);  // 0x88
            vec_tmp_data =
                _mm256_permute4x64_epi64(_mm256_castps_si256(vec_data),
                                         216);  // 11011000b
            vec_data = _mm256_castsi256_ps(vec_tmp_data);
            vec_dst = _mm256_fmadd_ps(
                _mm256_loadu_ps(dst_addr + 2 * output_w), vec_weight, vec_data);
            vec_dst_lo = _mm256_unpacklo_ps(vec_dst, vec_zero);
            vec_dst_hi = _mm256_unpackhi_ps(vec_dst, vec_zero);
            _mm256_maskstore_ps(
                src_addr_h2 + iw,
                vec_store_mask,
                _mm256_permute2f128_ps(vec_dst_lo, vec_dst_hi, 0x20));
            _mm256_maskstore_ps(
                src_addr_h2 + iw + 8,
                vec_store_mask,
                _mm256_permute2f128_ps(vec_dst_lo, vec_dst_hi, 0x31));

            vec_data_lo = _mm256_loadu_ps(src_addr_h3 + iw);
            vec_data_hi = _mm256_loadu_ps(src_addr_h3 + iw + 8);
            vec_data =
                _mm256_shuffle_ps(vec_data_lo, vec_data_hi, 136);  // 0x88
            vec_tmp_data =
                _mm256_permute4x64_epi64(_mm256_castps_si256(vec_data),
                                         216);  // 11011000b
            vec_data = _mm256_castsi256_ps(vec_tmp_data);
            vec_dst = _mm256_fmadd_ps(
                _mm256_loadu_ps(dst_addr + 3 * output_w), vec_weight, vec_data);
            vec_dst_lo = _mm256_unpacklo_ps(vec_dst, vec_zero);
            vec_dst_hi = _mm256_unpackhi_ps(vec_dst, vec_zero);
            _mm256_maskstore_ps(
                src_addr_h3 + iw,
                vec_store_mask,
                _mm256_permute2f128_ps(vec_dst_lo, vec_dst_hi, 0x20));
            _mm256_maskstore_ps(
                src_addr_h3 + iw + 8,
                vec_store_mask,
                _mm256_permute2f128_ps(vec_dst_lo, vec_dst_hi, 0x31));
          }

          for (; i < output_w; i++, iw += 2) {
            bool boundary_x = ((iw >= 0) && (iw < width));
            int src_offset = src_z + src_y + iw;
            int dst_offset = dst_z + dst_y + i;
            src[src_offset] += (boundary_x) * (boundary_y0)*dst[dst_offset] *
                               weights[weight_offset];
            src[src_offset + width * 2] +=
                (boundary_x) * (boundary_y1)*dst[dst_offset + output_w] *
                weights[weight_offset];
            src[src_offset + width * 4] +=
                (boundary_x) * (boundary_y2)*dst[dst_offset + output_w * 2] *
                weights[weight_offset];
            src[src_offset + width * 6] +=
                (boundary_x) * (boundary_y3)*dst[dst_offset + output_w * 3] *
                weights[weight_offset];
          }
        }
      }
    }
  }
  free(zero_ptr);
}

void fill_bias_act(float *tensor,
                   const float *bias,
                   int channel,
                   int channel_size,
                   bool flag_bias,
                   const operators::ActivationParam *act_param) {
  auto act_type = act_param->active_type;
  float local_alpha = 0.f;
  int len = channel * channel_size;

  if ((act_param != nullptr) && (act_param->has_active)) {
    if ((flag_bias) && (bias != nullptr)) {
      // activate and bias
      if (act_type == lite_api::ActivationType::kRelu) {
        activate_relu_inplace_bias(
            tensor, bias, channel, channel_size, local_alpha, 0);
      } else if (act_type == lite_api::ActivationType::kRelu6) {
        local_alpha = act_param->Relu_clipped_coef;
        activate_relu_inplace_bias(
            tensor, bias, channel, channel_size, local_alpha, 1);
      } else if (act_type == lite_api::ActivationType::kLeakyRelu) {
        local_alpha = act_param->Leaky_relu_alpha;
        activate_lrelu_inplace_bias(
            tensor, bias, channel, channel_size, local_alpha);
      } else if (act_type == lite_api::ActivationType::kHardSwish) {
        local_alpha = act_param->hard_swish_scale;
        activate_hardswish_inplace_bias(tensor,
                                        bias,
                                        channel,
                                        channel_size,
                                        local_alpha,
                                        act_param->hard_swish_threshold,
                                        act_param->hard_swish_offset);
      }
    } else {
      // activate
      if (act_type == lite_api::ActivationType::kRelu) {
        activate_relu_inplace(tensor, len, local_alpha, 0);
      } else if (act_type == lite_api::ActivationType::kRelu6) {
        local_alpha = act_param->Relu_clipped_coef;
        activate_relu_inplace(tensor, len, local_alpha, 1);
      } else if (act_type == lite_api::ActivationType::kLeakyRelu) {
        local_alpha = act_param->Leaky_relu_alpha;
        activate_lrelu_inplace(tensor, len, local_alpha);
      } else if (act_type == lite_api::ActivationType::kHardSwish) {
        local_alpha = act_param->hard_swish_scale;
        activate_hardswish_inplace(tensor,
                                   len,
                                   local_alpha,
                                   act_param->hard_swish_threshold,
                                   act_param->hard_swish_offset);
      }
    }
  } else {
    // only add bias
    if ((flag_bias) && (bias != nullptr))
      activate_none_inplace_bias(tensor, bias, channel, channel_size);
  }
}

#define DEPTHWISE_FUNCS                                                    \
  din_batch, weights, chout, hout, wout, kh, kw, paddings[0], paddings[1], \
      paddings[2], paddings[3], dilations[0], dilations[1], dout_batch, &ctx
      
// The convolution transpose operator consumes an input tensor and a filter, and computes the output.
void gautox_conv2d_transpose_ansi(
        const struct ggml_compute_params * params,
              struct ggml_tensor * dst) {

    INIT_PARAM
    bool flag_bias = (param.bias != nullptr);
    auto paddings = *param.paddings;
    auto dilations = *param.dilations;
    bool pads_equal =
        (paddings[0] == paddings[1]) && (paddings[2] == paddings[3]);

    int group_size_in = win * hin * chin / group;
    int group_size_weights = chin / group * chout / group * kw * kh;
    int group_size_coldata = m * n;
    bool pads_all_qual = pads_equal && (paddings[0] == paddings[2]);
    bool flag_1x1s1p1 = (kw == 1) && (kh == 1) && (param.strides[0] == 1) &&
                        (param.strides[1] == 1) && pads_all_qual &&
                        (paddings[0] == 0) && (dilations[0] == 1) &&
                        (dilations[1] == 1);
    auto din = param.x->data<float>();
    auto dout = param.output->mutable_data<float>();
    auto weights = param.filter->data<float>();
    auto act_param = param.activation_param;
    bool depthwise_s1 =
        depthwise_ && (param.strides[0] == 1 && param.strides[1] == 1);
    bool depthwise_s2 =
        depthwise_ && (param.strides[0] == 2 && param.strides[1] == 2);
    const float* bias_ptr =
        flag_bias ? static_cast<const float*>(param.bias->data<float>())
                    : nullptr;
    float* col_data = nullptr;

    if (!flag_1x1s1p1) {
        int col_size = param.groups * group_size_coldata;
        col_data =(float*)calloc(col_size * sizeof(float));
    }

    for (int i = 0; i < num; i++) {
        const float* din_batch = din + i * chin * hin * win;
        float* dout_batch = dout + i * chout * hout * wout;

        if (depthwise_s1) {
            conv_transpose_depthwise_s1(DEPTHWISE_FUNCS);
        } else if (depthwise_s2) {
            conv_transpose_depthwise_s2(DEPTHWISE_FUNCS);
        } else {
        paddle::lite::x86::math::Blas<lite::TargetType::kX86> matmul(ctx);
        if (flag_1x1s1p1) {
            col_data = dout_batch;
        }
        for (int g = 0; g < group; g++) {
            const float* din_group = din_batch + g * group_size_in;
            const float* weights_group = weights + g * group_size_weights;
            float* coldata_group = col_data + g * group_size_coldata;
            matmul.GEMM<float>(true,
                            false,
                            m,
                            n,
                            k,
                            1.f,
                            weights_group,
                            m,
                            din_group,
                            n,
                            0.f,
                            coldata_group,
                            n);
        }
        if (!flag_1x1s1p1) {
            col2im(col_data,
                                    chout,
                                    hout,
                                    wout,
                                    kh,
                                    kw,
                                    paddings[0],
                                    paddings[1],
                                    paddings[2],
                                    paddings[3],
                                    param.strides[0],
                                    param.strides[1],
                                    dilations[0],
                                    dilations[1],
                                    dout_batch);
        }
        }
        // bias and activate
        fill_bias_act(
            dout_batch, bias_ptr, chout, wout * hout, flag_bias, &act_param);
    }
    if (!flag_1x1s1p1) free(col_data);
}
