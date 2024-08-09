#include "../include/autox_nn.h"

#define INIT_PARAM                                   \
  int win = x_dims[3];                               \
  int hin = x_dims[2];                               \
  int chin = x_dims[1];                              \
  int num = x_dims[0];                               \
  int wout = o_dims[3];                              \
  int hout = o_dims[2];                              \
  int chout = o_dims[1];                             \
  int kw = w_dims[3];                                \
  int kh = w_dims[2];                                \
  /* deconv weights layout: chin * chout * kh * kw*/ \
  int m = chout * kw * kh / group;                   \
  int n = hin * win;                                 \
  int k = chin / group;

static bool is_a_ge_zero_and_a_lt_b(int a, int b) {
  return (unsigned)(a) < (unsigned)(b);
}

void col2im(const float* data_col,
            const int channels,
            const int height,
            const int width,
            const int kernel,
            const int pad,
            const int stride,
            const int dilation,
            float* data_im) {
  memset(data_im, 0, height * width * channels * sizeof(float));
  const int output_h =
      (height + pad + pad - (dilation * (kernel - 1) + 1)) /
          stride +
      1;
  const int output_w =
      (width + pad + pad - (dilation * (kernel - 1) + 1)) / stride +
      1;
  const int channel_size = height * width;
  for (int channel = channels; channel--; data_im += channel_size) {
    for (int kernel_row = 0; kernel_row < kernel; kernel_row++) {
      for (int kernel_col = 0; kernel_col < kernel; kernel_col++) {
        int input_row = -pad + kernel_row * dilation;
        for (int output_rows = output_h; output_rows; output_rows--) {
          if (!is_a_ge_zero_and_a_lt_b(input_row, height)) {
            data_col += output_w;
          } else {
            int input_col = -pad + kernel_col * dilation;
            for (int output_col = output_w; output_col; output_col--) {
              if (is_a_ge_zero_and_a_lt_b(input_col, width)) {
                data_im[input_row * width + input_col] += *data_col;
              }
              data_col++;
              input_col += stride;
            }
          }
          input_row += stride;
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
                                 const int kernel,
                                 const int pad,
                                 const int dilation,
                                 float* src) {
  memset(src, 0, height * width * channels * sizeof(float));
  const int output_h =
      (height + pad + pad - (dilation * (kernel - 1) + 1)) + 1;
  const int output_w =
      (width + pad + pad - (dilation * (kernel - 1) + 1)) + 1;
  float* zero_ptr =
      (float*)(malloc(width * sizeof(float)));
  memset(zero_ptr, 0, width * sizeof(float));
  const int ic_plane_size = height * width;
  const int oc_plane_size = output_h * output_w;
  const int rr_plane_size = kernel * kernel;

  __m256 vec_zero = _mm256_set1_ps(0.f);
  __m256 vec_width = _mm256_set1_ps(width * 1.0f);

  for (int c = 0; c < channels; c++) {
    int dst_z = c * oc_plane_size;
    int weight_z = c * rr_plane_size;
    int src_z = c * ic_plane_size;
    for (int ky = 0; ky < kernel; ky++) {
      int weight_y = ky * kernel;
      for (int kx = 0; kx < kernel; kx++) {
        int weight_offset = weight_z + weight_y + kx;
        const float* weight_addr = weights + weight_offset;
        for (int ih = -pad + ky * dilation, oh = 0; oh < output_h;
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
          int iw = -pad + kx * dilation;
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
                                 const int kernel,
                                 const int pad,
                                 const int dilation,
                                 float* src) {
  memset(src, 0, height * width * channels * sizeof(float));
  const int output_h =
      (height + pad + pad - (dilation * (kernel - 1) + 1)) / 2 + 1;
  const int output_w =
      (width + pad + pad - (dilation * (kernel - 1) + 1)) / 2 + 1;
  float* zero_ptr =
      (float*)(malloc(width * sizeof(float)));
  memset(zero_ptr, 0, width * sizeof(float));
  const int ic_plane_size = height * width;
  const int oc_plane_size = output_h * output_w;
  const int rr_plane_size = kernel * kernel;

  __m256 vec_zero = _mm256_set1_ps(0.f);
  __m256 vec_width = _mm256_set1_ps(width * 1.0f);
  const int mask_store[8] = {-1, 0, -1, 0, -1, 0, -1, 0};
  __m256i vec_store_mask = _mm256_loadu_si256((const __m256i*)&mask_store[0]);

  for (int c = 0; c < channels; c++) {
    int dst_z = c * oc_plane_size;
    int weight_z = c * rr_plane_size;
    int src_z = c * ic_plane_size;
    for (int ky = 0; ky < kernel; ky++) {
      int weight_y = ky * kernel;
      for (int kx = 0; kx < kernel; kx++) {
        int weight_offset = weight_z + weight_y + kx;
        const float* weight_addr = weights + weight_offset;
        for (int ih = -pad + ky * dilation, oh = 0; oh < output_h;
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
          int iw = -pad + kx * dilation;
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


#define DEPTHWISE_FUNCS                                                    \
  din_batch, weights, chout, hout, wout, kh, paddings, \
      dilations, dout_batch
      
// The convolution transpose operator consumes an input tensor and a filter, and computes the output.
void autox_conv2d_transpose(float* din, const float* bias, const float* weights, float* dout, uint16_t* x_dims, uint16_t* w_dims, uint16_t* o_dims,
    uint16_t group, uint8_t paddings, uint8_t strides, uint8_t dilations, int8_t act_type) {

    INIT_PARAM
    bool flag_bias = (bias != NULL);

    int group_size_in = win * hin * chin / group;
    int group_size_weights = chin / group * chout / group * kw * kh;
    int group_size_coldata = m * n;

    bool flag_1x1s1p1 = (kw == 1) && (kh == 1) && (strides == 1) &&
                        (paddings == 0) && (dilations == 1) &&
                        (dilations == 1);

    bool depthwise_s1 = (strides == 1);
    bool depthwise_s2 = (strides == 2);

    float* col_data = NULL;

    if (!flag_1x1s1p1) {
        int col_size = group * group_size_coldata;
        col_data =(float*)calloc(col_size, sizeof(float));
    }

    for (int i = 0; i < num; i++) {
        const float* din_batch = din + i * chin * hin * win;
        float* dout_batch = dout + i * chout * hout * wout;

        if (depthwise_s1) {
            conv_transpose_depthwise_s1(DEPTHWISE_FUNCS);
        } else if (depthwise_s2) {
            conv_transpose_depthwise_s2(DEPTHWISE_FUNCS);
        } else {

        if (flag_1x1s1p1) {
            col_data = dout_batch;
        }
        for (int g = 0; g < group; g++) {
            const float* din_group = din_batch + g * group_size_in;
            const float* weights_group = weights + g * group_size_weights;
            float* coldata_group = col_data + g * group_size_coldata;
            gemm_cpu(true,
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
                                    paddings,
                                    strides,
                                    dilations,
                                    dout_batch);
        }
        }
        // bias and activate
        fill_bias_act(
            dout_batch, bias, chout, wout * hout, flag_bias, act_type);
    }
    if (!flag_1x1s1p1) free(col_data);
}
