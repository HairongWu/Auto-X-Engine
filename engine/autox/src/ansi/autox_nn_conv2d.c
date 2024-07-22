#include <stdint.h>
#include "autox_nn.h"

void conv_depthwise_3x3s2_p01_direct(
    const float *din,
    float *dout,
    int num,
    int ch_out,
    int h_out,
    int w_out,
    int ch_in,
    int h_in,
    int w_in,
    const float *weights,
    const float *bias,
    int pad,
    int flag_bias,
    const int act_type) {

  bool right = false;  // for right result

  float *zero_ptr = static_cast<float *>(
      malloc(max(w_in * sizeof(float), 8 * sizeof(float))));
  memset(zero_ptr, 0, max(w_in * sizeof(float), 8 * sizeof(float)));
  float *write_ptr =
      static_cast<float *>(malloc(w_out * sizeof(float)));

  //! prepare for processing right result
  int rmask_o[4] = {0};
  float rmaskr[8] = {-1.f, -1.f, -1.f, -1.f, -1.f, -1.f, -1.f, -1.f};
  int ro = w_out % 3;
  int col = w_out / 3;
  if (ro > 0) col++;
  if (ro > 0) {
    for (int i = 0; i < 4; i++) {
      if (i < ro) {
        rmask_o[i] = 0x80000000;
      }
    }
    right = true;
  }
  int ri = (w_in - (1 - pad)) % 6;
  // [pad == 0 && w_out == 3 && win == 8] ===>>> [ri == 1 && ro == 0]
  // add condition ro > 0 for avoiding wrong rmaskr when pad == 0
  if (ri > 0 && (ro > 0 || pad == 1)) {
    for (int i = 0; i < 8; i++) {
      if (i <= ri) {
        rmaskr[i] = -1.f;
      } else {
        rmaskr[i] = 1.f;
      }
    }
  }

  int size_in_channel = w_in * h_in;
  int size_out_channel = w_out * h_out;
  int w_stride = 9;

  __m128 zero = _mm_set1_ps(0.f);
  __m256 zero_256 = _mm256_set1_ps(0.f);

  for (int n = 0; n < num; ++n) {
    const float *din_batch = din + n * ch_in * size_in_channel;
    float *dout_batch = dout + n * ch_in * size_out_channel;

    for (int c = 0; c < ch_in; c++) {
      float *dout_ptr = dout_batch + c * size_out_channel;
      const float *din_ch_ptr = din_batch + c * size_in_channel;

      float bias_val = flag_bias ? bias[c] : 0.f;
      __m256 v_bias = _mm256_set1_ps(bias_val);
      const float *wei_ptr = weights + c * w_stride;

      const float *dr0 = din_ch_ptr;
      const float *dr1 = dr0 + w_in;
      const float *dr2 = dr1 + w_in;
      const float *dr3 = dr2 + w_in;
      const float *dr4 = dr3 + w_in;

      const float *din_ptr0 = dr0;
      const float *din_ptr1 = dr1;
      const float *din_ptr2 = dr2;
      const float *din_ptr3 = dr3;
      const float *din_ptr4 = dr4;

      float *doutr0 = dout_ptr;
      float *doutr1 = doutr0 + w_out;

      // for shift input
      __m256i shift_0 = _mm256_set_epi32(7, 7, 6, 5, 4, 3, 2, 1);
      __m256i shift_1 = _mm256_set_epi32(7, 7, 7, 6, 5, 4, 3, 2);
      __m256i shift_3 = _mm256_set_epi32(6, 5, 4, 3, 2, 1, 0, 7);

      for (int i = 0; i + (1 - pad) < h_in; i += 4) {
        din_ptr0 = dr0;
        din_ptr1 = dr1;
        din_ptr2 = dr2;
        din_ptr3 = dr3;
        din_ptr4 = dr4;

        doutr0 = dout_ptr;
        doutr1 = doutr0 + w_out;

        //! process top pad
        if (i == 0 && pad == 1) {
          din_ptr0 = zero_ptr;
          din_ptr1 = dr0;
          din_ptr2 = dr1;
          din_ptr3 = dr2;
          din_ptr4 = dr3;
          dr0 = dr3;
          dr1 = dr0 + w_in;
        } else {
          dr0 = dr4;
          dr1 = dr0 + w_in;
        }
        dr2 = dr1 + w_in;
        dr3 = dr2 + w_in;
        dr4 = dr3 + w_in;

        //! process bottom pad
        if (i + 4 + (1 - pad) > h_in) {
          switch (i + 4 + (1 - pad) - h_in) {
            case 4:
              din_ptr1 = zero_ptr;
            case 3:
              din_ptr2 = zero_ptr;
            case 2:
              din_ptr3 = zero_ptr;
            case 1:
              din_ptr4 = zero_ptr;
            default:
              break;
          }
        }

        //! process bottom remain
        if (i / 2 + 2 > h_out) {
          switch (i / 2 + 2 - h_out) {
            case 2:
              doutr0 = write_ptr;
            case 1:
              doutr1 = write_ptr;
            default:
              break;
          }
        }

        for (int j = 0; j < col; j += 1) {
          __m256 i0 = _mm256_loadu_ps(din_ptr0);
          __m256 i2 = _mm256_loadu_ps(din_ptr2);
          __m256 i1 = _mm256_loadu_ps(din_ptr1);
          __m256 i3 = _mm256_loadu_ps(din_ptr3);
          __m256 i4 = _mm256_loadu_ps(din_ptr4);

          //! process left pad
          if (j == 0 && pad == 1) {
            din_ptr0 += 5;
            din_ptr1 += 5;
            din_ptr2 += 5;
            din_ptr3 += 5;
            din_ptr4 += 5;
            i0 = _mm256_blend_ps(zero_256, i0, 0b01111111);
            i0 = _mm256_permutevar8x32_ps(i0, shift_3);
            i1 = _mm256_blend_ps(zero_256, i1, 0b01111111);
            i1 = _mm256_permutevar8x32_ps(i1, shift_3);
            i2 = _mm256_blend_ps(zero_256, i2, 0b01111111);
            i2 = _mm256_permutevar8x32_ps(i2, shift_3);
            i3 = _mm256_blend_ps(zero_256, i3, 0b01111111);
            i3 = _mm256_permutevar8x32_ps(i3, shift_3);
            i4 = _mm256_blend_ps(zero_256, i4, 0b01111111);
            i4 = _mm256_permutevar8x32_ps(i4, shift_3);
          } else {
            din_ptr0 += 6;
            din_ptr1 += 6;
            din_ptr2 += 6;
            din_ptr3 += 6;
            din_ptr4 += 6;
          }

          //! process right remain
          __m128i mask = _mm_setr_epi32(0x80000000, 0x80000000, 0x80000000, 0);
          if (j + 1 == col) {
            __m256 rmask_ri = _mm256_loadu_ps(rmaskr);
            i0 = _mm256_blendv_ps(zero_256, i0, rmask_ri);
            i1 = _mm256_blendv_ps(zero_256, i1, rmask_ri);
            i2 = _mm256_blendv_ps(zero_256, i2, rmask_ri);
            i3 = _mm256_blendv_ps(zero_256, i3, rmask_ri);
            i4 = _mm256_blendv_ps(zero_256, i4, rmask_ri);
            dout_ptr = dout_ptr + 2 * w_out;
            if (right) {
              mask = _mm_setr_epi32(
                  rmask_o[0], rmask_o[1], rmask_o[2], rmask_o[3]);
            }
          }

          __m256 wei_00 = _mm256_set1_ps(*(wei_ptr));
          __m256 wei_01 = _mm256_set1_ps(*(wei_ptr + 1));
          __m256 wei_02 = _mm256_set1_ps(*(wei_ptr + 2));

          // r0 row0
          __m256 res0 = _mm256_fmadd_ps(i0, wei_00, v_bias);
          __m256 tmp = _mm256_permutevar8x32_ps(i0, shift_0);
          res0 = _mm256_fmadd_ps(tmp, wei_01, res0);
          tmp = _mm256_permutevar8x32_ps(i0, shift_1);
          res0 = _mm256_fmadd_ps(tmp, wei_02, res0);

          // r1 row0
          __m256 res1 = _mm256_fmadd_ps(i2, wei_00, v_bias);
          tmp = _mm256_permutevar8x32_ps(i2, shift_0);
          res1 = _mm256_fmadd_ps(tmp, wei_01, res1);
          tmp = _mm256_permutevar8x32_ps(i2, shift_1);
          res1 = _mm256_fmadd_ps(tmp, wei_02, res1);

          __m256 wei_10 = _mm256_set1_ps(*(wei_ptr + 3));
          __m256 wei_11 = _mm256_set1_ps(*(wei_ptr + 4));
          __m256 wei_12 = _mm256_set1_ps(*(wei_ptr + 5));

          // r0 row0 + row1
          res0 = _mm256_fmadd_ps(i1, wei_10, res0);
          tmp = _mm256_permutevar8x32_ps(i1, shift_0);
          res0 = _mm256_fmadd_ps(tmp, wei_11, res0);
          tmp = _mm256_permutevar8x32_ps(i1, shift_1);
          res0 = _mm256_fmadd_ps(tmp, wei_12, res0);

          // r1 row0 + row1
          res1 = _mm256_fmadd_ps(i3, wei_10, res1);
          tmp = _mm256_permutevar8x32_ps(i3, shift_0);
          res1 = _mm256_fmadd_ps(tmp, wei_11, res1);
          tmp = _mm256_permutevar8x32_ps(i3, shift_1);
          res1 = _mm256_fmadd_ps(tmp, wei_12, res1);

          __m256 wei_20 = _mm256_set1_ps(*(wei_ptr + 6));
          __m256 wei_21 = _mm256_set1_ps(*(wei_ptr + 7));
          __m256 wei_22 = _mm256_set1_ps(*(wei_ptr + 8));

          // r0 row0 + row1 + row2
          res0 = _mm256_fmadd_ps(i2, wei_20, res0);
          tmp = _mm256_permutevar8x32_ps(i2, shift_0);
          res0 = _mm256_fmadd_ps(tmp, wei_21, res0);
          tmp = _mm256_permutevar8x32_ps(i2, shift_1);
          res0 = _mm256_fmadd_ps(tmp, wei_22, res0);

          // r1 row0 + row1 + row2
          res1 = _mm256_fmadd_ps(i4, wei_20, res1);
          tmp = _mm256_permutevar8x32_ps(i4, shift_0);
          res1 = _mm256_fmadd_ps(tmp, wei_21, res1);
          tmp = _mm256_permutevar8x32_ps(i4, shift_1);
          res1 = _mm256_fmadd_ps(tmp, wei_22, res1);

          __m256i shift_2 = _mm256_set_epi32(6, 4, 2, 0, 6, 4, 2, 0);
          __m256 r0 = _mm256_permutevar8x32_ps(res0, shift_2);
          __m128 r0_128 = _mm256_extractf128_ps(r0, 0);

          __m256 r1 = _mm256_permutevar8x32_ps(res1, shift_2);
          __m128 r1_128 = _mm256_extractf128_ps(r1, 0);

            if (act_type == 1) {
              r0_128 = _mm_max_ps(r0_128, zero);
              r1_128 = _mm_max_ps(r1_128, zero);
            } else if (act_type == 2) {
              __m128 six = _mm_set1_ps(6.f);
              r0_128 = _mm_min_ps(_mm_max_ps(r0_128, zero), six);
              r1_128 = _mm_min_ps(_mm_max_ps(r1_128, zero), six);
            } 
			/*else if (act_type == lite_api::ActivationType::kLeakyRelu) {
              __m128 negative_slope = _mm_set1_ps(act_param.Leaky_relu_alpha);
              r0_128 = _mm_add_ps(
                  _mm_and_ps(_mm_cmple_ps(zero, r0_128), r0_128),
                  _mm_mul_ps(_mm_and_ps(_mm_cmplt_ps(r0_128, zero), r0_128),
                             negative_slope));
              r1_128 = _mm_add_ps(
                  _mm_and_ps(_mm_cmple_ps(zero, r1_128), r1_128),
                  _mm_mul_ps(_mm_and_ps(_mm_cmplt_ps(r1_128, zero), r1_128),
                             negative_slope));*/
            } else if (act_type == kHardSwish) {
              __m128 vscale = _mm_set1_ps(1.0 / act_param.hard_swish_scale);
              __m128 voffset = _mm_set1_ps(act_param.hard_swish_offset);
              __m128 vthreshold = _mm_set1_ps(act_param.hard_swish_threshold);
              r0_128 = _mm_mul_ps(
                  _mm_min_ps(vthreshold,
                             _mm_max_ps(zero, _mm_add_ps(r0_128, voffset))),
                  _mm_mul_ps(r0_128, vscale));
              r1_128 = _mm_mul_ps(
                  _mm_min_ps(vthreshold,
                             _mm_max_ps(zero, _mm_add_ps(r1_128, voffset))),
                  _mm_mul_ps(r1_128, vscale));
            }

          _mm_maskstore_ps(doutr0, mask, r0_128);
          _mm_maskstore_ps(doutr1, mask, r1_128);

          doutr0 = doutr0 + 3;
          doutr1 = doutr1 + 3;
        }
      }
    }
  }
  free(zero_ptr);
  free(write_ptr);
}
void conv_depthwise_3x3s1_p01_direct(
    const float *din,
    float *dout,
    int num,
    int ch_out,
    int h_out,
    int w_out,
    int ch_in,
    int h_in,
    int w_in,
    const float *weights,
    const float *bias,
    int pad,
    int flag_bias,
    const int act_type) {

  bool right = false;

  float *zero_ptr = static_cast<float *>(
      malloc(max(w_in * sizeof(float), 8)));
  memset(zero_ptr, 0, max(w_in * sizeof(float), 8));
  float *write_ptr =
      static_cast<float *>(malloc(w_out * sizeof(float)));

  //! prepare for processing right result
  int rmask_o[8] = {0, 0, 0, 0, 0, 0, 0, 0};
  float rmaskr[8] = {1.f, 1.f, 1.f, 1.f, 1.f, 1.f, 1.f, 1.f};
  int r = w_out % 6;
  int col = w_out / 6;
  if (r > 0) col++;
  if (r > 0) {
    for (int i = 0; i < 8; i++) {
      if (i < r) {
        rmask_o[i] = 0x80000000;
      }
      if (i <= r + (1 - pad)) {
        rmaskr[i] = -1.f;
      }
    }
    right = true;
  } else {
    for (int i = 0; i < 7 + (1 - pad); i++) {
      rmaskr[i] = -1.f;
    }
  }

  __m256i shift_1 = _mm256_set_epi32(7, 7, 6, 5, 4, 3, 2, 1);
  __m256i shift_2 = _mm256_set_epi32(7, 7, 7, 6, 5, 4, 3, 2);
  __m256i shift_3 = _mm256_set_epi32(6, 5, 4, 3, 2, 1, 0, 7);

  int size_in_channel = w_in * h_in;
  int size_out_channel = w_out * h_out;
  int w_stride = 9;

  __m256 zero = _mm256_set1_ps(0.f);

  for (int n = 0; n < num; ++n) {
    const float *din_batch = din + n * ch_in * size_in_channel;
    float *dout_batch = dout + n * ch_in * size_out_channel;

    for (int c = 0; c < ch_in; c++) {
      float *dout_ptr = dout_batch + c * size_out_channel;
      const float *din_ch_ptr = din_batch + c * size_in_channel;

      float bias_val = flag_bias ? bias[c] : 0.f;
      __m256 v_bias = _mm256_set1_ps(bias_val);
      const float *wei_ptr = weights + c * w_stride;

      float *doutr0 = dout_ptr;
      float *doutr1 = doutr0 + w_out;
      float *doutr2 = doutr1 + w_out;
      float *doutr3 = doutr2 + w_out;

      const float *dr0 = din_ch_ptr;
      const float *dr1 = dr0 + w_in;
      const float *dr2 = dr1 + w_in;
      const float *dr3 = dr2 + w_in;
      const float *dr4 = dr3 + w_in;
      const float *dr5 = dr4 + w_in;

      const float *din_ptr0 = dr0;
      const float *din_ptr1 = dr1;
      const float *din_ptr2 = dr2;
      const float *din_ptr3 = dr3;
      const float *din_ptr4 = dr4;
      const float *din_ptr5 = dr5;

      for (int i = 0; i < h_out; i += 4) {
        din_ptr0 = dr0;
        din_ptr1 = dr1;
        din_ptr2 = dr2;
        din_ptr3 = dr3;
        din_ptr4 = dr4;
        din_ptr5 = dr5;

        doutr0 = dout_ptr;
        doutr1 = doutr0 + w_out;
        doutr2 = doutr1 + w_out;
        doutr3 = doutr2 + w_out;

        //! process top pad
        if (i == 0 && pad == 1) {
          din_ptr0 = zero_ptr;
          din_ptr1 = dr0;
          din_ptr2 = dr1;
          din_ptr3 = dr2;
          din_ptr4 = dr3;
          din_ptr5 = dr4;
          dr0 = dr3;
          dr1 = dr4;
          dr2 = dr5;
        } else {
          dr0 = dr4;
          dr1 = dr5;
          dr2 = dr1 + w_in;
        }
        dr3 = dr2 + w_in;
        dr4 = dr3 + w_in;
        dr5 = dr4 + w_in;

        //! process bottom pad
        if (i + 5 + (1 - pad) > h_in) {
          switch (i + 5 + (1 - pad) - h_in) {
            case 5:
              din_ptr1 = zero_ptr;
            case 4:
              din_ptr2 = zero_ptr;
            case 3:
              din_ptr3 = zero_ptr;
            case 2:
              din_ptr4 = zero_ptr;
            case 1:
              din_ptr5 = zero_ptr;
            default:
              break;
          }
        }

        //! process bottom remain
        if (i + 4 > h_out) {
          switch (i + 4 - h_out) {
            case 3:
              doutr1 = write_ptr;
            case 2:
              doutr2 = write_ptr;
            case 1:
              doutr3 = write_ptr;
            default:
              break;
          }
        }

        for (int j = 0; j < col; j += 1) {
          __m256 i0 = _mm256_loadu_ps(din_ptr0);
          __m256 i1 = _mm256_loadu_ps(din_ptr1);
          __m256 i2 = _mm256_loadu_ps(din_ptr2);
          __m256 i3 = _mm256_loadu_ps(din_ptr3);
          __m256 i4 = _mm256_loadu_ps(din_ptr4);
          __m256 i5 = _mm256_loadu_ps(din_ptr5);

          //! process left pad
          if (j == 0 && pad == 1) {
            din_ptr0 += 5;
            din_ptr1 += 5;
            din_ptr2 += 5;
            din_ptr3 += 5;
            din_ptr4 += 5;
            din_ptr5 += 5;
            i0 = _mm256_blend_ps(zero, i0, 0b01111111);
            i0 = _mm256_permutevar8x32_ps(i0, shift_3);
            i1 = _mm256_blend_ps(zero, i1, 0b01111111);
            i1 = _mm256_permutevar8x32_ps(i1, shift_3);
            i2 = _mm256_blend_ps(zero, i2, 0b01111111);
            i2 = _mm256_permutevar8x32_ps(i2, shift_3);
            i3 = _mm256_blend_ps(zero, i3, 0b01111111);
            i3 = _mm256_permutevar8x32_ps(i3, shift_3);
            i4 = _mm256_blend_ps(zero, i4, 0b01111111);
            i4 = _mm256_permutevar8x32_ps(i4, shift_3);
            i5 = _mm256_blend_ps(zero, i5, 0b01111111);
            i5 = _mm256_permutevar8x32_ps(i5, shift_3);
          } else {
            din_ptr0 += 6;
            din_ptr1 += 6;
            din_ptr2 += 6;
            din_ptr3 += 6;
            din_ptr4 += 6;
            din_ptr5 += 6;
          }

          //! process right remain
          __m256i smask_ = _mm256_set_epi32(0,
                                            0,
                                            0x80000000,
                                            0x80000000,
                                            0x80000000,
                                            0x80000000,
                                            0x80000000,
                                            0x80000000);
          if (j + 1 == col) {
            __m256 rmask_i = _mm256_loadu_ps(rmaskr);
            i0 = _mm256_blendv_ps(zero, i0, rmask_i);
            i1 = _mm256_blendv_ps(zero, i1, rmask_i);
            i2 = _mm256_blendv_ps(zero, i2, rmask_i);
            i3 = _mm256_blendv_ps(zero, i3, rmask_i);
            i4 = _mm256_blendv_ps(zero, i4, rmask_i);
            i5 = _mm256_blendv_ps(zero, i5, rmask_i);
            dout_ptr = dout_ptr + 4 * w_out;
            if (right) {
              smask_ = _mm256_set_epi32(rmask_o[7],
                                        rmask_o[6],
                                        rmask_o[5],
                                        rmask_o[4],
                                        rmask_o[3],
                                        rmask_o[2],
                                        rmask_o[1],
                                        rmask_o[0]);
            }
          }

          __m256 wei_00 = _mm256_set1_ps(*(wei_ptr));
          __m256 wei_01 = _mm256_set1_ps(*(wei_ptr + 1));
          __m256 wei_02 = _mm256_set1_ps(*(wei_ptr + 2));

          // r0 row0
          __m256 r0 = _mm256_fmadd_ps(i0, wei_00, v_bias);
          __m256 tmp = _mm256_permutevar8x32_ps(i0, shift_1);
          r0 = _mm256_fmadd_ps(tmp, wei_01, r0);
          tmp = _mm256_permutevar8x32_ps(i0, shift_2);
          r0 = _mm256_fmadd_ps(tmp, wei_02, r0);

          // r1 row0
          __m256 r1 = _mm256_fmadd_ps(i1, wei_00, v_bias);
          tmp = _mm256_permutevar8x32_ps(i1, shift_1);
          r1 = _mm256_fmadd_ps(tmp, wei_01, r1);
          tmp = _mm256_permutevar8x32_ps(i1, shift_2);
          r1 = _mm256_fmadd_ps(tmp, wei_02, r1);

          // r2 row0
          __m256 r2 = _mm256_fmadd_ps(i2, wei_00, v_bias);
          tmp = _mm256_permutevar8x32_ps(i2, shift_1);
          r2 = _mm256_fmadd_ps(tmp, wei_01, r2);
          tmp = _mm256_permutevar8x32_ps(i2, shift_2);
          r2 = _mm256_fmadd_ps(tmp, wei_02, r2);

          // r3 row0
          __m256 r3 = _mm256_fmadd_ps(i3, wei_00, v_bias);
          tmp = _mm256_permutevar8x32_ps(i3, shift_1);
          r3 = _mm256_fmadd_ps(tmp, wei_01, r3);
          tmp = _mm256_permutevar8x32_ps(i3, shift_2);
          r3 = _mm256_fmadd_ps(tmp, wei_02, r3);

          __m256 wei_10 = _mm256_set1_ps(*(wei_ptr + 3));
          __m256 wei_11 = _mm256_set1_ps(*(wei_ptr + 4));
          __m256 wei_12 = _mm256_set1_ps(*(wei_ptr + 5));

          // r0 row0 + row1
          r0 = _mm256_fmadd_ps(i1, wei_10, r0);
          tmp = _mm256_permutevar8x32_ps(i1, shift_1);
          r0 = _mm256_fmadd_ps(tmp, wei_11, r0);
          tmp = _mm256_permutevar8x32_ps(i1, shift_2);
          r0 = _mm256_fmadd_ps(tmp, wei_12, r0);

          // r1 row0 + row1
          r1 = _mm256_fmadd_ps(i2, wei_10, r1);
          tmp = _mm256_permutevar8x32_ps(i2, shift_1);
          r1 = _mm256_fmadd_ps(tmp, wei_11, r1);
          tmp = _mm256_permutevar8x32_ps(i2, shift_2);
          r1 = _mm256_fmadd_ps(tmp, wei_12, r1);

          // r2 row0 + row1
          r2 = _mm256_fmadd_ps(i3, wei_10, r2);
          tmp = _mm256_permutevar8x32_ps(i3, shift_1);
          r2 = _mm256_fmadd_ps(tmp, wei_11, r2);
          tmp = _mm256_permutevar8x32_ps(i3, shift_2);
          r2 = _mm256_fmadd_ps(tmp, wei_12, r2);

          // r3 row0 + row1
          r3 = _mm256_fmadd_ps(i4, wei_10, r3);
          tmp = _mm256_permutevar8x32_ps(i4, shift_1);
          r3 = _mm256_fmadd_ps(tmp, wei_11, r3);
          tmp = _mm256_permutevar8x32_ps(i4, shift_2);
          r3 = _mm256_fmadd_ps(tmp, wei_12, r3);

          __m256 wei_20 = _mm256_set1_ps(*(wei_ptr + 6));
          __m256 wei_21 = _mm256_set1_ps(*(wei_ptr + 7));
          __m256 wei_22 = _mm256_set1_ps(*(wei_ptr + 8));

          // r0 row0 + row1 + row2
          r0 = _mm256_fmadd_ps(i2, wei_20, r0);
          tmp = _mm256_permutevar8x32_ps(i2, shift_1);
          r0 = _mm256_fmadd_ps(tmp, wei_21, r0);
          tmp = _mm256_permutevar8x32_ps(i2, shift_2);
          r0 = _mm256_fmadd_ps(tmp, wei_22, r0);

          // r1 row0 + row1 + row2
          r1 = _mm256_fmadd_ps(i3, wei_20, r1);
          tmp = _mm256_permutevar8x32_ps(i3, shift_1);
          r1 = _mm256_fmadd_ps(tmp, wei_21, r1);
          tmp = _mm256_permutevar8x32_ps(i3, shift_2);
          r1 = _mm256_fmadd_ps(tmp, wei_22, r1);

          // r2 row0 + row1 + row2
          r2 = _mm256_fmadd_ps(i4, wei_20, r2);
          tmp = _mm256_permutevar8x32_ps(i4, shift_1);
          r2 = _mm256_fmadd_ps(tmp, wei_21, r2);
          tmp = _mm256_permutevar8x32_ps(i4, shift_2);
          r2 = _mm256_fmadd_ps(tmp, wei_22, r2);

          // r3 row0 + row1 + row2
          r3 = _mm256_fmadd_ps(i5, wei_20, r3);
          tmp = _mm256_permutevar8x32_ps(i5, shift_1);
          r3 = _mm256_fmadd_ps(tmp, wei_21, r3);
          tmp = _mm256_permutevar8x32_ps(i5, shift_2);
          r3 = _mm256_fmadd_ps(tmp, wei_22, r3);

            if (act_type == kRelu) {
              r0 = _mm256_max_ps(r0, zero);
              r1 = _mm256_max_ps(r1, zero);
              r2 = _mm256_max_ps(r2, zero);
              r3 = _mm256_max_ps(r3, zero);
            } else if (act_type == kRelu6) {
              __m256 six = _mm256_set1_ps(6.0);
              r0 = _mm256_min_ps(_mm256_max_ps(r0, zero), six);
              r1 = _mm256_min_ps(_mm256_max_ps(r1, zero), six);
              r2 = _mm256_min_ps(_mm256_max_ps(r2, zero), six);
              r3 = _mm256_min_ps(_mm256_max_ps(r3, zero), six);
            } 
			/*else if (act_type == lite_api::ActivationType::kLeakyRelu) {
              __m256 negative_slope =
                  _mm256_set1_ps(act_param.Leaky_relu_alpha);
              r0 = _mm256_add_ps(
                  _mm256_and_ps(_mm256_cmp_ps(zero, r0, 18), r0),
                  _mm256_mul_ps(_mm256_and_ps(_mm256_cmp_ps(r0, zero, 17), r0),
                                negative_slope));
              r1 = _mm256_add_ps(
                  _mm256_and_ps(_mm256_cmp_ps(zero, r1, 18), r1),
                  _mm256_mul_ps(_mm256_and_ps(_mm256_cmp_ps(r1, zero, 17), r1),
                                negative_slope));
              r2 = _mm256_add_ps(
                  _mm256_and_ps(_mm256_cmp_ps(zero, r2, 18), r2),
                  _mm256_mul_ps(_mm256_and_ps(_mm256_cmp_ps(r2, zero, 17), r2),
                                negative_slope));
              r3 = _mm256_add_ps(
                  _mm256_and_ps(_mm256_cmp_ps(zero, r3, 18), r3),
                  _mm256_mul_ps(_mm256_and_ps(_mm256_cmp_ps(r3, zero, 17), r3),
                                negative_slope));*/
            } else if (act_type == kHardSwish) {
              __m256 vscale = _mm256_set1_ps(1.0 / act_param.hard_swish_scale);
              __m256 voffset = _mm256_set1_ps(act_param.hard_swish_offset);
              __m256 vthreshold =
                  _mm256_set1_ps(act_param.hard_swish_threshold);
              r0 = _mm256_mul_ps(
                  _mm256_min_ps(
                      vthreshold,
                      _mm256_max_ps(zero, _mm256_add_ps(r0, voffset))),
                  _mm256_mul_ps(r0, vscale));
              r1 = _mm256_mul_ps(
                  _mm256_min_ps(
                      vthreshold,
                      _mm256_max_ps(zero, _mm256_add_ps(r1, voffset))),
                  _mm256_mul_ps(r1, vscale));
              r2 = _mm256_mul_ps(
                  _mm256_min_ps(
                      vthreshold,
                      _mm256_max_ps(zero, _mm256_add_ps(r2, voffset))),
                  _mm256_mul_ps(r2, vscale));
              r3 = _mm256_mul_ps(
                  _mm256_min_ps(
                      vthreshold,
                      _mm256_max_ps(zero, _mm256_add_ps(r3, voffset))),
                  _mm256_mul_ps(r3, vscale));
            } else {
              LOG(FATAL) << "[X86] activation type: "
                         << static_cast<int>(act_type) << "not supported";
            }


          _mm256_maskstore_ps(doutr0, smask_, r0);
          _mm256_maskstore_ps(doutr1, smask_, r1);
          _mm256_maskstore_ps(doutr2, smask_, r2);
          _mm256_maskstore_ps(doutr3, smask_, r3);

          doutr0 = doutr0 + 6;
          doutr1 = doutr1 + 6;
          doutr2 = doutr2 + 6;
          doutr3 = doutr3 + 6;
        }
      }
    }
  }

  free(zero_ptr);
  free(write_ptr);
}

#define loadu_ps(a) _mm256_loadu_ps(a)
#define fmadd_ps(a, b, c) _mm256_fmadd_ps(a, b, c)
#define storeu_ps(a, b) _mm256_storeu_ps(a, b)
#define setzero_ps() _mm256_setzero_ps()
#define max_ps(a, b) _mm256_max_ps(a, b)
#define min_ps(a, b) _mm256_min_ps(a, b)
#define set1_ps(a) _mm256_set1_ps(a)
#define mul_ps(a, b) _mm256_mul_ps(a, b)
#define cmp_ps(a, b, c) _mm256_cmp_ps(a, b, c)
#define blendv_ps(a, b, c) _mm256_blendv_ps(a, b, c)
#define add_ps(a, b) _mm256_add_ps(a, b)
#define block_channel 8
#define Type __m256

#define ROUNDUP(a, b) ((((a) + (b)-1) / (b)) * (b))

void transpose4_ps(__m128* row0,
	__m128* row1,
	__m128* row2,
	__m128* row3) {
	__m128 tmp3, tmp2, tmp1, tmp0;
	tmp0 = _mm_unpacklo_ps((*row0), (*row1));
	tmp2 = _mm_unpacklo_ps((*row2), (*row3));
	tmp1 = _mm_unpackhi_ps((*row0), (*row1));
	tmp3 = _mm_unpackhi_ps((*row2), (*row3));
	*row0 = _mm_movelh_ps(tmp0, tmp2);
	*row1 = _mm_movehl_ps(tmp2, tmp0);
	*row2 = _mm_movelh_ps(tmp1, tmp3);
	*row3 = _mm_movehl_ps(tmp3, tmp1);
}

void packC4_common(const float* din,
	float* dout,
	const int pad,
	int h_in,
	int w_in,
	int channel) {
	int top = pad;
	int bottom = pad;
	int left = pad;
	int right = pad;
	int w_out = (w_in + left + right);
	int h_out = (h_in + top + bottom);
	int block_channel = 4;
	const float* din_init = din;
	float* dout_init = dout;

	for (int c = 0; c < channel; c += block_channel) {
		din = din_init + c * h_in * w_in;
		dout = dout_init + c * w_out * h_out;

		memset(dout, 0, top * w_out * block_channel * sizeof(float));
		float* dout_block = dout + top * w_out * block_channel;

		for (int i = 0; i < h_in; i++) {
			float* douth = dout_block + i * w_out * block_channel;
			const float* dinh = din + i * w_in;
			memset(douth, 0, left * block_channel * sizeof(float));
			douth += left * block_channel;
			int kernel_size = h_in * w_in;
			float* dinr0 = (float*)dinh;
			float* dinr1 = dinr0 + kernel_size;
			float* dinr2 = dinr1 + kernel_size;
			float* dinr3 = dinr2 + kernel_size;

			int j = 0;
			if (c + 3 < channel) {
				for (; j + 3 < w_in; j += 4) {
					__m128 _row0 = _mm_loadu_ps(dinr0);
					__m128 _row1 = _mm_loadu_ps(dinr1);
					__m128 _row2 = _mm_loadu_ps(dinr2);
					__m128 _row3 = _mm_loadu_ps(dinr3);
					transpose4_ps(&_row0, &_row1, &_row2, &_row3);
					_mm_storeu_ps(douth, _row0);
					_mm_storeu_ps(douth + 4, _row1);
					_mm_storeu_ps(douth + 8, _row2);
					_mm_storeu_ps(douth + 12, _row3);
					dinr0 += 4;
					dinr1 += 4;
					dinr2 += 4;
					dinr3 += 4;
					douth += 16;
				}

				for (; j < w_in; j++) {
					douth[0] = *dinr0++;
					douth[1] = *dinr1++;
					douth[2] = *dinr2++;
					douth[3] = *dinr3++;
					douth += 4;
				}
			}
			else {
				__m128 _row0 = _mm_setzero_ps();
				__m128 _row1 = _mm_setzero_ps();
				__m128 _row2 = _mm_setzero_ps();
				__m128 _row3 = _mm_setzero_ps();
				for (; j + 3 < w_in; j += 4) {
					_row0 = _mm_loadu_ps(dinr0);
					if (channel - c > 1) _row1 = _mm_loadu_ps(dinr1);
					if (channel - c > 2) _row2 = _mm_loadu_ps(dinr2);
					if (channel - c > 3) _row3 = _mm_loadu_ps(dinr3);
					transpose4_ps(&_row0, &_row1, &_row2, &_row3);
					_mm_storeu_ps(douth, _row0);
					_mm_storeu_ps(douth + 4, _row1);
					_mm_storeu_ps(douth + 8, _row2);
					_mm_storeu_ps(douth + 12, _row3);
					dinr0 += 4;
					dinr1 += 4;
					dinr2 += 4;
					dinr3 += 4;
					douth += 16;
				}

				for (; j < w_in; j++) {
					douth[0] = *dinr0++;
					douth[1] = channel - c > 1 ? *dinr1++ : 0;
					douth[2] = channel - c > 2 ? *dinr2++ : 0;
					douth[3] = channel - c > 3 ? *dinr3++ : 0;
					douth += 4;
				}
			}
			memset(douth, 0, right * block_channel * sizeof(float));
		}
		memset(dout + (h_in + top) * w_out * block_channel,
			0,
			bottom * w_out * block_channel * sizeof(float));
	}
}

void unpackC4_common(const float* din,
	float* dout,
	int size_out_channel,
	int channel) {
	int block_channel = 4;
	float* dout_init = dout;

	for (int c = 0; c < channel; c += block_channel) {
		dout = dout_init + c * size_out_channel;
		float* doutr0 = dout;
		float* doutr1 = doutr0 + size_out_channel;
		float* doutr2 = doutr1 + size_out_channel;
		float* doutr3 = doutr2 + size_out_channel;
		int j = 0;
		if (c + 3 < channel) {
			for (; j + 3 < size_out_channel; j += 4) {
				__m128 _row0 = _mm_loadu_ps(din);
				__m128 _row1 = _mm_loadu_ps(din + 4);
				__m128 _row2 = _mm_loadu_ps(din + 8);
				__m128 _row3 = _mm_loadu_ps(din + 12);
				transpose4_ps(&_row0, &_row1, &_row2, &_row3);
				_mm_storeu_ps(doutr0, _row0);
				_mm_storeu_ps(doutr1, _row1);
				_mm_storeu_ps(doutr2, _row2);
				_mm_storeu_ps(doutr3, _row3);
				doutr0 += 4;
				doutr1 += 4;
				doutr2 += 4;
				doutr3 += 4;
				din += 16;
			}

			for (; j < size_out_channel; j++) {
				*doutr0++ = *din++;
				*doutr1++ = *din++;
				*doutr2++ = *din++;
				*doutr3++ = *din++;
			}
		}
		else {
			for (; j + 3 < size_out_channel; j += 4) {
				__m128 _row0 = _mm_loadu_ps(din);
				__m128 _row1 = _mm_loadu_ps(din + 4);
				__m128 _row2 = _mm_loadu_ps(din + 8);
				__m128 _row3 = _mm_loadu_ps(din + 12);
				transpose4_ps(&_row0, &_row1, &_row2, &_row3);
				_mm_storeu_ps(doutr0, _row0);
				if (channel - c > 1) _mm_storeu_ps(doutr1, _row1);
				if (channel - c > 2) _mm_storeu_ps(doutr2, _row2);
				if (channel - c > 3) _mm_storeu_ps(doutr3, _row3);
				doutr0 += 4;
				doutr1 += 4;
				doutr2 += 4;
				doutr3 += 4;
				din += 16;
			}

			for (; j < size_out_channel; j++) {
				*doutr0++ = *din;
				if (channel - c > 1) *doutr1++ = *(din + 1);
				if (channel - c > 2) *doutr2++ = *(din + 2);
				if (channel - c > 3) *doutr3++ = *(din + 3);
				din += 4;
			}
		}
	}
}

void conv_depthwise_5x5s1(const float* din,
                          float* dout,
                          int num,
                          int ch_out,
                          int h_out,
                          int w_out,
                          int ch_in,
                          int h_in,
                          int w_in,
                          const float* weights,
                          const float* bias,
                          int pad,
                          int flag_bias,
                          const int act_type) {
	const int block_channel = 4;
  int size_in_channel = w_in * h_in;
  int size_out_channel = w_out * h_out;
  int in_len = block_channel * (2 * pad + w_in);

  int channel_num = ROUNDUP(ch_in, block_channel);
  float* pack_weight = (float*)(
      malloc(channel_num * 5 * 5 * sizeof(float)));
  float* pack_input = (float*)(malloc(
      (h_in + 2 * pad) * (w_in + 2 * pad) * block_channel * sizeof(float)));
  float* pack_out = (float*)(malloc(h_out * w_out * block_channel * sizeof(float)));

  packC8_common(weights, pack_weight, 0, 5, 5, ch_in);

  for (int n = 0; n < num; n++) {
    const float* din_batch = din + n * ch_in * size_in_channel;
    float* dout_batch = dout + n * ch_out * size_out_channel;

    for (int c = 0; c < ch_out; c += block_channel) {
      int real_block_channel = min(block_channel, ch_out - c);
      float* dout_ptr = dout_batch + c * size_out_channel;
      float* din_ptr = (float*)din_batch + c * size_in_channel;
      float* weights_data = pack_weight + c * 5 * 5;

      packC8_common(din_ptr,
                    pack_input,
                    {pad, pad, pad, pad},
                    h_in,
                    w_in,
                    real_block_channel);

      float bias_ptr[4] = {0.f};
      if (flag_bias) {
        for (int i = 0; i < block_channel; i++) {
          if (real_block_channel > i) {
            bias_ptr[i] = *(bias + c + i);
          }
        }
      }

      Type _bias = loadu_ps(bias_ptr);

      for (int i = 0; i < h_out; i++) {
        const float* block_inr0 = pack_input + i * in_len;
        const float* block_inr1 = block_inr0 + in_len;
        const float* block_inr2 = block_inr1 + in_len;
        const float* block_inr3 = block_inr2 + in_len;
        const float* block_inr4 = block_inr3 + in_len;
        int j = 0;
        float* dout_block = pack_out + i * w_out * block_channel;
        for (; j + 3 < w_out; j += 4) {
          Type i00 = loadu_ps(block_inr0);
          Type i01 = loadu_ps(block_inr0 + 1 * block_channel);
          Type i02 = loadu_ps(block_inr0 + 2 * block_channel);
          Type i03 = loadu_ps(block_inr0 + 3 * block_channel);
          Type i04 = loadu_ps(block_inr0 + 4 * block_channel);
          Type i05 = loadu_ps(block_inr0 + 5 * block_channel);
          Type i06 = loadu_ps(block_inr0 + 6 * block_channel);
          Type i07 = loadu_ps(block_inr0 + 7 * block_channel);

          Type w00 = loadu_ps(weights_data);
          Type r0 = fmadd_ps(i00, w00, _bias);
          Type r1 = fmadd_ps(i01, w00, _bias);
          Type r2 = fmadd_ps(i02, w00, _bias);
          Type r3 = fmadd_ps(i03, w00, _bias);

          Type w01 = loadu_ps(weights_data + block_channel);
          r0 = fmadd_ps(i01, w01, r0);
          r1 = fmadd_ps(i02, w01, r1);
          r2 = fmadd_ps(i03, w01, r2);
          r3 = fmadd_ps(i04, w01, r3);

          Type w02 = loadu_ps(weights_data + 2 * block_channel);
          r0 = fmadd_ps(i02, w02, r0);
          r1 = fmadd_ps(i03, w02, r1);
          r2 = fmadd_ps(i04, w02, r2);
          r3 = fmadd_ps(i05, w02, r3);

          Type w03 = loadu_ps(weights_data + 3 * block_channel);
          r0 = fmadd_ps(i03, w03, r0);
          r1 = fmadd_ps(i04, w03, r1);
          r2 = fmadd_ps(i05, w03, r2);
          r3 = fmadd_ps(i06, w03, r3);

          Type w04 = loadu_ps(weights_data + 4 * block_channel);
          r0 = fmadd_ps(i04, w04, r0);
          r1 = fmadd_ps(i05, w04, r1);
          r2 = fmadd_ps(i06, w04, r2);
          r3 = fmadd_ps(i07, w04, r3);

          Type i10 = loadu_ps(block_inr1);
          Type i11 = loadu_ps(block_inr1 + 1 * block_channel);
          Type i12 = loadu_ps(block_inr1 + 2 * block_channel);
          Type i13 = loadu_ps(block_inr1 + 3 * block_channel);
          Type i14 = loadu_ps(block_inr1 + 4 * block_channel);
          Type i15 = loadu_ps(block_inr1 + 5 * block_channel);
          Type i16 = loadu_ps(block_inr1 + 6 * block_channel);
          Type i17 = loadu_ps(block_inr1 + 7 * block_channel);

          Type w10 = loadu_ps(weights_data + 5 * block_channel);
          r0 = fmadd_ps(i10, w10, r0);
          r1 = fmadd_ps(i11, w10, r1);
          r2 = fmadd_ps(i12, w10, r2);
          r3 = fmadd_ps(i13, w10, r3);

          Type w11 = loadu_ps(weights_data + 6 * block_channel);
          r0 = fmadd_ps(i11, w11, r0);
          r1 = fmadd_ps(i12, w11, r1);
          r2 = fmadd_ps(i13, w11, r2);
          r3 = fmadd_ps(i14, w11, r3);

          Type w12 = loadu_ps(weights_data + 7 * block_channel);
          r0 = fmadd_ps(i12, w12, r0);
          r1 = fmadd_ps(i13, w12, r1);
          r2 = fmadd_ps(i14, w12, r2);
          r3 = fmadd_ps(i15, w12, r3);

          Type w13 = loadu_ps(weights_data + 8 * block_channel);
          r0 = fmadd_ps(i13, w13, r0);
          r1 = fmadd_ps(i14, w13, r1);
          r2 = fmadd_ps(i15, w13, r2);
          r3 = fmadd_ps(i16, w13, r3);

          Type w14 = loadu_ps(weights_data + 9 * block_channel);
          r0 = fmadd_ps(i14, w14, r0);
          r1 = fmadd_ps(i15, w14, r1);
          r2 = fmadd_ps(i16, w14, r2);
          r3 = fmadd_ps(i17, w14, r3);

          Type i20 = loadu_ps(block_inr2);
          Type i21 = loadu_ps(block_inr2 + 1 * block_channel);
          Type i22 = loadu_ps(block_inr2 + 2 * block_channel);
          Type i23 = loadu_ps(block_inr2 + 3 * block_channel);
          Type i24 = loadu_ps(block_inr2 + 4 * block_channel);
          Type i25 = loadu_ps(block_inr2 + 5 * block_channel);
          Type i26 = loadu_ps(block_inr2 + 6 * block_channel);
          Type i27 = loadu_ps(block_inr2 + 7 * block_channel);

          Type w20 = loadu_ps(weights_data + 10 * block_channel);
          r0 = fmadd_ps(i20, w20, r0);
          r1 = fmadd_ps(i21, w20, r1);
          r2 = fmadd_ps(i22, w20, r2);
          r3 = fmadd_ps(i23, w20, r3);

          Type w21 = loadu_ps(weights_data + 11 * block_channel);
          r0 = fmadd_ps(i21, w21, r0);
          r1 = fmadd_ps(i22, w21, r1);
          r2 = fmadd_ps(i23, w21, r2);
          r3 = fmadd_ps(i24, w21, r3);

          Type w22 = loadu_ps(weights_data + 12 * block_channel);
          r0 = fmadd_ps(i22, w22, r0);
          r1 = fmadd_ps(i23, w22, r1);
          r2 = fmadd_ps(i24, w22, r2);
          r3 = fmadd_ps(i25, w22, r3);

          Type w23 = loadu_ps(weights_data + 13 * block_channel);
          r0 = fmadd_ps(i23, w23, r0);
          r1 = fmadd_ps(i24, w23, r1);
          r2 = fmadd_ps(i25, w23, r2);
          r3 = fmadd_ps(i26, w23, r3);

          Type w24 = loadu_ps(weights_data + 14 * block_channel);
          r0 = fmadd_ps(i24, w24, r0);
          r1 = fmadd_ps(i25, w24, r1);
          r2 = fmadd_ps(i26, w24, r2);
          r3 = fmadd_ps(i27, w24, r3);

          Type i30 = loadu_ps(block_inr3);
          Type i31 = loadu_ps(block_inr3 + 1 * block_channel);
          Type i32 = loadu_ps(block_inr3 + 2 * block_channel);
          Type i33 = loadu_ps(block_inr3 + 3 * block_channel);
          Type i34 = loadu_ps(block_inr3 + 4 * block_channel);
          Type i35 = loadu_ps(block_inr3 + 5 * block_channel);
          Type i36 = loadu_ps(block_inr3 + 6 * block_channel);
          Type i37 = loadu_ps(block_inr3 + 7 * block_channel);

          Type w30 = loadu_ps(weights_data + 15 * block_channel);
          r0 = fmadd_ps(i30, w30, r0);
          r1 = fmadd_ps(i31, w30, r1);
          r2 = fmadd_ps(i32, w30, r2);
          r3 = fmadd_ps(i33, w30, r3);

          Type w31 = loadu_ps(weights_data + 16 * block_channel);
          r0 = fmadd_ps(i31, w31, r0);
          r1 = fmadd_ps(i32, w31, r1);
          r2 = fmadd_ps(i33, w31, r2);
          r3 = fmadd_ps(i34, w31, r3);

          Type w32 = loadu_ps(weights_data + 17 * block_channel);
          r0 = fmadd_ps(i32, w32, r0);
          r1 = fmadd_ps(i33, w32, r1);
          r2 = fmadd_ps(i34, w32, r2);
          r3 = fmadd_ps(i35, w32, r3);

          Type w33 = loadu_ps(weights_data + 18 * block_channel);
          r0 = fmadd_ps(i33, w33, r0);
          r1 = fmadd_ps(i34, w33, r1);
          r2 = fmadd_ps(i35, w33, r2);
          r3 = fmadd_ps(i36, w33, r3);

          Type w34 = loadu_ps(weights_data + 19 * block_channel);
          r0 = fmadd_ps(i34, w34, r0);
          r1 = fmadd_ps(i35, w34, r1);
          r2 = fmadd_ps(i36, w34, r2);
          r3 = fmadd_ps(i37, w34, r3);

          Type i40 = loadu_ps(block_inr4);
          Type i41 = loadu_ps(block_inr4 + 1 * block_channel);
          Type i42 = loadu_ps(block_inr4 + 2 * block_channel);
          Type i43 = loadu_ps(block_inr4 + 3 * block_channel);
          Type i44 = loadu_ps(block_inr4 + 4 * block_channel);
          Type i45 = loadu_ps(block_inr4 + 5 * block_channel);
          Type i46 = loadu_ps(block_inr4 + 6 * block_channel);
          Type i47 = loadu_ps(block_inr4 + 7 * block_channel);

          Type w40 = loadu_ps(weights_data + 20 * block_channel);
          r0 = fmadd_ps(i40, w40, r0);
          r1 = fmadd_ps(i41, w40, r1);
          r2 = fmadd_ps(i42, w40, r2);
          r3 = fmadd_ps(i43, w40, r3);

          Type w41 = loadu_ps(weights_data + 21 * block_channel);
          r0 = fmadd_ps(i41, w41, r0);
          r1 = fmadd_ps(i42, w41, r1);
          r2 = fmadd_ps(i43, w41, r2);
          r3 = fmadd_ps(i44, w41, r3);

          Type w42 = loadu_ps(weights_data + 22 * block_channel);
          r0 = fmadd_ps(i42, w42, r0);
          r1 = fmadd_ps(i43, w42, r1);
          r2 = fmadd_ps(i44, w42, r2);
          r3 = fmadd_ps(i45, w42, r3);

          Type w43 = loadu_ps(weights_data + 23 * block_channel);
          r0 = fmadd_ps(i43, w43, r0);
          r1 = fmadd_ps(i44, w43, r1);
          r2 = fmadd_ps(i45, w43, r2);
          r3 = fmadd_ps(i46, w43, r3);

          Type w44 = loadu_ps(weights_data + 24 * block_channel);
          r0 = fmadd_ps(i44, w44, r0);
          r1 = fmadd_ps(i45, w44, r1);
          r2 = fmadd_ps(i46, w44, r2);
          r3 = fmadd_ps(i47, w44, r3);

          Type zero = setzero_ps();

            if (act_type == kRelu) {
              r0 = max_ps(r0, zero);
              r1 = max_ps(r1, zero);
              r2 = max_ps(r2, zero);
              r3 = max_ps(r3, zero);
            } else if (act_type == kRelu6) {
              Type six = set1_ps(6.0);
              r0 = min_ps(max_ps(r0, zero), six);
              r1 = min_ps(max_ps(r1, zero), six);
              r2 = min_ps(max_ps(r2, zero), six);
              r3 = min_ps(max_ps(r3, zero), six);
            /*} 
			else if (act_type == lite_api::ActivationType::kLeakyRelu) {
              Type negative_slope = set1_ps(act_param.Leaky_relu_alpha);
              r0 = blendv_ps(
                  r0, mul_ps(negative_slope, r0), cmp_ps(r0, zero, 2));
              r1 = blendv_ps(
                  r1, mul_ps(negative_slope, r1), cmp_ps(r1, zero, 2));
              r2 = blendv_ps(
                  r2, mul_ps(negative_slope, r2), cmp_ps(r2, zero, 2));
              r3 = blendv_ps(
                  r3, mul_ps(negative_slope, r3), cmp_ps(r3, zero, 2));*/
            } else if (act_type == kHardSwish) {
              Type vscale = set1_ps(1.0 / 6);
              Type voffset = set1_ps(3);
              Type vthreshold = set1_ps(6);
              r0 = mul_ps(min_ps(vthreshold, max_ps(zero, add_ps(r0, voffset))),
                          mul_ps(r0, vscale));
              r1 = mul_ps(min_ps(vthreshold, max_ps(zero, add_ps(r1, voffset))),
                          mul_ps(r1, vscale));
              r2 = mul_ps(min_ps(vthreshold, max_ps(zero, add_ps(r2, voffset))),
                          mul_ps(r2, vscale));
              r3 = mul_ps(min_ps(vthreshold, max_ps(zero, add_ps(r3, voffset))),
                          mul_ps(r3, vscale));
            /*} else {
              LOG(FATAL) << " [X86] activation type "
                         << static_cast<int>(act_type) << " not supported ";*/
            }


          storeu_ps(dout_block, r0);
          storeu_ps(dout_block + block_channel, r1);
          storeu_ps(dout_block + 2 * block_channel, r2);
          storeu_ps(dout_block + 3 * block_channel, r3);
          dout_block += 4 * block_channel;

          block_inr0 += 4 * block_channel;
          block_inr1 = block_inr0 + in_len;
          block_inr2 = block_inr1 + in_len;
          block_inr3 = block_inr2 + in_len;
          block_inr4 = block_inr3 + in_len;
        }

        for (; j < w_out; j++) {
          Type r = _bias;
          for (int m = 0; m < 5; m++) {
            for (int n = 0; n < 5; n++) {
              Type weight = loadu_ps(weights_data + 5 * block_channel * m +
                                     block_channel * n);
              Type input = loadu_ps(block_inr0 + block_channel * (j % 4) +
                                    in_len * m + block_channel * n);
              r = fmadd_ps(input, weight, r);
            }
          }
          Type zero = setzero_ps();

            if (act_type == kRelu) {
              r = max_ps(r, zero);
            } else if (act_type == kRelu6) {
              Type six = set1_ps(6.0);
              r = min_ps(max_ps(r, zero), six);
            /*} 
			else if (act_type == lite_api::ActivationType::kLeakyRelu) {
              Type negative_slope = set1_ps(act_param.Leaky_relu_alpha);
              r = blendv_ps(r, mul_ps(negative_slope, r), cmp_ps(r, zero, 2));*/
            } else if (act_type == kHardSwish) {
              Type vscale = set1_ps(1.0 / 6);
              Type voffset = set1_ps(3);
              Type vthreshold = set1_ps(6);
              r = mul_ps(min_ps(vthreshold, max_ps(zero, add_ps(r, voffset))),
                         mul_ps(r, vscale));
           /* } else {
              LOG(FATAL) << " [X86] activation type "
                         << static_cast<int>(act_type) << " not supported ";*/
            }

          storeu_ps(dout_block, r);
          dout_block += block_channel;
        }
      }

      unpackC8_common(pack_out, dout_ptr, size_out_channel, real_block_channel);
    }
  }

  free(pack_weight);
  free(pack_input);
  free(pack_out);
}
void conv_depthwise_5x5s2(const float* din,
                          float* dout,
                          int num,
                          int ch_out,
                          int h_out,
                          int w_out,
                          int ch_in,
                          int h_in,
                          int w_in,
                          const float* weights,
                          const float* bias,
                          int pad,
                          int flag_bias,
                          const int act_type) {
	const int block_channel = 4;
  int size_in_channel = w_in * h_in;
  int size_out_channel = w_out * h_out;
  int in_len = block_channel * (2 * pad + w_in);

  int channel_num = ROUNDUP(ch_in, block_channel);
  float* pack_weight = (float*)(
      malloc(channel_num * 5 * 5 * sizeof(float)));
  float* pack_input = (float*)(malloc(
      (h_in + 2 * pad) * (w_in + 2 * pad) * block_channel * sizeof(float)));
  float* pack_out = (float*)(malloc(h_out * w_out * block_channel * sizeof(float)));

  packC8_common(weights, pack_weight, 0, 5, 5, ch_in);

  for (int n = 0; n < num; n++) {
    const float* din_batch = din + n * ch_in * size_in_channel;
    float* dout_batch = dout + n * ch_out * size_out_channel;

    for (int c = 0; c < ch_out; c += block_channel) {
      int real_block_channel = min(block_channel, ch_out - c);
      float* dout_ptr = dout_batch + c * size_out_channel;
      float* din_ptr = (float*)din_batch + c * size_in_channel;
      float* weights_data = pack_weight + c * 5 * 5;

      packC8_common(din_ptr,
                    pack_input,
                    {pad, pad, pad, pad},
                    h_in,
                    w_in,
                    real_block_channel);

      float bias_ptr[4] = {0.f};
      if (flag_bias) {
        for (int i = 0; i < block_channel; i++) {
          if (real_block_channel > i) {
            bias_ptr[i] = *(bias + c + i);
          }
        }
      }

      Type _bias = loadu_ps(bias_ptr);

      for (int i = 0; i < h_out; i++) {
        const float* block_inr0 = pack_input + i * 2 * in_len;
        const float* block_inr1 = block_inr0 + in_len;
        const float* block_inr2 = block_inr1 + in_len;
        const float* block_inr3 = block_inr2 + in_len;
        const float* block_inr4 = block_inr3 + in_len;
        int j = 0;
        float* dout_block = pack_out + i * w_out * block_channel;
        for (; j + 3 < w_out; j += 4) {
          Type i00 = loadu_ps(block_inr0);
          Type i01 = loadu_ps(block_inr0 + 1 * block_channel);
          Type i02 = loadu_ps(block_inr0 + 2 * block_channel);
          Type i03 = loadu_ps(block_inr0 + 3 * block_channel);
          Type i04 = loadu_ps(block_inr0 + 4 * block_channel);
          Type i05 = loadu_ps(block_inr0 + 5 * block_channel);
          Type i06 = loadu_ps(block_inr0 + 6 * block_channel);
          Type i07 = loadu_ps(block_inr0 + 7 * block_channel);
          Type i08 = loadu_ps(block_inr0 + 8 * block_channel);
          Type i09 = loadu_ps(block_inr0 + 9 * block_channel);
          Type i0a = loadu_ps(block_inr0 + 10 * block_channel);

          Type w00 = loadu_ps(weights_data);
          Type r0 = fmadd_ps(i00, w00, _bias);
          Type r1 = fmadd_ps(i02, w00, _bias);
          Type r2 = fmadd_ps(i04, w00, _bias);
          Type r3 = fmadd_ps(i06, w00, _bias);

          Type w01 = loadu_ps(weights_data + 1 * block_channel);
          r0 = fmadd_ps(i01, w01, r0);
          r1 = fmadd_ps(i03, w01, r1);
          r2 = fmadd_ps(i05, w01, r2);
          r3 = fmadd_ps(i07, w01, r3);

          Type w02 = loadu_ps(weights_data + 2 * block_channel);
          r0 = fmadd_ps(i02, w02, r0);
          r1 = fmadd_ps(i04, w02, r1);
          r2 = fmadd_ps(i06, w02, r2);
          r3 = fmadd_ps(i08, w02, r3);

          Type w03 = loadu_ps(weights_data + 3 * block_channel);
          r0 = fmadd_ps(i03, w03, r0);
          r1 = fmadd_ps(i05, w03, r1);
          r2 = fmadd_ps(i07, w03, r2);
          r3 = fmadd_ps(i09, w03, r3);

          Type w04 = loadu_ps(weights_data + 4 * block_channel);
          r0 = fmadd_ps(i04, w04, r0);
          r1 = fmadd_ps(i06, w04, r1);
          r2 = fmadd_ps(i08, w04, r2);
          r3 = fmadd_ps(i0a, w04, r3);

          Type i10 = loadu_ps(block_inr1);
          Type i11 = loadu_ps(block_inr1 + 1 * block_channel);
          Type i12 = loadu_ps(block_inr1 + 2 * block_channel);
          Type i13 = loadu_ps(block_inr1 + 3 * block_channel);
          Type i14 = loadu_ps(block_inr1 + 4 * block_channel);
          Type i15 = loadu_ps(block_inr1 + 5 * block_channel);
          Type i16 = loadu_ps(block_inr1 + 6 * block_channel);
          Type i17 = loadu_ps(block_inr1 + 7 * block_channel);
          Type i18 = loadu_ps(block_inr1 + 8 * block_channel);
          Type i19 = loadu_ps(block_inr1 + 9 * block_channel);
          Type i1a = loadu_ps(block_inr1 + 10 * block_channel);

          Type w10 = loadu_ps(weights_data + 5 * block_channel);
          r0 = fmadd_ps(i10, w10, r0);
          r1 = fmadd_ps(i12, w10, r1);
          r2 = fmadd_ps(i14, w10, r2);
          r3 = fmadd_ps(i16, w10, r3);

          Type w11 = loadu_ps(weights_data + 6 * block_channel);
          r0 = fmadd_ps(i11, w11, r0);
          r1 = fmadd_ps(i13, w11, r1);
          r2 = fmadd_ps(i15, w11, r2);
          r3 = fmadd_ps(i17, w11, r3);

          Type w12 = loadu_ps(weights_data + 7 * block_channel);
          r0 = fmadd_ps(i12, w12, r0);
          r1 = fmadd_ps(i14, w12, r1);
          r2 = fmadd_ps(i16, w12, r2);
          r3 = fmadd_ps(i18, w12, r3);

          Type w13 = loadu_ps(weights_data + 8 * block_channel);
          r0 = fmadd_ps(i13, w13, r0);
          r1 = fmadd_ps(i15, w13, r1);
          r2 = fmadd_ps(i17, w13, r2);
          r3 = fmadd_ps(i19, w13, r3);

          Type w14 = loadu_ps(weights_data + 9 * block_channel);
          r0 = fmadd_ps(i14, w14, r0);
          r1 = fmadd_ps(i16, w14, r1);
          r2 = fmadd_ps(i18, w14, r2);
          r3 = fmadd_ps(i1a, w14, r3);

          Type i20 = loadu_ps(block_inr2);
          Type i21 = loadu_ps(block_inr2 + 1 * block_channel);
          Type i22 = loadu_ps(block_inr2 + 2 * block_channel);
          Type i23 = loadu_ps(block_inr2 + 3 * block_channel);
          Type i24 = loadu_ps(block_inr2 + 4 * block_channel);
          Type i25 = loadu_ps(block_inr2 + 5 * block_channel);
          Type i26 = loadu_ps(block_inr2 + 6 * block_channel);
          Type i27 = loadu_ps(block_inr2 + 7 * block_channel);
          Type i28 = loadu_ps(block_inr2 + 8 * block_channel);
          Type i29 = loadu_ps(block_inr2 + 9 * block_channel);
          Type i2a = loadu_ps(block_inr2 + 10 * block_channel);

          Type w20 = loadu_ps(weights_data + 10 * block_channel);
          r0 = fmadd_ps(i20, w20, r0);
          r1 = fmadd_ps(i22, w20, r1);
          r2 = fmadd_ps(i24, w20, r2);
          r3 = fmadd_ps(i26, w20, r3);

          Type w21 = loadu_ps(weights_data + 11 * block_channel);
          r0 = fmadd_ps(i21, w21, r0);
          r1 = fmadd_ps(i23, w21, r1);
          r2 = fmadd_ps(i25, w21, r2);
          r3 = fmadd_ps(i27, w21, r3);

          Type w22 = loadu_ps(weights_data + 12 * block_channel);
          r0 = fmadd_ps(i22, w22, r0);
          r1 = fmadd_ps(i24, w22, r1);
          r2 = fmadd_ps(i26, w22, r2);
          r3 = fmadd_ps(i28, w22, r3);

          Type w23 = loadu_ps(weights_data + 13 * block_channel);
          r0 = fmadd_ps(i23, w23, r0);
          r1 = fmadd_ps(i25, w23, r1);
          r2 = fmadd_ps(i27, w23, r2);
          r3 = fmadd_ps(i29, w23, r3);

          Type w24 = loadu_ps(weights_data + 14 * block_channel);
          r0 = fmadd_ps(i24, w24, r0);
          r1 = fmadd_ps(i26, w24, r1);
          r2 = fmadd_ps(i28, w24, r2);
          r3 = fmadd_ps(i2a, w24, r3);

          Type i30 = loadu_ps(block_inr3);
          Type i31 = loadu_ps(block_inr3 + 1 * block_channel);
          Type i32 = loadu_ps(block_inr3 + 2 * block_channel);
          Type i33 = loadu_ps(block_inr3 + 3 * block_channel);
          Type i34 = loadu_ps(block_inr3 + 4 * block_channel);
          Type i35 = loadu_ps(block_inr3 + 5 * block_channel);
          Type i36 = loadu_ps(block_inr3 + 6 * block_channel);
          Type i37 = loadu_ps(block_inr3 + 7 * block_channel);
          Type i38 = loadu_ps(block_inr3 + 8 * block_channel);
          Type i39 = loadu_ps(block_inr3 + 9 * block_channel);
          Type i3a = loadu_ps(block_inr3 + 10 * block_channel);

          Type w30 = loadu_ps(weights_data + 15 * block_channel);
          r0 = fmadd_ps(i30, w30, r0);
          r1 = fmadd_ps(i32, w30, r1);
          r2 = fmadd_ps(i34, w30, r2);
          r3 = fmadd_ps(i36, w30, r3);

          Type w31 = loadu_ps(weights_data + 16 * block_channel);
          r0 = fmadd_ps(i31, w31, r0);
          r1 = fmadd_ps(i33, w31, r1);
          r2 = fmadd_ps(i35, w31, r2);
          r3 = fmadd_ps(i37, w31, r3);

          Type w32 = loadu_ps(weights_data + 17 * block_channel);
          r0 = fmadd_ps(i32, w32, r0);
          r1 = fmadd_ps(i34, w32, r1);
          r2 = fmadd_ps(i36, w32, r2);
          r3 = fmadd_ps(i38, w32, r3);

          Type w33 = loadu_ps(weights_data + 18 * block_channel);
          r0 = fmadd_ps(i33, w33, r0);
          r1 = fmadd_ps(i35, w33, r1);
          r2 = fmadd_ps(i37, w33, r2);
          r3 = fmadd_ps(i39, w33, r3);

          Type w34 = loadu_ps(weights_data + 19 * block_channel);
          r0 = fmadd_ps(i34, w34, r0);
          r1 = fmadd_ps(i36, w34, r1);
          r2 = fmadd_ps(i38, w34, r2);
          r3 = fmadd_ps(i3a, w34, r3);

          Type i40 = loadu_ps(block_inr4);
          Type i41 = loadu_ps(block_inr4 + 1 * block_channel);
          Type i42 = loadu_ps(block_inr4 + 2 * block_channel);
          Type i43 = loadu_ps(block_inr4 + 3 * block_channel);
          Type i44 = loadu_ps(block_inr4 + 4 * block_channel);
          Type i45 = loadu_ps(block_inr4 + 5 * block_channel);
          Type i46 = loadu_ps(block_inr4 + 6 * block_channel);
          Type i47 = loadu_ps(block_inr4 + 7 * block_channel);
          Type i48 = loadu_ps(block_inr4 + 8 * block_channel);
          Type i49 = loadu_ps(block_inr4 + 9 * block_channel);
          Type i4a = loadu_ps(block_inr4 + 10 * block_channel);

          Type w40 = loadu_ps(weights_data + 20 * block_channel);
          r0 = fmadd_ps(i40, w40, r0);
          r1 = fmadd_ps(i42, w40, r1);
          r2 = fmadd_ps(i44, w40, r2);
          r3 = fmadd_ps(i46, w40, r3);

          Type w41 = loadu_ps(weights_data + 21 * block_channel);
          r0 = fmadd_ps(i41, w41, r0);
          r1 = fmadd_ps(i43, w41, r1);
          r2 = fmadd_ps(i45, w41, r2);
          r3 = fmadd_ps(i47, w41, r3);

          Type w42 = loadu_ps(weights_data + 22 * block_channel);
          r0 = fmadd_ps(i42, w42, r0);
          r1 = fmadd_ps(i44, w42, r1);
          r2 = fmadd_ps(i46, w42, r2);
          r3 = fmadd_ps(i48, w42, r3);

          Type w43 = loadu_ps(weights_data + 23 * block_channel);
          r0 = fmadd_ps(i43, w43, r0);
          r1 = fmadd_ps(i45, w43, r1);
          r2 = fmadd_ps(i47, w43, r2);
          r3 = fmadd_ps(i49, w43, r3);

          Type w44 = loadu_ps(weights_data + 24 * block_channel);
          r0 = fmadd_ps(i44, w44, r0);
          r1 = fmadd_ps(i46, w44, r1);
          r2 = fmadd_ps(i48, w44, r2);
          r3 = fmadd_ps(i4a, w44, r3);

          Type zero = setzero_ps();

            if (act_type == kRelu) {
              r0 = max_ps(r0, zero);
              r1 = max_ps(r1, zero);
              r2 = max_ps(r2, zero);
              r3 = max_ps(r3, zero);
            } else if (act_type == kRelu6) {
              Type six = set1_ps(6.0);
              r0 = min_ps(max_ps(r0, zero), six);
              r1 = min_ps(max_ps(r1, zero), six);
              r2 = min_ps(max_ps(r2, zero), six);
              r3 = min_ps(max_ps(r3, zero), six);
            /*} 
			else if (act_type == lite_api::ActivationType::kLeakyRelu) {
              Type negative_slope = set1_ps(act_param.Leaky_relu_alpha);
              r0 = blendv_ps(
                  r0, mul_ps(negative_slope, r0), cmp_ps(r0, zero, 2));
              r1 = blendv_ps(
                  r1, mul_ps(negative_slope, r1), cmp_ps(r1, zero, 2));
              r2 = blendv_ps(
                  r2, mul_ps(negative_slope, r2), cmp_ps(r2, zero, 2));
              r3 = blendv_ps(
                  r3, mul_ps(negative_slope, r3), cmp_ps(r3, zero, 2));*/
            } else if (act_type == kHardSwish) {
              Type vscale = set1_ps(1.0 / 6);
              Type voffset = set1_ps(3);
              Type vthreshold = set1_ps(6);
              r0 = mul_ps(min_ps(vthreshold, max_ps(zero, add_ps(r0, voffset))),
                          mul_ps(r0, vscale));
              r1 = mul_ps(min_ps(vthreshold, max_ps(zero, add_ps(r1, voffset))),
                          mul_ps(r1, vscale));
              r2 = mul_ps(min_ps(vthreshold, max_ps(zero, add_ps(r2, voffset))),
                          mul_ps(r2, vscale));
              r3 = mul_ps(min_ps(vthreshold, max_ps(zero, add_ps(r3, voffset))),
                          mul_ps(r3, vscale));
           /* } else {
              LOG(FATAL) << " [X86] activation type "
                         << static_cast<int>(act_type) << " not supported ";*/
            }


          storeu_ps(dout_block, r0);
          storeu_ps(dout_block + 1 * block_channel, r1);
          storeu_ps(dout_block + 2 * block_channel, r2);
          storeu_ps(dout_block + 3 * block_channel, r3);
          dout_block += 4 * block_channel;

          block_inr0 += 8 * block_channel;
          block_inr1 = block_inr0 + in_len;
          block_inr2 = block_inr1 + in_len;
          block_inr3 = block_inr2 + in_len;
          block_inr4 = block_inr3 + in_len;
        }

        for (; j < w_out; j++) {
          Type r = _bias;
          for (int m = 0; m < 5; m++) {
            for (int n = 0; n < 5; n++) {
              Type weight = loadu_ps(weights_data + 5 * block_channel * m +
                                     block_channel * n);
              Type input = loadu_ps(block_inr0 + block_channel * (j % 4) * 2 +
                                    in_len * m + block_channel * n);
              r = fmadd_ps(input, weight, r);
            }
          }
          Type zero = setzero_ps();

            if (act_type == 1) {
              r = max_ps(r, zero);
            } else if (act_type == 2) {
              Type six = set1_ps(6.0);
              r = min_ps(max_ps(r, zero), six);
            /*} 
			else if (act_type == lite_api::ActivationType::kLeakyRelu) {
              Type negative_slope = set1_ps(act_param.Leaky_relu_alpha);
              r = blendv_ps(r, mul_ps(negative_slope, r), cmp_ps(r, zero, 2));*/
            } else if (act_type == kHardSwish) {
              Type vscale = set1_ps(1.0 / 6);
              Type voffset = set1_ps(3);
              Type vthreshold = set1_ps(6);
              r = mul_ps(min_ps(vthreshold, max_ps(zero, add_ps(r, voffset))),
                         mul_ps(r, vscale));
            /*} else {
              LOG(FATAL) << " [X86] activation type "
                         << static_cast<int>(act_type) << "not supported";*/
            }

          storeu_ps(dout_block, r);
          dout_block += block_channel;
        }
      }

      unpackC8_common(pack_out, dout_ptr, size_out_channel, real_block_channel);
    }
  }

  free(pack_weight);
  free(pack_input);
  free(pack_out);
}

static int8_t is_a_ge_zero_and_a_lt_b(int a, int b) {
	return (unsigned)(a) < (unsigned)(b);
}

void im2col_common(const float* data_im,
	int channels,
	int height,
	int width,
	int kernel_h,
	int kernel_w,
	int pad_top,
	int pad_bottom,
	int pad_left,
	int pad_right,
	int stride_h,
	int stride_w,
	int dilation_h,
	int dilation_w,
	float* data_col) {
	const int output_h =
		(height + pad_top + pad_bottom - (dilation_h * (kernel_h - 1) + 1)) /
		stride_h +
		1;
	const int output_w =
		(width + pad_left + pad_right - (dilation_w * (kernel_w - 1) + 1)) /
		stride_w +
		1;
	const int channel_size = height * width;
	for (int channel = channels; channel--; data_im += channel_size) {
		for (int kernel_row = 0; kernel_row < kernel_h; kernel_row++) {
			for (int kernel_col = 0; kernel_col < kernel_w; kernel_col++) {
				int input_row = -pad_top + kernel_row * dilation_h;
				for (int output_rows = output_h; output_rows; output_rows--) {
					if (!is_a_ge_zero_and_a_lt_b(input_row, height)) {
						for (int output_cols = output_w; output_cols; output_cols--) {
							*(data_col++) = 0;
						}
					}
					else {
						int input_col = -pad_left + kernel_col * dilation_w;
						for (int output_col = output_w; output_col; output_col--) {
							if (is_a_ge_zero_and_a_lt_b(input_col, width)) {
								*(data_col++) = data_im[input_row * width + input_col];
							}
							else {
								*(data_col++) = 0;
							}
							input_col += stride_w;
						}
					}
					input_row += stride_h;
				}
			}
		}
	}
}

void im2col_s1(const float* data_im,
	int channels,
	int height,
	int width,
	int kernel_h,
	int kernel_w,
	int pad_top,
	int pad_bottom,
	int pad_left,
	int pad_right,
	int dilation_h,
	int dilation_w,
	float* data_col) {
	const int output_h =
		(height + pad_top + pad_bottom - (dilation_h * (kernel_h - 1) + 1)) + 1;
	const int output_w =
		(width + pad_left + pad_right - (dilation_w * (kernel_w - 1) + 1)) + 1;
	const int in_channel_size = height * width;
	const int out_channel_size = output_h * output_w;
	const unsigned int output_plane_size =
		output_h * output_w * kernel_h * kernel_w;
	size_t tmp_size = static_cast<size_t>(output_plane_size);
	size_t mem_size = tmp_size * channels * sizeof(float);
	memset(data_col, 0, mem_size);

	for (int c = 0; c < channels; c++) {
		unsigned int data_im_z = static_cast<unsigned int>(c * in_channel_size);
		int data_col_z1 = c * output_plane_size;
		for (int ky = 0, h_offset = 0; ky < kernel_h;
			ky++, h_offset += dilation_h) {
			int data_col_z2 = ky * out_channel_size * kernel_w;
			for (int kx = 0, w_offset = 0; kx < kernel_w;
				kx++, w_offset += dilation_w) {
				int data_col_z3 = kx * out_channel_size;
				unsigned int data_col_z =
					static_cast<unsigned int>(data_col_z1 + data_col_z2 + data_col_z3);
				int oh_begin = max(((pad_top - h_offset)), 0);  // always >= 0
				int oh_end = min(((height + pad_bottom - h_offset)), output_h);
				oh_end = max(oh_begin, oh_end);
				int ow_begin = max(((pad_left - w_offset)), 0);
				int ow_end = min(((width + pad_right - w_offset)), output_w);
				ow_end = max(ow_begin, ow_end);
				int ih = oh_begin - pad_top + h_offset;
				for (int oh = oh_begin; oh < oh_end; ++oh, ++ih) {
					int iw = ow_begin - pad_left + w_offset;
					int ow = ow_begin;
					unsigned int data_im_offset = data_im_z + ih * width;
					unsigned int data_col_offset = data_col_z + oh * output_w;
					const float* data_im_ptr = data_im + data_im_offset;
					float* data_col_ptr = data_col + data_col_offset;

					for (; ow + 7 < ow_end; ow += 8, iw += 8) {
						__m256 vtmp = _mm256_loadu_ps(data_im_ptr + iw);
						_mm256_storeu_ps(data_col_ptr + ow, vtmp);
					}

					for (; ow < ow_end; ++ow, ++iw) {
						data_col[data_col_offset + ow] = data_im[data_im_offset + iw];
					}
				}
			}
		}
	}
}

void im2col_s2(const float* data_im,
	int channels,
	int height,
	int width,
	int kernel_h,
	int kernel_w,
	int pad_top,
	int pad_bottom,
	int pad_left,
	int pad_right,
	int dilation_h,
	int dilation_w,
	float* data_col) {
	const int output_h =
		(height + pad_top + pad_bottom - (dilation_h * (kernel_h - 1) + 1)) / 2 +
		1;
	const int output_w =
		(width + pad_left + pad_right - (dilation_w * (kernel_w - 1) + 1)) / 2 +
		1;
	const int in_channel_size = height * width;
	const unsigned int output_plane_size =
		output_h * output_w * kernel_h * kernel_w;
	size_t tmp_size = static_cast<size_t>(output_plane_size);
	size_t mem_size = tmp_size * channels * sizeof(float);
	memset(data_col, 0, mem_size);

	for (int c = 0; c < channels; c++) {
		unsigned int data_im_z = static_cast<unsigned int>(c * in_channel_size);
		int data_col_z1 = c * output_plane_size;
		for (int ky = 0, h_offset = 0; ky < kernel_h;
			ky++, h_offset += dilation_h) {
			int data_col_z2 = ky * output_h * output_w * kernel_w;
			for (int kx = 0, w_offset = 0; kx < kernel_w;
				kx++, w_offset += dilation_w) {
				int data_col_z3 = kx * output_h * output_w;
				unsigned int data_col_z =
					static_cast<unsigned int>(data_col_z1 + data_col_z2 + data_col_z3);
				int oh_begin = max(((pad_top - h_offset + 1) / 2), 0);
				int oh_end =
					min(((height + pad_bottom - h_offset + 1) / 2), output_h);
				oh_end = max(oh_begin, oh_end);
				int ow_begin = max(((pad_left - w_offset + 1) / 2), 0);
				int ow_end =
					min(((width + pad_right - w_offset + 1) / 2), output_w);
				ow_end = max(ow_begin, ow_end);
				int ih = oh_begin * 2 - pad_top + h_offset;
				for (int oh = oh_begin; oh < oh_end; ++oh, ih += 2) {
					int iw = ow_begin * 2 - pad_left + w_offset;
					int ow = ow_begin;
					unsigned int data_im_offset = data_im_z + ih * width;
					unsigned int data_col_offset = data_col_z + oh * output_w;
					const float* data_im_ptr = data_im + data_im_offset;
					float* data_col_ptr = data_col + data_col_offset;
					for (; ow + 3 < ow_end; ow += 4, iw += 8) {
						// a0a1a2a3
						__m128 vtmp0 = _mm_loadu_ps(data_im_ptr + iw);
						// a4a5a6a7
						__m128 vtmp1 = _mm_loadu_ps(data_im_ptr + iw + 4);
						// a0a2a4a6
						_mm_storeu_ps(data_col_ptr + ow,
							_mm_shuffle_ps(vtmp0, vtmp1, 0x88));
					}
					for (; ow < ow_end; ++ow, iw += 2) {
						data_col[data_col_offset + ow] = data_im[data_im_offset + iw];
					}
				}
			}
		}
	}
}

void im2col(const float* data_im,
	int channels,
	int height,
	int width,
	int kernel_h,
	int kernel_w,
	int pad_top,
	int pad_bottom,
	int pad_left,
	int pad_right,
	int stride_h,
	int stride_w,
	int dilation_h,
	int dilation_w,
	float* data_col) {
	bool pads_equal = ((pad_top == pad_bottom) && (pad_left == pad_right));
	bool pads_all_equal = (pads_equal && pad_top == pad_left);
	bool ks_equal = (stride_h == stride_w) && (kernel_h == kernel_w);
	bool no_dilation = (dilation_h == 1) && (dilation_w == 1);
	bool kspd = pads_all_equal && ks_equal && no_dilation;
	if (kspd && stride_h == 1) {
		im2col_s1(data_im,
			channels,
			height,
			width,
			kernel_h,
			kernel_w,
			pad_top,
			pad_bottom,
			pad_left,
			pad_right,
			dilation_h,
			dilation_w,
			data_col);
	}
	else if (kspd && stride_h == 2) {
		im2col_s2(data_im,
			channels,
			height,
			width,
			kernel_h,
			kernel_w,
			pad_top,
			pad_bottom,
			pad_left,
			pad_right,
			dilation_h,
			dilation_w,
			data_col);
	}
	else {
		im2col_common(data_im,
			channels,
			height,
			width,
			kernel_h,
			kernel_w,
			pad_top,
			pad_bottom,
			pad_left,
			pad_right,
			stride_h,
			stride_w,
			dilation_h,
			dilation_w,
			data_col);
	}
}

static void activate_relu_inplace(float *data, int len, float alpha, int mode) {
	int i = 0;

	if (0 == mode) {  // relu
		__m256 vec_zero = _mm256_set1_ps(0.f);
		for (; i + 7 < len; i += 8) {
			__m256 vec_data = _mm256_loadu_ps(data + i);
			_mm256_storeu_ps(data + i, _mm256_max_ps(vec_data, vec_zero));
		}

		for (; i < len; i++) {
			data[i] = data[i] > 0.f ? data[i] : 0.f;
		}
	}
	else {  // relu6

		__m256 vec_zero = _mm256_set1_ps(0.f);
		__m256 vec_alph = _mm256_set1_ps(alpha);
		for (; i + 7 < len; i += 8) {
			__m256 vec_data = _mm256_loadu_ps(data + i);
			_mm256_storeu_ps(
				data + i, _mm256_min_ps(_mm256_max_ps(vec_data, vec_zero), vec_alph));
		}

		for (; i < len; i++) {
			data[i] = data[i] > 0.f ? data[i] : 0.f;
			data[i] = data[i] < alpha ? data[i] : alpha;
		}
	}
}
static void activate_relu_inplace_bias(float *data,
	const float *bias,
	int channel,
	int channel_size,
	float alpha,
	int mode) {
	int i = 0;
	int j = 0;
	float *tmp_data = data;

	__m256 vec_zero = { 0.f };
	__m256 vec_bias = { 0.f };
	__m256 vec_data = { 0.f };
	__m256 vec_alph = _mm256_set1_ps(alpha);

	if (0 == mode) {  // relu
		for (j = 0; j < channel; j++) {
			i = 0;
			tmp_data = data + j * channel_size;

			vec_bias = _mm256_set1_ps(bias[j]);
			for (; i + 7 < channel_size; i += 8) {
				vec_data = _mm256_loadu_ps(tmp_data + i);
				vec_data = _mm256_add_ps(vec_bias, vec_data);
				_mm256_storeu_ps(tmp_data + i, _mm256_max_ps(vec_data, vec_zero));
			}

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

			vec_bias = _mm256_set1_ps(bias[j]);
			for (; i + 7 < channel_size; i += 8) {
				vec_data = _mm256_loadu_ps(tmp_data + i);
				vec_data = _mm256_add_ps(vec_bias, vec_data);
				_mm256_storeu_ps(
					tmp_data + i,
					_mm256_min_ps(_mm256_max_ps(vec_data, vec_zero), vec_alph));
			}

			for (; i < channel_size; i++) {
				tmp_data[i] += bias[j];
				tmp_data[i] = tmp_data[i] > 0.f ? tmp_data[i] : 0.f;
				tmp_data[i] = tmp_data[i] < alpha ? tmp_data[i] : alpha;
			}
		}
	}
}
static void activate_hardswish_inplace_bias(float *data,
	const float *bias,
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
		float *tmp_data = data + i * channel_size;

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
	float *data, int len, float scale, float threshold, float offset) {

	int cnt = len >> 5;
	int remain = len & 31;
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
	float *tmp_data = data;
	for (int i = 0; i < cnt; i++) {

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

static void activate_none_inplace_bias(float *data,
	const float *bias,
	int channel,
	int channel_size) {
	int i = 0;
	int j = 0;
	float *tmp_data = data;

	__m256 vec_bias = { 0.f };
	__m256 vec_data = { 0.f };

	for (j = 0; j < channel; j++) {
		i = 0;
		tmp_data = data + j * channel_size;

		vec_bias = _mm256_set1_ps(bias[j]);
		for (; i + 7 < channel_size; i += 8) {
			vec_data = _mm256_loadu_ps(tmp_data + i);
			vec_data = _mm256_add_ps(vec_bias, vec_data);
			_mm256_storeu_ps(tmp_data + i, vec_data);
		}

		for (; i < channel_size; i++) {
			tmp_data[i] += bias[j];
		}
	}
}

void fill_bias_act(float *tensor,
	const float *bias,
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
			if (act_type == 1) {
				activate_relu_inplace(tensor, len, relu_alpha, 0);
			}
			else if (act_type == 2) {
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

#define CONV_DW_PARAM                                                         \
  i_data, o_data, bs, oc, oh, ow, ic, ih, iw, w_data, b_data, paddings, flag_bias, \
      act_param
void DepthwiseConv(const float* i_data, float* o_data, const float* w_data, const float* b_data,
	uint16_t* x_dims, uint16_t* w_dims, uint16_t* o_dims, const int stride, int dilations, int paddings,int flag_bias,int act_param) {

	int iw = x_dims[3];
	int ih = x_dims[2];
	int ic = x_dims[1];
	int bs = x_dims[0];
	int oh = o_dims[2];
	int ow = o_dims[3];
	int oc = o_dims[1];
	int kh = w_dims[2];
	bool pad_less = paddings < 2;
	if (kh == 3) {
		if (dilations == 1 && pad_less) {
			if (stride == 1) {
				conv_depthwise_3x3s1_p01_direct(CONV_DW_PARAM);
			}
			else if (stride == 2) {
				conv_depthwise_3x3s2_p01_direct(CONV_DW_PARAM);
			}
		}
		//else {
		//	lite::x86::math::conv_depthwise_3x3_pack(
		//		param, &input_padding_, &input_pack_, &filter_pack_, &output_pack_);
		//}
	}
	else if (kh == 5) {
		if (stride == 1) {
			conv_depthwise_5x5s1(CONV_DW_PARAM);
		}
		else if (stride == 2) {
			conv_depthwise_5x5s2(CONV_DW_PARAM);
		}
	}
	
}


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
      (kh == 3) && (kw == 3) && (strides == 1 || strides == 2); \
  bool flag_dw_5x5 =                                                          \
      (kh == 5) && (kw == 5) && (strides == 1 || strides == 2);
	  
void autox_conv2d_ansi(float* din, float* dout, const float* bias, float* weights, uint16_t* x_dims, uint16_t* o_dims, uint16_t* w_dims,
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

	bool flag_p = paddings <= strides;
	//! select conv impl
	if (dw_kernel && flag_dw &&
		((flag_dw_5x5 && no_dilation) || (flag_dw_3x3 && (group & 3) == 0))) {
		DepthwiseConv(din, dout, weights, bias,
			x_dims, w_dims, o_dims, strides, dilations, paddings, bias!= NULL, act_type);
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
			im2col(din_batch,
				chin,
				hin,
				win,
				kh,
				kw,
				paddings,
				paddings,
				paddings,
				paddings,
				strides,
				strides,
				dilations,
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
				autox_matmul_v2(false,
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

