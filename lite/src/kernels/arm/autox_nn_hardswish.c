#include "autox_nn_ansi.h"

void autox_hardswish_fp16_arm(const float16_t* din,
                               float16_t* dout,
                               const int size,
                               const float threshold,
                               const float scale,
                               const float offset,
                               int threads) {
  int cnt = size >> 5;
  int remain = size & 31;
  float scale_r = 1. / scale;

  int cnt_8 = remain >> 3;
  int remain_8 = remain & 7;

  float16x8_t vzero_8 = vdupq_n_f16(float16_t(0));
  float16x8_t vthreshold_8 = vdupq_n_f16(float16_t(threshold));
  float16x8_t vscale_8 = vdupq_n_f16(float16_t(scale_r));
  float16x8_t voffset_8 = vdupq_n_f16(float16_t(offset));

  for (int i = 0; i < cnt; i++) {
    float16x8_t vdin0 = vld1q_f16(din);
    float16x8_t vdin1 = vld1q_f16(din + 8);
    float16x8_t vdin2 = vld1q_f16(din + 16);
    float16x8_t vdin3 = vld1q_f16(din + 24);
    float16x8_t vtmp0 = vminq_f16(
        vthreshold_8, vmaxq_f16(vzero_8, vaddq_f16(vdin0, voffset_8)));
    float16x8_t vsum0 = vmulq_f16(vscale_8, vdin0);
    float16x8_t vtmp1 = vminq_f16(
        vthreshold_8, vmaxq_f16(vzero_8, vaddq_f16(vdin1, voffset_8)));
    float16x8_t vsum1 = vmulq_f16(vscale_8, vdin1);
    float16x8_t vtmp2 = vminq_f16(
        vthreshold_8, vmaxq_f16(vzero_8, vaddq_f16(vdin2, voffset_8)));
    float16x8_t vsum2 = vmulq_f16(vscale_8, vdin2);
    float16x8_t vtmp3 = vminq_f16(
        vthreshold_8, vmaxq_f16(vzero_8, vaddq_f16(vdin3, voffset_8)));
    float16x8_t vsum3 = vmulq_f16(vscale_8, vdin3);
    float16x8_t vres0 = vmulq_f16(vsum0, vtmp0);
    float16x8_t vres1 = vmulq_f16(vsum1, vtmp1);
    float16x8_t vres2 = vmulq_f16(vsum2, vtmp2);
    float16x8_t vres3 = vmulq_f16(vsum3, vtmp3);
    vst1q_f16(dout, vres0);
    vst1q_f16(dout + 8, vres1);
    vst1q_f16(dout + 16, vres2);
    vst1q_f16(dout + 24, vres3);
    din += 32;
    dout += 32;
  }
  for (int i = 0; i < cnt_8; i++) {
    float16x8_t vdin0 = vld1q_f16(din);
    din += 8;
    float16x8_t vtmp0 = vminq_f16(
        vthreshold_8, vmaxq_f16(vzero_8, vaddq_f16(vdin0, voffset_8)));
    float16x8_t vsum0 = vmulq_f16(vscale_8, vdin0);
    float16x8_t vres0 = vmulq_f16(vsum0, vtmp0);
    vst1q_f16(dout, vres0);
    dout += 8;
  }
  for (int i = 0; i < remain_8; i++) {
    dout[0] =
        min(max(0.f, din[0] + offset), threshold) * din[0] * scale_r;
    din++;
    dout++;
  }
}

void autox_hardswish_f32_arm(const float* din,
                           float* dout,
                           int size,
                           float threshold,
                           float scale,
                           float offset,
                           int threads) {
  int nums_per_thread = size / threads;
  int remain = size - nums_per_thread * threads;
  int neon_loop_cnt_dim4 = nums_per_thread >> 2;
  int neon_loop_remain_dim4 = nums_per_thread - (neon_loop_cnt_dim4 << 2);

  const float* ptr_in = din;
  float* ptr_out = dout;
  float scale_r = 1. / scale;
  float32x4_t scale_v, offset_v, threshold_v, zero;
  offset_v = vdupq_n_f32(offset);
  scale_v = vdupq_n_f32(scale_r);
  zero = vdupq_n_f32(0.);
  threshold_v = vdupq_n_f32(threshold);

  LITE_PARALLEL_BEGIN(i, tid, threads) {
    const float* ptr_in_thread = ptr_in + i * nums_per_thread;
    float* ptr_out_thread = ptr_out + i * nums_per_thread;
    for (int j = 0; j < neon_loop_cnt_dim4; j++) {
      float32x4_t in = vld1q_f32(ptr_in_thread);
      float32x4_t in_add_offset = vaddq_f32(in, offset_v);
      float32x4_t tmp1 = vmaxq_f32(zero, in_add_offset);
      float32x4_t tmp2 = vminq_f32(threshold_v, tmp1);
      float32x4_t tmp3 = vmulq_f32(scale_v, in);
      float32x4_t tmp4 = vmulq_f32(tmp2, tmp3);
      vst1q_f32(ptr_out_thread, tmp4);
      ptr_in_thread += 4;
      ptr_out_thread += 4;
    }

    for (int j = 0; j < neon_loop_remain_dim4; j++) {
      ptr_out_thread[0] =
          min(max(0.f, ptr_in_thread[0] + offset), threshold) *
          ptr_in_thread[0] * scale_r;
      ptr_in_thread++;
      ptr_out_thread++;
    }
  }
  LITE_PARALLEL_END();
  ptr_out = dout + threads * nums_per_thread;
  ptr_in = din + threads * nums_per_thread;
  for (int i = 0; i < remain; i++) {
    ptr_out[0] = min(max(0.f, ptr_in[0] + offset), threshold) *
                 ptr_in[0] * scale_r;
    ptr_in++;
    ptr_out++;
  }
}
