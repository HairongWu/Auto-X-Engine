#include "autox_nn_ansi.h"

void autox_relu_fp16_arm(const float16_t* din,
                         float16_t* dout,
                         int size,
                         int threads) {
  int nums_per_thread = size / threads;
  int remain = size - threads * nums_per_thread;
  int neon_loop_cnt = nums_per_thread >> 5;
  int neon_loop_rem = nums_per_thread & 31;
  int neon_loop_rem_cnt = neon_loop_rem >> 3;
  int neon_loop_rem_rem = neon_loop_rem & 7;
  int stride = neon_loop_rem_cnt << 3;
  float16x8_t vzero = vdupq_n_f16(0.f);

  LITE_PARALLEL_BEGIN(i, tid, threads) {
    const float16_t* ptr_in_thread = din + i * nums_per_thread;
    float16_t* ptr_out_thread = dout + i * nums_per_thread;
    for (int j = 0; j < neon_loop_cnt; j++) {
      float16x8_t vin0 = vld1q_f16(ptr_in_thread);
      float16x8_t vin1 = vld1q_f16(ptr_in_thread + 8);
      float16x8_t vin2 = vld1q_f16(ptr_in_thread + 16);
      float16x8_t vin3 = vld1q_f16(ptr_in_thread + 24);
      ptr_in_thread += 32;
      vst1q_f16(ptr_out_thread, vmaxq_f16(vin0, vzero));
      vst1q_f16(ptr_out_thread + 8, vmaxq_f16(vin1, vzero));
      vst1q_f16(ptr_out_thread + 16, vmaxq_f16(vin2, vzero));
      vst1q_f16(ptr_out_thread + 24, vmaxq_f16(vin3, vzero));
      ptr_out_thread += 32;
    }
    for (int j = 0; j < neon_loop_rem_cnt; j++) {
      float16x8_t vin0 = vld1q_f16(ptr_in_thread);
      ptr_in_thread += 8;
      vst1q_f16(ptr_out_thread, vmaxq_f16(vin0, vzero));
      ptr_out_thread += 8;
    }
    for (int j = 0; j < neon_loop_rem_rem; ++j) {
      ptr_out_thread[0] = ptr_in_thread[0] > 0.f ? ptr_in_thread[0] : 0.f;
      ptr_in_thread++;
      ptr_out_thread++;
    }
  }
  LITE_PARALLEL_END()
  float16_t* out_ptr_remain = dout + threads * nums_per_thread;
  const float16_t* in_ptr_remain = din + threads * nums_per_thread;
  for (int j = 0; j < remain; ++j) {
    out_ptr_remain[0] = in_ptr_remain[0] > 0.f ? in_ptr_remain[0] : 0.f;
    in_ptr_remain++;
    out_ptr_remain++;
  }
}

void autox_relu_fp32_arm(const float* din, float* dout, int size, int threads) {
  int nums_per_thread = size / threads;
  int remain = size - threads * nums_per_thread;
  int neon_loop_cnt = nums_per_thread >> 4;
  int neon_loop_remain = nums_per_thread - (neon_loop_cnt << 4);
  float32x4_t vzero = vdupq_n_f32(0.f);
  LITE_PARALLEL_BEGIN(i, tid, threads) {
    const float* ptr_in_thread = din + i * nums_per_thread;
    float* ptr_out_thread = dout + i * nums_per_thread;
    int cnt = neon_loop_cnt;
#ifdef __aarch64__
    for (int num = 0; num < neon_loop_cnt; ++num) {
      float32x4_t vr0 = vld1q_f32(ptr_in_thread);
      ptr_in_thread += 4;
      float32x4_t vr1 = vld1q_f32(ptr_in_thread);
      ptr_in_thread += 4;
      float32x4_t vr2 = vld1q_f32(ptr_in_thread);
      ptr_in_thread += 4;
      float32x4_t vr3 = vld1q_f32(ptr_in_thread);
      ptr_in_thread += 4;
      vr0 = vmaxq_f32(vr0, vzero);
      vr1 = vmaxq_f32(vr1, vzero);
      vr2 = vmaxq_f32(vr2, vzero);
      vr3 = vmaxq_f32(vr3, vzero);
      vst1q_f32(ptr_out_thread, vr0);
      ptr_out_thread += 4;
      vst1q_f32(ptr_out_thread, vr1);
      ptr_out_thread += 4;
      vst1q_f32(ptr_out_thread, vr2);
      ptr_out_thread += 4;
      vst1q_f32(ptr_out_thread, vr3);
      ptr_out_thread += 4;
    }

#else
    if (cnt > 0) {
      asm volatile(
          "1:                                     @ loop header\n"
          "vld1.32  {d0-d3}, [%[din]]!            @ load din 0\n"
          "vld1.32  {d4-d7}, [%[din]]!            @ load din 0\n"

          "vmax.f32 q8, q0, %q[vzero]             @ relu\n"
          "vmax.f32 q9, q1, %q[vzero]             @ relu\n"
          "vmax.f32 q10, q2, %q[vzero]            @ relu\n"
          "vmax.f32 q11, q3, %q[vzero]            @ relu\n"

          "vst1.32  {d16-d19}, [%[dout]]!         @ store result, add pointer\n"
          "vst1.32  {d20-d23}, [%[dout]]!         @ store result, add pointer\n"

          "subs %[cnt], #1                        @ loop count minus 1\n"
          "bne    1b                              @ jump to main loop start "
          "point\n"
          : [dout] "+r"(ptr_out_thread),
            [din] "+r"(ptr_in_thread),
            [cnt] "+r"(cnt)
          : [vzero] "w"(vzero)
          : "cc", "memory", "q0", "q1", "q2", "q3", "q8", "q9", "q10", "q11");
    }
#endif
    for (int j = 0; j < neon_loop_remain; ++j) {
      ptr_out_thread[0] = ptr_in_thread[0] > 0.f ? ptr_in_thread[0] : 0.f;
      ptr_in_thread++;
      ptr_out_thread++;
    }
  }
  LITE_PARALLEL_END();
  float* out_ptr_remain = dout + threads * nums_per_thread;
  const float* in_ptr_remain = din + threads * nums_per_thread;
  for (int j = 0; j < remain; ++j) {
    out_ptr_remain[0] = in_ptr_remain[0] > 0.f ? in_ptr_remain[0] : 0.f;
    in_ptr_remain++;
    out_ptr_remain++;
  }
}