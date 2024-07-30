#include "autox_nn_ansi.h"

void autox_norm_ansi(const float* input,
          const int pre_n,
          const int n,
          const int post_n,
          const float epsilon,
          float* out) {
  for (int i = 0; i < pre_n; i++) {
    for (int k = 0; k < post_n; k++) {
      float sum = epsilon;
      const float* in_tmp = input + i * n * post_n + k;
      for (int j = 0; j < n; j++) {
        sum += in_tmp[j * post_n] * in_tmp[j * post_n];
      }
      sum = sqrtf(sum);
      float* out_tmp = out + i * n * post_n + k;
      for (int j = 0; j < n; j++) {
        out_tmp[j * post_n] = in_tmp[j * post_n] / sum;
      }
    }
  }
}