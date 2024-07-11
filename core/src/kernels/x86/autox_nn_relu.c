#include "autox_nn_ansi.h"

__m256 autox_relu_x86(const __m256 a) {
  __m256 tmp = _mm256_set1_ps(0.0f);
  return _mm256_max_ps(a, tmp);
}