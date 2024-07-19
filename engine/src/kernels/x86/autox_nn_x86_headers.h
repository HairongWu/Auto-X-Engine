#pragma once

#include <stdint.h>
#include <stdlib.h>

__m256 autox_relu_x86(const __m256 a);
void autox_hardswish_x86(const float* din,
                float* dout,
                int size,
                float scale,
                float offset,
                float threshold);