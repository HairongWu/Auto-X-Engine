#include "autox_nn.h"

    uint32_t count(const uint16_t *ddim, uint8_t start, uint8_t end)
    {
      start = max(start, 0);
      if (end < start) {
        return 0;
      }
      uint32_t sum = 1;
      for (uint8_t i = start; i < end; ++i) {
        sum *= ddim[i];
      }
      return sum;
    }