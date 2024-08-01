#include "../include/autox_nn.h"

void autox_concat(float** inputs, float* output, uint16_t* input_dims[], uint16_t* output_dims, 
      int8_t axis, uint8_t input_size, uint8_t dim_0_size)
{
  if (input_size == 1) {
    output = inputs[0];
    return;
  }
  const uint16_t* input0_dims = input_dims[0];
  if (axis < 0) {
    axis += dim_0_size;
  }

  int offset_concat_axis = 0;
  uint32_t num_concat = count(input0_dims, 0, axis);
  uint32_t concat_input_size = count(input0_dims, axis + 1, dim_0_size);
  const uint16_t top_concat_axis = output_dims[axis];

  for (int i = 0; i < input_size; ++i) {
    const float* bottom_data = inputs[i];
    const uint16_t bottom_concat_axis = input_dims[i][axis];
    for (uint32_t n = 0; n < num_concat; ++n) {
      memcpy(
          output +
              (n * top_concat_axis + offset_concat_axis) * concat_input_size,
          bottom_data + n * bottom_concat_axis * concat_input_size,
          (bottom_concat_axis * concat_input_size) * sizeof(float));
    }
    offset_concat_axis += bottom_concat_axis;
  }
}
