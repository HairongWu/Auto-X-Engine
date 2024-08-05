#include "../include/autox_nn.h"

// Transpose the input tensor similar to numpy.transpose. 
// For example, when perm=(1, 0, 2), given an input tensor of shape (1, 2, 3), the output shape will be (2, 1, 3).
void autox_transpose(const float* input_ptr, float* output_ptr, uint16_t* in_dim, uint16_t* out_dim, uint16_t *axis, int permute)
{
  // precompute inverted output dim and strides
  uint32_t rout_dim[6], strides[6];
  for (int i = 0; i < permute; ++i) {
    int k = permute - 1 - i;
    strides[k] = 1;
    for (uint16_t j = axis[i] + 1; j < permute; ++j) {
      strides[k] *= in_dim[j];
    }
    rout_dim[k] = out_dim[i];
  }

  // unroll the first 2 dimensions
  int reamin_dim = 1;
  for (int i = 2; i < permute; ++i) {
    reamin_dim *= out_dim[i];
  }

    for(uint16_t j=0; j< out_dim[1];j++) {
      uint32_t offset = j * strides[permute - 2];
      float* out_ptr = output_ptr + j * reamin_dim;
      int indics[4] = {0, 0, 0, 0};
      for (int k = 0; k < reamin_dim; ++k) {
        out_ptr[k] = input_ptr[offset];
        indics[0] += 1;
        offset += strides[0];
        for (int p = 0; p < permute - 3; ++p) {
          if (indics[p] == rout_dim[p]) {
            indics[p + 1] += 1;
            indics[p] = 0;
            offset += strides[p + 1];
            offset -= rout_dim[p] * strides[p];
          } else {
            break;
          }
        }
      }
    }
}

void autox_transpose2(const float* input_ptr, float* output_ptr, uint16_t* in_dim, uint16_t* out_dim, uint16_t* axis, int num_axes) {
    int from_inds[32] = { 0 };
    uint32_t size = count(in_dim, 0, num_axes);
    for (int index = 0; index < size; index++) {
        int from_index = index, to_index = 0;
        for (int i = 0; i < num_axes; i++) {
            from_inds[i] = from_index / in_dim[i];
            from_index = from_index % in_dim[i];
        }
        for (int i = 0; i < num_axes; i++) {
            to_index += from_inds[axis[i]] * out_dim[i];
        }

        *(output_ptr + to_index) = *(input_ptr + index);
    }
}