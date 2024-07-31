

void autox_split(float* input, float** output, uint16_t *in_dim, uint16_t input_dims_size, uint16_t output_ddim[][4], uint16_t* output_ddim_size, uint16_t output_size, int32_t axis)
{
  uint16_t* in_strides = (uint16_t*)calloc(input_dims_size, sizeof(uint16_t));
  for (int i = 0; i < input_dims_size; i++) {
	  in_strides[i] = 1;
  }
  in_strides[input_dims_size - 1] = in_dim[input_dims_size - 1];
  for (int i = input_dims_size - 2; i >= 0; --i) {
    in_strides[i] = in_strides[i + 1] * in_dim[i];
  }

  if (axis < 0) {
    axis += input_dims_size;
  }

  int input_offset = 0;
  for (uint16_t i=0;i<output_size;i++) {
    uint16_t *out_dim = output_ddim[i];
    uint16_t out_dim_size = output_ddim_size[i];
    uint32_t* out_strides = (uint32_t*)calloc(out_dim_size, sizeof(uint32_t));
    out_strides[out_dim_size - 1] = out_dim[out_dim_size - 1];
    for (int i = out_dim_size - 2; i >= 0; --i) {
      out_strides[i] = out_strides[i + 1] * out_dim[i];
    }

	  float* out_data = output[i];
    int before = out_strides[0] / out_strides[axis];
    int in_after = in_strides[axis];
    int out_after = out_strides[axis];

    const float* din_ptr = input + input_offset;

    for (int i = 0; i < before; ++i) {
      memcpy(out_data, din_ptr, sizeof(float) * out_after);
      din_ptr += in_after;
      out_data += out_after;
    }
    input_offset += out_strides[axis];

	  free(out_strides);
  }

  free(in_strides);
}
