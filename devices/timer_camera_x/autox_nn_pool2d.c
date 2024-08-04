#include "autox_nn.h"

uint16_t AdaptStartIndex(const uint16_t ph, const uint16_t input_size, const uint16_t output_size) {
	return (uint16_t)(
		floorf(ph * input_size) / output_size);
}

uint16_t AdaptEndIndex(const uint16_t ph, const uint16_t input_size, const uint16_t output_size) {
	return (uint16_t)(
		ceilf((ph + 1) * input_size) / output_size);
}

void autox_pool2d(const float *input_data, float *output_data, uint16_t* x_dims, uint16_t* o_dims, const uint8_t ksize, const uint8_t stride, const uint8_t padding, const uint8_t adaptive, const uint8_t type)
{
    uint16_t input_height = x_dims[2];
    uint16_t input_width = x_dims[3];
    uint16_t output_channels = o_dims[1];
    uint16_t output_height = o_dims[2];
    uint16_t output_width = o_dims[3];
    const uint32_t input_stride = input_height * input_width;
    const uint32_t output_stride = output_height * output_width;
    int32_t hstart, hend;
    int32_t wstart, wend;
    int output_stride_len = 0;
    for (uint16_t c = 0; c < output_channels; ++c) {
        for (uint16_t ph = 0; ph < output_height; ++ph) {
            if (adaptive) {
                hstart = AdaptStartIndex(ph, input_height, output_height);
                hend = AdaptEndIndex(ph, input_height, output_height);
            }
            for (uint16_t pw = 0; pw < output_width; ++pw) {
                int pool_size = 1;
                if (adaptive) {
                    wstart = AdaptStartIndex(pw, input_width, output_width);
                    wend = AdaptEndIndex(pw, input_width, output_width);
                }
                else {
                    hstart = ph * stride - padding;
                    wstart = pw * stride - padding;
                    hend = min(hstart + ksize,
                        input_height + padding);
                    wend =
                        min(wstart + ksize, input_width + padding);
                    pool_size = (hend - hstart) * (wend - wstart);

                    wstart = max(wstart, 0);
                    hstart = max(hstart, 0);
                    hend = min(hend, input_height);
                    wend = min(wend, input_width);
                }

                float ele = 0.0;
                if (type == 0)
                {
                    ele = -FLT_MAX;
                }
                else
                {

                }
                for (uint16_t h = hstart; h < hend; ++h) {
                    for (uint16_t w = wstart; w < wend; ++w) {
                        float x = input_data[h * input_width + w];
                        if (type == 0)
                        {
                            ele = ele > x ? ele : x;
                        }
                        else if (type == 1)
                        {
                            ele += x;
                        }
                        else
                        {

                        }
                    }
                }
                if (adaptive) {
                    pool_size = (hend - hstart) * (wend - wstart);
                }
                if (type == 1)
                {
                    ele /= pool_size;
                }
                else
                {

                }
                output_data[ph * output_width + pw] = ele;
            }
        }
        input_data += input_stride;
        output_data += output_stride;
        output_stride_len += output_stride;
    }
}
