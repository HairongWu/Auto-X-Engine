#include "autox_nn_ansi.h"
#include <float.h>

inline uint16_t AdaptStartIndex(const uint16_t ph, const uint16_t input_size, const uint16_t output_size) {
	return (uint16_t)(
		floorf(ph * input_size) / output_size);
}

inline uint16_t AdaptEndIndex(const uint16_t ph, const uint16_t input_size, const uint16_t output_size) {
	return (uint16_t)(
		ceilf((ph + 1) * input_size) / output_size);
}

void autox_pool2d_ansi(const float *input_data, float *output_data, const uint16_t input_height, const uint16_t input_width, const uint16_t output_channels, const uint16_t output_height,
    const uint16_t output_width, const uint8_t ksize, const uint8_t stride, const uint8_t padding, const uint8_t adaptive, const uint8_t type)
{

    const uint16_t input_stride = input_height * input_width;
    const uint16_t output_stride = output_height * output_width;
    int32_t hstart, hend;
    int32_t wstart, wend;
    int output_stride_len = 0;
    for (uint16_t c = 0; c < output_channels; ++c) {
        for (uint16_t ph = 0; ph < output_height; ++ph) {
            if (adaptive) {
                hstart = AdaptStartIndex(ph, input_height, output_height);
                hend = AdaptEndIndex(ph, input_height, output_height);
            }
            else {
                hstart = ph * stride - padding;
                hend = min(hstart + ksize, input_height);
                hstart = max(hstart, 0);
            }
            for (uint16_t pw = 0; pw < output_width; ++pw) {
                if (adaptive) {
                    wstart = AdaptStartIndex(pw, input_width, output_width);
                    wend = AdaptEndIndex(pw, input_width, output_width);
                }
                else {
                    wstart = pw * stride - padding;
                    wend = min(wstart + ksize, input_width);
                    wstart = max(wstart, 0);
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

                if (type == 1)
                {
                    uint16_t pool_size = (adaptive)
                        ? (hend - hstart) * (wend - wstart)
                        : ksize * ksize;
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
