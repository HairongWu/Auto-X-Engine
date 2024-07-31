

void autox_hwc2chw(const float* src, float* dst, uint16_t height, uint16_t width, uint8_t channels) {

	// Convert HWC to CHW
	for (int c = 0; c < channels; ++c)
	{
		for (int h = 0; h < height; ++h)
		{
			for (int w = 0; w < width; ++w)
			{
				int dstIdx = c * height * width + h * width + w;
				int srcIdx = h * width * channels + w * channels + c;
				dst[dstIdx] = src[srcIdx];
			}
		}
	}
}
