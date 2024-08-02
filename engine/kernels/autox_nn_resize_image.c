#include "../include/autox_nn.h"

static uint8_t get_pixel(const uint8_t* data, int m_h, int m_w, int x, int y, int c)
{
	return data[c*m_h*m_w + y * m_w + x];
}
static void set_pixel(uint8_t* data, int m_h, int m_w, int x, int y, int c, float val)
{
	if (x < 0 || y < 0 || c < 0 || x >= m_w || y >= m_h) return;

	data[c*m_h*m_w + y * m_w + x] = val;
}
static void add_pixel(uint8_t* data, int m_h, int m_w, int x, int y, int c, float val)
{
	data[c*m_h*m_w + y * m_w + x] += val;
}

void autox_resize_image(const uint8_t* src, uint8_t* dst, uint16_t ssize_h, uint16_t ssize_w, uint16_t dsize_h, uint16_t dsize_w)
{
	uint8_t* part = (uint8_t*)calloc(dsize_w*ssize_h*3, sizeof(uint8_t));
	int r, c, k;
	float w_scale = (float)(ssize_w - 1) / (dsize_w - 1);
	float h_scale = (float)(ssize_h - 1) / (dsize_h - 1);
	for (k = 0; k < 3; ++k) {
		for (r = 0; r < ssize_h; ++r) {
			for (c = 0; c < dsize_w; ++c) {
				float val = 0;
				if (c == dsize_w - 1 || ssize_w == 1) {
					val = get_pixel(src, ssize_h, ssize_w, ssize_w - 1, r, k);
				}
				else {
					float sx = c * w_scale;
					int ix = (int)sx;
					float dx = sx - ix;
					val = (1 - dx) * get_pixel(src, ssize_h, ssize_w, ix, r, k) + dx * get_pixel(src, ssize_h, ssize_w, ix + 1, r, k);
				}
				set_pixel(part, ssize_h, dsize_w, c, r, k, val);
			}
		}
	}
	for (k = 0; k < 3; ++k) {
		for (r = 0; r < dsize_h; ++r) {
			float sy = r * h_scale;
			int iy = (int)sy;
			float dy = sy - iy;
			for (c = 0; c < dsize_w; ++c) {
				float val = (1 - dy) * get_pixel(part, ssize_h, dsize_w, c, iy, k);
				set_pixel(dst, dsize_h, dsize_w, c, r, k, val);
			}
			if (r == dsize_h - 1 || ssize_h == 1) continue;
			for (c = 0; c < dsize_w; ++c) {
				float val = dy * get_pixel(part, ssize_h, dsize_w, c, iy + 1, k);
				add_pixel(dst, dsize_h, dsize_w, c, r, k, val);
			}
		}
	}

	free(part);
}
