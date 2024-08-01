#include "autox_nn_ansi.h"

void nearest_interp(const float* input_data,
	float* output_data,
	const float ratio_h,
	const float ratio_w,
	const int n,
	const int c,
	const int in_h,
	const int in_w,
	const int out_h,
	const int out_w,
	const int8_t align_corners) {
	int total_count = n * c;
	if (align_corners) {
		for (int i = 0; i < total_count; ++i) {
			for (int h = 0; h < out_h; ++h) {
				for (int w = 0; w < out_w; ++w) {
					const float* input_data_ptr = input_data + i * in_h * in_w;
					float* output_data_ptr =
						output_data + i * out_h * out_w + h * out_w + w;
					int near_y = (int)(ratio_h * h + 0.5);
					int near_x = (int)(ratio_w * w + 0.5);
					*output_data_ptr = input_data_ptr[near_y * in_w + near_x];
				}
			}
		}
	}
	else {
		for (int i = 0; i < total_count; ++i) {
			for (int h = 0; h < out_h; ++h) {
				for (int w = 0; w < out_w; ++w) {
					const float* input_data_ptr = input_data + i * in_h * in_w;
					float* output_data_ptr =
						output_data + i * out_h * out_w + h * out_w + w;
					int near_y = (int)(ratio_h * h);
					int near_x = (int)(ratio_w * w);
					*output_data_ptr = input_data_ptr[near_y * in_w + near_x];
				}
			}
		}
	}
}

void autox_nearest_interp_v2_ansi(float* X,
                       float* Out,
					   uint16_t* intput1_dims,
					   float scale,
					   int8_t align_corners) {
	int n = intput1_dims[0];
	int c = intput1_dims[1];
	int in_h = intput1_dims[2];
	int in_w = intput1_dims[3];
	float scale_h = scale;
	float scale_w = scale;
	int out_h = -1;
	int out_w = -1;
	if (scale_h > 0. && scale_w > 0.) {
		out_h = (int)(in_h * scale_h);
		out_w = (int)(in_w * scale_w);
	}
  float ratio_h = 0.f;
  float ratio_w = 0.f;
  if (out_h > 1) {
	  float new_scale_h = 0.f;
	  new_scale_h = (scale_h > 0) ? (float)(1. / scale_h)
		  : (float)(in_h) / out_h;
	  ratio_h = (align_corners) ? (float)(in_h - 1) / (out_h - 1)
		  : (float)(new_scale_h);
  }
  if (out_w > 1) {
	  float new_scale_w = 0.f;
	  new_scale_w = (scale_w > 0) ? (float)(1. / scale_w)
		  : (float)(in_w) / out_w;
	  ratio_w = (align_corners) ? (float)(in_w - 1) / (out_w - 1)
		  : (float)(new_scale_w);
  }
  nearest_interp(X,
	  Out,
	  ratio_h,
	  ratio_w,
	  n,
	  c,
	  in_h,
	  in_w,
	  out_h,
	  out_w,
	  align_corners);
}

void bilinear_interp(const float* input_data,
	float* output_data,
	const float ratio_h,
	const float ratio_w,
	const int h_in,
	const int w_in,
	const int n,
	const int c,
	const int h_out,
	const int w_out,
	const int8_t align_corners,
	const int8_t align_mode) {
	int* buf = (int*)(
		malloc(sizeof(int) * (w_out * 4 + h_out * 4)));
	int* xofs = buf;
	int* yofs = buf + w_out * 2;

	float* alpha = (float*)(buf + w_out * 2 + h_out * 2);
	float* beta = (float*)(buf + h_out * 2 + w_out * 4);

	float fx = 0.0f;
	float fy = 0.0f;
	int sx = 0;
	int sy = 0;
	if (align_corners) {
		// calculate x axis coordinate
		for (int dx = 0; dx < w_out; dx++) {
			fx = dx * ratio_w;
			sx = (int)(fx);
			fx -= sx;
			xofs[dx * 2] = sx;
			xofs[dx * 2 + 1] = (sx + 1) < w_in - 1 ? (sx + 1) : (w_in - 1);
			alpha[dx * 2] = 1.f - fx;
			alpha[dx * 2 + 1] = fx;
		}
		// calculate y axis coordinate
		for (int dy = 0; dy < h_out; dy++) {
			fy = dy * ratio_h;
			sy = (int)(fy);
			fy -= sy;
			yofs[dy * 2] = sy;
			yofs[dy * 2 + 1] = (sy + 1) < h_in - 1 ? (sy + 1) : (h_in - 1);
			beta[dy * 2] = 1.f - fy;
			beta[dy * 2 + 1] = fy;
		}
	}
	else {
		// calculate x axis coordinate
		for (int dx = 0; dx < w_out; dx++) {
			fx = align_mode ? ratio_w * dx : ratio_w * (dx + 0.5f) - 0.5f;
			fx = fx < 0 ? 0.f : fx;
			sx = (int)(fx);
			fx -= sx;
			xofs[dx * 2] = sx;
			xofs[dx * 2 + 1] = (sx + 1) < w_in - 1 ? (sx + 1) : (w_in - 1);
			alpha[dx * 2] = 1.f - fx;
			alpha[dx * 2 + 1] = fx;
		}
		// calculate y axis coordinate
		for (int dy = 0; dy < h_out; dy++) {
			fy = align_mode ? ratio_h * dy : ratio_h * (dy + 0.5f) - 0.5f;
			fy = fy < 0 ? 0.f : fy;
			sy = (int)(fy);
			fy -= sy;
			yofs[dy * 2] = sy;
			yofs[dy * 2 + 1] = (sy + 1) < h_in - 1 ? (sy + 1) : (h_in - 1);
			beta[dy * 2] = 1.f - fy;
			beta[dy * 2 + 1] = fy;
		}
	}
	// output w , h boundary
	int w_bound = w_out;
	int h_bound = h_out;
	if (ratio_w > 0 && ratio_h > 0) {
		if (align_corners) {
			w_bound = ceil((w_in - 1) / ratio_w);
			h_bound = ceil((h_in - 1) / ratio_h);
		}
		else {
			w_bound = ceil((w_in - 0.5f) / ratio_w - 0.5f);
			h_bound = ceil((h_in - 0.5f) / ratio_h - 0.5f);
		}
	}
	int in_stride = h_in * w_in;
	int out_stride = h_out * w_out;
	int total = n * c;

	for (int nc = 0; nc < total; ++nc) {
		const float* src = input_data + nc * in_stride;
		float* dst = output_data + nc * out_stride;
		const float* betap = beta;

		float* rowsbuf0 =
			(float*)(malloc(sizeof(int) * w_out));
		float* rowsbuf1 =
			(float*)(malloc(sizeof(int) * w_out));
		float* rows0 = rowsbuf0;
		float* rows1 = rowsbuf1;
		// h_bound loop
		for (int dy = 0; dy < h_bound; dy++) {
			int sy0 = yofs[dy * 2];
			int sy1 = yofs[dy * 2 + 1];

			const float* s0 = src + sy0 * w_in;
			const float* s1 = src + sy1 * w_in;

			const float* alphap = alpha;
			float* rows0p = rows0;
			float* rows1p = rows1;

			int dx = 0;
			// w_bound loop
			for (; dx + 3 < w_bound; dx += 4) {
				int x0 = xofs[dx * 2];
				int x1 = xofs[(dx + 1) * 2];
				int x2 = xofs[(dx + 2) * 2];
				int x3 = xofs[(dx + 3) * 2];
				int x01 = xofs[dx * 2 + 1];
				int x11 = xofs[(dx + 1) * 2 + 1];
				int x21 = xofs[(dx + 2) * 2 + 1];
				int x31 = xofs[(dx + 3) * 2 + 1];

				const float* s0p0 = s0 + x0;
				const float* s0p1 = s0 + x1;
				const float* s0p2 = s0 + x2;
				const float* s0p3 = s0 + x3;

				const float* s0p0_1 = s0 + x01;
				const float* s0p1_1 = s0 + x11;
				const float* s0p2_1 = s0 + x21;
				const float* s0p3_1 = s0 + x31;

				const float* s1p0 = s1 + x0;
				const float* s1p1 = s1 + x1;
				const float* s1p2 = s1 + x2;
				const float* s1p3 = s1 + x3;

				const float* s1p0_1 = s1 + x01;
				const float* s1p1_1 = s1 + x11;
				const float* s1p2_1 = s1 + x21;
				const float* s1p3_1 = s1 + x31;

				__m256 _a = _mm256_loadu_ps(alphap);

				__m256 _s0p0p3 = _mm256_set_ps(
					*s0p3_1, *s0p3, *s0p2_1, *s0p2, *s0p1_1, *s0p1, *s0p0_1, *s0p0);

				__m256 _ms0 = _mm256_mul_ps(_s0p0p3, _a);
				__m256 _s1p0p3 = _mm256_set_ps(
					*s1p3_1, *s1p3, *s1p2_1, *s1p2, *s1p1_1, *s1p1, *s1p0_1, *s1p0);
				__m256 _ms1 = _mm256_mul_ps(_s1p0p3, _a);

				__m256 _rows0 = _mm256_hadd_ps(_ms0, _ms0);
				__m256 _rows1 = _mm256_hadd_ps(_ms1, _ms1);

				__m256 _rs0 = _mm256_castpd_ps(
					_mm256_permute4x64_pd(_mm256_castps_pd(_rows0), 0b11011000));
				__m256 _rs1 = _mm256_castpd_ps(
					_mm256_permute4x64_pd(_mm256_castps_pd(_rows1), 0b11011000));
				_mm_storeu_ps(rows0p + dx, _mm256_castps256_ps128(_rs0));
				_mm_storeu_ps(rows1p + dx, _mm256_castps256_ps128(_rs1));

				alphap += 8;
		}

			// w_bound remain loop
			for (; dx < w_bound; ++dx) {
				int sx = xofs[dx * 2];
				int sx1 = xofs[dx * 2 + 1];
				const float* s0p = s0 + sx;
				const float* s1p = s1 + sx;
				const float* s0p1 = s0 + sx1;
				const float* s1p1 = s1 + sx1;

				float a0 = alphap[0];
				float a1 = alphap[1];
				rows0p[dx] = s0p[0] * a0 + s0p1[0] * a1;
				rows1p[dx] = s1p[0] * a0 + s1p1[0] * a1;
				alphap += 2;
			}

			float param0 = *(src + sy0 * w_in + w_in - 1);
			float param1 = *(src + sy1 * w_in + w_in - 1);
			const float buffer0[2] = { param0, param0 };
			const float buffer1[2] = { param1, param1 };

			__m256 _s0p0p3 = _mm256_set1_ps(param0);
			__m256 _s1p0p3 = _mm256_set1_ps(param1);
			for (; dx + 3 < w_out; dx += 4) {
				__m256 _a = _mm256_loadu_ps(alphap);

				__m256 _ms0 = _mm256_mul_ps(_s0p0p3, _a);
				__m256 _ms1 = _mm256_mul_ps(_s1p0p3, _a);

				__m256 _rows0 = _mm256_hadd_ps(_ms0, _ms0);
				__m256 _rows1 = _mm256_hadd_ps(_ms1, _ms1);

				__m256 _rs0 = _mm256_castpd_ps(
					_mm256_permute4x64_pd(_mm256_castps_pd(_rows0), 0b11011000));
				__m256 _rs1 = _mm256_castpd_ps(
					_mm256_permute4x64_pd(_mm256_castps_pd(_rows1), 0b11011000));
				_mm_storeu_ps(rows0p + dx, _mm256_castps256_ps128(_rs0));
				_mm_storeu_ps(rows1p + dx, _mm256_castps256_ps128(_rs1));

				alphap += 8;
			}

			// w_bound - w_out remain loop
			for (; dx < w_out; dx++) {
				const float* s0p = buffer0;
				const float* s1p = buffer1;

				float a0 = alphap[0];
				float a1 = alphap[1];
				rows0p[dx] = s0p[0] * a0 + s0p[1] * a1;
				rows1p[dx] = s1p[0] * a0 + s1p[1] * a1;

				alphap += 2;
			}

			float b0 = betap[0];
			float b1 = betap[1];

			// output pos
			float* dp = dst + dy * w_out;

			int nn = 0;

			// 8 float
			__m256 _b0 = _mm256_set1_ps(b0);
			__m256 _b1 = _mm256_set1_ps(b1);
			// calculate and store results
			for (; nn + 7 < w_out; nn += 8) {
				__m256 _rows0 = _mm256_loadu_ps(rows0p);
				__m256 _rows1 = _mm256_loadu_ps(rows1p);

				__m256 _d = _mm256_add_ps(_mm256_mul_ps(_rows0, _b0),
					_mm256_mul_ps(_rows1, _b1));
				_mm256_storeu_ps(dp, _d);

				dp += 8;
				rows0p += 8;
				rows1p += 8;
			}

			// 4 float
			__m128 _c0 = _mm_set1_ps(b0);
			__m128 _c1 = _mm_set1_ps(b1);
			for (; nn + 3 < w_out; nn += 4) {
				__m128 _rows0 = _mm_loadu_ps(rows0p);
				__m128 _rows1 = _mm_loadu_ps(rows1p);

				__m128 _d =
					_mm_add_ps(_mm_mul_ps(_rows0, _c0), _mm_mul_ps(_rows1, _c1));
				_mm_storeu_ps(dp, _d);

				dp += 4;
				rows0p += 4;
				rows1p += 4;
			}

			// calculate and store remain resluts
			for (; nn < w_out; ++nn) {
				*dp++ = *rows0p++ * b0 + *rows1p++ * b1;
			}
			betap += 2;
	}  // end h_bound loop

	// h_bound - h_out loop
		for (int dy = h_bound; dy < h_out; dy++) {
			int sy = h_in - 1;
			const float* s0 = src + sy * w_in;
			const float* alphap = alpha;
			float* rows0p = rows0;
			float* rows1p = rows1;

			int dx = 0;
			const float* s1 = s0;

			// w_bound loop
			for (; dx + 3 < w_bound; dx += 4) {
				int x0 = xofs[dx * 2];
				int x1 = xofs[(dx + 1) * 2];
				int x2 = xofs[(dx + 2) * 2];
				int x3 = xofs[(dx + 3) * 2];
				int x01 = xofs[dx * 2 + 1];
				int x11 = xofs[(dx + 1) * 2 + 1];
				int x21 = xofs[(dx + 2) * 2 + 1];
				int x31 = xofs[(dx + 3) * 2 + 1];

				const float* s0p0 = s0 + x0;
				const float* s0p1 = s0 + x1;
				const float* s0p2 = s0 + x2;
				const float* s0p3 = s0 + x3;

				const float* s0p0_1 = s0 + x01;
				const float* s0p1_1 = s0 + x11;
				const float* s0p2_1 = s0 + x21;
				const float* s0p3_1 = s0 + x31;

				const float* s1p0 = s1 + x0;
				const float* s1p1 = s1 + x1;
				const float* s1p2 = s1 + x2;
				const float* s1p3 = s1 + x3;

				const float* s1p0_1 = s1 + x01;
				const float* s1p1_1 = s1 + x11;
				const float* s1p2_1 = s1 + x21;
				const float* s1p3_1 = s1 + x31;

				__m256 _a = _mm256_loadu_ps(alphap);

				__m256 _s0p0p3 = _mm256_set_ps(
					*s0p3_1, *s0p3, *s0p2_1, *s0p2, *s0p1_1, *s0p1, *s0p0_1, *s0p0);
				__m256 _ms0 = _mm256_mul_ps(_s0p0p3, _a);
				__m256 _s1p0p3 = _mm256_set_ps(
					*s1p3_1, *s1p3, *s1p2_1, *s1p2, *s1p1_1, *s1p1, *s1p0_1, *s1p0);
				__m256 _ms1 = _mm256_mul_ps(_s1p0p3, _a);

				__m256 _rows0 = _mm256_hadd_ps(_ms0, _ms0);
				__m256 _rows1 = _mm256_hadd_ps(_ms1, _ms1);

				__m256 _rs0 = _mm256_castpd_ps(
					_mm256_permute4x64_pd(_mm256_castps_pd(_rows0), 0b11011000));
				__m256 _rs1 = _mm256_castpd_ps(
					_mm256_permute4x64_pd(_mm256_castps_pd(_rows1), 0b11011000));
				_mm_storeu_ps(rows0p + dx, _mm256_castps256_ps128(_rs0));
				_mm_storeu_ps(rows1p + dx, _mm256_castps256_ps128(_rs1));

				alphap += 8;
}

			// w_bound remain loop
			for (; dx < w_bound; ++dx) {
				int sx = xofs[dx * 2];
				int sx1 = xofs[dx * 2 + 1];
				const float* s0p = s0 + sx;
				const float* s0p1 = s0 + sx1;
				float a0 = alphap[0];
				float a1 = alphap[1];
				rows0p[dx] = s0p[0] * a0 + s0p1[0] * a1;
				rows1p[dx] = rows0p[dx];

				alphap += 2;
			}

			float param = *(src + sy * w_in + w_in - 1);
			const float buffer1[2] = { param, param };

			__m256 _s0p0p3 = _mm256_set1_ps(param);
			__m256 _s1p0p3 = _mm256_set1_ps(param);

			// w_bound - w_out loop
			for (; dx + 3 < w_out; dx += 4) {
				__m256 _a = _mm256_loadu_ps(alphap);

				__m256 _ms0 = _mm256_mul_ps(_s0p0p3, _a);
				__m256 _ms1 = _mm256_mul_ps(_s1p0p3, _a);

				__m256 _rows0 = _mm256_hadd_ps(_ms0, _ms0);
				__m256 _rows1 = _mm256_hadd_ps(_ms1, _ms1);

				__m256 _rs0 = _mm256_castpd_ps(
					_mm256_permute4x64_pd(_mm256_castps_pd(_rows0), 0b11011000));
				__m256 _rs1 = _mm256_castpd_ps(
					_mm256_permute4x64_pd(_mm256_castps_pd(_rows1), 0b11011000));
				_mm_storeu_ps(rows0p + dx, _mm256_castps256_ps128(_rs0));
				_mm_storeu_ps(rows1p + dx, _mm256_castps256_ps128(_rs1));

				alphap += 8;
			}

			// w_bound - wout remain loop
			for (; dx < w_out; dx++) {
				const float* s0p = buffer1;
				float a0 = alphap[0];
				float a1 = alphap[1];
				rows0p[dx] = s0p[0] * a0 + s0p[1] * a1;
				rows1p[dx] = rows0p[dx];
				alphap += 2;
			}

			float b0 = betap[0];
			float b1 = betap[1];

			float* dp = dst + dy * w_out;

			int nn = 0;

			// 8 float
			__m256 _b0 = _mm256_set1_ps(b0);
			__m256 _b1 = _mm256_set1_ps(b1);
			// calculate and store results
			for (; nn + 7 < w_out; nn += 8) {
				__m256 _rows0 = _mm256_loadu_ps(rows0p);
				__m256 _rows1 = _mm256_loadu_ps(rows1p);

				__m256 _d = _mm256_add_ps(_mm256_mul_ps(_rows0, _b0),
					_mm256_mul_ps(_rows1, _b1));
				_mm256_storeu_ps(dp, _d);

				dp += 8;
				rows0p += 8;
				rows1p += 8;
			}

			// 4 float
			__m128 _c0 = _mm_set1_ps(b0);
			__m128 _c1 = _mm_set1_ps(b1);
			for (; nn + 3 < w_out; nn += 4) {
				__m128 _rows0 = _mm_loadu_ps(rows0p);
				__m128 _rows1 = _mm_loadu_ps(rows1p);

				__m128 _d =
					_mm_add_ps(_mm_mul_ps(_rows0, _c0), _mm_mul_ps(_rows1, _c1));
				_mm_storeu_ps(dp, _d);

				dp += 4;
				rows0p += 4;
				rows1p += 4;
			}

			// calculate and store remain results
			for (; nn < w_out; ++nn) {
				*dp++ = *rows0p++ * b0 + *rows1p++ * b1;
			}

			betap += 2;
	}  // end h_bound - h_out loop
		free(rowsbuf0);
		free(rowsbuf1);
  }
	free(buf);
}

void autox_bilinear_interp_ansi(float* X,
	float* Out,
	uint16_t* intput1_dims,
	uint16_t* out_dims,
	float scale,
	int8_t align_corners,
	int align_mode) {
	int n = intput1_dims[0];
	int c = intput1_dims[1];
	int in_h = intput1_dims[2];
	int in_w = intput1_dims[3];
	int out_h = out_dims[2];
	int out_w = out_dims[3];
	if (scale > 0) {
		out_h = (int)(in_h * scale);
		out_w = (int)(in_w * scale);
	}

	float ratio_h = 0.f;
	float ratio_w = 0.f;
	if (out_h > 1) {
		ratio_h = (align_corners) ? (float)(in_h - 1) / (out_h - 1)
			: (float)(in_h) / out_h;
	}
	if (out_w > 1) {
		ratio_w = (align_corners) ? (float)(in_w - 1) / (out_w - 1)
			: (float)(in_w) / out_w;
	}
	bilinear_interp(X,
		Out,
		ratio_h,
		ratio_w,
		in_h,
		in_w,
		n,
		c,
		out_h,
		out_w,
		align_corners,
		align_mode);
}
