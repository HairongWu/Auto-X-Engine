#include "../include/autox_nn.h"


const int INTER_REMAP_COEF_SCALE = 1 << 15;

enum InterpolationMasks {
	INTER_BITS = 5,
	INTER_BITS2 = INTER_BITS * 2,
	INTER_TAB_SIZE = 1 << INTER_BITS,
	INTER_TAB_SIZE2 = INTER_TAB_SIZE * INTER_TAB_SIZE
};

static unsigned char NNDeltaTab_i[INTER_TAB_SIZE2][2];

static float BilinearTab_f[INTER_TAB_SIZE2][2][2];
static short BilinearTab_i[INTER_TAB_SIZE2][2][2];

static float BicubicTab_f[INTER_TAB_SIZE2][4][4];
static short BicubicTab_i[INTER_TAB_SIZE2][4][4];

static float Lanczos4Tab_f[INTER_TAB_SIZE2][8][8];
static short Lanczos4Tab_i[INTER_TAB_SIZE2][8][8];

//static const void* initInterTab2D(int method, bool fixpt)
//{
//	static bool inittab[INTER_MAX + 1] = { false };
//	float* tab = 0;
//	short* itab = 0;
//	int ksize = 0;
//	if (method == INTER_LINEAR) {
//		tab = BilinearTab_f[0][0], itab = BilinearTab_i[0][0], ksize = 2;
//	}
//	else if (method == INTER_CUBIC) {
//		tab = BicubicTab_f[0][0], itab = BicubicTab_i[0][0], ksize = 4;
//	}
//	else if (method == INTER_LANCZOS4) {
//		tab = Lanczos4Tab_f[0][0], itab = Lanczos4Tab_i[0][0], ksize = 8;
//	}
//	else {
//		FBC_Error("Unknown/unsupported interpolation type");
//	}
//
//	if (!inittab[method]) {
//		AutoBuffer<float> _tab(8 * INTER_TAB_SIZE);
//		int i, j, k1, k2;
//		initInterTab1D<float>(method, _tab, INTER_TAB_SIZE);
//		for (i = 0; i < INTER_TAB_SIZE; i++) {
//			for (j = 0; j < INTER_TAB_SIZE; j++, tab += ksize * ksize, itab += ksize * ksize) {
//				int isum = 0;
//				NNDeltaTab_i[i*INTER_TAB_SIZE + j][0] = j < INTER_TAB_SIZE / 2;
//				NNDeltaTab_i[i*INTER_TAB_SIZE + j][1] = i < INTER_TAB_SIZE / 2;
//
//				for (k1 = 0; k1 < ksize; k1++) {
//					float vy = _tab[i*ksize + k1];
//					for (k2 = 0; k2 < ksize; k2++) {
//						float v = vy * _tab[j*ksize + k2];
//						tab[k1*ksize + k2] = v;
//						isum += itab[k1*ksize + k2] = saturate_cast<short>(v*INTER_REMAP_COEF_SCALE);
//					}
//				}
//
//				if (isum != INTER_REMAP_COEF_SCALE) {
//					int diff = isum - INTER_REMAP_COEF_SCALE;
//					int ksize2 = ksize / 2, Mk1 = ksize2, Mk2 = ksize2, mk1 = ksize2, mk2 = ksize2;
//					for (k1 = ksize2; k1 < ksize2 + 2; k1++) {
//						for (k2 = ksize2; k2 < ksize2 + 2; k2++) {
//							if (itab[k1*ksize + k2] < itab[mk1*ksize + mk2])
//								mk1 = k1, mk2 = k2;
//							else if (itab[k1*ksize + k2] > itab[Mk1*ksize + Mk2])
//								Mk1 = k1, Mk2 = k2;
//						}
//					}
//					if (diff < 0)
//						itab[Mk1*ksize + Mk2] = (short)(itab[Mk1*ksize + Mk2] - diff);
//					else
//						itab[mk1*ksize + mk2] = (short)(itab[mk1*ksize + mk2] - diff);
//				}
//			}
//		}
//		tab -= INTER_TAB_SIZE2 * ksize*ksize;
//		itab -= INTER_TAB_SIZE2 * ksize*ksize;
//		inittab[method] = true;
//	}
//
//	return fixpt ? (const void*)itab : (const void*)tab;
//}
//
//static int remapBilinear(const Mat_<_Tp1, chs1>& _src, Mat_<_Tp1, chs1>& _dst,
//	const Mat_<_Tp2, chs2>& _xy, const Mat_<_Tp3, chs3>& _fxy, const void* _wtab, int borderType, const Scalar& _borderValue)
//{
//	typedef typename CastOp::rtype T;
//	typedef typename CastOp::type1 WT;
//	Size ssize = _src.size(), dsize = _dst.size();
//	int k, cn = _src.channels;
//	const AT* wtab = (const AT*)_wtab;
//	const T* S0 = (const T*)_src.ptr();
//	size_t sstep = _src.step / sizeof(S0[0]);
//	T cval[FBC_CN_MAX];
//	int dx, dy;
//	CastOp castOp;
//
//	for (k = 0; k < cn; k++)
//		cval[k] = saturate_cast<T>(_borderValue[k & 3]);
//
//	unsigned width1 = std::max(ssize.width - 1, 0), height1 = std::max(ssize.height - 1, 0);
//	FBC_Assert(ssize.area() > 0);
//
//	for (dy = 0; dy < dsize.height; dy++) {
//		T* D = (T*)_dst.ptr(dy);
//		const short* XY = (const short*)_xy.ptr(dy);
//		const ushort* FXY = (const ushort*)_fxy.ptr(dy);
//		int X0 = 0;
//		bool prevInlier = false;
//
//		for (dx = 0; dx <= dsize.width; dx++) {
//			bool curInlier = dx < dsize.width ? (unsigned)XY[dx * 2] < width1 && (unsigned)XY[dx * 2 + 1] < height1 : !prevInlier;
//			if (curInlier == prevInlier)
//				continue;
//
//			int X1 = dx;
//			dx = X0;
//			X0 = X1;
//			prevInlier = curInlier;
//
//			if (!curInlier) {
//				int len = 0;
//				D += len * cn;
//				dx += len;
//
//				if (cn == 1) {
//					for (; dx < X1; dx++, D++) {
//						int sx = XY[dx * 2], sy = XY[dx * 2 + 1];
//						const AT* w = wtab + FXY[dx] * 4;
//						const T* S = S0 + sy * sstep + sx;
//						*D = castOp(WT(S[0] * w[0] + S[1] * w[1] + S[sstep] * w[2] + S[sstep + 1] * w[3]));
//					}
//				}
//				else if (cn == 2) {
//					for (; dx < X1; dx++, D += 2) {
//						int sx = XY[dx * 2], sy = XY[dx * 2 + 1];
//						const AT* w = wtab + FXY[dx] * 4;
//						const T* S = S0 + sy * sstep + sx * 2;
//						WT t0 = S[0] * w[0] + S[2] * w[1] + S[sstep] * w[2] + S[sstep + 2] * w[3];
//						WT t1 = S[1] * w[0] + S[3] * w[1] + S[sstep + 1] * w[2] + S[sstep + 3] * w[3];
//						D[0] = castOp(t0); D[1] = castOp(t1);
//					}
//				}
//				else if (cn == 3) {
//					for (; dx < X1; dx++, D += 3) {
//						int sx = XY[dx * 2], sy = XY[dx * 2 + 1];
//						const AT* w = wtab + FXY[dx] * 4;
//						const T* S = S0 + sy * sstep + sx * 3;
//						WT t0 = S[0] * w[0] + S[3] * w[1] + S[sstep] * w[2] + S[sstep + 3] * w[3];
//						WT t1 = S[1] * w[0] + S[4] * w[1] + S[sstep + 1] * w[2] + S[sstep + 4] * w[3];
//						WT t2 = S[2] * w[0] + S[5] * w[1] + S[sstep + 2] * w[2] + S[sstep + 5] * w[3];
//						D[0] = castOp(t0); D[1] = castOp(t1); D[2] = castOp(t2);
//					}
//				}
//				else if (cn == 4) {
//					for (; dx < X1; dx++, D += 4) {
//						int sx = XY[dx * 2], sy = XY[dx * 2 + 1];
//						const AT* w = wtab + FXY[dx] * 4;
//						const T* S = S0 + sy * sstep + sx * 4;
//						WT t0 = S[0] * w[0] + S[4] * w[1] + S[sstep] * w[2] + S[sstep + 4] * w[3];
//						WT t1 = S[1] * w[0] + S[5] * w[1] + S[sstep + 1] * w[2] + S[sstep + 5] * w[3];
//						D[0] = castOp(t0); D[1] = castOp(t1);
//						t0 = S[2] * w[0] + S[6] * w[1] + S[sstep + 2] * w[2] + S[sstep + 6] * w[3];
//						t1 = S[3] * w[0] + S[7] * w[1] + S[sstep + 3] * w[2] + S[sstep + 7] * w[3];
//						D[2] = castOp(t0); D[3] = castOp(t1);
//					}
//				}
//				else {
//					for (; dx < X1; dx++, D += cn) {
//						int sx = XY[dx * 2], sy = XY[dx * 2 + 1];
//						const AT* w = wtab + FXY[dx] * 4;
//						const T* S = S0 + sy * sstep + sx * cn;
//						for (k = 0; k < cn; k++) {
//							WT t0 = S[k] * w[0] + S[k + cn] * w[1] + S[sstep + k] * w[2] + S[sstep + k + cn] * w[3];
//							D[k] = castOp(t0);
//						}
//					}
//				}
//			}
//			else {
//				if (borderType == BORDER_TRANSPARENT && cn != 3) {
//					D += (X1 - dx)*cn;
//					dx = X1;
//					continue;
//				}
//
//				if (cn == 1) {
//					for (; dx < X1; dx++, D++) {
//						int sx = XY[dx * 2], sy = XY[dx * 2 + 1];
//						if (borderType == BORDER_CONSTANT && (sx >= ssize.width || sx + 1 < 0 || sy >= ssize.height || sy + 1 < 0)) {
//							D[0] = cval[0];
//						}
//						else {
//							int sx0, sx1, sy0, sy1;
//							T v0, v1, v2, v3;
//							const AT* w = wtab + FXY[dx] * 4;
//							if (borderType == BORDER_REPLICATE) {
//								sx0 = clip(sx, 0, ssize.width);
//								sx1 = clip(sx + 1, 0, ssize.width);
//								sy0 = clip(sy, 0, ssize.height);
//								sy1 = clip(sy + 1, 0, ssize.height);
//								v0 = S0[sy0*sstep + sx0];
//								v1 = S0[sy0*sstep + sx1];
//								v2 = S0[sy1*sstep + sx0];
//								v3 = S0[sy1*sstep + sx1];
//							}
//							else {
//								sx0 = borderInterpolate<int>(sx, ssize.width, borderType);
//								sx1 = borderInterpolate<int>(sx + 1, ssize.width, borderType);
//								sy0 = borderInterpolate<int>(sy, ssize.height, borderType);
//								sy1 = borderInterpolate<int>(sy + 1, ssize.height, borderType);
//								v0 = sx0 >= 0 && sy0 >= 0 ? S0[sy0*sstep + sx0] : cval[0];
//								v1 = sx1 >= 0 && sy0 >= 0 ? S0[sy0*sstep + sx1] : cval[0];
//								v2 = sx0 >= 0 && sy1 >= 0 ? S0[sy1*sstep + sx0] : cval[0];
//								v3 = sx1 >= 0 && sy1 >= 0 ? S0[sy1*sstep + sx1] : cval[0];
//							}
//							D[0] = castOp(WT(v0*w[0] + v1 * w[1] + v2 * w[2] + v3 * w[3]));
//						}
//					}
//				}
//				else {
//					for (; dx < X1; dx++, D += cn) {
//						int sx = XY[dx * 2], sy = XY[dx * 2 + 1];
//						if (borderType == BORDER_CONSTANT && (sx >= ssize.width || sx + 1 < 0 || sy >= ssize.height || sy + 1 < 0)) {
//							for (k = 0; k < cn; k++)
//								D[k] = cval[k];
//						}
//						else {
//							int sx0, sx1, sy0, sy1;
//							const T *v0, *v1, *v2, *v3;
//							const AT* w = wtab + FXY[dx] * 4;
//							if (borderType == BORDER_REPLICATE) {
//								sx0 = clip(sx, 0, ssize.width);
//								sx1 = clip(sx + 1, 0, ssize.width);
//								sy0 = clip(sy, 0, ssize.height);
//								sy1 = clip(sy + 1, 0, ssize.height);
//								v0 = S0 + sy0 * sstep + sx0 * cn;
//								v1 = S0 + sy0 * sstep + sx1 * cn;
//								v2 = S0 + sy1 * sstep + sx0 * cn;
//								v3 = S0 + sy1 * sstep + sx1 * cn;
//							}
//							else if (borderType == BORDER_TRANSPARENT && ((unsigned)sx >= (unsigned)(ssize.width - 1) || (unsigned)sy >= (unsigned)(ssize.height - 1))) {
//								continue;
//							}
//							else {
//								sx0 = borderInterpolate<int>(sx, ssize.width, borderType);
//								sx1 = borderInterpolate<int>(sx + 1, ssize.width, borderType);
//								sy0 = borderInterpolate<int>(sy, ssize.height, borderType);
//								sy1 = borderInterpolate<int>(sy + 1, ssize.height, borderType);
//								v0 = sx0 >= 0 && sy0 >= 0 ? S0 + sy0 * sstep + sx0 * cn : &cval[0];
//								v1 = sx1 >= 0 && sy0 >= 0 ? S0 + sy0 * sstep + sx1 * cn : &cval[0];
//								v2 = sx0 >= 0 && sy1 >= 0 ? S0 + sy1 * sstep + sx0 * cn : &cval[0];
//								v3 = sx1 >= 0 && sy1 >= 0 ? S0 + sy1 * sstep + sx1 * cn : &cval[0];
//							}
//							for (k = 0; k < cn; k++)
//								D[k] = castOp(WT(v0[k] * w[0] + v1[k] * w[1] + v2[k] * w[2] + v3[k] * w[3]));
//						}
//					}
//				}
//			}
//		}
//	}
//
//	return 0;
//}
//
//static int remap_linear(const float* src, float* dst,
//	const float* map1, const float* map2, int borderMode)
//{
//	const void* ctab = 0;
//	bool fixpt = typeid(uchar).name() == typeid(_Tp1).name();
//	bool planar_input = map1.channels == 1;
//	ctab = initInterTab2D<_Tp1>(INTER_LINEAR, fixpt);
//
//	int x, y, x1, y1;
//	const int buf_size = 1 << 14;
//	int brows0 = min(128, dst.rows);
//	int bcols0 = min(buf_size / brows0, dst.cols);
//	brows0 = min(buf_size / bcols0, dst.rows);
//
//	Mat_<short, 2> _bufxy(brows0, bcols0);
//	Mat_<ushort, 1> _bufa(brows0, bcols0);
//	Mat_<short, 2> map1_tmp1(map1.rows, map1.cols, map1.data);
//
//	for (y = 0; y < dst.rows; y += brows0) {
//		for (x = 0; x < dst.cols; x += bcols0) {
//			int brows = min(brows0, range.end - y);
//			int bcols = min(bcols0, dst.cols - x);
//			Mat_<_Tp1, chs1> dpart;
//			dst.getROI(dpart, Rect(x, y, bcols, brows));
//			Mat_<short, 2> bufxy;
//			_bufxy.getROI(bufxy, Rect(0, 0, bcols, brows));
//			Mat_<ushort, 1> bufa;
//			_bufa.getROI(bufa, Rect(0, 0, bcols, brows));
//
//			for (y1 = 0; y1 < brows; y1++) {
//				short* XY = (short*)bufxy.ptr(y1);
//				ushort* A = (ushort*)bufa.ptr(y1);
//
//				if (map1.channels == 2 && typeid(short).name() == typeid(_Tp2).name() &&
//					(map2.channels == 1 && sizeof(_Tp3) == 2)) {
//					map1_tmp1.getROI(bufxy, Rect(x, y, bcols, brows));
//
//					const ushort* sA = (const ushort*)map2.ptr(y + y1) + x;
//					x1 = 0;
//
//					for (; x1 < bcols; x1++)
//						A[x1] = (ushort)(sA[x1] & (INTER_TAB_SIZE2 - 1));
//				}
//				else if (planar_input) {
//					const float* sX = (const float*)map1.ptr(y + y1) + x;
//					const float* sY = (const float*)map2.ptr(y + y1) + x;
//
//					x1 = 0;
//					for (; x1 < bcols; x1++) {
//						int sx = fbcRound(sX[x1] * INTER_TAB_SIZE);
//						int sy = fbcRound(sY[x1] * INTER_TAB_SIZE);
//						int v = (sy & (INTER_TAB_SIZE - 1))*INTER_TAB_SIZE + (sx & (INTER_TAB_SIZE - 1));
//						XY[x1 * 2] = saturate_cast<short>(sx >> INTER_BITS);
//						XY[x1 * 2 + 1] = saturate_cast<short>(sy >> INTER_BITS);
//						A[x1] = (ushort)v;
//					}
//				}
//				else {
//					const float* sXY = (const float*)map1.ptr(y + y1) + x * 2;
//					x1 = 0;
//					for (x1 = 0; x1 < bcols; x1++) {
//						int sx = fbcRound(sXY[x1 * 2] * INTER_TAB_SIZE);
//						int sy = fbcRound(sXY[x1 * 2 + 1] * INTER_TAB_SIZE);
//						int v = (sy & (INTER_TAB_SIZE - 1))*INTER_TAB_SIZE + (sx & (INTER_TAB_SIZE - 1));
//						XY[x1 * 2] = saturate_cast<short>(sx >> INTER_BITS);
//						XY[x1 * 2 + 1] = saturate_cast<short>(sy >> INTER_BITS);
//						A[x1] = (ushort)v;
//					}
//				}
//			}
//
//			if (typeid(_Tp1).name() == typeid(uchar).name()) { // uchar
//				remapBilinear<FixedPtCast<int, uchar, INTER_REMAP_COEF_BITS>, short, _Tp1, short, ushort, chs1, 2, 1>(src, dpart, bufxy, bufa, ctab, borderMode, borderValue);
//			}
//			else { // float
//				remapBilinear<Cast<float, float>, float, _Tp1, short, ushort, chs1, 2, 1>(src, dpart, bufxy, bufa, ctab, borderMode, borderValue);
//			}
//		}
//	}
//
//	return 0;
//}
//
//void warp_affine(const float* src, float* dst, const float* M_)
//{
//	double M[6];
//	Mat_<double, 1> matM(2, 3, M);
//	M_.convertTo(matM);
//
//	if (!(flags & WARP_INVERSE_MAP)) {
//		double D = M[0] * M[4] - M[1] * M[3];
//		D = D != 0 ? 1. / D : 0;
//		double A11 = M[4] * D, A22 = M[0] * D;
//		M[0] = A11; M[1] *= -D;
//		M[3] *= -D; M[4] = A22;
//		double b1 = -M[0] * M[2] - M[1] * M[5];
//		double b2 = -M[3] * M[2] - M[4] * M[5];
//		M[2] = b1; M[5] = b2;
//	}
//
//	int x;
//	AutoBuffer<int> _abdelta(dst.cols * 2);
//	int* adelta = &_abdelta[0], *bdelta = adelta + dst.cols;
//	const int AB_BITS = MAX(10, (int)INTER_BITS);
//	const int AB_SCALE = 1 << AB_BITS;
//
//	for (x = 0; x < dst.cols; x++) {
//		adelta[x] = saturate_cast<int>(M[0] * x*AB_SCALE);
//		bdelta[x] = saturate_cast<int>(M[3] * x*AB_SCALE);
//	}
//
//	const int BLOCK_SZ = 64;
//	short XY[BLOCK_SZ*BLOCK_SZ * 2], A[BLOCK_SZ*BLOCK_SZ];;
//	int round_delta = interpolation == INTER_NEAREST ? AB_SCALE / 2 : AB_SCALE / INTER_TAB_SIZE / 2, y, x1, y1;
//
//	int bh0 = min(BLOCK_SZ / 2, dst.rows);
//	int bw0 = min(BLOCK_SZ*BLOCK_SZ / bh0, dst.cols);
//	bh0 = min(BLOCK_SZ*BLOCK_SZ / bw0, dst.rows);
//
//	for (y = 0; y < dst.rows; y += bh0) {
//		for (x = 0; x < dst.cols; x += bw0) {
//			int bw = min(bw0, dst.cols - x);
//			int bh = min(bh0, range.end - y);
//
//			Mat_<short, 2> _XY(bh, bw, XY);
//			Mat_<_Tp1, chs1> dpart;
//			dst.getROI(dpart, Rect(x, y, bw, bh));
//
//			for (y1 = 0; y1 < bh; y1++) {
//				short* xy = XY + y1 * bw * 2;
//				int X0 = saturate_cast<int>((M[1] * (y + y1) + M[2])*AB_SCALE) + round_delta;
//				int Y0 = saturate_cast<int>((M[4] * (y + y1) + M[5])*AB_SCALE) + round_delta;
//
//				if (interpolation == INTER_NEAREST) {
//					x1 = 0;
//					for (; x1 < bw; x1++) {
//						int X = (X0 + adelta[x + x1]) >> AB_BITS;
//						int Y = (Y0 + bdelta[x + x1]) >> AB_BITS;
//						xy[x1 * 2] = saturate_cast<short>(X);
//						xy[x1 * 2 + 1] = saturate_cast<short>(Y);
//					}
//				}
//				else {
//					short* alpha = A + y1 * bw;
//					x1 = 0;
//					for (; x1 < bw; x1++) {
//						int X = (X0 + adelta[x + x1]) >> (AB_BITS - INTER_BITS);
//						int Y = (Y0 + bdelta[x + x1]) >> (AB_BITS - INTER_BITS);
//						xy[x1 * 2] = saturate_cast<short>(X >> INTER_BITS);
//						xy[x1 * 2 + 1] = saturate_cast<short>(Y >> INTER_BITS);
//						alpha[x1] = (short)((Y & (INTER_TAB_SIZE - 1))*INTER_TAB_SIZE +
//							(X & (INTER_TAB_SIZE - 1)));
//					}
//				}
//			}
//
//			{
//				Mat_<ushort, 1> _matA(bh, bw, A);
//				remap(src, dpart, _XY, _matA, interpolation, borderMode, borderValue);
//			}
//		}
//	}
//}
