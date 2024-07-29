#pragma once
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <assert.h>
#include <float.h>
#include <limits.h>

typedef union _m128i {
	int32_t i32[4];
	int16_t i16[8];
	int8_t i8[16];
	uint8_t u8[16];
} __m128i __attribute__((aligned(16)));

typedef union _m128 {
	float f32[4];
	int32_t i32[4];
} __m128 __attribute__((aligned(16)));

typedef union _m256i {
	__m128i m128i[2];
	int64_t i64[4];
	int32_t i32[8];
	int16_t i16[16];
	int8_t i8[32];
	uint16_t u16[16];
	uint8_t u8[32];
} __m256i __attribute__((aligned(32)));

typedef union _m256 {
	float f32[8];
	int32_t i32[8];
} __m256 __attribute__((aligned(32)));
typedef union __m64
{
	uint64_t    m64_u64;
	float               m64_f32[2];
	int8_t              m64_i8[8];
	int16_t             m64_i16[4];
	int32_t             m64_i32[2];
	int64_t             m64_i64;
	uint8_t     m64_u8[8];
	uint16_t    m64_u16[4];
	uint32_t    m64_u32[2];
} __m64;

inline int8_t saturate_cast_s32s8(int v) {
	return (int8_t)(((unsigned)(v - SCHAR_MIN) <= (unsigned)UCHAR_MAX
		? v
		: v > 0 ? SCHAR_MAX : SCHAR_MIN));
}

inline int8_t saturate_cast_i2s8(int v) {
	return (int8_t)(((unsigned)(v - SCHAR_MIN) <= (unsigned)UCHAR_MAX
		? v
		: v > 0 ? SCHAR_MAX : SCHAR_MIN));
}

inline int saturate_cast_f2i(float v) {
	return (int)(roundf(v));
}

inline int8_t saturate_cast_f32s8(float v) {
	int iv = saturate_cast_f2i((float)roundf(v));
	return saturate_cast_i2s8(iv);
}

inline int16_t saturate_cast_s32s16(int v) {
	return (int16_t)((unsigned)(v - SHRT_MIN) <= (unsigned)USHRT_MAX
		? v
		: v > 0 ? SHRT_MAX : SHRT_MIN);
}
inline uint8_t saturate_cast_s32u8(int v) {
	return (uint8_t)(
		((unsigned)v <= UCHAR_MAX ? v : v > 0 ? UCHAR_MAX : 0));
}

inline uint8_t saturate_cast_s16u8(int16_t v) {
	return saturate_cast_s32u8((int)(v));
}

inline int8_t saturate_cast_s16s8(int16_t v) {
	return saturate_cast_s32s8((int)(v));
}

/*******************************************************/
/* MACRO for shuffle parameter for _mm_shuffle_ps().   */
/* Argument fp3 is a digit[0123] that represents the fp*/
/* from argument "b" of mm_shuffle_ps that will be     */
/* placed in fp3 of result. fp2 is the same for fp2 in */
/* result. fp1 is a digit[0123] that represents the fp */
/* from argument "a" of mm_shuffle_ps that will be     */
/* places in fp1 of result. fp0 is the same for fp0 of */
/* result                                              */
/*******************************************************/
#define _MM_SHUFFLE(fp3,fp2,fp1,fp0) (((fp3) << 6) | ((fp2) << 4) | \
                                     ((fp1) << 2) | ((fp0)))
/*
 * Compare predicates for scalar and packed compare intrinsic functions
 */
#define _CMP_EQ_OQ     0x00  /* Equal (ordered, nonsignaling)               */
#define _CMP_LT_OS     0x01  /* Less-than (ordered, signaling)              */
#define _CMP_LE_OS     0x02  /* Less-than-or-equal (ordered, signaling)     */
#define _CMP_UNORD_Q   0x03  /* Unordered (nonsignaling)                    */
#define _CMP_NEQ_UQ    0x04  /* Not-equal (unordered, nonsignaling)         */
#define _CMP_NLT_US    0x05  /* Not-less-than (unordered, signaling)        */
#define _CMP_NLE_US    0x06  /* Not-less-than-or-equal (unordered,
														signaling)          */
#define _CMP_ORD_Q     0x07  /* Ordered (nonsignaling)                      */
#define _CMP_EQ_UQ     0x08  /* Equal (unordered, non-signaling)            */
#define _CMP_NGE_US    0x09  /* Not-greater-than-or-equal (unordered,
														   signaling)       */
#define _CMP_NGT_US    0x0A  /* Not-greater-than (unordered, signaling)     */
#define _CMP_FALSE_OQ  0x0B  /* False (ordered, nonsignaling)               */
#define _CMP_NEQ_OQ    0x0C  /* Not-equal (ordered, non-signaling)          */
#define _CMP_GE_OS     0x0D  /* Greater-than-or-equal (ordered, signaling)  */
#define _CMP_GT_OS     0x0E  /* Greater-than (ordered, signaling)           */
#define _CMP_TRUE_UQ   0x0F  /* True (unordered, non-signaling)             */
#define _CMP_EQ_OS     0x10  /* Equal (ordered, signaling)                  */
#define _CMP_LT_OQ     0x11  /* Less-than (ordered, nonsignaling)           */
#define _CMP_LE_OQ     0x12  /* Less-than-or-equal (ordered, nonsignaling)  */
#define _CMP_UNORD_S   0x13  /* Unordered (signaling)                       */
#define _CMP_NEQ_US    0x14  /* Not-equal (unordered, signaling)            */
#define _CMP_NLT_UQ    0x15  /* Not-less-than (unordered, nonsignaling)     */
#define _CMP_NLE_UQ    0x16  /* Not-less-than-or-equal (unordered,
														nonsignaling)       */
#define _CMP_ORD_S     0x17  /* Ordered (signaling)                         */
#define _CMP_EQ_US     0x18  /* Equal (unordered, signaling)                */
#define _CMP_NGE_UQ    0x19  /* Not-greater-than-or-equal (unordered,
														   nonsignaling)    */
#define _CMP_NGT_UQ    0x1A  /* Not-greater-than (unordered, nonsignaling)  */
#define _CMP_FALSE_OS  0x1B  /* False (ordered, signaling)                  */
#define _CMP_NEQ_OS    0x1C  /* Not-equal (ordered, signaling)              */
#define _CMP_GE_OQ     0x1D  /* Greater-than-or-equal (ordered,
													   nonsignaling)        */
#define _CMP_GT_OQ     0x1E  /* Greater-than (ordered, nonsignaling)        */
#define _CMP_TRUE_US   0x1F  /* True (unordered, signaling)                 */

													   // Sets the eight signed 16-bit integer values to w.
inline __m128i _mm_set1_epi16(short w)
{
	__m128i r;
	for (int i = 0; i < 8; i++)
		r.i16[i] = w;
	return r;
}
// Sets the four signed 32-bit integer values to i.
inline __m128i _mm_set1_epi32(int _i)
{
	__m128i r;
	for (int i = 0; i < 4; i++)
		r.i32[i] = _i;
	return r;
}
//  This returns a __m128 vector, where all four elements of the vector are set equal to a, i.e. the vector is [a,a,a,a].
inline __m128 _mm_set1_ps(float a)
{
	__m128 r;
	for (int i = 0; i < 4; i++)
		r.f32[i] = a;
	return r;
}

// Initializes a 256-bit vector with scalar integer values (8/16/32-bit values) as specified by the a parameter.
inline __m256i _mm256_set1_epi8(char a)
{
	__m256i r;
	for (int i = 0; i < 32; i++)
		r.i8[i] = a;
	return r;
}

inline __m256i _mm256_set1_epi16(short a)
{
	__m256i r;
	for (int i = 0; i < 16; i++)
		r.i16[i] = a;
	return r;
}

inline __m256i _mm256_set1_epi32(int a)
{
	__m256i r;
	for (int i = 0; i < 8; i++)
		r.i32[i] = a;
	return r;
}


inline __m256 _mm256_set1_ps(float a)
{
	__m256 r;
	for (int i = 0; i < 8; i++)
		r.f32[i] = a;
	return r;
}

// Sets the 128/256-bit value to zero.
inline __m128i _mm_setzero_si128(void)
{
	__m128i r;
	memset(r.i8, 0, 16);
	return r;
}

inline __m256i _mm256_setzero_si256(void)
{
	__m256i r;
	memset(r.i8, 0, 32);
	return r;
}
inline __m256 _mm256_setzero_ps(void)
{
	__m256 r;
	memset(r.f32, 0, 32);
	return r;
}
inline __m128 _mm_setzero_ps(void)
{
	__m128 r;
	memset(r.f32, 0, 16);
	return r;
}
// The _mm256_packus_epi16 intrinsic converts 16 packed signed word integers from source operands a and b into 32 packed unsigned byte integers. 
inline __m256i _mm256_packus_epi16(__m256i a, __m256i b)
{
	__m256i r;
	int i = 0;
	for (; i < 16; i++)
	{
		r.u8[i] = saturate_cast_s16u8(a.i16[i]);
		r.u8[i + 16] = saturate_cast_s16u8(b.i16[i]);
	}
	return r;
}

inline __m128i _mm_packs_epi16(__m128i a, __m128i b)
{
	__m128i r;
	int i = 0;
	for (; i < 8; i++)
	{
		r.i8[i] = saturate_cast_s16s8(a.i16[i]);
		r.i8[i + 8] = saturate_cast_s16s8(b.i16[i]);
	}
	return r;
}
inline __m128i _mm_packs_epi32(__m128i a, __m128i b)
{
	__m128i r;
	int i = 0;
	for (; i < 4; i++)
	{
		r.i16[i] = saturate_cast_s32s16(a.i32[i]);
		r.i16[i + 4] = saturate_cast_s32s16(b.i32[i]);
	}
	return r;
}
// Unpacks and interleaves the low-order signed or unsigned data elements (bytes, words, doublewords) of the source vector and the low-order signed or unsigned data elements (bytes, words, doublewords) in the destination operand. The high-order data elements are ignored.
inline __m256i _mm256_unpacklo_epi8(__m256i a, __m256i b)
{
	__m256i r;
	for (int i = 0; i < 32; i += 2)
	{
		r.i8[i] = a.i8[i / 2];
		r.i8[i + 1] = b.i8[i / 2];
	}

	return r;
}

inline __m256i _mm256_unpacklo_epi16(__m256i a, __m256i b)
{
	__m256i r;
	for (int i = 0; i < 16; i += 2)
	{
		r.i16[i] = a.i16[i / 2];
		r.i16[i + 1] = b.i16[i / 2];
	}

	return r;
}

inline __m256i _mm256_unpacklo_epi32(__m256i a, __m256i b)
{
	__m256i r;
	for (int i = 0; i < 8; i += 2)
	{
		r.i32[i] = a.i32[i / 2];
		r.i32[i + 1] = b.i32[i / 2];
	}

	return r;
}

// Unpacks and interleaves the high-order signed or unsigned data elements (bytes, words, doublewords) of the source vector and the high-order signed or unsigned data elements (bytes, words, doublewords) in the destination vector. The low-order data elements are ignored.
inline __m256i _mm256_unpackhi_epi8(__m256i a, __m256i b)
{
	__m256i r;
	for (int i = 0; i < 32; i += 2)
	{
		r.i8[i] = a.i8[16 + i / 2];
		r.i8[i + 1] = b.i8[16 + i / 2];
	}
	return r;
}

inline __m256i _mm256_unpackhi_epi16(__m256i a, __m256i b)
{
	__m256i r;
	for (int i = 0; i < 16; i += 2)
	{
		r.i16[i] = a.i16[8 + i / 2];
		r.i16[i + 1] = b.i16[8 + i / 2];
	}
	return r;
}


inline __m256i _mm256_unpackhi_epi32(__m256i a, __m256i b)
{
	__m256i r;
	for (int i = 0; i < 8; i += 2)
	{
		r.i32[i] = a.i32[4 + i / 2];
		r.i32[i + 1] = b.i32[4 + i / 2];
	}
	return r;
}
// Interleaves the lower two signed or unsigned 32-bit integers in a with the lower two signed or unsigned 32-bit integers in b.
inline __m128i _mm_unpacklo_epi32(__m128i a, __m128i b)
{
	__m128i r;
	for (int i = 0; i < 4; i += 2)
	{
		r.i32[i] = a.i32[i / 2];
		r.i32[i + 1] = b.i32[i / 2];
	}

	return r;
}
// Interleaves the upper two signed or unsigned 32-bit integers in a with the upper two signed or unsigned 32-bit integers in b.
inline __m128i _mm_unpackhi_epi32(__m128i a, __m128i b)
{
	__m128i r;
	for (int i = 0; i < 4; i += 2)
	{
		r.i32[i] = a.i32[2 + i / 2];
		r.i32[i + 1] = b.i32[2 + i / 2];
	}
	return r;
}
inline __m256 _mm256_unpacklo_ps(__m256 m1, __m256 m2)
{
	__m256 r;
	r.f32[0] = m1.f32[0];
	r.f32[1] = m2.f32[0];
	r.f32[2] = m1.f32[1];
	r.f32[3] = m2.f32[1];
	r.f32[4] = m1.f32[4];
	r.f32[5] = m2.f32[4];
	r.f32[6] = m1.f32[5];
	r.f32[7] = m2.f32[5];
	return r;
}
inline __m256 _mm256_unpackhi_ps(__m256 m1, __m256 m2)
{
	__m256 r;
	r.f32[0] = m1.f32[2];
	r.f32[1] = m2.f32[2];
	r.f32[2] = m1.f32[3];
	r.f32[3] = m2.f32[3];
	r.f32[4] = m1.f32[6];
	r.f32[5] = m2.f32[6];
	r.f32[6] = m1.f32[7];
	r.f32[7] = m2.f32[7];
	return r;
}
inline __m128 _mm_unpacklo_ps(__m128 m1, __m128 m2)
{
	__m128 r;
	r.f32[0] = m1.f32[0];
	r.f32[1] = m2.f32[0];
	r.f32[2] = m1.f32[1];
	r.f32[3] = m2.f32[1];
	return r;
}
inline __m128 _mm_unpackhi_ps(__m128 m1, __m128 m2)
{
	__m128 r;
	r.f32[0] = m1.f32[2];
	r.f32[1] = m2.f32[2];
	r.f32[2] = m1.f32[3];
	r.f32[3] = m2.f32[3];
	return r;
}

// Permutes 128-bit integer data from source vector a and source vector b using bits in the 8-bit immediate and stores results in the destination vector.
inline __m256i _mm256_permute2x128_si256(__m256i a, __m256i b, int control)
{
	__m256i r;
	if (control == 0x20)
	{
		r.m128i[0] = a.m128i[0];
		r.m128i[1] = b.m128i[1];
	}
	else if (control == 0x31)
	{
		r.m128i[0] = a.m128i[1];
		r.m128i[1] = b.m128i[1];
	}
	return r;
}

// Use two-bit index values in the immediate byte to select a qword integer element from the source vector val. The result element is copied to the corresponding element of destination vector. The intrinsic allows to copy the same element of the source vector to more than one element of the destination vector.
inline __m256i _mm256_permute4x64_epi64(__m256i val, const int control)
{
	__m256i r;
	// bibit_t c;
	// c.x = control;
	r.i64[0] = val.i64[0];
	r.i64[1] = val.i64[2];
	r.i64[2] = val.i64[1];
	r.i64[3] = val.i64[3];
	return r;
}
// Stores the lowest 32 bit float of a into memory.
inline void _mm_store_ss(float *p, __m128 a)
{
	*p = a.f32[0];
}
// Performs a store operation by moving packed single-precision floating point values (float32 values) from a float32 vector, b, to a 256-bit unaligned memory location, pointed to by a.
inline void _mm_storeu_ps(float *p, __m128 a)
{
	for (int i = 0; i < 4; i++)
		p[i] = a.f32[i];
}
// Stores 128-bits of integer data from a into memory.
inline void _mm_storeu_si128(__m128i *p, __m128i a)
{
	*p = a;
}
//a ����λ 2 �Ĥ΅g���ȸ���С���ゎ�򡢥��ɥ쥹 p �˥��ȥ����ޤ���
inline void _mm_storel_pi(__m64 *p, __m128 a)
{
	p[0].m64_i32[0] = a.f32[0];
	p[0].m64_i32[1] = a.f32[1];
}
// Performs a store operation by moving packed single-precision floating point values (float32 values) from a float32 vector, b, to a 256-bit unaligned memory location, pointed to by a.
inline void _mm256_storeu_ps(float *a, __m256 b)
{
	for (int i = 0; i < 8; i++)
		a[i] = b.f32[i];
}
// Performs a store operation by moving integer values from a 256-bit integer vector, b, to a 256-bit unaligned memory location, pointed to by a.
inline void _mm256_storeu_si256(__m256i *a, __m256i b)
{
	a[0] = b;
}
//Conditionally stores 32-bit data elements from the source vector into the corresponding elements of the vector in memory referenced by addr. 
//If an element of mask is 0, corresponding element of the result vector in memory stays unchanged. 
//Only the most significant bit of each element in the vector mask is used.
inline void _mm256_maskstore_epi32(int * addr, __m256i vmask, __m256i val)
{
	for (int i = 0; i < 8; i++)
	{
		if (vmask.i32[i] != 0)
			addr[i] = val.i32[i];
	}
}
// Loads 128-bits of integer data from memory into a new vector.
inline __m128i _mm_loadu_si128(__m128i const *p)
{
	__m128i r;
	r = *p;
	return r;
}
// Loads 64-bit integer from memory into first element of returned vector.
inline __m128i _mm_loadl_epi64(__m128i const*p)
{
	__m128i r;
	r.i32[0] = p->i32[0];
	r.i32[1] = p->i32[1];
	r.i32[2] = 0;
	r.i32[3] = 0;
	return r;
}
inline __m128 _mm_load1_ps(const float * p)
{
	__m128 r;
	r.f32[0] = *p;
	r.f32[1] = *p;
	r.f32[2] = *p;
	r.f32[3] = *p;
	return r;
}
// Loads integer values from the 256-bit unaligned memory location pointed to by *a, into a destination integer vector, which is returned by the intrinsic.
inline __m256i _mm256_loadu_si256(__m256i const * a)
{
	__m256i r;
	for (int i = 0; i < 8; i++)
		r.i32[i] = a->i32[i];
	return r;
}

// Loads packed single-precision floating point values (float32 values) from the 256-bit unaligned memory location pointed to by a, into a destination float32 vector, which is retured by the intrinsic.
inline __m256 _mm256_loadu_ps(float const *a)
{
	__m256 r;
	for (int i = 0; i < 8; i++)
		r.f32[i] = a[i];
	return r;
}
inline __m128 _mm_loadu_ps(float const * p)
{
	__m128 r;
	for (int i = 0; i < 4; i++)
		r.f32[i] = p[i];
	return r;
}
// Conditionally loads 32/64-bit data elements from the memory referenced by the addr and stores it into the corresponding data element of the result vector. 
// If an element of mask is 0, the 32/64-bit zero is written to the corresponding element of the result vector. 
// The mask bit for each data element is the most significant bit of that element in mask.
inline __m256i _mm256_maskload_epi32(int const * addr, __m256i mask)
{
	__m256i r;
	if (mask.i32[0] == 0)
		r.i32[0] = 0;
	else
		r.i32[0] = addr[0];

	if (mask.i32[1] == 0)
		r.i32[1] = 0;
	else
		r.i32[1] = addr[0];

	if (mask.i32[2] == 0)
		r.i32[2] = 0;
	else
		r.i32[2] = addr[0];

	if (mask.i32[3] == 0)
		r.i32[3] = 0;
	else
		r.i32[3] = addr[0];

	if (mask.i32[4] == 0)
		r.i32[4] = 0;
	else
		r.i32[4] = addr[0];

	if (mask.i32[5] == 0)
		r.i32[5] = 0;
	else
		r.i32[5] = addr[0];

	if (mask.i32[6] == 0)
		r.i32[6] = 0;
	else
		r.i32[6] = addr[0];

	if (mask.i32[7] == 0)
		r.i32[7] = 0;
	else
		r.i32[7] = addr[0];
	return r;
}

inline __m256i _mm256_maskload_epi64(int64_t const * addr, __m256i mask)
{
	__m256i r;
	if (mask.i64[0] == 0)
		r.i64[0] = 0;
	else
		r.i64[0] = addr[0];

	if (mask.i64[1] == 0)
		r.i64[1] = 0;
	else
		r.i64[1] = addr[0];

	if (mask.i64[2] == 0)
		r.i64[2] = 0;
	else
		r.i64[2] = addr[0];

	if (mask.i64[3] == 0)
		r.i64[3] = 0;
	else
		r.i64[3] = addr[0];
	return r;
}
// �������֥��`�ɤ򥤥�ǥå����ˤ�ä��x�k���줿�ѥå�����������Ҫ�ؤ��������ޤ���
inline int  _mm_extract_epi32(__m128i src, const int ndx)
{
	return src.i32[ndx];
}
// Extract 128 bits (composed of integer data) from a, selected with imm, and store the result in dst.
inline __m128i _mm256_extracti128_si256(__m256i a, int offset)
{
	return a.m128i[offset];
}
// Extracts 128-bit scalar integer values from the source vector m1, starting from the location specified by the value in the offset parameter.
inline __m128i _mm256_extractf128_si256(__m256i m1, const int offset)
{
	__m128i r;
	r = m1.m128i[offset];
	return r;
}
// Inserts 128-bits of packed integer data from the second source operand (third operand) into the destination operand (first operand) at a 128-bit offset from imm8[0].
// The remaining portions of the destination are written by the corresponding fields of the first source operand (second operand). 
// The high 7 bits of the immediate are ignored.
inline __m256i _mm256_inserti128_si256(__m256i a, __m128i b, const int mask)
{
	a.m128i[mask] = b;
	return a;
}

// Performs a SIMD compare of the packed signed byte integers in source vectors s1 and s2 and returns the maximum value for each pair of integers to the destination vector.
inline __m256i _mm256_max_epi8(__m256i s1, __m256i s2)
{
	__m256i r;
	for (int i = 0; i < 32; i++)
	{
		r.i8[i] = max(s1.i8[i], s2.i8[i]);
	}
	return r;
}
// Performs a SIMD compare of the packed single-precision floating-point (float32) elements in the first source vector m1 and the second source vector m2, and returns the maximum value for each pair.
inline __m256 _mm256_max_ps(__m256 m1, __m256 m2)
{
	__m256 r;
	for (int i = 0; i < 8; i++)
	{
		r.f32[i] = max(m1.f32[i], m2.f32[i]);
	}
	return r;
}
// Compares packed 32-bit integers in a and b, and returns packed maximum values.
inline  __m128i _mm_max_epi32(__m128i a, __m128i b)
{
	__m128i r;
	for (int i = 0; i < 4; i++)
	{
		r.i32[i] = max(a.i32[i], b.i32[i]);
	}
	return r;
}
// Compares packed single-precision (32-bit) floating-point elements in a and b, and return the corresponding maximum values.
inline __m128 _mm_max_ps(__m128 a, __m128 b)
{
	__m128 r;
	for (int i = 0; i < 4; i++)
	{
		r.f32[i] = max(a.f32[i], b.f32[i]);
	}
	return r;
}

inline __m256 _mm256_min_ps(__m256 a, __m256 b)
{
	__m256 r;
	for (int i = 0; i < 8; i++)
	{
		r.f32[i] = min(a.f32[i], b.f32[i]);
	}
	return r;
}
inline __m128 _mm_min_ps(__m128 a, __m128 b)
{
	__m128 r;
	for (int i = 0; i < 4; i++)
	{
		r.f32[i] = min(a.f32[i], b.f32[i]);
	}
	return r;
}

inline __m128i _mm_min_epi32(__m128i a, __m128i b)
{
	__m128i r;
	for (int i = 0; i < 4; i++)
	{
		r.i32[i] = min(a.i32[i], b.i32[i]);
	}
	return r;
}

inline __m256 _mm256_mul_ps(__m256 a, __m256 b)
{
	__m256 r;
	for (int i = 0; i < 8; i++)
	{
		r.f32[i] = a.f32[i] * b.f32[i];
	}
	return r;
}

inline  __m128 _mm_mul_ps(__m128 a, __m128 b)
{
	__m128 r;
	for (int i = 0; i < 4; i++)
	{
		r.f32[i] = a.f32[i] * b.f32[i];
	}
	return r;
}

inline __m256i _mm256_add_epi32(__m256i s1, __m256i s2)
{
	__m256i r;
	for (int i = 0; i < 8; i++)
	{
		r.i32[i] = s1.i32[i] + s2.i32[i];
	}
	return r;
}

inline __m256 _mm256_add_ps(__m256 m1, __m256 m2)
{
	__m256 r;
	for (int i = 0; i < 8; i++)
	{
		r.f32[i] = m1.f32[i] + m2.f32[i];
	}
	return r;
}

inline __m128i _mm_add_epi32(__m128i a, __m128i b)
{
	__m128i r;
	for (int i = 0; i < 4; i++)
	{
		r.i32[i] = a.i32[i] + b.i32[i];
	}
	return r;
}

inline __m128 _mm_add_ps(__m128 a, __m128 b)
{
	__m128 r;
	for (int i = 0; i < 4; i++)
	{
		r.f32[i] = a.f32[i] + b.f32[i];
	}
	return r;
}

inline __m128i _mm_madd_epi16(__m128i a, __m128i b)
{
	__m128i r;
	for (int i = 0; i < 4; i++)
	{
		int32_t t1 = (int32_t)a.i16[i * 2] * (int32_t)b.i16[i * 2];
		int32_t t2 = (int32_t)a.i16[i * 2 + 1] * (int32_t)b.i16[i * 2 + 1];
		r.i32[i] = t1 + t2;
	}
	return r;
}
// Multiplies vertically each unsigned byte of source vector s1 with the corresponding signed byte of source vector s2, producing intermediate, signed 16-bit integers. 
// Each adjacent pair of signed words is added, and the saturated result is packed to the destination vector.
inline __m256i _mm256_maddubs_epi16(__m256i s1, __m256i s2)
{
	__m256i r;
	for (int i = 0; i < 16; i++)
	{
		int32_t t1 = (int32_t)s1.u8[i * 2] * (int32_t)s2.i8[i * 2];
		int32_t t2 = (int32_t)s1.u8[i * 2 + 1] * (int32_t)s2.i8[i * 2 + 1];
		r.i16[i] = saturate_cast_s32s16(t1 + t2);
	}
	return r;
}

inline __m128i _mm_maddubs_epi16(__m128i a, __m128i b)
{
	__m128i r;
	for (int i = 0; i < 8; i++)
	{
		int32_t t1 = (int32_t)a.u8[i * 2] * (int32_t)b.i8[i * 2];
		int32_t t2 = (int32_t)a.u8[i * 2 + 1] * (int32_t)b.i8[i * 2 + 1];
		r.i16[i] = saturate_cast_s32s16(t1 + t2);
	}
	return r;
}

inline __m256 _mm256_cmp_ps(__m256 a, __m256 b, const int predicate)
{
	__m256 r;
	for (int i = 0; i < 8; i++)
	{
		if (predicate == _CMP_LE_OS)
			r.f32[i] = a.f32[i] <= b.f32[i] ? 1 : 0;
		else if (predicate == _CMP_GT_OS)
			r.f32[i] = a.f32[i] > b.f32[i] ? 1 : 0;
	}
	return r;
}
inline __m128 _mm_cmp_ps(__m128 a, __m128 b, const int predicate)
{
	__m128 r;
	for (int i = 0; i < 4; i++)
	{
		if (predicate == _CMP_LE_OS)
			r.f32[i] = a.f32[i] <= b.f32[i] ? 1 : 0;
		else if (predicate == _CMP_GT_OS)
			r.f32[i] = a.f32[i] > b.f32[i] ? 1 : 0;
	}
	return r;
}
inline __m128 _mm_cmple_ps(__m128 a, __m128 b)
{
	__m128 r;
	for (int i = 0; i < 4; i++)
	{
		r.f32[i] = a.f32[i] <= b.f32[i] ? 1 : 0;
	}
	return r;
}
inline __m256 _mm256_blendv_ps(__m256 a, __m256 b, __m256 mask)
{
	__m256 r;
	r.f32[0] = (mask.i32[0] & 0x80000000) ? b.f32[0] : a.f32[0];
	r.f32[1] = (mask.i32[1] & 0x80000000) ? b.f32[1] : a.f32[1];
	r.f32[2] = (mask.i32[2] & 0x80000000) ? b.f32[2] : a.f32[2];
	r.f32[3] = (mask.i32[3] & 0x80000000) ? b.f32[3] : a.f32[3];
	r.f32[4] = (mask.i32[4] & 0x80000000) ? b.f32[4] : a.f32[4];
	r.f32[5] = (mask.i32[5] & 0x80000000) ? b.f32[5] : a.f32[5];
	r.f32[6] = (mask.i32[6] & 0x80000000) ? b.f32[6] : a.f32[6];
	r.f32[7] = (mask.i32[7] & 0x80000000) ? b.f32[7] : a.f32[7];
	return r;
}

inline __m128 _mm_blendv_ps(
	__m128 a,
	__m128 b,
	__m128 mask
)
{
	__m128 r;
	r.f32[0] = (mask.i32[0] & 0x80000000) ? b.f32[0] : a.f32[0];
	r.f32[1] = (mask.i32[1] & 0x80000000) ? b.f32[1] : a.f32[1];
	r.f32[2] = (mask.i32[2] & 0x80000000) ? b.f32[2] : a.f32[2];
	r.f32[3] = (mask.i32[3] & 0x80000000) ? b.f32[3] : a.f32[3];
	return r;
}
/////////////////////////////////////////////////////////////////
inline __m256 _mm256_cvtepi32_ps(__m256i m1)
{
	__m256 r;
	for (int i = 0; i < 8; i++)
		r.f32[i] = (float)m1.i32[i];
	return r;
}

// The _mm256_packs_epi16 intrinsic converts 16 packed signed word integers from the first and the second source operands into 32 packed signed byte integers. 
inline __m256i _mm256_packs_epi16(__m256i a, __m256i b)
{
	__m256i r;
	for (int i = 0; i < 16; i++)
	{
		r.i8[i] = saturate_cast_s16s8(a.i16[i]);
		r.i8[i + 16] = saturate_cast_s16s8(b.i16[i]);
	}
	return r;
}
/////////////////////////////////////////////////////////////////
// The _mm256_packs_epi32 intrinsic converts eight packed signed doubleword integers from the first and the second source operands into 16 packed signed word integers.
inline __m256i _mm256_packs_epi32(__m256i a, __m256i b)
{
	__m256i r;
	for (int i = 0; i < 8; i++)
	{
		r.i16[i] = saturate_cast_s32s16(a.i32[i]);
		r.i16[i + 8] = saturate_cast_s32s16(b.i32[i]);
	}
	return r;
}
// Performs a SIMD add of the packed, signed, 16-bit integer data elements with saturation from the first source vector, s1, and corresponding elements of the second source vector, s2, and stores the packed integer results in the destination vector. 
// When an individual word result is beyond the range of a signed word integer (that is, greater than 7FFFH or less than 8000H), the saturated value of 7FFFH or 8000H, respectively, is written to the destination vector.
inline __m256i _mm256_adds_epi16(__m256i s1, __m256i s2)
{
	__m256i r;
	for (int i = 0; i < 8; i++)
	{
		int32_t t1 = (int32_t)s1.i16[i * 2] * (int32_t)s2.i16[i * 2];
		int32_t t2 = (int32_t)s1.i16[i * 2 + 1] * (int32_t)s2.i16[i * 2 + 1];
		r.i32[i] = saturate_cast_s32s16(t1 + t2);
	}
	return r;
}
// Multiplies individual, signed 16-bit integers of source vector s1 by the corresponding signed 16-bit integers of source vector s2, producing temporary, signed, 32-bit [doubleword] results. 
// The adjacent doubleword results are then summed and stored in the destination vector.
inline __m256i _mm256_madd_epi16(__m256i s1, __m256i s2)
{
	__m256i r;
	for (int i = 0; i < 8; i++)
	{
		int32_t t1 = (int32_t)s1.i16[i * 2] * (int32_t)s2.i16[i * 2];
		int32_t t2 = (int32_t)s1.i16[i * 2 + 1] * (int32_t)s2.i16[i * 2 + 1];
		r.i32[i] = t1 + t2;
	}
	return r;
}
/// @brief /////////////////////////////////////////
inline __m256 _mm256_shuffle_ps(__m256 m1, __m256 m2, const int select)
{
	__m256 r;
	r.f32[0] = m1.f32[0x3 & select];
	r.f32[1] = m1.f32[0x3 & (select >> 2)];
	r.f32[2] = m2.f32[0x3 & (select >> 4)];
	r.f32[3] = m2.f32[0x3 & (select >> 6)];

	r.f32[4] = m1.f32[0x3 & (select >> 8)];
	r.f32[5] = m1.f32[0x3 & (select >> 10)];
	r.f32[6] = m2.f32[0x3 & (select >> 12)];
	r.f32[7] = m2.f32[0x3 & (select >> 14)];
	return r;
}
inline __m128 _mm_shuffle_ps(__m128 m1, __m128 m2, unsigned int imm8)
{
	__m128 r;
	r.f32[0] = m1.f32[0x3 & imm8];
	r.f32[1] = m1.f32[0x3 & (imm8 >> 2)];
	r.f32[2] = m2.f32[0x3 & (imm8 >> 4)];
	r.f32[3] = m2.f32[0x3 & (imm8 >> 6)];
	return r;
}
inline __m256i _mm256_set_epi32(int e7, int e6, int e5, int e4, int e3, int e2, int e1, int e0)
{
	__m256i r;
	r.i32[0] = e0;
	r.i32[1] = e1;
	r.i32[2] = e2;
	r.i32[3] = e3;
	r.i32[4] = e4;
	r.i32[5] = e4;
	r.i32[6] = e6;
	r.i32[7] = e7;
	return r;
}
inline __m128i _mm_setr_epi32(int i0, int i1, int i2, int i3)
{
	__m128i r;
	r.i32[0] = i0;
	r.i32[1] = i1;
	r.i32[2] = i2;
	r.i32[3] = i3;
	return r;
}
inline __m256 _mm256_blend_ps(__m256 m1, __m256 m2, const int mask)
{
	__m256 r;
	if ((0x1 & mask) == 0)
	{
		r.f32[0] = m1.f32[0];
	}
	if ((0x1 & mask) == 1)
	{
		r.f32[0] = m2.f32[0];
	}

	if ((0x1 & (mask >> 1)) == 0)
	{
		r.f32[1] = m1.f32[1];
	}
	if ((0x1 & (mask >> 1)) == 1)
	{
		r.f32[1] = m2.f32[1];
	}

	if ((0x1 & (mask >> 2)) == 0)
	{
		r.f32[2] = m1.f32[2];
	}
	if ((0x1 & (mask >> 2)) == 1)
	{
		r.f32[2] = m2.f32[2];
	}

	if ((0x1 & (mask >> 3)) == 0)
	{
		r.f32[3] = m1.f32[3];
	}
	if ((0x1 & (mask >> 3)) == 1)
	{
		r.f32[3] = m2.f32[3];
	}

	if ((0x1 & (mask >> 4)) == 0)
	{
		r.f32[4] = m1.f32[4];
	}
	if ((0x1 & (mask >> 4)) == 1)
	{
		r.f32[4] = m2.f32[4];
	}

	if ((0x1 & (mask >> 5)) == 0)
	{
		r.f32[5] = m1.f32[5];
	}
	if ((0x1 & (mask >> 5)) == 1)
	{
		r.f32[5] = m2.f32[5];
	}

	if ((0x1 & (mask >> 6)) == 0)
	{
		r.f32[6] = m1.f32[6];
	}
	if ((0x1 & (mask >> 6)) == 1)
	{
		r.f32[6] = m2.f32[6];
	}

	if ((0x1 & (mask >> 7)) == 0)
	{
		r.f32[7] = m1.f32[7];
	}
	if ((0x1 & (mask >> 7)) == 1)
	{
		r.f32[7] = m2.f32[7];
	}
	return r;
}
inline __m128  _mm_blend_ps(__m128  m1, __m128  m2, const int mask)
{
	__m128 r;
	if ((0x1 & mask) == 0)
	{
		r.f32[0] = m1.f32[0];
	}
	if ((0x1 & mask) == 1)
	{
		r.f32[0] = m2.f32[0];
	}

	if ((0x1 & (mask >> 1)) == 0)
	{
		r.f32[1] = m1.f32[1];
	}
	if ((0x1 & (mask >> 1)) == 1)
	{
		r.f32[1] = m2.f32[1];
	}

	if ((0x1 & (mask >> 2)) == 0)
	{
		r.f32[2] = m1.f32[2];
	}
	if ((0x1 & (mask >> 2)) == 1)
	{
		r.f32[2] = m2.f32[2];
	}

	if ((0x1 & (mask >> 3)) == 0)
	{
		r.f32[3] = m1.f32[3];
	}
	if ((0x1 & (mask >> 3)) == 1)
	{
		r.f32[3] = m2.f32[3];
	}
	return r;
}
inline __m256 _mm256_permutevar8x32_ps(__m256 val, __m256i offsets)
{
	__m256 r;
	r.f32[0] = val.f32[offsets.i32[0] & 0x11];
	r.f32[1] = val.f32[offsets.i32[1] & 0x11];
	r.f32[2] = val.f32[offsets.i32[2] & 0x11];
	r.f32[3] = val.f32[offsets.i32[3] & 0x11];
	r.f32[4] = val.f32[offsets.i32[4] & 0x11];
	r.f32[5] = val.f32[offsets.i32[5] & 0x11];
	r.f32[6] = val.f32[offsets.i32[6] & 0x11];
	r.f32[7] = val.f32[offsets.i32[7] & 0x11];
	return r;
}
inline __m256 _mm256_permute2f128_ps(__m256 m1, __m256 m2, int control)
{
	__m256 r;
	if (control == 0x20)
	{
		r.f32[0] = m1.f32[0];
		r.f32[1] = m1.f32[1];
		r.f32[2] = m1.f32[2];
		r.f32[3] = m1.f32[3];
		r.f32[4] = m2.f32[0];
		r.f32[5] = m2.f32[1];
		r.f32[6] = m2.f32[2];
		r.f32[7] = m2.f32[3];
	}
	else 	if (control == 0x31)
	{
		r.f32[0] = m1.f32[4];
		r.f32[1] = m1.f32[5];
		r.f32[2] = m1.f32[6];
		r.f32[3] = m1.f32[7];
		r.f32[4] = m2.f32[4];
		r.f32[5] = m2.f32[5];
		r.f32[6] = m2.f32[6];
		r.f32[7] = m2.f32[7];
	}
	return r;
}
inline __m256 _mm256_fmadd_ps(__m256 a, __m256 b, __m256 c)
{
	__m256 r;
	for (int i = 0; i < 8; i++)
		r.f32[i] = a.f32[i] * b.f32[i] + c.f32[i];
	return r;
}
inline __m128 _mm256_extractf128_ps(__m256 m1, const int offset)
{
	__m128 r;
	if (offset == 0)
	{
		for (int i = 0; i < 4; i++)
			r.f32[i] = m1.f32[i];
	}
	else if (offset == 1)
		for (int i = 4; i < 8; i++)
			r.f32[i - 4] = m1.f32[i];
	return r;
}


inline void _mm256_maskstore_ps(float *a, __m256i mask, __m256 b)
{
	for (int i = 0; i < 8; i++)
	{
		if (mask.i32[i] != 0)
			a[i] = b.f32[i];
	}
}
inline void _mm_maskstore_ps(float *a, __m128i mask, __m128 b)
{
	for (int i = 0; i < 4; i++)
	{
		if (mask.i32[i] != 0)
			a[i] = b.f32[i];
	}
}
//////////////////////////////////////////////
inline __m128 _mm_movehl_ps(__m128 a, __m128 b)
{
	__m128 r;
	r.f32[0] = b.f32[2];
	r.f32[1] = b.f32[3];
	r.f32[2] = a.f32[2];
	r.f32[3] = a.f32[3];
	return r;
}
inline __m128 _mm_movelh_ps(__m128 a, __m128 b)
{
	__m128 r;
	r.f32[0] = a.f32[0];
	r.f32[1] = a.f32[1];
	r.f32[2] = b.f32[0];
	r.f32[3] = b.f32[1];
	return r;
}

