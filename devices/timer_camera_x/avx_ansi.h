
#pragma once
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <assert.h>
#include <float.h>
#include <limits.h>

#define max(x, y) (((x) > (y)) ? (x) : (y))
#define min(x, y) (((x) < (y)) ? (x) : (y))

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

int8_t saturate_cast_s32s8(int v);

int8_t saturate_cast_i2s8(int v);

int saturate_cast_f2i(float v);

int8_t saturate_cast_f32s8(float v);

int16_t saturate_cast_s32s16(int v);

uint8_t saturate_cast_s32u8(int v);

uint8_t saturate_cast_s16u8(int16_t v);

int8_t saturate_cast_s16s8(int16_t v);

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
__m128i _mm_set1_epi16(short w);
// Sets the four signed 32-bit integer values to i.
__m128i _mm_set1_epi32(int _i);
//  This returns a __m128 vector, where all four elements of the vector are set equal to a, i.e. the vector is [a,a,a,a].
__m128 _mm_set1_ps(float a);

// Initializes a 256-bit vector with scalar integer values (8/16/32-bit values) as specified by the a parameter.
__m256i _mm256_set1_epi8(char a);

__m256i _mm256_set1_epi16(short a);

__m256i _mm256_set1_epi32(int a);

__m256 _mm256_set1_ps(float a);

// Sets the 128/256-bit value to zero.
__m128i _mm_setzero_si128(void);

__m256i _mm256_setzero_si256(void);

__m256 _mm256_setzero_ps(void);

__m128 _mm_setzero_ps(void);

// The _mm256_packus_epi16 intrinsic converts 16 packed signed word integers from source operands a and b into 32 packed unsigned byte integers. 
__m256i _mm256_packus_epi16(__m256i a, __m256i b);

__m128i _mm_packs_epi16(__m128i a, __m128i b);

__m128i _mm_packs_epi32(__m128i a, __m128i b);

// Unpacks and interleaves the low-order signed or unsigned data elements (bytes, words, doublewords) of the source vector and the low-order signed or unsigned data elements (bytes, words, doublewords) in the destination operand. The high-order data elements are ignored.
__m256i _mm256_unpacklo_epi8(__m256i a, __m256i b);

__m256i _mm256_unpacklo_epi16(__m256i a, __m256i b);

__m256i _mm256_unpacklo_epi32(__m256i a, __m256i b);

// Unpacks and interleaves the high-order signed or unsigned data elements (bytes, words, doublewords) of the source vector and the high-order signed or unsigned data elements (bytes, words, doublewords) in the destination vector. The low-order data elements are ignored.
__m256i _mm256_unpackhi_epi8(__m256i a, __m256i b);

__m256i _mm256_unpackhi_epi16(__m256i a, __m256i b);

__m256i _mm256_unpackhi_epi32(__m256i a, __m256i b);

// Interleaves the lower two signed or unsigned 32-bit integers in a with the lower two signed or unsigned 32-bit integers in b.
__m128i _mm_unpacklo_epi32(__m128i a, __m128i b);

// Interleaves the upper two signed or unsigned 32-bit integers in a with the upper two signed or unsigned 32-bit integers in b.
__m128i _mm_unpackhi_epi32(__m128i a, __m128i b);

__m256 _mm256_unpacklo_ps(__m256 m1, __m256 m2);

__m256 _mm256_unpackhi_ps(__m256 m1, __m256 m2);

__m128 _mm_unpacklo_ps(__m128 m1, __m128 m2);

__m128 _mm_unpackhi_ps(__m128 m1, __m128 m2);

// Permutes 128-bit integer data from source vector a and source vector b using bits in the 8-bit immediate and stores results in the destination vector.
__m256i _mm256_permute2x128_si256(__m256i a, __m256i b, int control);

// Use two-bit index values in the immediate byte to select a qword integer element from the source vector val. The result element is copied to the corresponding element of destination vector. The intrinsic allows to copy the same element of the source vector to more than one element of the destination vector.
__m256i _mm256_permute4x64_epi64(__m256i val, const int control);

// Stores the lowest 32 bit float of a into memory.
void _mm_store_ss(float *p, __m128 a);

// Performs a store operation by moving packed single-precision floating point values (float32 values) from a float32 vector, b, to a 256-bit unaligned memory location, pointed to by a.
void _mm_storeu_ps(float *p, __m128 a);

// Stores 128-bits of integer data from a into memory.
void _mm_storeu_si128(__m128i *p, __m128i a);

void _mm_storel_pi(__m64 *p, __m128 a);

// Performs a store operation by moving packed single-precision floating point values (float32 values) from a float32 vector, b, to a 256-bit unaligned memory location, pointed to by a.
void _mm256_storeu_ps(float *a, __m256 b);

// Performs a store operation by moving integer values from a 256-bit integer vector, b, to a 256-bit unaligned memory location, pointed to by a.
void _mm256_storeu_si256(__m256i *a, __m256i b);

//Conditionally stores 32-bit data elements from the source vector into the corresponding elements of the vector in memory referenced by addr. 
//If an element of mask is 0, corresponding element of the result vector in memory stays unchanged. 
//Only the most significant bit of each element in the vector mask is used.
void _mm256_maskstore_epi32(int * addr, __m256i vmask, __m256i val);

// Loads 128-bits of integer data from memory into a new vector.
__m128i _mm_loadu_si128(__m128i const *p);

// Loads 64-bit integer from memory into first element of returned vector.
__m128i _mm_loadl_epi64(__m128i const*p);

__m128 _mm_load1_ps(const float * p);

// Loads integer values from the 256-bit unaligned memory location pointed to by *a, into a destination integer vector, which is returned by the intrinsic.
__m256i _mm256_loadu_si256(__m256i const * a);

// Loads packed single-precision floating point values (float32 values) from the 256-bit unaligned memory location pointed to by a, into a destination float32 vector, which is retured by the intrinsic.
__m256 _mm256_loadu_ps(float const *a);

__m128 _mm_loadu_ps(float const * p);

// Conditionally loads 32/64-bit data elements from the memory referenced by the addr and stores it into the corresponding data element of the result vector. 
// If an element of mask is 0, the 32/64-bit zero is written to the corresponding element of the result vector. 
// The mask bit for each data element is the most significant bit of that element in mask.
__m256i _mm256_maskload_epi32(int const * addr, __m256i mask);

__m256i _mm256_maskload_epi64(int64_t const * addr, __m256i mask);

int  _mm_extract_epi32(__m128i src, const int ndx);

// Extract 128 bits (composed of integer data) from a, selected with imm, and store the result in dst.
__m128i _mm256_extracti128_si256(__m256i a, int offset);

// Extracts 128-bit scalar integer values from the source vector m1, starting from the location specified by the value in the offset parameter.
__m128i _mm256_extractf128_si256(__m256i m1, const int offset);

// Inserts 128-bits of packed integer data from the second source operand (third operand) into the destination operand (first operand) at a 128-bit offset from imm8[0].
// The remaining portions of the destination are written by the corresponding fields of the first source operand (second operand). 
// The high 7 bits of the immediate are ignored.
__m256i _mm256_inserti128_si256(__m256i a, __m128i b, const int mask);

// Performs a SIMD compare of the packed signed byte integers in source vectors s1 and s2 and returns the maximum value for each pair of integers to the destination vector.
__m256i _mm256_max_epi8(__m256i s1, __m256i s2);

// Performs a SIMD compare of the packed single-precision floating-point (float32) elements in the first source vector m1 and the second source vector m2, and returns the maximum value for each pair.
__m256 _mm256_max_ps(__m256 m1, __m256 m2);

// Compares packed 32-bit integers in a and b, and returns packed maximum values.
 __m128i _mm_max_epi32(__m128i a, __m128i b);

// Compares packed single-precision (32-bit) floating-point elements in a and b, and return the corresponding maximum values.
__m128 _mm_max_ps(__m128 a, __m128 b);

__m256 _mm256_min_ps(__m256 a, __m256 b);

__m128 _mm_min_ps(__m128 a, __m128 b);

__m128i _mm_min_epi32(__m128i a, __m128i b);

__m256 _mm256_mul_ps(__m256 a, __m256 b);

 __m128 _mm_mul_ps(__m128 a, __m128 b);

__m256i _mm256_add_epi32(__m256i s1, __m256i s2);

__m256 _mm256_add_ps(__m256 m1, __m256 m2);

__m128i _mm_add_epi32(__m128i a, __m128i b);

__m128 _mm_add_ps(__m128 a, __m128 b);

__m128i _mm_madd_epi16(__m128i a, __m128i b);

// Multiplies vertically each unsigned byte of source vector s1 with the corresponding signed byte of source vector s2, producing intermediate, signed 16-bit integers. 
// Each adjacent pair of signed words is added, and the saturated result is packed to the destination vector.
__m256i _mm256_maddubs_epi16(__m256i s1, __m256i s2);

__m128i _mm_maddubs_epi16(__m128i a, __m128i b);

__m256 _mm256_cmp_ps(__m256 a, __m256 b, const int predicate);

__m128 _mm_cmp_ps(__m128 a, __m128 b, const int predicate);

__m128 _mm_cmple_ps(__m128 a, __m128 b);

__m256 _mm256_blendv_ps(__m256 a, __m256 b, __m256 mask);

__m128 _mm_blendv_ps(
	__m128 a,
	__m128 b,
	__m128 mask
);
/////////////////////////////////////////////////////////////////
__m256 _mm256_cvtepi32_ps(__m256i m1);

// The _mm256_packs_epi16 intrinsic converts 16 packed signed word integers from the first and the second source operands into 32 packed signed byte integers. 
__m256i _mm256_packs_epi16(__m256i a, __m256i b);

/////////////////////////////////////////////////////////////////
// The _mm256_packs_epi32 intrinsic converts eight packed signed doubleword integers from the first and the second source operands into 16 packed signed word integers.
__m256i _mm256_packs_epi32(__m256i a, __m256i b);
// Performs a SIMD add of the packed, signed, 16-bit integer data elements with saturation from the first source vector, s1, and corresponding elements of the second source vector, s2, and stores the packed integer results in the destination vector. 
// When an individual word result is beyond the range of a signed word integer (that is, greater than 7FFFH or less than 8000H), the saturated value of 7FFFH or 8000H, respectively, is written to the destination vector.
__m256i _mm256_adds_epi16(__m256i s1, __m256i s2);
// Multiplies individual, signed 16-bit integers of source vector s1 by the corresponding signed 16-bit integers of source vector s2, producing temporary, signed, 32-bit [doubleword] results. 
// The adjacent doubleword results are then summed and stored in the destination vector.
__m256i _mm256_madd_epi16(__m256i s1, __m256i s2);
/// @brief /////////////////////////////////////////
__m256 _mm256_shuffle_ps(__m256 m1, __m256 m2, const int select);

__m128 _mm_shuffle_ps(__m128 m1, __m128 m2, unsigned int imm8);

__m256i _mm256_set_epi32(int e7, int e6, int e5, int e4, int e3, int e2, int e1, int e0);

__m128i _mm_setr_epi32(int i0, int i1, int i2, int i3);

__m256 _mm256_blend_ps(__m256 m1, __m256 m2, const int mask);

__m128  _mm_blend_ps(__m128  m1, __m128  m2, const int mask);

__m256 _mm256_permutevar8x32_ps(__m256 val, __m256i offsets);

__m256 _mm256_permute2f128_ps(__m256 m1, __m256 m2, int control);

__m256 _mm256_fmadd_ps(__m256 a, __m256 b, __m256 c);

__m128 _mm256_extractf128_ps(__m256 m1, const int offset);

void _mm256_maskstore_ps(float *a, __m256i mask, __m256 b);

void _mm_maskstore_ps(float *a, __m128i mask, __m128 b);
//////////////////////////////////////////////
__m128 _mm_movehl_ps(__m128 a, __m128 b);

__m128 _mm_movelh_ps(__m128 a, __m128 b);

