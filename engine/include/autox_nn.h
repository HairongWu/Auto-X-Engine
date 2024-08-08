
#pragma once

#include <stdint.h>
#include <stdbool.h>
#include <stdlib.h>
#include <ctype.h>
#include <math.h>
#include <string.h>
#include <float.h>
#include <stdio.h>

typedef int autox_err_t;

/* Definitions for error constants. */
#define AUTOX_OK          0       /*!< autox_err_t value indicating success (no error) */
#define AUTOX_FAIL        -1      /*!< Generic autox_err_t code indicating failure */

#define AUTOX_ERR_NO_MEM              0x101   /*!< Out of memory */
#define AUTOX_ERR_INVALID_ARG         0x102   /*!< Invalid argument */
#define AUTOX_ERR_INVALID_STATE       0x103   /*!< Invalid state */
#define AUTOX_ERR_INVALID_SIZE        0x104   /*!< Invalid size */
#define AUTOX_ERR_NOT_FOUND           0x105   /*!< Requested resource not found */
#define AUTOX_ERR_NOT_SUPPORTED       0x106   /*!< Operation or feature not supported */
#define AUTOX_ERR_TIMEOUT             0x107   /*!< Operation timed out */
#define AUTOX_ERR_INVALID_RESPONSE    0x108   /*!< Received response was invalid */
#define AUTOX_ERR_INVALID_CRC         0x109   /*!< CRC or checksum was invalid */
#define AUTOX_ERR_INVALID_VERSION     0x10A   /*!< Version was invalid */
#define AUTOX_ERR_INVALID_MAC         0x10B   /*!< MAC address was invalid */
#define AUTOX_ERR_NOT_FINISHED        0x10C   /*!< Operation has not fully completed */
#define AUTOX_ERR_NOT_ALLOWED         0x10D   /*!< Operation is not allowed */

#define AUTOX_ERR_WIFI_BASE           0x3000  /*!< Starting number of WiFi error codes */
#define AUTOX_ERR_MESH_BASE           0x4000  /*!< Starting number of MESH error codes */
#define AUTOX_ERR_FLASH_BASE          0x6000  /*!< Starting number of flash error codes */
#define AUTOX_ERR_HW_CRYPTO_BASE      0xc000  /*!< Starting number of HW cryptography module error codes */
#define AUTOX_ERR_MEMPROT_BASE        0xd000  /*!< Starting number of Memory Protection API error codes */

enum ActivationType {
    kIndentity = 0,
    kRelu = 1,
    kRelu6 = 2,
    kPRelu = 3,
    kLeakyRelu = 4,
    kSigmoid = 5,
    kTanh = 6,
    kSwish = 7,
    kExp = 8,
    kAbs = 9,
    kHardSwish = 10,
    kReciprocal = 11,
    kThresholdedRelu = 12,
    kElu = 13,
    kHardSigmoid = 14,
    kLog = 15,
    kSigmoid_v2 = 16,
    kTanh_v2 = 17,
    kGelu = 18,
    kErf = 19,
    kSign = 20,
    kSoftPlus = 21,
    kMish = 22,
    kSilu = 23,
    kLog1p = 24,
    NUM = 25,
};
// ----------------------------------------------------------------------------
// The Sampler, which takes logits and returns a sampled token
// sampling can be done in a few ways: greedy argmax, sampling, top-p sampling

typedef struct {
    float prob;
    int index;
} ProbIndex; // struct used when sorting probabilities during top-p sampling

typedef struct {
    int vocab_size;
    ProbIndex* probindex; // buffer used in top-p sampling
    float temperature;
    float topp;
    unsigned long long rng_state;
} Sampler;
// ----------------------------------------------------------------------------
// The Byte Pair Encoding (BPE) Tokenizer that translates strings <-> tokens

typedef struct {
    char *str;
    int id;
} TokenIndex;

typedef struct {
    char** vocab;
    float* vocab_scores;
    TokenIndex *sorted_vocab;
    int vocab_size;
    unsigned int max_token_length;
    unsigned char byte_pieces[512]; // stores all single-byte strings
} Tokenizer;

typedef struct {
    int dim; // transformer dimension
    int hidden_dim; // for ffn layers
    int n_layers; // number of layers
    int n_heads; // number of query heads
    int n_kv_heads; // number of key/value heads (can be < query heads because of multiquery)
    int vocab_size; // vocabulary size, usually 256 (byte-level)
    int seq_len; // max sequence length
} Config;

typedef struct {
    // token embedding table
    float* token_embedding_table;    // (vocab_size, dim)
    // weights for rmsnorms
    float* rms_att_weight; // (layer, dim) rmsnorm weights
    float* rms_ffn_weight; // (layer, dim)
    // weights for matmuls. note dim == n_heads * head_size
    float* wq; // (layer, dim, n_heads * head_size)
    float* wk; // (layer, dim, n_kv_heads * head_size)
    float* wv; // (layer, dim, n_kv_heads * head_size)
    float* wo; // (layer, n_heads * head_size, dim)
    // weights for ffn
    float* w1; // (layer, hidden_dim, dim)
    float* w2; // (layer, dim, hidden_dim)
    float* w3; // (layer, hidden_dim, dim)
    // final rmsnorm
    float* rms_final_weight; // (dim,)
    // (optional) classifier weights for the logits, on the last layer
    float* wcls;
} TransformerWeights;

typedef struct {
    // current wave of activations
    float *x; // activation at current time stamp (dim,)
    float *xb; // same, but inside a residual branch (dim,)
    float *xb2; // an additional buffer just for convenience (dim,)
    float *hb; // buffer for hidden dimension in the ffn (hidden_dim,)
    float *hb2; // buffer for hidden dimension in the ffn (hidden_dim,)
    float *q; // query (dim,)
    float *k; // key (dim,)
    float *v; // value (dim,)
    float *att; // buffer for scores/attention values (n_heads, seq_len)
    float *logits; // output logits
    // kv cache
    float* key_cache;   // (layer, seq_len, dim)
    float* value_cache; // (layer, seq_len, dim)
} RunState;

typedef struct {
    Config config; // the hyperparameters of the architecture (the blueprint)
    TransformerWeights weights; // the weights of the model
    RunState state; // buffers for the "wave" of activations in the forward pass
    // some more state needed to properly clean up the memory mapping (sigh)
    int fd; // file descriptor for memory mapping
    float* data; // memory mapped data pointer
    uint32_t file_size; // size of the checkpoint file in bytes
} Transformer;

#define max(x, y) (((x) > (y)) ? (x) : (y))
#define min(x, y) (((x) < (y)) ? (x) : (y))

inline uint32_t count(const uint16_t *ddim, uint8_t start, uint8_t end)
{
  start = max(start, 0);
  if (end < start) {
    return 0;
  }
  uint32_t sum = 1;
  for (uint8_t i = start; i < end; ++i) {
    sum *= ddim[i];
  }
  return sum;
}

#if defined(_MSC_VER)
     /* Microsoft C/C++-compatible compiler */
     #include <intrin.h>
#elif defined(__GNUC__) && (defined(__x86_64__) || defined(__i386__))
     /* GCC-compatible compiler, targeting x86/x86-64 */
     #include <x86intrin.h>
#elif defined(__GNUC__) && defined(__ARM_NEON__)
     /* GCC-compatible compiler, targeting ARM with NEON */
     #include <arm_neon.h>
#elif defined(__GNUC__) && defined(__IWMMXT__)
     /* GCC-compatible compiler, targeting ARM with WMMX */
     #include <mmintrin.h>
#elif (defined(__GNUC__) || defined(__xlC__)) && (defined(__VEC__) || defined(__ALTIVEC__))
     /* XLC or GCC-compatible compiler, targeting PowerPC with VMX/VSX */
     #include <altivec.h>
#elif defined(__GNUC__) && defined(__SPE__)
     /* GCC-compatible compiler, targeting PowerPC with SPE */
     #include <spe.h>
#else
     #include "avx_ansi.h"

#endif

#ifdef  __cplusplus
extern "C" {
#endif

    void gemm_cpu(int TA, int TB, int M, int N, int K, float ALPHA,
        const float* A, int lda,
        const float* B, int ldb,
        float BETA,
        float* C, int ldc);

    void autox_hwc2chw(const float* src, float* dst, uint16_t height, uint16_t width, uint8_t channels);
    void autox_normalize_image(uint8_t* p, float* out, uint32_t p_h, uint32_t p_w, uint8_t p_c);
    void autox_resize_image(const uint8_t* src, uint8_t* dst, uint16_t ssize_h, uint16_t ssize_w, uint16_t dsize_h, uint16_t dsize_w);

    void autox_argmax(const float* input,
        float* output,
        const uint16_t* input_ddim,
        const uint16_t* output_ddim,
        const uint8_t input_ddim_size,
        const uint8_t output_ddim_size,
        int8_t axis);

    void autox_concat(float** inputs, float* output, uint16_t* input_dims[], uint16_t* output_dims,
        int8_t axis, uint8_t input_size, uint8_t dim_0_size);

    void autox_conv2d_depthwise(const float* i_data, float* o_data, const float* w_data, const float* b_data,
        uint16_t* x_dims, uint16_t* w_dims, uint16_t* o_dims, const int stride, int dilations, int paddings, int flag_bias, int act_param);

    void autox_conv2d(float* din, const float* bias, const float* weights, float* dout, uint16_t* x_dims, uint16_t* w_dims, uint16_t* o_dims,
        uint16_t group, uint8_t paddings, uint8_t strides, uint8_t dilations, int8_t act_type);

    void autox_elementwise_add(float* x_data,
        float* y_data,
        float* out_data, uint16_t* x_dims, uint16_t* y_dims, uint16_t* z_dims, int axis, uint16_t x_dims_size,
        uint16_t y_dims_size, uint16_t z_dims_size);

    void autox_elementwise_mul(float* x_data,
        float* y_data,
        float* out_data, uint16_t* x_dims, uint16_t* y_dims, uint16_t* z_dims, int axis, uint16_t x_dims_size,
        uint16_t y_dims_size, uint16_t z_dims_size);

    void autox_hard_sigmoid(float* data, uint16_t* dims, uint8_t dim_size, float offset, float slope);
    void autox_hard_swish(float* x_data, uint16_t* dims, uint8_t dim_size, float threshold, float scale, float offset);
    void autox_swish(float* x_data, uint16_t* dims, uint8_t dim_size, float beta);
    void autox_im2col(const float* data_im,
        int channels,
        int height,
        int width,
        int kernel,
        int pad,
        int stride,
        int dilation,
        float* data_col);

    void autox_bilinear_interp(float* X,
        float* Out,
        uint16_t* intput_dims,
        uint16_t* out_dims,
        float scale,
        int8_t align_corners,
        int align_mode);
    void autox_nearest_interp(float* X,
        float* Out,
        uint16_t* intput_dims,
        uint16_t* out_dims,
        float scale,
        int8_t align_corners);

    void autox_layer_norm(float* x,
        const float* bias,
        const float* scale,
        float* out,
        int height,
        const float epsilon,
        int right);

    void autox_matmul(const float* X, const float* Y, float* Out, uint16_t* x_dims,
        uint16_t* y_dims, uint16_t* o_dims, int8_t x_transpose, int8_t y_transpose,
        uint8_t x_dims_size, uint8_t y_dims_size, uint8_t o_dims_size);

    void autox_pool2d(const float* input_data, float* output_data, uint16_t* x_dims, uint16_t* o_dims, const uint8_t ksize,
        const uint8_t stride, const uint8_t padding, const uint8_t adaptive, const uint8_t type);
    void autox_relu(float* x_data, uint16_t* dims, uint8_t dim_size);
    void autox_relu_noreplace(float* x_data, float* y_data, uint16_t* dims, uint8_t dim_size);
    void autox_relu6(float* x_data, uint16_t* dims, uint8_t dim_size, float coef);

    void autox_reduce_mean(float* x_data,
        float* out_data, uint16_t* x_dims, uint16_t* y_dims);

    void autox_scale(float* data, uint16_t* dims, uint8_t dim_size, float bias, int8_t bias_before, float scale);
    void autox_sqrt(float* data, uint16_t* dims, uint8_t dim_size);
    void autox_sigmoid(float* data, uint16_t* dims, uint8_t dim_size);
    void autox_slice(float* x_data, float* o_data[], uint16_t* in_dims, uint16_t* out_dims, uint8_t in_dim_size, uint8_t out_dim_size);
    void autox_split(float* input, float** output, uint16_t* in_dim, uint16_t* output_ddim[], int32_t axis,
        uint16_t input_dims_size, uint16_t output_size, uint16_t output_ddim_size);

    void autox_transpose(const float* input_ptr, float* output_ptr, uint16_t* in_dim, uint16_t* out_dim, uint16_t* axis, int permute);
    void autox_transpose2(const float* input_ptr, float* output_ptr, uint16_t* in_dim, uint16_t* out_dim, uint16_t* axis, int permute);

    void autox_softmax(float* x, uint32_t height, uint32_t width);

    void autox_fusion_elementwise_add_activation(float* x_data,
        float* y_data,
        float* out_data, uint16_t* x_dims, uint16_t* y_dims, uint16_t* z_dims, int axis, uint16_t x_dims_size,
        uint16_t y_dims_size, uint16_t z_dims_size);
#ifdef  __cplusplus
}
#endif
