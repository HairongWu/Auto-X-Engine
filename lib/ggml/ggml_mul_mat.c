#include "../include/ggml.h"

// ggml_mul_mat

struct ggml_tensor* ggml_mul_mat(
    struct ggml_tensor* a,
    struct ggml_tensor* b) {

    const int64_t ne[4] = { a->ne[1], b->ne[1], b->ne[2], b->ne[3] };
    struct ggml_tensor* result = ggml_new_tensor(GGML_TYPE_F32, 4, ne);


    return result;
}
