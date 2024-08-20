#include "../include/ggml.h"

// ggml_transpose

struct ggml_tensor* ggml_transpose(
    struct ggml_tensor* a) {

    struct ggml_tensor* result = ggml_view_tensor(a);

    result->ne[0] = a->ne[1];
    result->ne[1] = a->ne[0];

    result->nb[0] = a->nb[1];
    result->nb[1] = a->nb[0];

    return result;
}