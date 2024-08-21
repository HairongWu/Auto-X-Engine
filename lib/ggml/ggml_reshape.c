#include "../include/ggml.h"

struct ggml_tensor* ggml_reshape_3d(
    struct ggml_tensor* a,
    int64_t               ne0,
    int64_t               ne1,
    int64_t               ne2) {

    const int64_t ne[3] = { ne0, ne1, ne2 };
    struct ggml_tensor* result = ggml_new_tensor_impl(a->type, 3, ne, a, 0);

    return result;
}

struct ggml_tensor* ggml_reshape_4d(
    struct ggml_tensor* a,
    int64_t               ne0,
    int64_t               ne1,
    int64_t               ne2,
    int64_t               ne3) {

    const int64_t ne[4] = { ne0, ne1, ne2, ne3 };
    struct ggml_tensor* result = ggml_new_tensor_impl(a->type, 4, ne, a, 0);

    return result;
}