#include "../include/ggml.h"

// ggml_compute_forward_cpy

static void ggml_compute_forward_cpy(
    struct ggml_tensor* a,
    struct ggml_tensor* dst) {
    ggml_compute_forward_dup(a, dst);
}

// ggml_cpy

static struct ggml_tensor* ggml_cpy_impl(
    struct ggml_tensor* a,
    struct ggml_tensor* b) {

    // make a view of the destination
    struct ggml_tensor* result = ggml_view_tensor(b);

    ggml_compute_forward_cpy(a, result);

    return result;
}

struct ggml_tensor* ggml_cpy(
    struct ggml_tensor* a,
    struct ggml_tensor* b) {
    return ggml_cpy_impl(a, b);
}