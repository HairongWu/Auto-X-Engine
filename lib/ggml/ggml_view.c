#include "../include/ggml.h"

static struct ggml_tensor* ggml_view_impl(
    struct ggml_tensor* a,
    int                   n_dims,
    const int64_t* ne,
    size_t                offset) {

    struct ggml_tensor* result = ggml_new_tensor_impl(a->type, n_dims, ne, a, offset);

    return result;
}

struct ggml_tensor* ggml_view_tensor(
    struct ggml_tensor* src) {
    struct ggml_tensor* result = ggml_new_tensor_impl(src->type, GGML_MAX_DIMS, src->ne, src, 0);

    for (int i = 0; i < GGML_MAX_DIMS; i++) {
        result->nb[i] = src->nb[i];
    }

    return result;
}

// ggml_view_1d

struct ggml_tensor* ggml_view_1d(
    struct ggml_tensor* a,
    int64_t               ne0,
    size_t                offset) {

    struct ggml_tensor* result = ggml_view_impl(a, 1, &ne0, offset);

    return result;
}

// ggml_view_2d

struct ggml_tensor* ggml_view_2d(
    struct ggml_tensor* a,
    int64_t               ne0,
    int64_t               ne1,
    size_t                nb1,
    size_t                offset) {

    const int64_t ne[2] = { ne0, ne1 };

    struct ggml_tensor* result = ggml_view_impl(a, 2, ne, offset);

    result->nb[1] = nb1;
    result->nb[2] = result->nb[1] * ne1;
    result->nb[3] = result->nb[2];

    return result;
}

// ggml_view_3d

struct ggml_tensor* ggml_view_3d(
    struct ggml_tensor* a,
    int64_t               ne0,
    int64_t               ne1,
    int64_t               ne2,
    size_t                nb1,
    size_t                nb2,
    size_t                offset) {

    const int64_t ne[3] = { ne0, ne1, ne2 };

    struct ggml_tensor* result = ggml_view_impl(a, 3, ne, offset);

    result->nb[1] = nb1;
    result->nb[2] = nb2;
    result->nb[3] = result->nb[2] * ne2;

    return result;
}

// ggml_view_4d

struct ggml_tensor* ggml_view_4d(
    struct ggml_tensor* a,
    int64_t               ne0,
    int64_t               ne1,
    int64_t               ne2,
    int64_t               ne3,
    size_t                nb1,
    size_t                nb2,
    size_t                nb3,
    size_t                offset) {

    const int64_t ne[4] = { ne0, ne1, ne2, ne3 };

    struct ggml_tensor* result = ggml_view_impl(a, 4, ne, offset);

    result->nb[1] = nb1;
    result->nb[2] = nb2;
    result->nb[3] = nb3;

    return result;
}
