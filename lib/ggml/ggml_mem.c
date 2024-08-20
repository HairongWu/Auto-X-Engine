#include "../include/ggml.h"

size_t ggml_row_size(enum ggml_type type, int64_t ne) {
    return ggml_type_size(type) * ne / ggml_blck_size(type);
}

static struct ggml_tensor* ggml_new_tensor_impl(
    enum   ggml_type      type,
    int                   n_dims,
    const int64_t* ne,
    struct ggml_tensor* view_src,
    size_t                view_offs)
{

    size_t data_size = ggml_row_size(type, ne[0]);
    for (int i = 1; i < n_dims; i++) {
        data_size *= ne[i];
    }

    struct ggml_tensor* const result = malloc(sizeof(struct ggml_tensor));
    void* data = malloc(data_size);

    *result = (struct ggml_tensor){
        /*.type         =*/ type,
        /*.ne           =*/ { 1, 1, 1, 1 },
        /*.nb           =*/ { 0, 0, 0, 0 },
        /*.data         =*/ data
    };

    // TODO: this should not be needed as long as we don't rely on aligned SIMD loads
    //GGML_ASSERT_ALIGNED(result->data);

    for (int i = 0; i < n_dims; i++) {
        result->ne[i] = ne[i];
    }

    result->nb[0] = ggml_type_size(type);
    result->nb[1] = result->nb[0] * (result->ne[0] / ggml_blck_size(type));
    for (int i = 2; i < GGML_MAX_DIMS; i++) {
        result->nb[i] = result->nb[i - 1] * result->ne[i - 1];
    }

    return result;
}

struct ggml_tensor* ggml_new_tensor(
    enum   ggml_type      type,
    int                   n_dims,
    const int64_t* ne) {
    return ggml_new_tensor_impl(type, n_dims, ne, NULL, 0);
}

struct ggml_tensor* ggml_new_tensor_1d(
    enum   ggml_type      type,
    int64_t ne0) {
    return ggml_new_tensor(type, 1, &ne0);
}

struct ggml_tensor* ggml_new_tensor_2d(
    enum   ggml_type      type,
    int64_t ne0,
    int64_t ne1) {
    const int64_t ne[2] = { ne0, ne1 };
    return ggml_new_tensor(type, 2, ne);
}

struct ggml_tensor* ggml_new_tensor_3d(
    enum   ggml_type      type,
    int64_t ne0,
    int64_t ne1,
    int64_t ne2) {
    const int64_t ne[3] = { ne0, ne1, ne2 };
    return ggml_new_tensor(type, 3, ne);
}

struct ggml_tensor* ggml_new_tensor_4d(
    enum   ggml_type type,
    int64_t ne0,
    int64_t ne1,
    int64_t ne2,
    int64_t ne3) {
    const int64_t ne[4] = { ne0, ne1, ne2, ne3 };
    return ggml_new_tensor(type, 4, ne);
}
struct ggml_tensor* ggml_dup_tensor(const struct ggml_tensor* src) {
    return ggml_new_tensor(src->type, GGML_MAX_DIMS, src->ne);
}

struct ggml_tensor* ggml_view_tensor(
    struct ggml_tensor* src) {
    struct ggml_tensor* result = ggml_new_tensor_impl(src->type, GGML_MAX_DIMS, src->ne, src, 0);

    for (int i = 0; i < GGML_MAX_DIMS; i++) {
        result->nb[i] = src->nb[i];
    }

    return result;
}

