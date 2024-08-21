#include "../include/ggml.h"

#if UINTPTR_MAX == 0xFFFFFFFF
#define GGML_MEM_ALIGN 4
#else
#define GGML_MEM_ALIGN 16
#endif

#define GGML_PAD(x, n) (((x) + (n) - 1) & ~((n) - 1))

enum ggml_object_type {
    GGML_OBJECT_TYPE_TENSOR,
    GGML_OBJECT_TYPE_GRAPH,
    GGML_OBJECT_TYPE_WORK_BUFFER
};

// ggml object
struct ggml_object {
    size_t offs;
    size_t size;

    struct ggml_object* next;

    enum ggml_object_type type;

    char padding[4];
};

static const size_t GGML_TENSOR_SIZE = sizeof(struct ggml_tensor);

static const size_t GGML_OBJECT_SIZE = sizeof(struct ggml_object);

size_t ggml_row_size(enum ggml_type type, int64_t ne) {
    return ggml_type_size(type) * ne / ggml_blck_size(type);
}

static struct ggml_object* ggml_new_object(enum ggml_object_type type, size_t size) {

    // align to GGML_MEM_ALIGN
    size_t size_needed = GGML_PAD(size, GGML_MEM_ALIGN);

    struct ggml_object* const obj_new = (struct ggml_object*)calloc(1, sizeof(struct ggml_object));

    *obj_new = (struct ggml_object){
        .offs = GGML_OBJECT_SIZE,
        .size = size_needed,
        .next = NULL,
        .type = type,
    };

    //printf("%s: inserted new object at %zu, size = %zu\n", __func__, cur_end, obj_new->size);

    return obj_new;
}

struct ggml_tensor* ggml_new_tensor_impl(
    enum   ggml_type      type,
    int                   n_dims,
    const int64_t* ne,
    struct ggml_tensor* view_src,
    size_t                view_offs) {

    size_t data_size = ggml_row_size(type, ne[0]);
    for (int i = 1; i < n_dims; i++) {
        data_size *= ne[i];
    }

    void* data = view_src != NULL ? view_src->data : NULL;
    if (data != NULL) {
        data = (char*)data + view_offs;
    }

    size_t obj_alloc_size = 0;

    if (view_src == NULL) {
        {
            // allocate tensor data in the context's memory pool
            obj_alloc_size = data_size;
        }
    }

    struct ggml_object* const obj_new = ggml_new_object(GGML_OBJECT_TYPE_TENSOR, GGML_TENSOR_SIZE + obj_alloc_size);

    // TODO: for recoverable errors, we would need to free the data allocated from the scratch buffer here

    struct ggml_tensor* const result = (struct ggml_tensor*)calloc(1, sizeof(struct ggml_tensor));

    * result = (struct ggml_tensor){
        /*.type         =*/ type,
        /*.ne           =*/ { 1, 1, 1, 1 },
        /*.nb           =*/ { 0, 0, 0, 0 },
        /*.data         =*/ obj_alloc_size > 0 ? (void*)(result + 1) : data,
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


