#include "../include/ggml.h"

// ggml_permute

struct ggml_tensor* ggml_permute(
    struct ggml_tensor* a,
    int                   axis0,
    int                   axis1,
    int                   axis2,
    int                   axis3) {

    struct ggml_tensor* result = ggml_view_tensor(a);

    int ne[GGML_MAX_DIMS];
    int nb[GGML_MAX_DIMS];

    ne[axis0] = a->ne[0];
    ne[axis1] = a->ne[1];
    ne[axis2] = a->ne[2];
    ne[axis3] = a->ne[3];

    nb[axis0] = a->nb[0];
    nb[axis1] = a->nb[1];
    nb[axis2] = a->nb[2];
    nb[axis3] = a->nb[3];

    result->ne[0] = ne[0];
    result->ne[1] = ne[1];
    result->ne[2] = ne[2];
    result->ne[3] = ne[3];

    result->nb[0] = nb[0];
    result->nb[1] = nb[1];
    result->nb[2] = nb[2];
    result->nb[3] = nb[3];

    int32_t params[] = { axis0, axis1, axis2, axis3 };
    // ggml_set_op_params(result, params, sizeof(params));

    return result;
}
