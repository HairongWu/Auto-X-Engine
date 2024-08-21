#include "../include/ggml.h"

// ggml_compute_forward_map_custom1

static void ggml_compute_forward_map_custom1(
    const struct ggml_tensor* src,
    struct ggml_tensor* dst) {

    const float* src_data = (float*)(src->data);
    float* dst_data = (float*)(dst->data);

    const int ne = (int)ggml_nelements(dst);
    const int dr = ne;
    const int ie0 = 0;
    const int ie1 = min(ie0 + dr, ne);

    for (int i = ie0; i < ie1; ++i) {
        dst_data[i] = cosf(src_data[i]);
    }
}

static struct ggml_tensor* ggml_map_custom1_impl(
    struct ggml_tensor* a) {

    struct ggml_tensor* result = ggml_dup_tensor(a);

    ggml_compute_forward_map_custom1(a, result);

    return result;
}

struct ggml_tensor* ggml_sam_cos(
    struct ggml_tensor* a) {
    return ggml_map_custom1_impl(a);
}