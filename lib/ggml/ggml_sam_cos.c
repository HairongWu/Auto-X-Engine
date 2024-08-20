#include "../include/ggml.h"

static void ggml_sam_cos(struct ggml_tensor* dst, const struct ggml_tensor* src) {

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
