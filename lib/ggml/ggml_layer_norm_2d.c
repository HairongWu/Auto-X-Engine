#include "../include/ggml.h"

struct ggml_tensor* ggml_layer_norm_2d(
    struct ggml_tensor* layer,
    int                   n_channels,
    struct ggml_tensor* w,
    struct ggml_tensor* b,
    float                 eps) {
    // LayerNorm2d
    // normalize along channel dimmension
    // TODO: better implementation
    layer = 
        ggml_norm(ggml_cont(layer), eps);

    layer = ggml_add(
        ggml_mul(
            ggml_repeat(w, layer),
            layer),
        ggml_repeat(b, layer));

    return layer;
}
