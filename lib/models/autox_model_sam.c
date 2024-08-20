#include "../include/autox_models.h"


#define ggml_QNT_VERSION_FACTOR 1000 // do not change this

int32_t n_enc_state = 768;
int32_t n_enc_layer = 12;
int32_t n_enc_head = 12;
int32_t n_enc_out_chans = 256;
int32_t n_pt_embd = 4;
int32_t n_dec_heads = 8;
int32_t ftype = 1;
float   mask_threshold = 0.f;
float   iou_threshold = 0.88f;
float   stability_score_threshold = 0.95f;
float   stability_score_offset = 1.0f;
float   eps = 1e-6f;
float   eps_decoder_transformer = 1e-5f;

static int32_t n_enc_head_dim() { return n_enc_state / n_enc_head; }
static int32_t n_img_size() { return 1024; }
static int32_t n_window_size() { return 14; }
static int32_t n_patch_size()  { return 16; }
static int32_t n_img_embd() { return n_img_size() / n_patch_size(); }

static uint32_t index = 0;


//int32_t* global_attn_indices() {
//    static int32_t ret[4];
//    switch (n_enc_state) {
//    case  768:
//        return { 2,  5,  8, 11 };
//    case 1024:
//        return { 5, 11, 17, 23 };
//    case 1280:
//        return { 7, 15, 23, 31 };
//    default:
//    {
//        fprintf(stderr, "%s: unsupported n_enc_state = %d\n", __func__, n_enc_state);
//    } break;
//    };
//
//    return ret;
//}
//
//static bool is_global_attn(int32_t layer) {
//    int32_t* indices = global_attn_indices();
//
//    for (int32_t idx = 0; idx < 4; idx++) {
//        if (layer == indices[idx]) {
//            return true;
//        }
//    }
//
//    return false;
//}

autox_err_t autox_sam(float* img1, void* weights, int nx, int ny, float x, float y)
{
    autox_err_t err = AUTOX_OK;

    mask_threshold = 0.f;
    iou_threshold = 0.88f;
    stability_score_threshold = 0.95f;
    stability_score_offset = 1.0f;
    eps = 1e-6f;
    eps_decoder_transformer = 1e-5f;

    // verify magic
    {
        uint32_t magic = (uint32_t*)((int8_t*)weights);
        index += sizeof(uint32_t);

        if (magic != 0x67676d6c) {
            return AUTOX_FAIL;
        }
    }

    // load hparams
    {
        n_enc_state = *(int32_t*)((int8_t*)weights + index);
        index += sizeof(n_enc_state);
        n_enc_layer = *(int32_t*)((int8_t*)weights + index);
        index += sizeof(n_enc_layer);
        n_enc_head = *(int32_t*)((int8_t*)weights + index);
        index += sizeof(n_enc_head);
        n_enc_out_chans = *(int32_t*)((int8_t*)weights + index);
        index += sizeof(n_enc_out_chans);
        n_pt_embd = *(int32_t*)((int8_t*)weights + index);
        index += sizeof(n_pt_embd);
        ftype = *(int32_t*)((int8_t*)weights + index);
        index += sizeof(ftype);

        const int32_t qntvr = ftype / ggml_QNT_VERSION_FACTOR;
        ftype %= ggml_QNT_VERSION_FACTOR;

    }
    struct ggml_tensor* embd_img = ggml_new_tensor_3d(GGML_TYPE_F32,
        n_img_embd(), n_img_embd(), n_enc_out_chans);

    struct ggml_tensor* low_res_masks = ggml_new_tensor_3d(GGML_TYPE_F32,
        n_enc_out_chans, n_enc_out_chans, 3);

    struct ggml_tensor* iou_predictions = ggml_new_tensor_1d(GGML_TYPE_F32, 3);

    //sam_encode_image(img1, embd_img, nx, ny, weights);

    //float* inp = (float*)calloc(2 * 2, sizeof(float));
    //float* pe;
    //float* embd_prompt_sparse;
    //float* embd_prompt_dense;
    //sam_encode_prompt(inp, pe, embd_prompt_sparse, embd_prompt_dense);


    //float *xy_embed_stacked = (float *)calloc(2*n_img_embd()*n_img_embd(), sizeof(float));
    //float* pe_img_dense = ggml_cont(ggml_permute(cur, 2, 0, 1, 3));
    //sam_fill_dense_pe(xy_embed_stacked, pe_img_dense);

    //if (!sam_decode_mask(model, enc_res, pe_img_dense, gf, state)) {
    //    fprintf(stderr, "%s: failed to decode mask\n", __func__);
    //    return AUTOX_FAIL;
    //}

    //// from sam_encode_prompt
    //{
    //    // transform points
    //    // ref: https://github.com/facebookresearch/segment-anything/blob/main/segment_anything/automatic_mask_generator.py#L276
    //    {
    //        const int nmax = max(nx, ny);

    //        const float scale = n_img_size() / (float)nmax;

    //        const int nx_new = (int)(nx * scale + 0.5f);
    //        const int ny_new = (int)(ny * scale + 0.5f);

    //        x = x * ((float)(nx_new) / nx) + 0.5f;
    //        y = y * ((float)(ny_new) / ny) + 0.5f;
    //    }

    //    struct ggml_tensor* inp = ggml_graph_get_tensor(gf, "prompt_input");
    //    // set the input by converting the [0, 1] coordinates to [-1, 1]
    //    float* data = (float*)inp->data;

    //    data[0] = 2.0f * (x / model.hparams.n_img_size()) - 1.0f;
    //    data[1] = 2.0f * (y / model.hparams.n_img_size()) - 1.0f;

    //    // padding
    //    // ref: https://github.com/facebookresearch/segment-anything/blob/main/segment_anything/modeling/prompt_encoder.py#L81-L85
    //    data[2] = 2.0f * (0.0f) - 1.0f;
    //    data[3] = 2.0f * (0.0f) - 1.0f;
    //}

    //// from sam_fill_dense_pe
    //{
    //    const float n_img_embd_inv = 1.0f / n_img_embd();
    //    float* data = (float*)ggml_get_data(xy_embed_stacked);
    //    for (int i = 0; i < n_img_embd; ++i) {
    //        const int row = 2 * i * n_img_embd();
    //        const float y_val = 2 * (i + 0.5f) * n_img_embd_inv - 1;
    //        for (int j = 0; j < n_img_embd; ++j) {
    //            const float x_val = 2 * (j + 0.5f) * n_img_embd_inv - 1;
    //            data[row + 2 * j + 0] = x_val;
    //            data[row + 2 * j + 1] = y_val;
    //        }
    //    }
    //}


	return err;
}