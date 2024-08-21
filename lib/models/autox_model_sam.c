#include "../include/autox_models.h"

#define _USE_MATH_DEFINES
#include <math.h>


#define GGML_QNT_VERSION_FACTOR 1000 // do not change this

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

struct prompt_encoder_result {
    struct ggml_tensor* embd_prompt_sparse;
    struct ggml_tensor* embd_prompt_dense;
};

struct  sam_layer_dec_transformer_attn {
    // q_proj
    struct ggml_tensor* q_w;
    struct ggml_tensor* q_b;

    // k_proj
    struct ggml_tensor* k_w;
    struct ggml_tensor* k_b;

    // v_proj
    struct ggml_tensor* v_w;
    struct ggml_tensor* v_b;

    // out_proj
    struct ggml_tensor* out_w;
    struct ggml_tensor* out_b;
};

struct sam_layer_dec_transformer {
    struct sam_layer_dec_transformer_attn self_attn;

    // norm1
    struct ggml_tensor* norm1_w;
    struct ggml_tensor* norm1_b;

    struct sam_layer_dec_transformer_attn cross_attn_token_to_img;

    // norm2
    struct ggml_tensor* norm2_w;
    struct ggml_tensor* norm2_b;

    // mlp.lin1
    struct ggml_tensor* mlp_lin1_w;
    struct ggml_tensor* mlp_lin1_b;

    // mlp.lin2
    struct ggml_tensor* mlp_lin2_w;
    struct ggml_tensor* mlp_lin2_b;

    // norm3
    struct ggml_tensor* norm3_w;
    struct ggml_tensor* norm3_b;

    // norm4
    struct ggml_tensor* norm4_w;
    struct ggml_tensor* norm4_b;

    struct sam_layer_dec_transformer_attn cross_attn_img_to_token;
};
struct sam_layer_dec_output_hypernet_mlps {
    // mlps_*.layers.0
    struct ggml_tensor* w_0;
    struct ggml_tensor* b_0;

    // mlps_*.layers.1
    struct ggml_tensor* w_1;
    struct ggml_tensor* b_1;

    // mlps_*.layers.2
    struct ggml_tensor* w_2;
    struct ggml_tensor* b_2;
};

static bool is_global_attn(int32_t layer) {
    if (n_enc_state == 768) {
        int32_t indices[4] = { 2,  5,  8, 11 };
        for (int32_t idx = 0; idx < 4; idx++) {
            if (layer == indices[idx]) {
                return true;
            }
        }
    }
    else if (n_enc_state == 1024) {
        int32_t indices[4] = { 5, 11, 17, 23 };
        for (int32_t idx = 0; idx < 4; idx++) {
            if (layer == indices[idx]) {
                return true;
            }
        }
    }
    else if (n_enc_state == 1280) {
        int32_t indices[4] = { 7, 15, 23, 31 };
        for (int32_t idx = 0; idx < 4; idx++) {
            if (layer == indices[idx]) {
                return true;
            }
        }
    }

    return false;
}

void sam_encode_image(
    const float* img, struct ggml_tensor* embd_img, const int nx, const int ny, void* weights) {

    struct ggml_tensor* inp = ggml_new_tensor_4d(GGML_TYPE_F32, n_img_size(), n_img_size(), 3, 1);
    {
        const int n = nx * ny;

        for (int k = 0; k < 3; k++) {
            for (int y = 0; y < ny; y++) {
                for (int x = 0; x < nx; x++) {
                    ((float*)(inp->data))[k * n + y * nx + x] = img[3 * (y * nx + x) + k];
                }
            }
        }
    }

    struct ggml_tensor* pe = ggml_new_tensor_4d(GGML_TYPE_F32, n_enc_state, n_img_embd(), n_img_embd(), 1);

    struct ggml_tensor* proj_w = ggml_new_tensor_4d(GGML_TYPE_F16, n_patch_size(), n_patch_size(), 3, n_enc_state);
    struct ggml_tensor* proj_b = ggml_new_tensor_3d(GGML_TYPE_F32, 1, 1, n_enc_state);

    /*int32_t n_dims = *(int32_t*)((int8_t*)weights + index);
    index += sizeof(n_dims);
    int32_t length = *(int32_t*)((int8_t*)weights + index);
    index += sizeof(length);
    int32_t ftype = *(int32_t*)((int8_t*)weights + index);
    index += sizeof(ftype);

    int64_t nelements = 1;
    int64_t ne[4] = { 1, 1, 1, 1 };
    for (int i = 0; i < n_dims; ++i) {
        int32_t ne_cur = *(int32_t*)((int8_t*)weights + index);
        index += sizeof(ne_cur);
        ne[i] = ne_cur;
        nelements *= ne[i];
    }*/
    // ref: https://github.com/facebookresearch/segment-anything/blob/main/segment_anything/modeling/image_encoder.py#L392
    struct ggml_tensor* cur = ggml_conv_2d_sk_p0(proj_w, inp);
    cur = ggml_add_inplace(
        cur,
        ggml_repeat(proj_b, cur));

    // ref: https://github.com/facebookresearch/segment-anything/blob/main/segment_anything/modeling/image_encoder.py#L394
    // keep in F32
    cur = ggml_cont(
        ggml_permute(cur, 1, 2, 0, 3));

    // convert to F16
    //cur = ggml_cpy(
    //        ggml_permute(cur, 1, 2, 0, 3),
    //        ggml_new_tensor_3d(GGML_TYPE_F16, n_enc_state, n_img_embd, n_img_embd));

    // ref: https://github.com/facebookresearch/segment-anything/blob/main/segment_anything/modeling/image_encoder.py#L108-L109
    cur = ggml_add_inplace(cur, pe);

    struct ggml_tensor* inpL = cur;

    for (int il = 0; il < n_enc_layer; ++il) {

        // norm
        // ref: https://github.com/facebookresearch/segment-anything/blob/main/segment_anything/modeling/image_encoder.py#L168
        {
            cur = ggml_norm(inpL, eps);

            struct ggml_tensor* norm1_w = ggml_new_tensor_1d(GGML_TYPE_F32, n_enc_state);
            struct ggml_tensor* norm1_b = ggml_new_tensor_1d(GGML_TYPE_F32, n_enc_state);
            // cur = ln_0_w*cur + ln_0_b
            cur = ggml_mul(cur, norm1_w);
            cur = ggml_add_inplace(cur, norm1_b);
        }

        const int64_t w0 = cur->ne[1];
        const int64_t h0 = cur->ne[2];

        if (is_global_attn(il) == false) {
            // local attention layer - apply window partition
            // ref: https://github.com/facebookresearch/segment-anything/blob/main/segment_anything/modeling/image_encoder.py#L169-L172
            cur = ggml_win_part(cur, n_window_size());
        }

        const int64_t W = cur->ne[1];
        const int64_t H = cur->ne[2];

        // self-attention
        {
            struct ggml_tensor* qkv_w = ggml_new_tensor_2d(GGML_TYPE_F16, n_enc_state, 3 * n_enc_state);
            struct ggml_tensor* qkv_b = ggml_new_tensor_1d(GGML_TYPE_F32, 3 * n_enc_state);
            cur = ggml_mul_mat(qkv_w, cur);
            cur = ggml_add_inplace(cur, qkv_b);

            // split qkv into separate tensors
            // ref: https://github.com/facebookresearch/segment-anything/blob/main/segment_anything/modeling/image_encoder.py#L225-L229
            const int B = cur->ne[3];

            cur = ggml_reshape_4d(cur, n_enc_state, 3, W * H, B);
            cur = ggml_cont(ggml_permute(cur, 0, 3, 1, 2));

            struct ggml_tensor* Q;
            struct ggml_tensor* K;
            struct ggml_tensor* V;

            Q = ggml_view_3d(cur, n_enc_state, W * H, B, cur->nb[1], cur->nb[2], 0 * cur->nb[3]);
            Q = ggml_reshape_4d(Q, n_enc_head_dim, n_enc_head, W * H, B);
            Q = ggml_cont(ggml_permute(Q, 0, 2, 1, 3));
            Q = ggml_reshape_3d(Q, n_enc_head_dim, W * H, B * n_enc_head);

            K = ggml_view_3d(cur, n_enc_state, W * H, B, cur->nb[1], cur->nb[2], 1 * cur->nb[3]);
            K = ggml_reshape_4d(K, n_enc_head_dim, n_enc_head, W * H, B);
            K = ggml_cont(ggml_permute(K, 0, 2, 1, 3));
            K = ggml_reshape_3d(K, n_enc_head_dim, W * H, B * n_enc_head);

            V = ggml_view_3d(cur, n_enc_state, W * H, B, cur->nb[1], cur->nb[2], 2 * cur->nb[3]);
            V = ggml_reshape_4d(V, n_enc_head_dim, n_enc_head, W * H, B);
            V = ggml_cont(ggml_permute(V, 1, 2, 0, 3)); // transposed
            V = ggml_reshape_3d(V, W * H, n_enc_head_dim, B * n_enc_head);

            struct ggml_tensor* KQ = ggml_mul_mat(K, Q);

            struct ggml_tensor* KQ_scaled =
                ggml_scale_inplace(
                    KQ,
                    1.0f / sqrtf(n_enc_head_dim()));

            struct ggml_tensor* rel_pos_w;
            struct ggml_tensor* rel_pos_h;
            if (is_global_attn(il)) {
                rel_pos_w = ggml_new_tensor_2d(GGML_TYPE_F16, n_enc_head_dim(), 2 * n_img_embd() - 1);
                rel_pos_h = ggml_new_tensor_2d(GGML_TYPE_F16, n_enc_head_dim(), 2 * n_img_embd() - 1);
            }
            else {
                rel_pos_w = ggml_new_tensor_2d(GGML_TYPE_F16, n_enc_head_dim(), 2 * n_window_size() - 1);
                rel_pos_h = ggml_new_tensor_2d(GGML_TYPE_F16, n_enc_head_dim(), 2 * n_window_size() - 1);
            }
            struct ggml_tensor* rw = ggml_get_rel_pos(rel_pos_w, W, W);
            struct ggml_tensor* rh = ggml_get_rel_pos(rel_pos_h, H, H);

            struct ggml_tensor* q_r = ggml_reshape_4d(Q, n_enc_head_dim, W, H, B * n_enc_head);

            struct ggml_tensor* rel_w = ggml_cont(ggml_permute(
                ggml_mul_mat(
                    rw,
                    ggml_cont(ggml_permute(q_r, 0, 2, 1, 3))),
                0, 2, 1, 3));
            struct ggml_tensor* rel_h = ggml_mul_mat(rh, q_r);

            struct ggml_tensor* attn = ggml_add_rel_pos_inplace(KQ_scaled, rel_w, rel_h);

            struct ggml_tensor* KQ_soft_max = ggml_soft_max_inplace(attn);

            struct ggml_tensor* KQV = ggml_mul_mat(V, KQ_soft_max);

            cur =
                ggml_reshape_4d(
                    ggml_cont(
                        ggml_permute(
                            ggml_reshape_4d(KQV, n_enc_head_dim, W * H, n_enc_head, B),
                            0, 2, 1, 3)),
                    n_enc_state, W, H, B);

            struct ggml_tensor* proj_w = ggml_new_tensor_2d(GGML_TYPE_F16, n_enc_state, n_enc_state);
            struct ggml_tensor* proj_b = ggml_new_tensor_1d(GGML_TYPE_F32, n_enc_state);
            cur = ggml_mul_mat(proj_w, cur);
            cur = ggml_add_inplace(cur, proj_b);
        }

        if (is_global_attn(il) == false) {
            // local attention layer - reverse window partition
            cur = ggml_win_unpart(cur, w0, h0, n_window_size);
        }

        cur = ggml_add_inplace(cur, inpL);

        struct ggml_tensor* inpFF = cur;

        // feed-forward network
        {
            // norm
            {
                cur = ggml_norm(inpFF, eps);

                struct ggml_tensor* norm2_w = ggml_new_tensor_1d(GGML_TYPE_F32, n_enc_state);
                struct ggml_tensor* norm2_b = ggml_new_tensor_1d(GGML_TYPE_F32, n_enc_state);
                // cur = mlp_ln_w*cur + mlp_ln_b
                cur = ggml_mul(cur, norm2_w);
                cur = ggml_add_inplace(cur, norm2_b);
            }

            struct ggml_tensor* mlp_lin1_w = ggml_new_tensor_2d(GGML_TYPE_F16, n_enc_state, 4 * n_enc_state);
            struct ggml_tensor* mlp_lin1_b = ggml_new_tensor_1d(GGML_TYPE_F32, 4 * n_enc_state);
            // fully connected
            cur = ggml_mul_mat(mlp_lin1_w, cur);
            cur = ggml_add_inplace(cur, mlp_lin1_b);

            // GELU activation
            cur = ggml_gelu(cur);

            struct ggml_tensor* mlp_lin2_w = ggml_new_tensor_2d(GGML_TYPE_F16, 4 * n_enc_state, n_enc_state);
            struct ggml_tensor* mlp_lin2_b = ggml_new_tensor_1d(GGML_TYPE_F32, n_enc_state);
            // projection
            cur = ggml_mul_mat(mlp_lin2_w, cur);
            cur = ggml_add_inplace(cur, mlp_lin2_b);
        }

        inpL = ggml_add(cur, inpFF);
    }

    cur = ggml_cont(ggml_permute(inpL, 2, 0, 1, 3));

    struct ggml_tensor* neck_conv_0 = ggml_new_tensor_4d(GGML_TYPE_F16, 1, 1, n_enc_state, n_enc_out_chans);
    cur = ggml_conv_2d_sk_p0(neck_conv_0, cur);

    struct ggml_tensor* neck_norm_0_w = ggml_new_tensor_1d(GGML_TYPE_F32, n_enc_out_chans);
    struct ggml_tensor* neck_norm_0_b = ggml_new_tensor_1d(GGML_TYPE_F32, n_enc_out_chans);
    cur = ggml_layer_norm_2d(cur, n_enc_out_chans, neck_norm_0_w, neck_norm_0_b, eps);

    struct ggml_tensor* neck_conv_1 = ggml_new_tensor_4d(GGML_TYPE_F16, 3, 3, n_enc_out_chans, n_enc_out_chans);
    cur = ggml_conv_2d_s1_ph(neck_conv_1, cur);

    struct ggml_tensor* neck_norm_1_w = ggml_new_tensor_1d(GGML_TYPE_F32, n_enc_out_chans);
    struct ggml_tensor* neck_norm_1_b = ggml_new_tensor_1d(GGML_TYPE_F32, n_enc_out_chans);
    cur = ggml_layer_norm_2d(cur, n_enc_out_chans, neck_norm_1_w, neck_norm_1_b, eps);

    cur = ggml_cpy(cur, embd_img);
}

// encode a prompt
//
// - points
// - boxes
// - masks
//
// TODO: currently just encode a single point for simplicity
//
struct prompt_encoder_result sam_encode_prompt(struct ggml_tensor* pe, struct ggml_tensor* inp) {

    struct ggml_tensor* cur = ggml_mul_mat(ggml_cont(ggml_transpose(pe)), inp);

    cur = ggml_scale(cur, (float)(2.0 * M_PI));

    // concat
    // ref: https://github.com/facebookresearch/segment-anything/blob/main/segment_anything/modeling/prompt_encoder.py#L192
    {
        struct ggml_tensor* t_sin = ggml_sam_sin(cur);
        struct ggml_tensor* t_cos = ggml_sam_cos(cur);

        cur = ggml_new_tensor_2d(GGML_TYPE_F32, t_sin->ne[0] + t_cos->ne[0], cur->ne[1]);

        ggml_cpy(t_sin, ggml_view_2d(cur, t_sin->ne[0], t_sin->ne[1], cur->nb[1], 0));
        ggml_cpy(t_cos, ggml_view_2d(cur, t_sin->ne[0], t_sin->ne[1], cur->nb[1], t_sin->nb[1]));

        // overwrite label == -1 with not_a_point_embed.weight
        // ref: https://github.com/facebookresearch/segment-anything/blob/main/segment_anything/modeling/prompt_encoder.py#L86
        // TODO: extend for multiple points
        struct ggml_tensor* not_a_pt_embd_w = ggml_new_tensor_1d(GGML_TYPE_F32, n_enc_out_chans);
        ggml_cpy(not_a_pt_embd_w, ggml_view_2d(cur, cur->ne[0], 1, cur->nb[1], cur->nb[1]));
    }

    // add point_embeddings[1] to label == 1
    // ref: https://github.com/facebookresearch/segment-anything/blob/main/segment_anything/modeling/prompt_encoder.py#L90
    struct ggml_tensor* v = ggml_view_2d(cur, cur->ne[0], 1, cur->nb[1], 0);
    struct ggml_tensor** pt_embd = calloc(n_pt_embd, sizeof(struct ggml_tensor*));
    for (int i = 0; i < n_pt_embd; i++) {
        pt_embd[i] = ggml_new_tensor_1d(GGML_TYPE_F32, n_enc_out_chans);
    }
    ggml_cpy(ggml_add_inplace(v, pt_embd[1]), v);

    struct ggml_tensor* embd_prompt_sparse = cur;

    struct ggml_tensor* no_mask_embd_w = ggml_new_tensor_1d(GGML_TYPE_F32, n_enc_out_chans);
    struct ggml_tensor* embd_prompt_dense = ggml_repeat(
        ggml_cont(
            ggml_view_3d(no_mask_embd_w,
                1, 1, no_mask_embd_w->ne[0], no_mask_embd_w->nb[0], no_mask_embd_w->nb[0], 0)),
        ggml_new_tensor_3d(GGML_TYPE_F32, n_img_embd(), n_img_embd(), n_enc_out_chans));
    
    struct prompt_encoder_result res;
    res.embd_prompt_sparse = embd_prompt_sparse;
    res.embd_prompt_dense = embd_prompt_dense;
    return res;
}

struct ggml_tensor* sam_fill_dense_pe(struct ggml_tensor* pe, struct ggml_tensor* xy_embed_stacked) {

    struct ggml_tensor* cur = ggml_mul_mat(ggml_cont(ggml_transpose(pe)), xy_embed_stacked);

    cur = ggml_scale(cur, (float)(2.0 * M_PI));

    // concat
    // ref: https://github.com/facebookresearch/segment-anything/blob/main/segment_anything/modeling/prompt_encoder.py#L192
    {
        struct ggml_tensor* t_sin = ggml_sam_sin(cur);
        struct ggml_tensor* t_cos = ggml_sam_cos(cur);

        cur = ggml_new_tensor_3d(GGML_TYPE_F32, t_sin->ne[0] + t_cos->ne[0], cur->ne[1], cur->ne[2]);

        ggml_cpy(t_sin, ggml_view_3d(cur, t_sin->ne[0], t_sin->ne[1], t_sin->ne[2], cur->nb[1], cur->nb[2], 0));
        ggml_cpy(t_cos, ggml_view_3d(cur, t_sin->ne[0], t_sin->ne[1], t_sin->ne[2], cur->nb[1], cur->nb[2], t_sin->nb[1]));
    }

    struct ggml_tensor* pe_img_dense = ggml_cont(ggml_permute(cur, 2, 0, 1, 3));

    return pe_img_dense;
}
struct ggml_tensor* sam_decode_mask_transformer_attn(
    const struct sam_layer_dec_transformer_attn attn,
    struct ggml_tensor* queries,
    struct ggml_tensor* keys,
    struct ggml_tensor* values) {

    const int n_head = n_dec_heads;

    struct ggml_tensor* Qcur;
    struct ggml_tensor* Kcur;
    struct ggml_tensor* Vcur;

    Qcur = ggml_mul_mat(attn.q_w, queries);
    Qcur = ggml_add_inplace(Qcur, attn.q_b);

    Kcur = ggml_mul_mat(attn.k_w, keys);
    Kcur = ggml_add_inplace(Kcur, attn.k_b);

    Vcur = ggml_mul_mat(attn.v_w, values);
    Vcur = ggml_add_inplace(Vcur, attn.v_b);

    struct ggml_tensor* Q;
    struct ggml_tensor* K;
    struct ggml_tensor* V;

    Q = ggml_reshape_4d(Qcur, Qcur->ne[0] / n_head, n_head, Qcur->ne[1], Qcur->ne[2]);
    Q = ggml_cont(ggml_permute(Q, 0, 2, 1, 3));

    K = ggml_reshape_4d(Kcur, Kcur->ne[0] / n_head, n_head, Kcur->ne[1], Kcur->ne[2]);
    K = ggml_cont(ggml_permute(K, 0, 2, 1, 3));

    V = ggml_reshape_4d(Vcur, Vcur->ne[0] / n_head, n_head, Vcur->ne[1], Vcur->ne[2]);
    V = ggml_cont(ggml_permute(V, 0, 2, 1, 3));

    // Q * K
    struct ggml_tensor* KQ = ggml_mul_mat(K, Q);

    struct ggml_tensor* KQ_scaled = ggml_scale_inplace(KQ, 1.0f / sqrt((float)(Q->ne[0])));

    struct ggml_tensor* KQ_soft_max = ggml_soft_max_inplace(KQ_scaled);

    struct ggml_tensor* KQV = ggml_mul_mat(KQ_soft_max, ggml_cont(ggml_transpose(V)));

    struct ggml_tensor* KQV_merged = ggml_cont(ggml_transpose(KQV));
    KQV_merged = ggml_cont(ggml_permute(KQV_merged, 0, 2, 1, 3));
    KQV_merged = ggml_reshape_3d(KQV_merged, KQV_merged->ne[0] * KQV_merged->ne[1], KQV_merged->ne[2], KQV_merged->ne[3]);
    KQV_merged = ggml_mul_mat(attn.out_w, KQV_merged);
    KQV_merged = ggml_add_inplace(KQV_merged, attn.out_b);

    return KQV_merged;
}
struct ggml_tensor* sam_decode_mask_mlp_relu_3(
    struct ggml_tensor* in,
    struct ggml_tensor* w_0,
    struct ggml_tensor* b_0,
    struct ggml_tensor* w_1,
    struct ggml_tensor* b_1,
    struct ggml_tensor* w_2,
    struct ggml_tensor* b_2) {

    struct ggml_tensor* cur;
    cur = ggml_mul_mat(w_0, in);
    cur = ggml_add_inplace(cur, b_0);

    cur = ggml_relu_inplace(cur);

    cur = ggml_mul_mat(w_1, cur);
    cur = ggml_add_inplace(cur, b_1);

    cur = ggml_relu_inplace(cur);

    cur = ggml_mul_mat(w_2, cur);
    cur = ggml_add_inplace(cur, b_2);

    return cur;
}
bool sam_decode_mask(
    const struct prompt_encoder_result prompt,
    struct ggml_tensor* pe_img,
    struct ggml_tensor* embd_img,
    struct ggml_tensor* iou_predictions,
    struct ggml_tensor* low_res_masks) {

    struct ggml_tensor* tokens;
    {
        // Concatenate output tokens
        // ref: https://github.com/facebookresearch/segment-anything/blob/6fdee8f2727f4506cfbbe553e23b895e27956588/segment_anything/modeling/mask_decoder.py#L120
        struct ggml_tensor* iou_token_w = ggml_new_tensor_2d(GGML_TYPE_F32, n_enc_out_chans, 1);
        struct ggml_tensor* mask_tokens_w = ggml_new_tensor_2d(GGML_TYPE_F32, n_enc_out_chans, n_pt_embd);
        tokens = ggml_new_tensor_3d(GGML_TYPE_F32, iou_token_w->ne[0], iou_token_w->ne[1] + mask_tokens_w->ne[1] + prompt.embd_prompt_sparse->ne[1], prompt.embd_prompt_sparse->ne[2]);

        const size_t offsets[3] = { 0, iou_token_w->ne[1] * tokens->nb[1], iou_token_w->ne[1] * tokens->nb[1] + mask_tokens_w->ne[1] * tokens->nb[1] };
        ggml_cpy(iou_token_w, ggml_view_2d(tokens, tokens->ne[0], iou_token_w->ne[1], tokens->nb[1], offsets[0]));
        ggml_cpy(mask_tokens_w, ggml_view_2d(tokens, tokens->ne[0], mask_tokens_w->ne[1], tokens->nb[1], offsets[1]));
        ggml_cpy(prompt.embd_prompt_sparse, ggml_view_2d(tokens, tokens->ne[0], prompt.embd_prompt_sparse->ne[1], tokens->nb[1], offsets[2]));
        // TODO: Sparse prompt embeddings can have more than one point
    }


    struct ggml_tensor* src;
    struct ggml_tensor* pos_src;
    int srcNE[4] = { 0, 0, 0, 0 };
    {
        // Expand per-image data in the batch direction to be per-mask
        // ref: https://github.com/facebookresearch/segment-anything/blob/6fdee8f2727f4506cfbbe553e23b895e27956588/segment_anything/modeling/mask_decoder.py#L125
        src = ggml_new_tensor_4d(GGML_TYPE_F32, embd_img->ne[0], embd_img->ne[1], embd_img->ne[2], tokens->ne[2]);

        src = ggml_add(
            ggml_repeat(
                embd_img,
                src),
            prompt.embd_prompt_dense);

        srcNE[0] = src->ne[0];
        srcNE[1] = src->ne[1];
        srcNE[2] = src->ne[2];
        srcNE[3] = src->ne[3];

        // flatten & permute
        // ref: https://github.com/facebookresearch/segment-anything/blob/6fdee8f2727f4506cfbbe553e23b895e27956588/segment_anything/modeling/transformer.py#L83
        src = ggml_cont(ggml_permute(
            ggml_view_3d(
                src,
                src->ne[0] * src->ne[1],
                src->ne[2],
                src->ne[3],
                src->nb[2],
                src->nb[3],
                0),
            1, 0, 2, 3));

        pos_src = ggml_new_tensor_4d(GGML_TYPE_F32, pe_img->ne[0], pe_img->ne[1], pe_img->ne[2], tokens->ne[2]);
        pos_src = ggml_repeat(
            pe_img,
            pos_src);

        // flatten & permute
        // ref: https://github.com/facebookresearch/segment-anything/blob/6fdee8f2727f4506cfbbe553e23b895e27956588/segment_anything/modeling/transformer.py#L83
        pos_src = ggml_cont(ggml_permute(
            ggml_view_3d(
                pos_src,
                pos_src->ne[0] * pos_src->ne[1],
                pos_src->ne[2],
                pos_src->ne[3],
                pos_src->nb[2],
                pos_src->nb[3],
                0),
            1, 0, 2, 3));
    }

    struct ggml_tensor* queries = tokens;
    struct ggml_tensor* keys = src;
    {
        // Run the transformer
        // ref: https://github.com/facebookresearch/segment-anything/blob/6fdee8f2727f4506cfbbe553e23b895e27956588/segment_anything/modeling/transformer.py#L62
        const int tfm_layers_count = 2;
        for (int i = 0; i < tfm_layers_count; ++i) {
            struct sam_layer_dec_transformer tfm_layer;
            tfm_layer.self_attn.q_w = ggml_new_tensor_2d(GGML_TYPE_F16, n_enc_out_chans, n_enc_out_chans);
            tfm_layer.self_attn.q_b = ggml_new_tensor_1d(GGML_TYPE_F32, n_enc_out_chans);
            tfm_layer.self_attn.k_w = ggml_new_tensor_2d(GGML_TYPE_F16, n_enc_out_chans, n_enc_out_chans);
            tfm_layer.self_attn.k_b = ggml_new_tensor_1d(GGML_TYPE_F32, n_enc_out_chans);
            tfm_layer.self_attn.v_w = ggml_new_tensor_2d(GGML_TYPE_F16, n_enc_out_chans, n_enc_out_chans);
            tfm_layer.self_attn.v_b = ggml_new_tensor_1d(GGML_TYPE_F32, n_enc_out_chans);
            tfm_layer.self_attn.out_w = ggml_new_tensor_2d(GGML_TYPE_F16, n_enc_out_chans, n_enc_out_chans);
            tfm_layer.self_attn.out_b = ggml_new_tensor_1d(GGML_TYPE_F32, n_enc_out_chans);

            tfm_layer.norm1_w = ggml_new_tensor_1d(GGML_TYPE_F32, n_enc_out_chans);
            tfm_layer.norm1_b = ggml_new_tensor_1d(GGML_TYPE_F32, n_enc_out_chans);

            tfm_layer.cross_attn_token_to_img.q_w = ggml_new_tensor_2d(GGML_TYPE_F16, n_enc_out_chans, n_enc_out_chans / 2);
            tfm_layer.cross_attn_token_to_img.q_b = ggml_new_tensor_1d(GGML_TYPE_F32, n_enc_out_chans / 2);
            tfm_layer.cross_attn_token_to_img.k_w = ggml_new_tensor_2d(GGML_TYPE_F16, n_enc_out_chans, n_enc_out_chans / 2);
            tfm_layer.cross_attn_token_to_img.k_b = ggml_new_tensor_1d(GGML_TYPE_F32, n_enc_out_chans / 2);
            tfm_layer.cross_attn_token_to_img.v_w = ggml_new_tensor_2d(GGML_TYPE_F16, n_enc_out_chans, n_enc_out_chans / 2);
            tfm_layer.cross_attn_token_to_img.v_b = ggml_new_tensor_1d(GGML_TYPE_F32, n_enc_out_chans / 2);
            tfm_layer.cross_attn_token_to_img.out_w = ggml_new_tensor_2d(GGML_TYPE_F16, n_enc_out_chans / 2, n_enc_out_chans);
            tfm_layer.cross_attn_token_to_img.out_b = ggml_new_tensor_1d(GGML_TYPE_F32, n_enc_out_chans);

            tfm_layer.norm2_w = ggml_new_tensor_1d(GGML_TYPE_F32, n_enc_out_chans);
            tfm_layer.norm2_b = ggml_new_tensor_1d(GGML_TYPE_F32, n_enc_out_chans);

            tfm_layer.mlp_lin1_w = ggml_new_tensor_2d(GGML_TYPE_F16, n_enc_out_chans, 8 * n_enc_out_chans);
            tfm_layer.mlp_lin1_b = ggml_new_tensor_1d(GGML_TYPE_F32, 8 * n_enc_out_chans);
            tfm_layer.mlp_lin2_w = ggml_new_tensor_2d(GGML_TYPE_F16, 8 * n_enc_out_chans, n_enc_out_chans);
            tfm_layer.mlp_lin2_b = ggml_new_tensor_1d(GGML_TYPE_F32, n_enc_out_chans);

            tfm_layer.norm3_w = ggml_new_tensor_1d(GGML_TYPE_F32, n_enc_out_chans);
            tfm_layer.norm3_b = ggml_new_tensor_1d(GGML_TYPE_F32, n_enc_out_chans);

            tfm_layer.norm4_w = ggml_new_tensor_1d(GGML_TYPE_F32, n_enc_out_chans);
            tfm_layer.norm4_b = ggml_new_tensor_1d(GGML_TYPE_F32, n_enc_out_chans);

            tfm_layer.cross_attn_img_to_token.q_w = ggml_new_tensor_2d(GGML_TYPE_F16, n_enc_out_chans, n_enc_out_chans / 2);
            tfm_layer.cross_attn_img_to_token.q_b = ggml_new_tensor_1d(GGML_TYPE_F32, n_enc_out_chans / 2);
            tfm_layer.cross_attn_img_to_token.k_w = ggml_new_tensor_2d(GGML_TYPE_F16, n_enc_out_chans, n_enc_out_chans / 2);
            tfm_layer.cross_attn_img_to_token.k_b = ggml_new_tensor_1d(GGML_TYPE_F32, n_enc_out_chans / 2);
            tfm_layer.cross_attn_img_to_token.v_w = ggml_new_tensor_2d(GGML_TYPE_F16, n_enc_out_chans, n_enc_out_chans / 2);
            tfm_layer.cross_attn_img_to_token.v_b = ggml_new_tensor_1d(GGML_TYPE_F32, n_enc_out_chans / 2);
            tfm_layer.cross_attn_img_to_token.out_w = ggml_new_tensor_2d(GGML_TYPE_F16, n_enc_out_chans / 2, n_enc_out_chans);
            tfm_layer.cross_attn_img_to_token.out_b = ggml_new_tensor_1d(GGML_TYPE_F32, n_enc_out_chans);

            // Self attention block
            // ref: https://github.com/facebookresearch/segment-anything/blob/6fdee8f2727f4506cfbbe553e23b895e27956588/segment_anything/modeling/transformer.py#L154
            const bool skip_first_layer_pe = i == 0;
            if (skip_first_layer_pe) {
                queries = sam_decode_mask_transformer_attn(tfm_layer.self_attn, queries, queries, queries);
            }
            else {
                struct ggml_tensor* q_0 = ggml_add(queries, tokens);

                struct ggml_tensor* self_attn = sam_decode_mask_transformer_attn(tfm_layer.self_attn, q_0, q_0, queries);
                queries = ggml_add(queries, self_attn);
            }

            queries = ggml_norm(queries, eps_decoder_transformer);
            queries = ggml_add_inplace(
                ggml_mul(queries, tfm_layer.norm1_w),
                tfm_layer.norm1_b);

            // Cross attention block, tokens attending to image embedding
            // ref: https://github.com/facebookresearch/segment-anything/blob/6fdee8f2727f4506cfbbe553e23b895e27956588/segment_anything/modeling/transformer.py#L163
            struct ggml_tensor* q_1 = ggml_add(queries, tokens);
            struct ggml_tensor* k_1 = ggml_add(keys, pos_src);

            struct ggml_tensor* cross_attn_token_to_img = sam_decode_mask_transformer_attn(tfm_layer.cross_attn_token_to_img, q_1, k_1, keys);

            queries = ggml_add_inplace(queries, cross_attn_token_to_img);
            queries = ggml_norm_inplace(queries, eps_decoder_transformer);
            queries = ggml_add_inplace(
                ggml_mul(queries, tfm_layer.norm2_w),
                tfm_layer.norm2_b);

            // MLP block
            // ref: https://github.com/facebookresearch/segment-anything/blob/6fdee8f2727f4506cfbbe553e23b895e27956588/segment_anything/modeling/transformer.py#L170
            struct ggml_tensor* mlp_out = ggml_mul_mat(
                tfm_layer.mlp_lin1_w,
                queries);

            mlp_out = ggml_add_inplace(mlp_out, tfm_layer.mlp_lin1_b);

            // RELU activation
            mlp_out = ggml_relu_inplace(mlp_out);
            mlp_out = ggml_mul_mat(tfm_layer.mlp_lin2_w, mlp_out);

            mlp_out = ggml_add_inplace(mlp_out, tfm_layer.mlp_lin2_b);

            queries = ggml_add_inplace(queries, mlp_out);
            queries = ggml_norm_inplace(queries, eps_decoder_transformer);
            queries = ggml_add_inplace(
                ggml_mul(queries, tfm_layer.norm3_w),
                tfm_layer.norm3_b);

            // Cross attention block, image embedding attending to tokens
            // ref: https://github.com/facebookresearch/segment-anything/blob/6fdee8f2727f4506cfbbe553e23b895e27956588/segment_anything/modeling/transformer.py#L175
            struct ggml_tensor* q_2 = ggml_add(queries, tokens);
            struct ggml_tensor* k_2 = ggml_add(keys, pos_src);

            struct ggml_tensor* cross_attn_img_to_token = sam_decode_mask_transformer_attn(tfm_layer.cross_attn_img_to_token, k_2, q_2, queries);
            keys = ggml_add_inplace(keys, cross_attn_img_to_token);
            keys = ggml_norm_inplace(keys, eps_decoder_transformer);
            keys = ggml_add_inplace(
                ggml_mul(keys, tfm_layer.norm4_w),
                tfm_layer.norm4_b);
        }

        // Apply the final attention layer from the points to the image
        // ref: https://github.com/facebookresearch/segment-anything/blob/6fdee8f2727f4506cfbbe553e23b895e27956588/segment_anything/modeling/transformer.py#L99
        struct ggml_tensor* q = ggml_add(queries, tokens);
        struct ggml_tensor* k = ggml_add(keys, pos_src);

        struct sam_layer_dec_transformer_attn transformer_final_attn_token_to_img;
        transformer_final_attn_token_to_img.q_w = ggml_new_tensor_2d(GGML_TYPE_F16, n_enc_out_chans, n_enc_out_chans / 2);
        transformer_final_attn_token_to_img.q_b = ggml_new_tensor_1d(GGML_TYPE_F32, n_enc_out_chans / 2);
        transformer_final_attn_token_to_img.k_w = ggml_new_tensor_2d(GGML_TYPE_F16, n_enc_out_chans, n_enc_out_chans / 2);
        transformer_final_attn_token_to_img.k_b = ggml_new_tensor_1d(GGML_TYPE_F32, n_enc_out_chans / 2);
        transformer_final_attn_token_to_img.v_w = ggml_new_tensor_2d(GGML_TYPE_F16, n_enc_out_chans, n_enc_out_chans / 2);
        transformer_final_attn_token_to_img.v_b = ggml_new_tensor_1d(GGML_TYPE_F32, n_enc_out_chans / 2);
        transformer_final_attn_token_to_img.out_w = ggml_new_tensor_2d(GGML_TYPE_F16, n_enc_out_chans / 2, n_enc_out_chans);
        transformer_final_attn_token_to_img.out_b = ggml_new_tensor_1d(GGML_TYPE_F32, n_enc_out_chans);
        struct ggml_tensor* final_attn_token_to_img = sam_decode_mask_transformer_attn(transformer_final_attn_token_to_img, q, k, keys);

        queries = ggml_add_inplace(queries, final_attn_token_to_img);
        queries = ggml_norm_inplace(queries, eps_decoder_transformer);

        struct ggml_tensor* transformer_norm_final_w = ggml_new_tensor_1d(GGML_TYPE_F32, n_enc_out_chans);
        struct ggml_tensor* transformer_norm_final_b = ggml_new_tensor_1d(GGML_TYPE_F32, n_enc_out_chans);
        queries = ggml_add_inplace(
            ggml_mul(queries, transformer_norm_final_w),
            transformer_norm_final_b);
    }


    struct ggml_tensor* iou_pred = ggml_view_2d(queries, queries->ne[0], queries->ne[2], queries->nb[2], 0);
    const int num_mask_tokens = 4; // num_multimask_outputs + 1
    struct ggml_tensor* mask_tokens_out = ggml_view_3d(queries, queries->ne[0], num_mask_tokens, queries->ne[2], queries->nb[1], num_mask_tokens * queries->nb[1], queries->nb[1]);

    // Upscale mask embeddings and predict masks using the mask tokens
    // ref: https://github.com/facebookresearch/segment-anything/blob/6fdee8f2727f4506cfbbe553e23b895e27956588/segment_anything/modeling/mask_decoder.py#L136
    keys = ggml_cont(ggml_transpose(keys));
    keys = ggml_view_4d(keys, srcNE[0], srcNE[1], srcNE[2], srcNE[3], srcNE[0] * keys->nb[0], keys->nb[1], keys->nb[2], 0);
    // ggml_build_forward_expand(gf, keys);
    struct ggml_tensor* upscaled_embedding;
    {
        struct ggml_tensor* output_upscaling_0_w = ggml_new_tensor_4d(GGML_TYPE_F16, 2, 2, n_img_embd(), n_enc_out_chans);
        struct ggml_tensor* output_upscaling_0_b = ggml_new_tensor_1d(GGML_TYPE_F32, n_img_embd());
        struct ggml_tensor* output_upscaling_1_w = ggml_new_tensor_1d(GGML_TYPE_F32, n_img_embd());
        struct ggml_tensor* output_upscaling_1_b = ggml_new_tensor_1d(GGML_TYPE_F32, n_img_embd());
        struct ggml_tensor* output_upscaling_3_w = ggml_new_tensor_4d(GGML_TYPE_F16, 2, 2, n_img_embd() / 2, n_img_embd());
        struct ggml_tensor* output_upscaling_3_b = ggml_new_tensor_1d(GGML_TYPE_F32, n_img_embd() / 2);
        // ConvTranspose2d
        keys = ggml_conv_transpose_2d_p0(output_upscaling_0_w, keys, 2);
        keys = ggml_add_inplace(keys, ggml_repeat(
            ggml_reshape_3d(output_upscaling_0_b, 1, 1, output_upscaling_0_b->ne[0]),
            keys));

        keys = ggml_layer_norm_2d(keys, n_img_embd, output_upscaling_1_w, output_upscaling_1_b, eps);

        // GELU activation
        keys = ggml_gelu_inplace(keys);

        // ConvTranspose2d
        keys = ggml_conv_transpose_2d_p0(output_upscaling_3_w, keys, 2);
        keys = ggml_add_inplace(ggml_repeat(
            ggml_reshape_3d(output_upscaling_3_b, 1, 1, output_upscaling_3_b->ne[0]),
            keys), keys);
        // GELU activation
        keys = ggml_gelu_inplace(keys);
        upscaled_embedding = ggml_reshape_3d(keys, keys->ne[0] * keys->ne[1], keys->ne[2], keys->ne[3]);
        upscaled_embedding = ggml_cont(ggml_transpose(upscaled_embedding)); // TODO: Shouldn't be needed
    }

    struct ggml_tensor* hyper_in = ggml_new_tensor_3d(GGML_TYPE_F32, n_img_embd() / 2, num_mask_tokens, mask_tokens_out->ne[2]);

    for (int i = 0; i < num_mask_tokens; ++i) {
        struct sam_layer_dec_output_hypernet_mlps mlp;
        mlp.w_0 = ggml_new_tensor_2d(GGML_TYPE_F16, n_enc_out_chans, n_enc_out_chans);
        mlp.b_0 = ggml_new_tensor_1d(GGML_TYPE_F32, n_enc_out_chans);
        mlp.w_1 = ggml_new_tensor_2d(GGML_TYPE_F16, n_enc_out_chans, n_enc_out_chans);
        mlp.b_1 = ggml_new_tensor_1d(GGML_TYPE_F32, n_enc_out_chans);
        mlp.w_2 = ggml_new_tensor_2d(GGML_TYPE_F16, n_enc_out_chans, n_img_embd() / 2);
        mlp.b_2 = ggml_new_tensor_1d(GGML_TYPE_F32, n_img_embd() / 2);
        struct ggml_tensor* in = ggml_view_2d(mask_tokens_out, mask_tokens_out->ne[0], mask_tokens_out->ne[2], mask_tokens_out->nb[1], i * mask_tokens_out->nb[1]);
        struct ggml_tensor* out = sam_decode_mask_mlp_relu_3(in, mlp.w_0, mlp.b_0, mlp.w_1, mlp.b_1, mlp.w_2, mlp.b_2);
        ggml_cpy(out, ggml_view_2d(hyper_in, hyper_in->ne[0], hyper_in->ne[2], hyper_in->nb[1], i * hyper_in->nb[1]));
    }

    struct ggml_tensor* masks = ggml_mul_mat(hyper_in, upscaled_embedding);
    masks = ggml_cont(ggml_transpose(masks)); // TODO: Shouldn't be needed
    masks = ggml_reshape_4d(masks, keys->ne[0], keys->ne[1], masks->ne[1], keys->ne[3]);

    struct ggml_tensor* iou_prediction_head_0_w = ggml_new_tensor_2d(GGML_TYPE_F16, n_enc_out_chans, n_enc_out_chans);
    struct ggml_tensor* iou_prediction_head_0_b = ggml_new_tensor_1d(GGML_TYPE_F32, n_enc_out_chans);
    struct ggml_tensor* iou_prediction_head_1_w = ggml_new_tensor_2d(GGML_TYPE_F16, n_enc_out_chans, n_enc_out_chans);
    struct ggml_tensor* iou_prediction_head_1_b = ggml_new_tensor_1d(GGML_TYPE_F32, n_enc_out_chans);
    struct ggml_tensor* iou_prediction_head_2_w = ggml_new_tensor_2d(GGML_TYPE_F16, n_enc_out_chans, n_pt_embd);
    struct ggml_tensor* iou_prediction_head_2_b = ggml_new_tensor_1d(GGML_TYPE_F32, n_pt_embd);
    // Generate mask quality predictions
    // ref: https://github.com/facebookresearch/segment-anything/blob/6fdee8f2727f4506cfbbe553e23b895e27956588/segment_anything/modeling/mask_decoder.py#L146
    iou_pred = sam_decode_mask_mlp_relu_3(iou_pred, iou_prediction_head_0_w, iou_prediction_head_0_b, iou_prediction_head_1_w, iou_prediction_head_1_b, iou_prediction_head_2_w, iou_prediction_head_2_b);

    // Select the correct mask or masks for output
    // ref: https://github.com/facebookresearch/segment-anything/blob/6fdee8f2727f4506cfbbe553e23b895e27956588/segment_anything/modeling/mask_decoder.py#L101
    iou_pred = ggml_cpy(ggml_view_1d(iou_pred, iou_pred->ne[0] - 1, iou_pred->nb[0]), iou_predictions);
    masks = ggml_view_4d(masks, masks->ne[0], masks->ne[1], masks->ne[2] - 1, masks->ne[3],
        masks->nb[1], masks->nb[2], masks->nb[3], masks->nb[2] /* offset*/);
    masks = ggml_cpy(masks, low_res_masks);

    return true;
}


bool sam_write_masks(int nx, int ny, struct ggml_tensor* low_res_masks, struct ggml_tensor* iou_predictions) {
    if (low_res_masks->ne[2] == 0) return true;
    if (low_res_masks->ne[2] != iou_predictions->ne[0]) {
        printf("Error: number of masks (%d) does not match number of iou predictions (%d)\n", (int)low_res_masks->ne[2], (int)iou_predictions->ne[0]);
        return false;
    }

    const float intersection_threshold = mask_threshold + stability_score_offset;
    const float union_threshold = mask_threshold - stability_score_offset;

    const int ne0 = low_res_masks->ne[0];
    const int ne1 = low_res_masks->ne[1];
    const int ne2 = low_res_masks->ne[2];

    // Remove padding and upscale masks to the original image size.
    // ref: https://github.com/facebookresearch/segment-anything/blob/efeab7296ab579d4a261e554eca80faf6b33924a/segment_anything/modeling/sam.py#L140

    const float preprocess_scale = max(nx, ny) / (float)(n_img_size());
    const int cropped_nx = (int)(nx / preprocess_scale + 0.5f);
    const int cropped_ny = (int)(ny / preprocess_scale + 0.5f);

    const float scale_x_1 = (float)ne0 / (float)n_img_size();
    const float scale_y_1 = (float)ne1 / (float)n_img_size();

    const float scale_x_2 = (float)(cropped_nx) / (float)(nx);
    const float scale_y_2 = (float)(cropped_ny) / (float)(ny);

    const float* iou_data = (float*)iou_predictions->data;

    for (int i = 0; i < ne2; ++i) {
        if (iou_threshold > 0.f && iou_data[i] < iou_threshold) {
            printf("Skipping mask %d with iou %f below threshold %f\n", i, iou_data[i], iou_threshold);
            continue; // Filtering masks with iou below the threshold
        }

        float* mask_data = calloc(n_img_size() * n_img_size(), sizeof(float));
        {
            const float* data = (float*)low_res_masks->data + i * ne0 * ne1;

            for (int iy = 0; iy < n_img_size(); ++iy) {
                for (int ix = 0; ix < n_img_size(); ++ix) {
                    const float sx = max(scale_x_1 * (ix + 0.5f) - 0.5f, 0.0f);
                    const float sy = max(scale_y_1 * (iy + 0.5f) - 0.5f, 0.0f);

                    const int x0 = max(0, (int)sx);
                    const int y0 = max(0, (int)sy);

                    const int x1 = min(x0 + 1, ne0 - 1);
                    const int y1 = min(y0 + 1, ne1 - 1);

                    const float dx = sx - x0;
                    const float dy = sy - y0;

                    const int j00 = y0 * ne0 + x0;
                    const int j01 = y0 * ne0 + x1;
                    const int j10 = y1 * ne0 + x0;
                    const int j11 = y1 * ne0 + x1;

                    const float v00 = data[j00];
                    const float v01 = data[j01];
                    const float v10 = data[j10];
                    const float v11 = data[j11];

                    const float v0 = (1 - dx) * v00 + dx * v01;
                    const float v1 = (1 - dx) * v10 + dx * v11;

                    const float v = (1 - dy) * v0 + dy * v1;

                    mask_data[iy * n_img_size() + ix] = v;
                }
            }
        }

        int intersections = 0;
        int unions = 0;
        int min_iy = ny;
        int max_iy = 0;
        int min_ix = nx;
        int max_ix = 0;
        {
            const float* data = mask_data;

            uint8_t* res = calloc(nx * ny, sizeof(uint8_t));

            for (int iy = 0; iy < ny; ++iy) {
                for (int ix = 0; ix < nx; ++ix) {
                    const float sx = max(scale_x_2 * (ix + 0.5f) - 0.5f, 0.0f);
                    const float sy = max(scale_y_2 * (iy + 0.5f) - 0.5f, 0.0f);

                    const int x0 = max(0, (int)sx);
                    const int y0 = max(0, (int)sy);

                    const int x1 = min(x0 + 1, cropped_nx - 1);
                    const int y1 = min(y0 + 1, cropped_ny - 1);

                    const float dx = sx - x0;
                    const float dy = sy - y0;

                    const int j00 = y0 * n_img_size() + x0;
                    const int j01 = y0 * n_img_size() + x1;
                    const int j10 = y1 * n_img_size() + x0;
                    const int j11 = y1 * n_img_size() + x1;

                    const float v00 = data[j00];
                    const float v01 = data[j01];
                    const float v10 = data[j10];
                    const float v11 = data[j11];

                    const float v0 = (1 - dx) * v00 + dx * v01;
                    const float v1 = (1 - dx) * v10 + dx * v11;

                    const float v = (1 - dy) * v0 + dy * v1;

                    if (v > intersection_threshold) {
                        intersections++;
                    }
                    if (v > union_threshold) {
                        unions++;
                    }
                    if (v > mask_threshold) {
                        min_iy = min(min_iy, iy);
                        max_iy = max(max_iy, iy);
                        min_ix = min(min_ix, ix);
                        max_ix = max(max_ix, ix);

                        res[iy * nx + ix] = 255;
                    }
                }
            }
        }

        const float stability_score = (float)(intersections) / (float)(unions);
        if (stability_score_threshold > 0.f && stability_score < stability_score_threshold) {
            printf("Skipping mask %d with stability score %f below threshold %f\n", i, stability_score, stability_score_threshold);
            continue; // Filtering masks with stability score below the threshold
        }

        printf("Mask %d: iou = %f, stability_score = %f, bbox (%d, %d), (%d, %d)\n",
            i, iou_data[i], stability_score, min_ix, max_ix, min_iy, max_iy);

    }


    return true;
}

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
        int32_t magic = *(int32_t*)((int8_t*)weights);
        index += sizeof(int32_t);

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

        const int32_t qntvr = ftype / GGML_QNT_VERSION_FACTOR;
        ftype %= GGML_QNT_VERSION_FACTOR;

    }
    struct ggml_tensor* embd_img = ggml_new_tensor_3d(GGML_TYPE_F32,
        n_img_embd(), n_img_embd(), n_enc_out_chans);

    struct ggml_tensor* low_res_masks = ggml_new_tensor_3d(GGML_TYPE_F32,
        n_enc_out_chans, n_enc_out_chans, 3);

    struct ggml_tensor* iou_predictions = ggml_new_tensor_1d(GGML_TYPE_F32, 3);

    sam_encode_image(img1, embd_img, nx, ny, weights);


    struct ggml_tensor* pe = ggml_new_tensor_2d(GGML_TYPE_F32, n_enc_out_chans / 2, 2);
    struct ggml_tensor* inp = ggml_new_tensor_2d(GGML_TYPE_F32, 2, 2);
    // from sam_encode_prompt
    {
        // transform points
        // ref: https://github.com/facebookresearch/segment-anything/blob/main/segment_anything/automatic_mask_generator.py#L276
        {
            const int nmax = max(nx, ny);

            const float scale = n_img_size() / (float)nmax;

            const int nx_new = (int)(nx * scale + 0.5f);
            const int ny_new = (int)(ny * scale + 0.5f);

            x = x * ((float)(nx_new) / nx) + 0.5f;
            y = y * ((float)(ny_new) / ny) + 0.5f;
        }

        // set the input by converting the [0, 1] coordinates to [-1, 1]
        float* data = (float*)inp->data;

        data[0] = 2.0f * (x / n_img_size()) - 1.0f;
        data[1] = 2.0f * (y / n_img_size()) - 1.0f;

        // padding
        // ref: https://github.com/facebookresearch/segment-anything/blob/main/segment_anything/modeling/prompt_encoder.py#L81-L85
        data[2] = 2.0f * (0.0f) - 1.0f;
        data[3] = 2.0f * (0.0f) - 1.0f;
    }
    struct prompt_encoder_result enc_res = sam_encode_prompt(pe, inp);
    if (!enc_res.embd_prompt_sparse || !enc_res.embd_prompt_dense) {
        fprintf(stderr, "%s: failed to encode prompt (%f, %f)\n", __func__, x, y);
        return AUTOX_FAIL;
    }

    struct ggml_tensor* xy_embed_stacked = ggml_new_tensor_3d(GGML_TYPE_F32, 2, n_img_embd(), n_img_embd());
    // from sam_fill_dense_pe
    {
        const float n_img_embd_inv = 1.0f / n_img_embd();
        float* data = (float*)xy_embed_stacked->data;
        for (int i = 0; i < n_img_embd; ++i) {
            const int row = 2 * i * n_img_embd();
            const float y_val = 2 * (i + 0.5f) * n_img_embd_inv - 1;
            for (int j = 0; j < n_img_embd; ++j) {
                const float x_val = 2 * (j + 0.5f) * n_img_embd_inv - 1;
                data[row + 2 * j + 0] = x_val;
                data[row + 2 * j + 1] = y_val;
            }
        }
    }

    struct ggml_tensor* pe_img_dense = sam_fill_dense_pe(pe, xy_embed_stacked);
    if (!pe_img_dense) {
        fprintf(stderr, "%s: failed to get dense positional encoding\n", __func__);
        return AUTOX_FAIL;
    }

    if (!sam_decode_mask(enc_res, pe_img_dense, embd_img, iou_predictions, low_res_masks)) {
        fprintf(stderr, "%s: failed to decode mask\n", __func__);
        return AUTOX_FAIL;
    }

    if (!sam_write_masks(nx, ny, low_res_masks, iou_predictions)) {
        fprintf(stderr, "%s: failed to write masks\n", __func__);
        return AUTOX_FAIL;
    }
	return err;
}