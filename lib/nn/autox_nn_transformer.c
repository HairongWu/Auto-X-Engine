#include "../include/autox_nn.h"
#include <string.h>

float *autox_transformer(Transformer *transformer, int token, int pos) {
    // a few convenience variables
    Config *p = &transformer->config;
    TransformerWeights *w = &transformer->weights;
    RunState *s = &transformer->state;
    float *x = s->x;
    int dim = p->dim;
    int kv_dim = (p->dim * p->n_kv_heads) / p->n_heads;
    int kv_mul = p->n_heads / p->n_kv_heads; // integer multiplier of the kv sharing in multiquery
    int hidden_dim = p->hidden_dim;
    int head_size = dim / p->n_heads;

    // copy the token embedding into x
    float *content_row = w->token_embedding_table + token * dim;
    memcpy(x, content_row, dim * sizeof(*x));
	uint32_t x_dims[1];
	uint32_t y_dims[2];
	uint32_t o_dims[1];
    // forward all the layers
    for (unsigned long long l = 0; l < p->n_layers; l++) {
        // attention rmsnorm
        autox_rmsnorm(s->xb, x, w->rms_att_weight + l * dim, dim);

        // key and value point to the kv cache
        int loff = l * p->seq_len * kv_dim; // kv cache layer offset for convenience
        s->k = s->key_cache + loff + pos * kv_dim;
        s->v = s->value_cache + loff + pos * kv_dim;

        // qkv matmuls for this position
		x_dims[0] = dim;
		y_dims[0] = dim;
		y_dims[1] = dim;
		o_dims[0] = dim;
		autox_matmul(s->xb, w->wq + l * dim*dim, s->q, x_dims, y_dims, o_dims, false, true, 1,2,1);
		x_dims[0] = dim;
		y_dims[0] = kv_dim;
		y_dims[1] = dim;
		o_dims[0] = kv_dim;
		autox_matmul(s->xb, w->wk + l * dim*kv_dim, s->k, x_dims, y_dims, o_dims, false, true, 1, 2, 1);
		autox_matmul(s->xb, w->wv + l * dim*kv_dim, s->v, x_dims, y_dims, o_dims, false, true, 1, 2, 1);

        // RoPE relative positional encoding: complex-valued rotate q and k in each head
        autox_rope_rotation(pos, s->q, s->k, dim, kv_dim, head_size);

        // multihead attention. iterate over all heads
        autox_multi_head_attention(p->n_heads,pos, p->seq_len, s->q, s->att, s->xb,
			s->key_cache, s->value_cache, kv_dim, kv_mul,
			head_size, loff);

        // final matmul to get the output of the attention
		x_dims[0] = dim;
		y_dims[0] = dim;
		y_dims[1] = dim;
		o_dims[0] = dim;
		autox_matmul(s->xb, w->wo + l * dim*dim, s->xb2, x_dims, y_dims, o_dims, false, true, 1, 2, 1);

        // residual connection back into x
        // autox_elementwise_add(x, s->xb2, x, dim);

        // ffn rmsnorm
        autox_rmsnorm(s->xb, x, w->rms_ffn_weight + l * dim, dim);

        // Now for FFN in PyTorch we have: self.w2(F.silu(self.w1(x)) * self.w3(x))
        // first calculate self.w1(x) and self.w3(x)
		x_dims[0] = dim;
		y_dims[0] = hidden_dim;
		y_dims[1] = dim;
		o_dims[0] = hidden_dim;
		autox_matmul(s->xb, w->w1 + l * dim*hidden_dim, s->hb, x_dims, y_dims, o_dims, false, true, 1, 2, 1);
		autox_matmul(s->xb, w->w3 + l * dim*hidden_dim, s->hb2, x_dims, y_dims, o_dims, false, true, 1, 2, 1);

        // SwiGLU non-linearity
        autox_swiglu(s->hb, s->hb2, hidden_dim);

        // final matmul to get the output of the ffn
		x_dims[0] = hidden_dim;
		y_dims[0] = dim;
		y_dims[1] = hidden_dim;
		o_dims[0] = dim;
		autox_matmul(s->hb, w->w2 + l * dim*hidden_dim, s->xb, x_dims, y_dims, o_dims, false, true, 1, 2, 1);

        // residual connection
       // autox_elementwise_add(x, s->xb, x, dim);
    }

    // final rmsnorm
    autox_rmsnorm(x, x, w->rms_final_weight, dim);

    // classifier into logits
	x_dims[0] = p->dim;
	y_dims[0] = p->vocab_size;
	y_dims[1] = p->dim;
	o_dims[0] = p->vocab_size;
	autox_matmul(x, w->wcls, s->logits, x_dims, y_dims, o_dims, false, true, 1, 2, 1);
    return s->logits;
}
