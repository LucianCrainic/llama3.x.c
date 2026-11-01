/* LLaMA 3 CPU inference, derived from Andrej Karpathy and James W. DeLancey.
 * See LICENSE. Numerical kernels intentionally remain independent per precision. */
#include "kernels.h"
#include "profile.h"
#include "runtime.h"
#include "shards.h"
#include <math.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>
#include <unistd.h>

typedef struct {
  // token embedding table
  const float *token_embedding_table; // (vocab_size, dim)
  // weights for rmsnorms
  const float *rms_att_weight; // (layer, dim) rmsnorm weights
  const float *rms_ffn_weight; // (layer, dim)
  // weights for matmuls. note dim == n_heads * head_size
  const float *wq; // (layer, dim, n_heads * head_size)
  const float *wk; // (layer, dim, n_kv_heads * head_size)
  const float *wv; // (layer, dim, n_kv_heads * head_size)
  const float *wo; // (layer, n_heads * head_size, dim)
  // weights for ffn
  const float *w1; // (layer, hidden_dim, dim)
  const float *w2; // (layer, dim, hidden_dim)
  const float *w3; // (layer, hidden_dim, dim)
  // final rmsnorm
  const float *rms_final_weight; // (dim,)
  // (optional) classifier weights for the logits, on the last layer
  const float *wcls;
} TransformerWeights;

typedef struct {
  // current wave of activations
  float *x;      // activation at current time stamp (dim,)
  float *xb;     // same, but inside a residual branch (dim,)
  float *hb;     // buffer for hidden dimension in the ffn (hidden_dim,)
  float *q;      // query (dim,)
  float *k;      // key (dim,)
  float *v;      // value (dim,)
  float *att;    // buffer for scores/attention values (n_heads, seq_len)
  float *logits; // output logits
  float *rope;   // RoPE cosines and signed sines for the current position (2 * head_size,)
  AttentionWork attention; // split-K attention partials
  // kv cache
  float *key_cache;   // (layer, seq_len, kv_dim); cache persists for this sequence
  float *value_cache; // (layer, seq_len, kv_dim); cache persists for this sequence
} RunState;

typedef struct {
  Config config;              // the hyperparameters of the architecture (the blueprint)
  TransformerWeights weights; // the weights of the model
  TensorManifest manifest;    // where each tensor lives in the file
  RunState state;             // buffers for the "wave" of activations in the forward pass
  // Mapping ownership: weights borrow this region until engine_free().
  int fd;           // file descriptor for memory mapping
  float *data;      // memory mapped data pointer
  size_t file_size; // size of the checkpoint file in bytes
} Transformer;

static void malloc_run_state(RunState *s, Config *p) {
  // Zeroed activations and cache are owned for the lifetime of this model.
  int kv_dim = (p->dim / p->n_heads) * p->n_kv_heads;
  s->x = memory_alloc(p->dim, sizeof(float), MEM_ACTIVATION);
  s->xb = memory_alloc(p->dim, sizeof(float), MEM_ACTIVATION);
  s->hb = memory_alloc(p->hidden_dim, sizeof(float), MEM_ACTIVATION);
  s->q = memory_alloc(p->dim, sizeof(float), MEM_ACTIVATION);
  s->key_cache = memory_alloc((size_t)p->n_layers * p->seq_len * kv_dim, sizeof(float), MEM_KV);
  s->value_cache = memory_alloc((size_t)p->n_layers * p->seq_len * kv_dim, sizeof(float), MEM_KV);
  s->att = memory_alloc((size_t)p->n_heads * p->seq_len, sizeof(float), MEM_ACTIVATION);
  s->logits = memory_alloc(p->vocab_size, sizeof(float), MEM_ACTIVATION);
  s->rope = memory_alloc(2 * (size_t)p->head_dim, sizeof(float), MEM_ACTIVATION);
  k_attention_init(&s->attention, p->n_heads, p->n_kv_heads, p->head_dim, p->seq_len);
}

static void free_run_state(RunState *s) {
  memory_free(s->x);
  memory_free(s->xb);
  memory_free(s->hb);
  memory_free(s->q);
  memory_free(s->att);
  memory_free(s->logits);
  memory_free(s->key_cache);
  memory_free(s->value_cache);
  memory_free(s->rope);
  k_attention_free(&s->attention);
}

/* Weight pointers are views into the file mapping at manifest offsets. Layers of a
 * role are contiguous, so layer 0 addresses the whole array. */
static const float *weight_view(const float *data, const TensorManifest *m, TensorRole role, int layer) {
  const TensorDesc *t = manifest_find(m, role, layer);
  if (!t)
    fail("tensor manifest has no %s tensor (layer %d)", manifest_role_name(role), layer);
  return (const float *)((const char *)data + t->offset);
}

static void memory_map_weights(TransformerWeights *w, const TensorManifest *m, const float *data) {
  w->token_embedding_table = weight_view(data, m, T_EMBED, -1);
  w->rms_att_weight = weight_view(data, m, T_RMS_ATT, 0);
  w->wq = weight_view(data, m, T_WQ, 0);
  w->wk = weight_view(data, m, T_WK, 0);
  w->wv = weight_view(data, m, T_WV, 0);
  w->wo = weight_view(data, m, T_WO, 0);
  w->rms_ffn_weight = weight_view(data, m, T_RMS_FFN, 0);
  w->w1 = weight_view(data, m, T_W1, 0);
  w->w2 = weight_view(data, m, T_W2, 0);
  w->w3 = weight_view(data, m, T_W3, 0);
  w->rms_final_weight = weight_view(data, m, T_RMS_FINAL, -1);
  w->wcls = weight_view(data, m, T_WCLS, -1);
}

static void read_checkpoint(const char *path, Config *config, TransformerWeights *weights, TensorManifest *manifest,
                            int *fd, float **data, size_t *file_size) {
  int shared, gs;
  double start = monotonic_seconds();
  checkpoint_map(path, 0, config, &shared, &gs, fd, data, file_size, manifest);
  memory_map_weights(weights, manifest, *data);
  metrics.checkpoint_s = monotonic_seconds() - start;
  metrics.checkpoint_ready = 1;
}

static void free_transformer(Transformer *t) {
  // close the memory mapping
  if (t->data != MAP_FAILED) {
    munmap(t->data, t->file_size);
  }
  if (t->fd != -1) {
    close(t->fd);
  }
  manifest_free(&t->manifest);
  // free the RunState buffers
  free_run_state(&t->state);
}

// ----------------------------------------------------------------------------
// neural net blocks; the dynamics of the Transformer

/* One autoregressive position: embeddings -> attention/RoPE over cached prefix
 * -> SwiGLU FFN per layer -> normalized vocabulary logits. Cache rows are
 * [layer][position][KV head * head_size]; GQA query heads share KV heads. */
static float *forward(Transformer *transformer, int token, int pos) {

  // a few convenience variables
  Config *p = &transformer->config;
  TransformerWeights *w = &transformer->weights;
  RunState *s = &transformer->state;
  float *x = s->x;
  int dim = p->dim;
  int kv_dim = (p->dim / p->n_heads) * p->n_kv_heads;
  int hidden_dim = p->hidden_dim;
  int head_size = dim / p->n_heads;

  // copy the token embedding into x
  const float *content_row = w->token_embedding_table + (size_t)token * dim;
  PROFILE(PHASE_EMBED, memcpy(x, content_row, dim * sizeof(*x)));
  // RoPE angles depend only on the position: compute them once for all layers
  PROFILE(PHASE_ROPE_TABLE, k_rope_table(p, pos, s->rope));

  // forward all the layers
  for (size_t l = 0; l < (size_t)p->n_layers; l++) {

    // attention rmsnorm
    PROFILE(PHASE_RMSNORM, k_rmsnorm(s->xb, x, w->rms_att_weight + l * dim, dim, p->norm_eps));

    // key and value point to the kv cache
    size_t loff = l * p->seq_len * kv_dim; // kv cache layer offset for convenience
    s->k = s->key_cache + loff + (size_t)pos * kv_dim;
    s->v = s->value_cache + loff + (size_t)pos * kv_dim;

    // qkv matmuls for this position, in one parallel region
    PROFILE(PHASE_QKV, k_matmul_qkv(s->q, s->k, s->v, s->xb, w->wq + l * dim * dim, w->wk + l * dim * kv_dim,
                                    w->wv + l * dim * kv_dim, dim, dim, kv_dim));

    PROFILE(PHASE_ROPE, k_rope_apply(s->q, s->k, s->rope, p->n_heads, p->n_kv_heads, head_size));

    // multihead attention over positions 0..pos, output into xb
    PROFILE(PHASE_ATTENTION, k_attention(s->xb, s->q, s->key_cache + loff, s->value_cache + loff, s->att,
                                         &s->attention, p->n_heads, p->n_kv_heads, head_size, p->seq_len, pos));

    // final matmul to get the output of the attention, with the residual connection back into x
    PROFILE(PHASE_WO, k_matmul_add(x, s->xb, w->wo + l * dim * dim, dim, dim));

    // ffn rmsnorm
    PROFILE(PHASE_RMSNORM, k_rmsnorm(s->xb, x, w->rms_ffn_weight + l * dim, dim, p->norm_eps));

    // Now for FFN in PyTorch we have: self.w2(F.silu(self.w1(x)) * self.w3(x))
    // w1 and w3 share one pass that applies the SwiGLU non-linearity
    PROFILE(PHASE_W1_W3, k_matmul_swiglu(s->hb, s->xb, w->w1 + l * dim * hidden_dim, w->w3 + l * dim * hidden_dim,
                                         dim, hidden_dim));

    // final matmul to get the output of the ffn, with the residual connection
    PROFILE(PHASE_W2, k_matmul_add(x, s->hb, w->w2 + l * dim * hidden_dim, hidden_dim, dim));
  }

  // final rmsnorm
  PROFILE(PHASE_RMSNORM, k_rmsnorm(x, x, w->rms_final_weight, dim, p->norm_eps));

  // classifier into logits
  PROFILE(PHASE_CLASSIFIER, k_matmul(s->logits, x, w->wcls, p->dim, p->vocab_size));
  return s->logits;
}

/* CLI adapter: each executable links exactly one numerical implementation. */
void engine_load(Model *model, const char *path) {
  engine_load_context(model, path, 0);
}
void engine_load_context(Model *model, const char *path, int context) {
  Transformer *t = memory_alloc(1, sizeof(*t), MEM_METADATA);
  t->fd = -1;
  model->implementation = t;
  kernels_init();
  read_checkpoint(path, &t->config, &t->weights, &t->manifest, &t->fd, &t->data, &t->file_size);
  model->config = t->config;
  model->mapped_bytes = t->file_size;
  if (context < 0 || context > t->config.seq_len)
    fail("requested context exceeds checkpoint context");
  t->config.seq_len = context ? context : (t->config.seq_len < 4096 ? t->config.seq_len : 4096);
  model->config = t->config;
  double start = monotonic_seconds();
  malloc_run_state(&t->state, &t->config);
  metrics.runtime_s = monotonic_seconds() - start;
  metrics.runtime_ready = 1;
}
float *engine_forward(Model *model, int token, int position) {
  if (token < 0 || token >= model->config.vocab_size || position < 0 || position >= model->config.seq_len)
    fail("forward input out of range: token=%d position=%d", token, position);
  return forward(model->implementation, token, position);
}
void engine_free(Model *model) {
  if (model->implementation) {
    free_transformer(model->implementation);
    memory_free(model->implementation);
    model->implementation = NULL;
  }
}
const char *engine_precision(void) {
  return "fp32";
}
int engine_group_size(void) {
  return 0;
}
