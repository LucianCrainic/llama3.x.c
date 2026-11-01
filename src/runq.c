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

static int GS; /* One model per CLI process; v2 checkpoint group size. */
/* LLAMA_Q8_ACTIVATIONS=fp32 multiplies dequantized weights by FP32 activations
 * (W8A32) instead of quantizing activations (the default W8A8 Q8_0 rule). */
static int fp32_activations;

typedef struct {
  int8_t *q; // quantized values
  float *s;  // scaling factors
} QuantizedTensor;

typedef struct {
  // token embedding table
  QuantizedTensor *q_tokens;    // (vocab_size, dim)
  float *token_embedding_table; // same, but dequantized

  // weights for rmsnorms
  float *rms_att_weight; // (layer, dim) rmsnorm weights
  float *rms_ffn_weight; // (layer, dim)
  // weights for matmuls. note dim == n_heads * head_size
  QuantizedTensor *wq; // (layer, dim, n_heads * head_size)
  QuantizedTensor *wk; // (layer, dim, n_kv_heads * head_size)
  QuantizedTensor *wv; // (layer, dim, n_kv_heads * head_size)
  QuantizedTensor *wo; // (layer, n_heads * head_size, dim)
  // weights for ffn
  QuantizedTensor *w1; // (layer, hidden_dim, dim)
  QuantizedTensor *w2; // (layer, dim, hidden_dim)
  QuantizedTensor *w3; // (layer, hidden_dim, dim)
  // final rmsnorm
  float *rms_final_weight; // (dim,)
  // (optional) classifier weights for the logits, on the last layer
  QuantizedTensor *wcls;
} TransformerWeights;

typedef struct {
  // current wave of activations
  float *x;           // activation at current time stamp (dim,)
  float *xb;          // same, but inside a residual branch (dim,)
  float *hb;          // buffer for hidden dimension in the ffn (hidden_dim,)
  QuantizedTensor xq; // quantized x (dim,)
  QuantizedTensor hq; // quantized hb (hidden_dim,)
  float *q;           // query (dim,)
  float *k;           // key (dim,)
  float *v;           // value (dim,)
  float *att;         // buffer for scores/attention values (n_heads, seq_len)
  float *logits;      // output logits
  float *rope;        // RoPE cosines and signed sines for the current position (2 * head_size,)
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
  s->xq = (QuantizedTensor){.q = memory_alloc(p->dim, sizeof(int8_t), MEM_ACTIVATION),
                            .s = memory_alloc(p->dim / GS, sizeof(float), MEM_ACTIVATION)};
  s->hq = (QuantizedTensor){.q = memory_alloc(p->hidden_dim, sizeof(int8_t), MEM_ACTIVATION),
                            .s = memory_alloc(p->hidden_dim / GS, sizeof(float), MEM_ACTIVATION)};
  s->q = memory_alloc(p->dim, sizeof(float), MEM_ACTIVATION);
  s->k = memory_alloc(kv_dim, sizeof(float), MEM_ACTIVATION);
  s->v = memory_alloc(kv_dim, sizeof(float), MEM_ACTIVATION);
  s->att = memory_alloc((size_t)p->n_heads * p->seq_len, sizeof(float), MEM_ACTIVATION);
  s->logits = memory_alloc(p->vocab_size, sizeof(float), MEM_ACTIVATION);
  s->key_cache = memory_alloc((size_t)p->n_layers * p->seq_len * kv_dim, sizeof(float), MEM_KV);
  s->value_cache = memory_alloc((size_t)p->n_layers * p->seq_len * kv_dim, sizeof(float), MEM_KV);
  s->rope = memory_alloc(2 * (size_t)p->head_dim, sizeof(float), MEM_ACTIVATION);
  k_attention_init(&s->attention, p->n_heads, p->n_kv_heads, p->head_dim, p->seq_len);
}

static void free_run_state(RunState *s) {
  memory_free(s->x);
  memory_free(s->xb);
  memory_free(s->hb);
  memory_free(s->xq.q);
  memory_free(s->xq.s);
  memory_free(s->hq.q);
  memory_free(s->hq.s);
  memory_free(s->q);
  memory_free(s->k);
  memory_free(s->v);
  memory_free(s->att);
  memory_free(s->logits);
  memory_free(s->key_cache);
  memory_free(s->value_cache);
  memory_free(s->rope);
  k_attention_free(&s->attention);
}

// ----------------------------------------------------------------------------
// Quantization functions

static void dequantize(const QuantizedTensor *qx, float *x, size_t n) {
  for (size_t i = 0; i < n; i++) {
    x[i] = qx->q[i] * qx->s[i / GS];
  }
}

/* Activations for one matmul input: quantized (W8A8) unless FP32 activations
 * were selected, in which case the FP32 vector is used directly (W8A32). */
static K8Input prepare_input(QuantizedTensor *qx, const float *x, int n) {
  if (fp32_activations)
    return (K8Input){NULL, NULL, x};
  k_quantize(qx->q, qx->s, x, n, GS);
  return (K8Input){qx->q, qx->s, x};
}

static K8Weight weight(const QuantizedTensor *t) {
  return (K8Weight){t->q, t->s};
}

static const TensorDesc *find_tensor(const TensorManifest *m, TensorRole role, int layer) {
  const TensorDesc *t = manifest_find(m, role, layer);
  if (!t)
    fail("tensor manifest has no %s tensor (layer %d)", manifest_role_name(role), layer);
  return t;
}

/* Views of quantized tensors of one role: `n` layers, or the single global tensor when n < 0. */
static QuantizedTensor *init_quantized_tensors(const char *data, const TensorManifest *m, TensorRole role, int n) {
  int count = n < 0 ? 1 : n;
  QuantizedTensor *res = memory_alloc(count, sizeof(QuantizedTensor), MEM_METADATA);
  for (int i = 0; i < count; i++) {
    const TensorDesc *t = find_tensor(m, role, n < 0 ? -1 : i);
    res[i].q = (int8_t *)(data + t->offset);
    res[i].s = (float *)(data + t->scale_offset);
  }
  return res;
}

static void memory_map_weights(TransformerWeights *w, const Config *p, const TensorManifest *m, const char *data) {
  w->rms_att_weight = (float *)(data + find_tensor(m, T_RMS_ATT, 0)->offset);
  w->rms_ffn_weight = (float *)(data + find_tensor(m, T_RMS_FFN, 0)->offset);
  w->rms_final_weight = (float *)(data + find_tensor(m, T_RMS_FINAL, -1)->offset);

  w->q_tokens = init_quantized_tensors(data, m, T_EMBED, -1);
  // dequantize token embedding table
  w->token_embedding_table = memory_alloc((size_t)p->vocab_size * p->dim, sizeof(float), MEM_EMBEDDING);
  dequantize(w->q_tokens, w->token_embedding_table, (size_t)p->vocab_size * p->dim);

  w->wq = init_quantized_tensors(data, m, T_WQ, p->n_layers);
  w->wk = init_quantized_tensors(data, m, T_WK, p->n_layers);
  w->wv = init_quantized_tensors(data, m, T_WV, p->n_layers);
  w->wo = init_quantized_tensors(data, m, T_WO, p->n_layers);

  w->w1 = init_quantized_tensors(data, m, T_W1, p->n_layers);
  w->w2 = init_quantized_tensors(data, m, T_W2, p->n_layers);
  w->w3 = init_quantized_tensors(data, m, T_W3, p->n_layers);

  w->wcls = find_tensor(m, T_WCLS, -1)->alias_of >= 0 ? w->q_tokens : init_quantized_tensors(data, m, T_WCLS, -1);
}

static void read_checkpoint(const char *path, Config *config, TransformerWeights *weights, TensorManifest *manifest,
                            int *fd, float **data, size_t *file_size) {
  int shared, gs;
  double start = monotonic_seconds();
  checkpoint_map(path, 1, config, &shared, &gs, fd, data, file_size, manifest);
  GS = gs;
  memory_map_weights(weights, config, manifest, (const char *)*data);
  metrics.checkpoint_s = monotonic_seconds() - start;
  metrics.checkpoint_ready = 1;
}

static void free_transformer(Transformer *t) {
  // free QuantizedTensors
  memory_free(t->weights.token_embedding_table);
  memory_free(t->weights.wq);
  memory_free(t->weights.wk);
  memory_free(t->weights.wv);
  memory_free(t->weights.wo);
  memory_free(t->weights.w1);
  memory_free(t->weights.w2);
  memory_free(t->weights.w3);
  if (t->weights.wcls != t->weights.q_tokens) {
    memory_free(t->weights.wcls);
  }
  memory_free(t->weights.q_tokens);
  manifest_free(&t->manifest);
  // close the memory mapping
  if (t->data != MAP_FAILED) {
    munmap(t->data, t->file_size);
  }
  if (t->fd != -1) {
    close(t->fd);
  }
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
  PROFILE(PHASE_EMBED, memcpy(x, w->token_embedding_table + (size_t)token * dim, dim * sizeof(float)));
  // RoPE angles depend only on the position: compute them once for all layers
  PROFILE(PHASE_ROPE_TABLE, k_rope_table(p, pos, s->rope));

  // forward all the layers
  for (size_t l = 0; l < (size_t)p->n_layers; l++) {

    // attention rmsnorm
    PROFILE(PHASE_RMSNORM, k_rmsnorm(s->xb, x, w->rms_att_weight + l * dim, dim, p->norm_eps));

    // qkv matmuls for this position, in one parallel region
    K8Input in;
    PROFILE(PHASE_QUANTIZE, in = prepare_input(&s->xq, s->xb, dim));
    PROFILE(PHASE_QKV, k_matmul_q8_qkv(s->q, s->k, s->v, &in, weight(w->wq + l), weight(w->wk + l), weight(w->wv + l),
                                       dim, dim, kv_dim, GS));

    PROFILE(PHASE_ROPE, k_rope_apply(s->q, s->k, s->rope, p->n_heads, p->n_kv_heads, head_size));

    // save key,value at this time step (pos) to our kv cache
    size_t loff = l * p->seq_len * kv_dim; // kv cache layer offset for convenience
    float *key_cache_row = s->key_cache + loff + (size_t)pos * kv_dim;
    float *value_cache_row = s->value_cache + loff + (size_t)pos * kv_dim;
    PROFILE(PHASE_KV_STORE, memcpy(key_cache_row, s->k, kv_dim * sizeof(*key_cache_row));
            memcpy(value_cache_row, s->v, kv_dim * sizeof(*value_cache_row)));

    // multihead attention over positions 0..pos, output into xb
    PROFILE(PHASE_ATTENTION, k_attention(s->xb, s->q, s->key_cache + loff, s->value_cache + loff, s->att,
                                         &s->attention, p->n_heads, p->n_kv_heads, head_size, p->seq_len, pos));

    // final matmul to get the output of the attention, with the residual connection back into x
    PROFILE(PHASE_QUANTIZE, in = prepare_input(&s->xq, s->xb, dim));
    PROFILE(PHASE_WO, k_matmul_q8_add(x, &in, weight(w->wo + l), dim, dim, GS));

    // ffn rmsnorm
    PROFILE(PHASE_RMSNORM, k_rmsnorm(s->xb, x, w->rms_ffn_weight + l * dim, dim, p->norm_eps));

    // Now for FFN in PyTorch we have: self.w2(F.silu(self.w1(x)) * self.w3(x))
    // w1 and w3 share one pass that applies the SwiGLU non-linearity
    PROFILE(PHASE_QUANTIZE, in = prepare_input(&s->xq, s->xb, dim));
    PROFILE(PHASE_W1_W3, k_matmul_q8_swiglu(s->hb, &in, weight(w->w1 + l), weight(w->w3 + l), dim, hidden_dim, GS));

    // final matmul to get the output of the ffn, with the residual connection
    PROFILE(PHASE_QUANTIZE, in = prepare_input(&s->hq, s->hb, hidden_dim));
    PROFILE(PHASE_W2, k_matmul_q8_add(x, &in, weight(w->w2 + l), hidden_dim, dim, GS));
  }

  // final rmsnorm
  PROFILE(PHASE_RMSNORM, k_rmsnorm(x, x, w->rms_final_weight, dim, p->norm_eps));

  // classifier into logits
  K8Input in;
  PROFILE(PHASE_QUANTIZE, in = prepare_input(&s->xq, x, dim));
  PROFILE(PHASE_CLASSIFIER, k_matmul_q8(s->logits, &in, weight(w->wcls), dim, p->vocab_size, GS));
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
  const char *activations = getenv("LLAMA_Q8_ACTIVATIONS");
  if (activations && *activations && strcmp(activations, "int8") && strcmp(activations, "fp32"))
    fail("LLAMA_Q8_ACTIVATIONS must be int8 or fp32");
  fp32_activations = activations && !strcmp(activations, "fp32");
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
  return fp32_activations ? "int8_q8_0_w8a32" : "int8_q8_0";
}
int engine_group_size(void) {
  return GS;
}
