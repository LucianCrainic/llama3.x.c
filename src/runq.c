/* LLaMA 3 CPU inference, derived from Andrej Karpathy and James W. DeLancey.
 * See LICENSE. Numerical kernels intentionally remain independent per precision. */
#include "kernels.h"
#include "loader.h"
#include "prefill.h"
#include "profile.h"
#include "runtime.h"
#include "shards.h"
#include "tp.h"
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
  QuantizedTensor *q_tokens; // (vocab_size, dim); rows are dequantized on demand

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
  // Row-major activations for up to M prompt tokens; decode uses row zero.
  float *x;                // (M, dim)
  float *xb;               // residual branch (M, dim)
  float *hb;               // FFN (M, hidden_dim)
  QuantizedTensor xq;      // quantized x (M, dim), per-row scales
  QuantizedTensor hq;      // quantized hb (M, hidden_dim), per-row scales
  float *q;                // queries (M, dim)
  float *k;                // key (dim,)
  float *v;                // value (dim,)
  float *att;              // buffer for scores/attention values (n_heads, seq_len)
  float *logits;           // output logits
  float *rope;             // RoPE tables (M, 2 * head_size)
  float *batch_logits;     // lazily allocated M x vocab diagnostic output
  AttentionWork attention; // split-K attention partials
  // kv cache
  float *key_cache;   // (layer, seq_len, kv_dim); cache persists for this sequence
  float *value_cache; // (layer, seq_len, kv_dim); cache persists for this sequence
} RunState;

typedef struct {
  Config config;              // the hyperparameters of the architecture (the blueprint)
  TransformerWeights weights; // the weights of the model
  TensorManifest manifest;    // where each tensor lives in the file
  WeightStore store;          // where those bytes live now: the mapping, or owned regions (A4)
  RunMemory mem;              // where the run state and KV caches live (A5)
  RunState state;             // buffers for the "wave" of activations in the forward pass
  Tp *tp;                     // LLAMA_SOCKETS=2: the two-socket engine, which owns weights, KV and activations (B6)
  // Mapping ownership: weights borrow this region until engine_free().
  int fd;           // file descriptor for memory mapping
  float *data;      // memory mapped data pointer
  size_t file_size; // size of the checkpoint file in bytes
} Transformer;

static void alloc_buffers(RunState *s, const Config *p, int max_batch, RunMemory *rm) {
  // Zeroed activations are owned for the lifetime of this model.
  int kv_dim = (p->dim / p->n_heads) * p->n_kv_heads;
  s->x = runmem_alloc(rm, checked_mul(max_batch, p->dim), sizeof(float));
  s->xb = runmem_alloc(rm, checked_mul(max_batch, p->dim), sizeof(float));
  s->hb = runmem_alloc(rm, checked_mul(max_batch, p->hidden_dim), sizeof(float));
  s->xq = (QuantizedTensor){.q = runmem_alloc(rm, checked_mul(max_batch, p->dim), sizeof(int8_t)),
                            .s = runmem_alloc(rm, checked_mul(max_batch, p->dim / GS), sizeof(float))};
  s->hq = (QuantizedTensor){.q = runmem_alloc(rm, checked_mul(max_batch, p->hidden_dim), sizeof(int8_t)),
                            .s = runmem_alloc(rm, checked_mul(max_batch, p->hidden_dim / GS), sizeof(float))};
  s->q = runmem_alloc(rm, checked_mul(max_batch, p->dim), sizeof(float));
  s->k = runmem_alloc(rm, checked_mul(max_batch, kv_dim), sizeof(float));
  s->v = runmem_alloc(rm, checked_mul(max_batch, kv_dim), sizeof(float));
  s->att = runmem_alloc(rm, checked_mul((size_t)p->n_heads, p->seq_len), sizeof(float));
  s->logits = runmem_alloc(rm, p->vocab_size, sizeof(float));
  s->rope = runmem_alloc(rm, checked_mul(max_batch, 2 * (size_t)p->head_dim), sizeof(float));
  runmem_attention(rm, &s->attention, p->n_heads, p->n_kv_heads, p->head_dim, p->seq_len, 1);
}

static void malloc_run_state(RunState *s, Config *p, int max_batch, RunMemory *rm) {
  int kv_dim = (p->dim / p->n_heads) * p->n_kv_heads;
  runmem_init(rm, checked_mul(checked_mul((size_t)p->n_layers, (size_t)p->seq_len), (size_t)kv_dim * sizeof(float)));
  alloc_buffers(s, p, max_batch, rm);
  if (runmem_sizing(rm)) { // owned placement: the first pass only measured; carve the work region now
    runmem_seal(rm);
    alloc_buffers(s, p, max_batch, rm);
  }
  s->key_cache = runmem_cache(rm, 0);
  s->value_cache = runmem_cache(rm, 1);
}

static void free_run_state(RunState *s, RunMemory *rm) {
  runmem_free(rm, s->x);
  runmem_free(rm, s->xb);
  runmem_free(rm, s->hb);
  runmem_free(rm, s->xq.q);
  runmem_free(rm, s->xq.s);
  runmem_free(rm, s->hq.q);
  runmem_free(rm, s->hq.s);
  runmem_free(rm, s->q);
  runmem_free(rm, s->k);
  runmem_free(rm, s->v);
  runmem_free(rm, s->att);
  runmem_free(rm, s->logits);
  runmem_free(rm, s->batch_logits);
  runmem_free(rm, s->key_cache);
  runmem_free(rm, s->value_cache);
  runmem_free(rm, s->rope);
  k_attention_free(&s->attention);
  runmem_release(rm);
}

// ----------------------------------------------------------------------------
// Quantization functions

/* Elements [first, first + n) of a tensor as FP32: value times its group scale. */
static void dequantize_row(const QuantizedTensor *qx, float *x, size_t first, size_t n) {
  for (size_t i = 0; i < n; i++)
    x[i] = qx->q[first + i] * qx->s[(first + i) / GS];
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
static QuantizedTensor *init_quantized_tensors(const WeightStore *data, const TensorManifest *m, TensorRole role, int n) {
  int count = n < 0 ? 1 : n;
  QuantizedTensor *res = memory_alloc(count, sizeof(QuantizedTensor), MEM_METADATA);
  for (int i = 0; i < count; i++) {
    const TensorDesc *t = find_tensor(m, role, n < 0 ? -1 : i);
    res[i].q = (int8_t *)weights_at(data, t->offset);
    res[i].s = (float *)weights_at(data, t->scale_offset);
  }
  return res;
}

static void memory_map_weights(TransformerWeights *w, const Config *p, const TensorManifest *m,
                               const WeightStore *data) {
  w->rms_att_weight = (float *)weights_at(data, find_tensor(m, T_RMS_ATT, 0)->offset);
  w->rms_ffn_weight = (float *)weights_at(data, find_tensor(m, T_RMS_FFN, 0)->offset);
  w->rms_final_weight = (float *)weights_at(data, find_tensor(m, T_RMS_FINAL, -1)->offset);

  w->q_tokens = init_quantized_tensors(data, m, T_EMBED, -1);

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
                            WeightStore *store, int *fd, float **data, size_t *file_size) {
  int shared, gs;
  double start = monotonic_seconds();
  checkpoint_map(path, 1, config, &shared, &gs, fd, data, file_size, manifest);
  GS = gs;
  metrics.checkpoint_s = monotonic_seconds() - start;
  /* Placement time is reported on its own, not as checkpoint_load. */
  if (tp_sockets_requested() == 2) { /* the shards are loaded by tp_create, straight from the file */
    metrics.checkpoint_ready = 1;
    return;
  }
  weights_open(store, *fd, data, *file_size, manifest);
  memory_map_weights(weights, config, manifest, store);
  metrics.checkpoint_ready = 1;
}

static void free_transformer(Transformer *t) {
  tp_free(t->tp);
  // free QuantizedTensors
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
  weights_close(&t->store);
  manifest_free(&t->manifest);
  // close the memory mapping
  if (t->data != MAP_FAILED) {
    munmap(t->data, t->file_size);
  }
  if (t->fd != -1) {
    close(t->fd);
  }
  // free the RunState buffers
  free_run_state(&t->state, &t->mem);
}

// ----------------------------------------------------------------------------
// neural net blocks; the dynamics of the Transformer

/* A row view borrows batched activations and shared serial attention scratch.
 * It owns nothing; the base RunState is the only object freed. */
static RunState row_state(RunState *s, const Config *p, int i) {
  RunState row = *s;
  row.x += (size_t)i * p->dim;
  row.xb += (size_t)i * p->dim;
  row.hb += (size_t)i * p->hidden_dim;
  row.q += (size_t)i * p->dim;
  row.rope += (size_t)i * 2 * p->head_dim;
  int kv_dim = p->head_dim * p->n_kv_heads;
  row.k += (size_t)i * kv_dim;
  row.v += (size_t)i * kv_dim;
  row.xq.q += (size_t)i * p->dim;
  row.xq.s += (size_t)i * (p->dim / GS);
  row.hq.q += (size_t)i * p->hidden_dim;
  row.hq.s += (size_t)i * (p->hidden_dim / GS);
  return row;
}

static void forward_embedding(Transformer *t, RunState *s, int token, int pos) {
  Config *p = &t->config;
  TransformerWeights *w = &t->weights;
  /* The embedding row is dequantized on demand; no FP32 copy of the table exists. */
  PROFILE(PHASE_EMBED, dequantize_row(w->q_tokens, s->x, (size_t)token * p->dim, p->dim));
  PROFILE(PHASE_ROPE_TABLE, k_rope_table(p, pos, s->rope));
}

/* Preserve the single-token kernel calls and their accumulation order. */
static void forward_layer(Transformer *t, RunState *s, size_t l, int pos) {
  Config *p = &t->config;
  TransformerWeights *w = &t->weights;
  float *x = s->x;
  int dim = p->dim;
  int kv_dim = p->head_dim * p->n_kv_heads;
  int hidden_dim = p->hidden_dim;
  int head_size = p->head_dim;

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
  PROFILE(PHASE_ATTENTION, k_attention(s->xb, s->q, s->key_cache + loff, s->value_cache + loff, s->att, &s->attention,
                                       p->n_heads, p->n_kv_heads, head_size, p->seq_len, pos));

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

static float *forward_logits(Transformer *t, RunState *s) {
  Config *p = &t->config;
  TransformerWeights *w = &t->weights;
  float *x = s->x;
  int dim = p->dim;
  PROFILE(PHASE_RMSNORM, k_rmsnorm(x, x, w->rms_final_weight, dim, p->norm_eps));

  // classifier into logits
  K8Input in;
  PROFILE(PHASE_QUANTIZE, in = prepare_input(&s->xq, x, dim));
  PROFILE(PHASE_CLASSIFIER, k_matmul_q8(s->logits, &in, weight(w->wcls), dim, p->vocab_size, GS));
  return s->logits;
}

/* Keep a token-at-a-time oracle and decode path, independent of batch ordering. */
static float *forward(Transformer *t, int token, int pos) {
  RunState *s = &t->state;
  forward_embedding(t, s, token, pos);
  for (size_t l = 0; l < (size_t)t->config.n_layers; l++)
    forward_layer(t, s, l, pos);
  return forward_logits(t, s);
}

float *engine_forward_batch(Model *model, const int *tokens, int count, int pos0, LogitsWanted want) {
  prefill_validate(model, tokens, count, pos0, want);
  Transformer *t = model->implementation;
  Config *p = &t->config;
  RunState *s = &t->state;
  if (t->tp) { /* per-token on two sockets; D1 adds the batched path */
    float *logits = NULL;
    if (want == LOGITS_ALL && !s->batch_logits)
      s->batch_logits = memory_alloc(checked_mul(model->max_batch, p->vocab_size), sizeof(float), MEM_ACTIVATION);
    for (int i = 0; i < count; i++) {
      logits = tp_forward(t->tp, tokens[i], pos0 + i);
      if (want == LOGITS_ALL)
        memcpy(s->batch_logits + (size_t)i * p->vocab_size, logits, (size_t)p->vocab_size * sizeof(float));
    }
    return want == LOGITS_NONE ? NULL : want == LOGITS_ALL ? s->batch_logits : logits;
  }
  if (want == LOGITS_ALL && !s->batch_logits)
    s->batch_logits = memory_alloc(checked_mul(model->max_batch, p->vocab_size), sizeof(float), MEM_ACTIVATION);
  for (int i = 0; i < count; i++) {
    RunState row = row_state(s, p, i);
    forward_embedding(t, &row, tokens[i], pos0 + i);
  }
  /* A row reads only its own activations and this layer's preceding KV rows.
   * Layer-major execution is therefore bitwise equivalent to token-major. */
  for (size_t l = 0; l < (size_t)p->n_layers; l++)
    for (int i = 0; i < count; i++) {
      RunState row = row_state(s, p, i);
      forward_layer(t, &row, l, pos0 + i);
    }
  if (want == LOGITS_NONE)
    return NULL;
  for (int i = want == LOGITS_ALL ? 0 : count - 1; i < count; i++) {
    RunState row = row_state(s, p, i);
    if (want == LOGITS_ALL)
      row.logits = s->batch_logits + (size_t)i * p->vocab_size;
    forward_logits(t, &row);
  }
  return want == LOGITS_ALL ? s->batch_logits : s->logits;
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
  kernels_backend_init();
  const char *activations = getenv("LLAMA_Q8_ACTIVATIONS");
  if (activations && *activations && strcmp(activations, "int8") && strcmp(activations, "fp32"))
    fail("LLAMA_Q8_ACTIVATIONS must be int8 or fp32");
  fp32_activations = activations && !strcmp(activations, "fp32");
  read_checkpoint(path, &t->config, &t->weights, &t->manifest, &t->store, &t->fd, &t->data, &t->file_size);
  model->config = t->config;
  model->file_bytes = t->file_size;
  model->mapped_bytes = t->data == MAP_FAILED ? 0 : t->file_size;
  if (context < 0 || context > t->config.seq_len)
    fail("requested context exceeds checkpoint context");
  t->config.seq_len = context ? context : (t->config.seq_len < 4096 ? t->config.seq_len : 4096);
  model->config = t->config;
  double start = monotonic_seconds();
  prefill_configure(model);
  if (tp_sockets_requested() == 2) {
    t->tp = tp_create(&t->config, &t->manifest, t->fd, 1, GS, fp32_activations);
    munmap(t->data, t->file_size); /* the shards were read with pread */
    t->data = MAP_FAILED;
    model->mapped_bytes = 0;
  } else {
    malloc_run_state(&t->state, &t->config, model->max_batch, &t->mem);
  }
  metrics.runtime_s = monotonic_seconds() - start;
  metrics.runtime_ready = 1;
}
void engine_kv_stats(Model *model, int position, int verify, KvStats *out) {
  Transformer *t = model->implementation;
  if (t->tp) {
    tp_kv_stats(t->tp, position, out);
    return;
  }
  size_t kv_dim = (size_t)t->config.head_dim * t->config.n_kv_heads;
  size_t active = checked_mul(checked_mul(checked_mul(2, (size_t)t->config.n_layers), (size_t)position), kv_dim * sizeof(float));
  runmem_kv_stats(&t->mem, active, verify, out);
}
float *engine_forward(Model *model, int token, int position) {
  if (token < 0 || token >= model->config.vocab_size || position < 0 || position >= model->config.seq_len)
    fail("forward input out of range: token=%d position=%d", token, position);
  Transformer *t = model->implementation;
  return t->tp ? tp_forward(t->tp, token, position) : forward(t, token, position);
}
void engine_free(Model *model) {
  if (model->implementation) {
    free_transformer(model->implementation);
    memory_free(model->implementation);
    model->implementation = NULL;
    kernels_backend_free();
  }
}
const char *engine_precision(void) {
  return fp32_activations ? "int8_q8_0_w8a32" : "int8_q8_0";
}
int engine_group_size(void) {
  return GS;
}
