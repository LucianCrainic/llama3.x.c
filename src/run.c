/* LLaMA 3 CPU inference, derived from Andrej Karpathy and James W. DeLancey.
 * See LICENSE. Numerical kernels intentionally remain independent per precision. */
#include "gemm.h"
#include "kernels.h"
#include "loader.h"
#include "prefill.h"
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
  // Row-major activations for up to M prompt tokens; decode uses row zero.
  float *x;                // (M, dim)
  float *xb;               // residual branch (M, dim)
  float *hb;               // FFN (M, hidden_dim)
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

/* The BF16 VNNI copy of one layer's matrices (C5), packed at load when LLAMA_PRECISION=bf16. */
typedef struct {
  const uint16_t *wq, *wk, *wv, *wo, *w1, *w2, *w3;
} PackedLayer;

static int bf16_mode; /* One model per CLI process: LLAMA_PRECISION=bf16 (C5). */

typedef struct {
  Config config;              // the hyperparameters of the architecture (the blueprint)
  TransformerWeights weights; // the weights of the model
  TensorManifest manifest;    // where each tensor lives in the file
  WeightStore store;          // where those bytes live now: the mapping, or owned regions (A4)
  RunMemory mem;              // where the run state and KV caches live (A5)
  RunState state;             // buffers for the "wave" of activations in the forward pass
  // BF16 mode (C5): packed weights, their stores, and the aligned panel the activations are converted into.
  PackedLayer *packed; // (layer)
  PackedStore packed_store, packed_classifier;
  const uint16_t *packed_wcls; // packed lazily, by the first LOGITS_ALL request
  AlignedBuffer panel;
  int bf16, state_batch; // BF16 mode; the largest batch the panel is sized for
  // Mapping ownership: weights borrow this region until engine_free().
  int fd;           // file descriptor for memory mapping
  float *data;      // memory mapped data pointer
  size_t file_size; // size of the checkpoint file in bytes
} Transformer;

static void alloc_buffers(RunState *s, const Config *p, int max_batch, RunMemory *rm) {
  // Zeroed activations are owned for the lifetime of this model.
  s->x = runmem_alloc(rm, checked_mul(max_batch, p->dim), sizeof(float));
  s->xb = runmem_alloc(rm, checked_mul(max_batch, p->dim), sizeof(float));
  s->hb = runmem_alloc(rm, checked_mul(max_batch, p->hidden_dim), sizeof(float));
  s->q = runmem_alloc(rm, checked_mul(max_batch, p->dim), sizeof(float));
  s->att = runmem_alloc(rm, checked_mul((size_t)p->n_heads, p->seq_len), sizeof(float));
  s->logits = runmem_alloc(rm, p->vocab_size, sizeof(float));
  s->rope = runmem_alloc(rm, checked_mul(max_batch, 2 * (size_t)p->head_dim), sizeof(float));
  runmem_attention(rm, &s->attention, p->n_heads, p->n_kv_heads, p->head_dim, p->seq_len, max_batch);
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
  runmem_free(rm, s->q);
  runmem_free(rm, s->att);
  runmem_free(rm, s->logits);
  runmem_free(rm, s->batch_logits);
  runmem_free(rm, s->key_cache);
  runmem_free(rm, s->value_cache);
  runmem_free(rm, s->rope);
  k_attention_free(&s->attention);
  runmem_release(rm);
}

/* Weight pointers are views into the file mapping at manifest offsets. Layers of a
 * role are contiguous, so layer 0 addresses the whole array. */
static const float *weight_view(const WeightStore *store, const TensorManifest *m, TensorRole role, int layer) {
  const TensorDesc *t = manifest_find(m, role, layer);
  if (!t)
    fail("tensor manifest has no %s tensor (layer %d)", manifest_role_name(role), layer);
  return (const float *)weights_at(store, t->offset);
}

static void memory_map_weights(TransformerWeights *w, const TensorManifest *m, const WeightStore *data) {
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
                            WeightStore *store, int *fd, float **data, size_t *file_size) {
  int shared, gs;
  double start = monotonic_seconds();
  checkpoint_map(path, 0, config, &shared, &gs, fd, data, file_size, manifest);
  metrics.checkpoint_s = monotonic_seconds() - start;
  /* Placement time is reported on its own, not as checkpoint_load. */
  weights_open(store, *fd, data, *file_size, manifest);
  memory_map_weights(weights, manifest, store);
  metrics.checkpoint_ready = 1;
}

static void free_transformer(Transformer *t) {
  memory_free(t->packed);
  packed_close(&t->packed_store);
  packed_close(&t->packed_classifier);
  if (t->panel.base)
    aligned_buffer_free(&t->panel);
  // close the memory mapping
  if (t->data != MAP_FAILED) {
    munmap(t->data, t->file_size);
  }
  if (t->fd != -1) {
    close(t->fd);
  }
  weights_close(&t->store);
  manifest_free(&t->manifest);
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
  return row;
}

/* ---- BF16 mode (C5) ---------------------------------------------------------------------------------------------- */

/* Packs every layer's seven matrices into one bound region, on the worker team, after the per-socket budget check. */
static void pack_weights(Transformer *t) {
  const Config *p = &t->config;
  const TransformerWeights *w = &t->weights;
  double start = monotonic_seconds();
  size_t dim = (size_t)p->dim, kv = (size_t)p->head_dim * p->n_kv_heads, hidden = (size_t)p->hidden_dim;
  const size_t rows[7] = {dim, kv, kv, dim, hidden, dim, hidden}, cols[7] = {dim, dim, dim, dim, dim, hidden, dim};
  size_t per_layer = 0;
  for (int i = 0; i < 7; i++)
    per_layer += bf16_vnni_elements((int)rows[i], (int)cols[i]) * sizeof(uint16_t);
  packed_open(&t->packed_store, checked_mul(per_layer, (size_t)p->n_layers), "the packed BF16 weights");
  t->packed = memory_alloc((size_t)p->n_layers, sizeof(PackedLayer), MEM_METADATA);
  PackBf16 *jobs = memory_alloc(checked_mul((size_t)p->n_layers, 7), sizeof(PackBf16), MEM_METADATA);
  for (size_t l = 0; l < (size_t)p->n_layers; l++) {
    const float *src[7] = {w->wq + l * dim * dim,   w->wk + l * kv * dim,     w->wv + l * kv * dim,
                           w->wo + l * dim * dim,   w->w1 + l * hidden * dim, w->w2 + l * dim * hidden,
                           w->w3 + l * hidden * dim};
    uint16_t *dst[7];
    for (int i = 0; i < 7; i++) {
      dst[i] = packed_take(&t->packed_store, bf16_vnni_elements((int)rows[i], (int)cols[i]) * sizeof(uint16_t));
      jobs[l * 7 + i] = (PackBf16){dst[i], src[i], (int)rows[i], (int)cols[i], cols[i]};
    }
    t->packed[l] = (PackedLayer){dst[0], dst[1], dst[2], dst[3], dst[4], dst[5], dst[6]};
  }
  pack_bf16_vnni_many(jobs, p->n_layers * 7);
  memory_free(jobs);
  packed_seal(&t->packed_store);
  size_t max_k = dim > hidden ? dim : hidden;
  aligned_buffer_alloc(&t->panel, k_gemm_bf16_panel_bytes(t->state_batch, (int)max_k), MEM_ACTIVATION);
  metrics.weight_pack_s += monotonic_seconds() - start;
  metrics.weight_packed = 1;
  LOG(LOG_INFO, "packed %zu bytes of BF16 weights in %.2f s", t->packed_store.region.logical_bytes,
      monotonic_seconds() - start);
}

/* The classifier is packed on the first request for all-position logits; the default last-row classifier stays FP32. */
static void pack_classifier(Transformer *t) {
  const Config *p = &t->config;
  double start = monotonic_seconds();
  packed_open(&t->packed_classifier, bf16_vnni_elements(p->vocab_size, p->dim) * sizeof(uint16_t),
              "the packed BF16 classifier");
  uint16_t *dst = packed_take(&t->packed_classifier, bf16_vnni_elements(p->vocab_size, p->dim) * sizeof(uint16_t));
  PackBf16 job = {dst, t->weights.wcls, p->vocab_size, p->dim, (size_t)p->dim};
  pack_bf16_vnni_many(&job, 1);
  packed_seal(&t->packed_classifier);
  t->packed_wcls = dst;
  metrics.weight_pack_s += monotonic_seconds() - start;
  LOG(LOG_INFO, "packed %zu bytes of BF16 classifier in %.2f s", t->packed_classifier.region.logical_bytes,
      monotonic_seconds() - start);
}

/* One projection: the FP32 GEMM, or in BF16 mode the same segments on the packed weights. */
static void project(Transformer *t, const float *a, int lda, int m, int k, const GemmSegment *seg, int count,
                    const uint16_t *const *packed, const uint16_t *const *packed3) {
  if (!t->bf16) {
    k_gemm_f32(a, lda, m, k, seg, count);
    return;
  }
  GemmBf16Segment b[GEMM_MAX_SEGMENTS];
  for (int i = 0; i < count; i++)
    b[i] = (GemmBf16Segment){.c = seg[i].c,
                             .ldc = seg[i].ldc,
                             .w = packed[i],
                             .w3 = packed3 ? packed3[i] : NULL,
                             .n = seg[i].n,
                             .kind = seg[i].kind};
  k_gemm_bf16(a, lda, m, k, b, count, &t->panel);
}

static void forward_embedding(Transformer *t, RunState *s, int token, int pos) {
  Config *p = &t->config;
  TransformerWeights *w = &t->weights;
  PROFILE(PHASE_EMBED, memcpy(s->x, w->token_embedding_table + (size_t)token * p->dim, p->dim * sizeof(float)));
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

  // key and value point to the kv cache
  size_t loff = l * p->seq_len * kv_dim; // kv cache layer offset for convenience
  s->k = s->key_cache + loff + (size_t)pos * kv_dim;
  s->v = s->value_cache + loff + (size_t)pos * kv_dim;

  // qkv matmuls for this position, in one parallel region
  PROFILE(PHASE_QKV, k_matmul_qkv(s->q, s->k, s->v, s->xb, w->wq + l * dim * dim, w->wk + l * dim * kv_dim,
                                  w->wv + l * dim * kv_dim, dim, dim, kv_dim));

  PROFILE(PHASE_ROPE, k_rope_apply(s->q, s->k, s->rope, p->n_heads, p->n_kv_heads, head_size));

  // multihead attention over positions 0..pos, output into xb
  PROFILE(PHASE_ATTENTION, k_attention(s->xb, s->q, s->key_cache + loff, s->value_cache + loff, s->att, &s->attention,
                                       p->n_heads, p->n_kv_heads, head_size, p->seq_len, pos));

  // final matmul to get the output of the attention, with the residual connection back into x
  PROFILE(PHASE_WO, k_matmul_add(x, s->xb, w->wo + l * dim * dim, dim, dim));

  // ffn rmsnorm
  PROFILE(PHASE_RMSNORM, k_rmsnorm(s->xb, x, w->rms_ffn_weight + l * dim, dim, p->norm_eps));

  // Now for FFN in PyTorch we have: self.w2(F.silu(self.w1(x)) * self.w3(x))
  // w1 and w3 share one pass that applies the SwiGLU non-linearity
  PROFILE(PHASE_W1_W3,
          k_matmul_swiglu(s->hb, s->xb, w->w1 + l * dim * hidden_dim, w->w3 + l * dim * hidden_dim, dim, hidden_dim));

  // final matmul to get the output of the ffn, with the residual connection
  PROFILE(PHASE_W2, k_matmul_add(x, s->hb, w->w2 + l * dim * hidden_dim, hidden_dim, dim));
}

/* One layer for `count` consecutive positions starting at pos0 through the GEMM: each weight matrix is streamed once for
 * all rows. K and V rows are written straight into the cache, so every row sees its own and all earlier positions
 * (later rows are ignored by the position bound of k_attention). */
static void forward_layer_batch(Transformer *t, RunState *s, size_t l, int pos0, int count) {
  Config *p = &t->config;
  TransformerWeights *w = &t->weights;
  int dim = p->dim, kv_dim = p->head_dim * p->n_kv_heads, hidden_dim = p->hidden_dim, head_size = p->head_dim;
  size_t loff = l * p->seq_len * kv_dim;
  float *kc = s->key_cache + loff, *vc = s->value_cache + loff;
  static const PackedLayer none;
  const PackedLayer *pl = t->bf16 ? &t->packed[l] : &none;

  for (int i = 0; i < count; i++)
    PROFILE(PHASE_RMSNORM, k_rmsnorm(s->xb + (size_t)i * dim, s->x + (size_t)i * dim, w->rms_att_weight + l * dim, dim,
                                     p->norm_eps));
  GemmSegment qkv[3] = {
      {.c = s->q, .ldc = dim, .w = w->wq + l * dim * dim, .ldw = (size_t)dim, .n = dim, .kind = GEMM_STORE},
      {.c = kc + (size_t)pos0 * kv_dim, .ldc = kv_dim, .w = w->wk + l * dim * kv_dim, .ldw = (size_t)dim, .n = kv_dim,
       .kind = GEMM_STORE},
      {.c = vc + (size_t)pos0 * kv_dim, .ldc = kv_dim, .w = w->wv + l * dim * kv_dim, .ldw = (size_t)dim, .n = kv_dim,
       .kind = GEMM_STORE}};
  PROFILE(PHASE_QKV,
          project(t, s->xb, dim, count, dim, qkv, 3, (const uint16_t *const[]){pl->wq, pl->wk, pl->wv}, NULL));
  for (int i = 0; i < count; i++)
    PROFILE(PHASE_ROPE, k_rope_apply(s->q + (size_t)i * dim, kc + (size_t)(pos0 + i) * kv_dim,
                                     s->rope + (size_t)i * 2 * head_size, p->n_heads, p->n_kv_heads, head_size));
  PROFILE(PHASE_ATTENTION, k_attention_batch(s->xb, s->q, kc, vc, s->att, &s->attention, p->n_heads, p->n_kv_heads,
                                             head_size, p->seq_len, pos0, count));
  GemmSegment wo = {.c = s->x, .ldc = dim, .w = w->wo + l * dim * dim, .ldw = (size_t)dim, .n = dim, .kind = GEMM_ADD};
  PROFILE(PHASE_WO, project(t, s->xb, dim, count, dim, &wo, 1, &pl->wo, NULL));

  for (int i = 0; i < count; i++)
    PROFILE(PHASE_RMSNORM, k_rmsnorm(s->xb + (size_t)i * dim, s->x + (size_t)i * dim, w->rms_ffn_weight + l * dim, dim,
                                     p->norm_eps));
  GemmSegment up = {.c = s->hb, .ldc = hidden_dim, .w = w->w1 + l * dim * hidden_dim, .w3 = w->w3 + l * dim * hidden_dim,
                    .ldw = (size_t)dim, .n = hidden_dim, .kind = GEMM_SWIGLU};
  PROFILE(PHASE_W1_W3, project(t, s->xb, dim, count, dim, &up, 1, &pl->w1, &pl->w3));
  GemmSegment down = {.c = s->x, .ldc = dim, .w = w->w2 + l * dim * hidden_dim, .ldw = (size_t)hidden_dim, .n = dim,
                      .kind = GEMM_ADD};
  PROFILE(PHASE_W2, project(t, s->hb, hidden_dim, count, hidden_dim, &down, 1, &pl->w2, NULL));
}

static float *forward_logits(Transformer *t, RunState *s) {
  Config *p = &t->config;
  TransformerWeights *w = &t->weights;
  float *x = s->x;
  int dim = p->dim;
  PROFILE(PHASE_RMSNORM, k_rmsnorm(x, x, w->rms_final_weight, dim, p->norm_eps));

  // classifier into logits
  PROFILE(PHASE_CLASSIFIER, k_matmul(s->logits, x, w->wcls, p->dim, p->vocab_size));
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
  if (want == LOGITS_ALL && !s->batch_logits)
    s->batch_logits = memory_alloc(checked_mul(model->max_batch, p->vocab_size), sizeof(float), MEM_ACTIVATION);
  for (int i = 0; i < count; i++) {
    RunState row = row_state(s, p, i);
    forward_embedding(t, &row, tokens[i], pos0 + i);
  }
  /* A row reads only its own activations and this layer's preceding KV rows.
   * Layer-major execution is therefore bitwise equivalent to token-major. */
  /* In BF16 mode even a single-row batch runs the BF16 GEMM, so a prompt's precision does not depend on how it was
   * split. */
  int gemm = model->prefill_gemm && (count > 1 || t->bf16);
  for (size_t l = 0; l < (size_t)p->n_layers; l++) {
    if (gemm) {
      forward_layer_batch(t, s, l, pos0, count);
      continue;
    }
    for (int i = 0; i < count; i++) {
      RunState row = row_state(s, p, i);
      forward_layer(t, &row, l, pos0 + i);
    }
  }
  if (want == LOGITS_NONE)
    return NULL;
  if (want == LOGITS_ALL && t->bf16) {
    /* All-position logits use the BF16 GEMM on a lazily packed classifier; the last-row classifier stays FP32. */
    if (!t->packed_wcls)
      pack_classifier(t);
    for (int i = 0; i < count; i++)
      PROFILE(PHASE_RMSNORM, k_rmsnorm(s->x + (size_t)i * p->dim, s->x + (size_t)i * p->dim,
                                       t->weights.rms_final_weight, p->dim, p->norm_eps));
    GemmBf16Segment cls = {
        .c = s->batch_logits, .ldc = p->vocab_size, .w = t->packed_wcls, .n = p->vocab_size, .kind = GEMM_STORE};
    PROFILE(PHASE_CLASSIFIER, k_gemm_bf16(s->x, p->dim, count, p->dim, &cls, 1, &t->panel));
    return s->batch_logits;
  }
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
  read_checkpoint(path, &t->config, &t->weights, &t->manifest, &t->store, &t->fd, &t->data, &t->file_size);
  model->config = t->config;
  model->file_bytes = t->file_size;
  model->mapped_bytes = t->data == MAP_FAILED ? 0 : t->file_size;
  if (context < 0 || context > t->config.seq_len)
    fail("requested context exceeds checkpoint context");
  t->config.seq_len = context ? context : (t->config.seq_len < 4096 ? t->config.seq_len : 4096);
  model->config = t->config;
  double start = monotonic_seconds();
  model->prefill_bf16 = prefill_precision(0);
  prefill_configure(model);
  malloc_run_state(&t->state, &t->config, model->max_batch, &t->mem);
  bf16_mode = t->bf16 = model->prefill_bf16;
  if (t->bf16) {
    t->state_batch = model->max_batch;
    pack_weights(t);
  }
  metrics.runtime_s = monotonic_seconds() - start;
  metrics.runtime_ready = 1;
}
void engine_kv_stats(Model *model, int position, int verify, KvStats *out) {
  Transformer *t = model->implementation;
  size_t kv_dim = (size_t)t->config.head_dim * t->config.n_kv_heads;
  size_t active = checked_mul(checked_mul(checked_mul(2, (size_t)t->config.n_layers), (size_t)position), kv_dim * sizeof(float));
  runmem_kv_stats(&t->mem, active, verify, out);
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
    kernels_backend_free();
  }
}
const char *engine_precision(void) {
  return bf16_mode ? "bf16_from_fp32" : "fp32";
}
int engine_group_size(void) {
  return 0;
}
