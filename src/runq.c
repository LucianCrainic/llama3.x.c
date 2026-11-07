/* LLaMA 3 CPU inference, derived from Andrej Karpathy and James W. DeLancey.
 * See LICENSE. Numerical kernels intentionally remain independent per precision. */
#include "gemm.h"
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

/* The VNNI-quad copy of one layer's int8 values for AMX-INT8 (C6); the group scales stay where the checkpoint has them.
 */
typedef struct {
  const int8_t *wq, *wk, *wv, *wo, *w1, *w2, *w3;
} PackedLayer;

typedef struct {
  Config config;              // the hyperparameters of the architecture (the blueprint)
  TransformerWeights weights; // the weights of the model
  TensorManifest manifest;    // where each tensor lives in the file
  WeightStore store;          // where those bytes live now: the mapping, or owned regions (A4)
  RunMemory mem;              // where the run state and KV caches live (A5)
  RunState state;             // buffers for the "wave" of activations in the forward pass
  Tp *tp;                     // LLAMA_SOCKETS=2: the two-socket engine, which owns weights, KV and activations (B6)
  PackedLayer *packed;        // (layer) AMX-INT8 packed weights, NULL unless amx is selected for W8A8 (C6)
  PackedStore packed_store;
  // Mapping ownership: weights borrow this region until engine_free().
  int fd;           // file descriptor for memory mapping
  float *data;      // memory mapped data pointer
  size_t file_size; // size of the checkpoint file in bytes
} Transformer;

static void alloc_buffers(RunState *s, const Config *p, int max_batch, RunMemory *rm) {
  // Zeroed activations are owned for the lifetime of this model.
  int kv_dim = (p->dim / p->n_heads) * p->n_kv_heads;
  /* The quantized activation rows are padded to whole 32-row blocks: the AMX kernel loads 16 rows at a time. */
  int padded = (max_batch + 31) / 32 * 32;
  s->x = runmem_alloc(rm, checked_mul(max_batch, p->dim), sizeof(float));
  s->xb = runmem_alloc(rm, checked_mul(max_batch, p->dim), sizeof(float));
  s->hb = runmem_alloc(rm, checked_mul(max_batch, p->hidden_dim), sizeof(float));
  s->xq = (QuantizedTensor){.q = runmem_alloc(rm, checked_mul(padded, p->dim), sizeof(int8_t)),
                            .s = runmem_alloc(rm, checked_mul(padded, p->dim / GS), sizeof(float))};
  s->hq = (QuantizedTensor){.q = runmem_alloc(rm, checked_mul(padded, p->hidden_dim), sizeof(int8_t)),
                            .s = runmem_alloc(rm, checked_mul(padded, p->hidden_dim / GS), sizeof(float))};
  s->q = runmem_alloc(rm, checked_mul(max_batch, p->dim), sizeof(float));
  s->k = runmem_alloc(rm, checked_mul(max_batch, kv_dim), sizeof(float));
  s->v = runmem_alloc(rm, checked_mul(max_batch, kv_dim), sizeof(float));
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
  memory_free(t->packed);
  packed_close(&t->packed_store);
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

/* ---- batched prefill (C6) ------------------------------------------------------------------------------------------
 */

/* Packs every layer's seven int8 matrices into VNNI quads for AMX-INT8, on the worker team, in one bound region that is
 * claimed against the per-socket budget first. The checkpoint's group scales are read in place. */
static void pack_weights(Transformer *t) {
  const Config *p = &t->config;
  const TransformerWeights *w = &t->weights;
  double start = monotonic_seconds();
  size_t dim = (size_t)p->dim, kv = (size_t)p->head_dim * p->n_kv_heads, hidden = (size_t)p->hidden_dim;
  const size_t rows[7] = {dim, kv, kv, dim, hidden, dim, hidden}, cols[7] = {dim, dim, dim, dim, dim, hidden, dim};
  size_t per_layer = 0;
  for (int i = 0; i < 7; i++)
    per_layer += s8_vnni_bytes((int)rows[i], (int)cols[i]);
  packed_open(&t->packed_store, checked_mul(per_layer, (size_t)p->n_layers), "the packed int8 weights");
  t->packed = memory_alloc((size_t)p->n_layers, sizeof(PackedLayer), MEM_METADATA);
  PackS8 *jobs = memory_alloc(checked_mul((size_t)p->n_layers, 7), sizeof(PackS8), MEM_METADATA);
  for (size_t l = 0; l < (size_t)p->n_layers; l++) {
    const int8_t *src[7] = {w->wq[l].q, w->wk[l].q, w->wv[l].q, w->wo[l].q, w->w1[l].q, w->w2[l].q, w->w3[l].q};
    int8_t *dst[7];
    for (int i = 0; i < 7; i++) {
      dst[i] = packed_take(&t->packed_store, s8_vnni_bytes((int)rows[i], (int)cols[i]));
      jobs[l * 7 + i] = (PackS8){dst[i], src[i], (int)rows[i], (int)cols[i], cols[i]};
    }
    t->packed[l] = (PackedLayer){dst[0], dst[1], dst[2], dst[3], dst[4], dst[5], dst[6]};
  }
  pack_s8_vnni_many(jobs, p->n_layers * 7);
  memory_free(jobs);
  packed_seal(&t->packed_store);
  metrics.weight_pack_s += monotonic_seconds() - start;
  metrics.weight_packed = 1;
  LOG(LOG_INFO, "packed %zu bytes of int8 weights for AMX-INT8 in %.2f s", t->packed_store.region.logical_bytes,
      monotonic_seconds() - start);
}

/* One layer for `count` consecutive positions through the batched Q8 GEMM (W8A8): each row is quantized exactly as the
 * per-token path quantizes it (k_quantize per row and group), each weight matrix is streamed once for all rows, and K
 * and V rows are written straight into the cache. */
static void forward_layer_batch(Transformer *t, RunState *s, size_t l, int pos0, int count) {
  Config *p = &t->config;
  TransformerWeights *w = &t->weights;
  int dim = p->dim, kv_dim = p->head_dim * p->n_kv_heads, hidden_dim = p->hidden_dim, head_size = p->head_dim;
  size_t loff = l * p->seq_len * kv_dim;
  float *kc = s->key_cache + loff, *vc = s->value_cache + loff;
  static const PackedLayer none;
  const PackedLayer *pl = t->packed ? &t->packed[l] : &none;

  for (int i = 0; i < count; i++)
    PROFILE(PHASE_RMSNORM,
            k_rmsnorm(s->xb + (size_t)i * dim, s->x + (size_t)i * dim, w->rms_att_weight + l * dim, dim, p->norm_eps));
  PROFILE(PHASE_QUANTIZE, k_quantize_rows(s->xq.q, s->xq.s, s->xb, (size_t)dim, count, dim, GS));
  GemmQ8Segment qkv[3] = {{.c = s->q, .ldc = dim, .w = weight(w->wq + l), .pw = pl->wq, .n = dim, .kind = GEMM_STORE},
                          {.c = kc + (size_t)pos0 * kv_dim,
                           .ldc = kv_dim,
                           .w = weight(w->wk + l),
                           .pw = pl->wk,
                           .n = kv_dim,
                           .kind = GEMM_STORE},
                          {.c = vc + (size_t)pos0 * kv_dim,
                           .ldc = kv_dim,
                           .w = weight(w->wv + l),
                           .pw = pl->wv,
                           .n = kv_dim,
                           .kind = GEMM_STORE}};
  PROFILE(PHASE_QKV, k_gemm_q8(s->xq.q, s->xq.s, count, dim, GS, qkv, 3));
  for (int i = 0; i < count; i++)
    PROFILE(PHASE_ROPE, k_rope_apply(s->q + (size_t)i * dim, kc + (size_t)(pos0 + i) * kv_dim,
                                     s->rope + (size_t)i * 2 * head_size, p->n_heads, p->n_kv_heads, head_size));
  PROFILE(PHASE_ATTENTION, k_attention_batch(s->xb, s->q, kc, vc, s->att, &s->attention, p->n_heads, p->n_kv_heads,
                                             head_size, p->seq_len, pos0, count));
  PROFILE(PHASE_QUANTIZE, k_quantize_rows(s->xq.q, s->xq.s, s->xb, (size_t)dim, count, dim, GS));
  GemmQ8Segment wo = {.c = s->x, .ldc = dim, .w = weight(w->wo + l), .pw = pl->wo, .n = dim, .kind = GEMM_ADD};
  PROFILE(PHASE_WO, k_gemm_q8(s->xq.q, s->xq.s, count, dim, GS, &wo, 1));

  for (int i = 0; i < count; i++)
    PROFILE(PHASE_RMSNORM,
            k_rmsnorm(s->xb + (size_t)i * dim, s->x + (size_t)i * dim, w->rms_ffn_weight + l * dim, dim, p->norm_eps));
  PROFILE(PHASE_QUANTIZE, k_quantize_rows(s->xq.q, s->xq.s, s->xb, (size_t)dim, count, dim, GS));
  GemmQ8Segment up = {.c = s->hb,
                      .ldc = hidden_dim,
                      .w = weight(w->w1 + l),
                      .w3 = weight(w->w3 + l),
                      .pw = pl->w1,
                      .pw3 = pl->w3,
                      .n = hidden_dim,
                      .kind = GEMM_SWIGLU};
  PROFILE(PHASE_W1_W3, k_gemm_q8(s->xq.q, s->xq.s, count, dim, GS, &up, 1));
  PROFILE(PHASE_QUANTIZE, k_quantize_rows(s->hq.q, s->hq.s, s->hb, (size_t)hidden_dim, count, hidden_dim, GS));
  GemmQ8Segment down = {.c = s->x, .ldc = dim, .w = weight(w->w2 + l), .pw = pl->w2, .n = dim, .kind = GEMM_ADD};
  PROFILE(PHASE_W2, k_gemm_q8(s->hq.q, s->hq.s, count, hidden_dim, GS, &down, 1));
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
   * Layer-major execution is therefore bitwise equivalent to token-major. With the batched Q8 GEMM (W8A8 only) results
   * agree within the Q8 budget, and bitwise on the scalar instruction set. */
  int gemm = model->prefill_gemm && count > 1;
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
  prefill_precision(1); /* LLAMA_PRECISION=bf16 is an error here */
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
    model->prefill_gemm = 0; /* the two-socket path runs per token; D1 adds the batched one */
    t->tp = tp_create(&t->config, &t->manifest, t->fd, 1, GS, fp32_activations);
    munmap(t->data, t->file_size); /* the shards were read with pread */
    t->data = MAP_FAILED;
    model->mapped_bytes = 0;
  } else {
    /* C6: which batched Q8 kernel runs. W8A32 never uses the batched GEMM: it stays on the per-row AVX-512 kernel. */
    if (fp32_activations) {
      model->prefill_gemm = 0;
      if (kernels_amx_active())
        LOG(LOG_INFO, "W8A32 prefill uses avx512: the per-row kernel, never AMX-INT8");
    } else if (model->prefill_gemm) {
      if (kernels_amx_active() && GS % 64 == 0) {
        model->prefill_kernel = "amx-int8";
      } else if (kernels_avx512_active() && GS % 64 == 0) {
        model->prefill_kernel = "avx512-q8";
      } else {
        model->prefill_kernel = "portable-q8";
        if (kernels_avx512_active())
          LOG(LOG_INFO, "W8A8 batched prefill: group size %d is not a multiple of 64, using the portable Q8 GEMM", GS);
      }
      if (kernels_amx_active() && GS % 64)
        LOG(LOG_INFO, "AMX-INT8 needs a group size that is a multiple of 64 (this checkpoint has %d): falling back", GS);
    }
    malloc_run_state(&t->state, &t->config, model->max_batch, &t->mem);
    if (model->prefill_kernel && !strcmp(model->prefill_kernel, "amx-int8"))
      pack_weights(t);
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
