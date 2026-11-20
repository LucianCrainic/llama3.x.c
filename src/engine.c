/* LLaMA 3 CPU inference, derived from Andrej Karpathy and James W. DeLancey. See LICENSE.
 *
 * The one-socket engine for both checkpoint formats (F3 step 1). run links it with engine_q8 = 0 (FP32 checkpoints,
 * optional BF16 prefill on packed weights); runq with engine_q8 = 1 (Q8_0 checkpoints, W8A8 or W8A32 activations). The
 * precision shows up in four places only: the embedding row, the per-token projections (GEMV), the batched projections
 * (GEMM) and the weight packing. Each keeps the kernel calls, their order and their profile phases of the former
 * run.c / runq.c, so results are bitwise those of the two former engines. Two sockets run in tp.c. */
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

/* One model per CLI process. */
static int GS;               /* Q8: the checkpoint's group size; 0 for FP32 */
static int fp32_activations; /* Q8: LLAMA_Q8_ACTIVATIONS=fp32 multiplies dequantized weights by FP32 activations (W8A32)
                                instead of quantizing activations (the default W8A8 Q8_0 rule) */
static int bf16_mode;        /* FP32: LLAMA_PRECISION=bf16 (C5) */

/* One weight matrix: FP32 values (f), or Q8_0 values (q) and their group scales (s). */
typedef struct {
  const float *f;
  const int8_t *q;
  const float *s;
} Matrix;

typedef struct {
  Matrix embed;                             // (vocab_size, dim); Q8 rows are dequantized on demand
  const float *rms_att, *rms_ffn;           // (layer, dim) rmsnorm weights
  Matrix *wq, *wk, *wv, *wo, *w1, *w2, *w3; // (layer); wq (dim, dim), wk/wv (kv_dim, dim), w1/w3 (hidden, dim), w2 (dim, hidden)
  const float *rms_final;                   // (dim,)
  Matrix wcls;                              // (vocab_size, dim); the embedding when the checkpoint ties them
} TransformerWeights;

typedef struct {
  int8_t *q; // quantized values
  float *s;  // per-row group scales
} QuantizedRows;

typedef struct {
  // Row-major activations for up to M prompt tokens; decode uses row zero.
  float *x;                // (M, dim)
  float *xb;               // residual branch (M, dim)
  float *hb;               // FFN (M, hidden_dim)
  QuantizedRows xq, hq;    // Q8: x and hb quantized (M rows padded to 32, per-row scales)
  float *q;                // queries (M, dim)
  float *k, *v;            // this position's key and value: its cache rows, or the stage below
  float *kstage, *vstage;  // LLAMA_KV_LAYOUT=head: K/V rows of the projections (M, kv_dim), stored into the cache per head
  float *att;              // buffer for scores/attention values (n_heads, seq_len)
  float *logits;           // output logits
  float *rope;             // RoPE tables (M, 2 * head_size)
  float *batch_logits;     // lazily allocated M x vocab diagnostic output
  AttentionWork attention; // split-K attention partials
  // kv cache
  float *key_cache;   // (layer, seq_len, kv_dim); cache persists for this sequence
  float *value_cache; // (layer, seq_len, kv_dim); cache persists for this sequence
} RunState;

/* One layer's matrices packed for the batched GEMM: BF16 VNNI pairs (FP32 checkpoint, LLAMA_PRECISION=bf16, C5) or
 * int8 VNNI quads for AMX-INT8 (Q8 checkpoint, C6; the group scales stay where the checkpoint has them). */
typedef struct {
  const void *wq, *wk, *wv, *wo, *w1, *w2, *w3;
} PackedLayer;

typedef struct {
  Config config;              // the hyperparameters of the architecture (the blueprint)
  TransformerWeights weights; // the weights of the model
  TensorManifest manifest;    // where each tensor lives in the file
  WeightStore store;          // where those bytes live now: the mapping, or owned regions (A4)
  RunMemory mem;              // where the run state and KV caches live (A5)
  RunState state;             // buffers for the "wave" of activations in the forward pass
  Tp *tp;                     // LLAMA_SOCKETS=2: the two-socket engine, which owns weights, KV and activations (B5, B6)
  PackedLayer *packed;        // (layer) NULL unless BF16 mode or AMX-INT8 prefill
  PackedStore packed_store, packed_classifier;
  const uint16_t *packed_wcls; // BF16 mode: packed lazily, by the first LOGITS_ALL request
  AlignedBuffer panel;         // BF16 mode: the aligned panel the activations are converted into
  int bf16, state_batch;       // BF16 mode; the largest batch the panel is sized for
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
  if (engine_q8) {
    /* The quantized activation rows are padded to whole 32-row blocks: the AMX kernel loads 16 rows at a time. */
    int padded = (max_batch + 31) / 32 * 32;
    s->xq = (QuantizedRows){.q = runmem_alloc(rm, checked_mul(padded, p->dim), sizeof(int8_t)),
                            .s = runmem_alloc(rm, checked_mul(padded, p->dim / GS), sizeof(float))};
    s->hq = (QuantizedRows){.q = runmem_alloc(rm, checked_mul(padded, p->hidden_dim), sizeof(int8_t)),
                            .s = runmem_alloc(rm, checked_mul(padded, p->hidden_dim / GS), sizeof(float))};
  }
  s->q = runmem_alloc(rm, checked_mul(max_batch, p->dim), sizeof(float));
  s->att = runmem_alloc(rm, checked_mul((size_t)p->n_heads, p->seq_len), sizeof(float));
  s->logits = runmem_alloc(rm, p->vocab_size, sizeof(float));
  s->rope = runmem_alloc(rm, checked_mul(max_batch, 2 * (size_t)p->head_dim), sizeof(float));
  if (kernels_kv_layout_head()) {
    size_t kv_dim = (size_t)p->head_dim * p->n_kv_heads;
    s->kstage = runmem_alloc(rm, checked_mul(max_batch, kv_dim), sizeof(float));
    s->vstage = runmem_alloc(rm, checked_mul(max_batch, kv_dim), sizeof(float));
  }
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
  runmem_free(rm, s->att);
  runmem_free(rm, s->logits);
  runmem_free(rm, s->batch_logits);
  runmem_free(rm, s->kstage);
  runmem_free(rm, s->vstage);
  runmem_free(rm, s->key_cache);
  runmem_free(rm, s->value_cache);
  runmem_free(rm, s->rope);
  k_attention_free(&s->attention);
  runmem_release(rm);
}

// ----------------------------------------------------------------------------
// Weights: views into the store at manifest offsets

static const TensorDesc *find_tensor(const TensorManifest *m, TensorRole role, int layer) {
  const TensorDesc *t = manifest_find(m, role, layer);
  if (!t)
    fail("tensor manifest has no %s tensor (layer %d)", manifest_role_name(role), layer);
  return t;
}

static Matrix matrix(const WeightStore *store, const TensorManifest *m, TensorRole role, int layer) {
  const TensorDesc *t = find_tensor(m, role, layer);
  if (!engine_q8)
    return (Matrix){.f = (const float *)weights_at(store, t->offset)};
  return (Matrix){.q = (const int8_t *)weights_at(store, t->offset),
                  .s = (const float *)weights_at(store, t->scale_offset)};
}

static Matrix *layer_matrices(const WeightStore *store, const TensorManifest *m, TensorRole role, int layers) {
  Matrix *res = memory_alloc(layers, sizeof(Matrix), MEM_METADATA);
  for (int l = 0; l < layers; l++)
    res[l] = matrix(store, m, role, l);
  return res;
}

/* Norm weights are FP32 in both formats. Layers of a role are contiguous, so layer 0 addresses the whole array. */
static const float *norm_view(const WeightStore *store, const TensorManifest *m, TensorRole role, int layer) {
  return (const float *)weights_at(store, find_tensor(m, role, layer)->offset);
}

static void memory_map_weights(TransformerWeights *w, const Config *p, const TensorManifest *m,
                               const WeightStore *data) {
  w->embed = matrix(data, m, T_EMBED, -1);
  w->rms_att = norm_view(data, m, T_RMS_ATT, 0);
  w->rms_ffn = norm_view(data, m, T_RMS_FFN, 0);
  w->rms_final = norm_view(data, m, T_RMS_FINAL, -1);
  w->wq = layer_matrices(data, m, T_WQ, p->n_layers);
  w->wk = layer_matrices(data, m, T_WK, p->n_layers);
  w->wv = layer_matrices(data, m, T_WV, p->n_layers);
  w->wo = layer_matrices(data, m, T_WO, p->n_layers);
  w->w1 = layer_matrices(data, m, T_W1, p->n_layers);
  w->w2 = layer_matrices(data, m, T_W2, p->n_layers);
  w->w3 = layer_matrices(data, m, T_W3, p->n_layers);
  w->wcls = matrix(data, m, T_WCLS, -1); /* a tied classifier's descriptor is a copy of the embedding's */
}

static void read_checkpoint(const char *path, Config *config, TransformerWeights *weights, TensorManifest *manifest,
                            WeightStore *store, int *fd, float **data, size_t *file_size) {
  int shared, gs;
  double start = monotonic_seconds();
  checkpoint_map(path, engine_q8, config, &shared, &gs, fd, data, file_size, manifest);
  GS = engine_q8 ? gs : 0;
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
  packed_close(&t->packed_classifier);
  if (t->panel.base)
    aligned_buffer_free(&t->panel);
  TransformerWeights *w = &t->weights;
  memory_free(w->wq);
  memory_free(w->wk);
  memory_free(w->wv);
  memory_free(w->wo);
  memory_free(w->w1);
  memory_free(w->w2);
  memory_free(w->w3);
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

/* ---- weight packing for the batched GEMM ------------------------------------------------------------------------- */

/* Packs every layer's seven matrices into one bound region, on the worker team, after the per-socket budget check: BF16
 * VNNI pairs in BF16 mode, int8 VNNI quads for AMX-INT8 (whose group scales are read in place). */
static void pack_weights(Transformer *t) {
  const Config *p = &t->config;
  const TransformerWeights *w = &t->weights;
  double start = monotonic_seconds();
  size_t dim = (size_t)p->dim, kv = (size_t)p->head_dim * p->n_kv_heads, hidden = (size_t)p->hidden_dim;
  const size_t rows[7] = {dim, kv, kv, dim, hidden, dim, hidden}, cols[7] = {dim, dim, dim, dim, dim, hidden, dim};
  size_t bytes[7], per_layer = 0;
  for (int i = 0; i < 7; i++) {
    bytes[i] = engine_q8 ? s8_vnni_bytes((int)rows[i], (int)cols[i])
                         : bf16_vnni_elements((int)rows[i], (int)cols[i]) * sizeof(uint16_t);
    per_layer += bytes[i];
  }
  packed_open(&t->packed_store, checked_mul(per_layer, (size_t)p->n_layers),
              engine_q8 ? "the packed int8 weights" : "the packed BF16 weights");
  t->packed = memory_alloc((size_t)p->n_layers, sizeof(PackedLayer), MEM_METADATA);
  size_t jobs = checked_mul((size_t)p->n_layers, 7);
  PackS8 *s8 = engine_q8 ? memory_alloc(jobs, sizeof(PackS8), MEM_METADATA) : NULL;
  PackBf16 *bf16 = engine_q8 ? NULL : memory_alloc(jobs, sizeof(PackBf16), MEM_METADATA);
  for (size_t l = 0; l < (size_t)p->n_layers; l++) {
    const Matrix *src[7] = {&w->wq[l], &w->wk[l], &w->wv[l], &w->wo[l], &w->w1[l], &w->w2[l], &w->w3[l]};
    void *dst[7];
    for (int i = 0; i < 7; i++) {
      dst[i] = packed_take(&t->packed_store, bytes[i]);
      if (engine_q8)
        s8[l * 7 + i] = (PackS8){dst[i], src[i]->q, (int)rows[i], (int)cols[i], cols[i]};
      else
        bf16[l * 7 + i] = (PackBf16){dst[i], src[i]->f, (int)rows[i], (int)cols[i], cols[i]};
    }
    t->packed[l] = (PackedLayer){dst[0], dst[1], dst[2], dst[3], dst[4], dst[5], dst[6]};
  }
  if (engine_q8)
    pack_s8_vnni_many(s8, (int)jobs);
  else
    pack_bf16_vnni_many(bf16, (int)jobs);
  memory_free(s8);
  memory_free(bf16);
  packed_seal(&t->packed_store);
  if (!engine_q8) {
    size_t max_k = dim > hidden ? dim : hidden;
    aligned_buffer_alloc(&t->panel, k_gemm_bf16_panel_bytes(t->state_batch, (int)max_k), MEM_ACTIVATION);
  }
  metrics.weight_pack_s += monotonic_seconds() - start;
  metrics.weight_packed = 1;
  LOG(LOG_INFO, "packed %zu bytes of %s in %.2f s", t->packed_store.region.logical_bytes,
      engine_q8 ? "int8 weights for AMX-INT8" : "BF16 weights", monotonic_seconds() - start);
}

/* BF16 mode: the classifier is packed on the first request for all-position logits; the default last-row classifier
 * stays FP32. */
static void pack_classifier(Transformer *t) {
  const Config *p = &t->config;
  double start = monotonic_seconds();
  packed_open(&t->packed_classifier, bf16_vnni_elements(p->vocab_size, p->dim) * sizeof(uint16_t),
              "the packed BF16 classifier");
  uint16_t *dst = packed_take(&t->packed_classifier, bf16_vnni_elements(p->vocab_size, p->dim) * sizeof(uint16_t));
  PackBf16 job = {dst, t->weights.wcls.f, p->vocab_size, p->dim, (size_t)p->dim};
  pack_bf16_vnni_many(&job, 1);
  packed_seal(&t->packed_classifier);
  t->packed_wcls = dst;
  metrics.weight_pack_s += monotonic_seconds() - start;
  LOG(LOG_INFO, "packed %zu bytes of BF16 classifier in %.2f s", t->packed_classifier.region.logical_bytes,
      monotonic_seconds() - start);
}

/* ---- projections: the only precision-specific arithmetic of the forward pass ------------------------------------- */

/* Q8: elements [first, first + n) of a matrix as FP32, value times its group scale. */
static void dequantize_row(const Matrix *m, float *x, size_t first, size_t n) {
  for (size_t i = 0; i < n; i++)
    x[i] = m->q[first + i] * m->s[(first + i) / GS];
}

static K8Weight k8(const Matrix *m) {
  return (K8Weight){m->q, m->s};
}

/* Q8 activations for one per-token matmul: quantized (W8A8), or the FP32 vector itself (W8A32). */
static K8Input prepare_input(QuantizedRows *qx, const float *x, int n) {
  if (fp32_activations)
    return (K8Input){NULL, NULL, x};
  k_quantize(qx->q, qx->s, x, n, GS);
  return (K8Input){qx->q, qx->s, x};
}

/* Per token: q, k, v = Wq x, Wk x, Wv x in one parallel region. */
static void project_qkv(RunState *s, size_t l, const TransformerWeights *w, const Config *p) {
  int dim = p->dim, kv_dim = p->head_dim * p->n_kv_heads;
  if (!engine_q8) {
    PROFILE(PHASE_QKV, k_matmul_qkv(s->q, s->k, s->v, s->xb, w->wq[l].f, w->wk[l].f, w->wv[l].f, dim, dim, kv_dim));
    return;
  }
  K8Input in;
  PROFILE(PHASE_QUANTIZE, in = prepare_input(&s->xq, s->xb, dim));
  PROFILE(PHASE_QKV, k_matmul_q8_qkv(s->q, s->k, s->v, &in, k8(&w->wq[l]), k8(&w->wk[l]), k8(&w->wv[l]), dim, dim,
                                     kv_dim, GS));
}

/* Per token: out += W x (n inputs, d outputs), the residual projections wo and w2. */
static void project_add(Phase phase, QuantizedRows *qx, float *out, const float *x, const Matrix *w, int n, int d) {
  if (!engine_q8) {
    PROFILE(phase, k_matmul_add(out, x, w->f, n, d));
    return;
  }
  K8Input in;
  PROFILE(PHASE_QUANTIZE, in = prepare_input(qx, x, n));
  PROFILE(phase, k_matmul_q8_add(out, &in, k8(w), n, d, GS));
}

/* Per token: out = silu(W1 x) * (W3 x) in one pass. */
static void project_swiglu(RunState *s, const Matrix *w1, const Matrix *w3, int n, int d) {
  if (!engine_q8) {
    PROFILE(PHASE_W1_W3, k_matmul_swiglu(s->hb, s->xb, w1->f, w3->f, n, d));
    return;
  }
  K8Input in;
  PROFILE(PHASE_QUANTIZE, in = prepare_input(&s->xq, s->xb, n));
  PROFILE(PHASE_W1_W3, k_matmul_q8_swiglu(s->hb, &in, k8(w1), k8(w3), n, d, GS));
}

/* Per token: logits = Wcls x. */
static void project_classifier(RunState *s, const Matrix *wcls, int n, int d) {
  if (!engine_q8) {
    PROFILE(PHASE_CLASSIFIER, k_matmul(s->logits, s->x, wcls->f, n, d));
    return;
  }
  K8Input in;
  PROFILE(PHASE_QUANTIZE, in = prepare_input(&s->xq, s->x, n));
  PROFILE(PHASE_CLASSIFIER, k_matmul_q8(s->logits, &in, k8(wcls), n, d, GS));
}

/* One output segment of a batched projection: n outputs per row into c, from w (and w3 for GEMM_SWIGLU), or from their
 * packed copies pw / pw3 where the GEMM uses them. */
typedef struct {
  float *c;
  int ldc;
  const Matrix *w, *w3;
  const void *pw, *pw3;
  int n;
  GemmKind kind;
} Projection;

/* Batched: `count` rows of `a` (k columns, row stride k) through up to GEMM_MAX_SEGMENTS projections, each weight
 * matrix streamed once for all rows. FP32: the FP32 GEMM, or the BF16 GEMM on the packed copies. Q8 (W8A8): the rows are
 * quantized into qx exactly as the per-token path quantizes them, then the Q8 GEMM (AMX-INT8 on the packed copies). */
static void project_rows(Transformer *t, Phase phase, QuantizedRows *qx, const float *a, int count, int k,
                         const Projection *pr, int segments) {
  if (engine_q8) {
    GemmQ8Segment seg[GEMM_MAX_SEGMENTS];
    for (int i = 0; i < segments; i++)
      seg[i] = (GemmQ8Segment){.c = pr[i].c,
                               .ldc = pr[i].ldc,
                               .w = k8(pr[i].w),
                               .w3 = pr[i].w3 ? k8(pr[i].w3) : (K8Weight){0},
                               .pw = pr[i].pw,
                               .pw3 = pr[i].pw3,
                               .n = pr[i].n,
                               .kind = pr[i].kind};
    PROFILE(PHASE_QUANTIZE, k_quantize_rows(qx->q, qx->s, a, (size_t)k, count, k, GS));
    PROFILE(phase, k_gemm_q8(qx->q, qx->s, count, k, GS, seg, segments));
  } else if (t->bf16) {
    GemmBf16Segment seg[GEMM_MAX_SEGMENTS];
    for (int i = 0; i < segments; i++)
      seg[i] = (GemmBf16Segment){
          .c = pr[i].c, .ldc = pr[i].ldc, .w = pr[i].pw, .w3 = pr[i].pw3, .n = pr[i].n, .kind = pr[i].kind};
    PROFILE(phase, k_gemm_bf16(a, k, count, k, seg, segments, &t->panel));
  } else {
    GemmSegment seg[GEMM_MAX_SEGMENTS];
    for (int i = 0; i < segments; i++)
      seg[i] = (GemmSegment){.c = pr[i].c,
                             .ldc = pr[i].ldc,
                             .w = pr[i].w->f,
                             .w3 = pr[i].w3 ? pr[i].w3->f : NULL,
                             .ldw = (size_t)k,
                             .n = pr[i].n,
                             .kind = pr[i].kind};
    PROFILE(phase, k_gemm_f32(a, k, count, k, seg, segments));
  }
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
  if (engine_q8) {
    row.xq.q += (size_t)i * p->dim;
    row.xq.s += (size_t)i * (p->dim / GS);
    row.hq.q += (size_t)i * p->hidden_dim;
    row.hq.s += (size_t)i * (p->hidden_dim / GS);
  }
  return row;
}

static void forward_embedding(Transformer *t, RunState *s, int token, int pos) {
  Config *p = &t->config;
  const Matrix *e = &t->weights.embed;
  size_t first = (size_t)token * p->dim;
  if (engine_q8) /* the row is dequantized on demand; no FP32 copy of the table exists */
    PROFILE(PHASE_EMBED, dequantize_row(e, s->x, first, p->dim));
  else
    PROFILE(PHASE_EMBED, memcpy(s->x, e->f + first, p->dim * sizeof(float)));
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
  PROFILE(PHASE_RMSNORM, k_rmsnorm(s->xb, x, w->rms_att + l * dim, dim, p->norm_eps));

  // key and value point to the kv cache
  size_t loff = l * p->seq_len * kv_dim; // kv cache layer offset for convenience
  int head_layout =
      kernels_kv_layout_head(); // LLAMA_KV_LAYOUT=head: the projections go to a row and are stored per head
  s->k = head_layout ? s->kstage : s->key_cache + loff + (size_t)pos * kv_dim;
  s->v = head_layout ? s->vstage : s->value_cache + loff + (size_t)pos * kv_dim;

  // qkv matmuls for this position, in one parallel region
  project_qkv(s, l, w, p);

  PROFILE(PHASE_ROPE, k_rope_apply(s->q, s->k, s->rope, p->n_heads, p->n_kv_heads, head_size));
  if (head_layout)
    PROFILE(PHASE_KV_STORE,
            k_kv_store(s->key_cache + loff, s->k, (size_t)kv_dim, pos, 1, p->n_kv_heads, head_size, p->seq_len);
            k_kv_store(s->value_cache + loff, s->v, (size_t)kv_dim, pos, 1, p->n_kv_heads, head_size, p->seq_len));

  // multihead attention over positions 0..pos, output into xb
  PROFILE(PHASE_ATTENTION, k_attention_kv(s->xb, s->q, s->key_cache + loff, s->value_cache + loff, s->att,
                                          &s->attention, p->n_heads, p->n_kv_heads, head_size, p->seq_len, pos));

  // final matmul to get the output of the attention, with the residual connection back into x
  project_add(PHASE_WO, &s->xq, x, s->xb, &w->wo[l], dim, dim);

  // ffn rmsnorm
  PROFILE(PHASE_RMSNORM, k_rmsnorm(s->xb, x, w->rms_ffn + l * dim, dim, p->norm_eps));

  // Now for FFN in PyTorch we have: self.w2(F.silu(self.w1(x)) * self.w3(x))
  // w1 and w3 share one pass that applies the SwiGLU non-linearity
  project_swiglu(s, &w->w1[l], &w->w3[l], dim, hidden_dim);

  // final matmul to get the output of the ffn, with the residual connection
  project_add(PHASE_W2, &s->hq, x, s->hb, &w->w2[l], hidden_dim, dim);
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
  /* rows: the projections are written straight into the cache rows. head: into K/V row buffers, stored per head after
   * RoPE. */
  int head_layout = kernels_kv_layout_head();
  float *kd = head_layout ? s->kstage : kc + (size_t)pos0 * kv_dim,
        *vd = head_layout ? s->vstage : vc + (size_t)pos0 * kv_dim;
  static const PackedLayer none;
  const PackedLayer *pl = t->packed ? &t->packed[l] : &none;

  PROFILE(PHASE_RMSNORM, k_rmsnorm_rows(s->xb, s->x, w->rms_att + l * dim, count, dim, p->norm_eps));
  Projection qkv[3] = {{.c = s->q, .ldc = dim, .w = &w->wq[l], .pw = pl->wq, .n = dim, .kind = GEMM_STORE},
                       {.c = kd, .ldc = kv_dim, .w = &w->wk[l], .pw = pl->wk, .n = kv_dim, .kind = GEMM_STORE},
                       {.c = vd, .ldc = kv_dim, .w = &w->wv[l], .pw = pl->wv, .n = kv_dim, .kind = GEMM_STORE}};
  project_rows(t, PHASE_QKV, &s->xq, s->xb, count, dim, qkv, 3);
  PROFILE(PHASE_ROPE, k_rope_apply_rows(s->q, (size_t)dim, kd, (size_t)kv_dim, s->rope, (size_t)2 * head_size, count,
                                        p->n_heads, p->n_kv_heads, head_size));
  if (head_layout)
    PROFILE(PHASE_KV_STORE, k_kv_store(kc, kd, (size_t)kv_dim, pos0, count, p->n_kv_heads, head_size, p->seq_len);
            k_kv_store(vc, vd, (size_t)kv_dim, pos0, count, p->n_kv_heads, head_size, p->seq_len));
  PROFILE(PHASE_ATTENTION, k_attention_batch_kv(s->xb, s->q, kc, vc, s->att, &s->attention, p->n_heads, p->n_kv_heads,
                                                head_size, p->seq_len, pos0, count));
  Projection wo = {.c = s->x, .ldc = dim, .w = &w->wo[l], .pw = pl->wo, .n = dim, .kind = GEMM_ADD};
  project_rows(t, PHASE_WO, &s->xq, s->xb, count, dim, &wo, 1);

  PROFILE(PHASE_RMSNORM, k_rmsnorm_rows(s->xb, s->x, w->rms_ffn + l * dim, count, dim, p->norm_eps));
  Projection up = {.c = s->hb,
                   .ldc = hidden_dim,
                   .w = &w->w1[l],
                   .w3 = &w->w3[l],
                   .pw = pl->w1,
                   .pw3 = pl->w3,
                   .n = hidden_dim,
                   .kind = GEMM_SWIGLU};
  project_rows(t, PHASE_W1_W3, &s->xq, s->xb, count, dim, &up, 1);
  Projection down = {.c = s->x, .ldc = dim, .w = &w->w2[l], .pw = pl->w2, .n = dim, .kind = GEMM_ADD};
  project_rows(t, PHASE_W2, &s->hq, s->hb, count, hidden_dim, &down, 1);
}

static float *forward_logits(Transformer *t, RunState *s) {
  Config *p = &t->config;
  PROFILE(PHASE_RMSNORM, k_rmsnorm(s->x, s->x, t->weights.rms_final, p->dim, p->norm_eps));
  // classifier into logits
  project_classifier(s, &t->weights.wcls, p->dim, p->vocab_size);
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

static float *forward_batch(Model *model, const int *tokens, int count, int pos0, LogitsWanted want) {
  prefill_validate(model, tokens, count, pos0, want);
  Transformer *t = model->implementation;
  Config *p = &t->config;
  RunState *s = &t->state;
  if (t->tp && model->prefill_gemm && (count > 1 || model->prefill_bf16))
    return tp_forward_batch(t->tp, tokens, count, pos0, want);
  if (t->tp) { /* a single row, W8A32, or the GEMM switched off: per token on two sockets */
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
  /* A row reads only its own activations and this layer's preceding KV rows. Layer-major execution is therefore bitwise
   * equivalent to token-major. With the batched GEMM, FP32 results agree within the GEMM's rounding and Q8 (W8A8 only)
   * within the Q8 budget, bitwise on the scalar instruction set. In BF16 mode even a single-row batch runs the BF16
   * GEMM, so a prompt's precision does not depend on how it was split. */
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
    PROFILE(PHASE_RMSNORM, k_rmsnorm_rows(s->x, s->x, t->weights.rms_final, count, p->dim, p->norm_eps));
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

/* ---- load: the prefill policy of each format (D2 §7) ------------------------------------------------------------- */

/* FP32: 128 rows per pass on one socket, 64 on two, 256 for the AMX-BF16 rung (LLAMA_PRECISION=bf16). */
static void configure_fp32(Model *model, Transformer *t) {
  model->prefill_bf16 = prefill_precision(0);
  prefill_configure(model, model->prefill_bf16 ? 256 : tp_sockets_requested() == 2 ? 64 : 128);
  bf16_mode = t->bf16 = model->prefill_bf16;
  model->prefill_kernel = t->bf16 ? (kernels_amx_active() ? "amx-bf16" : "bf16-reference") : kernels_name();
}

/* Q8: W8A8 prefill takes 256 rows per pass, on AMX-INT8 where it can run (C6); W8A32 never batches, it stays on the
 * per-row kernel. The rungs below AMX: the VNNI Q8 GEMM, and the portable one when the group size is not a multiple of
 * 64. */
static void configure_q8(Model *model) {
  prefill_configure(model, fp32_activations ? 1 : 256);
  if (!fp32_activations && model->prefill_gemm && GS % 64 == 0)
    kernels_select_amx_auto();
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
}

/* CLI adapter: the executable picks the checkpoint format (engine_q8). */
void engine_load(Model *model, const char *path) {
  engine_load_context(model, path, 0);
}
void engine_load_context(Model *model, const char *path, int context) {
  Transformer *t = memory_alloc(1, sizeof(*t), MEM_METADATA);
  t->fd = -1;
  model->implementation = t;
  kernels_init();
  kernels_backend_init();
  if (engine_q8) {
    prefill_precision(1); /* LLAMA_PRECISION=bf16 is an error here */
    const char *activations = getenv("LLAMA_Q8_ACTIVATIONS");
    if (activations && *activations && strcmp(activations, "int8") && strcmp(activations, "fp32"))
      fail("LLAMA_Q8_ACTIVATIONS must be int8 or fp32");
    fp32_activations = activations && !strcmp(activations, "fp32");
  }
  read_checkpoint(path, &t->config, &t->weights, &t->manifest, &t->store, &t->fd, &t->data, &t->file_size);
  model->config = t->config;
  model->file_bytes = t->file_size;
  model->mapped_bytes = t->data == MAP_FAILED ? 0 : t->file_size;
  if (context < 0 || context > t->config.seq_len)
    fail("requested context exceeds checkpoint context");
  t->config.seq_len = context ? context : (t->config.seq_len < 4096 ? t->config.seq_len : 4096);
  model->config = t->config;
  double start = monotonic_seconds();
  if (engine_q8)
    configure_q8(model);
  else
    configure_fp32(model, t);
  /* The packed copies the batched GEMM reads: BF16 (FP32 checkpoint, BF16 mode) or int8 for AMX-INT8. */
  int amx_int8 = engine_q8 && model->prefill_gemm && model->prefill_kernel && !strcmp(model->prefill_kernel, "amx-int8");
  if (tp_sockets_requested() == 2) {
    /* D1: with the GEMM on, microbatches run through the batched two-socket forward, else per token. In BF16 mode every
     * microbatch does, even a single row, so a prompt's precision does not depend on how it was split. */
    t->tp = tp_create(&t->config, &t->manifest, t->fd, engine_q8, GS, fp32_activations,
                      model->prefill_gemm ? model->max_batch : 1,
                      t->bf16 ? TP_PACK_BF16 : amx_int8 ? TP_PACK_INT8 : TP_PACK_NONE);
    munmap(t->data, t->file_size); /* the shards were read with pread */
    t->data = MAP_FAILED;
    model->mapped_bytes = 0;
  } else {
    malloc_run_state(&t->state, &t->config, model->max_batch, &t->mem);
    if (t->bf16 || amx_int8) {
      t->state_batch = model->max_batch;
      pack_weights(t);
    }
  }
  metrics.runtime_s = monotonic_seconds() - start;
  metrics.runtime_ready = 1;
}
void engine_kv_stats(Model *model, int position, int verify, KvStats *out) {
  Transformer *t = model->implementation;
  if (t->tp) {
    tp_kv_stats(t->tp, position, verify, out);
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
  float *logits = t->tp ? tp_forward(t->tp, token, position) : forward(t, token, position);
  park_if_failing();
  return logits;
}
float *engine_forward_batch(Model *model, const int *tokens, int count, int pos0, LogitsWanted want) {
  float *logits = forward_batch(model, tokens, count, pos0, want);
  park_if_failing();
  return logits;
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
  if (engine_q8)
    return fp32_activations ? "int8_q8_0_w8a32" : "int8_q8_0";
  return bf16_mode ? "bf16_from_fp32" : "fp32";
}
int engine_group_size(void) {
  return GS;
}
