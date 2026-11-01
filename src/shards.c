#include "shards.h"

typedef struct {
  TensorManifest *m;
  size_t cursor;
  int gs;
} Builder;

static const char *const role_names[T_ROLE_COUNT] = {
    "embed", "rms_att", "wq", "wk", "wv", "wo", "rms_ffn", "w1", "w2", "w3", "rms_final", "rope_legacy", "wcls"};

const char *manifest_role_name(TensorRole role) {
  return role >= 0 && role < T_ROLE_COUNT ? role_names[role] : "?";
}

static int add_tensor(Builder *b, TensorRole role, int layer, int rows, int cols, TensorDtype dtype) {
  TensorDesc *t = &b->m->t[b->m->count];
  size_t count = checked_mul((size_t)rows, (size_t)cols);
  *t = (TensorDesc){.role = role,
                    .layer = layer,
                    .rows = rows,
                    .cols = cols,
                    .dtype = dtype,
                    .group_size = dtype == DT_Q8 ? b->gs : 0,
                    .offset = b->cursor,
                    .bytes = tensor_bytes(count, dtype == DT_Q8, b->gs),
                    .alias_of = -1};
  if (dtype == DT_Q8)
    t->scale_offset = checked_add(t->offset, count);
  b->cursor = checked_add(b->cursor, t->bytes);
  return b->m->count++;
}

static void add_alias(Builder *b, TensorRole role, int target) {
  TensorDesc *t = &b->m->t[b->m->count];
  *t = b->m->t[target];
  t->role = role;
  t->alias_of = target;
  b->m->count++;
}

/* Layers of one role are contiguous: the loaders address layer l as layer 0 plus
 * l tensor sizes. */
static void add_layers(Builder *b, TensorRole role, int rows, int cols, TensorDtype dtype, int layers) {
  for (int l = 0; l < layers; l++)
    add_tensor(b, role, l, rows, cols, dtype);
}

/* FP32 order: embedding, rms_att[L], wq, wk, wv, wo, rms_ffn[L], w1, w2, w3,
 * rms_final, legacy RoPE tables (format 0 only), classifier unless tied. */
static void build_f32(Builder *b, const Config *p, int kv) {
  int layers = p->n_layers, d = p->dim, h = p->hidden_dim;
  int embed = add_tensor(b, T_EMBED, -1, p->vocab_size, d, DT_F32);
  add_layers(b, T_RMS_ATT, 1, d, DT_F32, layers);
  add_layers(b, T_WQ, d, d, DT_F32, layers);
  add_layers(b, T_WK, kv, d, DT_F32, layers);
  add_layers(b, T_WV, kv, d, DT_F32, layers);
  add_layers(b, T_WO, d, d, DT_F32, layers);
  add_layers(b, T_RMS_FFN, 1, d, DT_F32, layers);
  add_layers(b, T_W1, h, d, DT_F32, layers);
  add_layers(b, T_W2, d, h, DT_F32, layers);
  add_layers(b, T_W3, h, d, DT_F32, layers);
  add_tensor(b, T_RMS_FINAL, -1, 1, d, DT_F32);
  if (p->format == 0)
    add_tensor(b, T_ROPE_LEGACY, -1, p->seq_len, p->head_dim, DT_F32);
  if (p->shared)
    add_alias(b, T_WCLS, embed);
  else
    add_tensor(b, T_WCLS, -1, p->vocab_size, d, DT_F32);
}

/* Q8 order: FP32 norms first (rms_att[L], rms_ffn[L], rms_final), then each
 * quantized tensor as values followed by its scales, tokens before layers. */
static void build_q8(Builder *b, const Config *p, int kv) {
  int layers = p->n_layers, d = p->dim, h = p->hidden_dim;
  add_layers(b, T_RMS_ATT, 1, d, DT_F32, layers);
  add_layers(b, T_RMS_FFN, 1, d, DT_F32, layers);
  add_tensor(b, T_RMS_FINAL, -1, 1, d, DT_F32);
  int embed = add_tensor(b, T_EMBED, -1, p->vocab_size, d, DT_Q8);
  add_layers(b, T_WQ, d, d, DT_Q8, layers);
  add_layers(b, T_WK, kv, d, DT_Q8, layers);
  add_layers(b, T_WV, kv, d, DT_Q8, layers);
  add_layers(b, T_WO, d, d, DT_Q8, layers);
  add_layers(b, T_W1, h, d, DT_Q8, layers);
  add_layers(b, T_W2, d, h, DT_Q8, layers);
  add_layers(b, T_W3, h, d, DT_Q8, layers);
  if (p->shared)
    add_alias(b, T_WCLS, embed);
  else
    add_tensor(b, T_WCLS, -1, p->vocab_size, d, DT_Q8);
}

void manifest_build(TensorManifest *m, const Config *p, int quantized, int gs, size_t header_bytes) {
  /* 9 per-layer roles plus embedding, final norm, legacy RoPE and classifier. */
  int capacity = p->n_layers * 9 + 4;
  *m = (TensorManifest){.t = memory_alloc((size_t)capacity, sizeof(TensorDesc), MEM_METADATA)};
  Builder b = {.m = m, .cursor = header_bytes, .gs = quantized ? gs : 0};
  int kv = p->head_dim * p->n_kv_heads;
  if (quantized)
    build_q8(&b, p, kv);
  else
    build_f32(&b, p, kv);
  if (m->count > capacity)
    fail("tensor manifest overflow: %d descriptors, capacity %d", m->count, capacity);
  m->file_bytes = b.cursor;
  if (log_level >= LOG_TRACE) {
    LOG(LOG_TRACE, "tensor manifest: %d tensors, %zu bytes", m->count, m->file_bytes);
    for (int i = 0; i < m->count; i++) {
      const TensorDesc *t = &m->t[i];
      LOG(LOG_TRACE, "  %3d %-12s layer=%3d %dx%d %s gs=%d offset=%zu scales=%zu bytes=%zu alias=%d", i,
          manifest_role_name(t->role), t->layer, t->rows, t->cols, t->dtype == DT_Q8 ? "q8" : "f32", t->group_size,
          t->offset, t->scale_offset, t->bytes, t->alias_of);
    }
  }
}

const TensorDesc *manifest_find(const TensorManifest *m, TensorRole role, int layer) {
  for (int i = 0; i < m->count; i++)
    if (m->t[i].role == role && m->t[i].layer == layer)
      return &m->t[i];
  return NULL;
}

void manifest_free(TensorManifest *m) {
  memory_free(m->t);
  *m = (TensorManifest){0};
}
