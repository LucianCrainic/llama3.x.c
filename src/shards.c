#include "shards.h"

#include "loader.h"
#include "workers.h"
#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

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

/* ---- Shards (B1) ------------------------------------------------------------------------------------------------ */

enum { SCRATCH_DEFAULT = 64 << 20, SHARD_ALIGN = 64 };

static void split_range(int total, int sockets, int socket, int *begin, int *count) {
  int each = total / sockets;
  *begin = socket * each;
  *count = socket == sockets - 1 ? total - *begin : each;
}

static size_t align_up(size_t value) {
  return checked_add(value, SHARD_ALIGN - 1) & ~(size_t)(SHARD_ALIGN - 1);
}

static size_t value_bytes(const Shard *sh) {
  return checked_mul(checked_mul((size_t)sh->rows, (size_t)sh->cols), sh->src->dtype == DT_Q8 ? 1 : sizeof(float));
}
static size_t scale_bytes(const Shard *sh) {
  return sh->src->dtype == DT_Q8
             ? checked_mul(checked_mul((size_t)sh->rows, (size_t)(sh->cols / sh->src->group_size)), sizeof(float))
             : 0;
}

static int reject(char *err, size_t err_size, const char *format, ...) __attribute__((format(printf, 3, 4)));
static int reject(char *err, size_t err_size, const char *format, ...) {
  va_list args;
  va_start(args, format);
  vsnprintf(err, err_size, format, args);
  va_end(args);
  return -1;
}

/* Appends a shard of `t` for `socket` and gives it a place in the area (socket region or global) it belongs to. */
static void add_shard(ShardSet *set, const TensorDesc *t, ShardKind kind, int socket, int row0, int rows, int col0,
                      int cols, int alias) {
  Shard *sh = &set->t[set->count++];
  *sh = (Shard){.src = t,
                .kind = kind,
                .socket = socket,
                .row0 = row0,
                .rows = rows,
                .col0 = col0,
                .cols = cols,
                .alias = alias,
                .ld = (size_t)cols};
  size_t *cursor = kind == SHARD_GLOBAL ? &set->global_bytes : &set->region_bytes[socket];
  sh->q_offset = align_up(*cursor);
  *cursor = sh->q_offset + value_bytes(sh);
  if (t->dtype == DT_Q8) {
    sh->s_offset = align_up(*cursor);
    *cursor = sh->s_offset + scale_bytes(sh);
  }
}

int shards_plan_try(ShardSet *set, const TensorManifest *m, const Config *p, int sockets, int gs, char *err,
                    size_t err_size) {
  *set = (ShardSet){.sockets = sockets, .scratch_limit = SCRATCH_DEFAULT};
  const TensorDesc *probe = manifest_find(m, T_WQ, 0);
  if (!probe)
    return reject(err, err_size, "the manifest has no attention weights");
  set->quantized = probe->dtype == DT_Q8;
  set->gs = set->quantized ? probe->group_size : 0;
  if (set->quantized && gs && gs != probe->group_size)
    return reject(err, err_size, "group size %d does not match the checkpoint's %d", gs, probe->group_size);
  if (sockets < 1 || sockets > 2)
    return reject(err, err_size, "%d sockets: only 1 or 2 are supported", sockets);
  if (p->n_kv_heads % sockets)
    return reject(err, err_size,
                  "n_kv_heads=%d is not divisible by %d sockets: whole KV heads cannot be owned by a socket",
                  p->n_kv_heads, sockets);
  if (p->n_heads % sockets || (p->n_heads / sockets) % (p->n_kv_heads / sockets))
    return reject(err, err_size, "n_heads=%d cannot be split over %d sockets in complete GQA groups", p->n_heads,
                  sockets);
  if (p->hidden_dim % sockets)
    return reject(err, err_size, "hidden_dim=%d is not divisible by %d sockets", p->hidden_dim, sockets);
  if (p->dim % sockets || p->vocab_size < sockets)
    return reject(err, err_size, "dim=%d / vocab=%d cannot be split over %d sockets", p->dim, p->vocab_size, sockets);
  if (set->quantized) {
    if ((p->dim / sockets) % set->gs || (p->hidden_dim / sockets) % set->gs)
      return reject(err, err_size,
                    "group size %d does not divide the per-socket slices (dim/%d = %d, hidden/%d = %d); values are "
                    "never re-quantized",
                    set->gs, sockets, p->dim / sockets, sockets, p->hidden_dim / sockets);
  }
  /* 9 per-layer roles on every socket, plus final norm, classifier and one global embedding. */
  int capacity = sockets * (p->n_layers * 9 + 2) + 1;
  set->t = memory_alloc((size_t)capacity, sizeof(Shard), MEM_METADATA);
  static const TensorRole row_roles[] = {T_WQ, T_WK, T_WV, T_W1, T_W3};
  static const TensorRole col_roles[] = {T_WO, T_W2};
  static const TensorRole norm_roles[] = {T_RMS_ATT, T_RMS_FFN};
  for (int s = 0; s < sockets; s++) {
    int begin, count;
    for (int l = 0; l < p->n_layers; l++) {
      for (size_t r = 0; r < sizeof(norm_roles) / sizeof(*norm_roles); r++) {
        const TensorDesc *t = manifest_find(m, norm_roles[r], l);
        add_shard(set, t, SHARD_COPY, s, 0, t->rows, 0, t->cols, 0);
      }
      for (size_t r = 0; r < sizeof(row_roles) / sizeof(*row_roles); r++) {
        const TensorDesc *t = manifest_find(m, row_roles[r], l);
        split_range(t->rows, sockets, s, &begin, &count);
        add_shard(set, t, SHARD_ROWS, s, begin, count, 0, t->cols, 0);
      }
      for (size_t r = 0; r < sizeof(col_roles) / sizeof(*col_roles); r++) {
        const TensorDesc *t = manifest_find(m, col_roles[r], l);
        split_range(t->cols, sockets, s, &begin, &count);
        add_shard(set, t, SHARD_COLS, s, 0, t->rows, begin, count, 0);
      }
    }
    const TensorDesc *final = manifest_find(m, T_RMS_FINAL, -1), *cls = manifest_find(m, T_WCLS, -1);
    add_shard(set, final, SHARD_COPY, s, 0, final->rows, 0, final->cols, 0);
    split_range(cls->rows, sockets, s, &begin, &count);
    add_shard(set, cls, SHARD_ROWS, s, begin, count, 0, cls->cols, cls->alias_of >= 0);
  }
  const TensorDesc *embed = manifest_find(m, T_EMBED, -1), *cls = manifest_find(m, T_WCLS, -1);
  if (cls->alias_of < 0)
    add_shard(set, embed, SHARD_GLOBAL, 0, 0, embed->rows, 0, embed->cols, 0);
  if (set->count > capacity)
    fail("shard set overflow: %d shards, capacity %d", set->count, capacity);
  return 0;
}

void shards_plan(ShardSet *set, const TensorManifest *m, const Config *p, int sockets, int gs) {
  char err[512];
  if (shards_plan_try(set, m, p, sockets, gs, err, sizeof(err)))
    fail("%s", err);
}

void shards_free(ShardSet *set) {
  memory_free(set->t);
  *set = (ShardSet){0};
}

const Shard *shards_find(const ShardSet *set, TensorRole role, int layer, int socket) {
  for (int i = 0; i < set->count; i++) {
    const Shard *sh = &set->t[i];
    if (sh->src->role == role && sh->src->layer == layer && (sh->socket == socket || sh->kind == SHARD_GLOBAL))
      return sh;
  }
  return NULL;
}

void shards_bind(ShardSet *set, void *const socket_base[2], void *global_base) {
  for (int i = 0; i < set->count; i++) {
    Shard *sh = &set->t[i];
    char *base = sh->kind == SHARD_GLOBAL ? global_base : socket_base[sh->socket];
    if (!base || (uintptr_t)base % SHARD_ALIGN)
      fail("shard area is missing or not %d-byte aligned", SHARD_ALIGN);
    sh->q = base + sh->q_offset;
    sh->s = sh->src->dtype == DT_Q8 ? (float *)(base + sh->s_offset) : NULL;
  }
}

/* One contiguous read: file range to destination. */
typedef struct {
  size_t src;
  char *dst;
  size_t bytes;
} Piece;

typedef struct {
  int fd;
  const Piece *pieces;
  int count;
  size_t total;
  int *status; /* per worker: 0, an errno, or -1 for an unexpected end of file */
} RowsJob;

/* Each worker takes one slice of the pieces' bytes laid end to end, as the owned loader does. */
static void rows_slice(const void *job_, int worker, int workers) {
  const RowsJob *job = job_;
  size_t lo = (size_t)((unsigned long long)job->total * (unsigned)worker / (unsigned)workers);
  size_t hi = (size_t)((unsigned long long)job->total * (unsigned)(worker + 1) / (unsigned)workers);
  size_t base = 0;
  for (int i = 0; i < job->count && base < hi && !job->status[worker]; i++) {
    const Piece *pc = &job->pieces[i];
    size_t end = base + pc->bytes;
    if (end > lo) {
      size_t from = lo > base ? lo - base : 0, to = hi < end ? hi - base : pc->bytes;
      job->status[worker] = loader_read_exact(job->fd, pc->dst + from, to - from, pc->src + from, (size_t)1 << 30);
    }
    base = end;
  }
}

typedef struct {
  int fd;
  const Shard *shard;
  Team *team;
  char *scratch;
  int block_rows;
  size_t row_value_bytes, row_scale_bytes;
  int *status;
} ColsJob;

/* Every worker reads and packs its own rows of each block, so a block needs one barrier: the scratch is reused by the
 * next block. */
static void cols_pack(const void *job_, int worker, int workers) {
  const ColsJob *job = job_;
  const Shard *sh = job->shard;
  const TensorDesc *t = sh->src;
  int q8 = t->dtype == DT_Q8;
  size_t elem = q8 ? 1 : sizeof(float);
  int gs = t->group_size;
  char *scratch_v = job->scratch;
  char *scratch_s = job->scratch + (size_t)job->block_rows * job->row_value_bytes;
  for (int r0 = 0; r0 < sh->rows; r0 += job->block_rows) {
    int rows = sh->rows - r0 < job->block_rows ? sh->rows - r0 : job->block_rows;
    int a = (int)((long long)rows * worker / workers), b = (int)((long long)rows * (worker + 1) / workers);
    if (b > a && !job->status[worker]) {
      size_t src_row = (size_t)(sh->row0 + r0 + a);
      int status = loader_read_exact(job->fd, scratch_v + (size_t)a * job->row_value_bytes,
                                     (size_t)(b - a) * job->row_value_bytes, t->offset + src_row * job->row_value_bytes,
                                     (size_t)1 << 30);
      if (!status && q8)
        status = loader_read_exact(job->fd, scratch_s + (size_t)a * job->row_scale_bytes,
                                   (size_t)(b - a) * job->row_scale_bytes,
                                   t->scale_offset + src_row * job->row_scale_bytes, (size_t)1 << 30);
      job->status[worker] = status;
      for (int r = a; !status && r < b; r++) {
        size_t dst_row = (size_t)(r0 + r);
        memcpy((char *)sh->q + dst_row * sh->ld * elem,
               scratch_v + (size_t)r * job->row_value_bytes + (size_t)sh->col0 * elem, (size_t)sh->cols * elem);
        if (q8)
          memcpy(sh->s + dst_row * (size_t)(sh->cols / gs),
                 (const char *)(scratch_s + (size_t)r * job->row_scale_bytes) + (size_t)(sh->col0 / gs) * sizeof(float),
                 (size_t)(sh->cols / gs) * sizeof(float));
      }
    }
    if (job->team)
      team_barrier(job->team);
  }
}

static void check_status(const int *status, int workers, const char *what) {
  for (int w = 0; w < workers; w++)
    if (status[w]) {
      if (status[w] < 0)
        fail("unexpected end of file while loading %s", what);
      fail("cannot read the checkpoint for %s: %s", what, strerror(status[w]));
    }
}

void shards_load(ShardSet *set, int fd, int socket, Team *team) {
  int workers = team ? team_size(team) : 1;
  /* shards_load runs on the coordinator of one socket, possibly while the other socket's coordinator loads too, so it
   * avoids the engine's global accounting and uses plain allocations. */
  int *status = calloc((size_t)workers, sizeof(int));

  /* Contiguous shards first: one read of the values and one of the scales each, split among the workers. */
  Piece *pieces = calloc((size_t)set->count * 2, sizeof(Piece));
  int count = 0;
  size_t total = 0;
  double start = monotonic_seconds();
  for (int i = 0; i < set->count; i++) {
    const Shard *sh = &set->t[i];
    int mine = sh->kind == SHARD_GLOBAL ? socket == 0 : sh->socket == socket;
    if (!mine || sh->kind == SHARD_COLS)
      continue;
    const TensorDesc *t = sh->src;
    size_t elem = t->dtype == DT_Q8 ? 1 : sizeof(float);
    size_t row_values = checked_mul((size_t)t->cols, elem);
    pieces[count++] = (Piece){t->offset + (size_t)sh->row0 * row_values, sh->q, value_bytes(sh)};
    total += value_bytes(sh);
    if (t->dtype == DT_Q8) {
      size_t row_scales = checked_mul((size_t)(t->cols / t->group_size), sizeof(float));
      pieces[count++] = (Piece){t->scale_offset + (size_t)sh->row0 * row_scales, (char *)sh->s, scale_bytes(sh)};
      total += scale_bytes(sh);
    }
  }
  RowsJob rows = {.fd = fd, .pieces = pieces, .count = count, .total = total, .status = status};
  if (team)
    team_run(team, rows_slice, &rows);
  else
    rows_slice(&rows, 0, 1);
  check_status(status, workers, "the row shards");
  set->rows_seconds[socket] += monotonic_seconds() - start;
  free(pieces);

  /* Column shards through bounded scratch. */
  start = monotonic_seconds();
  for (int i = 0; i < set->count; i++) {
    const Shard *sh = &set->t[i];
    if (sh->kind != SHARD_COLS || sh->socket != socket)
      continue;
    const TensorDesc *t = sh->src;
    size_t elem = t->dtype == DT_Q8 ? 1 : sizeof(float);
    ColsJob job = {.fd = fd, .shard = sh, .team = team, .status = status};
    job.row_value_bytes = checked_mul((size_t)t->cols, elem);
    job.row_scale_bytes = t->dtype == DT_Q8 ? checked_mul((size_t)(t->cols / t->group_size), sizeof(float)) : 0;
    size_t row_bytes = job.row_value_bytes + job.row_scale_bytes;
    size_t block = set->scratch_limit / row_bytes;
    job.block_rows = block<1 ? 1 : block>(size_t) sh->rows ? sh->rows : (int)block;
    size_t scratch_bytes = (size_t)job.block_rows * row_bytes;
    if (scratch_bytes > set->scratch_peak[socket])
      set->scratch_peak[socket] = scratch_bytes;
    job.scratch = malloc(scratch_bytes);
    if (!status || !job.scratch)
      fail("cannot allocate %zu bytes of shard load scratch", scratch_bytes);
    memset(status, 0, (size_t)workers * sizeof(int));
    if (team)
      team_run(team, cols_pack, &job);
    else
      cols_pack(&job, 0, 1);
    check_status(status, workers, "a column shard");
    free(job.scratch);
  }
  set->cols_seconds[socket] += monotonic_seconds() - start;
  free(status);
}
