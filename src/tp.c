#include "tp.h"

#include "kernels.h"
#include "loader.h"
#include "numa.h"
#include "profile.h"
#include "workers.h"
#include <pthread.h>
#include <sched.h>
#include <stdatomic.h>
#include <stdlib.h>
#include <string.h>

enum { SOCKETS = 2, SPIN_LIMIT = 2048 };

typedef enum { CMD_SETUP, CMD_LOAD, CMD_FORWARD, CMD_STOP } Command;

typedef struct {
  int socket;
  Team *team;
  NumaRegion weights, kv_k, kv_v, work, exchange;
  float *x, *xb, *hb, *q, *att, *rope; /* socket-local activations, in the work region */
  float *key_cache, *value_cache;      /* [layer][seq][kv_dim / 2] */
  AttentionWork attention;
  float *debug; /* LLAMA_TP_CHECK: x after every layer */
  /* Q8 (B6) */
  int8_t *xq; /* quantized activations of the current projection input */
  float *xs;
  int8_t *aq; /* quantized slice that feeds Wo / W2 */
  float *as;
} TpSocket;

struct Tp {
  Config c;
  int quantized, gs, check, fp32_activations;
  ShardSet set;
  NumaRegion global;
  TpSocket s[SOCKETS];
  Exchange *exchange;
  float *logits;
  int kv_dim_local;
  /* Coordination with the socket-1 coordinator thread. */
  pthread_t thread;
  int started;
  _Alignas(64) atomic_uint command_generation;
  _Alignas(64) atomic_uint done_generation;
  unsigned issued;
  Command command;
  int token, pos, loader_fd;
};

int tp_sockets_requested(void) {
  const char *value = getenv("LLAMA_SOCKETS");
  if (!value || !*value || !strcmp(value, "1"))
    return 1;
  if (strcmp(value, "2"))
    fail("LLAMA_SOCKETS must be 1 or 2");
  const char *backend = getenv("LLAMA_BACKEND");
  if (!backend || strcmp(backend, "teams"))
    fail("LLAMA_SOCKETS=2 requires LLAMA_BACKEND=teams");
  return 2;
}

int tp_check_topology(const char *placement) {
  const Topology *topology;
  char err[512];
  if (numa_topology_try(&topology, err, sizeof(err)) ||
      numa_check_placement(topology, SOCKETS, placement, err, sizeof(err)))
    fail("LLAMA_SOCKETS=2: %s", err);
  return topology->nsockets;
}

/* ---- command channel to the socket-1 coordinator ---------------------------------------------------------------- */

static void relax(void) {
#if defined(__x86_64__) || defined(__i386__)
  __builtin_ia32_pause();
#elif defined(__aarch64__)
  __asm__ __volatile__("yield");
#endif
}

static void socket_command(Tp *tp, int socket);

static void *coordinator_main(void *p) {
  Tp *tp = p;
  kernels_set_thread_team(tp->s[1].team);
  unsigned seen = 0;
  for (;;) {
    unsigned now;
    for (unsigned spins = 0; (now = atomic_load_explicit(&tp->command_generation, memory_order_acquire)) == seen;
         spins++) {
      if (workers_aborted())
        return NULL;
      if (spins < SPIN_LIMIT)
        relax();
      else
        sched_yield();
    }
    seen = now;
    if (tp->command == CMD_STOP)
      return NULL;
    socket_command(tp, 1);
    atomic_store_explicit(&tp->done_generation, now, memory_order_release);
  }
}

/* Runs `command` on both sockets at once: socket 1 on its coordinator, socket 0 on the calling thread. */
static void run_both(Tp *tp, Command command) {
  tp->command = command;
  unsigned generation = ++tp->issued;
  atomic_store_explicit(&tp->command_generation, generation, memory_order_release);
  if (command == CMD_STOP)
    return;
  socket_command(tp, 0);
  for (unsigned spins = 0; atomic_load_explicit(&tp->done_generation, memory_order_acquire) != generation; spins++) {
    if (workers_aborted())
      return;
    if (spins < SPIN_LIMIT)
      relax();
    else
      sched_yield();
  }
}

/* ---- per-socket work -------------------------------------------------------------------------------------------- */

typedef struct {
  Tp *tp;
  int socket, elements;
  float *residual;
} ReduceJob;

static void reduce_fn(const void *p, int worker, int workers) {
  const ReduceJob *job = p;
  exchange_reduce_add(job->tp->exchange, job->socket, worker, workers, job->residual, job->elements);
}

static void reduce(Tp *tp, TpSocket *sk, float *residual, int elements) {
  ReduceJob job = {tp, sk->socket, elements, residual};
  team_run(sk->team, reduce_fn, &job);
}

static const float *norm_weight(const Tp *tp, TensorRole role, int layer, int socket) {
  const Shard *sh = shards_find(&tp->set, role, layer, socket);
  if (!sh)
    fail("missing norm shard %s layer %d", manifest_role_name(role), layer);
  return sh->q;
}

/* The shard and row that hold `token`'s embedding (tied: the owner's vocabulary shard; untied: the global shard). */
static const Shard *embedding_shard(const Tp *tp, int token, int *row) {
  const Shard *cls = shards_find(&tp->set, T_WCLS, -1, 0);
  if (cls->alias) {
    for (int s = 0; s < SOCKETS; s++) {
      const Shard *sh = shards_find(&tp->set, T_WCLS, -1, s);
      if (token >= sh->row0 && token < sh->row0 + sh->rows) {
        *row = token - sh->row0;
        return sh;
      }
    }
    fail("token %d is outside the vocabulary shards", token);
  }
  *row = token;
  return shards_find(&tp->set, T_EMBED, -1, 0);
}

static const float *embedding_row(const Tp *tp, int token) {
  const Shard *cls = shards_find(&tp->set, T_WCLS, -1, 0);
  if (cls->alias) { /* tied: the row lives in the vocabulary shard of its owner */
    for (int s = 0; s < SOCKETS; s++) {
      const Shard *sh = shards_find(&tp->set, T_WCLS, -1, s);
      if (token >= sh->row0 && token < sh->row0 + sh->rows)
        return (const float *)sh->q + (size_t)(token - sh->row0) * tp->c.dim;
    }
    fail("token %d is outside the vocabulary shards", token);
  }
  const Shard *embed = shards_find(&tp->set, T_EMBED, -1, 0);
  return (const float *)embed->q + (size_t)token * tp->c.dim;
}

/* The profile counters are shared, so only socket 0 times its phases; socket 1 does the same work untimed. */
#define SPROFILE(phase, ...)                                                                                           \
  do {                                                                                                                 \
    if (s == 0) {                                                                                                      \
      PROFILE(phase, __VA_ARGS__);                                                                                     \
    } else {                                                                                                           \
      __VA_ARGS__;                                                                                                     \
    }                                                                                                                  \
  } while (0)

static void forward_fp32(Tp *tp, int s) {
  TpSocket *sk = &tp->s[s];
  const Config *p = &tp->c;
  int dim = p->dim, hd = p->head_dim, hl = p->n_heads / SOCKETS, kl = p->n_kv_heads / SOCKETS, kvl = kl * hd;
  int hidden_l = p->hidden_dim / SOCKETS, token = tp->token, pos = tp->pos;
  SPROFILE(PHASE_EMBED, memcpy(sk->x, embedding_row(tp, token), (size_t)dim * sizeof(float)));
  SPROFILE(PHASE_ROPE_TABLE, k_rope_table(p, pos, sk->rope));
  for (int l = 0; l < p->n_layers; l++) {
    SPROFILE(PHASE_RMSNORM, k_rmsnorm(sk->xb, sk->x, norm_weight(tp, T_RMS_ATT, l, s), dim, p->norm_eps));
    size_t layer = (size_t)l * p->seq_len * kvl;
    float *k = sk->key_cache + layer + (size_t)pos * kvl, *v = sk->value_cache + layer + (size_t)pos * kvl;
    const float *wq = shards_find(&tp->set, T_WQ, l, s)->q, *wk = shards_find(&tp->set, T_WK, l, s)->q,
                *wv = shards_find(&tp->set, T_WV, l, s)->q;
    SPROFILE(PHASE_QKV, k_matmul_qkv(sk->q, k, v, sk->xb, wq, wk, wv, dim, dim / SOCKETS, kvl));
    SPROFILE(PHASE_ROPE, k_rope_apply(sk->q, k, sk->rope, hl, kl, hd));
    SPROFILE(PHASE_ATTENTION, k_attention(sk->xb, sk->q, sk->key_cache + layer, sk->value_cache + layer, sk->att,
                                          &sk->attention, hl, kl, hd, p->seq_len, pos));
    const Shard *wo = shards_find(&tp->set, T_WO, l, s);
    float *partial = exchange_partial(tp->exchange, s);
    if (!partial)
      return;
    SPROFILE(PHASE_WO, k_matmul_ld(partial, sk->xb, wo->q, wo->ld, dim / SOCKETS, dim, K_PARTIAL);
             reduce(tp, sk, sk->x, dim));

    SPROFILE(PHASE_RMSNORM, k_rmsnorm(sk->xb, sk->x, norm_weight(tp, T_RMS_FFN, l, s), dim, p->norm_eps));
    const float *w1 = shards_find(&tp->set, T_W1, l, s)->q, *w3 = shards_find(&tp->set, T_W3, l, s)->q;
    SPROFILE(PHASE_W1_W3, k_matmul_swiglu(sk->hb, sk->xb, w1, w3, dim, hidden_l));
    const Shard *w2 = shards_find(&tp->set, T_W2, l, s);
    partial = exchange_partial(tp->exchange, s);
    if (!partial)
      return;
    SPROFILE(PHASE_W2, k_matmul_ld(partial, sk->hb, w2->q, w2->ld, hidden_l, dim, K_PARTIAL);
             reduce(tp, sk, sk->x, dim));
    if (sk->debug)
      memcpy(sk->debug + (size_t)l * dim, sk->x, (size_t)dim * sizeof(float));
  }
  SPROFILE(PHASE_RMSNORM, k_rmsnorm(sk->x, sk->x, norm_weight(tp, T_RMS_FINAL, -1, s), dim, p->norm_eps));
  const Shard *cls = shards_find(&tp->set, T_WCLS, -1, s);
  SPROFILE(PHASE_CLASSIFIER, k_matmul(tp->logits + cls->row0, sk->x, cls->q, dim, cls->rows));
}

/* Q8_0 (B6): the same group size, scales and activation rule as runq.c. Row shards quantize the full normalized input,
 * which is identical on both sockets; Wo and W2 quantize only the owned slice, which starts on a group boundary, so
 * its groups are exactly the original groups. */
static K8Weight q8_weight(const Shard *sh) {
  return (K8Weight){(const int8_t *)sh->q, sh->s};
}

static K8Input q8_input(const Tp *tp, int8_t *q, float *scales, const float *x, int n) {
  if (tp->fp32_activations)
    return (K8Input){NULL, NULL, x};
  k_quantize(q, scales, x, n, tp->gs);
  return (K8Input){q, scales, x};
}

static void forward_q8(Tp *tp, int s) {
  TpSocket *sk = &tp->s[s];
  const Config *p = &tp->c;
  int dim = p->dim, hd = p->head_dim, gs = tp->gs, hl = p->n_heads / SOCKETS, kl = p->n_kv_heads / SOCKETS,
      kvl = kl * hd;
  int hidden_l = p->hidden_dim / SOCKETS, token = tp->token, pos = tp->pos;
  int row;
  const Shard *embed = embedding_shard(tp, token, &row);
  SPROFILE(PHASE_EMBED, for (int i = 0; i < dim; i++) sk->x[i] =
                            ((const int8_t *)embed->q)[(size_t)row * dim + i] * embed->s[((size_t)row * dim + i) / gs]);
  SPROFILE(PHASE_ROPE_TABLE, k_rope_table(p, pos, sk->rope));
  for (int l = 0; l < p->n_layers; l++) {
    SPROFILE(PHASE_RMSNORM, k_rmsnorm(sk->xb, sk->x, norm_weight(tp, T_RMS_ATT, l, s), dim, p->norm_eps));
    K8Input in;
    SPROFILE(PHASE_QUANTIZE, in = q8_input(tp, sk->xq, sk->xs, sk->xb, dim));
    size_t layer = (size_t)l * p->seq_len * kvl;
    float *k = sk->key_cache + layer + (size_t)pos * kvl, *v = sk->value_cache + layer + (size_t)pos * kvl;
    K8Weight wq = q8_weight(shards_find(&tp->set, T_WQ, l, s)), wk = q8_weight(shards_find(&tp->set, T_WK, l, s)),
             wv = q8_weight(shards_find(&tp->set, T_WV, l, s));
    SPROFILE(PHASE_QKV, k_matmul_q8_qkv(sk->q, k, v, &in, wq, wk, wv, dim, dim / SOCKETS, kvl, gs));
    SPROFILE(PHASE_ROPE, k_rope_apply(sk->q, k, sk->rope, hl, kl, hd));
    SPROFILE(PHASE_ATTENTION, k_attention(sk->xb, sk->q, sk->key_cache + layer, sk->value_cache + layer, sk->att,
                                          &sk->attention, hl, kl, hd, p->seq_len, pos));
    const Shard *wo = shards_find(&tp->set, T_WO, l, s);
    SPROFILE(PHASE_QUANTIZE, in = q8_input(tp, sk->aq, sk->as, sk->xb, dim / SOCKETS));
    float *partial = exchange_partial(tp->exchange, s);
    if (!partial)
      return;
    K8Weight wo_weight = q8_weight(wo);
    SPROFILE(PHASE_WO, k_matmul_q8_ld(partial, &in, &wo_weight, wo->ld, dim / SOCKETS, dim, gs, K_PARTIAL);
             reduce(tp, sk, sk->x, dim));

    SPROFILE(PHASE_RMSNORM, k_rmsnorm(sk->xb, sk->x, norm_weight(tp, T_RMS_FFN, l, s), dim, p->norm_eps));
    SPROFILE(PHASE_QUANTIZE, in = q8_input(tp, sk->xq, sk->xs, sk->xb, dim));
    K8Weight w1 = q8_weight(shards_find(&tp->set, T_W1, l, s)), w3 = q8_weight(shards_find(&tp->set, T_W3, l, s));
    SPROFILE(PHASE_W1_W3, k_matmul_q8_swiglu(sk->hb, &in, w1, w3, dim, hidden_l, gs));
    const Shard *w2 = shards_find(&tp->set, T_W2, l, s);
    SPROFILE(PHASE_QUANTIZE, in = q8_input(tp, sk->aq, sk->as, sk->hb, hidden_l));
    partial = exchange_partial(tp->exchange, s);
    if (!partial)
      return;
    K8Weight w2_weight = q8_weight(w2);
    SPROFILE(PHASE_W2, k_matmul_q8_ld(partial, &in, &w2_weight, w2->ld, hidden_l, dim, gs, K_PARTIAL);
             reduce(tp, sk, sk->x, dim));
    if (sk->debug)
      memcpy(sk->debug + (size_t)l * dim, sk->x, (size_t)dim * sizeof(float));
  }
  SPROFILE(PHASE_RMSNORM, k_rmsnorm(sk->x, sk->x, norm_weight(tp, T_RMS_FINAL, -1, s), dim, p->norm_eps));
  K8Input in;
  SPROFILE(PHASE_QUANTIZE, in = q8_input(tp, sk->xq, sk->xs, sk->x, dim));
  const Shard *cls = shards_find(&tp->set, T_WCLS, -1, s);
  SPROFILE(PHASE_CLASSIFIER, k_matmul_q8(tp->logits + cls->row0, &in, q8_weight(cls), dim, cls->rows, gs));
}

/* Carves the socket's activations out of its work region and sizes the attention work from its team. Runs on the
 * socket's coordinator, so the pages are first written by the socket that uses them. */
static void setup_socket(Tp *tp, int s) {
  TpSocket *sk = &tp->s[s];
  const Config *p = &tp->c;
  int hl = p->n_heads / SOCKETS, kl = p->n_kv_heads / SOCKETS;
  char *cursor = sk->work.data;
  size_t used = 0;
#define CARVE(pointer, count, type)                                                                                    \
  do {                                                                                                                 \
    used = (used + 63) / 64 * 64;                                                                                      \
    (pointer) = (type *)(cursor + used);                                                                               \
    used += (size_t)(count) * sizeof(type);                                                                            \
  } while (0)
  CARVE(sk->x, p->dim, float);
  CARVE(sk->xb, p->dim, float);
  CARVE(sk->hb, p->hidden_dim / SOCKETS, float);
  CARVE(sk->q, p->dim / SOCKETS, float);
  CARVE(sk->att, (size_t)hl * p->seq_len, float);
  CARVE(sk->rope, 2 * (size_t)p->head_dim, float);
  if (tp->quantized) {
    CARVE(sk->xq, p->dim, int8_t);
    CARVE(sk->xs, p->dim / tp->gs, float);
    int slice = (p->dim > p->hidden_dim ? p->dim : p->hidden_dim) / SOCKETS; /* attention output or FFN hidden */
    CARVE(sk->aq, slice, int8_t);
    CARVE(sk->as, slice / tp->gs, float);
  }
  float *attention_memory;
  CARVE(attention_memory, k_attention_bytes(hl, kl, p->head_dim, p->seq_len) / sizeof(float) + 1, float);
#undef CARVE
  if (used > sk->work.data_bytes)
    fail("socket %d work region holds %zu bytes, %zu needed", s, sk->work.data_bytes, used);
  k_attention_init_in(&sk->attention, hl, kl, p->head_dim, p->seq_len, attention_memory);
  sk->key_cache = sk->kv_k.data;
  sk->value_cache = sk->kv_v.data;
}

static size_t work_bytes(const Tp *tp) {
  const Config *p = &tp->c;
  int hl = p->n_heads / SOCKETS, kl = p->n_kv_heads / SOCKETS;
  size_t slice = (size_t)(p->dim > p->hidden_dim ? p->dim : p->hidden_dim) / SOCKETS;
  size_t bytes = 64 * 16; /* alignment of each buffer */
  bytes += (size_t)(2 * p->dim + p->hidden_dim / SOCKETS + p->dim / SOCKETS + 2 * p->head_dim) * sizeof(float);
  bytes += (size_t)hl * p->seq_len * sizeof(float);
  if (tp->quantized) /* xq, xs, aq, as */
    bytes += (size_t)p->dim + (size_t)p->dim / tp->gs * sizeof(float) + slice + slice / tp->gs * sizeof(float);
  bytes += k_attention_bytes(hl, kl, p->head_dim, p->seq_len) + 64;
  return bytes;
}

static void socket_command(Tp *tp, int socket) {
  switch (tp->command) {
  case CMD_SETUP:
    setup_socket(tp, socket);
    break;
  case CMD_LOAD:
    shards_load(&tp->set, tp->loader_fd, socket, tp->s[socket].team);
    break;
  case CMD_FORWARD:
    if (tp->quantized)
      forward_q8(tp, socket);
    else
      forward_fp32(tp, socket);
    break;
  case CMD_STOP:
    break;
  }
}

/* ---- creation, forward, teardown --------------------------------------------------------------------------------- */

static int placement_node(const char *mode, const SocketTopo *socket, int embedding) {
  if (!strcmp(mode, "file"))
    return -1;
  return !strcmp(mode, "hbm") && !embedding ? socket->hbm_node : socket->ddr_node;
}

static void verify_bound(NumaRegion *r, const char *mode) {
  if (r->node < 0)
    return;
  PageCheck check = {0};
  if (region_verify(r, region_verify_samples_default(), &check) != 0) {
    if (placement_strict(mode))
      fail("page placement of the %s region on node %d cannot be verified on this system", region_role_name(r->role),
           r->node);
    return;
  }
  size_t thp = 0;
  if (region_thp_bytes(r, &thp) != 0)
    thp = 0;
  region_record_verification(r, &check, thp);
  if ((check.wrong_node || check.missing || check.errors) && placement_strict(mode))
    fail("%s region of socket %d: %zu of %zu queried pages are not on node %d", region_role_name(r->role), r->socket,
         check.queried - check.on_node, check.queried, r->node);
}

Tp *tp_create(const Config *p, const TensorManifest *m, int fd, int quantized, int gs, int fp32_activations) {
  Tp *tp = memory_alloc(1, sizeof(*tp), MEM_METADATA);
  tp->c = *p;
  tp->quantized = quantized;
  tp->fp32_activations = fp32_activations;
  tp->gs = quantized ? gs : 0;
  tp->loader_fd = fd;
  const char *check = getenv("LLAMA_TP_CHECK");
  if (check && *check && strcmp(check, "0") && strcmp(check, "1"))
    fail("LLAMA_TP_CHECK must be 0 or 1");
  tp->check = check && !strcmp(check, "1");
  shards_plan(&tp->set, m, p, SOCKETS, gs);
  const char *mode = placement_mode();
  tp_check_topology(mode);
  const Topology *topology = numa_topology();
  int kvl = p->head_dim * p->n_kv_heads / SOCKETS;
  tp->kv_dim_local = kvl;
  size_t kv_bytes = checked_mul(checked_mul((size_t)p->n_layers, (size_t)p->seq_len), (size_t)kvl * sizeof(float));
  NumaRegion *exchange_regions[SOCKETS];
  for (int s = 0; s < SOCKETS; s++) {
    TpSocket *sk = &tp->s[s];
    sk->socket = s;
    sk->team = pool_team(s);
    if (!sk->team)
      fail("LLAMA_SOCKETS=2 needs a worker pool of two teams");
    int node = placement_node(mode, &topology->socket[s], 0), thp = region_thp_default();
    if (tp->set.region_bytes[s])
      region_reserve(&sk->weights, tp->set.region_bytes[s], s, node, ROLE_WEIGHTS, thp);
    region_reserve(&sk->kv_k, kv_bytes, s, node, ROLE_KV_K, thp);
    region_reserve(&sk->kv_v, kv_bytes, s, node, ROLE_KV_V, thp);
    region_reserve(&sk->work, work_bytes(tp), s, node, ROLE_WORK, 0);
    region_reserve(&sk->exchange, exchange_region_bytes(p->dim), s, node, ROLE_WORK, 0);
    exchange_regions[s] = &sk->exchange;
    region_mark_populated(&sk->kv_k);
    region_mark_populated(&sk->kv_v);
    region_mark_populated(&sk->work);
  }
  if (tp->set.global_bytes)
    region_reserve(&tp->global, tp->set.global_bytes, 0, placement_node(mode, &topology->socket[0], 1), ROLE_EMBEDDING,
                   region_thp_default());
  tp->exchange = exchange_create(p->dim, exchange_regions);
  tp->logits = memory_alloc((size_t)p->vocab_size, sizeof(float), MEM_ACTIVATION);
  if (tp->check)
    for (int s = 0; s < SOCKETS; s++)
      tp->s[s].debug = memory_alloc((size_t)p->n_layers * p->dim, sizeof(float), MEM_ACTIVATION);
  if (pthread_create(&tp->thread, NULL, coordinator_main, tp))
    fail("cannot start the coordinator thread of socket 1");
  tp->started = 1;

  run_both(tp, CMD_SETUP);
  void *base[SOCKETS] = {tp->s[0].weights.data, tp->s[1].weights.data};
  shards_bind(&tp->set, base, tp->global.data);
  double start = monotonic_seconds();
  run_both(tp, CMD_LOAD);
  metrics.placement_populate_s = monotonic_seconds() - start;
  for (int s = 0; s < SOCKETS; s++)
    if (tp->s[s].weights.mapping_base) {
      region_mark_populated(&tp->s[s].weights);
      region_protect(&tp->s[s].weights);
    }
  if (tp->global.mapping_base) {
    region_mark_populated(&tp->global);
    region_protect(&tp->global);
  }
  start = monotonic_seconds();
  for (int s = 0; s < SOCKETS; s++)
    verify_bound(&tp->s[s].weights, mode);
  verify_bound(&tp->global, mode);
  metrics.placement_verify_s = monotonic_seconds() - start;
  metrics.placement_owned = strcmp(mode, "file") != 0;
  metrics.sockets = SOCKETS;
  return tp;
}

float *tp_forward(Tp *tp, int token, int pos) {
  tp->token = token;
  tp->pos = pos;
  run_both(tp, CMD_FORWARD);
  if (tp->check && !workers_aborted())
    for (int l = 0; l < tp->c.n_layers; l++)
      if (memcmp(tp->s[0].debug + (size_t)l * tp->c.dim, tp->s[1].debug + (size_t)l * tp->c.dim,
                 (size_t)tp->c.dim * sizeof(float)))
        fail("LLAMA_TP_CHECK: the residual copies of the two sockets differ after layer %d (position %d)", l, pos);
  return tp->logits;
}

void tp_kv_stats(const Tp *tp, int position, KvStats *out) {
  size_t per_socket =
      checked_mul(checked_mul((size_t)tp->c.n_layers, (size_t)tp->c.seq_len), (size_t)tp->kv_dim_local * sizeof(float));
  *out = (KvStats){.reserved_bytes = checked_mul(checked_mul(2, SOCKETS), per_socket),
                   .active_bytes = checked_mul(checked_mul(checked_mul(2, SOCKETS), (size_t)tp->c.n_layers),
                                               checked_mul((size_t)position, (size_t)tp->kv_dim_local * sizeof(float))),
                   .node = tp->s[0].kv_k.node};
}

size_t tp_shard_bytes(const Tp *tp, int socket) {
  return tp->set.region_bytes[socket];
}

void tp_free(Tp *tp) {
  if (!tp)
    return;
  if (tp->started) {
    run_both(tp, CMD_STOP);
    pthread_join(tp->thread, NULL);
  }
  exchange_destroy(tp->exchange);
  for (int s = 0; s < SOCKETS; s++) {
    TpSocket *sk = &tp->s[s];
    memory_free(sk->debug);
    region_release(&sk->weights);
    region_release(&sk->kv_k);
    region_release(&sk->kv_v);
    region_release(&sk->work);
    region_release(&sk->exchange);
  }
  region_release(&tp->global);
  shards_free(&tp->set);
  memory_free(tp->logits);
  memory_free(tp);
}
