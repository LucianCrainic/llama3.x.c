#include "tp.h"

#include "gemm.h"
#include "kernels.h"
#include "loader.h"
#include "numa.h"
#include "profile.h"
#include "workers.h"
#include <limits.h>
#include <pthread.h>
#include <sched.h>
#include <stdatomic.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

enum { SOCKETS = 2, SPIN_LIMIT = 2048 };
/* Per-socket profile slots: the engine's phases, then the projections that end in a reduction, timed on their own. */
enum { TP_WO_MM = PHASE_COUNT, TP_W2_MM, TP_SLOTS };

typedef enum { CMD_SETUP, CMD_LOAD, CMD_PACK, CMD_PACK_CLS, CMD_FORWARD, CMD_FORWARD_BATCH, CMD_STOP } Command;

/* The seven matrices of a layer that get a packed copy, in the order of TpSocket.packed_at. */
static const TensorRole PACK_ROLES[7] = {T_WQ, T_WK, T_WV, T_WO, T_W1, T_W2, T_W3};

typedef struct {
  int socket;
  Team *team;
  NumaRegion weights, kv_k, kv_v, work, exchange, packed, packed_cls;
  void **packed_at;     /* [layer * 7 + PACK_ROLES index]: the packed copy in `packed`, or NULL */
  uint16_t *cls_bf16;   /* the packed classifier range, once all-position logits were asked for (BF16 mode) */
  AlignedBuffer panel;  /* BF16 mode: the activations converted to BF16 */
  float *x, *xb, *hb, *q, *att, *rope; /* socket-local activations, in the work region */
  float *key_cache, *value_cache; /* [layer][seq][kv_dim / 2]; LLAMA_KV_LAYOUT=head: [layer][kv head][seq][head_dim] */
  float *kstage,
      *vstage; /* LLAMA_KV_LAYOUT=head: the K/V rows of the projections (rows x kv_dim / 2), stored per head */
  AttentionWork attention;
  float *debug; /* LLAMA_TP_CHECK: x after every layer */
  uint64_t *debug_hash; /* LLAMA_TP_CHECK, batched forward: a hash of x (all rows) after every layer */
  int leader_cpu; /* CPU the socket's coordinator ran on during setup (-1 unknown) */
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
  TpPack pack;
  PackBf16 *bjobs[SOCKETS]; /* the packing work of each socket, planned on the main thread (regions are not thread-safe) */
  PackS8 *sjobs[SOCKETS];
  int njobs[SOCKETS];
  int cls_ready;
  int kv_dim_local;
  int rows; /* activation rows each socket holds: the batch capacity (1 for Q8, which has no batched path yet) */
  /* The batched forward (D1): its arguments, set before CMD_FORWARD_BATCH, and the all-position logits. */
  const int *btokens;
  int bcount, bpos0;
  LogitsWanted bwant;
  float *batch_logits;
  /* Coordination with the socket-1 coordinator thread. */
  pthread_t thread;
  int started;
  _Alignas(64) atomic_uint command_generation;
  _Alignas(64) atomic_uint done_generation;
  unsigned issued;
  /* LLAMA_PROFILE=1: seconds per socket, stage and slot; the exchange's five components (workers.h) per socket and
   * stage, as deltas of the exchange's running totals; forward passes per stage. */
  double prof[SOCKETS][STAGE_COUNT][TP_SLOTS];
  double xch[SOCKETS][STAGE_COUNT][5], xch_last[SOCKETS][5];
  size_t tokens[STAGE_COUNT];
  Command command;
  int token, pos, loader_fd;
};

int tp_hbm_pair(void) {
  const Topology *topology;
  char err[512];
  return !numa_topology_try(&topology, err, sizeof(err)) && topology->classification != CLASS_VIRTUAL &&
         !numa_check_placement(topology, SOCKETS, "hbm", err, sizeof(err));
}

static int knob_set(const char *name) {
  const char *value = getenv(name);
  return value && *value;
}

int tp_sockets_requested(void) {
  static int sockets;
  if (sockets)
    return sockets;
  const char *value = getenv("LLAMA_SOCKETS"), *topology = getenv("LLAMA_TOPOLOGY"), *placement = getenv("LLAMA_PLACEMENT");
  if (knob_set("LLAMA_SOCKETS")) {
    if (strcmp(value, "1") && strcmp(value, "2"))
      fail("LLAMA_SOCKETS must be 1 or 2");
    return sockets = value[0] - '0';
  }
  /* D2 §7: unset, both sockets of a node whose sockets each have an allowed HBM node, unless the other knobs ask for a
   * one-socket configuration (a given topology, file or DDR placement). */
  int auto_two = !(knob_set("LLAMA_TOPOLOGY") && strcmp(topology, "auto")) &&
                 !(knob_set("LLAMA_PLACEMENT") && strcmp(placement, "hbm")) && tp_hbm_pair();
  if (auto_two)
    LOG(LOG_INFO, "LLAMA_SOCKETS unset: both sockets have an HBM node, so the model runs on both (LLAMA_SOCKETS=1 keeps "
                  "one)");
  return sockets = auto_two ? 2 : 1;
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
  /* This thread inherited the creator's single-CPU mask; it is the leader of team 1 and belongs on team 1's first CPU. */
  team_adopt_leader(tp->s[1].team);
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

/* One projection that ends in the cross-socket reduction: Wo or W2, FP32 (`wq` NULL) or Q8. */
typedef struct {
  Tp *tp;
  int socket, n, d, gs;
  float *partial, *residual;
  size_t ld;
  const float *x, *w;               /* FP32 */
  const K8Input *input;             /* Q8 */
  const K8Weight *wq;               /* Q8; NULL for FP32 */
  double *leader_seconds;           /* LLAMA_PROFILE: the leader's own share of the projection */
  atomic_int claim;                 /* row counter shared by the members (zero before the team starts) */
} ProjectJob;

static void rows_done_fn(void *p, int r0, int r1) {
  const ProjectJob *job = p;
  exchange_rows_done(job->tp->exchange, job->socket, r0, r1);
}

/* Decode's Wo/W2: projection and reduction fused into one team job, with no team barrier. Every member computes its
 * rows of the partial and flags them (rows_done_fn), then adds its own element range of both sockets' partials as soon as
 * the producers of those rows, on both sockets, have flagged them (exchange_reduce_add_flags). Measured on the 8B model:
 * one dispatch and no barrier instead of two jobs and two barriers saves 17-19 us per reduction (B7 findings). */
static void project_reduce_fn(const void *p, int worker, int workers) {
  const ProjectJob *job = p;
  double start = worker == 0 && job->leader_seconds ? monotonic_seconds() : 0.0;
  if (job->wq)
    k_matmul_q8_ld_worker(job->partial, job->input, job->wq, job->ld, job->n, job->d, job->gs, K_PARTIAL, worker,
                          workers, (atomic_int *)&job->claim, rows_done_fn, (void *)job);
  else
    k_matmul_ld_worker(job->partial, job->x, job->w, job->ld, job->n, job->d, K_PARTIAL, worker, workers,
                       (atomic_int *)&job->claim, rows_done_fn, (void *)job);
  if (start)
    *job->leader_seconds = monotonic_seconds() - start;
  exchange_reduce_add_flags(job->tp->exchange, job->socket, worker, workers, job->residual, job->d, NULL);
}

static void project_reduce(Tp *tp, TpSocket *sk, ProjectJob *job, int slot) {
  job->tp = tp;
  atomic_init(&job->claim, 0);
  job->socket = sk->socket;
  job->residual = sk->x;
  double leader = 0.0;
  job->leader_seconds = profile_enabled ? &leader : NULL;
  team_run(sk->team, project_reduce_fn, job);
  exchange_reduce_finish(tp->exchange, sk->socket); /* the join above: every add of this generation is done */
  tp->prof[sk->socket][profile_stage][slot] += leader;
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

/* The engine's profile counters are shared, so only socket 0 feeds them; every socket also keeps its own in tp->prof. */
#define SPROFILE(phase, ...)                                                                                           \
  do {                                                                                                                 \
    double sprofile_start_ = profile_enabled ? monotonic_seconds() : 0.0;                                              \
    if (s == 0) {                                                                                                      \
      PROFILE(phase, __VA_ARGS__);                                                                                     \
    } else {                                                                                                           \
      __VA_ARGS__;                                                                                                     \
    }                                                                                                                  \
    if (profile_enabled)                                                                                               \
      tp->prof[s][profile_stage][phase] += monotonic_seconds() - sprofile_start_;                                      \
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
    int head_layout = kernels_kv_layout_head();
    float *k = head_layout ? sk->kstage : sk->key_cache + layer + (size_t)pos * kvl,
          *v = head_layout ? sk->vstage : sk->value_cache + layer + (size_t)pos * kvl;
    const float *wq = shards_find(&tp->set, T_WQ, l, s)->q, *wk = shards_find(&tp->set, T_WK, l, s)->q,
                *wv = shards_find(&tp->set, T_WV, l, s)->q;
    SPROFILE(PHASE_QKV, k_matmul_qkv(sk->q, k, v, sk->xb, wq, wk, wv, dim, dim / SOCKETS, kvl));
    SPROFILE(PHASE_ROPE, k_rope_apply(sk->q, k, sk->rope, hl, kl, hd));
    if (head_layout)
      SPROFILE(PHASE_KV_STORE, k_kv_store(sk->key_cache + layer, k, (size_t)kvl, pos, 1, kl, hd, p->seq_len);
               k_kv_store(sk->value_cache + layer, v, (size_t)kvl, pos, 1, kl, hd, p->seq_len));
    SPROFILE(PHASE_ATTENTION, k_attention_kv(sk->xb, sk->q, sk->key_cache + layer, sk->value_cache + layer, sk->att,
                                             &sk->attention, hl, kl, hd, p->seq_len, pos));
    const Shard *wo = shards_find(&tp->set, T_WO, l, s);
    float *partial = exchange_partial(tp->exchange, s);
    if (!partial)
      return;
    ProjectJob wo_job = {.n = dim / SOCKETS, .d = dim, .partial = partial, .ld = wo->ld, .x = sk->xb, .w = wo->q};
    SPROFILE(PHASE_WO, project_reduce(tp, sk, &wo_job, TP_WO_MM));

    SPROFILE(PHASE_RMSNORM, k_rmsnorm(sk->xb, sk->x, norm_weight(tp, T_RMS_FFN, l, s), dim, p->norm_eps));
    const float *w1 = shards_find(&tp->set, T_W1, l, s)->q, *w3 = shards_find(&tp->set, T_W3, l, s)->q;
    SPROFILE(PHASE_W1_W3, k_matmul_swiglu(sk->hb, sk->xb, w1, w3, dim, hidden_l));
    const Shard *w2 = shards_find(&tp->set, T_W2, l, s);
    partial = exchange_partial(tp->exchange, s);
    if (!partial)
      return;
    ProjectJob w2_job = {.n = hidden_l, .d = dim, .partial = partial, .ld = w2->ld, .x = sk->hb, .w = w2->q};
    SPROFILE(PHASE_W2, project_reduce(tp, sk, &w2_job, TP_W2_MM));
    if (sk->debug)
      memcpy(sk->debug + (size_t)l * dim, sk->x, (size_t)dim * sizeof(float));
  }
  SPROFILE(PHASE_RMSNORM, k_rmsnorm(sk->x, sk->x, norm_weight(tp, T_RMS_FINAL, -1, s), dim, p->norm_eps));
  const Shard *cls = shards_find(&tp->set, T_WCLS, -1, s);
  SPROFILE(PHASE_CLASSIFIER, k_matmul(tp->logits + cls->row0, sk->x, cls->q, dim, cls->rows));
}

/* Q8_0 (B6): the same group size, scales and activation rule as the one-socket engine (engine.c). Row shards quantize the full normalized input,
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
    int head_layout = kernels_kv_layout_head();
    float *k = head_layout ? sk->kstage : sk->key_cache + layer + (size_t)pos * kvl,
          *v = head_layout ? sk->vstage : sk->value_cache + layer + (size_t)pos * kvl;
    K8Weight wq = q8_weight(shards_find(&tp->set, T_WQ, l, s)), wk = q8_weight(shards_find(&tp->set, T_WK, l, s)),
             wv = q8_weight(shards_find(&tp->set, T_WV, l, s));
    SPROFILE(PHASE_QKV, k_matmul_q8_qkv(sk->q, k, v, &in, wq, wk, wv, dim, dim / SOCKETS, kvl, gs));
    SPROFILE(PHASE_ROPE, k_rope_apply(sk->q, k, sk->rope, hl, kl, hd));
    if (head_layout)
      SPROFILE(PHASE_KV_STORE, k_kv_store(sk->key_cache + layer, k, (size_t)kvl, pos, 1, kl, hd, p->seq_len);
               k_kv_store(sk->value_cache + layer, v, (size_t)kvl, pos, 1, kl, hd, p->seq_len));
    SPROFILE(PHASE_ATTENTION, k_attention_kv(sk->xb, sk->q, sk->key_cache + layer, sk->value_cache + layer, sk->att,
                                             &sk->attention, hl, kl, hd, p->seq_len, pos));
    const Shard *wo = shards_find(&tp->set, T_WO, l, s);
    SPROFILE(PHASE_QUANTIZE, in = q8_input(tp, sk->aq, sk->as, sk->xb, dim / SOCKETS));
    float *partial = exchange_partial(tp->exchange, s);
    if (!partial)
      return;
    K8Weight wo_weight = q8_weight(wo);
    ProjectJob wo_job = {.n = dim / SOCKETS, .d = dim, .gs = gs, .partial = partial, .ld = wo->ld, .input = &in,
                         .wq = &wo_weight};
    SPROFILE(PHASE_WO, project_reduce(tp, sk, &wo_job, TP_WO_MM));

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
    ProjectJob w2_job = {.n = hidden_l, .d = dim, .gs = gs, .partial = partial, .ld = w2->ld, .input = &in,
                         .wq = &w2_weight};
    SPROFILE(PHASE_W2, project_reduce(tp, sk, &w2_job, TP_W2_MM));
    if (sk->debug)
      memcpy(sk->debug + (size_t)l * dim, sk->x, (size_t)dim * sizeof(float));
  }
  SPROFILE(PHASE_RMSNORM, k_rmsnorm(sk->x, sk->x, norm_weight(tp, T_RMS_FINAL, -1, s), dim, p->norm_eps));
  K8Input in;
  SPROFILE(PHASE_QUANTIZE, in = q8_input(tp, sk->xq, sk->xs, sk->x, dim));
  const Shard *cls = shards_find(&tp->set, T_WCLS, -1, s);
  SPROFILE(PHASE_CLASSIFIER, k_matmul_q8(tp->logits + cls->row0, &in, q8_weight(cls), dim, cls->rows, gs));
}

/* ---- Batched forward (D1) -------------------------------------------------------------------------------------- */

/* Both sockets hold the same M x dim residual. Each runs the GEMMs of its own shards on its own team: row shards (QKV,
 * W1, W3) produce this socket's heads and hidden channels, the column shards (Wo, W2) a full M x dim partial that the
 * exchange adds to both residuals once. K/V rows go straight into the socket's own cache, and attention reads only
 * that cache, so no cross-socket barrier is needed besides the two reductions per layer. */
static void embed_rows(const Tp *tp, TpSocket *sk) {
  size_t dim = (size_t)tp->c.dim;
  for (int i = 0; i < tp->bcount; i++) {
    float *x = sk->x + (size_t)i * dim;
    if (!tp->quantized) {
      memcpy(x, embedding_row(tp, tp->btokens[i]), dim * sizeof(float));
      continue;
    }
    int row;
    const Shard *embed = embedding_shard(tp, tp->btokens[i], &row);
    for (size_t j = 0; j < dim; j++)
      x[j] = ((const int8_t *)embed->q)[(size_t)row * dim + j] * embed->s[((size_t)row * dim + j) / (size_t)tp->gs];
  }
}

static void rope_tables(const Tp *tp, TpSocket *sk) {
  for (int i = 0; i < tp->bcount; i++)
    k_rope_table(&tp->c, tp->bpos0 + i, sk->rope + (size_t)i * 2 * tp->c.head_dim);
}

static void rmsnorm_rows(const Tp *tp, float *out, const float *x, const float *weight, int rows) {
  k_rmsnorm_rows(out, x, weight, rows, tp->c.dim, tp->c.norm_eps);
}

/* The packed copy of matrix `index` (PACK_ROLES order) of layer l on this socket, or NULL. */
static void *packed_copy(const TpSocket *sk, int layer, int index) {
  return sk->packed_at ? sk->packed_at[(size_t)layer * 7 + (size_t)index] : NULL;
}

/* One product on FP32 activations: the FP32 GEMM, or in BF16 mode the same segments on the packed copies (the
 * activations are converted to BF16 into the socket's panel, as in the one-socket C5 path). */
static void gemm_f32_or_bf16(const Tp *tp, TpSocket *sk, const float *a, int lda, int k, const GemmSegment *seg,
                             int count, void *const *packed, void *const *packed3) {
  if (tp->pack != TP_PACK_BF16) {
    k_gemm_f32(a, lda, tp->bcount, k, seg, count);
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
  k_gemm_bf16(a, lda, tp->bcount, k, b, count, &sk->panel);
}

/* The partial product of this socket's K slice for all rows, into the exchange buffer, then the reduction into x.
 * `w` is a column shard (`packed` its packed copy, or NULL): FP32 products take the activations `a` (stride lda), W8A8
 * products the quantized rows `aq`/`as` (dense, stride k and k / gs). */
static int project_batch_reduce(Tp *tp, TpSocket *sk, const float *a, int lda, const int8_t *aq, const float *as, int k,
                                const Shard *w, void *packed) {
  float *partial = exchange_partial(tp->exchange, sk->socket);
  if (!partial)
    return 0;
  if (tp->quantized) {
    GemmQ8Segment seg = {.c = partial, .ldc = tp->c.dim, .w = q8_weight(w), .pw = packed, .n = tp->c.dim,
                         .kind = GEMM_PARTIAL};
    k_gemm_q8(aq, as, tp->bcount, k, tp->gs, &seg, 1);
  } else {
    GemmSegment seg = {.c = partial, .ldc = tp->c.dim, .w = w->q, .ldw = w->ld, .n = tp->c.dim, .kind = GEMM_PARTIAL};
    gemm_f32_or_bf16(tp, sk, a, lda, k, &seg, 1, &packed, NULL);
  }
  reduce(tp, sk, sk->x, tp->bcount * tp->c.dim);
  return !workers_aborted();
}

static uint64_t hash_floats(const float *x, size_t n) {
  uint64_t h = 1469598103934665603ull;
  const uint32_t *w = (const uint32_t *)x;
  for (size_t i = 0; i < n; i++)
    h = (h ^ w[i]) * 1099511628211ull;
  return h;
}

/* Quantizes `rows` rows of n values (stride ldx) of x for the next W8A8 product, exactly as the per-token path does it. */
static void quantize_batch(const Tp *tp, int8_t *q, float *scales, const float *x, size_t ldx, int n) {
  k_quantize_rows(q, scales, x, ldx, tp->bcount, n, tp->gs);
}

static void forward_batch(Tp *tp, int s) {
  TpSocket *sk = &tp->s[s];
  const Config *p = &tp->c;
  int dim = p->dim, hd = p->head_dim, hl = p->n_heads / SOCKETS, kl = p->n_kv_heads / SOCKETS, kvl = kl * hd;
  int ql = dim / SOCKETS, hidden_l = p->hidden_dim / SOCKETS, m = tp->bcount, pos0 = tp->bpos0, gs = tp->gs;
  int q8 = tp->quantized;
  SPROFILE(PHASE_EMBED, embed_rows(tp, sk));
  SPROFILE(PHASE_ROPE_TABLE, rope_tables(tp, sk));
  for (int l = 0; l < p->n_layers; l++) {
    SPROFILE(PHASE_RMSNORM, rmsnorm_rows(tp, sk->xb, sk->x, norm_weight(tp, T_RMS_ATT, l, s), m));
    size_t layer = (size_t)l * p->seq_len * kvl;
    float *kc = sk->key_cache + layer, *vc = sk->value_cache + layer;
    int head_layout = kernels_kv_layout_head();
    float *kd = head_layout ? sk->kstage : kc + (size_t)pos0 * kvl,
          *vd = head_layout ? sk->vstage : vc + (size_t)pos0 * kvl;
    const Shard *wq = shards_find(&tp->set, T_WQ, l, s), *wk = shards_find(&tp->set, T_WK, l, s),
                *wv = shards_find(&tp->set, T_WV, l, s);
    if (q8) {
      SPROFILE(PHASE_QUANTIZE, quantize_batch(tp, sk->xq, sk->xs, sk->xb, (size_t)dim, dim));
      GemmQ8Segment qkv[3] = {
          {.c = sk->q, .ldc = ql, .w = q8_weight(wq), .pw = packed_copy(sk, l, 0), .n = ql, .kind = GEMM_STORE},
          {.c = kd, .ldc = kvl, .w = q8_weight(wk), .pw = packed_copy(sk, l, 1), .n = kvl, .kind = GEMM_STORE},
          {.c = vd, .ldc = kvl, .w = q8_weight(wv), .pw = packed_copy(sk, l, 2), .n = kvl, .kind = GEMM_STORE}};
      SPROFILE(PHASE_QKV, k_gemm_q8(sk->xq, sk->xs, m, dim, gs, qkv, 3));
    } else {
      GemmSegment qkv[3] = {{.c = sk->q, .ldc = ql, .w = wq->q, .ldw = wq->ld, .n = ql, .kind = GEMM_STORE},
                            {.c = kd, .ldc = kvl, .w = wk->q, .ldw = wk->ld, .n = kvl, .kind = GEMM_STORE},
                            {.c = vd, .ldc = kvl, .w = wv->q, .ldw = wv->ld, .n = kvl, .kind = GEMM_STORE}};
      void *const pq[3] = {packed_copy(sk, l, 0), packed_copy(sk, l, 1), packed_copy(sk, l, 2)};
      SPROFILE(PHASE_QKV, gemm_f32_or_bf16(tp, sk, sk->xb, dim, dim, qkv, 3, pq, NULL));
    }
    SPROFILE(PHASE_ROPE,
             k_rope_apply_rows(sk->q, (size_t)ql, kd, (size_t)kvl, sk->rope, (size_t)2 * hd, m, hl, kl, hd));
    if (head_layout)
      SPROFILE(PHASE_KV_STORE, k_kv_store(kc, kd, (size_t)kvl, pos0, m, kl, hd, p->seq_len);
               k_kv_store(vc, vd, (size_t)kvl, pos0, m, kl, hd, p->seq_len));
    SPROFILE(PHASE_ATTENTION,
             k_attention_batch_kv(sk->xb, sk->q, kc, vc, sk->att, &sk->attention, hl, kl, hd, p->seq_len, pos0, m));
    int ok;
    if (q8)
      SPROFILE(PHASE_QUANTIZE, quantize_batch(tp, sk->aq, sk->as, sk->xb, (size_t)ql, ql));
    SPROFILE(PHASE_WO, ok = project_batch_reduce(tp, sk, sk->xb, ql, sk->aq, sk->as, ql,
                                                 shards_find(&tp->set, T_WO, l, s), packed_copy(sk, l, 3)));
    if (!ok)
      return;

    SPROFILE(PHASE_RMSNORM, rmsnorm_rows(tp, sk->xb, sk->x, norm_weight(tp, T_RMS_FFN, l, s), m));
    const Shard *w1 = shards_find(&tp->set, T_W1, l, s), *w3 = shards_find(&tp->set, T_W3, l, s);
    if (q8) {
      SPROFILE(PHASE_QUANTIZE, quantize_batch(tp, sk->xq, sk->xs, sk->xb, (size_t)dim, dim));
      GemmQ8Segment up = {.c = sk->hb, .ldc = hidden_l, .w = q8_weight(w1), .w3 = q8_weight(w3),
                          .pw = packed_copy(sk, l, 4), .pw3 = packed_copy(sk, l, 6), .n = hidden_l,
                          .kind = GEMM_SWIGLU};
      SPROFILE(PHASE_W1_W3, k_gemm_q8(sk->xq, sk->xs, m, dim, gs, &up, 1));
      SPROFILE(PHASE_QUANTIZE, quantize_batch(tp, sk->aq, sk->as, sk->hb, (size_t)hidden_l, hidden_l));
    } else {
      GemmSegment up = {.c = sk->hb, .ldc = hidden_l, .w = w1->q, .w3 = w3->q, .ldw = w1->ld, .n = hidden_l,
                        .kind = GEMM_SWIGLU};
      void *const p1 = packed_copy(sk, l, 4), *const p3 = packed_copy(sk, l, 6);
      SPROFILE(PHASE_W1_W3, gemm_f32_or_bf16(tp, sk, sk->xb, dim, dim, &up, 1, &p1, &p3));
    }
    SPROFILE(PHASE_W2, ok = project_batch_reduce(tp, sk, sk->hb, hidden_l, sk->aq, sk->as, hidden_l,
                                                 shards_find(&tp->set, T_W2, l, s), packed_copy(sk, l, 5)));
    if (!ok)
      return;
    if (sk->debug_hash)
      sk->debug_hash[l] = hash_floats(sk->x, (size_t)m * dim);
  }
  if (tp->bwant == LOGITS_NONE)
    return;
  /* The classifier is per row, exactly as the single-socket batch does it: only the last row for LOGITS_LAST. */
  const Shard *cls = shards_find(&tp->set, T_WCLS, -1, s);
  const float *final_norm = norm_weight(tp, T_RMS_FINAL, -1, s);
  if (tp->bwant == LOGITS_ALL && tp->pack == TP_PACK_BF16) {
    /* As in the one-socket BF16 mode: all-position logits are a BF16 GEMM on the packed classifier range, the default
     * last-row classifier below stays FP32. */
    SPROFILE(PHASE_RMSNORM, k_rmsnorm_rows(sk->x, sk->x, final_norm, m, dim, p->norm_eps));
    GemmBf16Segment seg = {.c = tp->batch_logits + cls->row0, .ldc = p->vocab_size, .w = sk->cls_bf16, .n = cls->rows,
                           .kind = GEMM_STORE};
    SPROFILE(PHASE_CLASSIFIER, k_gemm_bf16(sk->x, dim, m, dim, &seg, 1, &sk->panel));
    return;
  }
  for (int i = tp->bwant == LOGITS_ALL ? 0 : m - 1; i < m; i++) {
    float *x = sk->x + (size_t)i * dim;
    float *out = tp->bwant == LOGITS_ALL ? tp->batch_logits + (size_t)i * p->vocab_size : tp->logits;
    SPROFILE(PHASE_RMSNORM, k_rmsnorm(x, x, final_norm, dim, p->norm_eps));
    if (q8) {
      K8Input in;
      SPROFILE(PHASE_QUANTIZE, in = q8_input(tp, sk->xq, sk->xs, x, dim));
      SPROFILE(PHASE_CLASSIFIER, k_matmul_q8(out + cls->row0, &in, q8_weight(cls), dim, cls->rows, gs));
    } else {
      SPROFILE(PHASE_CLASSIFIER, k_matmul(out + cls->row0, x, cls->q, dim, cls->rows));
    }
  }
}

/* Carves the socket's activations out of its work region and sizes the attention work from its team. Runs on the
 * socket's coordinator, so the pages are first written by the socket that uses them. */
static void setup_socket(Tp *tp, int s) {
  TpSocket *sk = &tp->s[s];
  const Config *p = &tp->c;
  int hl = p->n_heads / SOCKETS, kl = p->n_kv_heads / SOCKETS;
  /* The socket that uses the work region writes all of it once, so every page exists on its own node and the page query
   * can verify the region like the others (a fresh mapping is zero already; this only makes the pages exist). */
  memset(sk->work.data, 0, sk->work.data_bytes);
  char *cursor = sk->work.data;
  size_t used = 0;
#define CARVE(pointer, count, type)                                                                                    \
  do {                                                                                                                 \
    used = (used + 63) / 64 * 64;                                                                                      \
    (pointer) = (type *)(cursor + used);                                                                               \
    used += (size_t)(count) * sizeof(type);                                                                            \
  } while (0)
  size_t rows = (size_t)tp->rows;
  CARVE(sk->x, rows * p->dim, float);
  CARVE(sk->xb, rows * p->dim, float);
  CARVE(sk->hb, rows * (p->hidden_dim / SOCKETS), float);
  CARVE(sk->q, rows * (p->dim / SOCKETS), float);
  CARVE(sk->att, (size_t)hl * p->seq_len, float);
  CARVE(sk->rope, rows * 2 * (size_t)p->head_dim, float);
  if (kernels_kv_layout_head()) {
    CARVE(sk->kstage, rows * (size_t)kl * p->head_dim, float);
    CARVE(sk->vstage, rows * (size_t)kl * p->head_dim, float);
  }
  if (tp->quantized) {
    /* Quantized rows are padded to whole 32-row blocks: the AMX kernel loads 16 rows at a time. */
    size_t padded = (rows + 31) / 32 * 32;
    CARVE(sk->xq, padded * p->dim, int8_t);
    CARVE(sk->xs, padded * (p->dim / tp->gs), float);
    size_t slice = (size_t)(p->dim > p->hidden_dim ? p->dim : p->hidden_dim) / SOCKETS; /* attention output or FFN hidden */
    CARVE(sk->aq, padded * slice, int8_t);
    CARVE(sk->as, padded * (slice / tp->gs), float);
  }
  float *attention_memory;
  CARVE(attention_memory, k_attention_bytes_rows(hl, kl, p->head_dim, p->seq_len, tp->rows) / sizeof(float) + 1, float);
#undef CARVE
  if (used > sk->work.data_bytes)
    fail("socket %d work region holds %zu bytes, %zu needed", s, sk->work.data_bytes, used);
  k_attention_init_rows(&sk->attention, hl, kl, p->head_dim, p->seq_len, tp->rows, attention_memory);
  sk->key_cache = sk->kv_k.data;
  sk->value_cache = sk->kv_v.data;
}

/* ---- region sizes -------------------------------------------------------------------------------------------- */

size_t tp_work_bytes(const Config *p, int quantized, int gs, int rows) {
  int hl = p->n_heads / SOCKETS, kl = p->n_kv_heads / SOCKETS;
  size_t slice = (size_t)(p->dim > p->hidden_dim ? p->dim : p->hidden_dim) / SOCKETS;
  size_t bytes = 64 * 16; /* alignment of each buffer */
  bytes += (size_t)rows * (size_t)(2 * p->dim + p->hidden_dim / SOCKETS + p->dim / SOCKETS + 2 * p->head_dim) *
           sizeof(float);
  bytes += (size_t)hl * p->seq_len * sizeof(float);
  if (kernels_kv_layout_head()) /* the K/V row buffers of LLAMA_KV_LAYOUT=head */
    bytes += 2 * (size_t)rows * (size_t)kl * (size_t)p->head_dim * sizeof(float);
  if (quantized) { /* xq, xs, aq, as, for whole 32-row blocks */
    size_t padded = ((size_t)rows + 31) / 32 * 32;
    bytes += padded * ((size_t)p->dim + (size_t)p->dim / gs * sizeof(float) + slice + slice / gs * sizeof(float));
  }
  bytes += k_attention_bytes_rows(hl, kl, p->head_dim, p->seq_len, rows) + 64;
  return bytes;
}

size_t tp_exchange_bytes(const Config *p, int rows) {
  return exchange_region_bytes(checked_mul((size_t)rows, (size_t)p->dim));
}

static size_t work_bytes(const Tp *tp) {
  return tp_work_bytes(&tp->c, tp->quantized, tp->gs, tp->rows);
}

static void socket_command(Tp *tp, int socket) {
  switch (tp->command) {
  case CMD_SETUP:
    setup_socket(tp, socket);
    tp->s[socket].leader_cpu = cpu_current();
    break;
  case CMD_LOAD:
    shards_load(&tp->set, tp->loader_fd, socket, tp->s[socket].team);
    break;
  case CMD_PACK:
    if (tp->pack == TP_PACK_BF16)
      pack_bf16_vnni_many(tp->bjobs[socket], tp->njobs[socket]);
    else
      pack_s8_vnni_many(tp->sjobs[socket], tp->njobs[socket]);
    break;
  case CMD_PACK_CLS:
    pack_bf16_vnni_many(tp->bjobs[socket], 1);
    break;
  case CMD_FORWARD:
    if (tp->quantized)
      forward_q8(tp, socket);
    else
      forward_fp32(tp, socket);
    break;
  case CMD_FORWARD_BATCH:
    forward_batch(tp, socket);
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
  /* A region that was never reserved (the global one of a model with a tied embedding) is zero-initialised, which reads
   * as node 0: there is nothing to verify. */
  if (r->node < 0 || !r->mapping_base)
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

/* ---- packed copies (D1) ------------------------------------------------------------------------------------------ */

static size_t pack_bytes_of(const Tp *tp, const Shard *sh) {
  return tp->pack == TP_PACK_BF16 ? bf16_vnni_elements(sh->rows, sh->cols) * sizeof(uint16_t)
                                  : s8_vnni_bytes(sh->rows, sh->cols);
}

static size_t align_line(size_t n) {
  return (n + 63) / 64 * 64;
}

/* What socket s's packed region must hold: its shard of every matrix of every layer, each on a cache line. */
static size_t packed_region_bytes(const Tp *tp, int s) {
  size_t total = 0;
  for (int l = 0; l < tp->c.n_layers; l++)
    for (int i = 0; i < 7; i++)
      total = align_line(total) + pack_bytes_of(tp, shards_find(&tp->set, PACK_ROLES[i], l, s));
  return total;
}

/* The per-socket HBM budget counts everything a socket holds: its weight shards (the FP32 copy stays for decode), the
 * packed copies, both KV caches, the work region and the exchange buffers. */
static void check_budget(const Tp *tp, int s, const char *mode, size_t kv_bytes, size_t packed_bytes) {
  size_t budget = placement_hbm_budget();
  if (strcmp(mode, "hbm"))
    return;
  size_t used = tp->set.region_bytes[s] + 2 * kv_bytes + work_bytes(tp) + tp_exchange_bytes(&tp->c, tp->rows) +
                packed_bytes;
  if (used > budget)
    fail("socket %d needs %zu bytes of HBM (%zu of shards, %zu of packed %s copies, %zu of KV caches) but "
         "LLAMA_HBM_BUDGET_GIB allows %zu per socket",
         s, used, tp->set.region_bytes[s], packed_bytes, tp->pack == TP_PACK_BF16 ? "BF16" : "int8", 2 * kv_bytes,
         budget);
}

/* Main thread, after the shards are bound: carves the packed region of every socket and plans the packing jobs. The
 * coordinators only run them (the region bookkeeping is not thread-safe). */
static void plan_packing(Tp *tp) {
  for (int s = 0; s < SOCKETS; s++) {
    TpSocket *sk = &tp->s[s];
    size_t count = (size_t)tp->c.n_layers * 7;
    sk->packed_at = memory_alloc(count, sizeof(void *), MEM_METADATA);
    if (tp->pack == TP_PACK_BF16)
      tp->bjobs[s] = memory_alloc(count, sizeof(PackBf16), MEM_METADATA);
    else
      tp->sjobs[s] = memory_alloc(count, sizeof(PackS8), MEM_METADATA);
    tp->njobs[s] = (int)count;
    size_t used = 0;
    for (int l = 0; l < tp->c.n_layers; l++)
      for (int i = 0; i < 7; i++) {
        const Shard *sh = shards_find(&tp->set, PACK_ROLES[i], l, s);
        used = align_line(used);
        char *dst = (char *)sk->packed.data + used;
        used += pack_bytes_of(tp, sh);
        size_t at = (size_t)l * 7 + (size_t)i;
        sk->packed_at[at] = dst;
        if (tp->pack == TP_PACK_BF16)
          tp->bjobs[s][at] = pack_bf16_job_from_shard((uint16_t *)dst, sh);
        else
          tp->sjobs[s][at] = pack_s8_job_from_shard((int8_t *)dst, sh);
      }
    if (used > sk->packed.logical_bytes)
      fail("socket %d packed region holds %zu bytes, %zu needed", s, sk->packed.logical_bytes, used);
  }
}

/* The classifier range of each socket in BF16, built when all-position logits are first asked for (the default
 * last-row classifier stays FP32, as on one socket). */
static void pack_classifier(Tp *tp) {
  const char *mode = placement_mode();
  const Topology *topology = numa_topology();
  double start = monotonic_seconds();
  for (int s = 0; s < SOCKETS; s++) {
    TpSocket *sk = &tp->s[s];
    const Shard *cls = shards_find(&tp->set, T_WCLS, -1, s);
    size_t bytes = bf16_vnni_elements(cls->rows, cls->cols) * sizeof(uint16_t);
    region_reserve(&sk->packed_cls, bytes, s, placement_node(mode, &topology->socket[s], 0), ROLE_PACKED,
                   region_thp_default());
    sk->cls_bf16 = sk->packed_cls.data;
    tp->bjobs[s][0] = pack_bf16_job_from_shard(sk->cls_bf16, cls);
  }
  run_both(tp, CMD_PACK_CLS);
  for (int s = 0; s < SOCKETS; s++) {
    region_mark_populated(&tp->s[s].packed_cls);
    region_protect(&tp->s[s].packed_cls);
    verify_bound(&tp->s[s].packed_cls, mode);
  }
  metrics.weight_pack_s += monotonic_seconds() - start;
  tp->cls_ready = 1;
}

Tp *tp_create(const Config *p, const TensorManifest *m, int fd, int quantized, int gs, int fp32_activations,
              int max_batch, TpPack pack) {
  Tp *tp = memory_alloc(1, sizeof(*tp), MEM_METADATA);
  if ((pack == TP_PACK_BF16 && quantized) || (pack == TP_PACK_INT8 && (!quantized || fp32_activations)))
    fail("packed shards: BF16 needs an FP32 checkpoint and int8 needs W8A8");
  tp->pack = pack;
  /* The batched forward takes FP32 and W8A8; W8A32 is per token, like the one-socket C6 rule, so it keeps one row. */
  tp->rows = (quantized && fp32_activations) || max_batch < 1 ? 1 : max_batch;
  if (checked_mul((size_t)tp->rows, (size_t)p->dim) > (size_t)INT_MAX)
    fail("LLAMA_PREFILL_BATCH=%d is too large for LLAMA_SOCKETS=2: the exchange holds %d x %d elements", tp->rows,
         tp->rows, p->dim);
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
    size_t packed_bytes = pack == TP_PACK_NONE ? 0 : packed_region_bytes(tp, s);
    check_budget(tp, s, mode, kv_bytes, packed_bytes);
    if (packed_bytes)
      region_reserve(&sk->packed, packed_bytes, s, node, ROLE_PACKED, thp);
    region_reserve(&sk->work, work_bytes(tp), s, node, ROLE_WORK, 0);
    region_reserve(&sk->exchange, tp_exchange_bytes(p, tp->rows), s, node, ROLE_WORK, 0);
    exchange_regions[s] = &sk->exchange;
    region_mark_populated(&sk->kv_k);
    region_mark_populated(&sk->kv_v);
    region_mark_populated(&sk->work);
  }
  if (tp->set.global_bytes)
    region_reserve(&tp->global, tp->set.global_bytes, 0, placement_node(mode, &topology->socket[0], 1), ROLE_EMBEDDING,
                   region_thp_default());
  /* The exchange buffers and flags start at zero; writing the whole region also makes its pages exist on their node. */
  for (int s = 0; s < SOCKETS; s++)
    if (tp->s[s].exchange.data) {
      memset(tp->s[s].exchange.data, 0, tp->s[s].exchange.data_bytes);
      region_mark_populated(&tp->s[s].exchange);
    }
  tp->exchange = exchange_create(tp->rows * p->dim, exchange_regions);
  if (tp->rows > 1)
    LOG(LOG_INFO, "two-socket batch capacity %d rows: work region %.1f MiB and exchange %.1f MiB per socket", tp->rows,
        (double)work_bytes(tp) / (1 << 20), (double)tp_exchange_bytes(p, tp->rows) / (1 << 20));
  if (pack == TP_PACK_BF16)
    for (int s = 0; s < SOCKETS; s++) {
      int max_k = p->dim > p->hidden_dim / SOCKETS ? p->dim : p->hidden_dim / SOCKETS;
      aligned_buffer_alloc(&tp->s[s].panel, k_gemm_bf16_panel_bytes(tp->rows, max_k), MEM_ACTIVATION);
    }
  if (profile_enabled)
    exchange_set_timing(tp->exchange, 1);
  tp->logits = memory_alloc((size_t)p->vocab_size, sizeof(float), MEM_ACTIVATION);
  if (tp->check)
    for (int s = 0; s < SOCKETS; s++) {
      tp->s[s].debug = memory_alloc((size_t)p->n_layers * p->dim, sizeof(float), MEM_ACTIVATION);
      tp->s[s].debug_hash = memory_alloc((size_t)p->n_layers, sizeof(uint64_t), MEM_ACTIVATION);
    }
  if (pthread_create(&tp->thread, NULL, coordinator_main, tp))
    fail("cannot start the coordinator thread of socket 1");
  tp->started = 1;

  run_both(tp, CMD_SETUP);
  for (int s = 0; s < SOCKETS; s++) {
    int cpu = tp->s[s].leader_cpu, want = team_worker_cpu(tp->s[s].team, 0);
    LOG(LOG_DEBUG, "socket %d coordinator runs on cpu %d (team leader assigned cpu %d)", s, cpu, want);
    if (want >= 0 && cpu >= 0 && cpu != want)
      LOG(LOG_WARN, "socket %d coordinator runs on cpu %d, not on its team leader cpu %d", s, cpu, want);
  }
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
  if (pack != TP_PACK_NONE) {
    start = monotonic_seconds();
    plan_packing(tp);
    run_both(tp, CMD_PACK);
    for (int s = 0; s < SOCKETS; s++) {
      region_mark_populated(&tp->s[s].packed);
      region_protect(&tp->s[s].packed);
    }
    metrics.weight_pack_s += monotonic_seconds() - start;
    metrics.weight_packed = 1;
    LOG(LOG_INFO, "packed %zu + %zu bytes of %s shard copies in %.2f s", tp->s[0].packed.logical_bytes,
        tp->s[1].packed.logical_bytes, pack == TP_PACK_BF16 ? "BF16" : "int8", monotonic_seconds() - start);
  }
  start = monotonic_seconds();
  for (int s = 0; s < SOCKETS; s++) {
    verify_bound(&tp->s[s].weights, mode);
    verify_bound(&tp->s[s].packed, mode);
    verify_bound(&tp->s[s].work, mode);
    verify_bound(&tp->s[s].exchange, mode);
  }
  verify_bound(&tp->global, mode);
  metrics.placement_verify_s = monotonic_seconds() - start;
  metrics.placement_owned = strcmp(mode, "file") != 0;
  metrics.sockets = SOCKETS;
  return tp;
}

/* LLAMA_PROFILE=1: one line per stage and socket, in milliseconds per forward pass. wo/w2 are the whole projection
 * including its reduction, wo_mm/w2_mm the leader's own share of the projection, so the slowest member's lateness shows up
 * as `straggler`. The exchange components are the leader's: straggler (waiting for the socket's last partial write), peer
 * (waiting for the other socket), add, close (final barrier) and buffer (waiting for a free partial buffer). */
static void log_profile(const Tp *tp) {
  static const char *const names[TP_SLOTS] = {"embedding", "rope_table", "rmsnorm", "quantize", "qkv", "rope",
                                              "kv_store",  "attention",  "wo",      "w1_w3",    "w2",  "classifier",
                                              "wo_mm",     "w2_mm"};
  static const char *const xnames[5] = {"straggler", "peer", "add", "close", "buffer"};
  for (int stage = 0; stage < STAGE_COUNT; stage++) {
    if (!tp->tokens[stage])
      continue;
    for (int s = 0; s < SOCKETS; s++) {
      char line[1024];
      size_t used = (size_t)snprintf(line, sizeof(line), "tp %s socket %d (%zu tokens, ms/token):",
                                     stage == STAGE_PREFILL ? "prefill" : "decode", s, tp->tokens[stage]);
      for (int i = 0; i < TP_SLOTS && used < sizeof(line); i++)
        if (tp->prof[s][stage][i] > 0)
          used += (size_t)snprintf(line + used, sizeof(line) - used, " %s=%.3f", names[i],
                                   tp->prof[s][stage][i] * 1e3 / (double)tp->tokens[stage]);
      for (int i = 0; i < 5 && used < sizeof(line); i++)
        used += (size_t)snprintf(line + used, sizeof(line) - used, " x_%s=%.3f", xnames[i],
                                 tp->xch[s][stage][i] * 1e3 / (double)tp->tokens[stage]);
      LOG(LOG_INFO, "%s", line);
    }
  }
}

float *tp_forward(Tp *tp, int token, int pos) {
  tp->token = token;
  tp->pos = pos;
  run_both(tp, CMD_FORWARD);
  if (profile_enabled && !workers_aborted()) {
    tp->tokens[profile_stage]++;
    for (int s = 0; s < SOCKETS; s++) {
      double now[5];
      exchange_timing_detail(tp->exchange, s, now);
      for (int i = 0; i < 5; i++) {
        tp->xch[s][profile_stage][i] += now[i] - tp->xch_last[s][i];
        tp->xch_last[s][i] = now[i];
      }
    }
  }
  if (tp->check && !workers_aborted())
    for (int l = 0; l < tp->c.n_layers; l++)
      if (memcmp(tp->s[0].debug + (size_t)l * tp->c.dim, tp->s[1].debug + (size_t)l * tp->c.dim,
                 (size_t)tp->c.dim * sizeof(float)))
        fail("LLAMA_TP_CHECK: the residual copies of the two sockets differ after layer %d (position %d)", l, pos);
  return tp->logits;
}

float *tp_forward_batch(Tp *tp, const int *tokens, int count, int pos0, LogitsWanted want) {
  if (count < 1 || count > tp->rows || pos0 < 0 || pos0 > tp->c.seq_len - count)
    fail("two-socket batch out of range: count=%d (capacity %d) position=%d", count, tp->rows, pos0);
  if (want == LOGITS_ALL && tp->pack == TP_PACK_BF16 && !tp->cls_ready)
    pack_classifier(tp);
  if (want == LOGITS_ALL && !tp->batch_logits)
    tp->batch_logits = memory_alloc(checked_mul((size_t)tp->rows, (size_t)tp->c.vocab_size), sizeof(float),
                                    MEM_ACTIVATION);
  tp->btokens = tokens;
  tp->bcount = count;
  tp->bpos0 = pos0;
  tp->bwant = want;
  run_both(tp, CMD_FORWARD_BATCH);
  if (profile_enabled && !workers_aborted()) {
    tp->tokens[profile_stage] += (size_t)count;
    for (int s = 0; s < SOCKETS; s++) {
      double now[5];
      exchange_timing_detail(tp->exchange, s, now);
      for (int i = 0; i < 5; i++) {
        tp->xch[s][profile_stage][i] += now[i] - tp->xch_last[s][i];
        tp->xch_last[s][i] = now[i];
      }
    }
  }
  if (tp->check && !workers_aborted())
    for (int l = 0; l < tp->c.n_layers; l++)
      if (tp->s[0].debug_hash[l] != tp->s[1].debug_hash[l])
        fail("LLAMA_TP_CHECK: the residual copies of the two sockets differ after layer %d (positions %d..%d)", l,
             pos0, pos0 + count - 1);
  return want == LOGITS_NONE ? NULL : want == LOGITS_ALL ? tp->batch_logits : tp->logits;
}

void tp_kv_stats(Tp *tp, int position, int verify, KvStats *out) {
  size_t per_socket =
      checked_mul(checked_mul((size_t)tp->c.n_layers, (size_t)tp->c.seq_len), (size_t)tp->kv_dim_local * sizeof(float));
  *out = (KvStats){.reserved_bytes = checked_mul(checked_mul(2, SOCKETS), per_socket),
                   .active_bytes = checked_mul(checked_mul(checked_mul(2, SOCKETS), (size_t)tp->c.n_layers),
                                               checked_mul((size_t)position, (size_t)tp->kv_dim_local * sizeof(float))),
                   .node = tp->s[0].kv_k.node};
  if (!verify || tp->s[0].kv_k.node < 0)
    return;
  /* As runmem_kv_stats does for one socket: a page-placement query of both caches of both sockets (the pages that exist),
   * so the report can show that each socket's K and V sit on its own node. */
  const char *mode = placement_mode();
  size_t page = (size_t)sysconf(_SC_PAGESIZE);
  double resident = 0;
  out->full = 1;
  for (int s = 0; s < SOCKETS; s++) {
    NumaRegion *regions[2] = {&tp->s[s].kv_k, &tp->s[s].kv_v};
    for (int i = 0; i < 2; i++) {
      PageCheck check = {0};
      if (region_verify(regions[i], region_verify_samples_default(), &check) != 0) {
        if (placement_strict(mode))
          fail("page placement of the KV caches of socket %d on node %d cannot be verified on this system", s,
               regions[i]->node);
        out->full = 0;
        return;
      }
      size_t thp = 0;
      if (region_thp_bytes(regions[i], &thp) != 0)
        thp = 0;
      region_record_verification(regions[i], &check, thp);
      out->full &= check.full;
      out->pages_queried += check.queried;
      out->pages_wrong_node += check.wrong_node;
      out->pages_missing += check.missing;
      size_t pages = (regions[i]->logical_bytes + page - 1) / page, present = check.queried - check.missing - check.errors;
      resident += check.queried ? (double)present / (double)check.queried * (double)pages * (double)page : 0;
      if ((check.wrong_node || check.errors) && placement_strict(mode))
        fail("KV cache of socket %d: %zu of %zu queried pages are not on node %d (%zu elsewhere, %zu errors)", s,
             check.queried - check.on_node - check.missing, check.queried, regions[i]->node, check.wrong_node,
             check.errors);
    }
  }
  out->verified = 1;
  out->resident_bytes = (size_t)(resident + 0.5);
}

void tp_free(Tp *tp) {
  if (!tp)
    return;
  if (tp->started) {
    run_both(tp, CMD_STOP);
    pthread_join(tp->thread, NULL);
  }
  if (profile_enabled)
    log_profile(tp);
  exchange_destroy(tp->exchange);
  for (int s = 0; s < SOCKETS; s++) {
    TpSocket *sk = &tp->s[s];
    memory_free(sk->debug);
    memory_free(sk->debug_hash);
    region_release(&sk->weights);
    region_release(&sk->packed);
    region_release(&sk->packed_cls);
    memory_free(sk->packed_at);
    memory_free(tp->bjobs[s]);
    memory_free(tp->sjobs[s]);
    if (sk->panel.base)
      aligned_buffer_free(&sk->panel);
    region_release(&sk->kv_k);
    region_release(&sk->kv_v);
    region_release(&sk->work);
    region_release(&sk->exchange);
  }
  region_release(&tp->global);
  shards_free(&tp->set);
  memory_free(tp->logits);
  memory_free(tp->batch_logits);
  memory_free(tp);
}
