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

/* LLAMA_TP_TRACE=FILE: one record per Wo/W2 projection+reduction and socket, written to FILE as CSV when the engine is
 * freed (microseconds relative to the leader's dispatch of the team job). Needs the fused form. Every worker stamps the
 * end of its share of the projection and the steps of exchange_reduce_add (workers.h); the leader folds them into a
 * record after the join. The stamps cost a few clock reads per worker, so a traced run is slightly slower. */
typedef struct {
  _Alignas(64) double start, proj_end;
  ReduceStamps r;
  double pad[2];
} WorkerStamp;

typedef struct {
  int kind, stage;                   /* 0 = Wo, 1 = W2; Stage */
  double disp;                       /* leader's clock (seconds) when it dispatched the team job */
  double start_mean, start_max;      /* workers entering the job */
  double end_first, end_mean, end_last; /* workers finishing their share of the projection */
  double b1_leader, b1_last;         /* leaving the first barrier */
  double peer_leader, peer_last;     /* seeing the other socket's partial */
  double add_mean, add_last;         /* finishing the add */
  double b2_leader, b2_last;         /* leaving the closing barrier */
  double ret;                        /* team_run returned */
  int last_worker;                   /* the worker whose projection ended last */
} TraceRec;
enum { TRACE_MAX_RECORDS = 40000, TRACE_MAX_WORKERS = 256 };

typedef enum { CMD_SETUP, CMD_LOAD, CMD_PACK, CMD_PACK_CLS, CMD_REPACK, CMD_FORWARD, CMD_FORWARD_BATCH, CMD_STOP } Command;

/* The seven matrices of a layer that get a packed copy, in the order of TpSocket.packed_at. */
static const TensorRole PACK_ROLES[7] = {T_WQ, T_WK, T_WV, T_WO, T_W1, T_W2, T_W3};

typedef struct {
  int socket;
  Team *team;
  NumaRegion weights, kv_k, kv_v, work, exchange, packed, packed_cls, repack;
  float **repack_at;    /* [layer * 7 + PACK_ROLES index]: the packed4 copy (LLAMA_FP32_LAYOUT=packed4) used by decode, or NULL */
  void **packed_at;     /* [layer * 7 + PACK_ROLES index]: the packed copy in `packed`, or NULL */
  uint16_t *cls_bf16;   /* the packed classifier range, once all-position logits were asked for (BF16 mode) */
  AlignedBuffer panel;  /* BF16 mode: the activations converted to BF16 */
  float *x, *xb, *hb, *q, *att, *rope; /* socket-local activations, in the work region */
  float *key_cache, *value_cache;      /* [layer][seq][kv_dim / 2] */
  AttentionWork attention;
  float *debug; /* LLAMA_TP_CHECK: x after every layer */
  uint64_t *debug_hash; /* LLAMA_TP_CHECK, batched forward: a hash of x (all rows) after every layer */
  int leader_cpu; /* CPU the socket's coordinator ran on during setup (-1 unknown) */
  /* Q8 (B6) */
  int8_t *xq; /* quantized activations of the current projection input */
  float *xs;
  int8_t *aq; /* quantized slice that feeds Wo / W2 */
  float *as;
  /* LLAMA_TP_TRACE */
  WorkerStamp *stamps;
  TraceRec *records;
  size_t record_count;
  unsigned last_count[2][TRACE_MAX_WORKERS];
  double late_sum[2][TRACE_MAX_WORKERS];
  size_t reductions[2];
} TpSocket;

struct Tp {
  Config c;
  int quantized, gs, check, fp32_activations;
  const char *trace_path;
  int reduce_flags; /* LLAMA_TP_REDUCE=flags (default, fused only): per-row flags instead of two team barriers */
  int fused; /* LLAMA_TP_FUSED (default 1): the partial projection and the reduction run in one team job */
  ShardSet set;
  NumaRegion global;
  TpSocket s[SOCKETS];
  Exchange *exchange;
  float *logits;
  TpPack pack;
  int packed4; /* LLAMA_FP32_LAYOUT=packed4: decode streams a packed4 copy of the FP32 shards (the shards stay for the GEMM) */
  Pack4Job *rjobs[SOCKETS];
  int rjobs_n[SOCKETS];
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
  const float *wp;                  /* FP32, packed4 copy of w (decode); NULL: stream w by rows */
  const K8Input *input;             /* Q8 */
  const K8Weight *wq;               /* Q8; NULL for FP32 */
  double *leader_seconds;           /* fused: the leader's own share of the projection */
  atomic_int claim;                 /* fused: row counter shared by the members (zero before the team starts) */
  WorkerStamp *stamps;              /* fused, traced: one per member */
} ProjectJob;

static void rows_done_fn(void *p, int r0, int r1) {
  const ProjectJob *job = p;
  exchange_rows_done(job->tp->exchange, job->socket, r0, r1);
}

static void project_share(const ProjectJob *job, int worker, int workers) {
  RowsDoneFn done = job->tp->reduce_flags ? rows_done_fn : NULL;
  if (job->wq)
    k_matmul_q8_ld_worker(job->partial, job->input, job->wq, job->ld, job->n, job->d, job->gs, K_PARTIAL, worker,
                          workers, (atomic_int *)&job->claim, done, (void *)job);
  else if (job->wp)
    k_matmul_p4_worker(job->partial, job->x, job->wp, job->n, job->d, K_PARTIAL, worker, workers,
                       (atomic_int *)&job->claim, done, (void *)job);
  else
    k_matmul_ld_worker(job->partial, job->x, job->w, job->ld, job->n, job->d, K_PARTIAL, worker, workers,
                       (atomic_int *)&job->claim, done, (void *)job);
}

/* Fused form: every member computes its rows of the partial and goes straight into the reduction, whose first barrier
 * is the join of the projection. One team dispatch and join instead of two. */
static void project_reduce_fn(const void *p, int worker, int workers) {
  const ProjectJob *job = p;
  WorkerStamp *stamp = job->stamps ? &job->stamps[worker] : NULL;
  double start = worker == 0 && job->leader_seconds ? monotonic_seconds() : 0.0;
  if (stamp)
    stamp->start = start ? start : monotonic_seconds();
  project_share(job, worker, workers);
  if (stamp)
    stamp->proj_end = monotonic_seconds();
  if (start)
    *job->leader_seconds = (stamp ? stamp->proj_end : monotonic_seconds()) - start;
  if (job->tp->reduce_flags)
    exchange_reduce_add_flags(job->tp->exchange, job->socket, worker, workers, job->residual, job->d,
                              stamp ? &stamp->r : NULL);
  else
    exchange_reduce_add_stamped(job->tp->exchange, job->socket, worker, workers, job->residual, job->d,
                                stamp ? &stamp->r : NULL);
}

/* Folds the members' stamps of the reduction that just ended into one record of the socket. */
static void trace_record(Tp *tp, TpSocket *sk, const ProjectJob *job, int kind, double disp, double ret) {
  if (sk->record_count >= TRACE_MAX_RECORDS)
    return;
  int n = team_size(sk->team);
  TraceRec *r = &sk->records[sk->record_count++];
  *r = (TraceRec){.kind = kind, .stage = (int)profile_stage, .disp = disp, .ret = ret, .end_first = 1e30};
  const WorkerStamp *w = job->stamps;
  for (int i = 0; i < n; i++) {
    r->start_mean += w[i].start;
    r->end_mean += w[i].proj_end;
    r->add_mean += w[i].r.after_add;
    if (w[i].start > r->start_max)
      r->start_max = w[i].start;
    if (w[i].proj_end < r->end_first)
      r->end_first = w[i].proj_end;
    if (w[i].proj_end > r->end_last) {
      r->end_last = w[i].proj_end;
      r->last_worker = i;
    }
    if (w[i].r.after_barrier1 > r->b1_last)
      r->b1_last = w[i].r.after_barrier1;
    if (w[i].r.after_peer > r->peer_last)
      r->peer_last = w[i].r.after_peer;
    if (w[i].r.after_add > r->add_last)
      r->add_last = w[i].r.after_add;
    if (w[i].r.after_barrier2 > r->b2_last)
      r->b2_last = w[i].r.after_barrier2;
  }
  r->start_mean /= n;
  r->end_mean /= n;
  r->add_mean /= n;
  r->b1_leader = w[0].r.after_barrier1;
  r->peer_leader = w[0].r.after_peer;
  r->b2_leader = w[0].r.after_barrier2;
  sk->reductions[kind]++;
  sk->last_count[kind][r->last_worker < TRACE_MAX_WORKERS ? r->last_worker : 0]++;
  for (int i = 0; i < n && i < TRACE_MAX_WORKERS; i++)
    sk->late_sum[kind][i] += w[i].proj_end - r->end_mean;
  (void)tp;
}

static void project_reduce(Tp *tp, TpSocket *sk, ProjectJob *job, int slot) {
  job->tp = tp;
  atomic_init(&job->claim, 0);
  job->socket = sk->socket;
  job->residual = sk->x;
  double *prof = &tp->prof[sk->socket][profile_stage][slot];
  if (tp->fused) {
    double leader = 0.0;
    job->leader_seconds = profile_enabled ? &leader : NULL;
    job->stamps = sk->stamps;
    double disp = sk->stamps ? monotonic_seconds() : 0.0;
    team_run(sk->team, project_reduce_fn, job);
    if (tp->reduce_flags)
      exchange_reduce_finish(tp->exchange, sk->socket); /* the join above: every add of this generation is done */
    if (sk->stamps)
      trace_record(tp, sk, job, slot - TP_WO_MM, disp, monotonic_seconds());
    *prof += leader;
    return;
  }
  double start = profile_enabled ? monotonic_seconds() : 0.0;
  if (job->wq)
    k_matmul_q8_ld(job->partial, job->input, job->wq, job->ld, job->n, job->d, job->gs, K_PARTIAL);
  else if (job->wp)
    k_matmul_p4(job->partial, job->x, job->wp, job->n, job->d, K_PARTIAL);
  else
    k_matmul_ld(job->partial, job->x, job->w, job->ld, job->n, job->d, K_PARTIAL);
  if (profile_enabled)
    *prof += monotonic_seconds() - start;
  reduce(tp, sk, sk->x, job->d);
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

/* The packed4 copy of matrix `index` (PACK_ROLES order) of layer l on this socket, or NULL. */
static const float *repack_copy(const TpSocket *sk, int layer, int index) {
  return sk->repack_at ? sk->repack_at[(size_t)layer * 7 + (size_t)index] : NULL;
}

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
    const float *rq = repack_copy(sk, l, 0), *rk = repack_copy(sk, l, 1), *rv = repack_copy(sk, l, 2);
    if (rq && rk && rv)
      SPROFILE(PHASE_QKV, k_matmul_qkv_p4(sk->q, k, v, sk->xb, rq, rk, rv, dim, dim / SOCKETS, kvl));
    else
      SPROFILE(PHASE_QKV, k_matmul_qkv(sk->q, k, v, sk->xb, wq, wk, wv, dim, dim / SOCKETS, kvl));
    SPROFILE(PHASE_ROPE, k_rope_apply(sk->q, k, sk->rope, hl, kl, hd));
    SPROFILE(PHASE_ATTENTION, k_attention(sk->xb, sk->q, sk->key_cache + layer, sk->value_cache + layer, sk->att,
                                          &sk->attention, hl, kl, hd, p->seq_len, pos));
    const Shard *wo = shards_find(&tp->set, T_WO, l, s);
    float *partial = exchange_partial(tp->exchange, s);
    if (!partial)
      return;
    ProjectJob wo_job = {.n = dim / SOCKETS, .d = dim, .partial = partial, .ld = wo->ld, .x = sk->xb, .w = wo->q,
                         .wp = repack_copy(sk, l, 3)};
    SPROFILE(PHASE_WO, project_reduce(tp, sk, &wo_job, TP_WO_MM));

    SPROFILE(PHASE_RMSNORM, k_rmsnorm(sk->xb, sk->x, norm_weight(tp, T_RMS_FFN, l, s), dim, p->norm_eps));
    const float *w1 = shards_find(&tp->set, T_W1, l, s)->q, *w3 = shards_find(&tp->set, T_W3, l, s)->q;
    const float *r1 = repack_copy(sk, l, 4), *r3 = repack_copy(sk, l, 6);
    if (r1 && r3)
      SPROFILE(PHASE_W1_W3, k_matmul_swiglu_p4(sk->hb, sk->xb, r1, r3, dim, hidden_l));
    else
      SPROFILE(PHASE_W1_W3, k_matmul_swiglu(sk->hb, sk->xb, w1, w3, dim, hidden_l));
    const Shard *w2 = shards_find(&tp->set, T_W2, l, s);
    partial = exchange_partial(tp->exchange, s);
    if (!partial)
      return;
    ProjectJob w2_job = {.n = hidden_l, .d = dim, .partial = partial, .ld = w2->ld, .x = sk->hb, .w = w2->q,
                         .wp = repack_copy(sk, l, 5)};
    SPROFILE(PHASE_W2, project_reduce(tp, sk, &w2_job, TP_W2_MM));
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
  for (int i = 0; i < rows; i++)
    k_rmsnorm(out + (size_t)i * tp->c.dim, x + (size_t)i * tp->c.dim, weight, tp->c.dim, tp->c.norm_eps);
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
    const Shard *wq = shards_find(&tp->set, T_WQ, l, s), *wk = shards_find(&tp->set, T_WK, l, s),
                *wv = shards_find(&tp->set, T_WV, l, s);
    if (q8) {
      SPROFILE(PHASE_QUANTIZE, quantize_batch(tp, sk->xq, sk->xs, sk->xb, (size_t)dim, dim));
      GemmQ8Segment qkv[3] = {
          {.c = sk->q, .ldc = ql, .w = q8_weight(wq), .pw = packed_copy(sk, l, 0), .n = ql, .kind = GEMM_STORE},
          {.c = kc + (size_t)pos0 * kvl, .ldc = kvl, .w = q8_weight(wk), .pw = packed_copy(sk, l, 1), .n = kvl,
           .kind = GEMM_STORE},
          {.c = vc + (size_t)pos0 * kvl, .ldc = kvl, .w = q8_weight(wv), .pw = packed_copy(sk, l, 2), .n = kvl,
           .kind = GEMM_STORE}};
      SPROFILE(PHASE_QKV, k_gemm_q8(sk->xq, sk->xs, m, dim, gs, qkv, 3));
    } else {
      GemmSegment qkv[3] = {
          {.c = sk->q, .ldc = ql, .w = wq->q, .ldw = wq->ld, .n = ql, .kind = GEMM_STORE},
          {.c = kc + (size_t)pos0 * kvl, .ldc = kvl, .w = wk->q, .ldw = wk->ld, .n = kvl, .kind = GEMM_STORE},
          {.c = vc + (size_t)pos0 * kvl, .ldc = kvl, .w = wv->q, .ldw = wv->ld, .n = kvl, .kind = GEMM_STORE}};
      void *const pq[3] = {packed_copy(sk, l, 0), packed_copy(sk, l, 1), packed_copy(sk, l, 2)};
      SPROFILE(PHASE_QKV, gemm_f32_or_bf16(tp, sk, sk->xb, dim, dim, qkv, 3, pq, NULL));
    }
    for (int i = 0; i < m; i++)
      SPROFILE(PHASE_ROPE, k_rope_apply(sk->q + (size_t)i * ql, kc + (size_t)(pos0 + i) * kvl,
                                        sk->rope + (size_t)i * 2 * hd, hl, kl, hd));
    SPROFILE(PHASE_ATTENTION, k_attention_batch(sk->xb, sk->q, kc, vc, sk->att, &sk->attention, hl, kl, hd,
                                                p->seq_len, pos0, m));
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
    for (int i = 0; i < m; i++)
      SPROFILE(PHASE_RMSNORM, k_rmsnorm(sk->x + (size_t)i * dim, sk->x + (size_t)i * dim, final_norm, dim, p->norm_eps));
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

/* ---- sizing (D1 §4.2) -------------------------------------------------------------------------------------------- */

size_t tp_work_bytes(const Config *p, int quantized, int gs, int rows) {
  int hl = p->n_heads / SOCKETS, kl = p->n_kv_heads / SOCKETS;
  size_t slice = (size_t)(p->dim > p->hidden_dim ? p->dim : p->hidden_dim) / SOCKETS;
  size_t bytes = 64 * 16; /* alignment of each buffer */
  bytes += (size_t)rows * (size_t)(2 * p->dim + p->hidden_dim / SOCKETS + p->dim / SOCKETS + 2 * p->head_dim) *
           sizeof(float);
  bytes += (size_t)hl * p->seq_len * sizeof(float);
  if (quantized) { /* xq, xs, aq, as, for whole 32-row blocks */
    size_t padded = ((size_t)rows + 31) / 32 * 32;
    bytes += padded * ((size_t)p->dim + (size_t)p->dim / gs * sizeof(float) + slice + slice / gs * sizeof(float));
  }
  bytes += k_attention_bytes_rows(hl, kl, p->head_dim, p->seq_len, rows) + 64;
  return bytes;
}

size_t tp_exchange_bytes(const Config *p, int rows) {
  return exchange_region_bytes(rows * p->dim);
}

size_t tp_exchange_payload_bytes(const Config *p, int rows) {
  return (size_t)rows * (size_t)p->dim * sizeof(float);
}

size_t tp_peer_bytes_per_token(const Config *p) {
  return 2 * (size_t)p->n_layers * (size_t)p->dim * sizeof(float);
}

static double exchange_us(const Config *p, int rows, const TpSizingInput *in) {
  return 2.0 * (in->exchange_latency_us + (double)tp_exchange_payload_bytes(p, rows) / (in->exchange_gbps * 1e3));
}

double tp_exchange_fraction(const Config *p, int rows, const TpSizingInput *in) {
  double exchange = exchange_us(p, rows, in), compute = rows * in->layer_us_per_row;
  if (compute < in->layer_weight_us)
    compute = in->layer_weight_us; /* below the ridge a layer costs what streaming its shard costs */
  return exchange / (exchange + compute);
}

int tp_choose_batch(const Config *p, int quantized, int gs, const TpSizingInput *in, const int *candidates, int count,
                    TpSizingRow *out) {
  int chosen = 0;
  for (int i = 0; i < count; i++) {
    int m = candidates[i];
    TpSizingRow *row = &out[i];
    row->rows = m;
    row->work_bytes = tp_work_bytes(p, quantized, gs, m);
    row->exchange_bytes = tp_exchange_bytes(p, m);
    row->fits = m >= 1 && (size_t)m * (size_t)p->dim <= (size_t)INT_MAX &&
                row->work_bytes + row->exchange_bytes <= in->work_budget;
    row->ridge = m >= in->ridge;
    row->fraction = tp_exchange_fraction(p, m, in);
    row->exchange_ok = row->fraction <= in->max_exchange_fraction;
    row->feasible = row->fits && row->ridge && row->exchange_ok;
    if (row->feasible && m > chosen)
      chosen = m;
  }
  return chosen;
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
  case CMD_REPACK:
    k_pack4_many(tp->rjobs[socket], tp->rjobs_n[socket]);
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
static void check_budget(const Tp *tp, int s, const char *mode, size_t kv_bytes, size_t packed_bytes,
                         size_t repack_bytes) {
  size_t budget = placement_hbm_budget();
  if (strcmp(mode, "hbm"))
    return;
  size_t used = tp->set.region_bytes[s] + 2 * kv_bytes + work_bytes(tp) + tp_exchange_bytes(&tp->c, tp->rows) +
                packed_bytes + repack_bytes;
  if (used > budget)
    fail("socket %d needs %zu bytes of HBM (%zu of shards, %zu of packed %s copies, %zu of packed4 decode copies, %zu of "
         "KV caches) but LLAMA_HBM_BUDGET_GIB allows %zu per socket",
         s, used, tp->set.region_bytes[s], packed_bytes, tp->pack == TP_PACK_BF16 ? "BF16" : "int8", repack_bytes,
         2 * kv_bytes, budget);
}

/* ---- packed4 decode copies (B7) ------------------------------------------------------------------------------------ */

/* Whether the shard can be packed in four-row groups. */
static int repackable_shape(const Shard *sh) {
  return sh && (sh->kind == SHARD_ROWS || sh->kind == SHARD_COLS) && sh->rows > 0 && sh->rows % 4 == 0 && sh->cols > 0;
}

/* What socket s's repack region must hold: its shard of every matrix of every layer that can be packed. */
static size_t repack_region_bytes(const Tp *tp, int s) {
  size_t total = 0;
  for (int l = 0; l < tp->c.n_layers; l++)
    for (int i = 0; i < 7; i++) {
      const Shard *sh = shards_find(&tp->set, PACK_ROLES[i], l, s);
      if (repackable_shape(sh))
        total = align_line(total) + k_pack4_floats(sh->rows, sh->cols) * sizeof(float);
    }
  return total;
}

/* Main thread, after the shards are bound: carves the region and plans the packing jobs the coordinators run. */
static void plan_repack(Tp *tp) {
  for (int s = 0; s < SOCKETS; s++) {
    TpSocket *sk = &tp->s[s];
    size_t count = (size_t)tp->c.n_layers * 7, used = 0;
    sk->repack_at = memory_alloc(count, sizeof(float *), MEM_METADATA);
    tp->rjobs[s] = memory_alloc(count, sizeof(Pack4Job), MEM_METADATA);
    for (int l = 0; l < tp->c.n_layers; l++)
      for (int i = 0; i < 7; i++) {
        const Shard *sh = shards_find(&tp->set, PACK_ROLES[i], l, s);
        if (!repackable_shape(sh) || !sh->q)
          continue;
        used = align_line(used);
        float *dst = (float *)((char *)sk->repack.data + used);
        used += k_pack4_floats(sh->rows, sh->cols) * sizeof(float);
        sk->repack_at[(size_t)l * 7 + (size_t)i] = dst;
        tp->rjobs[s][tp->rjobs_n[s]++] = (Pack4Job){dst, sh->q, sh->rows, sh->cols, sh->ld};
      }
    if (used > sk->repack.logical_bytes)
      fail("socket %d repack region holds %zu bytes, %zu needed", s, sk->repack.logical_bytes, used);
  }
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
  const char *fused = getenv("LLAMA_TP_FUSED");
  if (fused && *fused && strcmp(fused, "0") && strcmp(fused, "1"))
    fail("LLAMA_TP_FUSED must be 0 or 1");
  tp->fused = !(fused && !strcmp(fused, "0"));
  const char *reduce = getenv("LLAMA_TP_REDUCE");
  if (reduce && *reduce && strcmp(reduce, "flags") && strcmp(reduce, "barrier"))
    fail("LLAMA_TP_REDUCE must be flags or barrier");
  tp->reduce_flags = tp->fused && !(reduce && !strcmp(reduce, "barrier"));
  const char *layout = getenv("LLAMA_FP32_LAYOUT");
  if (layout && *layout && strcmp(layout, "rows") && strcmp(layout, "packed4"))
    fail("LLAMA_FP32_LAYOUT must be rows or packed4");
  tp->packed4 = layout && !strcmp(layout, "packed4");
  if (tp->packed4 && quantized)
    fail("LLAMA_FP32_LAYOUT=packed4 applies to FP32 checkpoints");
  const char *trace = getenv("LLAMA_TP_TRACE");
  tp->trace_path = trace && *trace ? trace : NULL;
  if (tp->trace_path && !tp->fused)
    fail("LLAMA_TP_TRACE needs the fused reduction (LLAMA_TP_FUSED=1)");
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
    size_t repack_bytes = tp->packed4 ? repack_region_bytes(tp, s) : 0;
    check_budget(tp, s, mode, kv_bytes, packed_bytes, repack_bytes);
    if (packed_bytes)
      region_reserve(&sk->packed, packed_bytes, s, node, ROLE_PACKED, thp);
    if (repack_bytes)
      region_reserve(&sk->repack, repack_bytes, s, node, ROLE_PACKED, thp);
    region_reserve(&sk->work, work_bytes(tp), s, node, ROLE_WORK, 0);
    region_reserve(&sk->exchange, exchange_region_bytes(tp->rows * p->dim), s, node, ROLE_WORK, 0);
    exchange_regions[s] = &sk->exchange;
    region_mark_populated(&sk->kv_k);
    region_mark_populated(&sk->kv_v);
    region_mark_populated(&sk->work);
    if (tp->trace_path) {
      int n = team_size(sk->team);
      if (n > TRACE_MAX_WORKERS)
        fail("LLAMA_TP_TRACE supports at most %d workers per socket", TRACE_MAX_WORKERS);
      if (posix_memalign((void **)&sk->stamps, 64, (size_t)n * sizeof(WorkerStamp)))
        fail("cannot allocate the trace stamps");
      sk->records = calloc(TRACE_MAX_RECORDS, sizeof(TraceRec));
      if (!sk->records)
        fail("cannot allocate the trace records");
    }
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
  if (tp->packed4) {
    start = monotonic_seconds();
    plan_repack(tp);
    run_both(tp, CMD_REPACK);
    for (int s = 0; s < SOCKETS; s++)
      if (tp->s[s].repack.mapping_base) {
        region_mark_populated(&tp->s[s].repack);
        region_protect(&tp->s[s].repack);
      }
    metrics.weight_pack_s += monotonic_seconds() - start;
    metrics.weight_packed = 1;
    LOG(LOG_INFO, "packed4 decode copies: %zu + %zu bytes in %.2f s", tp->s[0].repack.logical_bytes,
        tp->s[1].repack.logical_bytes, monotonic_seconds() - start);
  }
  start = monotonic_seconds();
  for (int s = 0; s < SOCKETS; s++) {
    verify_bound(&tp->s[s].weights, mode);
    verify_bound(&tp->s[s].packed, mode);
    verify_bound(&tp->s[s].repack, mode);
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
 * including its reduction, wo_mm/w2_mm the projection alone (unfused: its team job; fused: the leader's own share, so the
 * slowest member's lateness shows up as `straggler`). The exchange components are the leader's: straggler (waiting for
 * the socket's last partial write), peer (waiting for the other socket), add, close (final barrier) and buffer (waiting
 * for a free partial buffer). */
static void write_trace(const Tp *tp) {
  FILE *f = fopen(tp->trace_path, "w");
  if (!f) {
    LOG(LOG_WARN, "cannot write the reduction trace %s", tp->trace_path);
    return;
  }
  fprintf(f, "socket,stage,kind,index,disp_s,start_mean,start_max,end_first,end_mean,end_last,b1_leader,b1_last,"
             "peer_leader,peer_last,add_mean,add_last,b2_leader,b2_last,ret,last_worker\n");
  for (int s = 0; s < SOCKETS; s++)
    for (size_t i = 0; i < tp->s[s].record_count; i++) {
      const TraceRec *r = &tp->s[s].records[i];
#define US(field) ((r->field - r->disp) * 1e6)
      fprintf(f, "%d,%d,%d,%zu,%.9f,%.2f,%.2f,%.2f,%.2f,%.2f,%.2f,%.2f,%.2f,%.2f,%.2f,%.2f,%.2f,%.2f,%.2f,%d\n", s,
              r->stage, r->kind, i, r->disp, US(start_mean), US(start_max), US(end_first), US(end_mean), US(end_last),
              US(b1_leader), US(b1_last), US(peer_leader), US(peer_last), US(add_mean), US(add_last), US(b2_leader),
              US(b2_last), US(ret), r->last_worker);
#undef US
    }
  fclose(f);
  LOG(LOG_INFO, "reduction trace written to %s", tp->trace_path);
  /* The members that finish their share last most often, and how late the usual suspects are. */
  for (int s = 0; s < SOCKETS; s++)
    for (int kind = 0; kind < 2; kind++) {
      const TpSocket *sk = &tp->s[s];
      if (!sk->reductions[kind])
        continue;
      int n = team_size(sk->team), top[3] = {-1, -1, -1};
      for (int w = 0; w < n && w < TRACE_MAX_WORKERS; w++)
        for (int t = 0; t < 3; t++)
          if (top[t] < 0 || sk->last_count[kind][w] > sk->last_count[kind][top[t]]) {
            for (int u = 2; u > t; u--)
              top[u] = top[u - 1];
            top[t] = w;
            break;
          }
      char line[512];
      size_t used = (size_t)snprintf(line, sizeof(line), "trace socket %d %s (%zu reductions): last to finish the projection:", s,
                                     kind ? "w2" : "wo", sk->reductions[kind]);
      for (int t = 0; t < 3; t++)
        if (top[t] >= 0)
          used += (size_t)snprintf(line + used, sizeof(line) - used, " worker %d %.1f%% (mean %+.2f us)", top[t],
                                   100.0 * sk->last_count[kind][top[t]] / (double)sk->reductions[kind],
                                   sk->late_sum[kind][top[t]] / (double)sk->reductions[kind] * 1e6);
      LOG(LOG_INFO, "%s", line);
    }
}

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
      size_t used = (size_t)snprintf(line, sizeof(line), "tp %s socket %d (%zu tokens, ms/token, fused=%d, reduce=%s):",
                                     stage == STAGE_PREFILL ? "prefill" : "decode", s, tp->tokens[stage], tp->fused,
                                     tp->reduce_flags ? "flags" : "barrier");
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

int tp_batch_capacity(const Tp *tp) {
  return tp->rows;
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
  if (profile_enabled)
    log_profile(tp);
  if (tp->trace_path)
    write_trace(tp);
  exchange_destroy(tp->exchange);
  for (int s = 0; s < SOCKETS; s++) {
    TpSocket *sk = &tp->s[s];
    memory_free(sk->debug);
    memory_free(sk->debug_hash);
    free(sk->stamps);
    free(sk->records);
    region_release(&sk->weights);
    region_release(&sk->packed);
    region_release(&sk->packed_cls);
    region_release(&sk->repack);
    memory_free(sk->repack_at);
    memory_free(tp->rjobs[s]);
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
