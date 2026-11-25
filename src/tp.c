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

/* One layer's K and V caches, and with LLAMA_ATTN=amx under planned placement its persistent BF16 copy. Every access to
 * the caches goes through kv_layer(): a plan can split the layers over two tiers, so a layer's offset in one region is
 * not `layer * seq_len * kv_dim`. */
typedef struct {
  float *k, *v;
  uint16_t *bf16;
} KvLayer;

typedef struct {
  int socket;
  Team *team;
  KvLayer *kv_layers;
  void *load_scratch;
  NumaRegion weights, kv_k, kv_v, work, exchange, packed, packed_cls;
  void *work_data, *packed_data; /* where the activations and the packed copies are carved from (a region of this
                                    socket, or the one-socket loader's run memory and packed store) */
  size_t work_size, packed_size;
  void **packed_at;     /* [layer * 7 + PACK_ROLES index]: the packed copy in `packed`, or NULL */
  float **pscales_at;   /* the same index: the tiled Q8 kernel (LLAMA_Q8_GEMM=skewed|pipelined), the transposed scales after the packed copy */
  uint16_t *cls_bf16;   /* the packed classifier range, once all-position logits were asked for (BF16 mode) */
  AlignedBuffer panel;  /* BF16 mode: the activations converted to BF16 */
  float *x, *xb, *hb, *q, *att, *rope; /* socket-local activations, in the work region */
  /* kv_layers[l]: layer l's caches, [seq][kv_dim / 2]; LLAMA_KV_LAYOUT=head: [kv head][seq][head_dim] */
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

enum { HBM, DDR, TIERS };
typedef enum {
  A_WEIGHTS_HBM,
  A_WEIGHTS_DDR,
  A_K_HBM,
  A_K_DDR,
  A_V_HBM,
  A_V_DDR,
  A_KVB_HBM, /* the persistent BF16 K/V copy of LLAMA_ATTN=amx, tiered per layer like the caches */
  A_KVB_DDR,
  A_WORK,
  A_EXCHANGE,
  A_EMBED,
  A_LOGITS,
  A_PACK,
  A_PACK_CLS,
  A_LOAD_SCRATCH,
  A_COUNT
} Arena;

typedef struct {
  size_t bytes, rounded;
  int tier, node, thp;
  RegionRole role;
} ArenaPlan;

typedef struct {
  const char *mode;
  int sockets, simulated, admitted, weights_layers, kv_layers, classifier, packed;
  int threads[SOCKETS], node[SOCKETS][TIERS];
  size_t budget[SOCKETS][TIERS], free_bytes[SOCKETS][TIERS], charge[SOCKETS][TIERS], job_cap;
  size_t kv_layer_bytes, kv_bf16_layer_bytes, work_bytes[SOCKETS], panel_bytes;
  size_t logits_bytes;
  ArenaPlan arena[SOCKETS][A_COUNT];
} PlacementPlan;

/* The report outlives the model. Keep only the size and placement metadata. */
static PlacementPlan placement_summary;

struct Tp {
  Config c;
  int sockets; /* 1: the one-socket engine (tp_create_local); 2: tensor parallel over two sockets (tp_create) */
  int quantized, gs, check, fp32_activations;
  int tiled; /* LLAMA_Q8_GEMM=skewed|pipelined on packed int8: activations quantized in tile order, scales transposed */
  /* One socket: the run state and packed copies live where the one-socket loader places them (A4, A5, C5). */
  RunMemory mem;
  PackedStore local_packed, local_cls;
  ShardSet set;
  NumaRegion global;
  PlacementPlan plan;
  NumaRegion arenas[SOCKETS][A_COUNT];
  int batch_logits_rows;
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
  double xmat[SOCKETS][STAGE_COUNT][2][5]; /* the batched exchange components of Wo ([0]) and W2 ([1]) apart */
  size_t tokens[STAGE_COUNT];
  Command command;
  int token, pos, want_logits, loader_fd;
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
  if (tp->sockets == 1) { /* no coordinator thread: socket 0 is the calling thread */
    if (command != CMD_STOP)
      socket_command(tp, 0);
    return;
  }
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
    for (int s = 0; s < tp->sockets; s++) {
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
    for (int s = 0; s < tp->sockets; s++) {
      const Shard *sh = shards_find(&tp->set, T_WCLS, -1, s);
      if (token >= sh->row0 && token < sh->row0 + sh->rows)
        return (const float *)sh->q + (size_t)(token - sh->row0) * tp->c.dim;
    }
    fail("token %d is outside the vocabulary shards", token);
  }
  const Shard *embed = shards_find(&tp->set, T_EMBED, -1, 0);
  return (const float *)embed->q + (size_t)token * tp->c.dim;
}

/* The INT8 group of a product with input width k (per socket): the checkpoint's group size, or with per-channel INT8
 * (gs 0: one weight scale per output channel, one activation scale per token) the whole of k. */
static int tp_group(const Tp *tp, int k) {
  return tp->gs ? tp->gs : k;
}

static K8Weight q8_weight(const Shard *sh) {
  return (K8Weight){(const int8_t *)sh->q, sh->s};
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

/* Wo or W2 for one token, out += W x into the residual: on one socket the one-socket kernel itself (k_matmul_add,
 * k_matmul_q8_add); on two, this socket's column share into the exchange, reduced into both residuals. FP32 reads x,
 * Q8 the prepared input `in`. Returns 0 when the workers aborted. */
static int project_residual(Tp *tp, int s, Phase phase, int slot, const Shard *w, const float *x, const K8Input *in,
                            int n) {
  TpSocket *sk = &tp->s[s];
  int dim = tp->c.dim;
  if (tp->sockets == 1) {
    if (tp->quantized)
      SPROFILE(phase, k_matmul_q8_add(sk->x, in, q8_weight(w), n, dim, tp_group(tp, n)));
    else
      SPROFILE(phase, k_matmul_add(sk->x, x, w->q, n, dim));
    return 1;
  }
  float *partial = exchange_partial(tp->exchange, s);
  if (!partial)
    return 0;
  K8Weight wq = tp->quantized ? q8_weight(w) : (K8Weight){0};
  ProjectJob job = {.n = n,
                    .d = dim,
                    .gs = tp_group(tp, n),
                    .partial = partial,
                    .ld = w->ld,
                    .x = x,
                    .w = w->q,
                    .input = in,
                    .wq = tp->quantized ? &wq : NULL};
  SPROFILE(phase, project_reduce(tp, sk, &job, slot));
  return 1;
}

static KvLayer kv_layer(const TpSocket *sk, int layer) {
  return sk->kv_layers[layer];
}

static void forward_fp32(Tp *tp, int s) {
  TpSocket *sk = &tp->s[s];
  const Config *p = &tp->c;
  int S = tp->sockets, dim = p->dim, hd = p->head_dim, hl = p->n_heads / S, kl = p->n_kv_heads / S, kvl = kl * hd;
  int hidden_l = p->hidden_dim / S, token = tp->token, pos = tp->pos;
  SPROFILE(PHASE_EMBED, memcpy(sk->x, embedding_row(tp, token), (size_t)dim * sizeof(float)));
  SPROFILE(PHASE_ROPE_TABLE, k_rope_table(p, pos, sk->rope));
  for (int l = 0; l < p->n_layers; l++) {
    SPROFILE(PHASE_RMSNORM, k_rmsnorm(sk->xb, sk->x, norm_weight(tp, T_RMS_ATT, l, s), dim, p->norm_eps));
    KvLayer cache = kv_layer(sk, l);
    int head_layout = kernels_kv_layout_head();
    float *k = head_layout ? sk->kstage : cache.k + (size_t)pos * kvl,
          *v = head_layout ? sk->vstage : cache.v + (size_t)pos * kvl;
    const float *wq = shards_find(&tp->set, T_WQ, l, s)->q, *wk = shards_find(&tp->set, T_WK, l, s)->q,
                *wv = shards_find(&tp->set, T_WV, l, s)->q;
    SPROFILE(PHASE_QKV, k_matmul_qkv(sk->q, k, v, sk->xb, wq, wk, wv, dim, dim / S, kvl));
    SPROFILE(PHASE_ROPE, k_rope_apply(sk->q, k, sk->rope, hl, kl, hd));
    if (head_layout)
      SPROFILE(PHASE_KV_STORE, k_kv_store(cache.k, k, (size_t)kvl, pos, 1, kl, hd, p->seq_len);
               k_kv_store(cache.v, v, (size_t)kvl, pos, 1, kl, hd, p->seq_len));
    SPROFILE(PHASE_ATTENTION,
             k_attention_kv(sk->xb, sk->q, cache.k, cache.v, sk->att, &sk->attention, hl, kl, hd, p->seq_len, pos));
    if (!project_residual(tp, s, PHASE_WO, TP_WO_MM, shards_find(&tp->set, T_WO, l, s), sk->xb, NULL, dim / S))
      return;

    SPROFILE(PHASE_RMSNORM, k_rmsnorm(sk->xb, sk->x, norm_weight(tp, T_RMS_FFN, l, s), dim, p->norm_eps));
    const float *w1 = shards_find(&tp->set, T_W1, l, s)->q, *w3 = shards_find(&tp->set, T_W3, l, s)->q;
    SPROFILE(PHASE_W1_W3, k_matmul_swiglu(sk->hb, sk->xb, w1, w3, dim, hidden_l));
    if (!project_residual(tp, s, PHASE_W2, TP_W2_MM, shards_find(&tp->set, T_W2, l, s), sk->hb, NULL, hidden_l))
      return;
    if (sk->debug)
      memcpy(sk->debug + (size_t)l * dim, sk->x, (size_t)dim * sizeof(float));
  }
  if (!tp->want_logits)
    return;
  SPROFILE(PHASE_RMSNORM, k_rmsnorm(sk->x, sk->x, norm_weight(tp, T_RMS_FINAL, -1, s), dim, p->norm_eps));
  const Shard *cls = shards_find(&tp->set, T_WCLS, -1, s);
  SPROFILE(PHASE_CLASSIFIER, k_matmul(tp->logits + cls->row0, sk->x, cls->q, dim, cls->rows));
}

/* Q8_0 (B6): the checkpoint's group size and scales and the Q8_0 activation rule (W8A8, or FP32 inputs for W8A32). Row
 * shards quantize the full normalized input, which is identical on both sockets; Wo and W2 quantize only the owned
 * slice, which starts on a group boundary, so its groups are exactly the original groups. Per-channel INT8 has one
 * group per input: each socket's Wo/W2 slice gets its own per-token scale (as a tensor-parallel rank quantizes its
 * shard), so two sockets agree with one within the W8A8 noise, not bitwise. */
static K8Input q8_input(const Tp *tp, int8_t *q, float *scales, const float *x, int n) {
  if (tp->fp32_activations)
    return (K8Input){NULL, NULL, x};
  k_quantize(q, scales, x, n, tp_group(tp, n));
  return (K8Input){q, scales, x};
}

static void forward_q8(Tp *tp, int s) {
  TpSocket *sk = &tp->s[s];
  const Config *p = &tp->c;
  int S = tp->sockets, dim = p->dim, hd = p->head_dim, gs = tp_group(tp, dim), hl = p->n_heads / S,
      kl = p->n_kv_heads / S, kvl = kl * hd;
  int hidden_l = p->hidden_dim / S, token = tp->token, pos = tp->pos;
  int row;
  const Shard *embed = embedding_shard(tp, token, &row);
  SPROFILE(PHASE_EMBED, for (int i = 0; i < dim; i++) sk->x[i] =
                            ((const int8_t *)embed->q)[(size_t)row * dim + i] * embed->s[((size_t)row * dim + i) / gs]);
  SPROFILE(PHASE_ROPE_TABLE, k_rope_table(p, pos, sk->rope));
  for (int l = 0; l < p->n_layers; l++) {
    SPROFILE(PHASE_RMSNORM, k_rmsnorm(sk->xb, sk->x, norm_weight(tp, T_RMS_ATT, l, s), dim, p->norm_eps));
    K8Input in;
    SPROFILE(PHASE_QUANTIZE, in = q8_input(tp, sk->xq, sk->xs, sk->xb, dim));
    KvLayer cache = kv_layer(sk, l);
    int head_layout = kernels_kv_layout_head();
    float *k = head_layout ? sk->kstage : cache.k + (size_t)pos * kvl,
          *v = head_layout ? sk->vstage : cache.v + (size_t)pos * kvl;
    K8Weight wq = q8_weight(shards_find(&tp->set, T_WQ, l, s)), wk = q8_weight(shards_find(&tp->set, T_WK, l, s)),
             wv = q8_weight(shards_find(&tp->set, T_WV, l, s));
    SPROFILE(PHASE_QKV, k_matmul_q8_qkv(sk->q, k, v, &in, wq, wk, wv, dim, dim / S, kvl, gs));
    SPROFILE(PHASE_ROPE, k_rope_apply(sk->q, k, sk->rope, hl, kl, hd));
    if (head_layout)
      SPROFILE(PHASE_KV_STORE, k_kv_store(cache.k, k, (size_t)kvl, pos, 1, kl, hd, p->seq_len);
               k_kv_store(cache.v, v, (size_t)kvl, pos, 1, kl, hd, p->seq_len));
    SPROFILE(PHASE_ATTENTION,
             k_attention_kv(sk->xb, sk->q, cache.k, cache.v, sk->att, &sk->attention, hl, kl, hd, p->seq_len, pos));
    SPROFILE(PHASE_QUANTIZE, in = q8_input(tp, sk->aq, sk->as, sk->xb, dim / S));
    if (!project_residual(tp, s, PHASE_WO, TP_WO_MM, shards_find(&tp->set, T_WO, l, s), sk->xb, &in, dim / S))
      return;

    SPROFILE(PHASE_RMSNORM, k_rmsnorm(sk->xb, sk->x, norm_weight(tp, T_RMS_FFN, l, s), dim, p->norm_eps));
    SPROFILE(PHASE_QUANTIZE, in = q8_input(tp, sk->xq, sk->xs, sk->xb, dim));
    K8Weight w1 = q8_weight(shards_find(&tp->set, T_W1, l, s)), w3 = q8_weight(shards_find(&tp->set, T_W3, l, s));
    SPROFILE(PHASE_W1_W3, k_matmul_q8_swiglu(sk->hb, &in, w1, w3, dim, hidden_l, gs));
    SPROFILE(PHASE_QUANTIZE, in = q8_input(tp, sk->aq, sk->as, sk->hb, hidden_l));
    if (!project_residual(tp, s, PHASE_W2, TP_W2_MM, shards_find(&tp->set, T_W2, l, s), sk->hb, &in, hidden_l))
      return;
    if (sk->debug)
      memcpy(sk->debug + (size_t)l * dim, sk->x, (size_t)dim * sizeof(float));
  }
  if (!tp->want_logits)
    return;
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
typedef struct {
  const Tp *tp;
  TpSocket *sk;
} EmbedJob;

/* Rows [first, last) of the batch's embeddings: a copy, or the dequantized Q8 row (the per-token path's arithmetic). */
static void embed_chunks(const void *p, int segment, int first, int last) {
  (void)segment;
  const EmbedJob *job = p;
  const Tp *tp = job->tp;
  size_t dim = (size_t)tp->c.dim;
  for (int i = first; i < last; i++) {
    float *x = job->sk->x + (size_t)i * dim;
    if (!tp->quantized) {
      memcpy(x, embedding_row(tp, tp->btokens[i]), dim * sizeof(float));
      continue;
    }
    int row;
    const Shard *embed = embedding_shard(tp, tp->btokens[i], &row);
    for (size_t j = 0; j < dim; j++)
      x[j] = ((const int8_t *)embed->q)[(size_t)row * dim + j] *
             embed->s[((size_t)row * dim + j) / (size_t)tp_group(tp, (int)dim)];
  }
}

static void embed_rows(const Tp *tp, TpSocket *sk) {
  EmbedJob job = {tp, sk};
  int rows = tp->bcount;
  kernels_parallel_segments(1, &rows, embed_chunks, &job);
}

static void rope_chunks(const void *p, int segment, int first, int last) {
  (void)segment;
  const EmbedJob *job = p;
  for (int i = first; i < last; i++)
    k_rope_table(&job->tp->c, job->tp->bpos0 + i, job->sk->rope + (size_t)i * 2 * job->tp->c.head_dim);
}

static void rope_tables(const Tp *tp, TpSocket *sk) {
  EmbedJob job = {tp, sk};
  int rows = tp->bcount;
  kernels_parallel_segments(1, &rows, rope_chunks, &job);
}

static void rmsnorm_rows(const Tp *tp, float *out, const float *x, const float *weight, int rows) {
  k_rmsnorm_rows(out, x, weight, rows, tp->c.dim, tp->c.norm_eps);
}

/* The packed copy of matrix `index` (PACK_ROLES order) of layer l on this socket, or NULL. */
static void *packed_copy(const TpSocket *sk, int layer, int index) {
  return sk->packed_at ? sk->packed_at[(size_t)layer * 7 + (size_t)index] : NULL;
}

/* Its transposed scales (LLAMA_Q8_GEMM=skewed|pipelined), or NULL. */
static const float *packed_scales(const TpSocket *sk, int layer, int index) {
  return sk->pscales_at ? sk->pscales_at[(size_t)layer * 7 + (size_t)index] : NULL;
}

/* The W8A8 product on this batch's quantized activations: tile order for the pipelined kernel, else rows. */
static void gemm_q8(const Tp *tp, const int8_t *xq, const float *xs, int k, const GemmQ8Segment *seg, int count) {
  if (tp->tiled)
    k_gemm_q8_tiled(xq, xs, tp->bcount, k, tp_group(tp, k), seg, count);
  else
    k_gemm_q8(xq, xs, tp->bcount, k, tp_group(tp, k), seg, count);
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

/* LLAMA_TP_BATCH_REDUCE: 0 barrier, 1 wo (Wo interleaved, W2 barrier), 2 interleaved (both). */
static int batch_reduce_mode(void) {
  static int mode = -1;
  if (mode < 0) {
    const char *value = getenv("LLAMA_TP_BATCH_REDUCE");
    if (value && *value && strcmp(value, "interleaved") && strcmp(value, "wo") && strcmp(value, "barrier"))
      fail("LLAMA_TP_BATCH_REDUCE must be interleaved, wo or barrier");
    mode = !value || !*value || !strcmp(value, "interleaved") ? 2 : !strcmp(value, "wo") ? 1 : 0;
  }
  return mode;
}

const char *tp_batch_reduce_name(void) {
  static const char *const names[] = {"barrier", "wo", "interleaved"};
  return names[batch_reduce_mode()];
}

/* LLAMA_TP_BATCH_REDUCE=interleaved: the reduction of a W8A8 Wo/W2 partial runs inside the projection's team job. Every
 * worker flags each block of the partial it stored (GemmQ8Hooks.stored) and, between its compute units, adds up to
 * INTERLEAVE_BUDGET of its own stored blocks whose peer block is stored too (residual += p0 + p1, the operand order of
 * exchange_reduce_add, so the residual is bitwise the barrier form's), prefetching the next one; after its range it
 * waits for and adds the rest. A block becomes a candidate only INTERLEAVE_LAG blocks after the worker stored it, so
 * the peer has usually stored it too and one flag read suffices (lags 0-16 measured, job 5897997: 4 adds and reads
 * least). The cross-socket reads then spread over the projection instead of competing for the UPI links all at once
 * after it (Wo add 21 ms per 2048-token prefill, job 5892710). */
enum { INTERLEAVE_BUDGET = 2, INTERLEAVE_LAG = 4 };

typedef struct {
  Exchange *e;
  int socket, m, n, ld;
  float *residual;
  const float *p0, *p1;
  GemmQ8Hooks hooks;
  int timed;                /* LLAMA_PROFILE: the worker whose range starts at block 0 records its share */
  double check, add, wait, finish; /* seconds checking peer flags during the range, adding, waiting for peer blocks
                                    * after the range; when its range was done */
} Interleave;

static void interleave_rect(const Interleave *iv, int t, int *m0, int *mb, int *n0, int *nc) {
  *m0 = t % iv->hooks.mblocks * 32;
  *n0 = t / iv->hooks.mblocks * iv->hooks.chunk_n;
  *mb = iv->m - *m0 < 32 ? iv->m - *m0 : 32;
  *nc = iv->n - *n0 < iv->hooks.chunk_n ? iv->n - *n0 : iv->hooks.chunk_n;
}

static void interleave_add(const Interleave *iv, int t) {
  int m0, mb, n0, nc;
  interleave_rect(iv, t, &m0, &mb, &n0, &nc);
  for (int r = 0; r < mb; r++) {
    size_t at = (size_t)(m0 + r) * (size_t)iv->ld + (size_t)n0;
    float *x = iv->residual + at;
    const float *a = iv->p0 + at, *b = iv->p1 + at;
    for (int c = 0; c < nc; c++)
      x[c] += a[c] + b[c];
  }
}

/* The peer's block and the residual's, toward this core's caches, while the next compute unit runs. */
static void interleave_prefetch(const Interleave *iv, int t) {
  int m0, mb, n0, nc;
  interleave_rect(iv, t, &m0, &mb, &n0, &nc);
  const float *peer = iv->socket ? iv->p0 : iv->p1;
  for (int r = 0; r < mb; r++) {
    size_t at = (size_t)(m0 + r) * (size_t)iv->ld + (size_t)n0;
    for (int c = 0; c < nc; c += 16) {
      __builtin_prefetch(peer + at + c, 0, 3);
      __builtin_prefetch(iv->residual + at + c, 1, 3);
    }
  }
}

static void interleave_stored(void *ctx, int t0, int t1) {
  const Interleave *iv = ctx;
  exchange_blocks_done(iv->e, iv->socket, t0, t1);
}

static int interleave_turn(void *ctx, int first, int next, int done, int final) {
  Interleave *iv = ctx;
  int timed = iv->timed && first == 0;
  double start = timed ? monotonic_seconds() : 0.0;
  if (!final) {
    for (int i = 0; i < INTERLEAVE_BUDGET && next < done - INTERLEAVE_LAG; i++) {
      double c0 = timed ? monotonic_seconds() : 0.0;
      int ready = exchange_block_ready(iv->e, iv->socket, next);
      if (timed)
        iv->check += monotonic_seconds() - c0;
      if (!ready)
        break;
      interleave_add(iv, next++);
    }
    if (next < done - INTERLEAVE_LAG && exchange_block_ready(iv->e, iv->socket, next))
      interleave_prefetch(iv, next);
    if (timed)
      iv->add += monotonic_seconds() - start;
    return next;
  }
  for (; next < done; next++) {
    double t0 = timed ? monotonic_seconds() : 0.0;
    if (!exchange_block_wait(iv->e, iv->socket, next))
      return next;
    double t1 = timed ? monotonic_seconds() : 0.0;
    if (next + 1 < done && exchange_block_ready(iv->e, iv->socket, next + 1))
      interleave_prefetch(iv, next + 1);
    interleave_add(iv, next);
    if (timed) {
      iv->wait += t1 - t0;
      iv->add += monotonic_seconds() - t1;
    }
  }
  if (timed)
    iv->finish = monotonic_seconds();
  return next;
}

/* The two-socket W8A8 Wo/W2 with the interleaved reduction; the GEMM's team job is the whole reduction. */
static void project_interleaved(Tp *tp, TpSocket *sk, const int8_t *aq, const float *as, int k, GemmQ8Segment *seg,
                                int slot) {
  Interleave iv = {.e = tp->exchange,
                   .socket = sk->socket,
                   .m = tp->bcount,
                   .n = seg->n,
                   .ld = seg->ldc,
                   .residual = sk->x,
                   .hooks = {.stored = interleave_stored, .turn = interleave_turn},
                   .timed = profile_enabled};
  iv.hooks.ctx = &iv;
  exchange_buffers(tp->exchange, sk->socket, &iv.p0, &iv.p1);
  double start = profile_enabled ? monotonic_seconds() : 0.0;
  k_gemm_q8_interleaved(aq, as, tp->bcount, k, tp_group(tp, k), seg, tp->tiled, &iv.hooks);
  exchange_reduce_finish(tp->exchange, sk->socket); /* the join above: every add of this generation is done */
  if (profile_enabled) {
    double end = monotonic_seconds(), *x = tp->xmat[sk->socket][profile_stage][slot == TP_W2_MM];
    tp->prof[sk->socket][profile_stage][slot] += end - start;
    x[0] += iv.check;
    x[1] += iv.wait;
    x[2] += iv.add;
    x[3] += iv.finish > 0.0 ? end - iv.finish : 0.0;
  }
}

/* The partial product of this socket's K slice for all rows, into the exchange buffer, then the reduction into x.
 * `w` is a column shard (`packed` its packed copy, or NULL): FP32 products take the activations `a` (stride lda), W8A8
 * products the quantized rows `aq`/`as` (dense, stride k and k / gs). */
static int project_batch_reduce(Tp *tp, TpSocket *sk, const float *a, int lda, const int8_t *aq, const float *as, int k,
                                const Shard *w, void *packed, const float *pscales, int slot) {
  /* One socket adds the product straight into the residual (GEMM_ADD): no partial, no reduction. */
  int one = tp->sockets == 1;
  float *partial = one ? sk->x : exchange_partial(tp->exchange, sk->socket);
  if (!partial)
    return 0;
  GemmKind kind = one ? GEMM_ADD : GEMM_PARTIAL;
  double start = profile_enabled ? monotonic_seconds() : 0.0;
  if (tp->quantized) {
    GemmQ8Segment seg = {
        .c = partial, .ldc = tp->c.dim, .w = q8_weight(w), .pw = packed, .pws = pscales, .n = tp->c.dim, .kind = kind};
    if (!one && batch_reduce_mode() > (slot == TP_W2_MM)) {
      project_interleaved(tp, sk, aq, as, k, &seg, slot);
      return !workers_aborted();
    }
    gemm_q8(tp, aq, as, k, &seg, 1);
  } else {
    GemmSegment seg = {.c = partial, .ldc = tp->c.dim, .w = w->q, .ldw = w->ld, .n = tp->c.dim, .kind = kind};
    gemm_f32_or_bf16(tp, sk, a, lda, k, &seg, 1, &packed, NULL);
  }
  if (profile_enabled)
    tp->prof[sk->socket][profile_stage][slot] += monotonic_seconds() - start;
  if (!one) {
    /* LLAMA_PROFILE: this socket's coordinator is the exchange leader, so the components it just added are this
     * reduction's. */
    double before[5], after[5];
    if (profile_enabled)
      exchange_timing_detail(tp->exchange, sk->socket, before);
    reduce(tp, sk, sk->x, tp->bcount * tp->c.dim);
    if (profile_enabled) {
      exchange_timing_detail(tp->exchange, sk->socket, after);
      for (int i = 0; i < 5; i++)
        tp->xmat[sk->socket][profile_stage][slot == TP_W2_MM][i] += after[i] - before[i];
    }
  }
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
  if (tp->tiled)
    k_quantize_rows_tiled(q, scales, x, ldx, tp->bcount, n, tp_group(tp, n));
  else
    k_quantize_rows(q, scales, x, ldx, tp->bcount, n, tp_group(tp, n));
}

/* RMSNorm of the batch into xb, then (W8A8) its quantization for the next product. The pipelined kernel takes both
 * in one pass over the rows (fix 5 of docs/prefill-gap.md; the same k_rmsnorm and k_quantize, so bitwise equal); its
 * time is then counted as quantize. */
static void norm_quantize_batch(Tp *tp, TpSocket *sk, int s, const float *weight) {
  int m = tp->bcount, dim = tp->c.dim;
  if (tp->quantized && tp->tiled) {
    SPROFILE(PHASE_QUANTIZE, k_rmsnorm_quantize_rows_tiled(sk->xq, sk->xs, sk->xb, sk->x, weight, m, dim,
                                                           tp->c.norm_eps, tp_group(tp, dim)));
    return;
  }
  SPROFILE(PHASE_RMSNORM, rmsnorm_rows(tp, sk->xb, sk->x, weight, m));
  if (tp->quantized)
    SPROFILE(PHASE_QUANTIZE, quantize_batch(tp, sk->xq, sk->xs, sk->xb, (size_t)dim, dim));
}

static void forward_batch(Tp *tp, int s) {
  TpSocket *sk = &tp->s[s];
  const Config *p = &tp->c;
  int S = tp->sockets, dim = p->dim, hd = p->head_dim, hl = p->n_heads / S, kl = p->n_kv_heads / S, kvl = kl * hd;
  int ql = dim / S, hidden_l = p->hidden_dim / S, m = tp->bcount, pos0 = tp->bpos0, gs = tp_group(tp, dim);
  int q8 = tp->quantized;
  SPROFILE(PHASE_EMBED, embed_rows(tp, sk));
  SPROFILE(PHASE_ROPE_TABLE, rope_tables(tp, sk));
  for (int l = 0; l < p->n_layers; l++) {
    norm_quantize_batch(tp, sk, s, norm_weight(tp, T_RMS_ATT, l, s));
    KvLayer cache = kv_layer(sk, l);
    float *kc = cache.k, *vc = cache.v;
    int head_layout = kernels_kv_layout_head();
    float *kd = head_layout ? sk->kstage : kc + (size_t)pos0 * kvl,
          *vd = head_layout ? sk->vstage : vc + (size_t)pos0 * kvl;
    const Shard *wq = shards_find(&tp->set, T_WQ, l, s), *wk = shards_find(&tp->set, T_WK, l, s),
                *wv = shards_find(&tp->set, T_WV, l, s);
    if (q8) {
      GemmQ8Segment qkv[3] = {{.c = sk->q,
                               .ldc = ql,
                               .w = q8_weight(wq),
                               .pw = packed_copy(sk, l, 0),
                               .pws = packed_scales(sk, l, 0),
                               .n = ql,
                               .kind = GEMM_STORE},
                              {.c = kd,
                               .ldc = kvl,
                               .w = q8_weight(wk),
                               .pw = packed_copy(sk, l, 1),
                               .pws = packed_scales(sk, l, 1),
                               .n = kvl,
                               .kind = GEMM_STORE},
                              {.c = vd,
                               .ldc = kvl,
                               .w = q8_weight(wv),
                               .pw = packed_copy(sk, l, 2),
                               .pws = packed_scales(sk, l, 2),
                               .n = kvl,
                               .kind = GEMM_STORE}};
      SPROFILE(PHASE_QKV, gemm_q8(tp, sk->xq, sk->xs, dim, qkv, 3));
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
    SPROFILE(PHASE_WO,
             ok = project_batch_reduce(tp, sk, sk->xb, ql, sk->aq, sk->as, ql, shards_find(&tp->set, T_WO, l, s),
                                       packed_copy(sk, l, 3), packed_scales(sk, l, 3), TP_WO_MM));
    if (!ok)
      return;

    norm_quantize_batch(tp, sk, s, norm_weight(tp, T_RMS_FFN, l, s));
    const Shard *w1 = shards_find(&tp->set, T_W1, l, s), *w3 = shards_find(&tp->set, T_W3, l, s);
    if (q8) {
      GemmQ8Segment up = {.c = sk->hb,
                          .ldc = hidden_l,
                          .w = q8_weight(w1),
                          .w3 = q8_weight(w3),
                          .pw = packed_copy(sk, l, 4),
                          .pw3 = packed_copy(sk, l, 6),
                          .pws = packed_scales(sk, l, 4),
                          .pws3 = packed_scales(sk, l, 6),
                          .n = hidden_l,
                          .kind = GEMM_SWIGLU};
      SPROFILE(PHASE_W1_W3, gemm_q8(tp, sk->xq, sk->xs, dim, &up, 1));
      SPROFILE(PHASE_QUANTIZE, quantize_batch(tp, sk->aq, sk->as, sk->hb, (size_t)hidden_l, hidden_l));
    } else {
      GemmSegment up = {.c = sk->hb, .ldc = hidden_l, .w = w1->q, .w3 = w3->q, .ldw = w1->ld, .n = hidden_l,
                        .kind = GEMM_SWIGLU};
      void *const p1 = packed_copy(sk, l, 4), *const p3 = packed_copy(sk, l, 6);
      SPROFILE(PHASE_W1_W3, gemm_f32_or_bf16(tp, sk, sk->xb, dim, dim, &up, 1, &p1, &p3));
    }
    SPROFILE(PHASE_W2, ok = project_batch_reduce(tp, sk, sk->hb, hidden_l, sk->aq, sk->as, hidden_l,
                                                 shards_find(&tp->set, T_W2, l, s), packed_copy(sk, l, 5),
                                                 packed_scales(sk, l, 5), TP_W2_MM));
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

/* One slice of a region, zeroed by each worker of a team: the team creates the region's pages. */
typedef struct {
  void *data;
  size_t bytes;
} ZeroSpan;

static void zero_span(const void *p, int worker, int workers) {
  const ZeroSpan *z = p;
  size_t lo = (size_t)((unsigned long long)z->bytes * (unsigned)worker / (unsigned)workers);
  size_t hi = (size_t)((unsigned long long)z->bytes * (unsigned)(worker + 1) / (unsigned)workers);
  memset((char *)z->data + lo, 0, hi - lo);
}

/* The KV table of the legacy placements: layer l's caches are `l * seq_len * kv_dim` floats into one region per cache (the
 * socket's own region on two sockets, the run memory on one). It is built on the main thread, like the table of a plan,
 * because the engine's allocation counters are not thread-safe. */
static void kv_table_legacy(Tp *tp, TpSocket *sk, float *k, float *v) {
  size_t layer_floats = (size_t)tp->c.seq_len * tp->kv_dim_local;
  sk->kv_layers = memory_alloc((size_t)tp->c.n_layers, sizeof(KvLayer), MEM_METADATA);
  for (int l = 0; l < tp->c.n_layers; l++)
    sk->kv_layers[l] = (KvLayer){k + (size_t)l * layer_floats, v + (size_t)l * layer_floats, NULL};
}

/* Carves the socket's activations out of its work region and sizes the attention work from its team. Runs on the
 * socket's coordinator, so the pages are first written by the socket that uses them. */
static void setup_socket(Tp *tp, int s) {
  TpSocket *sk = &tp->s[s];
  const Config *p = &tp->c;
  int S = tp->sockets, hl = p->n_heads / S, kl = p->n_kv_heads / S;
  /* The socket that uses the work region writes all of it once, so every page exists on its own node and the page query
   * can verify the region like the others (a fresh mapping is zero already; this only makes the pages exist). One
   * socket's run memory was zeroed by its team when it was sealed (runmem_seal), or comes zeroed from the heap. */
  if (S == 2 && !tp->plan.mode)
    memset(sk->work_data, 0, sk->work_size);
  char *cursor = sk->work_data;
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
  CARVE(sk->hb, rows * (p->hidden_dim / S), float);
  CARVE(sk->q, rows * (p->dim / S), float);
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
    CARVE(sk->xs, padded * (size_t)(p->dim / tp_group(tp, p->dim)), float);
    size_t slice = (size_t)(p->dim > p->hidden_dim ? p->dim : p->hidden_dim) / S; /* attention output or FFN hidden */
    CARVE(sk->aq, padded * slice, int8_t);
    CARVE(sk->as, padded * (slice / (size_t)tp_group(tp, (int)slice)), float);
  }
  float *attention_memory;
  CARVE(attention_memory, k_attention_bytes_rows(hl, kl, p->head_dim, p->seq_len, tp->rows) / sizeof(float) + 1, float);
  /* The persistent BF16 K/V copy (LLAMA_ATTN=amx): legacy placement carves it from the work region; a plan gives it
   * arenas of its own, tiered per layer like the caches (KvLayer.bf16). */
  uint16_t *kv_bf16 = NULL;
  size_t kv_bf16_bytes = k_attention_kv_bf16_bytes(kl, p->head_dim, p->seq_len, p->n_layers);
  if (kv_bf16_bytes && !tp->plan.mode)
    CARVE(kv_bf16, kv_bf16_bytes / sizeof(uint16_t), uint16_t);
#undef CARVE
  if (used > sk->work_size)
    fail("socket %d work region holds %zu bytes, %zu needed", s, sk->work_size, used);
  k_attention_init_rows(&sk->attention, hl, kl, p->head_dim, p->seq_len, tp->rows, attention_memory);
  /* The KV table (kv_layers) was built on the main thread before this ran (kv_table_legacy, or tp_create_planned): the
   * coordinators of the two sockets must not touch the engine's allocation counters. */
  size_t layer_floats = (size_t)p->seq_len * tp->kv_dim_local;
  if (kv_bf16)
    k_attention_set_kv_bf16(&sk->attention, kv_bf16, kl, p->head_dim, p->seq_len, p->n_layers);
  else if (kv_bf16_bytes && tp->plan.mode) {
    if (!sk->kv_layers[0].bf16)
      fail("socket %d: the placement plan holds no arena for the BF16 K/V copy that LLAMA_ATTN=amx needs", s);
    uint16_t *memory[256];
    const float *keys[256];
    for (int l = 0; l < p->n_layers; l++) {
      memory[l] = sk->kv_layers[l].bf16;
      keys[l] = sk->kv_layers[l].k;
    }
    k_attention_set_kv_bf16_layers(&sk->attention, memory, keys, kl, p->head_dim, p->seq_len, p->n_layers);
  }
  /* The socket's team writes its KV caches (and the BF16 copy) once now, so their pages exist (on the socket's node,
   * created by its own threads) before the first prompt: otherwise each 4 KiB page is faulted in by the K/V store of the
   * prefill, inside the time to first token. A fresh mapping is zero already; the values do not change. A plan's tiers
   * are separate arenas, each written whole. */
  ZeroSpan spans[A_COUNT + 2];
  int nspans = 0;
  if (tp->plan.mode) {
    for (int i = A_K_HBM; i <= A_KVB_DDR; i++)
      if (tp->arenas[s][i].data)
        spans[nspans++] = (ZeroSpan){tp->arenas[s][i].data, tp->arenas[s][i].logical_bytes};
  } else {
    size_t kv_bytes = (size_t)p->n_layers * layer_floats * sizeof(float);
    spans[nspans++] = (ZeroSpan){sk->kv_layers[0].k, kv_bytes};
    spans[nspans++] = (ZeroSpan){sk->kv_layers[0].v, kv_bytes};
  }
  for (int i = 0; i < nspans; i++) {
    if (sk->team)
      team_run(sk->team, zero_span, &spans[i]);
    else
      zero_span(&spans[i], 0, 1);
  }
}

/* ---- region sizes -------------------------------------------------------------------------------------------- */

/* kv_bf16: count the persistent BF16 K/V copy of LLAMA_ATTN=amx in the work region (legacy placement carves it there);
 * a plan gives it arenas of its own. */
static size_t work_bytes_for(const Config *p, int sockets, int quantized, int gs, int rows, int workers, int kv_bf16) {
  int hl = p->n_heads / sockets, kl = p->n_kv_heads / sockets;
  size_t slice = (size_t)(p->dim > p->hidden_dim ? p->dim : p->hidden_dim) / sockets;
  size_t bytes = 64 * 16; /* alignment of each buffer */
  size_t d = (size_t)p->dim, hidden = (size_t)p->hidden_dim, head = (size_t)p->head_dim;
  size_t floats_per_row = checked_add(checked_mul(2, d), checked_add(hidden / sockets, d / sockets));
  floats_per_row = checked_add(floats_per_row, checked_mul(2, head));
  if (kernels_kv_layout_head()) /* the K/V row buffers of LLAMA_KV_LAYOUT=head */
    floats_per_row = checked_add(floats_per_row, checked_mul(2 * (size_t)kl, head));
  size_t scores = checked_mul((size_t)hl, (size_t)p->seq_len);
  size_t floats = checked_add(checked_mul((size_t)rows, floats_per_row), scores);
  bytes = checked_add(bytes, checked_mul(floats, sizeof(float)));
  if (quantized) { /* xq, xs, aq, as, for whole 32-row blocks */
    size_t padded = ((size_t)rows + 31) / 32 * 32;
    size_t groups = gs ? checked_add(d / (size_t)gs, slice / (size_t)gs) : 2; /* per-channel (gs 0): one scale per row each */
    size_t scales = checked_mul(groups, sizeof(float));
    bytes = checked_add(bytes, checked_mul(padded, checked_add(checked_add(d, slice), scales)));
  }
  size_t att = workers ? k_attention_bytes_rows_for(hl, kl, p->head_dim, p->seq_len, rows, workers)
                       : k_attention_bytes_rows(hl, kl, p->head_dim, p->seq_len, rows);
  bytes = checked_add(bytes, checked_add(att, 64));
  if (kv_bf16)
    bytes = checked_add(bytes, checked_add(k_attention_kv_bf16_bytes(kl, p->head_dim, p->seq_len, p->n_layers), 64));
  return bytes;
}

size_t tp_work_bytes(const Config *p, int quantized, int gs, int rows) {
  return work_bytes_for(p, SOCKETS, quantized, gs, rows, 0, 1);
}

size_t tp_exchange_bytes(const Config *p, int rows) {
  return exchange_region_bytes(checked_mul((size_t)rows, (size_t)p->dim));
}

static size_t work_bytes(const Tp *tp) {
  return work_bytes_for(&tp->c, tp->sockets, tp->quantized, tp->gs, tp->rows, 0, !tp->plan.mode);
}

static void socket_command(Tp *tp, int socket) {
  switch (tp->command) {
  case CMD_SETUP:
    if (tp->plan.mode) {
      const Arena touch[] = {A_WORK, A_LOGITS, A_LOAD_SCRATCH};
      for (size_t i = 0; i < sizeof(touch) / sizeof(*touch); i++) {
        NumaRegion *r = &tp->arenas[socket][touch[i]];
        if (r->data)
          memset(r->data, 0, r->logical_bytes);
      }
    }
    setup_socket(tp, socket);
    tp->s[socket].leader_cpu = cpu_current();
    break;
  case CMD_LOAD:
    shards_load_placed(&tp->set, tp->loader_fd, socket, tp->s[socket].team, tp->s[socket].load_scratch);
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

static size_t align_line(size_t n) {
  return checked_add(n, 63) & ~(size_t)63;
}

/* A packed copy: the VNNI tiles, and with the tiled Q8 kernel the transposed scales on the next cache line. */
static size_t pack_bytes_of(const Tp *tp, const Shard *sh) {
  if (tp->pack == TP_PACK_BF16)
    return bf16_vnni_elements(sh->rows, sh->cols) * sizeof(uint16_t);
  size_t tiles = s8_vnni_bytes(sh->rows, sh->cols);
  return tp->tiled ? align_line(tiles) + s8_scales_t_bytes(sh->rows, sh->cols, tp_group(tp, sh->cols)) : tiles;
}

/* What socket s's packed region must hold: its shard of every matrix of every layer, each on a cache line. */
static size_t packed_region_bytes(const Tp *tp, int s) {
  size_t total = 0;
  for (int l = 0; l < tp->c.n_layers; l++)
    for (int i = 0; i < 7; i++)
      total = checked_add(align_line(total), pack_bytes_of(tp, shards_find(&tp->set, PACK_ROLES[i], l, s)));
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
  for (int s = 0; s < tp->sockets; s++) {
    TpSocket *sk = &tp->s[s];
    size_t count = (size_t)tp->c.n_layers * 7;
    sk->packed_at = memory_alloc(count, sizeof(void *), MEM_METADATA);
    if (tp->tiled)
      sk->pscales_at = memory_alloc(count, sizeof(float *), MEM_METADATA);
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
        char *dst = (char *)sk->packed_data + used;
        used = checked_add(used, pack_bytes_of(tp, sh));
        size_t at = (size_t)l * 7 + (size_t)i;
        sk->packed_at[at] = dst;
        if (tp->pack == TP_PACK_BF16) {
          tp->bjobs[s][at] = pack_bf16_job_from_shard((uint16_t *)dst, sh);
        } else {
          tp->sjobs[s][at] = pack_s8_job_from_shard((int8_t *)dst, sh);
          if (tp->tiled) {
            PackS8 *job = &tp->sjobs[s][at];
            job->st = sk->pscales_at[at] = (float *)(dst + align_line(s8_vnni_bytes(sh->rows, sh->cols)));
            job->s = sh->s;
            job->groups = sh->cols / tp_group(tp, sh->cols);
          }
        }
      }
    if (used > sk->packed_size)
      fail("socket %d packed region holds %zu bytes, %zu needed", s, sk->packed_size, used);
  }
}

/* The classifier range of each socket in BF16, built when all-position logits are first asked for (the default
 * last-row classifier stays FP32, as on one socket). */
static void pack_classifier(Tp *tp) {
  double start = monotonic_seconds();
  if (tp->sockets == 1) { /* the one-socket packed store, on the weights' node */
    const Shard *cls = shards_find(&tp->set, T_WCLS, -1, 0);
    size_t bytes = bf16_vnni_elements(cls->rows, cls->cols) * sizeof(uint16_t);
    packed_open(&tp->local_cls, bytes, "the packed BF16 classifier");
    tp->s[0].cls_bf16 = packed_take(&tp->local_cls, bytes);
    tp->bjobs[0][0] = pack_bf16_job_from_shard(tp->s[0].cls_bf16, cls);
    run_both(tp, CMD_PACK_CLS);
    packed_seal(&tp->local_cls);
    metrics.weight_pack_s += monotonic_seconds() - start;
    LOG(LOG_INFO, "packed %zu bytes of BF16 classifier in %.2f s", tp->local_cls.region.logical_bytes,
        monotonic_seconds() - start);
    tp->cls_ready = 1;
    return;
  }
  const char *mode = placement_mode();
  const Topology *topology = numa_topology();
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

/* What both constructors share: the precision, the batch capacity and the packing mode. */
static Tp *tp_new(const Config *p, int sockets, int quantized, int gs, int fp32_activations, int max_batch,
                  TpPack pack) {
  Tp *tp = memory_alloc(1, sizeof(*tp), MEM_METADATA);
  if ((pack == TP_PACK_BF16 && quantized) || (pack == TP_PACK_INT8 && (!quantized || fp32_activations)))
    fail("packed shards: BF16 needs an FP32 checkpoint and int8 needs W8A8");
  tp->sockets = sockets;
  tp->pack = pack;
  /* The batched forward takes FP32 and W8A8; W8A32 is per token (the C6 rule), so it keeps one row. */
  tp->rows = (quantized && fp32_activations) || max_batch < 1 ? 1 : max_batch;
  tp->c = *p;
  tp->quantized = quantized;
  tp->fp32_activations = fp32_activations;
  tp->gs = quantized ? gs : 0;
  /* Per-channel INT8 (gs 0) has one group per product, and gs = 32 half k-steps: every per-socket input width must be
   * whole k-steps. */
  int widths = (p->dim / sockets) % 64 == 0 && (p->hidden_dim / sockets) % 64 == 0;
  int steps = gs ? gs % 64 == 0 || (gs == 32 && widths) : widths;
  tp->tiled = pack == TP_PACK_INT8 && steps && k_gemm_q8_tiled_active();
  return tp;
}

Tp *tp_create_local(const Config *p, const TensorManifest *m, const WeightStore *store, int quantized, int gs,
                    int fp32_activations, int max_batch, TpPack pack) {
  Tp *tp = tp_new(p, 1, quantized, gs, fp32_activations, max_batch, pack);
  TpSocket *sk = &tp->s[0];
  sk->team = pool_team(0);
  /* One socket's shards are whole tensors (shards.h), so each is a view into the store at its file offset: the mapping
   * in file mode, the owned regions otherwise. Nothing is copied here. */
  shards_plan(&tp->set, m, p, 1, gs);
  for (int i = 0; i < tp->set.count; i++) {
    Shard *sh = &tp->set.t[i];
    if (sh->row0 || sh->col0 || sh->rows != sh->src->rows || sh->cols != sh->src->cols)
      fail("one-socket shard of %s is not the whole tensor", manifest_role_name(sh->src->role));
    sh->q = (void *)weights_at(store, sh->src->offset);
    sh->s = sh->src->dtype == DT_Q8 ? (float *)weights_at(store, sh->src->scale_offset) : NULL;
  }
  /* The run state: KV caches and the work region where the loader places them (heap in file mode, else on the
   * weights' node, sized in a first pass and carved in a second; A5). */
  tp->kv_dim_local = p->head_dim * p->n_kv_heads;
  runmem_init(&tp->mem, checked_mul(checked_mul((size_t)p->n_layers, (size_t)p->seq_len),
                                    (size_t)tp->kv_dim_local * sizeof(float)));
  sk->work_size = work_bytes(tp);
  sk->work_data = runmem_alloc(&tp->mem, sk->work_size, 1);
  if (runmem_sizing(&tp->mem)) {
    runmem_seal(&tp->mem);
    sk->work_data = runmem_alloc(&tp->mem, sk->work_size, 1);
  }
  kv_table_legacy(tp, sk, runmem_cache(&tp->mem, 0), runmem_cache(&tp->mem, 1));
  setup_socket(tp, 0);
  tp->logits = memory_alloc((size_t)p->vocab_size, sizeof(float), MEM_ACTIVATION);
  if (pack == TP_PACK_BF16)
    aligned_buffer_alloc(&sk->panel, k_gemm_bf16_panel_bytes(tp->rows, p->dim > p->hidden_dim ? p->dim : p->hidden_dim),
                         MEM_ACTIVATION);
  if (pack != TP_PACK_NONE) {
    double start = monotonic_seconds();
    sk->packed_size = packed_region_bytes(tp, 0);
    packed_open(&tp->local_packed, sk->packed_size,
                pack == TP_PACK_BF16 ? "the packed BF16 weights" : "the packed int8 weights");
    sk->packed_data = packed_take(&tp->local_packed, sk->packed_size);
    plan_packing(tp);
    run_both(tp, CMD_PACK);
    packed_seal(&tp->local_packed);
    metrics.weight_pack_s += monotonic_seconds() - start;
    metrics.weight_packed = 1;
    LOG(LOG_INFO, "packed %zu bytes of %s in %.2f s", tp->local_packed.region.logical_bytes,
        pack == TP_PACK_BF16 ? "BF16 weights" : "int8 weights for AMX-INT8", monotonic_seconds() - start);
  }
  return tp;
}

Tp *tp_create(const Config *p, const TensorManifest *m, int fd, int quantized, int gs, int fp32_activations,
              int max_batch, TpPack pack) {
  Tp *tp = tp_new(p, SOCKETS, quantized, gs, fp32_activations, max_batch, pack);
  if (checked_mul((size_t)tp->rows, (size_t)p->dim) > (size_t)INT_MAX)
    fail("LLAMA_PREFILL_BATCH=%d is too large for LLAMA_SOCKETS=2: the exchange holds %d x %d elements", tp->rows,
         tp->rows, p->dim);
  tp->loader_fd = fd;
  const char *check = getenv("LLAMA_TP_CHECK");
  if (check && *check && strcmp(check, "0") && strcmp(check, "1"))
    fail("LLAMA_TP_CHECK must be 0 or 1");
  tp->check = check && !strcmp(check, "1");
  shards_plan(&tp->set, m, p, SOCKETS, gs);
  const char *mode = placement_mode();
  placement_record(mode);
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
    if (packed_bytes) {
      region_reserve(&sk->packed, packed_bytes, s, node, ROLE_PACKED, thp);
      sk->packed_data = sk->packed.data;
      sk->packed_size = sk->packed.logical_bytes;
    }
    region_reserve(&sk->work, work_bytes(tp), s, node, ROLE_WORK, 0);
    sk->work_data = sk->work.data;
    sk->work_size = sk->work.data_bytes;
    region_reserve(&sk->exchange, tp_exchange_bytes(p, tp->rows), s, node, ROLE_WORK, 0);
    exchange_regions[s] = &sk->exchange;
    region_mark_populated(&sk->kv_k);
    region_mark_populated(&sk->kv_v);
    region_mark_populated(&sk->work);
    kv_table_legacy(tp, sk, sk->kv_k.data, sk->kv_v.data);
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
  /* The interleaved reduction flags blocks of 32 rows x 16 or 32 channels in the per-element flag array. */
  if (batch_reduce_mode() &&
      (long long)((p->dim + 15) / 16) * ((tp->rows + 31) / 32) > exchange_block_capacity(tp->exchange))
    fail("LLAMA_TP_BATCH_REDUCE=%s: %d x %d blocks exceed the exchange flags", tp_batch_reduce_name(),
         (p->dim + 15) / 16, (tp->rows + 31) / 32);
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

static const char *tier_name(int tier) {
  return tier == HBM ? "hbm" : "ddr";
}

static const char *setting(const char *name) {
  const char *value = getenv(name);
  return value && *value ? value : NULL;
}

static int layer_setting(const char *name, int layers) {
  const char *value = setting(name);
  if (!value || !strcmp(value, "auto"))
    return -1;
  char *end;
  long number = strtol(value, &end, 10);
  if (strspn(value, "0123456789") != strlen(value) || *end || number < 0 || number > layers)
    fail("%s must be auto or between 0 and %d", name, layers);
  return (int)number;
}

static int tier_setting(const char *name, int fallback, int allow_auto) {
  const char *value = setting(name);
  if (!value || (allow_auto && !strcmp(value, "auto")))
    return fallback;
  if (!strcmp(value, "hbm"))
    return HBM;
  if (!strcmp(value, "ddr"))
    return DDR;
  fail("%s must be hbm or ddr%s", name, allow_auto ? " or auto" : "");
}

static size_t ddr_budget(void) {
  const char *value = setting("LLAMA_DDR_BUDGET_GIB");
  if (!value)
    return SIZE_MAX;
  char *end;
  double gib = strtod(value, &end);
  if (*end || !(gib > 0) || gib > 1 << 20)
    fail("LLAMA_DDR_BUDGET_GIB must be a positive number of GiB");
  return (size_t)(gib * (double)(1 << 30));
}

static size_t node_free_bytes(int node) {
  if (node < 0)
    return 0;
  char path[128], line[256];
  snprintf(path, sizeof(path), "/sys/devices/system/node/node%d/meminfo", node);
  FILE *f = fopen(path, "r");
  if (!f)
    return 0;
  size_t bytes = 0;
  while (fgets(line, sizeof(line), f)) {
    char *key = strstr(line, "MemFree:");
    if (key) {
      bytes = checked_mul((size_t)strtoull(key + 8, NULL, 10), 1024);
      break;
    }
  }
  fclose(f);
  return bytes;
}

static size_t job_memory_limit(void) {
#ifdef __linux__
  FILE *f = fopen("/proc/self/cgroup", "r");
  if (!f)
    return 0;
  char line[1024], path[1200];
  size_t limit = 0;
  while (fgets(line, sizeof(line), f)) {
    char *controllers = strchr(line, ':');
    char *group = controllers ? strchr(controllers + 1, ':') : NULL;
    if (!group || group[1] != '/')
      continue;
    *group++ = 0;
    const char *controller = controllers + 1;
    int unified = !*controller;
    if (!unified && strcmp(controller, "memory"))
      continue;
    group[strcspn(group, "\n")] = 0;
    for (;;) {
      snprintf(path, sizeof(path), "/sys/fs/cgroup%s%s/%s", unified ? "" : "/memory", group,
               unified ? "memory.max" : "memory.limit_in_bytes");
      FILE *cap = fopen(path, "r");
      unsigned long long bytes = 0;
      if (cap) {
        if (fscanf(cap, "%llu", &bytes) == 1 && bytes && bytes < SIZE_MAX && (!limit || bytes < limit))
          limit = (size_t)bytes;
        fclose(cap);
      }
      char *slash = strrchr(group, '/');
      if (!slash || !group[0])
        break;
      *slash = 0;
    }
  }
  fclose(f);
  return limit;
#else
  return 0;
#endif
}

static void arena_plan(PlacementPlan *plan, int s, Arena arena, int tier, size_t bytes, RegionRole role, int thp) {
  int node = plan->node[s][tier];
  size_t rounded = 0;
  if (bytes && (region_round(bytes, region_alignment(node, thp), &rounded) || rounded > PTRDIFF_MAX))
    fail("planned socket %d %s arena exceeds the addressable size", s, region_role_name(role));
  plan->arena[s][arena] =
      (ArenaPlan){.bytes = bytes, .rounded = rounded, .tier = tier, .node = node, .role = role, .thp = thp};
  plan->charge[s][tier] = checked_add(plan->charge[s][tier], rounded);
}

static void placement_layout(Tp *tp) {
  PlacementPlan *plan = &tp->plan;
  int ddr_only = !strcmp(plan->mode, "ddr"), hbm_only = !strcmp(plan->mode, "hbm-only");
  int hot = ddr_only ? DDR : HBM, cold = hbm_only ? HBM : DDR;
  int thp = region_thp_default();
  size_t weight_bytes[SOCKETS][TIERS];
  shards_place(&tp->set, plan->weights_layers, plan->classifier, hot, weight_bytes);
  memset(plan->charge, 0, sizeof(plan->charge));
  for (int s = 0; s < plan->sockets; s++) {
    for (int t = 0; t < TIERS; t++) {
      int layers = t == HBM ? plan->kv_layers : tp->c.n_layers - plan->kv_layers;
      size_t kv_bytes = checked_mul(plan->kv_layer_bytes, (size_t)layers);
      arena_plan(plan, s, A_WEIGHTS_HBM + t, t, weight_bytes[s][t], ROLE_WEIGHTS, thp);
      arena_plan(plan, s, A_K_HBM + t, t, kv_bytes, ROLE_KV_K, thp);
      arena_plan(plan, s, A_V_HBM + t, t, kv_bytes, ROLE_KV_V, thp);
      arena_plan(plan, s, A_KVB_HBM + t, t, checked_mul(plan->kv_bf16_layer_bytes, (size_t)layers), ROLE_KV_BF16, thp);
    }
    arena_plan(plan, s, A_WORK, hot, checked_add(align_line(plan->work_bytes[s]), plan->panel_bytes), ROLE_WORK, 0);
    arena_plan(plan, s, A_EXCHANGE, hot, plan->sockets == 2 ? tp_exchange_bytes(&tp->c, tp->rows) : 0, ROLE_EXCHANGE,
               0);
    arena_plan(plan, s, A_EMBED, cold, s ? 0 : tp->set.global_bytes, ROLE_EMBEDDING, thp);
    arena_plan(plan, s, A_LOGITS, hot, s ? 0 : plan->logits_bytes, ROLE_LOGITS, 0);
    size_t packed = tp->pack == TP_PACK_NONE ? 0 : packed_region_bytes(tp, s);
    arena_plan(plan, s, A_PACK, plan->packed, packed, ROLE_PACKED, thp);
    size_t packed_cls =
        tp->pack == TP_PACK_BF16 && tp->batch_logits_rows ? pack_bytes_of(tp, shards_find(&tp->set, T_WCLS, -1, s)) : 0;
    arena_plan(plan, s, A_PACK_CLS, plan->packed, packed_cls, ROLE_PACKED, thp);
    arena_plan(plan, s, A_LOAD_SCRATCH, cold, shards_scratch_bytes(&tp->set, s), ROLE_SCRATCH, 0);
  }
}

static int tier_fits(const PlacementPlan *plan, int tier) {
  for (int s = 0; s < plan->sockets; s++)
    if (plan->charge[s][tier] > plan->budget[s][tier])
      return 0;
  return 1;
}

/* The HBM charge is monotonic in either layer prefix, including rounded arena boundaries. */
static void largest_prefix(Tp *tp, int *prefix) {
  int lo = 0, hi = tp->c.n_layers;
  while (lo < hi) {
    *prefix = lo + (hi - lo + 1) / 2;
    placement_layout(tp);
    if (tier_fits(&tp->plan, HBM))
      lo = *prefix;
    else
      hi = *prefix - 1;
  }
  *prefix = lo;
}

static void placement_plan(Tp *tp, const Model *model, const PoolConfig *pool) {
  PlacementPlan *plan = &tp->plan;
  const Topology *topology = numa_topology();
  int hybrid = !strcmp(plan->mode, "hybrid"), ddr = !strcmp(plan->mode, "ddr");
  plan->sockets = pool->teams;
  plan->simulated = topology->classification == CLASS_VIRTUAL;
  if (setting("LLAMA_TOPOLOGY_ALLOW_FAR_HBM") && !strcmp(setting("LLAMA_TOPOLOGY_ALLOW_FAR_HBM"), "1"))
    fail("planned placement requires socket-local NUMA nodes");
  if (!plan->simulated) {
    char err[512];
    if ((!ddr && numa_check_placement(topology, pool->teams, "hbm", err, sizeof(err))) ||
        ((ddr || hybrid) && numa_check_placement(topology, pool->teams, "ddr", err, sizeof(err))))
      fail("%s", err);
  }
  if (!placement_strict(plan->mode))
    fail("LLAMA_PLACEMENT=%s requires strict placement", plan->mode);
  plan->job_cap = plan->simulated ? 0 : job_memory_limit();
  size_t cap[TIERS] = {placement_hbm_budget(), ddr_budget()};
  for (int s = 0; s < plan->sockets; s++) {
    const SocketTopo *socket = &topology->socket[s];
    plan->node[s][HBM] = socket->hbm_node;
    plan->node[s][DDR] = socket->ddr_node;
    for (int t = 0; t < TIERS; t++) {
      size_t total = t == HBM ? socket->hbm_bytes : socket->ddr_bytes;
      plan->budget[s][t] = !total || cap[t] < total ? cap[t] : total;
      plan->free_bytes[s][t] = node_free_bytes(plan->node[s][t]);
    }
    plan->threads[s] = pool->threads[s];
    plan->work_bytes[s] = work_bytes_for(&tp->c, tp->sockets, tp->quantized, tp->gs, tp->rows, pool->threads[s], 0);
  }
  plan->kv_layer_bytes = checked_mul((size_t)tp->c.seq_len, checked_mul((size_t)tp->kv_dim_local, sizeof(float)));
  /* LLAMA_ATTN=amx: one BF16 copy of K and V per layer (0 when the attention kernel does not keep one), planned with the
   * layer's caches: the layers that stay in HBM keep their copy there too. */
  plan->kv_bf16_layer_bytes = k_attention_kv_bf16_bytes(tp->c.n_kv_heads / tp->sockets, tp->c.head_dim, tp->c.seq_len, 1);
  if (plan->kv_bf16_layer_bytes && tp->c.n_layers > 256)
    plan->kv_bf16_layer_bytes = 0; /* the copy table holds 256 layers; k_attention_kv_bf16_bytes agrees */
  if (tp->pack == TP_PACK_BF16) {
    int hidden = tp->c.hidden_dim / tp->sockets;
    plan->panel_bytes = k_gemm_bf16_panel_bytes(tp->rows, hidden > tp->c.dim ? hidden : tp->c.dim);
  }
  tp->batch_logits_rows = model->prefill_logits == LOGITS_ALL ? model->max_batch : 0;
  plan->logits_bytes =
      checked_mul(checked_mul((size_t)tp->c.vocab_size, 1 + (size_t)tp->batch_logits_rows), sizeof(float));

  int layers = tp->c.n_layers;
  int weights = hybrid ? layer_setting("LLAMA_HYBRID_WEIGHTS_HBM_LAYERS", layers) : ddr ? 0 : layers;
  int kv = hybrid ? layer_setting("LLAMA_HYBRID_KV_HBM_LAYERS", layers) : ddr ? 0 : layers;
  int classifier = hybrid ? tier_setting("LLAMA_HYBRID_CLASSIFIER", -1, 1) : ddr ? DDR : HBM;
  plan->packed = hybrid ? tier_setting("LLAMA_HYBRID_PACKED", DDR, 0) : ddr ? DDR : HBM;
  if (hybrid && tp->pack == TP_PACK_NONE && setting("LLAMA_HYBRID_PACKED"))
    fail("LLAMA_HYBRID_PACKED requires a packed prefill kernel");
  plan->weights_layers = weights < 0 ? 0 : weights;
  plan->kv_layers = kv < 0 ? 0 : kv;
  plan->classifier = classifier < 0 ? HBM : classifier;
  if (classifier < 0) {
    placement_layout(tp);
    if (!tier_fits(plan, HBM))
      plan->classifier = DDR;
  }
  if (weights < 0)
    largest_prefix(tp, &plan->weights_layers);
  if (kv < 0)
    largest_prefix(tp, &plan->kv_layers);
  placement_layout(tp);
  placement_summary = *plan;
  size_t peak = 0;
  for (int s = 0; s < plan->sockets; s++)
    for (int t = 0; t < TIERS; t++) {
      size_t charge = plan->charge[s][t], budget = plan->budget[s][t];
      if (charge > budget)
        fail("socket %d node %d needs %zu bytes of %s (%zu over its %zu-byte budget)", s, plan->node[s][t], charge,
             tier_name(t), charge - budget, budget);
      peak = checked_add(peak, charge);
      if (plan->free_bytes[s][t] && charge > plan->free_bytes[s][t])
        LOG(LOG_WARN, "socket %d %s plan reserves %zu bytes with %zu currently free", s, tier_name(t), charge,
            plan->free_bytes[s][t]);
    }
  if (plan->job_cap && peak > plan->job_cap)
    fail("placement plan needs %zu bytes, above the cgroup memory limit %zu", peak, plan->job_cap);
  plan->admitted = 1;
  placement_summary = *plan;
  LOG(LOG_INFO, "%s plan: %d of %d weight layers and %d KV layers in HBM, classifier=%s, packed=%s", plan->mode,
      plan->weights_layers, layers, plan->kv_layers, tier_name(plan->classifier), tier_name(plan->packed));
  for (int s = 0; s < plan->sockets; s++)
    LOG(LOG_INFO, "socket %d: HBM %zu of %zu bytes, DDR %zu of %zu bytes", s, plan->charge[s][HBM],
        plan->budget[s][HBM], plan->charge[s][DDR], plan->budget[s][DDR]);
}

void tp_placement_report(FILE *f) {
  const PlacementPlan *plan = &placement_summary;
  if (!plan->mode) {
    fputs("null", f);
    return;
  }
  fprintf(f,
          "{\"mode\":\"%s\",\"admitted\":%s,\"simulated\":%s,\"weights_hbm_layers\":%d,\"kv_hbm_layers\":%d,"
          "\"classifier\":\"%s\",\"packed\":\"%s\",\"kv_layer_bytes\":%zu,\"kv_bf16_layer_bytes\":%zu,"
          "\"cgroup_limit_bytes\":%zu,\"nodes\":[",
          plan->mode, plan->admitted ? "true" : "false", plan->simulated ? "true" : "false", plan->weights_layers,
          plan->kv_layers, tier_name(plan->classifier), tier_name(plan->packed), plan->kv_layer_bytes,
          plan->kv_bf16_layer_bytes, plan->job_cap);
  for (int s = 0; s < plan->sockets; s++)
    for (int t = 0; t < TIERS; t++)
      fprintf(f,
              "%s{\"socket\":%d,\"threads\":%d,\"tier\":\"%s\",\"node\":%d,\"budget_bytes\":%zu,"
              "\"charged_bytes\":%zu,\"free_bytes_at_plan\":%zu}",
              s || t ? "," : "", s, plan->threads[s], tier_name(t), plan->node[s][t], plan->budget[s][t],
              plan->charge[s][t], plan->free_bytes[s][t]);
  fputs("],\"arenas\":[", f);
  int printed = 0;
  for (int s = 0; s < plan->sockets; s++)
    for (int i = 0; i < A_COUNT; i++) {
      const ArenaPlan *a = &plan->arena[s][i];
      if (a->bytes)
        fprintf(f,
                "%s{\"socket\":%d,\"tier\":\"%s\",\"node\":%d,\"role\":\"%s\",\"bytes\":%zu,"
                "\"rounded_bytes\":%zu}",
                printed++ ? "," : "", s, tier_name(a->tier), a->node, region_role_name(a->role), a->bytes, a->rounded);
    }
  fputs("]}", f);
}

static void placement_reserve(Tp *tp) {
  for (int s = 0; s < tp->sockets; s++)
    for (int i = 0; i < A_COUNT; i++) {
      const ArenaPlan *a = &tp->plan.arena[s][i];
      char err[512];
      if (a->bytes && region_reserve_try(&tp->arenas[s][i], a->bytes, s, a->node, a->role, a->thp, err, sizeof(err))) {
        for (int j = 0; j < tp->sockets; j++)
          for (int k = 0; k < A_COUNT; k++)
            region_release(&tp->arenas[j][k]);
        fail("placement reservation failed: %s", err);
      }
    }
}

Tp *tp_create_planned(const Model *model, const TensorManifest *m, int fd, int quantized, int gs, int fp32_activations,
                      TpPack pack, int plan_only) {
  PoolConfig pool;
  kernels_pool_config(&pool);
  const Config *p = &model->config;
  int sockets = pool.teams;
  Tp *tp = tp_new(p, sockets, quantized, gs, fp32_activations, model->prefill_gemm ? model->max_batch : 1, pack);
  tp->plan.mode = placement_mode();
  tp->loader_fd = fd;
  tp->kv_dim_local = p->head_dim * (p->n_kv_heads / sockets);
  metrics.sockets = sockets;
  if (sockets == 2 && checked_mul((size_t)tp->rows, (size_t)p->dim) > INT_MAX)
    fail("the two-socket exchange exceeds INT_MAX elements");
  const char *check = setting("LLAMA_TP_CHECK");
  if (check && strcmp(check, "0") && strcmp(check, "1"))
    fail("LLAMA_TP_CHECK must be 0 or 1");
  tp->check = sockets == 2 && check && !strcmp(check, "1");
  shards_plan(&tp->set, m, p, sockets, gs);
  placement_plan(tp, model, &pool);
  if (plan_only)
    return tp;

  kernels_backend_init();
  placement_reserve(tp);
  void *weight_bases[SOCKETS][TIERS] = {{0}};
  NumaRegion *exchange_regions[SOCKETS] = {0};
  for (int s = 0; s < sockets; s++) {
    TpSocket *sk = &tp->s[s];
    NumaRegion *a = tp->arenas[s];
    sk->team = pool_team(s);
    sk->socket = s;
    sk->work_data = a[A_WORK].data;
    sk->work_size = tp->plan.work_bytes[s];
    sk->load_scratch = a[A_LOAD_SCRATCH].data;
    sk->packed_data = a[A_PACK].data;
    sk->packed_size = a[A_PACK].logical_bytes;
    if (tp->plan.panel_bytes)
      sk->panel =
          (AlignedBuffer){.data = (char *)sk->work_data + align_line(sk->work_size), .bytes = tp->plan.panel_bytes};
    sk->kv_layers = memory_alloc((size_t)p->n_layers, sizeof(KvLayer), MEM_METADATA);
    int cut = tp->plan.kv_layers;
    for (int l = 0; l < p->n_layers; l++) {
      int tier = l < cut ? HBM : DDR;
      size_t at = (size_t)(tier == HBM ? l : l - cut) * tp->plan.kv_layer_bytes;
      size_t at16 = (size_t)(tier == HBM ? l : l - cut) * tp->plan.kv_bf16_layer_bytes;
      sk->kv_layers[l] = (KvLayer){
          (float *)((char *)a[A_K_HBM + tier].data + at), (float *)((char *)a[A_V_HBM + tier].data + at),
          tp->plan.kv_bf16_layer_bytes ? (uint16_t *)((char *)a[A_KVB_HBM + tier].data + at16) : NULL};
    }
    for (int t = 0; t < TIERS; t++)
      weight_bases[s][t] = a[A_WEIGHTS_HBM + t].data;
    exchange_regions[s] = &a[A_EXCHANGE];
    if (a[A_EXCHANGE].data)
      memset(a[A_EXCHANGE].data, 0, a[A_EXCHANGE].logical_bytes);
    if (tp->check) {
      sk->debug = memory_alloc((size_t)p->n_layers * p->dim, sizeof(float), MEM_ACTIVATION);
      sk->debug_hash = memory_alloc((size_t)p->n_layers, sizeof(uint64_t), MEM_ACTIVATION);
    }
  }
  tp->logits = tp->arenas[0][A_LOGITS].data;
  if (tp->batch_logits_rows)
    tp->batch_logits = tp->logits + p->vocab_size;
  shards_bind_placed(&tp->set, weight_bases, tp->arenas[0][A_EMBED].data);
  if (sockets == 2) {
    tp->exchange = exchange_create(tp->rows * p->dim, exchange_regions);
    /* The interleaved reduction flags blocks of 32 rows x 16 or 32 channels in the per-element flag array. */
    if (batch_reduce_mode() &&
        (long long)((p->dim + 15) / 16) * ((tp->rows + 31) / 32) > exchange_block_capacity(tp->exchange))
      fail("LLAMA_TP_BATCH_REDUCE=%s: %d x %d blocks exceed the exchange flags", tp_batch_reduce_name(),
           (p->dim + 15) / 16, (tp->rows + 31) / 32);
    if (profile_enabled)
      exchange_set_timing(tp->exchange, 1);
    if (pthread_create(&tp->thread, NULL, coordinator_main, tp))
      fail("cannot start the coordinator thread of socket 1");
    tp->started = 1;
  }
  run_both(tp, CMD_SETUP);
  for (int s = 0; s < sockets; s++)
    if (!tp->plan.simulated && (!team_affinity_verified(tp->s[s].team) || tp->s[s].leader_cpu != pool.cpus[s][0]))
      fail("socket %d worker team or coordinator affinity could not be verified", s);

  double start = monotonic_seconds();
  run_both(tp, CMD_LOAD);
  metrics.placement_populate_s = monotonic_seconds() - start;
  if (pack != TP_PACK_NONE) {
    start = monotonic_seconds();
    plan_packing(tp);
    run_both(tp, CMD_PACK);
    if (tp->arenas[0][A_PACK_CLS].data) {
      for (int s = 0; s < sockets; s++) {
        tp->s[s].cls_bf16 = tp->arenas[s][A_PACK_CLS].data;
        tp->bjobs[s][0] = pack_bf16_job_from_shard(tp->s[s].cls_bf16, shards_find(&tp->set, T_WCLS, -1, s));
      }
      run_both(tp, CMD_PACK_CLS);
      tp->cls_ready = 1;
    }
    metrics.weight_pack_s = monotonic_seconds() - start;
    metrics.weight_packed = 1;
  }
  start = monotonic_seconds();
  for (int s = 0; s < sockets; s++)
    for (int i = 0; i < A_COUNT; i++) {
      NumaRegion *r = &tp->arenas[s][i];
      if (!r->mapping_base)
        continue;
      region_mark_populated(r);
      if (i == A_LOAD_SCRATCH) {
        verify_bound(r, tp->plan.mode);
        region_release(r);
        tp->s[s].load_scratch = NULL;
        continue;
      }
      if (i >= A_K_HBM && i <= A_KVB_DDR)
        continue; /* K/V pages and their BF16 copy are checked once the prefix has been written. */
      if (i <= A_WEIGHTS_DDR || i == A_EMBED || i == A_PACK || i == A_PACK_CLS)
        region_protect(r);
      verify_bound(r, tp->plan.mode);
    }
  metrics.placement_verify_s = monotonic_seconds() - start;
  metrics.placement_owned = 1;
  return tp;
}

/* LLAMA_PROFILE=1: one line per stage and socket, in milliseconds per forward pass. wo/w2 are the whole projection
 * including its reduction, wo_mm/w2_mm the leader's own share of the projection (batched: the whole GEMM, its reduction
 * apart in the `exchange` line), so the slowest member's lateness shows up as `straggler`. The exchange components are
 * the leader's: straggler (waiting for the socket's last partial write), peer (waiting for the other socket), add,
 * close (final barrier) and buffer (waiting for a free partial buffer). With LLAMA_TP_BATCH_REDUCE=interleaved the
 * batched wo_mm/w2_mm are the whole fused job and the `exchange` line holds the leader's flag checks (in place of
 * straggler), its waits for peer blocks after its range (peer), its adds (add) and its wait for the rest of the team
 * (close). */
static void log_profile(const Tp *tp) {
  static const char *const names[TP_SLOTS] = {"embedding", "rope_table", "rmsnorm",   "quantize", "qkv",
                                              "rope",      "kv_store",   "attention", "wo",       "w1_w3",
                                              "w2",        "classifier", "wo_mm",     "w2_mm"};
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
      if (stage != STAGE_PREFILL)
        continue;
      used = (size_t)snprintf(line, sizeof(line), "tp prefill socket %d exchange (ms/token):", s);
      for (int m = 0; m < 2; m++)
        for (int i = 0; i < 4 && used < sizeof(line); i++)
          used += (size_t)snprintf(line + used, sizeof(line) - used, " %s_%s=%.4f", m ? "w2" : "wo", xnames[i],
                                   tp->xmat[s][stage][m][i] * 1e3 / (double)tp->tokens[stage]);
      LOG(LOG_INFO, "%s", line);
    }
  }
}

float *tp_forward(Tp *tp, int token, int pos, int logits) {
  tp->token = token;
  tp->pos = pos;
  tp->want_logits = logits;
  run_both(tp, CMD_FORWARD);
  if (tp->sockets == 2 && profile_enabled && !workers_aborted()) {
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
  return logits ? tp->logits : NULL;
}

float *tp_all_logits(Tp *tp, int rows) {
  if (rows > tp->batch_logits_rows) {
    if (tp->plan.mode)
      fail("all-position logits must be admitted with LLAMA_PREFILL_LOGITS=all");
    memory_free(tp->batch_logits);
    tp->batch_logits = memory_alloc((size_t)rows, checked_mul((size_t)tp->c.vocab_size, sizeof(float)), MEM_ACTIVATION);
    tp->batch_logits_rows = rows;
  }
  return tp->batch_logits;
}

float *tp_forward_batch(Tp *tp, const int *tokens, int count, int pos0, LogitsWanted want) {
  if (count < 1 || count > tp->rows || pos0 < 0 || pos0 > tp->c.seq_len - count)
    fail("two-socket batch out of range: count=%d (capacity %d) position=%d", count, tp->rows, pos0);
  if (want == LOGITS_ALL) {
    tp_all_logits(tp, tp->rows);
    if (tp->pack == TP_PACK_BF16 && !tp->cls_ready)
      pack_classifier(tp);
  }
  tp->btokens = tokens;
  tp->bcount = count;
  tp->bpos0 = pos0;
  tp->bwant = want;
  run_both(tp, CMD_FORWARD_BATCH);
  if (tp->sockets == 2 && profile_enabled && !workers_aborted()) {
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

static void planned_kv_stats(Tp *tp, int position, int verify, KvStats *out) {
  const Config *p = &tp->c;
  size_t all_layers = checked_mul(2 * (size_t)tp->sockets, (size_t)p->n_layers);
  size_t bytes_per_position = checked_mul((size_t)tp->kv_dim_local, sizeof(float));
  size_t bf16_total = ((size_t)p->seq_len + 31) / 32, bf16_blocks = ((size_t)position + 31) / 32;
  if (bf16_blocks > bf16_total)
    bf16_blocks = bf16_total;
  size_t bf16_head = checked_mul(bf16_total * 32, (size_t)p->head_dim); /* elements of one (K or V, head) segment */
  size_t bf16_used = checked_mul(checked_mul(bf16_blocks * 32, (size_t)p->head_dim), sizeof(uint16_t));
  size_t bf16_spans = checked_mul(2, (size_t)(p->n_kv_heads / tp->sockets)), bf16_stride = checked_mul(bf16_head, sizeof(uint16_t));
  *out = (KvStats){.reserved_bytes = checked_mul(all_layers, tp->plan.kv_layer_bytes),
                   .bf16_reserved_bytes = checked_mul(checked_mul((size_t)tp->sockets, (size_t)p->n_layers), tp->plan.kv_bf16_layer_bytes),
                   .bf16_active_bytes = tp->plan.kv_bf16_layer_bytes
                                            ? checked_mul(checked_mul(checked_mul((size_t)tp->sockets, (size_t)p->n_layers), bf16_spans),
                                                          bf16_used)
                                            : 0,
                   .active_bytes = checked_mul(checked_mul(all_layers, bytes_per_position), (size_t)position),
                   .active_prefix_only = 1,
                   .node = tp->plan.kv_layers == 0             ? tp->plan.node[0][DDR]
                           : tp->plan.kv_layers == p->n_layers ? tp->plan.node[0][HBM]
                                                               : -1};
  if (!verify || tp->plan.simulated)
    return;
  size_t spans = 1, row_bytes = bytes_per_position, page = region_alignment(-1, 0);
  if (kernels_kv_layout_head()) {
    spans = (size_t)(p->n_kv_heads / tp->sockets);
    row_bytes = checked_mul((size_t)p->head_dim, sizeof(float));
  }
  size_t stride = checked_mul(row_bytes, (size_t)p->seq_len);
  size_t used = checked_mul(row_bytes, (size_t)position), samples = region_verify_samples_default();
  out->full = 1;
  for (int s = 0; s < tp->sockets; s++)
    for (int i = A_K_HBM; i <= A_V_DDR; i++) {
      NumaRegion *r = &tp->arenas[s][i];
      if (!r->data)
        continue;
      size_t layers = r->logical_bytes / tp->plan.kv_layer_bytes;
      PageCheck check;
      size_t pages, thp = 0;
      if (region_verify_prefixes(r, checked_mul(layers, spans), stride, used, samples, &check, &pages))
        fail("cannot verify active %s pages on socket %d node %d", region_role_name(r->role), s, r->node);
      region_thp_bytes(r, &thp);
      region_record_verification(r, &check, thp);
      out->pages_queried += check.queried;
      out->pages_wrong_node += check.wrong_node;
      out->pages_missing += check.missing;
      out->full &= check.full;
      out->resident_bytes = checked_add(out->resident_bytes, checked_mul(pages, page));
      if (check.wrong_node || check.missing || check.errors)
        fail("active %s on socket %d node %d: %zu wrong-node, %zu missing, %zu page query errors",
             region_role_name(r->role), s, r->node, check.wrong_node, check.missing, check.errors);
    }
  for (int s = 0; s < tp->sockets; s++)
    for (int i = A_KVB_HBM; i <= A_KVB_DDR; i++) {
      NumaRegion *r = &tp->arenas[s][i];
      if (!r->data)
        continue;
      size_t layers = r->logical_bytes / tp->plan.kv_bf16_layer_bytes;
      PageCheck check;
      size_t pages, thp = 0;
      if (region_verify_prefixes(r, checked_mul(layers, bf16_spans), bf16_stride, bf16_used, samples, &check, &pages))
        fail("cannot verify active %s pages on socket %d node %d", region_role_name(r->role), s, r->node);
      region_thp_bytes(r, &thp);
      region_record_verification(r, &check, thp);
      out->pages_queried += check.queried;
      out->pages_wrong_node += check.wrong_node;
      out->pages_missing += check.missing;
      out->full &= check.full;
      out->bf16_resident_bytes = checked_add(out->bf16_resident_bytes, checked_mul(pages, page));
      if (check.wrong_node || check.missing || check.errors)
        fail("active %s on socket %d node %d: %zu wrong-node, %zu missing, %zu page query errors",
             region_role_name(r->role), s, r->node, check.wrong_node, check.missing, check.errors);
    }
  out->verified = 1;
}

void tp_kv_stats(Tp *tp, int position, int verify, KvStats *out) {
  if (tp->plan.mode) {
    planned_kv_stats(tp, position, verify, out);
    return;
  }
  if (tp->sockets == 1) {
    size_t active = checked_mul(checked_mul(checked_mul(2, (size_t)tp->c.n_layers), (size_t)position),
                                (size_t)tp->kv_dim_local * sizeof(float));
    runmem_kv_stats(&tp->mem, active, verify, out);
    return;
  }
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
  if (profile_enabled && tp->sockets == 2)
    log_profile(tp);
  exchange_destroy(tp->exchange);
  if (tp->sockets == 1 && !tp->plan.mode) {
    runmem_free(&tp->mem, tp->s[0].work_data);
    runmem_free(&tp->mem, tp->s[0].kv_layers[0].k);
    runmem_free(&tp->mem, tp->s[0].kv_layers[0].v);
    k_attention_free(&tp->s[0].attention);
    runmem_release(&tp->mem);
    packed_close(&tp->local_packed);
    packed_close(&tp->local_cls);
  }
  for (int s = 0; s < SOCKETS; s++) {
    TpSocket *sk = &tp->s[s];
    memory_free(sk->kv_layers);
    for (int i = 0; i < A_COUNT; i++)
      region_release(&tp->arenas[s][i]);
    memory_free(sk->debug);
    memory_free(sk->debug_hash);
    region_release(&sk->weights);
    region_release(&sk->packed);
    region_release(&sk->packed_cls);
    memory_free(sk->packed_at);
    memory_free(sk->pscales_at);
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
  if (!tp->plan.mode) {
    memory_free(tp->logits);
    memory_free(tp->batch_logits);
  }
  memory_free(tp);
}
