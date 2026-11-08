/* Where do Wo and W2 lose bandwidth? (measurement, not a test)
 *   reduce-bench [--sockets 1|2] [--threads N] [--model 8b|1b] [--precision fp32|q8] [--layers L] [--passes P]
 *                [--red fused|flags|bars|none|xonly] [--stamps 0|1] [--tier hbm|ddr] [--gap-us U]
 * Runs P passes over L synthetic decoder layers, one pinned team per socket, each layer being the three projections that
 * carry 85% of the weight bytes of a token, with the engine's kernels, row scheduling and exchange (src/tp.c does the same
 * sequence): quantize + Wo(+reduction), rmsnorm + quantize + W1/W3 (SwiGLU), quantize + W2(+reduction). Shard shapes are
 * those of one socket of a two-socket split. Weights are distinct per layer so nothing stays in cache.
 *   --red fused  the engine's fused form: every member projects its rows, then exchange_reduce_add (two sockets only)
 *   --red bars   the projection and the two local team barriers, no cross-socket exchange (isolates the local sync)
 *   --red none   the projection alone (isolates the kernel, its dispatch and join, and the row split)
 *   --red flags  the fused form with per-row flags instead of the two team barriers (exchange_reduce_add_flags)
 *   --red xonly  no projection: the exchange alone inside the same team job (the reduction's own latency)
 * With --stamps 1 every member timestamps its steps (start, end of projection, the exchange's four steps) and the leader
 * folds them after each job: the log breaks Wo and W2 into dispatch, projection spread, barrier 1, peer wait, add, barrier
 * 2 and join, and the skew between the sockets. Without LLAMA_TOPOLOGY pointing at two real sockets the threads are
 * unpinned and the memory unbound (a laptop run only checks that the program works). One JSON line per run. */
#include "kernels.h"
#include "numa.h"
#include "runtime.h"
#include "workers.h"
#include <assert.h>
#include <pthread.h>
#include <sched.h>
#include <stdatomic.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

enum { RED_FUSED, RED_BARS, RED_NONE, RED_XONLY, RED_FLAGS };
enum { P_QUANT_WO, P_WO, P_NORM_QUANT, P_W1W3, P_QUANT_W2, P_W2, P_COUNT };
static const char *const phase_name[P_COUNT] = {"quant_wo", "wo", "norm_quant", "w1_w3", "quant_w2", "w2"};
enum { GS = 64, MAX_ITER = 1 << 20, MAX_WORKERS = 256 };

typedef struct {
  _Alignas(64) double start, proj_end;
  ReduceStamps r;
  double pad[2];
} Stamp;

/* Folded stamps of one reduction job, seconds relative to the leader's dispatch, plus absolute times for the skew. */
typedef struct {
  double disp, start_max, end_first, end_mean, end_last, b1_last, peer_leader_wait, peer_last, add_last, b2_last, ret;
  double end_last_abs, b1_leader_abs;
} Fold;

typedef struct Context Context;
typedef struct {
  Context *c;
  int socket, red, q8;
  int n, d;
  float *partial, *residual;
  size_t ld;
  const float *x, *w;
  K8Input input;
  K8Weight wq;
  atomic_int claim;
  Stamp *stamps;
} Job;

struct Context {
  int sockets, threads, layers, passes, red, q8, stamps, dim, hidden, tier_ddr;
  double gap_us;
  Team *team[2];
  NumaRegion region[2], xregion[2];
  Exchange *exchange;
  /* per socket and layer: weights */
  struct {
    const float *wo, *w1, *w3, *w2;
    const int8_t *qo, *q1, *q3, *q2;
    const float *so, *s1, *s3, *s2;
  } layer[2][64];
  size_t shard_bytes[3]; /* wo, w1+w3, w2 in bytes per socket */
  float *x[2], *xb[2], *hb[2], *residual[2], *partial[2], *norm_weight;
  int8_t *xq[2], *aq[2];
  float *xs[2], *as[2];
  Stamp *stamp[2];
  /* results */
  double *phase_s[2]; /* [socket][(iter * P_COUNT) + phase] */
  double *layer_s[2];
  Fold *fold[2][2]; /* [socket][wo/w2][iter] */
  size_t iters;
  atomic_uint start_generation, done_generation, barrier_count, barrier_generation;
  unsigned issued;
  int stop;
};

static void socket_barrier(Context *c) {
  unsigned generation = atomic_load_explicit(&c->barrier_generation, memory_order_acquire);
  if (atomic_fetch_add_explicit(&c->barrier_count, 1, memory_order_acq_rel) + 1 == (unsigned)c->sockets) {
    atomic_store_explicit(&c->barrier_count, 0, memory_order_relaxed);
    atomic_store_explicit(&c->barrier_generation, generation + 1, memory_order_release);
  } else {
    while (atomic_load_explicit(&c->barrier_generation, memory_order_acquire) == generation)
      ;
  }
}

/* The exchange handle lives in the context; the job needs it too. */
static Exchange *g_exchange;
static Team *g_team[2];

static void bench_rows_done(void *p, int r0, int r1) {
  const Job *j = p;
  exchange_rows_done(g_exchange, j->socket, r0, r1);
}

static void job_run(const void *p, int worker, int workers) {
  Job *j = (Job *)p;
  Stamp *st = j->stamps ? &j->stamps[worker] : NULL;
  if (st)
    st->start = monotonic_seconds();
  if (j->red != RED_XONLY) {
    RowsDoneFn done = j->red == RED_FLAGS ? bench_rows_done : NULL;
    if (j->q8)
      k_matmul_q8_ld_worker(j->partial, &j->input, &j->wq, j->ld, j->n, j->d, GS, K_PARTIAL, worker, workers, &j->claim,
                            done, j);
    else
      k_matmul_ld_worker(j->partial, j->x, j->w, j->ld, j->n, j->d, K_PARTIAL, worker, workers, &j->claim, done, j);
  } else if (j->red == RED_XONLY) {
  }
  if (st) {
    st->proj_end = monotonic_seconds();
    st->r.after_barrier1 = st->r.after_peer = st->r.after_add = st->r.after_barrier2 = st->proj_end;
  }
  if (j->red == RED_FLAGS) {
    exchange_reduce_add_flags(g_exchange, j->socket, worker, workers, j->residual, j->d, st ? &st->r : NULL);
  } else if (j->red == RED_FUSED || j->red == RED_XONLY) {
    exchange_reduce_add_stamped(g_exchange, j->socket, worker, workers, j->residual, j->d, st ? &st->r : NULL);
  } else if (j->red == RED_BARS) {
    team_barrier(g_team[j->socket]);
    if (st)
      st->r.after_barrier1 = monotonic_seconds();
    team_barrier(g_team[j->socket]);
    if (st)
      st->r.after_barrier2 = st->r.after_add = st->r.after_peer = monotonic_seconds();
  }
}

static void fold_stamps(const Stamp *w, int n, double disp, double ret, Fold *f) {
  *f = (Fold){.disp = disp, .end_first = 1e30};
  for (int i = 0; i < n; i++) {
    double s = w[i].start, e = w[i].proj_end;
    f->end_mean += e;
    if (s > f->start_max)
      f->start_max = s;
    if (e < f->end_first)
      f->end_first = e;
    if (e > f->end_last)
      f->end_last = e;
    if (w[i].r.after_barrier1 > f->b1_last)
      f->b1_last = w[i].r.after_barrier1;
    if (w[i].r.after_peer > f->peer_last)
      f->peer_last = w[i].r.after_peer;
    if (w[i].r.after_add > f->add_last)
      f->add_last = w[i].r.after_add;
    if (w[i].r.after_barrier2 > f->b2_last)
      f->b2_last = w[i].r.after_barrier2;
  }
  f->end_mean /= n;
  f->peer_leader_wait = w[0].r.after_peer - w[0].r.after_barrier1;
  f->end_last_abs = f->end_last;
  f->b1_leader_abs = w[0].r.after_barrier1;
  f->start_max -= disp;
  f->end_first -= disp;
  f->end_mean -= disp;
  f->end_last -= disp;
  f->b1_last -= disp;
  f->peer_last -= disp;
  f->add_last -= disp;
  f->b2_last -= disp;
  f->ret = ret - disp;
}

static void quantize(Context *c, int8_t *q, float *scales, const float *x, int n) {
  if (c->q8)
    k_quantize(q, scales, x, n, GS);
}

/* One projection with its reduction (or the stand-in selected by --red). */
static void project(Context *c, int s, Job *job, int kind, size_t iter) {
  job->socket = s;
  job->red = c->red;
  job->q8 = c->q8;
  job->residual = c->residual[s];
  job->partial = c->partial[s];
  atomic_init(&job->claim, 0);
  job->stamps = c->stamps ? c->stamp[s] : NULL;
  if (c->red == RED_FUSED || c->red == RED_XONLY || c->red == RED_FLAGS) {
    job->partial = exchange_partial(c->exchange, s);
    assert(job->partial);
  }
  double disp = c->stamps ? monotonic_seconds() : 0.0;
  team_run(c->team[s], job_run, job);
  if (c->red == RED_FLAGS)
    exchange_reduce_finish(c->exchange, s);
  if (c->stamps && iter < c->iters)
    fold_stamps(c->stamp[s], c->threads, disp, monotonic_seconds(), &c->fold[s][kind][iter]);
}

static void layer_run(Context *c, int s, int l, size_t iter) {
  double t[P_COUNT + 1];
  Job job = {.c = c};
  const int dim = c->dim, hl = c->hidden / 2, dl = dim / 2;
  t[0] = monotonic_seconds();
  quantize(c, c->aq[s], c->as[s], c->xb[s], dl);
  t[1] = monotonic_seconds();
  job.n = dl;
  job.d = dim;
  job.ld = (size_t)dl;
  job.x = c->xb[s];
  job.w = c->layer[s][l].wo;
  job.wq = (K8Weight){c->layer[s][l].qo, c->layer[s][l].so};
  job.input = c->q8 ? (K8Input){c->aq[s], c->as[s], c->xb[s]} : (K8Input){NULL, NULL, c->xb[s]};
  project(c, s, &job, 0, iter);
  t[2] = monotonic_seconds();
  k_rmsnorm(c->xb[s], c->residual[s], c->norm_weight, dim, 1e-5f);
  quantize(c, c->xq[s], c->xs[s], c->xb[s], dim);
  t[3] = monotonic_seconds();
  K8Input in = c->q8 ? (K8Input){c->xq[s], c->xs[s], c->xb[s]} : (K8Input){NULL, NULL, c->xb[s]};
  if (c->q8) {
    K8Weight w1 = {c->layer[s][l].q1, c->layer[s][l].s1}, w3 = {c->layer[s][l].q3, c->layer[s][l].s3};
    k_matmul_q8_swiglu_ld(c->hb[s], &in, &w1, &w3, (size_t)dim, dim, hl, GS);
  } else {
    k_matmul_swiglu_ld(c->hb[s], c->xb[s], c->layer[s][l].w1, c->layer[s][l].w3, (size_t)dim, dim, hl);
  }
  t[4] = monotonic_seconds();
  quantize(c, c->aq[s], c->as[s], c->hb[s], hl);
  t[5] = monotonic_seconds();
  job.n = hl;
  job.d = dim;
  job.ld = (size_t)hl;
  job.x = c->hb[s];
  job.w = c->layer[s][l].w2;
  job.wq = (K8Weight){c->layer[s][l].q2, c->layer[s][l].s2};
  job.input = c->q8 ? (K8Input){c->aq[s], c->as[s], c->hb[s]} : (K8Input){NULL, NULL, c->hb[s]};
  project(c, s, &job, 1, iter);
  t[6] = monotonic_seconds();
  if (iter < c->iters) {
    double *out = c->phase_s[s] + iter * P_COUNT;
    for (int p = 0; p < P_COUNT; p++)
      out[p] = t[p + 1] - t[p];
    c->layer_s[s][iter] = t[6] - t[0];
  }
}

static void execute(Context *c, int s) {
  size_t iter = 0;
  for (int pass = 0; pass < c->passes + 1; pass++) { /* pass 0 warms up and is not recorded */
    socket_barrier(c);
    for (int l = 0; l < c->layers; l++) {
      layer_run(c, s, l, pass ? iter++ : c->iters);
      if (c->gap_us > 0) {
        double end = monotonic_seconds() + c->gap_us * 1e-6;
        while (monotonic_seconds() < end)
          ;
      }
    }
  }
}

static void *side_main(void *p) {
  Context *c = p;
  kernels_set_thread_team(c->team[1]);
  team_adopt_leader(c->team[1]);
  unsigned seen = 0;
  for (;;) {
    unsigned now;
    while ((now = atomic_load_explicit(&c->start_generation, memory_order_acquire)) == seen)
      sched_yield();
    seen = now;
    if (c->stop)
      return NULL;
    execute(c, 1);
    atomic_store_explicit(&c->done_generation, now, memory_order_release);
  }
}

static int cmp(const void *a, const void *b) {
  double x = *(const double *)a, y = *(const double *)b;
  return (x > y) - (x < y);
}

static double median(const double *v, size_t n, size_t stride, size_t offset, double *mean, double *p99) {
  double *tmp = malloc(n * sizeof(double));
  double sum = 0;
  for (size_t i = 0; i < n; i++) {
    tmp[i] = v[i * stride + offset];
    sum += tmp[i];
  }
  qsort(tmp, n, sizeof(double), cmp);
  double m = tmp[n / 2];
  if (mean)
    *mean = sum / (double)n;
  if (p99)
    *p99 = tmp[n * 99 / 100];
  free(tmp);
  return m;
}

typedef struct {
  float *base;
  size_t bytes;
  int q8;
} FillJob;

static void fill_fn(const void *p, int worker, int workers) {
  const FillJob *job = p;
  size_t i0 = job->bytes * (size_t)worker / (size_t)workers, i1 = job->bytes * (size_t)(worker + 1) / (size_t)workers;
  unsigned char *b = (unsigned char *)job->base;
  for (size_t i = i0; i < i1; i++)
    b[i] = (unsigned char)((i * 2654435761u) >> 24);
}

static void report(Context *c, const char *model, const char *red) {
  size_t n = c->iters;
  double gbps_phase[2][P_COUNT];
  printf("{\"model\": \"%s\", \"precision\": \"%s\", \"sockets\": %d, \"threads\": %d, \"red\": \"%s\", \"stamps\": %d, "
         "\"layers\": %d, \"iters\": %zu",
         model, c->q8 ? "q8" : "fp32", c->sockets, c->threads, red, c->stamps, c->layers, n);
  for (int s = 0; s < c->sockets; s++) {
    double layer_mean, layer_p99, layer_med = median(c->layer_s[s], n, 1, 0, &layer_mean, &layer_p99);
    printf(", \"socket%d\": {\"layer_us\": %.1f, \"layer_mean_us\": %.1f, \"layer_p99_us\": %.1f", s, layer_med * 1e6,
           layer_mean * 1e6, layer_p99 * 1e6);
    for (int p = 0; p < P_COUNT; p++) {
      double mean, p99, med = median(c->phase_s[s], n, P_COUNT, (size_t)p, &mean, &p99);
      printf(", \"%s_us\": %.1f, \"%s_mean_us\": %.1f", phase_name[p], med * 1e6, phase_name[p], mean * 1e6);
      /* effective GB/s as the engine's profile reports it: the phase's weight bytes of ALL sockets over its time */
      double bytes = p == P_WO ? (double)c->shard_bytes[0] : p == P_W1W3 ? (double)c->shard_bytes[1]
                     : p == P_W2 ? (double)c->shard_bytes[2] : 0.0;
      gbps_phase[s][p] = bytes ? bytes * c->sockets / med / 1e9 : 0.0;
      if (bytes)
        printf(", \"%s_gbps\": %.0f", phase_name[p], gbps_phase[s][p]);
    }
    printf("}");
  }
  printf("}\n");
  if (!c->stamps)
    return;
  /* Breakdown of the two reductions, microseconds, mean over the recorded jobs (and p99 of the total). */
  for (int kind = 0; kind < 2; kind++)
    for (int s = 0; s < c->sockets; s++) {
      const Fold *f = c->fold[s][kind];
      double sum[12] = {0}, skew = 0, abs_skew = 0, b1skew = 0, abs_b1skew = 0;
      double *tot = malloc(n * sizeof(double));
      for (size_t i = 0; i < n; i++) {
        double v[12] = {f[i].start_max, f[i].end_first, f[i].end_mean, f[i].end_last, f[i].b1_last, f[i].peer_leader_wait,
                        f[i].peer_last, f[i].add_last, f[i].b2_last, f[i].ret, 0, 0};
        for (int k = 0; k < 10; k++)
          sum[k] += v[k];
        tot[i] = f[i].ret;
        if (c->sockets == 2) {
          const Fold *o = &c->fold[1 - s][kind][i];
          double d = f[i].end_last_abs - o->end_last_abs, d1 = f[i].b1_leader_abs - o->b1_leader_abs;
          skew += d;
          abs_skew += d < 0 ? -d : d;
          b1skew += d1;
          abs_b1skew += d1 < 0 ? -d1 : d1;
        }
      }
      qsort(tot, n, sizeof(double), cmp);
      double us = 1e6 / (double)n;
      printf("{\"breakdown\": \"%s\", \"socket\": %d, \"dispatch_max_us\": %.1f, \"proj_first_us\": %.1f, "
             "\"proj_mean_us\": %.1f, \"proj_last_us\": %.1f, \"barrier1_done_us\": %.1f, \"peer_wait_us\": %.1f, "
             "\"peer_seen_last_us\": %.1f, \"add_done_us\": %.1f, \"barrier2_done_us\": %.1f, \"join_done_us\": %.1f, "
             "\"total_p50_us\": %.1f, \"total_p99_us\": %.1f, \"proj_last_skew_vs_peer_us\": %.1f, "
             "\"proj_last_abs_skew_us\": %.1f, \"ready_skew_vs_peer_us\": %.1f, \"ready_abs_skew_us\": %.1f}\n",
             kind ? "w2" : "wo", s, sum[0] * us, sum[1] * us, sum[2] * us, sum[3] * us, sum[4] * us, sum[5] * us,
             sum[6] * us, sum[7] * us, sum[8] * us, sum[9] * us, tot[n / 2] * 1e6, tot[n * 99 / 100] * 1e6, skew * us,
             abs_skew * us, b1skew * us, abs_b1skew * us);
      free(tot);
    }
}

int main(int argc, char **argv) {
  int sockets = 2, threads = 56, layers = 8, passes = 40, red = RED_FUSED, stamps = 0, q8 = 0, tier_ddr = 0;
  const char *model = "8b", *red_name = "fused";
  double gap = 0;
  for (int i = 1; i + 1 < argc; i += 2) {
    const char *k = argv[i], *v = argv[i + 1];
    if (!strcmp(k, "--sockets"))
      sockets = atoi(v);
    else if (!strcmp(k, "--threads"))
      threads = atoi(v);
    else if (!strcmp(k, "--model"))
      model = v;
    else if (!strcmp(k, "--precision"))
      q8 = !strcmp(v, "q8");
    else if (!strcmp(k, "--layers"))
      layers = atoi(v);
    else if (!strcmp(k, "--passes"))
      passes = atoi(v);
    else if (!strcmp(k, "--stamps"))
      stamps = atoi(v);
    else if (!strcmp(k, "--tier"))
      tier_ddr = !strcmp(v, "ddr");
    else if (!strcmp(k, "--gap-us"))
      gap = atof(v);
    else if (!strcmp(k, "--red")) {
      red_name = v;
      red = !strcmp(v, "fused") ? RED_FUSED : !strcmp(v, "bars") ? RED_BARS : !strcmp(v, "none") ? RED_NONE
            : !strcmp(v, "xonly") ? RED_XONLY : !strcmp(v, "flags") ? RED_FLAGS : -1;
    } else {
      fprintf(stderr, "unknown option %s\n", k);
      return 2;
    }
  }
  int dim = !strcmp(model, "8b") ? 4096 : !strcmp(model, "1b") ? 2048 : 0;
  int hidden = dim == 4096 ? 14336 : 8192;
  if (!dim || red < 0 || (sockets != 1 && sockets != 2) || threads < 1 || threads > MAX_WORKERS || layers < 1 ||
      layers > 64 || passes < 1 || ((red == RED_FUSED || red == RED_XONLY || red == RED_FLAGS) && sockets != 2)) {
    fprintf(stderr, "invalid options (fused and xonly need --sockets 2)\n");
    return 2;
  }
  kernels_init();
  const Topology *topology = numa_topology();
  const int real = topology->nsockets == 2 && topology->socket[0].cpus && topology->socket[1].cpus &&
                   topology->socket[0].ncpus >= threads && topology->socket[1].ncpus >= threads;
  static Context context;
  Context *c = &context;
  c->sockets = sockets;
  c->threads = threads;
  c->layers = layers;
  c->passes = passes;
  c->red = red;
  c->q8 = q8;
  c->stamps = stamps;
  c->dim = dim;
  c->hidden = hidden;
  c->gap_us = gap;
  c->iters = (size_t)layers * passes;
  PoolConfig config = {.teams = sockets,
                       .threads = {threads, threads},
                       .cpus = {real ? topology->socket[0].cpus : NULL, real ? topology->socket[1].cpus : NULL}};
  pool_start(&config);
  kernels_set_team(pool_team(0));
  for (int s = 0; s < sockets; s++)
    c->team[s] = g_team[s] = pool_team(s);

  /* Shapes of one socket's shard: Wo d=dim n=dim/2; W1 and W3 d=hidden/2 n=dim; W2 d=dim n=hidden/2. */
  const size_t wo_n = (size_t)dim * (dim / 2), w1_n = (size_t)(hidden / 2) * dim, w2_n = (size_t)dim * (hidden / 2);
  const size_t esize = q8 ? 1 : sizeof(float);
  const size_t scale_bytes[4] = {q8 ? wo_n / GS * 4 : 0, q8 ? w1_n / GS * 4 : 0, q8 ? w1_n / GS * 4 : 0,
                                 q8 ? w2_n / GS * 4 : 0};
  c->shard_bytes[0] = wo_n * esize + scale_bytes[0];
  c->shard_bytes[1] = 2 * w1_n * esize + scale_bytes[1] + scale_bytes[2];
  c->shard_bytes[2] = w2_n * esize + scale_bytes[3];
  size_t per_layer = 0;
  size_t part[8] = {wo_n * esize, scale_bytes[0], w1_n * esize, scale_bytes[1], w1_n * esize, scale_bytes[2], w2_n * esize,
                    scale_bytes[3]};
  for (int i = 0; i < 8; i++)
    per_layer += (part[i] + 4095) / 4096 * 4096;
  size_t total = per_layer * (size_t)layers;
  for (int s = 0; s < sockets; s++) {
    int node = real ? (tier_ddr ? topology->socket[s].ddr_node : topology->socket[s].hbm_node) : -1;
    region_reserve(&c->region[s], total, s, node, ROLE_WEIGHTS, region_thp_default());
    FillJob fill = {c->region[s].data, total, q8};
    team_run(c->team[s], fill_fn, &fill); /* first touch by the socket's own workers */
    region_mark_populated(&c->region[s]);
    char *cursor = c->region[s].data;
    for (int l = 0; l < layers; l++) {
      char *at = cursor + per_layer * (size_t)l;
      size_t off = 0;
      void *ptr[8];
      for (int i = 0; i < 8; i++) {
        ptr[i] = at + off;
        off += (part[i] + 4095) / 4096 * 4096;
      }
      if (q8) {
        /* int8 weights in [-127, 127] and positive scales: reinterpret the filler bytes' low 7 bits and a fixed scale */
        int8_t *qs[4] = {ptr[0], ptr[2], ptr[4], ptr[6]};
        size_t qn[4] = {wo_n, w1_n, w1_n, w2_n};
        for (int i = 0; i < 4; i++)
          for (size_t k = 0; k < qn[i]; k++)
            qs[i][k] = (int8_t)(qs[i][k] & 0x7f);
        float *ss[4] = {ptr[1], ptr[3], ptr[5], ptr[7]};
        size_t sn[4] = {scale_bytes[0] / 4, scale_bytes[1] / 4, scale_bytes[2] / 4, scale_bytes[3] / 4};
        for (int i = 0; i < 4; i++)
          for (size_t k = 0; k < sn[i]; k++)
            ss[i][k] = 0.001f;
        c->layer[s][l].qo = qs[0];
        c->layer[s][l].q1 = qs[1];
        c->layer[s][l].q3 = qs[2];
        c->layer[s][l].q2 = qs[3];
        c->layer[s][l].so = ss[0];
        c->layer[s][l].s1 = ss[1];
        c->layer[s][l].s3 = ss[2];
        c->layer[s][l].s2 = ss[3];
      } else {
        /* finite floats: the low mantissa bits of the filler give values around 1e-3..1 */
        float *fs[4] = {ptr[0], ptr[2], ptr[4], ptr[6]};
        size_t fn[4] = {wo_n, w1_n, w1_n, w2_n};
        for (int i = 0; i < 4; i++)
          for (size_t k = 0; k < fn[i]; k++) {
            unsigned bits;
            memcpy(&bits, &fs[i][k], 4);
            bits = 0x3c000000u | (bits & 0x007fffffu);
            memcpy(&fs[i][k], &bits, 4);
          }
        c->layer[s][l].wo = fs[0];
        c->layer[s][l].w1 = fs[1];
        c->layer[s][l].w3 = fs[2];
        c->layer[s][l].w2 = fs[3];
      }
    }
    PageCheck check = {0};
    if (node >= 0 && region_verify(&c->region[s], region_verify_samples_default(), &check) == 0)
      fprintf(stderr, "socket %d: %zu of %zu sampled pages on node %d\n", s, check.on_node, check.queried, node);
    c->x[s] = memory_alloc((size_t)dim, sizeof(float), MEM_ACTIVATION);
    c->xb[s] = memory_alloc((size_t)dim, sizeof(float), MEM_ACTIVATION);
    c->hb[s] = memory_alloc((size_t)hidden, sizeof(float), MEM_ACTIVATION);
    c->residual[s] = memory_alloc((size_t)dim, sizeof(float), MEM_ACTIVATION);
    c->partial[s] = memory_alloc((size_t)dim, sizeof(float), MEM_ACTIVATION);
    c->xq[s] = memory_alloc((size_t)dim, 1, MEM_ACTIVATION);
    c->xs[s] = memory_alloc((size_t)dim / GS + 1, sizeof(float), MEM_ACTIVATION);
    c->aq[s] = memory_alloc((size_t)hidden, 1, MEM_ACTIVATION);
    c->as[s] = memory_alloc((size_t)hidden / GS + 1, sizeof(float), MEM_ACTIVATION);
    for (int i = 0; i < dim; i++) {
      c->x[s][i] = 0.01f * (float)(i % 17);
      c->xb[s][i] = 0.01f * (float)(i % 13) - 0.05f;
      c->residual[s][i] = 0.01f * (float)(i % 7);
    }
    for (int i = 0; i < hidden; i++)
      c->hb[s][i] = 0.01f * (float)(i % 11) - 0.04f;
    if (posix_memalign((void **)&c->stamp[s], 64, (size_t)threads * sizeof(Stamp)))
      return 3;
    c->phase_s[s] = calloc(c->iters * P_COUNT, sizeof(double));
    c->layer_s[s] = calloc(c->iters, sizeof(double));
    for (int k = 0; k < 2; k++)
      c->fold[s][k] = calloc(c->iters, sizeof(Fold));
  }
  c->norm_weight = memory_alloc((size_t)dim, sizeof(float), MEM_ACTIVATION);
  for (int i = 0; i < dim; i++)
    c->norm_weight[i] = 1.0f;
  if (sockets == 2) {
    for (int s = 0; s < 2; s++) {
      int node = real ? (tier_ddr ? topology->socket[s].ddr_node : topology->socket[s].hbm_node) : -1;
      region_reserve(&c->xregion[s], exchange_region_bytes(dim), s, node, ROLE_WORK, 0);
      memset(c->xregion[s].data, 0, c->xregion[s].data_bytes);
    }
    NumaRegion *use[2] = {&c->xregion[0], &c->xregion[1]};
    c->exchange = g_exchange = exchange_create(dim, use);
  }
  pthread_t thread;
  if (sockets == 2)
    assert(!pthread_create(&thread, NULL, side_main, c));
  unsigned generation = ++c->issued;
  atomic_store_explicit(&c->start_generation, generation, memory_order_release);
  execute(c, 0);
  if (sockets == 2)
    while (atomic_load_explicit(&c->done_generation, memory_order_acquire) != generation)
      sched_yield();
  report(c, model, red_name);
  c->stop = 1;
  atomic_store_explicit(&c->start_generation, ++c->issued, memory_order_release);
  if (sockets == 2) {
    pthread_join(thread, NULL);
    exchange_destroy(c->exchange);
  }
  for (int s = 0; s < sockets; s++) {
    region_release(&c->region[s]);
    if (sockets == 2)
      region_release(&c->xregion[s]);
  }
  pool_stop();
  return 0;
}
