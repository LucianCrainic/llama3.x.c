/* Streaming-read ceiling of each socket's memory, and how close the engine's GEMV kernels come to it (measurement, not
 * a test).
 *   stream-bench [--sockets 1|2] [--threads N] [--mib M] [--reps R] [--tier hbm|ddr] [--only SUBSTRING]
 * One pinned team per socket reads a region bound to that socket's HBM (or DDR) node. With --sockets 2 both teams run at
 * the same time, as the tensor-parallel decode does, so the aggregate shows whether the two memories deliver their
 * bandwidth together. Experiments (one JSON line each, GB/s = 1e9 bytes/s, median of the repetitions):
 *   raw-chunk-pfD  every worker reads one contiguous chunk with 8 independent accumulators (D = software prefetch
 *                  distance in bytes, 0 = none): the ceiling of a pure stream
 *   raw-rows4-pfD  every worker reads four row streams at once (the access pattern of the FP32 GEMV), no arithmetic
 *                  beyond adding
 *   mini-gemv-rows4-pfD / mini-gemv-packed4-pfD   the GEMV arithmetic (four rows against x) over the engine's row layout
 *                  and over a packing in which four rows form one sequential stream (what a load-time repack would give)
 *   percall-*      the mini GEMV on 7168x4096 matrices, one team job per matrix as the engine issues them, statically
 *                  split or with workers claiming 8 or 32 rows at a time
 *   gemv-w1 / gemv-w2 / gemv-wo   the engine's k_matmul_ld on the 8B shapes of one socket's shard (7168x4096, 4096x7168,
 *                  4096x2048), rotating over copies so nothing stays in cache
 * Without LLAMA_TOPOLOGY pointing at two real sockets the threads are unpinned and the memory is unbound (a laptop
 * run only checks that the program works). */
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

#if defined(__x86_64__) && (defined(__GNUC__) || defined(__clang__))
#include <immintrin.h>
#define HAVE_X86 1
#define AVX512 __attribute__((target("avx512f,avx512dq,avx512bw,avx512vl,fma")))
#else
#define HAVE_X86 0
#endif

enum { MAX_REPS = 64 };
typedef struct Context Context;
typedef void (*WorkFn)(Context *c, int socket);

typedef struct {
  const char *name;
  WorkFn work;
  const void *arg;
  int passes;
  double bytes[2]; /* per socket and pass */
} Experiment;

struct Context {
  int sockets, threads, reps;
  Team *team[2];
  NumaRegion region[2];
  float *base[2];
  size_t floats; /* per socket */
  float *x[2], *out[2], *sink[2];
  /* coordination */
  atomic_uint start_generation, done_generation;
  atomic_uint barrier_count, barrier_generation;
  unsigned issued;
  Experiment *experiment;
  double seconds[2][MAX_REPS];
  int stop;
};

/* ---- the two-coordinator harness -------------------------------------------------------------------------------- */

static void barrier(Context *c) {
  unsigned generation = atomic_load_explicit(&c->barrier_generation, memory_order_acquire);
  if (atomic_fetch_add_explicit(&c->barrier_count, 1, memory_order_acq_rel) + 1 == (unsigned)c->sockets) {
    atomic_store_explicit(&c->barrier_count, 0, memory_order_relaxed);
    atomic_store_explicit(&c->barrier_generation, generation + 1, memory_order_release);
  } else {
    while (atomic_load_explicit(&c->barrier_generation, memory_order_acquire) == generation)
      ;
  }
}

static void execute(Context *c, int socket) {
  const Experiment *e = c->experiment;
  for (int rep = 0; rep < c->reps; rep++) {
    barrier(c);
    double start = monotonic_seconds();
    for (int pass = 0; pass < e->passes; pass++)
      e->work(c, socket);
    c->seconds[socket][rep] = monotonic_seconds() - start;
  }
}

typedef struct {
  Context *c;
} SideArg;

static void *side_main(void *p) {
  Context *c = ((SideArg *)p)->c;
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

static int compare_double(const void *a, const void *b) {
  double x = *(const double *)a, y = *(const double *)b;
  return (x > y) - (x < y);
}

static void run_experiment(Context *c, Experiment *e) {
  c->experiment = e;
  unsigned generation = ++c->issued;
  atomic_store_explicit(&c->start_generation, generation, memory_order_release);
  execute(c, 0);
  if (c->sockets == 2)
    while (atomic_load_explicit(&c->done_generation, memory_order_acquire) != generation)
      sched_yield();
  double socket_gbps[2] = {0, 0}, aggregate[MAX_REPS], per[2][MAX_REPS];
  for (int s = 0; s < c->sockets; s++)
    for (int r = 0; r < c->reps; r++)
      per[s][r] = e->bytes[s] * e->passes / c->seconds[s][r] / 1e9;
  for (int r = 0; r < c->reps; r++) {
    aggregate[r] = 0;
    for (int s = 0; s < c->sockets; s++)
      aggregate[r] += per[s][r];
  }
  for (int s = 0; s < c->sockets; s++) {
    qsort(per[s], (size_t)c->reps, sizeof(double), compare_double);
    socket_gbps[s] = per[s][c->reps / 2];
  }
  qsort(aggregate, (size_t)c->reps, sizeof(double), compare_double);
  printf("{\"experiment\": \"%s\", \"sockets\": %d, \"threads\": %d, \"socket_gbps\": [%.1f, %.1f], \"aggregate_gbps\": "
         "%.1f, \"aggregate_min\": %.1f, \"aggregate_max\": %.1f}\n",
         e->name, c->sockets, c->threads, socket_gbps[0], socket_gbps[1], aggregate[c->reps / 2], aggregate[0],
         aggregate[c->reps - 1]);
  fflush(stdout);
}

/* ---- raw read kernels ------------------------------------------------------------------------------------------- */

typedef struct {
  const float *base;
  size_t floats, row; /* row = floats per row of the rows4 pattern */
  int prefetch;       /* bytes ahead; 0 = none */
  float *sink;        /* one float per worker, a cache line apart */
  int mode;           /* READ_CHUNK, READ_ROWS4, READ_GEMV4 or READ_PACKED4 */
  const float *x;     /* GEMV modes: the activation vector */
  int chunk_rows;     /* 0 = static split; else workers claim this many rows at a time from `next` */
  atomic_int *next;
} ReadJob;
enum { READ_CHUNK, READ_ROWS4, READ_GEMV4, READ_PACKED4 };

#if HAVE_X86
AVX512 static float read_chunk(const float *p, size_t n, int prefetch) {
  __m512 a0 = _mm512_setzero_ps(), a1 = a0, a2 = a0, a3 = a0, a4 = a0, a5 = a0, a6 = a0, a7 = a0;
  const size_t ahead = (size_t)prefetch / sizeof(float);
  for (size_t i = 0; i + 128 <= n; i += 128) {
    if (ahead) {
      _mm_prefetch((const char *)(p + i + ahead), _MM_HINT_T0);
      _mm_prefetch((const char *)(p + i + ahead + 16), _MM_HINT_T0);
      _mm_prefetch((const char *)(p + i + ahead + 32), _MM_HINT_T0);
      _mm_prefetch((const char *)(p + i + ahead + 48), _MM_HINT_T0);
      _mm_prefetch((const char *)(p + i + ahead + 64), _MM_HINT_T0);
      _mm_prefetch((const char *)(p + i + ahead + 80), _MM_HINT_T0);
      _mm_prefetch((const char *)(p + i + ahead + 96), _MM_HINT_T0);
      _mm_prefetch((const char *)(p + i + ahead + 112), _MM_HINT_T0);
    }
    a0 = _mm512_add_ps(a0, _mm512_loadu_ps(p + i));
    a1 = _mm512_add_ps(a1, _mm512_loadu_ps(p + i + 16));
    a2 = _mm512_add_ps(a2, _mm512_loadu_ps(p + i + 32));
    a3 = _mm512_add_ps(a3, _mm512_loadu_ps(p + i + 48));
    a4 = _mm512_add_ps(a4, _mm512_loadu_ps(p + i + 64));
    a5 = _mm512_add_ps(a5, _mm512_loadu_ps(p + i + 80));
    a6 = _mm512_add_ps(a6, _mm512_loadu_ps(p + i + 96));
    a7 = _mm512_add_ps(a7, _mm512_loadu_ps(p + i + 112));
  }
  return _mm512_reduce_add_ps(
      _mm512_add_ps(_mm512_add_ps(_mm512_add_ps(a0, a1), _mm512_add_ps(a2, a3)), _mm512_add_ps(_mm512_add_ps(a4, a5), _mm512_add_ps(a6, a7))));
}

/* Four rows at a time, two cache lines from each per step: the stream layout of the FP32 GEMV. */
AVX512 static float read_rows4(const float *p, size_t rows, size_t row, int prefetch) {
  __m512 a[4] = {_mm512_setzero_ps(), _mm512_setzero_ps(), _mm512_setzero_ps(), _mm512_setzero_ps()};
  const size_t ahead = (size_t)prefetch / sizeof(float);
  for (size_t r = 0; r + 4 <= rows; r += 4) {
    const float *w0 = p + r * row;
    for (size_t j = 0; j + 32 <= row; j += 32) {
      for (int k = 0; k < 4; k++) {
        const float *w = w0 + (size_t)k * row + j;
        if (ahead) {
          _mm_prefetch((const char *)(w + ahead), _MM_HINT_T0);
          _mm_prefetch((const char *)(w + ahead + 16), _MM_HINT_T0);
        }
        a[k] = _mm512_add_ps(a[k], _mm512_add_ps(_mm512_loadu_ps(w), _mm512_loadu_ps(w + 16)));
      }
    }
  }
  return _mm512_reduce_add_ps(_mm512_add_ps(_mm512_add_ps(a[0], a[1]), _mm512_add_ps(a[2], a[3])));
}

/* The FP32 GEMV of the engine in miniature: four rows against x, two accumulators per row, rows `row` floats apart. */
AVX512 static float gemv_rows4(const float *p, size_t rows, size_t row, const float *x, int prefetch) {
  const size_t ahead = (size_t)prefetch / sizeof(float);
  float total = 0;
  for (size_t r = 0; r + 4 <= rows; r += 4) {
    const float *w[4] = {p + r * row, p + (r + 1) * row, p + (r + 2) * row, p + (r + 3) * row};
    __m512 a[4] = {_mm512_setzero_ps(), _mm512_setzero_ps(), _mm512_setzero_ps(), _mm512_setzero_ps()};
    __m512 b[4] = {_mm512_setzero_ps(), _mm512_setzero_ps(), _mm512_setzero_ps(), _mm512_setzero_ps()};
    for (size_t j = 0; j + 32 <= row; j += 32) {
      __m512 x0 = _mm512_loadu_ps(x + j), x1 = _mm512_loadu_ps(x + j + 16);
      for (int k = 0; k < 4; k++) {
        if (ahead) {
          _mm_prefetch((const char *)(w[k] + j + ahead), _MM_HINT_T0);
          _mm_prefetch((const char *)(w[k] + j + ahead + 16), _MM_HINT_T0);
        }
        a[k] = _mm512_fmadd_ps(_mm512_loadu_ps(w[k] + j), x0, a[k]);
        b[k] = _mm512_fmadd_ps(_mm512_loadu_ps(w[k] + j + 16), x1, b[k]);
      }
    }
    total += _mm512_reduce_add_ps(_mm512_add_ps(_mm512_add_ps(a[0], b[0]), _mm512_add_ps(_mm512_add_ps(a[1], b[1]),
             _mm512_add_ps(_mm512_add_ps(a[2], b[2]), _mm512_add_ps(a[3], b[3])))));
  }
  return total;
}

/* The same arithmetic over weights packed so that a block of four rows is ONE sequential stream: 16 floats of row 0,
 * of row 1, of row 2, of row 3, then the next 16 columns. Only the layout differs from gemv_rows4. */
AVX512 static float gemv_packed4(const float *p, size_t rows, size_t row, const float *x, int prefetch) {
  const size_t ahead = (size_t)prefetch / sizeof(float);
  float total = 0;
  for (size_t r = 0; r + 4 <= rows; r += 4) {
    const float *w = p + r * row;
    __m512 a[4] = {_mm512_setzero_ps(), _mm512_setzero_ps(), _mm512_setzero_ps(), _mm512_setzero_ps()};
    for (size_t j = 0; j + 16 <= row; j += 16) {
      __m512 x0 = _mm512_loadu_ps(x + j);
      const float *q = w + j * 4;
      if (ahead) {
        _mm_prefetch((const char *)(q + ahead), _MM_HINT_T0);
        _mm_prefetch((const char *)(q + ahead + 16), _MM_HINT_T0);
        _mm_prefetch((const char *)(q + ahead + 32), _MM_HINT_T0);
        _mm_prefetch((const char *)(q + ahead + 48), _MM_HINT_T0);
      }
      a[0] = _mm512_fmadd_ps(_mm512_loadu_ps(q), x0, a[0]);
      a[1] = _mm512_fmadd_ps(_mm512_loadu_ps(q + 16), x0, a[1]);
      a[2] = _mm512_fmadd_ps(_mm512_loadu_ps(q + 32), x0, a[2]);
      a[3] = _mm512_fmadd_ps(_mm512_loadu_ps(q + 48), x0, a[3]);
    }
    total += _mm512_reduce_add_ps(_mm512_add_ps(_mm512_add_ps(a[0], a[1]), _mm512_add_ps(a[2], a[3])));
  }
  return total;
}
#else
static float read_chunk(const float *p, size_t n, int prefetch) {
  (void)prefetch;
  float s = 0;
  for (size_t i = 0; i < n; i++)
    s += p[i];
  return s;
}
static float read_rows4(const float *p, size_t rows, size_t row, int prefetch) {
  return read_chunk(p, rows * row, prefetch);
}
static float gemv_rows4(const float *p, size_t rows, size_t row, const float *x, int prefetch) {
  (void)x;
  return read_chunk(p, rows * row, prefetch);
}
static float gemv_packed4(const float *p, size_t rows, size_t row, const float *x, int prefetch) {
  (void)x;
  return read_chunk(p, rows * row, prefetch);
}
#endif

static void read_fn(const void *p, int worker, int workers) {
  const ReadJob *job = p;
  float sum;
  if (job->mode == READ_CHUNK) {
    size_t lines = job->floats / 128;
    size_t i0 = lines * (size_t)worker / (size_t)workers * 128, i1 = lines * (size_t)(worker + 1) / (size_t)workers * 128;
    sum = read_chunk(job->base + i0, i1 - i0, job->prefetch);
  } else if (job->chunk_rows) {
    size_t rows = job->floats / job->row;
    sum = 0;
    for (;;) {
      size_t r0 = (size_t)atomic_fetch_add_explicit(job->next, job->chunk_rows, memory_order_relaxed);
      if (r0 >= rows)
        break;
      size_t r1 = r0 + (size_t)job->chunk_rows < rows ? r0 + (size_t)job->chunk_rows : rows;
      const float *w = job->base + r0 * job->row;
      sum += job->mode == READ_PACKED4 ? gemv_packed4(w, r1 - r0, job->row, job->x, job->prefetch)
                                       : gemv_rows4(w, r1 - r0, job->row, job->x, job->prefetch);
    }
  } else {
    size_t rows = job->floats / job->row;
    size_t r0 = rows * (size_t)worker / (size_t)workers / 4 * 4, r1 = rows * (size_t)(worker + 1) / (size_t)workers / 4 * 4;
    const float *w = job->base + r0 * job->row;
    if (job->mode == READ_ROWS4)
      sum = read_rows4(w, r1 - r0, job->row, job->prefetch);
    else if (job->mode == READ_GEMV4)
      sum = gemv_rows4(w, r1 - r0, job->row, job->x, job->prefetch);
    else
      sum = gemv_packed4(w, r1 - r0, job->row, job->x, job->prefetch);
  }
  job->sink[worker * 16] = sum;
}

static void work_read(Context *c, int socket) {
  const ReadJob *proto = ((const Experiment *)c->experiment)->arg;
  ReadJob job = *proto;
  job.base = c->base[socket];
  job.floats = c->floats;
  job.sink = c->sink[socket];
  job.x = c->x[socket];
  team_run(c->team[socket], read_fn, &job);
}

/* The mini GEMV one 7168x4096 matrix per team job, as the engine issues them (one dispatch and join per matrix),
 * statically split or with workers claiming `chunk_rows` rows at a time. */
typedef struct {
  ReadJob proto;
} MiniArg;

static void work_mini_percall(Context *c, int socket) {
  const ReadJob *proto = &((const MiniArg *)((const Experiment *)c->experiment)->arg)->proto;
  size_t matrix = (size_t)7168 * 4096, copies = c->floats / matrix;
  for (size_t k = 0; k < copies; k++) {
    atomic_int next;
    atomic_init(&next, 0);
    ReadJob job = *proto;
    job.base = c->base[socket] + k * matrix;
    job.floats = matrix;
    job.sink = c->sink[socket];
    job.x = c->x[socket];
    job.next = &next;
    team_run(c->team[socket], read_fn, &job);
  }
}

/* ---- GEMV on the engine's kernels ------------------------------------------------------------------------------- */

typedef struct {
  int d, n; /* rows, columns of one matrix */
} Shape;

static void work_gemv(Context *c, int socket) {
  const Shape *shape = ((const Experiment *)c->experiment)->arg;
  size_t matrix = (size_t)shape->d * shape->n;
  size_t copies = c->floats / matrix;
  for (size_t k = 0; k < copies; k++)
    k_matmul_ld(c->out[socket], c->x[socket], c->base[socket] + k * matrix, (size_t)shape->n, shape->n, shape->d, K_STORE);
}

/* ---- setup ------------------------------------------------------------------------------------------------------ */

typedef struct {
  float *base;
  size_t floats;
} FillJob;

static void fill_fn(const void *p, int worker, int workers) {
  const FillJob *job = p;
  size_t i0 = job->floats * (size_t)worker / (size_t)workers, i1 = job->floats * (size_t)(worker + 1) / (size_t)workers;
  for (size_t i = i0; i < i1; i++)
    job->base[i] = 0.001f * (float)(i % 1021) - 0.5f;
}

int main(int argc, char **argv) {
  int sockets = 2, threads = 56, reps = 7;
  size_t mib = 4096;
  const char *tier = "hbm", *only = NULL;
  for (int i = 1; i + 1 < argc; i += 2) {
    if (!strcmp(argv[i], "--sockets"))
      sockets = atoi(argv[i + 1]);
    else if (!strcmp(argv[i], "--threads"))
      threads = atoi(argv[i + 1]);
    else if (!strcmp(argv[i], "--mib"))
      mib = (size_t)atoll(argv[i + 1]);
    else if (!strcmp(argv[i], "--reps"))
      reps = atoi(argv[i + 1]);
    else if (!strcmp(argv[i], "--tier"))
      tier = argv[i + 1];
    else if (!strcmp(argv[i], "--only"))
      only = argv[i + 1];
    else {
      fprintf(stderr, "unknown option %s\n", argv[i]);
      return 2;
    }
  }
  if ((sockets != 1 && sockets != 2) || threads < 1 || reps < 1 || reps > MAX_REPS || mib < 64 ||
      (strcmp(tier, "hbm") && strcmp(tier, "ddr"))) {
    fprintf(stderr, "invalid options\n");
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
  c->reps = reps;
  PoolConfig config = {.teams = sockets,
                       .threads = {threads, threads},
                       .cpus = {real ? topology->socket[0].cpus : NULL, real ? topology->socket[1].cpus : NULL}};
  pool_start(&config);
  kernels_set_team(pool_team(0));
  /* A whole number of 7168x4096 matrices (112 MiB, the 8B W1 shard) when the region is that large, else of 128 rows. */
  size_t unit = (size_t)7168 * 4096, floats = (mib << 20) / sizeof(float);
  floats = floats >= unit ? floats / unit * unit : floats / (4096 * 128) * (4096 * 128);
  c->floats = floats;
  for (int s = 0; s < sockets; s++) {
    c->team[s] = pool_team(s);
    int node = real ? (!strcmp(tier, "hbm") ? topology->socket[s].hbm_node : topology->socket[s].ddr_node) : -1;
    region_reserve(&c->region[s], floats * sizeof(float), s, node, ROLE_WEIGHTS, region_thp_default());
    c->base[s] = c->region[s].data;
    FillJob fill = {c->base[s], floats};
    team_run(c->team[s], fill_fn, &fill); /* first touch by the socket's own workers */
    region_mark_populated(&c->region[s]);
    PageCheck check = {0};
    if (node >= 0 && region_verify(&c->region[s], region_verify_samples_default(), &check) == 0)
      fprintf(stderr, "socket %d: %zu of %zu sampled pages on node %d\n", s, check.on_node, check.queried, node);
    c->x[s] = memory_alloc(16384, sizeof(float), MEM_ACTIVATION);
    c->out[s] = memory_alloc(16384, sizeof(float), MEM_ACTIVATION);
    c->sink[s] = memory_alloc((size_t)threads * 16, sizeof(float), MEM_ACTIVATION);
    for (int i = 0; i < 16384; i++)
      c->x[s][i] = 0.01f * (float)(i % 17);
  }
  pthread_t thread;
  SideArg arg = {c};
  if (sockets == 2)
    assert(!pthread_create(&thread, NULL, side_main, &arg));

  const double bytes = (double)floats * sizeof(float);
  static const MiniArg mini[] = {
      {{.mode = READ_GEMV4, .row = 4096}},
      {{.mode = READ_GEMV4, .row = 4096, .chunk_rows = 32}},
      {{.mode = READ_GEMV4, .row = 4096, .chunk_rows = 8}},
      {{.mode = READ_PACKED4, .row = 4096, .prefetch = 4096}},
      {{.mode = READ_PACKED4, .row = 4096, .prefetch = 4096, .chunk_rows = 32}},
      {{.mode = READ_PACKED4, .row = 4096, .prefetch = 4096, .chunk_rows = 8}}};
  static const char *const mini_names[] = {"percall-rows4-static", "percall-rows4-dyn32", "percall-rows4-dyn8",
                                           "percall-packed4-pf4096-static", "percall-packed4-pf4096-dyn32",
                                           "percall-packed4-pf4096-dyn8"};
  static const Shape w1 = {7168, 4096}, w2 = {4096, 7168}, wo = {4096, 2048};
  const struct {
    const char *name;
    ReadJob read;
    const Shape *shape;
  } plan[] = {
      {"raw-chunk-pf0", {.mode = READ_CHUNK}, NULL},
      {"raw-chunk-pf1024", {.mode = READ_CHUNK, .prefetch = 1024}, NULL},
      {"raw-chunk-pf4096", {.mode = READ_CHUNK, .prefetch = 4096}, NULL},
      {"raw-rows4-pf0", {.mode = READ_ROWS4, .row = 4096}, NULL},
      {"raw-rows4-pf1024", {.mode = READ_ROWS4, .row = 4096, .prefetch = 1024}, NULL},
      {"raw-rows4-pf4096", {.mode = READ_ROWS4, .row = 4096, .prefetch = 4096}, NULL},
      {"mini-gemv-rows4-pf0", {.mode = READ_GEMV4, .row = 4096}, NULL},
      {"mini-gemv-rows4-pf1024", {.mode = READ_GEMV4, .row = 4096, .prefetch = 1024}, NULL},
      {"mini-gemv-rows4-pf2048", {.mode = READ_GEMV4, .row = 4096, .prefetch = 2048}, NULL},
      {"mini-gemv-rows4-pf4096", {.mode = READ_GEMV4, .row = 4096, .prefetch = 4096}, NULL},
      {"mini-gemv-packed4-pf0", {.mode = READ_PACKED4, .row = 4096}, NULL},
      {"mini-gemv-packed4-pf2048", {.mode = READ_PACKED4, .row = 4096, .prefetch = 2048}, NULL},
      {"mini-gemv-packed4-pf4096", {.mode = READ_PACKED4, .row = 4096, .prefetch = 4096}, NULL},
      {"mini-gemv-packed4-pf8192", {.mode = READ_PACKED4, .row = 4096, .prefetch = 8192}, NULL},
      {"gemv-w1", {0}, &w1},
      {"gemv-w2", {0}, &w2},
      {"gemv-wo", {0}, &wo}};
  for (size_t k = 0; k < sizeof(mini) / sizeof(*mini); k++) {
    if (only && !strstr(mini_names[k], only))
      continue;
    Experiment e = {.name = mini_names[k], .work = work_mini_percall, .arg = &mini[k], .passes = 3};
    e.bytes[0] = e.bytes[1] = (double)(floats / ((size_t)7168 * 4096)) * 7168 * 4096 * sizeof(float);
    run_experiment(c, &e);
  }
  for (size_t k = 0; k < sizeof(plan) / sizeof(*plan); k++) {
    if (only && !strstr(plan[k].name, only))
      continue;
    Experiment e = {.name = plan[k].name, .work = plan[k].shape ? work_gemv : work_read,
                    .arg = plan[k].shape ? (const void *)plan[k].shape : (const void *)&plan[k].read, .passes = 3};
    double per_pass = bytes;
    if (plan[k].shape)
      per_pass = (double)(floats / ((size_t)plan[k].shape->d * plan[k].shape->n)) * plan[k].shape->d * plan[k].shape->n *
                 sizeof(float);
    e.bytes[0] = e.bytes[1] = per_pass;
    run_experiment(c, &e);
  }
  c->stop = 1;
  atomic_store_explicit(&c->start_generation, ++c->issued, memory_order_release);
  if (sockets == 2)
    pthread_join(thread, NULL);
  for (int s = 0; s < sockets; s++)
    region_release(&c->region[s]);
  pool_stop();
  return 0;
}
