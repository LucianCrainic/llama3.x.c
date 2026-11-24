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
 * run only checks that the program works).
 *
 * Concurrent-tier research mode (docs/concurrent-tier-bandwidth.md): stream-bench --exp LABEL --cfgs FILE ... runs
 * windows in which a socket's cores read its HBM and its DDR at the same time (split teams, shared cores, page-interleaved
 * regions, or the engine's Q8 GEMV over a 70B shard split across the two tiers). It is selected by --exp and leaves the
 * experiments above untouched; its options are listed in the section "concurrent-tier research mode" below. */
#include "kernels.h"
#include "numa.h"
#include "runtime.h"
#include "workers.h"
#include <assert.h>
#include <errno.h>
#include <fcntl.h>
#include <pthread.h>
#include <sched.h>
#include <stdatomic.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <unistd.h>
#ifdef __linux__
#include <sys/mman.h>
#include <sys/syscall.h>
#include <sys/wait.h>
#endif

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

/* ---- concurrent-tier research mode ------------------------------------------------------------------------------ */
/* stream-bench --exp LABEL [--sockets 1|2] [--threads 56] [--page 2m|4k] [--region-mib 20480] [--reps R] [--gap S]
 *              [--cfgs FILE] [--cfg "k=v,k=v"]... [--perf-cmd SHELL --perf-out FILE --perf-stop FILE] [--verify-stride
 * N] Every configuration is a window: all workers of a socket start together (on a shared absolute start time, so both
 * sockets of an E4 run overlap), run for `warm` unmeasured seconds, then `seconds` measured ones (kind=gemv:
 * warm_passes and passes over 40 layers). With --reps R the loop is `for rep: for cfg`, so configurations are
 * interleaved. One JSON line per window and socket. Idle gaps (--gap) separate the windows; the perf process started by
 * --perf-cmd prints interval counters, so tools/concurrent_tier.py attributes each burst of memory traffic to its
 * window and converts CAS counts to bytes. Configuration keys (all optional except kind): kind=split   ch cores read
 * HBM and cd cores read DDR at once (cores beyond ch+cd idle; alt=1 spreads the DDR cores) kind=shared  every core
 * reads both tiers in the byte ratio a:b, as chunks of `unit` lines (unit>=8) or, with unit=1, as two interleaved
 * streams (a lines of HBM, b lines of DDR) kind=inter   one region whose pages are placed HBM:DDR = a:b (per page,
 * set_mempolicy before first touch); all cores read kind=gemv    the Q8 decode GEMV over a 70B per-socket shard,
 * gs=32|64, a fraction r of the rows of every matrix in the DDR copy, cdw workers on DDR rows (gshared=1: every worker
 * does both parts, alternating the order) streams=1|2|4|8 pf=bytes hint=0|1|2 (T0, T1, NTA) nt=0|1 (non-temporal load)
 * probe=0|1|2 (pointer chase on the last core in HBM|DDR) volume=MiB (split: fixed volume per worker instead of fixed
 * time) warm seconds warm_passes passes */
enum { CFG_SPLIT, CFG_SHARED, CFG_INTER, CFG_GEMV };
typedef struct {
  char name[96];
  int kind, ch, cd, a, b, unit, streams, hint, nt, pf, alt, probe, volume;
  int dstreams, dhint, dnt, dpf; /* the DDR group's own settings (-1 = those of the HBM group) */
  int gs, gshared, cdw, act, warm_passes, passes; /* act: active workers of a gemv window (0 = all) */
  double r, warm, seconds;
} Cfg;

enum { MIB = 1 << 20, LOG_BYTES = 64 << 20, LOG_MAX = 160, GEMV_LAYERS = 40, MAX_CFGS = 512 };
typedef struct {
  double t_begin, t_end, bytes_total[2], bytes_meas[2], probe_ns, probe_loads, lt[LOG_MAX];
  int nlog;
  long sink;
} __attribute__((aligned(128))) WRes;

typedef struct {
  int socket, hbm_node, ddr_node;
  Team *team;
  NumaRegion R[2]; /* 0 = HBM, 1 = DDR */
  char *base[2];
  size_t bytes; /* of R[t] */
  void *xmap;   /* the page-interleaved region of kind=inter */
  char *xbase;
  size_t xmap_bytes, xbytes;
  NumaRegion P[2]; /* pointer-chase regions */
  size_t pbytes;
  WRes res[128];
  double pass_t[256];
  int8_t *xq;
  float *xs, *out[7];
} Sock;

typedef struct {
  const Cfg *cfg;
  Sock *sock;
  double t_go;
} WinJob;

typedef struct {
  int sockets, threads, real, page4k, verify_stride;
  size_t inter_mib;
  const Topology *topology;
  Sock sock[2];
  atomic_uint go_gen, done_gen;
  const Cfg *cfg;
  double t_go;
  int stop;
} Run;

/* ---- the read kernels: S lockstep streams, software prefetch distance, hint and load flavor -----------------------
 */
typedef long (*RdFn)(const char *p, size_t lines, size_t pf);
#if HAVE_X86
#define DEFINE_RD(S, HN, HINT, NT)                                                                                     \
  AVX512 static long rd_##S##_##HN##_##NT(const char *p, size_t lines, size_t pf) {                                    \
    __m512i acc[4] = {_mm512_setzero_si512(), _mm512_setzero_si512(), _mm512_setzero_si512(), _mm512_setzero_si512()}; \
    size_t m = lines / (S);                                                                                            \
    for (size_t i = 0; i < m; i++) {                                                                                   \
      _Pragma("GCC unroll 8") for (int k = 0; k < (S); k++) {                                                          \
        const char *a = p + ((size_t)k * m + i) * 64;                                                                  \
        if (pf)                                                                                                        \
          _mm_prefetch(a + pf, HINT);                                                                                  \
        __m512i v = (NT) ? _mm512_stream_load_si512((void *)a) : _mm512_loadu_si512((const void *)a);                  \
        acc[k & 3] = _mm512_add_epi32(acc[k & 3], v);                                                                  \
      }                                                                                                                \
    }                                                                                                                  \
    return _mm512_reduce_add_epi32(                                                                                    \
        _mm512_add_epi32(_mm512_add_epi32(acc[0], acc[1]), _mm512_add_epi32(acc[2], acc[3])));                         \
  }
#define DEFINE_RD_S(S)                                                                                                 \
  DEFINE_RD(S, 0, _MM_HINT_T0, 0)                                                                                      \
  DEFINE_RD(S, 0, _MM_HINT_T0, 1) DEFINE_RD(S, 1, _MM_HINT_T1, 0) DEFINE_RD(S, 1, _MM_HINT_T1, 1)                      \
      DEFINE_RD(S, 2, _MM_HINT_NTA, 0) DEFINE_RD(S, 2, _MM_HINT_NTA, 1)
DEFINE_RD_S(1)
DEFINE_RD_S(2)
DEFINE_RD_S(4)
DEFINE_RD_S(8)
#define RD_ROW(S)                                                                                                      \
  {                                                                                                                    \
    {rd_##S##_0_0, rd_##S##_0_1}, {rd_##S##_1_0, rd_##S##_1_1}, {                                                      \
      rd_##S##_2_0, rd_##S##_2_1                                                                                       \
    }                                                                                                                  \
  }
static const RdFn rd_table[4][3][2] = {RD_ROW(1), RD_ROW(2), RD_ROW(4), RD_ROW(8)};

/* Two interleaved streams per core: a lines of the HBM region, then b lines of the DDR region, `steps` times. */
AVX512 static long rd_mix(const char *ph, const char *pd, size_t a, size_t b, size_t steps, size_t pf) {
  __m512i x = _mm512_setzero_si512(), y = x;
  for (size_t s = 0; s < steps; s++) {
    for (size_t i = 0; i < a; i++) {
      const char *q = ph + (s * a + i) * 64;
      if (pf)
        _mm_prefetch(q + pf, _MM_HINT_T0);
      x = _mm512_add_epi32(x, _mm512_loadu_si512((const void *)q));
    }
    for (size_t i = 0; i < b; i++) {
      const char *q = pd + (s * b + i) * 64;
      if (pf)
        _mm_prefetch(q + pf, _MM_HINT_T0);
      y = _mm512_add_epi32(y, _mm512_loadu_si512((const void *)q));
    }
  }
  return _mm512_reduce_add_epi32(_mm512_add_epi32(x, y));
}
#else
static long rd_scalar(const char *p, size_t lines, size_t pf) {
  (void)pf;
  long s = 0;
  for (size_t i = 0; i < lines; i++)
    s += p[i * 64];
  return s;
}
static const RdFn rd_table[4][3][2] = {{{rd_scalar, rd_scalar}, {rd_scalar, rd_scalar}, {rd_scalar, rd_scalar}},
                                       {{rd_scalar, rd_scalar}, {rd_scalar, rd_scalar}, {rd_scalar, rd_scalar}},
                                       {{rd_scalar, rd_scalar}, {rd_scalar, rd_scalar}, {rd_scalar, rd_scalar}},
                                       {{rd_scalar, rd_scalar}, {rd_scalar, rd_scalar}, {rd_scalar, rd_scalar}}};
static long rd_mix(const char *ph, const char *pd, size_t a, size_t b, size_t steps, size_t pf) {
  (void)pf;
  long s = 0;
  for (size_t i = 0; i < steps; i++)
    s += ph[i * a * 64] + pd[i * b * 64];
  return s;
}
#endif

static int stream_index(int s) {
  return s == 1 ? 0 : s == 2 ? 1 : s == 4 ? 2 : 3;
}

static void wait_until(double t) {
  while (monotonic_seconds() < t)
    ;
}

/* ---- accounting shared by the raw workers ------------------------------------------------------------------------ */
#define ACCOUNT(bh_, bd_)                                                                                              \
  do {                                                                                                                 \
    double now_ = monotonic_seconds();                                                                                 \
    r->bytes_total[0] += (bh_);                                                                                        \
    r->bytes_total[1] += (bd_);                                                                                        \
    if (counting) {                                                                                                    \
      r->bytes_meas[0] += (bh_);                                                                                       \
      r->bytes_meas[1] += (bd_);                                                                                       \
    } else if (now_ >= meas0) {                                                                                        \
      counting = 1;                                                                                                    \
      r->t_begin = now_;                                                                                               \
    }                                                                                                                  \
    if (now_ >= stop) {                                                                                                \
      r->t_end = now_;                                                                                                 \
      done = 1;                                                                                                        \
    }                                                                                                                  \
  } while (0)

/* The role of worker w in a split window: tier 0 (HBM), 1 (DDR) or -1 (idle), with its index in and the size of its
 * group. */
static int split_role(const Cfg *c, int w, int nstream, int *gi, int *gn) {
  int active = c->ch + c->cd;
  if (w >= nstream || w >= active)
    return -1;
  if (!c->alt) {
    if (w < c->ch) {
      *gi = w;
      *gn = c->ch;
      return 0;
    }
    *gi = w - c->ch;
    *gn = c->cd;
    return 1;
  }
  int is_d = ((w + 1) * c->cd / active) != (w * c->cd / active), idx = 0;
  for (int i = 0; i < w; i++)
    idx += (((i + 1) * c->cd / active) != (i * c->cd / active)) == is_d;
  *gi = idx;
  *gn = is_d ? c->cd : c->ch;
  return is_d;
}

static void slice_of(char *base, size_t bytes, int gi, int gn, char **p, size_t *len) {
  size_t blocks = bytes / MIB, b0 = blocks * (size_t)gi / (size_t)gn, b1 = blocks * (size_t)(gi + 1) / (size_t)gn;
  if (b1 == b0)
    b1 = b0 + 1;
  *p = base + b0 * MIB;
  *len = (b1 - b0) * MIB;
}

/* A dependent pointer chase on the last worker, in the HBM or the DDR chase region, while the others stream. */
static void probe_worker(const WinJob *j, WRes *r) {
  const Cfg *c = j->cfg;
  Sock *s = j->sock;
  const char *base = s->P[c->probe - 1].data;
  uint64_t idx = 0;
  double meas0 = j->t_go + c->warm, stop = meas0 + c->seconds;
  wait_until(j->t_go);
  int counting = 0;
  for (;;) {
    for (int k = 0; k < 2048; k++)
      idx = *(volatile const uint64_t *)(base + idx * 64);
    double now = monotonic_seconds();
    if (counting)
      r->probe_loads += 2048;
    else if (now >= meas0) {
      counting = 1;
      r->t_begin = now;
    }
    if (now >= stop) {
      r->t_end = now;
      break;
    }
  }
  r->probe_ns = r->probe_loads > 0 ? (r->t_end - r->t_begin) / r->probe_loads * 1e9 : 0;
  r->sink = (long)idx;
}

static void raw_worker(const void *p, int w, int workers) {
  const WinJob *j = p;
  const Cfg *c = j->cfg;
  Sock *s = j->sock;
  WRes *r = &s->res[w];
  memset(r, 0, sizeof(*r));
  if (c->probe && w == workers - 1) {
    probe_worker(j, r);
    return;
  }
  int nstream = workers - (c->probe ? 1 : 0), counting = 0, done = 0;
  double meas0 = j->t_go + c->warm, stop = meas0 + c->seconds;
  const RdFn rd_h = rd_table[stream_index(c->streams)][c->hint][c->nt];
  const RdFn rd_d = rd_table[stream_index(c->dstreams >= 0 ? c->dstreams : c->streams)]
                            [c->dhint >= 0 ? c->dhint : c->hint][c->dnt >= 0 ? c->dnt : c->nt];
  const size_t blk = MIB / 64, pf_h = (size_t)c->pf, pf_d = (size_t)(c->dpf >= 0 ? c->dpf : c->pf);
  RdFn rd = rd_h;
  size_t pf = pf_h;
  long sink = 0;
  if (c->kind == CFG_SPLIT || c->kind == CFG_INTER) {
    int gi = w, gn = nstream, tier = 0;
    char *base = c->kind == CFG_INTER ? s->xbase : NULL;
    size_t bytes = c->kind == CFG_INTER ? s->xbytes : s->bytes;
    if (c->kind == CFG_SPLIT) {
      tier = split_role(c, w, nstream, &gi, &gn);
      if (tier < 0)
        return;
      base = s->base[tier];
      rd = tier ? rd_d : rd_h;
      pf = tier ? pf_d : pf_h;
    }
    char *mine;
    size_t len;
    slice_of(base, bytes, gi, gn, &mine, &len);
    size_t blocks = len / MIB, bi = 0;
    wait_until(j->t_go);
    if (c->kind == CFG_SPLIT && c->volume) {
      /* Fixed volume per worker; progress is logged every LOG_BYTES so a common interval can be cut out afterwards. */
      double need = (double)c->volume * MIB, total = 0, next_log = LOG_BYTES;
      r->t_begin = monotonic_seconds();
      while (total < need) {
        sink ^= rd(mine + bi * MIB, blk, pf);
        bi = bi + 1 == blocks ? 0 : bi + 1;
        total += MIB;
        if (total >= next_log && r->nlog < LOG_MAX) {
          r->lt[r->nlog++] = monotonic_seconds();
          next_log += LOG_BYTES;
        }
      }
      r->t_end = monotonic_seconds();
      r->bytes_total[tier] = r->bytes_meas[tier] = total;
    } else {
      for (;;) {
        sink ^= rd(mine + bi * MIB, blk, pf);
        bi = bi + 1 == blocks ? 0 : bi + 1;
        ACCOUNT(c->kind == CFG_INTER ? MIB : (tier == 0 ? MIB : 0), c->kind == CFG_INTER ? 0 : (tier == 1 ? MIB : 0));
        if (done)
          break;
      }
    }
  } else { /* CFG_SHARED */
    char *hp, *dp;
    size_t hl, dl;
    slice_of(s->base[0], s->bytes, w, nstream, &hp, &hl);
    slice_of(s->base[1], s->bytes, w, nstream, &dp, &dl);
    size_t ua = (size_t)c->a * (size_t)c->unit, ub = (size_t)c->b * (size_t)c->unit;
    size_t steps = blk / (ua + ub ? ua + ub : 1);
    if (steps == 0)
      steps = 1;
    size_t ho = 0, dof = 0;
    wait_until(j->t_go);
    for (;;) {
      size_t bh = 0, bd = 0;
      if (c->unit == 1) {
        if (ho + steps * ua * 64 > hl)
          ho = 0;
        if (dof + steps * ub * 64 > dl)
          dof = 0;
        sink ^= rd_mix(hp + ho, dp + dof, ua, ub, steps, pf_h);
        bh = steps * ua * 64;
        bd = steps * ub * 64;
        ho += bh;
        dof += bd;
      } else {
        for (size_t k = 0; k < steps; k++) {
          if (ua) {
            if (ho + ua * 64 > hl)
              ho = 0;
            sink ^= rd_h(hp + ho, ua, pf_h);
            ho += ua * 64;
            bh += ua * 64;
          }
          if (ub) {
            if (dof + ub * 64 > dl)
              dof = 0;
            sink ^= rd_d(dp + dof, ub, pf_d);
            dof += ub * 64;
            bd += ub * 64;
          }
        }
      }
      ACCOUNT((double)bh, (double)bd);
      if (done)
        break;
    }
  }
  r->sink = sink;
}

/* ---- the 70B per-socket shard for kind=gemv ---------------------------------------------------------------------- */
typedef struct {
  int d, n;
} Mat;
static const Mat shard[7] = {{4096, 8192},  {512, 8192},   {512, 8192},  {8192, 4096},
                             {14336, 8192}, {14336, 8192}, {8192, 14336}};
static size_t mat_q_off[7], mat_s_off[7], layer_stride;

static size_t align_up(size_t x, size_t a) {
  return (x + a - 1) / a * a;
}

static void layout_shard(void) {
  size_t cursor = 0;
  for (int m = 0; m < 7; m++) {
    mat_q_off[m] = cursor;
    cursor += align_up((size_t)shard[m].d * shard[m].n, 4096);
    mat_s_off[m] = cursor; /* sized for gs = 32; gs = 64 uses the first half of the same rows */
    cursor += align_up((size_t)shard[m].d * (shard[m].n / 32) * 4, 4096);
  }
  layer_stride = cursor;
}

static int gemv_rows_d(const Cfg *c, int m) {
  int rows = (int)(c->r * shard[m].d + 0.5);
  return rows > shard[m].d ? shard[m].d : rows;
}

/* Bytes of one pass per tier: the rows each tier serves (int8 values plus FP32 group scales). */
static void gemv_bytes(const Cfg *c, int layers, double *bh, double *bd) {
  *bh = *bd = 0;
  for (int m = 0; m < 7; m++) {
    double row = shard[m].n + 4.0 * shard[m].n / c->gs;
    int rd_ = gemv_rows_d(c, m);
    *bd += rd_ * row * layers;
    *bh += (shard[m].d - rd_) * row * layers;
  }
}

static int gemv_layers(const Sock *s) {
  size_t fit = s->bytes / layer_stride;
  return fit > GEMV_LAYERS ? GEMV_LAYERS : (int)fit;
}

static void gemv_part(const Cfg *c, Sock *s, int m, int layer, int tier, int gi, int gn) {
  int rows_d = gemv_rows_d(c, m), rows_h = shard[m].d - rows_d;
  int rows = tier ? rows_d : rows_h, first = tier ? rows_h : 0;
  if (rows <= 0 || gn <= 0)
    return;
  size_t n = (size_t)shard[m].n, lay = (size_t)layer * layer_stride;
  K8Weight wt = {(const int8_t *)(s->base[tier] + lay + mat_q_off[m]) + (size_t)first * n,
                 (const float *)(s->base[tier] + lay + mat_s_off[m]) + (size_t)first * (n / (size_t)c->gs)};
  K8Input x = {s->xq, s->xs, NULL};
  k_matmul_q8_ld_worker(s->out[m] + first, &x, &wt, n, (int)n, rows, c->gs, K_STORE, gi, gn, NULL, NULL, NULL);
}

static void gemv_worker(const void *p, int w, int workers) {
  const WinJob *j = p;
  const Cfg *c = j->cfg;
  Sock *s = j->sock;
  Team *team = s->team;
  int active = c->act ? c->act : workers, layers = gemv_layers(s), cdw = c->gshared ? 0 : c->cdw, ch = active - cdw;
  int tier = w < ch ? 0 : 1, gi = tier ? w - ch : w, gn = tier ? cdw : ch;
  if (w >= active)
    gn = 0; /* idle member: only the barriers */
  static const int phases[4][2] = {{0, 3}, {3, 4}, {4, 6}, {6, 7}}; /* qkv | wo | w1 w3 | w2, a barrier after each */
  wait_until(j->t_go);
  for (int pass = 0; pass < c->warm_passes + c->passes; pass++) {
    for (int layer = 0; layer < layers; layer++)
      for (int ph = 0; ph < 4; ph++) {
        for (int m = phases[ph][0]; m < phases[ph][1]; m++) {
          if (c->gshared) {
            int first = w & 1; /* odd workers start on DDR, even ones on HBM */
            gemv_part(c, s, m, layer, first, w, w < active ? active : 0);
            gemv_part(c, s, m, layer, !first, w, w < active ? active : 0);
          } else {
            gemv_part(c, s, m, layer, tier, gi, gn);
          }
        }
        team_barrier(team);
      }
    if (w == 0)
      s->pass_t[pass] = monotonic_seconds();
  }
}

/* ---- setup ------------------------------------------------------------------------------------------------------- */
typedef struct {
  char *base;
  size_t bytes;
  int gemv_layers; /* 0 = raw pattern, else the shard layout over this many layers */
} FillJob2;

static uint64_t mix64(uint64_t x) {
  x ^= x >> 33;
  x *= 0xff51afd7ed558ccdULL;
  x ^= x >> 33;
  x *= 0xc4ceb9fe1a85ec53ULL;
  return x ^ (x >> 33);
}

static void fill2_fn(const void *p, int w, int workers) {
  const FillJob2 *job = p;
  if (!job->gemv_layers) {
    size_t words = job->bytes / 8, i0 = words * (size_t)w / (size_t)workers,
           i1 = words * (size_t)(w + 1) / (size_t)workers;
    uint64_t *q = (uint64_t *)job->base;
    for (size_t i = i0; i < i1; i++)
      q[i] = mix64(i + 1) & 0x7f7f7f7f7f7f7f7fULL;
    return;
  }
  for (int layer = 0; layer < job->gemv_layers; layer++)
    for (int m = 0; m < 7; m++) {
      int rows = shard[m].d, n = shard[m].n, r0 = (int)((long long)rows * w / workers),
          r1 = (int)((long long)rows * (w + 1) / workers);
      int8_t *q = (int8_t *)(job->base + (size_t)layer * layer_stride + mat_q_off[m]);
      float *sc = (float *)(job->base + (size_t)layer * layer_stride + mat_s_off[m]);
      for (int r = r0; r < r1; r++) {
        uint64_t h = mix64((uint64_t)(layer * 7 + m) << 32 | (uint64_t)r);
        for (int i = 0; i < n; i++) {
          h = h * 6364136223846793005ULL + 1442695040888963407ULL;
          int v = (int)((h >> 33) % 255) - 127;
          q[(size_t)r * n + i] = (int8_t)v;
        }
        for (int g = 0; g < n / 32; g++)
          sc[(size_t)r * (n / 32) + g] = 0.002f + 0.0001f * (float)((r + g) % 7);
      }
    }
  /* The rest of the region (past the last whole layer) is touched too, so every page of it is populated and verified. */
  size_t tail = (size_t)job->gemv_layers * layer_stride, words = (job->bytes - tail) / 8,
         i0 = words * (size_t)w / (size_t)workers, i1 = words * (size_t)(w + 1) / (size_t)workers;
  uint64_t *q = (uint64_t *)(job->base + tail);
  for (size_t i = i0; i < i1; i++)
    q[i] = mix64(i + 1) & 0x7f7f7f7f7f7f7f7fULL;
}

typedef struct {
  char *base;
  size_t pages, page;
  int a, b, hbm_node, ddr_node, real;
} InterJob;

#ifdef __linux__
static void bind_thread_to(int node, int max_node) {
  unsigned long mask[16] = {0};
  mask[node / (int)(8 * sizeof(unsigned long))] |= 1UL << (node % (int)(8 * sizeof(unsigned long)));
  syscall(SYS_set_mempolicy, 2 /* MPOL_BIND */, mask, (unsigned long)max_node + 2);
}
#endif

__attribute__((unused)) static int inter_node(const InterJob *j, size_t page) {
  size_t period = (size_t)(j->a + j->b), k = page % period;
  return k < (size_t)j->a ? j->hbm_node : j->ddr_node;
}

static void inter_fn(const void *p, int w, int workers) {
  const InterJob *j = p;
  size_t p0 = j->pages * (size_t)w / (size_t)workers, p1 = j->pages * (size_t)(w + 1) / (size_t)workers;
  int last = -2;
  (void)last;
  for (size_t pg = p0; pg < p1; pg++) {
#ifdef __linux__
    if (j->real) {
      int node = inter_node(j, pg);
      if (node != last) {
        bind_thread_to(node, numa_topology()->max_node);
        last = node;
      }
    }
#endif
    *(volatile char *)(j->base + pg * j->page) = 1;
  }
#ifdef __linux__
  if (j->real)
    syscall(SYS_set_mempolicy, 0 /* MPOL_DEFAULT */, NULL, 0UL);
#endif
}

static void inter_release(Sock *s) {
#ifdef __linux__
  if (s->xmap)
    munmap(s->xmap, s->xmap_bytes);
#endif
  s->xmap = NULL;
  s->xbase = NULL;
}

static void json_placement(const char *stage, int socket, const char *tier, int node, size_t bytes, int page4k,
                           const PageCheck *chk, size_t thp, const char *scheme) {
  printf(
      "{\"event\": \"placement\", \"stage\": \"%s\", \"socket\": %d, \"tier\": \"%s\", \"node\": %d, \"bytes\": %zu, "
      "\"page\": \"%s\", \"queried\": %zu, \"on_node\": %zu, \"wrong_node\": %zu, \"missing\": %zu, \"errors\": %zu, "
      "\"scheme\": \"%s\", \"thp_bytes\": %zu}\n",
      stage, socket, tier, node, bytes, page4k ? "4k" : "2m", chk->queried, chk->on_node, chk->wrong_node, chk->missing,
      chk->errors, scheme, thp);
  fflush(stdout);
}

/* Verify one interleaved region page by page against the pattern it was built with (move_pages, no migration). */
#ifdef __linux__
static void inter_query(void **addr, const int *expect, size_t n, PageCheck *chk) {
  int status[2048];
  if (syscall(SYS_move_pages, 0, (unsigned long)n, addr, NULL, status, 0)) {
    chk->errors += n;
    return;
  }
  for (size_t i = 0; i < n; i++) {
    chk->queried++;
    if (status[i] == expect[i])
      chk->on_node++;
    else if (status[i] < 0)
      chk->missing++;
    else
      chk->wrong_node++;
  }
}
#endif

static void inter_verify(Run *R, Sock *s, const Cfg *c, size_t page) {
  PageCheck chk = {0};
  size_t thp = 0;
  (void)c;
  (void)page;
#ifdef __linux__
  if (R->real) {
    InterJob job = {.a = c->a, .b = c->b, .hbm_node = s->hbm_node, .ddr_node = s->ddr_node};
    size_t pages = s->xbytes / page, n = 0;
    void *addr[2048];
    int expect[2048];
    for (size_t pg = 0; pg < pages; pg += (size_t)R->verify_stride) {
      addr[n] = s->xbase + pg * page;
      expect[n++] = inter_node(&job, pg);
      if (n == 2048) {
        inter_query(addr, expect, n, &chk);
        n = 0;
      }
    }
    if (n)
      inter_query(addr, expect, n, &chk);
    NumaRegion fake = {.data = s->xbase, .data_bytes = s->xbytes};
    region_thp_bytes(&fake, &thp);
  }
#endif
  json_placement("window", s->socket, "inter", -1, s->xbytes, R->page4k, &chk, thp,
                 R->verify_stride == 1 ? "full" : "stride");
  if (R->real && (chk.on_node != chk.queried || chk.errors))
    fail("interleaved region placement verification failed: %zu of %zu pages on the intended node", chk.on_node,
         chk.queried);
}

static void inter_prepare(Run *R, const Cfg *c) {
  for (int so = 0; so < R->sockets; so++) {
    Sock *s = &R->sock[so];
    size_t page = R->page4k ? 4096 : (size_t)2 * MIB, bytes = R->inter_mib * MIB;
    s->xbytes = bytes;
    s->xmap_bytes = bytes + 2 * MIB;
#ifdef __linux__
    void *m = mmap(NULL, s->xmap_bytes, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
    if (m == MAP_FAILED)
      fail("cannot map the interleaved region: %s", strerror(errno));
    s->xmap = m;
    s->xbase = (char *)(((uintptr_t)m + 2 * MIB - 1) / (2 * MIB) * (2 * MIB));
    madvise(s->xbase, bytes, R->page4k ? MADV_NOHUGEPAGE : MADV_HUGEPAGE);
#else
    s->xmap = s->xbase = malloc(s->xmap_bytes);
#endif
    InterJob job = {.base = s->xbase,
                    .pages = bytes / page,
                    .page = page,
                    .a = c->a,
                    .b = c->b,
                    .hbm_node = s->hbm_node,
                    .ddr_node = s->ddr_node,
                    .real = R->real};
    team_run(s->team, inter_fn, &job);
  }
  for (int so = 0; so < R->sockets; so++)
    inter_verify(R, &R->sock[so], c, R->page4k ? 4096 : (size_t)2 * MIB);
}

static void build_chase(Sock *s, int tier) {
  size_t lines = s->pbytes / 64;
  uint32_t *perm = malloc(lines * sizeof(uint32_t));
  if (!perm)
    fail("cannot allocate the chase permutation");
  for (size_t i = 0; i < lines; i++)
    perm[i] = (uint32_t)i;
  uint64_t rng = 0x9E3779B97F4A7C15ULL ^ (uint64_t)(tier + 1);
  for (size_t i = lines - 1; i > 0; i--) { /* Sattolo: one cycle through every line */
    rng = mix64(rng + i);
    size_t k = (size_t)(rng % i);
    uint32_t t = perm[i];
    perm[i] = perm[k];
    perm[k] = t;
  }
  char *base = s->P[tier].data;
  for (size_t i = 0; i < lines; i++)
    *(uint64_t *)(base + (size_t)perm[i] * 64) = perm[(i + 1) % lines];
  free(perm);
}

/* ---- the run loop ------------------------------------------------------------------------------------------------ */
static void *run_side(void *arg) {
  Run *R = arg;
  Sock *s = &R->sock[1];
  kernels_set_thread_team(s->team);
  team_adopt_leader(s->team);
  unsigned seen = 0;
  for (;;) {
    unsigned now;
    while ((now = atomic_load_explicit(&R->go_gen, memory_order_acquire)) == seen)
      sched_yield();
    seen = now;
    if (R->stop)
      return NULL;
    WinJob job = {R->cfg, s, R->t_go};
    team_run(s->team, R->cfg->kind == CFG_GEMV ? gemv_worker : raw_worker, &job);
    atomic_store_explicit(&R->done_gen, now, memory_order_release);
  }
}

static double realtime_now(void) {
  struct timespec ts;
  clock_gettime(CLOCK_REALTIME, &ts);
  return (double)ts.tv_sec + (double)ts.tv_nsec * 1e-9;
}

/* Cumulative bytes of a fixed-volume worker at time t, interpolated between its progress log entries. */
static double cum_at(const WRes *r, double t, double t0) {
  double total = r->bytes_total[0] + r->bytes_total[1];
  if (t >= r->t_end)
    return total;
  double prev_t = t0, prev_b = 0;
  for (int k = 0; k <= r->nlog; k++) {
    double kt = k < r->nlog ? r->lt[k] : r->t_end, kb = k < r->nlog ? (double)(k + 1) * LOG_BYTES : total;
    if (t <= kt)
      return prev_b + (kb - prev_b) * (t - prev_t) / (kt > prev_t ? kt - prev_t : 1e-9);
    prev_t = kt;
    prev_b = kb;
  }
  return total;
}

static void emit_window(Run *R, const Cfg *c, int rep, int index, double real_begin, double real_end, double t_go) {
  for (int so = 0; so < R->sockets; so++) {
    Sock *s = &R->sock[so];
    int workers = R->threads;
    double g[2] = {0, 0}, tot[2] = {0, 0}, dur = 0, probe = 0;
    int nrole[2] = {0, 0}, nd = 0;
    double vol_total = 0, vol_time = 0, tc = 1e30, cg[2] = {0, 0}, t_end_max = 0, t_beg_min = 1e30;
    if (c->kind == CFG_GEMV) {
      int layers = gemv_layers(s);
      double bh, bd;
      gemv_bytes(c, layers, &bh, &bd);
      double t0 = s->pass_t[c->warm_passes - 1], t1 = s->pass_t[c->warm_passes + c->passes - 1], best = 1e30;
      for (int k = c->warm_passes; k < c->warm_passes + c->passes; k++) {
        double d = s->pass_t[k] - s->pass_t[k - 1];
        best = d < best ? d : best;
      }
      double per = (t1 - t0) / c->passes;
      g[0] = bh / per / 1e9;
      g[1] = bd / per / 1e9;
      tot[0] = bh * (c->warm_passes + c->passes);
      tot[1] = bd * (c->warm_passes + c->passes);
      dur = per;
      vol_total = best; /* the fastest single pass, seconds */
      nrole[0] = c->gshared ? workers : workers - c->cdw;
      nrole[1] = c->gshared ? workers : c->cdw;
      t_beg_min = t0;
      t_end_max = t1;
    } else {
      for (int w = 0; w < workers; w++) {
        const WRes *r = &s->res[w];
        if (c->probe && w == workers - 1) {
          probe = r->probe_ns;
          continue;
        }
        if (r->t_end <= r->t_begin)
          continue;
        double d = r->t_end - r->t_begin;
        for (int t = 0; t < 2; t++) {
          if (r->bytes_meas[t] > 0) {
            g[t] += r->bytes_meas[t] / d / 1e9;
            nrole[t]++;
          }
          tot[t] += r->bytes_total[t];
        }
        dur += d;
        nd++;
        t_end_max = r->t_end > t_end_max ? r->t_end : t_end_max;
        t_beg_min = r->t_begin < t_beg_min ? r->t_begin : t_beg_min;
        tc = r->t_end < tc ? r->t_end : tc;
      }
      dur = nd ? dur / nd : 0;
      if (c->volume && nd) {
        double all = tot[0] + tot[1];
        vol_time = t_end_max - t_go;
        vol_total = all / vol_time / 1e9;
        for (int w = 0; w < workers; w++) {
          const WRes *r = &s->res[w];
          if (r->t_end <= r->t_begin)
            continue;
          int t = r->bytes_total[1] > 0;
          cg[t] += cum_at(r, tc, t_go);
        }
        cg[0] /= (tc - t_go) * 1e9;
        cg[1] /= (tc - t_go) * 1e9;
      }
    }
    printf(
        "{\"experiment\": \"%s\", \"name\": \"%s\", \"kind\": %d, \"socket\": %d, \"sockets\": %d, \"page\": \"%s\", "
        "\"rep\": %d, \"window\": %d, \"threads\": %d, \"ch\": %d, \"cd\": %d, \"a\": %d, \"b\": %d, \"unit\": %d, "
        "\"streams\": %d, \"pf\": %d, \"hint\": %d, \"nt\": %d, \"dstreams\": %d, \"dpf\": %d, \"dhint\": %d, \"dnt\": "
        "%d, \"alt\": %d, \"probe\": %d, \"volume\": %d, \"gs\": %d, "
        "\"r\": %.4f, \"cdw\": %d, \"act\": %d, \"gshared\": %d, \"warm\": %.3f, \"seconds\": %.3f, \"hbm_gbps\": %.3f, "
        "\"ddr_gbps\": %.3f, \"combined_gbps\": %.3f, \"hbm_bytes\": %.0f, \"ddr_bytes\": %.0f, \"workers_hbm\": %d, "
        "\"workers_ddr\": %d, \"mean_worker_s\": %.6f, \"probe_ns\": %.1f, \"volume_gbps\": %.3f, \"volume_s\": %.6f, "
        "\"common_hbm_gbps\": %.3f, \"common_ddr_gbps\": %.3f, \"common_s\": %.6f, \"gemv_pass_s\": %.6f, "
        "\"gemv_best_pass_s\": %.6f, \"layers\": %d, \"real_begin\": %.6f, \"real_end\": %.6f}\n",
        "concurrent-tier", c->name, c->kind, so, R->sockets, R->page4k ? "4k" : "2m", rep, index, workers, c->ch, c->cd,
        c->a, c->b, c->unit, c->streams, c->pf, c->hint, c->nt, c->dstreams, c->dpf, c->dhint, c->dnt, c->alt, c->probe,
        c->volume, c->gs, c->r, c->cdw, c->act, c->gshared, c->warm, c->seconds, g[0], g[1], g[0] + g[1], tot[0], tot[1],
        nrole[0], nrole[1], dur, probe, c->kind == CFG_GEMV ? 0.0 : vol_total, c->kind == CFG_GEMV ? 0.0 : vol_time,
        cg[0], cg[1], c->volume ? tc - t_go : 0.0, c->kind == CFG_GEMV ? dur : 0.0,
        c->kind == CFG_GEMV ? vol_total : 0.0, c->kind == CFG_GEMV ? gemv_layers(s) : 0, real_begin, real_end);
  }
  fflush(stdout);
}

static void run_window(Run *R, const Cfg *c, int rep, int index) {
  if (c->kind == CFG_INTER)
    inter_prepare(R, c);
  R->cfg = c;
  double lead = 0.08;
  R->t_go = monotonic_seconds() + lead;
  double real_begin = realtime_now() + lead;
  unsigned gen = atomic_load_explicit(&R->go_gen, memory_order_relaxed) + 1;
  atomic_store_explicit(&R->go_gen, gen, memory_order_release);
  WinJob job = {c, &R->sock[0], R->t_go};
  team_run(R->sock[0].team, c->kind == CFG_GEMV ? gemv_worker : raw_worker, &job);
  if (R->sockets == 2)
    while (atomic_load_explicit(&R->done_gen, memory_order_acquire) != gen)
      sched_yield();
  double real_end = realtime_now();
  emit_window(R, c, rep, index, real_begin, real_end, R->t_go);
  if (c->kind == CFG_INTER)
    for (int so = 0; so < R->sockets; so++)
      inter_release(&R->sock[so]);
}

/* ---- options ----------------------------------------------------------------------------------------------------- */
static int parse_cfg(const char *text, Cfg *c) {
  *c = (Cfg){.kind = CFG_SPLIT,
             .streams = 1,
             .pf = 4096,
             .unit = 128,
             .gs = 64,
             .warm = 0.2,
             .seconds = 1.0,
             .warm_passes = 2,
             .passes = 16,
             .volume = 0,
             .dstreams = -1,
             .dhint = -1,
             .dnt = -1,
             .dpf = -1};
  snprintf(c->name, sizeof(c->name), "%s", text);
  char buf[512];
  snprintf(buf, sizeof(buf), "%s", text);
  for (char *save = NULL, *tok = strtok_r(buf, ",", &save); tok; tok = strtok_r(NULL, ",", &save)) {
    char *eq = strchr(tok, '=');
    if (!eq)
      return -1;
    *eq = 0;
    const char *k = tok, *v = eq + 1;
    if (!strcmp(k, "name"))
      snprintf(c->name, sizeof(c->name), "%s", v);
    else if (!strcmp(k, "kind"))
      c->kind = !strcmp(v, "split")    ? CFG_SPLIT
                : !strcmp(v, "shared") ? CFG_SHARED
                : !strcmp(v, "inter")  ? CFG_INTER
                                       : CFG_GEMV;
    else if (!strcmp(k, "ch"))
      c->ch = atoi(v);
    else if (!strcmp(k, "cd"))
      c->cd = atoi(v);
    else if (!strcmp(k, "a"))
      c->a = atoi(v);
    else if (!strcmp(k, "b"))
      c->b = atoi(v);
    else if (!strcmp(k, "unit"))
      c->unit = atoi(v);
    else if (!strcmp(k, "streams") || !strcmp(k, "S"))
      c->streams = atoi(v);
    else if (!strcmp(k, "pf"))
      c->pf = atoi(v);
    else if (!strcmp(k, "dS"))
      c->dstreams = atoi(v);
    else if (!strcmp(k, "dpf"))
      c->dpf = atoi(v);
    else if (!strcmp(k, "dhint"))
      c->dhint = atoi(v);
    else if (!strcmp(k, "dnt"))
      c->dnt = atoi(v);
    else if (!strcmp(k, "hint"))
      c->hint = atoi(v);
    else if (!strcmp(k, "nt"))
      c->nt = atoi(v);
    else if (!strcmp(k, "alt"))
      c->alt = atoi(v);
    else if (!strcmp(k, "probe"))
      c->probe = atoi(v);
    else if (!strcmp(k, "volume"))
      c->volume = atoi(v);
    else if (!strcmp(k, "gs"))
      c->gs = atoi(v);
    else if (!strcmp(k, "r"))
      c->r = atof(v);
    else if (!strcmp(k, "cdw"))
      c->cdw = atoi(v);
    else if (!strcmp(k, "act"))
      c->act = atoi(v);
    else if (!strcmp(k, "gshared"))
      c->gshared = atoi(v);
    else if (!strcmp(k, "warm"))
      c->warm = atof(v);
    else if (!strcmp(k, "seconds"))
      c->seconds = atof(v);
    else if (!strcmp(k, "warm_passes"))
      c->warm_passes = atoi(v);
    else if (!strcmp(k, "passes"))
      c->passes = atoi(v);
    else
      return -1;
  }
  int dstreams = c->dstreams >= 0 ? c->dstreams : c->streams;
  int streams_ok = (c->streams == 1 || c->streams == 2 || c->streams == 4 || c->streams == 8) &&
                   (dstreams == 1 || dstreams == 2 || dstreams == 4 || dstreams == 8);
  if (!streams_ok || c->dhint > 2 || c->dnt > 1 || c->hint < 0 || c->hint > 2 || c->nt < 0 || c->nt > 1 || c->pf < 0 ||
      c->gs % 32 || c->gs > 64 || c->warm_passes < 1 || c->passes < 1 || c->warm_passes + c->passes > 255 || c->r < 0 ||
      c->r > 1 || (c->kind == CFG_SHARED && c->unit != 1 && (c->unit < 8 || c->unit % 8)) ||
      (c->probe < 0 || c->probe > 2))
    return -1;
  if (c->kind == CFG_SHARED || c->kind == CFG_INTER) {
    if (c->a < 0 || c->b < 0 || c->a + c->b == 0)
      return -1;
  }
  return 0;
}

static void check_cfg(const Run *R, const Cfg *c) {
  int stream = R->threads - (c->probe ? 1 : 0);
  if (c->kind == CFG_SPLIT && (c->ch < 0 || c->cd < 0 || c->ch + c->cd < 1 || c->ch + c->cd > stream))
    fail("config %s: ch + cd must be 1..%d", c->name, stream);
  if (c->kind == CFG_GEMV &&
      (c->cdw < 0 || c->cdw >= R->threads || (c->cdw == 0) != (c->r == 0 || c->gshared) || c->probe ||
                                 c->act > R->threads || (c->act && c->cdw >= c->act)))
    fail("config %s: cdw must be 0 exactly when r = 0 (or gshared)", c->name);
}

static pid_t perf_start(const char *cmd, const char *out) {
#ifdef __linux__
  pid_t pid = fork();
  if (pid < 0)
    fail("fork failed: %s", strerror(errno));
  if (pid == 0) {
    execl("/bin/sh", "sh", "-c", cmd, (char *)NULL);
    _exit(127);
  }
  for (int i = 0; i < 400; i++) {
    struct timespec ts = {0, 50 * 1000 * 1000};
    nanosleep(&ts, NULL);
    FILE *f = fopen(out, "r");
    if (f) {
      int ch = fgetc(f);
      fclose(f);
      if (ch != EOF)
        return pid;
    }
  }
  fail("perf did not start counting within 20 s");
#else
  (void)cmd;
  (void)out;
  return -1;
#endif
}

static void sleep_s(double s) {
  struct timespec ts = {(time_t)s, (long)((s - (double)(time_t)s) * 1e9)};
  nanosleep(&ts, NULL);
}

static void verify_tier(Run *R, Sock *s, int tier, const char *stage, size_t samples) {
  PageCheck chk = {0};
  size_t thp = 0;
  int ok = s->R[tier].node >= 0 && region_verify(&s->R[tier], samples, &chk) == 0;
  if (R->real) {
    if (!ok || chk.on_node != chk.queried)
      fail("socket %d %s placement verification failed (%s): %zu of %zu pages on node %d (%zu on another node, %zu "
           "missing, %zu query errors)",
           s->socket, tier ? "DDR" : "HBM", stage, chk.on_node, chk.queried, s->R[tier].node, chk.wrong_node,
           chk.missing, chk.errors);
    region_thp_bytes(&s->R[tier], &thp);
    if (!R->page4k && thp < s->R[tier].data_bytes * 95 / 100)
      LOG(LOG_WARN, "socket %d %s: only %zu of %zu bytes are in transparent huge pages", s->socket,
          tier ? "DDR" : "HBM", thp, s->R[tier].data_bytes);
    json_placement(stage, s->socket, tier ? "ddr" : "hbm", s->R[tier].node, s->R[tier].data_bytes, R->page4k, &chk, thp,
                   chk.scheme ? chk.scheme : "full");
  }
}

static int research_main(int argc, char **argv) {
  int sockets = 1, threads = 56, reps = 1;
  size_t region_mib = 20480, inter_mib = 32768;
  double gap = 0.6;
  const char *label = "run", *page = "2m", *perf_cmd = NULL, *perf_out = NULL, *perf_stop = NULL;
  static Cfg cfgs[MAX_CFGS];
  int ncfg = 0, verify_stride = 1;
  for (int i = 1; i + 1 < argc; i += 2) {
    const char *k = argv[i], *v = argv[i + 1];
    if (!strcmp(k, "--exp"))
      label = v;
    else if (!strcmp(k, "--sockets"))
      sockets = atoi(v);
    else if (!strcmp(k, "--threads"))
      threads = atoi(v);
    else if (!strcmp(k, "--reps"))
      reps = atoi(v);
    else if (!strcmp(k, "--page"))
      page = v;
    else if (!strcmp(k, "--region-mib"))
      region_mib = (size_t)atoll(v);
    else if (!strcmp(k, "--gap"))
      gap = atof(v);
    else if (!strcmp(k, "--inter-mib"))
      inter_mib = (size_t)atoll(v);
    else if (!strcmp(k, "--verify-stride"))
      verify_stride = atoi(v);
    else if (!strcmp(k, "--perf-cmd"))
      perf_cmd = v;
    else if (!strcmp(k, "--perf-out"))
      perf_out = v;
    else if (!strcmp(k, "--perf-stop"))
      perf_stop = v;
    else if (!strcmp(k, "--cfg") || !strcmp(k, "--cfgs")) {
      FILE *f = !strcmp(k, "--cfgs") ? fopen(v, "r") : NULL;
      if (!strcmp(k, "--cfgs") && !f)
        fail("cannot open %s", v);
      char line[512];
      const char *text = v;
      for (;;) {
        if (f) {
          if (!fgets(line, sizeof(line), f))
            break;
          line[strcspn(line, "\r\n")] = 0;
          if (!line[0] || line[0] == '#')
            continue;
          text = line;
        }
        if (ncfg == MAX_CFGS || parse_cfg(text, &cfgs[ncfg++]))
          fail("bad configuration '%s'", text);
        if (!f)
          break;
      }
      if (f)
        fclose(f);
    } else
      fail("unknown option %s", k);
  }
  if (!(argc % 2) || (sockets != 1 && sockets != 2) || threads < 2 || threads > 120 || reps < 1 || !ncfg ||
      (strcmp(page, "2m") && strcmp(page, "4k")) || verify_stride < 1 || region_mib < 64 ||
      (perf_cmd && (!perf_out || !perf_stop)))
    fail("invalid research options");
  kernels_init();
  layout_shard();
  static Run run;
  Run *R = &run;
  R->topology = numa_topology();
  const Topology *topology = R->topology;
  R->real = topology->nsockets == 2 && topology->socket[0].cpus && topology->socket[1].cpus &&
            topology->socket[0].ncpus >= threads && topology->socket[1].ncpus >= threads;
  R->sockets = sockets;
  R->threads = threads;
  R->page4k = !strcmp(page, "4k");
  R->verify_stride = verify_stride;
  R->inter_mib = inter_mib;
  PoolConfig config = {.teams = sockets,
                       .threads = {threads, threads},
                       .cpus = {R->real ? topology->socket[0].cpus : NULL, R->real ? topology->socket[1].cpus : NULL}};
  pool_start(&config);
  kernels_set_team(pool_team(0));
  int need_probe = 0, need_gemv = 0;
  for (int i = 0; i < ncfg; i++) {
    check_cfg(R, &cfgs[i]);
    need_probe |= cfgs[i].probe;
    need_gemv |= cfgs[i].kind == CFG_GEMV;
  }
  size_t bytes = region_mib * MIB;
  for (int so = 0; so < sockets; so++) {
    Sock *s = &R->sock[so];
    s->socket = so;
    s->team = pool_team(so);
    s->hbm_node = R->real ? topology->socket[so].hbm_node : -1;
    s->ddr_node = R->real ? topology->socket[so].ddr_node : -1;
    s->bytes = bytes;
    s->pbytes = (size_t)1024 * MIB;
    for (int t = 0; t < 2; t++) {
      region_reserve(&s->R[t], bytes, so, t ? s->ddr_node : s->hbm_node, ROLE_WEIGHTS, !R->page4k && R->real);
#ifdef __linux__
      if (R->page4k)
        madvise(s->R[t].data, s->R[t].data_bytes, MADV_NOHUGEPAGE);
#endif
      s->base[t] = s->R[t].data;
      FillJob2 fill = {s->base[t], bytes, need_gemv ? gemv_layers(s) : 0};
      if (need_gemv && fill.gemv_layers == 0)
        fail("region of %zu MiB holds no layer of the shard (%zu bytes per layer)", region_mib, layer_stride);
      team_run(s->team, fill2_fn, &fill);
      region_mark_populated(&s->R[t]);
      verify_tier(R, s, t, "setup", 0);
    }
    if (need_probe)
      for (int t = 0; t < 2; t++) {
        region_reserve(&s->P[t], s->pbytes, so, t ? s->ddr_node : s->hbm_node, ROLE_WEIGHTS, R->real);
        build_chase(s, t);
        region_mark_populated(&s->P[t]);
      }
    s->xq = memory_alloc(16384, 1, MEM_ACTIVATION);
    s->xs = memory_alloc(16384 / 32, sizeof(float), MEM_ACTIVATION);
    for (int i = 0; i < 16384; i++)
      s->xq[i] = (int8_t)((i * 37 % 255) - 127);
    for (int i = 0; i < 16384 / 32; i++)
      s->xs[i] = 0.01f;
    for (int m = 0; m < 7; m++)
      s->out[m] = memory_alloc(16384, sizeof(float), MEM_ACTIVATION);
  }
  pthread_t thread;
  if (sockets == 2) {
    int status = pthread_create(&thread, NULL, run_side, R);
    if (status)
      fail("cannot create the second coordinator: %s", strerror(status));
  }
  pid_t perf = -1;
  if (perf_cmd)
    perf = perf_start(perf_cmd, perf_out);
  (void)perf;
  sleep_s(0.8);
  printf("{\"event\": \"start\", \"experiment\": \"%s\", \"real_time\": %.6f, \"windows\": %d, \"reps\": %d, "
         "\"threads\": %d, "
         "\"sockets\": %d, \"page\": \"%s\", \"region_mib\": %zu, \"real\": %d, \"perf\": %d, \"layers\": %d, "
         "\"layer_stride\": %zu}\n",
         label, realtime_now(), ncfg * reps, reps, threads, sockets, page, region_mib, R->real, perf_cmd != NULL,
         need_gemv ? gemv_layers(&R->sock[0]) : 0, layer_stride);
  fflush(stdout);
  int index = 0;
  for (int rep = 0; rep < reps; rep++)
    for (int i = 0; i < ncfg; i++) {
      run_window(R, &cfgs[i], rep, index++);
      sleep_s(gap);
    }
  if (perf_cmd) {
    sleep_s(0.5);
    int fd = open(perf_stop, O_CREAT | O_WRONLY, 0644);
    if (fd >= 0)
      close(fd);
#ifdef __linux__
    int st;
    waitpid(perf, &st, 0);
#endif
  }
  for (int so = 0; so < sockets; so++)
    for (int t = 0; t < 2; t++)
      verify_tier(R, &R->sock[so], t, "end", R->page4k ? 1000000 : 0);
  R->stop = 1;
  atomic_store_explicit(&R->go_gen, atomic_load_explicit(&R->go_gen, memory_order_relaxed) + 1, memory_order_release);
  if (sockets == 2)
    pthread_join(thread, NULL);
  for (int so = 0; so < sockets; so++) {
    for (int t = 0; t < 2; t++) {
      region_release(&R->sock[so].R[t]);
      region_release(&R->sock[so].P[t]);
    }
  }
  pool_stop();
  return 0;
}

int main(int argc, char **argv) {
  for (int i = 1; i < argc; i++)
    if (!strcmp(argv[i], "--exp"))
      return research_main(argc, argv);
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
