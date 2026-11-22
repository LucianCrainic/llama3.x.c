/* Compute kernels: runtime ISA dispatch, the team the kernels run on, normalization and RoPE, and the execution
 * backend. The GEMV kernels are in gemv.c and the attention kernels in attention.c (split out in F8); the three share
 * kernels_internal.h. Scalar paths reproduce the original per-engine loops exactly. Team jobs are plain functions;
 * ISA-specific work runs in target-attributed helpers called once per row block, so the dispatch code never needs
 * AVX-512 itself. */
#include "gemm.h"
#include "kernels_internal.h"
#include "numa.h"
#include "tp.h"
#include "workers.h"
#include <math.h>
#include <stdatomic.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

KernelIsa kernels_isa = KERNELS_SCALAR;
static int initialized;
Team *kernels_default_team;
_Thread_local Team *kernels_thread_team;

void kernels_set_team(Team *team) {
  kernels_default_team = team;
}

void kernels_set_thread_team(Team *team) {
  kernels_thread_team = team;
}

static int supported(KernelIsa which) {
  if (which == KERNELS_SCALAR)
    return 1;
  if (which == KERNELS_AMX)
    return gemm_amx_unavailable() == NULL;
#if defined(KERNELS_EMULATE_AVX512)
  return 1;
#elif HAVE_AVX512
  __builtin_cpu_init();
  return __builtin_cpu_supports("avx512f") && __builtin_cpu_supports("avx512bw") &&
         __builtin_cpu_supports("avx512dq") && __builtin_cpu_supports("avx512vl") &&
         __builtin_cpu_supports("avx512vnni") && __builtin_cpu_supports("fma");
#else
  return 0;
#endif
}

static int amx_auto; /* LLAMA_KERNELS=auto on a machine where AMX can run: kernels_select_amx_auto may take it */

int kernels_select(KernelIsa which) {
  if (!supported(which))
    return 0;
  /* AMX runs on top of the AVX-512 kernels: the kernels_isa stays AVX-512, and the GEMM drivers look at gemm_amx_active(). */
  kernels_isa = which == KERNELS_AMX ? (supported(KERNELS_AVX512) ? KERNELS_AVX512 : KERNELS_SCALAR) : which;
  gemm_amx_enable(which == KERNELS_AMX);
  initialized = 1;
  return 1;
}


void kernels_init(void) {
  check_retired_knobs();
  const char *request = getenv("LLAMA_KERNELS");
  if (!request || !*request || !strcmp(request, "auto")) {
    kernels_select(supported(KERNELS_AVX512) ? KERNELS_AVX512 : KERNELS_SCALAR);
    amx_auto = 1; /* AMX itself (which implies AVX-512) is asked for only when an engine wants it (C4) */
  } else if (!strcmp(request, "scalar")) {
    kernels_select(KERNELS_SCALAR);
  } else if (!strcmp(request, "avx512")) {
    if (!kernels_select(KERNELS_AVX512))
      fail("LLAMA_KERNELS=avx512 requested, but AVX-512 F/BW/DQ/VL/VNNI is unavailable");
  } else if (!strcmp(request, "amx")) {
    if (!kernels_select(KERNELS_AMX))
      fail("LLAMA_KERNELS=amx requested, but AMX is unavailable: %s", gemm_amx_unavailable());
  } else {
    fail("LLAMA_KERNELS must be auto, scalar, avx512 or amx");
  }
  kernels_attention_init_env();
  gemm_q8_init_env();
  gemm_swiglu_init_env();
  LOG(LOG_DEBUG, "compute kernels: %s, attention %s, decode attention %s, K/V layout %s", kernels_name(),
      kernels_attention_name(), kernels_attention_decode_name(), kernels_kv_layout_name());
}

int kernels_select_amx_auto(void) {
  /* The tile-data permission is requested here, after the workers exist; Linux grants it to the whole process, and no
   * thread has executed a tile instruction yet. */
  if (amx_auto && !gemm_amx_active() && kernels_select(KERNELS_AMX))
    LOG(LOG_INFO, "LLAMA_KERNELS=auto: AMX runs the batched prefill GEMMs (LLAMA_KERNELS=avx512 keeps AVX-512)");
  return gemm_amx_active();
}

const char *kernels_name(void) {
  if (!initialized)
    return NULL;
  return gemm_amx_active() ? "amx" : kernels_isa == KERNELS_AVX512 ? "avx512" : "scalar";
}


int kernels_avx512_active(void) {
  return kernels_isa == KERNELS_AVX512;
}

int kernels_amx_active(void) {
  return gemm_amx_active();
}

typedef struct {
  SegmentFn fn;
  const void *job;
  int count;
  const int *chunks;
} SegmentRun;

static void run_segments(const SegmentRun *run, int worker, int workers) {
  for (int s = 0; s < run->count; s++) {
    int first = (int)((long long)run->chunks[s] * worker / workers);
    int last = (int)((long long)run->chunks[s] * (worker + 1) / workers);
    if (last > first)
      run->fn(run->job, s, first, last);
  }
}

static void segments_team(const void *p, int worker, int workers) {
  run_segments(p, worker, workers);
}

void kernels_parallel_segments(int count, const int *chunks, SegmentFn fn, const void *job) {
  SegmentRun run = {fn, job, count, chunks};
  if (current_team) {
    team_run(current_team, segments_team, &run);
    return;
  }
  run_segments(&run, 0, 1);
}


#if HAVE_AVX512
AVX512 static void rmsnorm_avx512(float *o, const float *x, const float *weight, int size, float epsilon) {
  __m512 a0 = _mm512_setzero_ps(), a1 = a0, a2 = a0, a3 = a0;
  int j = 0;
  for (; j + 64 <= size; j += 64) {
    __m512 v0 = _mm512_loadu_ps(x + j), v1 = _mm512_loadu_ps(x + j + 16);
    __m512 v2 = _mm512_loadu_ps(x + j + 32), v3 = _mm512_loadu_ps(x + j + 48);
    a0 = _mm512_fmadd_ps(v0, v0, a0);
    a1 = _mm512_fmadd_ps(v1, v1, a1);
    a2 = _mm512_fmadd_ps(v2, v2, a2);
    a3 = _mm512_fmadd_ps(v3, v3, a3);
  }
  for (; j < size; j += 16) {
    __m512 v = _mm512_maskz_loadu_ps(tail_mask(size - j), x + j);
    a0 = _mm512_fmadd_ps(v, v, a0);
  }
  float ss = _mm512_reduce_add_ps(_mm512_add_ps(_mm512_add_ps(a0, a1), _mm512_add_ps(a2, a3)));
  ss /= size;
  ss += epsilon;
  ss = 1.0f / sqrtf(ss);
  __m512 scale = _mm512_set1_ps(ss);
  for (j = 0; j < size; j += 16) {
    __mmask16 m = tail_mask(size - j);
    __m512 v = _mm512_mul_ps(_mm512_maskz_loadu_ps(m, weight + j), _mm512_mul_ps(scale, _mm512_maskz_loadu_ps(m, x + j)));
    _mm512_mask_storeu_ps(o + j, m, v);
  }
}

/* q*c + swap(q)*(-s, +s) without FMA, matching a*c - b*s and a*s + b*c exactly. */
AVX512 static void rope_avx512(float *v, const float *table, int heads, int head_dim) {
  const float *cosines = table, *sines = table + head_dim;
  for (int h = 0; h < heads; h++) {
    float *row = v + (size_t)h * head_dim;
    for (int j = 0; j < head_dim; j += 16) {
      __mmask16 m = tail_mask(head_dim - j);
      __m512 a = _mm512_maskz_loadu_ps(m, row + j);
      __m512 swapped = _mm512_permute_ps(a, 0xB1);
      __m512 r = _mm512_add_ps(_mm512_mul_ps(a, _mm512_maskz_loadu_ps(m, cosines + j)),
                               _mm512_mul_ps(swapped, _mm512_maskz_loadu_ps(m, sines + j)));
      _mm512_mask_storeu_ps(row + j, m, r);
    }
  }
}

#endif

// ----------------------------------------------------------------------------
// Normalization and RoPE

void k_rmsnorm(float *out, const float *x, const float *weight, int size, float epsilon) {
#if HAVE_AVX512
  if (kernels_isa == KERNELS_AVX512) {
    rmsnorm_avx512(out, x, weight, size, epsilon);
    return;
  }
#endif
  rmsnorm_eps(out, x, weight, size, epsilon);
}

void k_rope_table(const Config *p, int position, float *table) {
  for (int j = 0; j < p->head_dim; j += 2) {
    float angle = position * rope_frequency(p, j), c = cosf(angle), s = sinf(angle);
    table[j] = table[j + 1] = c;
    table[p->head_dim + j] = -s;
    table[p->head_dim + j + 1] = s;
  }
}

void k_rope_apply(float *q, float *k, const float *table, int n_heads, int n_kv_heads, int head_dim) {
#if HAVE_AVX512
  if (kernels_isa == KERNELS_AVX512) {
    rope_avx512(q, table, n_heads, head_dim);
    rope_avx512(k, table, n_kv_heads, head_dim);
    return;
  }
#endif
  const float *sines = table + head_dim;
  for (int h = 0; h < n_heads; h++) {
    for (int j = 0; j < head_dim; j += 2) {
      float c = table[j], s = sines[j + 1];
      int i = h * head_dim + j;
      float a = q[i], b = q[i + 1];
      q[i] = a * c - b * s;
      q[i + 1] = a * s + b * c;
      if (h < n_kv_heads) {
        a = k[i];
        b = k[i + 1];
        k[i] = a * c - b * s;
        k[i + 1] = a * s + b * c;
      }
    }
  }
}

/* ---- Row-parallel forms for batched prefill ---------------------------------------------------------------------------
 * A microbatch normalizes and rotates M independent rows. Run one after the other on the thread that leads the team, these
 * loops do not shrink with the number of workers or sockets: RMSNorm alone was 18% of a two-socket AMX-BF16 prefill token on
 * 8B (D1, job 5704694). Each row is exactly the single-row call, so the results are bitwise those of the serial loops. */
typedef struct {
  float *out;
  const float *x, *weight;
  int size;
  float epsilon;
} RmsRowsJob;

static void rmsnorm_rows_chunks(const void *p, int segment, int first, int last) {
  const RmsRowsJob *job = p;
  (void)segment;
  for (int i = first; i < last; i++)
    k_rmsnorm(job->out + (size_t)i * job->size, job->x + (size_t)i * job->size, job->weight, job->size, job->epsilon);
}

void k_rmsnorm_rows(float *out, const float *x, const float *weight, int rows, int size, float epsilon) {
  RmsRowsJob job = {out, x, weight, size, epsilon};
  if (rows < 2) {
    if (rows == 1)
      k_rmsnorm(out, x, weight, size, epsilon);
    return;
  }
  kernels_parallel_segments(1, &rows, rmsnorm_rows_chunks, &job);
}

typedef struct {
  float *q, *k;
  const float *table;
  size_t q_stride, k_stride, table_stride;
  int n_heads, n_kv_heads, head_dim;
} RopeRowsJob;

static void rope_rows_chunks(const void *p, int segment, int first, int last) {
  const RopeRowsJob *job = p;
  (void)segment;
  for (int i = first; i < last; i++)
    k_rope_apply(job->q + (size_t)i * job->q_stride, job->k + (size_t)i * job->k_stride,
                 job->table + (size_t)i * job->table_stride, job->n_heads, job->n_kv_heads, job->head_dim);
}

void k_rope_apply_rows(float *q, size_t q_stride, float *k, size_t k_stride, const float *table, size_t table_stride,
                       int rows, int n_heads, int n_kv_heads, int head_dim) {
  RopeRowsJob job = {q, k, table, q_stride, k_stride, table_stride, n_heads, n_kv_heads, head_dim};
  if (rows < 2) {
    if (rows == 1)
      k_rope_apply(q, k, table, n_heads, n_kv_heads, head_dim);
    return;
  }
  kernels_parallel_segments(1, &rows, rope_rows_chunks, &job);
}

// ----------------------------------------------------------------------------
// Execution backend

static int backend_started;

const char *kernels_backend_name(void) {
  return "teams";
}

void kernels_backend_init(void) {
  int sockets = tp_sockets_requested();
  /* Discover the topology now, from the job's full allowed sets, before any worker is pinned: the report
   * reads this cached result after the run. A failure is cached too and reported, not fatal here. */
  const Topology *topology_result;
  char topology_error[512];
  numa_topology_try(&topology_result, topology_error, sizeof(topology_error));
  if (backend_started)
    return;
  int threads = 0, available;
  const char *requested = getenv("LLAMA_THREADS");
  if (requested && *requested) {
    char *end;
    long value = strtol(requested, &end, 10);
    if (*end || value < 1 || value > TEAM_SUMMARY_THREADS)
      fail("LLAMA_THREADS must be an integer in 1..%d", TEAM_SUMMARY_THREADS);
    threads = (int)value;
  }
  if (sockets == 2) { /* B5: one team per socket, on that socket's CPUs (virtual: unpinned) */
    tp_check_topology("file");
    const Topology *two = numa_topology();
    PoolConfig config = {.teams = 2};
    for (int s = 0; s < 2; s++) {
      const SocketTopo *socket = &two->socket[s];
      config.threads[s] = threads ? threads : socket->ncpus;
      config.cpus[s] = socket->cpus;
      if (socket->cpus && config.threads[s] > socket->ncpus) {
        LOG(LOG_WARN, "LLAMA_THREADS=%d exceeds the %d CPUs of socket %d: its workers are not pinned", threads,
            socket->ncpus, s);
        config.cpus[s] = NULL;
      }
    }
    pool_start(&config);
    kernels_set_team(pool_team(0));
    backend_started = 1;
    LOG(LOG_DEBUG, "worker teams: %d + %d threads", config.threads[0], config.threads[1]);
    return;
  }
  static int affinity[1024];
  const int *cpus;
  const char *topology = getenv("LLAMA_TOPOLOGY");
  if (topology && *topology) {
    const SocketTopo *socket = &numa_topology()->socket[0]; /* one team on the first socket until B adds two */
    available = socket->ncpus;
    cpus = socket->cpus;
  } else {
    int allowed = numa_affinity_cpus(affinity, 1024);
    available = allowed > 0 ? allowed : (int)sysconf(_SC_NPROCESSORS_ONLN);
    cpus = allowed > 0 ? affinity : NULL;
  }
  if (available < 1)
    available = 1;
  if (!threads)
    threads = available;
  if (cpus && threads > available) {
    LOG(LOG_WARN, "LLAMA_THREADS=%d exceeds the %d allowed CPUs: workers are not pinned", threads, available);
    cpus = NULL;
  }
  PoolConfig config = {.teams = 1, .threads = {threads, 0}, .cpus = {cpus, NULL}};
  pool_start(&config);
  kernels_set_team(pool_team(0));
  backend_started = 1;
  LOG(LOG_DEBUG, "worker team: %d threads, %s", threads, cpus ? "pinned" : "unpinned");
}

void kernels_backend_free(void) {
  if (!backend_started)
    return;
  kernels_set_team(NULL);
  pool_stop();
  gemm_amx_thread_release(); /* the coordinator's tiles; the workers released theirs when they exited */
  backend_started = 0;
}
