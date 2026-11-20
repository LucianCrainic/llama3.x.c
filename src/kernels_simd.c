/* Compute kernels: runtime ISA dispatch, scalar references and AVX-512 paths.
 * Scalar paths reproduce the original per-engine loops exactly. OpenMP regions
 * live only in plain functions; ISA-specific work runs in target-attributed
 * helpers called once per row block, so no outlined region needs AVX-512. */
#include "gemm.h"
#include "kernels.h"
#include "numa.h"
#include "tp.h"
#include "workers.h"
#include <float.h>
#include <limits.h>
#include <math.h>
#include <stdatomic.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#ifdef _OPENMP
#include <omp.h>
#endif

#if defined(KERNELS_EMULATE_AVX512)
/* Test builds (make test-kernels-emul): the AVX-512 kernels of this file run on a scalar model of the intrinsics, so their
 * code, masks and tails are exercised on any host. Only speed and the hardware's exact rounding stay with real AVX-512. */
#define HAVE_AVX512 1
#include "../tests/avx512_emul.h"
#define AVX512
#elif defined(__x86_64__) && (defined(__GNUC__) || defined(__clang__))
#define HAVE_AVX512 1
#include <immintrin.h>
#define AVX512 __attribute__((target("avx512f,avx512bw,avx512dq,avx512vl,avx512vnni,fma")))
#else
#define HAVE_AVX512 0
#endif

static KernelIsa isa = KERNELS_SCALAR;
static int initialized;
/* Rows of one (kv group, chunk) task of the batched attention (E0 diagnostic, LLAMA_ATTN_ROW_BLOCK, read once by
 * kernels_init). Rows of a block are computed one after the other, each exactly as without the block, so the result is
 * bitwise independent of it. */
enum { BATCH_MAX_ROW_BLOCK = 16 };
static int attention_row_block = 4;
/* E2 tiled attention: query vectors per task (rows of a block x heads of the group, 1..16, LLAMA_ATTN_TILED_QUERIES) and
 * whether the tasks of a wave are claimed in plan order (default, LLAMA_ATTN_TILED_ORDER=natural) or longest first
 * (longest; E2 bench: natural was faster at 2048 and 8192 tokens). Neither changes how a
 * row is computed once its block is fixed; the block size sets the chunk plan, the order only the schedule. */
static int attention_tiled_queries = 16, attention_tiled_longest_first = 0;
/* With a team set, kernels run on it instead of opening OpenMP regions. */
static Team *default_team;
/* A coordinator thread that drives one socket's team (B2) binds its kernels to that team without touching the
 * process-wide default, so two coordinators can run kernels on two teams at the same time. */
static _Thread_local Team *thread_team;
#define current_team (thread_team ? thread_team : default_team)

void kernels_set_team(Team *team) {
  default_team = team;
}

void kernels_set_thread_team(Team *team) {
  thread_team = team;
}

static void attention_init_from_env(void);
static void kv_layout_init_from_env(void);

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

int kernels_select(KernelIsa which) {
  if (!supported(which))
    return 0;
  /* AMX runs on top of the AVX-512 kernels: the isa stays AVX-512, and the GEMM drivers look at gemm_amx_active(). */
  isa = which == KERNELS_AMX ? (supported(KERNELS_AVX512) ? KERNELS_AVX512 : KERNELS_SCALAR) : which;
  gemm_amx_enable(which == KERNELS_AMX);
  initialized = 1;
  return 1;
}

/* How a team splits the rows of one GEMV. Static: each member takes one contiguous share, so one slow member (a memory
 * channel under contention, an interrupt) holds the whole team at the join. Dynamic: members claim chunks of rows from a
 * shared counter, so the fast ones take over what the slow one has not reached, but no member streams a long contiguous
 * run any more. Hybrid: each member first takes a contiguous share of three quarters of the rows, then all claim the
 * rest in small chunks. A claim starts on a multiple of BLOCK rows, so every row sits in the same four-row group
 * whatever the team size or the timing. LLAMA_GEMV_SCHED=static|dynamic|hybrid|auto selects it for the FP32 and the
 * Q8 kernels together. Auto (default) is hybrid for large FP32 jobs and static otherwise, and static for Q8, whose
 * kernels prefetch along long contiguous runs and lose 6-25% to any claiming (measured on Xeon Max 9480, docs/plans). */
typedef enum { SCHED_STATIC, SCHED_DYNAMIC, SCHED_HYBRID, SCHED_AUTO } GemvSched;
static GemvSched gemv_sched_f32 = SCHED_AUTO, gemv_sched_q8 = SCHED_STATIC;
/* auto, FP32: hybrid for a job that streams at least this many weight bytes (the 8B shards: 117-235 MiB; measured +14% on
 * two sockets), static below it (1B: at most 64 MiB, where the claims cost 4-6%; Wo shards: 33 MiB). */
enum { AUTO_HYBRID_BYTES = 96 << 20 };

static GemvSched effective_f32(size_t bytes) {
  return gemv_sched_f32 == SCHED_AUTO ? (bytes >= AUTO_HYBRID_BYTES ? SCHED_HYBRID : SCHED_STATIC) : gemv_sched_f32;
}

void kernels_init(void) {
  const char *sched = getenv("LLAMA_GEMV_SCHED");
  gemv_sched_f32 = SCHED_AUTO;
  gemv_sched_q8 = SCHED_STATIC;
  if (sched && *sched && strcmp(sched, "auto")) {
    GemvSched which = !strcmp(sched, "static") ? SCHED_STATIC
                      : !strcmp(sched, "dynamic") ? SCHED_DYNAMIC
                      : !strcmp(sched, "hybrid") ? SCHED_HYBRID
                                                 : (GemvSched)-1;
    if ((int)which < 0)
      fail("LLAMA_GEMV_SCHED must be static, dynamic, hybrid or auto");
    gemv_sched_f32 = gemv_sched_q8 = which;
  }
  if (gemv_sched_q8 == SCHED_AUTO)
    gemv_sched_q8 = SCHED_STATIC;
  const char *rows = getenv("LLAMA_ATTN_ROW_BLOCK"); /* diagnostic: rows per batched-attention task, 1..16, default 4 */
  attention_row_block = 4;
  if (rows && *rows) {
    char *end;
    long value = strtol(rows, &end, 10);
    if (*end || value < 1 || value > BATCH_MAX_ROW_BLOCK)
      fail("LLAMA_ATTN_ROW_BLOCK must be an integer in 1..%d", BATCH_MAX_ROW_BLOCK);
    attention_row_block = (int)value;
  }
  const char *queries = getenv("LLAMA_ATTN_TILED_QUERIES"), *order = getenv("LLAMA_ATTN_TILED_ORDER");
  attention_tiled_queries = 16, attention_tiled_longest_first = 0;
  if (queries && *queries) {
    char *end;
    long value = strtol(queries, &end, 10);
    if (*end || value < 1 || value > 16)
      fail("LLAMA_ATTN_TILED_QUERIES must be an integer in 1..16");
    attention_tiled_queries = (int)value;
  }
  if (order && *order) {
    if (strcmp(order, "longest") && strcmp(order, "natural"))
      fail("LLAMA_ATTN_TILED_ORDER must be longest or natural");
    attention_tiled_longest_first = !strcmp(order, "longest");
  }
  const char *request = getenv("LLAMA_KERNELS");
  if (!request || !*request || !strcmp(request, "auto")) {
    kernels_select(supported(KERNELS_AVX512) ? KERNELS_AVX512 : KERNELS_SCALAR);
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
  attention_init_from_env();
  kv_layout_init_from_env();
  LOG(LOG_DEBUG, "compute kernels: %s, attention %s, decode attention %s, K/V layout %s", kernels_name(),
      kernels_attention_name(), kernels_attention_decode_name(), kernels_kv_layout_name());
}

const char *kernels_name(void) {
  if (!initialized)
    return NULL;
  return gemm_amx_active() ? "amx" : isa == KERNELS_AVX512 ? "avx512" : "scalar";
}

enum { BLOCK = 4 }; // rows per GEMV kernel call

/* Rows per claim: about four claims per member, a multiple of BLOCK, between 8 and 32 (a claim is a 128-512 KiB stream
 * of a large matrix, long enough that the shared counter is touched rarely). */
static int claim_rows(int total_rows, int workers) {
  int rows = total_rows / (workers * 4) / BLOCK * BLOCK;
  return rows < 8 ? 8 : rows > 32 ? 32 : rows;
}

int kernels_avx512_active(void) {
  return isa == KERNELS_AVX512;
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
#ifdef _OPENMP
#pragma omp parallel
  run_segments(&run, omp_get_thread_num(), omp_get_num_threads());
#else
  run_segments(&run, 0, 1);
#endif
}

/* Each thread receives one contiguous, balanced row range, so weight rows are
 * streamed sequentially per thread and hardware prefetch stays effective. */
static void parallel_rows(int rows, RowFn fn, const void *job) {
  if (current_team) {
    team_parallel_rows(current_team, rows, fn, job);
    return;
  }
#ifdef _OPENMP
#pragma omp parallel
  {
    int t = omp_get_thread_num(), nt = omp_get_num_threads();
    int r0 = (int)((long long)rows * t / nt), r1 = (int)((long long)rows * (t + 1) / nt);
    if (r1 > r0)
      fn(job, r0, r1);
  }
#else
  fn(job, 0, rows);
#endif
}

// ----------------------------------------------------------------------------
// Split-K partials

/* The partial of a chunk that holds no position of the row (tiled, E2): zero value sum, the lowest maximum, zero sum of
 * exponentials. A real chunk always has sum >= 1 (its maximum contributes exp(0)), so the test cannot match one. */
static int partial_empty(const float *chunk, int head_dim) {
  return chunk[head_dim + 1] == 0.0f && chunk[head_dim] == -FLT_MAX;
}

// ----------------------------------------------------------------------------
// AVX-512 helpers

#if HAVE_AVX512
static void partial_write_empty(float *o, int head_dim) {
  memset(o, 0, (size_t)head_dim * sizeof(float));
  o[head_dim] = -FLT_MAX;
  o[head_dim + 1] = 0.0f;
}

AVX512 static inline __mmask16 tail_mask(int count) {
  return count >= 16 ? (__mmask16)0xFFFF : (__mmask16)((1u << count) - 1u);
}

/* Cephes-style expf: range reduction by ln 2, degree-6 polynomial, vscalefps.
 * Inputs are clamped to the finite float range; relative error is ~2 ulp. */
AVX512 static inline __m512 exp512(__m512 x) {
  x = _mm512_min_ps(_mm512_max_ps(x, _mm512_set1_ps(-87.33654f)), _mm512_set1_ps(88.72283f));
  __m512 n = _mm512_roundscale_ps(_mm512_mul_ps(x, _mm512_set1_ps(1.44269504088896341f)),
                                  _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC);
  __m512 r = _mm512_fnmadd_ps(n, _mm512_set1_ps(0.693359375f), x);
  r = _mm512_fnmadd_ps(n, _mm512_set1_ps(-2.12194440e-4f), r);
  __m512 y = _mm512_set1_ps(1.9875691500e-4f);
  y = _mm512_fmadd_ps(y, r, _mm512_set1_ps(1.3981999507e-3f));
  y = _mm512_fmadd_ps(y, r, _mm512_set1_ps(8.3334519073e-3f));
  y = _mm512_fmadd_ps(y, r, _mm512_set1_ps(4.1665795894e-2f));
  y = _mm512_fmadd_ps(y, r, _mm512_set1_ps(1.6666665459e-1f));
  y = _mm512_fmadd_ps(y, r, _mm512_set1_ps(5.0000001201e-1f));
  y = _mm512_fmadd_ps(y, _mm512_mul_ps(r, r), _mm512_add_ps(r, _mm512_set1_ps(1.0f)));
  return _mm512_scalef_ps(y, n);
}

/* rows (1..4) dot products of W rows with x. Four rows share each x load and
 * provide independent FMA chains; two accumulators per row hide FMA latency. */
AVX512 static void dots_f32_avx512(const float *w, size_t stride, int rows, const float *x, int n, float *out) {
  if (rows == 4) {
    const float *w0 = w, *w1 = w + stride, *w2 = w + 2 * stride, *w3 = w + 3 * stride;
    __m512 a0 = _mm512_setzero_ps(), a1 = a0, a2 = a0, a3 = a0, b0 = a0, b1 = a0, b2 = a0, b3 = a0;
    int j = 0;
    for (; j + 32 <= n; j += 32) {
      __m512 x0 = _mm512_loadu_ps(x + j), x1 = _mm512_loadu_ps(x + j + 16);
      a0 = _mm512_fmadd_ps(_mm512_loadu_ps(w0 + j), x0, a0);
      b0 = _mm512_fmadd_ps(_mm512_loadu_ps(w0 + j + 16), x1, b0);
      a1 = _mm512_fmadd_ps(_mm512_loadu_ps(w1 + j), x0, a1);
      b1 = _mm512_fmadd_ps(_mm512_loadu_ps(w1 + j + 16), x1, b1);
      a2 = _mm512_fmadd_ps(_mm512_loadu_ps(w2 + j), x0, a2);
      b2 = _mm512_fmadd_ps(_mm512_loadu_ps(w2 + j + 16), x1, b2);
      a3 = _mm512_fmadd_ps(_mm512_loadu_ps(w3 + j), x0, a3);
      b3 = _mm512_fmadd_ps(_mm512_loadu_ps(w3 + j + 16), x1, b3);
    }
    for (; j < n; j += 16) {
      __mmask16 m = tail_mask(n - j);
      __m512 x0 = _mm512_maskz_loadu_ps(m, x + j);
      a0 = _mm512_fmadd_ps(_mm512_maskz_loadu_ps(m, w0 + j), x0, a0);
      a1 = _mm512_fmadd_ps(_mm512_maskz_loadu_ps(m, w1 + j), x0, a1);
      a2 = _mm512_fmadd_ps(_mm512_maskz_loadu_ps(m, w2 + j), x0, a2);
      a3 = _mm512_fmadd_ps(_mm512_maskz_loadu_ps(m, w3 + j), x0, a3);
    }
    out[0] = _mm512_reduce_add_ps(_mm512_add_ps(a0, b0));
    out[1] = _mm512_reduce_add_ps(_mm512_add_ps(a1, b1));
    out[2] = _mm512_reduce_add_ps(_mm512_add_ps(a2, b2));
    out[3] = _mm512_reduce_add_ps(_mm512_add_ps(a3, b3));
    return;
  }
  for (int r = 0; r < rows; r++) {
    const float *wr = w + r * stride;
    __m512 a0 = _mm512_setzero_ps(), a1 = a0, a2 = a0, a3 = a0;
    int j = 0;
    for (; j + 64 <= n; j += 64) {
      a0 = _mm512_fmadd_ps(_mm512_loadu_ps(wr + j), _mm512_loadu_ps(x + j), a0);
      a1 = _mm512_fmadd_ps(_mm512_loadu_ps(wr + j + 16), _mm512_loadu_ps(x + j + 16), a1);
      a2 = _mm512_fmadd_ps(_mm512_loadu_ps(wr + j + 32), _mm512_loadu_ps(x + j + 32), a2);
      a3 = _mm512_fmadd_ps(_mm512_loadu_ps(wr + j + 48), _mm512_loadu_ps(x + j + 48), a3);
    }
    for (; j < n; j += 16) {
      __mmask16 m = tail_mask(n - j);
      a0 = _mm512_fmadd_ps(_mm512_maskz_loadu_ps(m, wr + j), _mm512_maskz_loadu_ps(m, x + j), a0);
    }
    out[r] = _mm512_reduce_add_ps(_mm512_add_ps(_mm512_add_ps(a0, a1), _mm512_add_ps(a2, a3)));
  }
}

/* The same four dot products over the packed4 layout (see k_pack4): group g holds rows 4g..4g+3 as one sequential stream,
 * [column block b][row r][16 floats]. Every row sees exactly the operations of the rows == 4 path above (two accumulators,
 * 32 columns per step, the masked tail into the first), so the results are bitwise equal; only the addresses differ.
 * Weight lines are prefetched P4_PREFETCH_BYTES ahead along the stream (prefetching past the end never faults). */
enum { P4_PREFETCH_BYTES = 4096 };
AVX512 static void dots_p4_avx512(const float *p, const float *x, int n, float *out) {
  const size_t ahead = P4_PREFETCH_BYTES / sizeof(float);
  __m512 a0 = _mm512_setzero_ps(), a1 = a0, a2 = a0, a3 = a0, b0 = a0, b1 = a0, b2 = a0, b3 = a0;
  int j = 0;
  for (; j + 32 <= n; j += 32) {
    const float *q = p + (size_t)(j / 16) * 64, *q2 = q + 64;
    for (int line = 0; line < 8; line++)
      _mm_prefetch((const char *)(q + ahead + (size_t)line * 16), _MM_HINT_T0);
    __m512 x0 = _mm512_loadu_ps(x + j), x1 = _mm512_loadu_ps(x + j + 16);
    a0 = _mm512_fmadd_ps(_mm512_loadu_ps(q), x0, a0);
    b0 = _mm512_fmadd_ps(_mm512_loadu_ps(q2), x1, b0);
    a1 = _mm512_fmadd_ps(_mm512_loadu_ps(q + 16), x0, a1);
    b1 = _mm512_fmadd_ps(_mm512_loadu_ps(q2 + 16), x1, b1);
    a2 = _mm512_fmadd_ps(_mm512_loadu_ps(q + 32), x0, a2);
    b2 = _mm512_fmadd_ps(_mm512_loadu_ps(q2 + 32), x1, b2);
    a3 = _mm512_fmadd_ps(_mm512_loadu_ps(q + 48), x0, a3);
    b3 = _mm512_fmadd_ps(_mm512_loadu_ps(q2 + 48), x1, b3);
  }
  for (; j < n; j += 16) {
    __mmask16 m = tail_mask(n - j);
    const float *q = p + (size_t)(j / 16) * 64;
    __m512 x0 = _mm512_maskz_loadu_ps(m, x + j);
    a0 = _mm512_fmadd_ps(_mm512_maskz_loadu_ps(m, q), x0, a0);
    a1 = _mm512_fmadd_ps(_mm512_maskz_loadu_ps(m, q + 16), x0, a1);
    a2 = _mm512_fmadd_ps(_mm512_maskz_loadu_ps(m, q + 32), x0, a2);
    a3 = _mm512_fmadd_ps(_mm512_maskz_loadu_ps(m, q + 48), x0, a3);
  }
  out[0] = _mm512_reduce_add_ps(_mm512_add_ps(a0, b0));
  out[1] = _mm512_reduce_add_ps(_mm512_add_ps(a1, b1));
  out[2] = _mm512_reduce_add_ps(_mm512_add_ps(a2, b2));
  out[3] = _mm512_reduce_add_ps(_mm512_add_ps(a3, b3));
}

/* The Q8 bodies take a compile-time row count and are force-inlined into one
 * wrapper per count. Fixed-size accumulator arrays then live in registers;
 * with a runtime count they were kept on the stack, and every VNNI and FMA step
 * went through a store and a reload. */
#define Q8_INLINE AVX512 static inline __attribute__((always_inline))

/* Q8 work per weight byte is several times FP32's, so demand loads alone keep
 * too few cache misses in flight to saturate HBM; weight lines are prefetched
 * `ahead` bytes early. Short rows (a block of at most 8 KiB) prefetch the next
 * block at the same offset. That adds streams, which pays off while a core has
 * bandwidth to spare but costs HBM efficiency at a saturated socket, so longer
 * rows prefetch 1 KiB ahead within the row (W8A8) or not at all (W8A32, which
 * is compute-bound there). Measured on Xeon Max 9480 HBM, 1B and 8B Q8_0.
 * Prefetching past the end of an array never faults. */
enum { Q8_NEXT_BLOCK_MAX = 8192, Q8_IN_ROW = 1024 };
static inline size_t q8_ahead(int rows, size_t n, int fp32_activations) {
  size_t block = (size_t)rows * n;
  if (block <= Q8_NEXT_BLOCK_MAX)
    return block;
  return fp32_activations ? 0 : Q8_IN_ROW;
}

/* W8A8 with VNNI. vpdpbusd multiplies unsigned by signed bytes, so |x| is the
 * unsigned operand and w takes x's sign; exact because no value is -128. Each
 * group's int32 lanes are scaled once into FP32 accumulators. gs % 64 == 0. */
Q8_INLINE void dots_q8_body(const int8_t *wq, const float *ws, size_t n, size_t ld, const int rows, const int8_t *xq,
                            const float *xs, int gs, float *out) {
  const __m512i zero = _mm512_setzero_si512();
  const size_t ahead = q8_ahead(rows, n, 0);
  __m512 acc[BLOCK];
  for (int r = 0; r < rows; r++)
    acc[r] = _mm512_setzero_ps();
  size_t groups = n / gs, scale_ld = ld / gs;
  for (size_t g = 0; g < groups; g++) {
    __m512i iacc[BLOCK];
    for (int r = 0; r < rows; r++)
      iacc[r] = zero;
    for (int c = 0; c < gs; c += 64) {
      size_t off = g * gs + c;
      __m512i xv = _mm512_loadu_si512((const void *)(xq + off));
      __m512i ax = _mm512_abs_epi8(xv);
      __mmask64 negative = _mm512_movepi8_mask(xv);
      for (int r = 0; r < rows; r++) {
        _mm_prefetch((const char *)(wq + r * ld + off + ahead), _MM_HINT_T0);
        __m512i wv = _mm512_loadu_si512((const void *)(wq + r * ld + off));
        iacc[r] = _mm512_dpbusd_epi32(iacc[r], ax, _mm512_mask_sub_epi8(wv, negative, zero, wv));
      }
    }
    __m512 xscale = _mm512_set1_ps(xs[g]);
    for (int r = 0; r < rows; r++)
      acc[r] = _mm512_fmadd_ps(_mm512_mul_ps(_mm512_cvtepi32_ps(iacc[r]), xscale), _mm512_set1_ps(ws[r * scale_ld + g]),
                               acc[r]);
  }
  for (int r = 0; r < rows; r++)
    out[r] = _mm512_reduce_add_ps(acc[r]);
}

/* W8A32: weights are widened in registers and multiplied by FP32 activations;
 * each group's partial sum is scaled once. Two partial sums per row hide FMA
 * latency. gs % 16 == 0. */
Q8_INLINE void dots_q8f_body(const int8_t *wq, const float *ws, size_t n, size_t ld, const int rows, const float *x,
                             int gs, float *out) {
  const size_t ahead = q8_ahead(rows, n, 1);
  __m512 acc[BLOCK];
  for (int r = 0; r < rows; r++)
    acc[r] = _mm512_setzero_ps();
  size_t groups = n / gs, scale_ld = ld / gs;
  for (size_t g = 0; g < groups; g++) {
    __m512 ga[BLOCK], gb[BLOCK];
    for (int r = 0; r < rows; r++)
      ga[r] = gb[r] = _mm512_setzero_ps();
    int c = 0;
    for (; c + 32 <= gs; c += 32) {
      size_t off = g * gs + c;
      __m512 x0 = _mm512_loadu_ps(x + off), x1 = _mm512_loadu_ps(x + off + 16);
      for (int r = 0; r < rows; r++) {
        if (ahead && !(off & 63))
          _mm_prefetch((const char *)(wq + r * ld + off + ahead), _MM_HINT_T0);
        __m256i bytes = _mm256_loadu_si256((const __m256i *)(wq + r * ld + off));
        __m512 w0 = _mm512_cvtepi32_ps(_mm512_cvtepi8_epi32(_mm256_castsi256_si128(bytes)));
        __m512 w1 = _mm512_cvtepi32_ps(_mm512_cvtepi8_epi32(_mm256_extracti128_si256(bytes, 1)));
        ga[r] = _mm512_fmadd_ps(w0, x0, ga[r]);
        gb[r] = _mm512_fmadd_ps(w1, x1, gb[r]);
      }
    }
    for (; c < gs; c += 16) {
      size_t off = g * gs + c;
      __m512 x0 = _mm512_loadu_ps(x + off);
      for (int r = 0; r < rows; r++) {
        __m128i bytes = _mm_loadu_si128((const __m128i *)(wq + r * ld + off));
        ga[r] = _mm512_fmadd_ps(_mm512_cvtepi32_ps(_mm512_cvtepi8_epi32(bytes)), x0, ga[r]);
      }
    }
    for (int r = 0; r < rows; r++)
      acc[r] = _mm512_fmadd_ps(_mm512_add_ps(ga[r], gb[r]), _mm512_set1_ps(ws[r * scale_ld + g]), acc[r]);
  }
  for (int r = 0; r < rows; r++)
    out[r] = _mm512_reduce_add_ps(acc[r]);
}

#define Q8_WRAPPERS(rows)                                                                                              \
  AVX512 static void dots_q8_avx512_##rows(const int8_t *wq, const float *ws, size_t n, size_t ld, const int8_t *xq,   \
                                           const float *xs, int gs, float *out) {                                      \
    dots_q8_body(wq, ws, n, ld, rows, xq, xs, gs, out);                                                                \
  }                                                                                                                    \
  AVX512 static void dots_q8f_avx512_##rows(const int8_t *wq, const float *ws, size_t n, size_t ld, const float *x,    \
                                            int gs, float *out) {                                                      \
    dots_q8f_body(wq, ws, n, ld, rows, x, gs, out);                                                                    \
  }
Q8_WRAPPERS(1)
Q8_WRAPPERS(2)
Q8_WRAPPERS(3)
Q8_WRAPPERS(4)
#undef Q8_WRAPPERS

typedef void (*DotsQ8Fn)(const int8_t *, const float *, size_t, size_t, const int8_t *, const float *, int, float *);
typedef void (*DotsQ8fFn)(const int8_t *, const float *, size_t, size_t, const float *, int, float *);
static const DotsQ8Fn dots_q8_avx512[BLOCK + 1] = {NULL, dots_q8_avx512_1, dots_q8_avx512_2, dots_q8_avx512_3,
                                                   dots_q8_avx512_4};
static const DotsQ8fFn dots_q8f_avx512[BLOCK + 1] = {NULL, dots_q8f_avx512_1, dots_q8f_avx512_2, dots_q8f_avx512_3,
                                                     dots_q8f_avx512_4};

/* Bitwise-equal to the scalar quantizer: same max, division and clamp, with
 * roundf's half-away-from-zero rule rebuilt from truncation. gs % 16 == 0. */
AVX512 static void quantize_avx512(int8_t *q, float *s, const float *x, int n, int gs) {
  const __m512 abs_mask = _mm512_castsi512_ps(_mm512_set1_epi32(0x7fffffff));
  const __m512 sign_mask = _mm512_castsi512_ps(_mm512_set1_epi32((int)0x80000000u));
  const __m512 one = _mm512_set1_ps(1.0f), half = _mm512_set1_ps(0.5f);
  const __m512 lo = _mm512_set1_ps(-127.0f), hi = _mm512_set1_ps(127.0f), finite = _mm512_set1_ps(FLT_MAX);
  for (int group = 0; group < n / gs; group++) {
    const float *xg = x + (size_t)group * gs;
    __m512 wmax = _mm512_setzero_ps();
    for (int i = 0; i < gs; i += 16) {
      __m512 a = _mm512_and_ps(_mm512_loadu_ps(xg + i), abs_mask);
      if (_mm512_cmp_ps_mask(a, finite, _CMP_LE_OQ) != 0xFFFF)
        fail("nonfinite activation during INT8 quantization");
      wmax = _mm512_max_ps(wmax, a);
    }
    float scale = _mm512_reduce_max_ps(wmax) / 127.0f;
    s[group] = scale;
    for (int i = 0; i < gs; i += 16) {
      __m512 v = _mm512_setzero_ps();
      if (scale != 0.0f)
        v = _mm512_div_ps(_mm512_loadu_ps(xg + i), _mm512_set1_ps(scale));
      v = _mm512_max_ps(lo, _mm512_min_ps(hi, v));
      __m512 t = _mm512_roundscale_ps(v, _MM_FROUND_TO_ZERO | _MM_FROUND_NO_EXC);
      __mmask16 up = _mm512_cmp_ps_mask(_mm512_and_ps(_mm512_sub_ps(v, t), abs_mask), half, _CMP_GE_OQ);
      t = _mm512_mask_add_ps(t, up, t, _mm512_or_ps(_mm512_and_ps(v, sign_mask), one));
      _mm_storeu_si128((__m128i *)(q + (size_t)group * gs + i), _mm512_cvtepi32_epi8(_mm512_cvttps_epi32(t)));
    }
  }
}

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

/* One GQA group over positions [t0, t1): scores, local max/sum and the
 * unnormalized value sum for each query head. K/V chunks are at most 256
 * positions, so heads after the first re-read them from L2, not memory. */
AVX512 static void attention_chunk_avx512(const float *q, const float *kc, const float *vc, int kv_dim, int head_dim,
                                          int heads, int t0, int t1, float *scores, size_t score_stride,
                                          float *partial, size_t head_stride) {
  enum { MAX_VECTORS = 16 };
  int vectors = (head_dim + 15) / 16;
  __mmask16 last = tail_mask(head_dim - (vectors - 1) * 16);
  float divisor = sqrtf(head_dim);
  const __m512 abs_mask = _mm512_castsi512_ps(_mm512_set1_epi32(0x7fffffff)), finite = _mm512_set1_ps(FLT_MAX);
  for (int h = 0; h < heads; h++) {
    __m512 qv[MAX_VECTORS];
    for (int v = 0; v < vectors; v++)
      qv[v] = _mm512_maskz_loadu_ps(v + 1 == vectors ? last : 0xFFFF, q + (size_t)h * head_dim + v * 16);
    float *a = scores + (size_t)h * score_stride; /* a[t - t0] is the score of position t */
    int t = t0;
    for (; t + 4 <= t1; t += 4) {
      const float *k0 = kc + (size_t)t * kv_dim, *k1 = k0 + kv_dim, *k2 = k1 + kv_dim, *k3 = k2 + kv_dim;
      __m512 s0 = _mm512_setzero_ps(), s1 = s0, s2 = s0, s3 = s0;
      for (int v = 0; v < vectors; v++) {
        __mmask16 m = v + 1 == vectors ? last : 0xFFFF;
        s0 = _mm512_fmadd_ps(qv[v], _mm512_maskz_loadu_ps(m, k0 + v * 16), s0);
        s1 = _mm512_fmadd_ps(qv[v], _mm512_maskz_loadu_ps(m, k1 + v * 16), s1);
        s2 = _mm512_fmadd_ps(qv[v], _mm512_maskz_loadu_ps(m, k2 + v * 16), s2);
        s3 = _mm512_fmadd_ps(qv[v], _mm512_maskz_loadu_ps(m, k3 + v * 16), s3);
      }
      a[t - t0] = _mm512_reduce_add_ps(s0) / divisor;
      a[t - t0 + 1] = _mm512_reduce_add_ps(s1) / divisor;
      a[t - t0 + 2] = _mm512_reduce_add_ps(s2) / divisor;
      a[t - t0 + 3] = _mm512_reduce_add_ps(s3) / divisor;
    }
    for (; t < t1; t++) {
      const float *k0 = kc + (size_t)t * kv_dim;
      __m512 s0 = _mm512_setzero_ps();
      for (int v = 0; v < vectors; v++)
        s0 = _mm512_fmadd_ps(qv[v], _mm512_maskz_loadu_ps(v + 1 == vectors ? last : 0xFFFF, k0 + v * 16), s0);
      a[t - t0] = _mm512_reduce_add_ps(s0) / divisor;
    }
    __m512 vmax = _mm512_set1_ps(-FLT_MAX);
    for (t = t0; t < t1; t += 16) {
      __mmask16 m = tail_mask(t1 - t);
      __m512 s = _mm512_maskz_loadu_ps(m, a + (t - t0));
      if (_mm512_mask_cmp_ps_mask(m, _mm512_and_ps(s, abs_mask), finite, _CMP_LE_OQ) != m)
        fail("nonfinite value in softmax");
      vmax = _mm512_mask_max_ps(vmax, m, vmax, s);
    }
    float max_val = _mm512_reduce_max_ps(vmax);
    __m512 vsum = _mm512_setzero_ps(), shift = _mm512_set1_ps(max_val);
    for (t = t0; t < t1; t += 16) {
      __mmask16 m = tail_mask(t1 - t);
      __m512 e = exp512(_mm512_sub_ps(_mm512_maskz_loadu_ps(m, a + (t - t0)), shift));
      _mm512_mask_storeu_ps(a + (t - t0), m, e);
      vsum = _mm512_mask_add_ps(vsum, m, vsum, e);
    }
    __m512 ov[MAX_VECTORS];
    for (int v = 0; v < vectors; v++)
      ov[v] = _mm512_setzero_ps();
    for (t = t0; t < t1; t++) {
      const float *vr = vc + (size_t)t * kv_dim;
      __m512 p = _mm512_set1_ps(a[t - t0]);
      for (int v = 0; v < vectors; v++)
        ov[v] = _mm512_fmadd_ps(p, _mm512_maskz_loadu_ps(v + 1 == vectors ? last : 0xFFFF, vr + v * 16), ov[v]);
    }
    float *o = partial + (size_t)h * head_stride;
    for (int v = 0; v < vectors; v++)
      _mm512_mask_storeu_ps(o + v * 16, v + 1 == vectors ? last : 0xFFFF, ov[v]);
    o[head_dim] = max_val;
    o[head_dim + 1] = _mm512_reduce_add_ps(vsum);
  }
}

/* ---- E2: tiled batched attention -------------------------------------------------------------------------------------
 * One task = one KV group, one chunk [t0, t1) of positions and a block of `nrows` consecutive rows (row r sees the
 * positions below vis0 + r). The chunk is the same for every row of the block; a row sees only its own prefix of it, and
 * a row with no position in the chunk writes an empty partial. The block's query vectors are its rows x the group's
 * heads (at most 16); K is read once per tile of 32 positions for all of them. */
enum { TILED_POS = 32, TILED_MAX_Q = 16, TILED_MAX_CHUNK = 256, TILED_QG = 8 };

AVX512 static inline __mmask16 lane_mask(int count) {
  return count <= 0 ? (__mmask16)0 : count >= 16 ? (__mmask16)0xFFFF : (__mmask16)((1u << count) - 1u);
}

/* In-place transpose of a 16 x 16 tile held in 16 registers: 64 shuffles (unpack ps, unpack pd, two f32x4 shuffles). */
AVX512 static inline void transpose16(__m512 *r) {
  __m512 t[16], u[16];
  for (int i = 0; i < 8; i++) {
    t[2 * i] = _mm512_unpacklo_ps(r[2 * i], r[2 * i + 1]);
    t[2 * i + 1] = _mm512_unpackhi_ps(r[2 * i], r[2 * i + 1]);
  }
  for (int g = 0; g < 4; g++) {
    __m512d a = _mm512_castps_pd(t[4 * g]), b = _mm512_castps_pd(t[4 * g + 1]), c = _mm512_castps_pd(t[4 * g + 2]),
            d = _mm512_castps_pd(t[4 * g + 3]);
    u[4 * g] = _mm512_castpd_ps(_mm512_unpacklo_pd(a, c));
    u[4 * g + 1] = _mm512_castpd_ps(_mm512_unpackhi_pd(a, c));
    u[4 * g + 2] = _mm512_castpd_ps(_mm512_unpacklo_pd(b, d));
    u[4 * g + 3] = _mm512_castpd_ps(_mm512_unpackhi_pd(b, d));
  }
  for (int k = 0; k < 4; k++) {
    __m512 a0 = _mm512_shuffle_f32x4(u[k], u[4 + k], 0x88), a1 = _mm512_shuffle_f32x4(u[k], u[4 + k], 0xDD);
    __m512 b0 = _mm512_shuffle_f32x4(u[8 + k], u[12 + k], 0x88), b1 = _mm512_shuffle_f32x4(u[8 + k], u[12 + k], 0xDD);
    r[k] = _mm512_shuffle_f32x4(a0, b0, 0x88);
    r[4 + k] = _mm512_shuffle_f32x4(a1, b1, 0x88);
    r[8 + k] = _mm512_shuffle_f32x4(a0, b0, 0xDD);
    r[12 + k] = _mm512_shuffle_f32x4(a1, b1, 0xDD);
  }
}

/* Keys of positions t..t+cnt-1 (cnt <= 32) of one KV head as kt[d][position], so a score tile is a broadcast-FMA over d
 * with no horizontal reduction. Positions at and after cnt are zero; rows past them are never read. */
AVX512 static void tiled_transpose_keys(const float *kc, int kv_dim, int head_dim, int t, int cnt, float *kt) {
  int halves = cnt > 16 ? 2 : 1;
  for (int d0 = 0; d0 < head_dim; d0 += 16) {
    __mmask16 dm = lane_mask(head_dim - d0);
    int dn = head_dim - d0 < 16 ? head_dim - d0 : 16;
    for (int h = 0; h < halves; h++) {
      __m512 r[16];
      for (int p = 0; p < 16; p++)
        r[p] = h * 16 + p < cnt ? _mm512_maskz_loadu_ps(dm, kc + (size_t)(t + h * 16 + p) * kv_dim + d0)
                                : _mm512_setzero_ps();
      transpose16(r);
      for (int i = 0; i < dn; i++)
        _mm512_storeu_ps(kt + (size_t)(d0 + i) * TILED_POS + h * 16, r[i]);
    }
  }
}

/* Scores of TILED_QG query vectors (qd[d * 16 + j] is element d of vector j, already scaled: the 16 vectors of the task
 * are interleaved per d so the broadcast operands of one d sit in one cache line) against the transposed tile: 8 x 2
 * accumulators, per d two loads of kt and eight broadcast operands for 16 FMAs. `nv` is 1 or 2 position vectors. */
AVX512 static inline void tiled_scores8(const float *kt, const float *qd, int head_dim, int nv, float *out,
                                        size_t ostride, int cnt) {
  __m512 a[TILED_QG][2];
  for (int j = 0; j < TILED_QG; j++)
    a[j][0] = a[j][1] = _mm512_setzero_ps();
  if (nv == 2) {
    for (int d = 0; d < head_dim; d++, kt += TILED_POS, qd += TILED_MAX_Q) {
      __m512 k0 = _mm512_loadu_ps(kt), k1 = _mm512_loadu_ps(kt + 16);
      for (int j = 0; j < TILED_QG; j++) {
        __m512 b = _mm512_set1_ps(qd[j]);
        a[j][0] = _mm512_fmadd_ps(b, k0, a[j][0]);
        a[j][1] = _mm512_fmadd_ps(b, k1, a[j][1]);
      }
    }
  } else {
    for (int d = 0; d < head_dim; d++, kt += TILED_POS, qd += TILED_MAX_Q) {
      __m512 k0 = _mm512_loadu_ps(kt);
      for (int j = 0; j < TILED_QG; j++)
        a[j][0] = _mm512_fmadd_ps(_mm512_set1_ps(qd[j]), k0, a[j][0]);
    }
  }
  for (int j = 0; j < TILED_QG; j++) {
    _mm512_mask_storeu_ps(out + j * ostride, lane_mask(cnt), a[j][0]);
    if (nv == 2)
      _mm512_mask_storeu_ps(out + j * ostride + 16, lane_mask(cnt - 16), a[j][1]);
  }
}

/* P x V for TILED_QG query vectors and one slice of `dv` (1 or 2) vectors of the head: the accumulators come from the
 * task's output scratch (rescaled by alpha when the running maximum moved), one V row load feeds 8 x dv FMAs whose
 * broadcast operand is a probability in L1, and they go back to the scratch. 16 accumulators, two V registers. */
AVX512 static inline __attribute__((always_inline)) void tiled_pv(const float *p8, const float *v, int kv_dim, int npos,
                                                                   float *o8, size_t ostride, const float *alpha8,
                                                                   __mmask16 m0, __mmask16 m1, int dv) {
  __m512 a[TILED_QG][2];
  for (int j = 0; j < TILED_QG; j++) {
    a[j][0] = _mm512_loadu_ps(o8 + j * ostride);
    a[j][1] = dv == 2 ? _mm512_loadu_ps(o8 + j * ostride + 16) : _mm512_setzero_ps();
    if (alpha8) {
      __m512 alpha = _mm512_set1_ps(alpha8[j]);
      a[j][0] = _mm512_mul_ps(a[j][0], alpha);
      if (dv == 2)
        a[j][1] = _mm512_mul_ps(a[j][1], alpha);
    }
  }
  for (int pos = 0; pos < npos; pos++) {
    const float *vr = v + (size_t)pos * kv_dim;
    __m512 v0 = _mm512_maskz_loadu_ps(m0, vr), v1 = dv == 2 ? _mm512_maskz_loadu_ps(m1, vr + 16) : _mm512_setzero_ps();
    for (int j = 0; j < TILED_QG; j++) {
      __m512 b = _mm512_set1_ps(p8[j * TILED_POS + pos]);
      a[j][0] = _mm512_fmadd_ps(b, v0, a[j][0]);
      if (dv == 2)
        a[j][1] = _mm512_fmadd_ps(b, v1, a[j][1]);
    }
  }
  for (int j = 0; j < TILED_QG; j++) {
    _mm512_storeu_ps(o8 + j * ostride, a[j][0]);
    if (dv == 2)
      _mm512_storeu_ps(o8 + j * ostride + 16, a[j][1]);
  }
}

/* The task. Per tile of 32 positions: transpose K once; then per group of 8 query vectors scores (registers to an 8 x 32
 * L1 tile), online softmax (running maximum m and sum l per query vector; when the maximum moves, the accumulator and l
 * are rescaled by exp(m_old - m_new)), probabilities to an L1 tile, and PV into the output scratch. Nothing but K, V
 * and Q is read from beyond L1, and no score is ever written to memory the task does not own. Positions a row may not
 * see get probability exactly zero (the lanes are masked, not given a large negative score). */
AVX512 static void attention_tiled_avx512(const float *q, size_t q_row_stride, int nrows, int kv_mul, int head_dim,
                                          int vis0, int t0, int t1, const float *kc, const float *vc, int kv_dim,
                                          float *partial, size_t row_stride, size_t head_stride) {
  _Alignas(64) float qs[TILED_MAX_Q * 256], kt[256 * TILED_POS], os[TILED_MAX_Q * 256], ps[TILED_MAX_Q * TILED_POS],
      sc[TILED_QG * TILED_POS];
  float m[TILED_MAX_Q], l[TILED_MAX_Q], alpha[TILED_MAX_Q], lane[16];
  int vectors = (head_dim + 15) / 16, hdp = vectors * 16, nq = nrows * kv_mul;
  int nqp = (nq + TILED_QG - 1) / TILED_QG * TILED_QG;
  __mmask16 last = lane_mask(head_dim - (vectors - 1) * 16);
  /* Q for the block, scaled by 1/sqrt(head_dim) once (no division per score) and transposed to qs[d][query vector], with
   * zero vectors up to 16 (a 16 x 16 transpose per 16 elements of the head). */
  const __m512 scale = _mm512_set1_ps(1.0f / sqrtf((float)head_dim));
  for (int d0 = 0; d0 < hdp; d0 += 16) {
    __mmask16 dm = lane_mask(head_dim - d0);
    __m512 r[16];
    for (int qi = 0; qi < TILED_MAX_Q; qi++)
      r[qi] = qi < nq ? _mm512_mul_ps(_mm512_maskz_loadu_ps(dm, q + (size_t)(qi / kv_mul) * q_row_stride +
                                                                    (size_t)(qi % kv_mul) * head_dim + d0),
                                      scale)
                      : _mm512_setzero_ps();
    transpose16(r);
    for (int c = 0; c < 16; c++)
      _mm512_storeu_ps(qs + (size_t)(d0 + c) * TILED_MAX_Q, r[c]);
  }
  for (int qi = 0; qi < nqp; qi++)
    for (int v = 0; v < vectors; v++)
      _mm512_storeu_ps(os + (size_t)qi * hdp + v * 16, _mm512_setzero_ps());
  for (int qi = 0; qi < TILED_MAX_Q; qi++) {
    m[qi] = -FLT_MAX, l[qi] = 0.0f, alpha[qi] = 1.0f;
    for (int i = 0; i < TILED_POS; i += 16)
      _mm512_storeu_ps(ps + qi * TILED_POS + i, _mm512_setzero_ps());
  }
  const __m512 abs_mask = _mm512_castsi512_ps(_mm512_set1_epi32(0x7fffffff)), finite = _mm512_set1_ps(FLT_MAX);
  int tend = t1 < vis0 + nrows - 1 ? t1 : vis0 + nrows - 1; /* beyond the last row's prefix nothing is read */
  for (int t = t0; t < tend; t += TILED_POS) {
    int cnt = tend - t < TILED_POS ? tend - t : TILED_POS;
    tiled_transpose_keys(kc, kv_dim, head_dim, t, cnt, kt);
    for (int q0 = 0; q0 < nqp; q0 += TILED_QG) {
      /* the last real query vector of the group sees the most positions of this tile */
      int qlast = q0 + TILED_QG - 1 < nq ? q0 + TILED_QG - 1 : nq - 1, rlast = qlast / kv_mul;
      int gn = (vis0 + rlast < t + cnt ? vis0 + rlast : t + cnt) - t;
      if (gn <= 0)
        continue; /* no query vector of the group sees any position of this tile: nothing to add */
      tiled_scores8(kt, qs + q0, head_dim, gn > 16 ? 2 : 1, sc, TILED_POS, gn);
      int rescale = 0;
      for (int j = 0; j < TILED_QG; j++) {
        int qi = q0 + j, n = qi < nq ? (vis0 + qi / kv_mul < t + cnt ? vis0 + qi / kv_mul : t + cnt) - t : 0;
        float *p = ps + (size_t)qi * TILED_POS;
        alpha[qi] = 1.0f;
        if (n <= 0) { /* a row above the diagonal of this tile, or a padding row */
          _mm512_storeu_ps(p, _mm512_setzero_ps());
          _mm512_storeu_ps(p + 16, _mm512_setzero_ps());
          continue;
        }
        __mmask16 k0 = lane_mask(n), k1 = lane_mask(n - 16);
        __m512 s0 = _mm512_maskz_loadu_ps(k0, sc + j * TILED_POS), s1 = _mm512_maskz_loadu_ps(k1, sc + j * TILED_POS + 16);
        if ((_mm512_mask_cmp_ps_mask(k0, _mm512_and_ps(s0, abs_mask), finite, _CMP_LE_OQ) != k0) ||
            (_mm512_mask_cmp_ps_mask(k1, _mm512_and_ps(s1, abs_mask), finite, _CMP_LE_OQ) != k1))
          fail("nonfinite value in softmax");
        __m512 vmax = _mm512_set1_ps(-FLT_MAX);
        vmax = _mm512_mask_max_ps(vmax, k0, vmax, s0);
        vmax = _mm512_mask_max_ps(vmax, k1, vmax, s1);
        float tile_max = _mm512_reduce_max_ps(vmax), old = m[qi];
        if (tile_max > old) { /* the maximum moved: what is accumulated so far is scaled by exp(old - new) */
          _mm512_storeu_ps(lane, exp512(_mm512_set1_ps(old - tile_max)));
          alpha[qi] = lane[0];
          l[qi] *= lane[0];
          m[qi] = tile_max;
          rescale = 1;
        }
        __m512 shift = _mm512_set1_ps(m[qi]);
        __m512 e0 = _mm512_maskz_mov_ps(k0, exp512(_mm512_sub_ps(s0, shift)));
        __m512 e1 = _mm512_maskz_mov_ps(k1, exp512(_mm512_sub_ps(s1, shift)));
        _mm512_storeu_ps(p, e0);
        _mm512_storeu_ps(p + 16, e1);
        l[qi] += _mm512_reduce_add_ps(_mm512_add_ps(e0, e1));
      }
      for (int s = 0; s < vectors; s += 2) { /* slices of two vectors of the head */
        __mmask16 m0 = s + 1 == vectors ? last : 0xFFFF, m1 = s + 2 == vectors ? last : 0xFFFF;
        const float *pq = ps + (size_t)q0 * TILED_POS, *v = vc + (size_t)t * kv_dim + s * 16;
        float *o8 = os + (size_t)q0 * hdp + s * 16;
        if (s + 1 < vectors)
          tiled_pv(pq, v, kv_dim, gn, o8, (size_t)hdp, rescale ? alpha + q0 : NULL, m0, m1, 2);
        else
          tiled_pv(pq, v, kv_dim, gn, o8, (size_t)hdp, rescale ? alpha + q0 : NULL, m0, m1, 1);
      }
    }
  }
  for (int qi = 0; qi < nq; qi++) {
    int r = qi / kv_mul, h = qi % kv_mul, end = t1 < vis0 + r ? t1 : vis0 + r;
    float *o = partial + (size_t)r * row_stride + (size_t)h * head_stride;
    if (end <= t0) {
      partial_write_empty(o, head_dim);
      continue;
    }
    for (int v = 0; v < vectors; v++)
      _mm512_mask_storeu_ps(o + v * 16, v + 1 == vectors ? last : 0xFFFF,
                            _mm512_loadu_ps(os + (size_t)qi * hdp + v * 16));
    o[head_dim] = m[qi];
    o[head_dim + 1] = l[qi];
  }
}

// ----------------------------------------------------------------------------
// Group decode attention (E5)

/* Single-row attention for one KV group over one chunk [t0, t1) of positions: the group's `nq` query heads (at most 16)
 * are handled together, so each K and V row is read from memory once per group. Scores of 16 positions are computed
 * as 16 partial vectors (one per position, accumulated over the head with the pre-scaled Q) and reduced by one 16-way
 * transpose-add: no horizontal reduction and no division per score. Softmax is online per tile of 32 positions (running
 * maximum m and sum l per head, accumulator and l rescaled by exp(m_old - m_new)) and PV is fused into the same pass.
 * The result is a split-K partial [acc(head_dim), max, sum] per head, merged by merge_chunks like the split kernel's.
 */
enum { GROUP_POS = 32, GROUP_MAX_Q = 16, GROUP_MAX_HD = 256 };

/* Lane j of the result is the sum of the 16 lanes of a[j]: 30 shuffles and 15 adds for 16 sums (a full transpose takes
 * 64 shuffles). Levels: float unpack within 128-bit blocks, double unpack, then the four 128-bit blocks. */
AVX512 static inline __m512 group_reduce16(const __m512 *a) {
  __m512 s[8], u[4];
  for (int i = 0; i < 8; i++)
    s[i] = _mm512_add_ps(_mm512_unpacklo_ps(a[2 * i], a[2 * i + 1]), _mm512_unpackhi_ps(a[2 * i], a[2 * i + 1]));
  for (int i = 0; i < 4; i++) {
    __m512d x = _mm512_castps_pd(s[2 * i]), y = _mm512_castps_pd(s[2 * i + 1]);
    u[i] = _mm512_add_ps(_mm512_castpd_ps(_mm512_unpacklo_pd(x, y)), _mm512_castpd_ps(_mm512_unpackhi_pd(x, y)));
  }
  __m512 t0 = _mm512_add_ps(_mm512_shuffle_f32x4(u[0], u[1], 0x88), _mm512_shuffle_f32x4(u[0], u[1], 0xDD));
  __m512 t1 = _mm512_add_ps(_mm512_shuffle_f32x4(u[2], u[3], 0x88), _mm512_shuffle_f32x4(u[2], u[3], 0xDD));
  return _mm512_add_ps(_mm512_shuffle_f32x4(t0, t1, 0x88), _mm512_shuffle_f32x4(t0, t1, 0xDD));
}

/* Scores (already scaled through q) of 16 positions whose key rows are kp[0..15]; lane p is position p. */
AVX512 static inline __attribute__((always_inline)) __m512 group_scores16(const float *q, const float *const *kp,
                                                                          int vectors, __mmask16 last) {
  __m512 a[16];
  for (int p = 0; p < 16; p++)
    a[p] = _mm512_setzero_ps();
  for (int v = 0; v + 1 < vectors; v++) {
    __m512 qv = _mm512_loadu_ps(q + v * 16);
    for (int p = 0; p < 16; p++)
      a[p] = _mm512_fmadd_ps(qv, _mm512_loadu_ps(kp[p] + v * 16), a[p]);
  }
  __m512 qv = _mm512_loadu_ps(q + (vectors - 1) * 16);
  for (int p = 0; p < 16; p++)
    a[p] = _mm512_fmadd_ps(qv, _mm512_maskz_loadu_ps(last, kp[p] + (vectors - 1) * 16), a[p]);
  return group_reduce16(a);
}

/* P x V for NH query heads and a slice of up to DV vectors of the head (dv of them used): accumulators from the task's
 * output scratch (rescaled by alpha), one V row load feeds NH x dv FMAs with a probability broadcast, and they go back.
 */
AVX512 static inline __attribute__((always_inline)) void group_pv(const float *p, const float *v, int kv_dim, int npos,
                                                                  float *o, size_t ostride, const float *alpha,
                                                                  __mmask16 mlast, int dv, const int NH, const int DV) {
  __m512 a[8][8];
  __mmask16 mk[8];
  for (int x = 0; x < DV; x++)
    mk[x] = x + 1 == dv ? mlast : (__mmask16)0xFFFF;
  for (int j = 0; j < NH; j++) {
    __m512 al = _mm512_set1_ps(alpha[j]);
    for (int x = 0; x < DV; x++)
      a[j][x] = x < dv ? _mm512_mul_ps(_mm512_loadu_ps(o + j * ostride + x * 16), al) : _mm512_setzero_ps();
  }
  for (int pos = 0; pos < npos; pos++) {
    const float *vr = v + (size_t)pos * kv_dim;
    __m512 vv[8];
    for (int x = 0; x < DV; x++)
      vv[x] = x < dv ? _mm512_maskz_loadu_ps(mk[x], vr + x * 16) : _mm512_setzero_ps();
    for (int j = 0; j < NH; j++) {
      __m512 b = _mm512_set1_ps(p[j * GROUP_POS + pos]);
      for (int x = 0; x < DV; x++)
        a[j][x] = _mm512_fmadd_ps(b, vv[x], a[j][x]);
    }
  }
  for (int j = 0; j < NH; j++)
    for (int x = 0; x < DV; x++)
      if (x < dv)
        _mm512_storeu_ps(o + j * ostride + x * 16, a[j][x]);
}

AVX512 static void attention_group_avx512(const float *q, int nq, int head_dim, int t0, int t1, const float *kc,
                                          const float *vc, int kv_dim, float *partial, size_t head_stride) {
  _Alignas(64) float qs[GROUP_MAX_Q * GROUP_MAX_HD], os[GROUP_MAX_Q * GROUP_MAX_HD], ps[GROUP_MAX_Q * GROUP_POS],
      zero[GROUP_MAX_HD];
  float m[GROUP_MAX_Q], l[GROUP_MAX_Q], alpha[GROUP_MAX_Q], lane[16];
  int vectors = (head_dim + 15) / 16, hdp = vectors * 16;
  __mmask16 last = lane_mask(head_dim - (vectors - 1) * 16);
  const __m512 scale = _mm512_set1_ps(1.0f / sqrtf((float)head_dim));
  /* Q scaled once by 1/sqrt(head_dim); the tail of the last vector is zero so it never meets a stray key element. */
  for (int h = 0; h < nq; h++) {
    for (int v = 0; v < vectors; v++) {
      __mmask16 mk = v + 1 == vectors ? last : (__mmask16)0xFFFF;
      _mm512_storeu_ps(qs + (size_t)h * hdp + v * 16,
                       _mm512_mul_ps(_mm512_maskz_loadu_ps(mk, q + (size_t)h * head_dim + v * 16), scale));
      _mm512_storeu_ps(os + (size_t)h * hdp + v * 16, _mm512_setzero_ps());
    }
    m[h] = -FLT_MAX, l[h] = 0.0f, alpha[h] = 1.0f;
  }
  for (int v = 0; v < hdp; v += 16)
    _mm512_storeu_ps(zero + v, _mm512_setzero_ps());
  for (int i = 0; i < GROUP_MAX_Q * GROUP_POS; i += 16)
    _mm512_storeu_ps(ps + i, _mm512_setzero_ps());
  const __m512 abs_mask = _mm512_castsi512_ps(_mm512_set1_epi32(0x7fffffff)), finite = _mm512_set1_ps(FLT_MAX);
  for (int t = t0; t < t1; t += GROUP_POS) {
    int cnt = t1 - t < GROUP_POS ? t1 - t : GROUP_POS;
    for (int p = 0; p < cnt; p++) /* bring the tile's V rows in while the scores are computed */
      for (int d = 0; d < head_dim; d += 16)
        __builtin_prefetch(vc + (size_t)(t + p) * kv_dim + d, 0, 3);
    const float *kp[2][16];
    for (int p = 0; p < GROUP_POS; p++)
      kp[p / 16][p % 16] = p < cnt ? kc + (size_t)(t + p) * kv_dim : zero;
    __mmask16 k0 = lane_mask(cnt), k1 = lane_mask(cnt - 16);
    for (int h = 0; h < nq; h++) {
      const float *qh = qs + (size_t)h * hdp;
      __m512 s0 = group_scores16(qh, kp[0], vectors, last);
      __m512 s1 = cnt > 16 ? group_scores16(qh, kp[1], vectors, last) : _mm512_setzero_ps();
      if ((_mm512_mask_cmp_ps_mask(k0, _mm512_and_ps(s0, abs_mask), finite, _CMP_LE_OQ) != k0) ||
          (_mm512_mask_cmp_ps_mask(k1, _mm512_and_ps(s1, abs_mask), finite, _CMP_LE_OQ) != k1))
        fail("nonfinite value in softmax");
      __m512 vmax = _mm512_set1_ps(-FLT_MAX);
      vmax = _mm512_mask_max_ps(vmax, k0, vmax, s0);
      vmax = _mm512_mask_max_ps(vmax, k1, vmax, s1);
      float tile_max = _mm512_reduce_max_ps(vmax), old = m[h];
      alpha[h] = 1.0f;      /* 1 when the maximum did not move: one code path serves both cases */
      if (tile_max > old) { /* the maximum moved: what is accumulated so far is scaled by exp(old - new) */
        _mm512_storeu_ps(lane, exp512(_mm512_set1_ps(old - tile_max)));
        alpha[h] = lane[0];
        l[h] *= lane[0];
        m[h] = tile_max;
      }
      __m512 shift = _mm512_set1_ps(m[h]);
      __m512 e0 = _mm512_maskz_mov_ps(k0, exp512(_mm512_sub_ps(s0, shift)));
      __m512 e1 = _mm512_maskz_mov_ps(k1, exp512(_mm512_sub_ps(s1, shift)));
      _mm512_storeu_ps(ps + h * GROUP_POS, e0);
      _mm512_storeu_ps(ps + h * GROUP_POS + 16, e1);
      l[h] += _mm512_reduce_add_ps(_mm512_add_ps(e0, e1));
    }
    for (int h0 = 0; h0 < nq;) {
      int left = nq - h0, nh = left >= 8 ? 8 : left >= 4 ? 4 : left >= 2 ? 2 : 1;
      int dvmax = nh == 8 ? 2 : nh == 4 ? 4 : 8;
      for (int s = 0; s < vectors; s += dvmax) {
        int dv = vectors - s < dvmax ? vectors - s : dvmax;
        __mmask16 ml = s + dv == vectors ? last : (__mmask16)0xFFFF;
        const float *pp = ps + (size_t)h0 * GROUP_POS, *v = vc + (size_t)t * kv_dim + s * 16, *al = alpha + h0;
        float *o = os + (size_t)h0 * hdp + s * 16;
        switch (nh) {
        case 8:
          group_pv(pp, v, kv_dim, cnt, o, (size_t)hdp, al, ml, dv, 8, 2);
          break;
        case 4:
          group_pv(pp, v, kv_dim, cnt, o, (size_t)hdp, al, ml, dv, 4, 4);
          break;
        case 2:
          group_pv(pp, v, kv_dim, cnt, o, (size_t)hdp, al, ml, dv, 2, 8);
          break;
        default:
          group_pv(pp, v, kv_dim, cnt, o, (size_t)hdp, al, ml, dv, 1, 8);
          break;
        }
      }
      h0 += nh;
    }
  }
  for (int h = 0; h < nq; h++) {
    float *o = partial + (size_t)h * head_stride;
    if (t1 <= t0) {
      partial_write_empty(o, head_dim);
      continue;
    }
    for (int v = 0; v < vectors; v++)
      _mm512_mask_storeu_ps(o + v * 16, v + 1 == vectors ? last : (__mmask16)0xFFFF,
                            _mm512_loadu_ps(os + (size_t)h * hdp + v * 16));
    o[head_dim] = m[h];
    o[head_dim + 1] = l[h];
  }
}
#endif

// ----------------------------------------------------------------------------
// Normalization and RoPE

void k_rmsnorm(float *out, const float *x, const float *weight, int size, float epsilon) {
#if HAVE_AVX512
  if (isa == KERNELS_AVX512) {
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
  if (isa == KERNELS_AVX512) {
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
// GEMV drivers

/* PARTIAL stores like STORE, into a socket-local partial buffer that the cross-socket reduction adds to the residual
 * exactly once; it stays a separate kind so call sites state their intent. */
enum { STORE = K_STORE, ADD = K_ADD, PARTIAL = K_PARTIAL, SWIGLU = K_SWIGLU };

static float silu_gate(float value, float gate) {
  // silu(x)=x*σ(x), where σ(x) is the logistic sigmoid; elementwise multiply with w3(x)
  value *= (1.0f / (1.0f + expf(-value)));
  return value * gate;
}

static void f32_dots(const float *w, size_t ld, int rows, const float *x, int n, float *out) {
#if HAVE_AVX512
  if (isa == KERNELS_AVX512) {
    dots_f32_avx512(w, ld, rows, x, n, out);
    return;
  }
#endif
  for (int r = 0; r < rows; r++) {
    const float *wr = w + (size_t)r * ld;
    float val = 0.0f;
    for (int j = 0; j < n; j++) {
      val += wr[j] * x[j];
    }
    out[r] = val;
  }
}

/* Four rows against x over the packed4 layout; the scalar form sums each row in ascending column order, as f32_dots does. */
static void f32_dots_p4(const float *p, int n, const float *x, float *out) {
#if HAVE_AVX512
  if (isa == KERNELS_AVX512) {
    dots_p4_avx512(p, x, n, out);
    return;
  }
#endif
  for (int r = 0; r < BLOCK; r++) {
    float val = 0.0f;
    for (int j = 0; j < n; j++)
      val += p[(size_t)(j / 16) * 64 + (size_t)r * 16 + (size_t)(j % 16)] * x[j];
    out[r] = val;
  }
}

typedef struct {
  float *out;
  const float *w, *w3;
  int d;
} F32Segment;
typedef struct {
  const float *x;
  int n, kind, count;
  F32Segment segment[3];
  size_t ld; /* weight row stride in floats; 0 = n */
  atomic_int next; /* dynamic scheduling: the next unclaimed row (zero before the team starts) */
  RowsDoneFn done; /* worker entry points only (one segment): called after each range of rows is stored */
  void *done_ctx;
  int packed; /* the segments' weights are in the packed4 layout (k_pack4): every row range is whole four-row groups */
} F32Job;

static void f32_range(const F32Job *job, const F32Segment *seg, int r0, int r1) {
  float dots[BLOCK], gates[BLOCK];
  size_t ld = job->ld ? job->ld : (size_t)job->n;
  const size_t group = (size_t)((job->n + 15) / 16) * 64; /* floats of one packed four-row group */
  for (int i = r0; i < r1; i += BLOCK) {
    int rows = r1 - i < BLOCK ? r1 - i : BLOCK;
    if (job->packed) {
      if (rows != BLOCK || i % BLOCK)
        fail("packed4 GEMV: rows [%d, %d) are not whole four-row groups", i, r1);
      f32_dots_p4(seg->w + (size_t)(i / BLOCK) * group, job->n, job->x, dots);
      if (job->kind == SWIGLU)
        f32_dots_p4(seg->w3 + (size_t)(i / BLOCK) * group, job->n, job->x, gates);
    } else {
      f32_dots(seg->w + (size_t)i * ld, ld, rows, job->x, job->n, dots);
      if (job->kind == SWIGLU)
        f32_dots(seg->w3 + (size_t)i * ld, ld, rows, job->x, job->n, gates);
    }
    for (int r = 0; r < rows; r++) {
      if (job->kind == ADD)
        seg->out[i + r] += dots[r];
      else if (job->kind == SWIGLU)
        seg->out[i + r] = silu_gate(dots[r], gates[r]);
      else
        seg->out[i + r] = dots[r];
    }
  }
  if (job->done)
    job->done(job->done_ctx, r0, r1);
}

static void f32_rows(const void *p, int r0, int r1) {
  const F32Job *job = p;
  int base = 0;
  for (int s = 0; s < job->count; s++) {
    const F32Segment *seg = &job->segment[s];
    int lo = r0 > base ? r0 - base : 0, hi = r1 - base < seg->d ? r1 - base : seg->d;
    if (hi > lo)
      f32_range(job, seg, lo, hi);
    base += seg->d;
  }
}

/* Static split: every segment on its own, so one member's range never straddles Q, K and V. */
static void f32_team_static(const F32Job *job, int worker, int workers) {
  for (int s = 0; s < job->count; s++) {
    const F32Segment *seg = &job->segment[s];
    int r0 = (int)((long long)seg->d * worker / workers), r1 = (int)((long long)seg->d * (worker + 1) / workers);
    if (job->packed) { /* shares of whole four-row groups */
      int groups = seg->d / BLOCK;
      r0 = (int)((long long)groups * worker / workers) * BLOCK;
      r1 = (int)((long long)groups * (worker + 1) / workers) * BLOCK;
    }
    if (r1 > r0)
      f32_range(job, seg, r0, r1);
  }
}

/* Dynamic and hybrid splits over the concatenated rows of all segments; `next` is shared by the members of one job.
 * `range(job, r0, r1)` runs rows [r0, r1) of that concatenation. */
typedef void (*RangeFn)(const void *job, int r0, int r1);

static void claim_rows_run(RangeFn range, const void *job, int rows, atomic_int *next, int worker, int workers,
                           GemvSched sched) {
  int first = 0, chunk = claim_rows(rows, workers);
  if (sched == SCHED_HYBRID) {
    int pool = rows * 3 / 4 / BLOCK * BLOCK; /* the contiguous shares */
    int r0 = (int)((long long)pool * worker / workers) / BLOCK * BLOCK;
    int r1 = worker + 1 == workers ? pool : (int)((long long)pool * (worker + 1) / workers) / BLOCK * BLOCK;
    if (r1 > r0)
      range(job, r0, r1);
    first = pool;
    chunk = (rows - pool) / (workers * 2) / BLOCK * BLOCK;
    chunk = chunk < BLOCK ? BLOCK : chunk > 16 ? 16 : chunk;
  }
  for (;;) {
    int r0 = first + atomic_fetch_add_explicit(next, chunk, memory_order_relaxed);
    if (r0 >= rows)
      return;
    range(job, r0, r0 + chunk < rows ? r0 + chunk : rows);
  }
}

static int job_rows_f32(const F32Job *job) {
  int rows = 0;
  for (int s = 0; s < job->count; s++)
    rows += job->segment[s].d;
  return rows;
}

static size_t job_bytes_f32(const F32Job *job) {
  size_t row = job->packed ? (size_t)((job->n + 15) / 16) * 16 : (job->ld ? job->ld : (size_t)job->n);
  return (size_t)job_rows_f32(job) * row * sizeof(float) * (job->kind == SWIGLU ? 2 : 1);
}

static void f32_team_claim(const F32Job *job, atomic_int *next, int worker, int workers, GemvSched sched) {
  claim_rows_run(f32_rows, job, job_rows_f32(job), next, worker, workers, sched);
}

static void f32_team(const void *p, int worker, int workers) {
  F32Job *job = (F32Job *)p; /* the caller's job object is writable: only `next` changes */
  GemvSched sched = effective_f32(job_bytes_f32(job));
  if (sched != SCHED_STATIC && workers > 1)
    f32_team_claim(job, &job->next, worker, workers, sched);
  else
    f32_team_static(job, worker, workers);
}

/* Rows [4 * first, 4 * last) of the concatenated segments: the packed4 layout splits only between whole four-row groups. */
static void f32_group_rows(const void *p, int first, int last) {
  f32_rows(p, first * BLOCK, last * BLOCK);
}

static void f32_run(F32Job *job) {
  if (current_team) {
    team_run(current_team, f32_team, job);
    return;
  }
  int rows = 0;
  for (int s = 0; s < job->count; s++)
    rows += job->segment[s].d;
  if (job->packed)
    parallel_rows(rows / BLOCK, f32_group_rows, job); /* every segment has a multiple of four rows */
  else
    parallel_rows(rows, f32_rows, job);
}

void k_matmul(float *out, const float *x, const float *w, int n, int d) {
  // W (d,n) @ x (n,) -> xout (d,)
  F32Job job = {x, n, STORE, 1, {{out, w, NULL, d}}, 0, 0, NULL, NULL, 0};
  f32_run(&job);
}
void k_matmul_add(float *out, const float *x, const float *w, int n, int d) {
  F32Job job = {x, n, ADD, 1, {{out, w, NULL, d}}, 0, 0, NULL, NULL, 0};
  f32_run(&job);
}
void k_matmul_qkv(float *q, float *k, float *v, const float *x, const float *wq, const float *wk, const float *wv, int n,
                  int dq, int dkv) {
  F32Job job = {x, n, STORE, 3, {{q, wq, NULL, dq}, {k, wk, NULL, dkv}, {v, wv, NULL, dkv}}, 0, 0, NULL, NULL, 0};
  f32_run(&job);
}
void k_matmul_swiglu(float *out, const float *x, const float *w1, const float *w3, int n, int d) {
  F32Job job = {x, n, SWIGLU, 1, {{out, w1, w3, d}}, 0, 0, NULL, NULL, 0};
  f32_run(&job);
}

// ----------------------------------------------------------------------------
// Q8_0 drivers

static void q8_dots(K8Weight w, size_t ld, int row, int rows, const K8Input *x, int n, int gs, float *out) {
  size_t scale_ld = ld / gs;
  const int8_t *wq = w.q + (size_t)row * ld;
  const float *ws = w.s + (size_t)row * scale_ld;
#if HAVE_AVX512
  if (isa == KERNELS_AVX512) {
    if (x->q && gs % 64 == 0) {
      dots_q8_avx512[rows](wq, ws, (size_t)n, ld, x->q, x->s, gs, out);
      return;
    }
    if (!x->q && gs % 16 == 0) {
      dots_q8f_avx512[rows](wq, ws, (size_t)n, ld, x->f, gs, out);
      return;
    }
  }
#endif
  for (int r = 0; r < rows; r++) {
    const int8_t *wr = wq + (size_t)r * ld;
    const float *sr = ws + (size_t)r * scale_ld;
    float val = 0.0f;
    // do the matmul in groups of GS
    for (int j = 0; j <= n - gs; j += gs) {
      if (x->q) {
        int32_t ival = 0;
        for (int k = 0; k < gs; k++) {
          ival += ((int32_t)x->q[j + k]) * ((int32_t)wr[j + k]);
        }
        val += ((float)ival) * sr[j / gs] * x->s[j / gs];
      } else {
        float group = 0.0f;
        for (int k = 0; k < gs; k++) {
          group += (float)wr[j + k] * x->f[j + k];
        }
        val += group * sr[j / gs];
      }
    }
    out[r] = val;
  }
}

typedef struct {
  float *out;
  K8Weight w, w3;
  int d;
} Q8Segment;
typedef struct {
  const K8Input *x;
  int n, gs, kind, count;
  Q8Segment segment[3];
  size_t ld; /* weight row stride in values (a multiple of gs); 0 = n */
  atomic_int next;
  RowsDoneFn done;
  void *done_ctx;
} Q8Job;

static void q8_range(const Q8Job *job, const Q8Segment *seg, int r0, int r1) {
  float dots[BLOCK], gates[BLOCK];
  size_t ld = job->ld ? job->ld : (size_t)job->n;
  for (int i = r0; i < r1; i += BLOCK) {
    int rows = r1 - i < BLOCK ? r1 - i : BLOCK;
    q8_dots(seg->w, ld, i, rows, job->x, job->n, job->gs, dots);
    if (job->kind == SWIGLU)
      q8_dots(seg->w3, ld, i, rows, job->x, job->n, job->gs, gates);
    for (int r = 0; r < rows; r++) {
      if (job->kind == ADD)
        seg->out[i + r] += dots[r];
      else if (job->kind == SWIGLU)
        seg->out[i + r] = silu_gate(dots[r], gates[r]);
      else
        seg->out[i + r] = dots[r];
    }
  }
  if (job->done)
    job->done(job->done_ctx, r0, r1);
}

static void q8_rows(const void *p, int r0, int r1) {
  const Q8Job *job = p;
  int base = 0;
  for (int s = 0; s < job->count; s++) {
    const Q8Segment *seg = &job->segment[s];
    int lo = r0 > base ? r0 - base : 0, hi = r1 - base < seg->d ? r1 - base : seg->d;
    if (hi > lo)
      q8_range(job, seg, lo, hi);
    base += seg->d;
  }
}

static void q8_team_static(const Q8Job *job, int worker, int workers) {
  for (int s = 0; s < job->count; s++) {
    const Q8Segment *seg = &job->segment[s];
    int r0 = (int)((long long)seg->d * worker / workers), r1 = (int)((long long)seg->d * (worker + 1) / workers);
    if (r1 > r0)
      q8_range(job, seg, r0, r1);
  }
}

static void q8_team_claim(const Q8Job *job, atomic_int *next, int worker, int workers, GemvSched sched) {
  int rows = 0;
  for (int s = 0; s < job->count; s++)
    rows += job->segment[s].d;
  claim_rows_run(q8_rows, job, rows, next, worker, workers, sched);
}

static void q8_team(const void *p, int worker, int workers) {
  Q8Job *job = (Q8Job *)p;
  if (gemv_sched_q8 != SCHED_STATIC && workers > 1)
    q8_team_claim(job, &job->next, worker, workers, gemv_sched_q8);
  else
    q8_team_static(job, worker, workers);
}

static void q8_run(Q8Job *job) {
  if (current_team) {
    team_run(current_team, q8_team, job);
    return;
  }
  int rows = 0;
  for (int s = 0; s < job->count; s++)
    rows += job->segment[s].d;
  parallel_rows(rows, q8_rows, job);
}

void k_matmul_q8(float *out, const K8Input *x, K8Weight w, int n, int d, int gs) {
  Q8Job job = {x, n, gs, STORE, 1, {{out, w, {0}, d}}, 0, 0, NULL, NULL};
  q8_run(&job);
}
void k_matmul_q8_add(float *out, const K8Input *x, K8Weight w, int n, int d, int gs) {
  Q8Job job = {x, n, gs, ADD, 1, {{out, w, {0}, d}}, 0, 0, NULL, NULL};
  q8_run(&job);
}
void k_matmul_q8_qkv(float *q, float *k, float *v, const K8Input *x, K8Weight wq, K8Weight wk, K8Weight wv, int n,
                     int dq, int dkv, int gs) {
  Q8Job job = {x, n, gs, STORE, 3, {{q, wq, {0}, dq}, {k, wk, {0}, dkv}, {v, wv, {0}, dkv}}, 0, 0, NULL, NULL};
  q8_run(&job);
}
void k_matmul_q8_swiglu(float *out, const K8Input *x, K8Weight w1, K8Weight w3, int n, int d, int gs) {
  Q8Job job = {x, n, gs, SWIGLU, 1, {{out, w1, w3, d}}, 0, 0, NULL, NULL};
  q8_run(&job);
}

/* ---- Shard entry points (B2) ----------------------------------------------------------------------------------
 * Weight rows have stride ld (a shard packed by B1 has ld = its column count; ld may exceed n for padded rows).
 * Row slices need nothing more: pass the shard's rows. A column slice passes its own n and the slice of the input. */
static void check_epilogue(int kind, const char *name) {
  if (kind != STORE && kind != ADD && kind != PARTIAL)
    fail("%s: invalid epilogue kind %d", name, kind);
}

void k_matmul_ld(float *out, const float *x, const float *w, size_t ld, int n, int d, int kind) {
  check_epilogue(kind, "k_matmul_ld");
  if (ld < (size_t)n)
    fail("k_matmul_ld: row stride %zu is shorter than the %d inputs", ld, n);
  F32Job job = {x, n, kind, 1, {{out, w, NULL, d}}, ld, 0, NULL, NULL, 0};
  f32_run(&job);
}
/* The body of k_matmul_ld for one member of a team, to be called from inside a TeamFn on every member (the same row
 * split as the team-level call, so the results are bitwise equal). It lets the caller fuse the projection with a
 * following step into one team job. */
void k_matmul_ld_worker(float *out, const float *x, const float *w, size_t ld, int n, int d, int kind, int worker,
                        int workers, atomic_int *claim, RowsDoneFn done, void *done_ctx) {
  check_epilogue(kind, "k_matmul_ld_worker");
  if (ld < (size_t)n)
    fail("k_matmul_ld_worker: row stride %zu is shorter than the %d inputs", ld, n);
  F32Job job = {x, n, kind, 1, {{out, w, NULL, d}}, ld, 0, done, done_ctx, 0};
  GemvSched sched = effective_f32(job_bytes_f32(&job));
  if (sched != SCHED_STATIC && claim && workers > 1)
    f32_team_claim(&job, claim, worker, workers, sched);
  else
    f32_team_static(&job, worker, workers);
}
void k_matmul_swiglu_ld(float *out, const float *x, const float *w1, const float *w3, size_t ld, int n, int d) {
  if (ld < (size_t)n)
    fail("k_matmul_swiglu_ld: row stride %zu is shorter than the %d inputs", ld, n);
  F32Job job = {x, n, SWIGLU, 1, {{out, w1, w3, d}}, ld, 0, NULL, NULL, 0};
  f32_run(&job);
}
/* ---- Packed four-row layout (B7) -------------------------------------------------------------------------------------
 * The FP32 decode kernels stream each weight row as its own sequence, four rows at a time: four streams per core, and a
 * measured 408 GB/s per socket for one 112 MiB matrix per job against 518-529 GB/s when one team job covers 4 GiB. Packing
 * the four rows of a group into ONE sequential stream (16 floats of row 0, of row 1, 2, 3, then the next 16 columns)
 * measured 612 GB/s in tests/stream_bench.c (the raw stream is 601). The arithmetic of every row is unchanged, so the result
 * is bitwise equal to the row layout whenever the row layout also forms whole four-row groups (claiming schedules). */
size_t k_pack4_floats(int rows, int n) {
  return (size_t)(rows / BLOCK) * (size_t)((n + 15) / 16) * 64;
}

void k_pack4_groups(float *dst, const float *w, size_t ld, int n, int first, int last) {
  const size_t blocks = (size_t)((n + 15) / 16);
  for (int g = first; g < last; g++) {
    float *out = dst + (size_t)g * blocks * 64;
    for (size_t b = 0; b < blocks; b++) {
      int count = n - (int)b * 16 < 16 ? n - (int)b * 16 : 16;
      for (int r = 0; r < BLOCK; r++) {
        const float *row = w + ((size_t)g * BLOCK + (size_t)r) * ld + b * 16;
        float *to = out + b * 64 + (size_t)r * 16;
        for (int c = 0; c < 16; c++)
          to[c] = c < count ? row[c] : 0.0f;
      }
    }
  }
}

typedef struct {
  const Pack4Job *jobs;
  int count;
} Pack4All;

static void pack4_chunks(const void *p, int s, int first, int last) {
  const Pack4All *all = p;
  (void)s;
  int base = 0;
  for (int i = 0; i < all->count && base < last; i++) {
    const Pack4Job *job = &all->jobs[i];
    int groups = job->rows / BLOCK;
    if (base + groups > first)
      k_pack4_groups(job->dst, job->w, job->ld, job->n, first > base ? first - base : 0,
                     last < base + groups ? last - base : groups);
    base += groups;
  }
}

void k_pack4_many(const Pack4Job *jobs, int count) {
  Pack4All all = {jobs, count};
  long long groups = 0;
  for (int i = 0; i < count; i++) {
    if (jobs[i].rows % BLOCK || jobs[i].n < 1 || jobs[i].ld < (size_t)jobs[i].n)
      fail("k_pack4_many: %d rows x %d columns (stride %zu) cannot be packed in four-row groups", jobs[i].rows,
           jobs[i].n, jobs[i].ld);
    groups += jobs[i].rows / BLOCK;
  }
  if (groups > INT32_MAX)
    fail("too many weight groups to pack: %lld", groups);
  int chunks = (int)groups;
  kernels_parallel_segments(1, &chunks, pack4_chunks, &all);
}

static void check_p4(int d, const char *name) {
  if (d % BLOCK)
    fail("%s: %d rows are not a whole number of four-row groups", name, d);
}

void k_matmul_p4(float *out, const float *x, const float *wp, int n, int d, int kind) {
  check_epilogue(kind, "k_matmul_p4");
  check_p4(d, "k_matmul_p4");
  F32Job job = {x, n, kind, 1, {{out, wp, NULL, d}}, 0, 0, NULL, NULL, 1};
  f32_run(&job);
}
void k_matmul_p4_worker(float *out, const float *x, const float *wp, int n, int d, int kind, int worker, int workers,
                        atomic_int *claim, RowsDoneFn done, void *done_ctx) {
  check_epilogue(kind, "k_matmul_p4_worker");
  check_p4(d, "k_matmul_p4_worker");
  F32Job job = {x, n, kind, 1, {{out, wp, NULL, d}}, 0, 0, done, done_ctx, 1};
  GemvSched sched = effective_f32(job_bytes_f32(&job));
  if (sched != SCHED_STATIC && claim && workers > 1)
    f32_team_claim(&job, claim, worker, workers, sched);
  else
    f32_team_static(&job, worker, workers);
}
void k_matmul_qkv_p4(float *q, float *k, float *v, const float *x, const float *wq, const float *wk, const float *wv,
                     int n, int dq, int dkv) {
  check_p4(dq, "k_matmul_qkv_p4");
  check_p4(dkv, "k_matmul_qkv_p4");
  F32Job job = {x, n, STORE, 3, {{q, wq, NULL, dq}, {k, wk, NULL, dkv}, {v, wv, NULL, dkv}}, 0, 0, NULL, NULL, 1};
  f32_run(&job);
}
void k_matmul_swiglu_p4(float *out, const float *x, const float *w1, const float *w3, int n, int d) {
  check_p4(d, "k_matmul_swiglu_p4");
  F32Job job = {x, n, SWIGLU, 1, {{out, w1, w3, d}}, 0, 0, NULL, NULL, 1};
  f32_run(&job);
}
void k_matmul_q8_ld(float *out, const K8Input *x, const K8Weight *w, size_t ld, int n, int d, int gs, int kind) {
  check_epilogue(kind, "k_matmul_q8_ld");
  if (ld < (size_t)n || ld % (size_t)gs)
    fail("k_matmul_q8_ld: row stride %zu must be at least %d and a multiple of the group size %d", ld, n, gs);
  Q8Job job = {x, n, gs, kind, 1, {{out, *w, {0}, d}}, ld, 0, NULL, NULL};
  q8_run(&job);
}
void k_matmul_q8_ld_worker(float *out, const K8Input *x, const K8Weight *w, size_t ld, int n, int d, int gs, int kind,
                           int worker, int workers, atomic_int *claim, RowsDoneFn done, void *done_ctx) {
  check_epilogue(kind, "k_matmul_q8_ld_worker");
  if (ld < (size_t)n || ld % (size_t)gs)
    fail("k_matmul_q8_ld_worker: row stride %zu must be at least %d and a multiple of the group size %d", ld, n, gs);
  Q8Job job = {x, n, gs, kind, 1, {{out, *w, {0}, d}}, ld, 0, done, done_ctx};
  if (gemv_sched_q8 != SCHED_STATIC && claim && workers > 1)
    q8_team_claim(&job, claim, worker, workers, gemv_sched_q8);
  else
    q8_team_static(&job, worker, workers);
}
void k_matmul_q8_swiglu_ld(float *out, const K8Input *x, const K8Weight *w1, const K8Weight *w3, size_t ld, int n,
                           int d, int gs) {
  if (ld < (size_t)n || ld % (size_t)gs)
    fail("k_matmul_q8_swiglu_ld: row stride %zu must be at least %d and a multiple of the group size %d", ld, n, gs);
  Q8Job job = {x, n, gs, SWIGLU, 1, {{out, *w1, *w3, d}}, ld, 0, NULL, NULL};
  q8_run(&job);
}

void k_quantize(int8_t *q, float *s, const float *x, int n, int gs) {
#if HAVE_AVX512
  if (isa == KERNELS_AVX512 && gs % 16 == 0) {
    quantize_avx512(q, s, x, n, gs);
    return;
  }
#endif
  int num_groups = n / gs;
  float Q_MAX = 127.0f;

  for (int group = 0; group < num_groups; group++) {

    // find the max absolute value in the current group
    float wmax = 0.0;
    for (int i = 0; i < gs; i++) {
      if (!isfinite(x[group * gs + i]))
        fail("nonfinite activation during INT8 quantization");
      float val = fabsf(x[group * gs + i]);
      if (val > wmax) {
        wmax = val;
      }
    }

    // calculate and write the scaling factor
    float scale = wmax / Q_MAX;
    s[group] = scale;

    // calculate and write the quantized values
    for (int i = 0; i < gs; i++) {
      float quant_value = scale == 0.0f ? 0.0f : x[group * gs + i] / scale; // scale
      /* Clamp also handles subnormal-scale rounding without float-to-int UB. */
      int8_t quantized = (int8_t)roundf(fmaxf(-127.0f, fminf(127.0f, quant_value)));
      q[group * gs + i] = quantized;
    }
  }
}

// ----------------------------------------------------------------------------
// Attention

static int max_threads(void) {
  if (current_team)
    return team_size(current_team);
#ifdef _OPENMP
  return omp_get_max_threads();
#else
  return 1;
#endif
}

/* Enough chunks to occupy every thread, each of 32..256 positions where the
 * prefix allows, bounded by the workspace. */
static int attention_chunks(int positions, int n_kv_heads, int limit, int workers) {
  int chunks = (workers + n_kv_heads - 1) / n_kv_heads;
  int longest = (positions + 31) / 32, shortest = (positions + 255) / 256;
  if (chunks > longest)
    chunks = longest;
  if (chunks < shortest)
    chunks = shortest;
  if (chunks > limit)
    chunks = limit;
  return chunks < 1 ? 1 : chunks;
}
static int attention_max_chunks(int n_kv_heads, int seq_len, int workers) {
  int by_threads = (workers + n_kv_heads - 1) / n_kv_heads, by_length = (seq_len + 255) / 256;
  return by_threads > by_length ? by_threads : by_length;
}

static size_t attention_row_floats(int n_heads, int head_dim, int max_chunks) {
  return checked_mul(checked_mul((size_t)n_heads, (size_t)max_chunks), (size_t)head_dim + 2);
}

static size_t attention_budget_floats(void) {
  const char *value = getenv("LLAMA_ATTN_WORKSPACE_MIB");
  long mib = value && *value ? atol(value) : 32;
  return mib > 0 ? (size_t)mib * (1u << 20) / sizeof(float) : 1;
}

/* Rows of partials that fit: at least one, at most `rows`, otherwise as many as the budget holds. */
static size_t attention_rows_capacity(int n_heads, int head_dim, int max_chunks, int rows) {
  size_t per_row = attention_row_floats(n_heads, head_dim, max_chunks), fit = attention_budget_floats() / per_row;
  size_t want = rows > 1 ? (size_t)rows : 1;
  return fit < 1 ? 1 : fit < want ? fit : want;
}

size_t k_attention_bytes_rows(int n_heads, int n_kv_heads, int head_dim, int seq_len, int rows) {
  int chunks = attention_max_chunks(n_kv_heads, seq_len, max_threads());
  return checked_mul(checked_mul(attention_rows_capacity(n_heads, head_dim, chunks, rows),
                                 attention_row_floats(n_heads, head_dim, chunks)),
                     sizeof(float));
}

size_t k_attention_bytes(int n_heads, int n_kv_heads, int head_dim, int seq_len) {
  return k_attention_bytes_rows(n_heads, n_kv_heads, head_dim, seq_len, 1);
}

void k_attention_init_rows(AttentionWork *work, int n_heads, int n_kv_heads, int head_dim, int seq_len, int rows,
                           float *memory) {
  /* The team (or OpenMP thread count) must be set before this: it sizes the split-K workspace. */
  work->workers = max_threads();
  work->max_chunks = attention_max_chunks(n_kv_heads, seq_len, work->workers);
  work->n_heads = n_heads;
  work->head_dim = head_dim;
  work->external = memory != NULL;
  work->capacity = checked_mul(attention_rows_capacity(n_heads, head_dim, work->max_chunks, rows),
                               attention_row_floats(n_heads, head_dim, work->max_chunks));
  work->partial = memory ? memory : memory_alloc(work->capacity, sizeof(float), MEM_ACTIVATION);
}

void k_attention_init_in(AttentionWork *work, int n_heads, int n_kv_heads, int head_dim, int seq_len, float *memory) {
  k_attention_init_rows(work, n_heads, n_kv_heads, head_dim, seq_len, 1, memory);
}

void k_attention_init(AttentionWork *work, int n_heads, int n_kv_heads, int head_dim, int seq_len) {
  k_attention_init_in(work, n_heads, n_kv_heads, head_dim, seq_len, NULL);
}

int k_attention_chunk_count(const AttentionWork *work, int n_kv_heads, int position) {
  return attention_chunks(position + 1, n_kv_heads, work->max_chunks, work->workers);
}

void k_attention_free(AttentionWork *work) {
  if (!work->external)
    memory_free(work->partial);
  work->partial = NULL;
}

typedef struct {
  float *out;
  const float *q, *key_cache, *value_cache;
  float *att;
  int n_heads, n_kv_heads, head_size, seq_len, pos;
  size_t pos_stride, head_base; /* floats between positions of a head and between the bases of KV heads */
} ScalarAttention;

static void attention_scalar_head(const ScalarAttention *j, int h) {
  int head_size = j->head_size, pos = j->pos, kv_mul = j->n_heads / j->n_kv_heads;
  // get the query vector for this head
  const float *qh = j->q + h * head_size;
  // attention scores for this head
  float *a = j->att + (size_t)h * j->seq_len;
  // iterate over all timesteps, including the current one
  for (int t = 0; t <= pos; t++) {
    // get the key vector for this head and at this timestep
    const float *k = j->key_cache + (size_t)t * j->pos_stride + (size_t)(h / kv_mul) * j->head_base;
    // calculate the attention score as the dot product of q and k
    float score = 0.0f;
    for (int i = 0; i < head_size; i++) {
      score += qh[i] * k[i];
    }
    score /= sqrtf(head_size);
    // save the score to the attention buffer
    a[t] = score;
  }

  // softmax the scores to get attention weights, from 0..pos inclusively
  softmax(a, pos + 1);

  // weighted sum of the values, store back into xb
  float *xb = j->out + h * head_size;
  memset(xb, 0, head_size * sizeof(float));
  for (int t = 0; t <= pos; t++) {
    // get the value vector for this head and at this timestep
    const float *v = j->value_cache + (size_t)t * j->pos_stride + (size_t)(h / kv_mul) * j->head_base;
    // get the attention weight for this timestep
    float w = a[t];
    // accumulate the weighted value into xb
    for (int i = 0; i < head_size; i++) {
      xb[i] += w * v[i];
    }
  }
}

static void attention_scalar_rows(const void *p, int h0, int h1) {
  for (int h = h0; h < h1; h++)
    attention_scalar_head(p, h);
}

static void attention_scalar(float *out, const float *q, const float *key_cache, const float *value_cache, float *att,
                             int n_heads, int n_kv_heads, int head_size, int seq_len, int pos, size_t pos_stride,
                             size_t head_base) {
  ScalarAttention job = {out, q, key_cache, value_cache, att, n_heads, n_kv_heads, head_size, seq_len, pos, pos_stride,
                         head_base};
  if (current_team) {
    team_parallel_rows(current_team, n_heads, attention_scalar_rows, &job);
    return;
  }
  // multihead attention. iterate over all heads
  int h;
#ifdef _OPENMP
#pragma omp parallel for private(h)
#endif
  for (h = 0; h < n_heads; h++)
    attention_scalar_head(&job, h);
}

/* Merge chunk results: rescale each by exp(m_c - M) and normalize. `base` holds `chunks` slots of `slot` floats. Empty
 * partials are skipped, so their value sum is never read. */
static void merge_chunks(const float *base, size_t slot, int chunks, int head_dim, float *o) {
  float max_val = base[head_dim];
  for (int c = 1; c < chunks; c++)
    if (base[c * slot + head_dim] > max_val)
      max_val = base[c * slot + head_dim];
  float sum = 0.0f;
  memset(o, 0, head_dim * sizeof(float));
  for (int c = 0; c < chunks; c++) {
    const float *chunk = base + c * slot;
    if (partial_empty(chunk, head_dim))
      continue; /* a row with no position in this chunk (tiled, E2); split never writes one, so it is unaffected */
    float weight = expf(chunk[head_dim] - max_val);
    sum += weight * chunk[head_dim + 1];
    for (int i = 0; i < head_dim; i++)
      o[i] += weight * chunk[i];
  }
  for (int i = 0; i < head_dim; i++)
    o[i] /= sum;
}

void k_attention_merge_partials(const float *base, size_t slot, int chunks, int head_dim, float *o) {
  merge_chunks(base, slot, chunks, head_dim, o);
}

/* Opt-in per-task timing for the attention benchmark (E0). Unset (the default), a task costs one atomic load. */
static _Atomic(AttentionTaskHook) attention_task_hook;
static void *attention_task_ctx;

void k_attention_set_task_hook(AttentionTaskHook hook, void *ctx) {
  attention_task_ctx = ctx;
  atomic_store_explicit(&attention_task_hook, hook, memory_order_release);
}

/* worker < 0: the OpenMP thread number. */
static int attention_worker_id(int worker) {
#ifdef _OPENMP
  return worker < 0 ? omp_get_thread_num() : worker;
#else
  return worker < 0 ? 0 : worker;
#endif
}

#if HAVE_AVX512
typedef struct {
  float *out;
  const float *q, *key_cache, *value_cache;
  float *att, *partial;
  int n_heads, head_dim, seq_len, positions, kv_mul, kv_dim, chunks, tasks;
  int group; /* 1: the group kernel (E5) runs the (KV group, chunk) tasks instead of the per-head chunk kernel */
  size_t slot, head_stride;
  size_t pos_stride, head_base; /* K/V layout (E5.2): floats between positions of a head, and between KV heads' bases */
  atomic_int next_task; /* team-local queue over (kv group, chunk) tasks */
  Team *team;           /* the barrier's team: workers cannot look it up (thread_team is the coordinator's) */
} SplitKJob;

static void attention_task(const SplitKJob *j, int task) {
  int g = task / j->chunks, c = task % j->chunks;
  int t0 = (int)((long long)j->positions * c / j->chunks), t1 = (int)((long long)j->positions * (c + 1) / j->chunks);
  int h0 = g * j->kv_mul;
  if (j->group) {
    attention_group_avx512(j->q + (size_t)h0 * j->head_dim, j->kv_mul, j->head_dim, t0, t1,
                           j->key_cache + (size_t)g * j->head_base, j->value_cache + (size_t)g * j->head_base,
                           (int)j->pos_stride, j->partial + h0 * j->head_stride + c * j->slot, j->head_stride);
    return;
  }
  attention_chunk_avx512(j->q + (size_t)h0 * j->head_dim, j->key_cache + (size_t)g * j->head_base,
                         j->value_cache + (size_t)g * j->head_base, (int)j->pos_stride, j->head_dim, j->kv_mul, t0, t1,
                         j->att + (size_t)h0 * j->seq_len + t0, j->seq_len,
                         j->partial + h0 * j->head_stride + c * j->slot, j->head_stride);
}

static void attention_merge(const SplitKJob *j, int h) {
  merge_chunks(j->partial + h * j->head_stride, j->slot, j->chunks, j->head_dim, j->out + (size_t)h * j->head_dim);
}

/* The split-K task with the benchmark hook (see batch_run); a decode task is one row. */
static void attention_run(const SplitKJob *j, int task, int worker) {
  AttentionTaskHook hook = atomic_load_explicit(&attention_task_hook, memory_order_acquire);
  if (!hook) {
    attention_task(j, task);
    return;
  }
  int c = task % j->chunks, t0 = (int)((long long)j->positions * c / j->chunks),
      t1 = (int)((long long)j->positions * (c + 1) / j->chunks);
  AttentionTaskInfo info = {.worker = attention_worker_id(worker), .task = task, .rows = 1, .span = t1 - t0,
                            .positions = t1 - t0};
  info.begin = monotonic_seconds();
  attention_task(j, task);
  info.end = monotonic_seconds();
  hook(attention_task_ctx, &info);
}

static void attention_team(const void *p, int worker, int workers) {
  SplitKJob *j = (SplitKJob *)p;
  for (int task; (task = atomic_fetch_add_explicit(&j->next_task, 1, memory_order_relaxed)) < j->tasks;)
    attention_run(j, task, worker);
  team_barrier(j->team); /* every chunk of every head is in `partial` before any head is merged */
  for (int h = (int)((long long)j->n_heads * worker / workers);
       h < (int)((long long)j->n_heads * (worker + 1) / workers); h++)
    attention_merge(j, h);
}
#endif

/* The split-K decode path; `group` selects the per-group kernel (E5) for the (KV group, chunk) tasks. The chunk plan,
 * the partial slots and the merge are the same for both. */
static void attention_decode_splitk(float *out, const float *q, const float *key_cache, const float *value_cache,
                                    float *att, AttentionWork *work, int n_heads, int n_kv_heads, int head_dim,
                                    int seq_len, int position, int group, size_t pos_stride, size_t head_base) {
#if HAVE_AVX512
  if (isa == KERNELS_AVX512 && head_dim <= 256) {
    int positions = position + 1, kv_mul = n_heads / n_kv_heads, kv_dim = n_kv_heads * head_dim;
    int chunks = attention_chunks(positions, n_kv_heads, work->max_chunks, work->workers);
    SplitKJob job = {.out = out,
                     .q = q,
                     .key_cache = key_cache,
                     .value_cache = value_cache,
                     .att = att,
                     .partial = work->partial,
                     .n_heads = n_heads,
                     .head_dim = head_dim,
                     .seq_len = seq_len,
                     .positions = positions,
                     .kv_mul = kv_mul,
                     .kv_dim = kv_dim,
                     .pos_stride = pos_stride,
                     .head_base = head_base,
                     .chunks = chunks,
                     .tasks = n_kv_heads * chunks,
                     .group = group,
                     .slot = (size_t)head_dim + 2,
                     .head_stride = (size_t)work->max_chunks * ((size_t)head_dim + 2)};
    atomic_init(&job.next_task, 0);
    job.team = current_team;
    if (current_team) {
      team_run(current_team, attention_team, &job);
      return;
    }
#ifdef _OPENMP
#pragma omp parallel
#endif
    {
#ifdef _OPENMP
#pragma omp for schedule(dynamic, 1)
#endif
      for (int task = 0; task < job.tasks; task++)
        attention_run(&job, task, -1);
#ifdef _OPENMP
#pragma omp for
#endif
      for (int h = 0; h < n_heads; h++)
        attention_merge(&job, h);
    }
    return;
  }
#endif
  (void)work, (void)group;
  attention_scalar(out, q, key_cache, value_cache, att, n_heads, n_kv_heads, head_dim, seq_len, position, pos_stride,
                   head_base);
}

static void attention_decode_split(float *out, const float *q, const float *key_cache, const float *value_cache,
                                   float *att, AttentionWork *work, int n_heads, int n_kv_heads, int head_dim,
                                   int seq_len, int position, size_t pos_stride, size_t head_base) {
  attention_decode_splitk(out, q, key_cache, value_cache, att, work, n_heads, n_kv_heads, head_dim, seq_len, position,
                          0, pos_stride, head_base);
}

/* LLAMA_ATTN_DECODE=group (E5). One note per process when the call goes to `split`: same precision, different speed. */
static void attention_decode_group(float *out, const float *q, const float *key_cache, const float *value_cache,
                                   float *att, AttentionWork *work, int n_heads, int n_kv_heads, int head_dim,
                                   int seq_len, int position, size_t pos_stride, size_t head_base) {
  const char *why = NULL;
#if HAVE_AVX512
  if (isa != KERNELS_AVX512)
    why = "the AVX-512 kernels are not active";
  else if (head_dim > GROUP_MAX_HD || n_heads / n_kv_heads > GROUP_MAX_Q)
    why = "head_dim above 256 or more than 16 query heads per KV head is not supported";
#else
  why = "the AVX-512 kernels are not built";
#endif
  if (why) {
    static atomic_flag noted = ATOMIC_FLAG_INIT;
    if (!atomic_flag_test_and_set(&noted))
      LOG(LOG_INFO, "LLAMA_ATTN_DECODE=group: %s, running split attention instead", why);
  }
  attention_decode_splitk(out, q, key_cache, value_cache, att, work, n_heads, n_kv_heads, head_dim, seq_len, position,
                          why == NULL, pos_stride, head_base);
}

// ----------------------------------------------------------------------------
// Batched causal attention (C3)

typedef void (*ChunkFn)(const float *q, const float *kc, const float *vc, int kv_dim, int head_dim, int heads, int t0,
                        int t1, float *scores, size_t score_stride, float *partial, size_t head_stride);

/* Scalar reference of the chunk kernel: one GQA group over positions [t0, t1), writing per query head the
 * unnormalized value sum, the local maximum and the local sum of exponentials, like attention_chunk_avx512. */
static void attention_chunk_portable(const float *q, const float *kc, const float *vc, int kv_dim, int head_dim,
                                     int heads, int t0, int t1, float *scores, size_t score_stride, float *partial,
                                     size_t head_stride) {
  float divisor = sqrtf((float)head_dim);
  for (int h = 0; h < heads; h++) {
    const float *qh = q + (size_t)h * head_dim;
    float *a = scores + (size_t)h * score_stride, max_val = -FLT_MAX;
    for (int t = t0; t < t1; t++) {
      const float *k = kc + (size_t)t * kv_dim;
      float score = 0.0f;
      for (int i = 0; i < head_dim; i++)
        score += qh[i] * k[i];
      a[t - t0] = score / divisor;
      if (a[t - t0] > max_val)
        max_val = a[t - t0];
    }
    float *o = partial + (size_t)h * head_stride, sum = 0.0f;
    memset(o, 0, head_dim * sizeof(float));
    for (int t = t0; t < t1; t++) {
      float e = expf(a[t - t0] - max_val);
      sum += e;
      const float *v = vc + (size_t)t * kv_dim;
      for (int i = 0; i < head_dim; i++)
        o[i] += e * v[i];
    }
    o[head_dim] = max_val;
    o[head_dim + 1] = sum;
  }
}

enum { BATCH_MAX_GROUP = 16, BATCH_MAX_CHUNK = 256 };

void k_attention_set_tiled(int queries, int longest_first) {
  if (queries < 1 || queries > 16)
    fail("LLAMA_ATTN_TILED_QUERIES must be an integer in 1..16");
  attention_tiled_queries = queries;
  attention_tiled_longest_first = longest_first != 0;
}

void k_attention_set_row_block(int rows) {
  if (rows < 1 || rows > BATCH_MAX_ROW_BLOCK)
    fail("LLAMA_ATTN_ROW_BLOCK must be an integer in 1..%d", BATCH_MAX_ROW_BLOCK);
  attention_row_block = rows;
}

int k_attention_row_block(void) {
  return attention_row_block;
}

typedef struct {
  float *out;
  const float *q, *key_cache, *value_cache;
  float *partial;
  ChunkFn chunk;
  int n_heads, n_kv_heads, head_dim, kv_mul, pos_stride, dim, limit, workers;
  int pos0, row0, rows; /* this wave: rows row0..row0+rows-1 of the microbatch */
  int chunks_wave, blocks, tasks, row_block;
  size_t slot, head_stride, row_stride, head_base; /* K/V layout: floats between positions / between KV heads' bases */
  atomic_int next_task; /* team-local queue over (kv group, chunk, row block) tasks */
  Team *team;           /* the barrier's team: workers cannot look it up (thread_team is the coordinator's) */
} BatchJob;

static int batch_row_chunks(const BatchJob *j, int row) {
  return attention_chunks(j->pos0 + row + 1, j->n_kv_heads, j->limit, j->workers);
}

static void batch_split(const BatchJob *j, int task, int *g, int *c, int *first, int *last) {
  int block = task % j->blocks, rest = task / j->blocks;
  *c = rest % j->chunks_wave, *g = rest / j->chunks_wave;
  *first = block * j->row_block, *last = *first + j->row_block < j->rows ? *first + j->row_block : j->rows;
}

/* One (kv group, chunk, block of rows) task. Rows of a block read the same K/V chunk back to back, so it stays in L2;
 * each row uses exactly the chunk plan k_attention would use at its position, and positions past the row are masked
 * simply by never being part of its chunks. */
static void batch_task(const BatchJob *j, int task) {
  int g, c, first, last;
  batch_split(j, task, &g, &c, &first, &last);
  float scores[BATCH_MAX_GROUP * BATCH_MAX_CHUNK];
  for (int rr = first; rr < last; rr++) {
    int row = j->row0 + rr, positions = j->pos0 + row + 1, chunks = batch_row_chunks(j, row);
    if (c >= chunks)
      continue;
    int t0 = (int)((long long)positions * c / chunks), t1 = (int)((long long)positions * (c + 1) / chunks);
    int h0 = g * j->kv_mul;
    j->chunk(j->q + (size_t)row * j->dim + (size_t)h0 * j->head_dim, j->key_cache + (size_t)g * j->head_base,
             j->value_cache + (size_t)g * j->head_base, j->pos_stride, j->head_dim, j->kv_mul, t0, t1, scores,
             (size_t)(t1 - t0), j->partial + (size_t)rr * j->row_stride + (size_t)h0 * j->head_stride + c * j->slot,
             j->head_stride);
  }
}

/* What a task reads, for the hook: rows that take part, positions scored, and the K/V span the block streams (the rows'
 * chunks overlap, so the block pulls about min(t0)..max(t1) once). */
static void batch_extent(const BatchJob *j, int task, AttentionTaskInfo *info) {
  int g, c, first, last, lo = INT_MAX, hi = 0;
  batch_split(j, task, &g, &c, &first, &last);
  for (int rr = first; rr < last; rr++) {
    int positions = j->pos0 + j->row0 + rr + 1, chunks = batch_row_chunks(j, j->row0 + rr);
    if (c >= chunks)
      continue;
    int t0 = (int)((long long)positions * c / chunks), t1 = (int)((long long)positions * (c + 1) / chunks);
    info->rows++, info->positions += t1 - t0;
    lo = t0 < lo ? t0 : lo, hi = t1 > hi ? t1 : hi;
  }
  info->span = hi > lo ? hi - lo : 0;
}

static void batch_run(const BatchJob *j, int task, int worker) {
  AttentionTaskHook hook = atomic_load_explicit(&attention_task_hook, memory_order_acquire);
  if (!hook) {
    batch_task(j, task);
    return;
  }
  AttentionTaskInfo info = {.worker = attention_worker_id(worker), .wave = j->row0, .task = task};
  batch_extent(j, task, &info);
  info.begin = monotonic_seconds();
  batch_task(j, task);
  info.end = monotonic_seconds();
  hook(attention_task_ctx, &info);
}

static void batch_merge(const BatchJob *j, int item) {
  int rr = item / j->n_heads, h = item % j->n_heads, row = j->row0 + rr;
  merge_chunks(j->partial + (size_t)rr * j->row_stride + (size_t)h * j->head_stride, j->slot, batch_row_chunks(j, row),
               j->head_dim, j->out + (size_t)row * j->dim + (size_t)h * j->head_dim);
}

static void batch_team(const void *p, int worker, int workers) {
  BatchJob *j = (BatchJob *)p;
  for (int task; (task = atomic_fetch_add_explicit(&j->next_task, 1, memory_order_relaxed)) < j->tasks;)
    batch_run(j, task, worker);
  team_barrier(j->team); /* every chunk of every row and head is in `partial` before any is merged */
  int items = j->rows * j->n_heads;
  for (int item = (int)((long long)items * worker / workers); item < (int)((long long)items * (worker + 1) / workers);
       item++)
    batch_merge(j, item);
}

static void attention_batch_driver(float *out, const float *q, const float *key_cache, const float *value_cache,
                                   AttentionWork *work, int n_heads, int n_kv_heads, int head_dim, int pos0, int m,
                                   size_t pos_stride, size_t head_base, ChunkFn chunk) {
  BatchJob job = {.row_block = attention_row_block,
                  .out = out,
                  .q = q,
                  .key_cache = key_cache,
                  .value_cache = value_cache,
                  .partial = work->partial,
                  .chunk = chunk,
                  .n_heads = n_heads,
                  .n_kv_heads = n_kv_heads,
                  .head_dim = head_dim,
                  .kv_mul = n_heads / n_kv_heads,
                  .pos_stride = (int)pos_stride,
                  .head_base = head_base,
                  .dim = n_heads * head_dim,
                  .limit = work->max_chunks,
                  .workers = work->workers,
                  .pos0 = pos0,
                  .slot = (size_t)head_dim + 2};
  for (int row0 = 0; row0 < m;) {
    /* A wave holds as many rows of partials as the workspace allows; chunk counts only grow with the position, so the
     * last row of the wave sets the slots per head. */
    int rows = m - row0;
    for (;;) {
      job.row0 = row0, job.rows = rows;
      job.chunks_wave = batch_row_chunks(&job, row0 + rows - 1);
      if (rows == 1 || (size_t)rows * n_heads * job.chunks_wave * job.slot <= work->capacity)
        break;
      rows = (int)(work->capacity / ((size_t)n_heads * job.chunks_wave * job.slot));
      rows = rows < 1 ? 1 : rows;
    }
    job.head_stride = (size_t)job.chunks_wave * job.slot;
    job.row_stride = (size_t)n_heads * job.head_stride;
    job.blocks = (rows + job.row_block - 1) / job.row_block;
    job.tasks = n_kv_heads * job.chunks_wave * job.blocks;
    atomic_init(&job.next_task, 0);
    job.team = current_team;
    if (current_team) {
      team_run(current_team, batch_team, &job);
    } else {
#ifdef _OPENMP
#pragma omp parallel
#endif
      {
#ifdef _OPENMP
#pragma omp for schedule(dynamic, 1)
#endif
        for (int task = 0; task < job.tasks; task++)
          batch_run(&job, task, -1);
#ifdef _OPENMP
#pragma omp for
#endif
        for (int item = 0; item < job.rows * n_heads; item++)
          batch_merge(&job, item);
      }
    }
    row0 += rows;
  }
}

static int batch_supported(int n_heads, int n_kv_heads, int head_dim, int seq_len, int pos0, int m) {
  (void)seq_len;
  return head_dim <= 256 && n_heads / n_kv_heads <= BATCH_MAX_GROUP && m > 1 && pos0 >= 0;
}

static void attention_batch_split(float *out, const float *q, const float *key_cache, const float *value_cache,
                                  float *att, AttentionWork *work, int n_heads, int n_kv_heads, int head_dim,
                                  int seq_len, int pos0, int m, size_t pos_stride, size_t head_base) {
#if HAVE_AVX512
  if (isa == KERNELS_AVX512 && batch_supported(n_heads, n_kv_heads, head_dim, seq_len, pos0, m)) {
    attention_batch_driver(out, q, key_cache, value_cache, work, n_heads, n_kv_heads, head_dim, pos0, m, pos_stride,
                           head_base, attention_chunk_avx512);
    return;
  }
#endif
  /* The scalar kernels have no cross-row sharing to exploit: one row at a time, exactly the per-row call. */
  size_t dim = (size_t)n_heads * head_dim;
  for (int i = 0; i < m; i++)
    attention_decode_split(out + i * dim, q + i * dim, key_cache, value_cache, att, work, n_heads, n_kv_heads, head_dim,
                           seq_len, pos0 + i, pos_stride, head_base);
}

void k_attention_batch_portable_layout(float *out, const float *q, const float *key_cache, const float *value_cache,
                                       AttentionWork *work, int n_heads, int n_kv_heads, int head_dim, int seq_len,
                                       int pos0, int m, size_t pos_stride, size_t head_base) {
  if (!batch_supported(n_heads, n_kv_heads, head_dim, seq_len, pos0, m))
    fail("portable batched attention needs head_dim <= 256, at most %d heads per KV head and m > 1", BATCH_MAX_GROUP);
  if (pos_stride < (size_t)head_dim || pos_stride > INT_MAX)
    fail("k_attention_batch_portable_layout: position stride %zu is not in [head_dim, INT_MAX]", pos_stride);
  attention_batch_driver(out, q, key_cache, value_cache, work, n_heads, n_kv_heads, head_dim, pos0, m, pos_stride,
                         head_base, attention_chunk_portable);
}

void k_attention_batch_portable(float *out, const float *q, const float *key_cache, const float *value_cache,
                                AttentionWork *work, int n_heads, int n_kv_heads, int head_dim, int seq_len, int pos0,
                                int m) {
  k_attention_batch_portable_layout(out, q, key_cache, value_cache, work, n_heads, n_kv_heads, head_dim, seq_len, pos0,
                                    m, (size_t)n_kv_heads * head_dim, head_dim);
}

// ----------------------------------------------------------------------------
// Tiled batched attention (E2)

#if HAVE_AVX512
typedef struct {
  int g, block, chunk, cost;
} TiledTask;

typedef struct {
  float *out;
  const float *q, *key_cache, *value_cache;
  float *partial;
  int n_heads, n_kv_heads, head_dim, kv_mul, pos_stride, dim, limit, workers;
  int pos0, m, row0, rows; /* this wave: rows row0..row0+rows-1 of the microbatch of m rows */
  int rblock, chunks_wave, tasks;
  const TiledTask *order; /* the tasks of the wave, in the order they are claimed */
  size_t slot, head_stride, row_stride, head_base; /* K/V layout: floats between positions / between KV heads' bases */
  atomic_int next_task;
  Team *team;
} TiledJob;

/* The chunk plan of a block of rows: chunk length (a multiple of the position tile) and chunk count. It depends only on
 * the block's nominal last row, never on the wave that happens to contain the block, so results do not depend on how a
 * microbatch is cut into waves. */
static void tiled_plan(const TiledJob *j, int block, int *length, int *chunks) {
  int end = (block + 1) * j->rblock < j->m ? (block + 1) * j->rblock : j->m, positions = j->pos0 + end;
  int wanted = attention_chunks(positions, j->n_kv_heads, j->limit, j->workers);
  int len = ((positions + wanted - 1) / wanted + TILED_POS - 1) / TILED_POS * TILED_POS;
  *length = len;
  *chunks = (positions + len - 1) / len;
}

/* What a task covers: the first and last row of its block that belong to the wave, and its chunk [t0, t1). */
static void tiled_extent(const TiledJob *j, const TiledTask *t, int *first, int *last, int *t0, int *t1) {
  int length, chunks;
  tiled_plan(j, t->block, &length, &chunks);
  int block_end = (t->block + 1) * j->rblock < j->m ? (t->block + 1) * j->rblock : j->m;
  *first = t->block * j->rblock > j->row0 ? t->block * j->rblock : j->row0;
  *last = block_end < j->row0 + j->rows ? block_end : j->row0 + j->rows;
  *t0 = t->chunk * length;
  *t1 = (t->chunk + 1) * length < j->pos0 + block_end ? (t->chunk + 1) * length : j->pos0 + block_end;
}

static void tiled_task(const TiledJob *j, const TiledTask *t) {
  int first, last, t0, t1;
  tiled_extent(j, t, &first, &last, &t0, &t1);
  int h0 = t->g * j->kv_mul, rr = first - j->row0;
  attention_tiled_avx512(j->q + (size_t)first * j->dim + (size_t)h0 * j->head_dim, (size_t)j->dim, last - first,
                         j->kv_mul, j->head_dim, j->pos0 + first + 1, t0, t1,
                         j->key_cache + (size_t)t->g * j->head_base, j->value_cache + (size_t)t->g * j->head_base,
                         j->pos_stride,
                         j->partial + (size_t)rr * j->row_stride + (size_t)h0 * j->head_stride + t->chunk * j->slot,
                         j->row_stride, j->head_stride);
}

static void tiled_run(const TiledJob *j, int index, int worker) {
  const TiledTask *t = &j->order[index];
  AttentionTaskHook hook = atomic_load_explicit(&attention_task_hook, memory_order_acquire);
  if (!hook) {
    tiled_task(j, t);
    return;
  }
  AttentionTaskInfo info = {.worker = attention_worker_id(worker), .wave = j->row0, .task = index};
  int first, last, t0, t1;
  tiled_extent(j, t, &first, &last, &t0, &t1);
  for (int row = first; row < last; row++) {
    int end = t1 < j->pos0 + row + 1 ? t1 : j->pos0 + row + 1;
    if (end > t0)
      info.rows++, info.positions += end - t0;
  }
  info.span = t1 > t0 ? t1 - t0 : 0;
  info.begin = monotonic_seconds();
  tiled_task(j, t);
  info.end = monotonic_seconds();
  hook(attention_task_ctx, &info);
}

static void tiled_merge(const TiledJob *j, int item) {
  int rr = item / j->n_heads, h = item % j->n_heads, row = j->row0 + rr, length, chunks;
  tiled_plan(j, row / j->rblock, &length, &chunks);
  merge_chunks(j->partial + (size_t)rr * j->row_stride + (size_t)h * j->head_stride, j->slot, chunks, j->head_dim,
               j->out + (size_t)row * j->dim + (size_t)h * j->head_dim);
}

static void tiled_team(const void *p, int worker, int workers) {
  TiledJob *j = (TiledJob *)p;
  for (int task; (task = atomic_fetch_add_explicit(&j->next_task, 1, memory_order_relaxed)) < j->tasks;)
    tiled_run(j, task, worker);
  team_barrier(j->team); /* every chunk of every row and head is in `partial` before any is merged */
  int items = j->rows * j->n_heads;
  for (int item = (int)((long long)items * worker / workers); item < (int)((long long)items * (worker + 1) / workers);
       item++)
    tiled_merge(j, item);
}

/* The tasks of a wave in the order they are claimed: the most positions first (the late row blocks, whose chunks are the
 * longest), then late blocks, early chunks, low groups. Slots are disjoint, so the order never changes a result. */
static int tiled_compare(const void *a, const void *b) {
  const TiledTask *x = a, *y = b;
  if (x->cost != y->cost)
    return x->cost > y->cost ? -1 : 1;
  if (x->block != y->block)
    return x->block > y->block ? -1 : 1;
  if (x->chunk != y->chunk)
    return x->chunk < y->chunk ? -1 : 1;
  return x->g - y->g;
}

static TiledTask *tiled_tasks(TiledJob *j) {
  int block0 = j->row0 / j->rblock, block1 = (j->row0 + j->rows - 1) / j->rblock, count = 0;
  TiledTask *list = NULL;
  for (int pass = 0; pass < 2; pass++) {
    if (pass) {
      list = malloc((size_t)count * sizeof(*list));
      if (!list)
        fail("out of memory for the attention task list");
      count = 0;
    }
    for (int block = block0; block <= block1; block++) {
      int length, chunks;
      tiled_plan(j, block, &length, &chunks);
      int end = j->pos0 + ((block + 1) * j->rblock < j->m ? (block + 1) * j->rblock : j->m);
      for (int g = 0; g < j->n_kv_heads; g++)
        for (int c = 0; c < chunks; c++) {
          if (pass) {
            int span = (c + 1) * length < end ? length : end - c * length;
            list[count] = (TiledTask){.g = g, .block = block, .chunk = c, .cost = span};
          }
          count++;
        }
    }
  }
  j->tasks = count;
  if (attention_tiled_longest_first)
    qsort(list, (size_t)count, sizeof(*list), tiled_compare);
  return list;
}

static void attention_tiled_driver(float *out, const float *q, const float *key_cache, const float *value_cache,
                                   AttentionWork *work, int n_heads, int n_kv_heads, int head_dim, int pos0, int m,
                                   size_t pos_stride, size_t head_base) {
  int kv_mul = n_heads / n_kv_heads, queries = attention_tiled_queries / kv_mul;
  TiledJob job = {.out = out,
                  .q = q,
                  .key_cache = key_cache,
                  .value_cache = value_cache,
                  .partial = work->partial,
                  .n_heads = n_heads,
                  .n_kv_heads = n_kv_heads,
                  .head_dim = head_dim,
                  .kv_mul = kv_mul,
                  .pos_stride = (int)pos_stride,
                  .head_base = head_base,
                  .dim = n_heads * head_dim,
                  .limit = work->max_chunks,
                  .workers = work->workers,
                  .pos0 = pos0,
                  .m = m,
                  .rblock = queries < 1 ? 1 : queries,
                  .slot = (size_t)head_dim + 2};
  for (int row0 = 0; row0 < m;) {
    /* A wave holds as many rows of partials as the workspace allows. The slots per head follow the plan of the last
     * block of the wave; chunk counts only grow with the position. */
    int rows = m - row0;
    for (;;) {
      job.row0 = row0, job.rows = rows;
      int last_end = ((row0 + rows - 1) / job.rblock + 1) * job.rblock;
      job.chunks_wave = attention_chunks(pos0 + (last_end < m ? last_end : m), n_kv_heads, job.limit, job.workers);
      if (rows == 1 || (size_t)rows * n_heads * job.chunks_wave * job.slot <= work->capacity)
        break;
      rows = (int)(work->capacity / ((size_t)n_heads * job.chunks_wave * job.slot));
      rows = rows < 1 ? 1 : rows;
    }
    job.head_stride = (size_t)job.chunks_wave * job.slot;
    job.row_stride = (size_t)n_heads * job.head_stride;
    TiledTask *order = tiled_tasks(&job);
    job.order = order;
    atomic_init(&job.next_task, 0);
    job.team = current_team;
    if (current_team) {
      team_run(current_team, tiled_team, &job);
    } else {
#ifdef _OPENMP
#pragma omp parallel
#endif
      {
#ifdef _OPENMP
#pragma omp for schedule(dynamic, 1)
#endif
        for (int task = 0; task < job.tasks; task++)
          tiled_run(&job, task, -1);
#ifdef _OPENMP
#pragma omp for
#endif
        for (int item = 0; item < job.rows * n_heads; item++)
          tiled_merge(&job, item);
      }
    }
    free(order);
    row0 += rows;
  }
}
#endif

/* Set when the batched variant was chosen explicitly (LLAMA_ATTN or kernels_attention_select), not by default. */
static int attention_batch_requested;

/* One note per process when `tiled` hands a call to `split`: the precision is the same, only the speed differs. It is
 * an INFO note when tiled was asked for and a DEBUG note when it is only the default (every host without AVX-512). */
static void attention_tiled_fallback(const char *reason) {
  static atomic_flag noted = ATOMIC_FLAG_INIT;
  if (!atomic_flag_test_and_set(&noted))
    LOG(attention_batch_requested ? LOG_INFO : LOG_DEBUG, "LLAMA_ATTN=tiled: %s, running split attention instead",
        reason);
}

static void attention_batch_tiled(float *out, const float *q, const float *key_cache, const float *value_cache,
                                  float *att, AttentionWork *work, int n_heads, int n_kv_heads, int head_dim,
                                  int seq_len, int pos0, int m, size_t pos_stride, size_t head_base) {
  if (m > 1) { /* a single row is the decode path, whatever the variant */
#if HAVE_AVX512
    if (isa != KERNELS_AVX512)
      attention_tiled_fallback("the AVX-512 kernels are not active");
    else if (head_dim > 256 || n_heads / n_kv_heads > TILED_MAX_Q)
      attention_tiled_fallback("head_dim above 256 or more than 16 query heads per KV head is not supported");
    else if (batch_supported(n_heads, n_kv_heads, head_dim, seq_len, pos0, m)) {
      attention_tiled_driver(out, q, key_cache, value_cache, work, n_heads, n_kv_heads, head_dim, pos0, m, pos_stride,
                             head_base);
      return;
    }
#else
    attention_tiled_fallback("the AVX-512 kernels are not built");
#endif
  }
  attention_batch_split(out, q, key_cache, value_cache, att, work, n_heads, n_kv_heads, head_dim, seq_len, pos0, m,
                        pos_stride, head_base);
}

// ----------------------------------------------------------------------------
// Attention variant dispatch (E1)

/* Each public attention entry point forwards to one entry of a table, chosen once by kernels_init from LLAMA_ATTN
 * (batched, prefill) and LLAMA_ATTN_DECODE (single row). A later variant is one new table entry plus its function;
 * run.c, runq.c and tp.c need no change. An entry without a function is a known name that is not built yet: it is
 * refused with a message rather than silently running the default. The first entry is the default: tiled for the
 * batched path since E4 (owner decision, 2026-11-19; split stays selectable with LLAMA_ATTN=split), split for decode. */
typedef void (*AttentionBatchFn)(float *out, const float *q, const float *key_cache, const float *value_cache,
                                 float *att, AttentionWork *work, int n_heads, int n_kv_heads, int head_dim,
                                 int seq_len, int pos0, int m, size_t pos_stride, size_t head_base);
typedef void (*AttentionDecodeFn)(float *out, const float *q, const float *key_cache, const float *value_cache,
                                  float *att, AttentionWork *work, int n_heads, int n_kv_heads, int head_dim,
                                  int seq_len, int position, size_t pos_stride, size_t head_base);
typedef struct {
  const char *name;
  AttentionBatchFn run;
  const char *phase; /* where the variant lands, for the refusal message */
} AttentionBatchEntry;
typedef struct {
  const char *name;
  AttentionDecodeFn run;
  const char *phase;
} AttentionDecodeEntry;

static const AttentionBatchEntry attention_batch_table[] = {
    {"tiled", attention_batch_tiled, NULL},
    {"split", attention_batch_split, NULL},
    {"amx", NULL, "E3"},
};
static const AttentionDecodeEntry attention_decode_table[] = {
    {"split", attention_decode_split, NULL},
    {"group", attention_decode_group, NULL},
};
static const AttentionBatchEntry *attention_batch_active = &attention_batch_table[0];
static const AttentionDecodeEntry *attention_decode_active = &attention_decode_table[0];

int kernels_attention_select(const char *batch, const char *decode) {
  const AttentionBatchEntry *b = NULL;
  const AttentionDecodeEntry *d = NULL;
  for (size_t i = 0; batch && i < sizeof(attention_batch_table) / sizeof(*attention_batch_table); i++)
    if (!strcmp(batch, attention_batch_table[i].name))
      b = &attention_batch_table[i];
  for (size_t i = 0; decode && i < sizeof(attention_decode_table) / sizeof(*attention_decode_table); i++)
    if (!strcmp(decode, attention_decode_table[i].name))
      d = &attention_decode_table[i];
  if ((batch && !(b && b->run)) || (decode && !(d && d->run)))
    return 0;
  if (b) {
    attention_batch_active = b;
    attention_batch_requested = 1;
  }
  if (d)
    attention_decode_active = d;
  return 1;
}

const char *kernels_attention_name(void) {
  return attention_batch_active->name;
}

const char *kernels_attention_decode_name(void) {
  return attention_decode_active->name;
}

/* LLAMA_ATTN / LLAMA_ATTN_DECODE, read once by kernels_init. Unset, empty or "auto" keep the default. */
static void attention_init_from_env(void) {
  const char *batch = getenv("LLAMA_ATTN"), *decode = getenv("LLAMA_ATTN_DECODE");
  attention_batch_active = &attention_batch_table[0];
  attention_decode_active = &attention_decode_table[0];
  attention_batch_requested = 0;
  if (batch && *batch && strcmp(batch, "auto")) {
    const AttentionBatchEntry *found = NULL;
    for (size_t i = 0; i < sizeof(attention_batch_table) / sizeof(*attention_batch_table); i++)
      if (!strcmp(batch, attention_batch_table[i].name))
        found = &attention_batch_table[i];
    if (!found)
      fail("LLAMA_ATTN must be split, tiled or amx");
    if (!found->run)
      fail("LLAMA_ATTN=%s is not implemented yet (%s)", found->name, found->phase);
    attention_batch_active = found;
    attention_batch_requested = 1;
  }
  if (decode && *decode && strcmp(decode, "auto")) {
    const AttentionDecodeEntry *found = NULL;
    for (size_t i = 0; i < sizeof(attention_decode_table) / sizeof(*attention_decode_table); i++)
      if (!strcmp(decode, attention_decode_table[i].name))
        found = &attention_decode_table[i];
    if (!found)
      fail("LLAMA_ATTN_DECODE must be split or group");
    if (!found->run)
      fail("LLAMA_ATTN_DECODE=%s is not implemented yet (%s)", found->name, found->phase);
    attention_decode_active = found;
  }
}

void k_attention(float *out, const float *q, const float *key_cache, const float *value_cache, float *att,
                 AttentionWork *work, int n_heads, int n_kv_heads, int head_dim, int seq_len, int position) {
  attention_decode_active->run(out, q, key_cache, value_cache, att, work, n_heads, n_kv_heads, head_dim, seq_len,
                               position, (size_t)n_kv_heads * head_dim, head_dim);
}

void k_attention_layout(float *out, const float *q, const float *key_cache, const float *value_cache, float *att,
                        AttentionWork *work, int n_heads, int n_kv_heads, int head_dim, int seq_len, int position,
                        size_t pos_stride, size_t head_base) {
  if (pos_stride < (size_t)head_dim || pos_stride > INT_MAX)
    fail("k_attention_layout: position stride %zu is not in [head_dim, INT_MAX]", pos_stride);
  attention_decode_active->run(out, q, key_cache, value_cache, att, work, n_heads, n_kv_heads, head_dim, seq_len,
                               position, pos_stride, head_base);
}

void k_attention_batch(float *out, const float *q, const float *key_cache, const float *value_cache, float *att,
                       AttentionWork *work, int n_heads, int n_kv_heads, int head_dim, int seq_len, int pos0, int m) {
  attention_batch_active->run(out, q, key_cache, value_cache, att, work, n_heads, n_kv_heads, head_dim, seq_len, pos0,
                              m, (size_t)n_kv_heads * head_dim, head_dim);
}

void k_attention_batch_layout(float *out, const float *q, const float *key_cache, const float *value_cache, float *att,
                              AttentionWork *work, int n_heads, int n_kv_heads, int head_dim, int seq_len, int pos0,
                              int m, size_t pos_stride, size_t head_base) {
  if (pos_stride < (size_t)head_dim || pos_stride > INT_MAX)
    fail("k_attention_batch_layout: position stride %zu is not in [head_dim, INT_MAX]", pos_stride);
  attention_batch_active->run(out, q, key_cache, value_cache, att, work, n_heads, n_kv_heads, head_dim, seq_len, pos0,
                              m, pos_stride, head_base);
}

// ----------------------------------------------------------------------------
// K/V cache layout (E5.4): LLAMA_KV_LAYOUT=rows|head

/* head (the default since E5.4): a layer's cache is [kv_head][position][head_dim], each head with room for seq_len
 * positions. rows (the layout before E5.4): [position][kv_head][head_dim]. The cache is the same size and the cache
 * pointer of a layer is the same in both; the results are bitwise equal. */
static int kv_layout_head = 1;

int kernels_kv_layout_head(void) {
  return kv_layout_head;
}

const char *kernels_kv_layout_name(void) {
  return kv_layout_head ? "head" : "rows";
}

int kernels_kv_layout_select(const char *name) {
  if (!strcmp(name, "rows"))
    kv_layout_head = 0;
  else if (!strcmp(name, "head"))
    kv_layout_head = 1;
  else
    return 0;
  return 1;
}

static void kv_layout_init_from_env(void) {
  const char *layout = getenv("LLAMA_KV_LAYOUT");
  kv_layout_head = 1; /* unset, empty and auto are head */
  if (layout && *layout && strcmp(layout, "auto") && !kernels_kv_layout_select(layout))
    fail("LLAMA_KV_LAYOUT must be rows or head");
}

void k_kv_strides(int n_kv_heads, int head_dim, int seq_len, size_t *pos_stride, size_t *head_base) {
  *pos_stride = kv_layout_head ? (size_t)head_dim : (size_t)n_kv_heads * head_dim;
  *head_base = kv_layout_head ? (size_t)seq_len * head_dim : (size_t)head_dim;
}

void k_kv_store(float *cache, const float *src, size_t src_ld, int pos0, int count, int n_kv_heads, int head_dim,
                int seq_len) {
  size_t kv_dim = (size_t)n_kv_heads * head_dim;
  if (!kv_layout_head) {
    for (int r = 0; r < count; r++)
      memcpy(cache + (size_t)(pos0 + r) * kv_dim, src + (size_t)r * src_ld, kv_dim * sizeof(float));
    return;
  }
  for (int r = 0; r < count; r++)
    for (int g = 0; g < n_kv_heads; g++)
      memcpy(cache + (size_t)g * seq_len * head_dim + (size_t)(pos0 + r) * head_dim,
             src + (size_t)r * src_ld + (size_t)g * head_dim, (size_t)head_dim * sizeof(float));
}

void k_attention_kv(float *out, const float *q, const float *key_cache, const float *value_cache, float *att,
                    AttentionWork *work, int n_heads, int n_kv_heads, int head_dim, int seq_len, int position) {
  if (!kv_layout_head) {
    k_attention(out, q, key_cache, value_cache, att, work, n_heads, n_kv_heads, head_dim, seq_len, position);
    return;
  }
  k_attention_layout(out, q, key_cache, value_cache, att, work, n_heads, n_kv_heads, head_dim, seq_len, position,
                     (size_t)head_dim, (size_t)seq_len * head_dim);
}

void k_attention_batch_kv(float *out, const float *q, const float *key_cache, const float *value_cache, float *att,
                          AttentionWork *work, int n_heads, int n_kv_heads, int head_dim, int seq_len, int pos0,
                          int m) {
  if (!kv_layout_head) {
    k_attention_batch(out, q, key_cache, value_cache, att, work, n_heads, n_kv_heads, head_dim, seq_len, pos0, m);
    return;
  }
  k_attention_batch_layout(out, q, key_cache, value_cache, att, work, n_heads, n_kv_heads, head_dim, seq_len, pos0, m,
                           (size_t)head_dim, (size_t)seq_len * head_dim);
}

// ----------------------------------------------------------------------------
// Execution backend

static int backend_started, backend_used;

const char *kernels_backend_name(void) {
  return backend_used ? "teams" : "omp";
}

void kernels_backend_init(void) {
  int sockets = tp_sockets_requested(); /* LLAMA_SOCKETS=2 without LLAMA_BACKEND=teams fails here */
  /* Discover the topology now, from the job's full allowed sets, before any worker is pinned: the report
   * reads this cached result after the run. A failure is cached too and reported, not fatal here. */
  const Topology *topology_result;
  char topology_error[512];
  numa_topology_try(&topology_result, topology_error, sizeof(topology_error));
  const char *backend = getenv("LLAMA_BACKEND");
  if (!backend || !*backend || !strcmp(backend, "omp"))
    return;
  if (strcmp(backend, "teams"))
    fail("LLAMA_BACKEND must be omp or teams");
  if (backend_started)
    return;
#ifdef _OPENMP
  LOG(LOG_INFO, "LLAMA_BACKEND=teams: OpenMP regions are unused");
#endif
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
    backend_started = backend_used = 1;
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
  const char *bind = getenv("OMP_PROC_BIND");
  if (!(topology && *topology) && available == 1 && bind && *bind && strcmp(bind, "false") && strcmp(bind, "FALSE"))
    LOG(LOG_WARN,
        "only one CPU is allowed and OMP_PROC_BIND=%s is set: the OpenMP runtime pins the launching thread to one "
        "core at load. Unset it (or set OMP_PROC_BIND=false) for LLAMA_BACKEND=teams",
        bind);
  if (!threads)
    threads = available;
  if (cpus && threads > available) {
    LOG(LOG_WARN, "LLAMA_THREADS=%d exceeds the %d allowed CPUs: workers are not pinned", threads, available);
    cpus = NULL;
  }
  PoolConfig config = {.teams = 1, .threads = {threads, 0}, .cpus = {cpus, NULL}};
  pool_start(&config);
  kernels_set_team(pool_team(0));
  backend_started = backend_used = 1;
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
