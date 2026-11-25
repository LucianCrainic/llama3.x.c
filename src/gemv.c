/* GEMV kernels: the FP32 and Q8_0 row-dot kernels (scalar and AVX-512), their team drivers (static and hybrid
 * splits, fused QKV and SwiGLU, worker entry points for tp.c) and activation quantization. Moved out of
 * kernels_simd.c unchanged (F8). */
#include "kernels_internal.h"
#include "testing.h"
#include <math.h>
#include <stdatomic.h>
#include <stdlib.h>
#include <string.h>

/* How a team splits the rows of one GEMV. Static: each member takes one contiguous share, so one slow member (a memory
 * channel under contention, an interrupt) holds the whole team at the join. Dynamic: members claim chunks of rows from a
 * shared counter, so the fast ones take over what the slow one has not reached, but no member streams a long contiguous
 * run any more. Hybrid: each member first takes a contiguous share of three quarters of the rows, then all claim the
 * rest in small chunks. A claim starts on a multiple of BLOCK rows, so every row sits in the same four-row group
 * whatever the team size or the timing. FP32 runs hybrid for large jobs and static otherwise; Q8 always runs static, as
 * its kernels prefetch along long contiguous runs and lose 6-25% to any claiming (measured on Xeon Max 9480, docs/plans).
 * Dynamic, and the LLAMA_GEMV_SCHED knob that forced one policy everywhere, were removed after the sweep (F5). */
typedef enum { SCHED_STATIC, SCHED_HYBRID } GemvSched;
/* FP32: hybrid for a job that streams at least this many weight bytes (the 8B shards: 117-235 MiB; measured +14% on two
 * sockets), static below it (1B: at most 64 MiB, where the claims cost 4-6%; Wo shards: 33 MiB). */
enum { AUTO_HYBRID_BYTES = 96 << 20 };
static int gemv_force_hybrid; /* test hook: run every FP32 team GEMV hybrid, so small fixtures reach the claiming path */

void k_gemv_force_hybrid(int on) {
  gemv_force_hybrid = on != 0;
}

static GemvSched effective_f32(size_t bytes) {
  return gemv_force_hybrid || bytes >= AUTO_HYBRID_BYTES ? SCHED_HYBRID : SCHED_STATIC;
}

#if HAVE_AVX512
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

/* W8A8 with VNNI at gs = 32 (the Q8_0 GGUF block): one 64-byte load spans a pair of groups, so int32 lanes 0-7 hold
 * group g's dot and lanes 8-15 group g + 1's. The scales are spread the same way (two adjacent floats, lanes 0-7 and
 * 8-15), and each lane is rescaled as in dots_q8_body. n % 64 == 0. */
Q8_INLINE void dots_q8_pair_body(const int8_t *wq, const float *ws, size_t n, size_t ld, const int rows,
                                 const int8_t *xq, const float *xs, float *out) {
  const __m512i zero = _mm512_setzero_si512();
  const size_t ahead = q8_ahead(rows, n, 0);
  __m512 acc[BLOCK];
  for (int r = 0; r < rows; r++)
    acc[r] = _mm512_setzero_ps();
  size_t scale_ld = ld / 32;
#define Q8_PAIR(p) _mm512_mask_mov_ps(_mm512_set1_ps((p)[0]), 0xff00, _mm512_set1_ps((p)[1]))
  for (size_t off = 0, g = 0; off < n; off += 64, g += 2) {
    __m512i xv = _mm512_loadu_si512((const void *)(xq + off));
    __m512i ax = _mm512_abs_epi8(xv);
    __mmask64 negative = _mm512_movepi8_mask(xv);
    __m512 xscale = Q8_PAIR(xs + g);
    for (int r = 0; r < rows; r++) {
      _mm_prefetch((const char *)(wq + r * ld + off + ahead), _MM_HINT_T0);
      __m512i wv = _mm512_loadu_si512((const void *)(wq + r * ld + off));
      __m512i dot = _mm512_dpbusd_epi32(zero, ax, _mm512_mask_sub_epi8(wv, negative, zero, wv));
      acc[r] = _mm512_fmadd_ps(_mm512_mul_ps(_mm512_cvtepi32_ps(dot), xscale), Q8_PAIR(ws + r * scale_ld + g), acc[r]);
    }
  }
#undef Q8_PAIR
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
  AVX512 static void dots_q8p_avx512_##rows(const int8_t *wq, const float *ws, size_t n, size_t ld, const int8_t *xq,  \
                                            const float *xs, int gs, float *out) {                                     \
    (void)gs;                                                                                                          \
    dots_q8_pair_body(wq, ws, n, ld, rows, xq, xs, out);                                                               \
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
static const DotsQ8Fn dots_q8p_avx512[BLOCK + 1] = {NULL, dots_q8p_avx512_1, dots_q8p_avx512_2, dots_q8p_avx512_3,
                                                    dots_q8p_avx512_4};
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

#endif

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
  if (kernels_isa == KERNELS_AVX512) {
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
  atomic_int next; /* hybrid scheduling: the next unclaimed row (zero before the team starts) */
  RowsDoneFn done; /* worker entry points only (one segment): called after each range of rows is stored */
  void *done_ctx;
} F32Job;

static void f32_range(const F32Job *job, const F32Segment *seg, int r0, int r1) {
  float dots[BLOCK], gates[BLOCK];
  size_t ld = job->ld ? job->ld : (size_t)job->n;
  for (int i = r0; i < r1; i += BLOCK) {
    int rows = r1 - i < BLOCK ? r1 - i : BLOCK;
    f32_dots(seg->w + (size_t)i * ld, ld, rows, job->x, job->n, dots);
    if (job->kind == SWIGLU)
      f32_dots(seg->w3 + (size_t)i * ld, ld, rows, job->x, job->n, gates);
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
    if (r1 > r0)
      f32_range(job, seg, r0, r1);
  }
}

/* The hybrid split over the concatenated rows of all segments; `next` is shared by the members of one job.
 * `range(job, r0, r1)` runs rows [r0, r1) of that concatenation. */
typedef void (*RangeFn)(const void *job, int r0, int r1);

static void hybrid_rows_run(RangeFn range, const void *job, int rows, atomic_int *next, int worker, int workers) {
  int pool = rows * 3 / 4 / BLOCK * BLOCK; /* the contiguous shares */
  int r0 = (int)((long long)pool * worker / workers) / BLOCK * BLOCK;
  int r1 = worker + 1 == workers ? pool : (int)((long long)pool * (worker + 1) / workers) / BLOCK * BLOCK;
  if (r1 > r0)
    range(job, r0, r1);
  int first = pool, chunk = (rows - pool) / (workers * 2) / BLOCK * BLOCK;
  chunk = chunk < BLOCK ? BLOCK : chunk > 16 ? 16 : chunk;
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
  size_t row = job->ld ? job->ld : (size_t)job->n;
  return (size_t)job_rows_f32(job) * row * sizeof(float) * (job->kind == SWIGLU ? 2 : 1);
}

static void f32_team_claim(const F32Job *job, atomic_int *next, int worker, int workers) {
  hybrid_rows_run(f32_rows, job, job_rows_f32(job), next, worker, workers);
}

static void f32_team(const void *p, int worker, int workers) {
  F32Job *job = (F32Job *)p; /* the caller's job object is writable: only `next` changes */
  if (effective_f32(job_bytes_f32(job)) == SCHED_HYBRID && workers > 1)
    f32_team_claim(job, &job->next, worker, workers);
  else
    f32_team_static(job, worker, workers);
}

static void f32_run(F32Job *job) {
  if (current_team) {
    team_run(current_team, f32_team, job);
    return;
  }
  parallel_rows(job_rows_f32(job), f32_rows, job);
}

void k_matmul(float *out, const float *x, const float *w, int n, int d) {
  // W (d,n) @ x (n,) -> xout (d,)
  F32Job job = {x, n, STORE, 1, {{out, w, NULL, d}}, 0, 0, NULL, NULL};
  f32_run(&job);
}
void k_matmul_add(float *out, const float *x, const float *w, int n, int d) {
  F32Job job = {x, n, ADD, 1, {{out, w, NULL, d}}, 0, 0, NULL, NULL};
  f32_run(&job);
}
void k_matmul_qkv(float *q, float *k, float *v, const float *x, const float *wq, const float *wk, const float *wv, int n,
                  int dq, int dkv) {
  F32Job job = {x, n, STORE, 3, {{q, wq, NULL, dq}, {k, wk, NULL, dkv}, {v, wv, NULL, dkv}}, 0, 0, NULL, NULL};
  f32_run(&job);
}
void k_matmul_swiglu(float *out, const float *x, const float *w1, const float *w3, int n, int d) {
  F32Job job = {x, n, SWIGLU, 1, {{out, w1, w3, d}}, 0, 0, NULL, NULL};
  f32_run(&job);
}

// ----------------------------------------------------------------------------
// Q8_0 drivers

static void q8_dots(K8Weight w, size_t ld, int row, int rows, const K8Input *x, int n, int gs, float *out) {
  size_t scale_ld = ld / gs;
  const int8_t *wq = w.q + (size_t)row * ld;
  const float *ws = w.s + (size_t)row * scale_ld;
#if HAVE_AVX512
  if (kernels_isa == KERNELS_AVX512) {
    if (x->q && gs % 64 == 0) {
      dots_q8_avx512[rows](wq, ws, (size_t)n, ld, x->q, x->s, gs, out);
      return;
    }
    if (x->q && gs == 32 && n % 64 == 0) {
      dots_q8p_avx512[rows](wq, ws, (size_t)n, ld, x->q, x->s, gs, out);
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
  RowsDoneFn done;
  void *done_ctx;
} Q8Job;

/* The row loop of one range, from block `rot` (counted from r0) round to block rot - 1. Every block is the same BLOCK
 * rows at the same offset from r0 as in the unrotated loop and writes its own outputs, so the results are bitwise those of
 * rot = 0; only the order in which the blocks are read changes. */
static void q8_blocks(const Q8Job *job, const Q8Segment *seg, int r0, int r1, int rot) {
  float dots[BLOCK], gates[BLOCK];
  size_t ld = job->ld ? job->ld : (size_t)job->n;
  int blocks = (r1 - r0 + BLOCK - 1) / BLOCK;
  for (int k = 0; k < blocks; k++) {
    int i = r0 + (k + rot < blocks ? k + rot : k + rot - blocks) * BLOCK;
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

static void q8_range(const Q8Job *job, const Q8Segment *seg, int r0, int r1) {
  q8_blocks(job, seg, r0, r1, 0);
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

/* LLAMA_GEMV_STAGGER=1: member w of a team starts its share at block w (of BLOCK rows) and wraps, so simultaneous
 * members read addresses that differ by w blocks (32 KiB at the 8192-byte rows of the 70B shard) instead of by multiples of
 * their share. On 2 MiB pages shares of exactly 2 MiB put every member on the same stripe of the HBM address
 * interleave in barrier lockstep (docs/concurrent-tier-bandwidth.md 5.1, docs/gemv-stagger.md). Bitwise unchanged. */
static int gemv_stagger;

void gemv_init_env(void) {
  const char *value = getenv("LLAMA_GEMV_STAGGER");
  gemv_stagger = 0;
  if (value && *value && strcmp(value, "0")) {
    if (strcmp(value, "1"))
      fail("LLAMA_GEMV_STAGGER must be 0 or 1");
    gemv_stagger = 1;
  }
}
void k_gemv_stagger(int on) {
  gemv_stagger = on != 0;
}
int k_gemv_stagger_active(void) {
  return gemv_stagger;
}

static void q8_team_static(const Q8Job *job, int worker, int workers) {
  for (int s = 0; s < job->count; s++) {
    const Q8Segment *seg = &job->segment[s];
    int r0 = (int)((long long)seg->d * worker / workers), r1 = (int)((long long)seg->d * (worker + 1) / workers);
    if (r1 > r0) {
      int blocks = (r1 - r0 + BLOCK - 1) / BLOCK;
      q8_blocks(job, seg, r0, r1, gemv_stagger ? worker % blocks : 0);
    }
  }
}

static void q8_team(const void *p, int worker, int workers) {
  q8_team_static((const Q8Job *)p, worker, workers);
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
  Q8Job job = {x, n, gs, STORE, 1, {{out, w, {0}, d}}, 0, NULL, NULL};
  q8_run(&job);
}
void k_matmul_q8_add(float *out, const K8Input *x, K8Weight w, int n, int d, int gs) {
  Q8Job job = {x, n, gs, ADD, 1, {{out, w, {0}, d}}, 0, NULL, NULL};
  q8_run(&job);
}
void k_matmul_q8_qkv(float *q, float *k, float *v, const K8Input *x, K8Weight wq, K8Weight wk, K8Weight wv, int n,
                     int dq, int dkv, int gs) {
  Q8Job job = {x, n, gs, STORE, 3, {{q, wq, {0}, dq}, {k, wk, {0}, dkv}, {v, wv, {0}, dkv}}, 0, NULL, NULL};
  q8_run(&job);
}
void k_matmul_q8_swiglu(float *out, const K8Input *x, K8Weight w1, K8Weight w3, int n, int d, int gs) {
  Q8Job job = {x, n, gs, SWIGLU, 1, {{out, w1, w3, d}}, 0, NULL, NULL};
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
  F32Job job = {x, n, kind, 1, {{out, w, NULL, d}}, ld, 0, NULL, NULL};
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
  F32Job job = {x, n, kind, 1, {{out, w, NULL, d}}, ld, 0, done, done_ctx};
  if (effective_f32(job_bytes_f32(&job)) == SCHED_HYBRID && claim && workers > 1)
    f32_team_claim(&job, claim, worker, workers);
  else
    f32_team_static(&job, worker, workers);
}
void k_matmul_swiglu_ld(float *out, const float *x, const float *w1, const float *w3, size_t ld, int n, int d) {
  if (ld < (size_t)n)
    fail("k_matmul_swiglu_ld: row stride %zu is shorter than the %d inputs", ld, n);
  F32Job job = {x, n, SWIGLU, 1, {{out, w1, w3, d}}, ld, 0, NULL, NULL};
  f32_run(&job);
}
void k_matmul_q8_ld(float *out, const K8Input *x, const K8Weight *w, size_t ld, int n, int d, int gs, int kind) {
  check_epilogue(kind, "k_matmul_q8_ld");
  if (ld < (size_t)n || ld % (size_t)gs)
    fail("k_matmul_q8_ld: row stride %zu must be at least %d and a multiple of the group size %d", ld, n, gs);
  Q8Job job = {x, n, gs, kind, 1, {{out, *w, {0}, d}}, ld, NULL, NULL};
  q8_run(&job);
}
void k_matmul_q8_ld_worker(float *out, const K8Input *x, const K8Weight *w, size_t ld, int n, int d, int gs, int kind,
                           int worker, int workers, atomic_int *claim, RowsDoneFn done, void *done_ctx) {
  check_epilogue(kind, "k_matmul_q8_ld_worker");
  if (ld < (size_t)n || ld % (size_t)gs)
    fail("k_matmul_q8_ld_worker: row stride %zu must be at least %d and a multiple of the group size %d", ld, n, gs);
  Q8Job job = {x, n, gs, kind, 1, {{out, *w, {0}, d}}, ld, done, done_ctx};
  (void)claim; /* Q8 always splits statically (see GemvSched) */
  q8_team_static(&job, worker, workers);
}
void k_matmul_q8_swiglu_ld(float *out, const K8Input *x, const K8Weight *w1, const K8Weight *w3, size_t ld, int n,
                           int d, int gs) {
  if (ld < (size_t)n || ld % (size_t)gs)
    fail("k_matmul_q8_swiglu_ld: row stride %zu must be at least %d and a multiple of the group size %d", ld, n, gs);
  Q8Job job = {x, n, gs, SWIGLU, 1, {{out, *w1, *w3, d}}, ld, NULL, NULL};
  q8_run(&job);
}

void k_quantize(int8_t *q, float *s, const float *x, int n, int gs) {
#if HAVE_AVX512
  if (kernels_isa == KERNELS_AVX512 && gs % 16 == 0) {
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
