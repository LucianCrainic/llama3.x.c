/* Compute kernels: runtime ISA dispatch, scalar references and AVX-512 paths.
 * Scalar paths reproduce the original per-engine loops exactly. OpenMP regions
 * live only in plain functions; ISA-specific work runs in target-attributed
 * helpers called once per row block, so no outlined region needs AVX-512. */
#include "kernels.h"
#include <float.h>
#include <math.h>
#include <stdlib.h>
#include <string.h>
#ifdef _OPENMP
#include <omp.h>
#endif

#if defined(__x86_64__) && (defined(__GNUC__) || defined(__clang__))
#define HAVE_AVX512 1
#include <immintrin.h>
#define AVX512 __attribute__((target("avx512f,avx512bw,avx512dq,avx512vl,avx512vnni,fma")))
#else
#define HAVE_AVX512 0
#endif

static KernelIsa isa = KERNELS_SCALAR;
static int initialized;

static int supported(KernelIsa which) {
  if (which == KERNELS_SCALAR)
    return 1;
#if HAVE_AVX512
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
  isa = which;
  initialized = 1;
  return 1;
}

void kernels_init(void) {
  const char *request = getenv("LLAMA_KERNELS");
  if (!request || !*request || !strcmp(request, "auto")) {
    kernels_select(supported(KERNELS_AVX512) ? KERNELS_AVX512 : KERNELS_SCALAR);
  } else if (!strcmp(request, "scalar")) {
    kernels_select(KERNELS_SCALAR);
  } else if (!strcmp(request, "avx512")) {
    if (!kernels_select(KERNELS_AVX512))
      fail("LLAMA_KERNELS=avx512 requested, but AVX-512 F/BW/DQ/VL/VNNI is unavailable");
  } else {
    fail("LLAMA_KERNELS must be auto, scalar or avx512");
  }
  LOG(LOG_DEBUG, "compute kernels: %s", kernels_name());
}

const char *kernels_name(void) {
  if (!initialized)
    return NULL;
  return isa == KERNELS_AVX512 ? "avx512" : "scalar";
}

enum { BLOCK = 4 }; // rows per GEMV kernel call

/* Each thread receives one contiguous, balanced row range, so weight rows are
 * streamed sequentially per thread and hardware prefetch stays effective. */
typedef void (*RowFn)(const void *job, int r0, int r1);
static void parallel_rows(int rows, RowFn fn, const void *job) {
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
// AVX-512 helpers

#if HAVE_AVX512
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

/* The Q8 bodies take a compile-time row count and are force-inlined into one
 * wrapper per count. Fixed-size accumulator arrays then live in registers;
 * with a runtime count they were kept on the stack, and every VNNI and FMA step
 * went through a store and a reload. */
#define Q8_INLINE AVX512 static inline __attribute__((always_inline))

/* W8A8 with VNNI. vpdpbusd multiplies unsigned by signed bytes, so |x| is the
 * unsigned operand and w takes x's sign; exact because no value is -128. Each
 * group's int32 lanes are scaled once into FP32 accumulators. gs % 64 == 0. */
Q8_INLINE void dots_q8_body(const int8_t *wq, const float *ws, size_t n, const int rows, const int8_t *xq,
                            const float *xs, int gs, float *out) {
  const __m512i zero = _mm512_setzero_si512();
  __m512 acc[BLOCK];
  for (int r = 0; r < rows; r++)
    acc[r] = _mm512_setzero_ps();
  size_t groups = n / gs;
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
        __m512i wv = _mm512_loadu_si512((const void *)(wq + r * n + off));
        iacc[r] = _mm512_dpbusd_epi32(iacc[r], ax, _mm512_mask_sub_epi8(wv, negative, zero, wv));
      }
    }
    __m512 xscale = _mm512_set1_ps(xs[g]);
    for (int r = 0; r < rows; r++)
      acc[r] = _mm512_fmadd_ps(_mm512_mul_ps(_mm512_cvtepi32_ps(iacc[r]), xscale), _mm512_set1_ps(ws[r * groups + g]),
                               acc[r]);
  }
  for (int r = 0; r < rows; r++)
    out[r] = _mm512_reduce_add_ps(acc[r]);
}

/* W8A32: weights are widened in registers and multiplied by FP32 activations;
 * each group's partial sum is scaled once. Two partial sums per row hide FMA
 * latency. gs % 16 == 0. */
Q8_INLINE void dots_q8f_body(const int8_t *wq, const float *ws, size_t n, const int rows, const float *x, int gs,
                             float *out) {
  __m512 acc[BLOCK];
  for (int r = 0; r < rows; r++)
    acc[r] = _mm512_setzero_ps();
  size_t groups = n / gs;
  for (size_t g = 0; g < groups; g++) {
    __m512 ga[BLOCK], gb[BLOCK];
    for (int r = 0; r < rows; r++)
      ga[r] = gb[r] = _mm512_setzero_ps();
    int c = 0;
    for (; c + 32 <= gs; c += 32) {
      size_t off = g * gs + c;
      __m512 x0 = _mm512_loadu_ps(x + off), x1 = _mm512_loadu_ps(x + off + 16);
      for (int r = 0; r < rows; r++) {
        __m256i bytes = _mm256_loadu_si256((const __m256i *)(wq + r * n + off));
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
        __m128i bytes = _mm_loadu_si128((const __m128i *)(wq + r * n + off));
        ga[r] = _mm512_fmadd_ps(_mm512_cvtepi32_ps(_mm512_cvtepi8_epi32(bytes)), x0, ga[r]);
      }
    }
    for (int r = 0; r < rows; r++)
      acc[r] = _mm512_fmadd_ps(_mm512_add_ps(ga[r], gb[r]), _mm512_set1_ps(ws[r * groups + g]), acc[r]);
  }
  for (int r = 0; r < rows; r++)
    out[r] = _mm512_reduce_add_ps(acc[r]);
}

#define Q8_WRAPPERS(rows)                                                                                            \
  AVX512 static void dots_q8_avx512_##rows(const int8_t *wq, const float *ws, size_t n, const int8_t *xq,           \
                                           const float *xs, int gs, float *out) {                                  \
    dots_q8_body(wq, ws, n, rows, xq, xs, gs, out);                                                                 \
  }                                                                                                                 \
  AVX512 static void dots_q8f_avx512_##rows(const int8_t *wq, const float *ws, size_t n, const float *x, int gs,    \
                                            float *out) {                                                           \
    dots_q8f_body(wq, ws, n, rows, x, gs, out);                                                                     \
  }
Q8_WRAPPERS(1)
Q8_WRAPPERS(2)
Q8_WRAPPERS(3)
Q8_WRAPPERS(4)
#undef Q8_WRAPPERS

typedef void (*DotsQ8Fn)(const int8_t *, const float *, size_t, const int8_t *, const float *, int, float *);
typedef void (*DotsQ8fFn)(const int8_t *, const float *, size_t, const float *, int, float *);
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
                                          int heads, int t0, int t1, float *att, int seq_len, float *partial,
                                          size_t head_stride) {
  enum { MAX_VECTORS = 16 };
  int vectors = (head_dim + 15) / 16;
  __mmask16 last = tail_mask(head_dim - (vectors - 1) * 16);
  float divisor = sqrtf(head_dim);
  const __m512 abs_mask = _mm512_castsi512_ps(_mm512_set1_epi32(0x7fffffff)), finite = _mm512_set1_ps(FLT_MAX);
  for (int h = 0; h < heads; h++) {
    __m512 qv[MAX_VECTORS];
    for (int v = 0; v < vectors; v++)
      qv[v] = _mm512_maskz_loadu_ps(v + 1 == vectors ? last : 0xFFFF, q + (size_t)h * head_dim + v * 16);
    float *a = att + (size_t)h * seq_len;
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
      a[t] = _mm512_reduce_add_ps(s0) / divisor;
      a[t + 1] = _mm512_reduce_add_ps(s1) / divisor;
      a[t + 2] = _mm512_reduce_add_ps(s2) / divisor;
      a[t + 3] = _mm512_reduce_add_ps(s3) / divisor;
    }
    for (; t < t1; t++) {
      const float *k0 = kc + (size_t)t * kv_dim;
      __m512 s0 = _mm512_setzero_ps();
      for (int v = 0; v < vectors; v++)
        s0 = _mm512_fmadd_ps(qv[v], _mm512_maskz_loadu_ps(v + 1 == vectors ? last : 0xFFFF, k0 + v * 16), s0);
      a[t] = _mm512_reduce_add_ps(s0) / divisor;
    }
    __m512 vmax = _mm512_set1_ps(-FLT_MAX);
    for (t = t0; t < t1; t += 16) {
      __mmask16 m = tail_mask(t1 - t);
      __m512 s = _mm512_maskz_loadu_ps(m, a + t);
      if (_mm512_mask_cmp_ps_mask(m, _mm512_and_ps(s, abs_mask), finite, _CMP_LE_OQ) != m)
        fail("nonfinite value in softmax");
      vmax = _mm512_mask_max_ps(vmax, m, vmax, s);
    }
    float max_val = _mm512_reduce_max_ps(vmax);
    __m512 vsum = _mm512_setzero_ps(), shift = _mm512_set1_ps(max_val);
    for (t = t0; t < t1; t += 16) {
      __mmask16 m = tail_mask(t1 - t);
      __m512 e = exp512(_mm512_sub_ps(_mm512_maskz_loadu_ps(m, a + t), shift));
      _mm512_mask_storeu_ps(a + t, m, e);
      vsum = _mm512_mask_add_ps(vsum, m, vsum, e);
    }
    __m512 ov[MAX_VECTORS];
    for (int v = 0; v < vectors; v++)
      ov[v] = _mm512_setzero_ps();
    for (t = t0; t < t1; t++) {
      const float *vr = vc + (size_t)t * kv_dim;
      __m512 p = _mm512_set1_ps(a[t]);
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

// ----------------------------------------------------------------------------
// GEMV drivers

enum { STORE, ADD, SWIGLU };

static float silu_gate(float value, float gate) {
  // silu(x)=x*σ(x), where σ(x) is the logistic sigmoid; elementwise multiply with w3(x)
  value *= (1.0f / (1.0f + expf(-value)));
  return value * gate;
}

static void f32_dots(const float *w, int rows, const float *x, int n, float *out) {
#if HAVE_AVX512
  if (isa == KERNELS_AVX512) {
    dots_f32_avx512(w, (size_t)n, rows, x, n, out);
    return;
  }
#endif
  for (int r = 0; r < rows; r++) {
    const float *wr = w + (size_t)r * n;
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
} F32Job;

static void f32_range(const F32Job *job, const F32Segment *seg, int r0, int r1) {
  float dots[BLOCK], gates[BLOCK];
  for (int i = r0; i < r1; i += BLOCK) {
    int rows = r1 - i < BLOCK ? r1 - i : BLOCK;
    f32_dots(seg->w + (size_t)i * job->n, rows, job->x, job->n, dots);
    if (job->kind == SWIGLU)
      f32_dots(seg->w3 + (size_t)i * job->n, rows, job->x, job->n, gates);
    for (int r = 0; r < rows; r++) {
      if (job->kind == ADD)
        seg->out[i + r] += dots[r];
      else if (job->kind == SWIGLU)
        seg->out[i + r] = silu_gate(dots[r], gates[r]);
      else
        seg->out[i + r] = dots[r];
    }
  }
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

static void f32_run(F32Job *job) {
  int rows = 0;
  for (int s = 0; s < job->count; s++)
    rows += job->segment[s].d;
  parallel_rows(rows, f32_rows, job);
}

void k_matmul(float *out, const float *x, const float *w, int n, int d) {
  // W (d,n) @ x (n,) -> xout (d,)
  F32Job job = {x, n, STORE, 1, {{out, w, NULL, d}}};
  f32_run(&job);
}
void k_matmul_add(float *out, const float *x, const float *w, int n, int d) {
  F32Job job = {x, n, ADD, 1, {{out, w, NULL, d}}};
  f32_run(&job);
}
void k_matmul_qkv(float *q, float *k, float *v, const float *x, const float *wq, const float *wk, const float *wv, int n,
                  int dq, int dkv) {
  F32Job job = {x, n, STORE, 3, {{q, wq, NULL, dq}, {k, wk, NULL, dkv}, {v, wv, NULL, dkv}}};
  f32_run(&job);
}
void k_matmul_swiglu(float *out, const float *x, const float *w1, const float *w3, int n, int d) {
  F32Job job = {x, n, SWIGLU, 1, {{out, w1, w3, d}}};
  f32_run(&job);
}

// ----------------------------------------------------------------------------
// Q8_0 drivers

static void q8_dots(K8Weight w, int row, int rows, const K8Input *x, int n, int gs, float *out) {
  size_t groups = (size_t)n / gs;
  const int8_t *wq = w.q + (size_t)row * n;
  const float *ws = w.s + (size_t)row * groups;
#if HAVE_AVX512
  if (isa == KERNELS_AVX512) {
    if (x->q && gs % 64 == 0) {
      dots_q8_avx512[rows](wq, ws, (size_t)n, x->q, x->s, gs, out);
      return;
    }
    if (!x->q && gs % 16 == 0) {
      dots_q8f_avx512[rows](wq, ws, (size_t)n, x->f, gs, out);
      return;
    }
  }
#endif
  for (int r = 0; r < rows; r++) {
    const int8_t *wr = wq + (size_t)r * n;
    const float *sr = ws + (size_t)r * groups;
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
} Q8Job;

static void q8_range(const Q8Job *job, const Q8Segment *seg, int r0, int r1) {
  float dots[BLOCK], gates[BLOCK];
  for (int i = r0; i < r1; i += BLOCK) {
    int rows = r1 - i < BLOCK ? r1 - i : BLOCK;
    q8_dots(seg->w, i, rows, job->x, job->n, job->gs, dots);
    if (job->kind == SWIGLU)
      q8_dots(seg->w3, i, rows, job->x, job->n, job->gs, gates);
    for (int r = 0; r < rows; r++) {
      if (job->kind == ADD)
        seg->out[i + r] += dots[r];
      else if (job->kind == SWIGLU)
        seg->out[i + r] = silu_gate(dots[r], gates[r]);
      else
        seg->out[i + r] = dots[r];
    }
  }
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

static void q8_run(Q8Job *job) {
  int rows = 0;
  for (int s = 0; s < job->count; s++)
    rows += job->segment[s].d;
  parallel_rows(rows, q8_rows, job);
}

void k_matmul_q8(float *out, const K8Input *x, K8Weight w, int n, int d, int gs) {
  Q8Job job = {x, n, gs, STORE, 1, {{out, w, {0}, d}}};
  q8_run(&job);
}
void k_matmul_q8_add(float *out, const K8Input *x, K8Weight w, int n, int d, int gs) {
  Q8Job job = {x, n, gs, ADD, 1, {{out, w, {0}, d}}};
  q8_run(&job);
}
void k_matmul_q8_qkv(float *q, float *k, float *v, const K8Input *x, K8Weight wq, K8Weight wk, K8Weight wv, int n,
                     int dq, int dkv, int gs) {
  Q8Job job = {x, n, gs, STORE, 3, {{q, wq, {0}, dq}, {k, wk, {0}, dkv}, {v, wv, {0}, dkv}}};
  q8_run(&job);
}
void k_matmul_q8_swiglu(float *out, const K8Input *x, K8Weight w1, K8Weight w3, int n, int d, int gs) {
  Q8Job job = {x, n, gs, SWIGLU, 1, {{out, w1, w3, d}}};
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
#ifdef _OPENMP
  return omp_get_max_threads();
#else
  return 1;
#endif
}

#if HAVE_AVX512
/* Enough chunks to occupy every thread, each of 32..256 positions where the
 * prefix allows, bounded by the workspace. */
static int attention_chunks(int positions, int n_kv_heads, int limit) {
  int chunks = (max_threads() + n_kv_heads - 1) / n_kv_heads;
  int longest = (positions + 31) / 32, shortest = (positions + 255) / 256;
  if (chunks > longest)
    chunks = longest;
  if (chunks < shortest)
    chunks = shortest;
  if (chunks > limit)
    chunks = limit;
  return chunks < 1 ? 1 : chunks;
}
#endif

void k_attention_init(AttentionWork *work, int n_heads, int n_kv_heads, int head_dim, int seq_len) {
  int by_threads = (max_threads() + n_kv_heads - 1) / n_kv_heads, by_length = (seq_len + 255) / 256;
  work->max_chunks = by_threads > by_length ? by_threads : by_length;
  work->n_heads = n_heads;
  work->head_dim = head_dim;
  work->partial = memory_alloc(checked_mul(checked_mul((size_t)n_heads, (size_t)work->max_chunks), (size_t)head_dim + 2),
                               sizeof(float), MEM_ACTIVATION);
}

void k_attention_free(AttentionWork *work) {
  memory_free(work->partial);
  work->partial = NULL;
}

static void attention_scalar(float *out, const float *q, const float *key_cache, const float *value_cache, float *att,
                             int n_heads, int n_kv_heads, int head_size, int seq_len, int pos) {
  int kv_dim = n_kv_heads * head_size, kv_mul = n_heads / n_kv_heads;
  // multihead attention. iterate over all heads
  int h;
#ifdef _OPENMP
#pragma omp parallel for private(h)
#endif
  for (h = 0; h < n_heads; h++) {
    // get the query vector for this head
    const float *qh = q + h * head_size;
    // attention scores for this head
    float *a = att + (size_t)h * seq_len;
    // iterate over all timesteps, including the current one
    for (int t = 0; t <= pos; t++) {
      // get the key vector for this head and at this timestep
      const float *k = key_cache + (size_t)t * kv_dim + (h / kv_mul) * head_size;
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
    float *xb = out + h * head_size;
    memset(xb, 0, head_size * sizeof(float));
    for (int t = 0; t <= pos; t++) {
      // get the value vector for this head and at this timestep
      const float *v = value_cache + (size_t)t * kv_dim + (h / kv_mul) * head_size;
      // get the attention weight for this timestep
      float w = a[t];
      // accumulate the weighted value into xb
      for (int i = 0; i < head_size; i++) {
        xb[i] += w * v[i];
      }
    }
  }
}

void k_attention(float *out, const float *q, const float *key_cache, const float *value_cache, float *att,
                 AttentionWork *work, int n_heads, int n_kv_heads, int head_dim, int seq_len, int position) {
#if HAVE_AVX512
  if (isa == KERNELS_AVX512 && head_dim <= 256) {
    int positions = position + 1, kv_mul = n_heads / n_kv_heads, kv_dim = n_kv_heads * head_dim;
    int chunks = attention_chunks(positions, n_kv_heads, work->max_chunks), tasks = n_kv_heads * chunks;
    size_t slot = (size_t)head_dim + 2, head_stride = (size_t)work->max_chunks * slot;
    float *partial = work->partial;
#ifdef _OPENMP
#pragma omp parallel
#endif
    {
#ifdef _OPENMP
#pragma omp for schedule(dynamic, 1)
#endif
      for (int task = 0; task < tasks; task++) {
        int g = task / chunks, c = task % chunks;
        int t0 = (int)((long long)positions * c / chunks), t1 = (int)((long long)positions * (c + 1) / chunks);
        int h0 = g * kv_mul;
        attention_chunk_avx512(q + (size_t)h0 * head_dim, key_cache + (size_t)g * head_dim,
                               value_cache + (size_t)g * head_dim, kv_dim, head_dim, kv_mul, t0, t1,
                               att + (size_t)h0 * seq_len, seq_len, partial + h0 * head_stride + c * slot,
                               head_stride);
      }
      /* Merge chunk results: rescale each by exp(m_c - M) and normalize. */
#ifdef _OPENMP
#pragma omp for
#endif
      for (int h = 0; h < n_heads; h++) {
        const float *base = partial + h * head_stride;
        float max_val = base[head_dim];
        for (int c = 1; c < chunks; c++)
          if (base[c * slot + head_dim] > max_val)
            max_val = base[c * slot + head_dim];
        float *o = out + (size_t)h * head_dim, sum = 0.0f;
        memset(o, 0, head_dim * sizeof(float));
        for (int c = 0; c < chunks; c++) {
          const float *chunk = base + c * slot;
          float weight = expf(chunk[head_dim] - max_val);
          sum += weight * chunk[head_dim + 1];
          for (int i = 0; i < head_dim; i++)
            o[i] += weight * chunk[i];
        }
        for (int i = 0; i < head_dim; i++)
          o[i] /= sum;
      }
    }
    return;
  }
#endif
  (void)work;
  attention_scalar(out, q, key_cache, value_cache, att, n_heads, n_kv_heads, head_dim, seq_len, position);
}
