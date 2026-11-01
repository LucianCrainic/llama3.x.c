/* Batched FP32 GEMM (C2). One driver, two microkernels: a portable one (also the scalar instruction set) and an
 * AVX-512 4x4 dot-product tile. The driver splits every segment's N into chunks of NC weight rows that are
 * distributed over the workers; a chunk loops over blocks of M rows and blocks of K, collects each block's tile sums
 * in a small scratch and applies the epilogue once all of K has been summed. */
#include "gemm.h"
#include "kernels.h"
#include <math.h>
#include <stdlib.h>
#include <string.h>

#if defined(GEMM_EMULATE_AVX512)
#include "../tests/avx512_emul.h"
#define GEMM_AVX512 1
#define AVX512
#elif defined(__x86_64__) && (defined(__GNUC__) || defined(__clang__))
#define GEMM_AVX512 1
#include <immintrin.h>
#define AVX512 __attribute__((target("avx512f,avx512bw,avx512dq,avx512vl,avx512vnni,fma")))
#else
#define GEMM_AVX512 0
#endif

enum { NC = 16, MB = 256, MR = 4, NR = 4, DEFAULT_KC = 1024 };

/* tile[r * NR + c] = sum over the kb values of A row r and W row c. */
typedef void (*Micro)(const float *a, size_t lda, const float *w, size_t ldw, int mr, int nr, int kb, float *tile);

typedef struct {
  const float *a;
  int lda, m, k, kc, count;
  GemmSegment segment[GEMM_MAX_SEGMENTS];
  Micro micro;
} GemmJob;

/* Same expression as silu_gate in kernels_simd.c, so the SWIGLU epilogue rounds like the GEMV's. */
static float silu_gate(float value, float gate) {
  value *= (1.0f / (1.0f + expf(-value)));
  return value * gate;
}

static void micro_portable(const float *a, size_t lda, const float *w, size_t ldw, int mr, int nr, int kb,
                           float *tile) {
  for (int r = 0; r < mr; r++)
    for (int c = 0; c < nr; c++) {
      const float *ar = a + (size_t)r * lda, *wc = w + (size_t)c * ldw;
      float val = 0.0f;
      for (int j = 0; j < kb; j++)
        val += wc[j] * ar[j];
      tile[r * NR + c] = val;
    }
}

#if GEMM_AVX512
AVX512 static inline __mmask16 tail_mask(int count) {
  return count >= 16 ? (__mmask16)0xFFFF : (__mmask16)((1u << count) - 1u);
}

/* Four A rows by four W rows share every load: 8 loads feed 16 FMAs per 16 values of K, with 16 independent chains. */
AVX512 static void micro_avx512(const float *a, size_t lda, const float *w, size_t ldw, int mr, int nr, int kb,
                                float *tile) {
  if (mr == MR && nr == NR) {
    const float *a0 = a, *a1 = a + lda, *a2 = a + 2 * lda, *a3 = a + 3 * lda;
    const float *w0 = w, *w1 = w + ldw, *w2 = w + 2 * ldw, *w3 = w + 3 * ldw;
    __m512 c00 = _mm512_setzero_ps(), c01 = c00, c02 = c00, c03 = c00, c10 = c00, c11 = c00, c12 = c00, c13 = c00,
           c20 = c00, c21 = c00, c22 = c00, c23 = c00, c30 = c00, c31 = c00, c32 = c00, c33 = c00;
#define TILE_STEP(LOAD, M, JJ)                                                                                         \
  do {                                                                                                                 \
    __m512 x0 = LOAD(M, a0 + (JJ)), x1 = LOAD(M, a1 + (JJ)), x2 = LOAD(M, a2 + (JJ)), x3 = LOAD(M, a3 + (JJ));        \
    __m512 y0 = LOAD(M, w0 + (JJ)), y1 = LOAD(M, w1 + (JJ)), y2 = LOAD(M, w2 + (JJ)), y3 = LOAD(M, w3 + (JJ));        \
    c00 = _mm512_fmadd_ps(x0, y0, c00), c01 = _mm512_fmadd_ps(x0, y1, c01);                                            \
    c02 = _mm512_fmadd_ps(x0, y2, c02), c03 = _mm512_fmadd_ps(x0, y3, c03);                                            \
    c10 = _mm512_fmadd_ps(x1, y0, c10), c11 = _mm512_fmadd_ps(x1, y1, c11);                                            \
    c12 = _mm512_fmadd_ps(x1, y2, c12), c13 = _mm512_fmadd_ps(x1, y3, c13);                                            \
    c20 = _mm512_fmadd_ps(x2, y0, c20), c21 = _mm512_fmadd_ps(x2, y1, c21);                                            \
    c22 = _mm512_fmadd_ps(x2, y2, c22), c23 = _mm512_fmadd_ps(x2, y3, c23);                                            \
    c30 = _mm512_fmadd_ps(x3, y0, c30), c31 = _mm512_fmadd_ps(x3, y1, c31);                                            \
    c32 = _mm512_fmadd_ps(x3, y2, c32), c33 = _mm512_fmadd_ps(x3, y3, c33);                                            \
  } while (0)
#define LOAD_FULL(M, P) _mm512_loadu_ps(P)
#define LOAD_MASKED(M, P) _mm512_maskz_loadu_ps(M, P)
    int j = 0;
    for (; j + 16 <= kb; j += 16)
      TILE_STEP(LOAD_FULL, 0, j);
    if (j < kb) {
      __mmask16 m = tail_mask(kb - j);
      TILE_STEP(LOAD_MASKED, m, j);
    }
#undef TILE_STEP
#undef LOAD_FULL
#undef LOAD_MASKED
    tile[0] = _mm512_reduce_add_ps(c00), tile[1] = _mm512_reduce_add_ps(c01);
    tile[2] = _mm512_reduce_add_ps(c02), tile[3] = _mm512_reduce_add_ps(c03);
    tile[4] = _mm512_reduce_add_ps(c10), tile[5] = _mm512_reduce_add_ps(c11);
    tile[6] = _mm512_reduce_add_ps(c12), tile[7] = _mm512_reduce_add_ps(c13);
    tile[8] = _mm512_reduce_add_ps(c20), tile[9] = _mm512_reduce_add_ps(c21);
    tile[10] = _mm512_reduce_add_ps(c22), tile[11] = _mm512_reduce_add_ps(c23);
    tile[12] = _mm512_reduce_add_ps(c30), tile[13] = _mm512_reduce_add_ps(c31);
    tile[14] = _mm512_reduce_add_ps(c32), tile[15] = _mm512_reduce_add_ps(c33);
    return;
  }
  /* Edge tiles (M or N not a multiple of four): one accumulator per output, same arithmetic. */
  for (int r = 0; r < mr; r++)
    for (int c = 0; c < nr; c++) {
      const float *ar = a + (size_t)r * lda, *wc = w + (size_t)c * ldw;
      __m512 acc = _mm512_setzero_ps();
      int j = 0;
      for (; j + 16 <= kb; j += 16)
        acc = _mm512_fmadd_ps(_mm512_loadu_ps(wc + j), _mm512_loadu_ps(ar + j), acc);
      if (j < kb) {
        __mmask16 m = tail_mask(kb - j);
        acc = _mm512_fmadd_ps(_mm512_maskz_loadu_ps(m, wc + j), _mm512_maskz_loadu_ps(m, ar + j), acc);
      }
      tile[r * NR + c] = _mm512_reduce_add_ps(acc);
    }
}
#endif

/* Sums A[m0..m0+mb) . W[n0..n0+nc)^T over all of K into acc[mb][NC], one K block at a time. */
static void accumulate(const GemmJob *job, const float *w, size_t ldw, int n0, int nc, int m0, int mb, float *acc) {
  memset(acc, 0, (size_t)mb * NC * sizeof(float));
  float tile[MR * NR];
  for (int k0 = 0; k0 < job->k; k0 += job->kc) {
    int kb = job->k - k0 < job->kc ? job->k - k0 : job->kc;
    for (int mi = 0; mi < mb; mi += MR) {
      int mr = mb - mi < MR ? mb - mi : MR;
      const float *a = job->a + (size_t)(m0 + mi) * job->lda + k0;
      for (int ni = 0; ni < nc; ni += NR) {
        int nr = nc - ni < NR ? nc - ni : NR;
        job->micro(a, (size_t)job->lda, w + (size_t)(n0 + ni) * ldw + k0, ldw, mr, nr, kb, tile);
        for (int r = 0; r < mr; r++)
          for (int c = 0; c < nr; c++)
            acc[(size_t)(mi + r) * NC + ni + c] += tile[r * NR + c];
      }
    }
  }
}

static void gemm_chunks(const void *p, int s, int first, int last) {
  const GemmJob *job = p;
  const GemmSegment *seg = &job->segment[s];
  float acc[MB * NC], gate[MB * NC];
  for (int chunk = first; chunk < last; chunk++) {
    int n0 = chunk * NC, nc = seg->n - n0 < NC ? seg->n - n0 : NC;
    for (int m0 = 0; m0 < job->m; m0 += MB) {
      int mb = job->m - m0 < MB ? job->m - m0 : MB;
      accumulate(job, seg->w, seg->ldw, n0, nc, m0, mb, acc);
      if (seg->kind == GEMM_SWIGLU)
        accumulate(job, seg->w3, seg->ldw, n0, nc, m0, mb, gate);
      for (int r = 0; r < mb; r++) {
        float *out = seg->c + (size_t)(m0 + r) * seg->ldc + n0;
        const float *sum = acc + (size_t)r * NC, *g = gate + (size_t)r * NC;
        for (int c = 0; c < nc; c++) {
          if (seg->kind == GEMM_ADD)
            out[c] += sum[c];
          else if (seg->kind == GEMM_SWIGLU)
            out[c] = silu_gate(sum[c], g[c]);
          else
            out[c] = sum[c];
        }
      }
    }
  }
}

static void check_segments(const GemmSegment *segments, int count) {
  if (count < 1 || count > GEMM_MAX_SEGMENTS)
    fail("GEMM needs 1..%d segments, got %d", GEMM_MAX_SEGMENTS, count);
  for (int s = 0; s < count; s++)
    if (segments[s].kind == GEMM_SWIGLU && !segments[s].w3)
      fail("GEMM_SWIGLU needs a second weight matrix");
}

static void gemm_run(const float *a, int lda, int m, int k, const GemmSegment *segments, int count, int kc,
                     Micro micro) {
  check_segments(segments, count);
  if (m <= 0 || k <= 0)
    return;
  GemmJob job = {.a = a, .lda = lda, .m = m, .k = k, .kc = kc > 0 && kc < k ? kc : k, .count = count, .micro = micro};
  int chunks[GEMM_MAX_SEGMENTS];
  for (int s = 0; s < count; s++) {
    job.segment[s] = segments[s];
    chunks[s] = (segments[s].n + NC - 1) / NC;
  }
  kernels_parallel_segments(count, chunks, gemm_chunks, &job);
}

int k_gemm_kc(void) {
#if GEMM_AVX512 && !defined(GEMM_EMULATE_AVX512)
  if (kernels_avx512_active()) {
    const char *value = getenv("LLAMA_GEMM_KC");
    int kc = value && *value ? atoi(value) : DEFAULT_KC;
    return kc >= 16 ? kc : DEFAULT_KC;
  }
#endif
  return 0; /* one block: ascending k, as the scalar GEMV sums */
}

void k_gemm_f32(const float *a, int lda, int m, int k, const GemmSegment *segments, int count) {
#if GEMM_AVX512 && !defined(GEMM_EMULATE_AVX512)
  if (kernels_avx512_active()) {
    gemm_run(a, lda, m, k, segments, count, k_gemm_kc(), micro_avx512);
    return;
  }
#endif
  gemm_run(a, lda, m, k, segments, count, 0, micro_portable);
}

int k_gemm_f32_simd_test(const float *a, int lda, int m, int k, const GemmSegment *segments, int count, int kc) {
#if GEMM_AVX512
  gemm_run(a, lda, m, k, segments, count, kc, micro_avx512);
  return 1;
#else
  (void)a, (void)lda, (void)m, (void)k, (void)segments, (void)count, (void)kc;
  return 0;
#endif
}

void k_gemm_f32_portable(const float *a, int lda, int m, int k, const GemmSegment *segments, int count, int kc) {
  gemm_run(a, lda, m, k, segments, count, kc, micro_portable);
}
