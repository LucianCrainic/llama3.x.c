#ifndef LLAMA_KERNELS_INTERNAL_H
#define LLAMA_KERNELS_INTERNAL_H

/* Private to the kernel files (kernels_simd.c, gemv.c, attention.c): the instruction-set macros, the selected
 * instruction set, the team the kernels run on, and small helpers they share. Not part of the engine API. */
#include "kernels.h"
#include "workers.h"
#include <float.h>
#include <stddef.h>

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

/* The instruction set kernels_select chose (kernels_simd.c). */
extern KernelIsa kernels_isa;
/* With a team set, kernels run on it; without one (unit tests) they run serially on the calling thread. A
 * coordinator thread that drives one socket's team (B2) binds its kernels to that team without touching the
 * process-wide default, so two coordinators can run kernels on two teams at the same time. */
extern Team *kernels_default_team;
extern _Thread_local Team *kernels_thread_team;
#define current_team (kernels_thread_team ? kernels_thread_team : kernels_default_team)

enum { BLOCK = 4 }; // rows per GEMV kernel call

/* Each thread receives one contiguous, balanced row range, so weight rows are
 * streamed sequentially per thread and hardware prefetch stays effective. */
static inline void parallel_rows(int rows, RowFn fn, const void *job) {
  if (current_team) {
    team_parallel_rows(current_team, rows, fn, job);
    return;
  }
  fn(job, 0, rows);
}

static inline int max_threads(void) {
  if (current_team)
    return team_size(current_team);
  return 1;
}

/* LLAMA_ATTN, LLAMA_ATTN_DECODE and LLAMA_KV_LAYOUT, read once by kernels_init (attention.c). */
void kernels_attention_init_env(void);

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

#endif

#endif
