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

#include "simd_exp.h"

#endif

#endif
