#ifndef LLAMA_GEMM_H
#define LLAMA_GEMM_H

#include <stddef.h>

/* Batched FP32 GEMM on the checkpoint's W[out][in] layout (C2): C[m][n] (op)= A[m][k] . W[n][k]^T for M activation
 * rows at once, so every weight row is streamed once per microbatch instead of once per token. The epilogues match
 * the GEMV kinds. Segments share A and run in one parallel region (QKV is three segments). Summation order differs
 * from the GEMV on SIMD hardware, so results agree within the kernel budget, not bitwise; with the scalar instruction
 * set (one K block, ascending k) they equal the per-row GEMV bitwise. */
typedef enum {
  GEMM_STORE,  /* c = acc */
  GEMM_ADD,    /* c += acc (residual) */
  GEMM_SWIGLU, /* c = silu(acc of w) * acc of w3 */
  GEMM_PARTIAL /* c = acc of a partial K slice, reduced later (B2); computed exactly like STORE here */
} GemmKind;

typedef struct {
  float *c;       /* m x n output, row-major, leading dimension ldc */
  int ldc;
  const float *w; /* n x k weights, row-major, leading dimension ldw */
  const float *w3; /* GEMM_SWIGLU only: the second weight matrix, same shape and leading dimension */
  size_t ldw;
  int n;
  GemmKind kind;
} GemmSegment;

enum { GEMM_MAX_SEGMENTS = 3 };

void k_gemm_f32(const float *a, int lda, int m, int k, const GemmSegment *segments, int count);

/* Test hooks: the same driver with a portable microkernel and an explicit K block, to exercise blocking, tails and
 * epilogues on any host. kc <= 0 uses one block. */
void k_gemm_f32_portable(const float *a, int lda, int m, int k, const GemmSegment *segments, int count, int kc);
/* Runs the SIMD microkernel (real AVX-512 on x86, an emulation in the GEMM_EMULATE_AVX512 test build) regardless of
 * the selected instruction set; returns 0 when none is compiled in. The caller must know the CPU can run it. */
int k_gemm_f32_simd_test(const float *a, int lda, int m, int k, const GemmSegment *segments, int count, int kc);
/* The K block the active instruction set uses (LLAMA_GEMM_KC overrides the AVX-512 default of 1024). */
int k_gemm_kc(void);

#endif
