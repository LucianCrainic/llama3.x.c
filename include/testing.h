#ifndef LLAMA_TESTING_H
#define LLAMA_TESTING_H

/* Test hooks: entry points that only the unit tests and the benchmarks under tests/ call. They select variants
 * directly, reach paths the engine takes only on large shapes, or run one backend regardless of the selected
 * instruction set. Only the files that implement them include it alongside the tests; no engine path calls them. */

#include "gemm.h"
#include "kernels.h"
#include <stddef.h>
#include <stdint.h>

/* ---- Kernel selection --------------------------------------------------------------------------------------------- */

/* Pick the attention variants directly (NULL leaves one unchanged); returns 0 for an unknown name. */
int kernels_attention_select(const char *batch, const char *decode);
/* Pick the K/V cache layout ("rows" or "head"); returns 0 for an unknown name. */
int kernels_kv_layout_select(const char *name);

/* ---- GEMV ---------------------------------------------------------------------------------------------------------- */

/* Run every FP32 team GEMV with the hybrid split (contiguous shares, then claimed chunks), which the engine uses only
 * for jobs of 96 MiB and more, so the unit tests reach it on small matrices. Results are bitwise unchanged. */
void k_gemv_force_hybrid(int on);
/* LLAMA_GEMV_STAGGER from a test: the Q8 team GEMV starts member w at block w of its share. Results are bitwise unchanged. */
void k_gemv_stagger(int on);

/* ---- Attention ----------------------------------------------------------------------------------------------------- */

/* The merge of `chunks` split-K partials [value sum (head_dim), maximum, sum] into one normalized row. A partial with
 * maximum -FLT_MAX and sum 0 (a row with no position in that chunk, tiled attention) is skipped. */
void k_attention_merge_partials(const float *base, size_t slot, int chunks, int head_dim, float *o);
/* The batched split driver with a portable chunk kernel, on any host (rows layout, or the strides of
 * k_attention_layout). */
void k_attention_batch_portable(float *out, const float *q, const float *key_cache, const float *value_cache,
                                AttentionWork *work, int n_heads, int n_kv_heads, int head_dim, int seq_len, int pos0,
                                int m);
void k_attention_batch_portable_layout(float *out, const float *q, const float *key_cache, const float *value_cache,
                                       AttentionWork *work, int n_heads, int n_kv_heads, int head_dim, int seq_len,
                                       int pos0, int m, size_t pos_stride, size_t head_base);
/* The batched-attention workspace in MiB (32 by default; 0 = one row per wave), so tests run several waves on small
 * shapes. Takes effect for workspaces sized after the call. A negative value restores the default. */
void k_attention_set_workspace_mib(long mib);
/* Opt-in timing of the attention tasks for tests/attention_bench.c: while a hook is set, every (kv group, chunk[, row
 * block]) task reports who ran it, what it read and when (monotonic_seconds). The hook runs on the worker threads; set
 * and clear it between kernel calls. With no hook (the default) the kernels are unchanged. */
typedef struct {
  int worker;        /* team member (0 when serial) */
  int wave;          /* first row of the wave inside the microbatch (0 for decode) */
  int task;          /* index in the wave */
  int rows;          /* rows of the block that take part */
  int span;          /* K/V positions the block streams: min(t0)..max(t1) over its rows */
  long positions;    /* positions scored, summed over the rows */
  double begin, end; /* monotonic_seconds around the task */
} AttentionTaskInfo;
typedef void (*AttentionTaskHook)(void *ctx, const AttentionTaskInfo *info);
void k_attention_set_task_hook(AttentionTaskHook hook, void *ctx);

/* ---- GEMM ---------------------------------------------------------------------------------------------------------- */

/* The FP32 GEMM driver with a portable microkernel and an explicit K block, to exercise blocking, tails and epilogues
 * on any host. kc <= 0 uses one block. */
void k_gemm_f32_portable(const float *a, int lda, int m, int k, const GemmSegment *segments, int count, int kc);
/* Runs the SIMD microkernel (real AVX-512 on x86, an emulation in the GEMM_EMULATE_AVX512 test build) regardless of
 * the selected instruction set; returns 0 when none is compiled in. The caller must know the CPU can run it. */
int k_gemm_f32_simd_test(const float *a, int lda, int m, int k, const GemmSegment *segments, int count, int kc);
/* One named Q8 backend (0 when not compiled in; the caller must know the CPU can run it), and the exact int32 group
 * dots of the AMX chunk kernel, dots[(g * m + r) * n + c]. */
int k_gemm_q8_backend_test(Q8GemmBackend backend, const int8_t *xq, const float *xs, int m, int k, int gs,
                           const GemmQ8Segment *segments, int count);
int k_gemm_q8_amx_dots_test(int32_t *dots, const int8_t *xq, int m, int k, int gs, const int8_t *packed, int n);

#endif
