#ifndef LLAMA_KERNELS_H
#define LLAMA_KERNELS_H

#include "runtime.h"

/* Compute kernels shared by both engines. kernels_init() selects one instruction
 * set per process: AVX-512 (F/BW/DQ/VL/VNNI + FMA) when the CPU and OS support it,
 * otherwise the scalar reference, which keeps the original operation order.
 * LLAMA_KERNELS=auto|scalar|avx512|amx overrides the automatic choice; an explicit
 * unsupported request fails. SIMD reductions change FP32 summation order, so
 * results agree with the scalar reference within tolerance, not bitwise.
 * KERNELS_AMX (C4) keeps every kernel on its AVX-512 path and additionally lets the batched GEMM use AMX tiles; it
 * needs the CPU feature and the kernel's tile-data permission, and `auto` never selects it. */
typedef enum { KERNELS_SCALAR, KERNELS_AVX512, KERNELS_AMX } KernelIsa;
void kernels_init(void);
int kernels_select(KernelIsa isa); /* 0 when unsupported; used by tests */
const char *kernels_name(void);    /* NULL before kernels_init() */

/* Execution backend. With LLAMA_BACKEND=teams (default omp) the engine starts one persistent worker team
 * (LLAMA_THREADS threads, else the allowed CPUs; LLAMA_TOPOLOGY selects its CPUs) and every kernel runs on it
 * instead of opening OpenMP regions. kernels_set_team routes kernels to a team (NULL = OpenMP); set it before
 * k_attention_init, which sizes the split-K workspace from the worker count. */
struct Team;
void kernels_set_team(struct Team *team);
void kernels_backend_init(void);
void kernels_backend_free(void);
const char *kernels_backend_name(void); /* "omp" or "teams"; stays "teams" after the pool stops */

/* Internal to gemm.c: the active instruction set, and the team-or-OpenMP choice. fn(job, segment, first, last) runs
 * for each segment's balanced share [first, last) of its chunks[segment] chunks; a worker's share never straddles
 * two segments. Returns after every share has run. */
int kernels_avx512_active(void);
int kernels_amx_active(void); /* LLAMA_KERNELS=amx was selected */
typedef void (*SegmentFn)(const void *job, int segment, int first, int last);
void kernels_parallel_segments(int count, const int *chunks, SegmentFn fn, const void *job);

void k_rmsnorm(float *out, const float *x, const float *weight, int size, float epsilon);

/* One position's RoPE table: 2 * head_dim floats (cosines duplicated per pair,
 * then sines as -s, +s). Values equal apply_rope's per-pair cosf/sinf. */
void k_rope_table(const Config *p, int position, float *table);
void k_rope_apply(float *q, float *k, const float *table, int n_heads, int n_kv_heads, int head_dim);

/* FP32 GEMV with row-major W (d, n). Fused variants run in one parallel region:
 * _add accumulates into out (residual), _qkv computes three projections of the
 * same input, _swiglu writes silu(W1 x) * (W3 x). */
void k_matmul(float *out, const float *x, const float *w, int n, int d);
void k_matmul_add(float *out, const float *x, const float *w, int n, int d);
void k_matmul_qkv(float *q, float *k, float *v, const float *x, const float *wq, const float *wk, const float *wv, int n,
                  int dq, int dkv);
void k_matmul_swiglu(float *out, const float *x, const float *w1, const float *w3, int n, int d);

/* Q8_0: int8 rows with one FP32 scale per group of gs inputs. Values must lie in
 * [-127, 127] (the exporter and k_quantize guarantee it). An input with q == NULL
 * multiplies dequantized weights by the FP32 activations f (W8A32); otherwise
 * activations are the quantized q/s pair (W8A8, the repository's Q8_0 rule). */
typedef struct {
  const int8_t *q;
  const float *s;
} K8Weight;
typedef struct {
  const int8_t *q;
  const float *s;
  const float *f;
} K8Input;
void k_quantize(int8_t *q, float *s, const float *x, int n, int gs);
void k_matmul_q8(float *out, const K8Input *x, K8Weight w, int n, int d, int gs);
void k_matmul_q8_add(float *out, const K8Input *x, K8Weight w, int n, int d, int gs);
void k_matmul_q8_qkv(float *q, float *k, float *v, const K8Input *x, K8Weight wq, K8Weight wk, K8Weight wv, int n,
                     int dq, int dkv, int gs);
void k_matmul_q8_swiglu(float *out, const K8Input *x, K8Weight w1, K8Weight w3, int n, int d, int gs);

/* Causal single-token attention over a layer's cache rows [position][kv_dim].
 * The AVX-512 path processes each GQA group together and splits positions into
 * chunks merged with a numerically stable softmax (split-K decoding). */
typedef struct {
  float *partial;
  int max_chunks, n_heads, head_dim, workers;
  int external;    /* partial was supplied by the caller (a placement region) and is not freed here */
  size_t capacity; /* floats in partial; batched attention processes as many rows per wave as fit */
} AttentionWork;
void k_attention_init(AttentionWork *work, int n_heads, int n_kv_heads, int head_dim, int seq_len);
/* Same, with the split-K workspace in caller memory of k_attention_bytes() bytes (zeroed, float-aligned). */
size_t k_attention_bytes(int n_heads, int n_kv_heads, int head_dim, int seq_len);
void k_attention_init_in(AttentionWork *work, int n_heads, int n_kv_heads, int head_dim, int seq_len, float *memory);
/* The same for batched prefill of up to `rows` rows per call: room for as many rows of split-K partials as the budget
 * LLAMA_ATTN_WORKSPACE_MIB (default 32) allows, never less than one row. Longer batches run in waves of that many rows. */
size_t k_attention_bytes_rows(int n_heads, int n_kv_heads, int head_dim, int seq_len, int rows);
void k_attention_init_rows(AttentionWork *work, int n_heads, int n_kv_heads, int head_dim, int seq_len, int rows,
                           float *memory);
/* Causal attention for `m` query rows at positions pos0..pos0+m-1 whose K/V rows are already in the cache. Row i
 * attends to positions 0..pos0+i and is computed exactly like k_attention at that position, so results are bitwise
 * equal to the per-row calls; the rows of one task share K/V loads. out and q hold m rows of n_heads*head_dim floats;
 * att is the per-row scratch of the fallback path. */
void k_attention_batch(float *out, const float *q, const float *key_cache, const float *value_cache, float *att,
                       AttentionWork *work, int n_heads, int n_kv_heads, int head_dim, int seq_len, int pos0, int m);
/* Test hook: the same driver with a portable chunk kernel, on any host. */
void k_attention_batch_portable(float *out, const float *q, const float *key_cache, const float *value_cache,
                                AttentionWork *work, int n_heads, int n_kv_heads, int head_dim, int seq_len, int pos0,
                                int m);
void k_attention_free(AttentionWork *work);
void k_attention(float *out, const float *q, const float *key_cache, const float *value_cache, float *att,
                 AttentionWork *work, int n_heads, int n_kv_heads, int head_dim, int seq_len, int position);

#endif
