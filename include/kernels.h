#ifndef LLAMA_KERNELS_H
#define LLAMA_KERNELS_H

#include "runtime.h"

/* Compute kernels shared by both engines. kernels_init() selects one instruction
 * set per process: AVX-512 (F/BW/DQ/VL/VNNI + FMA) when the CPU and OS support it,
 * otherwise the scalar reference, which keeps the original operation order.
 * LLAMA_KERNELS=auto|scalar|avx512 overrides the automatic choice; an explicit
 * unsupported request fails. SIMD reductions change FP32 summation order, so
 * results agree with the scalar reference within tolerance, not bitwise. */
typedef enum { KERNELS_SCALAR, KERNELS_AVX512 } KernelIsa;
void kernels_init(void);
int kernels_select(KernelIsa isa); /* 0 when unsupported; used by tests */
const char *kernels_name(void);    /* NULL before kernels_init() */

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
  int max_chunks, n_heads, head_dim;
} AttentionWork;
void k_attention_init(AttentionWork *work, int n_heads, int n_kv_heads, int head_dim, int seq_len);
void k_attention_free(AttentionWork *work);
void k_attention(float *out, const float *q, const float *key_cache, const float *value_cache, float *att,
                 AttentionWork *work, int n_heads, int n_kv_heads, int head_dim, int seq_len, int position);

#endif
