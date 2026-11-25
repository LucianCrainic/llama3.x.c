#ifndef LLAMA_KERNELS_H
#define LLAMA_KERNELS_H

#include "runtime.h"
#include <stdatomic.h>

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

/* Execution backend: the engine starts one persistent worker team per socket (LLAMA_THREADS threads, else the allowed
 * CPUs; LLAMA_TOPOLOGY selects its CPUs) and every kernel runs on it (F2; the OpenMP backend was removed).
 * kernels_set_team routes kernels to a team (NULL = run serially on the calling thread, as the unit tests do); set it
 * before k_attention_init, which sizes the split-K workspace from the worker count. */
struct Team;
void kernels_set_team(struct Team *team);
/* Binds the calling thread's kernels to a team, overriding kernels_set_team for this thread only (NULL = back to the
 * default). Each socket's coordinator thread does this once, so two sockets can run kernels concurrently (B2). */
void kernels_set_thread_team(struct Team *team);
void kernels_backend_init(void);
void kernels_backend_free(void);
const char *kernels_backend_name(void); /* "teams", for the report */

/* Internal to gemm.c: the active instruction set, and the team-or-serial choice. fn(job, segment, first, last) runs
 * for each segment's balanced share [first, last) of its chunks[segment] chunks; a worker's share never straddles
 * two segments. Returns after every share has run. */
int kernels_avx512_active(void);
int kernels_amx_active(void); /* LLAMA_KERNELS=amx was selected */
/* LLAMA_KERNELS unset or auto, and AMX usable (this requests the tile permission): selects it. Called for
 * batched prefill in W8A8 (runq, D2 §7) and in the explicit BF16 mode (prefill_configure); an FP32 checkpoint without
 * LLAMA_PRECISION=bf16 never asks, so its permission stays "not requested" (C4). Returns kernels_amx_active(). */
int kernels_select_amx_auto(void);
/* Attention variants (E1). kernels_init reads LLAMA_ATTN (batched, prefill: split|tiled|amx) and LLAMA_ATTN_DECODE
 * (single row: split|group) once. The defaults are tiled for the batched path (since E4; it runs split where AVX-512
 * is not active) and split, the split-K kernel, for decode. The names are recorded in the run report. Tests pick a
 * variant with kernels_attention_select (testing.h). */
const char *kernels_attention_name(void);
const char *kernels_attention_decode_name(void);
/* W8A8 engines (runq) prefer LLAMA_ATTN=amx when LLAMA_ATTN is unset (prefill-gap fix 7: it passes the W8A8 gate and
 * is faster on one and two sockets); amx runs tiled where AMX is not active. FP32 and BF16 keep tiled. */
void kernels_attention_prefer_amx(void);
/* K/V cache layout (E5.4). kernels_init reads LLAMA_KV_LAYOUT=rows|head (default head since E5.4). rows: a layer's cache is
 * [position][kv_head][head_dim]; head: [kv_head][position][head_dim] with room for seq_len positions per head (the same
 * bytes). Engines address the cache through the helpers at the end of this header; the results are bitwise the same in
 * both layouts. Tests pick one with kernels_kv_layout_select (testing.h). */
int kernels_kv_layout_head(void);
const char *kernels_kv_layout_name(void);
typedef void (*SegmentFn)(const void *job, int segment, int first, int last);
void kernels_parallel_segments(int count, const int *chunks, SegmentFn fn, const void *job);

void k_rmsnorm(float *out, const float *x, const float *weight, int size, float epsilon);
/* `rows` consecutive rows of `size` values (out and x have row stride `size`; out may equal x), one team job; each row is
 * exactly k_rmsnorm, so the result is bitwise that of the serial loop. */
void k_rmsnorm_rows(float *out, const float *x, const float *weight, int rows, int size, float epsilon);
/* k_rope_apply for `rows` rows with the given strides (in floats) of q, k and the rope table, one team job; bitwise the serial loop. */
void k_rope_apply_rows(float *q, size_t q_stride, float *k, size_t k_stride, const float *table, size_t table_stride,
                       int rows, int n_heads, int n_kv_heads, int head_dim);

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

/* Shard entry points (B2). Weight rows have stride ld elements (ld >= n; a packed shard has ld = its columns). Row
 * slices of a tensor use the ordinary entry points on the shard's rows; a column slice (Wo, W2) passes n = its column
 * count and the matching slice of the input. `kind` is K_STORE, K_ADD or K_PARTIAL: PARTIAL writes W*x like STORE into
 * a socket-local partial buffer, and the cross-socket reduction adds the residual once, so a two-socket Wo or W2 must
 * not use K_ADD. SWIGLU keeps its pair form. For Q8, ld must be a multiple of gs. */
typedef enum { K_STORE, K_ADD, K_PARTIAL, K_SWIGLU } KernelEpilogue;
void k_matmul_ld(float *out, const float *x, const float *w, size_t ld, int n, int d, int kind);
/* The share of k_matmul_ld / k_matmul_q8_ld of member `worker` of `workers`, for a caller that is already inside a team
 * job (every member calls it; the union is the team-level result, bitwise). Teams backend only. `claim` is a counter
 * shared by the members of this one job and zero before the team starts; a large FP32 job claims the last quarter of
 * its rows from it (the hybrid split). NULL means a static split; Q8 always splits statically and ignores it. */
/* `done`, when not NULL, is called as done(done_ctx, r0, r1) by the member that computed output rows [r0, r1), after
 * those rows are stored, once per contiguous range it computed (a static split: once; claiming: once per claim). A
 * following step can use it to start on rows whose producers are finished without waiting for the whole team. */
typedef void (*RowsDoneFn)(void *ctx, int r0, int r1);
void k_matmul_ld_worker(float *out, const float *x, const float *w, size_t ld, int n, int d, int kind, int worker,
                        int workers, atomic_int *claim, RowsDoneFn done, void *done_ctx);
void k_matmul_swiglu_ld(float *out, const float *x, const float *w1, const float *w3, size_t ld, int n, int d);
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
/* Shard variants of the above. A column slice's activations are quantized from the owned slice only, at the original
 * group boundaries (col0 % gs == 0), so W8A8 and W8A32 both see whole groups. */
void k_matmul_q8_ld(float *out, const K8Input *x, const K8Weight *w, size_t ld, int n, int d, int gs, int kind);
void k_matmul_q8_ld_worker(float *out, const K8Input *x, const K8Weight *w, size_t ld, int n, int d, int gs, int kind,
                           int worker, int workers, atomic_int *claim, RowsDoneFn done, void *done_ctx);
void k_matmul_q8_swiglu_ld(float *out, const K8Input *x, const K8Weight *w1, const K8Weight *w3, size_t ld, int n,
                           int d, int gs);

/* Causal single-token attention over a layer's cache rows [position][kv_dim].
 * The AVX-512 path processes each GQA group together and splits positions into
 * chunks merged with a numerically stable softmax (split-K decoding). */
typedef struct {
  float *partial;
  int max_chunks, n_heads, head_dim, workers;
  int external;    /* partial was supplied by the caller (a placement region) and is not freed here */
  size_t capacity; /* floats in partial; batched attention processes as many rows per wave as fit */
  uint16_t *kv_bf16; /* LLAMA_ATTN=amx, batched: the BF16 K/V tile operands, after the partials (else NULL) */
  /* LLAMA_ATTN=amx with a persistent BF16 copy (k_attention_set_kv_bf16): one copy per layer, keyed by the layer's key
   * cache pointer, packed incrementally (positions [0, kv_valid) of a slot are current). */
  uint16_t *kv_cache_bf16;
  size_t kv_layer_elems;
  int kv_layers, kv_used;
  const float *kv_key[256];
  int kv_valid[256];
  uint16_t *kv_slot[256]; /* k_attention_set_kv_bf16_layers: the BF16 copy of each layer in memory of its own (else NULL) */
} AttentionWork;
/* Bytes of the persistent BF16 K/V copy of `layers` layers for LLAMA_ATTN=amx (0 when not selected), and its setup. */
size_t k_attention_kv_bf16_bytes(int n_kv_heads, int head_dim, int seq_len, int layers);
void k_attention_set_kv_bf16(AttentionWork *work, uint16_t *memory, int n_kv_heads, int head_dim, int seq_len,
                             int layers);
/* The same with each layer's copy in memory of its own (a placement tier per layer): layer l's key cache is
 * key_caches[l] and its copy memory[l] (k_attention_kv_bf16_bytes(.., 1) bytes each). The cache pointers are bound now,
 * so a layer's slot never depends on the order the layers first run in. */
void k_attention_set_kv_bf16_layers(AttentionWork *work, uint16_t *const *memory, const float *const *key_caches,
                                    int n_kv_heads, int head_dim, int seq_len, int layers);
/* The persistent BF16 copy of the layer whose key cache is `key_cache` is current only below `position` from now on (no
 * effect without a copy). For writers of the FP32 cache that bypass the attention entry points. */
void k_attention_kv_invalidate(AttentionWork *work, const float *key_cache, int position);
void k_attention_init(AttentionWork *work, int n_heads, int n_kv_heads, int head_dim, int seq_len);
/* Same, with the split-K workspace in caller memory of k_attention_bytes() bytes (zeroed, float-aligned). */
size_t k_attention_bytes(int n_heads, int n_kv_heads, int head_dim, int seq_len);
void k_attention_init_in(AttentionWork *work, int n_heads, int n_kv_heads, int head_dim, int seq_len, float *memory);
/* The same for batched prefill of up to `rows` rows per call: room for as many rows of split-K partials as the budget
 * (32 MiB) allows, never less than one row. Longer batches run in waves of that many rows. */
size_t k_attention_bytes_rows_for(int n_heads, int n_kv_heads, int head_dim, int seq_len, int rows, int workers);
size_t k_attention_bytes_rows(int n_heads, int n_kv_heads, int head_dim, int seq_len, int rows);
void k_attention_init_rows(AttentionWork *work, int n_heads, int n_kv_heads, int head_dim, int seq_len, int rows,
                           float *memory);
/* Causal attention for `m` query rows at positions pos0..pos0+m-1 whose K/V rows are already in the cache. Row i
 * attends to positions 0..pos0+i and is computed exactly like k_attention at that position, so results are bitwise
 * equal to the per-row calls; the rows of one task share K/V loads. out and q hold m rows of n_heads*head_dim floats;
 * att is the per-row scratch of the fallback path. */
void k_attention_batch(float *out, const float *q, const float *key_cache, const float *value_cache, float *att,
                       AttentionWork *work, int n_heads, int n_kv_heads, int head_dim, int seq_len, int pos0, int m);
/* Position chunks the split-K attention uses for `position` with this work's team size and the given (local) KV head
 * count, for the report and for choosing contexts where a team has too little work. */
int k_attention_chunk_count(const AttentionWork *work, int n_kv_heads, int position);
void k_attention_free(AttentionWork *work);
void k_attention(float *out, const float *q, const float *key_cache, const float *value_cache, float *att,
                 AttentionWork *work, int n_heads, int n_kv_heads, int head_dim, int seq_len, int position);
/* k_attention on a caller-chosen K/V layout (k_attention_kv uses it for the head-major cache). Position t of KV head g starts at
 * cache + t * pos_stride + g * head_base floats. k_attention is pos_stride = n_kv_heads * head_dim, head_base = head_dim
 * (rows [position][kv_head][head_dim]); the head-major layout [kv_head][position][head_dim] is pos_stride = head_dim,
 * head_base = seq_len * head_dim. The same decode kernel (LLAMA_ATTN_DECODE) runs, and the result is bitwise the one the
 * rows layout gives on the same values. */
void k_attention_layout(float *out, const float *q, const float *key_cache, const float *value_cache, float *att,
                        AttentionWork *work, int n_heads, int n_kv_heads, int head_dim, int seq_len, int position,
                        size_t pos_stride, size_t head_base);

/* The batched attention on a caller-chosen K/V layout (the strides of k_attention_layout). k_attention_batch is
 * pos_stride = n_kv_heads * head_dim, head_base = head_dim; results are bitwise those of the rows layout. */
void k_attention_batch_layout(float *out, const float *q, const float *key_cache, const float *value_cache, float *att,
                              AttentionWork *work, int n_heads, int n_kv_heads, int head_dim, int seq_len, int pos0,
                              int m, size_t pos_stride, size_t head_base);

/* Engine-facing K/V cache helpers (LLAMA_KV_LAYOUT). `cache` is one layer's K or V cache of the (local) KV heads.
 * k_kv_strides gives the pos_stride / head_base of the active layout. k_kv_store copies `count` rows of
 * n_kv_heads * head_dim floats (row stride src_ld) to positions pos0.. . k_attention_kv / k_attention_batch_kv are
 * k_attention / k_attention_batch on the active layout (the plain calls themselves for rows). */
void k_kv_strides(int n_kv_heads, int head_dim, int seq_len, size_t *pos_stride, size_t *head_base);
void k_kv_store(float *cache, const float *src, size_t src_ld, int pos0, int count, int n_kv_heads, int head_dim,
                int seq_len);
void k_attention_kv(float *out, const float *q, const float *key_cache, const float *value_cache, float *att,
                    AttentionWork *work, int n_heads, int n_kv_heads, int head_dim, int seq_len, int position);
void k_attention_batch_kv(float *out, const float *q, const float *key_cache, const float *value_cache, float *att,
                          AttentionWork *work, int n_heads, int n_kv_heads, int head_dim, int seq_len, int pos0, int m);

#endif
