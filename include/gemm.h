#ifndef LLAMA_GEMM_H
#define LLAMA_GEMM_H

#include "kernels.h"
#include "runtime.h"
#include "shards.h"
#include <stddef.h>
#include <stdint.h>

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

/* The K block the active instruction set uses (1024 on AVX-512, 0 = one block otherwise). */
int k_gemm_kc(void);

/* ---- AMX (C4) ------------------------------------------------------------------------------------------------------
 * LLAMA_KERNELS=amx selects AMX for the batched GEMM tiles only; everything else keeps running on the AVX-512 kernels.
 * Tile instructions need CPU support and, on Linux, the kernel's permission for tile data (arch_prctl). The amx level
 * is never chosen by `auto`. */
/* NULL when AMX is usable, else the reason. The first call asks the kernel for the tile-data permission; make it
 * before any worker thread can execute a tile instruction. */
const char *gemm_amx_unavailable(void);
/* "not requested", "granted", "denied" or "unsupported", for the report. */
const char *gemm_amx_permission(void);
void gemm_amx_enable(int on); /* set by the kernel selection */
int gemm_amx_active(void);
int gemm_amx_emulated(void); /* the build runs AMX code on the software tile model (test builds) */
/* Releases the calling thread's tile state. Worker threads release it themselves when they exit. */
void gemm_amx_thread_release(void);

/* FP32 to BF16 with round to nearest even (see gemm.c for NaN, denormal and overflow behaviour), and back. */
uint16_t bf16_from_f32(float x);
float f32_from_bf16(uint16_t v);

/* Packing for the tile microkernels. W is rows x cols row-major (W[out][in]) with leading dimension ld (elements).
 * Rows are grouped in tiles of 16 and zero-padded, as is K (to 32 for BF16 pairs, 64 for int8 quads). The helpers take
 * plain (rows, cols, ld) pointers; pack_*_job_from_shard (below) builds the job of a loaded B1 Shard. The _tiles forms pack tiles [first, last) so a
 * worker team can split the work. */
size_t bf16_vnni_elements(int rows, int cols);
size_t s8_vnni_bytes(int rows, int cols);
void pack_bf16_vnni(uint16_t *dst, const float *w, int rows, int cols, size_t ld);
void pack_bf16_vnni_tiles(uint16_t *dst, const float *w, int rows, int cols, size_t ld, int first, int last);
void pack_s8_vnni(int8_t *dst, const int8_t *w, int rows, int cols, size_t ld);
void pack_s8_vnni_tiles(int8_t *dst, const int8_t *w, int rows, int cols, size_t ld, int first, int last);

/* Tile microkernels: C[m][n] = A[m][k] . B[n][k]^T for m, n <= 32. A is row-major with leading dimension lda in
 * elements and must be zero-padded and readable up to ceil(m, 16) rows by ceil(k, 32) (BF16) or ceil(k, 64) (int8)
 * columns; bp is the packed B for the same k (at most two 16-column tiles); C has room for whole 16 x 16 tiles. The
 * BF16 result is an FP32 sum in ascending K; the int8 result is exact int32. Both fail unless AMX code is compiled in,
 * and the caller must have made sure the CPU can run it (gemm_amx_unavailable() == NULL). */
void amx_bf16_tile_gemm(float *c, size_t ldc, const uint16_t *a, size_t lda, const uint16_t *bp, int m, int n, int k);
void amx_s8_tile_gemm(int32_t *c, size_t ldc, const int8_t *a, size_t lda, const int8_t *bp, int m, int n, int k);

/* A 64-byte-aligned workspace carved out of an accounted allocation (memory_alloc itself aligns to 16 bytes). */
typedef struct {
  void *base, *data;
  size_t bytes;
} AlignedBuffer;
void aligned_buffer_alloc(AlignedBuffer *buffer, size_t bytes, MemoryKind kind);
void aligned_buffer_free(AlignedBuffer *buffer);

/* ---- BF16 GEMM (C5) ------------------------------------------------------------------------------------------------
 * The activations are converted to BF16 (round to nearest even) into a 64-byte-aligned panel, the weights come
 * pre-packed by pack_bf16_vnni, the sums are FP32 and the epilogues are the FP32 ones. With LLAMA_KERNELS=amx the tile
 * microkernel runs; otherwise a portable reference with the same result (bit for bit on the software tile model). */
typedef struct {
  float *c; /* m x n output, row-major, leading dimension ldc */
  int ldc;
  const uint16_t *w;  /* n x k weights packed by pack_bf16_vnni for this k */
  const uint16_t *w3; /* GEMM_SWIGLU only: the second packed matrix */
  int n;
  GemmKind kind;
} GemmBf16Segment;
/* Bytes the panel must hold for up to m rows of k values. */
size_t k_gemm_bf16_panel_bytes(int m, int k);
void k_gemm_bf16(const float *a, int lda, int m, int k, const GemmBf16Segment *segments, int count,
                 AlignedBuffer *panel);

/* Packs many matrices with the active worker team (or serially without one). */
typedef struct {
  uint16_t *dst;
  const float *w;
  int rows, cols;
  size_t ld;
} PackBf16;
void pack_bf16_vnni_many(const PackBf16 *jobs, int count);

/* ---- Q8_0 batched GEMM (C6) ----------------------------------------------------------------------------------------
 * W8A8 with the repository's Q8_0 rule: int8 weights with one FP32 scale per group of gs inputs, activations quantized
 * per token row and group by k_quantize, per group `(float)(int32 dot) * ws * xs` accumulated in FP32. Three backends
 * with that semantic: the portable one (the scalar GEMV's arithmetic, so on the scalar instruction set it equals the
 * per-row GEMV bitwise), a tiled AVX-512 VNNI kernel (the "avx512-q8" rung; agrees with the GEMV within the Q8 budget),
 * and AMX-INT8 on weights packed by pack_s8_vnni (exact int32 group dots; same budget). AMX-INT8 takes gs % 64 == 0
 * and gs = 32 (half-shape tiles, k % 64 == 0), the VNNI kernel gs % 64 == 0; other group sizes use the portable
 * backend. W8A32 is not routed through any of them. */
typedef struct {
  float *c; /* m x n output, row-major, leading dimension ldc */
  int ldc;
  K8Weight w, w3;         /* n x k int8 values (row stride k) and n x (k / gs) scales; w3 for GEMM_SWIGLU */
  const int8_t *pw, *pw3; /* the same values packed by pack_s8_vnni; NULL unless AMX may be used */
  const float *pws, *pws3; /* the scales transposed by pack_s8_scales_t; only k_gemm_q8_tiled reads them */
  int n;
  GemmKind kind;
} GemmQ8Segment;
typedef enum { Q8_GEMM_PORTABLE, Q8_GEMM_AVX512, Q8_GEMM_AMX } Q8GemmBackend;

/* Quantizes m rows of n values (row stride ldx) with k_quantize per row, in parallel: row r of q (stride n) and of s
 * (stride n / gs) is bitwise what the per-token path produces. */
void k_quantize_rows(int8_t *q, float *s, const float *x, size_t ldx, int m, int n, int gs);
/* C[m][n] (op)= Xq[m][k] . Wq[n][k]^T with activation scales xs[m][k / gs]; xq and xs are dense (row strides k and
 * k / gs). For AMX, xq must be readable up to the next multiple of 32 rows. Segments share the activations. */
void k_gemm_q8(const int8_t *xq, const float *xs, int m, int k, int gs, const GemmQ8Segment *segments, int count);
/* The backend k_gemm_q8 would choose (for logging and reports). */
Q8GemmBackend k_gemm_q8_pick(int gs, const GemmQ8Segment *segments, int count);

/* The tiled AMX-INT8 kernel (LLAMA_Q8_GEMM=skewed, the default, or pipelined): the same Q8_0 arithmetic as the AMX
 * chunk kernel, bitwise, on activations quantized straight into tile order and weight scales transposed at pack time,
 * with the FP32 rescale of one 16-row half-step overlapped with the tile instructions of the next; skewed also issues
 * each half-step's tile store after the next half-step's tile products.
 * Tile order of A (k % 64 == 0): row r, bytes [64 t, 64 t + 64) live at
 *   ((r / 32 * 2 + r % 32 / 16) * (k / 64) + t) * 1024 + (r % 16) * 64,
 * so a 32-row block is 32 * k contiguous bytes and every 16 x 64 tile is one 1 KiB load. Rows are padded to whole
 * 32-row blocks. Scales stay dense [row][k / gs]. Transposed weight scales: st[(c / 32 * groups + g) * 32 + c % 32] for
 * channel c and group g, channels padded to 32 with zeros (s8_scales_t_bytes). */
void gemm_q8_init_env(void); /* reads LLAMA_Q8_GEMM (kernels_init) */
void gemv_init_env(void);    /* reads LLAMA_GEMV_STAGGER=0|1 (kernels_init) */
int k_gemv_stagger_active(void);
/* LLAMA_SWIGLU_EXP=vector|libm: the batched Q8 GEMM's SwiGLU epilogue with the AVX-512 exponential of the attention
 * kernels (default, where AVX-512 runs) or libm expf (bitwise the GEMV's silu_gate); read by kernels_init. */
void gemm_swiglu_init_env(void);
const char *k_gemm_swiglu_exp_name(void);
const char *k_gemm_q8_mode_name(void); /* "skewed", "pipelined" or "chunk" */
const char *k_gemm_q8_kb_order_name(void); /* LLAMA_Q8_KB_ORDER: "rows" or "chunks" */
int k_gemm_q8_tiled_active(void);      /* skewed or pipelined asked for and AMX active */
size_t s8_scales_t_bytes(int rows, int cols, int gs);
void pack_s8_scales_t(float *st, const float *s, int rows, int groups, int first_tile, int last_tile);
/* k_quantize_rows with the int8 values in tile order (scales as k_quantize_rows); q holds whole 32-row blocks. */
void k_quantize_rows_tiled(int8_t *q, float *s, const float *x, size_t ldx, int m, int n, int gs);
/* k_rmsnorm_rows (rows of n, stride n) into xb, then k_quantize_rows_tiled of xb, in one pass over the rows. */
void k_rmsnorm_quantize_rows_tiled(int8_t *q, float *s, float *xb, const float *x, const float *weight, int m, int n,
                                   float eps, int gs);
/* k_gemm_q8 on tile-ordered activations; every segment needs pw/pws (and pw3/pws3 for GEMM_SWIGLU) and AMX. */
void k_gemm_q8_tiled(const int8_t *xq, const float *xs, int m, int k, int gs, const GemmQ8Segment *segments, int count);

/* Work interleaved with one segment's product (the two-socket batched Wo/W2 reduction in tp.c). The product is a list
 * of blocks: block t covers rows [t % mblocks * 32, +32) and channels [t / mblocks * chunk_n, +chunk_n), clipped to m
 * and n (both set by the GEMM before it starts). Every worker owns a contiguous range [first, last) of blocks and
 * stores them in ascending order; after each block's epilogue (a K-blocked product: each piece's) it calls stored(ctx,
 * t0, t1) for the blocks just stored, and after every unit of compute turn(ctx, first, next, done, 0), which may use
 * blocks [next, done) (stored, not yet used) and returns the new `next`. After its range it calls turn(ctx, first,
 * next, last, 1), which must use every remaining block. The product's own results are bitwise those of
 * k_gemm_q8(_tiled). */
typedef struct {
  void (*stored)(void *ctx, int t0, int t1);
  int (*turn)(void *ctx, int first, int next, int done, int final);
  void *ctx;
  int mblocks, chunk_n; /* set by the GEMM */
} GemmQ8Hooks;
/* k_gemm_q8 (tiled = 0) or k_gemm_q8_tiled (tiled = 1) on one segment, with interleaved work. */
void k_gemm_q8_interleaved(const int8_t *xq, const float *xs, int m, int k, int gs, const GemmQ8Segment *segment,
                           int tiled, GemmQ8Hooks *hooks);

/* Packs many int8 matrices with the active worker team (or serially without one). */
typedef struct {
  int8_t *dst;
  const int8_t *w;
  int rows, cols;
  size_t ld;
  float *st;      /* when non-NULL, also the transposed scales of `s` (pack_s8_scales_t) */
  const float *s; /* dense rows x groups scales */
  int groups;
} PackS8;
void pack_s8_vnni_many(const PackS8 *jobs, int count);

/* Packing jobs for a loaded B1 Shard (shards.h), so one socket's slice is packed where it lives: the job packs the
 * shard's rows x cols block at its own stride (`ld`) into `dst`, which must hold bf16_vnni_elements / s8_vnni_bytes
 * of (shard->rows, shard->cols). The shard must be a ROWS or COLS shard of the right dtype (F32 for BF16, Q8 for
 * int8) and bound (and loaded before the job runs); anything else is an error. A Q8 shard's group scales stay in
 * shard->s, rows x (cols / gs), exactly as a K8Weight reads them; they are not packed. Pack as many shards as you
 * like with one pack_*_many call, then use the result with GemmBf16Segment.w / GemmQ8Segment.pw. */
PackBf16 pack_bf16_job_from_shard(uint16_t *dst, const Shard *shard);
PackS8 pack_s8_job_from_shard(int8_t *dst, const Shard *shard);

#endif
