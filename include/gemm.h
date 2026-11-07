#ifndef LLAMA_GEMM_H
#define LLAMA_GEMM_H

#include "runtime.h"
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

/* Test hooks: the same driver with a portable microkernel and an explicit K block, to exercise blocking, tails and
 * epilogues on any host. kc <= 0 uses one block. */
void k_gemm_f32_portable(const float *a, int lda, int m, int k, const GemmSegment *segments, int count, int kc);
/* Runs the SIMD microkernel (real AVX-512 on x86, an emulation in the GEMM_EMULATE_AVX512 test build) regardless of
 * the selected instruction set; returns 0 when none is compiled in. The caller must know the CPU can run it. */
int k_gemm_f32_simd_test(const float *a, int lda, int m, int k, const GemmSegment *segments, int count, int kc);
/* The K block the active instruction set uses (LLAMA_GEMM_KC overrides the AVX-512 default of 1024). */
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
int gemm_amx_compiled(void); /* the build has AMX code (real or emulated) */
int gemm_amx_emulated(void); /* ... and it runs on the software tile model (test builds) */
/* Releases the calling thread's tile state. Worker threads release it themselves when they exit. */
void gemm_amx_thread_release(void);

/* FP32 to BF16 with round to nearest even (see gemm.c for NaN, denormal and overflow behaviour), and back. */
uint16_t bf16_from_f32(float x);
float f32_from_bf16(uint16_t v);

/* Packing for the tile microkernels. W is rows x cols row-major (W[out][in]) with leading dimension ld (elements).
 * Rows are grouped in tiles of 16 and zero-padded, as is K (to 32 for BF16 pairs, 64 for int8 quads). The helpers take
 * plain (rows, cols, ld) pointers; adapt them to a Shard once B1 lands. The _tiles forms pack tiles [first, last) so a
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

/* Packs many matrices with the active worker team (or OpenMP, or serially). */
typedef struct {
  uint16_t *dst;
  const float *w;
  int rows, cols;
  size_t ld;
} PackBf16;
void pack_bf16_vnni_many(const PackBf16 *jobs, int count);

#endif
