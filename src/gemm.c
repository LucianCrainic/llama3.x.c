/* Batched FP32 GEMM (C2). One driver, two microkernels: a portable one (also the scalar instruction set) and an
 * AVX-512 4x4 dot-product tile. The driver splits every segment's N into chunks of NC weight rows that are
 * distributed over the workers; a chunk loops over blocks of M rows and blocks of K, collects each block's tile sums
 * in a small scratch and applies the epilogue once all of K has been summed.
 *
 * AMX (C4): ISA detection and the Linux tile-data permission, per-thread tile configuration, the BF16 and INT8 tile
 * microkernels and the VNNI packing helpers. The `amx` target attribute is used only in this file. */
#include "gemm.h"
#include "kernels.h"
#include <math.h>
#include <pthread.h>
#include <stdint.h>
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

/* ---- AMX (C4) ---------------------------------------------------------------------------------------------------
 * GEMM_EMULATE_AMX swaps the tile instructions for the software model in tests/amx_emul.h, so the loops below run
 * on any host. The macros take literal tile numbers, as the hardware intrinsics require. */
#if defined(GEMM_EMULATE_AMX)
#include "../tests/amx_emul.h"
#define GEMM_AMX 1
#define GEMM_AMX_EMULATED 1
#define AMX_TARGET
#define AMX_LOADCFG(p) emul_tile_loadconfig(p)
#define AMX_RELEASE() emul_tile_release()
#define AMX_ZERO(t) emul_tile_zero(t)
#define AMX_LOADD(t, p, stride) emul_tile_loadd(t, p, (long)(stride))
#define AMX_STORED(t, p, stride) emul_tile_stored(t, p, (long)(stride))
#define AMX_DPBF16PS(c, a, b) emul_tile_dpbf16ps(c, a, b)
#define AMX_DPBSSD(c, a, b) emul_tile_dpbssd(c, a, b)
#elif defined(__x86_64__) && (defined(__GNUC__) || defined(__clang__)) &&                                              \
    (__has_include(<amxtileintrin.h>) || __has_include(<amxintrin.h>))
#include <cpuid.h>
#include <immintrin.h>
#define GEMM_AMX 1
#define GEMM_AMX_EMULATED 0
#define AMX_TARGET __attribute__((target("amx-tile,amx-bf16,amx-int8")))
#define AMX_LOADCFG(p) _tile_loadconfig(p)
#define AMX_RELEASE() _tile_release()
#define AMX_ZERO(t) _tile_zero(t)
#define AMX_LOADD(t, p, stride) _tile_loadd(t, p, stride)
#define AMX_STORED(t, p, stride) _tile_stored(t, p, stride)
#define AMX_DPBF16PS(c, a, b) _tile_dpbf16ps(c, a, b)
#define AMX_DPBSSD(c, a, b) _tile_dpbssd(c, a, b)
#else
#define GEMM_AMX 0
#define GEMM_AMX_EMULATED 0
#endif

#if GEMM_AMX && !GEMM_AMX_EMULATED && defined(__linux__)
#include <errno.h>
#include <sys/syscall.h>
#include <unistd.h>
#if __has_include(<asm/prctl.h>)
#include <asm/prctl.h>
#endif
/* The documented values (Linux 5.16 and later), used when the installed headers predate them. */
#ifndef ARCH_GET_XCOMP_PERM
#define ARCH_GET_XCOMP_PERM 0x1022
#endif
#ifndef ARCH_REQ_XCOMP_PERM
#define ARCH_REQ_XCOMP_PERM 0x1023
#endif
#define XFEATURE_XTILEDATA 18
_Static_assert(ARCH_GET_XCOMP_PERM == 0x1022 && ARCH_REQ_XCOMP_PERM == 0x1023, "arch_prctl XCOMP_PERM request codes");
#define AMX_PERMISSION_SYSCALL 1
#endif

static int amx_active;
static const char *amx_permission_name = "not requested";
#if GEMM_AMX && !GEMM_AMX_EMULATED
static int amx_checked;
static char amx_reason[256];
#endif

#if GEMM_AMX && !GEMM_AMX_EMULATED
/* CPUID leaf 7, subleaf 0, EDX: bit 22 AMX-BF16, bit 24 AMX-TILE, bit 25 AMX-INT8. Read directly rather than through
 * __builtin_cpu_supports, whose AMX names depend on the compiler version. */
static int cpu_has_amx(void) {
  unsigned a, b, c, d;
  if (__get_cpuid_max(0, NULL) < 7)
    return 0;
  __cpuid_count(7, 0, a, b, c, d);
  const unsigned need = (1u << 22) | (1u << 24) | (1u << 25);
  return (d & need) == need;
}
#endif

/* NULL when AMX can be used, else why not. Requests the kernel's tile-data permission on the first call; that must
 * happen before any worker thread executes a tile instruction, and kernels_init() is that point. */
const char *gemm_amx_unavailable(void) {
#if !GEMM_AMX
  return "this build has no AMX support (it needs x86-64 and a compiler with the AMX intrinsics, GCC 11 or later)";
#elif GEMM_AMX_EMULATED
  return NULL;
#else
  if (amx_checked)
    return amx_reason[0] ? amx_reason : NULL;
  amx_checked = 1;
  __builtin_cpu_init();
  if (!cpu_has_amx()) {
    amx_permission_name = "unsupported";
    snprintf(amx_reason, sizeof(amx_reason), "this CPU does not report AMX-TILE, AMX-BF16 and AMX-INT8");
  } else if (!(__builtin_cpu_supports("avx512f") && __builtin_cpu_supports("avx512bw") &&
               __builtin_cpu_supports("avx512dq") && __builtin_cpu_supports("avx512vl") &&
               __builtin_cpu_supports("avx512vnni"))) {
    snprintf(amx_reason, sizeof(amx_reason), "AVX-512 F/BW/DQ/VL/VNNI is unavailable");
  } else {
#ifdef AMX_PERMISSION_SYSCALL
    unsigned long permitted = 0;
    if (syscall(SYS_arch_prctl, ARCH_REQ_XCOMP_PERM, XFEATURE_XTILEDATA) != 0) {
      amx_permission_name = "denied";
      snprintf(amx_reason, sizeof(amx_reason),
               "the kernel refused permission for AMX tile data (arch_prctl ARCH_REQ_XCOMP_PERM: %s)", strerror(errno));
    } else if (syscall(SYS_arch_prctl, ARCH_GET_XCOMP_PERM, &permitted) != 0 ||
               !(permitted & (1ul << XFEATURE_XTILEDATA))) {
      amx_permission_name = "denied";
      snprintf(amx_reason, sizeof(amx_reason), "arch_prctl ARCH_GET_XCOMP_PERM does not show XTILEDATA as permitted");
    } else {
      amx_permission_name = "granted";
    }
#else
    amx_permission_name = "unsupported";
    snprintf(amx_reason, sizeof(amx_reason), "the tile-data permission request is implemented for Linux only");
#endif
  }
  return amx_reason[0] ? amx_reason : NULL;
#endif
}

const char *gemm_amx_permission(void) {
  return amx_permission_name;
}

void gemm_amx_enable(int on) {
  amx_active = on;
}

int gemm_amx_active(void) {
  return amx_active;
}

int gemm_amx_compiled(void) {
  return GEMM_AMX;
}

int gemm_amx_emulated(void) {
  return GEMM_AMX_EMULATED;
}

/* ---- BF16 conversion and VNNI packing (portable) ---------------------------------------------------------------- */

/* Round to nearest even on the 16 dropped bits; denormals convert like any other value (they are not flushed); a NaN
 * stays a NaN (quieted, sign and top payload bits kept); infinities are exact; values at or above the midpoint to
 * 2^128 round to infinity. */
uint16_t bf16_from_f32(float x) {
  uint32_t u;
  memcpy(&u, &x, sizeof(u));
  if ((u & 0x7fffffffu) > 0x7f800000u)
    return (uint16_t)((u >> 16) | 0x0040u);
  u += 0x7fffu + ((u >> 16) & 1u);
  return (uint16_t)(u >> 16);
}

float f32_from_bf16(uint16_t v) {
  uint32_t u = (uint32_t)v << 16;
  float x;
  memcpy(&x, &u, sizeof(x));
  return x;
}

static size_t round_up(size_t v, size_t to) {
  return (v + to - 1) / to * to;
}

size_t bf16_vnni_elements(int rows, int cols) {
  return round_up((size_t)rows, 16) * round_up((size_t)cols, 32);
}

size_t s8_vnni_bytes(int rows, int cols) {
  return round_up((size_t)rows, 16) * round_up((size_t)cols, 64);
}

/* Tile `nt` holds weight rows [16 nt, 16 nt + 16). Inside it, K pair p (columns 2p, 2p+1) is a row of 16 channel
 * pairs: dst[((nt * kp/2 + p) * 16 + c) * 2 + e] = bf16(w[16 nt + c][2p + e]), with kp = cols rounded up to 32. A
 * tile load of 16 pair rows (32 values of K) then starts at element nt * kp * 16 + (k / 32) * 512 and is contiguous. */
void pack_bf16_vnni_tiles(uint16_t *dst, const float *w, int rows, int cols, size_t ld, int first, int last) {
  size_t kp = round_up((size_t)cols, 32);
  for (int nt = first; nt < last; nt++)
    for (size_t p = 0; p < kp / 2; p++)
      for (int c = 0; c < 16; c++)
        for (int e = 0; e < 2; e++) {
          size_t row = (size_t)nt * 16 + (size_t)c, col = 2 * p + (size_t)e;
          dst[((size_t)nt * (kp / 2) + p) * 32 + (size_t)c * 2 + (size_t)e] =
              row < (size_t)rows && col < (size_t)cols ? bf16_from_f32(w[row * ld + col]) : 0;
        }
}

void pack_bf16_vnni(uint16_t *dst, const float *w, int rows, int cols, size_t ld) {
  pack_bf16_vnni_tiles(dst, w, rows, cols, ld, 0, (rows + 15) / 16);
}

/* The same for int8 with VNNI quads: dst[((nt * kp/4 + q) * 16 + c) * 4 + e] = w[16 nt + c][4q + e], kp = cols
 * rounded up to 64, so a tile load starts at byte nt * kp * 16 + (k / 64) * 1024. */
void pack_s8_vnni_tiles(int8_t *dst, const int8_t *w, int rows, int cols, size_t ld, int first, int last) {
  size_t kp = round_up((size_t)cols, 64);
  for (int nt = first; nt < last; nt++)
    for (size_t q = 0; q < kp / 4; q++)
      for (int c = 0; c < 16; c++)
        for (int e = 0; e < 4; e++) {
          size_t row = (size_t)nt * 16 + (size_t)c, col = 4 * q + (size_t)e;
          dst[((size_t)nt * (kp / 4) + q) * 64 + (size_t)c * 4 + (size_t)e] =
              row < (size_t)rows && col < (size_t)cols ? w[row * ld + col] : 0;
        }
}

void pack_s8_vnni(int8_t *dst, const int8_t *w, int rows, int cols, size_t ld) {
  pack_s8_vnni_tiles(dst, w, rows, cols, ld, 0, (rows + 15) / 16);
}

/* ---- aligned workspace ------------------------------------------------------------------------------------------ */

void aligned_buffer_alloc(AlignedBuffer *buffer, size_t bytes, MemoryKind kind) {
  buffer->base = memory_alloc(checked_add(bytes, 63), 1, kind);
  buffer->data = (void *)(((uintptr_t)buffer->base + 63) & ~(uintptr_t)63);
  buffer->bytes = bytes;
}

void aligned_buffer_free(AlignedBuffer *buffer) {
  memory_free(buffer->base);
  *buffer = (AlignedBuffer){0};
}

/* ---- tile configuration and microkernels ------------------------------------------------------------------------ */

#if GEMM_AMX
/* Every tile is 16 rows of 64 bytes: a 16 x 32 BF16 or 16 x 64 int8 A block, a K-pair or K-quad B block of 16 columns,
 * and a 16 x 16 FP32 or int32 C block. Tiles 0-3 hold C (m block x n block), 4-5 the A blocks, 6-7 the B blocks. */
typedef struct {
  uint8_t palette_id, start_row, reserved[14];
  uint16_t colsb[16];
  uint8_t rows[16];
} TileConfig;
_Static_assert(sizeof(TileConfig) == 64, "the tile configuration is 64 bytes");

static _Thread_local int amx_thread_ready;
static pthread_key_t amx_key;
static pthread_once_t amx_key_once = PTHREAD_ONCE_INIT;

AMX_TARGET static void amx_thread_exit(void *unused) {
  (void)unused;
  AMX_RELEASE();
}

static void amx_key_create(void) {
  pthread_key_create(&amx_key, amx_thread_exit);
}

/* Configures this thread's tiles at its first AMX job. A worker's tile state ends with the thread (the key's destructor
 * releases it explicitly); the coordinator thread calls gemm_amx_thread_release() when the engine is torn down. */
AMX_TARGET static void amx_thread_configure(void) {
  if (amx_thread_ready)
    return;
  TileConfig config __attribute__((aligned(64))) = {.palette_id = 1};
  for (int t = 0; t < 8; t++) {
    config.rows[t] = 16;
    config.colsb[t] = 64;
  }
  AMX_LOADCFG(&config);
  amx_thread_ready = 1;
  pthread_once(&amx_key_once, amx_key_create);
  pthread_setspecific(amx_key, &amx_thread_ready);
}

AMX_TARGET void gemm_amx_thread_release(void) {
  if (!amx_thread_ready)
    return;
  AMX_RELEASE();
  amx_thread_ready = 0;
  pthread_setspecific(amx_key, NULL);
}

/* C[m][n] = sum_k A[m][k] * B[n][k] for m, n <= 32, with 2 x 2 C tiles in flight. A is BF16 row-major with leading
 * dimension lda (elements) and must be readable, zero-padded, up to ceil(m, 16) rows and ceil(k, 32) columns. B is
 * packed by pack_bf16_vnni for the same k; bp points at the first of its (at most two) 16-column tiles. C needs room
 * for whole 16 x 16 tiles (ceil(m, 16) rows by ceil(n, 16) columns); only the first m x n values are meaningful. */
AMX_TARGET void amx_bf16_tile_gemm(float *c, size_t ldc, const uint16_t *a, size_t lda, const uint16_t *bp, int m,
                                   int n, int k) {
  amx_thread_configure();
  const size_t kp = round_up((size_t)k, 32), tile = kp * 16;
  const int wide = n > 16, tall = m > 16;
  AMX_ZERO(0);
  if (wide)
    AMX_ZERO(1);
  if (tall) {
    AMX_ZERO(2);
    if (wide)
      AMX_ZERO(3);
  }
  for (size_t k0 = 0; k0 < kp; k0 += 32) {
    AMX_LOADD(4, a + k0, lda * sizeof(uint16_t));
    if (tall)
      AMX_LOADD(5, a + 16 * lda + k0, lda * sizeof(uint16_t));
    AMX_LOADD(6, bp + (k0 / 32) * 512, 64);
    if (wide)
      AMX_LOADD(7, bp + tile + (k0 / 32) * 512, 64);
    AMX_DPBF16PS(0, 4, 6);
    if (wide)
      AMX_DPBF16PS(1, 4, 7);
    if (tall) {
      AMX_DPBF16PS(2, 5, 6);
      if (wide)
        AMX_DPBF16PS(3, 5, 7);
    }
  }
  AMX_STORED(0, c, ldc * sizeof(float));
  if (wide)
    AMX_STORED(1, c + 16, ldc * sizeof(float));
  if (tall) {
    AMX_STORED(2, c + 16 * ldc, ldc * sizeof(float));
    if (wide)
      AMX_STORED(3, c + 16 * ldc + 16, ldc * sizeof(float));
  }
}

/* The same for signed int8: C[m][n] = sum_k A[m][k] * B[n][k] as exact int32, K in blocks of 64. A is readable up to
 * ceil(k, 64) columns, zero-padded; B is packed by pack_s8_vnni. */
AMX_TARGET void amx_s8_tile_gemm(int32_t *c, size_t ldc, const int8_t *a, size_t lda, const int8_t *bp, int m, int n,
                                 int k) {
  amx_thread_configure();
  const size_t kp = round_up((size_t)k, 64), tile = kp * 16;
  const int wide = n > 16, tall = m > 16;
  AMX_ZERO(0);
  if (wide)
    AMX_ZERO(1);
  if (tall) {
    AMX_ZERO(2);
    if (wide)
      AMX_ZERO(3);
  }
  for (size_t k0 = 0; k0 < kp; k0 += 64) {
    AMX_LOADD(4, a + k0, lda);
    if (tall)
      AMX_LOADD(5, a + 16 * lda + k0, lda);
    AMX_LOADD(6, bp + (k0 / 64) * 1024, 64);
    if (wide)
      AMX_LOADD(7, bp + tile + (k0 / 64) * 1024, 64);
    AMX_DPBSSD(0, 4, 6);
    if (wide)
      AMX_DPBSSD(1, 4, 7);
    if (tall) {
      AMX_DPBSSD(2, 5, 6);
      if (wide)
        AMX_DPBSSD(3, 5, 7);
    }
  }
  AMX_STORED(0, c, ldc * sizeof(int32_t));
  if (wide)
    AMX_STORED(1, c + 16, ldc * sizeof(int32_t));
  if (tall) {
    AMX_STORED(2, c + 16 * ldc, ldc * sizeof(int32_t));
    if (wide)
      AMX_STORED(3, c + 16 * ldc + 16, ldc * sizeof(int32_t));
  }
}
#else
void gemm_amx_thread_release(void) {}
void amx_bf16_tile_gemm(float *c, size_t ldc, const uint16_t *a, size_t lda, const uint16_t *bp, int m, int n, int k) {
  (void)c, (void)ldc, (void)a, (void)lda, (void)bp, (void)m, (void)n, (void)k;
  fail("AMX tile kernels are not compiled into this build");
}
void amx_s8_tile_gemm(int32_t *c, size_t ldc, const int8_t *a, size_t lda, const int8_t *bp, int m, int n, int k) {
  (void)c, (void)ldc, (void)a, (void)lda, (void)bp, (void)m, (void)n, (void)k;
  fail("AMX tile kernels are not compiled into this build");
}
#endif

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

/* ---- BF16 GEMM (C5) --------------------------------------------------------------------------------------------- */

/* The reference tile kernel: the same contract and the same result as amx_bf16_tile_gemm, in plain C. Every product of
 * two BF16 values is exact in FP32, and the sum runs in ascending K, so this is the scalar BF16 reference ("convert,
 * then multiply in FP32") and it equals the software tile model bit for bit. */
static void bf16_micro_portable(float *c, size_t ldc, const uint16_t *a, size_t lda, const uint16_t *bp, int m, int n,
                                int k) {
  const size_t kp = round_up((size_t)k, 32);
  for (int i = 0; i < m; i++)
    for (int j = 0; j < n; j++) {
      const uint16_t *b = bp + (size_t)(j / 16) * kp * 16 + (size_t)(j % 16) * 2;
      float sum = 0.0f;
      for (int l = 0; l < k; l++)
        sum += f32_from_bf16(a[(size_t)i * lda + l]) * f32_from_bf16(b[(size_t)(l / 2) * 32 + (size_t)(l & 1)]);
      c[(size_t)i * ldc + j] = sum;
    }
}

typedef void (*Bf16Micro)(float *c, size_t ldc, const uint16_t *a, size_t lda, const uint16_t *bp, int m, int n, int k);

typedef struct {
  const float *src;
  uint16_t *panel;
  size_t src_ld, panel_ld;
  int m, k, count;
  GemmBf16Segment segment[GEMM_MAX_SEGMENTS];
  Bf16Micro micro;
} Bf16Job;

/* One chunk is one panel row: the activations of one token converted to BF16 (round to nearest even), the K padding
 * zeroed, and the padding rows below the last token zeroed too. */
static void bf16_convert_rows(const void *p, int s, int first, int last) {
  const Bf16Job *job = p;
  (void)s;
  for (int row = first; row < last; row++) {
    uint16_t *dst = job->panel + (size_t)row * job->panel_ld;
    int done = 0;
    if (row < job->m) {
      const float *src = job->src + (size_t)row * job->src_ld;
      for (; done < job->k; done++)
        dst[done] = bf16_from_f32(src[done]);
    }
    memset(dst + done, 0, ((size_t)job->panel_ld - (size_t)done) * sizeof(uint16_t));
  }
}

enum { BF16_NC = 32, BF16_MB = 32 };

static void bf16_chunks(const void *p, int s, int first, int last) {
  const Bf16Job *job = p;
  const GemmBf16Segment *seg = &job->segment[s];
  const size_t tile = round_up((size_t)job->k, 32) * 16;
  _Alignas(64) float acc[BF16_MB * 32], gate[BF16_MB * 32];
  for (int chunk = first; chunk < last; chunk++) {
    int n0 = chunk * BF16_NC, nc = seg->n - n0 < BF16_NC ? seg->n - n0 : BF16_NC;
    const uint16_t *w = seg->w + (size_t)(n0 / 16) * tile;
    for (int m0 = 0; m0 < job->m; m0 += BF16_MB) {
      int mb = job->m - m0 < BF16_MB ? job->m - m0 : BF16_MB;
      const uint16_t *a = job->panel + (size_t)m0 * job->panel_ld;
      job->micro(acc, 32, a, job->panel_ld, w, mb, nc, job->k);
      if (seg->kind == GEMM_SWIGLU)
        job->micro(gate, 32, a, job->panel_ld, seg->w3 + (size_t)(n0 / 16) * tile, mb, nc, job->k);
      for (int r = 0; r < mb; r++) {
        float *out = seg->c + (size_t)(m0 + r) * seg->ldc + n0;
        const float *sum = acc + (size_t)r * 32, *g = gate + (size_t)r * 32;
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

size_t k_gemm_bf16_panel_bytes(int m, int k) {
  return round_up((size_t)(m > 0 ? m : 1), 16) * (round_up((size_t)(k > 0 ? k : 1), 32) + 32) * sizeof(uint16_t);
}

void k_gemm_bf16(const float *a, int lda, int m, int k, const GemmBf16Segment *segments, int count,
                 AlignedBuffer *panel) {
  if (count < 1 || count > GEMM_MAX_SEGMENTS)
    fail("GEMM needs 1..%d segments, got %d", GEMM_MAX_SEGMENTS, count);
  for (int s = 0; s < count; s++)
    if (!segments[s].w || (segments[s].kind == GEMM_SWIGLU && !segments[s].w3))
      fail("BF16 GEMM segment %d has no packed weights", s);
  if (m <= 0 || k <= 0)
    return;
  if (panel->bytes < k_gemm_bf16_panel_bytes(m, k))
    fail("BF16 activation panel holds %zu bytes but %zu are needed", panel->bytes, k_gemm_bf16_panel_bytes(m, k));
  Bf16Job job = {.src = a,
                 .panel = panel->data,
                 .src_ld = (size_t)lda,
                 .panel_ld = round_up((size_t)k, 32) + 32,
                 .m = m,
                 .k = k,
                 .count = count,
                 .micro = gemm_amx_active() ? amx_bf16_tile_gemm : bf16_micro_portable};
  int rows = (int)round_up((size_t)m, 16);
  kernels_parallel_segments(1, &rows, bf16_convert_rows, &job);
  int chunks[GEMM_MAX_SEGMENTS];
  for (int s = 0; s < count; s++) {
    job.segment[s] = segments[s];
    chunks[s] = (segments[s].n + BF16_NC - 1) / BF16_NC;
  }
  kernels_parallel_segments(count, chunks, bf16_chunks, &job);
}

/* ---- packing many matrices on the worker team ----------------------------------------------------------------------
 */

typedef struct {
  const PackBf16 *jobs;
  int count;
} PackAll;

/* A chunk is one 16-row tile; chunk numbers run over the tiles of all matrices laid end to end. */
static void pack_bf16_chunks(const void *p, int s, int first, int last) {
  const PackAll *all = p;
  (void)s;
  int base = 0;
  for (int i = 0; i < all->count && base < last; i++) {
    const PackBf16 *job = &all->jobs[i];
    int tiles = (job->rows + 15) / 16;
    if (base + tiles > first)
      pack_bf16_vnni_tiles(job->dst, job->w, job->rows, job->cols, job->ld, first > base ? first - base : 0,
                           last < base + tiles ? last - base : tiles);
    base += tiles;
  }
}

void pack_bf16_vnni_many(const PackBf16 *jobs, int count) {
  PackAll all = {jobs, count};
  long long tiles = 0;
  for (int i = 0; i < count; i++)
    tiles += (jobs[i].rows + 15) / 16;
  if (tiles > INT32_MAX)
    fail("too many weight tiles to pack: %lld", tiles);
  int chunks = (int)tiles;
  kernels_parallel_segments(1, &chunks, pack_bf16_chunks, &all);
}
