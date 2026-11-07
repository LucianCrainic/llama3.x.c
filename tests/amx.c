/* AMX enablement (C4). Built twice:
 *   test-amx        against the real kernels: the packing helpers run everywhere; the tile microkernels run only on a
 *                   CPU with AMX and the kernel's permission, otherwise the test prints "amx unavailable" and skips
 * them. test-amx-emul   with GEMM_EMULATE_AMX: the same microkernel code on the software tile model of
 * tests/amx_emul.h, so the tile loops, VNNI indexing and accumulation order are executed and checked on any host. The
 * BF16 kernel is compared with a double reference on BF16-rounded inputs (accumulation error only, 2e-6 of sum |w x|);
 * the int8 kernel must equal a scalar int32 reference exactly. */
#include "gemm.h"
#include "kernels.h"
#include "workers.h"
#include <assert.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static _Thread_local unsigned long long rng = 88172645463325252ull; /* per thread: worker jobs draw numbers too */
static int failures, checks;

static unsigned long long next(void) {
  rng ^= rng << 13;
  rng ^= rng >> 7;
  rng ^= rng << 17;
  return rng;
}
static float uniform(void) {
  return (float)((double)(next() >> 40) / (double)(1ull << 24)) * 2.0f - 1.0f;
}
static void *zeroed(size_t count, size_t size) {
  void *p = calloc(count ? count : 1, size);
  assert(p);
  return p;
}
static void fail_check(const char *what, int m, int n, int k, const char *detail) {
  if (failures++ < 20)
    fprintf(stderr, "%s m=%d n=%d k=%d: %s\n", what, m, n, k, detail);
}
static size_t up(size_t v, size_t to) {
  return (v + to - 1) / to * to;
}

/* ---- packing: unpack through the tile geometry and compare bitwise ------------------------------------------------
 */

/* Reads the packed buffer the way the hardware reads a tile: tile nt, K block kb, pair row r holds, for channel c,
 * the two values of W[16 nt + c][kb + 2r], W[16 nt + c][kb + 2r + 1] at element 2c, 2c + 1. */
static void test_pack_bf16(int rows, int cols, size_t ld) {
  float *w = zeroed((size_t)rows * ld, sizeof(float));
  for (size_t i = 0; i < (size_t)rows * ld; i++)
    w[i] = uniform() * 4;
  size_t elements = bf16_vnni_elements(rows, cols);
  uint16_t *packed = zeroed(elements, 2), *again = zeroed(elements, 2);
  memset(packed, 0xAB, elements * 2); /* every element must be written, padding included */
  pack_bf16_vnni(packed, w, rows, cols, ld);
  size_t kp = up((size_t)cols, 32);
  int ntiles = (rows + 15) / 16, bad = 0;
  checks++;
  if (elements != (size_t)ntiles * 16 * kp)
    bad++;
  for (int nt = 0; nt < ntiles; nt++)
    for (size_t kb = 0; kb < kp; kb += 32) {
      const uint16_t *tile = packed + (size_t)nt * kp * 16 + (kb / 32) * 512;
      for (int r = 0; r < 16; r++)
        for (int c = 0; c < 16; c++)
          for (int e = 0; e < 2; e++) {
            size_t row = (size_t)nt * 16 + c, col = kb + 2 * (size_t)r + e;
            uint16_t want = row < (size_t)rows && col < (size_t)cols ? bf16_from_f32(w[row * ld + col]) : 0;
            bad += tile[r * 32 + 2 * c + e] != want;
          }
    }
  /* The tile-range form, split anywhere, must give the same bytes. */
  memset(again, 0xCD, elements * 2);
  int split = ntiles / 2;
  pack_bf16_vnni_tiles(again, w, rows, cols, ld, split, ntiles);
  pack_bf16_vnni_tiles(again, w, rows, cols, ld, 0, split);
  bad += memcmp(packed, again, elements * 2) != 0;
  if (bad)
    fail_check("pack_bf16_vnni", rows, 0, cols, "differs from the tile-geometry reference");
  free(w), free(packed), free(again);
}

static void test_pack_s8(int rows, int cols, size_t ld) {
  int8_t *w = zeroed((size_t)rows * ld, 1);
  for (size_t i = 0; i < (size_t)rows * ld; i++)
    w[i] = (int8_t)((int)(next() % 255) - 127);
  size_t bytes = s8_vnni_bytes(rows, cols);
  int8_t *packed = zeroed(bytes, 1), *again = zeroed(bytes, 1);
  memset(packed, 0x55, bytes);
  pack_s8_vnni(packed, w, rows, cols, ld);
  size_t kp = up((size_t)cols, 64);
  int ntiles = (rows + 15) / 16, bad = 0;
  checks++;
  if (bytes != (size_t)ntiles * 16 * kp)
    bad++;
  for (int nt = 0; nt < ntiles; nt++)
    for (size_t kb = 0; kb < kp; kb += 64) {
      const int8_t *tile = packed + (size_t)nt * kp * 16 + (kb / 64) * 1024;
      for (int r = 0; r < 16; r++)
        for (int c = 0; c < 16; c++)
          for (int e = 0; e < 4; e++) {
            size_t row = (size_t)nt * 16 + c, col = kb + 4 * (size_t)r + e;
            int8_t want = row < (size_t)rows && col < (size_t)cols ? w[row * ld + col] : 0;
            bad += tile[r * 64 + 4 * c + e] != want;
          }
    }
  memset(again, 0x66, bytes);
  int split = ntiles / 2;
  pack_s8_vnni_tiles(again, w, rows, cols, ld, split, ntiles);
  pack_s8_vnni_tiles(again, w, rows, cols, ld, 0, split);
  bad += memcmp(packed, again, bytes) != 0;
  if (bad)
    fail_check("pack_s8_vnni", rows, 0, cols, "differs from the tile-geometry reference");
  free(w), free(packed), free(again);
}

static void test_packing(void) {
  const int rows[] = {1, 15, 16, 17, 33, 64}, cols[] = {1, 2, 3, 31, 32, 33, 63, 64, 65, 129};
  for (size_t i = 0; i < sizeof(rows) / sizeof(*rows); i++)
    for (size_t j = 0; j < sizeof(cols) / sizeof(*cols); j++) {
      test_pack_bf16(rows[i], cols[j], (size_t)cols[j]);
      test_pack_bf16(rows[i], cols[j], (size_t)cols[j] + 7); /* a padded leading dimension */
      test_pack_s8(rows[i], cols[j], (size_t)cols[j]);
      test_pack_s8(rows[i], cols[j], (size_t)cols[j] + 5);
    }
}

static void test_aligned_buffer(void) {
  size_t before = memory_current[MEM_ACTIVATION];
  AlignedBuffer b;
  for (size_t bytes = 1; bytes < 300; bytes += 37) {
    aligned_buffer_alloc(&b, bytes, MEM_ACTIVATION);
    checks++;
    if (((uintptr_t)b.data & 63) || memory_current[MEM_ACTIVATION] < before + bytes)
      fail_check("aligned_buffer", (int)bytes, 0, 0, "not 64-byte aligned or not accounted");
    memset(b.data, 1, bytes); /* the whole range is usable */
    aligned_buffer_free(&b);
    checks++;
    if (memory_current[MEM_ACTIVATION] != before)
      fail_check("aligned_buffer", (int)bytes, 0, 0, "accounting not returned");
  }
}

/* ---- tile microkernels ---------------------------------------------------------------------------------------------
 */

typedef struct {
  int m, n, k;
  int bf16;
  int quiet; /* worker threads: no shared state, no accounting */
  int bad;
  double worst; /* BF16: worst error as a fraction of sum |w x|; int8 unused */
} KernelCase;

/* Aligned memory without the engine's accounting, which only the coordinator thread may touch. */
static void plain_buffer(AlignedBuffer *b, size_t bytes) {
  b->base = zeroed(bytes + 63, 1);
  b->data = (void *)(((uintptr_t)b->base + 63) & ~(uintptr_t)63);
  b->bytes = bytes;
}
static void plain_free(AlignedBuffer *b) {
  free(b->base);
}

/* One shape end to end: build padded operands in aligned memory, pack B, run the kernel, compare. */
static void run_case(KernelCase *kc) {
  const int m = kc->m, n = kc->n, k = kc->k;
  const size_t lda_pad = 32; /* a leading dimension wider than the padded K, so tile rows are not contiguous */
  AlignedBuffer abuf, cbuf, bbuf;
  if (kc->bf16) {
    size_t kp = up((size_t)k, 32), lda = kp + lda_pad, rows = up((size_t)m, 16);
    plain_buffer(&abuf, rows * lda * 2);
    plain_buffer(&cbuf, rows * 32 * sizeof(float));
    plain_buffer(&bbuf, bf16_vnni_elements(n, k) * 2);
    float *w = zeroed((size_t)n * k, sizeof(float)), *x = zeroed((size_t)m * k, sizeof(float));
    for (int i = 0; i < n * k; i++)
      w[i] = uniform();
    for (int i = 0; i < m * k; i++)
      x[i] = uniform() * 2;
    uint16_t *a = abuf.data;
    memset(a, 0, rows * lda * 2);
    for (int i = 0; i < m; i++)
      for (int j = 0; j < k; j++)
        a[(size_t)i * lda + j] = bf16_from_f32(x[(size_t)i * k + j]);
    pack_bf16_vnni(bbuf.data, w, n, k, (size_t)k);
    float *c = cbuf.data;
    for (size_t i = 0; i < rows * 32; i++)
      c[i] = -12345.0f; /* poison: the kernel writes whole tiles, the rest must stay */
    amx_bf16_tile_gemm(c, 32, a, lda, bbuf.data, m, n, k);
    for (int i = 0; i < m; i++)
      for (int j = 0; j < n; j++) {
        double ref = 0, mag = 0;
        for (int l = 0; l < k; l++) { /* BF16-rounded operands, double products */
          double p = (double)f32_from_bf16(bf16_from_f32(x[(size_t)i * k + l])) *
                     (double)f32_from_bf16(bf16_from_f32(w[(size_t)j * k + l]));
          ref += p, mag += fabs(p);
        }
        double err = fabs((double)c[(size_t)i * 32 + j] - ref);
        if (!(err <= 2e-6 * mag + 1e-30)) {
          kc->bad++;
          if (!kc->quiet)
            fail_check("amx_bf16_tile_gemm", m, n, k, "outside 2e-6 sum|w x|");
        }
        if (mag > 0 && err / mag > kc->worst)
          kc->worst = err / mag;
      }
    free(w), free(x);
  } else {
    size_t kp = up((size_t)k, 64), lda = kp + lda_pad, rows = up((size_t)m, 16);
    plain_buffer(&abuf, rows * lda);
    plain_buffer(&cbuf, rows * 32 * sizeof(int32_t));
    plain_buffer(&bbuf, s8_vnni_bytes(n, k));
    int8_t *w = zeroed((size_t)n * k, 1), *x = zeroed((size_t)m * k, 1);
    for (int i = 0; i < n * k; i++)
      w[i] = (int8_t)((int)(next() % 256) - 128); /* -128 included: the instruction is signed x signed */
    for (int i = 0; i < m * k; i++)
      x[i] = (int8_t)((int)(next() % 256) - 128);
    int8_t *a = abuf.data;
    memset(a, 0, rows * lda);
    for (int i = 0; i < m; i++)
      memcpy(a + (size_t)i * lda, x + (size_t)i * k, (size_t)k);
    pack_s8_vnni(bbuf.data, w, n, k, (size_t)k);
    int32_t *c = cbuf.data;
    for (size_t i = 0; i < rows * 32; i++)
      c[i] = -123456789;
    amx_s8_tile_gemm(c, 32, a, lda, bbuf.data, m, n, k);
    for (int i = 0; i < m; i++)
      for (int j = 0; j < n; j++) {
        int32_t ref = 0;
        for (int l = 0; l < k; l++)
          ref += (int32_t)x[(size_t)i * k + l] * (int32_t)w[(size_t)j * k + l];
        if (c[(size_t)i * 32 + j] != ref) {
          kc->bad++;
          if (!kc->quiet)
            fail_check("amx_s8_tile_gemm", m, n, k, "int32 result not exact");
        }
      }
    free(w), free(x);
  }
  plain_free(&abuf), plain_free(&cbuf), plain_free(&bbuf);
}

static void run_cases(int bf16) {
  const int ms[] = {1, 5, 15, 16, 17, 31, 32}, ns[] = {1, 3, 16, 17, 31, 32},
            ks[] = {1, 2, 31, 32, 33, 63, 64, 65, 100, 257, 1024};
  double worst = 0;
  int bad = 0;
  for (size_t i = 0; i < sizeof(ms) / sizeof(*ms); i++)
    for (size_t j = 0; j < sizeof(ns) / sizeof(*ns); j++)
      for (size_t l = 0; l < sizeof(ks) / sizeof(*ks); l++) {
        KernelCase kc = {.m = ms[i], .n = ns[j], .k = ks[l], .bf16 = bf16};
        run_case(&kc);
        checks++;
        bad += kc.bad;
        if (kc.worst > worst)
          worst = kc.worst;
      }
  if (bf16)
    printf("amx: bf16 tile gemm %s (worst error %.2e of sum|w x|, budget 2e-6)\n", bad ? "FAILED" : "ok", worst);
  else
    printf("amx: int8 tile gemm %s (exact int32)\n", bad ? "FAILED" : "ok");
}

/* Every worker configures its own tiles at its first job: several threads run the kernels at once. */
static int worker_failures;
static void worker_job(const void *job, int worker, int workers) {
  (void)job, (void)workers;
  KernelCase kc = {.m = 32 - worker % 3, .n = 32 - worker % 5, .k = 100 + worker, .bf16 = worker & 1, .quiet = 1};
  run_case(&kc);
  __atomic_fetch_add(&worker_failures, kc.bad, __ATOMIC_RELAXED);
}

static void test_teams(void) {
  PoolConfig config = {.teams = 1, .threads = {6, 0}, .cpus = {NULL, NULL}};
  pool_start(&config);
  for (int round = 0; round < 3; round++)
    team_run(pool_team(0), worker_job, NULL);
  checks++;
  pool_stop(); /* the workers release their tiles as they exit */
  if (worker_failures)
    fail_check("worker threads", 0, 0, 0, "kernel results wrong on a worker");
  printf("amx: 6 worker threads, per-thread tile configuration %s\n", worker_failures ? "FAILED" : "ok");
}

static void test_selection(void) {
  checks++;
  setenv("LLAMA_KERNELS", "auto", 1);
  kernels_init();
  if (!strcmp(kernels_name(), "amx"))
    fail_check("kernels_init", 0, 0, 0, "auto selected amx");
  if (gemm_amx_unavailable()) {
    checks++;
    if (kernels_select(KERNELS_AMX) || gemm_amx_active())
      fail_check("kernels_select", 0, 0, 0, "amx selected although unavailable");
    return;
  }
  checks += 2;
  if (!kernels_select(KERNELS_AMX) || strcmp(kernels_name(), "amx") || !kernels_amx_active())
    fail_check("kernels_select", 0, 0, 0, "amx not selected although available");
  kernels_select(KERNELS_SCALAR);
  if (kernels_amx_active() || !strcmp(kernels_name(), "amx"))
    fail_check("kernels_select", 0, 0, 0, "amx still active after selecting scalar");
}

int main(void) {
  test_packing();
  test_aligned_buffer();
  printf("amx: packing helpers %s\n", failures ? "FAILED" : "ok");
  const char *why = gemm_amx_unavailable();
  if (why) {
    printf("amx unavailable: %s\n", why);
  } else {
    printf("amx: %s tiles, permission %s\n", gemm_amx_emulated() ? "emulated" : "hardware", gemm_amx_permission());
    run_cases(1);
    run_cases(0);
    test_teams();
  }
  test_selection();
  assert(memory_current[MEM_ACTIVATION] == 0 && memory_current[MEM_WEIGHTS_PACKED] == 0);
  gemm_amx_thread_release();
  printf("amx: %d checks, %d failures\n", checks, failures);
  return failures != 0;
}
