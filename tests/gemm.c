/* GEMM regression (C2): every variant of the driver against a double-precision reference, plus bitwise agreement with
 * the per-row GEMV on the scalar instruction set. Built twice: against the real kernels (the AVX-512 microkernel runs
 * where the CPU has it) and with GEMM_EMULATE_AVX512, where the same microkernel code runs on an emulation.
 *   test-gemm             uses LLAMA_TEST_TEAMS (default 1,3,7) for worker teams on top of the single-thread path */
#include "gemm.h"
#include "kernels.h"
#include "workers.h"
#include <assert.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static unsigned long long rng = 88172645463325252ull;
static int failures, checks;
static char variant[96] = "";

static float uniform(void) {
  rng ^= rng << 13;
  rng ^= rng >> 7;
  rng ^= rng << 17;
  return (float)((double)(rng >> 40) / (double)(1ull << 24)) * 2.0f - 1.0f;
}
static void fill(float *x, size_t n, float scale) {
  for (size_t i = 0; i < n; i++)
    x[i] = uniform() * scale;
}
static void *zeroed(size_t count, size_t size) {
  void *p = calloc(count ? count : 1, size);
  assert(p);
  return p;
}
static void check(const char *what, double got, double expected, double tolerance, int m, int n, int k) {
  checks++;
  if (!(fabs(got - expected) <= tolerance) && failures++ < 20)
    fprintf(stderr, "%s [%s] m=%d n=%d k=%d: got %.9g expected %.9g (tolerance %.3g)\n", what, variant, m, n, k, got,
            expected, tolerance);
}
static double silu(double v) {
  return v / (1.0 + exp(-v));
}

typedef enum { RUN_DEFAULT, RUN_PORTABLE, RUN_SIMD } Runner;
static void run(Runner runner, int kc, const float *a, int lda, int m, int k, const GemmSegment *seg, int count) {
  if (runner == RUN_DEFAULT)
    k_gemm_f32(a, lda, m, k, seg, count);
  else if (runner == RUN_PORTABLE)
    k_gemm_f32_portable(a, lda, m, k, seg, count, kc);
  else
    assert(k_gemm_f32_simd_test(a, lda, m, k, seg, count, kc));
}

/* One shape, every kind, with padded leading dimensions and sentinels around the output. */
static void test_shape(Runner runner, int kc, int m, int n, int k) {
  int lda = k + 3, ldc = n + 7;
  size_t ldw = (size_t)k + 5;
  float *a = zeroed((size_t)m * lda, 4), *w = zeroed(ldw * n, 4), *w3 = zeroed(ldw * n, 4);
  float *c = zeroed((size_t)m * ldc, 4), *base = zeroed((size_t)m * ldc, 4);
  fill(a, (size_t)m * lda, 1), fill(w, ldw * n, 1), fill(w3, ldw * n, 1), fill(base, (size_t)m * ldc, 4);
  double *ref = zeroed((size_t)m * n, 8), *mag = zeroed((size_t)m * n, 8), *ref3 = zeroed((size_t)m * n, 8),
         *mag3 = zeroed((size_t)m * n, 8);
  for (int i = 0; i < m; i++)
    for (int j = 0; j < n; j++)
      for (int l = 0; l < k; l++) {
        double p = (double)a[(size_t)i * lda + l] * w[(size_t)j * ldw + l];
        double q = (double)a[(size_t)i * lda + l] * w3[(size_t)j * ldw + l];
        ref[(size_t)i * n + j] += p, mag[(size_t)i * n + j] += fabs(p);
        ref3[(size_t)i * n + j] += q, mag3[(size_t)i * n + j] += fabs(q);
      }
  const GemmKind kinds[] = {GEMM_STORE, GEMM_ADD, GEMM_SWIGLU, GEMM_PARTIAL};
  for (int t = 0; t < 4; t++) {
    memcpy(c, base, (size_t)m * ldc * 4);
    GemmSegment seg = {.c = c, .ldc = ldc, .w = w, .w3 = w3, .ldw = ldw, .n = n, .kind = kinds[t]};
    run(runner, kc, a, lda, m, k, &seg, 1);
    for (int i = 0; i < m; i++) {
      for (int j = 0; j < n; j++) {
        size_t x = (size_t)i * n + j, y = (size_t)i * ldc + j;
        double got = c[y], r = ref[x], g = ref3[x];
        if (kinds[t] == GEMM_ADD)
          check("gemm.add", got, base[y] + r, 2e-5 * (mag[x] + fabs(base[y])) + 1e-6, m, n, k);
        else if (kinds[t] == GEMM_SWIGLU)
          check("gemm.swiglu", got, silu(r) * g, 2e-5 * (mag[x] * (fabs(g) + 1) + mag3[x] * (fabs(r) + 1)) + 1e-6, m, n,
                k);
        else
          check(kinds[t] == GEMM_STORE ? "gemm.store" : "gemm.partial", got, r, 2e-5 * mag[x] + 1e-6, m, n, k);
      }
      for (int j = n; j < ldc; j++) /* the padding between rows is not ours to write */
        check("gemm.padding", c[(size_t)i * ldc + j], base[(size_t)i * ldc + j], 0, m, n, k);
    }
  }
  free(a), free(w), free(w3), free(c), free(base), free(ref), free(mag), free(ref3), free(mag3);
}

/* QKV: three segments of different widths share A and one parallel region. */
static void test_segments(Runner runner, int kc, int m, int k) {
  const int widths[3] = {40, 9, 9};
  float *a = zeroed((size_t)m * k, 4);
  fill(a, (size_t)m * k, 1);
  float *w[3], *c[3];
  GemmSegment seg[3];
  for (int s = 0; s < 3; s++) {
    w[s] = zeroed((size_t)widths[s] * k, 4), c[s] = zeroed((size_t)m * widths[s], 4);
    fill(w[s], (size_t)widths[s] * k, 1);
    seg[s] = (GemmSegment){.c = c[s], .ldc = widths[s], .w = w[s], .ldw = (size_t)k, .n = widths[s], .kind = GEMM_STORE};
  }
  run(runner, kc, a, k, m, k, seg, 3);
  for (int s = 0; s < 3; s++) {
    for (int i = 0; i < m; i++)
      for (int j = 0; j < widths[s]; j++) {
        double r = 0, g = 0;
        for (int l = 0; l < k; l++) {
          double p = (double)a[(size_t)i * k + l] * w[s][(size_t)j * k + l];
          r += p, g += fabs(p);
        }
        check("gemm.segment", c[s][(size_t)i * widths[s] + j], r, 2e-5 * g + 1e-6, m, widths[s], k);
      }
    free(w[s]), free(c[s]);
  }
  free(a);
}

/* With the scalar instruction set one K block and ascending k reproduce the per-row GEMV exactly. */
static void test_bitwise_with_gemv(int m, int n, int k) {
  float *a = zeroed((size_t)m * k, 4), *w = zeroed((size_t)n * k, 4), *w3 = zeroed((size_t)n * k, 4);
  float *c = zeroed((size_t)m * n, 4), *g = zeroed((size_t)m * n, 4), *base = zeroed((size_t)m * n, 4);
  fill(a, (size_t)m * k, 1), fill(w, (size_t)n * k, 1), fill(w3, (size_t)n * k, 1), fill(base, (size_t)m * n, 4);
  for (int t = 0; t < 3; t++) {
    memcpy(c, base, (size_t)m * n * 4), memcpy(g, base, (size_t)m * n * 4);
    GemmSegment seg = {.c = c, .ldc = n, .w = w, .w3 = w3, .ldw = (size_t)k, .n = n,
                       .kind = t == 0 ? GEMM_STORE : t == 1 ? GEMM_ADD : GEMM_SWIGLU};
    k_gemm_f32(a, k, m, k, &seg, 1);
    for (int i = 0; i < m; i++) {
      if (t == 0)
        k_matmul(g + (size_t)i * n, a + (size_t)i * k, w, k, n);
      else if (t == 1)
        k_matmul_add(g + (size_t)i * n, a + (size_t)i * k, w, k, n);
      else
        k_matmul_swiglu(g + (size_t)i * n, a + (size_t)i * k, w, w3, k, n);
    }
    checks++;
    if (memcmp(c, g, (size_t)m * n * 4) && failures++ < 20)
      fprintf(stderr, "gemm vs gemv [%s] kind=%d m=%d n=%d k=%d: not bitwise equal\n", variant, t, m, n, k);
  }
  free(a), free(w), free(w3), free(c), free(g), free(base);
}

static void test_all(Runner runner, int kc) {
  const int ms[] = {1, 3, 4, 5, 16, 31, 64}, ns[] = {1, 17, 64, 100}, ks[] = {1, 16, 100, 257};
  for (size_t i = 0; i < sizeof(ms) / sizeof(*ms); i++)
    for (size_t j = 0; j < sizeof(ns) / sizeof(*ns); j++)
      for (size_t l = 0; l < sizeof(ks) / sizeof(*ks); l++)
        test_shape(runner, kc, ms[i], ns[j], ks[l]);
  test_shape(runner, kc, 260, 33, 100); /* more rows than one M block */
  test_shape(runner, kc, 64, 300, 1100); /* several chunks and K blocks */
  test_segments(runner, kc, 5, 100);
  test_segments(runner, kc, 33, 70);
}

static void test_variants(const char *name, int simd_ok) {
  snprintf(variant, sizeof(variant), "%s default", name);
  test_all(RUN_DEFAULT, 0);
  const int kcs[] = {0, 7, 16, 100, 1024};
  for (size_t i = 0; i < sizeof(kcs) / sizeof(*kcs); i++) {
    snprintf(variant, sizeof(variant), "%s portable kc=%d", name, kcs[i]);
    test_all(RUN_PORTABLE, kcs[i]);
  }
  if (simd_ok) {
    const int simd_kcs[] = {16, 17, 100, 1024};
    for (size_t i = 0; i < sizeof(simd_kcs) / sizeof(*simd_kcs); i++) {
      snprintf(variant, sizeof(variant), "%s simd kc=%d", name, simd_kcs[i]);
      test_all(RUN_SIMD, simd_kcs[i]);
    }
  }
}

int main(void) {
#ifdef GEMM_EMULATE_AVX512
  const int emulated = 1;
#else
  const int emulated = 0;
#endif
  const KernelIsa isas[2] = {KERNELS_SCALAR, KERNELS_AVX512};
  int team_sizes[8] = {1, 3, 7};
  size_t team_size_count = 3;
  const char *override = getenv("LLAMA_TEST_TEAMS");
  if (override && *override) {
    team_size_count = 0;
    for (char *copy = strdup(override), *token = strtok(copy, ","); token && team_size_count < 8;
         token = strtok(NULL, ","))
      team_sizes[team_size_count++] = atoi(token);
  }
  for (int i = 0; i < 2; i++) {
    if (!kernels_select(isas[i])) {
      printf("gemm: %s unavailable on this CPU, skipped\n", i ? "avx512" : "scalar");
      continue;
    }
    int simd_ok = emulated || i == 1;
    int before = failures;
    test_variants(kernels_name(), simd_ok);
    if (i == 0) {
      snprintf(variant, sizeof(variant), "scalar bitwise");
      test_bitwise_with_gemv(7, 33, 100);
      test_bitwise_with_gemv(64, 20, 257);
      test_bitwise_with_gemv(1, 5, 16);
    }
    printf("gemm: %s%s %s\n", kernels_name(), emulated ? " (emulated microkernel)" : "", failures == before ? "ok" : "FAILED");
    for (size_t t = 0; t < team_size_count; t++) {
      PoolConfig config = {.teams = 1, .threads = {team_sizes[t], 0}, .cpus = {NULL, NULL}};
      pool_start(&config);
      kernels_set_team(pool_team(0));
      before = failures;
      char label[48];
      snprintf(label, sizeof(label), "%s teams:%d", kernels_name(), team_sizes[t]);
      test_variants(label, simd_ok);
      if (i == 0) {
        snprintf(variant, sizeof(variant), "scalar bitwise %s", label);
        test_bitwise_with_gemv(7, 33, 100);
      }
      printf("gemm: %s %s\n", label, failures == before ? "ok" : "FAILED");
      kernels_set_team(NULL);
      pool_stop();
    }
  }
  for (int i = 0; i < MEM_COUNT; i++)
    assert(memory_current[i] == 0);
  printf("gemm: %d checks, %d failures\n", checks, failures);
  return failures != 0;
}
