/* BF16 mode kernels (C5): the FP32 -> BF16 conversion against an independent reference, and the BF16 GEMM driver
 * against a double reference computed on BF16-rounded operands (accumulation error only: 2e-6 of sum |w x|).
 * Built twice, like test-amx: against the real kernels (the portable reference GEMM everywhere, the AMX tiles on a CPU
 * that has them) and with GEMM_EMULATE_AMX, where the AMX path runs on the software tile model and must equal the
 * portable reference bit for bit. */
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
static void bad(const char *what, const char *detail) {
  if (failures++ < 20)
    fprintf(stderr, "%s: %s\n", what, detail);
}
static float from_bits(uint32_t u) {
  float f;
  memcpy(&f, &u, sizeof(f));
  return f;
}
static uint32_t to_bits(float f) {
  uint32_t u;
  memcpy(&u, &f, sizeof(u));
  return u;
}

/* ---- conversion ------------------------------------------------------------------------------------------------ */

/* Round to nearest even from the definition: the two neighbouring BF16 values, exact distances in double, a tie goes
 * to the neighbour with an even last mantissa bit. 2^128 stands in for the value past the largest finite BF16. */
static uint16_t reference_bf16(float x) {
  uint32_t u = to_bits(x), sign = u & 0x80000000u, magnitude = u & 0x7fffffffu;
  if (magnitude > 0x7f800000u) /* NaN: stays a NaN, sign kept, quiet bit set; the payload is not specified */
    return (uint16_t)((sign | 0x7fc00000u) >> 16);
  if (magnitude == 0x7f800000u)
    return (uint16_t)((sign | magnitude) >> 16);
  uint32_t lo = magnitude & 0xffff0000u, hi = lo + 0x10000u;
  double value = fabs((double)x), below = (double)from_bits(lo);
  double above = hi >= 0x7f800000u ? ldexp(1.0, 128) : (double)from_bits(hi);
  double d_lo = value - below, d_hi = above - value;
  uint32_t pick = d_lo < d_hi ? lo : d_hi < d_lo ? hi : ((lo >> 16) & 1) == 0 ? lo : hi;
  return (uint16_t)((sign | pick) >> 16);
}

static int is_nan_bf16(uint16_t v) {
  return (v & 0x7f80) == 0x7f80 && (v & 0x007f);
}

static void check_conversion(uint32_t u) {
  float x = from_bits(u);
  uint16_t got = bf16_from_f32(x), want = reference_bf16(x);
  checks++;
  /* A NaN must come back as a NaN of the same sign with the quiet bit; the payload is not compared. */
  if (isnan(x) ? !(is_nan_bf16(got) && (got & 0x8000) == (want & 0x8000) && (got & 0x40)) : got != want) {
    char detail[96];
    snprintf(detail, sizeof(detail), "bf16_from_f32(0x%08x) = 0x%04x, expected 0x%04x", u, got, want);
    bad("conversion", detail);
  }
  checks++;
  if (!isnan(x) && to_bits(f32_from_bf16(got)) != ((uint32_t)got << 16))
    bad("conversion", "f32_from_bf16 is not the exact widening");
}

static void test_conversion(void) {
  /* Ties and their neighbours at every exponent, for even and odd upper mantissa bits, both signs. */
  for (uint32_t exponent = 0; exponent < 255; exponent++)
    for (uint32_t mantissa_hi = 0; mantissa_hi < 128; mantissa_hi += 1)
      for (uint32_t low = 0; low < 0x10000; low += 0x7fff) /* 0, 0x7fff, 0xfffe */
        for (uint32_t extra = 0; extra < 3; extra++) {
          uint32_t u = (exponent << 23) | (mantissa_hi << 16) | (low == 0xfffe ? 0xffff : low) | 0;
          check_conversion(u + extra), check_conversion(0x80000000u | (u + extra));
        }
  for (uint32_t exponent = 0; exponent < 256; exponent += 1)
    for (uint32_t mantissa_hi = 0; mantissa_hi < 128; mantissa_hi += 5)
      for (int delta = -1; delta <= 1; delta++) { /* the tie 0x8000 and its two neighbours */
        uint32_t u = (exponent << 23) | (mantissa_hi << 16) | (uint32_t)(0x8000 + delta);
        check_conversion(u), check_conversion(u | 0x80000000u);
      }
  /* Specials: zeros, infinities, NaNs (quiet, signalling, payload only below the BF16 bits), the largest finite
   * values and the denormal range. */
  const uint32_t specials[] = {0x00000000, 0x80000000, 0x7f800000, 0xff800000, 0x7fc00000, 0xffc00000, 0x7f800001,
                               0xff800001, 0x7f80ffff, 0x7fffffff, 0xffffffff, 0x7f7fffff, 0xff7fffff, 0x7f7f8000,
                               0x7f7f7fff, 0x7f7f0000, 0x00000001, 0x00007fff, 0x00008000, 0x00008001, 0x00018000,
                               0x0000ffff, 0x00010000, 0x007fffff, 0x00800000, 0x807fffff, 0x80000001, 0x80008000};
  for (size_t i = 0; i < sizeof(specials) / sizeof(*specials); i++)
    check_conversion(specials[i]);
  checks++;
  if (bf16_from_f32(from_bits(0x7f7fffff)) != 0x7f80 || bf16_from_f32(from_bits(0xff7fffff)) != 0xff80 ||
      bf16_from_f32(from_bits(0x7f7f8000)) != 0x7f80 || bf16_from_f32(from_bits(0x7f7f7fff)) != 0x7f7f ||
      bf16_from_f32(from_bits(0x00008000)) != 0x0000 || bf16_from_f32(from_bits(0x00018000)) != 0x0002 ||
      bf16_from_f32(from_bits(0x7f800001)) != 0x7fc0)
    bad("conversion", "a documented corner case is wrong");
  /* A stride sweep over all of the 32-bit space and random values. */
  for (uint64_t u = 0; u < (1ull << 32); u += 0x3f1)
    check_conversion((uint32_t)u);
  for (int i = 0; i < 1000000; i++)
    check_conversion((uint32_t)next());
  printf("bf16: conversion %s\n", failures ? "FAILED" : "ok");
}

/* ---- GEMM driver ----------------------------------------------------------------------------------------------- */

static double silu(double v) {
  return v / (1.0 + exp(-v));
}
static double rounded(float x) {
  return (double)f32_from_bf16(bf16_from_f32(x));
}

typedef struct {
  float *c;
  size_t elements;
} Output;

/* One shape and kind. The result is returned so the caller can compare two paths bit for bit. */
static Output run_shape(int m, int n, int k, GemmKind kind, int compare) {
  int lda = k + 3, ldc = n + 7;
  float *a = zeroed((size_t)m * lda, 4), *w = zeroed((size_t)n * k, 4), *w3 = zeroed((size_t)n * k, 4);
  float *base = zeroed((size_t)m * ldc, 4);
  for (size_t i = 0; i < (size_t)m * lda; i++)
    a[i] = uniform() * 2;
  for (size_t i = 0; i < (size_t)n * k; i++)
    w[i] = uniform(), w3[i] = uniform();
  for (size_t i = 0; i < (size_t)m * ldc; i++)
    base[i] = uniform() * 4;
  uint16_t *pw = zeroed(bf16_vnni_elements(n, k), 2), *pw3 = zeroed(bf16_vnni_elements(n, k), 2);
  pack_bf16_vnni(pw, w, n, k, (size_t)k);
  pack_bf16_vnni(pw3, w3, n, k, (size_t)k);
  AlignedBuffer panel;
  aligned_buffer_alloc(&panel, k_gemm_bf16_panel_bytes(m, k), MEM_ACTIVATION);
  Output out = {zeroed((size_t)m * ldc, 4), (size_t)m * ldc};
  memcpy(out.c, base, (size_t)m * ldc * 4);
  GemmBf16Segment seg = {.c = out.c, .ldc = ldc, .w = pw, .w3 = pw3, .n = n, .kind = kind};
  k_gemm_bf16(a, lda, m, k, &seg, 1, &panel);
  if (compare)
    for (int i = 0; i < m; i++) {
      for (int j = 0; j < n; j++) {
        double ref = 0, mag = 0, ref3 = 0, mag3 = 0;
        for (int l = 0; l < k; l++) {
          double x = rounded(a[(size_t)i * lda + l]), p = x * rounded(w[(size_t)j * k + l]),
                 q = x * rounded(w3[(size_t)j * k + l]);
          ref += p, mag += fabs(p), ref3 += q, mag3 += fabs(q);
        }
        double got = out.c[(size_t)i * ldc + j], old = base[(size_t)i * ldc + j], want, tol;
        if (kind == GEMM_ADD)
          want = old + ref, tol = 2e-6 * mag + 2e-7 * fabs(old) + 1e-6;
        else if (kind == GEMM_SWIGLU)
          want = silu(ref) * ref3, tol = 2e-5 * (mag * (fabs(ref3) + 1) + mag3 * (fabs(ref) + 1)) + 1e-6;
        else
          want = ref, tol = 2e-6 * mag + 1e-6;
        checks++;
        if (!(fabs(got - want) <= tol)) {
          char detail[160];
          snprintf(detail, sizeof(detail), "m=%d n=%d k=%d kind=%d (%d,%d): got %.9g expected %.9g", m, n, k, kind, i,
                   j, got, want);
          bad("k_gemm_bf16", detail);
        }
      }
      for (int j = n; j < ldc; j++) { /* the padding between rows is not ours to write */
        checks++;
        if (out.c[(size_t)i * ldc + j] != base[(size_t)i * ldc + j])
          bad("k_gemm_bf16", "wrote into the row padding");
      }
    }
  aligned_buffer_free(&panel);
  free(a), free(w), free(w3), free(base), free(pw), free(pw3);
  return out;
}

static void run_all(const char *label, Output *saved, int *saved_count, int record) {
  const int ms[] = {1, 3, 16, 17, 33, 64}, ns[] = {1, 17, 32, 33, 100}, ks[] = {1, 31, 32, 33, 100, 257};
  const GemmKind kinds[] = {GEMM_STORE, GEMM_ADD, GEMM_SWIGLU, GEMM_PARTIAL};
  int before = failures, index = 0;
  for (size_t i = 0; i < sizeof(ms) / sizeof(*ms); i++)
    for (size_t j = 0; j < sizeof(ns) / sizeof(*ns); j++)
      for (size_t l = 0; l < sizeof(ks) / sizeof(*ks); l++)
        for (int t = 0; t < 4; t++) {
          unsigned long long state = rng; /* the same operands in every run of the matrix */
          Output out = run_shape(ms[i], ns[j], ks[l], kinds[t], 1);
          rng = state;
          (void)next();
          if (record) {
            saved[index] = out;
          } else {
            checks++;
            if (memcmp(saved[index].c, out.c, out.elements * 4))
              bad("bf16 gemm", "differs from the portable reference");
            free(out.c);
          }
          index++;
        }
  *saved_count = index;
  printf("bf16: gemm %s %s\n", label, failures == before ? "ok" : "FAILED");
}

/* The operands of a case must not depend on what ran before it: reseed identically per matrix run. */
static void test_gemm(void) {
  enum { MAX_CASES = 6 * 5 * 6 * 4 };
  Output *portable = zeroed(MAX_CASES, sizeof(Output));
  int count = 0, scratch;
  const unsigned long long seed = rng;
  kernels_select(KERNELS_SCALAR);
  run_all("portable reference", portable, &count, 1);
  const int team_sizes[] = {1, 3, 7};
  for (size_t t = 0; t < 3; t++) {
    PoolConfig config = {.teams = 1, .threads = {team_sizes[t], 0}, .cpus = {NULL, NULL}};
    pool_start(&config);
    kernels_set_team(pool_team(0));
    rng = seed;
    char label[64];
    snprintf(label, sizeof(label), "portable reference teams:%d", team_sizes[t]);
    run_all(label, portable, &scratch, 0);
    kernels_set_team(NULL);
    pool_stop();
  }
  if (gemm_amx_unavailable()) {
    printf("amx unavailable: %s; the AMX BF16 path was not run\n", gemm_amx_unavailable());
  } else {
    kernels_select(KERNELS_AMX);
    for (size_t t = 0; t < 3; t++) {
      PoolConfig config = {.teams = 1, .threads = {team_sizes[t], 0}, .cpus = {NULL, NULL}};
      pool_start(&config);
      kernels_set_team(pool_team(0));
      rng = seed;
      char label[64];
      snprintf(label, sizeof(label), "%s teams:%d", gemm_amx_emulated() ? "amx (tile model)" : "amx", team_sizes[t]);
      if (gemm_amx_emulated())
        run_all(label, portable, &scratch, 0); /* bit for bit */
      else {
        Output *again = zeroed(MAX_CASES, sizeof(Output));
        run_all(label, again, &scratch, 1); /* checked against the double reference inside */
        for (int i = 0; i < scratch; i++)
          free(again[i].c);
        free(again);
      }
      kernels_set_team(NULL);
      pool_stop();
    }
    kernels_select(KERNELS_SCALAR);
    gemm_amx_thread_release();
  }
  for (int i = 0; i < count; i++)
    free(portable[i].c);
  free(portable);
}

/* pack_bf16_vnni_many on a team equals matrix-by-matrix packing. */
static void test_pack_many(void) {
  const int rows[] = {5, 16, 33, 100}, cols[] = {7, 32, 65, 130};
  PackBf16 jobs[4];
  uint16_t *want[4], *got[4];
  float *w[4];
  for (int i = 0; i < 4; i++) {
    w[i] = zeroed((size_t)rows[i] * cols[i], 4);
    for (size_t j = 0; j < (size_t)rows[i] * cols[i]; j++)
      w[i][j] = uniform() * 3;
    size_t elements = bf16_vnni_elements(rows[i], cols[i]);
    want[i] = zeroed(elements, 2), got[i] = zeroed(elements, 2);
    memset(got[i], 0xAB, elements * 2);
    pack_bf16_vnni(want[i], w[i], rows[i], cols[i], (size_t)cols[i]);
    jobs[i] = (PackBf16){got[i], w[i], rows[i], cols[i], (size_t)cols[i]};
  }
  PoolConfig config = {.teams = 1, .threads = {5, 0}, .cpus = {NULL, NULL}};
  pool_start(&config);
  kernels_set_team(pool_team(0));
  pack_bf16_vnni_many(jobs, 4);
  kernels_set_team(NULL);
  pool_stop();
  for (int i = 0; i < 4; i++) {
    checks++;
    if (memcmp(want[i], got[i], bf16_vnni_elements(rows[i], cols[i]) * 2))
      bad("pack_bf16_vnni_many", "differs from the per-matrix packing");
    free(w[i]), free(want[i]), free(got[i]);
  }
  printf("bf16: pack_bf16_vnni_many %s\n", failures ? "FAILED" : "ok");
}

int main(void) {
  test_conversion();
  test_pack_many();
  test_gemm();
  for (int i = 0; i < MEM_COUNT; i++)
    assert(memory_current[i] == 0);
  printf("bf16: %d checks, %d failures\n", checks, failures);
  return failures != 0;
}
