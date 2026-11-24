/* Batched Q8_0 GEMM (C6). Built twice, like test-gemm and test-amx:
 *   test-q8        the portable backend everywhere; AVX-512 and AMX-INT8 on CPUs that have them.
 *   test-q8-emul   with GEMM_EMULATE_AVX512 and GEMM_EMULATE_AMX: the VNNI tile kernel and the AMX chunk kernel run on
 *                  scalar models of their instructions, so their exact loops are checked on any host.
 * Checks: the batched quantizer is bitwise the per-token one; the portable backend equals the per-row GEMV bitwise on
 * the scalar instruction set; every backend is within the Q8 budget (2e-5 sum|w x| + 1e-6) of a double reference built
 * from the dequantized operands; the AMX int32 group dots are exactly the scalar int32 dots; gs % 64 != 0 falls back.
 */
#include "gemm.h"
#include "kernels.h"
#include "testing.h"
#include "workers.h"
#include <assert.h>
#include <math.h>
#include <stdatomic.h>
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
static double silu(double v) {
  return v / (1.0 + exp(-v));
}
static size_t up(size_t v, size_t to) {
  return (v + to - 1) / to * to;
}

typedef struct {
  int m, n, k, gs;
  float *x;   /* m x k FP32 activations */
  int8_t *xq; /* quantized, rows padded to a multiple of 32 with garbage */
  float *xs;
  int8_t *wq, *wq3; /* n x k */
  float *ws, *ws3;  /* n x (k / gs) */
  int8_t *packed, *packed3;
} Case;

static void make_case(Case *c, int m, int n, int k, int gs) {
  *c = (Case){.m = m, .n = n, .k = k, .gs = gs};
  int groups = k / gs;
  c->x = zeroed((size_t)m * k, 4);
  for (size_t i = 0; i < (size_t)m * k; i++)
    c->x[i] = uniform() * (1.0f + (float)(i / k % 3));
  size_t rows = up((size_t)m, 32);
  c->xq = zeroed(rows * k, 1);
  for (size_t i = 0; i < rows * k; i++)
    c->xq[i] = (int8_t)((int)(next() % 255) - 127); /* garbage in the padding rows */
  c->xs = zeroed(rows * groups, 4);
  k_quantize_rows(c->xq, c->xs, c->x, (size_t)k, m, k, gs);
  c->wq = zeroed((size_t)n * k, 1), c->wq3 = zeroed((size_t)n * k, 1);
  c->ws = zeroed((size_t)n * groups, 4), c->ws3 = zeroed((size_t)n * groups, 4);
  for (size_t i = 0; i < (size_t)n * k; i++)
    c->wq[i] = (int8_t)((int)(next() % 255) - 127), c->wq3[i] = (int8_t)((int)(next() % 255) - 127);
  for (size_t i = 0; i < (size_t)n * groups; i++)
    c->ws[i] = (i % 7 == 3 ? 0.0f : fabsf(uniform()) * 0.02f), c->ws3[i] = fabsf(uniform()) * 0.02f;
  c->packed = zeroed(s8_vnni_bytes(n, k), 1), c->packed3 = zeroed(s8_vnni_bytes(n, k), 1);
  pack_s8_vnni(c->packed, c->wq, n, k, (size_t)k);
  pack_s8_vnni(c->packed3, c->wq3, n, k, (size_t)k);
}

static void free_case(Case *c) {
  free(c->x), free(c->xq), free(c->xs), free(c->wq), free(c->wq3), free(c->ws), free(c->ws3), free(c->packed),
      free(c->packed3);
}

/* sum and magnitude of one output from the dequantized operands, in double. */
static void reference(const Case *c, int i, int j, int use3, double *sum, double *mag) {
  const int groups = c->k / c->gs;
  const int8_t *w = (use3 ? c->wq3 : c->wq) + (size_t)j * c->k;
  const float *ws = (use3 ? c->ws3 : c->ws) + (size_t)j * groups;
  *sum = *mag = 0;
  for (int l = 0; l < c->k; l++) {
    double p =
        ((double)c->xq[(size_t)i * c->k + l] * c->xs[(size_t)i * groups + l / c->gs]) * ((double)w[l] * ws[l / c->gs]);
    *sum += p, *mag += fabs(p);
  }
}

static const char *backend_name(Q8GemmBackend b) {
  return b == Q8_GEMM_AMX ? "amx" : b == Q8_GEMM_AVX512 ? "avx512" : "portable";
}

/* Runs one backend on all four kinds and checks against the double reference. Returns the outputs (STORE) for
 * comparisons between backends, or NULL when the backend is not compiled in. */
static float *check_backend(const Case *c, Q8GemmBackend backend, int via_pick) {
  const int m = c->m, n = c->n, ldc = n + 5;
  float *base = zeroed((size_t)m * ldc, 4), *out = zeroed((size_t)m * ldc, 4), *store = NULL;
  for (size_t i = 0; i < (size_t)m * ldc; i++)
    base[i] = uniform() * 3;
  const GemmKind kinds[] = {GEMM_STORE, GEMM_ADD, GEMM_SWIGLU, GEMM_PARTIAL};
  for (int t = 0; t < 4; t++) {
    memcpy(out, base, (size_t)m * ldc * 4);
    GemmQ8Segment seg = {.c = out, .ldc = ldc, .w = {c->wq, c->ws}, .w3 = {c->wq3, c->ws3}, .n = n, .kind = kinds[t]};
    if (backend == Q8_GEMM_AMX)
      seg.pw = c->packed, seg.pw3 = c->packed3;
    if (via_pick)
      k_gemm_q8(c->xq, c->xs, m, c->k, c->gs, &seg, 1);
    else if (!k_gemm_q8_backend_test(backend, c->xq, c->xs, m, c->k, c->gs, &seg, 1)) {
      free(base), free(out);
      return NULL;
    }
    for (int i = 0; i < m; i++) {
      for (int j = 0; j < n; j++) {
        double r, rm, g, gm;
        reference(c, i, j, 0, &r, &rm);
        if (kinds[t] == GEMM_SWIGLU)
          reference(c, i, j, 1, &g, &gm);
        double got = out[(size_t)i * ldc + j], old = base[(size_t)i * ldc + j], want, tol;
        if (kinds[t] == GEMM_ADD)
          want = old + r, tol = 2e-5 * (rm + fabs(old)) + 1e-6;
        else if (kinds[t] == GEMM_SWIGLU)
          want = silu(r) * g, tol = 2e-5 * (rm * (fabs(g) + 1) + gm * (fabs(r) + 1)) + 1e-6;
        else
          want = r, tol = 2e-5 * rm + 1e-6;
        checks++;
        if (!(fabs(got - want) <= tol)) {
          char detail[200];
          snprintf(detail, sizeof(detail), "[%s] m=%d n=%d k=%d gs=%d kind=%d (%d,%d): got %.9g expected %.9g",
                   backend_name(backend), m, n, c->k, c->gs, kinds[t], i, j, got, want);
          bad("k_gemm_q8", detail);
        }
      }
      for (int j = n; j < ldc; j++) {
        checks++;
        if (out[(size_t)i * ldc + j] != base[(size_t)i * ldc + j])
          bad("k_gemm_q8", "wrote into the row padding");
      }
    }
    if (t == 0) {
      store = zeroed((size_t)m * ldc, 4);
      memcpy(store, out, (size_t)m * ldc * 4);
    }
  }
  free(base), free(out);
  return store;
}

/* With the scalar instruction set the portable backend is the per-row GEMV, bit for bit, for every kind. */
static void check_bitwise_with_gemv(const Case *c) {
  const int m = c->m, n = c->n, groups = c->k / c->gs;
  float *base = zeroed((size_t)m * n, 4), *got = zeroed((size_t)m * n, 4), *want = zeroed((size_t)m * n, 4);
  for (size_t i = 0; i < (size_t)m * n; i++)
    base[i] = uniform() * 3;
  const GemmKind kinds[] = {GEMM_STORE, GEMM_ADD, GEMM_SWIGLU};
  for (int t = 0; t < 3; t++) {
    memcpy(got, base, (size_t)m * n * 4), memcpy(want, base, (size_t)m * n * 4);
    GemmQ8Segment seg = {.c = got, .ldc = n, .w = {c->wq, c->ws}, .w3 = {c->wq3, c->ws3}, .n = n, .kind = kinds[t]};
    k_gemm_q8(c->xq, c->xs, m, c->k, c->gs, &seg, 1);
    for (int i = 0; i < m; i++) {
      K8Input in = {c->xq + (size_t)i * c->k, c->xs + (size_t)i * groups, NULL};
      K8Weight w = {c->wq, c->ws}, w3 = {c->wq3, c->ws3};
      if (t == 0)
        k_matmul_q8(want + (size_t)i * n, &in, w, c->k, n, c->gs);
      else if (t == 1)
        k_matmul_q8_add(want + (size_t)i * n, &in, w, c->k, n, c->gs);
      else
        k_matmul_q8_swiglu(want + (size_t)i * n, &in, w, w3, c->k, n, c->gs);
    }
    checks++;
    if (memcmp(got, want, (size_t)m * n * 4))
      bad("k_gemm_q8 vs GEMV", "portable backend is not bitwise equal on the scalar instruction set");
  }
  free(base), free(got), free(want);
}

static void check_quantize(const Case *c) {
  const int groups = c->k / c->gs;
  int8_t *q = zeroed((size_t)c->m * c->k, 1);
  float *s = zeroed((size_t)c->m * groups, 4);
  for (int i = 0; i < c->m; i++)
    k_quantize(q + (size_t)i * c->k, s + (size_t)i * groups, c->x + (size_t)i * c->k, c->k, c->gs);
  checks++;
  if (memcmp(q, c->xq, (size_t)c->m * c->k) || memcmp(s, c->xs, (size_t)c->m * groups * 4))
    bad("k_quantize_rows", "not bitwise equal to the per-token quantizer");
  free(q), free(s);
}

/* The AMX int32 dots equal the scalar int32 dots exactly. */
static void check_amx_dots(const Case *c) {
  const int groups = c->k / c->gs;
  int32_t *dots = zeroed((size_t)groups * c->m * c->n, 4);
  if (!k_gemm_q8_amx_dots_test(dots, c->xq, c->m, c->k, c->gs, c->packed, c->n)) {
    free(dots);
    return;
  }
  int wrong = 0;
  for (int g = 0; g < groups; g++)
    for (int i = 0; i < c->m; i++)
      for (int j = 0; j < c->n; j++) {
        int32_t ref = 0;
        for (int l = g * c->gs; l < (g + 1) * c->gs; l++)
          ref += (int32_t)c->xq[(size_t)i * c->k + l] * (int32_t)c->wq[(size_t)j * c->k + l];
        wrong += dots[((size_t)g * c->m + i) * c->n + j] != ref;
      }
  checks++;
  if (wrong)
    bad("amx group dots", "int32 group dots are not exact");
  free(dots);
}

static int amx_usable(void) {
  return gemm_amx_unavailable() == NULL;
}

/* GemmQ8Hooks (the two-socket interleaved reduction): every block is stored exactly once, handed to turn only after it
 * was stored, each exactly once, all by the end of the product; the product stays bitwise k_gemm_q8(_tiled)'s. */
typedef struct {
  int blocks;
  _Atomic int *stored, *used;
  _Atomic int bad;
  GemmQ8Hooks hooks;
} HookLog;

static void hook_stored(void *ctx, int t0, int t1) {
  HookLog *log = ctx;
  for (int t = t0; t < t1; t++)
    if (t < 0 || t >= log->blocks || atomic_fetch_add(&log->stored[t], 1))
      log->bad = 1;
}

static int hook_turn(void *ctx, int first, int next, int done, int final) {
  HookLog *log = ctx;
  (void) final;
  if (next < first || done < next)
    log->bad = 1;
  for (int t = next; t < done; t++)
    if (t < 0 || t >= log->blocks || !atomic_load(&log->stored[t]) || atomic_fetch_add(&log->used[t], 1))
      log->bad = 1;
  return done;
}

static void check_interleaved(const Case *c, const int8_t *xq, const float *xs, const float *st, int tiled) {
  const int m = c->m, n = c->n, ldc = n + 3;
  float *want = zeroed((size_t)m * ldc, 4), *got = zeroed((size_t)m * ldc, 4);
  GemmQ8Segment seg = {
      .c = want, .ldc = ldc, .w = {c->wq, c->ws}, .pw = c->packed, .pws = st, .n = n, .kind = GEMM_PARTIAL};
  if (tiled)
    k_gemm_q8_tiled(xq, xs, m, c->k, c->gs, &seg, 1);
  else
    k_gemm_q8(xq, xs, m, c->k, c->gs, &seg, 1);
  int blocks = (n + 15) / 16 * ((m + 31) / 32); /* an upper bound for either chunk width */
  HookLog log = {.blocks = blocks,
                 .stored = zeroed((size_t)blocks, sizeof(_Atomic int)),
                 .used = zeroed((size_t)blocks, sizeof(_Atomic int))};
  log.hooks = (GemmQ8Hooks){.stored = hook_stored, .turn = hook_turn, .ctx = &log};
  seg.c = got;
  k_gemm_q8_interleaved(xq, xs, m, c->k, c->gs, &seg, tiled, &log.hooks);
  int total = (n + log.hooks.chunk_n - 1) / log.hooks.chunk_n * log.hooks.mblocks;
  for (int t = 0; t < blocks; t++)
    if (atomic_load(&log.stored[t]) != (t < total) || atomic_load(&log.used[t]) != (t < total))
      log.bad = 1;
  checks += 2;
  char detail[160];
  snprintf(detail, sizeof(detail), "m=%d n=%d k=%d gs=%d tiled=%d", m, n, c->k, c->gs, tiled);
  if (log.hooks.mblocks != (m + 31) / 32 || log.bad)
    bad("k_gemm_q8_interleaved: blocks not stored and handed exactly once", detail);
  if (memcmp(want, got, (size_t)m * ldc * 4))
    bad("k_gemm_q8_interleaved: product differs", detail);
  free(want), free(got), free((void *)log.stored), free((void *)log.used);
}

/* The tiled kernels: the tile-order quantizer writes the same int8 values and scales as k_quantize_rows, the
 * transposed scales are the scales, and both tiled kernels (LLAMA_Q8_GEMM=skewed and =pipelined) in both K-block
 * orders (LLAMA_Q8_KB_ORDER=rows and =chunks) equal the AMX chunk kernel bit for bit, for every kind. */
static void check_tiled(const Case *c) {
  const int m = c->m, n = c->n, k = c->k, gs = c->gs, groups = k / gs, ldc = n + 3;
  const size_t rows = up((size_t)m, 32);
  int8_t *tq = zeroed(rows * k, 1);
  float *ts = zeroed(rows * groups, 4);
  for (size_t i = 0; i < rows * k; i++)
    tq[i] = (int8_t)((int)(next() % 255) - 127); /* garbage in the padding rows */
  k_quantize_rows_tiled(tq, ts, c->x, (size_t)k, m, k, gs);
  for (int r = 0; r < m; r++) {
    checks++;
    if (memcmp(ts + (size_t)r * groups, c->xs + (size_t)r * groups, (size_t)groups * 4))
      bad("k_quantize_rows_tiled", "scales differ from k_quantize_rows");
    for (int t = 0; t < k / 64; t++) {
      size_t at = ((size_t)r / 32 * 2 + (size_t)r % 32 / 16) * (size_t)(k / 64) * 1024 + (size_t)t * 1024 +
                  (size_t)(r % 16) * 64;
      checks++;
      if (memcmp(tq + at, c->xq + (size_t)r * k + (size_t)t * 64, 64))
        bad("k_quantize_rows_tiled", "int8 values are not k_quantize_rows' values in tile order");
    }
  }
  /* The fused RMSNorm + tile-order quantizer: xb equals k_rmsnorm_rows and q/s equal the tiled quantizer of xb. */
  float *norm = zeroed((size_t)k, 4), *xb = zeroed((size_t)m * k, 4), *xb_want = zeroed((size_t)m * k, 4);
  int8_t *fq = zeroed(rows * k, 1), *fq_want = zeroed(rows * k, 1);
  float *fs = zeroed(rows * groups, 4), *fs_want = zeroed(rows * groups, 4);
  for (int i = 0; i < k; i++)
    norm[i] = 0.5f + uniform();
  k_rmsnorm_rows(xb_want, c->x, norm, m, k, 1e-5f);
  k_quantize_rows_tiled(fq_want, fs_want, xb_want, (size_t)k, m, k, gs);
  k_rmsnorm_quantize_rows_tiled(fq, fs, xb, c->x, norm, m, k, 1e-5f, gs);
  checks += 3;
  if (memcmp(xb, xb_want, (size_t)m * k * 4))
    bad("k_rmsnorm_quantize_rows_tiled", "normalised rows differ from k_rmsnorm_rows");
  if (memcmp(fq, fq_want, rows * k) || memcmp(fs, fs_want, (size_t)m * groups * 4))
    bad("k_rmsnorm_quantize_rows_tiled", "int8 values or scales differ from k_quantize_rows_tiled of the norm");
  free(norm), free(xb), free(xb_want), free(fq), free(fq_want), free(fs), free(fs_want);
  float *st = zeroed(s8_scales_t_bytes(n, k, gs), 1), *st3 = zeroed(s8_scales_t_bytes(n, k, gs), 1);
  for (size_t i = 0; i < s8_scales_t_bytes(n, k, gs) / 4; i++)
    st[i] = st3[i] = NAN; /* every entry must be written, padding included */
  int tiles = (n + 15) / 16;
  pack_s8_scales_t(st, c->ws, n, groups, 0, tiles / 2); /* in two ranges, as a team splits it */
  pack_s8_scales_t(st, c->ws, n, groups, tiles / 2, tiles);
  pack_s8_scales_t(st3, c->ws3, n, groups, 0, tiles);
  for (int j = 0; j < (int)up((size_t)n, 32); j++)
    for (int g = 0; g < groups; g++) {
      float want = j < n ? c->ws[(size_t)j * groups + g] : 0.0f,
            got = st[((size_t)(j / 32) * groups + g) * 32 + j % 32];
      checks++;
      if (memcmp(&got, &want, 4))
        bad("pack_s8_scales_t", "transposed scale differs");
    }
  float *base = zeroed((size_t)m * ldc, 4), *want = zeroed((size_t)m * ldc, 4), *got = zeroed((size_t)m * ldc, 4);
  for (size_t i = 0; i < (size_t)m * ldc; i++)
    base[i] = uniform() * 3;
  const GemmKind kinds[] = {GEMM_STORE, GEMM_ADD, GEMM_SWIGLU, GEMM_PARTIAL};
  gemm_amx_enable(1);
  static const char *const modes[][2] = {{"skewed", "rows"}, {"pipelined", "rows"}, {"skewed", "chunks"}};
  for (int mode = 0; mode < 3; mode++)
    for (int t = 0; t < 4; t++) {
      setenv("LLAMA_Q8_GEMM", modes[mode][0], 1);
      setenv("LLAMA_Q8_KB_ORDER", modes[mode][1], 1);
      gemm_q8_init_env();
      GemmQ8Segment seg = {.ldc = ldc,
                           .w = {c->wq, c->ws},
                           .w3 = {c->wq3, c->ws3},
                           .pw = c->packed,
                           .pw3 = c->packed3,
                           .pws = st,
                           .pws3 = st3,
                           .n = n,
                           .kind = kinds[t]};
      memcpy(want, base, (size_t)m * ldc * 4);
      memcpy(got, base, (size_t)m * ldc * 4);
      seg.c = want;
      k_gemm_q8_backend_test(Q8_GEMM_AMX, c->xq, c->xs, m, k, gs, &seg, 1);
      seg.c = got;
      k_gemm_q8_tiled(tq, ts, m, k, gs, &seg, 1);
      checks++;
      if (memcmp(want, got, (size_t)m * ldc * 4)) {
        char detail[160];
        snprintf(detail, sizeof(detail), "m=%d n=%d k=%d gs=%d kind=%d: %s/%s differs from the chunk kernel", m, n, k,
                 gs, kinds[t], modes[mode][0], modes[mode][1]);
        bad("k_gemm_q8_tiled", detail);
      }
    }
  unsetenv("LLAMA_Q8_GEMM");
  unsetenv("LLAMA_Q8_KB_ORDER");
  gemm_q8_init_env();
  check_interleaved(c, tq, ts, st, 1);
  check_interleaved(c, c->xq, c->xs, NULL, 0);
  gemm_amx_enable(0);
  free(tq), free(ts), free(st), free(st3), free(base), free(want), free(got);
}

static void run_matrix(const char *label) {
  const int ms[] = {1, 2, 3, 17, 33, 40}, ns[] = {1, 5, 16, 17, 33};
  const int gss[] = {2, 4, 32, 64, 128};
  int before = failures;
  int avx_ran = 0, amx_ran = 0;
  for (size_t gi = 0; gi < sizeof(gss) / sizeof(*gss); gi++)
    for (int mult = 0; mult < 2; mult++) {
      int gs = gss[gi], k = gs * (mult ? 5 : 2);
      for (size_t i = 0; i < sizeof(ms) / sizeof(*ms); i++)
        for (size_t j = 0; j < sizeof(ns) / sizeof(*ns); j++) {
          Case c;
          make_case(&c, ms[i], ns[j], k, gs);
          check_quantize(&c);
          check_interleaved(&c, c.xq, c.xs, NULL, 0);
          float *portable = check_backend(&c, Q8_GEMM_PORTABLE, 0);
          float *avx = check_backend(&c, Q8_GEMM_AVX512, 0);
          avx_ran += avx != NULL;
          free(portable), free(avx);
          if (amx_usable()) {
            float *amx = check_backend(&c, Q8_GEMM_AMX, 0);
            amx_ran += amx != NULL;
            free(amx);
            if (gs % 64 == 0) {
              check_amx_dots(&c);
              check_tiled(&c);
            }
            /* The pick: AMX when active and gs % 64 == 0, the fallback otherwise, the result within budget either way.
             */
            gemm_amx_enable(1);
            GemmQ8Segment probe = {.n = ns[j], .pw = c.packed, .kind = GEMM_STORE};
            Q8GemmBackend picked = k_gemm_q8_pick(gs, &probe, 1);
            checks++;
            if ((gs % 64 == 0) != (picked == Q8_GEMM_AMX))
              bad("k_gemm_q8_pick", "amx chosen for an unsupported group size or not chosen for a supported one");
            free(check_backend(&c, Q8_GEMM_PORTABLE, 1));
            gemm_amx_enable(0);
          }
          free_case(&c);
        }
    }
  printf("q8: %s: portable%s%s ran, %s\n", label, avx_ran ? ", avx512 tile kernel" : "", amx_ran ? ", amx-int8" : "",
         failures == before ? "all within budget" : "FAILED");
}

/* Real layers are far larger than the matrix above (K up to 14336, N up to 14336, M = 64 microbatches): sizes that cross
 * several M blocks, N chunks and many K groups, with edges that are not multiples of the tile or chunk sizes. */
static void run_large(const char *label) {
  const int shapes[][4] = {{64, 130, 2048, 64},
                           {64, 33, 8192, 64},
                           {37, 520, 1024, 64},
                           {128, 70, 2048, 128},
                           {64, 600, 4096, 64},
                           {96, 257, 3072, 64},
                           /* K sweep up to the largest layer input of the 8B model, then M and N edges */
                           {16, 48, 128, 64},
                           {16, 48, 512, 64},
                           {16, 48, 1024, 64},
                           {16, 48, 2048, 64},
                           {16, 48, 4096, 64},
                           {16, 48, 8192, 64},
                           {16, 48, 14336, 64},
                           {64, 4096, 2048, 64},
                           {64, 14336, 4096, 64},
                           {53, 1024, 2048, 64},
                           /* per-channel INT8: one group over all of K (gs = K, wider than the quantizer's piece) */
                           {64, 130, 4096, 4096},
                           {40, 48, 14336, 14336},
                           {33, 70, 2048, 2048},
                           {17, 33, 128, 128},
                           {16, 48, 4096, 4096},
                           {40, 17, 3072, 3072}};
  int before = failures;
  for (size_t i = 0; i < sizeof(shapes) / sizeof(*shapes); i++) {
    Case c;
    make_case(&c, shapes[i][0], shapes[i][1], shapes[i][2], shapes[i][3]);
    check_quantize(&c);
    check_interleaved(&c, c.xq, c.xs, NULL, 0);
    free(check_backend(&c, Q8_GEMM_PORTABLE, 0));
    free(check_backend(&c, Q8_GEMM_AVX512, 0));
    if (amx_usable()) {
      free(check_backend(&c, Q8_GEMM_AMX, 0));
      check_amx_dots(&c);
      check_tiled(&c);
    }
    free_case(&c);
  }
  printf("q8: %s: real-size shapes: %s\n", label, failures == before ? "all within budget" : "FAILED");
}

int main(void) {
  /* LLAMA_SWIGLU_EXP=libm (the way back from the vector default) keeps the epilogue bitwise the GEMV's silu_gate. */
  setenv("LLAMA_SWIGLU_EXP", "libm", 1);
  gemm_swiglu_init_env();
  /* Scalar instruction set: portable batched Q8 == per-row GEMV, bit for bit. */
  kernels_select(KERNELS_SCALAR);
  {
    const int shapes[][4] = {{5, 7, 16, 4}, {33, 20, 128, 64}, {1, 9, 32, 2}, {40, 33, 192, 64}};
    for (size_t i = 0; i < sizeof(shapes) / sizeof(*shapes); i++) {
      Case c;
      make_case(&c, shapes[i][0], shapes[i][1], shapes[i][2], shapes[i][3]);
      check_bitwise_with_gemv(&c);
      free_case(&c);
    }
    printf("q8: portable batched GEMM equals the per-row GEMV bitwise on the scalar instruction set: %s\n",
           failures ? "FAILED" : "ok");
  }
  run_matrix("single thread");
  run_large("single thread");
  /* The default LLAMA_SWIGLU_EXP=vector: the AVX-512 exponential in the SwiGLU epilogue, within the same budget. */
  unsetenv("LLAMA_SWIGLU_EXP");
  gemm_swiglu_init_env();
  run_matrix("vector exp");
  int team_sizes[8] = {3, 7};
  size_t team_count = 2;
  const char *override = getenv("LLAMA_TEST_TEAMS"); /* e.g. "28,56" for full-socket teams on the cluster */
  if (override && *override) {
    team_count = 0;
    for (const char *q = override; *q && team_count < 8;) {
      team_sizes[team_count++] = atoi(q);
      while (*q && *q != ',')
        q++;
      q += *q == ',';
    }
  }
  for (size_t t = 0; t < team_count; t++) {
    PoolConfig config = {.teams = 1, .threads = {team_sizes[t], 0}, .cpus = {NULL, NULL}};
    pool_start(&config);
    kernels_set_team(pool_team(0));
    char label[32];
    snprintf(label, sizeof(label), "teams:%d", team_sizes[t]);
    run_matrix(label);
    run_large(label);
    kernels_set_team(NULL);
    pool_stop();
  }
  if (!amx_usable())
    printf("amx unavailable: %s; the AMX-INT8 kernel was not run\n", gemm_amx_unavailable());
  for (int i = 0; i < MEM_COUNT; i++)
    assert(memory_current[i] == 0);
  gemm_amx_thread_release();
  printf("q8: %d checks, %d failures\n", checks, failures);
  return failures != 0;
}
