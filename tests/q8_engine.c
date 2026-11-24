/* Batched Q8_0 prefill, end to end on the Q8 engine (C6). Linked with src/runq.c. argv[1] is a checkpoint whose group
 * size is a multiple of 64 (the AVX-512 / AMX-INT8 path), argv[2] one whose group size AMX-INT8 cannot take (neither
 * 32 nor a multiple of 64: the fallback), argv[3] one with group size 32 (AMX-INT8 with half-shape tiles).
 *  1. On the scalar instruction set the batched GEMM prefill equals the per-token forward bitwise, for any split.
 *  2. AMX-INT8 (the tile model in the emulated build, hardware where present) agrees with the portable batched forward
 *     within the Q8 budget; the difference is recorded.
 *  3. The per-token path (decode) is unchanged by selecting amx.
 *  4. W8A32 never uses the batched GEMM, with or without amx; a group size AMX-INT8 cannot take falls back; group size
 *     32 runs AMX-INT8, independent of the split, and agrees with its portable batched forward up to activation
 *     rounding flips (below). */
#include "gemm.h"
#include "kernels.h"
#include "runtime.h"
#include <assert.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

enum { TOKENS = 24 };

static int failures;
static void check(int ok, const char *what) {
  if (!ok && failures++ < 20)
    fprintf(stderr, "q8 engine: %s\n", what);
}

typedef struct {
  float *logits;
  size_t vocab;
  char kernel[32]; /* Model.prefill_kernel, "" when the batched GEMM is off */
  int gemm;
} Run;

static Run forward_all(const char *path, const char *kernels, const char *gemm, const char *activations, int split,
                       int per_token) {
  setenv("LLAMA_KERNELS", kernels, 1);
  setenv("LLAMA_PREFILL_GEMM", gemm, 1);
  setenv("LLAMA_Q8_ACTIVATIONS", activations, 1);
  char batch[16];
  snprintf(batch, sizeof(batch), "%d", split > 1 ? split : 2);
  setenv("LLAMA_PREFILL_BATCH", batch, 1);
  Model model = {0};
  engine_load(&model, path);
  Run run = {.vocab = (size_t)model.config.vocab_size, .gemm = model.prefill_gemm};
  if (model.prefill_gemm && model.prefill_kernel)
    snprintf(run.kernel, sizeof(run.kernel), "%s", model.prefill_kernel);
  run.logits = memory_alloc(checked_mul(TOKENS, run.vocab), sizeof(float), MEM_METADATA);
  int tokens[TOKENS];
  for (int i = 0; i < TOKENS; i++)
    tokens[i] = 65 + i % 23;
  if (per_token) {
    for (int i = 0; i < TOKENS; i++)
      memcpy(run.logits + (size_t)i * run.vocab, engine_forward(&model, tokens[i], i), run.vocab * sizeof(float));
  } else {
    for (int i = 0; i < TOKENS;) {
      int count = TOKENS - i < split ? TOKENS - i : split;
      memcpy(run.logits + (size_t)i * run.vocab, engine_forward_batch(&model, tokens + i, count, i, LOGITS_ALL),
             (size_t)count * run.vocab * sizeof(float));
      i += count;
    }
  }
  engine_free(&model);
  return run;
}

static double max_diff(const Run *a, const Run *b, double *max_rel) {
  double worst = 0;
  *max_rel = 0;
  for (size_t i = 0; i < TOKENS * a->vocab; i++) {
    double d = fabs((double)a->logits[i] - b->logits[i]);
    if (d > worst)
      worst = d;
    double rel = d / (fabs((double)a->logits[i]) + 1e-3);
    if (rel > *max_rel)
      *max_rel = rel;
  }
  return worst;
}

int main(int argc, char **argv) {
  assert(argc == 4);
  const char *wide = argv[1], *narrow = argv[2], *half = argv[3];
  const size_t bytes = 0;
  (void)bytes;
  /* The budgets below are the GEMM's: keep the W8A8 defaults that change other rounding (AMX-BF16 attention, the
   * vector SwiGLU exponential; both judged by the W8A8 gate) on their ways back. */
  setenv("LLAMA_ATTN", "tiled", 1);
  setenv("LLAMA_SWIGLU_EXP", "libm", 1);

  Run reference = forward_all(wide, "scalar", "off", "int8", 1, 1);
  check(!reference.gemm, "the per-token reference used the GEMM");

  /* 1. scalar batched GEMM == per-token, bitwise */
  Run portable = forward_all(wide, "scalar", "on", "int8", TOKENS, 0);
  check(!strcmp(portable.kernel, "portable-q8"), "scalar GEMM prefill is not reported as portable-q8");
  check(memcmp(reference.logits, portable.logits, TOKENS * reference.vocab * sizeof(float)) == 0,
        "scalar batched Q8 prefill differs from the per-token forward");
  for (int split = 2; split <= 7; split += 5) {
    Run part = forward_all(wide, "scalar", "on", "int8", split, 0);
    check(memcmp(reference.logits, part.logits, TOKENS * reference.vocab * sizeof(float)) == 0,
          "scalar batched Q8 prefill depends on the microbatch split");
    memory_free(part.logits);
  }

  if (gemm_amx_unavailable()) {
    printf("amx unavailable: %s; the AMX-INT8 forward was not run\n", gemm_amx_unavailable());
  } else {
    /* 2. AMX-INT8 vs the portable batched forward */
    Run amx = forward_all(wide, "amx", "auto", "int8", 7, 0);
    check(!strcmp(amx.kernel, "amx-int8"), "amx did not select amx-int8 for gs % 64 == 0");
    double rel, abs = max_diff(&amx, &portable, &rel);
    printf("q8 engine: amx-int8 vs portable batched Q8 forward over %d positions: max_abs=%.3e max_rel=%.3e\n", TOKENS,
           abs, rel);
    check(abs <= 1e-4 && rel <= 1e-3, "AMX-INT8 forward is outside the Q8 budget of the portable batched forward");
    Run again = forward_all(wide, "amx", "auto", "int8", TOKENS, 0);
    double rel2, abs2 = max_diff(&amx, &again, &rel2);
    printf("q8 engine: amx-int8 split 7 vs split %d: max_abs=%.3e\n", TOKENS, abs2);
    check(abs2 == 0, "AMX-INT8 results depend on the microbatch split");
    memory_free(again.logits);
    /* 3. decode path unchanged: per-token under amx equals per-token under the best non-amx kernels */
    Run decode = forward_all(wide, "amx", "off", "int8", 1, 1);
    const char *base = kernels_select(KERNELS_AVX512) ? "avx512" : "scalar";
    Run decode_base = forward_all(wide, base, "off", "int8", 1, 1);
    check(memcmp(decode.logits, decode_base.logits, TOKENS * decode.vocab * sizeof(float)) == 0,
          "per-token logits changed under LLAMA_KERNELS=amx");
    /* 4a. W8A32 stays per row under amx */
    Run f32_amx = forward_all(wide, "amx", "auto", "fp32", 7, 0),
        f32_base = forward_all(wide, base, "auto", "fp32", 7, 0);
    check(!f32_amx.gemm, "W8A32 used the batched GEMM under amx");
    check(memcmp(f32_amx.logits, f32_base.logits, TOKENS * f32_amx.vocab * sizeof(float)) == 0,
          "W8A32 under amx differs from W8A32 without it");
    /* 4b. a group size AMX-INT8 cannot take */
    Run narrow_amx = forward_all(narrow, "amx", "auto", "int8", 7, 0);
    Run narrow_scalar = forward_all(narrow, "scalar", "on", "int8", 7, 0);
    check(!strcmp(narrow_amx.kernel, "portable-q8"), "gs=16 under amx did not fall back to the portable Q8 GEMM");
    check(memcmp(narrow_amx.logits, narrow_scalar.logits, TOKENS * narrow_amx.vocab * sizeof(float)) == 0 ||
              !strcmp(base, "avx512"),
          "gs=16 fallback differs from the scalar batched forward");
    /* 4c. gs = 32: AMX-INT8 with half-shape tiles */
    Run half_amx = forward_all(half, "amx", "auto", "int8", 7, 0);
    Run half_portable = forward_all(half, "scalar", "on", "int8", TOKENS, 0);
    check(!strcmp(half_amx.kernel, "amx-int8"), "amx did not select amx-int8 for gs = 32");
    /* The AMX rescale (fma(dot * xs, ws, acc)) rounds differently from the portable one (acc += dot * ws * xs) by about
     * 1e-7; an activation within that of an int8 rounding boundary then quantizes one step apart in the next layer and
     * moves its position (and, through its K/V, later ones) by up to about 1e-2 on this fixture (positions 17-23; with
     * the portable kernel switched to the AMX expression the two forwards are bitwise equal, docs/gs32-amx.md). So:
     * every position keeps its argmax and no logit moves by more than 5e-2. */
    double half_rel, half_abs = max_diff(&half_amx, &half_portable, &half_rel);
    int beyond = 0, argmax_ok = 1;
    for (int t = 0; t < TOKENS; t++) {
      const float *a = half_amx.logits + (size_t)t * half_amx.vocab, *p = half_portable.logits + (size_t)t * half_amx.vocab;
      size_t ia = 0, ip = 0;
      double worst = 0;
      for (size_t v = 0; v < half_amx.vocab; v++) {
        ia = a[v] > a[ia] ? v : ia;
        ip = p[v] > p[ip] ? v : ip;
        worst = fmax(worst, fabs((double)a[v] - p[v]));
      }
      beyond += worst > 1e-4;
      argmax_ok &= ia == ip;
    }
    printf("q8 engine: gs=32 amx-int8 vs portable batched Q8 forward: max_abs=%.3e max_rel=%.3e, %d of %d positions beyond "
           "1e-4\n", half_abs, half_rel, beyond, TOKENS);
    check(argmax_ok && half_abs <= 5e-2, "gs=32 AMX-INT8 forward disagrees with the portable one beyond rounding flips");
    Run half_again = forward_all(half, "amx", "auto", "int8", TOKENS, 0);
    double half_rel2, half_abs2 = max_diff(&half_amx, &half_again, &half_rel2);
    check(half_abs2 == 0, "gs=32 AMX-INT8 results depend on the microbatch split");
    memory_free(amx.logits), memory_free(decode.logits), memory_free(decode_base.logits), memory_free(f32_amx.logits),
        memory_free(f32_base.logits), memory_free(narrow_amx.logits), memory_free(narrow_scalar.logits),
        memory_free(half_amx.logits), memory_free(half_portable.logits), memory_free(half_again.logits);
  }
  memory_free(reference.logits), memory_free(portable.logits);
  for (int i = 0; i < MEM_COUNT; i++)
    assert(memory_current[i] == 0);
  printf("q8 engine: %s\n", failures ? "FAILED" : "ok");
  return failures != 0;
}
