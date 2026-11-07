/* BF16 precision mode, end to end on the FP32 engine (C5). Linked with src/run.c; argv[1] is an FP32 checkpoint.
 *  1. Decode is unchanged: in BF16 mode the per-token path (engine_forward, the decode path) equals the FP32 model
 *     bitwise, because it never touches the packed weights.
 *  2. BF16 batched logits (all positions) are independent of how the prompt is split into microbatches, bit for bit.
 *  3. The AMX path (tile model in the emulated build, hardware where present) agrees with the portable scalar BF16
 *     reference forward within the kernel budget; the error against FP32 is recorded, not gated.
 * The environment selects the batch size (LLAMA_PREFILL_BATCH) as in the other prefill tests. */
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
    fprintf(stderr, "bf16 engine: %s\n", what);
}

/* Loads a model under the given environment, runs `body`, frees it. Returns logits for all TOKENS positions. */
static float *forward_all(const char *path, const char *kernels, const char *precision, int split, int per_token) {
  setenv("LLAMA_KERNELS", kernels, 1);
  if (precision)
    setenv("LLAMA_PRECISION", precision, 1);
  else
    unsetenv("LLAMA_PRECISION");
  char batch[16];
  snprintf(batch, sizeof(batch), "%d", split > 1 ? split : (precision ? 2 : 1));
  setenv("LLAMA_PREFILL_BATCH", batch, 1);
  Model model = {0};
  engine_load(&model, path);
  size_t vocab = (size_t)model.config.vocab_size;
  float *logits = memory_alloc(checked_mul(TOKENS, vocab), sizeof(float), MEM_METADATA);
  int tokens[TOKENS];
  for (int i = 0; i < TOKENS; i++)
    tokens[i] = 65 + i % 23;
  if (per_token) {
    for (int i = 0; i < TOKENS; i++)
      memcpy(logits + (size_t)i * vocab, engine_forward(&model, tokens[i], i), vocab * sizeof(float));
  } else {
    for (int i = 0; i < TOKENS;) {
      int count = TOKENS - i < split ? TOKENS - i : split;
      memcpy(logits + (size_t)i * vocab, engine_forward_batch(&model, tokens + i, count, i, LOGITS_ALL),
             (size_t)count * vocab * sizeof(float));
      i += count;
    }
  }
  engine_free(&model);
  return logits;
}

int main(int argc, char **argv) {
  assert(argc == 2);
  const size_t vocab = 128010; /* the synthetic fixtures' vocabulary */
  float *fp32 = forward_all(argv[1], "scalar", NULL, 1, 1);

  /* 1. decode path unchanged */
  float *decode = forward_all(argv[1], "scalar", "bf16", 2, 1);
  check(memcmp(fp32, decode, TOKENS * vocab * sizeof(float)) == 0, "per-token logits differ from FP32 in BF16 mode");

  /* 2. microbatch splits */
  float *whole = forward_all(argv[1], "scalar", "bf16", TOKENS, 0);
  for (int split = 2; split <= 7; split += 5) {
    float *part = forward_all(argv[1], "scalar", "bf16", split, 0);
    check(memcmp(whole, part, TOKENS * vocab * sizeof(float)) == 0, "BF16 logits depend on the microbatch split");
    memory_free(part);
  }

  /* BF16 against FP32: recorded. */
  double max_abs = 0, norm = 0;
  int agree = 0;
  for (int i = 0; i < TOKENS; i++) {
    const float *a = fp32 + (size_t)i * vocab, *b = whole + (size_t)i * vocab;
    int arg_a = 0, arg_b = 0;
    for (size_t v = 0; v < vocab; v++) {
      check(isfinite(b[v]), "non-finite BF16 logit");
      double d = fabs((double)a[v] - b[v]);
      if (d > max_abs)
        max_abs = d;
      norm += d * d;
      if (a[v] > a[arg_a])
        arg_a = (int)v;
      if (b[v] > b[arg_b])
        arg_b = (int)v;
    }
    agree += arg_a == arg_b;
  }
  printf("bf16 engine: vs fp32 over %d positions: max_abs=%.3e rms=%.3e argmax agreement %d/%d (recorded, not gated)\n",
         TOKENS, max_abs, sqrt(norm / ((double)TOKENS * vocab)), agree, TOKENS);
  check(max_abs < 0.5, "BF16 logits are implausibly far from FP32"); /* a sanity bound for the tiny fixture only */

  /* 3. AMX against the portable scalar BF16 reference */
  if (gemm_amx_unavailable()) {
    printf("amx unavailable: %s; the AMX forward was not run\n", gemm_amx_unavailable());
  } else {
    float *amx_logits = forward_all(argv[1], "amx", "bf16", 7, 0);
    double worst = 0;
    for (size_t i = 0; i < TOKENS * vocab; i++) {
      double d = fabs((double)amx_logits[i] - whole[i]);
      if (d > worst)
        worst = d;
      check(d <= 1e-5 + 1e-5 * fabs(whole[i]), "AMX forward differs from the scalar BF16 reference beyond the budget");
    }
    printf("bf16 engine: amx vs scalar BF16 reference forward: max_abs=%.3e\n", worst);
    memory_free(amx_logits);
  }
  memory_free(fp32), memory_free(decode), memory_free(whole);
  for (int i = 0; i < MEM_COUNT; i++)
    assert(memory_current[i] == 0);
  printf("bf16 engine: %s\n", failures ? "FAILED" : "ok");
  return failures != 0;
}
