#include "kernels.h"
#include "prefill.h"

#include <errno.h>
#include <limits.h>
#include <stdlib.h>
#include <string.h>

int prefill_precision(int quantized) {
  const char *value = getenv("LLAMA_PRECISION");
  int bf16 = 0;
  if (value && *value) {
    if (!strcmp(value, "bf16"))
      bf16 = 1;
    else if (strcmp(value, "fp32"))
      fail("LLAMA_PRECISION must be fp32 or bf16");
  }
  const char *drop = getenv("LLAMA_BF16_DROP_FP32");
  if (bf16 && drop && *drop && strcmp(drop, "0"))
    fail("LLAMA_BF16_DROP_FP32 is reserved for a future BF16 decode GEMV and is not supported: decode reads the FP32 "
         "weights");
  if (bf16 && quantized)
    fail("LLAMA_PRECISION=bf16 applies to FP32 checkpoints (run); Q8_0 checkpoints keep their int8 semantics (runq)");
  if (!quantized && !bf16 && kernels_amx_active())
    fail("LLAMA_KERNELS=amx on an FP32 checkpoint requires LLAMA_PRECISION=bf16: FP32 execution never changes "
         "precision silently");
  return bf16;
}

void prefill_configure(Model *model) {
  const char *batch = getenv("LLAMA_PREFILL_BATCH");
  long count = 1;
  if (batch && *batch) {
    /* Require decimal digits only (strtol alone also accepts whitespace/signs). */
    if (strspn(batch, "0123456789") != strlen(batch))
      fail("LLAMA_PREFILL_BATCH must be a positive integer <= INT_MAX");
    errno = 0;
    char *end;
    count = strtol(batch, &end, 10);
    if (errno || *end || count < 1 || count > INT_MAX)
      fail("LLAMA_PREFILL_BATCH must be a positive integer <= INT_MAX");
  }
  const char *logits = getenv("LLAMA_PREFILL_LOGITS");
  if (logits && *logits && strcmp(logits, "last") && strcmp(logits, "all"))
    fail("LLAMA_PREFILL_LOGITS must be last or all");
  /* LLAMA_PREFILL_GEMM: auto = the GEMM where the AVX-512 kernels or AMX run, on = always (the scalar GEMM sums like
   * the GEMV), off = the per-row reference. The Q8 engine has a W8A8 GEMM (C6); W8A32 always runs per row. */
  const char *gemm = getenv("LLAMA_PREFILL_GEMM");
  if (gemm && *gemm && strcmp(gemm, "auto") && strcmp(gemm, "on") && strcmp(gemm, "off"))
    fail("LLAMA_PREFILL_GEMM must be auto, on or off");
  model->max_batch = count < model->config.seq_len ? (int)count : model->config.seq_len;
  model->prefill_gemm = (gemm && !strcmp(gemm, "on")    ? 1
                         : gemm && !strcmp(gemm, "off") ? 0
                                                        : kernels_avx512_active() || kernels_amx_active()) &&
                        model->max_batch > 1;
  if (model->prefill_bf16) {
    if (gemm && !strcmp(gemm, "off"))
      fail("LLAMA_PRECISION=bf16 needs the batched GEMM, but LLAMA_PREFILL_GEMM=off disables it");
    if (model->max_batch < 2)
      fail("LLAMA_PRECISION=bf16 applies to batched prefill: set LLAMA_PREFILL_BATCH to 2 or more");
    model->prefill_gemm = 1;
    if (!kernels_amx_active())
      LOG(LOG_WARN, "LLAMA_PRECISION=bf16 without LLAMA_KERNELS=amx runs the portable BF16 reference GEMM (slow)");
  }
  model->prefill_logits = logits && !strcmp(logits, "all") ? LOGITS_ALL : LOGITS_LAST;
  LOG(LOG_DEBUG, "prefill: maximum batch=%d, logits=%s, gemm=%s, precision=%s", model->max_batch,
      model->prefill_logits == LOGITS_ALL ? "all" : "last", model->prefill_gemm ? "on" : "off",
      model->prefill_bf16 ? "bf16" : "fp32");
}

int engine_max_batch(const Model *model) {
  return model->max_batch;
}

void prefill_validate(const Model *model, const int *tokens, int count, int pos0, LogitsWanted want) {
  /* Subtract after checking pos0, so even INT_MAX inputs cannot overflow. */
  if (!tokens || count < 1 || count > model->max_batch || pos0 < 0 || pos0 >= model->config.seq_len ||
      count > model->config.seq_len - pos0)
    fail("batch input out of range: count=%d position=%d maximum=%d", count, pos0, model->max_batch);
  if (want != LOGITS_NONE && want != LOGITS_LAST && want != LOGITS_ALL)
    fail("invalid batch logits request: %d", (int)want);
  /* Validate every token before writing any KV entry. */
  for (int i = 0; i < count; i++)
    if (tokens[i] < 0 || tokens[i] >= model->config.vocab_size)
      fail("forward input out of range: token=%d position=%d", tokens[i], pos0 + i);
}
