#include "prefill.h"

#include <errno.h>
#include <limits.h>
#include <stdlib.h>
#include <string.h>

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
  model->max_batch = count < model->config.seq_len ? (int)count : model->config.seq_len;
  model->prefill_logits = logits && !strcmp(logits, "all") ? LOGITS_ALL : LOGITS_LAST;
  LOG(LOG_DEBUG, "prefill: maximum batch=%d, logits=%s", model->max_batch,
      model->prefill_logits == LOGITS_ALL ? "all" : "last");
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
