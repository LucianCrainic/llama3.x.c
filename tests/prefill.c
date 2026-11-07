/* Bitwise oracle: token-major forward versus layer-major prefill, including
 * continuation/decode and classifier-call counts. No model-library dependency. */
#include "profile.h"
#include "runtime.h"
#include <assert.h>
#include <limits.h>
#include <stdlib.h>
#include <string.h>

static void invalid_call(Model *model, const char *name) {
  int tokens[] = {65, 66};
  int count = 1, pos = 0;
  LogitsWanted want = LOGITS_LAST;
  if (!strcmp(name, "null"))
    engine_forward_batch(model, NULL, count, pos, want);
  else {
    if (!strcmp(name, "zero"))
      count = 0;
    else if (!strcmp(name, "negative-count"))
      count = -1;
    else if (!strcmp(name, "large-count"))
      count = INT_MAX;
    else if (!strcmp(name, "negative-pos"))
      pos = -1;
    else if (!strcmp(name, "large-pos"))
      pos = INT_MAX;
    else if (!strcmp(name, "context")) {
      count = 2;
      pos = model->config.seq_len - 1;
    } else if (!strcmp(name, "token")) {
      count = 2;
      tokens[1] = model->config.vocab_size;
    } else if (!strcmp(name, "negative-token"))
      tokens[0] = -1;
    else if (!strcmp(name, "policy"))
      want = (LogitsWanted)99;
    else
      abort();
    engine_forward_batch(model, tokens, count, pos, want);
  }
  abort(); /* Invalid inputs must exit through fail(), never reach here. */
}

int main(int argc, char **argv) {
  assert(argc == 2 || argc == 3);
  Model model = {0};
  engine_load(&model, argv[1]);
  if (argc == 3)
    invalid_call(&model, argv[2]);
  assert(engine_max_batch(&model) > 0 && engine_max_batch(&model) <= model.config.seq_len);
  const int context = model.config.seq_len, vocab = model.config.vocab_size;
  int *tokens = memory_alloc(context, sizeof(int), MEM_PROMPT);
  float *reference = memory_alloc(checked_mul(context, vocab), sizeof(float), MEM_METADATA);
  for (int i = 0; i < context; i++) {
    tokens[i] = 65 + i % 23;
    memcpy(reference + (size_t)i * vocab, engine_forward(&model, tokens[i], i), vocab * sizeof(float));
  }
  assert(setenv("LLAMA_PROFILE", "1", 1) == 0);
  const int lengths[] = {1, 5, 31, 32, 33, context};
  for (int prefix = 0; prefix <= 7; prefix += 7)
    for (size_t c = 0; c < sizeof(lengths) / sizeof(lengths[0]); c++) {
      int length = lengths[c];
      if (length > context - prefix)
        length = context - prefix;
      for (LogitsWanted want = LOGITS_NONE; want <= LOGITS_ALL; want++) {
        for (int i = 0; i < prefix; i++)
          engine_forward(&model, tokens[i], i);
        profile_init();
        profile_stage = STAGE_PREFILL;
        float *out = NULL;
        for (int i = 0; i < length;) {
          int count = length - i;
          if (count > engine_max_batch(&model))
            count = engine_max_batch(&model);
          LogitsWanted request = want == LOGITS_LAST && i + count < length ? LOGITS_NONE : want;
          out = engine_forward_batch(&model, tokens + prefix + i, count, prefix + i, request);
          if (request == LOGITS_NONE)
            assert(out == NULL);
          else if (request == LOGITS_ALL)
            assert(memcmp(out, reference + (size_t)(prefix + i) * vocab, (size_t)count * vocab * sizeof(float)) == 0);
          else
            assert(memcmp(out, reference + (size_t)(prefix + i + count - 1) * vocab, vocab * sizeof(float)) == 0);
          i += count;
        }
        size_t expected = want == LOGITS_NONE ? 0 : (want == LOGITS_LAST ? 1 : (size_t)length);
        assert(profile.calls[STAGE_PREFILL][PHASE_CLASSIFIER] == expected);
        /* Future rows left by previous cases must not leak into attention. */
        for (int i = prefix + length; i < context && i < prefix + length + 3; i++) {
          out = engine_forward(&model, tokens[i], i);
          assert(memcmp(out, reference + (size_t)i * vocab, vocab * sizeof(float)) == 0);
        }
      }
    }
  memory_free(reference);
  memory_free(tokens);
  engine_free(&model);
  for (int i = 0; i < MEM_COUNT; i++)
    assert(memory_current[i] == 0);
  return 0;
}
