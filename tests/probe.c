/* Small subprocess bridge for numerical validation; no sampling or printing
 * transformations are applied to the forward logits. */
#include "runtime.h"
#include <stdlib.h>
#include <string.h>
/* A text argument starting with '@' names a file to read: long prompts do not fit in one argv string. */
static char *text_argument(const char *arg) {
  if (arg[0] != '@')
    return strdup(arg);
  FILE *file = fopen(arg + 1, "rb");
  if (!file)
    exit(2);
  size_t size = 0, capacity = 1 << 16;
  char *text = malloc(capacity);
  for (size_t got; text && (got = fread(text + size, 1, capacity - size - 1, file)) > 0;) {
    size += got;
    if (size + 1 >= capacity)
      text = realloc(text, capacity *= 2);
  }
  fclose(file);
  if (!text)
    exit(2);
  text[size] = 0;
  return text;
}

int main(int argc, char **argv) {
  if (argc < 3)
    return 2;
  if (!strcmp(argv[1], "tokenize") || !strcmp(argv[1], "chat")) {
    if (argc < 5)
      return 2;
    Tokenizer t = {0};
    build_tokenizer(&t, argv[2], atoi(argv[3]));
    int n, *tokens;
    char *text = text_argument(argv[4]);
    if (!strcmp(argv[1], "chat"))
      tokens = chat_prompt(&t, argc > 5 ? argv[5] : "", text, 1, &n);
    else {
      tokens = memory_alloc(strlen(text) + 2, sizeof(int), MEM_PROMPT);
      encode(&t, text, 1, 0, tokens, &n);
    }
    free(text);
    for (int i = 0; i < n; i++)
      printf("%d\n", tokens[i]);
    memory_free(tokens);
    free_tokenizer(&t);
  } else if (!strcmp(argv[1], "batch")) {
    if (argc < 4)
      return 2;
    Model model = {0};
    engine_load(&model, argv[2]);
    int n = argc - 3;
    int *tokens = memory_alloc(n, sizeof(int), MEM_PROMPT);
    for (int i = 0; i < n; i++)
      tokens[i] = atoi(argv[i + 3]);
    for (int i = 0; i < n;) {
      int count = n - i;
      if (count > engine_max_batch(&model))
        count = engine_max_batch(&model);
      LogitsWanted want =
          model.prefill_logits == LOGITS_ALL ? LOGITS_ALL : (i + count == n ? LOGITS_LAST : LOGITS_NONE);
      float *logits = engine_forward_batch(&model, tokens + i, count, i, want);
      size_t values = (want == LOGITS_ALL ? (size_t)count : want == LOGITS_LAST ? 1u : 0u) * model.config.vocab_size;
      if (values && fwrite(logits, sizeof(float), values, stdout) != values)
        return 1;
      i += count;
    }
    memory_free(tokens);
    engine_free(&model);
  } else if (!strcmp(argv[1], "generate")) {
    if (argc < 5)
      return 2;
    Model model = {0};
    engine_load(&model, argv[2]);
    int steps = atoi(argv[3]), pos = 0;
    float *logits = NULL;
    /* With LLAMA_PREFILL_BATCH > 1 the prompt goes through the batched prefill (C5/C6); decode stays per token. */
    if (engine_max_batch(&model) > 1) {
      int n = argc - 4;
      int *prompt = memory_alloc(n, sizeof(int), MEM_PROMPT);
      for (int i = 0; i < n; i++)
        prompt[i] = atoi(argv[i + 4]);
      for (int i = 0; i < n;) {
        int count = n - i < engine_max_batch(&model) ? n - i : engine_max_batch(&model);
        logits = engine_forward_batch(&model, prompt + i, count, i, i + count == n ? LOGITS_LAST : LOGITS_NONE);
        i += count;
      }
      pos = n;
      memory_free(prompt);
    } else
      for (int i = 4; i < argc; i++)
        logits = engine_forward(&model, atoi(argv[i]), pos++);
    for (int i = 0; i < steps; i++) {
      int token = sample_argmax(logits, model.config.vocab_size);
      printf("%d\n", token);
      if (token == model.config.eos_id || token == model.config.eot_id || token == model.config.eom_id ||
          i + 1 == steps || pos == model.config.seq_len)
        break;
      logits = engine_forward(&model, token, pos++);
    }
    engine_free(&model);
  } else if (!strcmp(argv[1], "rope")) {
    Config p = {.head_dim = 128,
                .rope_theta = 500000,
                .rope_type = 1,
                .rope_factor = (float)atof(argv[2]),
                .rope_low = 1,
                .rope_high = 4,
                .rope_original = 8192};
    for (int j = 0; j < 128; j += 2)
      printf("%.9g\n", rope_frequency(&p, j));
  } else {
    Model model = {0};
    engine_load(&model, argv[1]);
    for (int i = 2; i < argc; i++) {
      float *logits = engine_forward(&model, atoi(argv[i]), i - 2);
      if (fwrite(logits, sizeof(float), model.config.vocab_size, stdout) != (size_t)model.config.vocab_size)
        return 1;
    }
    engine_free(&model);
  }
  for (int i = 0; i < MEM_COUNT; i++)
    if (memory_current[i])
      return 3;
  return 0;
}
