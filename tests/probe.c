/* Small subprocess bridge for numerical validation; no sampling or printing
 * transformations are applied to the forward logits. */
#include "runtime.h"
#include <stdlib.h>
#include <string.h>
int main(int argc, char **argv) {
  if (argc < 3)
    return 2;
  if (!strcmp(argv[1], "tokenize") || !strcmp(argv[1], "chat")) {
    if (argc < 5)
      return 2;
    Tokenizer t = {0};
    build_tokenizer(&t, argv[2], atoi(argv[3]));
    int n, *tokens;
    if (!strcmp(argv[1], "chat"))
      tokens = chat_prompt(&t, argc > 5 ? argv[5] : "", argv[4], 1, &n);
    else {
      tokens = memory_alloc(strlen(argv[4]) + 2, sizeof(int), MEM_PROMPT);
      encode(&t, argv[4], 1, 0, tokens, &n);
    }
    for (int i = 0; i < n; i++)
      printf("%d\n", tokens[i]);
    memory_free(tokens);
    free_tokenizer(&t);
  } else if (!strcmp(argv[1], "generate")) {
    if (argc < 5)
      return 2;
    Model model = {0};
    engine_load(&model, argv[2]);
    int steps = atoi(argv[3]), pos = 0;
    float *logits = NULL;
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
