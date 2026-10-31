#include "runtime.h"
#include <assert.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static void unit_tests(void) {
  assert(checked_mul(12, 8) == 96 && checked_add(12, 8) == 20);
  double a = monotonic_seconds(), b = monotonic_seconds();
  assert(b >= a);
  float x[] = {1000, 1001, 1002};
  softmax(x, 3);
  assert(fabsf(x[0] + x[1] + x[2] - 1) < 1e-6f && sample_argmax(x, 3) == 2);
  float input[] = {1, 2, 3, 4}, weights[] = {1, 1, 1, 1}, out[4];
  rmsnorm(out, input, weights, 4);
  assert(fabsf(out[3] - 4 / sqrtf(7.5f + 1e-5f)) < 1e-6f);
  Sampler first = {0}, second = {0};
  build_sampler(&first, 3, 1, .9f, 123);
  build_sampler(&second, 3, 1, .9f, 123);
  for (int i = 0; i < 100; i++) {
    float u[] = {.3f, .8f, -.2f}, v[] = {.3f, .8f, -.2f};
    assert(sample(&first, u) == sample(&second, v));
  }
  free_sampler(&first);
  free_sampler(&second);
  Tokenizer t = {.vocab_size = 4, .max_token_length = 3};
  t.vocab = memory_alloc(4, sizeof(char *), MEM_TOKENIZER);
  t.vocab_scores = memory_alloc(4, sizeof(float), MEM_TOKENIZER);
  const char *pieces[] = {"A", "B", "C", "ABC"};
  for (int i = 0; i < 4; i++) {
    t.vocab[i] = memory_alloc(strlen(pieces[i]) + 1, 1, MEM_TOKENIZER);
    strcpy(t.vocab[i], pieces[i]);
    t.vocab_scores[i] = (float)i;
  }
  int tokens[11], n;
  encode(&t, "ABCABCABC", 0, 0, tokens, &n);
  assert(n == 3 && tokens[0] == 3 && tokens[1] == 3 && tokens[2] == 3);
  encode(&t, "", 0, 0, tokens, &n);
  assert(n == 0);
  free_tokenizer(&t);
  for (int i = 0; i < MEM_COUNT; i++)
    assert(memory_current[i] == 0);
}
int main(int argc, char **argv) {
  if (argc == 2 && !strcmp(argv[1], "--overflow")) {
    checked_mul(SIZE_MAX, 2);
    return 1;
  }
  if (argc == 2 && !strcmp(argv[1], "--add-overflow")) {
    checked_add(SIZE_MAX, 1);
    return 1;
  }
  unit_tests();
  if (argc == 2) {
    Model model = {0};
    engine_load(&model, argv[1]);
    for (int p = 0; p < 4; p++) {
      float *logits = engine_forward(&model, 65 + p, p);
      for (int i = 0; i < 12; i++)
        printf("%.9g\n", logits[i]);
    }
    engine_free(&model);
    for (int i = 0; i < MEM_COUNT; i++)
      assert(memory_current[i] == 0);
  }
  return 0;
}
