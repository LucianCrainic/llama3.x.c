#ifndef LLAMA_APP_H
#define LLAMA_APP_H
#include "runtime.h"
typedef struct {
  const char *checkpoint, *tokenizer, *prompt, *prompt_file, *mode, *system_prompt, *json;
  int plan_only;
  int chat_template, context, steps, generation_length, benchmark, no_text;
  int warm_context; /* --warm-context N: forward N untimed tokens before the prompt (B7), 0 = off */
#ifdef LLAMA_ENABLE_SYNTHETIC_KV
  int synthetic_kv, synthetic_kv_prime; /* SYNTHETIC_KV=1 builds: LLAMA_SYNTHETIC_KV, LLAMA_SYNTHETIC_KV_PRIME */
  double synthetic_kv_min_s;            /* LLAMA_SYNTHETIC_KV_MIN_SECONDS: least real compute before the fill */
#endif
  unsigned long long seed;
  float temperature, topp;
} Options;
extern Options options;
extern Model model;
void write_report(void);
#endif
