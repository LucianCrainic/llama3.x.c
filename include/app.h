#ifndef LLAMA_APP_H
#define LLAMA_APP_H
#include "runtime.h"
typedef struct {
  const char *checkpoint, *tokenizer, *prompt, *prompt_file, *mode, *system_prompt, *json;
  int chat_template, context, steps, generation_length, benchmark, no_text;
  unsigned long long seed;
  float temperature, topp;
} Options;
extern Options options;
extern Model model;
void write_report(void);
#endif
