#ifndef LLAMA_PREFILL_H
#define LLAMA_PREFILL_H

#include "runtime.h"

/* Shared policy/validation; numerical work stays in the precision engines. */
void prefill_configure(Model *model);
void prefill_validate(const Model *model, const int *tokens, int count, int pos0, LogitsWanted want);

#endif
