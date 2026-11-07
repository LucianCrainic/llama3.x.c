#ifndef LLAMA_PREFILL_H
#define LLAMA_PREFILL_H

#include "runtime.h"

/* Shared policy/validation; numerical work stays in the precision engines. */
void prefill_configure(Model *model);
/* LLAMA_PRECISION (C5): fp32 (default) or bf16, the explicit BF16 mode for FP32 checkpoints. Returns 1 for bf16.
 * `quantized` is 1 for the Q8_0 engine, where bf16 is an error. Also enforces the mode guard: LLAMA_KERNELS=amx on an
 * FP32 checkpoint without bf16 is an error, because FP32 execution never changes precision silently. Call after
 * kernels_init() and set Model.prefill_bf16 from the result before prefill_configure(). */
int prefill_precision(int quantized);
void prefill_validate(const Model *model, const int *tokens, int count, int pos0, LogitsWanted want);

#endif
