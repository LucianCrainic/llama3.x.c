# Batched prefill

C1 introduces a layer-major reference path for one sequence on one socket. It
uses the current scalar/AVX-512 GEMV and attention kernels, processing rows in
causal order within each layer. It supports FP32, Q8 W8A8, and Q8 W8A32. This is
the numerical reference for the later GEMM and AMX phases; it does not yet reuse
weights through GEMM or execute AMX instructions.

```sh
make -j4
LLAMA_PREFILL_BATCH=32 build/release/run checkpoint.bin -z tokenizer.bin \
  --prompt-file prompt.txt --generation-length 32 --json result.json
```

`runq` accepts the same settings. `LLAMA_Q8_ACTIVATIONS=fp32` continues to select
W8A32. Existing kernel selection and checkpoint formats are unchanged.

## Configuration and behavior

| Setting | Default | Behavior |
| --- | --- | --- |
| `LLAMA_PREFILL_BATCH` | `1` | Positive decimal integer up to `INT_MAX`; effective capacity is capped at the allocated context |
| `LLAMA_PREFILL_LOGITS` | `last` | `last` computes only the final prompt's classifier; `all` computes and retains every row within each microbatch |

Unset or empty values select the defaults. Invalid nonempty values fail at model
initialization. The batch size also respects the remaining prompt and `-n` limit.
A truncated prompt produces no sampled output. Prompt echo emits the same bytes,
but flushes after each microbatch. Decode and chat EOT caching use single-token
forward calls.

Even at the default batch size of one, unused prompt classifiers are now skipped.
For timing comparisons with the previous per-token behavior, select
`LLAMA_PREFILL_LOGITS=all`. No performance claim has yet been measured on the target
hardware.

JSON reports add `parameters.prefill_batch`, `parameters.prefill_logits` and
`prefill.microbatches`. `tokens.forward_passes` still counts tokens, rather than
calls. TTFT includes tokenization, all prefill, and the first non-stop sample as
before. Diagnostic all-row logits are not emitted in the performance JSON.

## C API and memory

`engine_forward_batch(model, tokens, count, pos0, want)` appends tokens at positions
`pos0..pos0+count-1`. The caller must have populated the preceding KV prefix.
All input tokens and the complete range are validated before KV writes.

| Request | Return |
| --- | --- |
| `LOGITS_NONE` | `NULL`; only activations and KV are updated |
| `LOGITS_LAST` | One vocabulary-sized vector for the final row |
| `LOGITS_ALL` | `count × vocab_size` floats in row order |

Output storage is model-owned and must not be retained across another forward
call. `LOGITS_ALL` can be requested irrespective of the environment setting.
Its `M × vocab_size` buffer is allocated on first use; last-only execution never
allocates it. Activations and RoPE tables are allocated for M rows, while KV and
serial attention scratch do not grow with M. All allocations use the existing
tracked allocator and are released by `engine_free`.

## Validation tools

```sh
make -j4 test
make -j4 MODE=sanitize test
make build/release/probe-fp32 build/release/probe-int8
python3 tools/compare_engines.py --probe build/release/probe-fp32 \
  --checkpoint checkpoint.bin --tokens 128000,9906,11,1917 \
  --batch 32 --output comparison.json
```

The comparison runs the single-token oracle and batched diagnostic path in
separate processes, with identical kernel and Q8 settings. Its gate requires
finite, bitwise-identical logits at every position. `--reference-probe` can point
to a probe from an earlier build. Token IDs must match the checkpoint. Full
logits are spooled to temporary files; `TMPDIR` controls their location and
required disk space is twice `prompt_length × vocab_size × 4` bytes. Comparison
memory is bounded to a pair of rows.

The optional `make test-reference PYTHON=/path/to/python` suite uses the existing
ML dependencies to compare synthetic Transformers models and now also checks
bitwise batch agreement at head dimensions 64 and 128. Real-model runs belong in
an allocated compute job. See [C1 validation](prefill-c1-validation.md) for the
completed checks and remaining hardware gate.
