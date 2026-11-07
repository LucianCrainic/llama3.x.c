# C5. AMX-BF16 precision mode

## 1. Goal and status

Add an **explicit** BF16 precision mode for FP32 checkpoints, used by batched prefill
with AMX-BF16.

- **Weights.** Packed once at load into BF16 VNNI layout, per shard and owned per
  socket.
- **Activations.** Converted to BF16 with round-to-nearest-even before each GEMM.
- **Accumulation, epilogues, norms, attention and reductions** stay in FP32.

FP32 execution is never changed silently. Without `LLAMA_PRECISION=bf16`, nothing in
this phase is active.

Status: **implemented locally; cluster check pending** (2026-11-07). Branch: `feature/prefill-c5-amx-bf16` (worked
on in the `tp-local-c` worktree). The mode, packing, GEMM driver, scalar reference forward, report fields and guards
are in and tested on macOS (portable reference and the software tile model). The AMX hardware run, the error
characterization against FP32 and HF, and therefore the final error gate, are open: §6.

## 2. Depends on / unblocks

- Depends on: C3 (complete batched forward) and C4 (AMX enablement and packing
  helpers).
- Unblocks: C6, C7, D1.

## 3. Reuse

| Code | Location | Note |
| --- | --- | --- |
| Shards and loader | B1 (`Shard`, `shards_load`), A4 | Packing runs in the same owner-team pass |
| GEMM driver and epilogues | C2 | The AMX tile path is a different microkernel under the same driver |
| BF16 widening in tools | `tools/export.py:618`, `tools/validate.py:4` | Shows that the HF sources are BF16, so FP32 checkpoints are exact widenings of BF16 weights |
| `engine_precision` | `src/run.c` | Reports `"fp32"` today |

## 4. Changes

### 4.1 Mode semantics

| Item | Behaviour with `LLAMA_PRECISION=bf16` (`run` only) |
| --- | --- |
| `engine_precision` | Returns `"bf16_from_fp32"`. The report's `runtime.precision_mode` is `bf16`. |
| Prefill projections (QKV, Wo, W1/W3, W2) | AMX-BF16 GEMM on packed weights; FP32 accumulation; FP32 epilogues |
| Activations into GEMM | FP32 → BF16, RNE, converted per microbatch into a 64-byte-aligned A panel (`_mm512_cvtne2ps_pbh` on AVX512-BF16, or an integer-rounding fallback) |
| Classifier | GEMV on the last row in FP32 (unchanged) by default. `LOGITS_ALL` uses AMX-BF16. |
| Decode | **Unchanged FP32 GEMV** on the retained FP32 weights |
| Norms, RoPE, attention, residuals, reductions | FP32 |

### 4.2 Memory budget

The budget is per socket, and for 8B it is **modeled** from study §8.1:

| Item | One socket | Each socket when S = 2 |
| --- | ---: | ---: |
| Retained FP32 matrices and norms | 27.96 GiB | 13.98 GiB |
| Packed BF16 matrices | 13.98 GiB | 6.99 GiB |
| Total weights in HBM | 41.94 GiB | 20.97 GiB |

The one-socket total of 41.94 GiB is under 56 GiB but leaves less room for KV. The
loader checks the per-socket budget (A2) before packing and reports
`packed_weights_peak` separately.

`LLAMA_BF16_DROP_FP32=1` is reserved for a future BF16 decode GEMV. It is rejected in
this phase.

### 4.3 Error gate (provisional)

1. **Compare.** On the first `check` run, compare BF16 mode against the FP32 one-socket
   reference and against HF. Record max_abs, rms and argmax agreement at all prompt
   positions, plus greedy generation agreement for 8 steps.
2. **Fix the gate.** Set it from the observed distribution, with margin. The initial
   proposal: argmax agreement at least 99% across all positions of the 4
   `validate.py` prompts, and identical greedy continuations on at least 3 of 4
   prompts.
3. **Record.** Write the gate into this document and into `docs/` results. It is never
   compared against the FP32 1e-3 budget.

## 5. Local tests

These run on macOS without AMX. A scalar BF16 reference GEMM (convert, then multiply
in FP32) runs everywhere.

- **Kernel accumulation.** The AMX BF16 GEMM (on AMX hosts) vs the scalar BF16
  reference within `2e-6·Σ|w·x|` on BF16-rounded inputs.
- **Conversion.** RNE conversion is bitwise equal to a reference for ties, NaN, ±Inf,
  denormals and the max value.
- **Fixture.** BF16-mode logits vs FP32 are recorded in the test output, not gated.
  The scalar BF16 reference forward and the AMX forward agree within the kernel budget.
- **Mode guard.** `LLAMA_KERNELS=amx` on an FP32 checkpoint without
  `LLAMA_PRECISION=bf16` fails with the documented message.

## 6. Cluster run

`cresco8-prefill.sbatch STAGE=check PHASE=C5` (with `MODELS=...` for the model part; `PRECISIONS=fp32` is enough).
**Estimated** at 60 minutes. It needs the user's go-ahead to submit. **Open, not submitted**: `phase_C5` is written and
verified with `DRY_RUN=1`. It records (it does not gate) every BF16 comparison and writes `bf16-error.md` with max_abs, rms
and argmax agreement per prompt and for the FP32 AVX-512 GEMM reference next to it. Two small tool changes serve it:
`tools/validate.py --batch M --record-only` (logits through the batched path; exit 0 whatever the comparison shows) and
`tests/probe.c generate` (prefills the prompt through the batched path when `LLAMA_PREFILL_BATCH` > 1, so the greedy
agreement is BF16 too). The packing-time run uses `LLAMA_PLACEMENT=hbm` and so depends on the A4/A5 cluster checks.

| Run | Gate or record |
| --- | --- |
| 1B and 8B: `compare_engines.py` with FP32 per-token as A and BF16 AMX at M = 64 as B; `all` logits, 4 prompts | Recorded; sets the provisional gate |
| `validate.py --hf` in BF16 mode | Recorded vs HF |
| Packing time, packed bytes, per-socket budget | **Measured** |
| Short sweep: 8B, 2048-token prompt, M ∈ {32, 64, 128, 256} | Prefill tok/s recorded |

### Local results (2026-11-07, macOS arm64, no AMX)

| Check | Result |
| --- | --- |
| `test-bf16`: FP32 to BF16 conversion vs an independent round-to-nearest-even reference built from the two neighbouring values | bitwise equal on ties at every exponent (and ±1 around each tie), ±0, ±Inf, NaN (signalling, payload below the BF16 bits, quieted, sign kept), the largest finite value and its tie to infinity, the denormal range, a stride sweep of the whole 32-bit space and 1e6 random values |
| `test-bf16`: `k_gemm_bf16` (STORE, ADD, SWIGLU, PARTIAL; m, n, k tails; padded `ldc`) vs a double reference on BF16-rounded operands, 1, 3 and 7 workers | within `2e-6·Σ\|w·x\|` (SWIGLU: the composite bound of `tests/gemm.c`); padding between rows untouched |
| `test-bf16-emul`: the AMX path on the software tile model vs the portable reference | **bit for bit** equal on all of the above (the model adds every K pair in ascending order, and a BF16 product is exact in FP32) |
| `test-bf16-engine[-emul]`, synthetic fixture, 24 positions, all-position logits | per-token (decode) logits in BF16 mode equal the FP32 model **bitwise**; BF16 logits are bitwise independent of the microbatch split (M = 2, 7, whole); emulated AMX equals the scalar BF16 reference forward (max\|Δ\| = 0). BF16 vs FP32 on this fixture: max_abs 1.5e-3, rms 3.7e-4, argmax agreement 1/24 (the fixture's logits are near-flat noise; recorded, not gated) |
| `tests/test_runtime.py` | mode validation, guards, report fields, packed-region accounting (`packed_weights_peak` equals the padded size formula under `LLAMA_PLACEMENT=ddr`), FP32 default unchanged |
| `make test`, `make MODE=sanitize test` | green on macOS arm64, release and sanitize builds (2026-11-07, after C7) |

Not exercised locally: the real `tdpbf16ps` instruction (its in-pair summation order may differ from the model's in the
last bit, which is the reason for the 2e-6 budget on hardware), the 1B and 8B models, the Hugging Face comparison, and
everything about speed.

### Implementation notes

- **Selection.** `LLAMA_PRECISION=bf16` (default `fp32`; `run` only, `runq` rejects it). It needs `LLAMA_PREFILL_BATCH` of
  2 or more and is incompatible with `LLAMA_PREFILL_GEMM=off`, both rejected with a message. With `LLAMA_KERNELS=amx`
  the AMX tiles run; with any other kernel set the portable BF16 GEMM runs (logged as slow, reported as
  `kernels_prefill: "bf16-reference"`, and it is what the tests use as the scalar reference). `LLAMA_KERNELS=amx` on an
  FP32 checkpoint without `LLAMA_PRECISION=bf16` fails: `LLAMA_KERNELS=amx on an FP32 checkpoint requires
  LLAMA_PRECISION=bf16: FP32 execution never changes precision silently`. `LLAMA_BF16_DROP_FP32` is rejected.
- **What runs in BF16.** QKV, Wo, W1/W3, W2 of the batched path, including a batch of one row (so a prompt's result does
  not depend on how it was split; the decode path `engine_forward` never uses the packed weights). The last-row
  classifier stays FP32; `LOGITS_ALL` runs one BF16 GEMM on a classifier packed on first use (accounted, logged).
- **Packing.** At load, on the worker team, into one `ROLE_PACKED` region (`packed_open/take/seal` in `src/loader.c`):
  bound to the placement mode's weights node, claimed against `LLAMA_HBM_BUDGET_GIB` before any byte is packed,
  read-only afterwards and page-verified like the other regions. `packed_weights_peak` is the region size,
  `timing_seconds.weight_pack` the time. The B1 `Shard` descriptor is not used yet: the loop packs the whole matrices;
  with shards, the same `pack_bf16_vnni_many` job list takes one entry per shard (see the C4 note).
- **Report.** `runtime.precision_mode` is `bf16` (the engine's `engine_precision()` returns `bf16_from_fp32`),
  `runtime.kernels_prefill` is `amx-bf16` or `bf16-reference`, `build.amx_permission`, `memory_bytes.packed_weights_peak`,
  `timing_seconds.weight_pack`.
- **Conversion.** Integer round-to-nearest-even; denormals are converted, not flushed; NaN stays NaN (quieted). The
  AVX512-BF16 instruction `vcvtneps2bf16` is deliberately not used: it flushes denormal inputs, which would break the
  bitwise equality with the reference. Measure the conversion cost on the cluster (§8) before deciding whether a
  vectorized integer version is needed.

## 7. Exit criteria

- The BF16 mode is explicit and reported.
- Its error is characterized, and the provisional gate is written down.
- The AMX path matches the scalar BF16 reference within budget.
- Decode is unchanged; checked bitwise against FP32 decode after the same prompt
  state.

On that last point: BF16 prefill writes K/V computed in BF16 mode, so the decode
logits differ through the cache. The bitwise check is therefore made with a decode-only
comparison starting from the same cache.

## 8. Risks and open questions

- **Cache contents differ.** The KV cache written by BF16 prefill differs from FP32
  prefill. That is inherent to the mode, and is why the gate is end to end, not per
  token.
- **Conversion cost.** Activation conversion per microbatch is small relative to the
  GEMM (M·K vs N·K), but measure it.

## 9. Not in this phase

- BF16 checkpoints.
- BF16 KV.
- BF16 decode GEMV.
- Dropping the FP32 copy.
