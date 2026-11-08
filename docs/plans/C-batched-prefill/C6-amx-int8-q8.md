# C6. AMX-INT8 for Q8_0 (W8A8)

## 1. Goal and status

Run batched Q8_0 prefill on AMX-INT8 tiles while **preserving the existing Q8_0
semantics**:

- the same per-row, per-group weight scales;
- the same `k_quantize` activation quantization (per token row, per group);
- the same group size.

The result must agree with the AVX-512 Q8 batched path within the kernel budget. W8A8
comes first. W8A32 stays a separate AVX-512 path.

Status: **implemented; cluster check passed on kernels and speed; W8A8 logit agreement recorded against a statistical gate, to be confirmed by a re-run** (see Cluster results).

## 2. Depends on / unblocks

- Depends on: C5 (ladder order, packing pass, budget reporting) and C4.
- Unblocks: C7, D1.

## 3. Reuse

| Code | Location | Note |
| --- | --- | --- |
| `k_quantize` | `src/kernels_simd.c:611` | Bitwise identical across ISAs. Applied per token row. |
| W8A8 semantics | `dots_q8_avx512`, `src/kernels_simd.c:163` | Per group: `int32 dot × ws[r,g] × xs[g]`, FMA into FP32. Requires `gs % 64 == 0`. |
| W8A32 | `dots_q8f_avx512`, `src/kernels_simd.c:190` | Stays on AVX-512 |
| Scalar Q8 | `src/kernels_simd.c:511-519` | Oracle |
| INT8 packing helper | C4 `pack_s8_vnni`, B1 `Shard` | Same descriptors for one or two sockets |

## 4. Changes

### 4.1 AVX-512 Q8 batched oracle

First, upgrade C2's per-row Q8 reference to a tiled AVX-512 VNNI batched kernel with
the same per-group semantics. This is both the oracle and the "avx512-q8" ladder rung
in C7.

### 4.2 AMX-INT8 kernel, for each N block × M block

```text
for each group g (K-step of gs = 64 int8 = one tile K-depth):
    C_i32 (16x16 tiles) = 0                       # zero
    _tile_dpbssd(C_i32, A_s8[m, g], B_s8[g, n])   # signed x signed, one group
    store C_i32 → workspace; C_f32[m][n] += C_i32[m][n] * ws[n, g] * xs[m, g]
```

- **Signed × signed.** `tdpbssd` multiplies signed values directly, so the |x|/sign
  trick that VNNI `dpbusd` needs is not needed. The int32 result per group equals the
  AVX-512 path's exactly.
- **Rescale order.** The FP32 rescale per group uses the same operand order as
  `dots_q8_avx512` (`acc += (float)i32 * (ws * xs)`), so results agree within the
  budget. They are not bitwise, because FMA vs separate multiply-add may differ; this
  is recorded.
- **Groups wider than one tile.** For `gs = 128`, accumulate two tile steps before the
  rescale. Any `gs % 64 == 0` is supported. Other group sizes fall back to the AVX-512
  batched path, with the fallback logged.

### 4.3 Packing

The weight int8 values are packed to VNNI quads per shard. Scales stay in their
existing row-major `[row][group]` layout, read per N tile. Packed bytes per socket for
8B:

| | Bytes |
| --- | ---: |
| Packed int8 copy, one socket (**modeled**) | about 7 GiB |
| Retained Q8 for decode | 7.43 GiB |
| Total | about 14.4 GiB |

### 4.4 W8A32

`LLAMA_Q8_ACTIVATIONS=fp32` with `amx` logs `W8A32 prefill uses avx512` and uses the
AVX-512 W8A32 batched kernel. It is never routed through AMX-INT8, since that would
change the activation semantics.

## 5. Local tests

These run on all platforms with scalar oracles; AMX cases run on AMX hosts only.

- **Activations.** `k_quantize` per row in the batched path gives int8 and scales
  bitwise equal to the per-token path.
- **Integer exactness.** Each group's int32 dot from the AMX tiles is exactly equal to
  the scalar int32 dot.
- **Final outputs.** Within `2e-5·Σ|w·x| + 1e-6` of the double reference built from
  the dequantized operands, which is the existing Q8 budget (`tests/kernels.c:126-167`).
  Covers `gs` ∈ {64, 128} and fallback for `gs = 32`.
- **Fixture.** AMX-INT8 forward vs AVX-512 Q8 batched forward within the fixture
  tolerance.

## 6. Cluster run

`cresco8-prefill.sbatch STAGE=check PHASE=C6` (with `MODELS=...`; `PRECISIONS=int8` is enough). **Estimated** at 60 minutes.
It needs the user's go-ahead to submit. **Open, not submitted**: `phase_C6` is written and verified with `DRY_RUN=1`. The
first time the AVX-512 VNNI tile kernel and the AMX-INT8 kernel run on hardware is `test-q8` there (it records PASS only
if both ran). At the 2048-token prompt only the last position is compared (`--last-only` needs the per-token path as A), so
AMX-INT8 is compared with the per-token Q8 path there and with the batched AVX-512 path on the 86-token prompt.

| Run | Gate or record |
| --- | --- |
| `compare_engines.py` 1B and 8B INT8: AVX-512 Q8 batched (M = 64) vs AMX-INT8 (M = 64), `all` logits | max\|Δlogit\| ≤ 1e-3, argmax 100% |
| `compare_engines.py` AVX-512 Q8 batched vs per-token Q8 | Same gate |
| `validate.py` Q8 recording | max_abs vs HF ≤ per-token Q8 value + 1e-3 |
| Short sweep: 8B Q8, 2048-token prompt, M ∈ {32, 64, 128, 256} | Prefill tok/s recorded |

### Local results (2026-11-07, macOS arm64, no AVX-512, no AMX)

| Check | Result |
| --- | --- |
| `test-q8`: `k_quantize_rows` (parallel, one call for M rows) vs the per-token quantizer | bitwise equal int8 values and scales |
| `test-q8`, scalar instruction set: portable batched GEMM vs the per-row GEMV (`k_matmul_q8`, `_add`, `_swiglu`) | **bitwise** equal |
| `test-q8` / `test-q8-emul`: portable, AVX-512 VNNI tile kernel and AMX-INT8 against a double reference from the dequantized operands, `gs` ∈ {2, 4, 32, 64, 128}, m, n tails, padded `ldc`, 1, 3, 7 workers, all four kinds | within `2e-5·Σ\|w·x\| + 1e-6`; padding untouched |
| `test-q8-emul`: AMX int32 group dots vs the scalar int32 dots | **exactly** equal (with activations and weights in [-127, 127]); 3 deliberate mutations of the kernels were caught |
| `test-q8-emul`: pick logic | AMX only for `gs % 64 == 0`; otherwise the portable backend, within budget |
| `test-q8-engine[-emul]`, fixture (dim 128, hidden 256, `gs` 64, plus a `gs` 32 checkpoint), 24 positions, all logits | scalar batched prefill equals the per-token forward **bitwise**, for any split; AMX-INT8 (tile model) vs the portable batched forward: max_abs 6.0e-7, max_rel 7.5e-5; bitwise independent of the split; per-token path unchanged by `amx`; W8A32 stays per row; `gs = 32` falls back to the portable GEMM |
| `tests/test_runtime.py` | `runq-amx-emul` report: `kernels_prefill = amx-int8`, `packed_weights_peak` equals the padded size formula; W8A32 logs `W8A32 prefill uses avx512` and reports no batched kernel |

### Implementation notes

- **Where it runs.** `k_gemm_q8` in `src/gemm.c` is the batched W8A8 driver; `runq.c` gets `forward_layer_batch` (quantize
  each row with `k_quantize` through `k_quantize_rows`, QKV / Wo / W1+W3 / W2 as GEMM segments, K and V written straight
  into the cache, `k_attention_batch`). It is used when `prefill_gemm` is on (`LLAMA_PREFILL_BATCH` > 1 and the AVX-512 or
  AMX kernels, or `LLAMA_PREFILL_GEMM=on`) and the activations are W8A8. The attention workspace is now sized for
  `max_batch` rows, and the quantized activation rows are padded to whole 32-row blocks (the AMX kernel loads 16 rows per
  tile).
- **Backends.** `amx-int8` (`gs % 64 == 0`, packed weights), `avx512-q8` (`gs % 64 == 0`; the oracle and the "avx512-q8"
  ladder rung of C7: 2 activation rows by 4 weight rows per VNNI tile, `|x|` unsigned and the weight carrying `x`'s sign,
  as in `dots_q8_body`), and `portable-q8` (any `gs`; the scalar GEMV's arithmetic). `Model.prefill_kernel` names the one in
  use and is the report's `runtime.kernels_prefill`.
- **Rescale order.** Per group, `acc = fmaf((float)i32 * xs, ws, acc)`, the operand order of `dots_q8_body`
  (`fmadd(mul(cvt(i32), xs), ws, acc)`), not the `ws * xs` the first draft of this document wrote. Not bitwise equal to the
  GEMV, because the GEMV keeps 16 lane accumulators and reduces at the end; the AMX kernel sums whole-group dots.
- **Packing.** `pack_s8_vnni_many` on the worker team into a `ROLE_PACKED` region (budget-checked, read-only, verified), the
  checkpoint's `[row][group]` scales read in place. The report shows `packed_weights_peak` and `weight_pack`.
- **W8A32.** `LLAMA_Q8_ACTIVATIONS=fp32` turns the batched GEMM off for the Q8 engine altogether (the per-row AVX-512 W8A32
  kernel runs, as today) and logs `W8A32 prefill uses avx512` when amx is selected. There is no batched W8A32 kernel.
- **Fallback.** `gs % 64 != 0` with amx or AVX-512 logs the fallback and uses the portable batched GEMM (which is what the
  GEMV itself does for W8A8 at such group sizes).
- **Classifier.** The all-position classifier of `runq` stays on the per-row Q8 GEMV.
- **Rescale overhead (§8).** First version, not tuned: each group stores the 16 x 16 int32 tiles to a stack buffer and rescales
  in plain C that the compiler vectorizes (the kernel's target includes AVX-512). The mitigations of §8 are the first thing
  to try if the cluster measurement is far from the roofline.

### Cluster results (2026-11-08)

cresco8-hbm14 (Flat mode, 2 × Xeon Max 9480), gcc 11.4.1, `OPENMP=1`, `STAGE=check`, `HF_GATE=0`. Jobs 5694396 (commit `44ce282`), 5694401 and 5703346 (commit `4677bd5`, diagnostics).

- **Kernels on hardware.** `test-q8` passes with the AVX-512 VNNI tile kernel and AMX-INT8 on 1, 28 and 56 workers at
  real layer sizes (K up to 14336, N up to 14336, M up to 128; 34,360,924 checks, 0 failures); the batched AVX-512 Q8 path
  is bitwise identical to the per-token path on 1B and 8B (86 and 2048 tokens).
- **Prefill speed** (2048-token prompt, 56 threads, median of 3, W8A8):

  | Model | per-token | AVX-512 Q8, M = 128 | AMX-INT8 M = 32 | M = 64 | M = 128 | M = 256 |
  | --- | ---: | ---: | ---: | ---: | ---: | ---: |
  | 1B, tok/s | 277 | 1126 | 1961 | 2232 | 2351 | 2432 |
  | 8B, tok/s | 48 | 200 | 437 | 495 | 527 | 563 |

  AMX-INT8 is 2.1x (1B) to 2.7x (8B) the AVX-512 Q8 batched path at M = 256, and 11.7x the per-token path on 8B; decode is unchanged (8B 41 tok/s).
- **The 1e-3 gate failed, and the cause is not the kernel.** AMX-INT8 against the AVX-512 path gives max |Δlogit| 0.42 and rms
  0.063 (1B), 1.41 and 0.086 (8B), argmax 98.1%. The deviation is the same at 1, 4 and 56 threads (no race) and the same as
  between paths that share no AMX code: scalar against AVX-512 per-token on 1B gives rms 0.063, max 0.34, argmax 96.2%,
  and scalar against AMX 0.062 / 0.36 / 94.3%, while W8A32 agrees to 3e-5 on all of them. W8A8 requantizes its
  activations, so any change of FP32 order moves the logits by this noise floor. The comparison now uses `W8A8_GATE`
  (max 2.0, rms 0.15, argmax >= 0.9, from these values), and a strict W8A32 comparison of the AMX selection was added; both
  are in `phase_C6` and have not been run yet.
- **Not run.** The Hugging Face recording of the Q8 variants (`HF_GATE=0`).

## 7. Exit criteria

- AMX-INT8 W8A8 agrees with the AVX-512 Q8 batched path and the per-token path.
- Q8 semantics are unchanged, with bitwise activation quantization.
- W8A32 is untouched.
- Packing is reported in the per-socket budget.

## 8. Risks and open questions

- **Rescale overhead.** With `gs = 64`, every tile `dpbssd` (16×16×64 MACs) is
  followed by a 16×16 store, convert and FMA. That overhead can cap AMX-INT8 well below
  the **measured** 69 TOPS library rate. Mitigations, in order:
  1. Keep C tiles for 2×2 blocks and rescale while the next tile op is in flight.
  2. Rescale in registers after `_tile_stored` into an L1-resident buffer.
  3. Measure, and report the achieved fraction.

  Changing the Q8 format (a larger `gs`) is out of scope.

## 9. Not in this phase

- W8A32 on AMX.
- New quantization formats.
- INT8 attention.
