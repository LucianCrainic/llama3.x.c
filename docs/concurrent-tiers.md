# Concurrent HBM and DDR streaming of the decode GEMV weights (`LLAMA_HYBRID_SPLIT=rows`)

Branch `feature/concurrent-tiers`, from `develop` 59f6bc6. Opt-in prototype; no default changes. Measurements, job ids and
raw files are in [concurrent-tiers-validation.md](concurrent-tiers-validation.md). Labels: *measured* (a job), *derived*
(arithmetic on measured values or on the code), *modelled* (a model, not run).

## 1. Why

The P1 study ([concurrent-tier-bandwidth.md](concurrent-tier-bandwidth.md), gate G1 marginal) found that raw concurrent reads
of HBM and DDR never beat HBM alone, but that the engine's own Q8 GEMV over the 70B shard, with 20% of the output rows in
DDR served by 14 of the 56 workers and 42 reading HBM, ran 1.4 to 3.9% faster than all-HBM (E5, p = 0.006). The use that
matters is the spill case: when the model does not fit in HBM (70B on one socket, or a long context on two) serial layer
hybrid reads its DDR layers at DDR speed while HBM idles. A row split keeps both tiers busy during every GEMV.

## 2. Knobs (all opt-in; see [configuration.md](configuration.md))

| knob | default | meaning |
| --- | --- | --- |
| `LLAMA_HYBRID_SPLIT` | `layers` | `rows` splits the output rows of each large matrix over HBM and DDR; `layers` is today's behaviour |
| `LLAMA_HYBRID_DDR_FRACTION` | `auto` | share of each matrix's per-socket output rows in DDR (0 to 0.9), or `auto` |
| `LLAMA_HYBRID_DDR_CORES` | 14 of 56, scaled | team members that serve the DDR rows (`1 <= n < threads`) |

Refusals (all at plan time, before any memory is reserved): `rows` without `LLAMA_PLACEMENT=hybrid`, with an FP32 or BF16
checkpoint, together with `LLAMA_HYBRID_WEIGHTS_HBM_LAYERS`; a fraction or core count outside its range or not a number;
`DDR_FRACTION` or `DDR_CORES` without `rows`. Tested in `tests/test_tp.py` (`test_row_split_knob_parsing_and_refusals`).

## 3. Design

**Which rows.** Per socket, per layer, the shards of Wq, Wk, Wv, W1, W3 (row shards) and Wo, W2 (column shards on two
sockets; one socket holds the whole matrix) each keep their first `rows - ddr_rows` rows in the socket's HBM weights arena
and the last `ddr_rows` rows in its DDR weights arena. `ddr_rows = floor(rows x fraction)` rounded down to a multiple of 32
(one pair of AMX tiles, so a packed copy splits at a tile boundary and a transposed scale chunk starts there), at most
15/16 of the rows; matrices of fewer than 32 rows stay whole. The classifier, norms and embedding are never split. Shard
fields: `ddr_rows`, `q2`, `s2` (`include/shards.h`). The row stride and layout inside each part are unchanged.

**Decode, one team call.** A GEMV segment whose weights are split (`K8Rows`, `include/kernels.h`) is served inside the same
team job as before: the last `n` members (`LLAMA_HYBRID_DDR_CORES`) take the DDR rows, shared statically, the others take the
HBM rows, shared statically; the QKV and SwiGLU fused jobs do this per segment. A team of one serves both parts itself.
`LLAMA_GEMV_STAGGER=1` rotates the block order inside each group exactly as it does inside a whole team. The rows-done
callback of the fused Wo/W2 reduction reports every row once, whatever group computed it. The residual, the activations,
the quantizer and the reductions are untouched. (`src/gemv.c`, `q8_team_split`.)

**Bitwise.** Each output row is a dot product of one weight row with the activation, computed by `q8_dots` alone: the
accumulation order depends only on the row's bytes and scales, not on which other rows share its block, which member
computes it, or which arena holds it. Moving a row to another arena or member therefore changes no bit (*derived*; also
tested bitwise at three levels: the kernels, the whole forward pass on the virtual topology, and on hardware).

**Prefill.** The batched GEMM does not use the team split. `gemm_q8_rows` runs the product on the HBM rows of all segments,
then on the DDR rows of the split ones (`c + rows`, `n = ddr_rows`, weights `q2/s2`, packed tiles at
`s8_vnni_bytes(first, k)`, transposed scales at `first / 32 x groups x 32` floats); each output column belongs to exactly one
call and its arithmetic does not depend on `n`. The interleaved Wo/W2 reduction flags the blocks of one GEMM, so a row-split
Wo or W2 takes the barrier form of the reduction (the same bits, [tensor-parallel.md](tensor-parallel.md)). The AMX packed
copy is packed from both parts (`pack_s8_jobs_from_shard`) and stays where `LLAMA_HYBRID_PACKED` puts it. The loader reads
the two parts of each shard with the same piece mechanism as before (`src/shards.c`).

**Planner.** `shards_place` counts the HBM part and the DDR part in the two per-socket weights arenas (64-byte aligned parts,
values then scales). Every other arena is the layer plan's. The charge of a node is the sum of the rounded arenas, as before
(`tests/test_tp.py: test_row_split_plan_arithmetic` recomputes the DDR arena in Python). `auto`:

* the smallest fraction on a 0.1% grid at which every socket's HBM charge fits its budget (bisection; the charge falls
  monotonically with the fraction, rounded arenas included), with KV caches of every layer in HBM (`auto` KV layers) and the
  classifier in HBM;
* when everything fits already, 0.2, the best fraction of the P1 engine-shaped study (a constant; section 5 of the validation
  document compares it with the measured optimum of this prototype).

If even 0.9 does not fit, the ordinary over-budget error names the node. The DDR budget (`LLAMA_DDR_BUDGET_GIB`) bounds the DDR
share. `--plan-only` and the `--json` report show the plan: `placement_plan.split`, `ddr_fraction`, `ddr_fraction_auto`,
`ddr_cores`, and the weights arenas of both tiers as before; `tools/check_placement.py --planned` verifies the pages of every
arena, the DDR weights arena included, against the plan.

## 4. What this does not do

* No change to attention, KV placement (the layer knobs apply as in serial hybrid), the residual or the reductions.
* No row split of the classifier, the embedding or the norms. FP32 and BF16 checkpoints are refused.
* The default `LLAMA_HYBRID_DDR_FRACTION=auto` balance constant is the P1 value, not a tuned one.
* A row-split Wo/W2 batched prefill uses the barrier reduction instead of the interleaved one: a measurable prefill cost is
  possible and was not studied (prefill is out of scope here).
