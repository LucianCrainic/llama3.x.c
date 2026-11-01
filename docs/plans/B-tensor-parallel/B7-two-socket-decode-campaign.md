# B7. Two-socket decode campaign and decision

## 1. Goal and status

Measure whether the two-socket split improves single-request decode latency over the
**optimized one-socket baseline** (A6), and explain where the time goes. The split is
judged only by the study §10.4 criteria.

Status: **planned**. Branch: `feature/tp-b7-campaign` (scripts and results docs only).

## 2. Depends on / unblocks

- Depends on: B6, A6 (baseline and THP default) and F3.
- Unblocks: D2.

## 3. Reuse

| Code | Location | Use |
| --- | --- | --- |
| A6 matrix and baseline | A6 | One-socket `anon-hbm` best configuration per model and precision |
| Old two-socket runs | `scripts/cresco8-kernels.sbatch:166-168` | Measured "112 threads without TP" (6.42 and 8.17 tok/s FP32 8B); kept as a control |
| Library, summary | F3 | — |
| Study matrix | study §10.3 | Normative experiment list |

## 4. Changes

- A `PHASE=B7` matrix in `scripts/cresco8-tp.sbatch`.
- An optional all-output-row comparator, `LLAMA_TP_PARTITION=rows`, implemented here
  only if the time budget allows (study §6.4).
  - All matrices, including Wo and W2, are split by output rows. This requires an
    all-gather of attention/FFN input vectors instead of a sum.
  - It reuses existing full-row kernels and the same placement.
- Results docs: `docs/tensor-parallel.md` and `docs/tensor-parallel-validation.md`.

### 4.1 Run matrix

All runs use `LLAMA_BACKEND=teams`, HBM placement, the THP setting chosen by A6, and
3 interleaved repeats.

| Dimension | Values |
| --- | --- |
| Configurations | one socket ×56 (A6 best); two sockets 2 × {14, 28, 42, 56}; control: 112 threads, no TP, interleaved HBM (old style) |
| Models | 1B, 8B |
| Precision | FP32, Q8 W8A8; W8A32 at 2 × 56 only |
| Context | Retained-context starts at 512, 8192 and 32768 tokens (prefill not timed for decode), then 128 decode tokens |
| Profiling | One extra repeat per configuration with `LLAMA_PROFILE=1`, kept separate from the timed medians |
| Counters | perf HBM/DDR CAS reads and UPI flits, per socket, counted once per node (study §10.3) |
| Comparator | `LLAMA_TP_PARTITION=rows` at 2 × 56, 8B FP32/Q8, context 512 (if implemented) |

Long-context decode is measured from a comparable retained context, never averaged
with an empty-cache run (study §10.3). That requires a `--warm-context N` benchmark
option (prefill N tokens untimed, then time decode). It is added here, in `app.c`
`generate`.

**Estimated** duration is 4–6 hours, so it may need to be split into two jobs by
model. It needs the user's go-ahead to submit.

## 5. Local tests

- A script dry run.
- `--warm-context` unit test on the fixture: timed decode excludes the warm prefill,
  and the report records `warm_context`.

## 6. Cluster run

As in §4.1. Outputs:

- `summary.md` with medians and ranges.
- A per-phase table for each socket: qkv, attention, wo, ffn, classifier, reduce,
  barrier_wait and imbalance = |s0 − s1|.
- **Modeled** vs **measured** time: the study §8.3 model ("sum of the per-layer max
  plus exchange") compared with the observed time.

## 7. Exit criteria (study §10.4)

**Improvement.** Two-socket decode counts as an improvement only if it beats the best
one-socket configuration beyond run variability, with every numerical and placement
gate passing. Report the result per precision, model and context.

**The crossover table records:**
- where two sockets win;
- where synchronization dominates (likely 1B Q8);
- where attention dominates (long context).

**Revisit triggers.** The design must be revisited if any of the following holds
(study §10.4):
- reductions plus waits exceed about 20% of token time;
- one socket waits on the other more than about 10% of the time;
- UPI counters show attention or weight traffic crossing sockets.

**Capacity.** A capacity-only gain, such as a context that fits only with two
sockets' HBM, is reported separately.

## 8. Risks and open questions

- **Bandwidth stops being the limit.** The ceiling modelled from **measured** stream
  rates is 27.19 ms per token for FP32 8B on two sockets (study §8.3). The kernels
  currently reach 81% of the one-socket bandwidth; if the two-socket kernels do not,
  compute or synchronization becomes the limit.
- **Interference.** The flat node is shared with other campaigns. Every result records
  the exclusivity and job ID.

## 9. Not in this phase

- Prefill on two sockets (D1, D2).
- Multi-request decode.
- KV tiering.
