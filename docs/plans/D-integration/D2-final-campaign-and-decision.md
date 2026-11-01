# D2. Final campaign and decision

## 1. Goal and status

Run one controlled campaign covering the complete feature set, and decide for each
feature whether it becomes a default, an option, or is documented only as a capacity
result:

- placement;
- the two-socket split;
- batched prefill at each rung;
- their combination.

Prefill and decode are always measured and reported separately.

Status: **planned**. Branch: `feature/integ-d2-final-campaign` (scripts and results
docs only).

## 2. Depends on / unblocks

- Depends on: D1, B7 and C7.
- Unblocks: nothing. This phase closes the plan (milestone M5).

## 3. Reuse

| Source | Use |
| --- | --- |
| A6, B7, C7 matrices and summaries | Baselines. D2 reruns only the headline points, so every number comes from one commit and one node state. |
| F3 `summarize_campaign.py` | Tables and ranges |
| Study §10.3–10.4 | Experiment list and decision criteria |

## 4. Changes

- A `PHASE=D2` matrix in `scripts/cresco8-prefill.sbatch`, which covers decode-only
  points too.
- Results docs: `docs/numa-tp-prefill-results.md` and
  `docs/numa-tp-prefill-results-validation.md`.
- Updates to `docs/kernels.md` "Not in this change" and to the
  [plans README](../README.md) milestones.

### 4.1 Headline matrix

Every configuration runs 3 interleaved repeats, with placement verified:

| Axis | Values |
| --- | --- |
| Sockets | 1 (A6 best placement), 2 (B7 best workers) |
| Precision / rung | FP32 per-token, FP32 AVX-512 GEMM, BF16 AMX, Q8 per-token, Q8 AVX-512 VNNI, Q8 AMX-INT8 |
| Models | 1B, 8B |
| Prompt | 86, 2048, 8192 |
| M | Per rung: the C7 choice for one socket, the D1 choice for two sockets |
| Decode | 128 tokens from retained contexts of 512 and 8192, separately timed |
| Counters | HBM/DDR CAS and UPI per socket, counted once per node |

**Estimated** duration is 5–6 hours, which may be two jobs split by model. It needs
the user's go-ahead to submit.

### 4.2 Report contents

1. **Prefill.** Prefill tok/s, TTFT and speedup over the per-token one-socket baseline,
   for each precision × sockets × prompt cell.
2. **Decode.** Decode tok/s for each precision × sockets × context cell, compared with
   `docs/kernels.md` and A6.
3. **Crossover tables** (study §10.4): where two sockets win for decode and for
   prefill, by precision, model and context; where synchronization dominates; where
   attention dominates.
4. **Placement evidence.** Region tables and verification coverage. THP backing.
5. **Numerical evidence.** Every gate's observed error. The BF16 provisional gate is
   confirmed or revised.
6. **Modeled vs measured.** The roofline (`docs/hbm-roofline.md` §3–4) and study §8.3
   models against the measured results, with the gap explained.

## 5. Local tests

Run the script dry run and the summarizer on synthetic v2 reports.

## 6. Cluster run

As in §4.1, on the Flat node, `--exclusive`.

## 7. Exit criteria and decision rules

| Feature | Becomes default when | Otherwise |
| --- | --- | --- |
| Engine-owned placement (`LLAMA_PLACEMENT=hbm`) | Not slower than staged HBM, and verified | Option, for verification and capacity |
| Two sockets (`LLAMA_SOCKETS=2`) | Lower latency than the best one-socket configuration beyond variability, for that model/precision/context | Option; crossover documented; capacity result reported separately |
| Batched prefill (`LLAMA_PREFILL_BATCH=M`) | Faster TTFT than per-token beyond variability, with gates passing | Option |
| AMX-INT8 for Q8 prefill | Faster than AVX-512 Q8 batched, with the gate passing | Option |
| AMX-BF16 | **Never a silent default for FP32 checkpoints.** It stays an explicit precision mode (user decision). Recommended in documentation when it is faster within its gate. | — |

The plan is complete when every row has a decision backed by D2 evidence, and the
README milestones show M1–M5 done.

## 8. Risks and open questions

- **Node availability.** If the Flat node's availability limits D2, prioritize 8B and
  the 2048-token prompt, and record what was skipped.

## 9. Not in this phase

- New features. Follow-ups, such as BF16 checkpoints or KV, KV tiering, multi-request
  batching, exchange/compute overlap or AMX attention, are listed in the results doc as
  candidates for a later plan.
