# Validation of the planned-placement merge (24 November 2026)

Companion of `docs/hbm-ddr-placement.md`. Tree under test: the sources of `33a7fa4` (merge `feature/hbm-ddr-placement`
1bc9189 into `develop` b9b9bbb, plus the fix, test and script commits; later commits only add this document and the
reworded merge message, so `git diff 33a7fa4 HEAD -- src include tests tools scripts` is empty). Numbers are measured
unless labelled. Hardware: CRESCO8 Flat node (dual Xeon Max 9480, gcc 11.4, `-O3`), 8B checkpoints (int8 gs 64 `int8-v3`,
FP32 `fp32-v3`, gs 32 `q8-gguf-8b-v3`, per-channel `int8-pc-v4`).

## 1. Jobs

| job | node | what | tree | outcome |
|---|---|---|---|---|
| 5916198 | hbm13 | `p2-merge-ab`: bitwise logits and token traces develop against merged, interleaved 5-repeat benchmark, planned modes with `check_placement.py --planned`, gs 32 and per-channel files, 70B plan-only | b9b9bbb, 33a7fa4 | 122 PASS, 2 FAIL (both a defect of the job script: `plan-only` of `hbm-only` and `ddr` was given `LLAMA_HYBRID_KV_HBM_LAYERS`, which the engine refuses outside `hybrid`; rerun in the gate job); results `p2-20261124-merge-ab` |
| 5916200 | hbm13 | `p2-merge-gate`: `make test` on socket 0, ThreadSanitizer (gcc-toolset-14) of `test_tp.py`, `test-q8`, `test-q8-emul`, `test-q8-engine-emul`; plan-only of the three placements | 33a7fa4 | GATE_PENDING |

Local (macOS arm64, software AVX-512 and AMX models, virtual topologies): `make test` rc 0 on 33a7fa4: kernels, GEMM, AMX,
BF16, Q8 C tests 0 failures; `test_runtime` 40 OK; `test_numa` 28 OK (4 skipped); `test_tp` 47 OK; `test_tools` 49 OK
(1 skipped); `test_xengine` 28 OK. The state of the merge commit alone (0fb89ab, textual resolution only) fails the new
tests: a per-channel plan wrote through a NULL KV pointer or divided by zero, the work region was 31 KB short of the BF16 copy.

## 2. Hazards and the tests that cover them

| # | hazard | fix | test (all in `tests/test_tp.py` unless named) | fails on 0fb89ab |
|---|---|---|---|---|
| 1 | persistent BF16 K/V copy allocated outside the planner | `kv.bf16` arenas per tier, `k_attention_set_kv_bf16_layers`, prefix verification, `kv.bf16_copy` in `--json` | `test_planned_bf16_kv_copy_has_arenas_in_both_budgets` (arena per tier and socket, size per layer, node charge = rounded arenas, HBM and DDR budgets refuse a plan that omits the copy, none without `LLAMA_ATTN=amx`); `test_planned_bf16_kv_copy_follows_its_layer_across_tiers` (split 0/1/2 layers, 1S and 2S, head and rows layouts, bitwise against the unplanned run; report carries the copy); `tests/test_tools.py::test_planned_report_requires_the_bf16_kv_copy_to_be_reported` | yes (work region short) |
| 2 | per-channel format 4 (gs 0) | work size counts one scale per row and input | `test_per_channel_checkpoint_is_planned_and_run` (three placements admitted on scalar and tile model, work arena differs from gs 64 by exactly the scale term, planned logits bitwise the unplanned ones, 1S and 2S); `tests/sizing.c` | yes (SIGSEGV on 1S) |
| 3 | gs 32 and gs 64 packed layouts | none needed (shared `pack_bytes_of`); tested | `test_hybrid_packed_tier_with_every_group_size` (gs 64, gs 32, per-channel; `LLAMA_HYBRID_PACKED=hbm` and `ddr`: arena on the requested tier, equal size in both, logits bitwise the unplanned run) | yes (1S) |
| 4 | fused RMSNorm + quantizer with `kv_layer()` | merge keeps both | `test_fused_norm_quantizer_runs_with_a_split_kv_region` (pipelined = chunk = unplanned, split KV, three file types, 1S and 2S; source asserts the two fused calls and the accessor) | yes |
| 5 | raw `key_cache`/`value_cache` | table for every placement | `test_no_raw_kv_indexing_outside_the_kv_table` (section 3) | n/a |
| 6 | interleaved reduction, exchange, prefault, `plan.mode` guards | planned prefault, exchange flag check | `test_planned_split_kv_prefault_and_exchange` (barrier = wo = interleaved bitwise under a split plan and against `hbm-only`; exchange arena size; the 13 guards enumerated) | yes (NULL prefault) |
| 7 | `hbm-ddr.sbatch` forces `LLAMA_KERNELS=avx512`, M = 64 | options, `LEGACY=1` | `test_hbm_ddr_script_follows_engine_defaults` | n/a |

Virtual topologies are `simulated`: their page queries are skipped, so the page-level checks of the BF16 copy are covered by
the hardware reports of job 5916198 (section 5) and by `test_planned_report_requires_the_bf16_kv_copy_to_be_reported`
on a synthetic report.

## 3. Raw KV indexing

```
$ grep -nE '\b(key|value)_cache\b' src/tp.c        # no output
$ grep -n 'kv_layers\[' src/tp.c
  setup_socket:   legacy table (one `l * layer_floats` offset per layer), BF16 slot binding, prefault base
  tp_create_planned: planned table (tier arena + (l or l - cut) * layer bytes), K, V and BF16
  forward_fp32, forward_q8, forward_batch: kv_layer(sk, l) only
```

(`kv_layer(sk, l)` is the only accessor of the forward functions.) `attention.c` receives the layer's pointers and indexes
inside them; it never sees a layer number.

## 4. Bitwise comparison of the default path (job 5916198)

Default environment, 8B, develop b9b9bbb against the merged tree, SHA-256 of the raw logits of the 86-token chat prompt
(probe, last row, all rows batched, per token) and token traces of 64 greedy tokens (`--seed 42 -t 0`):

| config | int8 batch last / all / per token | fp32 batch last / all / per token | token trace int8 / fp32 |
|---|---|---|---|
| 1S | identical / identical / identical | identical / identical / identical | identical / identical |
| 2S | identical / identical / identical | identical / identical / identical | identical / identical |

Raw hashes: `p2-20261124-merge-ab/bitwise/*.sha256`. int8 AMX attention (the BF16 copy) is in the default path.

Benchmark (P2048 + 128 generated, int8, `runq`, 5 interleaved repeats, order alternating; medians [min, max]):

| | TTFT develop | TTFT merged | decode develop | decode merged |
|---|---|---|---|---|
| 1S (bound to socket 0, default placement) | 1.505 s [1.369, 1.522] | 1.386 s [1.358, 1.508] | 13.98 tok/s [13.97, 13.99] | 13.99 tok/s [13.96, 14.00] |
| 2S | 0.753 s [0.742, 0.814] | 0.743 s [0.737, 0.754] | 89.56 tok/s [89.19, 90.23] | 90.13 tok/s [89.56, 90.37] |

The TTFT spreads are the known bimodality of the prefill (`docs/bimodality.md`: 1S 1.36-1.38 against 1.50-1.52 s in both
trees); the merged tree falls in both modes. Decode differs by less than the spread. No default changed.
Raw: `bench/p2048-int8-s{1,2}-{dev,new}-{1..5}.json`, `bench/summary.txt`.

## 5. Planned modes on 8B (job 5916198)

`run` of `runq` int8, 64 greedy tokens, 1S and 2S, reference: the merged tree at default placement. Every row below has an
identical token trace and a passing `check_placement.py --planned` (region inventory equals the plan, every region verified
`full` on its node, BF16 copy reported, worker affinity verified):

`hbm-only`; `ddr`; hybrid with weights HBM layers 32, 24, 16, 8, 0 (KV 16); hybrid with KV HBM layers 32, 24, 8, 1, 0
(weights 16); hybrid packed `hbm`/`ddr` x classifier `hbm`/`ddr` (weights 8, KV 8); hybrid `auto`; hybrid with
`LLAMA_ATTN=tiled` (own reference); FP32 `hbm-only` and `hybrid`; BF16 prefill hybrid with packed `ddr` (2S); the gs 32 and
per-channel files with `hbm-only` and hybrid packed `hbm`/`ddr` (1S and 2S), whose probe logits are also bitwise the develop
logits. 51 traces and 51 placement checks. Raw: `planned/*.json`, `*.placement`, `*.tokens.json`; `results.txt`.

Plan-only admission of the 70B files (derived from the plan JSON in `plan70/`, 2S): gs 64 and gs 32 at context 98304 under
`hybrid` are admitted (HBM 55 of 56 GiB per socket, the BF16 copy included; auto KV layers in HBM 37 and 33, weights 80);
`hbm-only` is refused at context 8192 and 98304 (HBM need 73.4 GiB against 56: the packed copies double the weights, not
the BF16 copy).

## 6. Deviations and what is not verified

1. The two plan-only failures of job 5916198 are in the job script (see section 1), not in the engine.
2. The merge message is reworded after the gate with its job id; hashes after it differ from the tested ones, the sources do not.
3. Not tested on hardware: the 70B run (plan-only only, by instruction), `LLAMA_THP=on` with the BF16 copy, 256 or more layers.
4. The BF16 copy is zeroed whole at setup (also under a plan), so every page of every arena exists before the first prompt:
   a 70B hybrid plan therefore faults the DDR tier's full copy; this matches the legacy two-socket prefault and is not a
   change of the default path.
