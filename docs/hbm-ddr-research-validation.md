# Hybrid HBM and DDR for 70B: validation record

**Status: carried over, not re-run.** Provenance of [hbm-ddr-research.md](hbm-ddr-research.md), and what was and was not checked when the earlier 70B comparison work was ported into this tree. Every measurement below was made on an older tree (before the prefill, Wo/W2 and gs32/gs64 kernel work); no cluster job was run for the port (0 node-hours). Raw data stays outside the repository.

## Provenance

| Item | Value |
| --- | --- |
| Hardware | CRESCO8 Flat/Quadrant nodes `cresco8-hbm13` and `cresco8-hbm14`, 2 x Xeon Max 9480, 56 GiB HBM budget per socket |
| Model | Llama 3.1 70B Instruct, NousResearch mirror `d50656ee28e2c2906d317cbbb6fcb55eb4055a84`, v3 Q8_0 with 64-element groups (export job 5892714, SHA-256 in the document) |
| Engine | `1bc9189` (the placement branch, an ancestor of this tree) for the pilot; the "pre-port revision" of each later job is the earlier 70B comparison work as it stood at that job. Its hashes do not exist in this tree, so the document names the job instead |
| Raw data | `hbm-ddr-pilots/followup-20261122/70b` and `results/hbm-ddr-research-20261123` in cluster storage; the analysis tools accept both layouts |

## Jobs

| Job | Node | What it measured | Section of the document |
| --- | --- | --- | --- |
| 5892711 | hbm13 | 8B serial-model check, 4 GiB raw read streams | Original pilot results |
| 5892714 | | 70B checkpoint export | Original pilot results |
| 5892715 | hbm13 | 70B pilot: HBM-only, hybrid 40/40, DDR-only (`1bc9189`) | Original pilot results |
| 5904620 | hbm14 | Baseline repeats, rotated order, weight and KV spill cells | Matched 70B replication |
| 5904621 | hbm14 | 8,192 warmed positions, 95 decode steps, different reservations and tier policies | Fixed active history, different reservations and tier policies |
| 5904622 | second Flat node | AMX pair, three repeats, 2,113-token prompt | AMX packs and reservation pilot |
| 5904726 | | Serial and concurrent local-pair read control; decode attention kernel microbenchmark | Controlled concurrency and the decode attention kernel |
| 5904727 | | Profiled hybrid run that exposed the warm-up exchange-clock bug | Matched 70B replication |
| 5904755 | | The same request after the fix (about 10.9 ms of peer wait) | Matched 70B replication |
| 5904742 | | 10 ms CAS-read windows, 40/40 placement | Sub-token controller activity |
| 5904744 | hbm13 | Real 73,728-position warm-up, weight-first plan | Real workload beyond the practical HBM budget |
| 5904751 | | First unprimed synthetic comparison (rejected: 3.93 against 9.36 tok/s) | Startup pause and proxy rejection |
| 5904757 | | 0 and 1,024 primer positions with the same synthetic prefix | Startup pause and proxy rejection |
| 5904760 | hbm14 | Primed synthetic validation at 73,728 and 8,192 positions | Validated capacity frontier |
| 5904761 | | Grouped decode attention on the primed setup | Validated capacity frontier; Real grouped-attention validation |
| 5904763 | hbm13 | Real 73,728 positions, split against grouped attention, same executable as 5904744 | Real grouped-attention validation |
| 5904765 | | 130,944 primed synthetic positions, 131,072 reservation | Near the full 128K reservation |

The figures [decode](figures/hbm-ddr-decode.svg), [windows](figures/hbm-ddr-windows.svg) and [capacity](figures/hbm-ddr-capacity.svg) are rendered from the raw directories by `tools/plot_hbm_ddr.py`; they were not regenerated for the port.

## Checks made for the port (this tree, no cluster)

| Check | Result |
| --- | --- |
| Every research commit hash that does not exist in this tree was replaced by "pre-port revision" and the job id | done; `1bc9189` and `5c0dd56` are in this tree's history and stay |
| Banner and metric labels (measured, modelled, derived) added; no number changed | done |
| The tools run on this tree: `tools/hbm_ddr_results.py` and `tools/plot_hbm_ddr.py` import `tools/xengine_summary.py`, whose event names gained the CAS write, context-switch and migration counters | `tests/test_hbm_ddr.py`: phase windows, joint tier bins, low-read spans, the logical bytes per step of the 70B request (73.856 GB of weights and 0.400 GB of KV, as in the document), the new perf event names, the worker sampler on a fake task directory; the plot tool is only compiled (it needs matplotlib) |
| The research script on this tree: `DRY_RUN=1` environments in `tests/test_hbm_ddr.py` (engine defaults not forced, `LEGACY=1` and explicit options, missing options refused) | pass; no job submitted |
| Synthetic KV on this tree (rewritten to go through `kv_layer()`) | `tests/test_tp.py` (needs a `SYNTHETIC_KV=1` build, skipped otherwise): active-prefix KV accounting on hybrid, DDR and HBM-only plans, primer 0/3/13, invalid knobs, a stop token forwarded as an ordinary input; and bitwise independence of the prefill after the fill from the primer length under `LLAMA_ATTN=amx` (emulated AMX, one and two sockets, legacy and split-tier KV). With the invalidation removed that test fails, so it covers the BF16 copy; `tests/test_tools.py` covers the refusal in `compare_generation.py` |

## Not validated

- No number in the document was re-measured or recounted from raw data; they are the earlier work's figures.
- `stream-bench --local-pair` (job 5904726) was not ported: the merged [concurrent-tier bandwidth study](concurrent-tier-bandwidth.md) has a superseding research mode in `tests/stream_bench.c`. That control cannot be reproduced with this script.
- The synthetic KV fill was reimplemented for the planned placements of this tree. On hardware it is not yet compared against a real warm-up: run a primed synthetic against a real fill on the 8B model first (decode rate within 3%, controller read volume within 5%, identical token traces and placement), as the document's proxy rule requires.
- On the current tree the default kernels differ from those of every job above (AMX gs64 prefill, fused Wo/W2 reduction). Use `LEGACY=1` of the script to approximate the old configuration; it does not recreate the old kernels.
