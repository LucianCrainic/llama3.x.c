# 70B cross-engine comparison: validation record

**Status: carried over; the llama3.x.c Q8 DDR rows were re-run on `ee24659` (jobs 5929406, 5929407) and HBM rows (job 5935580, a different configuration; see "Re-run on the current tree"), everything else is not re-run.** Provenance of [xengine-70b.md](xengine-70b.md) and of the harness ported with it (`tools/xengine70/`, `scripts/xengine70.sbatch`, the `xengine70-fp32` and `xengine70-q8` targets of the Makefile). The measurements were made on an older tree; the port ran no cluster job (0 node-hours).

## Provenance

| Item | Value |
| --- | --- |
| Hardware | CRESCO8 Flat/Quadrant node `cresco8-hbm14` for every result job, 2 x Xeon Max 9480, 55 cores per socket available to each engine |
| Model | Llama 3.1 70B Instruct, NousResearch mirror `d50656ee28e2c2906d317cbbb6fcb55eb4055a84`; FP32 widened exactly from BF16 |
| Engines | llama3.x.c: the pre-port revision of the jobs below (hybrid research tree on `1bc9189`); llama.cpp `b1b770d90d94cad9e9ea5b993bdbc383a522fb28`; vLLM `0.34.0+cpu` from the pinned image |
| Requests | 547 and 2,113 token IDs (SHA-256 in the document), 128 greedy tokens, context 8,192, batch 256, one warm-up and three measured requests |

## Jobs

| Job | What | Result in the document |
| --- | --- | --- |
| 5910127 | FP32 v3 checkpoint conversion | input |
| 5910128 | GGUF conversion (Q8_0 and F32) | input |
| 5910181 | Q8_0 repack to the C format | input |
| 5910133 | llama3.x.c FP32 on DDR | FP32 table |
| 5910177 | vLLM FP32 on DDR (stopped during container shutdown after all eight traces were written) | FP32 table |
| 5912354 | llama.cpp FP32 on DDR | FP32 table |
| 5910211 | Q8_0 DDR cells for both engines and the C HBM cell; reached its time limit in the llama.cpp HBM cell | Q8_0 table |
| 5913407 | llama.cpp Q8_0 HBM cell, repeated on the same node | Q8_0 table |

## Re-run on the current tree

| Item | Value |
| --- | --- |
| Tree | `ee24659` (develop), `scripts/xengine70.sbatch` modes `c-q8` and `c-q8-hbm`, engine defaults, no `LEGACY` (`LLAMA_KERNELS=auto`, AMX prefill) |
| Node | `cresco8-hbm14`, exclusive, after the gate job 5929402 |
| Jobs | 5929406 (gs32 checkpoint, the file of the old rows): DDR cell completed, `hbm-only` cell failed at startup; 5929407 (native gs64 checkpoint): same. 6 min 1 s and 4 min 48 s. 5935580: hybrid HBM cell for both files, 7 min 41 s. About 0.31 node-hours in all |
| Workloads | the 547- and 2,113-token prompts of the earlier jobs (same SHA-256), 128 greedy tokens, context 8,192, batch 256, one warm-up and three measured requests, two sockets of 55 threads |
| DDR gs32 | TTFT 2.689 s and 10.222 s, decode 5.928 and 5.755 tok/s (old 84.236 s, 326.208 s, 5.068, 4.489) |
| DDR gs64 | TTFT 2.154 s and 8.065 s, decode 6.216 and 6.011 tok/s (no earlier value) |
| Token check | gs32: all 8 requests (both workloads, warm-up included) equal the 128 IDs of job 5910211 and each other, a bitwise match across the kernel change; gs64: long equal to it, short diverges at position 3 (zero-based) from the gs32 trace (different checkpoint) |
| `hbm-only` cell | not runnable on this tree: strict `hbm-only` needs 83.0 GB (gs32) or 78.6 GB (gs64) per socket because the AMX packed prefill copy must also be held in HBM, 22.9 GB and 18.5 GB over the 56 GiB per-socket budget; the old tree had no packed copy (43 GB per socket) |
| HBM weights+KV, packed prefill copy in DDR (hybrid 80/80), job 5935580 | `LLAMA_PLACEMENT=hybrid LLAMA_HYBRID_WEIGHTS_HBM_LAYERS=80 LLAMA_HYBRID_KV_HBM_LAYERS=80 LLAMA_HYBRID_PACKED=ddr`, strict. Plan-only checked before submission (virtual 2 x 55 topology, AMX model, header of the real file): HBM 43.25 GB per socket (gs32), 41.07 GB (gs64), within 60.13 GB; the run log's plan matches (43,247,030,272 and 43,246,514,176 bytes). gs32 TTFT 2.598 s and 9.917 s, decode 10.966 and 10.120 tok/s (old, `hbm-only`, portable kernels: 84.484 s, 326.997 s, 5.343, 5.290); gs64 TTFT 2.124 s and 8.009 s, decode 11.583 and 10.709 tok/s. Token IDs: all requests equal the DDR run of the same checkpoint; gs32 equals the old job's 128 IDs on both workloads. `scripts/xengine70.sbatch` exports `LLAMA_PLACEMENT` itself (`hbm-only` for `c-q8-hbm`), so the cell was run through a wrapper that repeats the script's `c-q8-hbm` branch (same build, driver, prompts, `taskset`, batch and threads) with the hybrid knobs; the script needs an overridable `placement` to take them |
| Raw results | cluster result directory `xengine70-20261125-cq8` (cells `gs32-c-q8`, `gs32-c-q8-hbm`, `gs64-c-q8`, `gs64-c-q8-hbm`, each with `runs.jsonl`, `engine.log`, `environment.txt`, `command.sh`); not in the repository |

## Staleness

The llama3.x.c Q8_0 rows were replaced by the re-run above (the HBM rows with a different configuration). The superseded values (TTFT 84 s and 326-327 s, decode 4.5-5.3 tok/s) ran the portable group-32 kernels, before the gs32 AMX-INT8 prefill and VNNI decode work ([gs32-amx.md](gs32-amx.md): 8B decode 24.6 to 48.0 tok/s on one socket, prefill many times faster). The FP32 rows used the then-current kernels and are also older than the prefill and Wo/W2 work. The llama.cpp and vLLM rows do not depend on this tree.

## Checks made for the port (this tree, no cluster)

| Check | Result |
| --- | --- |
| The llama3.x.c commit hash of the earlier work (it does not exist in this tree) replaced by "pre-port revision" plus the job ids | done |
| Banner, STALE marks, corrected sentence on the portable matmuls | done |
| `make xengine70-fp32 xengine70-q8` | built as part of `make test` (macOS, no AVX-512; the drivers use only the public engine API) |
| The harness script: engine defaults no longer forced | `tests/test_xengine70.py`; no job submitted |
| Prompt fixtures (547 and 2,113 ids, SHA-256 as in the document), `tools/xengine70/summarize.py` on synthetic request files, the native driver on a tiny model (four requests per workload, identical greedy output across repeats), the script's syntax and that it forces no engine default | `tests/test_xengine70.py` |

## Not validated

- The old `hbm-only` configuration of the C engine HBM cell on the current tree: it does not fit the per-socket HBM budget (see the re-run table). The new HBM rows are the hybrid 80/80 configuration, not a like-for-like repeat.
- llama.cpp and vLLM were not re-run.
