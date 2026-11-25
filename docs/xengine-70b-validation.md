# 70B cross-engine comparison: validation record

**Status: carried over, not re-run.** Provenance of [xengine-70b.md](xengine-70b.md) and of the harness ported with it (`tools/xengine70/`, `scripts/xengine70.sbatch`, the `xengine70-fp32` and `xengine70-q8` targets of the Makefile). The measurements were made on an older tree; the port ran no cluster job (0 node-hours).

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

## Staleness

The llama3.x.c Q8_0 rows (TTFT 84 s and 326 s, decode 4.5-5.3 tok/s) are STALE: they ran the portable group-32 kernels, before the gs32 AMX-INT8 prefill and VNNI decode work ([gs32-amx.md](gs32-amx.md): 8B decode 24.6 to 48.0 tok/s on one socket, prefill many times faster). The FP32 rows used the then-current kernels and are also older than the prefill and Wo/W2 work. The llama.cpp and vLLM rows do not depend on this tree.

## Checks made for the port (this tree, no cluster)

| Check | Result |
| --- | --- |
| The llama3.x.c commit hash of the earlier work (it does not exist in this tree) replaced by "pre-port revision" plus the job ids | done |
| Banner, STALE marks, corrected sentence on the portable matmuls | done |
| `make xengine70-fp32 xengine70-q8` | built as part of `make test` (macOS, no AVX-512; the drivers use only the public engine API) |
| The harness script: engine defaults no longer forced | `tests/test_xengine70.py`; no job submitted |
| Prompt fixtures (547 and 2,113 ids, SHA-256 as in the document), `tools/xengine70/summarize.py` on synthetic request files, the native driver on a tiny model (four requests per workload, identical greedy output across repeats), the script's syntax and that it forces no engine default | `tests/test_xengine70.py` |

## Not validated

- The harness has not been run on this tree. Run `c-q8` on the same node first; compare the output IDs with the 128-token traces of the earlier jobs.
- A re-run on this tree will use the current defaults (AMX prefill, VNNI decode, fused reductions); it will not reproduce the stale rows, and that is the point.
