# Staggered Q8 decode GEMV: validation record

Companion of [gemv-stagger.md](gemv-stagger.md). Jobs, commits, files and gates. No host names or cluster paths are given; a "results directory" is the per-job directory the
job script creates on the cluster's results file system. All jobs ran `scripts/gemv-stagger.sbatch` exclusively on one Flat node of the HBM partition (the same node,
"node A", for every job; the other node was used by other work), from a staged copy of the tracked files of the commit named (a `COMMIT` file in the staged tree holds the
full hash; the job writes the sha256 of every source file into its results directory).

## 1. Commits

| commit | content |
|---|---|
| a6d642e | `LLAMA_GEMV_STAGGER`, unit test, script, summary tool |
| 6e4e376, 6a2454e, f5327eb | script fixes only (probe make targets, gate under one-socket binding, plan-only for the planned configuration only) |
| 2c8bc12 | `LLAMA_THP=never` (`MADV_NOHUGEPAGE`) and the cells that use it: the engine code of the final tree |
| the commit of this document | this document and `docs/configuration.md` text (documentation only) |

The engine code (`src/`, `include/`) of every A/B and gate job after the first is the code of 2c8bc12; the gate and bitwise job of the final tree is job 5934865.

## 2. Jobs

| job | stage | commit | result | elapsed |
|---|---|---|---|---|
| 5929336 | check (first try) | 6e4e376 | failed by script error: `make test` ran without the one-socket binding the campaign gate uses (84 failures all `LLAMA_SOCKETS unset: both sockets ...`), and the reference build target was misnamed | 0:41 |
| 5929381 | check: gate and logits | 6a2454e | `make test` rc 0; logits identical (section 3) | 0:59 |
| 5933865 | ab 8B 1S | f5327eb | failed: `--plan-only` needs a planned placement (script error); 5933866, 5933867 cancelled before use | 0:01 |
| 5933868 | ab 8B 1S, cells THP unset / on x knob | f5327eb | 20 runs + 4 profiled, 24 token traces identical; both THP cells were 2 MiB pages (section 4 of gemv-stagger.md) | 0:05 |
| 5934862 | ab 8B 1S, cells THP unset / never x knob | 2c8bc12 | 24 token traces identical | 0:05 |
| 5934863 | ab 8B 2S, same cells | 2c8bc12 | 24 token traces identical | 0:05 |
| 5934864 | ab 70B 2S (hybrid, all HBM), same cells | 2c8bc12 | 24 token traces identical | 0:37 |
| 5934865 | check: gate and logits of the final engine code | 2c8bc12 | `make test` rc 0; logits identical (1 distinct digest of 8, both configurations) | 1:00 |

Jobs 5933869 and 5933870 (the follow-ups of the first chain) were cancelled before they ran. Node time used: 3.6 node-hours against the 12 allowed.

Setup of every A/B job: `LLAMA_BACKEND=teams LLAMA_THREADS=56 LLAMA_PLACEMENT_STRICT=1 LLAMA_VERIFY_SAMPLES=0 LLAMA_KV_VERIFY=1`, explicit topology `0/2/0,1/3/1`; 8B one socket
`LLAMA_PLACEMENT=hbm LLAMA_SOCKETS=1` under `numactl --cpunodebind=0`; 8B two sockets `LLAMA_PLACEMENT=hbm LLAMA_SOCKETS=2`; 70B gs64 `LLAMA_PLACEMENT=hybrid LLAMA_SOCKETS=2
LLAMA_HYBRID_WEIGHTS_HBM_LAYERS=80 LLAMA_HYBRID_KV_HBM_LAYERS=80 LLAMA_HYBRID_CLASSIFIER=hbm LLAMA_HYBRID_PACKED=ddr` (the P2 note of the hybrid-memory campaign ledger: the
packed copy does not fit HBM; per-socket HBM use 39.4 GB of 60.1 GB, plan check passed). Engine defaults otherwise (AMX for prefill, `LLAMA_KERNELS=auto`; decode GEMV is
the VNNI kernel). Command: `--benchmark --chat-template --no-text -t 0 -n 0 --context-length (4096 + 256 + extra) --warm-context 4096 --generation-length 256 (8B) or 128 (70B)`
with the prime-number prompt of `scripts/hbm-ddr.sbatch`. Every `LLAMA_*` variable of the caller is cleared before a cell. Each repetition runs the four cells in a
rotated order (repetition k starts with cell k mod 4). After the repetitions one run of each cell with `LLAMA_PROFILE=1`.

## 3. Bitwise checks

* **Unit test** `test_gemv_stagger` (`tests/kernels.c`): for gs 32 and 64, d in {1, 3, 4, 5, 13, 100, 257}, both W8A8 and W8A32 inputs, every team size of the test
  matrix (virtual topology teams), `k_matmul_q8`, `_add`, `_qkv`, `_swiglu` and `_ld` (PARTIAL) with the knob on are `memcmp`-equal to knob off; the worker entry point
  `k_matmul_q8_ld_worker` run serially for 1, 5 and 9 workers equals the team result, and every row is reported to the rows-done callback exactly once.
* **Logits** (job 5929381, tree 6a2454e; job 5934865, tree 2c8bc12): `probe-int8` prints the raw float32 logits at every position of a fixed 97-token input (BOS and ids
  1009 + 1301 k); the sha256 of the output is the same for 8 variants per configuration: the develop binary (8fec0c8, no knob), and the tree with knob off and on, each with
  `LLAMA_THP` unset and `never` (the first job used unset and `on`). 8B one socket: 1 distinct digest of 8; 8B two sockets: 1 distinct digest of 8.
* **Token traces** (`LLAMA_TOKENS_OUT`) of every benchmark run (24 per job, all four cells, five repetitions and the profiled runs): identical in all four jobs including 70B
  (`tokens-identical.txt`).

## 4. Gates

* Local (macOS arm64, software AVX-512 and AMX models, virtual topologies): `make test` rc 0 on 2c8bc12 (the same run on a6d642e: rc 0).
* Cluster: `make test` rc 0 on 6a2454e (job 5929381), under `numactl --cpunodebind=0 --membind=0` with the engine knobs unset as the campaign check stage runs it; on the
  final engine code, job 5934865: `make test` rc 0.
* The HF logit gate was not run: the arithmetic is unchanged (the bitwise checks above are stronger than the gate, which the knob cannot move).

## 5. Raw files

Each results directory holds `environment.txt`, `source.sha256`, `bin.build.log`, for each run `rK-thp=T-stagger=S.{json,tokens.json,stdout,stderr,launch,end}` and
`prof-thp=T-stagger=S.json`, and `tokens-identical.txt`; the check jobs hold `make-test.log`, `make-test.rc` and `logits-CONFIG-{base,tree}-thpT-staggerS.sha256`. The tables of
the results section are `tools/gemv_stagger_summary.py` over the A/B directories.

## 6. Per-repetition values (tok/s, measured)

Repetition order 1 to 5; cells are `THP / knob` (THP `off` is the engine default, 2 MiB pages on this node).

| config | off / 0 | off / 1 | never / 0 | never / 1 |
|---|---|---|---|---|
| 8B 1S (5934862) | 42.34 42.21 42.41 42.18 41.99 | 43.97 44.40 44.16 44.33 44.31 | 43.32 43.92 44.05 43.88 43.85 | 43.75 43.85 43.84 43.81 43.66 |
| 8B 2S (5934863) | 74.45 74.87 73.53 72.26 73.25 | 76.15 75.61 74.95 74.45 74.92 | 74.88 74.72 74.81 74.87 74.12 | 74.30 75.31 74.43 74.76 74.35 |
| 70B 2S (5934864) | 10.02 10.02 9.98 10.00 9.99 | 10.60 10.62 10.60 10.60 10.55 | 10.50 10.51 10.47 10.49 10.48 | 10.54 10.53 10.53 10.50 10.48 |

Decode profile of the 4 KiB cells (ms per token): 8B one socket W1/W3 8.087 and 8.132, W2 3.967 and 3.983 (knob off and on); 8B two sockets W1/W3 4.082 and 4.097, W2
2.441 and 2.356; 70B W1/W3 40.324 and 40.627, W2 21.134 and 20.842: knob and page size give the same phase times once the aliasing is gone.
