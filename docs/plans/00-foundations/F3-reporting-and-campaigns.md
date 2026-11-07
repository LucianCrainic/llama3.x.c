# F3. Reporting schema v2 and campaign infrastructure

## 1. Goal and status

This phase delivers the evidence plumbing that every later campaign needs:

- **Report schema v2.** The JSON report gains placement, team and prefill fields.
- **Shared Slurm library.** Common steps move out of the existing Flat-mode script into
  one library, used by three per-track scripts.
- **Engine-vs-engine comparison.** A logit comparison tool between two engine
  configurations.
- **Generalized summarizer.**

Each phase then adds only the run matrix for its own experiments.

Status: **implemented; cluster check run; `make test` green on Linux (acceptance comparison with the kernels script not done)**.

Implementation notes:

- **Schema v2** (`src/report.c`): `schema_version` is 2 and the report carries `runtime`,
  `topology` (+ `topology_error`), `teams`, `regions`, `kv`, `phase_seconds`, `prefill`, and
  `timing_seconds.{weight_pack,placement_populate,placement_verify}`. Producers that exist
  fill them (backend, THP, precision mode, A1 topology, F2 teams with pinning read-back,
  A2 regions and, once a verification is recorded, A3 evidence); the rest are `null`.
  `placement` is always `file` until A4 adds `LLAMA_PLACEMENT`. `teams` is `null` for the
  `omp` backend, `regions` is `[]` when nothing was reserved, and a topology that cannot be
  discovered is `null` with the message in `topology_error` (never fatal).
- **Discovery before pinning.** `kernels_backend_init` runs `numa_topology_try` first, so the
  cached topology reflects the job's full allowed sets even though the pool pins the leader;
  `TeamSummary` and the region history survive `pool_stop` / `engine_free` for the report,
  which is written from an `atexit` handler.
- **Tools.** `tools/compare_engines.py` (raw probe under env A and B, every position, JSON
  like `validate.py`'s `comparisons[]`, non-zero exit over budget; numpy optional),
  `tools/summarize_campaign.py` (v2 labels and fields, median and full range, tolerant of
  v1 reports) and `tools/make_prompts.py` (fixed-length prompts from the Dickens opening
  paragraph, counts and SHA-256 in `prompts.json`). `tests/probe.c` accepts `@file` text
  because 32768 tokens do not fit one argv string on Linux.
- **Scripts.** `scripts/lib/cresco8-common.sh` provides the functions listed above, and
  `cresco8-{placement,tp,prefill}.sbatch` source it through `$ROOT` (Slurm runs a copy of
  the script, so its own path is useless). `scripts/cresco8-kernels.sbatch` is left
  untouched for old campaigns, so the library is a copy of its steps, not a refactor of it;
  the cluster `check` is what shows the outputs are identical. Run labels come from
  `run_label` and are parsed by `summarize_campaign.py`. Memory policy is only imposed when
  `MEM_BIND` is set, so engine-owned placement runs can leave it unset.
- **Local tests.** v2 keys and `null`s, the error-path JSON, `compare_engines` with identical
  and perturbed environments, the summarizer on a synthetic directory, prompt lengths and
  reproducibility, `bash -n`, and the library's environment checks. `shellcheck` was not
  installed locally, so that test is skipped.

## 2. Depends on / unblocks

- Depends on: nothing. Fields are added as their producers land, and are `null` until
  then.
- Unblocks: every `check` and `bench` run (A1–A6, B1–B7, C1–C7, D1–D2).

## 3. Reuse

| Code | Location | Use |
| --- | --- | --- |
| `write_report` | `src/report.c` (schema at :101) | JSON writer; `schema_version: 1` |
| Linux affinity and NUMA nodes | `src/report.c:156-186` | Extend instead of duplicating |
| `MemoryKind` names | `src/report.c:265` `names[]`, `include/runtime.h:34-43` | Positional; must grow in step with the enum |
| Flat-mode campaign | `scripts/cresco8-kernels.sbatch` | Flat check :54-67; build and test :70-78; perf uncore events :81-84; validation gate :90-112; workload :116-136; summary :177 |
| Summarizer | `tools/summarize_kernels.py` | Label regex, GB/s model, perf CSV columns |
| Probe | `tests/probe.c` | Raw float32 logits on stdout, exit 3 on a leak |
| Real-model validation | `tools/validate.py` | HF comparison and affinity capture |

## 4. Changes

### 4.1 Report schema v2 (`src/report.c`)

New top-level objects. A producer that does not exist yet writes `null`.

```json
"runtime": {
  "backend": "omp|teams", "sockets": 1, "placement": "file|ddr|hbm",
  "thp": "off|on", "precision_mode": "fp32|bf16|int8_q8_0|int8_q8_0_w8a32",
  "prefill_batch": 1, "prefill_logits": "last|all", "kernels_prefill": "avx512|amx"
},
"topology": { "sockets": [ { "cpus": [..], "ddr_node": 0, "hbm_node": 2,
                             "classification": "numaperf|explicit|virtual" } ],
              "mems_allowed": [0,1,2,3], "cpus_allowed": [..] },
"teams": [ { "index": 0, "threads": 56, "cpus": [..], "affinity_verified": true } ],
"regions": [ { "role": "weights.layer|weights.classifier|embedding|kv.k|kv.v|work",
               "socket": 0, "node": 2, "bytes": 0, "thp_bytes": 0,
               "verified": "full|sampled|unverified", "pages_queried": 0,
               "pages_mismatched": 0, "pages_missing": 0 } ],
"kv": { "reserved_bytes": 0, "resident_bytes": 0, "active_bytes": 0 },
"phase_seconds": { "socket0": { "qkv":0, "attention":0, "wo":0, "ffn":0, "classifier":0,
                                "reduce":0, "barrier_wait":0 }, "socket1": {..} },
"prefill": { "microbatches": 0, "tokens": 0, "seconds": 0, "logits_rows": 0 },
"timing_seconds": { ..., "weight_pack": 0, "placement_populate": 0, "placement_verify": 0 }
```

Rules for the new fields:

- **Schema version.** `schema_version` becomes `2`. The `test_runtime.py:51` assertion
  and `docs/benchmark-example.json` are updated, and `docs/phase1.md` gets a "Schema v2
  additions" table.
- **Phase timing cost.** `phase_seconds` is filled only when `LLAMA_PROFILE=1`. Each
  socket leader times its own work, so the instrumentation costs one
  `monotonic_seconds` pair per phase. Socket rates are never added together (study
  §10.3).
- **Memory accounting.** `memory_bytes` gains `owned_weights_peak` and
  `packed_weights_peak`, and keeps `mapped_checkpoint` separate (study §10.3).

### 4.2 Campaign scripts

**`scripts/lib/cresco8-common.sh`** is sourced by each sbatch script. It provides
functions extracted unchanged from `cresco8-kernels.sbatch`:

| Function | Purpose |
| --- | --- |
| `require_env` | Requires `ROOT`, `OUT` (must not exist yet), `PYTHON`, `MODELS` |
| `record_environment` | lscpu, `numactl -H`, scontrol, commit, and `source-*.sha256` |
| `check_flat_mode` | Writes `flat-mode.json`; fails if not Flat |
| `build_tree` | `make CC=gcc OPENMP=$1 MODE=$2` |
| `perf_available` / `perf_attach` | Uncore events `ddr_cas_rd`, `hbm_cas_rd`, `unc_upi_rxl_flits.all_data` |
| `stage_checkpoint` | `/dev/shm` staging under a numactl policy; kept for baselines |
| `run_engine` | Runs one label with env, `--json`, perf and exit status |
| `validate_gate` | Runs `tools/validate.py`; FP32 failure is fatal, Q8 is recorded |
| `make_prompt` | `short.txt` (86 tokens) and fixed-length prompts of 512, 2048, 8192 and 32768 tokens, built deterministically from a repeated public-domain paragraph and truncated by the tokenizer |

**Three skeleton scripts** share the same `#SBATCH` header as the kernels campaign:
112 CPUs, `--exclusive`, `--mem=0`, and partition and nodelist passed at submission:

- `scripts/cresco8-placement.sbatch`
- `scripts/cresco8-tp.sbatch`
- `scripts/cresco8-prefill.sbatch`

Each script accepts `STAGE=check|bench`, `REPEATS`, `PRECISIONS` and a phase selector
`PHASE=A3|A6|...`, which picks the run matrix described in that phase's document.

**Run labels** extend the existing scheme with a short key for each knob:

```
{model}-{precision}-{backend}-{placement}-s{sockets}-t{threads}[-m{M}][-p{prompt}][-c{ctx}][-k{kernel}]-{rep}.json
```

### 4.3 Tools

- **`tools/compare_engines.py`.** Runs `tests/probe.c` twice, under env A and env B,
  on the same checkpoint and prompt tokens. It compares logits at **all** positions
  (A uses `LLAMA_PREFILL_LOGITS=all` once C1 exists) and reports max_abs, rms, max_rel
  and argmax agreement per position. It writes JSON like `validate.py`'s
  `comparisons[]` and exits non-zero when it is over budget. B5, B6, C1–C6 and D1 use
  it for C-vs-C gates.
- **`tools/summarize_campaign.py`**, a generalization of `summarize_kernels.py` (which
  stays for old campaigns). It parses the v2 labels and fields and adds these columns:
  sockets, placement plus verification status, THP, M, kernel, precision, prompt
  length, TTFT, prefill tok/s, decode tok/s, a per-phase breakdown (when profiled),
  reduce/wait share, and the HBM/DDR/UPI counters. It reports the median and the full
  range across repeats.

## 5. Local tests

- `test_runtime.py` checks the v2 schema keys, that unproduced fields are `null`, and
  the JSON validity of every error path.
- `test_tools.py`:
  - `compare_engines.py` runs on the fixtures with identical env (expects zero
    difference) and with a perturbed env (expects a non-zero difference to be
    reported).
  - `summarize_campaign.py` runs on a synthetic directory of v2 reports.
- `bash -n` and `shellcheck` (if available) on the library and the three skeletons.

## 6. Cluster run

`STAGE=check` of each skeleton with `PHASE=none`. It builds, runs the Flat-mode check,
records the environment and runs `make test`, with no benchmarks. **Estimated** at
15 minutes. It needs the user's go-ahead to submit.

Acceptance: the outputs (`environment.txt`, `flat-mode.json`, `test.log`) are identical
in content to a `cresco8-kernels.sbatch STAGE=check` run of the same commit.

### Cluster results (2026-11-01)

The library, `check_stage`, the Flat-mode check, `make test` and the F2/A phases ran in jobs 5622314 and 5622320 on
cresco8-hbm14 (`environment.txt`, `flat-mode.json`, `test.log`, `phase-results.txt`, `build-config-*.txt` per job);
the Flat-mode check passed and the v2 reports of the 24 benchmark runs were summarized by `summarize_campaign.py`
(`F2/summary.md`). `PHASE` lists, `expect`, `HF_GATE` and the optional `MODELS` came out of these runs. Not done: the
acceptance comparison with a `cresco8-kernels.sbatch STAGE=check` run of the same commit. It is not a criterion of
milestone M1 and stays open. `make test` is green on Linux after the last test fix (jobs 5622322, commit `bc7da43`,
and 5693907, commit `4385a8a`: `phase-results.txt` records `PASS make-test`; the second job's `test.log` ends with
the Python tool tests 22 OK, 1 skipped). Job 5693907 also added `GATE_ENV`/`GATE_TAG` to `validate_models`, so the
Hugging Face gate can run the engine under another backend environment.
## 7. Exit criteria

- Schema v2 is emitted and the v1 consumers are updated.
- The script library reproduces the existing Flat-mode checks without behaviour
  changes.
- `compare_engines.py` and `summarize_campaign.py` are tested locally.

## 8. Risks and open questions

- **Perf availability.** `sudo -n perf` works on the Flat node (hw-bench
  `docs/06_COUNTERS.md`) but may not elsewhere. The library must treat perf as optional
  and record whether it ran.
- **Prompt construction for long contexts.** Prompts must be reproducible and avoid
  copyrighted text. Generate them from a fixed public-domain or synthetic source,
  record the token counts, and check them in the report.

## 9. Not in this phase

- Any placement or kernel change.
- Phase-specific run matrices. Each phase defines its own.
