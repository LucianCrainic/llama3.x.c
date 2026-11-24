# Hybrid memory campaign: ledger

Standing instructions: [`docs/plans/hybrid-memory-manager-prompt.md`](../../plans/hybrid-memory-manager-prompt.md)
(local, not committed). This file is the manager's memory across sessions. Started 24 November 2026 from `develop` at
`b9b9bbb`.

**Resume:** read this file, `git fetch`, `git log develop`, `squeue -u $USER`, then continue from **Next action**.

## Next action

P1 engineer, P0 recon (inventory, docs), P2 merge and P2b inventory agents launched 24 Nov. Wait for their returns;
verify P1 (§8) and run the independent review before G1; start downloads and the 70B gs64 conversion once P0 reports
internet access and existing checkpoints; send the P2b table to the owner for approval.

## Compute limits (D2, measured 24 Nov)

| Item | Value | Source |
| --- | --- | --- |
| Per-job wall time, `cresco8_hbm` | 24 h (`MaxTime=1-00:00:00`, QOS `cresco8_hbm_h24`) | `scontrol show partition`, `sacctmgr show qos` |
| Per-user node limit, `cresco8_hbm` | 5 nodes (QOS `MaxTRESPU node=5`), partition has 14 nodes `hbm[01-14]` | `sacctmgr show qos` |
| Debug partition `cresco8_hbm_dbg` | 1 h, 2 nodes (`hbm[15-16]`), memory mode unknown | `scontrol show partition` |
| Account allocation | **no Slurm cap**: association `enea`, QOS `normal`, empty `GrpTRESMins`; fair share 0.053 (low priority, not a limit). Site tools found are Lustre quota only | `sacctmgr show assoc`, `sshare -U -l`, `/usr/local/bin` |
| Conclusion | The "24 hours" is the per-job wall time, not an allocation. Plan every job under 24 h | |
| Node-hours used before the campaign | 124.2 (1-24 Nov, all partitions) | `sacct -X`, elapsed x nodes |

## Phases

| Phase | Status | Branch | Agent | Jobs | Node-h |
| --- | --- | --- | --- | --- | --- |
| P0 recon | done (reference page: `reference-numbers.md`, spot-checked) | none | haiku x2 | 5916176-78 (debug, ~0.05) | ~0.05 |
| P1 concurrent bandwidth | in work | `perf/concurrent-tier-bandwidth` | sonnet | | |
| G1 decision | open | | manager | | |
| P2 merge placement | in work | `merge/hbm-ddr-placement` | sonnet | | |
| P2b 70B port | approved; porting starts once P2 is merged into develop | `feature/70b-comparison` | sonnet | | |
| Downloads / 70B gs64 conversion | **not needed**: 70B gs64 and gs32 engine files exist (headers checked by manager); sha256 manifest job in work (cpu partition) | | haiku | | |
| P3 concurrent placement | waiting on G1 | `feature/concurrent-tiers` | | | |
| P4 limits + baseline | not started | | | | |
| P5 gap loop | not started | | | | |

## Decisions

| Date | Decision | Reason |
| --- | --- | --- |
| 24 Nov | Removed the three existing mentions of the earlier branch prefix from `docs/` (gs32-amx, C1 plan, C1 validation) on this branch | Owner rule §2: the name must not appear in `develop`. Commit history on `develop` has none (checked `git log` messages and authors) |
| 24 Nov | P1 starts from the earlier concurrent-stream code (measured, older tree: job 5904726) rather than from scratch | It already pins two core groups to separate tiers with verified placement; E1-E6 extend it. Code is re-committed under the campaign's rules |
| 24 Nov | P2b porting waits for the P2 merge, then one sonnet agent ports steps 1-6 in order on `feature/70b-comparison`; step 7 (70B Q8 re-run) after the gs64/gs32 70B files exist | Plan §P2b: port onto the planner; avoids porting onto a moving merge branch |
| 24 Nov | No downloads: the 70B HF snapshot (community mirror, rev d50656ee, 30 shards hashed at download) and the gs64 (22 Nov) and gs32 (24 Nov) engine files already exist; only a sha256 manifest and provenance check are run | D1 asks for files, not for re-downloads |
| 24 Nov | E3 uses `mbind` per chunk | Flat-node kernel is 5.14 (RHEL 9.4), no `weighted_interleave` sysfs |
| 24 Nov | Node allocation: P1 owns Flat node hbm14; P2 gates use hbm13; P1's cross-node replication on hbm13 is scheduled by the manager | One job per Flat node; benchmark A/Bs stay on one node |

## P0 findings (24 Nov)

| Item | Finding |
| --- | --- |
| Internet | Compute nodes (cpu_dbg, hbm_dbg): none, no proxy. Login node reaches huggingface.co, github, pypi; HF LFS CDN name does not resolve. No transfer partition. No HF token on the cluster |
| Kernel | 5.14.0-427 (RHEL 9.4) on hbm13-16: no `MPOL_WEIGHTED_INTERLEAVE` |
| Memory mode | hbm15 (debug partition) shows 2 NUMA nodes only: Cache mode. hbm13/14 Flat per earlier jobs (confirmed again by P1/P2 job records) |
| CPU | Max 9480, 56 cores/socket, 1 thread/core, AMX-INT8/BF16, VNNI, AVX512-FP16; THP `always` |
| 70B files | HF snapshot 132 GB (community mirror rev d50656ee); engine v3 INT8 gs64 75.0 GB; engine v3 INT8 gs32 (from Q8_0 GGUF) 79.4 GB; FP32 v3 282 GB; Q8_0 and F32 GGUFs. All in scratch |
| 8B files | v3 FP32, v3 gs64, v3 gs32, v4 per-channel, HF snapshot |
| Storage | scratch is the large area; the store area reports ~45 GB free (unconfirmed), too small for the 70B engine files |
| Earlier 70B result dirs | placement pilots (22 Nov), research runs (23 Nov), cross-engine runs (24 Nov), all in scratch/store |

## P2b inventory (verified 24 Nov)

- Source: tip of the earlier 70B comparison work only (the research branch is its ancestor; the synthetic-KV branch holds
  only a duplicate and a near-duplicate). 41 commits beyond the placement branch; no occurrence of the forbidden name in
  content, messages or author fields (manager re-checked).
- The warm-up exchange-timing bug it fixes is live on `develop` (`src/app.c` warm_up only clears `profile_enabled`).
- Docs cite research commit hashes that will not exist after the port: rewrite or label as pre-port.
- C-engine 70B Q8 rows (TTFT 84 / 326 s, decode 4.5-5.3 tok/s) predate gs32 AMX/VNNI: stale.

## Prior evidence (measured, older tree; not campaign numbers)

- Job 5904726 (earlier 70B comparison work, revision of 23 Nov): one socket, 28 cores on HBM + 28 cores on DDR,
  4.2 GB each. Concurrent aggregate 490.7 GB/s [469.1-499.1] vs serial 308.7; per group 298.7 HBM / 245.3 DDR while
  both ran. Below the 552 GB/s 56-core HBM-only ceiling: E1 must sweep the split to answer H-concurrent.
- Job 5904742: layer-serial 40/40 hybrid decode at 8.07 tok/s (70B); HBM and DDR CAS rates anti-correlated (r = -0.986)
  at 10 ms bins: the serial model, as expected.

## Gap table

| Model | S | Placement | Context | Metric | Ceiling (source) | Measured median [range] | Eff. | Top phases | Status | Owner | Jobs |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| | | | | | | | | | | | |

## Open questions for the owner

| # | Question | State |
| --- | --- | --- |
| Q1 | The P2b inventory: port all of the earlier 70B comparison work from its tip in 7 steps (profile fix, synthetic KV + primer, tools, research sbatch, 70B cross-engine harness, docs with older-tree banner, optional 70B Q8 re-run); the two duplicate commits of the synthetic-KV variant dropped | **approved by owner 24 Nov: all seven steps, including the optional 70B Q8 re-run** |
