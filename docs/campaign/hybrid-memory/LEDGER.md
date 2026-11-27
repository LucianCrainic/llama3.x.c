# Hybrid memory campaign: ledger

Standing instructions: [`docs/plans/hybrid-memory-manager-prompt.md`](../../plans/hybrid-memory-manager-prompt.md)
(local, not committed). This file is the manager's memory across sessions. Started 24 November 2026 from `develop` at
`b9b9bbb`.

## G1 decision (25 Nov)

**Marginal.** Verified: raw split/shared reads never beat HBM-only (569.1; 578.3 lockstep-staggered); interleaved pages
+1-3% (two nodes, not robust); engine Q8 GEMV row split (r=0.2, 14 DDR workers, 4 KiB) +1.4% and +2.3% on node A, +3.9% on
node B (wall-clock +1.0/+3.0/+3.3%), p 0.006 per test, 0.066 best-of-11. Socket at ~350 W in every run. Actions: (1)
P3 as a bounded prototype (decode GEMV row split, opt-in) measured end to end on 70B against HBM-only, serial hybrid and
DDR, with the spill case (70B 1S, long-context 2S) as the main motivation: modelled concurrent vs serial +3 to +33%;
(2) first the 2 MiB GEMV stagger fix (perf/gemv-stagger, opt-in), which is the HBM-only baseline P3 must beat.

**Resume:** read this file, `git fetch`, `git log develop`, `squeue -u $USER`, then continue from **Next action**.

## Next action

25 Nov: G1 reported (marginal). Gate on develop 8fec0c8 **passed** (5929332, hbm14: make test rc 0, 3342 s, 0 FAIL/ERROR; manager-read). Running: perf/gemv-stagger 5929336 (hbm13),
TSAN q8-emul 5929318 (cpu), P2b cluster checks (hbm14, queued after 5929332). Push develop once TSAN 5929318 is clean (owner name-rule greps first).
P2b merged ee24659 + step 7 docs dbd8e14 + script fix 119d912 (local; docs/script-only after gate 5929402). perf/gemv-stagger merged 59f6bc6 (opt-in; manager re-derived medians from raw JSON: 2 MiB pages 8B 1S 42.21->44.31, 2S 73.53->74.95, 70B 10.00->10.60 tok/s; 4 KiB no effect; logits sha256 equal across 16 variants). Gate 5935721 on 59f6bc6 (hbm14). TSAN 5929318 **clean** (test-q8 rc 0 13268 s, test-q8-emul rc 0 16287 s, 0 race reports; manager-read). Gate 5935721 on 59f6bc6 **passed** (rc 0, 7 suites OK). **develop pushed cb28b94** (25 Nov; b9b9bbb..cb28b94; tree + log name greps empty). Stagger replicated (5935737, hbm14, manager-read): gs64 8B 1S +4.3%, 2S +1.2%, 70B +5.9%; gs32 +5.2/+1.6/+12.0%; 4 KiB no effect. **Default flip approved by owner (25 Nov)**: bd18928 (default 1, `=0` restores); gate 5935882 **passed** (rc 0, 8 suites OK); merged and pushed with develop. Then 70B C Q8 re-run (step 7), P3, P4.

Older: P1 engineer, P0 recon (inventory, docs), P2 merge and P2b inventory agents launched 24 Nov. Wait for their returns;
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
| P1 concurrent bandwidth | **done**, merged 8fec0c8 (local); gate on develop tip running | `perf/concurrent-tier-bandwidth` | sonnet | 5916187, 5916190, 5916193 (smoke), 5916195 (A), 5916213 (C), 5916429 (B), 5923181 (D), 5923753 (summary) | ~2.0 + B |
| G1 decision | **MARGINAL** (25 Nov, verified): engine GEMV row split +1.4/+2.3% (node A), +3.9% (node B); raw reads no gain beyond 1-3% | | manager | | |
| P2 merge placement | fixes done (7 hazards, test each), A/B bitwise PASS 122/122; gate 5916531: TSAN 0 race reports; test-tp and test-q8 clean, test-q8-emul timed out (rc 124) and reruns on cresco8_cpu; make test fails only on the stale guard test fixed in ce43988; 5923182 on ce43988: make test rc 0, A/B equal (1S HBM decode 44.74 vs 44.77, 2S 89.55 vs 89.72; manager-verified); carry-over diff 719232e..ce43988 = tests + docs only. **Open: test-q8-emul TSAN never completed; 20 h cpu-partition rerun refused by the permission system, asked owner** | `merge/hbm-ddr-placement` @719232e | sonnet | 5916198, 5916200 (TSAN caught a real race, fixed 32d93c2), 5916430, 5916531 | ~3.5 |
| P2b 70B port | **merged** ee24659 (local): steps 1-6 + fidelity note 301df53; owner name-rule greps empty; 5929376 (hbm14): default-path tokens bitwise equal to develop (1S/2S, warm 0/512, profile on); profile fix not visible on 8B (develop exchange not inflated; classifier phase 0.46 vs 0.04 ms/token, cause unconfirmed); 5929377: synthetic decode 50.44/50.49 vs real 52.01 tok/s (-3.0/-2.9%), amx -0.8%, CAS within 1.2%, placement + KV accounting identical (manager-read raw JSON). Gate 5929402 on ee24659 **passed** (make test rc 0, 0 FAIL/ERROR, 7 suites OK). Step 7 done: 70B C Q8 re-run merged dbd8e14 (jobs 5929406/5929407 DDR, 5935580 HBM = hybrid 80/80 packed=ddr; manager re-derived medians from runs.jsonl): DDR gs32 TTFT 2.69/10.22 s, decode 5.93/5.76; HBM 2.60/9.92 s, 10.97/10.12 tok/s; gs64 HBM 11.58/10.71; ids equal to 5910211. Script PLACEMENT/HYBRID_* options 119d912 | `feature/70b-comparison` | sonnet | 5929376, 5929377 | 0.35 |
| Downloads / 70B gs64 conversion | **not needed**: 70B gs64 and gs32 engine files exist (headers checked by manager); sha256 manifest done (job 5916180, cpu partition, 0.2 node-h; 70B gs64 and tokenizer hashes match the record made at conversion) | | haiku | 5916180 | 0.2 |
| P3 concurrent placement | prototype committed on feature/concurrent-tiers (58cbbc7: LLAMA_HYBRID_SPLIT=rows, DDR_FRACTION, DDR_CORES; one decode team call, last n members serve DDR rows). 5935734 check: logits sha256 1 digest over hbm-only/80-80/r0.1-0.3 (8B 1S/2S), 70B traces equal; its make test failed only on a test div-by-zero fixed in 58cbbc7. Gate 5935771 on 357d58f: make test rc 0, logits 1 digest (8B 1S/2S decode+batch), traces 0 differ (8B 1S/2S, 70B 2S) — manager-read. 5935769 failed on a script check (fixed 357d58f); 5935827 fits-8b-1s-b (manager-read, 5 reps, stagger on): hbm-only 44.05, best rows r0.15/c8 43.55 (-1.1%), P1 point r0.2/c14 42.90 (-2.6%), ddr 24.98: **no gain in the fits case at 8B 1S**. **Spill 5935860 (70B 1S, warm 4096, manager-read): rows-auto f0.254 c20 5.370 tok/s vs serial (59/80 weight layers in HBM) 4.834 = +11.1%; c14 5.321, f0.35-c20 5.333, c28 4.608; ddr 3.173; 36 traces equal. Limits model (cal) predicted serial 4.67, concurrent 5.52 (measured within -3%/+3%).** **Fits 2S (manager-read, 5 reps, all completed, traces 0 differ): 5936152 8B 2S hyb-all-hbm 74.40, best rows r0.15/c8 73.49 (-1.2%), ddr 44.75; 5936167 70B 2S (hybrid 80/80, packed ddr) 10.51, best r0.1/c8 10.27 (-2.3%), ddr 5.88. No gain in the fits case on 1S or 2S; mismatched fraction/cores lose up to 34%.** 5949316 superseded (sbatch --export split the comma lists; only serial ran). **Spill 2S long 5949331 (70B 2S, 64K warm, synthetic KV, manager-read, 5 reps, 24 traces 0 differ, placement PASS): serial (planner: weights 80, KV 37 in HBM) 2.458, serial-kv80 control (weights 23, KV 80) 2.369, rows-auto-c20 (f 0.708) 2.500 = +5.5% vs control, +1.7% vs serial (inside spread); c14 2.249.** Validation doc fc83661; P3 ~6.0 node-h. **Merged 5c86d9d onto develop 08aa16d (opt-in; defaults and default numerics unchanged); gate 5949471 on 5c86d9d passed (make test rc 0, 8 suites OK, hbm14).** Use only for weight spill (DDR_FRACTION=auto, 14-20 DDR cores) | `feature/concurrent-tiers` → develop | | | |
| P4 limits + baseline | limits doc merged (develop, after 59f6bc6; tests + --check-doc rc 0, TTFT ceiling re-derived by manager: 285 TOP, 0.63 s at 2048). Key: 70B TTFT 12-16x AMX peak; concurrent beats serial only in spill (1S: +15-32% modelled). Campaign started 26 Nov (owner: "Start P4") on feature/hybrid-campaign from develop 0dd18db: C0-C4, C6, C7; C5 deferred (no BF16/INT8 K/V yet, P5); C8 skipped (no Cache-mode node in cresco8_hbm). Reuses P3 5936167 (C3 @4096) and 5935860 (C6 @4096). Cap 40 node-h | `feature/hybrid-campaign` | sonnet | | |
| P5 gap loop | TTFT diagnosis merged (develop, local): 70B hybrid 80/80 packed=ddr gs64, driver medians (manager-read) P=2113 7.915 s @M=256 vs 5.603 @M=1024, P=547 2.106 vs 1.609; GEMM 89% at 40.8 TOP/s (9% AMX peak), 57.5 at M=1024; W1/W3 carries the per-batch cost (0.37 s/batch); packed copy read ~2.4x per M=256 batch from DDR; 8B packed hbm=ddr. Next: W1/W3 shard-shape gemm_bench (L3 capacity hypothesis), xengine row at M=1024 | | sonnet | 5935737, 5935864 | ~1.8 |

## Decisions

| Date | Decision | Reason |
| --- | --- | --- |
| 25 Nov | `LLAMA_GEMV_STAGGER=1` becomes the engine default (owner: "Yes, make default") | Bitwise unchanged; +1.2 to +12% decode on 2 MiB pages on two Flat nodes, gs64 and gs32; no effect on 4 KiB pages (D4: A/B, gates, owner OK) |
| 24 Nov | Removed the three existing mentions of the earlier branch prefix from `docs/` (gs32-amx, C1 plan, C1 validation) on this branch | Owner rule §2: the name must not appear in `develop`. Commit history on `develop` has none (checked `git log` messages and authors) |
| 24 Nov | P1 starts from the earlier concurrent-stream code (measured, older tree: job 5904726) rather than from scratch | It already pins two core groups to separate tiers with verified placement; E1-E6 extend it. Code is re-committed under the campaign's rules |
| 24 Nov | P2b porting waits for the P2 merge, then one sonnet agent ports steps 1-6 in order on `feature/70b-comparison`; step 7 (70B Q8 re-run) after the gs64/gs32 70B files exist | Plan §P2b: port onto the planner; avoids porting onto a moving merge branch |
| 24 Nov | No downloads: the 70B HF snapshot (community mirror, rev d50656ee, 30 shards hashed at download) and the gs64 (22 Nov) and gs32 (24 Nov) engine files already exist; only a sha256 manifest and provenance check are run | D1 asks for files, not for re-downloads |
| 24 Nov | Engine checkpoints stay in scratch; protection against a purge is the manifest plus the recipe (HF rev d50656ee, whose 30 shard hashes match the hub's; conversion jobs 5892714 gs64, 5910127/5910128/5910181 FP32/GGUF/gs32) | Store area quota is 100 GB hard with ~47 GB free; the two 70B Q8 files are 154 GB |
| 24 Nov | E3 uses `mbind` per chunk | Flat-node kernel is 5.14 (RHEL 9.4), no `weighted_interleave` sysfs |
| 25 Nov | Gate 5916531 (tree 719232e) HW make test rc=2, only the stale guard-list test fixed in ce43988; HW make test reruns on ce43988, TSAN carries over if src unchanged | §8: gates on the final tree |
| 25 Nov | P2 lands by `merge --no-ff merge/hbm-ddr-placement` into develop with the gate id in that message; no history rewrite of the branch | §10: no rewrites |
| 25 Nov | P2 needs an extra A/B on the HBM path (1S HBM, ~51 tok/s) | Its 1S A/B used the default file/DDR placement (~14 tok/s) |
| 25 Nov | P2 merged into develop (117cd0d) on owner's word with test-q8-emul TSAN incomplete; ledger branch merged (066a00d); both owner name-rule greps empty; push held until TSAN 5929318 is clean | D5: push only what passed every gate |
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

## Manifest (D1)

| File | Bytes | sha256 (first 16) | Source |
| --- | ---: | --- | --- |
| 70B int8-v3 gs64 | 74967187712 | 39016780130c2e4f | export job 5892714 from HF rev d50656ee; exporter commit not recorded |
| 70B tokenizer-v2 | 1864864 | ae7eab7f00edbd82 | same |
| 70B q8-gguf-v3 gs32 | 79376711936 | 7d59133b707e4209 | Q8_0 GGUF (job 5910128) repacked by job 5910181 |
| 70B q8_0.gguf | 74975050112 | 18b23ebb06099766 | job 5910128 |
| 70B fp32-v3 | 282214826240 | a4b504bc585e1af9 | job 5910127 |
| 8B int8-v3 gs64 | 8532934912 | 8d23a3b55c2dc46b | phase-2 export (matches earlier record) |
| 8B fp32-v3 | 32121045248 | 5ed04fb1046b7023 | same |
| 8B int8-pc-v4 | 8037591296 | 64cd68ffa9bbbb4a | 21 Nov |
| 8B tokenizer-v2 | 1864927 | 9b7aaa19a7abe3a5 | phase-2 export |
| 8B q8-gguf gs32 | 9034809600 | 68a6ac940702f205 | gs32 work, 24 Nov |

Full hashes are in the cluster manifest next to the models. The gated upstream repository hides its LFS hashes;
the mirror's file list and sizes match it.

## P2b inventory (verified 24 Nov)

- Source: tip of the earlier 70B comparison work only (the research branch is its ancestor; the synthetic-KV branch holds
  only a duplicate and a near-duplicate). 41 commits beyond the placement branch; no occurrence of the forbidden name in
  content, messages or author fields (manager re-checked).
- The warm-up exchange-timing bug it fixes is live on `develop` (`src/app.c` warm_up only clears `profile_enabled`).
- Docs cite research commit hashes that will not exist after the port: rewrite or label as pre-port.
- C-engine 70B Q8 rows (TTFT 84 / 326 s, decode 4.5-5.3 tok/s) predate gs32 AMX/VNNI: stale.

## P1 interim from job A/C (agent-reported, 7 reps, NOT yet verified by manager)

- Best HBM-only 569.1 [554.2-571.2] GB/s/socket (2M), still rising at 56 cores; DDR saturates 265.8 at 28 cores.
- E1: no split above HBM-only (54+2 562.4, 48+8 554.8, 28+28 501.1). E2 best ~561 (8:1..3:1), overlapping HBM-only.
  Replication on hbm13 within 0-1.6%.
- Package 351-354 W in every E1 window incl. DDR-only; core clock 1.77 GHz HBM-only, 1.44-1.57 mixed, 2.30 DDR-only.
- E5 all-HBM GEMV gs64: 317 (32 w), 398 (48), 418 (52), 401 (56) GB/s: below stream ceiling, 56 < 52 workers.
- Counter/computed exceptions: 6 base windows up to 1.47, 7 DDR windows up to 1.07: must be explained in validation.
- Loaded latency: DDR probe 132 -> 366 ns under 55 DDR streaming cores.

## P1 job D (agent-reported, under verification)

- E5 shuffled: best all-HBM GEMV 447.2 [440.6-451.7] (4K, 56 w, = 12.1 tok/s-equivalent 70B gs64); best row split
  453.4 [442.5-459.8] (+1.4%, overlapping). 2M all-HBM 400.9 (56 w) / ~422 (52 w): page-size effect ~11%, not alignment.
- E3 random: max 573.4 [567.0-577.4] at 8:1 (overlaps HBM-only). E4 node: HBM-only 1127.8, best mixed 1167.8
  [1113.1-1180.2] (overlaps). UPI 1.5 GB/s in E4 interleave unexplained.
- Zero data reads 6% faster than random (600.9 vs 565.8) at equal 351 W, core clock 1.75 vs 1.64 GHz.
- **Independent re-derivation of D (25 Nov): all bandwidth medians/ranges confirmed** (E5c full table, E3, E4, fillcmp),
  placement 100%, THP shares, in-process shuffle. Not confirmed: core clock (cycles/56 gives 2.10 vs 2.17 GHz, not 1.64 vs
  1.75); 351 W is socket 0 only (socket 1 ~180 W); CAS closure fails for e3 HBM (0.72, interleave bytes booked as HBM),
  e4 DDR (1.21-1.24), fill DDR (1.18); fill mode of e3/e4/e5c not recorded; e5off 4K spread 2.2%. Sent to P1 to fix.
- **Independent sonnet review (25 Nov):** numbers, placement, timing, cache, counters pass. Blocking: E6 was run at cd=0
  (best_points bug), so tuning at a real split never happened. Must-fix: small same-sign excess in E2-E5 (+0.8-3%, E5 +0.8/
  +1.4/+2.0% over three jobs, p 0.002-0.06) to be stated, not called noise; HBM-only grid stopped at pf 16K; reference
  must be e1-cd0; E3/E4 single node; GEMV is weight-stream only (upper bound). 2M-vs-4K GEMV effect is real (all jobs) but
  'not alignment' untested: likely lockstep stripe aliasing (w1/w3 rows per worker = exactly 2 MiB at 28/56 workers).
- Decision: before G1, P1 runs job E (E6 at real splits, pf 32-128K, aliasing test, cdw +-2) on hbm14 and job F
  (E3/E4/E5c replication) on hbm13, after fixing best_points, CPU-observed record, perf pinning. P1 cap +15 node-h.
- **Job E (5924010, agent-reported, verification running):** extended HBM-only best 567.3 (reference 569.1 stands);
  E6 at real splits: spreading DDR cores +1-2%, still <= HBM-only. E5 +/-2 DDR workers, 4K: all-HBM 459.0, best split
  r=0.2 / 14 DDR workers 469.8 (+2.3%, p=0.006); 12 or 16 DDR workers lose. Engine-shaped gain now +0.8 to +2.3% in four
  runs: small, reproducible, needs an exact worker split. **Aliasing confirmed:** 2M all-HBM GEMV 411 at 56 workers,
  437-447 at 53-55, 447.5 with a 4 KiB x worker stagger; raw lockstep 2M-aligned stripes 403 vs 554-577 staggered.
- **Job E re-derived (25 Nov): all medians/ranges confirmed**, pinning 3136/3136, random fill, placement. New: raw
  56-stream HBM read with stripes staggered 1 MiB+4 KiB = 576.7 [541.8-586.6] > e1-cd0 569.1 (the E1 reference is partly
  aliased). Reported vs bytes/wall GB/s differ (0.976-1.066); E5 best split +2.3% reported, +3.1% recomputed; best-of-11
  Bonferroni p ~0.013. Decision: job G (staggered HBM-only reference vs staggered splits) before G1.
- Manager's provisional G1 reading: **No** (E5 +1.4% within noise, far below 10%; raw streams no gain). Final after review.

## P1 interim from job B (agent-reported, unverified)

- E5 gs64 2M: best all-HBM 418 (52 w), best split ~447 (r 0.25-0.3) = +6.8%; 4K: all-HBM 444 (56 w), split 455-456 = ~+1%.
  Against the best all-HBM over both page sizes the gain is ~0-1%: G1 tends to No, pending shuffled confirmation (job D).
- E3 all-HBM read 603 vs E1 569: E3 regions zero-filled vs random in E1; data-dependent power under the cap suspected.
  Job D tests random vs zero (`fillcmp`) and reruns E3/E4 with random data.
- **Side finding for P5:** all-HBM GEMV 4K pages 444 vs 2M 403 GB/s at 56 workers (+10%); engine weights use THP. Asked
  for a shuffled 2M/4K GEMV pair in job D.
- 96 outlier windows (NTA hint up to 1.47, small-DDR-volume points up to 1.09) now listed; run-level HBM 0.993-1.007.

## P2 notes for P4

- 70B `hbm-only` overflows because of the AMX packed copy (per socket at ctx 8192: weights 34.4 GiB + packed 33.9 + rest
  5.0 = 73.2 GiB vs 56 GiB budget; without packed ~39.3). `hbm-only` rejects `LLAMA_HYBRID_PACKED`. **The 70B HBM
  baseline (C1) is therefore `hybrid` with all weight and KV layers in HBM and packed=ddr**, labelled as such.
- BF16 K/V copy is 15 GiB/socket at ctx 98304: it competes with KV for HBM under hybrid.
- `hybrid` admits gs64 and gs32 at ctx 98304 with HBM 55/56 GiB, auto KV layers 37 (gs64) / 33 (gs32).
- The 2S planned prefault writes the whole DDR-tier KV and BF16 arenas at setup (as legacy): setup cost at 70B.

## P1 verification of A/C (independent haiku re-derivation, 25 Nov)

- Confirmed exactly: E1 medians and ranges (all 12 splits, both pages), E5 HBM-only GEMV scan, 7 reps, placement 456/456
  events 100%, run-level CAS bytes 0.999 HBM / 1.001 DDR, package power ~351 W in all configurations.
- Corrections sent to P1: HBM-only core clock median 1.71 GHz (1.77 is the max); `common_s = 0` so E1 numbers are
  whole-window values; per-window CAS ratios depend on perf alignment (extremes not reproduced); interleaving only across
  page sizes; no E3 claim before job B.
- Replication hbm13 (C) vs hbm14 (A) re-derived: combined E1 points within -1.56..+0.39% (2M and 4K). Confirmed.

## P1 smoke (1 repetition, superseded)

- HBM-only 56 cores ~560-615 GB/s/socket (above the old 552); DDR-only ~253-260; split 48/8 539-582; 28/28 465-540;
  shared 3:1 552-615; pages 4:1 604-615: no split clearly above HBM-only so far.
- All-HBM Q8 GEMV gs64 ~406-457 GB/s: below the stream ceiling (core/power-bound?); r=0.25 similar.
- CAS bytes / computed 0.993-1.02 over 70 windows; placement 100%.
- Package ~350 W (TDP) in HBM and split windows, core clock 1.5-1.7 GHz vs 2.6 GHz DDR-only: power cap is the leading
  hypothesis. Prefetcher and uncore-frequency controls are not available (sudo allows perf stat, ipmitool, turbostat).
- Manager asked for RAPL (both sockets) and frequency per window, GEMV vs cores/frequency, best-HBM baseline with spread.

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
| Q2 | Is there a non-purged area larger than 100 GB for the 70B engine files, and what is the scratch purge policy? (not blocking) | open |
| Q3 | test-q8-emul TSAN: owner said merge now; run afterwards | answered 25 Nov: merged 117cd0d; TSAN job 5929318 (cresco8_cpu, 12 h) submitted; push of develop held until it reports 0 races |
| Q1 | The P2b inventory: port all of the earlier 70B comparison work from its tip in 7 steps (profile fix, synthetic KV + primer, tools, research sbatch, 70B cross-engine harness, docs with older-tree banner, optional 70B Q8 re-run); the two duplicate commits of the synthetic-KV variant dropped | **approved by owner 24 Nov: all seven steps, including the optional 70B Q8 re-run** |

## Synthetic K/V fidelity failure (26 Nov, manager-read)

- 5950121 (70B 2S hyb80, tree 47c08fb, hbm13): decode at 32768 real 7.217/7.202/7.150 against synthetic (PRIME 1024)
  3.870/3.898/3.688 tok/s, **-46%**, not the documented ~3%. Same windows (35 prompt, 95 steps), same placement and 2 MiB
  coverage, same CAS bytes per step (~47 GB per socket at 32 B per HBM CAS); **corrected:** not slower compute. The
  synthetic window holds one ~11-14 s near-idle span (HBM reads ~1 GB/s) starting ~5 s into decode; outside it both runs
  stream at ~360 GB/s per socket (manager re-derived from r1 perf.csv; agent from the profiled run). The span starts
  ~8.6 s after the process's first forward in both a synthetic run and the real t2048 run: the known unresolved "startup
  pause" of hbm-ddr-research.md. A real 32768 warm-up (171 s) absorbs it; a short primer puts it inside the decode window.
  Any short run whose timed window overlaps compute time ~8.6-23 s can be hit. Diagnostic 5950126 (8B 2S, hbm01 Cache mode).
- Consequences: every synthetic cell is void until fixed: P4 c1 d65536s and c2 d131072s in 5950121, the planned c3/c4/c6
  synthetic cells (held), and the P3 long-context run 5949331 (its relative result is suspect and is so marked here).
  Real-fill cells (4096, 32768, TTFT) are unaffected.
- Diagnosis agent on fix/synthetic-kv-fidelity (opt-in build only; default path untouched).

## P4 C0 (8B) verified (26 Nov)

- 5950120 (hbm14, tree 47c08fb, 1:37:00): 180 timed runs completed, placement PASS, traces 1 distinct per group, pause
  scan 239/240 clean (one 2S TTFT run 0.31 s idle in prefill). Manager re-derived 8 decode and 3 TTFT medians from raw
  JSON: all equal to the agent's table (e.g. 1S hbm-only d4096 44.13 [43.97-44.14], 2S ddr d32768 25.54, 1S rows r0.1
  c14 d32768 21.25, 2S hbm-only TTFT 2048 0.822 [0.700-0.879]).
- Serial model holds: serial cells 0.95-1.09 (1S) and 0.87-1.01 (2S) of the calibrated ceiling; CAS DDR share matches
  the planned byte fraction within 0.5 pp. No concurrent cell beats all-HBM (best -3.0% 1S, -3.5% 2S), as in P3.
- Stop rule: 1S rows r0.1/r0.2 at 32768 are 1.36/1.35 of the library concurrent ceiling (faster than modelled). Cause
  (derived): the library charges K/V at the reduced per-group rate, the engine splits only the GEMV and runs attention
  on all cores. Engine-form ceiling added to the summary tool (aae7b69), rows cells 0.92-1.15; library formula to be
  corrected in hybrid-limits (pending).
- Fix fix/synthetic-kv-fidelity 571e182 + docs 813d222 (LLAMA_SYNTHETIC_KV_MIN_SECONDS, default 30; SYNTHETIC_KV build
  only). Validation 5950963 (hbm14, diag script, manager-read): real 7.166/7.141, min30 7.188/7.230/7.197 (+0.9%),
  but the min0 control 7.220 did **not** pause (0/6 runs idle > 0.1 s), against 4/4 paused runs under
  hybrid-campaign.sbatch in 5950121. Suspect: something the campaign script runs next to the engine; analysis
  delegated. Gate 5950966 on 813d222 (hbm14). P4 synthetic cells stay held.
- **Correction (manager re-read):** the min0 control of 5950963 did pause, in its prefill (6.34 s against 0.70-0.82 s in
  the other 5 runs; TTFT 6.40 s). Per-0.5 s /proc samples show one all-CPU kernel-mode span (sys ~80% of ticks, all
  threads R, no vmstat counter moving) in every run of 5950963, 5950121 and the 8B hbm01 jobs: 6.5-14 s long, starting
  3.4-10.3 s after the first forward; not script-, perf-, node- or data-specific. Cause open (needs root for stacks).
  The fix is validated: min30 put the span in the primer in 3/3 runs (prefill 0.81-0.82 s, decode 7.188-7.230,
  slowest step <= 0.19 s). Docs correction 888e676.
- **Consequence for TTFT cells:** a short default-build run can have the span inside its timed prefill (70B 6.4 s TTFT
  against 0.8 s). Every TTFT cell must pass the pause scan on its prefill window; a guard for the default build is a
  design question for P4 (opt-in only).

## P4 C1/C2/C7 verified (27 Nov)

- 5950121 (hbm13, 47c08fb, 2:02:25, 64 runs, 0 failed, traces equal, placement PASS). Manager re-derived all 14 cell
  medians from raw JSON; all equal the agent's table. Decode (clean, real fills): c1 hyb80 d4096 10.51 [10.41-10.52]
  (0.95 of calibrated ceiling), d32768 7.20 (0.89); c2 ddr d4096 5.89 (0.98), d32768 4.65 (0.95). CAS bytes per token
  at 32768 ~13% above weights + K/V (96 vs 84.6 GB, derived, unexplained: open item).
- c1 at 65536 is not admitted by the planner (K/V + BF16 copy 9.6 GB over the 56 GiB budget): the all-HBM baseline ends
  below 65536. Synthetic cells of this job void (old build).
- TTFT: pause inside the timed prefill of all t8192 and avx512 runs (0/18 clean) and 2/5 c7 amx t2048. Clean t2048:
  c1 6.556 (3/3), c2 ddr 6.210 (2/3), c7 amx 7.038 (3/5). Decision (manager): opt-in LLAMA_BENCH_PREWARM_SECONDS
  (feature/bench-prewarm, bitwise-equivalence test required), TTFT cells rerun with it; cold-process TTFT reported
  separately.
- Running: 5962577 (hbm13: c3, c4, synthetic reruns; tree 787a714), 5962578 (hbm14: c6). ~11 of 40 node-h committed.
- C6 5962578 (hbm14, 787a714, 1:02:27, manager-read, 5 reps, d32768): serial (w64 kv0) real 3.433 / synthetic 3.460
  (+0.8%, fix holds, slowest step <= 0.42 s); serial-kv80 (w28 kv80) 2.975 (-14%); rows-auto-c20 (w80 kv80, f 0.643)
  2.769 (-20%); ddr 2.480 (-28%). At 1S long context the planner default (weights HBM, K/V DDR) beats every K/V-in-HBM
  cell; rows-kvmatch (K/V as serial) added for a fair test, queued on hbm14 after the prewarm gate.
- Prewarm knob feature/bench-prewarm e73408d (opt-in LLAMA_BENCH_PREWARM_SECONDS): check 5962596 running (hbm14),
  gate 5962597 queued.
