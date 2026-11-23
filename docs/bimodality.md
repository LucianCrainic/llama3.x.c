# Prefill TTFT bimodality (23–24 November 2026)

W8A8 Q8_0 gs=64 P2048 prefill (8B, `int8-v3.bin`, tree 5c0dd56, default knobs). Each process run lands in a fast or a slow mode:

| Socket count | Fast mode | Slow mode | Gap |
| --- | --- | --- | --- |
| 1S (56 threads) | 1.27–1.33 s | 1.37–1.51 s | about +110 to +160 ms (+9 to +12%) |
| 2S | 0.71–0.78 s | 0.80–0.86 s | about +60 ms (+8%) |

Long prompts show the same split: the interleaved L 2S runs sit at 3.75 s or at 3.97–4.0 s (jobs 5898029, 5898041).

**Status: cause not found.** Seven measurement jobs rule out the memory side, the scheduler, interrupts, other processes and page sizes. Slow runs do the same memory work and spin longer at the team's barriers. One worker or more is slow while running on its own core. What slows it is still open.

## 1. What a slow run looks like

**It is a per-process state.** In the earlier 15-repeat A/Bs (5898029, 5898041) the slow runs are scattered through the run order, with no clustering in time.

**How often it happens changes between jobs.** Slow runs out of 30:

| Job | 1S | 2S |
| --- | --- | --- |
| 5904733 | 4 | 5 (with perf), 12 (without) |
| 5904762 | 9 (56 threads), 8 (55 threads) | 1 (112 threads), 15 (110 threads) |
| 5904985 | 2 | 1 |

**Slow runs spin more** (5904733, hbm13; 2S figures are for socket 0):

| Counter over the prefill window | 1S fast | 1S slow | 2S fast | 2S slow |
| --- | ---: | ---: | ---: | ---: |
| Core cycles (G) | 141 | 166 | 86 | 96 |
| Instructions (G) | 232 | 297 (+28%) | 137 | 162 (+18%) |
| IPC | 1.64 | 1.80 | 1.60 | 1.68 |
| L1D replacements (G) | 19.6 | 19.6 | 10.1 | 10.1 |
| L2 lines in (G) | 8.34 | 8.38 | 4.28 | 4.29 |
| Fraction of cycles stalled on L2 misses | 0.176 | 0.154 | 0.165 | 0.159 |
| HBM bytes read (GB) | 26.3 | 26.2 | 17.5 | 18.0 |

What the table shows:
- The work is unchanged: L1, L2 and HBM traffic are the same in both modes.
- The extra cycles retire extra instructions at a higher IPC, with fewer cycles stalled on L2 misses. That is the signature of spin-waiting on a flag that sits in L1.
- So in a slow run the team waits longer for one worker or more, as the profiled run of 5898042 showed: W13 +41 ms on both sockets, quantize +41 ms on socket 1.

## 2. What is ruled out

| Hypothesis | Test | Job | Result |
| --- | --- | --- | --- |
| Uncore/mesh clock | CHA clockticks over the prefill window | 5904519 (hbm14, 2S) | 1.69/1.76 GHz fast against 1.71/1.80 GHz slow (sockets 0/1): not lower |
| Memory latency | CHA TOR occupancy over inserts, IA demand-read misses | 5904519 | 145–158 ns in both modes |
| Extra traffic | HBM/DDR CAS, UPI flits, LLC misses | 5904519, 5904733 | the same in both modes |
| Measurement perturbation | 2S with socket-level perf against 2S without | 5904733 | median 0.768 against 0.771 s, same two modes |
| Preemption by another task | per-thread `/proc` schedstat run-queue delay, nonvoluntary switches | 5904762 | the same in both modes (1S: total 80–120 ms, worst thread 9–10 ms, at start-up) |
| Daemons, interrupts | other processes' CPU time per run; `/proc/interrupts` and `/proc/softirqs` per CPU | 5904762 | 10–30 ms of other processes' CPU per run, no slow-mode pattern; interrupts scale with run length only |
| A busy core | leave one core per socket free (55 threads) | 5904762 | 1S still 8/30 slow; 2S 15/30 slow |
| 4 KiB pages | per-mapping `AnonHugePages` at peak Rss; free 2 MiB blocks per node before each run | 5904985 | 15.8 GB anonymous memory, about 50 MB on 4 KiB pages in both modes; about 490 GB of free 2 MiB blocks |
| Core clock | cycles over ref-cycles | 5904519, 5904733 | not lower in slow runs; slightly higher, as expected of spinning cores |

## 3. Instruments that failed

- **Per-CPU counting** (`perf stat -A -I 50`, 56 CPUs, 5904766) slowed every 1S run to 4.8–5.0 s, so it cannot show a 10% effect.
  - CPU 0 retires 0.19 of the median core's instructions in every run. It hosts the team leader and its serial work, so it does not mark a straggler.
  - No other core stood out: the next lowest was at 0.94–0.97 of the median.
- **Scheduler tracing** (`perf record -e sched:sched_switch`, 5904743) is not available, because sudo on the compute nodes allows `perf stat` only (also `turbostat` and `ipmitool sensor`).
- **The first window analysis** of 5904519 keyed on a start-up counter spike. It was redone with the prefill window placed by HBM reads (`bimodal_an.py`).
- **5904519's 1S arm** ran without the 56-thread, explicit-topology, `numactl --cpunodebind=0` launch, so one team spread over both sockets (1.78 s). That arm is discarded.

## 4. What is left to try

- **Per-worker busy time in the engine.** Under a profiling knob, accumulate each worker's time inside its chunks for each phase with `rdtsc`, with no per-chunk logging. This names the late worker and phase, and shows whether the late worker is the same CPU or thread index across runs. The existing `LLAMA_PROFILE=1` slows 2S runs to 0.92–1.05 s, so the new knob must cost less.
- **Per-core clock.** Run `turbostat` (allowed by sudo) at 1 s intervals over many runs, for each core's busy MHz.
- **The node's state.** Mode fractions changed from 3% to 50% between jobs a few hours apart. Record the package power and temperature (`turbostat`, `ipmitool sensor`) per run.

## 5. Consequences for measurements

- **A/Bs at P must interleave configurations** and read paired diffs and per-mode medians, with at least 15 repeats (as in 5898029/5898041). Five repeats cannot resolve 15–30 ms.
- **A 1S median of 3–5 repeats can land on either side.** For example, 5892420 gave 1.474 s against 1.299 s in 5892421.

## Scripts and data

`R/prefill-gap/scripts/bimodal{2..7}.sbatch`, `bimodal_an.py`, `bimodal5_an.py`, `bimodal7_an.py`; per-run data under `R/prefill-gap/bimodal{2..7}-<job>/` (`R` is the campaign results directory on the cluster).
