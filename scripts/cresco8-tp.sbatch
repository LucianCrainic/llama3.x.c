#!/usr/bin/env bash
#SBATCH --job-name=llama-tp
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=112
#SBATCH --exclusive
#SBATCH --mem=0
#SBATCH --time=04:00:00
# Flat Mode campaign skeleton for tensor parallelism (track B) on a dual Xeon Max 9480 node.
# Submit with a verified --partition/--nodelist and separate -o/-e paths; every submission needs the
# owner's go-ahead for that job. Required: ROOT (tree under test), OUT (new directory), PYTHON. Optional: MODELS
# ("NAME:ARTIFACT_DIR:HF_SNAPSHOT ..."; the artifact directory holds fp32-v3.bin, int8-v3.bin, tokenizer-v2.bin);
# without it the model gates are skipped and recorded. GATES_ONLY=1 makes B7 stop after its correctness gates.
# PHASE is a space-separated list of phase ids ("B1 B4"); phase_<ID> functions below (or in the library) hold each
# phase's matrix. PHASE=none runs only the common check. The exit status is non-zero if any expectation failed.
# B1-B6 are `check` phases: B1 and B4 need no model; B5 and B6 compare LLAMA_SOCKETS=2 with one socket on MODELS. B7 is
# the decode campaign (STAGE=bench; see phase_B7 for the matrix and the time it needs). DRY_RUN=1 prints the matrix
# without running anything.
set -euo pipefail
# Slurm runs a copy of this script, so the library is found through ROOT, not through the script's own path.
: "${ROOT:?tree under test required}"
# shellcheck source=lib/cresco8-common.sh
source "$ROOT/scripts/lib/cresco8-common.sh"
require_env
job_begin
# On hbm14: S0 = cores 0-55, DDR 0, HBM 2; S1 = cores 56-111, DDR 1, HBM 3 (explicit map C/H/D = package/HBM/DDR).
EXPLICIT='explicit:0/2/0,1/3/1'
# Two sockets run with no job-wide CPU or memory binding: each team pins itself and each region binds itself.
TP_ENV=(LLAMA_BACKEND=teams LLAMA_THREADS=56 OMP_PROC_BIND=false "LLAMA_TOPOLOGY=$EXPLICIT" LLAMA_PLACEMENT=hbm)

# ---- B1: shard descriptors and packing ---------------------------------------------------------------------------
# The synthetic reconstruction tests on the hardware, then each model sharded over both sockets from a copy staged on
# DDR node 0 into HBM 2 and 3 (teams pinned), compared with the file bitwise, pages verified, load time recorded.
phase_B1() {
  local dir="$OUT/B1" entry name artifacts hf precision staged
  expect shards-synthetic ok "$dir" -- "$ROOT/build/release/test-shards"
  [[ -n $MODELS ]] || { echo 'MODELS is empty: no B1 model run was made' | tee "$dir/no-models.txt"; return 0; }
  need_shm
  for entry in $MODELS; do
    IFS=: read -r name artifacts hf <<< "$entry"
    for precision in $PRECISIONS; do
      staged=$(stage_model_on 0 "$name" "$precision" "$artifacts")
      expect "verify-$name-$precision" ok "$dir" -- env "LLAMA_TOPOLOGY=$EXPLICIT" "$ROOT/build/release/test-shards" verify \
        "$staged" "$precision" hbm 56
      if ! dry_run; then
        if python3 - "$dir/verify-$name-$precision.out" << 'PY'; then
import json, sys
d = json.load(open(sys.argv[1]))
assert d['match'] == 1 and d['placement_ok'] == 1, d
assert d['regions'][0]['node'] == 2 and d['regions'][1]['node'] == 3, d['regions']
for r in d['regions'][:2]:
    assert r['verified'] and r['queried'] > 0 and r['on_node'] == r['queried'] and r['missing'] == 0, r
print('rows_seconds', d['rows_seconds'], 'cols_seconds', d['cols_seconds'], 'scratch_peak', d['scratch_peak'])
PY
          record_result "shards-$name-$precision" PASS
        else
          record_result "shards-$name-$precision" FAIL
        fi
      fi
      unstage_model_on 0 "$name" "$precision"
    done
  done
}

# ---- B2: partial-output kernels ----------------------------------------------------------------------------------
phase_B2() {
  local dir="$OUT/B2"
  expect kernels-teams-1-28-56 ok "$dir" -- env LLAMA_TEST_TEAMS=1,28,56 "$ROOT/build/release/test-kernels"
  if dry_run; then return 0; fi
  if grep -q 'avx512 teams:56 ok' "$dir/kernels-teams-1-28-56.out" && grep -q 'avx512 ok' "$dir/kernels-teams-1-28-56.out"; then
    record_result kernels-avx512-ok PASS
  else
    record_result kernels-avx512-ok FAIL 'avx512 and avx512 teams:56 were not both exercised and ok'
  fi
}

# ---- B3: socket-local KV and attention ---------------------------------------------------------------------------
phase_B3() {
  local dir="$OUT/B3" threads
  expect attention-split-2x56 ok "$dir" -- env LLAMA_TEST_TEAMS=1,56 "$ROOT/build/release/test-kernels"
  for threads in 28 56; do
    expect "attention-bench-2x$threads" ok "$dir" -- env "LLAMA_TOPOLOGY=$EXPLICIT" "$ROOT/build/release/test-kernels" \
      bench-attention "$threads"
  done
}

# ---- B4: cross-socket reduction ----------------------------------------------------------------------------------
# Stress at the hardware's team sizes, then the latency microbenchmark with pinned workers and the partial buffers on
# HBM 2 and 3. The per-token budget: 64 reductions against about 10% of the modeled 27 ms two-socket FP32 token time.
phase_B4() {
  local dir="$OUT/B4" bin="$ROOT/build/release/test-workers" threads
  for threads in 14 28 56; do
    expect "exchange-stress-2x$threads" ok "$dir" -- "$bin" exchange 20000 "$threads" "$threads" 4096
    expect "exchange-latency-2x$threads" ok "$dir" -- env "LLAMA_TOPOLOGY=$EXPLICIT" "$bin" bench-exchange 100000 "$threads"
  done
  expect exchange-fail-worker ok "$dir" -- bash -c "! $bin exchange-fail-worker"
  if dry_run; then return 0; fi
  python3 - "$dir" << 'PY' | tee "$dir/summary.txt"
import json, sys
from pathlib import Path
budget_us = 0.10 * 27e3 / 64      # 10% of 27 ms over 64 reductions
for threads in (14, 28, 56):
    p = Path(sys.argv[1]) / f'exchange-latency-2x{threads}.out'
    if not p.exists():
        continue
    for line in p.read_text().splitlines():
        r = json.loads(line)
        if r['elements'] == 4096:
            print(f"{threads} workers per socket, E=4096: median {r['median_us']} us, p99 {r['p99_us']} us; "
                  f"64 reductions = {64 * r['median_us'] / 1e3:.2f} ms ({'within' if r['median_us'] <= budget_us else 'OVER'} "
                  f"the {budget_us:.1f} us budget); pinned={r['pinned']}")
PY
}

# b_smoke NAME PRECISION ENGINE STAGED TOKENIZER TAG ENV...: one profiled two-socket run (recorded, not a gate) and its
# placement evidence; PHASE_ID selects the results directory.
b_smoke() {
  local name=$1 precision=$2 engine=$3 staged=$4 tokenizer=$5 tag=$6 label
  shift 6
  label=$(run_label "$name" "$precision" teams hbm 2 56 "x=$tag")
  GENERATION=128 CONTEXT=2048 REP=1 run_engine "$label" "$engine" "$staged" "$tokenizer" "$OUT/short.txt" 1 0 "$@" \
    LLAMA_SOCKETS=2 LLAMA_PROFILE=1
  dry_run && return 0
  if tp_placement "$OUT/$label-1.json" > "$OUT/B${PHASE_ID}/placement-$name-$precision-$tag.txt"; then
    record_result "placement-$name-$precision-$tag" PASS
  else
    record_result "placement-$name-$precision-$tag" FAIL "$(grep ^FAIL "$OUT/B${PHASE_ID}/placement-$name-$precision-$tag.txt" | head -3)"
  fi
}

# ---- B5: two-socket FP32 forward ---------------------------------------------------------------------------------
phase_B5() {
  local dir="$OUT/B5" entry name artifacts hf precision staged
  if [[ -z $MODELS ]]; then
    echo 'MODELS is empty: no B5 model run was made' | tee "$dir/no-models.txt"
    return 0
  fi
  need_shm
  a_models "$dir"
  for entry in $MODELS; do
    IFS=: read -r name artifacts hf <<< "$entry"
    for precision in $PRECISIONS; do
      [[ $precision == fp32 ]] || continue
      staged=$(stage_model_on 0 "$name" "$precision" "$artifacts")
      # Logits at every prompt position: one socket (A6's configuration) against two, residual copies checked.
      b_compare "$dir" "$name" "$precision" "$staged" 1s-vs-2s "${TP_ENV[@]}" LLAMA_SOCKETS=1 -- "${TP_ENV[@]}" \
        LLAMA_SOCKETS=2 LLAMA_TP_CHECK=1
      if [[ ${HF_GATE:-1} != 0 ]]; then
        tp_validate "$name" "$artifacts" "$hf" fp32 fp32-2s "${TP_ENV[@]}" LLAMA_SOCKETS=2 && record_result "hf-$name-fp32-2s" PASS ||
          record_result "hf-$name-fp32-2s" FAIL
      else
        echo "HF_GATE=0: the Hugging Face gate with LLAMA_SOCKETS=2 was not run for $name" | tee -a "$dir/no-hf-gate.txt"
      fi
      PHASE_ID=5 b_smoke "$name" "$precision" run "$staged" "$artifacts/tokenizer-v2.bin" smoke "${TP_ENV[@]}"
      unstage_model_on 0 "$name" "$precision"
    done
  done
  dry_run || "$PYTHON" "$ROOT/tools/summarize_campaign.py" "$OUT" > "$dir/summary.md"
}

# ---- B6: two-socket Q8 -------------------------------------------------------------------------------------------
# W8A8 and W8A32 against the one-socket path (a C-against-C gate, strict), then the Hugging Face deviation recorded next
# to the one-socket values, then the smoke benchmark.
phase_B6() {
  local dir="$OUT/B6" entry name artifacts hf precision staged
  if [[ -z $MODELS ]]; then
    echo 'MODELS is empty: no B6 model run was made' | tee "$dir/no-models.txt"
    return 0
  fi
  need_shm
  a_models "$dir"
  for entry in $MODELS; do
    IFS=: read -r name artifacts hf <<< "$entry"
    for precision in $PRECISIONS; do
      [[ $precision == int8 ]] || continue
      staged=$(stage_model_on 0 "$name" "$precision" "$artifacts")
      B_FLAGS=("${W8A8_GATE[@]}")
      b_compare "$dir" "$name" "$precision" "$staged" w8a8-1s-vs-2s "${TP_ENV[@]}" LLAMA_SOCKETS=1 -- "${TP_ENV[@]}" \
        LLAMA_SOCKETS=2 LLAMA_TP_CHECK=1
      B_FLAGS=()
      b_compare "$dir" "$name" "$precision" "$staged" w8a32-1s-vs-2s "${TP_ENV[@]}" LLAMA_SOCKETS=1 \
        LLAMA_Q8_ACTIVATIONS=fp32 -- "${TP_ENV[@]}" LLAMA_SOCKETS=2 LLAMA_TP_CHECK=1 LLAMA_Q8_ACTIVATIONS=fp32
      if [[ ${HF_GATE:-1} != 0 ]]; then
        tp_validate "$name" "$artifacts" "$hf" int8 w8a8-2s "${TP_ENV[@]}" LLAMA_SOCKETS=2 || true
        tp_validate "$name" "$artifacts" "$hf" int8 w8a32-2s "${TP_ENV[@]}" LLAMA_SOCKETS=2 LLAMA_Q8_ACTIVATIONS=fp32 || true
      else
        echo "HF_GATE=0: the Hugging Face deviation with LLAMA_SOCKETS=2 was not recorded for $name" | tee -a "$dir/no-hf-gate.txt"
      fi
      PHASE_ID=6 b_smoke "$name" "$precision" runq "$staged" "$artifacts/tokenizer-v2.bin" smoke "${TP_ENV[@]}"
      unstage_model_on 0 "$name" "$precision"
    done
  done
  dry_run || "$PYTHON" "$ROOT/tools/summarize_campaign.py" "$OUT" > "$dir/summary.md"
}

# ---- B7: two-socket decode campaign ------------------------------------------------------------------------------
# Decode of 128 tokens from a retained context (--warm-context N: an untimed batched prefill of N tokens, then the timed
# 86-token chat prompt and the timed decode) on LLAMA_BACKEND=teams with engine-owned HBM placement. Configurations: one
# socket x 56 (the A6 baseline: anon-hbm), two sockets x {14, 28, 42, 56} (workers per socket), and the control of the
# old kernel campaign, 112 OpenMP threads without tensor parallelism on a copy interleaved over HBM 2 and 3. Precisions
# FP32 and Q8 W8A8, and W8A32 at two sockets x 56 only. Contexts 512, 8192 and 32768 for the 56-worker configurations; the
# worker-count sweep and the control run at 512 only, which keeps the campaign inside its time budget (the grid of
# study 10.3 is trimmed the same way as C7's). FP32 also runs two sockets x 56 from the packed4 copy of the shards
# (LLAMA_FP32_LAYOUT=packed4, decision L8; tag p4), at every context, with its own profiled run. REPEATS interleaved repeats (configurations alternate within a repeat),
# plus one LLAMA_PROFILE=1 run per configuration and context whose per-socket profile lines (log level info) go to the
# .stderr files, and perf counters on the first repeat. The LLAMA_TP_PARTITION=rows comparator is not built (decision L8:
# only if B7 shows reductions plus waits above 20%). Gates first: one socket against two with batched prefill (the warm-up
# path) and the strict W8A32 and statistical W8A8 gates; a failed gate stops the benchmark. THP follows A6: set B7_THP
# (default off) to what A6 chose. Estimated at 4-6 hours for 1B and 8B; submit with --time=06:00:00 or split MODELS.
B7_THP=${B7_THP:-off}
B7_BATCH=${B7_BATCH:-64}
B7_DECODE=128
B7_CONTEXTS="512 8192 32768"

# b7_config CONFIG PRECISION: sets B7_SOCKETS, B7_TPS (workers per socket, or total for the control), B7_BACKEND, B7_CPU,
# B7_MEM, B7_PREC (the label's precision) and B7_ENV for a configuration name: 1s56, 2sT or ctl.
b7_config() {
  local config=$1
  B7_MEM= B7_BACKEND=teams B7_PREC=$2
  [[ $2 != w8a32 ]] || B7_ENV_EXTRA=(LLAMA_Q8_ACTIVATIONS=fp32)
  [[ $2 == w8a32 ]] || B7_ENV_EXTRA=()
  # A p4 suffix (FP32 only) streams the packed4 copy of the shards in decode (LLAMA_FP32_LAYOUT=packed4, decision L8).
  if [[ $config == *p4 ]]; then
    config=${config%p4}
    B7_ENV_EXTRA+=(LLAMA_FP32_LAYOUT=packed4)
  fi
  case $config in
    1s56 | 2s[0-9]*)
      B7_SOCKETS=${config:0:1} B7_TPS=${config:2} B7_CPU=
      [[ $B7_SOCKETS == 2 ]] || B7_CPU=0
      B7_ENV=(LLAMA_BACKEND=teams "LLAMA_THREADS=$B7_TPS" OMP_PROC_BIND=false "LLAMA_TOPOLOGY=$EXPLICIT" LLAMA_PLACEMENT=hbm
        "LLAMA_THP=$B7_THP" LLAMA_PREFILL_GEMM=on "LLAMA_PREFILL_BATCH=$B7_BATCH" "LLAMA_SOCKETS=$B7_SOCKETS") ;;
    ctl)
      B7_SOCKETS=2 B7_TPS=112 B7_BACKEND=omp B7_CPU=0,1 B7_MEM=2,3
      B7_ENV=(LLAMA_BACKEND=omp OMP_NUM_THREADS=112 OMP_PROC_BIND=close OMP_PLACES=cores "LLAMA_PREFILL_BATCH=$B7_BATCH") ;;
    *) die "unknown B7 configuration '$config'" ;;
  esac
  B7_ENV+=(${B7_ENV_EXTRA[@]+"${B7_ENV_EXTRA[@]}"})
}

# b7_launch NAME PRECISION CONFIG CONTEXT REP STAGED TOKENIZER [LABEL_WORDS...] [-- ENV_WORDS...]: one decode run from a
# retained context of CONTEXT tokens; the label carries the context as p=. A word x=prof makes it the profiled run.
b7_launch() {
  local name=$1 precision=$2 config=$3 ctx=$4 rep=$5 staged=$6 tokenizer=$7 label_words=() env_words=() side=label label
  local engine=run perf=0 log=warn
  shift 7
  while (($#)); do
    if [[ $1 == -- ]]; then side=env; elif [[ $side == label ]]; then label_words+=("$1"); else env_words+=("$1"); fi
    shift
  done
  [[ $precision == fp32 ]] || engine=runq
  b7_config "$config" "$precision"
  [[ $rep != 1 ]] || perf=$PERF
  [[ " ${label_words[*]-} " != *" x="*"prof "* ]] || log=info
  label=$(run_label "$name" "$B7_PREC" "$B7_BACKEND" hbm "$B7_SOCKETS" "$B7_TPS" "p=$ctx" ${label_words[@]+"${label_words[@]}"})
  [[ $config != ctl ]] || label=$(run_label "$name" "$B7_PREC" omp file 2 112 "p=$ctx" x=ctl)
  CPU_BIND=$B7_CPU MEM_BIND=$B7_MEM REP=$rep CONTEXT=$((ctx + 256)) GENERATION=$B7_DECODE WARM=$ctx LOG_LEVEL=$log \
    run_engine "$label" "$engine" "$staged" "$tokenizer" "$OUT/short.txt" 1 "$perf" "${B7_ENV[@]}" ${env_words[@]+"${env_words[@]}"}
}

# b7_gates DIR NAME ARTIFACTS: the correctness gates, on a copy staged on DDR node 0.
b7_gates() {
  local dir=$1 name=$2 artifacts=$3 precision staged
  for precision in $PRECISIONS; do
    staged=$(stage_model_on 0 "$name" "$precision" "$artifacts")
    if [[ $precision == fp32 ]]; then
      B_FLAGS=(--max-abs 1e-3 --batch-a "$B7_BATCH" --batch-b "$B7_BATCH")
      b_compare "$dir" "$name" fp32 "$staged" b7-batched-1s-vs-2s "${TP_ENV[@]}" LLAMA_SOCKETS=1 LLAMA_PREFILL_GEMM=on -- \
        "${TP_ENV[@]}" LLAMA_SOCKETS=2 LLAMA_TP_CHECK=1 LLAMA_PREFILL_GEMM=on
    else
      B_FLAGS=("${W8A8_GATE[@]}" --batch-a "$B7_BATCH" --batch-b "$B7_BATCH")
      b_compare "$dir" "$name" int8 "$staged" b7-w8a8-batched-1s-vs-2s "${TP_ENV[@]}" LLAMA_SOCKETS=1 LLAMA_PREFILL_GEMM=on -- \
        "${TP_ENV[@]}" LLAMA_SOCKETS=2 LLAMA_TP_CHECK=1 LLAMA_PREFILL_GEMM=on
      B_FLAGS=(--max-abs 1e-3)
      b_compare "$dir" "$name" int8 "$staged" b7-w8a32-1s-vs-2s "${TP_ENV[@]}" LLAMA_SOCKETS=1 LLAMA_Q8_ACTIVATIONS=fp32 -- \
        "${TP_ENV[@]}" LLAMA_SOCKETS=2 LLAMA_TP_CHECK=1 LLAMA_Q8_ACTIVATIONS=fp32
    fi
    if [[ $precision == fp32 ]]; then
      # The packed4 decode layout does the same arithmetic as the row layout: bitwise, on the AVX-512 kernels, under a
      # claiming schedule (the row layout then also forms whole four-row groups), on one socket's worth and on two.
      B_FLAGS=(--bitwise)
      b_compare "$dir" "$name" fp32 "$staged" b7-packed4-vs-rows-2s "${TP_ENV[@]}" LLAMA_SOCKETS=2 LLAMA_GEMV_SCHED=dynamic -- \
        "${TP_ENV[@]}" LLAMA_SOCKETS=2 LLAMA_GEMV_SCHED=dynamic LLAMA_FP32_LAYOUT=packed4 LLAMA_TP_CHECK=1
    fi
    B_FLAGS=()
    unstage_model_on 0 "$name" "$precision"
  done
}

phase_B7() {
  local dir="$OUT/B7" entry name artifacts hf precision staged tokenizer rep ctx config configs control
  if [[ -z $MODELS ]]; then
    echo 'MODELS is empty: no B7 model run was made' | tee "$dir/no-models.txt"
    return 0
  fi
  need_shm
  a_models "$dir"
  for entry in $MODELS; do
    IFS=: read -r name artifacts hf <<< "$entry"
    b7_gates "$dir" "$name" "$artifacts"
  done
  if ! dry_run && grep -q '^FAIL compare-.*-b7-' "$OUT/phase-results.txt" 2> /dev/null; then
    echo 'a B7 correctness gate failed: no benchmark was run' | tee "$dir/gate-failed.txt"
    return 0
  fi
  if [[ ${GATES_ONLY:-0} == 1 ]]; then
    echo 'GATES_ONLY=1: the gates ran, the benchmark did not' | tee "$dir/gates-only.txt"
    return 0
  fi
  for entry in $MODELS; do
    IFS=: read -r name artifacts hf <<< "$entry"
    tokenizer="$artifacts/tokenizer-v2.bin"
    for precision in $PRECISIONS; do
      staged=$(stage_model_on 0 "$name" "$precision" "$artifacts")
      for ((rep = 1; rep <= REPEATS; rep++)); do
        for ctx in $B7_CONTEXTS; do
          configs="1s56 2s56"
          [[ $ctx != 512 ]] || configs="1s56 2s14 2s28 2s42 2s56"
          for config in $configs; do
            b7_launch "$name" "$([[ $precision == fp32 ]] && echo fp32 || echo int8)" "$config" "$ctx" "$rep" "$staged" "$tokenizer"
          done
          # FP32: the same two-socket decode from the packed4 copy of the shards (decision L8).
          if [[ $precision == fp32 ]]; then
            b7_launch "$name" fp32 2s56p4 "$ctx" "$rep" "$staged" "$tokenizer" x=p4
          fi
          # W8A32 at two sockets x 56 only; its warm-up runs per token, so not at the longest context.
          if [[ $precision == int8 && $ctx != 32768 ]]; then
            b7_launch "$name" w8a32 2s56 "$ctx" "$rep" "$staged" "$tokenizer"
          fi
        done
      done
      # One profiled run per configuration and context (per-socket phase times in the .stderr files).
      for ctx in $B7_CONTEXTS; do
        configs="1s56 2s56"
        [[ $ctx != 512 ]] || configs="1s56 2s14 2s28 2s42 2s56"
        for config in $configs; do
          b7_launch "$name" "$([[ $precision == fp32 ]] && echo fp32 || echo int8)" "$config" "$ctx" 1 "$staged" "$tokenizer" \
            x=prof -- LLAMA_PROFILE=1
        done
        if [[ $precision == fp32 ]]; then
          b7_launch "$name" fp32 2s56p4 "$ctx" 1 "$staged" "$tokenizer" x=p4prof -- LLAMA_PROFILE=1
        fi
      done
      unstage_model_on 0 "$name" "$precision"
      # The control: 112 threads, no tensor parallelism, the copy interleaved over HBM 2 and 3 (the old kernel campaign).
      control="$shm/$name-$precision-hbm23.bin"
      if dry_run; then :; else stage_checkpoint "$artifacts/$precision-v3.bin" "$control" --interleave=2,3; fi
      for ((rep = 1; rep <= REPEATS; rep++)); do
        b7_launch "$name" "$([[ $precision == fp32 ]] && echo fp32 || echo int8)" ctl 512 "$rep" "$control" "$tokenizer"
      done
      dry_run || rm -f -- "$control" "$control.numa.txt"
    done
  done
  dry_run || "$PYTHON" "$ROOT/tools/summarize_campaign.py" "$OUT" > "$dir/summary.md"
}

# ---- E5: decode attention, LLAMA_ATTN_DECODE=group ---------------------------------------------------------------------
# Opt-in single-row attention kernel (one task per KV group and position chunk, scores of 16 positions reduced by one
# transpose-add, online softmax and PV fused, split-K partials merged as before); the default stays split. Both stages
# need the models; submit them one after the other. Estimates (not measurements): check 1.5-2.5 h, bench 5-7 h for 1B and 8B,
# FP32 and W8A8 (split the job with MODELS / PRECISIONS if the allocation is shorter; DRY_RUN=1 prints every launch).
# STAGE=check (E5/compare-*.json, E5/gen-*.json, correctness-*-group*.json): after the common `make test`,
#   per-token path (no batched prefill) at every position, LLAMA_ATTN_DECODE=split against group, on every model:
#     FP32 gate (max |dlogit| <= 1e-3, argmax 100%) at 86 and 2048 tokens, one socket; one socket split against two sockets
#     group (LLAMA_TP_CHECK=1) at 86 tokens; W8A8 (W8A8_GATE) at 86 and 2048 tokens one socket, 86 two sockets;
#   Hugging Face gate (validate.py, per-token) FP32 with group: one socket (OpenMP, as the common gate) and two sockets
#     (teams, LLAMA_TP_CHECK=1); HF_GATE=0 skips and records it;
#   determinism (tools/compare_generation.py, recorded, not gated): 128 greedy tokens from a warm 8192-token context, split
#     against group, one socket and two sockets, FP32 and W8A8, E5/gen-*.json.
# STAGE=bench (reports $OUT/*-xattn{split,group}-N.json, -xattn{split,group}prof-1.json with LLAMA_PROFILE=1): the B7 subset,
#   1B and 8B, FP32 and W8A8, contexts 512 / 8192 / 32768, one socket and two sockets x 56 workers, split against group,
#   alternating inside each repeat, REPEATS repeats plus one profiled repeat; then the attention bench decode cells
#   (E5/bench-decode-KERNEL-SHAPE-kvN.out and E5/bench-decode.jsonl: 8B and 1B, all KV heads and half of them as one socket
#   of two holds, 56 threads, 1000 repetitions: the bench caps --reps at 1024).
# Knobs of the bench stage (E5 follow-ups; every default reproduces the matrix above, DRY_RUN=1 prints it):
#   E5_CONTEXTS ("512 8192 32768") contexts of the engine runs and, by default, of the microbenchmark; E5_KERNELS
#   ("split group"), the order within a repeat; E5_SOCKETS ("1 2") socket counts of the engine runs; E5_PROFILED (1)
#   profiled repeats per configuration, context and kernel (-xattnKERNELprof-1 ... -N); E5_ALTERNATE (0) 1 reverses the
#   kernel order in even repeats, so neither kernel is always launched second; E5_SKIP_ENGINE (0) 1 skips the engine runs;
#   E5_SKIP_MICRO (0) 1 skips the microbenchmark; E5_MICRO_SHAPES ("8b 1b"), E5_MICRO_KV ("8 4": local KV heads),
#   E5_MICRO_CONTEXTS (E5_CONTEXTS), E5_MICRO_REPS (1000, at most 1024) and E5_LAYOUTS ("rows"): K/V layouts of the
#   microbenchmark, rows (the engine's) and head ([kv_head][position][head_dim], the E5.2 probe). With the default layout
#   the cells are those of the first E5 job; otherwise every cell takes --kv-layout and a -LAYOUT name suffix.
#   MODELS and PRECISIONS restrict the engine runs as in every phase.
# Knobs of the K/V layout study (E5.4, LLAMA_KV_LAYOUT=head; every default leaves the matrices above unchanged):
#   E5_KV_LAYOUTS (empty): "rows", "head" or "rows head" adds LLAMA_KV_LAYOUT=LAYOUT to the engine runs of the bench stage,
#   as a cell of its own per (kernel, layout) variant (label tag attnKERNELkvLAYOUT, profiled runs ...prof), so a repeat runs
#   every variant once, in the reverse order in even repeats with E5_ALTERNATE=1; the layouts of the engine runs, not of the
#   microbenchmark (E5_LAYOUTS). E5_ATTN (empty, split or tiled) sets LLAMA_ATTN (the batched prefill kernel, which builds
#   the warm contexts and runs the timed prompt) for the engine runs; empty keeps the job-wide default. E5_PREFILL_PROMPTS
#   (empty, e.g. "2048 8192") adds the time-to-first-token cells (V4): one socket, one cell per prompt length, rung and
#   layout, REPEATS repeats alternating the layout order, tiled prefill (E5_ATTN, default tiled), 32 generated tokens;
#   E5_PREFILL_RUNGS ("avx512_fp32 amx_bf16 avx512_q8 amx_int8", those of the precision of the checkpoint run) names the
#   rungs and M the one-socket choice of C7 (128, 256, 256, 256). E5_KV_CHECK (0): STAGE=check runs the correctness gates
#   of the layout (C2) instead of the split-against-group gates: rows against head, bitwise, on every model, precision,
#   socket count and decode kernel; a short prompt per token, a warm 8192-token context (tiled batched prefill, last row),
#   the decode path after it (tools/compare_engines.py --tail), 128 greedy tokens, the AMX rungs, and the Hugging Face
#   gate of FP32 with the head layout. E5_SKIP_DECODE (0): 1 leaves out the decode runs of the engine stage (a job of the
#   time-to-first-token cells alone); the stage writes the prompt files of E5_PREFILL_PROMPTS itself.
E5_KV_LAYOUTS=${E5_KV_LAYOUTS:-}
E5_ATTN=${E5_ATTN:-}
E5_PREFILL_PROMPTS=${E5_PREFILL_PROMPTS:-}
E5_PREFILL_RUNGS=${E5_PREFILL_RUNGS:-"avx512_fp32 amx_bf16 avx512_q8 amx_int8"}
E5_KV_CHECK=${E5_KV_CHECK:-0}
E5_SKIP_DECODE=${E5_SKIP_DECODE:-0}
E5_CONTEXTS=${E5_CONTEXTS:-"512 8192 32768"}
E5_KERNELS=${E5_KERNELS:-"split group"}
E5_SOCKETS=${E5_SOCKETS:-"1 2"}
E5_PROFILED=${E5_PROFILED:-1}
E5_ALTERNATE=${E5_ALTERNATE:-0}
E5_SKIP_ENGINE=${E5_SKIP_ENGINE:-0}
E5_SKIP_MICRO=${E5_SKIP_MICRO:-0}
E5_MICRO_SHAPES=${E5_MICRO_SHAPES:-"8b 1b"}
E5_MICRO_KV=${E5_MICRO_KV:-"8 4"}
E5_MICRO_CONTEXTS=${E5_MICRO_CONTEXTS:-$E5_CONTEXTS}
E5_MICRO_REPS=${E5_MICRO_REPS:-1000}
E5_LAYOUTS=${E5_LAYOUTS:-rows}

# e5_knobs: the E5 knobs are words from fixed sets or positive integers; anything else stops the phase before a launch.
e5_knobs() {
  local word
  [[ -n $E5_CONTEXTS && -n $E5_KERNELS && -n $E5_SOCKETS && -n $E5_LAYOUTS ]] || die "E5 knob lists must not be empty"
  for word in $E5_CONTEXTS $E5_MICRO_CONTEXTS $E5_MICRO_KV; do
    [[ $word =~ ^[1-9][0-9]*$ ]] || die "E5 contexts and KV heads must be positive integers, got '$word'"
  done
  for word in $E5_KERNELS; do [[ $word == split || $word == group ]] || die "E5_KERNELS words are split or group, got '$word'"; done
  for word in $E5_SOCKETS; do [[ $word == 1 || $word == 2 ]] || die "E5_SOCKETS words are 1 or 2, got '$word'"; done
  for word in $E5_LAYOUTS; do [[ $word == rows || $word == head ]] || die "E5_LAYOUTS words are rows or head, got '$word'"; done
  for word in $E5_MICRO_SHAPES; do
    [[ $word =~ ^(8b|1b|[1-9][0-9]*/[1-9][0-9]*/[1-9][0-9]*)$ ]] || die "bad E5_MICRO_SHAPES word '$word'"
  done
  for word in $E5_KV_LAYOUTS; do [[ $word == rows || $word == head ]] || die "E5_KV_LAYOUTS words are rows or head, got '$word'"; done
  [[ -z $E5_ATTN || $E5_ATTN == split || $E5_ATTN == tiled ]] || die "E5_ATTN is empty, split or tiled, got '$E5_ATTN'"
  for word in $E5_PREFILL_PROMPTS; do
    [[ $word =~ ^[1-9][0-9]*$ ]] || die "E5_PREFILL_PROMPTS words are prompt lengths, got '$word'"
  done
  [[ -z $E5_PREFILL_PROMPTS || -n $E5_KV_LAYOUTS ]] || die "E5_PREFILL_PROMPTS needs E5_KV_LAYOUTS (the cells compare layouts)"
  for word in $E5_PREFILL_RUNGS; do
    [[ $word == avx512_fp32 || $word == amx_bf16 || $word == avx512_q8 || $word == amx_int8 ]] ||
      die "E5_PREFILL_RUNGS words are avx512_fp32, amx_bf16, avx512_q8 or amx_int8, got '$word'"
  done
  [[ $E5_KV_CHECK == 0 || $E5_KV_CHECK == 1 ]] || die "E5_KV_CHECK is 0 or 1, got '$E5_KV_CHECK'"
  [[ $E5_SKIP_DECODE == 0 || $E5_SKIP_DECODE == 1 ]] || die "E5_SKIP_DECODE is 0 or 1, got '$E5_SKIP_DECODE'"
  [[ $E5_PROFILED =~ ^[0-9]+$ ]] || die "E5_PROFILED must be a non-negative integer, got '$E5_PROFILED'"
  [[ $E5_MICRO_REPS =~ ^[1-9][0-9]*$ ]] && ((E5_MICRO_REPS <= 1024)) || die "E5_MICRO_REPS must be 1..1024, got '$E5_MICRO_REPS'"
  for word in "$E5_ALTERNATE" "$E5_SKIP_ENGINE" "$E5_SKIP_MICRO"; do
    [[ $word == 0 || $word == 1 ]] || die "E5_ALTERNATE, E5_SKIP_ENGINE and E5_SKIP_MICRO are 0 or 1, got '$word'"
  done
}

# e5_pair DIR NAME PRECISION STAGED LABEL SOCKETS_A SOCKETS_B: per-token logits with LLAMA_ATTN_DECODE=split on SOCKETS_A against
# group on SOCKETS_B; the gate flags come from B_FLAGS (b_compare's default is the FP32 gate), the tokens from CMP_TOKENS.
e5_pair() {
  local dir=$1 name=$2 precision=$3 staged=$4 label=$5 a=$6 b=$7 check=()
  [[ $b != 2 ]] || check=(LLAMA_TP_CHECK=1)
  b_compare "$dir" "$name" "$precision" "$staged" "$label" "${TP_ENV[@]}" "LLAMA_SOCKETS=$a" LLAMA_ATTN_DECODE=split -- \
    "${TP_ENV[@]}" "LLAMA_SOCKETS=$b" ${check[@]+"${check[@]}"} LLAMA_ATTN_DECODE=group
}

# e5_generation DIR NAME PRECISION STAGED TOKENIZER SOCKETS: 128 greedy tokens from a warm 8192-token context, split against
# group on the same number of sockets (B7's configuration and warm-up path), recorded by compare_generation.py.
e5_generation() {
  local dir=$1 name=$2 precision=$3 staged=$4 tokenizer=$5 sockets=$6 engine=run probe=probe-fp32 args=() word
  [[ $precision == fp32 ]] || { engine=runq; probe=probe-int8; }
  b7_config 1s56 "$precision"
  for word in "${B7_ENV[@]}"; do
    [[ $word == LLAMA_SOCKETS=* ]] || args+=(--env-a "$word" --env-b "$word")
  done
  args+=(--env-a "LLAMA_SOCKETS=$sockets" --env-b "LLAMA_SOCKETS=$sockets" --env-a LLAMA_ATTN_DECODE=split
    --env-b LLAMA_ATTN_DECODE=group)
  [[ $sockets != 2 ]] || args+=(--env-a LLAMA_TP_CHECK=1 --env-b LLAMA_TP_CHECK=1)
  expect "gen-$name-$precision-s$sockets-split-vs-group" ok "$dir" -- "$PYTHON" "$ROOT/tools/compare_generation.py" \
    --engine "$ROOT/build/release/$engine" --checkpoint "$staged" --tokenizer "$tokenizer" --prompt-file "$OUT/short.txt" \
    --steps 128 --warm-context 8192 --engine-args "--context-length 8448 --chat-template" "${args[@]}" \
    --probe "$ROOT/build/release/$probe" --probe-batch "$B7_BATCH" \
    --output "$dir/gen-$name-$precision-s$sockets-split-vs-group.json"
}

e5_check() {
  local dir="$OUT/E5" entry name artifacts hf precision staged tokenizer
  a_models "$dir" 2048
  for entry in $MODELS; do
    IFS=: read -r name artifacts hf <<< "$entry"
    tokenizer="$artifacts/tokenizer-v2.bin"
    tokens_of "$name" "$OUT/prompt-2048.txt" "$dir/$name-tokens-2048.txt"
    for precision in $PRECISIONS; do
      staged=$(stage_model_on 0 "$name" "$precision" "$artifacts")
      if [[ $precision == fp32 ]]; then
        B_FLAGS=(--max-abs 1e-3)
        e5_pair "$dir" "$name" fp32 "$staged" e5-pertoken-1s 1 1
        e5_pair "$dir" "$name" fp32 "$staged" e5-pertoken-s1split-vs-s2group 1 2
        CMP_TOKENS="$dir/$name-tokens-2048.txt" e5_pair "$dir" "$name" fp32 "$staged" e5-pertoken-2048-1s 1 1
      else
        B_FLAGS=("${W8A8_GATE[@]}")
        e5_pair "$dir" "$name" int8 "$staged" e5-w8a8-pertoken-1s 1 1
        e5_pair "$dir" "$name" int8 "$staged" e5-w8a8-pertoken-s1split-vs-s2group 1 2
        CMP_TOKENS="$dir/$name-tokens-2048.txt" e5_pair "$dir" "$name" int8 "$staged" e5-w8a8-pertoken-2048-1s 1 1
      fi
      B_FLAGS=()
      e5_generation "$dir" "$name" "$precision" "$staged" "$tokenizer" 1
      e5_generation "$dir" "$name" "$precision" "$staged" "$tokenizer" 2
      unstage_model_on 0 "$name" "$precision"
    done
    # the Hugging Face gate of the FP32 per-token path with group (the common prelude ran the default kernel)
    if [[ " $PRECISIONS " == *" fp32 "* ]]; then
      if [[ ${HF_GATE:-1} != 0 ]]; then
        if validate_gate "$name" "$artifacts" "$hf" fp32 fp32-group LLAMA_ATTN_DECODE=group; then
          record_result "hf-$name-fp32-group" PASS
        else
          record_result "hf-$name-fp32-group" FAIL
        fi
        if tp_validate "$name" "$artifacts" "$hf" fp32 fp32-group-2s "${TP_ENV[@]}" LLAMA_SOCKETS=2 LLAMA_TP_CHECK=1 \
          LLAMA_ATTN_DECODE=group; then
          record_result "hf-$name-fp32-group-2s" PASS
        else
          record_result "hf-$name-fp32-group-2s" FAIL
        fi
      else
        echo "HF_GATE=0: the Hugging Face gate with LLAMA_ATTN_DECODE=group was not run for $name" | tee -a "$dir/no-hf-gate.txt"
      fi
    fi
  done
  dry_run || "$PYTHON" "$ROOT/tools/summarize_generation.py" "$dir" > "$dir/summary.md"
}

# ---- E5.4: correctness gates of LLAMA_KV_LAYOUT=head (C2) --------------------------------------------------------------------
# e5_kv_pair DIR NAME PRECISION STAGED LABEL SOCKETS ENV...: logits with LLAMA_KV_LAYOUT=rows against head, ENV on both
# sides, one or two sockets (LLAMA_TP_CHECK=1 on two); the gate flags come from B_FLAGS (always --bitwise here) and the
# tokens from CMP_TOKENS.
e5_kv_pair() {
  local dir=$1 name=$2 precision=$3 staged=$4 label=$5 sockets=$6 check=()
  shift 6
  [[ $sockets != 2 ]] || check=(LLAMA_TP_CHECK=1)
  b_compare "$dir" "$name" "$precision" "$staged" "$label" "${TP_ENV[@]}" "LLAMA_SOCKETS=$sockets" ${check[@]+"${check[@]}"} "$@" \
    LLAMA_KV_LAYOUT=rows -- "${TP_ENV[@]}" "LLAMA_SOCKETS=$sockets" ${check[@]+"${check[@]}"} "$@" LLAMA_KV_LAYOUT=head
}

# e5_kv_generation DIR NAME PRECISION STAGED TOKENIZER SOCKETS KERNEL: 128 greedy tokens from a warm 8192-token context and the
# short prompt (tiled batched prefill), rows against head with decode KERNEL, which must be identical (compare_generation.py).
e5_kv_generation() {
  local dir=$1 name=$2 precision=$3 staged=$4 tokenizer=$5 sockets=$6 kernel=$7 engine=run probe=probe-fp32 args=() word
  [[ $precision == fp32 ]] || { engine=runq; probe=probe-int8; }
  b7_config 1s56 "$precision"
  for word in "${B7_ENV[@]}"; do
    [[ $word == LLAMA_SOCKETS=* ]] || args+=(--env-a "$word" --env-b "$word")
  done
  args+=(--env-a "LLAMA_SOCKETS=$sockets" --env-b "LLAMA_SOCKETS=$sockets" --env-a "LLAMA_ATTN_DECODE=$kernel"
    --env-b "LLAMA_ATTN_DECODE=$kernel" --env-a LLAMA_ATTN=tiled --env-b LLAMA_ATTN=tiled --env-a LLAMA_KV_LAYOUT=rows
    --env-b LLAMA_KV_LAYOUT=head)
  [[ $sockets != 2 ]] || args+=(--env-a LLAMA_TP_CHECK=1 --env-b LLAMA_TP_CHECK=1)
  expect "gen-$name-$precision-s$sockets-$kernel-rows-vs-head" ok "$dir" -- "$PYTHON" "$ROOT/tools/compare_generation.py" \
    --engine "$ROOT/build/release/$engine" --checkpoint "$staged" --tokenizer "$tokenizer" --prompt-file "$OUT/short.txt" \
    --steps 128 --warm-context 8192 --engine-args "--context-length 8448 --chat-template" "${args[@]}" \
    --probe "$ROOT/build/release/$probe" --probe-batch "$B7_BATCH" --require-identical \
    --output "$dir/gen-$name-$precision-s$sockets-$kernel-rows-vs-head.json"
}

e5_kv_check() {
  local dir="$OUT/E5" entry name artifacts hf precision staged tokenizer sockets kernel batch=("--batch-a" "$B7_BATCH" "--batch-b" "$B7_BATCH")
  a_models "$dir" 2048,8192
  for entry in $MODELS; do
    IFS=: read -r name artifacts hf <<< "$entry"
    tokenizer="$artifacts/tokenizer-v2.bin"
    tokens_of "$name" "$OUT/prompt-8192.txt" "$dir/$name-tokens-8192.txt"
    for precision in $PRECISIONS; do
      staged=$(stage_model_on 0 "$name" "$precision" "$artifacts")
      for sockets in 1 2; do
        for kernel in split group; do
          # the short prompt per token (every position), and the decode path on a warm 8192-token context: the first
          # 8184 tokens through the tiled batched prefill, the last 8 per token
          B_FLAGS=(--bitwise)
          CMP_TOKENS= e5_kv_pair "$dir" "$name" "$precision" "$staged" "kv-short-s$sockets-$kernel" "$sockets" \
            "LLAMA_ATTN_DECODE=$kernel"
          B_FLAGS=(--bitwise "${batch[@]}" --tail 8)
          CMP_TOKENS="$dir/$name-tokens-8192.txt" e5_kv_pair "$dir" "$name" "$precision" "$staged" "kv-tail8192-s$sockets-$kernel" \
            "$sockets" "LLAMA_ATTN_DECODE=$kernel" LLAMA_ATTN=tiled LLAMA_PREFILL_GEMM=on "LLAMA_PREFILL_BATCH=$B7_BATCH"
          e5_kv_generation "$dir" "$name" "$precision" "$staged" "$tokenizer" "$sockets" "$kernel"
        done
        # the logits after the tiled batched prefill of the 8192 tokens (the last row), and the AMX rungs that write K/V
        B_FLAGS=(--bitwise "${batch[@]}" --last-only)
        CMP_TOKENS="$dir/$name-tokens-8192.txt" e5_kv_pair "$dir" "$name" "$precision" "$staged" "kv-prefill8192-s$sockets" \
          "$sockets" LLAMA_ATTN=tiled LLAMA_PREFILL_GEMM=on "LLAMA_PREFILL_BATCH=$B7_BATCH"
        if [[ $precision == fp32 ]]; then
          B_FLAGS=(--bitwise --batch-a 256 --batch-b 256 --last-only)
          CMP_TOKENS="$dir/$name-tokens-8192.txt" e5_kv_pair "$dir" "$name" "$precision" "$staged" "kv-amxbf16-8192-s$sockets" \
            "$sockets" LLAMA_ATTN=tiled LLAMA_KERNELS=amx LLAMA_PRECISION=bf16 LLAMA_PREFILL_BATCH=256
        else
          B_FLAGS=(--bitwise --batch-a 256 --batch-b 256 --last-only)
          CMP_TOKENS="$dir/$name-tokens-8192.txt" e5_kv_pair "$dir" "$name" "$precision" "$staged" "kv-amxint8-8192-s$sockets" \
            "$sockets" LLAMA_ATTN=tiled LLAMA_KERNELS=amx LLAMA_PREFILL_BATCH=256
        fi
      done
      B_FLAGS=()
      unstage_model_on 0 "$name" "$precision"
    done
    # the Hugging Face gate of the FP32 per-token path with the head-major cache, one socket and two sockets
    if [[ " $PRECISIONS " == *" fp32 "* ]]; then
      if [[ ${HF_GATE:-1} != 0 ]]; then
        if validate_gate "$name" "$artifacts" "$hf" fp32 fp32-head LLAMA_KV_LAYOUT=head; then
          record_result "hf-$name-fp32-head" PASS
        else
          record_result "hf-$name-fp32-head" FAIL
        fi
        if tp_validate "$name" "$artifacts" "$hf" fp32 fp32-head-2s "${TP_ENV[@]}" LLAMA_SOCKETS=2 LLAMA_TP_CHECK=1 \
          LLAMA_KV_LAYOUT=head; then
          record_result "hf-$name-fp32-head-2s" PASS
        else
          record_result "hf-$name-fp32-head-2s" FAIL
        fi
      else
        echo "HF_GATE=0: the Hugging Face gate with LLAMA_KV_LAYOUT=head was not run for $name" | tee -a "$dir/no-hf-gate.txt"
      fi
    fi
  done
  dry_run || "$PYTHON" "$ROOT/tools/summarize_generation.py" "$dir" > "$dir/summary.md"
}

# e5_bench_cell NAME KERNEL SHAPE KV [LAYOUT]: one attention-bench decode launch on socket 0's HBM, all microbenchmark
# contexts; a LAYOUT adds --kv-layout (E5.2).
e5_bench_cell() {
  local name=$1 kernel=$2 shape=$3 kv=$4 layout=${5:-} layout_args=()
  [[ -z $layout ]] || layout_args=(--kv-layout "$layout")
  expect "$name" ok "$OUT/E5" -- numactl --cpunodebind=0 env LLAMA_BACKEND=teams LLAMA_THREADS=56 OMP_PROC_BIND=false \
    LLAMA_KERNELS=avx512 "LLAMA_ATTN_DECODE=$kernel" "$ROOT/build/release/attention-bench" --node 2 --mode decode \
    --shape "$shape" --context "${E5_MICRO_CONTEXTS// /,}" --reps "$E5_MICRO_REPS" --kv-heads-local "$kv" \
    ${layout_args[@]+"${layout_args[@]}"}
}

# e5_variants REVERSE: the (kernel, layout) variants of a repeat as words "kernel:layout" in E5_VARIANTS ("kernel:" when
# E5_KV_LAYOUTS is empty), in the reverse order when REVERSE is 1 (E5_ALTERNATE, even repeats).
e5_variants() {
  local kernel layout word
  E5_VARIANTS=()
  for kernel in $E5_KERNELS; do
    for layout in ${E5_KV_LAYOUTS:-@}; do
      word="$kernel:${layout/@/}"
      if [[ ${1:-0} == 1 ]]; then E5_VARIANTS=("$word" ${E5_VARIANTS[@]+"${E5_VARIANTS[@]}"}); else E5_VARIANTS+=("$word"); fi
    done
  done
}

# e5_decode_launch NAME PRECISION CONFIG CONTEXT REP STAGED TOKENIZER VARIANT PROFILED: one decode run of a variant.
e5_decode_launch() {
  local name=$1 precision=$2 config=$3 ctx=$4 rep=$5 staged=$6 tokenizer=$7 variant=$8 profiled=$9
  local kernel=${variant%%:*} layout=${variant#*:} tag env_words
  tag="attn$kernel"
  env_words=("LLAMA_ATTN_DECODE=$kernel")
  [[ -z $layout ]] || { tag+="kv$layout"; env_words+=("LLAMA_KV_LAYOUT=$layout"); }
  [[ -z $E5_ATTN ]] || env_words+=("LLAMA_ATTN=$E5_ATTN")
  [[ $profiled != 1 ]] || { tag+=prof; env_words+=(LLAMA_PROFILE=1); }
  b7_launch "$name" "$precision" "$config" "$ctx" "$rep" "$staged" "$tokenizer" "x=$tag" -- "${env_words[@]}"
}

# e5_rung RUNG M: RUNG_PREC, RUNG_ENGINE, RUNG_ENV of the prefill rungs of the layout cells (as c_rung of the prefill script).
e5_rung() {
  local m=$2
  case $1 in
    avx512_fp32) RUNG_PREC=fp32 RUNG_ENGINE=run RUNG_ENV=(LLAMA_KERNELS=avx512 LLAMA_PREFILL_GEMM=on "LLAMA_PREFILL_BATCH=$m") ;;
    amx_bf16) RUNG_PREC=bf16 RUNG_ENGINE=run RUNG_ENV=(LLAMA_KERNELS=amx LLAMA_PRECISION=bf16 "LLAMA_PREFILL_BATCH=$m") ;;
    avx512_q8) RUNG_PREC=int8 RUNG_ENGINE=runq RUNG_ENV=(LLAMA_KERNELS=avx512 LLAMA_PREFILL_GEMM=on "LLAMA_PREFILL_BATCH=$m") ;;
    amx_int8) RUNG_PREC=int8 RUNG_ENGINE=runq RUNG_ENV=(LLAMA_KERNELS=amx "LLAMA_PREFILL_BATCH=$m") ;;
    *) die "unknown prefill rung '$1'" ;;
  esac
}

# e5_prefill_runs NAME PRECISION STAGED TOKENIZER: the time-to-first-token cells (V4) of one checkpoint: one socket, every
# prompt, rung of the precision and layout, REPEATS repeats; the layouts alternate their order in even repeats (E5_ALTERNATE).
e5_prefill_runs() {
  local name=$1 precision=$2 staged=$3 tokenizer=$4 rep n rung m layout layouts prompt
  for ((rep = 1; rep <= REPEATS; rep++)); do
    layouts=$E5_KV_LAYOUTS
    if [[ $E5_ALTERNATE == 1 && $((rep % 2)) == 0 ]]; then
      layouts=
      for layout in $E5_KV_LAYOUTS; do layouts="$layout $layouts"; done
    fi
    for n in $E5_PREFILL_PROMPTS; do
      prompt="$OUT/prompt-$n.txt"
      for rung in $E5_PREFILL_RUNGS; do
        case $rung in
          avx512_fp32 | amx_bf16) [[ $precision == fp32 ]] || continue ;;
          *) [[ $precision == int8 ]] || continue ;;
        esac
        m=256
        [[ $rung != avx512_fp32 ]] || m=128
        for layout in $layouts; do
          e5_rung "$rung" "$m"
          b7_config 1s56 "$RUNG_PREC"
          CPU_BIND=$B7_CPU REP=$rep CONTEXT=$((n + 256)) GENERATION=32 LOG_LEVEL=warn run_engine \
            "$(run_label "$name" "$RUNG_PREC" teams hbm 1 56 "m=$m" "p=$n" "k=$rung" "x=kv$layout")" "$RUNG_ENGINE" "$staged" \
            "$tokenizer" "$prompt" 1 "$([[ $rep == 1 ]] && echo "$PERF" || echo 0)" "${B7_ENV[@]}" "${RUNG_ENV[@]}" \
            "LLAMA_ATTN=${E5_ATTN:-tiled}" "LLAMA_KV_LAYOUT=$layout"
        done
      done
    done
  done
}

# e5_engine_runs: the engine decode runs of the bench stage.
e5_engine_runs() {
  local entry name artifacts hf precision staged tokenizer rep ctx config variant engine_precision configs=()
  for ctx in $E5_SOCKETS; do configs+=("${ctx}s56"); done
  for entry in $MODELS; do
    IFS=: read -r name artifacts hf <<< "$entry"
    tokenizer="$artifacts/tokenizer-v2.bin"
    for precision in $PRECISIONS; do
      engine_precision=int8
      [[ $precision != fp32 ]] || engine_precision=fp32
      staged=$(stage_model_on 0 "$name" "$precision" "$artifacts")
      for ((rep = 1; rep <= REPEATS && E5_SKIP_DECODE == 0; rep++)); do
        # E5_ALTERNATE: even repeats launch the variants in the reverse order
        e5_variants "$([[ $E5_ALTERNATE == 1 && $((rep % 2)) == 0 ]] && echo 1 || echo 0)"
        for ctx in $E5_CONTEXTS; do
          for config in "${configs[@]}"; do
            for variant in "${E5_VARIANTS[@]}"; do
              e5_decode_launch "$name" "$engine_precision" "$config" "$ctx" "$rep" "$staged" "$tokenizer" "$variant" 0
            done
          done
        done
      done
      # Profiled repeats per configuration and context (the per-socket profile lines go to the .stderr files).
      e5_variants 0
      for ((rep = 1; rep <= E5_PROFILED && E5_SKIP_DECODE == 0; rep++)); do
        for ctx in $E5_CONTEXTS; do
          for config in "${configs[@]}"; do
            for variant in "${E5_VARIANTS[@]}"; do
              e5_decode_launch "$name" "$engine_precision" "$config" "$ctx" "$rep" "$staged" "$tokenizer" "$variant" 1
            done
          done
        done
      done
      [[ -z $E5_PREFILL_PROMPTS ]] || e5_prefill_runs "$name" "$precision" "$staged" "$tokenizer"
      unstage_model_on 0 "$name" "$precision"
    done
  done
}

e5_bench() {
  local dir="$OUT/E5" shape kv kernel layout
  e5_knobs
  [[ $E5_SKIP_ENGINE == 1 ]] || e5_engine_runs
  # the attention bench: the layer's decode attention alone, all KV heads and the half one socket of two holds
  if [[ $E5_SKIP_MICRO != 1 ]]; then
    for shape in $E5_MICRO_SHAPES; do
      for kv in $E5_MICRO_KV; do
        for kernel in $E5_KERNELS; do
          if [[ $E5_LAYOUTS == rows ]]; then
            e5_bench_cell "bench-decode-$kernel-${shape//\//x}-kv$kv" "$kernel" "$shape" "$kv"
          else
            for layout in $E5_LAYOUTS; do
              e5_bench_cell "bench-decode-$kernel-${shape//\//x}-kv$kv-$layout" "$kernel" "$shape" "$kv" "$layout"
            done
          fi
        done
      done
    done
  fi
  if ! dry_run; then
    if compgen -G "$dir/bench-decode-*.out" > /dev/null; then cat "$dir"/bench-decode-*.out > "$dir/bench-decode.jsonl"; else : > "$dir/bench-decode.jsonl"; fi
    "$PYTHON" "$ROOT/tools/summarize_campaign.py" "$OUT" > "$dir/summary.md"
  fi
}

phase_E5() {
  local dir="$OUT/E5"
  if [[ -z $MODELS ]]; then
    echo 'MODELS is empty: no E5 model run was made' | tee "$dir/no-models.txt"
    return 0
  fi
  [[ $E5_KV_CHECK == 0 || $E5_KV_CHECK == 1 ]] || die "E5_KV_CHECK is 0 or 1, got '$E5_KV_CHECK'"
  need_shm
  if [[ $STAGE == check ]]; then
    if [[ $E5_KV_CHECK == 1 ]]; then e5_knobs && e5_kv_check; else e5_check; fi
  else
    if [[ -n $E5_PREFILL_PROMPTS ]]; then
      # the time-to-first-token cells read prompt-N.txt (only the check stage used to write them: jobs 5876582 and 5876584)
      dry_run && echo "E5 prompts: ${E5_PREFILL_PROMPTS// /,}"
      a_models "$dir" "${E5_PREFILL_PROMPTS// /,}"
    else
      a_models "$dir"
    fi
    e5_bench
  fi
}

require_phases
if ! dry_run; then
  check_stage
  validate_models
fi
run_phases
exit "$OVERALL"
