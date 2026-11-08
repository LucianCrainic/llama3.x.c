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
# without it the model gates are skipped and recorded.
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

if ! dry_run; then
  check_stage
  validate_models
fi
run_phases
exit "$OVERALL"
