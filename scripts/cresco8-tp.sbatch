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
# B1-B6 are `check` phases: B1 and B4 need no model; B5 and B6 compare LLAMA_SOCKETS=2 with one socket on MODELS. B7
# (the decode campaign) is not here yet. DRY_RUN=1 prints the matrix without running anything.
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
B_FLAGS=() # b_compare's budget when it is not the strict 1e-3 (W8A8 sets the statistical gate)
TP_ENV=(LLAMA_BACKEND=teams LLAMA_THREADS=56 OMP_PROC_BIND=false "LLAMA_TOPOLOGY=$EXPLICIT" LLAMA_PLACEMENT=hbm)

# tp_validate NAME ARTIFACTS HF PRECISION LABEL [env...]: the Hugging Face gate without the one-socket numactl binding
# of validate_gate. FP32 failure is fatal; INT8 is recorded against today's unchanged budget.
tp_validate() {
  local name=$1 artifacts=$2 hf=$3 precision=$4 label=$5 status=0
  shift 5
  if dry_run; then
    printf 'VALIDATE %s %s %s env=%s\n' "$name" "$precision" "$label" "$*" | tee -a "$OUT/dry-run.txt"
    return 0
  fi
  local extra=()
  [[ $precision != int8 ]] || extra+=(--max-abs 1.0 --rtol 0.05)
  (cd "$ROOT" && env "$@" "$PYTHON" tools/validate.py --hf "$hf" --checkpoint "$artifacts/$precision-v3.bin" \
    --tokenizer "$artifacts/tokenizer-v2.bin" --precision "$precision" --threads 56 \
    --output "$OUT/correctness-$name-$label.json" "${extra[@]}") \
    > "$OUT/correctness-$name-$label.stdout" 2> "$OUT/correctness-$name-$label.stderr" || status=$?
  printf '%s %s exit=%s env=%s\n' "$name" "$label" "$status" "$*" >> "$OUT/correctness-status.txt"
  return "$status"
}

# tp_placement REPORT: both sockets' weights regions fully verified on HBM 2 and HBM 3, an untied embedding on DDR 0.
tp_placement() {
  python3 - "$1" << 'PY'
import json, sys
report = json.load(open(sys.argv[1]))
regions = report.get('regions') or []
bad = 0
def check(name, ok, detail=''):
    global bad
    bad += not ok
    print(('PASS' if ok else 'FAIL'), name, detail)
check('run completed', report.get('status') == 'completed', str(report.get('status')))
check('two sockets', (report.get('runtime') or {}).get('sockets') == 2)
for socket, node in ((0, 2), (1, 3)):
    found = [r for r in regions if r['role'] == 'weights' and r['socket'] == socket]
    check(f'socket {socket} weights: one region on node {node}, fully verified, nothing elsewhere or missing',
          len(found) == 1 and found[0]['node'] == node and found[0]['verified'] == 'full' and
          (found[0]['pages_queried'] or 0) > 0 and found[0]['pages_mismatched'] == 0 and found[0]['pages_missing'] == 0,
          str(found))
for r in regions:
    if r['role'] == 'embedding':
        check('global embedding on DDR node 0', r['node'] == 0 and r['verified'] == 'full' and r['pages_mismatched'] == 0, str(r))
sys.exit(1 if bad else 0)
PY
}

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

# b_compare DIR NAME PRECISION STAGED LABEL ENVA... -- ENVB...: logits of one configuration against another at every
# position of the 86-token prompt (the gate of compare_engines.py: max |delta logit| <= 1e-3).
b_compare() {
  local dir=$1 name=$2 precision=$3 staged=$4 label=$5 side=a args=()
  shift 5
  while (($#)); do
    if [[ $1 == -- ]]; then side=b; else args+=("--env-$side" "$1"); fi
    shift
  done
  # One socket against two, C against C: no job-wide binding, so both teams can pin themselves. B_FLAGS replaces the strict
  # budget (W8A8 uses the statistical gate of the library).
  local flags=(--max-abs 1e-3)
  [[ ${#B_FLAGS[@]} -eq 0 ]] || flags=("${B_FLAGS[@]}")
  expect "compare-$name-$precision-$label" ok "$dir" -- "$PYTHON" "$ROOT/tools/compare_engines.py" --probe \
    "$ROOT/build/release/probe-$precision" --checkpoint "$staged" --tokens-file "$dir/$name-tokens.txt" "${args[@]}" \
    "${flags[@]}" --output "$dir/compare-$name-$precision-$label.json"
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

if ! dry_run; then
  check_stage
  validate_models
fi
run_phases
exit "$OVERALL"
