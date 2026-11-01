# shellcheck shell=bash
# Shared steps of the CRESCO8 Flat-mode campaigns (placement, tensor parallel, prefill). Source this file from an
# sbatch script after `set -euo pipefail`. The functions follow scripts/cresco8-kernels.sbatch step for step; only
# the tree variables differ (one ROOT instead of SIMD_ROOT and BASE_ROOT). Hosts, gateways and credentials never
# belong here, and model locations only ever arrive through MODELS="NAME:ARTIFACTS:HF ...".

die() {
  printf '%s\n' "$*" >&2
  exit 1
}

# Required: ROOT (tree under test), OUT (must not exist yet), PYTHON. Optional: MODELS (without it the Hugging Face
# gate and every model-based run is skipped and recorded as such), STAGE=check|bench (default bench), REPEATS
# (default 3), PRECISIONS (default "fp32 int8"), PHASE (default none): a space-separated list of phase ids.
require_env() {
  : "${SLURM_JOB_ID:?submit through Slurm}"
  : "${ROOT:?tree under test required}"
  : "${OUT:?unique results directory required}"
  : "${PYTHON:?validation Python required}"
  MODELS=${MODELS:-}
  STAGE=${STAGE:-bench}
  REPEATS=${REPEATS:-3}
  PRECISIONS=${PRECISIONS:-"fp32 int8"}
  PHASE=${PHASE:-none}
  [[ $STAGE == check || $STAGE == bench ]] || die "STAGE must be check or bench, got '$STAGE'"
  [[ $REPEATS =~ ^[1-9][0-9]*$ ]] || die "REPEATS must be a positive integer, got '$REPEATS'"
  local phase
  for phase in $PHASE; do
    [[ $phase =~ ^(none|[A-DF][0-9]+)$ ]] || die "PHASE must be none or phase ids such as 'A1 A3 F2', got '$PHASE'"
  done
  [[ ! -e $OUT ]] || die "OUT '$OUT' already exists"
  for entry in $MODELS; do
    [[ $entry =~ ^[^:]+:[^:]+:[^:]+$ ]] || die "MODELS entry '$entry' is not NAME:ARTIFACTS:HF"
  done
  export STAGE REPEATS PRECISIONS PHASE MODELS
}

# Creates OUT, records the exit status, removes the staging directory, and tees job output into OUT.
job_begin() {
  mkdir "$OUT"
  shm=
  finish() {
    local status=$?
    printf 'exit_code=%s\n' "$status" > "$OUT/exit-status.txt"
    [[ -z $shm ]] || rm -rf -- "$shm"
    exit "$status"
  }
  trap finish EXIT
  exec > >(tee "$OUT/job.stdout") 2> >(tee "$OUT/job.stderr" >&2)
  # OMP_PROC_BIND is not exported job-wide: with it set, the OpenMP runtime pins the launching thread of every
  # process that links it to one core at load, and teams workers would inherit that single-CPU mask. run_engine sets
  # it for omp runs only.
  export OMP_DYNAMIC=FALSE
  export USE_TF=0 USE_FLAX=0 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
}

record_environment() {
  {
    date -u +%FT%TZ; hostname; uname -a; lscpu; numactl -H
    scontrol show job "$SLURM_JOB_ID"
    gcc --version; pkg-config --modversion libpcre2-8
    printf "%s %s\n" "$ROOT" "$(cat "$ROOT/COMMIT" 2> /dev/null || git -C "$ROOT" rev-parse HEAD)"
    printf 'STAGE=%s REPEATS=%s PRECISIONS=%s PHASE=%s MODELS=%s\n' "$STAGE" "$REPEATS" "$PRECISIONS" "$PHASE" "$MODELS"
  } > "$OUT/environment.txt"
  (cd "$ROOT" && find src include tests tools scripts Makefile -type f ! -path '*/__pycache__/*' -print0 | sort -z |
    xargs -0 sha256sum) > "$OUT/source-$(basename "$ROOT").sha256"
}

# Flat Mode evidence: no ACPI memory-side cache and two CPU-less ~64 GiB HBM nodes.
check_flat_mode() {
  python3 - << 'PY' > "$OUT/flat-mode.json"
import glob, json
from pathlib import Path
assert not glob.glob('/sys/devices/system/node/node*/memory_side_cache/index*/size'), 'memory-side cache present'
nodes = {}
for path in sorted(glob.glob('/sys/devices/system/node/node[0-9]*')):
    total = int(Path(path, 'meminfo').read_text().split('MemTotal:')[1].split()[0]) * 1024
    nodes[Path(path).name] = dict(cpus=Path(path, 'cpulist').read_text().strip(), mem_bytes=total)
assert set(nodes) == {'node0', 'node1', 'node2', 'node3'}, nodes
assert nodes['node2']['cpus'] == '' and nodes['node3']['cpus'] == '', nodes
assert all(60e9 < nodes[n]['mem_bytes'] < 70e9 for n in ('node2', 'node3')), nodes
assert 'Max 9480' in Path('/proc/cpuinfo').read_text(), 'unexpected CPU model'
print(json.dumps(dict(mode='flat', ddr_nodes=[0, 1], hbm_nodes=[2, 3], nodes=nodes), indent=2))
PY
}

# build_tree OPENMP MODE [targets...]; the log and the build configuration go to OUT.
build_tree() {
  local openmp=$1 mode=$2
  shift 2
  make -C "$ROOT" CC=gcc OPENMP="$openmp" MODE="$mode" all "$@" > "$OUT/build-$(basename "$ROOT").log"
  cp "$ROOT/build/$mode/config" "$OUT/build-config-$(basename "$ROOT").txt"
}

# Uncore CAS counters (64-byte lines) show which tier served the reads. Perf is optional: whether it ran is recorded.
EVENTS='uncore_imc/event=0x05,umask=0xcf,name=ddr_cas_rd/,uncore_type_14/event=0x05,umask=0xcf,name=hbm_cas_rd/,unc_upi_rxl_flits.all_data'
PERF=0
perf_available() {
  PERF=0
  if sudo -n /usr/bin/perf stat -a -x , -e "$EVENTS" -- true 2> /dev/null; then PERF=1; fi
  printf 'perf_uncore=%s\n' "$PERF" >> "$OUT/environment.txt"
}

# perf_attach PID PREFIX: counts until PID exits; never fails the run.
perf_attach() {
  [[ $PERF == 1 ]] || return 0
  sudo -n /usr/bin/perf stat -a --per-socket -x , -e "$EVENTS" -- tail -s 0.1 --pid="$1" -f /dev/null \
    2> "$2.perf.csv" || true
}

# stage_checkpoint SOURCE DESTINATION numactl-policy...: tmpfs staging under an explicit memory policy. Kept for
# baselines; the engine-owned placement of track A does not use it.
stage_checkpoint() {
  local source=$1 destination=$2
  shift 2
  numactl "$@" cp "$source" "$destination"
  numactl -H > "$destination.numa.txt"
}

# run_label MODEL PRECISION BACKEND PLACEMENT SOCKETS THREADS [m=M] [p=PROMPT] [c=CTX] [k=KERNEL]
#   {model}-{precision}-{backend}-{placement}-s{sockets}-t{threads}[-m{M}][-p{prompt}][-c{ctx}][-k{kernel}]
# (the repeat suffix -{rep} is added by run_engine; tools/summarize_campaign.py parses these labels).
run_label() {
  local label="$1-$2-$3-$4-s$5-t$6" extra
  shift 6
  for extra in m p c k; do
    local value
    for arg in "$@"; do
      [[ $arg == "$extra="* ]] && value=${arg#*=} && label+="-$extra$value"
    done
  done
  printf '%s\n' "$label"
}

# run_engine LABEL ENGINE CHECKPOINT TOKENIZER PROMPT_FILE REPEATS PERF [env...]
# One benchmark label, REPEATS independent launches, with the workload of the kernel campaigns.
# The caller sets the CPU and memory binding through CPU_BIND and MEM_BIND (numactl arguments), or leaves them unset
# for engine-owned placement, where the job must not impose a memory policy.
run_engine() {
  local label=$1 engine=$2 checkpoint=$3 tokenizer=$4 prompt=$5 repeats=$6 perf=$7
  shift 7
  local binding=()
  [[ -z ${CPU_BIND:-} ]] || binding+=(--cpunodebind="$CPU_BIND")
  [[ -z ${MEM_BIND:-} ]] || binding+=(--membind="$MEM_BIND")
  local launcher=()
  [[ ${#binding[@]} -eq 0 ]] || launcher=(numactl "${binding[@]}")
  local r
  for ((r = 1; r <= repeats; r++)); do
    local prefix="$OUT/$label-$r"
    env OMP_PROC_BIND=close OMP_PLACES=cores "$@" "${launcher[@]}" \
      "$ROOT/build/release/$engine" "$checkpoint" -z "$tokenizer" --context-length "${CONTEXT:-2048}" --benchmark \
      --chat-template --prompt-file "$prompt" -t 0 --seed 42 --generation-length "${GENERATION:-128}" --no-text \
      --log-level warn --json "$prefix.json" > "$prefix.stdout" 2> "$prefix.stderr" &
    local pid=$!
    if [[ $perf == 1 && $r == 1 ]]; then perf_attach "$pid" "$prefix"; fi
    wait "$pid"
  done
}

# validate_gate NAME ARTIFACTS HF PRECISION LABEL [env...]
# HF validation with the engine under test. FP32 failure is fatal; INT8 results are recorded against the unchanged
# budget without gating (W8A8 requantizes activations, so FP32 reordering moves its HF deviation).
validate_gate() {
  local name=$1 artifacts=$2 hf=$3 precision=$4 label=$5 status=0
  shift 5
  local extra=()
  [[ $precision != int8 ]] || extra+=(--max-abs 1.0 --rtol 0.05)
  (cd "$ROOT" && env "$@" OMP_PROC_BIND=false OMP_NUM_THREADS=56 numactl --cpunodebind=0 --membind=0 \
    "$PYTHON" tools/validate.py --hf "$hf" --checkpoint "$artifacts/$precision-v3.bin" \
    --tokenizer "$artifacts/tokenizer-v2.bin" --precision "$precision" --threads 56 \
    --output "$OUT/correctness-$name-$label.json" "${extra[@]}") \
    > "$OUT/correctness-$name-$label.stdout" 2> "$OUT/correctness-$name-$label.stderr" || status=$?
  printf '%s %s exit=%s\n' "$name" "$label" "$status" >> "$OUT/correctness-status.txt"
  return "$status"
}

# make_prompt PROBE TOKENIZER VOCAB [LENGTHS|none]: short.txt (86 tokens) and prompt-{512,2048,8192,32768}.txt in OUT,
# with their achieved token counts and SHA-256 in prompts.json. LENGTHS is a comma list, or none for short.txt only.
make_prompt() {
  python3 "$ROOT/tools/make_prompts.py" --probe "$1" --tokenizer "$2" --vocab "$3" --out "$OUT" \
    --lengths "${4:-512,2048,8192,32768}" > /dev/null
}

# validate_models: the correctness gate of every model and precision, as in the kernel campaign. HF_GATE=0 skips it
# (and records that) where no Python environment with torch and transformers exists.
validate_models() {
  local entry name artifacts hf precision
  if [[ ${HF_GATE:-1} == 0 ]]; then
    echo 'HF_GATE=0: Hugging Face validation was not run (no torch/transformers environment)' | tee "$OUT/no-hf-gate.txt"
    return 0
  fi
  if [[ -z $MODELS ]]; then
    echo 'MODELS is empty: Hugging Face validation and model benchmarks were not run' | tee "$OUT/no-models.txt"
    return 0
  fi
  for entry in $MODELS; do
    IFS=: read -r name artifacts hf <<< "$entry"
    for precision in $PRECISIONS; do
      if [[ $precision == fp32 ]]; then
        validate_gate "$name" "$artifacts" "$hf" fp32 fp32
      else
        validate_gate "$name" "$artifacts" "$hf" int8 w8a8 || true
        validate_gate "$name" "$artifacts" "$hf" int8 w8a32 LLAMA_Q8_ACTIVATIONS=fp32 || true
      fi
    done
  done
}

# OVERALL is the job's exit status: any failed expectation or unit test makes it non-zero, but the evidence that is
# still collectable is collected.
OVERALL=0

# record_result NAME PASS|FAIL [detail]: one line in phase-results.txt.
record_result() {
  printf '%s %s %s\n' "$2" "$1" "${3:-}" >> "$OUT/phase-results.txt"
  [[ $2 == PASS ]] || OVERALL=1
}

# expect NAME ok|fail|any DIR -- COMMAND...: runs the command with output in DIR/NAME.{out,err} and checks its exit
# status against the expectation ("any" records whatever happened); the command never stops the job.
expect() {
  local name=$1 want=$2 dir=$3 rc=0
  shift 4
  "$@" > "$dir/$name.out" 2> "$dir/$name.err" || rc=$?
  if [[ $want == any ]] || [[ $want == ok && $rc -eq 0 ]] || [[ $want == fail && $rc -ne 0 ]]; then
    record_result "$name" PASS "exit=$rc expected=$want"
  else
    record_result "$name" FAIL "exit=$rc expected=$want"
  fi
  return 0
}

# run_phases: runs phase_<ID> for each id in PHASE; phase_none does nothing. An unknown phase is an error.
run_phases() {
  local phase
  for phase in $PHASE; do
    if [[ $phase == none ]]; then
      continue
    elif declare -F "phase_$phase" > /dev/null; then
      mkdir -p "$OUT/$phase"
      "phase_$phase"
    else
      die "phase $phase has no run matrix in $(basename "${BASH_SOURCE[1]:-this script}") yet"
    fi
  done
}

# F2, model-free part: kernels on teams at full-socket size on the hardware ISA, the pinning read-back, and the
# per-barrier cost. The real-model gate and the omp-versus-teams decode comparison need MODELS and a Hugging Face
# Python environment and are not part of this phase's model-free check.
phase_F2() {
  local dir="$OUT/F2" bin="$ROOT/build/release"
  expect kernels-teams ok "$dir" -- numactl --cpunodebind=0 env LLAMA_TEST_TEAMS=1,56 "$bin/test-kernels"
  if grep -q 'avx512 teams:56 ok' "$dir/kernels-teams.out"; then
    record_result kernels-teams-avx512-56 PASS
  else
    record_result kernels-teams-avx512-56 FAIL 'avx512 teams:56 was not exercised'
  fi
  expect affinity ok "$dir" -- numactl --cpunodebind=0 "$bin/test-workers" affinity 56
  if python3 - "$dir/affinity.out" << 'PY'; then
import json, sys
data = json.load(open(sys.argv[1]))
assert data['supported'] and data['verified'] == 1 and len(data['workers']) == 56, data
assert all(w['assigned'] == w['observed'] for w in data['workers'])
PY
    record_result affinity-readback PASS 'every worker read back as assigned'
  else
    record_result affinity-readback FAIL
  fi
  expect barrier-cost ok "$dir" -- numactl --cpunodebind=0 "$bin/test-workers" bench-barrier 200000 56
  expect barrier-stress ok "$dir" -- numactl --cpunodebind=0 "$bin/test-workers" barrier 1000000 56
  [[ -z $MODELS ]] || phase_F2_models
}

# stage_model NAME PRECISION ARTIFACTS: copies a checkpoint into tmpfs ($shm, created by the caller in the parent
# shell) with its pages on HBM node 2 (socket 0's HBM) and prints the staged path.
stage_model() {
  local staged="$shm/$1-$2.bin"
  [[ -e $staged ]] || stage_checkpoint "$3/$2-v3.bin" "$staged" --membind=2
  printf '%s\n' "$staged"
}

# F2 with models, one socket (56 threads, weights staged on its HBM): the omp and teams backends agree at every
# position of an 86-token prompt (C against C, same ISA), and decode speed of teams against omp over REPEATS runs.
# The Hugging Face gate of a backend that agrees with the validated one is not repeated here (HF_GATE=0 records it).
phase_F2_models() {
  local dir="$OUT/F2" bin="$ROOT/build/release" entry name artifacts hf precision engine staged tokenizer label
  [[ -n $shm ]] || shm=$(mktemp -d "/dev/shm/llama-campaign-$SLURM_JOB_ID-XXXXXX")
  for entry in $MODELS; do
    IFS=: read -r name artifacts hf <<< "$entry"
    tokenizer="$artifacts/tokenizer-v2.bin"
    make_prompt "$bin/probe-fp32" "$tokenizer" 128256 none
    "$bin/probe-fp32" tokenize "$tokenizer" 128256 "@$OUT/short.txt" > "$dir/$name-tokens.txt"
    for precision in $PRECISIONS; do
      engine=run
      [[ $precision == fp32 ]] || engine=runq
      staged=$(stage_model "$name" "$precision" "$artifacts")
      expect "compare-$name-$precision-omp-vs-teams" ok "$dir" -- numactl --cpunodebind=0 --membind=2 "$PYTHON" \
        "$ROOT/tools/compare_engines.py" --probe "$bin/probe-$precision" --checkpoint "$staged" \
        --tokens-file "$dir/$name-tokens.txt" --env-a LLAMA_BACKEND=omp --env-a OMP_NUM_THREADS=56 \
        --env-b LLAMA_BACKEND=teams --env-b LLAMA_THREADS=56 --env-b OMP_PROC_BIND=false --output "$dir/compare-$name-$precision.json"
      export CPU_BIND=0 MEM_BIND=2
      label=$(run_label "$name" "$precision" omp file 1 56)
      run_engine "$label" "$engine" "$staged" "$tokenizer" "$OUT/short.txt" "$REPEATS" 0 LLAMA_BACKEND=omp OMP_NUM_THREADS=56
      label=$(run_label "$name" "$precision" teams file 1 56)
      run_engine "$label" "$engine" "$staged" "$tokenizer" "$OUT/short.txt" "$REPEATS" 0 LLAMA_BACKEND=teams \
        LLAMA_THREADS=56 OMP_PROC_BIND=false
      unset CPU_BIND MEM_BIND
      rm -f -- "$staged" # HBM node 2 holds 64 GiB: one staged checkpoint at a time
    done
  done
  "$PYTHON" "$ROOT/tools/summarize_campaign.py" "$OUT" > "$dir/summary.md"
}

# check_stage: everything every skeleton does before a phase matrix: environment, Flat-mode check, build, tests.
check_stage() {
  record_environment
  check_flat_mode
  build_tree 1 release build/release/probe-fp32 build/release/probe-int8
  local status=0
  (cd "$ROOT" && OMP_NUM_THREADS=56 numactl --cpunodebind=0 --membind=0 make CC=gcc OPENMP=1 test) > "$OUT/test.log" 2>&1 ||
    status=$?
  if [[ $status -eq 0 ]]; then record_result make-test PASS; else record_result make-test FAIL "exit=$status"; fi
  perf_available
}
