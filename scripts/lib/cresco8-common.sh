# shellcheck shell=bash
# Shared steps of the CRESCO8 Flat-mode campaigns (placement, tensor parallel, prefill). Source this file from an
# sbatch script after `set -euo pipefail`. The functions follow scripts/cresco8-kernels.sbatch step for step; only
# the tree variables differ (one ROOT instead of SIMD_ROOT and BASE_ROOT). Hosts, gateways and credentials never
# belong here, and model locations only ever arrive through MODELS="NAME:ARTIFACTS:HF ...".

die() {
  printf '%s\n' "$*" >&2
  exit 1
}

# Required: ROOT (tree under test), OUT (must not exist yet), PYTHON, MODELS. Optional: STAGE=check|bench (default
# bench), REPEATS (default 3), PRECISIONS (default "fp32 int8"), PHASE (default none).
require_env() {
  : "${SLURM_JOB_ID:?submit through Slurm}"
  : "${ROOT:?tree under test required}"
  : "${OUT:?unique results directory required}"
  : "${PYTHON:?validation Python required}"
  : "${MODELS:?NAME:ARTIFACTS:HF entries required}"
  STAGE=${STAGE:-bench}
  REPEATS=${REPEATS:-3}
  PRECISIONS=${PRECISIONS:-"fp32 int8"}
  PHASE=${PHASE:-none}
  [[ $STAGE == check || $STAGE == bench ]] || die "STAGE must be check or bench, got '$STAGE'"
  [[ $REPEATS =~ ^[1-9][0-9]*$ ]] || die "REPEATS must be a positive integer, got '$REPEATS'"
  [[ $PHASE =~ ^(none|[A-D][0-9]+)$ ]] || die "PHASE must be none or a phase id such as A3, got '$PHASE'"
  [[ ! -e $OUT ]] || die "OUT '$OUT' already exists"
  for entry in $MODELS; do
    [[ $entry =~ ^[^:]+:[^:]+:[^:]+$ ]] || die "MODELS entry '$entry' is not NAME:ARTIFACTS:HF"
  done
  export STAGE REPEATS PRECISIONS PHASE
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
  export OMP_DYNAMIC=FALSE OMP_PROC_BIND=close OMP_PLACES=cores
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
    env "$@" "${launcher[@]}" \
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

# make_prompt PROBE TOKENIZER VOCAB: short.txt (86 tokens) and prompt-{512,2048,8192,32768}.txt in OUT, with their
# achieved token counts and SHA-256 in prompts.json.
make_prompt() {
  python3 "$ROOT/tools/make_prompts.py" --probe "$1" --tokenizer "$2" --vocab "$3" --out "$OUT" > /dev/null
}

# validate_models: the correctness gate of every model and precision, as in the kernel campaign.
validate_models() {
  local entry name artifacts hf precision
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

# check_stage: everything every skeleton does before a phase matrix: environment, Flat-mode check, build, tests.
check_stage() {
  record_environment
  check_flat_mode
  build_tree 1 release build/release/probe-fp32 build/release/probe-int8
  (cd "$ROOT" && OMP_NUM_THREADS=56 numactl --cpunodebind=0 --membind=0 make CC=gcc OPENMP=1 test) > "$OUT/test.log" 2>&1
  perf_available
}
