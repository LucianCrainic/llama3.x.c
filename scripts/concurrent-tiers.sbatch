#!/usr/bin/env bash
#SBATCH --job-name=concurrent-tiers
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=112
#SBATCH --exclusive
#SBATCH --mem=0
#SBATCH --time=06:00:00

# Concurrent placement of the decode GEMV weights (docs/concurrent-tiers.md): LLAMA_HYBRID_SPLIT=rows against hbm-only,
# serial layer-hybrid and ddr. Submit to one Flat node with both HBM nodes allowed; the cells of one job share the node.
# Required: ROOT (tree under test), OUT (new directory), STAGE, PYTHON.
#   STAGE=check   build ROOT, `make test`, then the bitwise checks: the per-position logits (decode and batched prefill) of the
#                 8B checkpoint through hbm-only, the layer hybrid and row splits, byte for byte, on one and two sockets, and
#                 the token traces of 8B hbm-only against a row split, and of the 70B all-HBM hybrid against a row split.
#   STAGE=bench   CONFIG cells, REPEATS interleaved repetitions, then one profiled run of each cell. CONFIG:
#       8b-1s, 8b-2s, 70b-2s   fits: reference (hbm-only for 8B, the 80/80 hybrid for 70B), ddr, the 80/80 hybrid with row
#                              splits r x DDR cores (R_LIST x C_LIST)
#       70b-1s                 spill: serial hybrid (most weight layers that fit), row split with the smallest fraction that
#                              fits x DDR cores (C_LIST), ddr, and EXTRA_CELLS "fraction:cores ..." at fixed fractions
#       70b-2s-long            spill by the KV caches at CONTEXT, synthetic K/V (needs a SYNTHETIC_KV build, done here)
# Required for the stages that load a model: MODEL_8B and MODEL_70B (directories holding int8-v3.bin and tokenizer-v2.bin;
# the stage uses the one it needs).
# Optional: REPEATS=5, WARM=4096, GENERATION (8B 256, 70B 96), R_LIST="0.1 0.15 0.2 0.25 0.3", C_LIST="8 14 20", EXTRA_CELLS,
# CONTEXT and WARM for 70b-2s-long, ONLY_CELLS (a list of cell names, for a repeat), DRY_RUN=1.
# Every cell has LLAMA_GEMV_STAGGER=1. Nothing here names a host, a path or a credential: locations arrive through the
# environment.
set -euo pipefail

: "${ROOT:?}" "${OUT:?}" "${STAGE:?}" "${PYTHON:?}"
repeats=${REPEATS:-5}
warm=${WARM:-4096}
r_list=${R_LIST:-0.1 0.15 0.2 0.25 0.3}
c_list=${C_LIST:-8 14 20}
prompt="Print a long numbered list of prime numbers. Keep listing primes for at least 1000 tokens. Start at 2."
[[ $STAGE == check || $STAGE == bench ]] || { echo "STAGE must be check or bench" >&2; exit 2; }
[[ ! -e $OUT ]] || { echo "OUT '$OUT' already exists" >&2; exit 2; }

base=(LLAMA_BACKEND=teams LLAMA_THREADS=56 LLAMA_PLACEMENT_STRICT=1 LLAMA_VERIFY_SAMPLES=16384 LLAMA_KV_VERIFY=1
  LLAMA_GEMV_STAGGER=1)
topology=explicit:0/2/0,1/3/1
cfg=()
bind=()
cells=()          # names, in order; cell_env[name] holds the cell's LLAMA_* settings
declare -A cell_env
reference=
checkpoint=
tokenizer=
generation=
context=
syn_env=()
syn_build=0
layers=0
sockets=1

add_cell() { cells+=("$1"); cell_env[$1]=$2; }

select_config() { # sets cfg, bind, checkpoint, tokenizer, generation, context, layers, cells, reference
  local c=$1 r k
  cells=(); cell_env=(); syn_env=(); syn_build=0
  case $c in
    8b-1s | 8b-2s)
      [[ -n ${MODEL_8B:-} ]] || { echo "MODEL_8B is required" >&2; exit 2; }
      checkpoint=$MODEL_8B/int8-v3.bin; tokenizer=$MODEL_8B/tokenizer-v2.bin
      layers=32; generation=${GENERATION:-256}; context=$((warm + 256))
      ;;
    70b-2s | 70b-1s | 70b-2s-long)
      [[ -n ${MODEL_70B:-} ]] || { echo "MODEL_70B is required" >&2; exit 2; }
      checkpoint=$MODEL_70B/int8-v3.bin; tokenizer=$MODEL_70B/tokenizer-v2.bin
      layers=80; generation=${GENERATION:-96}; context=$((warm + 256))
      ;;
    *) echo "CONFIG must be 8b-1s, 8b-2s, 70b-2s, 70b-1s or 70b-2s-long" >&2; exit 2 ;;
  esac
  cfg=("${base[@]}" "LLAMA_TOPOLOGY=$topology")
  bind=()
  case $c in
    8b-1s | 70b-1s) cfg+=(LLAMA_SOCKETS=1); bind=(numactl --cpunodebind=0); sockets=1 ;;
    *) cfg+=(LLAMA_SOCKETS=2); sockets=2 ;;
  esac
  local all_hbm="LLAMA_PLACEMENT=hybrid LLAMA_HYBRID_WEIGHTS_HBM_LAYERS=$layers LLAMA_HYBRID_KV_HBM_LAYERS=$layers LLAMA_HYBRID_CLASSIFIER=hbm LLAMA_HYBRID_PACKED=ddr"
  local rows_base="LLAMA_PLACEMENT=hybrid LLAMA_HYBRID_KV_HBM_LAYERS=$layers LLAMA_HYBRID_CLASSIFIER=hbm LLAMA_HYBRID_PACKED=ddr LLAMA_HYBRID_SPLIT=rows"
  case $c in
    8b-1s | 8b-2s)
      add_cell hbm-only "LLAMA_PLACEMENT=hbm-only"
      add_cell hyb-all-hbm "$all_hbm"
      add_cell ddr "LLAMA_PLACEMENT=ddr"
      for r in $r_list; do for k in $c_list; do add_cell "rows-r$r-c$k" "$rows_base LLAMA_HYBRID_DDR_FRACTION=$r LLAMA_HYBRID_DDR_CORES=$k"; done; done
      reference=hbm-only
      ;;
    70b-2s)
      add_cell hyb-all-hbm "$all_hbm"
      add_cell ddr "LLAMA_PLACEMENT=ddr"
      for r in $r_list; do for k in $c_list; do add_cell "rows-r$r-c$k" "$rows_base LLAMA_HYBRID_DDR_FRACTION=$r LLAMA_HYBRID_DDR_CORES=$k"; done; done
      reference=hyb-all-hbm
      ;;
    70b-1s)
      # the KV caches stay in HBM in every cell but ddr; the classifier is left to the planner (HBM when it fits)
      add_cell serial "LLAMA_PLACEMENT=hybrid LLAMA_HYBRID_KV_HBM_LAYERS=$layers LLAMA_HYBRID_PACKED=ddr"
      for k in ${C_LIST:-14 20 28}; do
        add_cell "rows-auto-c$k" "LLAMA_PLACEMENT=hybrid LLAMA_HYBRID_KV_HBM_LAYERS=$layers LLAMA_HYBRID_PACKED=ddr LLAMA_HYBRID_SPLIT=rows LLAMA_HYBRID_DDR_CORES=$k"
      done
      for x in ${EXTRA_CELLS:-}; do
        add_cell "rows-f${x%%:*}-c${x##*:}" "LLAMA_PLACEMENT=hybrid LLAMA_HYBRID_KV_HBM_LAYERS=$layers LLAMA_HYBRID_PACKED=ddr LLAMA_HYBRID_SPLIT=rows LLAMA_HYBRID_DDR_FRACTION=${x%%:*} LLAMA_HYBRID_DDR_CORES=${x##*:}"
      done
      add_cell ddr "LLAMA_PLACEMENT=ddr"
      reference=serial
      ;;
    70b-2s-long)
      warm=${WARM:-65536}; context=${CONTEXT:-98304}; generation=${GENERATION:-64}
      syn_build=1; syn_env=(LLAMA_SYNTHETIC_KV=1 LLAMA_SYNTHETIC_KV_PRIME=1024)
      add_cell serial "LLAMA_PLACEMENT=hybrid LLAMA_HYBRID_PACKED=ddr"
      for k in ${C_LIST:-14 20 28}; do
        add_cell "rows-auto-c$k" "LLAMA_PLACEMENT=hybrid LLAMA_HYBRID_KV_HBM_LAYERS=$layers LLAMA_HYBRID_PACKED=ddr LLAMA_HYBRID_SPLIT=rows LLAMA_HYBRID_DDR_CORES=$k"
      done
      for x in ${EXTRA_CELLS:-}; do
        add_cell "rows-f${x%%:*}-c${x##*:}" "LLAMA_PLACEMENT=hybrid LLAMA_HYBRID_KV_HBM_LAYERS=$layers LLAMA_HYBRID_PACKED=ddr LLAMA_HYBRID_SPLIT=rows LLAMA_HYBRID_DDR_FRACTION=${x%%:*} LLAMA_HYBRID_DDR_CORES=${x##*:}"
      done
      add_cell ddr "LLAMA_PLACEMENT=ddr"
      reference=serial
      ;;
  esac
  if [[ -n ${ONLY_CELLS:-} ]]; then
    local keep=() n
    for n in "${cells[@]}"; do [[ " $ONLY_CELLS " == *" $n "* ]] && keep+=("$n"); done
    cells=("${keep[@]}")
  fi
}

if [[ ${DRY_RUN:-0} == 1 ]]; then
  for c in ${CONFIG:-8b-1s}; do
    select_config "$c"
    echo "config=$c stage=$STAGE repeats=$repeats context=$context warm=$warm generation=$generation reference=$reference"
    printf 'env: %s\nbind: %s\n' "${cfg[*]}" "${bind[*]-}"
    for n in "${cells[@]}"; do printf 'cell %s: %s\n' "$n" "${cell_env[$n]}"; done
  done
  exit 0
fi
mkdir "$OUT"
exec > >(tee "$OUT/job.stdout") 2> >(tee "$OUT/job.stderr" >&2)
{
  printf 'job=%s\n' "${SLURM_JOB_ID:-}"
  hostname; date -u; uname -a; gcc --version | head -1; numactl -H
  printf '%s %s\n' "$ROOT" "$(cat "$ROOT/COMMIT" 2> /dev/null || git -C "$ROOT" rev-parse HEAD)"
  printf 'stage=%s config=%s repeats=%s warm=%s\n' "$STAGE" "${CONFIG:-}" "$repeats" "$warm"
  printf 'models: 8b=%s 70b=%s\n' "${MODEL_8B:-}" "${MODEL_70B:-}"
  grep -E '^(Cpus|Mems)_allowed_list' /proc/self/status
  perf --version || true
} > "$OUT/environment.txt"
(cd "$ROOT" && find src include tests tools scripts Makefile -type f ! -path '*/__pycache__/*' -print0 | sort -z |
  xargs -0 sha256sum) > "$OUT/source.sha256"

# Every LLAMA_* variable of the caller is cleared, so a cell sees exactly the environment written down.
unset_args=()
while IFS= read -r name; do unset_args+=(-u "$name"); done < <(env | sed -n 's/^\(LLAMA_[A-Za-z0-9_]*\)=.*/\1/p')
clean() { env "${unset_args[@]}" "$@"; }

# Intel PCM uses 32 bytes per HBM CAS on Sapphire Rapids and 64 bytes per DDR CAS.
reads='uncore_imc/event=0x05,umask=0xcf,name=ddr_cas_rd/,uncore_type_14/event=0x05,umask=0xcf,name=hbm_cas_rd/'
writes='uncore_imc/event=0x05,umask=0xf0,name=ddr_cas_wr/,uncore_type_14/event=0x05,umask=0xf0,name=hbm_cas_wr/'
events="$reads,$writes,unc_upi_rxl_flits.all_data,unc_upi_txl_flits.all_data,power/energy-pkg/,power/energy-ram/,cycles,ref-cycles"
interval=100
perf=(sudo -n /usr/bin/perf stat -a --per-socket -I "$interval" -x, -e "$events")

stage_check() {
  mkdir "$OUT/bin"
  # the gate as the campaign check stage runs it: engine knobs unset, one socket (CPUs and memory of node 0)
  clean numactl --cpunodebind=0 --membind=0 make -C "$ROOT" -j 56 CC=gcc BUILD_DIR="$OUT/bin" test > "$OUT/make-test.log" 2>&1 &&
    gate=0 || gate=$?
  printf 'make test rc=%s\n' "$gate" | tee "$OUT/make-test.rc"
  tail -5 "$OUT/make-test.log"
  status=$gate
  # the raw logits: a fixed 97-token input, decode path (probe) and batched prefill path (probe batch, every position)
  tokens=(128000 $(seq 1009 1301 125000))
  for c in 8b-1s 8b-2s; do
    select_config "$c"
    local probe_cells=(hbm-only hyb-all-hbm rows-r0.2-c14 rows-r0.3-c8 rows-r0.1-c20)
    for mode in decode batch; do
      for n in "${probe_cells[@]}"; do
        if [[ $mode == batch ]]; then
          clean "${cfg[@]}" ${cell_env[$n]} LLAMA_PREFILL_LOGITS=all LLAMA_PREFILL_BATCH=32 ${bind[@]+"${bind[@]}"} "$OUT/bin/probe-int8" batch "$checkpoint" "${tokens[@]}"
        else
          clean "${cfg[@]}" ${cell_env[$n]} ${bind[@]+"${bind[@]}"} "$OUT/bin/probe-int8" "$checkpoint" "${tokens[@]}"
        fi | sha256sum | cut -d' ' -f1 > "$OUT/logits-$c-$mode-$n.sha256"
      done
      sort -u "$OUT"/logits-"$c"-"$mode"-*.sha256 | wc -l > "$OUT/logits-distinct-$c-$mode.txt"
      printf 'logits %s %s: %s distinct digest(s) over %s cells\n' "$c" "$mode" "$(cat "$OUT/logits-distinct-$c-$mode.txt")" "${#probe_cells[@]}"
      [[ $(cat "$OUT/logits-distinct-$c-$mode.txt") == 1 ]] || status=1
      ! grep -q '^e3b0c442' "$OUT"/logits-"$c"-"$mode"-*.sha256 || status=1 # the digest of no output: a probe failed
    done
  done
  # token traces: 8B hbm-only against the row split of r = 0.2, on one and two sockets; 70B all-HBM hybrid against the split
  generation=64; warm=1024
  for c in 8b-1s 8b-2s 70b-2s; do
    select_config "$c"
    [[ $c != 70b-2s ]] && generation=64 || generation=32
    context=$((warm + 256))
    for n in "$reference" rows-r0.2-c14 rows-r0.2-c20; do
      run_cell "trace-$c-$n" "$n" || status=1
    done
    "$PYTHON" - "$OUT" "$c" << 'PY' || status=1
import json, sys
from pathlib import Path
out, config = Path(sys.argv[1]), sys.argv[2]
traces = {p.name: json.loads(p.read_text()) for p in sorted(out.glob(f'trace-{config}-*.tokens.json'))}
reference = next(iter(traces.values()))
bad = [n for n, t in traces.items() if t != reference]
line = f'{config}: {len(traces)} token traces, {len(bad)} differ: {bad}'
print(line)
with open(out / 'tokens-identical.txt', 'a') as f:
    f.write(line + '\n')
sys.exit(1 if bad or len(traces) < 3 else 0)
PY
  done
  return "$status"
}

# run_cell LABEL CELL [EXTRA_ENV...]: one benchmark run with perf counters, the plan and its placement check
run_cell() {
  local label=$1 name=$2 prefix status=0 pid perfpid
  shift 2
  prefix="$OUT/$label"
  local binary="$OUT/bin/runq"
  [[ $syn_build == 0 ]] || binary="$OUT/bin-syn/runq"
  printf '%s %s\n' "$(date -Is)" "$label"
  clean "${cfg[@]}" ${cell_env[$name]} "$binary" "$checkpoint" --context-length "$context" --plan-only --json "$prefix.plan.json" \
    > "$prefix.plan.stdout" 2> "$prefix.plan.stderr" || { cat "$prefix.plan.stderr" >&2; return 1; }
  date +%s%N > "$prefix.launch"
  clean "${cfg[@]}" ${cell_env[$name]} ${syn_env[@]+"${syn_env[@]}"} "$@" LLAMA_TOKENS_OUT="$prefix.tokens.json" \
    ${bind[@]+"${bind[@]}"} "$binary" "$checkpoint" -z "$tokenizer" --benchmark --chat-template --no-text -t 0 -n 0 \
    --context-length "$context" --warm-context "$warm" --generation-length "$generation" --prompt "$prompt" \
    --json "$prefix.json" > "$prefix.stdout" 2> "$prefix.stderr" &
  pid=$!
  date +%s%N > "$prefix.perf-start"
  "${perf[@]}" -- tail -s 0.05 --pid="$pid" -f /dev/null > /dev/null 2> "$prefix.perf.csv" &
  perfpid=$!
  wait "$pid" || status=$?
  date +%s%N > "$prefix.end"
  wait "$perfpid" || true
  date +%s%N > "$prefix.perf-end"
  if [[ $status != 0 ]]; then
    cat "$prefix.stderr" >&2
    return "$status"
  fi
  "$PYTHON" "$ROOT/tools/check_placement.py" "$prefix.json" --planned --allow-sampled > "$prefix.placement" ||
    { cat "$prefix.placement" >&2; return 1; }
  "$PYTHON" - "$prefix.json" "$generation" "${#syn_env[@]}" << 'PY'
import json, sys
with open(sys.argv[1]) as stream:
    report = json.load(stream)
expected = int(sys.argv[2]) - 1
if bool(report["parameters"].get("synthetic_kv")) != bool(int(sys.argv[3])):
    raise SystemExit(f"{sys.argv[1]}: unexpected KV population method")
# the model may end its answer (end of text) before the limit: accept a run with at least 64 timed decode steps
if report["tokens"]["decode_steps"] < min(expected, 64) or report["stop_reason"] not in ("generation_limit", "step_limit", "end_of_text", "eos"):
    raise SystemExit(f"{sys.argv[1]}: stop {report['stop_reason']}, {report['tokens']['decode_steps']} decode steps (wanted {expected})")
PY
}

stage_bench() {
  select_config "$CONFIG"
  mkdir "$OUT/bin"
  make -C "$ROOT" -j 56 CC=gcc BUILD_DIR="$OUT/bin" all > "$OUT/build.log" 2>&1 || { tail -30 "$OUT/build.log"; exit 1; }
  if [[ $syn_build == 1 ]]; then
    mkdir "$OUT/bin-syn"
    make -C "$ROOT" -j 56 CC=gcc SYNTHETIC_KV=1 BUILD_DIR="$OUT/bin-syn" all > "$OUT/build-syn.log" 2>&1 || { tail -30 "$OUT/build-syn.log"; exit 1; }
  fi
  sha256sum "$OUT/bin/runq" "$0" > "$OUT/executables.sha256"
  printf 'config=%s context=%s warm=%s generation=%s reference=%s\ncells: %s\nengine env: %s\n' "$CONFIG" "$context" "$warm" \
    "$generation" "$reference" "${cells[*]}" "${cfg[*]}" >> "$OUT/environment.txt"
  for n in "${cells[@]}"; do printf 'cell %s: %s\n' "$n" "${cell_env[$n]}" >> "$OUT/environment.txt"; done
  printf 'interval_ms=%s\nevents=%s\n' "$interval" "$events" >> "$OUT/environment.txt"
  "${perf[@]}" -- sleep 5 2> "$OUT/idle.perf.csv"
  local rep i n
  for ((rep = 0; rep < repeats; rep++)); do
    for ((i = 0; i < ${#cells[@]}; i++)); do
      n=${cells[(rep + i) % ${#cells[@]}]}
      run_cell "r$((rep + 1))-$n" "$n"
    done
  done
  for n in "${cells[@]}"; do
    run_cell "prof-$n" "$n" LLAMA_PROFILE=1
  done
  "$PYTHON" - "$OUT" << 'PY'
import json, sys
from pathlib import Path
out = Path(sys.argv[1])
traces = {p.name: json.loads(p.read_text()) for p in sorted(out.glob('*.tokens.json'))}
reference = next(iter(traces.values()))
bad = [n for n, t in traces.items() if t != reference]
(out / 'tokens-identical.txt').write_text(f'{len(traces)} token traces, {len(bad)} differ: {bad}\n')
print(f'{len(traces)} token traces, {len(bad)} differ')
sys.exit(1 if bad else 0)
PY
}

if [[ $STAGE == check ]]; then
  # the check stage builds the probes; the benchmark binaries come from the same build directory
  stage_check
else
  stage_bench
fi
