# shellcheck shell=bash
# The steps of scripts/xengine.sbatch, the cross-engine campaign (llama3.x.c, llama.cpp, vLLM CPU) on a CRESCO8 Flat or
# Cache node (plan: docs/plans/xengine-comparison.md). Source this file after `set -euo pipefail`. Host names, gateways,
# credentials and cluster paths never belong here: every location arrives through an environment variable.
#
# Reused from cresco8-common.sh (unchanged): die, dry_run, job_begin, record_environment, check_flat_mode, build_tree,
# need_shm, stage_model_on, unstage_model_on, record_result and OVERALL. The DRY_RUN pattern is the same: every cell prints
# its commands (to OUT/dry-run.txt too) and nothing runs.
#
# Layout of one run: OUT/runs/<engine>-s<S>-<mem>-<workload>-r<rep>/
#   stdout stderr report.json (the engine's own report) perf.csv perf.err perf.cmd anchors markers.json numa_maps.json
#   numastat.txt check.txt cmd.txt cell.json (its manifest entry; OUT/manifest.json merges them)
#
# Adding an engine (llamacpp, vllm) is one function plus a few optional hooks, all named by the engine id:
#   run_<engine>_cell S MEM WL REP   required. Sets CELL_CMD (the full command line, arrays only), and optionally CELL_ENV
#                                    (VAR=value words), CELL_VERSION, CELL_CONFIG (text for the manifest), CELL_GEN (tokens to
#                                    generate), CELL_PROC (process name, for the wrapper perf mode), CELL_LOAD_PATTERN (a
#                                    string its stderr shows when loading is over), CELL_CHECK (a command that verifies the
#                                    result, exit status 0 = pass), then calls xe_execute_cell. Output files go to $CELL_DIR.
#   <engine>_matrix                  optional; prints "S MEM WL..." lines (default: every MEM x every SOCKETS x WORKLOADS)
#   <engine>_prepare                 optional; builds, stages the model, checks inputs once per job (non-zero = skip engine)
#   <engine>_begin_config S MEM / <engine>_end_config S MEM   optional; start and stop a server once per configuration
#                                    (XE_CONFIG_VARIANT and XE_CONFIG_WLS describe the configuration; a matrix line may name a
#                                    variant as MEM:VARIANT, which becomes part of the cell label and the manifest)
#   <engine>_describe_config S MEM   optional; prints the server command for DRY_RUN
#   <engine>_finalize                optional; runs after the process ended, before markers and checks (e.g. builds report.json)
#   CELL_ANCHORS (array of key=value) adds anchors; CELL_PERF_ATTACH (a function printing pids) makes the Cache node's
#                                    user-space perf attach with -p to those pids instead of wrapping the command
#   <engine>_estimate_s S MEM WL REP and <engine>_config_overhead_s S MEM   seconds, for the DRY_RUN estimate
#   <engine>_environment             optional; writes versions to stdout (appended to environment.txt)
# and a reader of its report in tools/xengine_summary.py (PHASE_READERS) plus a marker deriver in tools/xengine_cell.py.

# shellcheck source=cresco8-common.sh
source "${BASH_SOURCE[0]%/*}/cresco8-common.sh"

# Counter groups, no multiplexing (verified on the Flat nodes). MEM: DDR and HBM CAS reads, UPI data flits in both
# directions, RAPL package and DRAM energy. COMPUTE: core cycles, reference cycles, instructions, AMX activity, RAPL.
# AMX_OPS_RETIRED.INT8 reads 0 on this CPU but did not cause multiplexing; XE_AMX_OPS=0 leaves it out.
XE_MEM_EVENTS='uncore_imc/event=0x05,umask=0xcf,name=ddr_cas_rd/,uncore_type_14/event=0x05,umask=0xcf,name=hbm_cas_rd/,unc_upi_rxl_flits.all_data,unc_upi_txl_flits.all_data,power/energy-pkg/,power/energy-ram/'
XE_COMPUTE_EVENTS='cycles,ref-cycles,instructions,EXE.AMX_BUSY,power/energy-pkg/,power/energy-ram/'
# The Cache node has no privileged counters: per-process, user-space only.
XE_USER_EVENTS='cycles,instructions,EXE.AMX_BUSY'
XE_PERF_INTERVAL_MS=200

xe_now_ns() {
  date +%s%N
}

xe_in_list() {
  local needle=$1 item
  shift
  for item in "$@"; do [[ $item == "$needle" ]] && return 0; done
  return 1
}

# Required: ROOT, OUT (must not exist), NODE_ROLE (A|B|cache). Optional: ENGINES (default "c"; the full set is
# "c llamacpp vllm"), SOCKETS ("1 2"), MEMS (role default: A hbm, B ddr, cache cache), WORKLOADS ("P D L"), REPEATS (3),
# PYTHON (python3), DRY_RUN, ANCHOR (role B: add the c / 1S / HBM / D anchor cell, default 1), XE_PLACEMENT_REPS (repeats
# that record numa_maps and numastat, default "1"), XE_AMX_OPS (1), PERF_BIN (/usr/bin/perf), UPERF (user-space perf,
# Cache role). Engine inputs: ARTIFACTS (int8-v3.bin and tokenizer-v2.bin), PROMPTS_DIR (short-86.txt, prompt-2048.txt,
# prompt-8192.txt), C_ENGINE (default $ROOT/build/release/runq, built in the job when missing), GGUF, LLAMACPP_BIN, VLLM_*.
xe_require_env() {
  : "${SLURM_JOB_ID:?submit through Slurm}"
  : "${ROOT:?staged repository copy required}"
  : "${OUT:?unique campaign directory required}"
  : "${NODE_ROLE:?NODE_ROLE required (A, B or cache)}"
  DRY_RUN=${DRY_RUN:-0}
  [[ $DRY_RUN == 0 || $DRY_RUN == 1 ]] || die "DRY_RUN must be 0 or 1, got '$DRY_RUN'"
  PYTHON=${PYTHON:-python3}
  ENGINES=${ENGINES:-c}
  SOCKETS=${SOCKETS:-"1 2"}
  WORKLOADS=${WORKLOADS:-"P D L"}
  REPEATS=${REPEATS:-3}
  ANCHOR=${ANCHOR:-1}
  XE_PLACEMENT_REPS=${XE_PLACEMENT_REPS:-1}
  XE_AMX_OPS=${XE_AMX_OPS:-1}
  PERF_BIN=${PERF_BIN:-/usr/bin/perf}
  case $NODE_ROLE in
    A) MEMS=${MEMS:-hbm} ;;
    B) MEMS=${MEMS:-ddr} ;;
    cache) MEMS=${MEMS:-cache} ;;
    *) die "NODE_ROLE must be A, B or cache, got '$NODE_ROLE'" ;;
  esac
  local value
  for value in $ENGINES; do xe_in_list "$value" c llamacpp vllm || die "ENGINES must list c, llamacpp and/or vllm, got '$ENGINES'"; done
  for value in $SOCKETS; do [[ $value == 1 || $value == 2 ]] || die "SOCKETS must list 1 and/or 2, got '$SOCKETS'"; done
  for value in $WORKLOADS; do xe_in_list "$value" P D L || die "WORKLOADS must list P, D and/or L, got '$WORKLOADS'"; done
  for value in $MEMS; do
    if [[ $NODE_ROLE == cache ]]; then
      [[ $value == cache ]] || die "MEMS on a cache node must be 'cache', got '$MEMS'"
    else
      [[ $value == hbm || $value == ddr ]] || die "MEMS on a Flat node must list hbm and/or ddr, got '$MEMS'"
    fi
  done
  [[ $REPEATS =~ ^[1-9][0-9]*$ ]] || die "REPEATS must be a positive integer, got '$REPEATS'"
  [[ $ANCHOR == 0 || $ANCHOR == 1 ]] || die "ANCHOR must be 0 or 1, got '$ANCHOR'"
  [[ ! -e $OUT ]] || die "OUT '$OUT' already exists"
  if [[ $DRY_RUN == 1 ]]; then
    ARTIFACTS=${ARTIFACTS:-'<ARTIFACTS>'}
    PROMPTS_DIR=${PROMPTS_DIR:-'<PROMPTS_DIR>'}
    UPERF=${UPERF:-'<UPERF>'}
    VLLM_SIF=${VLLM_SIF:-'<VLLM_SIF>'}
    VLLM_MODEL=${VLLM_MODEL:-'<VLLM_MODEL>'}
    VLLM_PATCHED_WORKER=${VLLM_PATCHED_WORKER:-'<VLLM_PATCHED_WORKER>'}
  else
    for value in $ENGINES; do
      if [[ $value == c ]]; then
        : "${ARTIFACTS:?ARTIFACTS (directory with int8-v3.bin and tokenizer-v2.bin) required for the c engine}"
        : "${PROMPTS_DIR:?PROMPTS_DIR (short-86.txt, prompt-2048.txt, prompt-8192.txt) required for the c engine}"
      fi
    done
    for value in $ENGINES; do
      if [[ $value == vllm ]]; then
        : "${VLLM_SIF:?VLLM_SIF (Apptainer image of vLLM 0.34.0 CPU) required for vllm}"
        : "${VLLM_MODEL:?VLLM_MODEL (directory of the W8A8 checkpoint) required for vllm}"
        : "${PROMPTS_DIR:?PROMPTS_DIR (the *.ids.json files) required for vllm}"
        [[ $NODE_ROLE == cache ]] || : "${VLLM_PATCHED_WORKER:?VLLM_PATCHED_WORKER (patched cpu_worker.py, scripts/vllm/numa-hbm.patch applied) required for vllm on a Flat node}"
      fi
    done
    [[ $NODE_ROLE != cache ]] || : "${UPERF:?UPERF (user-space perf binary) required on the Cache node}"
  fi
  XE_CELL_PY="$ROOT/tools/xengine_cell.py"
  C_ENGINE=${C_ENGINE:-$ROOT/build/release/runq}
  C_COMMIT=$(cat "$ROOT/COMMIT" 2> /dev/null || git -C "$ROOT" rev-parse HEAD 2> /dev/null || echo unknown)
  if [[ $XE_AMX_OPS == 1 ]]; then
    XE_COMPUTE_EVENTS='cycles,ref-cycles,instructions,AMX_OPS_RETIRED.INT8,EXE.AMX_BUSY,power/energy-pkg/,power/energy-ram/'
  fi
  if [[ $DRY_RUN == 0 ]] && command -v taskset > /dev/null; then
    XE_HELPER_CPU=${XE_HELPER_CPU-$(($(getconf _NPROCESSORS_ONLN) - 1))}
  fi
  export NODE_ROLE ENGINES SOCKETS MEMS WORKLOADS REPEATS DRY_RUN
  return 0
}

# job_begin, plus: the EXIT trap also merges the per-run manifest entries into OUT/manifest.json and builds runs.csv, and a
# SIGTERM (the Slurm time limit) ends the script through that trap.
xe_job_begin() {
  job_begin
  finish() {
    local status=$?
    printf 'exit_code=%s\n' "$status" > "$OUT/exit-status.txt"
    if ! dry_run; then
      "$PYTHON" "$XE_CELL_PY" assemble "$OUT" --campaign "node_role=$NODE_ROLE" --campaign "job=$SLURM_JOB_ID" \
        --campaign "engines=$ENGINES" --campaign "sockets=$SOCKETS" --campaign "mems=$MEMS" \
        --campaign "workloads=$WORKLOADS" --campaign "repeats=$REPEATS" || true
      "$PYTHON" "$ROOT/tools/xengine_summary.py" "$OUT" -o "$OUT/runs.csv" || true
    fi
    if declare -F vllm_stop > /dev/null; then vllm_stop || true; fi
    [[ -z $shm ]] || rm -rf -- "$shm"
    exit "$status"
  }
  trap 'exit 143' TERM
}

xe_record_environment() {
  if dry_run; then
    printf 'ENV   record_environment, python freeze, engine versions -> environment.txt\n'
    return 0
  fi
  STAGE=xengine REPEATS=$REPEATS PRECISIONS=int8 MODELS=
  record_environment
  {
    printf 'NODE_ROLE=%s ENGINES=%s SOCKETS=%s MEMS=%s WORKLOADS=%s REPEATS=%s ANCHOR=%s\n' "$NODE_ROLE" "$ENGINES" "$SOCKETS" \
      "$MEMS" "$WORKLOADS" "$REPEATS" "$ANCHOR"
    printf 'perf: %s\n' "$("$PERF_BIN" --version 2> /dev/null || echo unavailable)"
    local e
    for e in $ENGINES; do
      if declare -F "${e}_environment" > /dev/null; then "${e}_environment"; fi
    done
  } >> "$OUT/environment.txt"
  "$PYTHON" -m pip freeze > "$OUT/python-freeze.txt" 2> /dev/null || true
}

# A and B: Flat mode (the check of the existing job: 4 NUMA nodes, two CPU-less ~64 GB HBM nodes). cache: two NUMA nodes;
# the memory-side-cache sizes found are recorded in cache-mode.json.
xe_check_node() {
  if dry_run; then
    printf 'CHECK node role %s: %s\n' "$NODE_ROLE" "$([[ $NODE_ROLE == cache ]] && echo '2 NUMA nodes (Cache mode)' || echo 'Flat mode, 4 NUMA nodes, 2 CPU-less ~64 GB')"
    return 0
  fi
  if [[ $NODE_ROLE == cache ]]; then
    python3 - << 'PY' > "$OUT/cache-mode.json"
import glob, json
from pathlib import Path
nodes = sorted(Path(p).name for p in glob.glob('/sys/devices/system/node/node[0-9]*'))
assert nodes == ['node0', 'node1'], 'Cache node must have exactly 2 NUMA nodes, found %s' % nodes
cache = {Path(p).parts[-3]: Path(p).read_text().strip() for p in sorted(glob.glob('/sys/devices/system/node/node*/memory_side_cache/index*/size'))}
print(json.dumps(dict(mode='cache', numa_nodes=nodes, memory_side_cache_bytes=cache), indent=2))
PY
  else
    check_flat_mode
  fi
}

# Privileged counters through sudo (Flat nodes); the Cache node has the user-space wrapper only.
XE_PERF=0
xe_perf_available() {
  XE_PERF=0
  if dry_run; then
    XE_PERF=1
    return 0
  fi
  if [[ $NODE_ROLE == cache ]]; then
    "$UPERF" stat -e cycles,instructions -- true > /dev/null 2>&1 && XE_PERF=1
  elif sudo -n "$PERF_BIN" stat -a -x , -e "$XE_MEM_EVENTS" -- true > /dev/null 2>&1; then
    XE_PERF=1
  fi
  printf 'perf_available=%s\n' "$XE_PERF" >> "$OUT/environment.txt"
}

# 10 s of the MEM group, system-wide, before anything runs: idle power for the energy-per-token subtraction. Flat only.
xe_idle_baseline() {
  [[ $NODE_ROLE != cache ]] || return 0
  if dry_run; then
    printf 'IDLE  sudo -n %s stat -a --per-socket -I 1000 -x, -e <MEM group> -o OUT/idle-baseline.perf.csv -- sleep 10\n' "$PERF_BIN"
    return 0
  fi
  [[ $XE_PERF == 1 ]] || {
    record_result idle-baseline FAIL 'perf unavailable'
    return 0
  }
  sudo -n "$PERF_BIN" stat -a --per-socket -I 1000 -x, -e "$XE_MEM_EVENTS" -o "$OUT/idle-baseline.perf.csv" -- sleep 10 \
    2> "$OUT/idle-baseline.perf.err" || record_result idle-baseline FAIL 'perf failed'
}

# Repeat 1 = counters for memory (MEM), repeat 2 = compute and AMX (COMPUTE), repeat 3 = no perf (timing only).
# Cache node: repeat 1 = user-space per-process counters (USER), the rest timing only.
xe_perf_group_for_rep() {
  case $NODE_ROLE:$1 in
    A:1 | B:1) echo MEM ;;
    A:2 | B:2) echo COMPUTE ;;
    cache:1) echo USER ;;
    *) echo none ;;
  esac
}

# ---- one cell ------------------------------------------------------------------------------------------------------
CELL_TODO=
# xe_run_cell ENGINE S MEM WL REP: sets the cell globals and hands over to run_<engine>_cell.
xe_run_cell() {
  CELL_ENGINE=$1 CELL_S=$2 CELL_MEM=$3 CELL_WL=$4 CELL_REP=$5 CELL_VARIANT=${6:-}
  CELL_LABEL="$CELL_ENGINE-s$CELL_S-$CELL_MEM${CELL_VARIANT:+-$CELL_VARIANT}-$CELL_WL-r$CELL_REP"
  CELL_DIR="$OUT/runs/$CELL_LABEL"
  CELL_PERF_GROUP=$(xe_perf_group_for_rep "$CELL_REP")
  CELL_ENV=() CELL_CMD=() CELL_CHECK=() CELL_ANCHORS=()
  CELL_PERF_ATTACH=
  CELL_VERSION= CELL_CONFIG= CELL_GEN= CELL_PROC= CELL_LOAD_PATTERN= CELL_TODO=
  "run_${CELL_ENGINE}_cell" "$CELL_S" "$CELL_MEM" "$CELL_WL" "$CELL_REP"
}

xe_estimate_cell() {
  "${CELL_ENGINE}_estimate_s" "$CELL_S" "$CELL_MEM" "$CELL_WL" "$CELL_REP"
}

xe_perf_command_text() {
  case $1 in
    MEM) printf 'sudo -n %s stat -a --per-socket -I %s -x, -e %s -o %s/perf.csv -- tail -s 0.1 --pid=<engine pid> -f /dev/null' "$PERF_BIN" "$XE_PERF_INTERVAL_MS" "$XE_MEM_EVENTS" "$CELL_DIR" ;;
    COMPUTE) printf 'sudo -n %s stat -a --per-socket -I %s -x, -e %s -o %s/perf.csv -- tail -s 0.1 --pid=<engine pid> -f /dev/null' "$PERF_BIN" "$XE_PERF_INTERVAL_MS" "$XE_COMPUTE_EVENTS" "$CELL_DIR" ;;
    USER)
      if [[ -n $CELL_PERF_ATTACH ]]; then
        printf '%s stat -I %s -x, -e %s -p <worker pids> -o %s/perf.csv -- tail -s 0.1 --pid=<client pid> -f /dev/null' "${UPERF:-<UPERF>}" "$XE_PERF_INTERVAL_MS" "$XE_USER_EVENTS" "$CELL_DIR"
      else
        printf '%s stat -I %s -x, -e %s -o %s/perf.csv -- <command>' "${UPERF:-<UPERF>}" "$XE_PERF_INTERVAL_MS" "$XE_USER_EVENTS" "$CELL_DIR"
      fi
      ;;
    *) printf 'none' ;;
  esac
}

xe_print_cell() {
  local est=$1
  {
    printf 'CELL %s perf=%s est=%ss%s\n' "$CELL_LABEL" "$CELL_PERF_GROUP" "$est" "${CELL_TODO:+ [$CELL_TODO]}"
    if [[ -z $CELL_TODO ]]; then
      printf '  env:   %s\n' "${CELL_ENV[*]-}"
      printf '  cmd:   %s\n' "${CELL_CMD[*]}"
      printf '  perf:  %s\n' "$(xe_perf_command_text "$CELL_PERF_GROUP")"
      printf '  check: %s\n' "${CELL_CHECK[*]-}"
    fi
  } | tee -a "$OUT/dry-run.txt"
  printf '%s %s\n' "$CELL_ENGINE" "$est" >> "$OUT/estimate.tsv"
}

# xe_write_cell STATUS DETAIL RC: the manifest entry of the current cell (OUT/runs/<label>/cell.json).
xe_write_cell() {
  local env_args=() word
  for word in ${CELL_ENV[@]+"${CELL_ENV[@]}"}; do env_args+=(--env "$word"); done
  "$PYTHON" "$XE_CELL_PY" write-cell "$CELL_DIR" --field "label=$CELL_LABEL" --field "node_role=$NODE_ROLE" \
    --field "engine=$CELL_ENGINE" --field "variant=$CELL_VARIANT" --field "version=$CELL_VERSION" --field "config=$CELL_CONFIG" --field "sockets=$CELL_S" \
    --field "mem=$CELL_MEM" --field "workload=$CELL_WL" --field "repeat=$CELL_REP" --field "generation=$CELL_GEN" \
    --field "perf_group=$CELL_PERF_GROUP" --field "perf_events=$(xe_perf_events "$CELL_PERF_GROUP")" \
    --field "job=$SLURM_JOB_ID" --field "status=$1" --field "status_detail=$2" --field "rc=$3" \
    ${env_args[@]+"${env_args[@]}"} --command "${CELL_CMD[*]}"
}

xe_perf_events() {
  case $1 in
    MEM) printf '%s' "$XE_MEM_EVENTS" ;;
    COMPUTE) printf '%s' "$XE_COMPUTE_EVENTS" ;;
    USER) printf '%s' "$XE_USER_EVENTS" ;;
  esac
}

# Background helper of one run: notes when loading is over (the engine's stderr shows CELL_LOAD_PATTERN), then, on the
# repeats in XE_PLACEMENT_REPS, summarises /proc/PID/numa_maps (end of load) and runs numastat -p about XE_MIDRUN_S later
# (a run that is already over is noted). It runs on the last CPU so it does not steal a worker's core on one-socket runs.
xe_sampler() {
  local d=$1 pid=$2 mode=$3 target=$2
  if [[ -n ${XE_HELPER_CPU:-} ]]; then taskset -cp "$XE_HELPER_CPU" "$BASHPID" > /dev/null 2>&1 || true; fi
  [[ -n $CELL_LOAD_PATTERN ]] || return 0
  until [[ -r $d/stderr ]] && grep -qF -- "$CELL_LOAD_PATTERN" "$d/stderr"; do
    kill -0 "$pid" 2> /dev/null || return 0
    sleep 0.02
  done
  xe_now_ns > "$d/load_end_seen_ns"
  xe_in_list "$CELL_REP" $XE_PLACEMENT_REPS || return 0
  if [[ $mode == wrap ]]; then target=$(pgrep -n -x "$CELL_PROC" || true); fi
  [[ -n $target ]] || return 0
  "$PYTHON" "$XE_CELL_PY" numa-maps "$target" > "$d/numa_maps.json" 2> /dev/null || true
  sleep "${XE_MIDRUN_S:-1.0}"
  if kill -0 "$target" 2> /dev/null; then
    { printf '# numastat -p %s, %s s after the numa_maps summary (load end)\n' "$target" "${XE_MIDRUN_S:-1.0}"; numastat -p "$target"; } \
      > "$d/numastat.txt" 2>&1 || true
  else
    echo '# process already finished at the numastat time' > "$d/numastat.txt"
  fi
}

# xe_execute_cell: runs the command in CELL_CMD with CELL_ENV, perf by repeat, markers, placement evidence, checks and the
# manifest entry. Never fails the job: a failed cell is recorded (record_result) and the next one runs.
xe_execute_cell() {
  local d=$CELL_DIR est
  est=$(xe_estimate_cell)
  if dry_run; then
    xe_print_cell "$est"
    return 0
  fi
  if [[ -n $CELL_TODO ]]; then
    record_result "cell-$CELL_LABEL" FAIL "$CELL_TODO"
    return 0
  fi
  mkdir -p "$d"
  local mode=none events
  events=$(xe_perf_events "$CELL_PERF_GROUP")
  case $CELL_PERF_GROUP in
    MEM | COMPUTE) mode=sys ;;
    USER) mode=wrap ;;
  esac
  [[ $mode != wrap || -z $CELL_PERF_ATTACH ]] || mode=attach
  if [[ $mode != none && $XE_PERF != 1 ]]; then
    mode=none CELL_PERF_GROUP=none events=
  fi
  xe_write_cell started '' ''
  printf '%s\n' "${CELL_ENV[*]-} ${CELL_CMD[*]}" > "$d/cmd.txt"
  xe_perf_command_text "$CELL_PERF_GROUP" > "$d/perf.cmd"
  local cmd=(env ${CELL_ENV[@]+"${CELL_ENV[@]}"} "${CELL_CMD[@]}")
  [[ $mode != wrap ]] || cmd=("$UPERF" stat -I "$XE_PERF_INTERVAL_MS" -x, -e "$events" -o "$d/perf.csv" -- "${cmd[@]}")
  local launch_ns perf_start_ns='' end_ns rc=0 pid perfpid='' samp
  launch_ns=$(xe_now_ns)
  "${cmd[@]}" < /dev/null > "$d/stdout" 2> "$d/stderr" &
  pid=$!
  if [[ $mode == sys ]]; then
    perf_start_ns=$(xe_now_ns)
    sudo -n "$PERF_BIN" stat -a --per-socket -I "$XE_PERF_INTERVAL_MS" -x, -e "$events" -o "$d/perf.csv" -- \
      tail -s 0.1 --pid="$pid" -f /dev/null < /dev/null > /dev/null 2> "$d/perf.err" &
    perfpid=$!
  fi
  if [[ $mode == attach ]]; then
    perf_start_ns=$(xe_now_ns)
    "$UPERF" stat -I "$XE_PERF_INTERVAL_MS" -x, -e "$events" -p "$("$CELL_PERF_ATTACH")" -o "$d/perf.csv" -- \
      tail -s 0.1 --pid="$pid" -f /dev/null < /dev/null > /dev/null 2> "$d/perf.err" &
    perfpid=$!
  fi
  xe_sampler "$d" "$pid" "$mode" < /dev/null &
  samp=$!
  wait "$pid" || rc=$?
  end_ns=$(xe_now_ns)
  if [[ -n $perfpid ]]; then wait "$perfpid" || true; fi
  wait "$samp" || true
  {
    printf 'launch_ns=%s\nend_ns=%s\nrc=%s\nperf_mode=%s\n' "$launch_ns" "$end_ns" "$rc" "$mode"
    [[ -z $perf_start_ns ]] || printf 'perf_start_ns=%s\n' "$perf_start_ns"
    printf '%s\n' ${CELL_ANCHORS[@]+"${CELL_ANCHORS[@]}"}
    [[ ! -s $d/load_end_seen_ns ]] || printf 'load_end_seen_ns=%s\n' "$(< "$d/load_end_seen_ns")"
  } > "$d/anchors"
  if declare -F "${CELL_ENGINE}_finalize" > /dev/null; then "${CELL_ENGINE}_finalize" || true; fi
  "$PYTHON" "$XE_CELL_PY" markers "$d" 2> "$d/markers.err" || true
  local status=ok detail='' crc=0
  if ((${#CELL_CHECK[@]})); then "${CELL_CHECK[@]}" > "$d/check.txt" 2>&1 || crc=$?; fi
  if [[ $rc -ne 0 ]]; then
    status=failed detail="exit=$rc"
  elif [[ $crc -ne 0 ]]; then
    status=failed detail="check: $(grep -m1 '^FAIL' "$d/check.txt" | cut -c1-200)"
  elif [[ $mode != none ]] && ! grep -qE '^[0-9S]' "$d/perf.csv" 2> /dev/null; then
    status=failed detail='perf wrote no data'
  fi
  xe_write_cell "$status" "$detail" "$rc"
  if [[ $status == ok ]]; then record_result "cell-$CELL_LABEL" PASS; else record_result "cell-$CELL_LABEL" FAIL "$detail"; fi
  return 0
}

# ---- the matrix ----------------------------------------------------------------------------------------------------
# Default matrix of an engine: one line "S MEM WL..." per socket count and memory tier.
xe_matrix_default() {
  local mem s
  for mem in $MEMS; do
    for s in $SOCKETS; do printf '%s %s %s\n' "$s" "$mem" "$WORKLOADS"; done
  done
}

# xe_run_engine ENGINE: prepare, then per configuration (begin hook, REPEATS x workloads, end hook).
xe_run_engine() {
  local engine=$1 s mem memtok wls rep wl variant
  declare -F "run_${engine}_cell" > /dev/null || die "engine '$engine' has no run_${engine}_cell"
  if declare -F "${engine}_prepare" > /dev/null; then
    "${engine}_prepare" || {
      record_result "prepare-$engine" FAIL 'engine skipped'
      return 0
    }
  fi
  while read -r s memtok wls; do
    [[ -n $s ]] || continue
    mem=${memtok%%:*} variant=
    [[ $memtok != *:* ]] || variant=${memtok#*:}
    XE_CONFIG_VARIANT=$variant XE_CONFIG_WLS=$wls
    if dry_run && declare -F "${engine}_config_overhead_s" > /dev/null; then
      printf 'CONFIG %s s%s %s%s begin/end (server start, etc.) est=%ss\n' "$engine" "$s" "$mem" "${variant:+ ($variant)}" "$("${engine}_config_overhead_s" "$s" "$mem")" |
        tee -a "$OUT/dry-run.txt"
      printf '%s %s\n' "$engine" "$("${engine}_config_overhead_s" "$s" "$mem")" >> "$OUT/estimate.tsv"
    fi
    if dry_run && declare -F "${engine}_describe_config" > /dev/null; then
      "${engine}_describe_config" "$s" "$mem" | tee -a "$OUT/dry-run.txt"
    fi
    if declare -F "${engine}_begin_config" > /dev/null && ! dry_run; then
      "${engine}_begin_config" "$s" "$mem" || {
        record_result "config-$engine-s$s-$mem" FAIL 'begin_config failed; configuration skipped'
        continue
      }
    fi
    for ((rep = 1; rep <= REPEATS; rep++)); do
      for wl in $wls; do xe_run_cell "$engine" "$s" "$mem" "$wl" "$rep" "$variant"; done
    done
    if declare -F "${engine}_end_config" > /dev/null && ! dry_run; then "${engine}_end_config" "$s" "$mem" || true; fi
  done < <(if declare -F "${engine}_matrix" > /dev/null; then "${engine}_matrix"; else xe_matrix_default; fi)
}

xe_print_estimate() {
  dry_run || return 0
  local fixed=300 budget=180
  [[ $NODE_ROLE != cache ]] || fixed=180
  {
    printf '\nESTIMATE role %s (placeholders for llamacpp and vllm; the c engine load is %s s per run)\n' "$NODE_ROLE" "${XE_C_LOAD_S:-45}"
    awk -v fixed="$fixed" -v budget="$budget" '
      { s[$1] += $2; n[$1]++; tot += $2 }
      END {
        for (e in s) printf "  %-9s %3d items %6.1f min\n", e, n[e], s[e] / 60
        printf "  %-9s %17.1f min (environment record, staging, idle baseline, checks)\n", "fixed", fixed / 60
        printf "  %-9s %17.1f min of the %d min limit\n", "TOTAL", (tot + fixed) / 60, budget
      }' "$OUT/estimate.tsv"
  } | tee -a "$OUT/dry-run.txt"
}

# ---- llama3.x.c ----------------------------------------------------------------------------------------------------
C_EXPLICIT='explicit:0/2/0,1/3/1' # package/HBM/DDR per socket on a Flat node: S0 = cores 0-55, DDR 0, HBM 2; S1 = 56-111, DDR 1, HBM 3
C_STAGED=
C_COMMIT=

c_matrix() {
  xe_matrix_default
  # Anchor (plan section 2): c, 1S, HBM, decode-heavy, on the DDR node as well, to measure the node-to-node difference.
  if [[ $NODE_ROLE == B && $ANCHOR == 1 ]] && xe_in_list 1 $SOCKETS && xe_in_list D $WORKLOADS && ! xe_in_list hbm $MEMS; then
    printf '1 hbm D\n'
  fi
}

c_prepare() {
  need_shm
  if dry_run; then
    C_STAGED=$(stage_model_on 0 llama31-8b int8 "$ARTIFACTS")
    printf 'PREP  c: build %s if missing, stage int8-v3.bin in tmpfs on node 0 -> %s, copy prompts manifest\n' "$C_ENGINE" "$C_STAGED"
    return 0
  fi
  if [[ ! -x $C_ENGINE ]]; then
    build_tree release build/release/runq || return 1
  fi
  printf 'c engine binary sha256 %s\n' "$(sha256sum "$C_ENGINE" | cut -d' ' -f1)" >> "$OUT/environment.txt"
  C_STAGED=$(stage_model_on 0 llama31-8b int8 "$ARTIFACTS") || return 1
  cp "$PROMPTS_DIR/prompts.manifest.json" "$OUT/prompts.manifest.json" 2> /dev/null || true
}

c_environment() {
  printf 'c engine: commit %s binary %s sha256 %s\n' "$C_COMMIT" "$C_ENGINE" "$(sha256sum "${C_ENGINE:-/nonexistent}" 2> /dev/null | cut -d' ' -f1)"
}

# Seconds per run (placeholders from the setup smoke runs): load 30-60 s (checkpoint pack, placement populate and verify),
# then P 2-3 s, D 3-6 s, L about 20 s, plus about 2 s of harness work.
c_estimate_s() {
  local run
  case $3 in P) run=3 ;; D) run=6 ;; L) run=22 ;; esac
  echo $((${XE_C_LOAD_S:-45} + run + 2))
}

# Workloads (--benchmark, greedy, raw prompts without --chat-template so every engine sees the same exact token ids):
#   P  prompt-2048 (2048 tokens), G = 1                         context 2112
#   D  short-86 (86 tokens), G = 128                            context 278
#   L  prompt-8192 (8192 tokens), G = 128 (decode from depth 8192; its TTFT is the 8192 prefill)   context 8448
# 1S needs numactl --cpunodebind=0, or the engine takes all 112 CPUs as one team. The engine binds its own memory
# (no --membind). Cache node: 1S plain (no placement knob), 2S LLAMA_PLACEMENT=ddr, no explicit topology.
run_c_cell() {
  local sockets=$1 mem=$2 wl=$3 prompt ctx ptok
  case $wl in
    P) prompt=prompt-2048 CELL_GEN=1 ctx=2112 ptok=2048 ;;
    D) prompt=short-86 CELL_GEN=128 ctx=278 ptok=86 ;;
    L) prompt=prompt-8192 CELL_GEN=128 ctx=8448 ptok=8192 ;;
  esac
  CELL_ENV=(LLAMA_BACKEND=teams LLAMA_THREADS=56)
  case $mem in
    hbm | ddr) CELL_ENV+=("LLAMA_TOPOLOGY=$C_EXPLICIT" "LLAMA_PLACEMENT=$mem" "LLAMA_SOCKETS=$sockets") ;;
    cache)
      CELL_ENV+=("LLAMA_SOCKETS=$sockets")
      if [[ $sockets == 2 ]]; then CELL_ENV+=(LLAMA_PLACEMENT=ddr); fi
      if [[ $sockets == 1 && -n ${C_CACHE_1S_PLACEMENT:-} ]]; then CELL_ENV+=("LLAMA_PLACEMENT=$C_CACHE_1S_PLACEMENT"); fi
      ;;
  esac
  local launcher=()
  [[ $sockets != 1 ]] || launcher=(numactl --cpunodebind=0)
  CELL_CMD=(${launcher[@]+"${launcher[@]}"} "${C_ENGINE:-<runq>}" "${C_STAGED:-<staged int8-v3.bin>}" -z "$ARTIFACTS/tokenizer-v2.bin"
    --context-length "$ctx" --benchmark --prompt-file "$PROMPTS_DIR/$prompt.txt" -t 0 --seed 42 -n 0
    --generation-length "$CELL_GEN" --no-text --log-level info --json "$CELL_DIR/report.json")
  CELL_VERSION=$C_COMMIT
  CELL_CONFIG="threads_per_socket=56 prompt=$prompt context=$ctx generation=$CELL_GEN greedy seed=42 no-chat-template"
  CELL_PROC=runq
  CELL_LOAD_PATTERN='placement=' # the engine's info line right after initialisation, before it tokenises the prompt
  CELL_CHECK=("$PYTHON" "$XE_CELL_PY" check-c "$CELL_DIR" --sockets "$sockets" --mem "$mem" --generation "$CELL_GEN"
    --prompt-tokens "$ptok")
  xe_execute_cell
}

# ---- llama.cpp: section to be filled in with its verified command line ----------------------------------------------------
# Placeholder only: the estimate follows the plan (section 8); the run function records the cell as not implemented.
llamacpp_estimate_s() {
  local run
  case $3 in P) run=6 ;; D) run=14 ;; L) run=45 ;; esac
  echo $((${XE_LLAMACPP_LOAD_S:-15} + run + 2))
}
run_llamacpp_cell() {
  CELL_TODO='llama.cpp section not filled in yet (command line pending)'
  xe_execute_cell
}

# ---- vLLM 0.34.0 CPU backend (Apptainer image, RedHatAI W8A8 checkpoint) ------------------------------------------------
# One server per configuration, started in vllm_begin_config and stopped in vllm_end_config; every cell is one separate
# streaming request of tools/xengine_vllm_client.py (exact token ids of the c engine's prompt files, greedy, ignore_eos),
# so perf attaches to the request: system-wide on Flat nodes (tail --pid=<client>), user-space `perf -p <workers>` on the
# Cache node. Compiled mode (default), 55 OMP threads per socket (cores 55 and 111 stay free for the frontend and the
# client), prefix caching off and proven off (vllm:prefix_cache_hits_total == 0 at the end of every configuration).
# Configurations (MEM[:VARIANT] in the matrix): hbm:patched = ranks bound to the HBM nodes by the disclosed one-line patch
# (scripts/vllm/numa-hbm.patch, mounted over cpu_worker.py, VLLM_HBM_NODE_OFFSET=2); hbm:stock-membind = stock vLLM under
# `numactl --membind=2` (it rebinds itself to DDR; kept as a finding, D only); ddr and cache = stock.
# Env: VLLM_SIF, VLLM_MODEL, VLLM_PATCHED_WORKER (Flat), optional VLLM_WORK (node-local scratch), VLLM_PORT, VLLM_VERSION,
# VLLM_MODEL_REV, VLLM_CLIENT_CPU (55), XE_VLLM_START_TIMEOUT_S (1500), APPTAINER_CACHEDIR / APPTAINER_TMPDIR.
VLLM_PORT=${VLLM_PORT:-8000}
VLLM_SERVED=llama31-8b-w8a8
VLLM_VERSION=${VLLM_VERSION:-0.34.0+cpu}
VLLM_STARTUP_S=
VLLM_T0_NS=
VLLM_SERVER_PID=
VLLM_CFG_DIR=
VLLM_CFG_TAG=

vllm_matrix() {
  local mem s
  for mem in $MEMS; do
    for s in $SOCKETS; do
      if [[ $mem == hbm ]]; then printf '%s hbm:patched %s\n' "$s" "$WORKLOADS"; else printf '%s %s %s\n' "$s" "$mem" "$WORKLOADS"; fi
    done
  done
  if [[ $NODE_ROLE == A && ${XE_VLLM_MEMBIND_CONTROL:-1} == 1 ]] && xe_in_list 1 $SOCKETS && xe_in_list D $WORKLOADS; then
    printf '1 hbm:stock-membind D\n'
  fi
}

# Seconds per request including about 6 s of client start-up; per configuration: server start-up (about 190 s compiled), a
# warm-up request per workload, placement evidence and shutdown.
vllm_estimate_s() {
  local run
  case $3 in P) run=10 ;; D) run=20 ;; L) run=35 ;; esac
  echo "$run"
}
vllm_config_overhead_s() {
  echo "${XE_VLLM_START_S:-260}"
}

# vllm_build_server S VARIANT: sets VLLM_SERVER_CMD (an array) for the configuration.
vllm_build_server() {
  local s=$1 variant=$2 bind wrap=() binds=() envs=()
  local local_dir="${VLLM_WORK:-/tmp/vllm-xengine}/$SLURM_JOB_ID-$VLLM_CFG_TAG"
  if [[ $s == 1 ]]; then
    bind=0-54
    wrap=(numactl --cpunodebind=0)
    [[ $variant != stock-membind ]] || wrap+=(--membind=2)
  else
    bind='0-54|56-110'
  fi
  if [[ $variant == patched ]]; then
    binds+=(--bind "$VLLM_PATCHED_WORKER:/opt/venv/lib/python3.12/site-packages/vllm/v1/worker/cpu_worker.py:ro")
    envs+=(--env VLLM_HBM_NODE_OFFSET=2)
  fi
  VLLM_SERVER_CMD=(${wrap[@]+"${wrap[@]}"} apptainer exec --cleanenv --no-home --bind "$VLLM_MODEL:/model:ro" --bind "$local_dir:/cache"
    ${binds[@]+"${binds[@]}"} --env "VLLM_CPU_OMP_THREADS_BIND=$bind" --env VLLM_CACHE_ROOT=/cache/vllm
    --env TORCHINDUCTOR_CACHE_DIR=/cache/inductor --env HF_HOME=/cache/hf --env HF_HUB_OFFLINE=1 --env VLLM_LOGGING_LEVEL=INFO
    --env TOKENIZERS_PARALLELISM=false --env CXX=g++ --env HOME=/cache/home --env TRITON_CACHE_DIR=/cache/triton
    --env XDG_CACHE_HOME=/cache/xdg ${envs[@]+"${envs[@]}"} "$VLLM_SIF" vllm serve /model --served-model-name "$VLLM_SERVED"
    --dtype bfloat16 --tensor-parallel-size "$s" --max-model-len 8704 --max-num-seqs 1 --max-num-batched-tokens 8192
    --no-enable-prefix-caching --seed 0 --generation-config vllm --kv-cache-memory-bytes 4G --port "$VLLM_PORT")
  VLLM_LOCAL=$local_dir
}

vllm_set_cfg() {
  VLLM_CFG_TAG="s$1-$2${XE_CONFIG_VARIANT:+-$XE_CONFIG_VARIANT}"
  VLLM_CFG_DIR="$OUT/vllm-config-$VLLM_CFG_TAG"
}

vllm_describe_config() {
  vllm_set_cfg "$1" "$2"
  vllm_build_server "$1" "$XE_CONFIG_VARIANT"
  printf 'SERVER vllm %s (workloads %s): start once, wait for /health, one warm-up request per workload, numastat -p per worker\n' \
    "$VLLM_CFG_TAG" "$XE_CONFIG_WLS"
  printf '  cmd:   %s\n' "${VLLM_SERVER_CMD[*]}"
}

# vllm_workload WL: sets VLLM_PROMPT (ids file stem), VLLM_G, VLLM_PTOK.
vllm_workload() {
  case $1 in
    P) VLLM_PROMPT=prompt-2048 VLLM_G=1 VLLM_PTOK=2048 ;;
    D) VLLM_PROMPT=short-86 VLLM_G=128 VLLM_PTOK=86 ;;
    L) VLLM_PROMPT=prompt-8192 VLLM_G=128 VLLM_PTOK=8192 ;;
  esac
}

# vllm_client_cmd OUTDIR WL: VLLM_CLIENT_CMD (array), the client pinned to a spare core, inside the image (its python and
# tokenizers); OUTDIR is mounted as /out.
vllm_client_cmd() {
  vllm_workload "$2"
  VLLM_CLIENT_CMD=(taskset -c "${VLLM_CLIENT_CPU:-55}" apptainer exec --cleanenv --no-home --bind "$VLLM_MODEL:/model:ro"
    --bind "$ROOT/tools:/res:ro" --bind "$PROMPTS_DIR:/prompts:ro" --bind "$1:/out" "$VLLM_SIF" python3 /res/xengine_vllm_client.py
    --tokenizer /model --model "$VLLM_SERVED" --port "$VLLM_PORT" --workload "$2" --ids-file "/prompts/$VLLM_PROMPT.ids.json"
    --gen "$VLLM_G")
}

vllm_worker_pids() {
  pgrep -u "$(id -un)" -f 'VLLM::Worker|VLLM::.*Worker' | tr '\n' ' '
}
vllm_worker_pids_csv() {
  vllm_worker_pids | sed 's/ *$//; s/ /,/g'
}

vllm_prepare() {
  if dry_run; then
    printf 'PREP  vllm: check image, checkpoint, patched worker, ids files; one server per configuration\n'
    return 0
  fi
  local f
  for f in "$VLLM_SIF" "$VLLM_MODEL/config.json" "$PROMPTS_DIR/short-86.ids.json" "$PROMPTS_DIR/prompt-2048.ids.json" \
    "$PROMPTS_DIR/prompt-8192.ids.json" "$ROOT/tools/xengine_vllm_client.py"; do
    [[ -e $f ]] || {
      echo "vllm_prepare: missing $f" >&2
      return 1
    }
  done
  [[ $NODE_ROLE == cache ]] || [[ -e $VLLM_PATCHED_WORKER ]] || return 1
  mkdir -p "${VLLM_WORK:-/tmp/vllm-xengine}"
}

vllm_environment() {
  printf 'vllm %s image sha256 %s checkpoint revision %s patch sha256 %s\n' "$VLLM_VERSION" \
    "$(sha256sum "$VLLM_SIF" 2> /dev/null | cut -d' ' -f1)" "${VLLM_MODEL_REV:-unrecorded}" \
    "$(sha256sum "$ROOT/scripts/vllm/numa-hbm.patch" 2> /dev/null | cut -d' ' -f1)"
  printf 'apptainer: %s\n' "$(apptainer --version 2> /dev/null || echo unavailable)"
}

vllm_stop() {
  [[ -z $VLLM_SERVER_PID ]] && return 0
  pkill -TERM -P "$VLLM_SERVER_PID" 2> /dev/null || true
  kill -TERM "$VLLM_SERVER_PID" 2> /dev/null || true
  local i
  for i in $(seq 1 30); do
    pgrep -u "$(id -un)" -f 'vllm serve' > /dev/null || break
    sleep 1
  done
  pkill -KILL -u "$(id -un)" -f 'vllm serve' 2> /dev/null || true
  pkill -KILL -u "$(id -un)" -f 'VLLM::' 2> /dev/null || true
  wait "$VLLM_SERVER_PID" 2> /dev/null || true
  VLLM_SERVER_PID=
  rm -rf -- "${VLLM_LOCAL:-/nonexistent-xengine}"
  sleep 2
}

# Starts the server (start-up time = launch to /health), runs one discarded warm-up request per workload, then records
# the placement evidence (numastat -p and a numa_maps summary of every worker) and the metrics baseline.
vllm_begin_config() {
  local s=$1 mem=$2 d wl pid now
  vllm_set_cfg "$s" "$mem"
  d=$VLLM_CFG_DIR
  mkdir -p "$d"
  vllm_build_server "$s" "$XE_CONFIG_VARIANT"
  rm -rf "$VLLM_LOCAL"
  mkdir -p "$VLLM_LOCAL/home" "$VLLM_LOCAL/triton" "$VLLM_LOCAL/xdg"
  printf '%s\n' "${VLLM_SERVER_CMD[*]}" > "$d/serve-cmd.txt"
  VLLM_T0_NS=$(xe_now_ns)
  "${VLLM_SERVER_CMD[@]}" < /dev/null > "$d/server.log" 2>&1 &
  VLLM_SERVER_PID=$!
  until curl -sf "http://127.0.0.1:$VLLM_PORT/health" > /dev/null; do
    if ! kill -0 "$VLLM_SERVER_PID" 2> /dev/null; then
      tail -30 "$d/server.log" >&2
      VLLM_SERVER_PID=
      return 1
    fi
    now=$(xe_now_ns)
    if ((now - VLLM_T0_NS > ${XE_VLLM_START_TIMEOUT_S:-1500} * 1000000000)); then
      vllm_stop
      return 1
    fi
    sleep 1
  done
  now=$(xe_now_ns)
  VLLM_STARTUP_S=$(awk -v a="$VLLM_T0_NS" -v b="$now" 'BEGIN { printf "%.1f", (b - a) / 1e9 }')
  printf 'startup_seconds=%s\nserver_launch_ns=%s\n' "$VLLM_STARTUP_S" "$VLLM_T0_NS" > "$d/startup.txt"
  for wl in $XE_CONFIG_WLS; do
    vllm_client_cmd "$d" "$wl"
    "${VLLM_CLIENT_CMD[@]}" --warmups 1 --repeats 0 --out "/out/warmup-$wl.jsonl" < /dev/null > "$d/warmup-$wl.stdout" 2>&1 ||
      record_result "warmup-vllm-$VLLM_CFG_TAG-$wl" FAIL 'warm-up request failed'
  done
  for pid in $(vllm_worker_pids); do
    numastat -p "$pid" > "$d/numastat-$pid.txt" 2>&1 || true
    "$PYTHON" "$XE_CELL_PY" numa-maps "$pid" > "$d/numa_maps-$pid.json" 2> /dev/null || true
  done
  curl -sf "http://127.0.0.1:$VLLM_PORT/metrics" > "$d/metrics-before.txt" || true
}

vllm_end_config() {
  local d=$VLLM_CFG_DIR
  curl -sf "http://127.0.0.1:$VLLM_PORT/metrics" > "$d/metrics-after.txt" || true
  if "$PYTHON" "$XE_CELL_PY" check-prefix "$d" > "$d/prefix-check.txt" 2>&1; then
    record_result "prefix-cache-vllm-$VLLM_CFG_TAG" PASS
  else
    record_result "prefix-cache-vllm-$VLLM_CFG_TAG" FAIL "$(cat "$d/prefix-check.txt")"
  fi
  vllm_stop
}

run_vllm_cell() {
  local s=$1 wl=$3
  vllm_set_cfg "$s" "$2"
  vllm_client_cmd "$CELL_DIR" "$wl"
  CELL_GEN=$VLLM_G
  CELL_CMD=("${VLLM_CLIENT_CMD[@]}" --warmups 0 --repeats 1 --out /out/client.jsonl --tag "$CELL_LABEL")
  CELL_VERSION=$VLLM_VERSION
  CELL_CONFIG="tp=$s omp_threads_per_socket=55 variant=${CELL_VARIANT:-stock} compiled prefix_caching=off max_num_seqs=1 kv_cache=4G prompt=$VLLM_PROMPT generation=$VLLM_G greedy ignore_eos startup_s=${VLLM_STARTUP_S:-n/a} server_cmd_file=$VLLM_CFG_TAG/serve-cmd.txt"
  CELL_ANCHORS=("load_start_ns=${VLLM_T0_NS:-0}")
  CELL_PERF_ATTACH=vllm_worker_pids_csv
  CELL_CHECK=("$PYTHON" "$XE_CELL_PY" check-vllm "$CELL_DIR" --generation "$VLLM_G" --prompt-tokens "$VLLM_PTOK")
  xe_execute_cell
}

vllm_finalize() {
  "$PYTHON" "$XE_CELL_PY" vllm-report "$CELL_DIR" 2> /dev/null
}
