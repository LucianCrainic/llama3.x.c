#!/usr/bin/env bash
#SBATCH --job-name=concurrent-tier
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=112
#SBATCH --exclusive
#SBATCH --mem=0
#SBATCH --time=08:00:00
# Concurrent HBM + DDR read bandwidth on one CRESCO8 Flat node (dual Xeon Max 9480): does a socket's combined read
# bandwidth exceed its best HBM-only bandwidth when its cores read both tiers at once, and does the engine's own Q8 decode GEMV
# with its rows split across the tiers beat all-HBM? The experiments and their analysis are in
# docs/concurrent-tier-bandwidth.md; the program is tests/stream_bench.c (--exp mode), the configurations and the
# analysis are tools/concurrent_tier.py.
#
# Required: ROOT (staged tree), OUT (new directory), PYTHON. Optional:
#   STAGES   (lists such as STAGES, PAGES, PRIOR may use '+' for spaces) stages in order (default smoke): smoke base e1 e2 e3 e4 e5 e5hbm e6 lat freq rep rep-e5
#            smoke  one pass over a handful of configurations, one and two sockets, both page sizes (counter format, placement,
#                   JSON); base  HBM-only and DDR-only sweeps (cores x streams x prefetch distance, hint, non-temporal), picks
#                   BEST_*;  e1 split teams;  e2 shared cores;  e3 page-interleaved regions;  e4 both sockets at the best
#                   E1/E2/E3 settings;  e5 the Q8 GEMV over the 70B shard;  e5hbm all-HBM GEMV with fewer workers;
#                   e6 sensitivity at the best E1 point;  lat loaded latency;  freq core/uncore frequency (turbostat) under long windows;
#                   fillcmp reads over random and zero-filled regions, e5c shuffled E5 confirmation (both page sizes),
#                   e5off GEMV alignment test, e6b real-split sensitivity, e1g e3g staggered HBM stripes, hbmx HBM-only baseline extension, e5alias stripe-aliasing tests,
#                   e5c2 row-split worker-count scan, repe3 repe4 repe5 the shuffled replication on the second node;
#                   rep, rep-e5 the replication points
#   REPS     repetitions per point of the measured stages (default 7);  BASE_REPS for base (default 3)
#   PAGES    "2m 4k" (default)
#   BEST_ENV a BEST_* file of an earlier job (otherwise the base stage of this job writes OUT/best.env)
#   PRIOR    results directories of earlier jobs, read by e4 and e6 to find the best E1/E2/E3 points
#   FILL     random (default), zero or ones: the content of the tier regions (stream-bench --fill); SHUFFLE=1 shuffles the order of
#            the configurations inside every process
#   SUMMARY_ARGS  extra arguments of the final summarize step (for example --zero-fill DIR); with PRIOR it pools earlier results
#   REGION_MIB  tier region per socket (default 20480 = 20 GiB, above the 16 GiB floor); INTER_MIB (32768)
#   DRY_RUN=1   print the commands of the processes, run nothing
# Each process runs every configuration of a stage once; the repetitions loop outside it with the page sizes alternating, so
# configurations and page sizes are interleaved in time. Perf (uncore CAS counters, UPI, RAPL) runs for the whole process and
# is cut into one burst per window by tools/concurrent_tier.py. Hosts, paths and credentials never appear here.
set -euo pipefail
: "${ROOT:?tree under test required}"
# shellcheck source=lib/cresco8-common.sh
source "$ROOT/scripts/lib/cresco8-common.sh"
require_env
STAGES=${STAGES:-smoke}
REPS=${REPS:-7}
BASE_REPS=${BASE_REPS:-3}
PAGES=${PAGES:-"2m 4k"}
REGION_MIB=${REGION_MIB:-20480}
INTER_MIB=${INTER_MIB:-32768}
PRIOR=${PRIOR:-}
FILL=${FILL:-random}
SHUFFLE=${SHUFFLE:-0}
BEST_ENV=${BEST_ENV:-}
# sbatch --export cannot carry spaces: lists may be given with '+' instead.
STAGES=${STAGES//+/ }
PAGES=${PAGES//+/ }
PRIOR=${PRIOR//+/ }
SUMMARY_ARGS=${SUMMARY_ARGS:-}
SUMMARY_ARGS=${SUMMARY_ARGS//+/ }
job_begin
# Fixed explicit map: S0 = cores 0-55, DDR node 0, HBM node 2; S1 = cores 56-111, DDR node 1, HBM node 3.
TOPOLOGY='explicit:0/2/0,1/3/1'
# DDR CAS reads count 64-byte lines, HBM CAS reads 32-byte units (job 5892172); UPI data flits; RAPL package and DRAM energy;
# unhalted and reference cycles (the average core frequency of the window).
MEM_EVENTS='uncore_imc/event=0x05,umask=0xcf,name=ddr_cas_rd/,uncore_type_14/event=0x05,umask=0xcf,name=hbm_cas_rd/,unc_upi_rxl_flits.all_data,unc_upi_txl_flits.all_data,power/energy-pkg/,power/energy-ram/,cycles,ref-cycles'
mkdir -p "$OUT/raw" "$OUT/cfg"
failures=0
BIN=stream-bench

environment_extra() {
  {
    printf '\n== concurrent-tier environment ==\n'
    printf 'stages=%s reps=%s base_reps=%s pages=%s region_mib=%s inter_mib=%s prior=%s best_env=%s\n' \
      "$STAGES" "$REPS" "$BASE_REPS" "$PAGES" "$REGION_MIB" "$INTER_MIB" "$PRIOR" "$BEST_ENV"
    printf 'kernel=%s\n' "$(uname -r)"
    printf 'weighted_interleave_sysfs=%s\n' "$([[ -d /sys/kernel/mm/mempolicy/weighted_interleave ]] && echo present || echo absent)"
    local f
    for f in enabled defrag shmem_enabled khugepaged/defrag; do
      printf 'thp_%s=%s\n' "${f//\//_}" "$(cat "/sys/kernel/mm/transparent_hugepage/$f" 2> /dev/null || echo n/a)"
    done
    printf 'numa_balancing=%s\n' "$(cat /proc/sys/kernel/numa_balancing 2> /dev/null || echo n/a)"
    printf 'hugepages_total=%s\n' "$(grep -E '^HugePages_Total' /proc/meminfo | tr -d ' ')"
    printf 'perf=%s\n' "$(perf --version 2> /dev/null || echo n/a)"
    printf 'sudo_list: %s\n' "$(sudo -n -l 2>&1 | tr '\n' ' ' | cut -c1-2000)"
    printf 'msr_device=%s rdmsr=%s wrmsr=%s\n' "$([[ -e /dev/cpu/0/msr ]] && echo present || echo absent)" \
      "$(command -v rdmsr || echo absent)" "$(command -v wrmsr || echo absent)"
    printf 'prefetch_msr_0x1a4: %s\n' "$(rdmsr -p 0 0x1a4 2>&1 | head -1 || true)"
    local d
    for d in /sys/devices/system/cpu/intel_uncore_frequency/package_*_die_*; do
      [[ -d $d ]] || continue
      printf 'uncore %s min_khz=%s max_khz=%s cur_khz=%s writable=%s\n' "$(basename "$d")" "$(cat "$d/min_freq_khz" 2> /dev/null)" \
        "$(cat "$d/max_freq_khz" 2> /dev/null)" "$(cat "$d/current_freq_khz" 2> /dev/null)" "$([[ -w $d/max_freq_khz ]] && echo yes || echo no)"
    done
    printf 'cpufreq_governor=%s\n' "$(cat /sys/devices/system/cpu/cpu0/cpufreq/scaling_governor 2> /dev/null || echo n/a)"
    printf 'cpufreq_cur_khz_cpu0=%s\n' "$(cat /sys/devices/system/cpu/cpu0/cpufreq/scaling_cur_freq 2> /dev/null || echo n/a)"
    printf 'ulimit_locked_kb=%s\n' "$(ulimit -l)"
  } >> "$OUT/environment.txt"
}

build_bench() {
  make -C "$ROOT" CC=gcc MODE=release build/release/stream-bench > "$OUT/build.log" 2>&1
  mkdir -p "$OUT/bin"
  cp "$ROOT/build/release/stream-bench" "$OUT/bin/stream-bench"
  cp "$ROOT/build/release/config" "$OUT/build-config.txt"
  sha256sum "$OUT/bin/stream-bench" > "$OUT/executables.sha256"
  BIN="$OUT/bin/stream-bench"
}

# run_proc LABEL PAGE SOCKETS CFGFILE [extra stream-bench arguments]: one process (one repetition of every configuration of
# the file), its stdout JSON lines, stderr, and the perf interval series.
run_proc() {
  local label=$1 page=$2 sockets=$3 cfg=$4
  shift 4
  local base="$OUT/raw/$label" stop="$OUT/raw/$label.stop"
  local args=(--exp "$label" --sockets "$sockets" --threads 56 --page "$page" --reps 1 --cfgs "$cfg" --region-mib "$REGION_MIB"
    --inter-mib "$INTER_MIB" --fill "$FILL" "$@")
  # SHUFFLE=1: the configurations of every process run in their own pseudo-random order (seeded by the repetition number).
  [[ $SHUFFLE == 0 ]] || args+=(--shuffle "${label##*-r}")
  if [[ $PERF == 1 ]]; then
    args+=(--perf-cmd "sudo -n /usr/bin/perf stat -a --per-socket -I 100 -x, -e '$MEM_EVENTS' -- sh -c 'until [ -e $stop ]; do sleep 0.05; done' 2> '$base.perf.csv'"
      --perf-out "$base.perf.csv" --perf-stop "$stop")
  fi
  if dry_run; then
    printf 'RUN %s: LLAMA_TOPOLOGY=%s stream-bench %s\n' "$label" "$TOPOLOGY" "${args[*]}"
    return 0
  fi
  printf '%s %s start\n' "$(date -Is)" "$label"
  local status=0
  LLAMA_TOPOLOGY=$TOPOLOGY LC_ALL=C "$BIN" "${args[@]}" > "$base.jsonl" 2> "$base.stderr" || status=$?
  rm -f "$stop"
  printf '%s\t%s\n' "$label" "$status" >> "$OUT/process-status.tsv"
  if [[ $status != 0 ]]; then
    printf 'process %s failed with status %s: %s\n' "$label" "$status" "$(tail -n 3 "$base.stderr" | tr '\n' ' ')" >&2
    failures=$((failures + 1))
  fi
  return 0
}

ct() {
  "$PYTHON" "$ROOT/tools/concurrent_tier.py" "$@"
}

# rounds STAGE REPS SOCKETS [extra arguments]: the configurations of STAGE, REPS times, page sizes alternating inside a repetition.
rounds() {
  local stage=$1 reps=$2 sockets=$3
  shift 3
  local cfg="$OUT/cfg/$stage.cfg" rep page
  local best=()
  [[ -z $BEST_ENV ]] || best=(--best-env "$BEST_ENV")
  # shellcheck disable=SC2086  # PRIOR is a list of directories
  ct cfgs "$stage" ${best[@]+"${best[@]}"} --results "$OUT" $PRIOR > "$cfg"
  printf 'stage %s: %s configurations, %s repetitions, pages %s\n' "$stage" "$(wc -l < "$cfg")" "$reps" "$PAGES"
  for rep in $(seq 1 "$reps"); do
    for page in $PAGES; do
      run_proc "$stage-$page-r$rep" "$page" "$sockets" "$cfg" "$@"
    done
  done
}

dry_run || {
  record_environment
  check_flat_mode
  environment_extra
  build_bench
}
if dry_run; then
  PERF=1
else
  EVENTS=$MEM_EVENTS
  perf_available
fi
printf 'perf_events=%s\n' "$MEM_EVENTS" >> "$OUT/environment.txt"
[[ -z $BEST_ENV ]] || cp "$BEST_ENV" "$OUT/best-input.env"

for stage in $STAGES; do
  case $stage in
    smoke)
      rounds smoke 1 1
      best=()
      [[ -z $BEST_ENV ]] || best=(--best-env "$BEST_ENV")
      ct cfgs smoke-inter ${best[@]+"${best[@]}"} >"$OUT/cfg/smoke-inter.cfg"
      for page in $PAGES; do
        run_proc "smoke-inter-$page-r1" "$page" 1 "$OUT/cfg/smoke-inter.cfg" --region-mib 2048 --inter-mib 8192
      done
      for page in $PAGES; do
        run_proc "smoke2-$page-r1" "$page" 2 "$OUT/cfg/smoke.cfg"
      done
      ;;
    base)
      rounds base "$BASE_REPS" 1
      if ! dry_run; then
        ct best "$OUT" > "$OUT/best.env"
        BEST_ENV="$OUT/best.env"
        cat "$BEST_ENV"
      fi
      ;;
    e1) rounds e1 "$REPS" 1 ;;
    e2) rounds e2 "$REPS" 1 ;;
    e3) rounds e3 "$REPS" 1 --region-mib 1024 ;;
    e4) rounds e4 "$REPS" 2 ;;
    e5) rounds e5 "$REPS" 1 ;;
    e5c) rounds e5c "$REPS" 1 ;;
    e5off) rounds e5off "$REPS" 1 ;;
    e1g) rounds e1g "$REPS" 1 ;;
    e3g) PAGES=2m rounds e3g "$REPS" 1 --region-mib 1024 ;;
    e6b) rounds e6b "$REPS" 1 ;;
    hbmx) rounds hbmx "$REPS" 1 ;;
    e5alias) rounds e5alias "$REPS" 1 ;;
    e5c2) rounds e5c2 "$REPS" 1 ;;
    repe3) rounds repe3 "$REPS" 1 --region-mib 1024 ;;
    repe4) rounds repe4 "$REPS" 2 ;;
    repe5) rounds repe5 "$REPS" 1 ;;
    fillcmp)
      # The same reads over pseudo-random and over zero-filled tier regions, alternating inside each repetition.
      for tag in random zero; do ct cfgs "fillcmp-$tag" ${BEST_ENV:+--best-env "$BEST_ENV"} > "$OUT/cfg/fillcmp-$tag.cfg"; done
      for rep in $(seq 1 "$REPS"); do
        for tag in random zero; do
          run_proc "fc$tag-2m-r$rep" 2m 1 "$OUT/cfg/fillcmp-$tag.cfg" --fill "$tag"
        done
      done
      ;;
    e5hbm) rounds e5hbm "$REPS" 1 ;;
    e6) rounds e6 "$REPS" 1 ;;
    lat) rounds lat "$REPS" 1 ;;
    freq)
      # Core and uncore frequency (turbostat, the only privileged tool besides perf this account may run) under long HBM-only,
      # split and DDR-only windows; every line carries the wall-clock time it was read, to be matched with real_begin/real_end.
      best=()
      [[ -z $BEST_ENV ]] || best=(--best-env "$BEST_ENV")
      # shellcheck disable=SC2086
      ct cfgs freq ${best[@]+"${best[@]}"} --results "$OUT" $PRIOR > "$OUT/cfg/freq.cfg"
      if ! dry_run; then
        (sudo -n /usr/bin/turbostat --quiet --interval 1 --num_iterations 90 --cpu 0,1,56,57 2>&1 |
          while IFS= read -r line; do printf '%s %s\n' "$(date +%s.%N)" "$line"; done > "$OUT/raw/freq.turbostat.txt") &
        ts_pid=$!
      fi
      run_proc "freq-2m-r1" 2m 1 "$OUT/cfg/freq.cfg"
      dry_run || wait "$ts_pid" || true
      ;;
    rep) rounds rep "$REPS" 1 ;;
    rep-e5) rounds rep-e5 "$REPS" 1 ;;
    *) die "unknown stage '$stage'" ;;
  esac
done
dry_run || ct summarize "$OUT" $PRIOR ${SUMMARY_ARGS:-} -o "$OUT/summary" || printf 'summarize failed (the raw files are complete)\n' >&2
printf 'failed processes: %s\n' "$failures"
[[ $failures == 0 ]]
