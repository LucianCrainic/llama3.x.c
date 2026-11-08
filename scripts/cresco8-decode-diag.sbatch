#!/usr/bin/env bash
#SBATCH --job-name=llama-decode-diag
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=112
#SBATCH --exclusive
#SBATCH --mem=0
#SBATCH --time=01:00:00
# Diagnosis of the two-socket decode (plan B7, first steps). First the streaming ceiling of the memory: tests/stream_bench.c
# reads HBM with one pinned team per socket (raw streams, the GEMV's four-row pattern, and the engine's own kernels on
# the 8B shard shapes), on one socket and on both at once. Then, interleaved over REPEATS launches and with
# LLAMA_PROFILE=1, the same short decode in these configurations on each model/precision of MODELS:
#   1s      one socket (the A6 configuration: 56 teams workers, weights on HBM node 2)           -- tree ROOT
#   2s-old  LLAMA_SOCKETS=2 with the tree OLD_ROOT (the build before the leader-pinning fix)   -- optional
#   2s-sep  LLAMA_SOCKETS=2, LLAMA_TP_FUSED=0 (projection and reduction as two team jobs)
#   2s      LLAMA_SOCKETS=2, fused (the default)
# The info log of every two-socket launch carries the per-socket phase times and the split of the reduction.
# While every two-socket launch runs, the affinity mask and the running CPU of each of its threads are sampled from
# /proc, so a leader pinned to the wrong CPU shows up directly. Required: ROOT, OUT (new directory), PYTHON, MODELS
# ("NAME:ARTIFACT_DIR:HF_SNAPSHOT ..."). Optional: OLD_ROOT (an already built tree), PRECISIONS, REPEATS, DRY_RUN=1.
# STREAM=0 skips the streaming benchmark, DECODE_MODELS= (empty) skips the decode runs. Every submission needs the owner's go-ahead.
set -euo pipefail
: "${ROOT:?tree under test required}"
source "$ROOT/scripts/lib/cresco8-common.sh"
require_env
[[ -n $MODELS ]] || die "MODELS is required"
job_begin
record_environment
dir="$OUT/D"
mkdir "$dir"
# On hbm14: S0 = cores 0-55, DDR 0, HBM 2; S1 = cores 56-111, DDR 1, HBM 3.
EXPLICIT='explicit:0/2/0,1/3/1'
OLD_ROOT=${OLD_ROOT:-}
CONFIGS=${CONFIGS:-"1s 2s-sep 2s"} # any of: 1s 2s 2s-sep and {1s,2s}-{static,dynamic,hybrid} (LLAMA_GEMV_SCHED; plain 1s and 2s use auto)
SAMPLE=${SAMPLE:-0} # 1 samples every thread's CPU from /proc during two-socket launches (it perturbs the timing)
TWO_ENV=(LLAMA_BACKEND=teams LLAMA_THREADS=56 OMP_PROC_BIND=false "LLAMA_TOPOLOGY=$EXPLICIT" LLAMA_PLACEMENT=hbm LLAMA_SOCKETS=2
  LLAMA_PROFILE=1)
ONE_ENV=(LLAMA_BACKEND=teams LLAMA_THREADS=56 OMP_PROC_BIND=false LLAMA_PROFILE=1)

if ! dry_run; then
  build_tree 1 release build/release/probe-fp32 build/release/stream-bench
  if [[ -n $OLD_ROOT && ! -x $OLD_ROOT/build/release/run ]]; then
    make -C "$OLD_ROOT" CC=gcc OPENMP=1 MODE=release all > "$dir/build-old.log"
  fi
fi

if ! dry_run && [[ ${STREAM:-1} == 1 ]]; then
  for sockets in 1 2; do
    for threads in 56 28; do
      LLAMA_TOPOLOGY=$EXPLICIT "$ROOT/build/release/stream-bench" --sockets "$sockets" --threads "$threads" --mib 4096 \
        --reps 7 > "$dir/stream-s$sockets-t$threads.jsonl" 2> "$dir/stream-s$sockets-t$threads.err" ||
        echo "stream-bench $sockets/$threads failed" >> "$dir/failures.txt"
    done
  done
fi

# sample_threads PID FILE: every 4 s, one line per thread: tid, allowed CPU list, CPU it last ran on.
sample_threads() {
  local pid=$1 file=$2 n=0 t
  while kill -0 "$pid" 2> /dev/null && ((n < 12)); do
    sleep 4
    n=$((n + 1))
    {
      echo "## t+$((n * 4))s"
      for t in /proc/"$pid"/task/*; do
        printf '%s %s %s\n' "$(basename "$t")" "$(awk '/^Cpus_allowed_list/ {print $2}' "$t/status" 2> /dev/null)" \
          "$(awk '{print $39}' "$t/stat" 2> /dev/null)"
      done
    } >> "$file" 2> /dev/null || true
  done
}

# diag_run TAG TREE LABEL ENGINE CHECKPOINT TOKENIZER REP SAMPLE CPU_BIND MEM_BIND ENV...
diag_run() {
  local tag=$1 tree=$2 label=$3 engine=$4 checkpoint=$5 tokenizer=$6 rep=$7 sample=$8 cpu=$9 mem=${10}
  shift 10
  local prefix="$dir/$label-$rep" launcher=()
  [[ -z $cpu ]] || launcher=(numactl --cpunodebind="$cpu" --membind="$mem")
  if dry_run; then
    printf 'RUN %s rep=%s tree=%s sample=%s cpu=%s mem=%s env=%s\n' "$label" "$rep" "$tree" "$sample" "$cpu" "$mem" "$*" |
      tee -a "$OUT/dry-run.txt"
    return 0
  fi
  env OMP_PROC_BIND=false "$@" "${launcher[@]}" "$tree/build/release/$engine" "$checkpoint" -z "$tokenizer" \
    --context-length 2048 --benchmark --chat-template --prompt-file "$OUT/short.txt" -t 0 --seed 42 \
    --generation-length 128 --no-text --log-level info --json "$prefix.json" > "$prefix.stdout" 2> "$prefix.stderr" &
  local pid=$!
  [[ $sample == 1 ]] && sample_threads "$pid" "$prefix.threads" &
  local sampler=$!
  wait "$pid" || echo "run $label-$rep failed" >> "$dir/failures.txt"
  [[ $sample == 1 ]] && wait "$sampler" 2> /dev/null || true
}

need_shm
for entry in ${DECODE_MODELS-$MODELS}; do
  IFS=: read -r name artifacts hf <<< "$entry"
  tokenizer="$artifacts/tokenizer-v2.bin"
  make_prompt "$ROOT/build/release/probe-fp32" "$tokenizer" 128256 none
  for precision in $PRECISIONS; do
    engine=run
    [[ $precision == fp32 ]] || engine=runq
    one=$(stage_model_on 2 "$name" "$precision" "$artifacts")
    two=$(stage_model_on 0 "$name" "$precision" "$artifacts")
    # The configurations rotate through the three positions of a repeat, so no configuration always runs first or last.
    for ((rep = 1; rep <= REPEATS; rep++)); do
      order=($CONFIGS)
      [[ -z $OLD_ROOT ]] || order+=(2s-old)
      for ((k = 0; k < ${#order[@]}; k++)); do
        config=${order[$(((k + rep - 1) % ${#order[@]}))]}
        case $config in
          1s) diag_run 1s "$ROOT" "$name-$precision-1s" "$engine" "$one" "$tokenizer" "$rep" 0 0 2 "${ONE_ENV[@]}" ;;
          1s-static | 1s-dynamic | 1s-hybrid)
            diag_run "$config" "$ROOT" "$name-$precision-$config" "$engine" "$one" "$tokenizer" "$rep" 0 0 2 "${ONE_ENV[@]}" \
              "LLAMA_GEMV_SCHED=${config#1s-}" ;;
          2s-old) diag_run 2s-old "$OLD_ROOT" "$name-$precision-2s-old" "$engine" "$two" "$tokenizer" "$rep" "$SAMPLE" "" "" "${TWO_ENV[@]}" ;;
          2s-sep) diag_run 2s-sep "$ROOT" "$name-$precision-2s-sep" "$engine" "$two" "$tokenizer" "$rep" "$SAMPLE" "" "" "${TWO_ENV[@]}" LLAMA_TP_FUSED=0 ;;
          2s) diag_run 2s "$ROOT" "$name-$precision-2s" "$engine" "$two" "$tokenizer" "$rep" "$SAMPLE" "" "" "${TWO_ENV[@]}" ;;
          2s-static | 2s-dynamic | 2s-hybrid)
            diag_run "$config" "$ROOT" "$name-$precision-$config" "$engine" "$two" "$tokenizer" "$rep" "$SAMPLE" "" "" \
              "${TWO_ENV[@]}" "LLAMA_GEMV_SCHED=${config#2s-}" ;;
          *) die "unknown configuration $config" ;;
        esac
      done
    done
    unstage_model_on 2 "$name" "$precision"
    unstage_model_on 0 "$name" "$precision"
  done
done

if ! dry_run; then
  "$PYTHON" "$ROOT/tools/summarize_decode_diag.py" "$dir" > "$dir/summary.md" || true
  "$PYTHON" "$ROOT/tools/summarize_profile.py" "$dir" > "$dir/profile.md" || true
fi
