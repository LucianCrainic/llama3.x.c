#!/usr/bin/env bash
#SBATCH --job-name=ttft-diagnosis
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=112
#SBATCH --exclusive
#SBATCH --mem=0
#SBATCH --time=03:00:00

# TTFT diagnosis of the hybrid 70B and 8B prefill (docs/ttft-70b-diagnosis.md). One Flat node, hybrid placement, two sockets of 55 threads
# (the conventions of scripts/xengine70.sbatch), engine defaults otherwise (AMX-INT8 prefill, LLAMA_KERNELS=auto).
# Required environment: ROOT (staged tree, tracked files only), OUT (new directory), PYTHON (with transformers),
#   CHECKPOINT_70B, TOKENIZER_70B,
#   CHECKPOINT_8B, TOKENIZER_8B.
# Optional: DRY_RUN=1 (print the cells), CELLS (words of: plan 8b 70b avx512; default all).
# Per cell:
#   drv-*   tools/xengine70 (xengine70-q8): both prompts as exact id arrays, 1 warm-up + 3 measured requests of each, 128 greedy
#           tokens; the perf counters (DDR and HBM CAS reads and writes, UPI flits) run over the whole process and the prefill
#           windows are cut from the request start epoch and TTFT (tools/ttft_diagnosis_summary.py).
#   prof-*  tools/xengine70/native_profile.c, the same requests with the engine's phase profile reset before and read after each
#           prefill (warm process, exact fixture ids), 8 greedy tokens; the perf group is the compute events (cycles, instructions,
#           EXE.AMX_BUSY). It is built here with an extra makefile fragment: no engine file is touched.
# Nothing here names a host, a path or a credential: locations arrive through the environment.
set -uo pipefail

: "${ROOT:?}" "${OUT:?}" "${PYTHON:?}" "${CHECKPOINT_70B:?}" "${CHECKPOINT_8B:?}"
cells=${CELLS:-plan 8b 70b avx512}
context=8192
generation=128
cpus=0-54,56-110
perf=${PERF_BIN:-/usr/bin/perf}
interval=50
# Intel PCM: 32 bytes per HBM CAS and 64 per DDR CAS on Sapphire Rapids (docs/hbm-ddr-research.md).
mem_events='uncore_imc/event=0x05,umask=0xcf,name=ddr_cas_rd/,uncore_type_14/event=0x05,umask=0xcf,name=hbm_cas_rd/,uncore_imc/event=0x05,umask=0xf0,name=ddr_cas_wr/,uncore_type_14/event=0x05,umask=0xf0,name=hbm_cas_wr/,unc_upi_rxl_flits.all_data,unc_upi_txl_flits.all_data'
cmp_events='cycles,ref-cycles,instructions,EXE.AMX_BUSY'

if [[ ${DRY_RUN:-0} == 1 ]]; then
  echo "cells: $cells"
  echo "context=$context generation=$generation cpus=$cpus interval_ms=$interval"
  exit 0
fi
[[ ! -e $OUT ]] || { echo "OUT '$OUT' already exists" >&2; exit 2; }
mkdir "$OUT"
exec > >(tee "$OUT/job.stdout") 2> >(tee "$OUT/job.stderr" >&2)
export LC_ALL=C
{
  printf 'job=%s\n' "${SLURM_JOB_ID:-none}"
  hostname; date -Is; uname -a; gcc --version | head -1; numactl -H
  grep -E '^(Cpus|Mems)_allowed_list' /proc/self/status
  lscpu
  cat /sys/kernel/mm/transparent_hugepage/enabled
  "$perf" --version
  printf '%s %s\n' "$ROOT" "$(cat "$ROOT/COMMIT" 2> /dev/null || git -C "$ROOT" rev-parse HEAD)"
  stat -c '%n %s' "$CHECKPOINT_70B" "$CHECKPOINT_8B"
  printf 'cells=%s\ncontext=%s\ngeneration=%s\ncpus=%s\ninterval_ms=%s\nmem_events=%s\ncmp_events=%s\n' "$cells" "$context" "$generation" "$cpus" "$interval" "$mem_events" "$cmp_events"
} > "$OUT/environment.txt"
(cd "$ROOT" && find src include tests tools scripts Makefile -type f ! -path '*/__pycache__/*' -print0 | sort -z | xargs -0 sha256sum) > "$OUT/source.sha256"

# Every LLAMA_* variable of the caller is cleared, so a run sees exactly the environment of its command.
unset_args=()
while IFS= read -r name; do unset_args+=(-u "$name"); done < <(env | sed -n 's/^\(LLAMA_[A-Za-z0-9_]*\)=.*/\1/p')

build="$OUT/bin"
mkdir "$OUT/extra"
cp "$ROOT/tools/xengine70/native_profile.c" "$OUT/extra/native_profile.c" 2> /dev/null || cp "${PROFILE_SRC:?native_profile.c not in the staged tree: set PROFILE_SRC}" "$OUT/extra/native_profile.c"
cat > "$OUT/extra/profile.mk" << 'MK'
$(BUILD_DIR)/xengine70-profile-q8: $(PROFILE_SRC) src/runq.c src/engine.c $(COMMON) $(HEADERS) $(BUILD_DIR)/config
	$(CC) $(CPPFLAGS) $(CFLAGS) $(PROFILE_SRC) src/runq.c src/engine.c $(COMMON) $(LDFLAGS) $(LDLIBS) -o $@
MK
make -C "$ROOT" -f Makefile -f "$OUT/extra/profile.mk" -j 56 CC=gcc BUILD_DIR="$build" PROFILE_SRC="$OUT/extra/native_profile.c" \
  "$build/runq" "$build/xengine70-q8" "$build/xengine70-profile-q8" > "$OUT/build.log" 2>&1 || { tail -30 "$OUT/build.log"; exit 1; }
sha256sum "$build/runq" "$build/xengine70-q8" "$build/xengine70-profile-q8" "$OUT/extra/native_profile.c" > "$OUT/executables.sha256"

# Prompts: the exact id arrays, in the whitespace-delimited form the native drivers read.
prompts="$OUT/prompts"
mkdir "$prompts"
cp "$ROOT"/tools/xengine70/prompts/*.ids.json "$prompts/"
"$PYTHON" - "$prompts" << 'PY'
import json, sys
from pathlib import Path
p = Path(sys.argv[1])
for name in ("short", "long"):
    ids = json.loads((p / f"{name}.ids.json").read_text())
    (p / f"{name}.ids.txt").write_text(" ".join(map(str, ids)) + "\n")
PY

# ---- perf --------------------------------------------------------------------------------------------------------------------------
# A sentinel process stands for the workload: perf stat -a runs until the sentinel ends. The perf clock starts when the first interval
# line is printed (its time stamp is the time since the counters were enabled), so t0 = epoch at the first line - that stamp.
perf_ok_mem=1
perf_ok_cmp=1
if ! sudo -n "$perf" stat -a --per-socket -I 100 -x, -e "$mem_events" -- sleep 1 > /dev/null 2> "$OUT/perf-check-mem.txt"; then perf_ok_mem=0; fi
if ! sudo -n "$perf" stat -a --per-socket -I 100 -x, -e "$cmp_events" -- sleep 1 > /dev/null 2> "$OUT/perf-check-cmp.txt"; then perf_ok_cmp=0; fi
printf 'perf_ok_mem=%s perf_ok_cmp=%s\n' "$perf_ok_mem" "$perf_ok_cmp" | tee -a "$OUT/environment.txt"
sentinel=
perfpid=
start_perf() { # PREFIX GROUP
  local prefix=$1 group=$2 events ok
  case $group in mem) events=$mem_events ok=$perf_ok_mem ;; cmp) events=$cmp_events ok=$perf_ok_cmp ;; *) return 0 ;; esac
  [[ $ok == 1 ]] || return 0
  sleep 7200 &
  sentinel=$!
  sudo -n "$perf" stat -a --per-socket -I "$interval" -x, -e "$events" -- tail -s 0.05 --pid="$sentinel" -f /dev/null \
    > /dev/null 2> "$prefix.perf.csv" &
  perfpid=$!
  local waited=0
  until grep -q -E '^[0-9]' "$prefix.perf.csv" 2> /dev/null; do
    sleep 0.005
    waited=$((waited + 1))
    ((waited < 4000)) || { echo "perf produced no interval for $prefix" >&2; return 1; }
  done
  local now first
  now=$(date +%s%N)
  first=$(grep -m1 -E '^[0-9]' "$prefix.perf.csv" | cut -d, -f1)
  awk -v n="$now" -v f="$first" 'BEGIN { printf "%d\n", n - f * 1e9 }' > "$prefix.perf-t0"
}
stop_perf() {
  [[ -z $sentinel ]] || { kill "$sentinel" 2> /dev/null; wait "$sentinel" 2> /dev/null; }
  [[ -z $perfpid ]] || wait "$perfpid" 2> /dev/null
  sentinel=
  perfpid=
}

[[ $perf_ok_mem == 0 ]] || { start_perf "$OUT/idle" mem; sleep 5; stop_perf; }

# ---- runs --------------------------------------------------------------------------------------------------------------------------
engine_env() { # MODEL (70b|8b) BATCH PACKED  -> LLAMA_* assignments, one per line
  local model=$1 batch=$2 packed=$3 layers=80
  [[ $model == 70b ]] || layers=32
  printf '%s\n' LLAMA_SOCKETS=2 LLAMA_THREADS=55 "LLAMA_PREFILL_BATCH=$batch" LLAMA_PLACEMENT=hybrid LLAMA_PLACEMENT_STRICT=1 \
    LLAMA_VERIFY_SAMPLES=16384 "LLAMA_HYBRID_WEIGHTS_HBM_LAYERS=$layers" "LLAMA_HYBRID_KV_HBM_LAYERS=$layers" \
    LLAMA_HYBRID_CLASSIFIER=hbm
  [[ $packed == none ]] || printf 'LLAMA_HYBRID_PACKED=%s\n' "$packed" # the AVX-512 kernels have no packed copy: the knob is an error there
}

execute() { # PREFIX GROUP ENV... -- COMMAND...
  local prefix=$1 group=$2
  shift 2
  local assignments=()
  while [[ $1 != -- ]]; do assignments+=("$1"); shift; done
  shift
  start_perf "$prefix" "$group" || true
  date +%s%N > "$prefix.launch"
  env "${unset_args[@]}" "${assignments[@]}" "$@" > "$prefix.stdout" 2> "$prefix.stderr" &
  local pid=$!
  if [[ $prefix == *drv* ]]; then
    until [[ -s $prefix.stdout ]] || ! kill -0 "$pid" 2> /dev/null; do sleep 2; done
    numastat -p "$pid" > "$prefix.numastat" 2>&1 || true
  fi
  wait "$pid"
  local status=$?
  date +%s%N > "$prefix.end"
  stop_perf
  printf '%s rc=%s %s\n' "$(date -Is)" "$status" "$(basename "$prefix")" | tee -a "$OUT/runs.log"
  return "$status"
}

driver() { # LABEL MODEL(70b|8b) BATCH PACKED [EXTRA_ENV...]
  local label=$1 model=$2 batch=$3 packed=$4
  shift 4
  local ckpt=$CHECKPOINT_70B
  [[ $model == 70b ]] || ckpt=$CHECKPOINT_8B
  local assignments
  mapfile -t assignments < <(engine_env "$model" "$batch" "$packed")
  execute "$OUT/drv-$label" mem "${assignments[@]}" "$@" -- taskset -c "$cpus" "$build/xengine70-q8" "$ckpt" \
    "$prompts/short.ids.txt" "$prompts/long.ids.txt" "$context" "$batch"
}

prof() { # LABEL MODEL BATCH PACKED WORKLOADS GROUP [EXTRA_ENV...]
  local label=$1 model=$2 batch=$3 packed=$4 workloads=$5 group=$6
  shift 6
  local ckpt=$CHECKPOINT_70B
  [[ $model == 70b ]] || ckpt=$CHECKPOINT_8B
  local assignments
  mapfile -t assignments < <(engine_env "$model" "$batch" "$packed")
  execute "$OUT/prof-$label" "$group" "${assignments[@]}" "$@" -- taskset -c "$cpus" "$build/xengine70-profile-q8" "$ckpt" \
    "$prompts/short.ids.txt" "$prompts/long.ids.txt" "$context" "$batch" "$workloads" 8
}

plan() { # LABEL MODEL BATCH PACKED
  local label=$1 model=$2 batch=$3 packed=$4 ckpt=$CHECKPOINT_70B
  [[ $model == 70b ]] || ckpt=$CHECKPOINT_8B
  local assignments
  mapfile -t assignments < <(engine_env "$model" "$batch" "$packed")
  env "${unset_args[@]}" "${assignments[@]}" taskset -c "$cpus" "$build/runq" "$ckpt" --context-length "$context" --plan-only \
    --json "$OUT/plan-$label.json" > "$OUT/plan-$label.stdout" 2> "$OUT/plan-$label.stderr"
  local status=$?
  printf '%s rc=%s plan-%s\n' "$(date -Is)" "$status" "$label" | tee -a "$OUT/runs.log"
}

for cell in $cells; do
  case $cell in
    plan)
      for batch in 256 1024; do plan "70b-b$batch" 70b "$batch" ddr; done
      ;;
    8b)
      # 8B on two sockets, everything in HBM except (for ddr) the AMX packed copy: the cheap pair of the packed-copy question
      for batch in 256 1024; do
        for packed in hbm ddr; do
          driver "8b-$packed-b$batch" 8b "$batch" "$packed"
          prof "8b-$packed-b$batch" 8b "$batch" "$packed" long cmp
        done
      done
      ;;
    70b)
      for batch in 256 1024; do
        driver "70b-ddr-b$batch" 70b "$batch" ddr
        prof "70b-ddr-b$batch" 70b "$batch" ddr both cmp
      done
      ;;
    avx512)
      # the non-AMX reference: AVX-512 prefill kernels, same placement, the 547-token prompt through the command line
      prof "70b-b256-avx512" 70b 256 none short cmp LLAMA_KERNELS=avx512
      ;;
    *) echo "unknown cell '$cell'" >&2 ;;
  esac
done
echo finished
