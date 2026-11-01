#!/usr/bin/env bash
#SBATCH --job-name=llama-kernels
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=112
#SBATCH --exclusive
#SBATCH --mem=0
#SBATCH --time=04:00:00
# Flat Mode kernel campaign on a dual Xeon Max 9480 node. Submit with a verified
# --partition/--nodelist and separate -o/-e paths.
# Required: SIMD_ROOT (tree under test), BASE_ROOT (comparison tree), OUT (new
# directory), PYTHON (conversion/validation environment) and MODELS, a space-
# separated list of NAME:ARTIFACT_DIR:HF_SNAPSHOT entries whose artifact
# directory holds fp32-v3.bin, int8-v3.bin and tokenizer-v2.bin.
# STAGE=check builds, tests and validates only; STAGE=bench (default) adds the
# benchmark matrix. REPEATS defaults to three for the headline configurations.
set -euo pipefail
: "${SLURM_JOB_ID:?submit through Slurm}"
: "${SIMD_ROOT:?tree under test required}"
: "${BASE_ROOT:?comparison tree required}"
: "${OUT:?unique results directory required}"
: "${PYTHON:?validation Python required}"
: "${MODELS:?NAME:ARTIFACTS:HF entries required}"
STAGE=${STAGE:-bench}
REPEATS=${REPEATS:-3}
PRECISIONS=${PRECISIONS:-"fp32 int8"}
[[ $STAGE == check || $STAGE == bench ]]
[[ $REPEATS =~ ^[1-9][0-9]*$ ]]
mkdir "$OUT"
shm=
finish() {
  status=$?
  printf 'exit_code=%s\n' "$status" > "$OUT/exit-status.txt"
  [[ -z $shm ]] || rm -rf -- "$shm"
  exit "$status"
}
trap finish EXIT
exec > >(tee "$OUT/job.stdout") 2> >(tee "$OUT/job.stderr" >&2)
export OMP_DYNAMIC=FALSE OMP_PROC_BIND=close OMP_PLACES=cores
export USE_TF=0 USE_FLAX=0 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false

{
  date -u +%FT%TZ; hostname; uname -a; lscpu; numactl -H
  scontrol show job "$SLURM_JOB_ID"
  gcc --version; pkg-config --modversion libpcre2-8
  for tree in "$SIMD_ROOT" "$BASE_ROOT"; do printf "%s %s\n" "$tree" "$(cat "$tree/COMMIT" 2>/dev/null || git -C "$tree" rev-parse HEAD)"; done
  printf 'STAGE=%s REPEATS=%s PRECISIONS=%s MODELS=%s\n' "$STAGE" "$REPEATS" "$PRECISIONS" "$MODELS"
} > "$OUT/environment.txt"
for tree in "$SIMD_ROOT" "$BASE_ROOT"; do
  (cd "$tree" && find src include tests tools scripts Makefile -type f ! -path '*/__pycache__/*' -print0 | sort -z |
    xargs -0 sha256sum) > "$OUT/source-$(basename "$tree").sha256"
done

# Flat Mode evidence: no ACPI memory-side cache and two CPU-less ~64 GiB HBM nodes.
python3 - <<'PY' > "$OUT/flat-mode.json"
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

for tree in "$BASE_ROOT" "$SIMD_ROOT"; do
  make -C "$tree" CC=gcc OPENMP=1 all build/release/probe-fp32 build/release/probe-int8 > "$OUT/build-$(basename "$tree").log"
  cp "$tree/build/release/config" "$OUT/build-config-$(basename "$tree").txt"
done
(cd "$SIMD_ROOT" && OMP_NUM_THREADS=56 numactl --cpunodebind=0 --membind=0 make CC=gcc OPENMP=1 test) > "$OUT/test.log" 2>&1
for threads in 1 56 112; do
  OMP_NUM_THREADS=$threads "$SIMD_ROOT/build/release/test-kernels" > "$OUT/test-kernels-t$threads.txt"
done
grep -q 'avx512 ok' "$OUT/test-kernels-t56.txt" || { echo 'AVX-512 kernels were not exercised' >&2; exit 1; }

# Uncore CAS counters (64-byte lines) show which tier served the reads.
EVENTS='uncore_imc/event=0x05,umask=0xcf,name=ddr_cas_rd/,uncore_type_14/event=0x05,umask=0xcf,name=hbm_cas_rd/,unc_upi_rxl_flits.all_data'
PERF=0
if sudo -n /usr/bin/perf stat -a -x , -e "$EVENTS" -- true 2> /dev/null; then PERF=1; fi
printf 'perf_uncore=%s\n' "$PERF" >> "$OUT/environment.txt"

# HF validation with the kernels under test. FP32 and 1B W8A8 must pass their
# Phase 2 budgets; the other INT8 results are recorded without gating.
validate() { # name artifacts hf precision label [env...]
  local name=$1 artifacts=$2 hf=$3 precision=$4 label=$5 status=0
  shift 5
  local extra=()
  [[ $precision != int8 ]] || extra+=(--max-abs 1.0 --rtol 0.05)
  (cd "$SIMD_ROOT" && env "$@" OMP_PROC_BIND=false OMP_NUM_THREADS=56 numactl --cpunodebind=0 --membind=0 \
    "$PYTHON" tools/validate.py --hf "$hf" --checkpoint "$artifacts/$precision-v3.bin" \
    --tokenizer "$artifacts/tokenizer-v2.bin" --precision "$precision" --threads 56 \
    --output "$OUT/correctness-$name-$label.json" "${extra[@]}") \
    > "$OUT/correctness-$name-$label.stdout" 2> "$OUT/correctness-$name-$label.stderr" || status=$?
  printf '%s %s exit=%s\n' "$name" "$label" "$status" >> "$OUT/correctness-status.txt"
  return "$status"
}
for entry in $MODELS; do
  IFS=: read -r name artifacts hf <<< "$entry"
  for precision in $PRECISIONS; do
    if [[ $precision == fp32 ]]; then
      validate "$name" "$artifacts" "$hf" fp32 fp32
    else
      if [[ $name == 1b ]]; then validate "$name" "$artifacts" "$hf" int8 w8a8; else
        validate "$name" "$artifacts" "$hf" int8 w8a8 || true; fi
      validate "$name" "$artifacts" "$hf" int8 w8a32 LLAMA_Q8_ACTIVATIONS=fp32 || true
    fi
  done
done
[[ $STAGE == bench ]] || exit 0

sentence='Explain how a CPU executes instructions and uses memory. '
printf '%s%s%s%s%s' "$sentence" "$sentence" "$sentence" "$sentence" "$sentence" > "$OUT/short.txt"
shm=$(mktemp -d "/dev/shm/llama-kernels-$SLURM_JOB_ID-XXXXXX")

# The checkpoint is staged into tmpfs under an explicit memory policy, so its
# mapped pages live on the chosen tier; the process heap uses the same policy.
stage() { # source destination numactl-policy...
  local source=$1 destination=$2
  shift 2
  numactl "$@" cp "$source" "$destination"
  numactl -H > "$destination.numa.txt"
}
run() { # label tree engine checkpoint tokenizer cpunodes memnodes threads repeats perf [env...]
  local label=$1 tree=$2 engine=$3 checkpoint=$4 tokenizer=$5 cpus=$6 mems=$7 threads=$8 repeats=$9 perf=${10}
  shift 10
  for ((r = 1; r <= repeats; r++)); do
    local prefix="$OUT/$label-$r"
    env "$@" OMP_NUM_THREADS="$threads" numactl --cpunodebind="$cpus" --membind="$mems" \
      "$tree/build/release/$engine" "$checkpoint" -z "$tokenizer" --context-length 2048 --benchmark \
      --chat-template --prompt-file "$OUT/short.txt" -t 0 --seed 42 --generation-length 128 --no-text \
      --log-level warn --json "$prefix.json" > "$prefix.stdout" 2> "$prefix.stderr" &
    local pid=$!
    if [[ $PERF == 1 && $perf == 1 && $r == 1 ]]; then
      sudo -n /usr/bin/perf stat -a --per-socket -x , -e "$EVENTS" -- tail -s 0.1 --pid="$pid" -f /dev/null \
        2> "$prefix.perf.csv" || true
    fi
    wait "$pid"
  done
}

for entry in $MODELS; do
  IFS=: read -r name artifacts hf <<< "$entry"
  tokenizer="$artifacts/tokenizer-v2.bin"
  for precision in $PRECISIONS; do
    engine=run; [[ $precision == fp32 ]] || engine=runq
    hbm="$shm/$name-$precision-hbm.bin" ddr="$shm/$name-$precision-ddr.bin" both="$shm/$name-$precision-hbm23.bin"
    stage "$artifacts/$precision-v3.bin" "$hbm" --membind=2
    stage "$artifacts/$precision-v3.bin" "$ddr" --membind=0
    stage "$artifacts/$precision-v3.bin" "$both" --interleave=2,3
    tag="$name-$precision"
    # Headline: one full socket, local HBM versus local DDR, old and new kernels.
    run "$tag-base-hbm-t56" "$BASE_ROOT" "$engine" "$hbm" "$tokenizer" 0 2 56 "$REPEATS" 1
    run "$tag-base-ddr-t56" "$BASE_ROOT" "$engine" "$ddr" "$tokenizer" 0 0 56 "$REPEATS" 1
    run "$tag-simd-hbm-t56" "$SIMD_ROOT" "$engine" "$hbm" "$tokenizer" 0 2 56 "$REPEATS" 1
    run "$tag-simd-ddr-t56" "$SIMD_ROOT" "$engine" "$ddr" "$tokenizer" 0 0 56 "$REPEATS" 1
    # Thread scaling on one socket.
    for threads in 14 28; do
      run "$tag-simd-hbm-t$threads" "$SIMD_ROOT" "$engine" "$hbm" "$tokenizer" 0 2 "$threads" 1 0
      run "$tag-simd-ddr-t$threads" "$SIMD_ROOT" "$engine" "$ddr" "$tokenizer" 0 0 "$threads" 1 0
    done
    # Two sockets without tensor parallelism: interleaved HBM, and all weights remote for socket 1.
    run "$tag-simd-hbm23-t112" "$SIMD_ROOT" "$engine" "$both" "$tokenizer" 0,1 2,3 112 1 1
    run "$tag-simd-hbm2-t112" "$SIMD_ROOT" "$engine" "$hbm" "$tokenizer" 0,1 2 112 1 1
    if [[ $precision == int8 ]]; then
      run "$tag-simd-w8a32-hbm-t56" "$SIMD_ROOT" "$engine" "$hbm" "$tokenizer" 0 2 56 "$REPEATS" 1 \
        LLAMA_Q8_ACTIVATIONS=fp32
    fi
    run "$tag-simd-hbm-t56-active" "$SIMD_ROOT" "$engine" "$hbm" "$tokenizer" 0 2 56 1 0 OMP_WAIT_POLICY=active
    rm -f -- "$hbm" "$ddr" "$both"
  done
done
python3 "$SIMD_ROOT/tools/summarize_kernels.py" "$OUT" > "$OUT/summary.md"
cat "$OUT/summary.md"
