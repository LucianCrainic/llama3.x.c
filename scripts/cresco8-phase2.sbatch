#!/usr/bin/env bash
#SBATCH --job-name=llama-phase2
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=02:00:00
# Submit with verified --partition/--nodelist/--exclude and separate -o/-e paths.
# Required: HF_MODEL, ARTIFACTS, OUT. Optional: ROOT, PYTHON, PRECISION, REPEATS.
set -euo pipefail
: "${SLURM_JOB_ID:?submit through Slurm}"
: "${HF_MODEL:?local HF snapshot required}"
: "${ARTIFACTS:?converted model directory required}"
: "${OUT:?unique results directory required}"
ROOT=${ROOT:-$SLURM_SUBMIT_DIR}
PYTHON=${PYTHON:-python3}
PRECISION=${PRECISION:-fp32}
REPEATS=${REPEATS:-3}
WORKLOADS=${WORKLOADS:-"short long"}
LOCAL_STAGE=${LOCAL_STAGE:-1}
SMOKE_ONLY=${SMOKE_ONLY:-0}
[[ $SMOKE_ONLY == 0 || $SMOKE_ONLY == 1 ]]
[[ $LOCAL_STAGE == 0 || $LOCAL_STAGE == 1 ]]
stage_dir=
[[ $PRECISION == fp32 || $PRECISION == int8 ]]
[[ $REPEATS =~ ^[1-9][0-9]*$ ]]
mkdir -p "$ARTIFACTS"
# Never overwrite a previous experiment.
mkdir "$OUT"
finish() {
  status=$?
  printf 'exit_code=%s\n' "$status" > "$OUT/exit-status.txt"
  [[ -z $stage_dir ]] || rm -rf -- "$stage_dir"
  exit "$status"
}
trap finish EXIT
cd "$ROOT"
exec > >(tee "$OUT/job.stdout") 2> >(tee "$OUT/job.stderr" >&2)
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK OMP_DYNAMIC=FALSE OMP_PROC_BIND=close OMP_PLACES=cores
export USE_TF=0 USE_FLAX=0
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
{
  date -u +%FT%TZ; hostname; uname -a; lscpu; numactl -H
  scontrol show job "$SLURM_JOB_ID"
  git rev-parse HEAD; git status --short; git diff --binary HEAD
  "${CC:-gcc}" --version; pkg-config --modversion libpcre2-8
  printf 'HF_MODEL=%s\nPRECISION=%s\nOMP_NUM_THREADS=%s\n' "$HF_MODEL" "$PRECISION" "$OMP_NUM_THREADS"
  printf 'LOCAL_STAGE=%s SMOKE_ONLY=%s; validation and smoke precede repeats; no cache eviction\n' "$LOCAL_STAGE" "$SMOKE_ONLY"
} > "$OUT/environment.txt"
# ACPI memory-side cache evidence is required, rather than a node-name guess.
"$PYTHON" - <<'PY' > "$OUT/cache-mode.json"
import glob, json
from pathlib import Path
sizes = {p: int(Path(p).read_text()) for p in glob.glob('/sys/devices/system/node/node*/memory_side_cache/index*/size')}
assert len(sizes) >= 2 and sum(sizes.values()) >= 128 * 1024**3, 'HBM Cache Mode evidence missing'
assert 'Max 9480' in Path('/proc/cpuinfo').read_text(), 'unexpected CPU model'
print(json.dumps(dict(memory_side_cache_bytes=sizes, mode='cache'), indent=2))
PY
make CC="${CC:-gcc}" OPENMP=1 all build/release/probe-fp32 build/release/probe-int8
make CC="${CC:-gcc}" OPENMP=1 test
# Preserve the exact source and build recipe even for an uncommitted checkout.
find src include tools tests scripts -type f ! -path '*/__pycache__/*' -print0 | sort -z | xargs -0 sha256sum > "$OUT/source.sha256"
cp Makefile "$OUT/Makefile"; cp build/release/config "$OUT/build-config.txt"
"$PYTHON" -m pip freeze > "$OUT/python-freeze.txt"
checkpoint="$ARTIFACTS/$PRECISION-v3.bin"
tokenizer="$ARTIFACTS/tokenizer-v2.bin"
if [[ ! -s $tokenizer ]]; then
  "$PYTHON" tools/tokenizer.py --hf "$HF_MODEL" -o "$tokenizer.tmp"
  mv "$tokenizer.tmp" "$tokenizer"
fi
if [[ ! -s $checkpoint ]]; then
  extra=(); [[ $PRECISION != int8 ]] || extra+=(--int8)
  "$PYTHON" tools/export.py "$checkpoint.tmp" --hf "$HF_MODEL" --version 3 "${extra[@]}"
  mv "$checkpoint.tmp" "$checkpoint"
fi
sha256sum "$checkpoint" "$tokenizer" > "$OUT/artifacts.sha256"
# Stage once to node-local storage so repeated mmap page-ins do not repeatedly
# measure shared-filesystem latency. This is file I/O, not NUMA memory placement.
if [[ $LOCAL_STAGE == 1 ]]; then
  stage_dir=$(mktemp -d "${SLURM_TMPDIR:-/tmp}/llama-phase2-${SLURM_JOB_ID}-XXXXXX")
  "$PYTHON" - "$checkpoint" "$stage_dir" <<'PY_STAGE'
import os, shutil, sys
from pathlib import Path
source, stage = map(Path, sys.argv[1:])
assert shutil.disk_usage(stage).free > source.stat().st_size * 1.1, 'insufficient node-local checkpoint staging space'
PY_STAGE
  /usr/bin/time -p -o "$OUT/staging-time.txt" cp "$checkpoint" "$stage_dir/checkpoint.bin"
  checkpoint="$stage_dir/checkpoint.bin"
  sha256sum "$checkpoint" > "$OUT/staged-artifact.sha256"
  read -r original_hash original_path < "$OUT/artifacts.sha256"
  read -r staged_hash staged_path < "$OUT/staged-artifact.sha256"
  [[ $original_hash == "$staged_hash" ]] || { echo 'staged checkpoint checksum mismatch' >&2; exit 1; }
  df -h "$stage_dir" > "$OUT/staging-filesystem.txt"
fi
# Numerical validation always precedes smoke/performance runs.
extra=(); [[ $PRECISION != int8 ]] || extra+=(--max-abs "${INT8_ATOL:-1.0}" --rtol "${INT8_RTOL:-0.05}")
srun --cpu-bind=cores env OMP_PROC_BIND=false "$PYTHON" tools/validate.py --hf "$HF_MODEL" --checkpoint "$checkpoint" \
  --tokenizer "$tokenizer" --precision "$PRECISION" --threads "$OMP_NUM_THREADS" \
  --output "$OUT/correctness.json" "${extra[@]}" > "$OUT/correctness.stdout" 2> "$OUT/correctness.stderr"
runner=build/release/run; [[ $PRECISION != int8 ]] || runner=build/release/runq
srun --cpu-bind=cores "$runner" "$checkpoint" -z "$tokenizer" --context-length 256 --benchmark --chat-template \
  -i 'Write one short sentence about the sky.' -t 0 --seed 42 --generation-length 16 \
  --json "$OUT/smoke.json" > "$OUT/smoke.stdout" 2> "$OUT/smoke.stderr"
if [[ $SMOKE_ONLY == 1 ]]; then exit 0; fi
"$PYTHON" - "$OUT" <<'PY'
import sys
from pathlib import Path
out = Path(sys.argv[1])
sentence = 'Explain how a CPU executes instructions and uses memory. '
(out/'short.txt').write_text(sentence * 5)
facts = [
    'The program counter identifies the next instruction to fetch.',
    'The instruction decoder determines which operation and operands are needed.',
    'Registers hold a small set of values close to the execution units.',
    'Arithmetic units add, multiply, compare and combine register values.',
    'Loads read values from memory and stores write values back to memory.',
    'Caches retain recently used data and instructions near the processor.',
    'A cache miss requires a request to a more distant level of memory.',
    'Conditional branches choose which instruction sequence follows a comparison.',
    'Dependencies constrain when instructions can execute relative to each other.',
    'Operating systems schedule programs and maintain their address spaces.',
]
context = '\n'.join(f'{i+1}. {facts[i % len(facts)]}' for i in range(30))
(out/'long.txt').write_text('Reference notes for an introductory computing tutorial:\n' + context +
    '\nUsing these notes, write a detailed tutorial of at least 400 words explaining a complete instruction cycle, '
    'registers, caches, main memory, branches and dependencies. Connect the steps in paragraphs with concrete examples.')
PY
for workload in $WORKLOADS; do
  [[ $workload == short || $workload == long ]]
  for ((run=1; run<=REPEATS; run++)); do
    srun --cpu-bind=cores "$runner" "$checkpoint" -z "$tokenizer" --context-length 2048 \
      --benchmark --chat-template --prompt-file "$OUT/$workload.txt" -t 0 --seed 42 \
      --generation-length 128 --no-text --log-level warn --json "$OUT/$workload-$run.json" \
      > "$OUT/$workload-$run.stdout" 2> "$OUT/$workload-$run.stderr"
  done
done
"$PYTHON" tools/summarize.py "$OUT" > "$OUT/summary.json"
