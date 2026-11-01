#!/usr/bin/env bash
#SBATCH --job-name=llamacpp-compare
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=128G
#SBATCH --time=01:30:00
# Required: SOURCE (pinned llama.cpp checkout), HARNESS (this scripts directory),
# HF_MODEL, BASELINE (reports with short/long-1.json), C_PROBE, C_TOKENIZER,
# PYTHON (conversion environment), MODEL_LABEL, THREADS, ARTIFACTS, OUT.
# Select the same verified hardware and allocation policy as the original run
# on submission.
set -euo pipefail
: "${SLURM_JOB_ID:?submit through Slurm}"
if [[ -n ${TOOLCHAIN_MODULE:-} ]]; then
  set +eu
  source /etc/profile
  module load "$TOOLCHAIN_MODULE"
  module_status=$?
  set -eu
  [[ $module_status == 0 ]]
fi
for name in SOURCE HARNESS HF_MODEL BASELINE C_PROBE C_TOKENIZER PYTHON MODEL_LABEL THREADS ARTIFACTS OUT; do
  [[ -n ${!name:-} ]] || { echo "missing $name" >&2; exit 1; }
done
mkdir -p "$ARTIFACTS"
mkdir "$OUT"
cp "${BASH_SOURCE[0]}" "$OUT/job.sbatch"
stage=
finish() {
  status=$?
  echo "$status" > "$OUT/exit-status.txt"
  [[ -z $stage ]] || rm -rf -- "$stage"
  exit "$status"
}
trap finish EXIT
exec > >(tee "$OUT/job.stdout") 2> >(tee "$OUT/job.stderr" >&2)
export OMP_NUM_THREADS=$THREADS OMP_DYNAMIC=FALSE OMP_PROC_BIND=false
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
export USE_TF=0 USE_FLAX=0
{
  date -u +%FT%TZ; hostname; lscpu; numactl -H
  scontrol show job "$SLURM_JOB_ID"
  git -C "$SOURCE" rev-parse HEAD; git -C "$SOURCE" status --short
  "$PYTHON" -m pip freeze
} > "$OUT/environment.txt"
"$PYTHON" - <<'PY' > "$OUT/cache-mode.json"
import glob, json
from pathlib import Path
sizes = {p: int(Path(p).read_text()) for p in glob.glob('/sys/devices/system/node/node*/memory_side_cache/index*/size')}
assert len(sizes) >= 2 and sum(sizes.values()) >= 128 * 1024**3
assert 'Max 9480' in Path('/proc/cpuinfo').read_text()
print(json.dumps(sizes, indent=2))
PY
build="$ARTIFACTS/build"
cmake -S "$SOURCE" -B "$build" -DCMAKE_BUILD_TYPE=Release \
  -DGGML_NATIVE=ON -DGGML_BLAS=OFF -DGGML_CUDA=OFF -DGGML_OPENMP=ON \
  -DLLAMA_BUILD_TESTS=OFF -DLLAMA_BUILD_TOOLS=OFF -DLLAMA_BUILD_EXAMPLES=OFF \
  -DLLAMA_BUILD_SERVER=OFF -DLLAMA_OPENSSL=OFF
cmake --build "$build" --target llama -j "$SLURM_CPUS_PER_TASK"
cp "$build/CMakeCache.txt" "$OUT/"
c++ -O3 -std=c++17 "$HARNESS/llamacpp-compare.cpp" \
  -I"$SOURCE/include" -I"$SOURCE/ggml/include" -L"$build/bin" \
  -Wl,-rpath,"$build/bin" -lllama -o "$ARTIFACTS/compare"
sha256sum "$HARNESS"/llamacpp-* "$ARTIFACTS/compare" > "$OUT/source.sha256"
"$PYTHON" "$HARNESS/llamacpp-prepare.py" "$HF_MODEL" "$BASELINE" "$C_PROBE" "$C_TOKENIZER" "$OUT/prompts"
model="$ARTIFACTS/$MODEL_LABEL-f32.gguf"
stage=$(mktemp -d "${SLURM_TMPDIR:-/tmp}/llamacpp-compare-${SLURM_JOB_ID}-XXXXXX")
if [[ ! -s $model ]]; then
  # Lazy tensor conversion does many small reads; stage the source snapshot
  # first to avoid repeatedly accessing the shared filesystem.
  mkdir "$stage/hf"
  for input in "$HF_MODEL"/*.json "$HF_MODEL"/*.safetensors "$HF_MODEL"/*.model; do
    if [[ -f $input ]]; then cp -L "$input" "$stage/hf/"; fi
  done
  "$PYTHON" "$SOURCE/convert_hf_to_gguf.py" "$stage/hf" --outtype f32 --outfile "$stage/model.gguf"
  cp "$stage/model.gguf" "$model.tmp"
  mv "$model.tmp" "$model"
else
  /usr/bin/time -p -o "$OUT/staging-time.txt" cp "$model" "$stage/model.gguf"
fi
sha256sum "$model" > "$OUT/model.sha256"
sha256sum "$stage/model.gguf" > "$OUT/staged.sha256"
[[ $(cut -d' ' -f1 "$OUT/model.sha256") == $(cut -d' ' -f1 "$OUT/staged.sha256") ]]
for batch in 512 1; do
  for workload in short long; do
    # Separate warm-up process; all measured processes start with empty KV state.
    for run in warmup 1 2 3; do
      srun --cpus-per-task="$THREADS" --cpu-bind=cores /usr/bin/time -v -o "$OUT/b${batch}-${workload}-${run}.time" \
        "$ARTIFACTS/compare" "$stage/model.gguf" "$OUT/prompts/$workload.txt" \
        "$OUT/prompts/$workload.ids" "$THREADS" "$batch" 128 \
        > "$OUT/b${batch}-${workload}-${run}.json" 2> "$OUT/b${batch}-${workload}-${run}.stderr"
      "$PYTHON" - "$OUT/b${batch}-${workload}-${run}.json" <<'PY'
import json, sys
r = json.load(open(sys.argv[1]))
assert r['output'] == 128 and r['decode_steps'] == 127 and not r['stopped_eog'], r
assert len(r['affinity_cpus']) == r['threads'], r
print(sys.argv[1], r['prefill_tok_s'], r['decode_tok_s'], r['peak_rss_bytes'])
PY
    done
  done
done
