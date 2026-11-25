#!/usr/bin/env bash
#SBATCH --job-name=xengine70
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=112
#SBATCH --exclusive
#SBATCH --mem=0
#SBATCH --time=01:20:00

# 70B cross-engine comparison (docs/xengine-70b.md). Usage: ROOT OUT SNAPSHOT CPP VLLM_IMAGE PROMPTS MODE
# MODE: c-fp32, c-q8, c-q8-hbm, cpp-fp32, cpp-q8, cpp-q8-hbm, vllm-fp32. PROMPTS holds short.ids.txt and long.ids.txt (the
# whitespace-delimited form of tools/xengine70/prompts/*.ids.json); the checkpoints are in ../models next to it.
# The C engine's own defaults apply to everything this comparison does not fix: it sets the socket and thread counts, the
# prefill batch (the driver checks it equals the request batch), the placement and the page verification. Optional environment:
# TOPOLOGY (LLAMA_TOPOLOGY; unset: discovered), KERNELS, ATTN, ATTN_DECODE, KV_LAYOUT, THP, Q8_ACTIVATIONS (each becomes the
# LLAMA_* knob of the same name), and LEGACY=1 for the configuration of the earlier 70B comparison work (KERNELS=avx512
# ATTN=tiled ATTN_DECODE=group KV_LAYOUT=head THP=off Q8_ACTIVATIONS=int8; any option set explicitly wins over it). The
# Q8 rows of docs/xengine-70b.md were measured before the gs32 AMX and VNNI kernels existed; re-run the c-q8 modes to refresh.
set -euo pipefail
root=$(realpath "$1")
out=$2
snapshot=$(realpath "$3")
cpp=$(realpath "$4")
image=$(realpath "$5")
prompts=$(realpath "$6")
mode=$7
mkdir "$out"
export LC_ALL=C
{
  date -Is
  hostname
  numactl -H
  lscpu
  grep -E '^(Cpus|Mems)_allowed_list' /proc/self/status
  printf 'job=%s\nmode=%s\n' "${SLURM_JOB_ID:?}" "$mode"
  sha256sum "$prompts"/*.ids.json
} > "$out/environment.txt"

context=8192
batch=256
generation=128
case "$mode" in
  vllm-fp32)
    port=18027
    cache=$(mktemp -d)
    mkdir -p "$cache"/{home,triton,xdg}
    cleanup() {
      if [[ -n ${server:-} ]]; then
        kill -TERM -- "-$server" 2>/dev/null || true
        for ((attempt = 0; attempt < 15; attempt++)); do
          kill -0 -- "-$server" 2>/dev/null || break
          sleep 1
        done
        kill -KILL -- "-$server" 2>/dev/null || true
        if ! kill -0 "$server" 2>/dev/null; then
          wait "$server" || true
        fi
      fi
    }
    trap cleanup EXIT
    server_cmd=(apptainer exec --cleanenv --no-home --bind "$snapshot:/model:ro,$cache:/cache"
      --env 'VLLM_CPU_OMP_THREADS_BIND=0-54|56-110' --env VLLM_CACHE_ROOT=/cache/vllm
      --env TORCHINDUCTOR_CACHE_DIR=/cache/inductor --env HF_HOME=/cache/hf --env HF_HUB_OFFLINE=1
      --env TOKENIZERS_PARALLELISM=false --env HOME=/cache/home --env XDG_CACHE_HOME=/cache/xdg
      --env TRITON_CACHE_DIR=/cache/triton --env CXX=g++
      "$image" vllm serve /model --served-model-name matched70 --dtype float32 --tensor-parallel-size 2
      --max-model-len "$context" --max-num-seqs 1 --max-num-batched-tokens "$batch" --enable-chunked-prefill
      --no-enable-prefix-caching --seed 1 --generation-config vllm --kv-cache-memory-bytes 4G
      --host 127.0.0.1 --port "$port")
    printf '%q ' "${server_cmd[@]}" > "$out/command.sh"
    printf '\n' >> "$out/command.sh"
    sha256sum "$image" >> "$out/environment.txt"
    setsid "${server_cmd[@]}" > "$out/server.log" 2>&1 &
    server=$!
    deadline=$((SECONDS+1800))
    until curl -fsS "http://127.0.0.1:$port/health" > /dev/null 2>&1; do
      kill -0 "$server" 2>/dev/null && (( SECONDS < deadline )) || {
        tail -35 "$out/server.log" >&2
        exit 1
      }
      sleep 2
    done
    for workload in short long; do
      taskset -c 55 python3 "$root/tools/xengine_vllm_client.py" --host 127.0.0.1 --port "$port" --model matched70 \
        --tokenizer "$snapshot" --ids-file "$prompts/$workload.ids.json" --gen "$generation" \
        --warmups 1 --repeats 3 --timeout 1500 --out "$out/$workload.jsonl" > "$out/$workload.client.log"
      python3 - "$out/$workload.jsonl" <<'PY'
import json
import sys
from pathlib import Path
rows = [json.loads(line) for line in Path(sys.argv[1]).read_text().splitlines()]
assert len(rows) == 4
for row in rows:
    assert row["ok"] and len(row["token_ids"]) == len(row["token_arrivals"]) == row["gen_requested"], row
PY
    done
    curl -fsS "http://127.0.0.1:$port/metrics" > "$out/metrics.txt"
    ps -o pid,ppid,pgid,comm -g "$server" > "$out/processes.txt"
    while read -r pid _; do
      [[ $pid =~ ^[0-9]+$ ]] || continue
      numastat -p "$pid" > "$out/numastat-$pid.txt" 2>&1 || true
      grep -E '^(Name|Cpus_allowed_list|Mems_allowed_list)' "/proc/$pid/status" > "$out/status-$pid.txt" || true
    done < "$out/processes.txt"
    ;;
  c-fp32|c-q8|c-q8-hbm|cpp-fp32|cpp-q8|cpp-q8-hbm)
    models="$(dirname "$prompts")/models"
    ids=("$prompts/short.ids.txt" "$prompts/long.ids.txt" "$context" "$batch")
    placement=ddr
    nodes=0,1
    if [[ $mode == *-hbm ]]; then
      placement=hbm-only
      nodes=2,3
    fi
    if [[ $mode == c-* ]]; then
      build="$out/bin"
      precision=fp32
      model="$models/fp32-v3.bin"
      if [[ $mode == c-q8* ]]; then
        precision=q8
        model="$models/q8-gguf-v3.bin"
      fi
      make -C "$root" -j 8 CC=gcc BUILD_DIR="$build" "$build/xengine70-$precision" > "$out/build.log" 2>&1
      unset LLAMA_TOPOLOGY LLAMA_KERNELS LLAMA_PRECISION LLAMA_Q8_ACTIVATIONS LLAMA_KV_LAYOUT LLAMA_ATTN LLAMA_ATTN_DECODE LLAMA_THP
      if [[ ${LEGACY:-0} == 1 ]]; then
        : "${KERNELS=avx512}" "${ATTN=tiled}" "${ATTN_DECODE=group}" "${KV_LAYOUT=head}" "${THP=off}" "${Q8_ACTIVATIONS=int8}"
      fi
      [[ -z ${TOPOLOGY:-} ]] || export LLAMA_TOPOLOGY=$TOPOLOGY
      [[ -z ${KERNELS:-} ]] || export LLAMA_KERNELS=$KERNELS
      [[ -z ${ATTN:-} ]] || export LLAMA_ATTN=$ATTN
      [[ -z ${ATTN_DECODE:-} ]] || export LLAMA_ATTN_DECODE=$ATTN_DECODE
      [[ -z ${KV_LAYOUT:-} ]] || export LLAMA_KV_LAYOUT=$KV_LAYOUT
      [[ -z ${THP:-} ]] || export LLAMA_THP=$THP
      [[ -z ${Q8_ACTIVATIONS:-} ]] || export LLAMA_Q8_ACTIVATIONS=$Q8_ACTIVATIONS
      export LLAMA_SOCKETS=2 LLAMA_THREADS=55 LLAMA_PREFILL_BATCH="$batch"
      export LLAMA_PLACEMENT="$placement" LLAMA_PLACEMENT_STRICT=1 LLAMA_VERIFY_SAMPLES=16384
      env | grep '^LLAMA_' | sort >> "$out/environment.txt"
      command=(taskset -c 0-54,56-110 "$build/xengine70-$precision" "$model" "${ids[@]}")
    else
      source /opt/rh/gcc-toolset-14/enable
      lib="$cpp/build-g14/bin"
      g++ -O3 -std=c++17 -I"$cpp/include" -I"$cpp/ggml/include" "$root/tools/xengine70/native_cpp.cpp" \
        -Wl,-rpath,"$lib" "$lib/libllama.so.1.0.0" "$lib/libggml.so.1.0.0" \
        "$lib/libggml-base.so.1.0.0" "$lib/libggml-cpu.so.1.0.0" -o "$out/native-cpp" > "$out/build.log" 2>&1
      model="$models/f32.gguf"
      [[ $mode != cpp-q8* ]] || model="$models/q8_0.gguf"
      command=(taskset -c 0-54,56-110 numactl --interleave="$nodes" "$out/native-cpp" "$model" "${ids[@]}" 110)
      git -C "$cpp" rev-parse HEAD >> "$out/environment.txt"
    fi
    test -s "$model"
    stat -c '%n %s' "$model" >> "$out/environment.txt"
    printf '%q ' "${command[@]}" > "$out/command.sh"
    printf '\n' >> "$out/command.sh"
    "${command[@]}" > "$out/runs.jsonl" 2> "$out/engine.log" &
    pid=$!
    until [[ -s "$out/runs.jsonl" ]] || ! kill -0 "$pid" 2>/dev/null; do sleep 2; done
    numastat -p "$pid" > "$out/numastat.txt" 2>&1 || true
    wait "$pid"
    ;;
  *) echo "Unknown benchmark mode: $mode" >&2; exit 2 ;;
esac
