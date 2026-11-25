#!/usr/bin/env bash
#SBATCH --job-name=llama-tiers
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=112
#SBATCH --exclusive
#SBATCH --mem=0
#SBATCH --time=00:50:00

# Flat/Quadrant placement pilot. Submit to one node with both local tiers allowed.
# Required: ROOT, OUT (new directory), CHECKPOINT, TOKENIZER.
# Optional: ENGINE=runq, SOCKETS=2, REPEATS=3, CONTEXT=4096, WARM=512, GENERATION=128,
# WEIGHTS_HBM_LAYERS=16, KV_HBM_LAYERS=16 (set to match this checkpoint), VERIFY_SAMPLES=0, PROMPT.
# The engine's own defaults apply unless an option below is set (the kernels, the prefill batch, attention, the K/V
# layout and the page size are not forced): KERNELS, PREFILL_BATCH, PREFILL_LOGITS, KV_LAYOUT, ATTN, ATTN_DECODE, THP
# (each becomes the LLAMA_* knob of the same name). LEGACY=1 restores the values this script used before the AMX gs32 work
# and the M=1024 batch (KERNELS=avx512 PREFILL_BATCH=64 PREFILL_LOGITS=last KV_LAYOUT=head ATTN=tiled ATTN_DECODE=split
# THP=off); any option set explicitly wins over it. DRY_RUN=1 prints the LLAMA_* environment the runs would see and exits.
set -euo pipefail

: "${ROOT:?}" "${OUT:?}" "${CHECKPOINT:?}" "${TOKENIZER:?}"
engine=${ENGINE:-runq}
sockets=${SOCKETS:-2}
repeats=${REPEATS:-3}
context=${CONTEXT:-4096}
warm=${WARM:-512}
generation=${GENERATION:-128}
weights_layers=${WEIGHTS_HBM_LAYERS:-16}
kv_layers=${KV_HBM_LAYERS:-16}
samples=${VERIFY_SAMPLES:-0}
threads=${THREADS:-56}
prompt=(--prompt "${PROMPT:-Print a long numbered list of prime numbers. Keep listing primes for at least 1000 tokens. Start at 2.}")
PYTHON=${PYTHON:-python3}

export LLAMA_SOCKETS="$sockets" LLAMA_THREADS="$threads" LLAMA_PLACEMENT_STRICT=1
export LLAMA_VERIFY_SAMPLES="$samples" LLAMA_KV_VERIFY=1
unset LLAMA_KERNELS LLAMA_PREFILL_BATCH LLAMA_PREFILL_LOGITS LLAMA_KV_LAYOUT LLAMA_ATTN LLAMA_ATTN_DECODE LLAMA_THP
unset LLAMA_HYBRID_WEIGHTS_HBM_LAYERS LLAMA_HYBRID_KV_HBM_LAYERS LLAMA_HYBRID_CLASSIFIER LLAMA_HYBRID_PACKED
if [[ ${LEGACY:-0} == 1 ]]; then
  : "${KERNELS=avx512}" "${PREFILL_BATCH=64}" "${PREFILL_LOGITS=last}" "${KV_LAYOUT=head}" "${ATTN=tiled}" \
    "${ATTN_DECODE=split}" "${THP=off}"
fi
# An option left unset leaves the engine default in force.
[[ -z ${KERNELS:-} ]] || export LLAMA_KERNELS=$KERNELS
[[ -z ${PREFILL_BATCH:-} ]] || export LLAMA_PREFILL_BATCH=$PREFILL_BATCH
[[ -z ${PREFILL_LOGITS:-} ]] || export LLAMA_PREFILL_LOGITS=$PREFILL_LOGITS
[[ -z ${KV_LAYOUT:-} ]] || export LLAMA_KV_LAYOUT=$KV_LAYOUT
[[ -z ${ATTN:-} ]] || export LLAMA_ATTN=$ATTN
[[ -z ${ATTN_DECODE:-} ]] || export LLAMA_ATTN_DECODE=$ATTN_DECODE
[[ -z ${THP:-} ]] || export LLAMA_THP=$THP
if [[ ${DRY_RUN:-0} == 1 ]]; then
  env | grep '^LLAMA_' | sort
  exit 0
fi
mkdir "$OUT"

{
  hostname
  date -u
  uname -a
  gcc --version
  numactl -H
  grep -E '^(Cpus|Mems)_allowed_list' /proc/self/status
  printf 'engine=%s sockets=%s threads=%s context=%s warm=%s generation=%s weights_hbm_layers=%s kv_hbm_layers=%s\n' \
    "$engine" "$sockets" "$threads" "$context" "$warm" "$generation" "$weights_layers" "$kv_layers"
  printf 'checkpoint=%s\ntokenizer=%s\n' "$CHECKPOINT" "$TOKENIZER"
  env | grep '^LLAMA_' | sort
} > "$OUT/environment.txt"

make -C "$ROOT" -j "$threads" CC=gcc BUILD_DIR="$OUT/bin" > "$OUT/build.log" 2>&1
binary="$OUT/bin/$engine"

# Raw event definitions match the existing Flat-node calibration. HBM CAS is 32 bytes, DDR CAS is 64.
events='uncore_imc/event=0x05,umask=0xcf,name=ddr_cas_rd/,uncore_type_14/event=0x05,umask=0xcf,name=hbm_cas_rd/,unc_upi_rxl_flits.all_data,unc_upi_txl_flits.all_data'
perf=0
if sudo -n /usr/bin/perf stat -a -x, -e "$events" -- true 2> "$OUT/perf-probe.csv"; then
  perf=1
  sudo -n /usr/bin/perf stat -a --per-socket -I 200 -x, -e "$events" -- sleep 3 2> "$OUT/idle.perf.csv" || true
fi
printf 'perf_uncore=%s\n' "$perf" >> "$OUT/environment.txt"

modes=(hbm-only ddr hybrid)
baseline=
for ((rep=0; rep<repeats; rep++)); do
  for ((i=0; i<3; i++)); do
    mode=${modes[(rep+i)%3]}
    prefix="$OUT/$mode-$((rep+1))"
    knobs=(LLAMA_PLACEMENT="$mode")
    if [[ $mode == hybrid ]]; then
      knobs+=(LLAMA_HYBRID_WEIGHTS_HBM_LAYERS="$weights_layers" LLAMA_HYBRID_KV_HBM_LAYERS="$kv_layers")
    fi
    if [[ $rep == 0 ]]; then
      env "${knobs[@]}" "$binary" "$CHECKPOINT" --context-length "$context" --plan-only --json "$prefix.plan.json"
    fi
    date +%s%N > "$prefix.launch"
    env "${knobs[@]}" LLAMA_TOKENS_OUT="$prefix.tokens.json" \
      "$binary" "$CHECKPOINT" -z "$TOKENIZER" --benchmark --chat-template --no-text -t 0 -n 0 \
      --context-length "$context" --warm-context "$warm" --generation-length "$generation" "${prompt[@]}" --json "$prefix.json" \
      > "$prefix.stdout" 2> "$prefix.stderr" &
    pid=$!
    perfpid=
    if [[ $perf == 1 && $rep == 0 ]]; then
      date +%s%N > "$prefix.perf-start"
      sudo -n /usr/bin/perf stat -a --per-socket -I 200 -x, -e "$events" \
        -- tail -s 0.1 --pid="$pid" -f /dev/null > /dev/null 2> "$prefix.perf.csv" &
      perfpid=$!
    fi
    status=0
    wait "$pid" || status=$?
    date +%s%N > "$prefix.end"
    [[ -z $perfpid ]] || wait "$perfpid" || true
    if [[ $status != 0 ]]; then
      cat "$prefix.stderr" >&2
      exit "$status"
    fi
    verify=(--planned)
    [[ $samples == 0 ]] || verify+=(--allow-sampled)
    "$PYTHON" "$ROOT/tools/check_placement.py" "$prefix.json" "${verify[@]}" > "$prefix.placement"
    if [[ -z $baseline ]]; then baseline="$prefix.tokens.json"; fi
    "$PYTHON" - "$baseline" "$prefix.tokens.json" <<'PY'
import json
import sys
from pathlib import Path
left, right = [json.loads(Path(p).read_text()) for p in sys.argv[1:]]
if left != right:
    raise SystemExit("token traces differ across otherwise matched placements")
PY
  done
done
