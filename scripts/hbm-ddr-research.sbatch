#!/usr/bin/env bash
#SBATCH --job-name=llama-tier-research
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=112
#SBATCH --exclusive
#SBATCH --mem=0

# Research campaigns of the 70B HBM and DDR study (docs/hbm-ddr-research.md). Submit on a dual-socket Flat/Quadrant node,
# one campaign per job; each job writes a new run directory. Required: ROOT (staged source tree), OUT (new directory),
# MODEL_DIR (holds int8-v3.bin, tokenizer-v2.bin and a non-empty q8-ready.txt), CAMPAIGN. Optional: COMMIT (recorded in the
# environment file; default: ROOT/COMMIT or the HEAD of ROOT), SOCKETS=2, THREADS=56, HBM_BUDGET_GIB=56 (the study's
# reservation, 56 GiB per socket), TOPOLOGY (LLAMA_TOPOLOGY; unset: discovered), DRY_RUN=1 (print the LLAMA_* environment the
# runs would see and exit).
# The engine's own defaults apply unless an option below is set (the kernels, the prefill batch, attention, the K/V layout and
# the page size are not forced): KERNELS, PREFILL_BATCH, PREFILL_LOGITS, KV_LAYOUT, ATTN, ATTN_DECODE, THP, Q8_ACTIVATIONS
# (each becomes the LLAMA_* knob of the same name). LEGACY=1 restores the configuration of the earlier 70B comparison work
# (KERNELS=avx512 PREFILL_BATCH=128 PREFILL_LOGITS=last KV_LAYOUT=head ATTN=tiled ATTN_DECODE=split THP=off
# Q8_ACTIVATIONS=int8); any option set explicitly wins over it. The grouped-attention, profile and AMX cells name their own
# knobs below because those knobs are what the cell varies.
# CAMPAIGN: baseline, packs, budget, diagnostics, profile, profile-fixed, tier-window, edge-auto, edge-exchange, edge-ddr,
#   edge-group-real, stress, stress-group, stress-max, prime-check
# The stress, stress-group, stress-max and prime-check campaigns build with SYNTHETIC_KV=1 and fill the warm context with
# LLAMA_SYNTHETIC_KV=1 (primed by LLAMA_SYNTHETIC_KV_PRIME real positions). The packs, edge-* and stress cells that the study
# ran with AMX-INT8 prefill check the report: they fail unless the engine selected the AMX kernels (set KERNELS=amx to force).
# diagnostics: the serial/concurrent local-pair control of job 5904726 needed a stream-bench option that this tree does not
# have (see docs/hbm-ddr-research.md); only the attention-bench half is run.
set -euo pipefail

: "${ROOT:?}" "${OUT:?}" "${MODEL_DIR:?}" "${CAMPAIGN:?}"
root=$(realpath "$ROOT")
commit=${COMMIT:-$(cat "$root/COMMIT" 2> /dev/null || git -C "$root" rev-parse HEAD)}
out=$OUT
model=$(realpath "$MODEL_DIR")
campaign=$CAMPAIGN
checkpoint="$model/int8-v3.bin"
tokenizer="$model/tokenizer-v2.bin"
sockets=${SOCKETS:-2}
threads=${THREADS:-56}
budget=${HBM_BUDGET_GIB:-56}

export LC_ALL=C
export LLAMA_SOCKETS="$sockets" LLAMA_THREADS="$threads" LLAMA_PLACEMENT_STRICT=1
export LLAMA_KV_VERIFY=1 LLAMA_VERIFY_SAMPLES=16384 LLAMA_HBM_BUDGET_GIB="$budget" LLAMA_PROFILE=0
unset LLAMA_TOPOLOGY LLAMA_KERNELS LLAMA_PREFILL_BATCH LLAMA_PREFILL_LOGITS LLAMA_KV_LAYOUT LLAMA_ATTN LLAMA_ATTN_DECODE \
  LLAMA_THP LLAMA_Q8_ACTIVATIONS
unset LLAMA_DDR_BUDGET_GIB LLAMA_TOPOLOGY_ALLOW_FAR_HBM LLAMA_SYNTHETIC_KV LLAMA_SYNTHETIC_KV_PRIME
if [[ ${LEGACY:-0} == 1 ]]; then
  : "${KERNELS=avx512}" "${PREFILL_BATCH=128}" "${PREFILL_LOGITS=last}" "${KV_LAYOUT=head}" "${ATTN=tiled}" \
    "${ATTN_DECODE=split}" "${THP=off}" "${Q8_ACTIVATIONS=int8}"
fi
# An option left unset leaves the engine default in force.
[[ -z ${TOPOLOGY:-} ]] || export LLAMA_TOPOLOGY=$TOPOLOGY
[[ -z ${KERNELS:-} ]] || export LLAMA_KERNELS=$KERNELS
[[ -z ${PREFILL_BATCH:-} ]] || export LLAMA_PREFILL_BATCH=$PREFILL_BATCH
[[ -z ${PREFILL_LOGITS:-} ]] || export LLAMA_PREFILL_LOGITS=$PREFILL_LOGITS
[[ -z ${KV_LAYOUT:-} ]] || export LLAMA_KV_LAYOUT=$KV_LAYOUT
[[ -z ${ATTN:-} ]] || export LLAMA_ATTN=$ATTN
[[ -z ${ATTN_DECODE:-} ]] || export LLAMA_ATTN_DECODE=$ATTN_DECODE
[[ -z ${THP:-} ]] || export LLAMA_THP=$THP
[[ -z ${Q8_ACTIVATIONS:-} ]] || export LLAMA_Q8_ACTIVATIONS=$Q8_ACTIVATIONS
if [[ ${DRY_RUN:-0} == 1 ]]; then
  env | grep '^LLAMA_' | sort
  exit 0
fi
test -r "$checkpoint" && test -r "$tokenizer" && test -s "$model/q8-ready.txt"
mkdir "$out"
build="$out/bin"
build_args=()
run_env=()
if [[ $campaign == stress || $campaign == stress-group || $campaign == stress-max || $campaign == prime-check ]]; then
  build_args=(SYNTHETIC_KV=1)
  run_env=(LLAMA_SYNTHETIC_KV=1 LLAMA_SYNTHETIC_KV_PRIME=1024)
fi
unset LLAMA_HYBRID_WEIGHTS_HBM_LAYERS LLAMA_HYBRID_KV_HBM_LAYERS LLAMA_HYBRID_CLASSIFIER LLAMA_HYBRID_PACKED

{
  printf 'job=%s\ncommit=%s\ncampaign=%s\n' "${SLURM_JOB_ID:?}" "$commit" "$campaign"
  date -Is
  hostname
  uname -a
  numactl -H
  grep -E '^(Cpus|Mems)_allowed_list' /proc/self/status
  lscpu
  cat /proc/sys/kernel/numa_balancing || true
  perf --version
  gcc -dumpfullversion
  stat -c '%n %s' "$checkpoint" "$tokenizer"
  cat "$model/q8-ready.txt"
  env | sort | grep '^LLAMA_'
} > "$out/environment.txt"

make -C "$root" -j 8 "${build_args[@]}" CC=gcc BUILD_DIR="$build" "$build/runq" "$build/stream-bench" "$build/attention-bench" \
  > "$out/build.log" 2>&1
binary="$build/runq"
sha256sum "$binary" "$0" > "$out/executables.sha256"

# Intel PCM uses 32 bytes per HBM CAS on Sapphire Rapids and 64 bytes per DDR CAS.
reads='uncore_imc/event=0x05,umask=0xcf,name=ddr_cas_rd/,uncore_type_14/event=0x05,umask=0xcf,name=hbm_cas_rd/'
writes='uncore_imc/event=0x05,umask=0xf0,name=ddr_cas_wr/,uncore_type_14/event=0x05,umask=0xf0,name=hbm_cas_wr/'
events="$reads,$writes,unc_upi_rxl_flits.all_data,unc_upi_txl_flits.all_data,power/energy-pkg/,power/energy-ram/,cycles,ref-cycles"
interval=200
case "$campaign" in
  profile|profile-fixed) events="$events,context-switches,cpu-migrations" ;;
  tier-window)
    events="$reads"
    interval=10
    ;;
esac
perf=(sudo -n /usr/bin/perf stat -a --per-socket -I "$interval" -x, -e "$events")
printf 'interval_ms=%s\nevents=%s\n' "$interval" "$events" >> "$out/environment.txt"
"${perf[@]}" -- sleep 5 2> "$out/idle.perf.csv"

short_prompt='Print a long numbered list of prime numbers. Keep listing primes for at least 2000 tokens. Start at 2.'
prompt=(--prompt "$short_prompt")
expect_amx=0 # 1 in the cells of the study that ran AMX-INT8 prefill: run_cell then requires build.kernels == amx
printf 'label\tbackend\tattention\tmode\tcapacity\twarm\tgeneration\tweight_hbm_layers\tkv_hbm_layers\tprofile\n' > "$out/cells.tsv"

run_cell() (
  label=$1 mode=$2 context=$3 warm=$4 generation=$5
  export LLAMA_PLACEMENT="$mode"
  if [[ $mode == hybrid ]]; then
    export LLAMA_HYBRID_WEIGHTS_HBM_LAYERS=${6:-auto} LLAMA_HYBRID_KV_HBM_LAYERS=${7:-auto}
  fi
  prefix="$out/$label"
  export LLAMA_TOKENS_OUT="$prefix.tokens.json"
  printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
    "$label" "${LLAMA_KERNELS:-default}" "${LLAMA_ATTN_DECODE:-default}" "$mode" "$context" "$warm" "$generation" \
    "${LLAMA_HYBRID_WEIGHTS_HBM_LAYERS:--}" "${LLAMA_HYBRID_KV_HBM_LAYERS:--}" "$LLAMA_PROFILE" >> "$out/cells.tsv"
  printf '%s %s\n' "$(date -Is)" "$label"
  "$binary" "$checkpoint" --context-length "$context" --plan-only --json "$prefix.plan.json" \
    > "$prefix.plan.stdout" 2> "$prefix.plan.stderr"

  date +%s%N > "$prefix.launch"
  env "${run_env[@]}" "$binary" "$checkpoint" -z "$tokenizer" --benchmark --chat-template --no-text -t 0 -n 0 \
    --context-length "$context" --warm-context "$warm" --generation-length "$generation" \
    "${prompt[@]}" --json "$prefix.json" > "$prefix.stdout" 2> "$prefix.stderr" &
  pid=$!
  date +%s%N > "$prefix.perf-start"
  "${perf[@]}" -- tail -s 0.05 --pid="$pid" -f /dev/null > /dev/null 2> "$prefix.perf.csv" &
  perfpid=$!
  sampler_pid=
  if [[ $campaign == prime-check ]]; then
    python3 "$root/tools/hbm_worker_samples.py" "$pid" > "$prefix.worker-samples.jsonl" 2> "$prefix.worker-samples.stderr" &
    sampler_pid=$!
  fi
  status=0
  wait "$pid" || status=$?
  date +%s%N > "$prefix.end"
  wait "$perfpid"
  date +%s%N > "$prefix.perf-end"
  [[ -z $sampler_pid ]] || wait "$sampler_pid" || true
  if [[ $status != 0 ]]; then
    cat "$prefix.stderr" >&2
    exit "$status"
  fi
  python3 "$root/tools/check_placement.py" "$prefix.json" --planned --allow-sampled > "$prefix.placement"
  python3 - "$prefix.json" "$generation" "${#run_env[@]}" "${expect_amx:-0}" <<'PY'
import json
import sys
with open(sys.argv[1]) as stream:
    report = json.load(stream)
expected = int(sys.argv[2]) - 1
synthetic = bool(int(sys.argv[3]))
if bool(report["parameters"].get("synthetic_kv")) != synthetic:
    raise SystemExit(f"{sys.argv[1]}: unexpected KV population method")
if int(sys.argv[4]) and report["build"]["kernels"] != "amx":
    raise SystemExit(f"{sys.argv[1]}: this cell needs the AMX kernels, the engine selected {report['build']['kernels']}")
if report["stop_reason"] != "generation_limit" or report["tokens"]["decode_steps"] != expected:
    raise SystemExit(f"{sys.argv[1]}: expected {expected} timed decode steps")
PY
)

case "$campaign" in
  baseline)
    for tier in hbm ddr; do
      "$build/stream-bench" --sockets 2 --threads 56 --mib 4096 --reps 7 --tier "$tier" --only raw-chunk \
        > "$out/stream-$tier.jsonl" 2> "$out/stream-$tier.stderr" &
      stream_pid=$!
      date +%s%N > "$out/stream-$tier.perf-start"
      "${perf[@]}" -- tail -s 0.05 --pid="$stream_pid" -f /dev/null 2> "$out/stream-$tier.perf.csv" &
      stream_perf_pid=$!
      wait "$stream_pid"
      wait "$stream_perf_pid"
    done
    for rep in 1 2 3; do
      order=(hbm-only hybrid ddr)
      case $rep in
        2) order=(ddr hbm-only hybrid) ;;
        3) order=(hybrid ddr hbm-only) ;;
      esac
      for mode in "${order[@]}"; do
        run_cell "$mode-$rep" "$mode" 8192 512 128 40 40
        cmp "$out/hbm-only-1.tokens.json" "$out/$mode-$rep.tokens.json"
      done
    done
    run_cell weight-spill hybrid 8192 512 128 40 80
    run_cell kv-spill hybrid 8192 512 128 80 40
    for mode in weight-spill kv-spill; do
      cmp "$out/hbm-only-1.tokens.json" "$out/$mode.tokens.json"
    done
    ;;
  packs)
    expect_amx=1
    python3 - "$out/prefill-prompt.txt" <<'PY'
import sys
from pathlib import Path
body = "\n".join(f"Entry {i}: The key measuring instrument is the memory controller." for i in range(1, 161))
instructions = "\nNow print a long numbered list of prime numbers, starting at 2, for at least 2000 tokens."
Path(sys.argv[1]).write_text(body + instructions)
PY
    prompt=(--prompt-file "$out/prefill-prompt.txt")
    for rep in 1 2 3; do
      order=(hybrid ddr)
      [[ $rep != 2 ]] || order=(ddr hybrid)
      for mode in "${order[@]}"; do
        run_cell "$mode-$rep" "$mode" 8192 0 128
        cmp "$out/hybrid-1.tokens.json" "$out/$mode-$rep.tokens.json"
      done
    done
    ;;
  budget)
    run_cell auto-short hybrid 98304 8192 96
    run_cell exchange-short hybrid 98304 8192 96 60 79
    run_cell hbm-short hbm-only 16384 8192 96
    cmp "$out/auto-short.tokens.json" "$out/exchange-short.tokens.json"
    cmp "$out/auto-short.tokens.json" "$out/hbm-short.tokens.json"
    ;;
  diagnostics)
    for kernel in split group; do
      for node in 2 0; do
        LLAMA_SOCKETS=1 LLAMA_ATTN_DECODE="$kernel" numactl --cpunodebind=0 \
          "$build/attention-bench" --mode decode --shape 64/8/128 --kv-heads-local 4 --kv-layout head \
          --node "$node" --context 8192,32768,73728,98304 --cold-mib 1536 --reps 63 \
          > "$out/attention-$kernel-node$node.jsonl" 2> "$out/attention-$kernel-node$node.stderr"
      done
    done
    ;;
  profile)
    export LLAMA_PROFILE=1
    for rep in 1 2 3; do
      run_cell "hybrid-$rep" hybrid 8192 512 128 40 40
      cmp "$out/hybrid-1.tokens.json" "$out/hybrid-$rep.tokens.json"
    done
    ;;
  profile-fixed)
    export LLAMA_PROFILE=1
    run_cell hybrid-fixed hybrid 8192 512 128 40 40
    ;;
  tier-window)
    run_cell hybrid-10ms hybrid 8192 512 128 40 40
    ;;
  prime-check)
    run_env=(LLAMA_SYNTHETIC_KV=1)
    run_cell unprimed hybrid 98304 8192 96
    run_env+=(LLAMA_SYNTHETIC_KV_PRIME=1024)
    run_cell primed hybrid 98304 8192 96
    cmp "$out/unprimed.tokens.json" "$out/primed.tokens.json"
    ;;
  stress)
    run_cell validate-short hybrid 98304 8192 96
    run_cell near-hbm hbm-only 65536 64512 64
    run_cell edge-auto hybrid 98304 73728 64
    run_cell edge-exchange hybrid 98304 73728 64 60 79
    run_cell edge-ddr ddr 98304 73728 64
    cmp "$out/edge-auto.tokens.json" "$out/edge-exchange.tokens.json"
    cmp "$out/edge-auto.tokens.json" "$out/edge-ddr.tokens.json"
    run_cell edge-exchange-2 hybrid 98304 73728 64 60 79
    run_cell edge-auto-2 hybrid 98304 73728 64
    cmp "$out/edge-auto.tokens.json" "$out/edge-exchange-2.tokens.json"
    cmp "$out/edge-auto.tokens.json" "$out/edge-auto-2.tokens.json"
    run_cell max-auto hybrid 98304 98176 64
    run_cell max-exchange hybrid 98304 98176 64 60 79
    cmp "$out/max-auto.tokens.json" "$out/max-exchange.tokens.json"
    run_cell max-exchange-2 hybrid 98304 98176 64 60 79
    run_cell max-auto-2 hybrid 98304 98176 64
    cmp "$out/max-auto.tokens.json" "$out/max-exchange-2.tokens.json"
    cmp "$out/max-auto.tokens.json" "$out/max-auto-2.tokens.json"
    expect_amx=1
    run_env=(LLAMA_SYNTHETIC_KV=1 LLAMA_SYNTHETIC_KV_PRIME=4096)
    run_cell validate-edge-amx hybrid 98304 73728 64
    ;;
  stress-group)
    export LLAMA_ATTN_DECODE=group
    run_cell edge-auto-group hybrid 98304 73728 64
    run_cell edge-exchange-group hybrid 98304 73728 64 60 79
    cmp "$out/edge-auto-group.tokens.json" "$out/edge-exchange-group.tokens.json"
    run_cell edge-exchange-group-2 hybrid 98304 73728 64 60 79
    run_cell edge-auto-group-2 hybrid 98304 73728 64
    cmp "$out/edge-auto-group.tokens.json" "$out/edge-exchange-group-2.tokens.json"
    cmp "$out/edge-auto-group.tokens.json" "$out/edge-auto-group-2.tokens.json"
    run_cell max-auto-group hybrid 98304 98176 64
    run_env=()
    run_cell short-real-group hybrid 98304 8192 96
    ;;
  stress-max)
    run_cell limit-auto hybrid 131072 130944 64
    export LLAMA_ATTN_DECODE=group
    run_cell limit-auto-group hybrid 131072 130944 64
    run_cell limit-exchange-group hybrid 131072 130944 64 60 59
    run_cell limit-ddr-group ddr 131072 130944 64
    cmp "$out/limit-auto-group.tokens.json" "$out/limit-exchange-group.tokens.json"
    cmp "$out/limit-auto-group.tokens.json" "$out/limit-ddr-group.tokens.json"
    ;;
  edge-auto|edge-exchange|edge-ddr|edge-group-real)
    # AMX packs stay in DDR in every cell. The long warm-up forwards real tokens.
    expect_amx=1
    export LLAMA_PROFILE=1
    case "$campaign" in
      edge-auto) run_cell auto-edge hybrid 98304 73728 64 ;;
      edge-exchange) run_cell exchange-edge hybrid 98304 73728 64 60 79 ;;
      edge-ddr) run_cell ddr-edge ddr 98304 73728 64 ;;
      edge-group-real)
        export LLAMA_ATTN_DECODE=group
        run_cell auto-edge-group hybrid 98304 73728 64
        ;;
    esac
    ;;
  *)
    printf 'Unknown campaign %s\n' "$campaign" >&2
    exit 2
    ;;
esac
