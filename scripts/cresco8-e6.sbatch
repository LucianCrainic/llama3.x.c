#!/usr/bin/env bash
#SBATCH --job-name=llama-e6
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=112
#SBATCH --exclusive
#SBATCH --mem=0
#SBATCH --time=04:00:00
# Flat Mode campaign skeleton for the E6 batched-decode experiment (follow-up track E) on a dual Xeon Max 9480 node.
# Submit with a verified --partition/--nodelist and separate -o/-e paths. Required: ROOT (tree under test), OUT (new
# directory), PYTHON. Optional: MODELS ("NAME:ARTIFACT_DIR:HF_SNAPSHOT ..."; the artifact directory holds fp32-v3.bin,
# int8-v3.bin, tokenizer-v2.bin); without it the engine cells of the experiment are skipped and recorded.
# PHASE=E6 (STAGE=bench). The experiment needs no scheduler and no batched engine: it times the kernels decode would call.
#   1. gemm-bench at M = 1, 2, 4, 8, 16, 32 on the decode weight shapes (qkv, wo, w1/w3, w2, classifier; the list comes from
#      tools/e6_model.py shapes) of 8B and 1B, weights cold on HBM, 56 threads: AVX-512 FP32 (with the per-row GEMV as the
#      baseline), AVX-512 VNNI Q8 (W8A8, with its GEMV), AMX-INT8 and AMX-BF16. One socket with the full shapes, and two
#      sockets with the per-socket shard shapes, both sockets running at once.
#   2. amx-bench at M = 1..32 (the tile unit alone, operands in L2), 1 and 56 workers.
#   3. attention-bench decode at contexts 512 / 2048 / 8192 / 32768, 8B and 1B, all KV heads and half of them, split and
#      group (E5 measured 512 / 8192 / 32768 on the other node of the partition; 2048 is new; the 8B kv8 8192 split cell
#      is the node cross-check against E5's 0.372 ms per layer).
#   4. engine decode runs at N = 1 (the existing engine) from retained contexts of 512 and 2048 tokens, one and two sockets,
#      FP32 and W8A8, 1B and 8B, REPEATS repeats plus one profiled run at 2048: the measured N = 1 token the model must
#      reproduce, and the node cross-check against B7 (512 tokens).
# DRY_RUN=1 prints the matrix without running anything. Estimated at 2-3 hours with both models.
set -euo pipefail
# Slurm runs a copy of this script, so the library is found through ROOT, not through the script's own path.
: "${ROOT:?tree under test required}"
# shellcheck source=lib/cresco8-common.sh
source "$ROOT/scripts/lib/cresco8-common.sh"
require_env
job_begin
# S0 = cores 0-55, DDR 0, HBM 2; S1 = cores 56-111, DDR 1, HBM 3 (explicit map C/H/D = package/HBM/DDR).
EXPLICIT='explicit:0/2/0,1/3/1'
E6_MS=${E6_MS:-1,2,4,8,16,32}
E6_REPS=${E6_REPS:-100}
E6_COLD_MIB=${E6_COLD_MIB:-1024}
E6_CONTEXTS=${E6_CONTEXTS:-"512 2048"}
E6_MODELS_BENCH=${E6_MODELS_BENCH:-"8b 1b"}
E6_ATTN_CONTEXTS=${E6_ATTN_CONTEXTS:-512,2048,8192,32768}
E6_THP=${E6_THP:-off}
E6_BATCH=${E6_BATCH:-64}

# e6_bench NAME SOCKET BIN ENV_WORDS... -- BENCH_ARGS...: one bench launch on SOCKET's cores and HBM, output in E6/NAME.out;
# prints the pid so that two sockets can run at once. The command never stops the job; the exit status is recorded at wait.
e6_start() {
  local name=$1 socket=$2 bin=$3 env_words=() args=() side=env
  shift 3
  while (($#)); do
    if [[ $1 == -- ]]; then side=args; elif [[ $side == env ]]; then env_words+=("$1"); else args+=("$1"); fi
    shift
  done
  numactl "--cpunodebind=$socket" env LLAMA_BACKEND=teams LLAMA_THREADS=56 \
    ${env_words[@]+"${env_words[@]}"} "$ROOT/build/release/$bin" "${args[@]}" > "$OUT/E6/$name.out" 2> "$OUT/E6/$name.err" &
  E6_PID=$!
}

# e6_gemm TAG KERNELS KIND MODEL SOCKETS [--gemv]: gemm-bench on the decode shapes of MODEL; with SOCKETS=2 the per-socket shard
# shapes run on both sockets at once (outputs gemm-TAG-MODEL-s2.out for socket 0's node 2 and ...-s2-n3.out for node 3).
e6_gemm() {
  local tag=$1 kernels=$2 kind=$3 model=$4 sockets=$5 gemv=${6:-} shapes pid0 pid1 rc
  shapes=$(python3 "$ROOT/tools/e6_model.py" shapes "$model" "$sockets")
  local args=(--kernel "$kind" --m "$E6_MS" --shapes "$shapes" --reps "$E6_REPS" --cold-mib "$E6_COLD_MIB")
  [[ -z $gemv ]] || args+=("$gemv")
  if dry_run; then
    printf 'GEMM %s %s s%s kernels=%s: gemm-bench %s\n' "$tag" "$model" "$sockets" "$kernels" "${args[*]} --node 2[,3]" | tee -a "$OUT/dry-run.txt"
    return 0
  fi
  e6_start "gemm-$tag-$model-s$sockets" 0 gemm-bench "LLAMA_KERNELS=$kernels" -- --node 2 "${args[@]}"
  pid0=$E6_PID
  if [[ $sockets == 2 ]]; then
    e6_start "gemm-$tag-$model-s$sockets-n3" 1 gemm-bench "LLAMA_KERNELS=$kernels" -- --node 3 "${args[@]}"
    pid1=$E6_PID
  fi
  rc=0
  wait "$pid0" || rc=$?
  [[ $sockets != 2 ]] || wait "$pid1" || rc=$?
  if [[ $rc -eq 0 ]]; then record_result "gemm-$tag-$model-s$sockets" PASS; else record_result "gemm-$tag-$model-s$sockets" FAIL "exit=$rc"; fi
}

e6_kernels() {
  local model sockets
  for model in $E6_MODELS_BENCH; do
    for sockets in 1 2; do
      e6_gemm f32 avx512 f32 "$model" "$sockets" --gemv
      e6_gemm q8v avx512 q8 "$model" "$sockets" --gemv
      e6_gemm q8amx amx q8 "$model" "$sockets"
      e6_gemm bf16amx amx bf16 "$model" "$sockets"
    done
  done
  local workers k
  for workers in 1 56; do
    for k in 512 4096; do
      expect "amx-t$workers-k$k" ok "$OUT/E6" -- numactl --cpunodebind=0 env LLAMA_BACKEND=teams "LLAMA_THREADS=$workers" \
        LLAMA_KERNELS=amx "$ROOT/build/release/amx-bench" --kind both --k "$k" --m "$E6_MS,24" --reps 5000
    done
  done
}

e6_attention() {
  local shape kv kernel
  for shape in 8b 1b; do
    for kv in 8 4; do
      for kernel in split group; do
        expect "attention-$kernel-$shape-kv$kv" ok "$OUT/E6" -- numactl --cpunodebind=0 env LLAMA_BACKEND=teams LLAMA_THREADS=56 \
          LLAMA_KERNELS=avx512 "LLAMA_ATTN_DECODE=$kernel" "$ROOT/build/release/attention-bench" --node 2 \
          --mode decode --shape "$shape" --context "$E6_ATTN_CONTEXTS" --reps 1000 --kv-heads-local "$kv"
      done
    done
  done
}

# e6_launch NAME PRECISION SOCKETS CONTEXT REP STAGED TOKENIZER [x=LABEL]: one N = 1 decode run from a retained context.
e6_launch() {
  local name=$1 precision=$2 sockets=$3 ctx=$4 rep=$5 staged=$6 tokenizer=$7 extra=${8:-x=e6} engine=runq label cpu= log=warn perf=0
  local env_words=(LLAMA_BACKEND=teams LLAMA_THREADS=56 "LLAMA_TOPOLOGY=$EXPLICIT" LLAMA_PLACEMENT=hbm
    "LLAMA_THP=$E6_THP" LLAMA_PREFILL_GEMM=on "LLAMA_PREFILL_BATCH=$E6_BATCH" "LLAMA_SOCKETS=$sockets" LLAMA_ATTN_DECODE=split)
  [[ $precision != fp32 ]] || engine=run
  [[ $sockets == 2 ]] || cpu=0
  [[ $extra != *prof ]] || { log=info; env_words+=(LLAMA_PROFILE=1); }
  [[ $rep != 1 || $extra == *prof ]] || perf=$PERF
  label=$(run_label "$name" "$precision" teams hbm "$sockets" 56 "p=$ctx" "$extra")
  CPU_BIND=$cpu MEM_BIND= REP=$rep CONTEXT=$((ctx + 256)) GENERATION=128 WARM=$ctx LOG_LEVEL=$log \
    run_engine "$label" "$engine" "$staged" "$tokenizer" "$OUT/short.txt" 1 "$perf" "${env_words[@]}"
}

e6_engine() {
  local dir="$OUT/E6" entry name artifacts hf precision staged tokenizer rep ctx sockets
  if [[ -z $MODELS ]]; then
    echo 'MODELS is empty: no E6 engine run was made' | tee "$dir/no-models.txt"
    return 0
  fi
  need_shm
  a_models "$dir"
  for entry in $MODELS; do
    IFS=: read -r name artifacts hf <<< "$entry"
    tokenizer="$artifacts/tokenizer-v2.bin"
    for precision in $PRECISIONS; do
      staged=$(stage_model_on 0 "$name" "$precision" "$artifacts")
      for ((rep = 1; rep <= REPEATS; rep++)); do
        for ctx in $E6_CONTEXTS; do
          for sockets in 1 2; do
            e6_launch "$name" "$precision" "$sockets" "$ctx" "$rep" "$staged" "$tokenizer"
          done
        done
      done
      for sockets in 1 2; do
        e6_launch "$name" "$precision" "$sockets" 2048 1 "$staged" "$tokenizer" x=e6prof
      done
      unstage_model_on 0 "$name" "$precision"
    done
  done
}

phase_E6() {
  local dir="$OUT/E6"
  dry_run || snapshot_memory "$dir/memory-before.txt"
  e6_kernels
  e6_attention
  e6_engine
  if ! dry_run; then
    "$PYTHON" "$ROOT/tools/summarize_campaign.py" "$OUT" > "$dir/campaign-summary.md" || true
    "$PYTHON" "$ROOT/tools/e6_model.py" model "$dir" > "$dir/model.md" || true
  fi
}

require_phases
if ! dry_run; then
  check_stage
  validate_models
fi
run_phases
exit "$OVERALL"
