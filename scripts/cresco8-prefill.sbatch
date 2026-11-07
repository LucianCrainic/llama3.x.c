#!/usr/bin/env bash
#SBATCH --job-name=llama-prefill
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=112
#SBATCH --exclusive
#SBATCH --mem=0
#SBATCH --time=04:00:00
# Flat Mode campaign skeleton for batched prefill and its two-socket integration (tracks C and D) on a dual Xeon Max 9480 node.
# Submit with a verified --partition/--nodelist and separate -o/-e paths; every submission needs the
# owner's go-ahead for that job. Required: ROOT (tree under test), OUT (new directory), PYTHON. Optional: MODELS
# ("NAME:ARTIFACT_DIR:HF_SNAPSHOT ..."; the artifact directory holds fp32-v3.bin, int8-v3.bin, tokenizer-v2.bin);
# without it the model gates are skipped and recorded.
# PHASE is a space-separated list of phase ids ("A1 A2"); phase_<ID> functions below (or in the library) hold each
# phase's matrix. PHASE=none runs only the common check. The exit status is non-zero if any expectation failed.
set -euo pipefail
# Slurm runs a copy of this script, so the library is found through ROOT, not through the script's own path.
: "${ROOT:?tree under test required}"
# shellcheck source=lib/cresco8-common.sh
source "$ROOT/scripts/lib/cresco8-common.sh"
require_env
job_begin
# ---- shared helpers of the C phases --------------------------------------------------------------------------------
# c_prompts DIR LENGTHS: writes the fixed-length prompts and tokenizes short.txt and every prompt-N.txt per model.
c_prompts() {
  local dir=$1 lengths=$2 entry name artifacts hf n
  a_models "$dir" "$lengths"
  for entry in $MODELS; do
    IFS=: read -r name artifacts hf <<< "$entry"
    for n in ${lengths//,/ }; do
      tokens_of "$name" "$OUT/prompt-$n.txt" "$dir/$name-tokens-$n.txt"
    done
  done
}

# c_run NAME PRECISION SOURCE TOKENIZER REP PROMPT BACKEND LABEL_WORDS... -- ENV_WORDS...: one benchmark launch on
# socket 0 against a checkpoint staged on HBM node 2 (the baseline of the kernel campaigns), interleavable by REP.
c_run() {
  local name=$1 precision=$2 source=$3 tokenizer=$4 rep=$5 prompt=$6 label_words=() env_words=() side=label label engine=run
  shift 6
  while (($#)); do
    if [[ $1 == -- ]]; then side=env; elif [[ $side == label ]]; then label_words+=("$1"); else env_words+=("$1"); fi
    shift
  done
  [[ $precision == fp32 ]] || engine=runq
  label=$(run_label "$name" "$precision" teams file 1 56 "${label_words[@]}")
  CPU_BIND=0 MEM_BIND=2 REP=$rep run_engine "$label" "$engine" "$source" "$tokenizer" "$prompt" 1 "$PERF" \
    LLAMA_BACKEND=teams LLAMA_THREADS=56 OMP_PROC_BIND=false "${env_words[@]}"
}

# ---- C1: batched forward API --------------------------------------------------------------------------------------
# The reference batched path must equal the per-token path bit for bit at every position, and skipping the prompt
# classifiers is the only speed difference expected.
phase_C1() {
  local dir="$OUT/C1" entry name artifacts hf precision staged tokenizer backend m rep benv gemm_env
  if [[ -z $MODELS ]]; then
    echo 'MODELS is empty: no C1 model run was made' | tee "$dir/no-models.txt"
    return 0
  fi
  need_shm
  c_prompts "$dir" 2048
  for entry in $MODELS; do
    IFS=: read -r name artifacts hf <<< "$entry"
    tokenizer="$artifacts/tokenizer-v2.bin"
    for precision in $PRECISIONS; do
      staged=$(stage_model_on 0 "$name" "$precision" "$artifacts")
      # The bitwise criterion is about the reference batched path. On AVX-512 hosts the default FP32 batched path is the
      # C2 GEMM, which sums in another order than the GEMV (its agreement is C2's budget gate), so FP32 pins it off here.
      gemm_env=()
      [[ $precision != fp32 ]] || gemm_env=(LLAMA_PREFILL_GEMM=off)
      for backend in omp teams; do
        read -r -a benv <<< "$(backend_env "$backend" 56 | tr '\n' ' ')"
        for m in 7 32; do
          CMP_FLAGS=(--bitwise --batch-b "$m")
          a_bitwise "$dir" "$name" "$precision" "$staged" "$backend-m1-vs-m$m" "${benv[@]}" -- "${benv[@]}" \
            ${gemm_env[@]+"${gemm_env[@]}"}
        done
      done
      if [[ $precision == int8 ]]; then
        read -r -a benv <<< "$(backend_env teams 56 | tr '\n' ' ')"
        CMP_FLAGS=(--bitwise --batch-b 32)
        a_bitwise "$dir" "$name" "$precision" "$staged" teams-w8a32-m1-vs-m32 "${benv[@]}" LLAMA_Q8_ACTIVATIONS=fp32 -- \
          "${benv[@]}" LLAMA_Q8_ACTIVATIONS=fp32
      fi
      CMP_FLAGS=(--bitwise)
      unstage_model_on 0 "$name" "$precision"
    done
    # Speed: only the classifier skipped, 8B FP32 (HBM-staged), 86- and 2048-token prompts, interleaved repeats.
    # C1_SPEED=0 re-runs only the comparisons above.
    [[ ${C1_SPEED:-1} != 0 ]] || continue
    staged=$(stage_model_on 2 "$name" fp32 "$artifacts")
    for ((rep = 1; rep <= REPEATS; rep++)); do
      for logits in all last; do
        c_run "$name" fp32 "$staged" "$tokenizer" "$rep" "$OUT/short.txt" p=86 "x=l$logits" -- LLAMA_PREFILL_BATCH=1 \
          LLAMA_PREFILL_LOGITS=$logits
        CONTEXT=2304 GENERATION=16 c_run "$name" fp32 "$staged" "$tokenizer" "$rep" "$OUT/prompt-2048.txt" p=2048 \
          c=2304 "x=l$logits" -- LLAMA_PREFILL_BATCH=1 LLAMA_PREFILL_LOGITS=$logits
      done
    done
    unstage_model_on 2 "$name" fp32
  done
  dry_run || "$PYTHON" "$ROOT/tools/summarize_campaign.py" "$OUT" > "$dir/summary.md"
}

# ---- C2: AVX-512 FP32 GEMM -------------------------------------------------------------------------------------------
phase_C2() {
  local dir="$OUT/C2" bin="$ROOT/build/release" entry name artifacts hf staged tokenizer n m rep benv
  expect kernels-teams-1-28-56 ok "$dir" -- numactl --cpunodebind=0 env LLAMA_TEST_TEAMS=1,28,56 "$bin/test-kernels"
  expect gemm-teams-1-28-56 ok "$dir" -- numactl --cpunodebind=0 env LLAMA_TEST_TEAMS=1,28,56 "$bin/test-gemm"
  if ! dry_run; then
    grep -q 'gemm: avx512 teams:56 ok' "$dir/gemm-teams-1-28-56.out" && record_result gemm-avx512-56 PASS ||
      record_result gemm-avx512-56 FAIL 'avx512 teams:56 was not exercised'
  fi
  # Weights bound to HBM node 2; the GEMV-per-row line is the per-token baseline at the same M.
  expect gemm-bench ok "$dir" -- numactl --cpunodebind=0 env LLAMA_BACKEND=teams LLAMA_THREADS=56 OMP_PROC_BIND=false \
    "$bin/gemm-bench" --node 2 --gemv
  if ! dry_run && [[ -s $dir/gemm-bench.out ]]; then
    cp "$dir/gemm-bench.out" "$dir/gemm-bench.jsonl"
    python3 "$ROOT/tools/summarize_gemm.py" "$dir/gemm-bench.jsonl" > "$dir/gemm-bench.md"
  fi
  if [[ -z $MODELS ]]; then
    echo 'MODELS is empty: no C2 model run was made' | tee "$dir/no-models.txt"
    return 0
  fi
  need_shm
  c_prompts "$dir" 128,256,512,2048
  for entry in $MODELS; do
    IFS=: read -r name artifacts hf <<< "$entry"
    tokenizer="$artifacts/tokenizer-v2.bin"
    staged=$(stage_model_on 0 "$name" fp32 "$artifacts")
    read -r -a benv <<< "$(backend_env teams 56 | tr '\n' ' ')"
    # GEMM batches of 64 against the per-token path at every position of four prompts: within the logit budget.
    CMP_FLAGS=(--max-abs 1e-3 --batch-b 64)
    for n in short 128 256 512; do
      CMP_TOKENS="$dir/$name-tokens$([[ $n == short ]] || echo "-$n").txt"
      a_bitwise "$dir" "$name" fp32 "$staged" "gemm-m64-prompt$n" "${benv[@]}" -- "${benv[@]}" LLAMA_PREFILL_GEMM=on
    done
    CMP_FLAGS=(--bitwise) CMP_TOKENS=
    unstage_model_on 0 "$name" fp32
    # Short prefill sweep on the HBM-staged baseline; C7 is the full campaign.
    staged=$(stage_model_on 2 "$name" fp32 "$artifacts")
    for ((rep = 1; rep <= REPEATS; rep++)); do
      for m in 16 32 64 128; do
        CONTEXT=2304 GENERATION=16 c_run "$name" fp32 "$staged" "$tokenizer" "$rep" "$OUT/prompt-2048.txt" "m=$m" p=2048 \
          c=2304 -- LLAMA_PREFILL_BATCH=$m LLAMA_PREFILL_GEMM=on
      done
    done
    unstage_model_on 2 "$name" fp32
  done
  dry_run || "$PYTHON" "$ROOT/tools/summarize_campaign.py" "$OUT" > "$dir/summary.md"
}

# ---- C3: batched causal attention and KV append -----------------------------------------------------------------------
phase_C3() {
  local dir="$OUT/C3" bin="$ROOT/build/release" entry name artifacts hf staged tokenizer n label benv
  expect kernels-attention-teams-1-28-56 ok "$dir" -- numactl --cpunodebind=0 env LLAMA_TEST_TEAMS=1,28,56 "$bin/test-kernels"
  if [[ -z $MODELS ]]; then
    echo 'MODELS is empty: no C3 model run was made' | tee "$dir/no-models.txt"
    return 0
  fi
  need_shm
  c_prompts "$dir" 512,2048,8192
  for entry in $MODELS; do
    IFS=: read -r name artifacts hf <<< "$entry"
    tokenizer="$artifacts/tokenizer-v2.bin"
    staged=$(stage_model_on 0 "$name" fp32 "$artifacts")
    read -r -a benv <<< "$(backend_env teams 56 | tr '\n' ' ')"
    # GEMM plus batched attention, M = 64, against the per-token path: all positions of the 86- and 512-token
    # prompts, and the final position of the 2048-token prompt (its all-position logits are 1 GB per side).
    CMP_FLAGS=(--max-abs 1e-3 --batch-b 64)
    CMP_TOKENS="$dir/$name-tokens.txt"
    a_bitwise "$dir" "$name" fp32 "$staged" attention-m64-prompt86 "${benv[@]}" -- "${benv[@]}" LLAMA_PREFILL_GEMM=on
    CMP_TOKENS="$dir/$name-tokens-512.txt"
    a_bitwise "$dir" "$name" fp32 "$staged" attention-m64-prompt512 "${benv[@]}" -- "${benv[@]}" LLAMA_PREFILL_GEMM=on
    CMP_FLAGS=(--max-abs 1e-3 --batch-b 64 --last-only)
    CMP_TOKENS="$dir/$name-tokens-2048.txt"
    a_bitwise "$dir" "$name" fp32 "$staged" attention-m64-prompt2048-last "${benv[@]}" -- "${benv[@]}" LLAMA_PREFILL_GEMM=on
    CMP_FLAGS=(--bitwise) CMP_TOKENS=
    unstage_model_on 0 "$name" fp32
    # Where attention overtakes the GEMMs: the phase profile of prefill at 2048 and 8192 tokens, M = 64.
    staged=$(stage_model_on 2 "$name" fp32 "$artifacts")
    for n in 2048 8192; do
      CONTEXT=8448 GENERATION=1 c_run "$name" fp32 "$staged" "$tokenizer" 1 "$OUT/prompt-$n.txt" m=64 "p=$n" c=8448 x=prof -- \
        LLAMA_PREFILL_BATCH=64 LLAMA_PREFILL_GEMM=on LLAMA_PROFILE=1
    done
    unstage_model_on 2 "$name" fp32
    if ! dry_run; then
      "$PYTHON" - "$OUT" "$name" > "$dir/attention-share-$name.txt" << 'PY'
import json, sys
from pathlib import Path
out, name = Path(sys.argv[1]), sys.argv[2]
for path in sorted(out.glob(f'{name}-fp32-teams-file-s1-t56-m64-p*-c8448-xprof-1.json')):
    stage = (json.loads(path.read_text()).get('profile_seconds') or {}).get('prefill') or {}
    forward = stage.get('forward') or 0
    parts = {k: v for k, v in stage.items() if k not in ('forward', 'forward_passes') and isinstance(v, (int, float))}
    shares = ', '.join(f'{k} {100 * v / forward:.0f}%' for k, v in sorted(parts.items(), key=lambda kv: -kv[1])[:6]) if forward else 'no profile'
    print(f'{path.stem}: forward {forward:.2f}s; {shares}')
PY
    fi
  done
  dry_run || "$PYTHON" "$ROOT/tools/summarize_campaign.py" "$OUT" > "$dir/summary.md"
}

if ! dry_run; then
  check_stage
  validate_models
fi
run_phases
exit "$OVERALL"
