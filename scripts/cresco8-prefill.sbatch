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

# ---- shared helpers of C4-C7 ---------------------------------------------------------------------------------------
# c_record DIR NAME PRECISION STAGED LABEL ENVA_WORDS... -- ENVB_WORDS...: a_bitwise that records instead of gating (the
# BF16 mode has no budget yet): the comparison JSON carries max_abs, rms and the argmax agreement, whatever they are.
c_record() {
  local dir=$1 name=$2 precision=$3 staged=$4 label=$5 side=a args=() tokens=${CMP_TOKENS:-$1/$2-tokens.txt}
  shift 5
  while (($#)); do
    if [[ $1 == -- ]]; then side=b; else args+=("--env-$side" "$1"); fi
    shift
  done
  expect "compare-$name-$precision-$label" any "$dir" -- numactl --cpunodebind=0 "$PYTHON" \
    "$ROOT/tools/compare_engines.py" --probe "$ROOT/build/release/probe-$precision" --checkpoint "$staged" \
    --tokens-file "$tokens" "${args[@]}" "${CMP_FLAGS[@]}" --max-abs 1e30 --output "$dir/compare-$name-$precision-$label.json"
}

# c_rung RUNG M: the ladder of the plans as environment. Sets RUNG_PREC (the precision field of the run label), RUNG_ENGINE
# and RUNG_ENV for a prefill microbatch of M tokens. A rung is one kernel set; "per_token" is the baseline of its precision.
#   per_token / per_token_q8   LLAMA_PREFILL_BATCH=1 on the AVX-512 kernels (today's path)
#   avx512_fp32                AVX-512 FP32 GEMM
#   amx_bf16                   AMX-BF16 (the explicit BF16 precision mode)
#   avx512_q8                  tiled AVX-512 VNNI batched Q8, W8A8
#   amx_int8                   AMX-INT8 W8A8
#   avx512_q8_w8a32            W8A32, always per row on AVX-512
c_rung() {
  local m=$2
  case $1 in
    per_token) RUNG_PREC=fp32 RUNG_ENGINE=run RUNG_ENV=(LLAMA_KERNELS=avx512 LLAMA_PREFILL_BATCH=1) ;;
    avx512_fp32) RUNG_PREC=fp32 RUNG_ENGINE=run RUNG_ENV=(LLAMA_KERNELS=avx512 LLAMA_PREFILL_GEMM=on "LLAMA_PREFILL_BATCH=$m") ;;
    amx_bf16) RUNG_PREC=bf16 RUNG_ENGINE=run RUNG_ENV=(LLAMA_KERNELS=amx LLAMA_PRECISION=bf16 "LLAMA_PREFILL_BATCH=$m") ;;
    per_token_q8) RUNG_PREC=int8 RUNG_ENGINE=runq RUNG_ENV=(LLAMA_KERNELS=avx512 LLAMA_PREFILL_BATCH=1) ;;
    avx512_q8) RUNG_PREC=int8 RUNG_ENGINE=runq RUNG_ENV=(LLAMA_KERNELS=avx512 LLAMA_PREFILL_GEMM=on "LLAMA_PREFILL_BATCH=$m") ;;
    amx_int8) RUNG_PREC=int8 RUNG_ENGINE=runq RUNG_ENV=(LLAMA_KERNELS=amx "LLAMA_PREFILL_BATCH=$m") ;;
    avx512_q8_w8a32)
      RUNG_PREC=w8a32 RUNG_ENGINE=runq RUNG_ENV=(LLAMA_KERNELS=avx512 LLAMA_Q8_ACTIVATIONS=fp32 "LLAMA_PREFILL_BATCH=$m") ;;
    *) die "unknown rung '$1'" ;;
  esac
}

# c_rungs PRECISION: the rungs measured on a checkpoint of that precision. c_rung_ms RUNG: its microbatch sizes (the
# baselines run one token at a time; W8A32 prefill is per row whatever the batch, so two sizes show that it does not matter).
c_rungs() {
  if [[ $1 == fp32 ]]; then echo per_token avx512_fp32 amx_bf16; else echo per_token_q8 avx512_q8 amx_int8 avx512_q8_w8a32; fi
}
c_rung_ms() {
  case $1 in
    per_token | per_token_q8) echo 1 ;;
    avx512_q8_w8a32) echo 1 64 ;;
    *) echo 16 32 64 128 256 ;;
  esac
}

# c_launch NAME RUNG M THREADS REP PROMPT STAGED TOKENIZER [LABEL_WORDS...] [-- ENV_WORDS...]: one launch of one rung on
# socket 0 against a checkpoint staged on its HBM (node 2), REP selecting the repetition so a caller can interleave.
c_launch() {
  local name=$1 rung=$2 m=$3 threads=$4 rep=$5 prompt=$6 staged=$7 tokenizer=$8 label_words=() env_words=() side=label label
  shift 8
  while (($#)); do
    if [[ $1 == -- ]]; then side=env; elif [[ $side == label ]]; then label_words+=("$1"); else env_words+=("$1"); fi
    shift
  done
  c_rung "$rung" "$m"
  label=$(run_label "$name" "$RUNG_PREC" teams file 1 "$threads" "m=$m" "k=$rung" ${label_words[@]+"${label_words[@]}"})
  CPU_BIND=0 MEM_BIND=2 REP=$rep run_engine "$label" "$RUNG_ENGINE" "$staged" "$tokenizer" "$prompt" 1 "$PERF" \
    LLAMA_BACKEND=teams "LLAMA_THREADS=$threads" OMP_PROC_BIND=false "${RUNG_ENV[@]}" ${env_words[@]+"${env_words[@]}"}
}

# c_fixtures DIR: the synthetic checkpoints the engine-level AMX tests read (dim 128, group sizes 64 and 32).
c_fixtures() {
  dry_run && { echo "fixtures: $1/fixture-gs64 $1/fixture-gs32 (dry run)"; return 0; }
  "$PYTHON" - "$ROOT/tests" "$1" << 'PY'
import sys
sys.path.insert(0, sys.argv[1])
from fixtures import create
for gs in (64, 32):
    create(f'{sys.argv[2]}/fixture-gs{gs}', config=(128, 256, 2, 2, 1, 512, 64), gs=gs, seq_len=64)
create(f'{sys.argv[2]}/fixture-fp32', seq_len=64)
PY
}

# c_grep NAME FILE PATTERN [DETAIL]: PASS when the test output shows the line, so a skipped hardware path cannot pass.
c_grep() {
  dry_run && return 0
  if grep -q -- "$3" "$2"; then record_result "$1" PASS "${4:-}"; else record_result "$1" FAIL "no '$3' in $(basename "$2")"; fi
}

# ---- C4: AMX enablement -------------------------------------------------------------------------------------------
# Model-free. test-amx, test-kernels and test-gemm run the tile kernels on the real instructions; the software tile
# model (test-amx-emul) runs as the control. `make test` in the common check has already run all of them once, and
# its test.log shows "amx unavailable" if the node lacks AMX or the kernel refuses the permission. The microbenchmark
# records TFLOP/s and TOPS for one core and one socket against the measured peaks of docs/hbm-roofline.md.
phase_C4() {
  local dir="$OUT/C4" bin="$ROOT/build/release" kind threads
  expect test-amx ok "$dir" -- numactl --cpunodebind=0 "$bin/test-amx"
  c_grep amx-hardware-tiles "$dir/test-amx.out" 'amx: hardware tiles, permission granted' 'ARCH_GET_XCOMP_PERM shows XTILEDATA'
  c_grep amx-bf16-tile-gemm "$dir/test-amx.out" 'amx: bf16 tile gemm ok'
  c_grep amx-int8-tile-gemm "$dir/test-amx.out" 'amx: int8 tile gemm ok'
  c_grep amx-per-thread-tiles "$dir/test-amx.out" 'tile configuration ok'
  expect test-amx-emul ok "$dir" -- numactl --cpunodebind=0 "$bin/test-amx-emul"
  for threads in 1 56; do
    expect "kernels-amx-$threads" ok "$dir" -- numactl --cpunodebind=0 env LLAMA_TEST_TEAMS=$threads "$bin/test-kernels"
    c_grep "kernels-amx-teams-$threads" "$dir/kernels-amx-$threads.out" "kernels: amx teams:$threads ok" 'amx ok'
    expect "gemm-amx-$threads" ok "$dir" -- numactl --cpunodebind=0 env LLAMA_TEST_TEAMS=$threads "$bin/test-gemm"
    c_grep "gemm-amx-teams-$threads" "$dir/gemm-amx-$threads.out" "gemm: amx teams:$threads ok"
  done
  # Tile microbenchmark, operands in L2: TFLOP/s (BF16) and TOPS (int8) per worker and in total.
  for threads in 1 56; do
    for kind in bf16 int8; do
      expect "amx-bench-$kind-t$threads" ok "$dir" -- numactl --cpunodebind=0 env LLAMA_KERNELS=amx LLAMA_BACKEND=teams \
        LLAMA_THREADS=$threads OMP_PROC_BIND=false "$bin/amx-bench" --kind "$kind"
    done
  done
  dry_run || cat "$dir"/amx-bench-*.out > "$dir/amx-bench.jsonl" 2> /dev/null || true
}

# ---- C5: AMX-BF16 precision mode ------------------------------------------------------------------------------------
# The BF16 mode has no budget until this phase has run: every model comparison is recorded, not gated, and bf16-error.md
# lists the distribution the provisional gate of C5 section 4.3 is to be set from.
phase_C5() {
  local dir="$OUT/C5" bin="$ROOT/build/release" entry name artifacts hf staged tokenizer n m benv label rep status=0
  expect test-bf16 ok "$dir" -- numactl --cpunodebind=0 "$bin/test-bf16"
  c_grep bf16-amx-gemm "$dir/test-bf16.out" 'bf16: gemm amx teams:7 ok' 'AMX BF16 GEMM within 2e-6 of the double reference on 1, 3, 7 workers'
  expect test-bf16-emul ok "$dir" -- numactl --cpunodebind=0 "$bin/test-bf16-emul"
  c_fixtures "$dir"
  expect bf16-engine-fixture ok "$dir" -- numactl --cpunodebind=0 "$bin/test-bf16-engine" "$dir/fixture-fp32/fp32.bin"
  c_grep bf16-engine-amx "$dir/bf16-engine-fixture.out" 'amx vs scalar BF16 reference forward' 'AMX forward vs the scalar BF16 forward'
  if [[ -z $MODELS || " $PRECISIONS " != *" fp32 "* ]]; then
    echo 'MODELS is empty or PRECISIONS has no fp32: no C5 model run was made' | tee "$dir/no-models.txt"
    return 0
  fi
  need_shm
  c_prompts "$dir" 128,256,512,2048
  for entry in $MODELS; do
    IFS=: read -r name artifacts hf <<< "$entry"
    tokenizer="$artifacts/tokenizer-v2.bin"
    staged=$(stage_model_on 0 "$name" fp32 "$artifacts")
    read -r -a benv <<< "$(backend_env teams 56 | tr '\n' ' ')"
    # FP32 per-token (A) against BF16 on AMX at M = 64 (B), every position of four prompts: recorded.
    CMP_FLAGS=(--batch-b 64)
    for n in short 128 256 512; do
      CMP_TOKENS="$dir/$name-tokens$([[ $n == short ]] || echo "-$n").txt"
      c_record "$dir" "$name" fp32 "$staged" "bf16-amx-m64-prompt$n" "${benv[@]}" LLAMA_KERNELS=avx512 -- "${benv[@]}" \
        LLAMA_KERNELS=amx LLAMA_PRECISION=bf16
    done
    # The FP32 AVX-512 GEMM against per-token, for the same prompts, is the reference the BF16 error is read against.
    CMP_FLAGS=(--batch-b 64)
    for n in short 128 256 512; do
      CMP_TOKENS="$dir/$name-tokens$([[ $n == short ]] || echo "-$n").txt"
      c_record "$dir" "$name" fp32 "$staged" "fp32-avx512-m64-prompt$n" "${benv[@]}" LLAMA_KERNELS=avx512 -- "${benv[@]}" \
        LLAMA_KERNELS=avx512 LLAMA_PREFILL_GEMM=on
    done
    CMP_FLAGS=(--bitwise) CMP_TOKENS=
    # Hugging Face in BF16 mode: logits through the batched path, greedy generation prefilled through it: recorded.
    if [[ ${HF_GATE:-1} == 0 ]]; then
      echo 'HF_GATE=0: the BF16 Hugging Face comparison was not run' | tee -a "$dir/no-hf-gate.txt"
    elif dry_run; then
      printf 'VALIDATE %s bf16 env=LLAMA_KERNELS=amx LLAMA_PRECISION=bf16 LLAMA_PREFILL_BATCH=64\n' "$name" |
        tee -a "$OUT/dry-run.txt"
    else
      (cd "$ROOT" && env LLAMA_KERNELS=amx LLAMA_PRECISION=bf16 LLAMA_PREFILL_BATCH=64 LLAMA_BACKEND=teams \
        LLAMA_THREADS=56 OMP_PROC_BIND=false OMP_NUM_THREADS=56 numactl --cpunodebind=0 --membind=0 "$PYTHON" \
        tools/validate.py --hf "$hf" --checkpoint "$artifacts/fp32-v3.bin" --tokenizer "$tokenizer" --precision fp32 \
        --threads 56 --batch 64 --record-only --max-abs 1.0 --rtol 0.05 --output "$dir/correctness-$name-bf16.json") \
        > "$dir/correctness-$name-bf16.stdout" 2> "$dir/correctness-$name-bf16.stderr" || status=$?
      printf '%s bf16 exit=%s\n' "$name" "$status" >> "$dir/correctness-status.txt"
    fi
    unstage_model_on 0 "$name" fp32
    # Packing time, packed bytes and the per-socket budget under engine-owned HBM placement (needs the A4 and A5 checks),
    # then the short M sweep on the staged baseline. Placement is the engine's here, so no memory policy is imposed.
    staged=$(stage_model_on 2 "$name" fp32 "$artifacts")
    label=$(run_label "$name" bf16 teams hbm 1 56 m=64 p=86 k=amx_bf16 x=pack)
    CONTEXT=2048 GENERATION=4 CPU_BIND=0 run_engine "$label" run "$staged" "$tokenizer" "$OUT/short.txt" 1 "$PERF" \
      LLAMA_BACKEND=teams LLAMA_THREADS=56 OMP_PROC_BIND=false LLAMA_PLACEMENT=hbm LLAMA_KERNELS=amx LLAMA_PRECISION=bf16 \
      LLAMA_PREFILL_BATCH=64
    for ((rep = 1; rep <= REPEATS; rep++)); do
      for m in 32 64 128 256; do
        CONTEXT=2304 GENERATION=16 c_launch "$name" amx_bf16 "$m" 56 "$rep" "$OUT/prompt-2048.txt" "$staged" "$tokenizer" p=2048 c=2304
      done
    done
    unstage_model_on 2 "$name" fp32
  done
  if ! dry_run; then
    "$PYTHON" - "$dir" > "$dir/bf16-error.md" << 'PY'
import glob, json, sys
rows = []
for path in sorted(glob.glob(f'{sys.argv[1]}/compare-*-fp32-*prompt*.json')):
    data = json.load(open(path))
    summary = data.get('summary') or {}
    rows.append((path.rsplit('/', 1)[-1][len('compare-'):-5], data.get('tokens'), summary.get('max_abs'),
                 summary.get('rms'), summary.get('argmax_agreement')))
print('| comparison | tokens | max_abs | rms | argmax agreement |\n| --- | ---: | ---: | ---: | ---: |')
for label, tokens, max_abs, rms, agree in rows:
    print(f'| {label} | {tokens} | {max_abs:.3e} | {rms:.3e} | {agree:.4f} |' if max_abs is not None else f'| {label} | - | - | - | - |')
PY
    "$PYTHON" "$ROOT/tools/summarize_campaign.py" "$OUT" > "$dir/summary.md"
  fi
}

# ---- C6: AMX-INT8 for Q8_0 --------------------------------------------------------------------------------------------
# Gates: the batched AVX-512 Q8 path against the per-token Q8 path, and AMX-INT8 against the batched AVX-512 Q8 path, all
# positions of the 86-token prompt and the final position of the 2048-token one, within the logit budget with argmax 100%.
phase_C6() {
  local dir="$OUT/C6" bin="$ROOT/build/release" entry name artifacts hf staged tokenizer n m benv rep rung variant words status=0
  expect test-q8 ok "$dir" -- numactl --cpunodebind=0 "$bin/test-q8"
  c_grep q8-amx-int8 "$dir/test-q8.out" 'amx-int8 ran' 'AMX-INT8 kernel within budget; int32 group dots exact'
  c_grep q8-avx512-vnni "$dir/test-q8.out" 'avx512 tile kernel' 'the VNNI tile kernel ran on the CPU'
  expect test-q8-emul ok "$dir" -- numactl --cpunodebind=0 "$bin/test-q8-emul"
  c_fixtures "$dir"
  expect q8-engine-fixture ok "$dir" -- numactl --cpunodebind=0 "$bin/test-q8-engine" "$dir/fixture-gs64/int8.bin" \
    "$dir/fixture-gs32/int8.bin"
  c_grep q8-engine-amx "$dir/q8-engine-fixture.out" 'amx-int8 vs portable batched Q8 forward' 'AMX-INT8 forward vs the portable one'
  if [[ -z $MODELS || " $PRECISIONS " != *" int8 "* ]]; then
    echo 'MODELS is empty or PRECISIONS has no int8: no C6 model run was made' | tee "$dir/no-models.txt"
    return 0
  fi
  need_shm
  c_prompts "$dir" 512,2048
  for entry in $MODELS; do
    IFS=: read -r name artifacts hf <<< "$entry"
    tokenizer="$artifacts/tokenizer-v2.bin"
    staged=$(stage_model_on 0 "$name" int8 "$artifacts")
    read -r -a benv <<< "$(backend_env teams 56 | tr '\n' ' ')"
    CMP_FLAGS=(--max-abs 1e-3 --batch-b 64)
    CMP_TOKENS="$dir/$name-tokens.txt"
    a_bitwise "$dir" "$name" int8 "$staged" avx512-q8-m64-vs-per-token-prompt86 "${benv[@]}" LLAMA_KERNELS=avx512 -- "${benv[@]}" \
      LLAMA_KERNELS=avx512 LLAMA_PREFILL_GEMM=on
    CMP_FLAGS=("${W8A8_GATE[@]}" --batch-a 64 --batch-b 64)
    a_bitwise "$dir" "$name" int8 "$staged" amx-int8-m64-vs-avx512-q8-prompt86 "${benv[@]}" LLAMA_KERNELS=avx512 \
      LLAMA_PREFILL_GEMM=on -- "${benv[@]}" LLAMA_KERNELS=amx
    CMP_FLAGS=(--max-abs 1e-3 --batch-b 64 --last-only)
    CMP_TOKENS="$dir/$name-tokens-2048.txt"
    a_bitwise "$dir" "$name" int8 "$staged" avx512-q8-m64-vs-per-token-prompt2048-last "${benv[@]}" LLAMA_KERNELS=avx512 -- \
      "${benv[@]}" LLAMA_KERNELS=avx512 LLAMA_PREFILL_GEMM=on
    # AMX-INT8 against the AVX-512 batched path at the last position of the long prompt: both sides batched, last-only needs
    # A to be per-token, so the AMX side is compared with the per-token path instead (same budget).
    CMP_FLAGS=("${W8A8_GATE_LAST[@]}" --batch-b 64 --last-only)
    a_bitwise "$dir" "$name" int8 "$staged" amx-int8-m64-vs-per-token-prompt2048-last "${benv[@]}" LLAMA_KERNELS=avx512 -- \
      "${benv[@]}" LLAMA_KERNELS=amx
    # The arithmetic itself: with FP32 activations there is no requantization, so the AMX selection (which keeps W8A32 on
    # AVX-512) agrees with the per-token path to the strict budget.
    CMP_FLAGS=(--max-abs 1e-3 --batch-b 64)
    CMP_TOKENS="$dir/$name-tokens.txt"
    a_bitwise "$dir" "$name" int8 "$staged" amx-w8a32-m64-vs-per-token-prompt86 "${benv[@]}" LLAMA_KERNELS=avx512 \
      LLAMA_Q8_ACTIVATIONS=fp32 -- "${benv[@]}" LLAMA_KERNELS=amx LLAMA_Q8_ACTIVATIONS=fp32
    CMP_FLAGS=(--bitwise) CMP_TOKENS=
    # Hugging Face, recorded as for the other Q8 comparisons: per-token Q8 and AMX-INT8 batched; read the two max_abs
    # values against each other (AMX-INT8 <= per-token + 1e-3).
    if [[ ${HF_GATE:-1} == 0 ]]; then
      echo 'HF_GATE=0: the Q8 Hugging Face recording was not run' | tee -a "$dir/no-hf-gate.txt"
    elif dry_run; then
      printf 'VALIDATE %s int8 per-token and amx-int8 batch=64\n' "$name" | tee -a "$OUT/dry-run.txt"
    else
      for variant in per-token amx-int8; do
        words=(LLAMA_KERNELS=avx512)
        [[ $variant == per-token ]] || words=(LLAMA_KERNELS=amx LLAMA_PREFILL_BATCH=64)
        status=0
        (cd "$ROOT" && env "${words[@]}" LLAMA_BACKEND=teams LLAMA_THREADS=56 OMP_PROC_BIND=false OMP_NUM_THREADS=56 \
          numactl --cpunodebind=0 --membind=0 "$PYTHON" tools/validate.py --hf "$hf" --checkpoint "$artifacts/int8-v3.bin" \
          --tokenizer "$tokenizer" --precision int8 --threads 56 --record-only --max-abs 1.0 --rtol 0.05 \
          $([[ $variant == per-token ]] || echo "--batch 64") --output "$dir/correctness-$name-q8-$variant.json") \
          > "$dir/correctness-$name-q8-$variant.stdout" 2> "$dir/correctness-$name-q8-$variant.stderr" || status=$?
        printf '%s q8 %s exit=%s\n' "$name" "$variant" "$status" >> "$dir/correctness-status.txt"
      done
    fi
    unstage_model_on 0 "$name" int8
    # Short sweep on the HBM-staged baseline; C7 is the full campaign.
    staged=$(stage_model_on 2 "$name" int8 "$artifacts")
    for ((rep = 1; rep <= REPEATS; rep++)); do
      CONTEXT=2304 GENERATION=16 c_launch "$name" per_token_q8 1 56 "$rep" "$OUT/prompt-2048.txt" "$staged" "$tokenizer" p=2048 c=2304
      for m in 32 64 128 256; do
        for rung in avx512_q8 amx_int8; do
          CONTEXT=2304 GENERATION=16 c_launch "$name" "$rung" "$m" 56 "$rep" "$OUT/prompt-2048.txt" "$staged" "$tokenizer" \
            p=2048 c=2304
        done
      done
    done
    unstage_model_on 2 "$name" int8
  done
  dry_run || "$PYTHON" "$ROOT/tools/summarize_campaign.py" "$OUT" > "$dir/summary.md"
}

# ---- C7: one-socket prefill campaign ------------------------------------------------------------------------------------
# The rungs of the ladder (c_rung) by model, prompt length, microbatch size and thread count, all on socket 0 with
# LLAMA_BACKEND=teams against checkpoints staged on its HBM, REPEATS interleaved repeats, 32 generated tokens so decode is
# measured but not dominant. The gates of C2, C3, C5 and C6 run first on the 2048-token prompt; a failed gate stops the
# benchmark. Passes: (1) gates; (2) M sweep at 2048; (3) the best two M per rung from (2); (4) the other prompt lengths at
# M = 1 and those two; (5) 14 and 28 threads at the best M; (6) separate decode runs, 128 tokens from a retained
# context of 2048 tokens (--warm-context, an untimed prefill in the rung's own setting); (7) one LLAMA_PROFILE=1 repeat per rung. The per-token baselines
# run once at 8192 tokens and on 14 and 28 threads. Estimated at about 5 hours for 1B and 8B (the baselines dominate: 8B FP32
# per token is about 85 minutes of it): submit with a --time of at least 06:00:00 (the script header asks for 04:00:00).

# c7_best NAME RUNG: the two microbatch sizes with the best median prefill throughput at the 2048-token prompt, best first.
c7_best() {
  if dry_run; then
    echo "64 128"
    return 0
  fi
  "$PYTHON" - "$OUT" "$1" "$2" << 'PY'
import glob, json, re, statistics, sys
out, name, rung = sys.argv[1:4]
rates = {}
for path in glob.glob(f'{out}/{name}-*-teams-file-s1-t56-m*-p2048-c2304-k{rung}-[0-9]*.json'):
    match = re.search(r'-m(\d+)-p2048', path)
    rate = ((json.load(open(path)).get('throughput_tokens_per_second')) or {}).get('prefill')
    if match and isinstance(rate, (int, float)):
        rates.setdefault(int(match.group(1)), []).append(rate)
ranked = sorted(((statistics.median(v), m) for m, v in rates.items()), reverse=True)
print(' '.join(str(m) for _, m in ranked[:2]) or '64')
PY
}

# c7_gates NAME ARTIFACTS: the correctness gates of every rung on the 2048-token prompt (last position) and the 86-token one
# (all positions), at the extremes of the microbatch range.
c7_gates() {
  local name=$1 artifacts=$2 precision staged benv m n
  read -r -a benv <<< "$(backend_env teams 56 | tr '\n' ' ')"
  for precision in $PRECISIONS; do
    staged=$(stage_model_on 0 "$name" "$precision" "$artifacts")
    for m in 64 256; do
      for n in short 2048; do
        CMP_TOKENS="$OUT/C7/$name-tokens$([[ $n == short ]] || echo "-$n").txt"
        CMP_FLAGS=(--max-abs 1e-3 --batch-b "$m")
        [[ $n == short ]] || CMP_FLAGS+=(--last-only)
        if [[ $precision == fp32 ]]; then
          a_bitwise "$OUT/C7" "$name" fp32 "$staged" "c7-avx512-fp32-m$m-prompt$n" "${benv[@]}" LLAMA_KERNELS=avx512 -- "${benv[@]}" \
            LLAMA_KERNELS=avx512 LLAMA_PREFILL_GEMM=on
          local strictf=("${CMP_FLAGS[@]}")
          if [[ $n == short ]]; then CMP_FLAGS=("${BF16_GATE[@]}" --batch-b "$m"); else CMP_FLAGS=("${BF16_GATE_LAST[@]}" --batch-b "$m" --last-only); fi
          a_bitwise "$OUT/C7" "$name" fp32 "$staged" "c7-bf16-amx-m$m-prompt$n" "${benv[@]}" LLAMA_KERNELS=avx512 -- "${benv[@]}" \
            LLAMA_KERNELS=amx LLAMA_PRECISION=bf16
          CMP_FLAGS=("${strictf[@]}")
        else
          a_bitwise "$OUT/C7" "$name" int8 "$staged" "c7-avx512-q8-m$m-prompt$n" "${benv[@]}" LLAMA_KERNELS=avx512 -- "${benv[@]}" \
            LLAMA_KERNELS=avx512 LLAMA_PREFILL_GEMM=on
          # AMX-INT8 sums in another order than the per-token kernel, and W8A8 requantizes its activations: the deviation is the
          # noise floor of W8A8_GATE (job 5713471 failed this gate at the strict 1e-3, observing max 0.2-1.4 and rms 0.04-0.09).
          local strict=("${CMP_FLAGS[@]}")
          if [[ $n == short ]]; then CMP_FLAGS=("${W8A8_GATE[@]}" --batch-b "$m"); else CMP_FLAGS=("${W8A8_GATE_LAST[@]}" --batch-b "$m" --last-only); fi
          a_bitwise "$OUT/C7" "$name" int8 "$staged" "c7-amx-int8-m$m-prompt$n" "${benv[@]}" LLAMA_KERNELS=avx512 -- "${benv[@]}" \
            LLAMA_KERNELS=amx
          CMP_FLAGS=("${strict[@]}")
        fi
      done
    done
    CMP_FLAGS=(--bitwise) CMP_TOKENS=
    unstage_model_on 0 "$name" "$precision"
  done
}

phase_C7() {
  local dir="$OUT/C7" entry name artifacts hf precision staged tokenizer rung m rep n best best_m second threads prompt reps
  if [[ -z $MODELS ]]; then
    echo 'MODELS is empty: no C7 model run was made' | tee "$dir/no-models.txt"
    return 0
  fi
  need_shm
  c_prompts "$dir" 512,2048,8192
  # (1) gates, on staged-on-node-0 copies, before any benchmark.
  for entry in $MODELS; do
    IFS=: read -r name artifacts hf <<< "$entry"
    c7_gates "$name" "$artifacts"
  done
  if ! dry_run && grep -q '^FAIL compare-.*-c7-' "$OUT/phase-results.txt" 2> /dev/null; then
    echo 'a C7 correctness gate failed: no benchmark was run' | tee "$dir/gate-failed.txt"
    return 0
  fi
  for entry in $MODELS; do
    IFS=: read -r name artifacts hf <<< "$entry"
    tokenizer="$artifacts/tokenizer-v2.bin"
    for precision in $PRECISIONS; do
      staged=$(stage_model_on 2 "$name" "$precision" "$artifacts")
      # (2) microbatch sweep at 2048 tokens, repeats interleaved across rungs and sizes.
      for ((rep = 1; rep <= REPEATS; rep++)); do
        for rung in $(c_rungs "$precision"); do
          for m in $(c_rung_ms "$rung"); do
            CONTEXT=2304 GENERATION=32 c_launch "$name" "$rung" "$m" 56 "$rep" "$OUT/prompt-2048.txt" "$staged" "$tokenizer" \
              p=2048 c=2304
          done
        done
      done
      for rung in $(c_rungs "$precision"); do
        # (3) the best two microbatch sizes of this rung (a baseline has one size).
        if [[ $(c_rung_ms "$rung" | wc -w) -eq 1 ]]; then best_m=$(c_rung_ms "$rung"); else best_m=$(c7_best "$name" "$rung"); fi
        read -r best second <<< "$best_m"
        # The per-token baselines take minutes per launch at 8192 tokens and on few threads, so they run once there.
        reps=$REPEATS
        [[ $(c_rung_ms "$rung" | wc -w) -ne 1 ]] || reps=1
        # (4) the other prompt lengths at M = 1 (the baseline rungs) and the best two sizes (the others).
        for n in 86 512 8192; do
          prompt="$OUT/prompt-$n.txt"
          [[ $n != 86 ]] || prompt="$OUT/short.txt"
          for m in $best_m; do
            for ((rep = 1; rep <= (n == 8192 ? reps : REPEATS); rep++)); do
              CONTEXT=$((n > 86 ? n + 256 : 2304)) GENERATION=32 c_launch "$name" "$rung" "$m" 56 "$rep" "$prompt" "$staged" \
                "$tokenizer" "p=$n" "c=$((n > 86 ? n + 256 : 2304))"
            done
          done
        done
        # (5) fewer threads at the best size, 2048 tokens.
        for threads in 14 28; do
          for ((rep = 1; rep <= reps; rep++)); do
            CONTEXT=2304 GENERATION=32 c_launch "$name" "$rung" "$best" "$threads" "$rep" "$OUT/prompt-2048.txt" "$staged" \
              "$tokenizer" p=2048 c=2304
          done
        done
        # (6) decode: 128 generated tokens from a retained context of 2048 tokens (--warm-context, untimed, in the rung's own
        # prefill setting), then the 86-token chat prompt.
        for ((rep = 1; rep <= REPEATS; rep++)); do
          CONTEXT=2304 GENERATION=128 WARM=2048 c_launch "$name" "$rung" "$best" 56 "$rep" "$OUT/short.txt" "$staged" "$tokenizer" \
            p=2048 c=2304 x=dec
        done
        # (7) one profiled repeat: GEMM, attention, conversion, rescale and packing time.
        for n in 2048 8192; do
          CONTEXT=$((n + 256)) GENERATION=1 c_launch "$name" "$rung" "$best" 56 1 "$OUT/prompt-$n.txt" "$staged" "$tokenizer" \
            "p=$n" "c=$((n + 256))" x=prof -- LLAMA_PROFILE=1
        done
      done
      unstage_model_on 2 "$name" "$precision"
    done
  done
  if ! dry_run; then
    "$PYTHON" "$ROOT/tools/summarize_campaign.py" "$OUT" > "$dir/summary.md"
    "$PYTHON" - "$OUT" > "$dir/phase-shares.txt" << 'PY'
import glob, json, sys
from pathlib import Path
for path in sorted(glob.glob(f'{sys.argv[1]}/*-xprof-1.json')):
    stage = (json.loads(Path(path).read_text()).get('profile_seconds') or {}).get('prefill') or {}
    forward = stage.get('forward') or 0
    parts = {k: v for k, v in stage.items() if k not in ('forward', 'forward_passes') and isinstance(v, (int, float))}
    shares = ', '.join(f'{k} {100 * v / forward:.0f}%' for k, v in sorted(parts.items(), key=lambda kv: -kv[1])[:6]) if forward else 'no profile'
    print(f'{Path(path).stem}: forward {forward:.2f}s; {shares}')
PY
  fi
}

# ---- D: two-socket batched prefill and the final campaign -------------------------------------------------------------
# Two sockets run with no job-wide CPU or memory binding (each team pins itself, each region binds itself); one socket is
# bound to socket 0. Both use engine-owned HBM placement (A6's anon-hbm) from a checkpoint staged on DDR node 0, so the one
# and two socket numbers of a phase differ only in LLAMA_SOCKETS. D_THP follows A6 (default off).
EXPLICIT='explicit:0/2/0,1/3/1'
D_THP=${D_THP:-off}
D_ENV=(LLAMA_BACKEND=teams LLAMA_THREADS=56 OMP_PROC_BIND=false "LLAMA_TOPOLOGY=$EXPLICIT" LLAMA_PLACEMENT=hbm "LLAMA_THP=$D_THP")

# d_launch NAME RUNG M SOCKETS REP PROMPT STAGED TOKENIZER [LABEL_WORDS...] [-- ENV_WORDS...]: one launch of a rung on one
# or two sockets, REP selecting the repetition so a caller can interleave. A label word x=prof makes the run profiled
# (the caller adds LLAMA_PROFILE=1); its per-socket lines are then logged.
d_launch() {
  local name=$1 rung=$2 m=$3 sockets=$4 rep=$5 prompt=$6 staged=$7 tokenizer=$8 label_words=() env_words=() side=label label
  local cpu= perf=0 log=warn
  shift 8
  while (($#)); do
    if [[ $1 == -- ]]; then side=env; elif [[ $side == label ]]; then label_words+=("$1"); else env_words+=("$1"); fi
    shift
  done
  c_rung "$rung" "$m"
  [[ $sockets == 2 ]] || cpu=0
  [[ $rep != 1 ]] || perf=$PERF
  [[ " ${label_words[*]-} " != *" x=prof "* ]] || log=info
  label=$(run_label "$name" "$RUNG_PREC" teams hbm "$sockets" 56 "m=$m" "k=$rung" ${label_words[@]+"${label_words[@]}"})
  CPU_BIND=$cpu REP=$rep LOG_LEVEL=$log run_engine "$label" "$RUNG_ENGINE" "$staged" "$tokenizer" "$prompt" 1 "$perf" \
    "${D_ENV[@]}" "LLAMA_SOCKETS=$sockets" "${RUNG_ENV[@]}" ${env_words[@]+"${env_words[@]}"}
}

# d_rungs PRECISION: the batched rungs of D1 (the baselines and W8A32 stay per token; their agreement is in the gates).
d_rungs() {
  if [[ $1 == fp32 ]]; then echo avx512_fp32 amx_bf16; else echo avx512_q8 amx_int8; fi
}

# ---- D1: two-socket batched prefill ---------------------------------------------------------------------------------
# (1) Agreement of two sockets with one socket at the same rung, M = 64, prompts of 86 tokens (every position) and 2048
# (last position): FP32 1e-3 strict; W8A8 (AVX-512 VNNI and AMX-INT8) the statistical gate of the library; W8A32 strict
# (it stays per token); BF16 recorded only until C5 has fixed its gate. (2) The Hugging Face gate of FP32 with
# LLAMA_SOCKETS=2 LLAMA_PREFILL_BATCH=64. (3) Placement of each two-socket rung: shards, packed shards and KV caches
# verified on their sockets' nodes. (4) The M sweep at 2048 tokens: two sockets at M in {16, 32, 64, 128, 256}, one socket
# at {64, 256} for the same node state, REPEATS interleaved repeats, then one profiled run per two-socket point (per-socket
# GEMM, attention, exchange and wait times in the .stderr files). (5) Decode after a two-socket prefill, 128 tokens
# after the 2048-token prompt, to compare with B7. About 2 hours for 1B and 8B; a failed gate stops the benchmark.
# GATES_ONLY=1 runs only the gates and the placement checks (about 30 minutes).
d1_gates() {
  local dir=$1 name=$2 artifacts=$3 hf=$4 precision staged rung m=64 n tokens
  local a b
  for precision in $PRECISIONS; do
    staged=$(stage_model_on 0 "$name" "$precision" "$artifacts")
    for rung in $(d_rungs "$precision") $([[ $precision == fp32 ]] || echo avx512_q8_w8a32); do
      c_rung "$rung" "$m"
      a=("${D_ENV[@]}" LLAMA_SOCKETS=1 "${RUNG_ENV[@]}")
      b=("${D_ENV[@]}" LLAMA_SOCKETS=2 LLAMA_TP_CHECK=1 "${RUNG_ENV[@]}")
      for n in short 2048; do
        CMP_TOKENS="$dir/$name-tokens$([[ $n == short ]] || echo "-$n").txt"
        B_FLAGS=(--max-abs 1e-3 --batch-a "$m" --batch-b "$m")
        B_EXPECT=ok
        case $rung in
          avx512_q8 | amx_int8) B_FLAGS=("${W8A8_GATE[@]}" --batch-a "$m" --batch-b "$m") ;;
          avx512_q8_w8a32) B_FLAGS=(--max-abs 1e-3 --batch-a "$m" --batch-b "$m") ;; # batched interface, per row inside; --last-only needs it
          amx_bf16)
            B_EXPECT=any
            B_FLAGS+=(--max-abs 1e30)
            ;;
        esac
        if [[ $n != short ]]; then
          case $rung in
            avx512_q8 | amx_int8) B_FLAGS=("${W8A8_GATE_LAST[@]}" --batch-a "$m" --batch-b "$m" --last-only) ;;
            *) B_FLAGS+=(--last-only) ;;
          esac
        fi
        b_compare "$dir" "$name" "$([[ $precision == fp32 ]] && echo fp32 || echo int8)" "$staged" "d1-$rung-m$m-prompt$n" "${a[@]}" -- "${b[@]}"
      done
    done
    B_FLAGS=() B_EXPECT=ok CMP_TOKENS=
    if [[ $precision == fp32 && ${HF_GATE:-1} != 0 ]]; then
      # --batch 64: the logits come through the two-socket batched prefill (without it validate.py reads them per token and
      # only the generation steps use the batched path; K4 of 17 November 2026 showed the logits identical to the per-token gate)
      TP_VALIDATE_FLAGS=(--batch 64)
      tp_validate "$name" "$artifacts" "$hf" fp32 fp32-2s-b64 "${D_ENV[@]}" LLAMA_SOCKETS=2 \
        LLAMA_PREFILL_BATCH=64 LLAMA_PREFILL_GEMM=on && record_result "hf-$name-fp32-2s-b64" PASS || record_result "hf-$name-fp32-2s-b64" FAIL
      TP_VALIDATE_FLAGS=()
    elif [[ $precision == fp32 ]]; then
      echo "HF_GATE=0: the Hugging Face gate with LLAMA_SOCKETS=2 LLAMA_PREFILL_BATCH=64 was not run for $name" |
        tee -a "$dir/no-hf-gate.txt"
    fi
    unstage_model_on 0 "$name" "$precision"
  done
}

# d1_placement DIR NAME PRECISION STAGED TOKENIZER: one two-socket run per batched rung, its regions checked.
d1_placement() {
  local dir=$1 name=$2 precision=$3 staged=$4 tokenizer=$5 rung label opts
  for rung in $(d_rungs "$precision"); do
    CONTEXT=2304 GENERATION=16 d_launch "$name" "$rung" 64 2 1 "$OUT/short.txt" "$staged" "$tokenizer" x=place
    dry_run && continue
    c_rung "$rung" 64
    label=$(run_label "$name" "$RUNG_PREC" teams hbm 2 56 m=64 "k=$rung" x=place)
    opts="kv scratch"
    [[ $rung != amx_bf16 && $rung != amx_int8 ]] || opts="kv scratch packed"
    # shellcheck disable=SC2086
    if tp_placement "$OUT/$label-1.json" $opts > "$dir/placement-$name-$rung.txt"; then
      record_result "placement-$name-$rung" PASS
    else
      record_result "placement-$name-$rung" FAIL "$(grep ^FAIL "$dir/placement-$name-$rung.txt" | head -3)"
    fi
  done
}

phase_D1() {
  local dir="$OUT/D1" entry name artifacts hf precision staged tokenizer rung m rep reps=$((REPEATS < 3 ? REPEATS : 3))
  if [[ -z $MODELS ]]; then
    echo 'MODELS is empty: no D1 model run was made' | tee "$dir/no-models.txt"
    return 0
  fi
  need_shm
  c_prompts "$dir" 2048
  for entry in $MODELS; do
    IFS=: read -r name artifacts hf <<< "$entry"
    d1_gates "$dir" "$name" "$artifacts" "$hf"
  done
  if ! dry_run && grep -q '^FAIL \(compare-.*-d1-\|hf-.*-2s-b64\)' "$OUT/phase-results.txt" 2> /dev/null; then
    echo 'a D1 correctness gate failed: no benchmark was run' | tee "$dir/gate-failed.txt"
    return 0
  fi
  for entry in $MODELS; do
    IFS=: read -r name artifacts hf <<< "$entry"
    tokenizer="$artifacts/tokenizer-v2.bin"
    for precision in $PRECISIONS; do
      staged=$(stage_model_on 0 "$name" "$precision" "$artifacts")
      d1_placement "$dir" "$name" "$precision" "$staged" "$tokenizer"
      if [[ ${GATES_ONLY:-0} == 1 ]]; then # the gates and the placement checks, without the sweep
        echo 'GATES_ONLY=1: the gates and placement checks ran, the sweep did not' > "$dir/gates-only.txt"
        unstage_model_on 0 "$name" "$precision"
        continue
      fi
      # (4) the M sweep, interleaved: every point once per repeat.
      for ((rep = 1; rep <= reps; rep++)); do
        for rung in $(d_rungs "$precision"); do
          for m in 16 32 64 128 256; do
            CONTEXT=2304 GENERATION=32 d_launch "$name" "$rung" "$m" 2 "$rep" "$OUT/prompt-2048.txt" "$staged" "$tokenizer" p=2048
          done
          for m in 64 256; do
            CONTEXT=2304 GENERATION=32 d_launch "$name" "$rung" "$m" 1 "$rep" "$OUT/prompt-2048.txt" "$staged" "$tokenizer" p=2048
          done
        done
      done
      for rung in $(d_rungs "$precision"); do
        for m in 16 32 64 128 256; do
          CONTEXT=2304 GENERATION=1 d_launch "$name" "$rung" "$m" 2 1 "$OUT/prompt-2048.txt" "$staged" "$tokenizer" p=2048 x=prof -- \
            LLAMA_PROFILE=1
        done
        # (5) decode from a retained context of 2048 tokens built by the rung's own two-socket prefill.
        for ((rep = 1; rep <= reps; rep++)); do
          CONTEXT=2560 GENERATION=128 WARM=2048 d_launch "$name" "$rung" 64 2 "$rep" "$OUT/short.txt" "$staged" "$tokenizer" p=2048 x=dec
        done
      done
      unstage_model_on 0 "$name" "$precision"
    done
  done
  dry_run || "$PYTHON" "$ROOT/tools/summarize_campaign.py" "$OUT" > "$dir/summary.md"
}

# ---- D2: the final campaign ------------------------------------------------------------------------------------------
# The headline points of every feature from one commit and one node state: sockets {1, 2} x the rungs of both precisions x
# prompts {86, 2048, 8192} (prefill, TTFT; 32 generated tokens), and decode from retained contexts of 512 and 8192 tokens
# (128 tokens, --warm-context) for FP32, W8A8 and W8A32. REPEATS interleaved repeats, placement verified in every report,
# perf counters on the first repeat. M per rung is the C7 choice for one socket and the D1 choice for two: set
# D2_M1="avx512_fp32=64 amx_bf16=256 ..." and D2_M2 likewise (default 64 for every rung, which the evidence replaces).
# Estimated at 5-6 hours for 1B and 8B (the per-token baselines at 8192 tokens dominate): submit with --time=07:00:00 or
# split MODELS. If node time is short, run 8B at 2048 tokens first and record what was skipped.
d2_m() { # SOCKETS RUNG
  local words=$([[ $1 == 1 ]] && echo "${D2_M1:-}" || echo "${D2_M2:-}") word
  for word in $words; do
    [[ $word != "$2="* ]] || { echo "${word#*=}"; return 0; }
  done
  echo 64
}

d2_rungs() { # PRECISION
  if [[ $1 == fp32 ]]; then echo per_token avx512_fp32 amx_bf16; else echo per_token_q8 avx512_q8 amx_int8; fi
}

phase_D2() {
  local dir="$OUT/D2" entry name artifacts hf precision staged tokenizer rung m rep n sockets ctx prompt decode_rung
  if [[ -z $MODELS ]]; then
    echo 'MODELS is empty: no D2 model run was made' | tee "$dir/no-models.txt"
    return 0
  fi
  need_shm
  c_prompts "$dir" 2048,8192
  for entry in $MODELS; do
    IFS=: read -r name artifacts hf <<< "$entry"
    tokenizer="$artifacts/tokenizer-v2.bin"
    for precision in $PRECISIONS; do
      staged=$(stage_model_on 0 "$name" "$precision" "$artifacts")
      for ((rep = 1; rep <= REPEATS; rep++)); do
        for n in 86 2048 8192; do
          prompt="$OUT/prompt-$n.txt"
          [[ $n != 86 ]] || prompt="$OUT/short.txt"
          for sockets in 1 2; do
            for rung in $(d2_rungs "$precision"); do
              m=1
              [[ $rung == per_token || $rung == per_token_q8 ]] || m=$(d2_m "$sockets" "$rung")
              ctx=$((n > 86 ? n + 256 : 2304))
              CONTEXT=$ctx GENERATION=32 d_launch "$name" "$rung" "$m" "$sockets" "$rep" "$prompt" "$staged" "$tokenizer" "p=$n"
            done
          done
        done
        # Decode from retained contexts, in the rung that builds the context fastest (the decode kernel is the same).
        for ctx in 512 8192; do
          for sockets in 1 2; do
            if [[ $precision == fp32 ]]; then
              CONTEXT=$((ctx + 256)) GENERATION=128 WARM=$ctx d_launch "$name" avx512_fp32 64 "$sockets" "$rep" "$OUT/short.txt" \
                "$staged" "$tokenizer" "p=$ctx" x=dec
            else
              CONTEXT=$((ctx + 256)) GENERATION=128 WARM=$ctx d_launch "$name" avx512_q8 64 "$sockets" "$rep" "$OUT/short.txt" \
                "$staged" "$tokenizer" "p=$ctx" x=dec
              CONTEXT=$((ctx + 256)) GENERATION=128 WARM=$ctx d_launch "$name" avx512_q8_w8a32 1 "$sockets" "$rep" "$OUT/short.txt" \
                "$staged" "$tokenizer" "p=$ctx" x=dec
            fi
          done
        done
      done
      unstage_model_on 0 "$name" "$precision"
    done
  done
  dry_run || "$PYTHON" "$ROOT/tools/summarize_campaign.py" "$OUT" > "$dir/summary.md"
}

# ---- D3: where does the W8A8 deviation of two-socket AMX-INT8 come from? (diagnostic, not a plan phase) ---------------------
# D1's gate compares two-socket AMX-INT8 with one-socket AMX-INT8 on the 86-token prompt; on 8B it showed one position at
# max |dlogit| 4.12 (budget 2.0) and rms 0.136, against 0.58 and 0.063 for AVX-512 VNNI. This records, without gating, the
# deviation of every pair that isolates one factor: the kernel (one socket AMX against one socket AVX-512), the split (two
# sockets against one at the same kernel), and the microbatch size (M = 64 against 16 on two sockets). All use W8A8 and the
# statistical summary; read d3-summary.txt.
d3_case() { # LABEL SOCKETS_A KERNEL_A M_A SOCKETS_B KERNEL_B M_B
  local label=$1 a b kernel
  local env_for=() 
  for side in a b; do
    local sockets kernel m
    if [[ $side == a ]]; then sockets=$2 kernel=$3; else sockets=$5 kernel=$6; fi
    local words=("${D_ENV[@]}" "LLAMA_SOCKETS=$sockets" "LLAMA_KERNELS=$kernel")
    [[ $kernel != avx512 ]] || words+=(LLAMA_PREFILL_GEMM=on)
    [[ $sockets != 2 ]] || words+=(LLAMA_TP_CHECK=1)
    if [[ $side == a ]]; then a=("${words[@]}"); else b=("${words[@]}"); fi
  done
  B_EXPECT=any B_FLAGS=(--max-abs 1e30 --batch-a "$4" --batch-b "$7")
  b_compare "$D3_DIR" "$D3_NAME" int8 "$D3_STAGED" "d3-$label" "${a[@]}" -- "${b[@]}"
}

phase_D3() {
  local dir="$OUT/D3" entry name artifacts hf n
  if [[ -z $MODELS ]]; then
    echo 'MODELS is empty: no D3 run was made' | tee "$dir/no-models.txt"
    return 0
  fi
  need_shm
  c_prompts "$dir" 2048
  D3_DIR=$dir
  for entry in $MODELS; do
    IFS=: read -r name artifacts hf <<< "$entry"
    D3_NAME=$name
    D3_STAGED=$(stage_model_on 0 "$name" int8 "$artifacts")
    for n in short 2048; do
      CMP_TOKENS="$dir/$name-tokens$([[ $n == short ]] || echo "-$n").txt"
      suffix=$([[ $n == short ]] || echo -prompt2048)
      d3_case "1s-amx-vs-1s-avx512$suffix" 1 amx 64 1 avx512 64
      d3_case "2s-amx-vs-1s-avx512$suffix" 2 amx 64 1 avx512 64
      d3_case "2s-amx-vs-2s-avx512$suffix" 2 amx 64 2 avx512 64
      d3_case "2s-amx-vs-1s-amx$suffix" 2 amx 64 1 amx 64
      d3_case "2s-amx-m64-vs-m16$suffix" 2 amx 64 2 amx 16
    done
    B_FLAGS=() B_EXPECT=ok CMP_TOKENS=
    unstage_model_on 0 "$name" int8
  done
  dry_run || python3 - "$dir" > "$OUT/d3-summary.txt" << 'PY'
import glob, json, os, sys
print(f'{"comparison":58} {"max_abs":>8} {"rms":>7} {"argmax":>7} {"bitwise":>8}')
for path in sorted(glob.glob(os.path.join(sys.argv[1], 'compare-*-d3-*.json'))):
    d = json.load(open(path))
    s = d['summary']
    print(f'{os.path.basename(path)[8:-5]:58} {s["max_abs"]:8.3f} {s["rms"]:7.4f} {s["argmax_agreement"]:7.3f} {str(s["bitwise"]):>8}')
PY
}

# ---- E0: measure attention before changing it ---------------------------------------------------------------------------
# tests/attention_bench.c on socket 0 with K/V in a region on its HBM node (2): the prefill matrix (H1, H2, H4), the row-block
# sweep (H3), the half-KV-head shapes (H6), decode (H5), hardware counters, one profiled engine run that shows whether the
# bench represents the engine, and the bitwise gate of the LLAMA_ATTN_ROW_BLOCK knob. Everything but the last two needs no
# model. Results land in E0/ as jsonl; tools/summarize_attention_bench.py builds E0/summary.md and E0/counters.csv.
E0_CORE_EVENTS='cycles,instructions,l1d.replacement,l2_rqsts.miss,mem_load_retired.l3_miss'
E0_PERF=0

# E0_DIR (default $OUT/E0) is where the three helpers below write; phase_E2 points it at E2/. E0_ATTN, when set, adds
# LLAMA_ATTN=$E0_ATTN to the counted launch of e0_counted (e0_bench takes its environment words from the caller).
# e0_bench NAME THREADS [ENV_WORDS...] -- BENCH_ARGS...: one attention-bench launch, output in E0_DIR/NAME.out.
e0_bench() {
  local name=$1 threads=$2 env_words=() args=() side=env
  shift 2
  while (($#)); do
    if [[ $1 == -- ]]; then side=args; elif [[ $side == env ]]; then env_words+=("$1"); else args+=("$1"); fi
    shift
  done
  expect "$name" ok "${E0_DIR:-$OUT/E0}" -- numactl --cpunodebind=0 env LLAMA_BACKEND=teams "LLAMA_THREADS=$threads" OMP_PROC_BIND=false \
    LLAMA_KERNELS=avx512 ${env_words[@]+"${env_words[@]}"} "$ROOT/build/release/attention-bench" --node 2 "${args[@]}"
}

# e0_collect FILE PATTERN: the jsonl of every E0/PATTERN.out, concatenated.
e0_collect() {
  dry_run && return 0
  local dir=${E0_DIR:-$OUT/E0}
  if compgen -G "$dir/$2" > /dev/null; then cat "$dir"/$2 > "$dir/$1"; else : > "$dir/$1"; fi
}

# e0_counted NAME THREADS BENCH_ARGS...: one bench launch counted by perf, system-wide per socket, the way run_engine counts an
# engine run (the uncore events too when the common check found them). Counts include the start-up of the bench; the cells use
# enough repetitions that the passes dominate. Never fails the phase.
e0_counted() {
  local name=$1 threads=$2 events=$E0_CORE_EVENTS pid
  shift 2
  [[ $PERF != 1 ]] || events+=",$EVENTS"
  if dry_run; then
    printf 'COUNTERS %s threads=%s attn=%s events=%s: %s\n' "$name" "$threads" "${E0_ATTN:-default}" "$events" "$*" | tee -a "$OUT/dry-run.txt"
    return 0
  fi
  [[ $E0_PERF == 1 ]] || return 0
  local dir=${E0_DIR:-$OUT/E0}
  numactl --cpunodebind=0 env LLAMA_BACKEND=teams "LLAMA_THREADS=$threads" OMP_PROC_BIND=false LLAMA_KERNELS=avx512 \
    ${E0_ATTN:+"LLAMA_ATTN=$E0_ATTN"} "$ROOT/build/release/attention-bench" --node 2 "$@" > "$dir/counters-$name.jsonl" \
    2> "$dir/counters-$name.err" &
  pid=$!
  sudo -n /usr/bin/perf stat -a --per-socket -x , -e "$events" -- tail -s 0.1 --pid="$pid" -f /dev/null \
    2> "$dir/counters-$name.perf.csv" || true
  wait "$pid" || record_result "counters-$name" FAIL 'attention-bench failed under perf'
}

phase_E0() {
  local dir="$OUT/E0" shape threads rb kv entry name artifacts hf tokenizer staged benv profile
  if ! dry_run && sudo -n /usr/bin/perf stat -a --per-socket -x , -e "$E0_CORE_EVENTS" -- true 2> /dev/null; then E0_PERF=1; fi
  dry_run || printf 'perf_core=%s\n' "$E0_PERF" >> "$OUT/environment.txt"
  # (1) the prefill matrix: the whole prompt of one layer, microbatches as run.c cuts them.
  for shape in 8b 1b; do
    for threads in 14 28 56; do
      e0_bench "prefill-$shape-t$threads" "$threads" -- --mode prefill --shape "$shape" --prompt 512,2048,8192 --m 16,64,256 \
        --reps "$REPEATS"
    done
  done
  e0_collect bench-prefill.jsonl 'prefill-*.out'
  # (2) H3: rows per task, same results bit for bit.
  for rb in 1 2 4 8 16; do
    e0_bench "rowblock-$rb" 56 "LLAMA_ATTN_ROW_BLOCK=$rb" -- --mode prefill --shape 8b --prompt 8192 --m 256 --reps "$REPEATS"
  done
  e0_collect rowblock.jsonl 'rowblock-*.out'
  # (3) H6: one socket's half of the model (4 of 8 KV heads) at the sizes of the two-socket runs, 56 threads.
  for shape in 8b 1b; do
    e0_bench "h6-prefill-$shape-kv4" 56 -- --mode prefill --shape "$shape" --prompt 2048,8192 --m 64,256 --reps "$REPEATS" \
      --kv-heads-local 4
  done
  e0_collect bench-prefill-kv4.jsonl 'h6-prefill-*.out'
  # (4) H5 and H6: decode attention at the last position of a full context, all KV heads and half of them.
  for shape in 8b 1b; do
    for kv in 8 4; do
      e0_bench "decode-$shape-kv$kv" 56 -- --mode decode --shape "$shape" --context 8192,32768 --reps 200 --kv-heads-local "$kv"
    done
  done
  e0_collect bench-decode.jsonl 'decode-*.out'
  # (5) counters on the 8b/8192/M256 prefill cell and the 1b/32768 decode cell.
  e0_counted prefill-8b-8192-m256 56 --mode prefill --shape 8b --prompt 8192 --m 256 --reps "$REPEATS"
  e0_counted decode-1b-32768 56 --mode decode --shape 1b --context 32768 --reps 1000
  # (6) engine runs: the bitwise gate of the knob, and one profiled run for the cross-check.
  if [[ -z $MODELS ]]; then
    echo 'MODELS is empty: no E0 engine run was made' | tee "$dir/no-models.txt"
  else
    need_shm
    c_prompts "$dir" 2048,8192
    for entry in $MODELS; do
      IFS=: read -r name artifacts hf <<< "$entry"
      # the 8B model only (E0_MODEL=1b for a rehearsal on another model)
      [[ $(printf '%s' "$name" | tr '[:upper:]' '[:lower:]') == *"${E0_MODEL:-8b}"* ]] || continue
      tokenizer="$artifacts/tokenizer-v2.bin"
      staged=$(stage_model_on 0 "$name" fp32 "$artifacts")
      read -r -a benv <<< "$(backend_env teams 56 | tr '\n' ' ')"
      # Both sides batched (M 64), last row of the 2048-token prompt, the knob at 4 against 16: bit-identical logits.
      CMP_FLAGS=(--bitwise --batch-a 64 --batch-b 64 --last-only)
      CMP_TOKENS="$dir/$name-tokens-2048.txt"
      a_bitwise "$dir" "$name" fp32 "$staged" e0-rowblock-4-vs-16-m64-prompt2048 "${benv[@]}" LLAMA_KERNELS=avx512 \
        LLAMA_PREFILL_GEMM=on LLAMA_ATTN_ROW_BLOCK=4 -- "${benv[@]}" LLAMA_KERNELS=avx512 LLAMA_PREFILL_GEMM=on \
        LLAMA_ATTN_ROW_BLOCK=16
      CMP_FLAGS=(--bitwise) CMP_TOKENS=
      unstage_model_on 0 "$name" fp32
      # The profiled engine run of C7 (8B AMX-BF16, one socket, 56 threads, checkpoint on HBM node 2): the attention seconds
      # of its report are what the bench's ms per layer times 32 must match.
      staged=$(stage_model_on 2 "$name" fp32 "$artifacts")
      CONTEXT=8448 GENERATION=1 c_launch "$name" amx_bf16 256 56 1 "$OUT/prompt-8192.txt" "$staged" "$tokenizer" p=8192 \
        c=8448 x=prof -- LLAMA_PROFILE=1
      unstage_model_on 2 "$name" fp32
      if ! dry_run; then
        for profile in "$OUT/$name"-bf16-*-xprof-1.json; do
          [[ ! -e $profile ]] || cp "$profile" "$dir/engine-profile.json"
        done
      fi
    done
  fi
  dry_run || "$PYTHON" "$ROOT/tools/summarize_attention_bench.py" "$dir" --counters-csv "$dir/counters.csv" > "$dir/summary.md"
}

# ---- E1: attention dispatch check (default unchanged), item 2b and the FP32 determinism data -------------------------------
# STAGE=check PHASE=E1; `make test` already ran in the common check stage. Three parts, about an hour for 1B and 8B:
# (1) the bitwise gate: with BASELINE_ROOT (a copy of an earlier commit, e.g. the D2 build, whose build/release/probe-fp32 an
#     earlier job built) the logits of that build and of this tree with LLAMA_ATTN=split must be bit-identical on the 8B model
#     (E1_MODEL, default 8b) at the 2048-token prompt, last row: FP32 M = 64 batched, FP32 per token, two sockets M = 64. A shim
#     written into E1/ runs $PROBE_DIR/probe-fp32, so one compare_engines invocation can compare two binaries. Without
#     BASELINE_ROOT this part is skipped and recorded (not a failure).
# (2) item 2b, recorded: the 1B model (E1_SMALL_MODEL, default 1b), W8A8 and W8A32 from a retained context of 8192 tokens,
#     one socket against two (LLAMA_TP_CHECK=1), and one socket against one socket as the run-to-run control, REPEATS times,
#     128 steps, greedy: E1/gen-<case>-<rep>.json. (3) FP32 determinism-gate data on every model: the 2048-token and the short
#     prompt, one socket per token against two sockets and against batched M = 64, one repeat. tools/summarize_generation.py
#     reads the gen-*.json files and writes E1/summary.md. None of the generation comparisons gates the job unless a run fails.
e1_is() { # NAME PATTERN: does the lower-cased model name contain PATTERN
  [[ $(printf '%s' "$1" | tr '[:upper:]' '[:lower:]') == *"$2"* ]]
}

# e1_shim DIR PRECISION: writes DIR/probe-shim-PRECISION.sh, which runs $PROBE_DIR/probe-PRECISION, and prints its path.
e1_shim() {
  local shim="$1/probe-shim-$2.sh"
  if ! dry_run; then
    printf '#!/usr/bin/env bash\nexec "$PROBE_DIR/probe-%s" "$@"\n' "$2" > "$shim"
    chmod +x "$shim"
  fi
  printf '%s\n' "$shim"
}

# e1_bitwise DIR NAME STAGED: the three bitwise comparisons of the baseline probe against this build (needs c_prompts 2048).
e1_bitwise() {
  local dir=$1 name=$2 staged=$3 shim benv base="$BASELINE_ROOT/build/release" here="$ROOT/build/release" two
  shim=$(e1_shim "$dir" fp32)
  read -r -a benv <<< "$(backend_env teams 56 | tr '\n' ' ')"
  CMP_PROBE=$shim CMP_TOKENS="$dir/$name-tokens-2048.txt"
  CMP_FLAGS=(--bitwise --batch-a 64 --batch-b 64 --last-only)
  a_bitwise "$dir" "$name" fp32 "$staged" e1-m64-prompt2048 "${benv[@]}" LLAMA_KERNELS=avx512 LLAMA_PREFILL_GEMM=on \
    "PROBE_DIR=$base" -- "${benv[@]}" LLAMA_KERNELS=avx512 LLAMA_PREFILL_GEMM=on LLAMA_ATTN=split "PROBE_DIR=$here"
  # per token: the batched path at M = 1 returning the last row only (the per-token reference, without 2048 rows of logits)
  CMP_FLAGS=(--bitwise --batch-a 1 --batch-b 1 --last-only)
  a_bitwise "$dir" "$name" fp32 "$staged" e1-pertoken-prompt2048 "${benv[@]}" LLAMA_KERNELS=avx512 "PROBE_DIR=$base" -- \
    "${benv[@]}" LLAMA_KERNELS=avx512 LLAMA_ATTN=split "PROBE_DIR=$here"
  # two sockets, as D2 runs them (teams, 56 threads each), exchange state checked
  two=("${D_ENV[@]}" LLAMA_SOCKETS=2 LLAMA_TP_CHECK=1 LLAMA_KERNELS=avx512 LLAMA_PREFILL_GEMM=on)
  B_FLAGS=(--bitwise --batch-a 64 --batch-b 64 --last-only) B_EXPECT=ok
  b_compare "$dir" "$name" fp32 "$staged" e1-s2-m64-prompt2048 "${two[@]}" "PROBE_DIR=$base" -- "${two[@]}" LLAMA_ATTN=split \
    "PROBE_DIR=$here"
  B_FLAGS=() CMP_FLAGS=(--bitwise) CMP_TOKENS= CMP_PROBE=
}

# e1_side SOCKETS RUNG M: sets E1_SIDE to the environment words of one configuration (D2's placement words, the rung's kernels).
e1_side() {
  c_rung "$2" "$3"
  E1_SIDE=("${D_ENV[@]}" "LLAMA_SOCKETS=$1")
  [[ $1 != 2 ]] || E1_SIDE+=(LLAMA_TP_CHECK=1)
  E1_SIDE+=("${RUNG_ENV[@]}")
}

# e1_case CASE REP SOCKETS_A RUNG_A M_A SOCKETS_B RUNG_B M_B: one recorded generation comparison with tools/compare_generation.py
# (A is the reference, and its probe gives the top-2 margin at the divergence). The run is described by E1_ENGINE, E1_PROBE,
# E1_STAGED, E1_TOKENIZER, E1_PROMPT, E1_CTX (context length), E1_WARM (retained context, 0 for none) and E1_PBATCH.
e1_case() {
  local case=$1 rep=$2 a b warm=() args=() word
  e1_side "$3" "$4" "$5"
  a=("${E1_SIDE[@]}")
  e1_side "$6" "$7" "$8"
  b=("${E1_SIDE[@]}")
  [[ $E1_WARM == 0 ]] || warm=(--warm-context "$E1_WARM")
  for word in "${a[@]}"; do args+=(--env-a "$word"); done
  for word in "${b[@]}"; do args+=(--env-b "$word"); done
  expect "gen-$case-$rep" ok "$OUT/E1" -- "$PYTHON" "$ROOT/tools/compare_generation.py" --engine "$ROOT/build/release/$E1_ENGINE" \
    --checkpoint "$E1_STAGED" --tokenizer "$E1_TOKENIZER" --prompt-file "$E1_PROMPT" --steps 128 ${warm[@]+"${warm[@]}"} \
    --engine-args "--context-length $E1_CTX --chat-template" "${args[@]}" \
    --probe "$ROOT/build/release/$E1_PROBE" --probe-batch "$E1_PBATCH" --output "$OUT/E1/gen-$case-$rep.json"
}

phase_E1() {
  local dir="$OUT/E1" entry name artifacts hf n rep bitwise_done=0
  if [[ -z $MODELS ]]; then
    echo 'MODELS is empty: no E1 run was made' | tee "$dir/no-models.txt"
    return 0
  fi
  need_shm
  c_prompts "$dir" 2048
  if [[ -z ${BASELINE_ROOT:-} ]]; then
    echo 'baseline not given: the bitwise default check against the baseline build was skipped' | tee "$dir/baseline-not-given.txt"
  elif ! dry_run && [[ ! -x $BASELINE_ROOT/build/release/probe-fp32 ]]; then
    record_result e1-baseline-probe FAIL 'BASELINE_ROOT has no build/release/probe-fp32'
    BASELINE_ROOT=
  fi
  for entry in $MODELS; do
    IFS=: read -r name artifacts hf <<< "$entry"
    E1_TOKENIZER="$artifacts/tokenizer-v2.bin"
    # FP32 on every model: the bitwise gate (8B only) and the determinism-gate data.
    E1_STAGED=$(stage_model_on 0 "$name" fp32 "$artifacts")
    if [[ -n ${BASELINE_ROOT:-} ]] && e1_is "$name" "${E1_MODEL:-8b}"; then
      e1_bitwise "$dir" "$name" "$E1_STAGED"
      bitwise_done=1
    fi
    E1_ENGINE=run E1_PROBE=probe-fp32 E1_WARM=0 E1_PBATCH=1
    for n in short 2048; do
      if [[ $n == short ]]; then E1_PROMPT="$OUT/short.txt" E1_CTX=512; else E1_PROMPT="$OUT/prompt-2048.txt" E1_CTX=2304; fi
      e1_case "fp32-$name-$n-s1-vs-s2" 1 1 per_token 1 2 per_token 1
      e1_case "fp32-$name-$n-pertoken-vs-m64" 1 1 per_token 1 1 avx512_fp32 64
    done
    unstage_model_on 0 "$name" fp32
    # item 2b on the 1B model: W8A8 and W8A32 from a retained context of 8192 tokens.
    if e1_is "$name" "${E1_SMALL_MODEL:-1b}"; then
      E1_STAGED=$(stage_model_on 0 "$name" int8 "$artifacts")
      E1_ENGINE=runq E1_PROBE=probe-int8 E1_WARM=8192 E1_CTX=8448 E1_PBATCH=64 E1_PROMPT="$OUT/short.txt"
      for ((rep = 1; rep <= REPEATS; rep++)); do
        e1_case "item2b-$name-w8a8-s1-vs-s2" "$rep" 1 avx512_q8 "$(d2_m 1 avx512_q8)" 2 avx512_q8 "$(d2_m 2 avx512_q8)"
        e1_case "item2b-$name-w8a32-s1-vs-s2" "$rep" 1 avx512_q8_w8a32 1 2 avx512_q8_w8a32 1
        e1_case "item2b-$name-w8a8-s1-vs-s1" "$rep" 1 avx512_q8 "$(d2_m 1 avx512_q8)" 1 avx512_q8 "$(d2_m 1 avx512_q8)"
      done
      unstage_model_on 0 "$name" int8
    fi
  done
  if [[ $bitwise_done == 0 && -n ${BASELINE_ROOT:-} ]]; then
    echo "no MODELS entry matches E1_MODEL=${E1_MODEL:-8b}: the bitwise default check was skipped" | tee "$dir/no-bitwise-model.txt"
  fi
  dry_run || "$PYTHON" "$ROOT/tools/summarize_generation.py" "$dir" > "$dir/summary.md"
}

# ---- E2: FP32 register-tiled fused attention (LLAMA_ATTN=tiled) -----------------------------------------------------------
# Both stages need the models. STAGE=check is one job of about 1.5-2.5 hours (the one-socket per-token references at 8192 tokens
# dominate: submit with --time=03:00:00), STAGE=bench one of about 2 hours. The check job's `make test` is the common job start
# (check_stage); with SKIP_TEST=1 it is not repeated and the phase records that. The bench job follows the check with SKIP_TEST=1
# (and SKIP_PRELUDE_HF=1 if the Hugging Face prelude is not wanted again); GATES_ONLY is not used by this phase.
# check (E2/compare-*.json, summary.md from tools/summarize_e2.py): every comparison is a compare_engines.py run, A first.
#   FP32 gate (max |dlogit| <= 1e-3, argmax 100%), LLAMA_ATTN=tiled with M = 64 against the one-socket per-token reference, on every
#     model at 86 tokens (all positions), 2048 and 8192 (last row); at 8192 once more with LLAMA_ATTN_TILED_QUERIES=4;
#   AMX-BF16 (BF16_GATE): tiled against split, M = 64, 86 tokens (all positions) and 2048 (last row);
#   two real sockets (8B only, E2_MODEL): tiled M = 64 with LLAMA_TP_CHECK=1 against one socket per token, 2048 tokens, FP32 gate;
#   W8A8 (W8A8_GATE, probe-int8, int8 checkpoint): tiled against split, M = 64, 2048 tokens (last row);
#   `split` stays bitwise: with BASELINE_ROOT, E1's three bitwise comparisons (e1_bitwise) on the 8B model.
# bench (E2/): bench-split.jsonl and bench-tiled.jsonl (the E0 prefill matrix and its half-KV-head cells, 56 threads included
#   at 8B/8192/M256), bench-tiled-sweep.jsonl (queries 4, 8, 16 x order longest, natural on 8B 8192/M256 and 2048/M64),
#   counters-{split,tiled}-prefill-8b-8192-m256.*, engine TTFT of 8B and 1B (AVX-512 FP32 and AMX-BF16, one socket, 2048 and 8192
#   tokens, split against tiled, REPEATS repeats plus one LLAMA_PROFILE=1 repeat; reports $OUT/*-xattn{split,tiled}-N.json and
#   -xprof{split,tiled}-1.json, summarised in E2/ttft.csv), one decode cell with the default attention (8B FP32, 8192 retained
#   context, x=dec), E2/bench-summary.md. M of the engine runs is D2_M1's (default 64 per rung): set
#   D2_M1="avx512_fp32=64 amx_bf16=256" at submission.
E2_FP32_GATE=(--max-abs 1e-3) # argmax agreement stays at the compare_engines default of 100%

# e2_fp32_case DIR NAME STAGED N LABEL [ENV_WORDS...]: LLAMA_ATTN=tiled (and ENV_WORDS) at M = 64 against the one-socket per-token
# reference (M = 1 batched interface, the e1 per-token convention) at N tokens: every position at 86, the last row above.
e2_fp32_case() {
  local dir=$1 name=$2 staged=$3 n=$4 label=$5 benv
  shift 5
  read -r -a benv <<< "$(backend_env teams 56 | tr '\n' ' ')"
  CMP_FLAGS=("${E2_FP32_GATE[@]}" --batch-a 1 --batch-b 64)
  CMP_TOKENS=
  if [[ $n != 86 ]]; then
    CMP_FLAGS+=(--last-only)
    CMP_TOKENS="$dir/$name-tokens-$n.txt"
  fi
  a_bitwise "$dir" "$name" fp32 "$staged" "e2-$label-prompt$n" "${benv[@]}" LLAMA_KERNELS=avx512 -- "${benv[@]}" \
    LLAMA_KERNELS=avx512 LLAMA_PREFILL_GEMM=on LLAMA_ATTN=tiled "$@"
  CMP_FLAGS=(--bitwise) CMP_TOKENS=
}

# e2_pair_case DIR NAME PRECISION STAGED N LABEL RUNG GATE...: tiled against split at the rung (M = 64), both sides batched.
e2_pair_case() {
  local dir=$1 name=$2 precision=$3 staged=$4 n=$5 label=$6 rung=$7 benv
  shift 7
  read -r -a benv <<< "$(backend_env teams 56 | tr '\n' ' ')"
  c_rung "$rung" 64
  CMP_FLAGS=("$@" --batch-a 64 --batch-b 64)
  CMP_TOKENS=
  if [[ $n != 86 ]]; then
    CMP_FLAGS+=(--last-only)
    CMP_TOKENS="$dir/$name-tokens-$n.txt"
  fi
  a_bitwise "$dir" "$name" "$precision" "$staged" "e2-$label-prompt$n" "${benv[@]}" "${RUNG_ENV[@]}" LLAMA_ATTN=split -- \
    "${benv[@]}" "${RUNG_ENV[@]}" LLAMA_ATTN=tiled
  CMP_FLAGS=(--bitwise) CMP_TOKENS=
}

e2_check() {
  local dir=$1 entry name artifacts hf n staged two bitwise_done=0
  if ! dry_run && ! grep -q '^PASS make-test' "$OUT/phase-results.txt" 2> /dev/null; then
    echo 'make test did not pass in this job (SKIP_TEST=1, or it failed): the check relies on an earlier passing run' |
      tee "$dir/make-test-not-verified.txt"
  fi
  if [[ -z ${BASELINE_ROOT:-} ]]; then
    echo 'baseline not given: the bitwise split check against the baseline build was skipped' | tee "$dir/baseline-not-given.txt"
  elif ! dry_run && [[ ! -x $BASELINE_ROOT/build/release/probe-fp32 ]]; then
    record_result e2-baseline-probe FAIL 'BASELINE_ROOT has no build/release/probe-fp32'
    BASELINE_ROOT=
  fi
  c_prompts "$dir" 2048,8192
  for entry in $MODELS; do
    IFS=: read -r name artifacts hf <<< "$entry"
    staged=$(stage_model_on 0 "$name" fp32 "$artifacts")
    # (1) real-model FP32, tiled M = 64 against one-socket per token; the block-size knob at 8192.
    for n in 86 2048 8192; do
      e2_fp32_case "$dir" "$name" "$staged" "$n" tiled-m64
    done
    e2_fp32_case "$dir" "$name" "$staged" 8192 tiled-q4-m64 LLAMA_ATTN_TILED_QUERIES=4
    # (2) AMX-BF16, tiled against split.
    e2_pair_case "$dir" "$name" fp32 "$staged" 86 bf16-tiled-vs-split amx_bf16 "${BF16_GATE[@]}"
    e2_pair_case "$dir" "$name" fp32 "$staged" 2048 bf16-tiled-vs-split amx_bf16 "${BF16_GATE_LAST[@]}"
    if e1_is "$name" "${E2_MODEL:-8b}"; then
      # (3) two real sockets against one socket per token, 2048 tokens, exchange state checked.
      two=("${D_ENV[@]}" LLAMA_SOCKETS=2 LLAMA_TP_CHECK=1 LLAMA_KERNELS=avx512 LLAMA_PREFILL_GEMM=on LLAMA_ATTN=tiled)
      CMP_TOKENS="$dir/$name-tokens-2048.txt"
      B_FLAGS=("${E2_FP32_GATE[@]}" --batch-a 1 --batch-b 64 --last-only) B_EXPECT=ok
      b_compare "$dir" "$name" fp32 "$staged" e2-s2-tiled-m64-prompt2048 "${D_ENV[@]}" LLAMA_SOCKETS=1 LLAMA_KERNELS=avx512 -- \
        "${two[@]}"
      B_FLAGS=() CMP_TOKENS=
      # (4) split stays bitwise against the baseline build (E1's helper).
      if [[ -n ${BASELINE_ROOT:-} ]]; then
        e1_bitwise "$dir" "$name" "$staged"
        bitwise_done=1
      fi
    fi
    unstage_model_on 0 "$name" fp32
    # (5) W8A8, tiled against split.
    staged=$(stage_model_on 0 "$name" int8 "$artifacts")
    e2_pair_case "$dir" "$name" int8 "$staged" 2048 w8a8-tiled-vs-split avx512_q8 "${W8A8_GATE_LAST[@]}"
    unstage_model_on 0 "$name" int8
  done
  if [[ $bitwise_done == 0 && -n ${BASELINE_ROOT:-} ]]; then
    echo "no MODELS entry matches E2_MODEL=${E2_MODEL:-8b}: the bitwise split check was skipped" | tee "$dir/no-bitwise-model.txt"
  fi
  # the exit status of the summary is the failure count; the failures are already in phase-results.txt
  dry_run || "$PYTHON" "$ROOT/tools/summarize_e2.py" checks "$dir" > "$dir/summary.md" || true
}

e2_bench() {
  local dir=$1 attn shape threads q order cell n m rung rep entry name artifacts hf staged tokenizer
  E0_DIR=$dir
  if ! dry_run && sudo -n /usr/bin/perf stat -a --per-socket -x , -e "$E0_CORE_EVENTS" -- true 2> /dev/null; then E0_PERF=1; fi
  dry_run || printf 'perf_core=%s\n' "$E0_PERF" >> "$OUT/environment.txt"
  # (1) the E0 prefill matrix and the half-KV-head cells, per attention kernel (decode is untouched, so not repeated).
  for attn in split tiled; do
    for shape in 8b 1b; do
      for threads in 14 28 56; do
        e0_bench "bench-$attn-prefill-$shape-t$threads" "$threads" "LLAMA_ATTN=$attn" -- --mode prefill --shape "$shape" \
          --prompt 512,2048,8192 --m 16,64,256 --reps "$REPEATS"
      done
      e0_bench "bench-$attn-h6-$shape-kv4" 56 "LLAMA_ATTN=$attn" -- --mode prefill --shape "$shape" --prompt 2048,8192 \
        --m 64,256 --reps "$REPEATS" --kv-heads-local 4
    done
    e0_collect "bench-$attn.jsonl" "bench-$attn-*.out"
  done
  # (2) the tiled knobs on the two cells that matter; every collected line carries its knobs.
  for cell in 8192:256 2048:64; do
    n=${cell%:*} m=${cell#*:}
    for q in 4 8 16; do
      for order in longest natural; do
        e0_bench "sweep-q$q-$order-p$n-m$m" 56 LLAMA_ATTN=tiled "LLAMA_ATTN_TILED_QUERIES=$q" "LLAMA_ATTN_TILED_ORDER=$order" -- \
          --mode prefill --shape 8b --prompt "$n" --m "$m" --reps "$REPEATS"
      done
    done
  done
  if ! dry_run; then
    : > "$dir/bench-tiled-sweep.jsonl"
    for cell in "$dir"/sweep-q*-*-p*-m*.out; do
      [[ -e $cell ]] || continue
      q=${cell##*/sweep-q} q=${q%%-*}
      order=${cell#*/sweep-q$q-} order=${order%%-p*}
      sed "s/^{/{\"tiled_queries\": $q, \"tiled_order\": \"$order\", /" "$cell" >> "$dir/bench-tiled-sweep.jsonl"
    done
  fi
  # (3) counters on the 8b/8192/M256 prefill cell for both kernels.
  for attn in split tiled; do
    E0_ATTN=$attn e0_counted "$attn-prefill-8b-8192-m256" 56 --mode prefill --shape 8b --prompt 8192 --m 256 --reps "$REPEATS"
  done
  unset E0_DIR
  # (4) engine TTFT, one socket, checkpoint on HBM node 2: the rungs of C7 at 2048 and 8192 tokens, split against tiled.
  if [[ -z $MODELS ]]; then
    echo 'MODELS is empty: no E2 engine run was made' | tee "$dir/no-models.txt"
  else
    need_shm
    c_prompts "$dir" 2048,8192
    for entry in $MODELS; do
      IFS=: read -r name artifacts hf <<< "$entry"
      tokenizer="$artifacts/tokenizer-v2.bin"
      staged=$(stage_model_on 2 "$name" fp32 "$artifacts")
      for rung in avx512_fp32 amx_bf16; do
        m=$(d2_m 1 "$rung")
        for ((rep = 1; rep <= REPEATS; rep++)); do
          for n in 2048 8192; do
            for attn in split tiled; do
              CONTEXT=$((n + 256)) GENERATION=32 c_launch "$name" "$rung" "$m" 56 "$rep" "$OUT/prompt-$n.txt" "$staged" \
                "$tokenizer" "p=$n" "c=$((n + 256))" "x=attn$attn" -- "LLAMA_ATTN=$attn"
            done
          done
        done
        for n in 2048 8192; do
          for attn in split tiled; do
            CONTEXT=$((n + 256)) GENERATION=1 c_launch "$name" "$rung" "$m" 56 1 "$OUT/prompt-$n.txt" "$staged" "$tokenizer" \
              "p=$n" "c=$((n + 256))" "x=prof$attn" -- LLAMA_PROFILE=1 "LLAMA_ATTN=$attn"
          done
        done
      done
      # (5) decode is untouched: one cell (8B FP32 from a retained context of 8192 tokens, the C7/D2 decode cell), default settings.
      if e1_is "$name" "${E2_MODEL:-8b}"; then
        for ((rep = 1; rep <= REPEATS; rep++)); do
          CONTEXT=8448 GENERATION=128 WARM=8192 c_launch "$name" avx512_fp32 "$(d2_m 1 avx512_fp32)" 56 "$rep" "$OUT/short.txt" \
            "$staged" "$tokenizer" p=8192 c=8448 x=dec
        done
      fi
      unstage_model_on 2 "$name" fp32
    done
  fi
  # (6) summaries.
  if ! dry_run; then
    "$PYTHON" "$ROOT/tools/summarize_e2.py" ttft "$OUT" > "$dir/ttft.csv"
    {
      "$PYTHON" "$ROOT/tools/summarize_attention_bench.py" --compare "$dir/bench-split.jsonl" "$dir/bench-tiled.jsonl" \
        --sweep "$dir/bench-tiled-sweep.jsonl"
      for attn in split tiled; do
        printf '\n# attention=%s\n\n' "$attn"
        "$PYTHON" "$ROOT/tools/summarize_attention_bench.py" "$dir" --prefill-file "$dir/bench-$attn.jsonl" --no-counters
      done
      printf '\n# Counters\n\n'
      "$PYTHON" "$ROOT/tools/summarize_attention_bench.py" "$dir" --prefill-file /dev/null --counters-csv "$dir/counters.csv" |
        sed -n '/^## Counters/,$p'
      printf '\n# Engine TTFT (ttft.csv)\n\n```\n'
      cat "$dir/ttft.csv"
      printf '```\n'
    } > "$dir/bench-summary.md"
  fi
}

phase_E2() {
  local dir="$OUT/E2"
  if [[ $STAGE == bench ]]; then
    e2_bench "$dir"
  elif [[ -z $MODELS ]]; then
    echo 'MODELS is empty: no E2 check was made' | tee "$dir/no-models.txt"
  else
    need_shm
    e2_check "$dir"
  fi
}

require_phases
if ! dry_run; then
  check_stage
  validate_models
fi
run_phases
exit "$OVERALL"
