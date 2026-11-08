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
          c_record "$OUT/C7" "$name" fp32 "$staged" "c7-bf16-amx-m$m-prompt$n" "${benv[@]}" LLAMA_KERNELS=avx512 -- "${benv[@]}" \
            LLAMA_KERNELS=amx LLAMA_PRECISION=bf16
        else
          a_bitwise "$OUT/C7" "$name" int8 "$staged" "c7-avx512-q8-m$m-prompt$n" "${benv[@]}" LLAMA_KERNELS=avx512 -- "${benv[@]}" \
            LLAMA_KERNELS=avx512 LLAMA_PREFILL_GEMM=on
          a_bitwise "$OUT/C7" "$name" int8 "$staged" "c7-amx-int8-m$m-prompt$n" "${benv[@]}" LLAMA_KERNELS=avx512 -- "${benv[@]}" \
            LLAMA_KERNELS=amx
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
      tp_validate "$name" "$artifacts" "$hf" fp32 fp32-2s-b64 "${D_ENV[@]}" LLAMA_SOCKETS=2 LLAMA_PREFILL_BATCH=64 \
        LLAMA_PREFILL_GEMM=on && record_result "hf-$name-fp32-2s-b64" PASS || record_result "hf-$name-fp32-2s-b64" FAIL
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
    opts=kv
    [[ $rung != amx_bf16 && $rung != amx_int8 ]] || opts="kv packed"
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

require_phases
if ! dry_run; then
  check_stage
  validate_models
fi
run_phases
exit "$OVERALL"
