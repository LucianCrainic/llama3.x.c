#!/usr/bin/env bash
#SBATCH --job-name=llama
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=112
#SBATCH --exclusive
#SBATCH --mem=0
#SBATCH --time=03:00:00
# The CRESCO8 Flat-node job for this tree (dual Xeon Max 9480): STAGE=check validates it, STAGE=bench measures it, both at
# the engine's defaults (architecture.md section 2.1) unless the submitter exports engine knobs job-wide. Submit with a
# verified --partition/--nodelist and separate -o/-e paths. Required: ROOT (tree under test), OUT (new directory),
# PYTHON. Optional: MODELS ("NAME:ARTIFACT_DIR:HF_SNAPSHOT ..."; the artifact directory holds fp32-v3.bin, int8-v3.bin,
# tokenizer-v2.bin; without it only the build and `make test` run), PRECISIONS ("fp32 int8"), REPEATS (3), DRY_RUN=1
# (print the matrix, run nothing). The exit status is non-zero if any gate failed.
#
# STAGE=check (about 1.5 h for 1B and 8B): build, `make test` on socket 0 (SKIP_TEST=1 skips it), then per model on the
# 86-token chat prompt, logits at every position (tools/compare_engines.py):
#   FP32  one socket against two, per token (max |dlogit| <= 1e-3, LLAMA_TP_CHECK=1); the defaults (two sockets, batched
#         prefill M = 64, tiled attention) against the one-socket per-token reference (1e-3); the BF16 prefill
#         (LLAMA_PRECISION=bf16, M = 256, AMX-BF16) against the same reference (the BF16 gate); the Hugging Face gate at
#         the defaults (tools/validate.py); a profiled smoke run at the defaults and one in BF16, with their placement
#         checked (weights on HBM 2 and 3, K/V caches, packed shards).
#   Q8_0  W8A8 one socket against two (the W8A8 statistical gate) and W8A32 (1e-3); the defaults (two sockets, M = 256,
#         AMX-INT8) against the one-socket per-token W8A8 reference (W8A8 gate); the Hugging Face deviation of W8A8 and
#         W8A32 recorded, not gated; a profiled smoke run at the defaults with its placement checked.
# STAGE=bench: per model, checkpoint and REPEATS (interleaved: every cell once per repeat, one and two sockets in turn),
#   BENCH_SOCKETS ("1 2") x
#   prefill cells, BENCH_PROMPTS ("512 2048 8192") tokens, 32 generated: fp32, bf16 (LLAMA_PRECISION=bf16), int8 (W8A8);
#   decode cells from a retained context (--warm-context), BENCH_CONTEXTS ("512 8192 32768"), 128 generated: fp32, int8,
#   and w8a32 (LLAMA_Q8_ACTIVATIONS=fp32) up to 8192 (its warm-up runs per token).
#   One socket binds to socket 0; both use 56 workers per socket and engine-owned HBM placement. BENCH_PROFILE=1 (default)
#   adds one LLAMA_PROFILE=1 run per cell and socket count (tags prof / decprof, per-socket phase lines in the .stderr).
#   Perf uncore counters on the first repeat when available. tools/summarize_campaign.py writes OUT/summary.md.
#   Estimated at 3-4 h for 1B and 8B with the defaults; restrict MODELS, PRECISIONS or the lists for a shorter job.
#
# The scripts of the finished campaigns (phases A1-A6, B1-B7, C1-C7, D1-D3, E0-E6) are at tag campaigns-2026-11; each
# results document cites the commit it was produced with.
set -euo pipefail
# Slurm runs a copy of this script, so the library is found through ROOT, not through the script's own path.
: "${ROOT:?tree under test required}"
# shellcheck source=lib/cresco8-common.sh
source "$ROOT/scripts/lib/cresco8-common.sh"
require_env
BENCH_SOCKETS=${BENCH_SOCKETS:-"1 2"}
BENCH_PROMPTS=${BENCH_PROMPTS:-"512 2048 8192"}
BENCH_CONTEXTS=${BENCH_CONTEXTS:-"512 8192 32768"}
BENCH_PROFILE=${BENCH_PROFILE:-1}
for value in $BENCH_SOCKETS; do [[ $value == 1 || $value == 2 ]] || die "BENCH_SOCKETS must list 1 and/or 2, got '$BENCH_SOCKETS'"; done
for value in $BENCH_PROMPTS $BENCH_CONTEXTS; do [[ $value =~ ^[1-9][0-9]*$ ]] || die "BENCH_PROMPTS and BENCH_CONTEXTS are token counts, got '$value'"; done
[[ $BENCH_PROFILE == 0 || $BENCH_PROFILE == 1 ]] || die "BENCH_PROFILE must be 0 or 1, got '$BENCH_PROFILE'"
job_begin
# On a Flat node: S0 = cores 0-55, DDR 0, HBM 2; S1 = cores 56-111, DDR 1, HBM 3 (explicit map C/H/D = package/HBM/DDR).
EXPLICIT='explicit:0/2/0,1/3/1'
# The configuration the gates name explicitly: 56 workers per socket, engine-owned HBM placement. No job-wide CPU or
# memory binding: each team pins itself and each region binds itself.
TP_ENV=(LLAMA_BACKEND=teams LLAMA_THREADS=56 "LLAMA_TOPOLOGY=$EXPLICIT" LLAMA_PLACEMENT=hbm)

# smoke DIR NAME PRECISION ENGINE STAGED TOKENIZER TAG CHECKS [env...]: one profiled run (recorded, not a gate) and its
# placement evidence; CHECKS are the check_placement options ("kv", "kv packed").
smoke() {
  local dir=$1 name=$2 precision=$3 engine=$4 staged=$5 tokenizer=$6 tag=$7 checks=$8 label
  shift 8
  label=$(run_label "$name" "$precision" teams hbm 2 56 "x=$tag")
  GENERATION=128 CONTEXT=2048 REP=1 LOG_LEVEL=info run_engine "$label" "$engine" "$staged" "$tokenizer" "$OUT/short.txt" 1 0 \
    "$@" LLAMA_PROFILE=1
  dry_run && return 0
  # shellcheck disable=SC2086  # CHECKS is a word list
  if check_placement "$OUT/$label-1.json" $checks > "$dir/placement-$name-$precision-$tag.txt"; then
    record_result "placement-$name-$precision-$tag" PASS
  else
    record_result "placement-$name-$precision-$tag" FAIL "$(grep ^FAIL "$dir/placement-$name-$precision-$tag.txt" | head -3)"
  fi
}

stage_check() {
  local dir="$OUT/check" entry name artifacts hf precision staged tokenizer
  mkdir -p "$dir"
  if [[ -z $MODELS ]]; then
    echo 'MODELS is empty: only the build and make test ran' | tee "$dir/no-models.txt"
    return 0
  fi
  need_shm
  prompts "$dir"
  for entry in $MODELS; do
    IFS=: read -r name artifacts hf <<< "$entry"
    tokenizer="$artifacts/tokenizer-v2.bin"
    for precision in $PRECISIONS; do
      staged=$(stage_model_on 0 "$name" "$precision" "$artifacts")
      if [[ $precision == fp32 ]]; then
        compare "$dir" "$name" fp32 "$staged" 1s-vs-2s -- "${TP_ENV[@]}" LLAMA_SOCKETS=1 -- "${TP_ENV[@]}" LLAMA_SOCKETS=2 \
          LLAMA_TP_CHECK=1
        compare "$dir" "$name" fp32 "$staged" reference-vs-defaults --batch-b 64 -- "${TP_ENV[@]}" LLAMA_SOCKETS=1 -- \
          LLAMA_TP_CHECK=1
        compare "$dir" "$name" fp32 "$staged" reference-vs-bf16 "${BF16_GATE[@]}" --batch-b 256 -- "${TP_ENV[@]}" \
          LLAMA_SOCKETS=1 -- LLAMA_PRECISION=bf16
        hf_gate "$name" "$artifacts" "$hf" fp32 fp32-defaults
        smoke "$dir" "$name" fp32 run "$staged" "$tokenizer" smoke kv
        smoke "$dir" "$name" bf16 run "$staged" "$tokenizer" smoke "kv packed" LLAMA_PRECISION=bf16
      else
        compare "$dir" "$name" int8 "$staged" w8a8-1s-vs-2s "${W8A8_GATE[@]}" -- "${TP_ENV[@]}" LLAMA_SOCKETS=1 -- \
          "${TP_ENV[@]}" LLAMA_SOCKETS=2 LLAMA_TP_CHECK=1
        compare "$dir" "$name" int8 "$staged" w8a32-1s-vs-2s -- "${TP_ENV[@]}" LLAMA_SOCKETS=1 LLAMA_Q8_ACTIVATIONS=fp32 -- \
          "${TP_ENV[@]}" LLAMA_SOCKETS=2 LLAMA_TP_CHECK=1 LLAMA_Q8_ACTIVATIONS=fp32
        compare "$dir" "$name" int8 "$staged" w8a8-reference-vs-defaults "${W8A8_GATE[@]}" --batch-b 256 -- "${TP_ENV[@]}" \
          LLAMA_SOCKETS=1 -- LLAMA_TP_CHECK=1
        hf_gate "$name" "$artifacts" "$hf" int8 w8a8-defaults
        hf_gate "$name" "$artifacts" "$hf" int8 w8a32-defaults LLAMA_Q8_ACTIVATIONS=fp32
        smoke "$dir" "$name" int8 runq "$staged" "$tokenizer" smoke "kv packed"
      fi
      unstage_model_on 0 "$name" "$precision"
    done
  done
}

# bench_launch NAME LABEL_PRECISION ENGINE SOCKETS REP STAGED TOKENIZER PROMPT_FILE [LABEL_WORDS...] [-- ENV_WORDS...]
bench_launch() {
  local name=$1 precision=$2 engine=$3 sockets=$4 rep=$5 staged=$6 tokenizer=$7 prompt=$8 label_words=() env_words=()
  local side=label cpu= perf=0 log=warn label
  shift 8
  while (($#)); do
    if [[ $1 == -- ]]; then side=env; elif [[ $side == label ]]; then label_words+=("$1"); else env_words+=("$1"); fi
    shift
  done
  [[ $sockets == 2 ]] || cpu=0
  [[ $rep != 1 ]] || perf=$PERF
  [[ " ${label_words[*]-} " != *"prof "* ]] || log=info
  label=$(run_label "$name" "$precision" teams hbm "$sockets" 56 ${label_words[@]+"${label_words[@]}"})
  CPU_BIND=$cpu REP=$rep LOG_LEVEL=$log run_engine "$label" "$engine" "$staged" "$tokenizer" "$prompt" 1 "$perf" \
    "${TP_ENV[@]}" "LLAMA_SOCKETS=$sockets" ${env_words[@]+"${env_words[@]}"}
}

# bench_cells NAME CHECKPOINT STAGED TOKENIZER REP TAG_SUFFIX [ENV_WORDS...]: every cell of one checkpoint once, one and
# two sockets in turn; TAG_SUFFIX (empty or prof) and ENV_WORDS (LLAMA_PROFILE=1) make the profiled repeat.
bench_cells() {
  local name=$1 checkpoint=$2 staged=$3 tokenizer=$4 rep=$5 suffix=$6 n ctx sockets variant engine words
  shift 6
  local variants=(fp32) decode_variants=(fp32)
  engine=run
  if [[ $checkpoint == int8 ]]; then
    variants=(int8) decode_variants=(int8 w8a32) engine=runq
  else
    variants=(fp32 bf16)
  fi
  for n in $BENCH_PROMPTS; do
    for variant in "${variants[@]}"; do
      words=()
      [[ $variant != bf16 ]] || words=(LLAMA_PRECISION=bf16)
      for sockets in $BENCH_SOCKETS; do
        CONTEXT=$((n + 256)) GENERATION=32 bench_launch "$name" "$variant" "$engine" "$sockets" "$rep" "$staged" "$tokenizer" \
          "$OUT/prompt-$n.txt" "p=$n" ${suffix:+"x=$suffix"} -- ${words[@]+"${words[@]}"} "$@"
      done
    done
  done
  for ctx in $BENCH_CONTEXTS; do
    for variant in "${decode_variants[@]}"; do
      words=()
      if [[ $variant == w8a32 ]]; then
        ((ctx <= 8192)) || continue
        words=(LLAMA_Q8_ACTIVATIONS=fp32)
      fi
      for sockets in $BENCH_SOCKETS; do
        CONTEXT=$((ctx + 256)) GENERATION=128 WARM=$ctx bench_launch "$name" "$variant" "$engine" "$sockets" "$rep" "$staged" \
          "$tokenizer" "$OUT/short.txt" "p=$ctx" "x=dec$suffix" -- ${words[@]+"${words[@]}"} "$@"
      done
    done
  done
}

stage_bench() {
  local dir="$OUT/bench" entry name artifacts hf precision staged tokenizer rep
  mkdir -p "$dir"
  if [[ -z $MODELS ]]; then
    echo 'MODELS is empty: no benchmark was run' | tee "$dir/no-models.txt"
    return 0
  fi
  need_shm
  prompts "$dir" "$(tr ' ' ',' <<< "$BENCH_PROMPTS")"
  for entry in $MODELS; do
    IFS=: read -r name artifacts hf <<< "$entry"
    tokenizer="$artifacts/tokenizer-v2.bin"
    for precision in $PRECISIONS; do
      staged=$(stage_model_on 0 "$name" "$precision" "$artifacts")
      for ((rep = 1; rep <= REPEATS; rep++)); do
        bench_cells "$name" "$precision" "$staged" "$tokenizer" "$rep" ''
      done
      [[ $BENCH_PROFILE == 0 ]] || bench_cells "$name" "$precision" "$staged" "$tokenizer" 1 prof LLAMA_PROFILE=1
      unstage_model_on 0 "$name" "$precision"
    done
  done
}

dry_run || check_stage
"stage_$STAGE"
dry_run || "$PYTHON" "$ROOT/tools/summarize_campaign.py" "$OUT" > "$OUT/summary.md"
exit "$OVERALL"
