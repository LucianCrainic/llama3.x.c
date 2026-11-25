#!/usr/bin/env bash
#SBATCH --job-name=gemv-stagger
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=112
#SBATCH --exclusive
#SBATCH --mem=0
#SBATCH --time=06:00:00

# LLAMA_GEMV_STAGGER and 2 MiB pages (docs/gemv-stagger.md). Submit to one Flat node with both HBM nodes allowed.
# Required: ROOT (tree under test), OUT (new directory), STAGE, PYTHON.
#   STAGE=check   build ROOT (and BASE_ROOT, the reference tree, when given), `make test` in ROOT, then the bitwise probes:
#                 the per-position logits of the checkpoint through the reference binary, the tree with the knob off and the
#                 tree with the knob on, byte for byte, on 4 KiB and on 2 MiB pages.
#   STAGE=ab      decode benchmark cells (page size x knob), interleaved over REPEATS repetitions, then one profiled run of
#                 each cell; token traces of every run are compared.
# Required for the stages that load a model: CONFIG (8b-1s, 8b-2s, 70b-2s; STAGE=check takes a list, CONFIG="8b-1s 8b-2s"),
# CHECKPOINT, TOKENIZER.
# Optional: REPEATS=5, WARM=4096, GENERATION=128, EXTRA=0 (context length = WARM + 256 + EXTRA), BASE_ROOT, DRY_RUN=1.
# Nothing here names a host, a path or a credential: locations arrive through the environment.
set -euo pipefail

: "${ROOT:?}" "${OUT:?}" "${STAGE:?}" "${PYTHON:?}"
repeats=${REPEATS:-5}
warm=${WARM:-4096}
generation=${GENERATION:-128}
context=$((warm + 256 + ${EXTRA:-0}))
prompt="Print a long numbered list of prime numbers. Keep listing primes for at least 1000 tokens. Start at 2."
[[ $STAGE == check || $STAGE == ab ]] || { echo "STAGE must be check or ab" >&2; exit 2; }
[[ ! -e $OUT ]] || { echo "OUT '$OUT' already exists" >&2; exit 2; }

base=(LLAMA_BACKEND=teams LLAMA_THREADS=56 LLAMA_PLACEMENT_STRICT=1 LLAMA_VERIFY_SAMPLES=0 LLAMA_KV_VERIFY=1)
topology=explicit:0/2/0,1/3/1
select_config() {
  bind=()
  case $1 in
    8b-1s) cfg=("${base[@]}" "LLAMA_TOPOLOGY=$topology" LLAMA_PLACEMENT=hbm LLAMA_SOCKETS=1); bind=(numactl --cpunodebind=0) ;;
    8b-2s) cfg=("${base[@]}" "LLAMA_TOPOLOGY=$topology" LLAMA_PLACEMENT=hbm LLAMA_SOCKETS=2) ;;
    # 70B does not fit HBM with the AMX packed copy: every weight and KV layer in HBM, the packed copy in DDR
    # (docs/campaign/hybrid-memory/LEDGER.md, P2 notes for P4)
    70b-2s) cfg=("${base[@]}" "LLAMA_TOPOLOGY=$topology" LLAMA_PLACEMENT=hybrid LLAMA_SOCKETS=2
      LLAMA_HYBRID_WEIGHTS_HBM_LAYERS=80 LLAMA_HYBRID_KV_HBM_LAYERS=80 LLAMA_HYBRID_CLASSIFIER=hbm LLAMA_HYBRID_PACKED=ddr) ;;
    *) echo "CONFIG must be 8b-1s, 8b-2s or 70b-2s" >&2; exit 2 ;;
  esac
}
cfg=()
bind=()
[[ -z ${CONFIG:-} || $STAGE == check ]] || select_config "$CONFIG"
# Cells: page size and knob. THP=off is the engine default (4 KiB pages).
cells=("thp=off stagger=0" "thp=off stagger=1" "thp=on stagger=0" "thp=on stagger=1")

if [[ ${DRY_RUN:-0} == 1 ]]; then
  echo "config=${CONFIG:-} stage=$STAGE repeats=$repeats context=$context warm=$warm generation=$generation"
  printf 'env: %s\nbind: %s\n' "${cfg[*]-}" "${bind[*]-}"
  printf 'cell: %s\n' "${cells[@]}"
  exit 0
fi
mkdir "$OUT"
exec > >(tee "$OUT/job.stdout") 2> >(tee "$OUT/job.stderr" >&2)
{
  hostname; date -u; uname -a; gcc --version | head -1; numactl -H
  printf '%s %s\n' "$ROOT" "$(cat "$ROOT/COMMIT" 2> /dev/null || git -C "$ROOT" rev-parse HEAD)"
  [[ -z ${BASE_ROOT:-} ]] || printf '%s %s\n' "$BASE_ROOT" "$(cat "$BASE_ROOT/COMMIT" 2> /dev/null)"
  printf 'stage=%s config=%s repeats=%s context=%s warm=%s generation=%s\n' "$STAGE" "${CONFIG:-}" "$repeats" "$context" \
    "$warm" "$generation"
  printf 'checkpoint=%s\ntokenizer=%s\n' "${CHECKPOINT:-}" "${TOKENIZER:-}"
  printf 'engine env: %s\n' "${cfg[*]-}"
} > "$OUT/environment.txt"
(cd "$ROOT" && find src include tests tools scripts Makefile -type f ! -path '*/__pycache__/*' -print0 | sort -z |
  xargs -0 sha256sum) > "$OUT/source.sha256"

# Every LLAMA_* variable of the caller is cleared, so a cell sees exactly the engine environment written down above.
clean() { env "${unset_args[@]}" "$@"; }
unset_args=()
while IFS= read -r name; do unset_args+=(-u "$name"); done < <(env | sed -n 's/^\(LLAMA_[A-Za-z0-9_]*\)=.*/\1/p')

cell_env() { # cell_env "thp=off stagger=1" -> LLAMA_THP=off LLAMA_GEMV_STAGGER=1 (the reference binary ignores the knob)
  local thp=${1#thp=}; thp=${thp%% *}; local stagger=${1##*stagger=}
  printf 'LLAMA_THP=%s\nLLAMA_GEMV_STAGGER=%s\n' "$thp" "$stagger"
}

build() { # build TREE DIR
  make -C "$1" -j 56 CC=gcc BUILD_DIR="$2" all "$2/probe-int8" > "$2.build.log" 2>&1 || { tail -30 "$2.build.log"; exit 1; }
}

stage_check() {
  mkdir "$OUT/bin"
  build_dir="$OUT/bin"
  # the gate as the campaign check stage runs it: engine knobs unset, one socket (CPUs and memory of node 0)
  clean numactl --cpunodebind=0 --membind=0 make -C "$ROOT" -j 56 CC=gcc BUILD_DIR="$build_dir" test > "$OUT/make-test.log" 2>&1 &&
    gate=0 || gate=$?
  printf 'make test rc=%s\n' "$gate" | tee "$OUT/make-test.rc"
  tail -5 "$OUT/make-test.log"
  [[ -z ${BASE_ROOT:-} ]] || make -C "$BASE_ROOT" -j 56 CC=gcc BUILD_DIR="$OUT/bin-base" all "$OUT/bin-base/probe-int8" > "$OUT/build-base.log" 2>&1
  [[ -n ${CONFIG:-} ]] || return "$gate"
  status=0
  for name in $CONFIG; do
    select_config "$name"
    probe_config "$name" || status=1
  done
  return $((gate || status))
}

probe_config() {
  local CONFIG=$1
  # a fixed 97-token input (BOS and ids spread over the vocabulary); the probe prints every position's logits as raw float32
  tokens=(128000 $(seq 1009 1301 125000))
  local status=0
  for cell in off0 off1 on0 on1; do
    thp=${cell%[01]}; stagger=${cell: -1}
    for who in base tree; do
      [[ $who == tree || -n ${BASE_ROOT:-} ]] || continue
      binary=$OUT/bin/probe-int8; [[ $who == tree ]] || binary=$OUT/bin-base/probe-int8
      clean "${cfg[@]}" LLAMA_THP="$thp" LLAMA_GEMV_STAGGER="$stagger" ${bind[@]+"${bind[@]}"} "$binary" "$CHECKPOINT" "${tokens[@]}" \
        | sha256sum | cut -d' ' -f1 > "$OUT/logits-$CONFIG-$who-thp$thp-stagger$stagger.sha256"
      # the reference binary has no knob: its second variant is the same program, which also shows run-to-run determinism
    done
  done
  sort -u "$OUT"/logits-"$CONFIG"-*.sha256 | wc -l > "$OUT/logits-distinct-$CONFIG.txt"
  [[ $(cat "$OUT/logits-distinct-$CONFIG.txt") == 1 ]] || status=1
  ! grep -q '^e3b0c442' "$OUT"/logits-"$CONFIG"-*.sha256 || status=1 # the digest of no output: a probe failed
  printf 'distinct logit digests over %s files: %s\n' "$(ls "$OUT"/logits-"$CONFIG"-*.sha256 | wc -l)" \
    "$(cat "$OUT/logits-distinct-$CONFIG.txt")"
  return "$status"
}

run_cell() { # run_cell LABEL CELL [EXTRA_ENV...]
  local label=$1 cell=$2 prefix thp stagger
  shift 2
  prefix="$OUT/$label"
  thp=$(cell_env "$cell" | sed -n 1p); stagger=$(cell_env "$cell" | sed -n 2p)
  date +%s%N > "$prefix.launch"
  clean "${cfg[@]}" "$thp" "$stagger" "$@" LLAMA_TOKENS_OUT="$prefix.tokens.json" ${bind[@]+"${bind[@]}"} "$OUT/bin/runq" "$CHECKPOINT" \
    -z "$TOKENIZER" --benchmark --chat-template --no-text -t 0 -n 0 --context-length "$context" --warm-context "$warm" \
    --generation-length "$generation" --prompt "$prompt" --json "$prefix.json" > "$prefix.stdout" 2> "$prefix.stderr" ||
    { cat "$prefix.stderr" >&2; return 1; }
  date +%s%N > "$prefix.end"
}

stage_ab() {
  mkdir "$OUT/bin"
  build "$ROOT" "$OUT/bin"
  if [[ $CONFIG == 70b-2s ]]; then # a plan that does not fit HBM fails here, before the first run (--plan-only needs a planned placement)
    clean "${cfg[@]}" "$OUT/bin/runq" "$CHECKPOINT" --context-length "$context" --plan-only --json "$OUT/plan.json"
  fi
  for ((rep = 0; rep < repeats; rep++)); do
    for ((i = 0; i < ${#cells[@]}; i++)); do
      cell=${cells[(rep + i) % ${#cells[@]}]}
      run_cell "r$((rep + 1))-${cell// /-}" "$cell"
    done
  done
  for cell in "${cells[@]}"; do
    run_cell "prof-${cell// /-}" "$cell" LLAMA_PROFILE=1
  done
  "$PYTHON" - "$OUT" << 'PY'
import json, sys
from pathlib import Path
out = Path(sys.argv[1])
traces = {p.name: json.loads(p.read_text()) for p in sorted(out.glob('*.tokens.json'))}
reference = next(iter(traces.values()))
bad = [n for n, t in traces.items() if t != reference]
(out / 'tokens-identical.txt').write_text(f'{len(traces)} token traces, {len(bad)} differ: {bad}\n')
print(f'{len(traces)} token traces, {len(bad)} differ')
sys.exit(1 if bad else 0)
PY
}

"stage_$STAGE"
