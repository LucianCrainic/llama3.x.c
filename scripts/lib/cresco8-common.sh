# shellcheck shell=bash
# The steps of scripts/cresco8.sbatch (STAGE=check and STAGE=bench on a CRESCO8 Flat node). Source this file after
# `set -euo pipefail`. Hosts, gateways and credentials never belong here, and model locations only ever arrive through
# MODELS="NAME:ARTIFACTS:HF ...". The phase scripts of the finished campaigns (A1-A6, B1-B7, C1-C7, D1-D3, E0-E6) are at
# the tag campaigns-2026-11.

die() {
  printf '%s\n' "$*" >&2
  exit 1
}

# DRY_RUN=1 prints every command a stage would run instead of running it, so a run matrix can be checked on a laptop.
dry_run() {
  [[ ${DRY_RUN:-0} == 1 ]]
}

# Required: ROOT (tree under test), OUT (must not exist yet), PYTHON. Optional: MODELS (without it every model-based step
# is skipped and recorded as such), STAGE=check|bench (default check), REPEATS (default 3), PRECISIONS (default
# "fp32 int8": the checkpoints; the bench stage adds the BF16 prefill of the FP32 checkpoint).
require_env() {
  : "${SLURM_JOB_ID:?submit through Slurm}"
  : "${ROOT:?tree under test required}"
  : "${OUT:?unique results directory required}"
  : "${PYTHON:?validation Python required}"
  MODELS=${MODELS:-}
  STAGE=${STAGE:-check}
  REPEATS=${REPEATS:-3}
  PRECISIONS=${PRECISIONS:-"fp32 int8"}
  DRY_RUN=${DRY_RUN:-0}
  [[ $DRY_RUN == 0 || $DRY_RUN == 1 ]] || die "DRY_RUN must be 0 or 1, got '$DRY_RUN'"
  [[ $STAGE == check || $STAGE == bench ]] || die "STAGE must be check or bench, got '$STAGE'"
  [[ $REPEATS =~ ^[1-9][0-9]*$ ]] || die "REPEATS must be a positive integer, got '$REPEATS'"
  local precision entry
  for precision in $PRECISIONS; do
    [[ $precision == fp32 || $precision == int8 ]] || die "PRECISIONS must list fp32 and/or int8, got '$PRECISIONS'"
  done
  [[ -z ${PHASE:-} ]] || die "PHASE is gone (F6): STAGE=check or STAGE=bench; the phase scripts are at tag campaigns-2026-11"
  [[ ! -e $OUT ]] || die "OUT '$OUT' already exists"
  for entry in $MODELS; do
    [[ $entry =~ ^[^:]+:[^:]+:[^:]+$ ]] || die "MODELS entry '$entry' is not NAME:ARTIFACTS:HF"
  done
  export STAGE REPEATS PRECISIONS MODELS DRY_RUN
}

# Creates OUT, records the exit status, removes the staging directory, and tees job output into OUT. Engine knobs a
# submitter exports apply to every run of the job (record_environment writes them down); unset, runs take the engine's
# defaults (architecture.md section 2.1), except where a step names its configuration.
job_begin() {
  mkdir "$OUT"
  shm=
  finish() {
    local status=$?
    printf 'exit_code=%s\n' "$status" > "$OUT/exit-status.txt"
    [[ -z $shm ]] || rm -rf -- "$shm"
    exit "$status"
  }
  trap finish EXIT
  exec > >(tee "$OUT/job.stdout") 2> >(tee "$OUT/job.stderr" >&2)
  export USE_TF=0 USE_FLAX=0 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
}

record_environment() {
  {
    date -u +%FT%TZ; hostname; uname -a; lscpu; numactl -H
    scontrol show job "$SLURM_JOB_ID"
    gcc --version; pkg-config --modversion libpcre2-8
    printf "%s %s\n" "$ROOT" "$(cat "$ROOT/COMMIT" 2> /dev/null || git -C "$ROOT" rev-parse HEAD)"
    printf 'STAGE=%s REPEATS=%s PRECISIONS=%s MODELS=%s\n' "$STAGE" "$REPEATS" "$PRECISIONS" "$MODELS"
    printf 'engine knobs exported job-wide: %s\n' "$(env | grep '^LLAMA_' | sort | tr '\n' ' ')"
  } > "$OUT/environment.txt"
  (cd "$ROOT" && find src include tests tools scripts Makefile -type f ! -path '*/__pycache__/*' -print0 | sort -z |
    xargs -0 sha256sum) > "$OUT/source-$(basename "$ROOT").sha256"
}

# Flat Mode evidence: no ACPI memory-side cache and two CPU-less ~64 GiB HBM nodes.
check_flat_mode() {
  python3 - << 'PY' > "$OUT/flat-mode.json"
import glob, json
from pathlib import Path
assert not glob.glob('/sys/devices/system/node/node*/memory_side_cache/index*/size'), 'memory-side cache present'
nodes = {}
for path in sorted(glob.glob('/sys/devices/system/node/node[0-9]*')):
    total = int(Path(path, 'meminfo').read_text().split('MemTotal:')[1].split()[0]) * 1024
    nodes[Path(path).name] = dict(cpus=Path(path, 'cpulist').read_text().strip(), mem_bytes=total)
assert set(nodes) == {'node0', 'node1', 'node2', 'node3'}, nodes
assert nodes['node2']['cpus'] == '' and nodes['node3']['cpus'] == '', nodes
assert all(60e9 < nodes[n]['mem_bytes'] < 70e9 for n in ('node2', 'node3')), nodes
assert 'Max 9480' in Path('/proc/cpuinfo').read_text(), 'unexpected CPU model'
print(json.dumps(dict(mode='flat', ddr_nodes=[0, 1], hbm_nodes=[2, 3], nodes=nodes), indent=2))
PY
}

# build_tree MODE [targets...]; the log and the build configuration go to OUT.
build_tree() {
  local mode=$1
  shift
  make -C "$ROOT" CC=gcc MODE="$mode" all "$@" > "$OUT/build-$(basename "$ROOT").log"
  cp "$ROOT/build/$mode/config" "$OUT/build-config-$(basename "$ROOT").txt"
}

# Uncore CAS counters (64-byte lines) show which tier served the reads. Perf is optional: whether it ran is recorded.
EVENTS='uncore_imc/event=0x05,umask=0xcf,name=ddr_cas_rd/,uncore_type_14/event=0x05,umask=0xcf,name=hbm_cas_rd/,unc_upi_rxl_flits.all_data'
PERF=0
perf_available() {
  PERF=0
  if sudo -n /usr/bin/perf stat -a -x , -e "$EVENTS" -- true 2> /dev/null; then PERF=1; fi
  printf 'perf_uncore=%s\n' "$PERF" >> "$OUT/environment.txt"
}

# perf_attach PID PREFIX: counts until PID exits; never fails the run.
perf_attach() {
  [[ $PERF == 1 ]] || return 0
  sudo -n /usr/bin/perf stat -a --per-socket -x , -e "$EVENTS" -- tail -s 0.1 --pid="$1" -f /dev/null \
    2> "$2.perf.csv" || true
}

# run_label MODEL PRECISION BACKEND PLACEMENT SOCKETS THREADS [m=M] [p=PROMPT] [c=CTX] [k=KERNEL] [x=TAG]
#   {model}-{precision}-{backend}-{placement}-s{sockets}-t{threads}[-m{M}][-p{prompt}][-c{ctx}][-k{kernel}][-x{tag}]
# (the repeat suffix -{rep} is added by run_engine; tools/summarize_campaign.py parses these labels).
run_label() {
  local label="$1-$2-$3-$4-s$5-t$6" extra
  shift 6
  for extra in m p c k x; do
    local value
    for arg in "$@"; do
      [[ $arg == "$extra="* ]] && value=${arg#*=} && label+="-$extra$value"
    done
  done
  printf '%s\n' "$label"
}

# run_engine LABEL ENGINE CHECKPOINT TOKENIZER PROMPT_FILE REPEATS PERF [env...]
# One benchmark label, REPEATS independent launches. CPU_BIND and MEM_BIND (numactl arguments) bind the run; left unset,
# the engine places itself. CONTEXT and GENERATION size the run; WARM=N adds --warm-context N (an untimed prefill of N
# tokens, so the timed decode starts from a retained context of that length: CONTEXT must exceed N plus the prompt and
# the generation); LOG_LEVEL (default warn) is info for runs whose per-socket profile lines (LLAMA_PROFILE=1) are wanted.
run_engine() {
  local label=$1 engine=$2 checkpoint=$3 tokenizer=$4 prompt=$5 repeats=$6 perf=$7
  shift 7
  local binding=()
  [[ -z ${CPU_BIND:-} ]] || binding+=(--cpunodebind="$CPU_BIND")
  [[ -z ${MEM_BIND:-} ]] || binding+=(--membind="$MEM_BIND")
  local launcher=()
  [[ ${#binding[@]} -eq 0 ]] || launcher=(numactl "${binding[@]}")
  # REP=n runs only repetition n of this label, so a caller can interleave configurations across repeats.
  local r first=1 last=$repeats
  [[ -z ${REP:-} ]] || first=$REP last=$REP
  if dry_run; then
    printf 'RUN %s reps=%s-%s cpu=%s mem=%s ctx=%s gen=%s warm=%s env=%s\n' "$label" "$first" "$last" "${CPU_BIND:-}" \
      "${MEM_BIND:-}" "${CONTEXT:-2048}" "${GENERATION:-128}" "${WARM:-0}" "$*" | tee -a "$OUT/dry-run.txt"
    return 0
  fi
  for ((r = first; r <= last; r++)); do
    local prefix="$OUT/$label-$r"
    env "$@" "${launcher[@]}" \
      "$ROOT/build/release/$engine" "$checkpoint" -z "$tokenizer" --context-length "${CONTEXT:-2048}" --benchmark \
      --chat-template --prompt-file "$prompt" -t 0 --seed 42 -n 0 --generation-length "${GENERATION:-128}" --no-text \
      ${WARM:+--warm-context "$WARM"} --log-level "${LOG_LEVEL:-warn}" --json "$prefix.json" > "$prefix.stdout" 2> "$prefix.stderr" &
    local pid=$!
    if [[ $perf == 1 && $r == "$first" ]]; then perf_attach "$pid" "$prefix"; fi
    # A failed launch is recorded and the job goes on (its report says status error, the summary shows it as failed, and
    # OVERALL makes the job exit non-zero): one bad configuration must not cost the hours of the rest.
    local rc=0
    wait "$pid" || rc=$?
    [[ $rc -eq 0 ]] || record_result "run-$label-$r" FAIL "exit=$rc"
  done
}

# make_prompt PROBE TOKENIZER VOCAB [LENGTHS|none]: short.txt (86 tokens) and prompt-{N}.txt in OUT, with their achieved
# token counts and SHA-256 in prompts.json. LENGTHS is a comma list, or none for short.txt only.
make_prompt() {
  dry_run && return 0
  python3 "$ROOT/tools/make_prompts.py" --probe "$1" --tokenizer "$2" --vocab "$3" --out "$OUT" \
    --lengths "${4:-512,2048,8192,32768}" > /dev/null
}

# OVERALL is the job's exit status: any failed expectation or unit test makes it non-zero, but the evidence that is
# still collectable is collected.
OVERALL=0

# record_result NAME PASS|FAIL [detail]: one line in phase-results.txt.
record_result() {
  printf '%s %s %s\n' "$2" "$1" "${3:-}" >> "$OUT/phase-results.txt"
  [[ $2 == PASS ]] || OVERALL=1
}

# expect NAME ok|fail|any DIR -- COMMAND...: runs the command with output in DIR/NAME.{out,err} and checks its exit
# status against the expectation ("any" records whatever happened); the command never stops the job.
expect() {
  local name=$1 want=$2 dir=$3 rc=0
  shift 4
  if dry_run; then
    printf 'EXPECT %s (%s): %s\n' "$name" "$want" "$*" | tee -a "$OUT/dry-run.txt"
    return 0
  fi
  "$@" > "$dir/$name.out" 2> "$dir/$name.err" || rc=$?
  if [[ $want == any ]] || [[ $want == ok && $rc -eq 0 ]] || [[ $want == fail && $rc -ne 0 ]]; then
    record_result "$name" PASS "exit=$rc expected=$want"
  else
    record_result "$name" FAIL "exit=$rc expected=$want"
  fi
  return 0
}

# prompts DIR [LENGTHS]: writes the prompts (short.txt, prompt-N.txt) and tokenizes short.txt per model into
# DIR/NAME-tokens.txt (the token ids the logit gates compare at).
prompts() {
  local dir=$1 entry name artifacts hf
  for entry in $MODELS; do
    IFS=: read -r name artifacts hf <<< "$entry"
    make_prompt "$ROOT/build/release/probe-fp32" "$artifacts/tokenizer-v2.bin" 128256 "${2:-none}"
    dry_run || "$ROOT/build/release/probe-fp32" tokenize "$artifacts/tokenizer-v2.bin" 128256 "@$OUT/short.txt" \
      > "$dir/$name-tokens.txt"
  done
}

# W8A8 requantizes its activations: any change in FP32 operation order (another ISA, AMX, a split over two sockets) moves
# the logits by a noise floor that is the same whatever the cause, and that grows with the model and the context. Measured
# on 8B (D1/D3, job 5704685): AMX against AVX-512 rms 0.086 / max 1.41 on 86 tokens and 0.115 / 7.58 on the last row of
# 2048; two sockets against one 0.136 / 4.12 and 0.112 / 8.86; W8A32 agrees to 7e-5 in the same runs. The bounds sit a
# factor of 1.5-2 above the largest observed values: rms and argmax carry the judgment, the max only catches a
# catastrophic error.
W8A8_GATE=(--max-abs 16 --rms-max 0.2 --argmax-min 0.9)
# The AMX-BF16 gate (C5, fixed from the records of K1, jobs 5830324 and 5694396): AMX-BF16 prefill against the FP32
# per-token path, all positions: max 0.057-1.48, rms 0.0096-0.024, argmax agreement 98.1-100%. The bounds sit 1.3-2x
# above the observed values. BF16 is never judged against the FP32 1e-3 budget.
BF16_GATE=(--max-abs 2 --rms-max 0.05 --argmax-min 0.97)

# stage_model_on NODE NAME PRECISION ARTIFACTS: copies a checkpoint into tmpfs ($shm, created by need_shm in the parent
# shell) with its pages on NODE, and prints the staged path.
stage_model_on() {
  local node=$1 staged="$shm/$2-$3-node$1.bin"
  if dry_run; then
    printf '%s\n' "$staged"
    return 0
  fi
  if [[ ! -e $staged ]]; then
    numactl --membind="$node" cp "$4/$3-v3.bin" "$staged"
    numactl -H > "$staged.numa.txt"
  fi
  printf '%s\n' "$staged"
}

# unstage_model_on NODE NAME PRECISION: frees the tmpfs copy (copies must not accumulate).
unstage_model_on() {
  dry_run || rm -f -- "$shm/$2-$3-node$1.bin" "$shm/$2-$3-node$1.bin.numa.txt"
}

# need_shm: creates the tmpfs staging directory in the caller's shell (finish() removes it).
need_shm() {
  if dry_run; then
    shm=/dev/shm/dry-run
  elif [[ -z $shm ]]; then
    shm=$(mktemp -d "/dev/shm/llama-campaign-$SLURM_JOB_ID-XXXXXX")
  fi
}

# check_stage: environment, Flat-mode check, build, `make test` (with the engine knobs unset, so the tests see the
# engine's own defaults). SKIP_TEST=1 builds but does not run `make test` (recorded as skipped): for a follow-up job on
# a tree whose full test run already passed on the same hardware.
check_stage() {
  record_environment
  check_flat_mode
  build_tree release build/release/probe-fp32 build/release/probe-int8
  local status=0
  if [[ ${SKIP_TEST:-0} == 1 ]]; then
    echo 'SKIP_TEST=1: make test was not run in this job' | tee "$OUT/test-skipped.txt"
  else
    (cd "$ROOT" && env -u LLAMA_ATTN -u LLAMA_ATTN_DECODE -u LLAMA_KV_LAYOUT -u LLAMA_SOCKETS -u LLAMA_PREFILL_BATCH \
      -u LLAMA_KERNELS -u LLAMA_BACKEND -u LLAMA_PRECISION -u LLAMA_PLACEMENT numactl --cpunodebind=0 --membind=0 \
      make CC=gcc test) > "$OUT/test.log" 2>&1 || status=$?
    if [[ $status -eq 0 ]]; then record_result make-test PASS; else record_result make-test FAIL "exit=$status"; fi
  fi
  perf_available
}

# hf_gate NAME ARTIFACTS HF PRECISION LABEL [env...]: tools/validate.py against Hugging Face with the engine under test
# (per token, no job-wide binding). FP32 failure is fatal for the job; INT8 is recorded against its budget without gating
# (W8A8 requantizes activations, so FP32 reordering moves its Hugging Face deviation).
hf_gate() {
  local name=$1 artifacts=$2 hf=$3 precision=$4 label=$5 status=0
  shift 5
  if dry_run; then
    printf 'VALIDATE %s %s %s env=%s\n' "$name" "$precision" "$label" "$*" | tee -a "$OUT/dry-run.txt"
    return 0
  fi
  local extra=()
  [[ $precision != int8 ]] || extra+=(--max-abs 1.0 --rtol 0.05)
  (cd "$ROOT" && env "$@" "$PYTHON" tools/validate.py --hf "$hf" --checkpoint "$artifacts/$precision-v3.bin" \
    --tokenizer "$artifacts/tokenizer-v2.bin" --precision "$precision" --threads 56 \
    --output "$OUT/correctness-$name-$label.json" "${extra[@]}") \
    > "$OUT/correctness-$name-$label.stdout" 2> "$OUT/correctness-$name-$label.stderr" || status=$?
  printf '%s %s exit=%s env=%s\n' "$name" "$label" "$status" "$*" >> "$OUT/correctness-status.txt"
  if [[ $precision == fp32 ]]; then
    if [[ $status -eq 0 ]]; then record_result "hf-$name-$label" PASS; else record_result "hf-$name-$label" FAIL "exit=$status"; fi
  fi
}

# check_placement REPORT [kv] [packed]: both sockets' weights regions fully verified on HBM 2 and HBM 3, an untied
# embedding on DDR 0; with `kv`, the K/V caches of both sockets; with `packed`, one packed (BF16 or int8 shard copies)
# region per socket on its node, fully verified.
check_placement() {
  python3 - "$@" << 'PY'
import json, sys
report = json.load(open(sys.argv[1]))
regions = report.get('regions') or []
bad = 0
def check(name, ok, detail=''):
    global bad
    bad += not ok
    print(('PASS' if ok else 'FAIL'), name, detail)
check('run completed', report.get('status') == 'completed', str(report.get('status')))
check('two sockets', (report.get('runtime') or {}).get('sockets') == 2)
for socket, node in ((0, 2), (1, 3)):
    found = [r for r in regions if r['role'] == 'weights' and r['socket'] == socket]
    check(f'socket {socket} weights: one region on node {node}, fully verified, nothing elsewhere or missing',
          len(found) == 1 and found[0]['node'] == node and found[0]['verified'] == 'full' and
          (found[0]['pages_queried'] or 0) > 0 and found[0]['pages_mismatched'] == 0 and found[0]['pages_missing'] == 0,
          str(found))
if 'kv' in sys.argv[2:]:
    kv = report.get('kv') or {}
    check('KV caches of both sockets verified, none on another node, none missing', bool(kv.get('verified')) and
          kv.get('pages_wrong_node') == 0 and (kv.get('pages_queried') or 0) > 0, str({k: kv.get(k) for k in
          ('verified', 'full', 'pages_queried', 'pages_wrong_node', 'pages_missing')}))
if 'packed' in sys.argv[2:]:
    for socket, node in ((0, 2), (1, 3)):
        found = [r for r in regions if r['role'] == 'packed' and r['socket'] == socket]
        check(f'socket {socket} packed shards: on node {node}, fully verified',
              len(found) >= 1 and all(r['node'] == node and r['verified'] == 'full' and r['pages_mismatched'] == 0 and
                                      r['pages_missing'] == 0 for r in found), str(found))
for r in regions:
    if r['role'] == 'embedding':
        check('global embedding on DDR node 0', r['node'] == 0 and r['verified'] == 'full' and r['pages_mismatched'] == 0, str(r))
sys.exit(1 if bad else 0)
PY
}

# compare DIR NAME PRECISION STAGED LABEL [FLAGS...] -- ENVA... -- ENVB...: logits of configuration A against B at every
# position of the 86-token prompt (tools/compare_engines.py; default budget max |delta logit| <= 1e-3, FLAGS replace it
# and may add --batch-a/--batch-b M). No job-wide binding, so the engine places and pins itself on both sides.
compare() {
  local dir=$1 name=$2 precision=$3 staged=$4 label=$5 side=flags flags=() args=()
  shift 5
  while (($#)); do
    if [[ $1 == -- ]]; then
      [[ $side == flags ]] && side=a || side=b
    elif [[ $side == flags ]]; then
      flags+=("$1")
    else
      args+=("--env-$side" "$1")
    fi
    shift
  done
  [[ " ${flags[*]-} " == *" --max-abs "* ]] || flags+=(--max-abs 1e-3)
  expect "compare-$name-$precision-$label" ok "$dir" -- "$PYTHON" "$ROOT/tools/compare_engines.py" --probe \
    "$ROOT/build/release/probe-$precision" --checkpoint "$staged" --tokens-file "$dir/$name-tokens.txt" \
    ${args[@]+"${args[@]}"} "${flags[@]}" --output "$dir/compare-$name-$precision-$label.json"
}
