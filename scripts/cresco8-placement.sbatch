#!/usr/bin/env bash
#SBATCH --job-name=llama-placement
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=112
#SBATCH --exclusive
#SBATCH --mem=0
#SBATCH --time=04:00:00
# Flat Mode campaign skeleton for NUMA placement (tracks A) on a dual Xeon Max 9480 node.
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

# ---- A1: topology discovery -------------------------------------------------------------------------------------
# On hbm14: S0 = cores 0-55, DDR 0, HBM 2; S1 = cores 56-111, DDR 1, HBM 3 (explicit map C/H/D = package/HBM/DDR).
EXPLICIT='explicit:0/2/0,1/3/1'
phase_A1() {
  local dir="$OUT/A1" probe="$ROOT/build/release/numa_probe"
  numactl -H > "$dir/numactl-H.txt"
  expect topology-auto any "$dir" -- "$probe"            # numaperf data, or a refusal that names the explicit form
  expect topology-explicit ok "$dir" -- "$probe" --topology "$EXPLICIT"
  expect topology-cpunodebind0 ok "$dir" -- numactl --cpunodebind=0 "$probe" --topology "$EXPLICIT"
  expect topology-srun-cpu-bind ok "$dir" -- srun --ntasks=1 --cpus-per-task=112 --cpu-bind=cores "$probe" \
    --topology "$EXPLICIT"
  expect topology-membind-0-2 ok "$dir" -- numactl --membind=0,2 "$probe" --topology "$EXPLICIT"
  # numactl --membind sets a task memory policy; it does not change the cpuset-allowed nodes that
  # get_mempolicy(MPOL_F_MEMS_ALLOWED) reports, so the allowed set (and a two-socket hbm request) are unchanged.
  expect two-sockets-hbm-under-membind ok "$dir" -- numactl --membind=0,2 "$probe" --topology "$EXPLICIT" --sockets 2 \
    --placement hbm
  expect two-sockets-hbm-full-set ok "$dir" -- "$probe" --topology "$EXPLICIT" --sockets 2 --placement hbm
  if python3 - "$dir" "$OUT/flat-mode.json" << 'PY' | tee "$dir/summary.txt"; then
import json, re, sys
from pathlib import Path
d = Path(sys.argv[1])
flat = json.loads(Path(sys.argv[2]).read_text())
bad = 0
def check(name, ok, detail=''):
    global bad
    bad += not ok
    print(('PASS' if ok else 'FAIL'), name, detail)
def load(name):
    p = d / f'{name}.out'
    return json.loads(p.read_text()) if p.exists() and p.stat().st_size else None
def cpus(lst):
    out = []
    for part in lst.split(','):
        if part:
            a, _, b = part.partition('-')
            out += range(int(a), int(b or a) + 1)
    return out
full = [list(range(0, 56)), list(range(56, 112))]
nodes = {int(k[4:]): v for k, v in flat['nodes'].items()}
# numactl -H must agree with sysfs (flat-mode.json) and with the discovered DDR nodes.
text = (d / 'numactl-H.txt').read_text()
for node, info in nodes.items():
    m = re.search(rf'node {node} cpus:([0-9 ]*)', text)
    check(f'numactl-H node {node} cpus match sysfs', m is not None and sorted(map(int, m.group(1).split())) == cpus(info['cpus']))
auto = load('topology-auto')
auto_err = (d / 'topology-auto.err').read_text()
print('INFO auto classification:', auto['classification'] if auto else 'refused: ' + auto_err.strip()[:200])
if auto:
    check('auto: numaperf layout', auto['classification'] == 'numaperf' and [s['cpus'] for s in auto['sockets']] == full and
          [(s['ddr_node'], s['hbm_node']) for s in auto['sockets']] == [(0, 2), (1, 3)])
else:
    check('auto without numaperf names the explicit form', 'explicit:C0/H0/D0,C1/H1/D1' in auto_err)
t = load('topology-explicit')
check('explicit layout', t['classification'] == 'explicit' and [s['cpus'] for s in t['sockets']] == full and
      [(s['ddr_node'], s['hbm_node']) for s in t['sockets']] == [(0, 2), (1, 3)] and t['mems_allowed'] == [0, 1, 2, 3])
check('explicit: DDR/HBM bytes plausible', all(60e9 < s['hbm_bytes'] < 70e9 for s in t['sockets']))
t = load('topology-cpunodebind0')
check('numactl --cpunodebind=0: one socket, cores 0-55', t['nsockets'] == 1 and t['sockets'][0]['cpus'] == full[0] and t['sockets'][0]['hbm_node'] == 2)
t = load('topology-srun-cpu-bind')
print('INFO srun --cpu-bind=cores allowed cpus:', [len(s['cpus']) for s in t['sockets']], 'sockets', t['nsockets'])
check('srun step: every reported CPU belongs to its socket', all(set(s['cpus']) <= set(full[i if t['nsockets'] == 2 else s['package']]) for i, s in enumerate(t['sockets'])))
t = load('topology-membind-0-2')
check('numactl --membind=0,2 leaves the cpuset-allowed set unchanged', t['mems_allowed'] == [0, 1, 2, 3] and t['proc_mems_allowed_list'] == '0-3')
sys.exit(1 if bad else 0)
PY
    record_result A1-summary PASS
  else
    record_result A1-summary FAIL
  fi
}

# ---- A2: bound regions --------------------------------------------------------------------------------------------
phase_A2() {
  local dir="$OUT/A2" regions="$ROOT/build/release/test-regions" n
  for n in 0 1 2 3; do
    expect "bind-1GiB-node$n" ok "$dir" -- numactl --cpunodebind=0 "$regions" bind "$n" 1073741824
  done
  # numactl --membind is a task policy; a VMA policy set by mbind takes precedence, so every node still binds. A node
  # outside the cpuset-allowed set is refused only by the kernel (not reproducible here without a cpuset); the
  # refusal text is exercised through fault injection below.
  for n in 0 1 2 3; do
    expect "membind02-bind-node$n" ok "$dir" -- numactl --cpunodebind=0 --membind=0,2 "$regions" bind "$n" 1073741824
  done
  expect thp-node2 ok "$dir" -- numactl --cpunodebind=0 "$regions" bind 2 1073741824 thp
  grep -q 'thp_requested=1' "$dir/thp-node2.out" && record_result thp-node2-madvise PASS ||
    record_result thp-node2-madvise FAIL "$(cat "$dir/thp-node2.err")"
  expect maps-after-failed-startup ok "$dir" -- numactl --cpunodebind=0 "$regions" maps 2
  python3 - "$dir/maps-after-failed-startup.out" << 'PY' && record_result no-leaked-mappings PASS || record_result no-leaked-mappings FAIL
import json, sys
d = json.load(open(sys.argv[1]))
assert d['second_failed'] == 1 and d['live_regions'] == 0, d
assert d['maps_before'] == d['maps_after'] and d['mapped_bytes_before'] == d['mapped_bytes_after'], d
PY
  expect fault-injection-message fail "$dir" -- env LLAMA_NUMA_FAULT=EPERM numactl --cpunodebind=0 "$regions" bind 2 4096
  grep -q 'cannot bind' "$dir/fault-injection-message.err" && record_result fault-injection-text PASS ||
    record_result fault-injection-text FAIL
}

# ---- A3: placement verification -----------------------------------------------------------------------------------
phase_A3() {
  local dir="$OUT/A3" regions="$ROOT/build/release/test-regions" n gib8=8589934592
  cat /sys/kernel/mm/transparent_hugepage/enabled > "$dir/thp-system.txt" 2>&1 || true
  for n in 0 1 2 3; do
    expect "verify-8GiB-node$n" ok "$dir" -- numactl --cpunodebind=0 "$regions" verify "$n" "$gib8" all threads=56
    expect "verify-8GiB-node$n-thp" ok "$dir" -- numactl --cpunodebind=0 "$regions" verify "$n" "$gib8" all threads=56 thp
  done
  expect verify-8GiB-node2-from-socket1 ok "$dir" -- numactl --cpunodebind=1 "$regions" verify 2 "$gib8" all threads=56
  expect verify-control-membind0-expect2 ok "$dir" -- numactl --cpunodebind=0 --membind=0 "$regions" verify -1 "$gib8" \
    all threads=56 expect=2
  expect verify-bound-node1-under-membind0 ok "$dir" -- numactl --cpunodebind=0 --membind=0 "$regions" verify 1 "$gib8" \
    all threads=56
  expect verify-sampled-node2 ok "$dir" -- numactl --cpunodebind=0 "$regions" verify 2 "$gib8" all threads=56 samples=4096
  if python3 - "$dir" << 'PY' | tee "$dir/summary.txt"; then
import json, sys
from pathlib import Path
d = Path(sys.argv[1])
bad = 0
def check(name, ok, detail=''):
    global bad
    bad += not ok
    print(('PASS' if ok else 'FAIL'), name, detail)
def load(name):
    p = d / f'{name}.out'
    return json.loads(p.read_text()) if p.exists() and p.stat().st_size else None
for n in range(4):
    for suffix in ('', '-thp'):
        v = load(f'verify-8GiB-node{n}{suffix}')
        if v is None:
            check(f'node {n}{suffix} ran', False)
            continue
        check(f'node {n}{suffix}: 100% on node, 0 missing', v['full'] == 1 and v['on_node'] == v['queried'] > 0 and v['missing'] == 0 and v['errors'] == 0,
              f"queried={v['queried']} verify={v['verify_seconds']:.2f}s populate={v['populate_gbps']:.1f} GB/s thp_bytes={v['thp_bytes']} policy={v['numa_maps_policy']}")
v = load('verify-8GiB-node2-from-socket1')
check('node 2 populated by a socket-1 team is still node 2', v and v['on_node'] == v['queried'] > 0, f"populate={v['populate_gbps']:.1f} GB/s" if v else '')
v = load('verify-control-membind0-expect2')
check('negative control: portable region under membind=0 is reported off node 2', v and v['wrong_node'] == v['queried'] > 0 and v['on_node'] == 0,
      f"numa_maps={v['numa_maps_node_pages']}" if v else '')
v = load('verify-sampled-node2')
check('sampled check is not labelled full', v and v['full'] == 0 and v['scheme'] == 'stride+extent-endpoints' and v['on_node'] == v['queried'], f"queried={v['queried']}" if v else '')
v = load('verify-bound-node1-under-membind0')
check('mbind wins over a job-wide numactl --membind=0', v and v['on_node'] == v['queried'] > 0 and v['missing'] == 0,
      f"policy={v['numa_maps_policy']} numa_maps={v['numa_maps_node_pages']}" if v else '')
sys.exit(1 if bad else 0)
PY
    record_result A3-summary PASS
  else
    record_result A3-summary FAIL
  fi
}

# ---- A4: owned weight loader --------------------------------------------------------------------------------------
# Source checkpoints are staged in tmpfs on DDR node 0 (never on HBM node 2, which must have room for the owned copy).
# The `file` baseline of the bitwise gate reads the same staged file through the mapping; where the source lives
# dominates load time but not inference, and is recorded in the report directory.
phase_A4() {
  local dir="$OUT/A4" entry name artifacts hf precision engine staged tokenizer backend placement label benv
  if [[ -z $MODELS ]]; then
    echo 'MODELS is empty: no A4 model run was made' | tee "$dir/no-models.txt"
    return 0
  fi
  need_shm
  a_models "$dir"
  for entry in $MODELS; do
    IFS=: read -r name artifacts hf <<< "$entry"
    tokenizer="$artifacts/tokenizer-v2.bin"
    for precision in $PRECISIONS; do
      engine=run
      [[ $precision == fp32 ]] || engine=runq
      staged=$(stage_model_on 0 "$name" "$precision" "$artifacts")
      # 1. bitwise logits at every prompt position: file against hbm and ddr, on each backend.
      for backend in omp teams; do
        read -r -a benv <<< "$(backend_env "$backend" 56 | tr '\n' ' ')"
        for placement in hbm ddr; do
          a_bitwise "$dir" "$name" "$precision" "$staged" "$backend-file-vs-$placement" "${benv[@]}" \
            LLAMA_PLACEMENT=file -- "${benv[@]}" LLAMA_PLACEMENT=$placement
        done
      done
      if [[ $precision == int8 ]]; then # W8A32 takes the same placement code through different kernels
        read -r -a benv <<< "$(backend_env teams 56 | tr '\n' ' ')"
        a_bitwise "$dir" "$name" "$precision" "$staged" teams-w8a32-file-vs-hbm "${benv[@]}" LLAMA_PLACEMENT=file \
          LLAMA_Q8_ACTIVATIONS=fp32 -- "${benv[@]}" LLAMA_PLACEMENT=hbm LLAMA_Q8_ACTIVATIONS=fp32
      fi
      # 2. startup breakdown and full verification, one launch per placement; memory before and after.
      for placement in hbm ddr; do
        snapshot_memory "$dir/$name-$precision-$placement.meminfo-before"
        export CPU_BIND=0
        label=$(run_label "$name" "$precision" teams "$placement" 1 56 x=startup)
        GENERATION=8 CONTEXT=2048 REP=1 run_engine "$label" "$engine" "$staged" "$tokenizer" "$OUT/short.txt" 1 0 \
          LLAMA_BACKEND=teams LLAMA_THREADS=56 OMP_PROC_BIND=false LLAMA_PLACEMENT=$placement
        unset CPU_BIND
        snapshot_memory "$dir/$name-$precision-$placement.meminfo-after"
        if dry_run; then continue; fi
        if [[ $placement == hbm ]]; then
          if "$PYTHON" "$ROOT/tools/check_placement.py" "$OUT/$label-1.json" --placement hbm --weights-node 2 \
            --embedding-node 0 > "$dir/placement-$name-$precision-$placement.txt"; then
            record_result "placement-$name-$precision-hbm" PASS
          else
            record_result "placement-$name-$precision-hbm" FAIL "$(grep ^FAIL "$dir/placement-$name-$precision-$placement.txt" | head -3)"
          fi
        else
          if "$PYTHON" "$ROOT/tools/check_placement.py" "$OUT/$label-1.json" --placement ddr --weights-node 0 \
            --embedding-node 0 > "$dir/placement-$name-$precision-$placement.txt"; then
            record_result "placement-$name-$precision-ddr" PASS
          else
            record_result "placement-$name-$precision-ddr" FAIL "$(grep ^FAIL "$dir/placement-$name-$precision-$placement.txt" | head -3)"
          fi
        fi
      done
      # 3. the Hugging Face gate with the engine under LLAMA_PLACEMENT=hbm (FP32 gated, INT8 recorded), when available.
      if [[ ${HF_GATE:-1} != 0 && $precision == fp32 ]]; then
        validate_gate "$name" "$artifacts" "$hf" fp32 fp32-hbm LLAMA_PLACEMENT=hbm && record_result "hf-$name-fp32-hbm" PASS ||
          record_result "hf-$name-fp32-hbm" FAIL
      elif [[ ${HF_GATE:-1} != 0 ]]; then
        validate_gate "$name" "$artifacts" "$hf" int8 w8a8-hbm LLAMA_PLACEMENT=hbm || true
        validate_gate "$name" "$artifacts" "$hf" int8 w8a32-hbm LLAMA_PLACEMENT=hbm LLAMA_Q8_ACTIVATIONS=fp32 || true
      else
        echo "HF_GATE=0: the Hugging Face gate under LLAMA_PLACEMENT=hbm was not run for $name $precision" |
          tee -a "$dir/no-hf-gate.txt"
      fi
      unstage_model_on 0 "$name" "$precision"
    done
  done
  dry_run || "$PYTHON" "$ROOT/tools/summarize_campaign.py" "$OUT" > "$dir/summary.md"
}

# ---- A5: KV cache and workspace placement -------------------------------------------------------------------------
# 8B with a 32768-token context, a 2048-token prompt and 128 generated tokens on HBM: KV pages must be on node 2 after
# the prompt and at the end, and the batched path (C1) must stay bitwise equal under owned placement.
phase_A5() {
  local dir="$OUT/A5" entry name artifacts hf precision engine staged tokenizer label thp benv layers
  if [[ -z $MODELS ]]; then
    echo 'MODELS is empty: no A5 model run was made' | tee "$dir/no-models.txt"
    return 0
  fi
  need_shm
  for entry in $MODELS; do
    IFS=: read -r name artifacts hf <<< "$entry"
    tokenizer="$artifacts/tokenizer-v2.bin"
    make_prompt "$ROOT/build/release/probe-fp32" "$tokenizer" 128256 2048
    dry_run || "$ROOT/build/release/probe-fp32" tokenize "$tokenizer" 128256 "@$OUT/short.txt" > "$dir/$name-tokens.txt"
    for precision in $PRECISIONS; do
      engine=run
      [[ $precision == fp32 ]] || engine=runq
      staged=$(stage_model_on 0 "$name" "$precision" "$artifacts")
      read -r -a benv <<< "$(backend_env teams 56 | tr '\n' ' ')"
      # Owned KV and work regions change nothing a kernel computes: bitwise against the mapping, per token and batched.
      a_bitwise "$dir" "$name" "$precision" "$staged" teams-file-vs-hbm "${benv[@]}" LLAMA_PLACEMENT=file -- \
        "${benv[@]}" LLAMA_PLACEMENT=hbm
      a_bitwise "$dir" "$name" "$precision" "$staged" teams-file-vs-hbm-m32 "${benv[@]}" LLAMA_PLACEMENT=file -- \
        "${benv[@]}" LLAMA_PLACEMENT=hbm LLAMA_PREFILL_BATCH=32
      for thp in off on; do
        export CPU_BIND=0
        label=$(run_label "$name" "$precision" teams hbm 1 56 p=2048 c=32768 "x=kv$thp")
        CONTEXT=32768 GENERATION=128 REP=1 run_engine "$label" "$engine" "$staged" "$tokenizer" "$OUT/prompt-2048.txt" 1 0 \
          LLAMA_BACKEND=teams LLAMA_THREADS=56 OMP_PROC_BIND=false LLAMA_PLACEMENT=hbm LLAMA_THP=$thp
        unset CPU_BIND
        if dry_run; then continue; fi
        layers=$("$PYTHON" -c "import json,sys; print(json.load(open(sys.argv[1]))['model']['layers'])" "$OUT/$label-1.json" 2> /dev/null || echo 0)
        if "$PYTHON" "$ROOT/tools/check_placement.py" "$OUT/$label-1.json" --placement hbm --weights-node 2 \
          --embedding-node 0 --kv-node 2 --layers "$layers" > "$dir/placement-$name-$precision-thp$thp.txt"; then
          record_result "kv-placement-$name-$precision-thp$thp" PASS
        else
          record_result "kv-placement-$name-$precision-thp$thp" FAIL "$(grep ^FAIL "$dir/placement-$name-$precision-thp$thp.txt" | head -3)"
        fi
      done
      unstage_model_on 0 "$name" "$precision"
    done
  done
  dry_run || "$PYTHON" "$ROOT/tools/summarize_campaign.py" "$OUT" > "$dir/summary.md"
}

# ---- A6: one-socket placement campaign ---------------------------------------------------------------------------
# Bench phase: 1B and 8B on socket 0, teams backend, REPEATS interleaved passes over every configuration. The `file`
# baselines read a copy staged on HBM node 2 and are created and removed in each pass, because the owned HBM copy needs
# the room; the anon runs load from a copy staged on DDR node 0. Gate first: a failed correctness check stops the benchmark.
a6_run() { # a6_run NAME PRECISION ENGINE SOURCE TOKENIZER REP LABELKEY PLACEMENT THREADS ENVWORDS...
  local name=$1 precision=$2 engine=$3 source=$4 tokenizer=$5 rep=$6 key=$7 placement=$8 threads=$9 label
  shift 9
  case $key in
    file) label=$(run_label "$name" "$precision" teams file 1 "$threads") ;;
    omp-file) label=$(run_label "$name" "$precision" omp file 1 "$threads") ;;
    ddr) label=$(run_label "$name" "$precision" teams ddr 1 "$threads") ;;
    hbm) label=$(run_label "$name" "$precision" teams hbm 1 "$threads") ;;
    hbm-thp) label=$(run_label "$name" "$precision" teams hbm 1 "$threads" x=thp) ;;
    remote) label=$(run_label "$name" "$precision" teams hbm 1 "$threads" x=remote) ;;
    w8a32) label=$(run_label "$name" w8a32 teams hbm 1 "$threads") ;;
    long) label=$(run_label "$name" "$precision" teams hbm 1 "$threads" p=8192 c=8448) ;;
    *) die "unknown A6 configuration $key" ;;
  esac
  REP=$rep run_engine "$label" "$engine" "$source" "$tokenizer" "${PROMPT:-$OUT/short.txt}" 1 "$PERF" "$@"
}

phase_A6() {
  local dir="$OUT/A6" entry name artifacts hf precision engine tokenizer src0 src2 rep gate=1 benv
  if [[ $STAGE != bench ]] && ! dry_run; then
    echo 'A6 is a bench phase: run it with STAGE=bench' | tee "$dir/not-bench.txt"
    return 0
  fi
  if [[ -z $MODELS ]]; then
    echo 'MODELS is empty: no A6 model run was made' | tee "$dir/no-models.txt"
    return 0
  fi
  need_shm
  a_models "$dir" 8192
  # Correctness gate: anon-hbm against the staged baseline, bitwise, before any timing.
  for entry in $MODELS; do
    IFS=: read -r name artifacts hf <<< "$entry"
    for precision in $PRECISIONS; do
      src0=$(stage_model_on 0 "$name" "$precision" "$artifacts")
      read -r -a benv <<< "$(backend_env teams 56 | tr '\n' ' ')"
      a_bitwise "$dir" "$name" "$precision" "$src0" gate-file-vs-hbm "${benv[@]}" LLAMA_PLACEMENT=file -- \
        "${benv[@]}" LLAMA_PLACEMENT=hbm
      if ! dry_run && grep -q "^FAIL compare-$name-$precision-gate-file-vs-hbm " "$OUT/phase-results.txt"; then
        gate=0
      fi
    done
  done
  if [[ $gate == 0 ]]; then
    echo 'A6 correctness gate failed: no benchmark was run' | tee "$dir/gate-failed.txt"
    return 0
  fi
  for entry in $MODELS; do
    IFS=: read -r name artifacts hf <<< "$entry"
    tokenizer="$artifacts/tokenizer-v2.bin"
    for precision in $PRECISIONS; do
      engine=run
      [[ $precision == fp32 ]] || engine=runq
      src0=$(stage_model_on 0 "$name" "$precision" "$artifacts")
      for ((rep = 1; rep <= REPEATS; rep++)); do
        # Staged-file baselines (HBM node 2 holds the copy only for the duration of these two runs).
        src2=$(stage_model_on 2 "$name" "$precision" "$artifacts")
        export CPU_BIND=0 MEM_BIND=2
        a6_run "$name" "$precision" "$engine" "$src2" "$tokenizer" "$rep" file file 56 LLAMA_BACKEND=teams \
          LLAMA_THREADS=56 OMP_PROC_BIND=false LLAMA_PLACEMENT=file
        a6_run "$name" "$precision" "$engine" "$src2" "$tokenizer" "$rep" omp-file file 56 LLAMA_BACKEND=omp \
          OMP_NUM_THREADS=56
        unset CPU_BIND MEM_BIND
        unstage_model_on 2 "$name" "$precision"
        # Engine-owned placements: no job-wide memory policy.
        export CPU_BIND=0
        a6_run "$name" "$precision" "$engine" "$src0" "$tokenizer" "$rep" ddr ddr 56 LLAMA_BACKEND=teams LLAMA_THREADS=56 \
          OMP_PROC_BIND=false LLAMA_PLACEMENT=ddr
        a6_run "$name" "$precision" "$engine" "$src0" "$tokenizer" "$rep" hbm hbm 56 LLAMA_BACKEND=teams LLAMA_THREADS=56 \
          OMP_PROC_BIND=false LLAMA_PLACEMENT=hbm
        a6_run "$name" "$precision" "$engine" "$src0" "$tokenizer" "$rep" hbm-thp hbm 56 LLAMA_BACKEND=teams \
          LLAMA_THREADS=56 OMP_PROC_BIND=false LLAMA_PLACEMENT=hbm LLAMA_THP=on
        # Remote control: socket 0's threads read HBM node 3 (an explicit map points socket 0 at it).
        a6_run "$name" "$precision" "$engine" "$src0" "$tokenizer" "$rep" remote hbm 56 LLAMA_BACKEND=teams \
          LLAMA_THREADS=56 OMP_PROC_BIND=false LLAMA_PLACEMENT=hbm LLAMA_TOPOLOGY=explicit:0/3/0,1/2/1
        for threads in 14 28; do
          a6_run "$name" "$precision" "$engine" "$src0" "$tokenizer" "$rep" hbm hbm "$threads" LLAMA_BACKEND=teams \
            LLAMA_THREADS=$threads OMP_PROC_BIND=false LLAMA_PLACEMENT=hbm
        done
        if [[ $precision == int8 ]]; then # W8A32 on the owned HBM placement only
          a6_run "$name" "$precision" "$engine" "$src0" "$tokenizer" "$rep" w8a32 hbm 56 LLAMA_BACKEND=teams \
            LLAMA_THREADS=56 OMP_PROC_BIND=false LLAMA_PLACEMENT=hbm LLAMA_Q8_ACTIVATIONS=fp32
        fi
        if [[ $precision == fp32 && $name == 8[bB] ]]; then # the long-context point
          CONTEXT=8448 PROMPT="$OUT/prompt-8192.txt" a6_run "$name" "$precision" "$engine" "$src0" "$tokenizer" "$rep" \
            long hbm 56 LLAMA_BACKEND=teams LLAMA_THREADS=56 OMP_PROC_BIND=false LLAMA_PLACEMENT=hbm
        fi
        unset CPU_BIND
      done
      unstage_model_on 0 "$name" "$precision"
    done
  done
  dry_run || "$PYTHON" "$ROOT/tools/summarize_campaign.py" "$OUT" > "$dir/summary.md"
}

if ! dry_run; then
  check_stage
  validate_models
fi
run_phases
exit "$OVERALL"
