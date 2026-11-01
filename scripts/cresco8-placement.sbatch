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
check_stage

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
  expect two-sockets-hbm-needs-node3 fail "$dir" -- numactl --membind=0,2 "$probe" --topology "$EXPLICIT" --sockets 2 \
    --placement hbm
  expect two-sockets-hbm-full-set ok "$dir" -- "$probe" --topology "$EXPLICIT" --sockets 2 --placement hbm
  python3 - "$dir" "$OUT/flat-mode.json" << 'PY' | tee "$dir/summary.txt"
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
check('membind 0,2: allowed nodes [0, 2]', t['mems_allowed'] == [0, 2] and t['sockets'][0]['hbm_allowed'] and not t['sockets'][1]['ddr_allowed'] and not t['sockets'][1]['hbm_allowed'])
err = (d / 'two-sockets-hbm-needs-node3.err').read_text()
check('membind 0,2: two-socket hbm refused naming node 1 or 3', 'not in this job' in err, err.strip()[:160])
sys.exit(1 if bad else 0)
PY
  [[ ${PIPESTATUS[0]} -eq 0 ]] && record_result A1-summary PASS || record_result A1-summary FAIL
}

# ---- A2: bound regions --------------------------------------------------------------------------------------------
phase_A2() {
  local dir="$OUT/A2" regions="$ROOT/build/release/test-regions" n
  for n in 0 1 2 3; do
    expect "bind-1GiB-node$n" ok "$dir" -- numactl --cpunodebind=0 "$regions" bind "$n" 1073741824
  done
  for n in 0 2; do
    expect "membind02-bind-node$n" ok "$dir" -- numactl --cpunodebind=0 --membind=0,2 "$regions" bind "$n" 1073741824
  done
  for n in 1 3; do
    expect "membind02-bind-node$n-refused" fail "$dir" -- numactl --cpunodebind=0 --membind=0,2 "$regions" bind "$n" \
      1073741824
    grep -q 'allowed memory nodes: 0,2' "$dir/membind02-bind-node$n-refused.err" &&
      record_result "membind02-node$n-message" PASS || record_result "membind02-node$n-message" FAIL
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
  expect verify-sampled-node2 ok "$dir" -- numactl --cpunodebind=0 "$regions" verify 2 "$gib8" all threads=56 samples=4096
  python3 - "$dir" << 'PY' | tee "$dir/summary.txt"
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
sys.exit(1 if bad else 0)
PY
  [[ ${PIPESTATUS[0]} -eq 0 ]] && record_result A3-summary PASS || record_result A3-summary FAIL
}

validate_models
run_phases
exit "$OVERALL"
