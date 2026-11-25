#!/usr/bin/env python3
"""Tables of one scripts/ttft-diagnosis.sbatch results directory (docs/ttft-70b-diagnosis.md).

    ttft_diagnosis_summary.py DIRECTORY [--peak-tops 450] [--json out.json]

Stdlib only. Inputs per run (names are the labels the job script writes):
  drv-*      the tools/xengine70 driver: .stdout holds one JSON line per request (workload, repeat, ttft_s, start_epoch_ns),
             .perf.csv/.perf-t0 the interval counters of the whole process; the prefill window of a request is
             [start_epoch_ns, start_epoch_ns + ttft_s].
  cli-*      the runq command line, LLAMA_PROFILE=1: the report .json (prefill seconds, per-phase profile, phase_epoch_ns),
             .perf.csv/.perf-t0 of the run (group mem: CAS reads and writes; group cmp: cycles, instructions, EXE.AMX_BUSY).
  plan-*     --plan-only reports.
Perf time stamps are seconds since the counters were enabled; perf-t0 is the epoch (ns) of that moment. A window sums the
intervals that overlap it, prorated by overlap (the rate is assumed uniform inside a 50 ms interval).
"""
import argparse
import json
import re
import statistics
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from xengine_summary import DDR_CAS_BYTES, HBM_CAS_BYTES, parse_perf, window_sum  # noqa: E402

SOCKETS = (0, 1)
PHASES = ['embedding', 'rope_table', 'rmsnorm', 'quantize', 'qkv', 'rope', 'kv_store', 'attention', 'wo', 'w1_w3_swiglu', 'w2',
          'classifier']
GEMMS = ['qkv', 'wo', 'w1_w3_swiglu', 'w2', 'classifier']


def load(path):
    return json.loads(Path(path).read_text())


def span(values, digits=3):
    if not values:
        return '-'
    return f'{statistics.median(values):.{digits}f} [{min(values):.{digits}f}-{max(values):.{digits}f}]'


def perf_series(prefix):
    csv, t0 = Path(f'{prefix}.perf.csv'), Path(f'{prefix}.perf-t0')
    if not csv.is_file() or not t0.is_file():
        return None, None
    return parse_perf(csv), int(t0.read_text().strip())


def window(series, t0, first_ns, last_ns):
    """Counts of the window [first_ns, last_ns] (epoch ns): {name: sum over sockets}, per-socket detail, worst coverage, pct."""
    a, b = (first_ns - t0) / 1e9, (last_ns - t0) / 1e9
    total, per_socket, cover, pct = {}, {}, 1.0, 100.0
    for (socket, name) in sorted(series):
        value, covered, running = window_sum(series, (socket, name), a, b)
        total[name] = total.get(name, 0.0) + value
        per_socket.setdefault(name, {})[socket] = value
        cover = min(cover, covered / (b - a))
        if running is not None:
            pct = min(pct, running)
    return total, per_socket, cover, pct


def read_bytes(counts):
    return {'ddr_gb': counts.get('ddr', 0) * DDR_CAS_BYTES / 1e9, 'hbm_gb': counts.get('hbm', 0) * HBM_CAS_BYTES / 1e9,
            'ddr_wr_gb': counts.get('ddr_wr', 0) * DDR_CAS_BYTES / 1e9, 'hbm_wr_gb': counts.get('hbm_wr', 0) * HBM_CAS_BYTES / 1e9}


def model_dims(report):
    c = report['model']['configuration']
    kv = c['n_kv_heads'] * c['head_dim']
    macs = {'qkv': c['dim'] * (c['dim'] + 2 * kv), 'wo': c['dim'] * c['dim'], 'w1_w3_swiglu': 2 * c['dim'] * c['hidden_dim'],
            'w2': c['dim'] * c['hidden_dim']}
    return c, {k: v * c['n_layers'] for k, v in macs.items()}


def gemm_ops(report, rows):
    """{phase: operations} for `rows` prompt rows (2 per multiply-add); the classifier runs on the last row only."""
    c, macs = model_dims(report)
    ops = {k: 2.0 * v * rows for k, v in macs.items()}
    ops['classifier'] = 2.0 * c['vocab_size'] * c['dim']
    return ops


def attention_ops(report, rows):
    c = report['model']['configuration']
    return 4.0 * (rows * rows / 2) * c['n_heads'] * c['head_dim'] * c['n_layers']


# ---- drivers ------------------------------------------------------------------------------------------------------------------------
def drivers(out, packed_gb):
    rows = []
    for stdout in sorted(out.glob('drv-*.stdout')):
        prefix = stdout.with_suffix('')
        requests = [json.loads(line) for line in stdout.read_text().splitlines() if line.startswith('{')]
        series, t0 = perf_series(prefix)
        for r in requests:
            row = {'run': prefix.name, 'workload': r['workload'], 'repeat': r['repeat'], 'warmup': r['warmup'],
                   'tokens': r['prompt_tokens'], 'ttft_s': r['ttft_s'], 'decode_s': r['decode_s']}
            if series:
                first = r['start_epoch_ns']
                last = first + round(r['ttft_s'] * 1e9)
                counts, per_socket, cover, pct = window(series, t0, first, last)
                row.update(read_bytes(counts))
                row['coverage'] = cover
                row['min_pct_running'] = pct
                row['ddr_gb_per_socket'] = [per_socket.get('ddr', {}).get(s, 0) * DDR_CAS_BYTES / 1e9 for s in SOCKETS]
                row['upi_rx_gb'] = counts.get('upi_rx', 0) * 64 / 9 / 1e9
            rows.append(row)
    return rows


def driver_table(rows, packed_gb, batches):
    print('| driver run | prompt | batches | warm-up TTFT s | measured TTFT s median [min-max] | DDR read GB / request (median) | '
          'DDR read / (batches x packed copy of both sockets) | DDR GB/s | HBM read GB / request | HBM GB/s | DDR write GB |')
    print('| --- | ---: | ---: | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: |')
    runs = sorted({r['run'] for r in rows})
    result = {}
    for run in runs:
        for workload in ('short', 'long'):
            mine = [r for r in rows if r['run'] == run and r['workload'] == workload]
            if not mine:
                continue
            warm = [r for r in mine if r['warmup']]
            meas = [r for r in mine if not r['warmup']]
            tokens = mine[0]['tokens']
            batch = batches.get(run)
            count = -(-tokens // batch) if batch else None
            ttft = [r['ttft_s'] for r in meas]
            ddr = [r['ddr_gb'] for r in meas if 'ddr_gb' in r]
            hbm = [r['hbm_gb'] for r in meas if 'hbm_gb' in r]
            ddrw = [r['ddr_wr_gb'] for r in meas if 'ddr_wr_gb' in r]
            packed = packed_gb.get(run)
            ratio = statistics.median(ddr) / (count * packed * 2) if ddr and packed and count else None  # packed is GB per socket, two sockets
            rate_d = statistics.median([r['ddr_gb'] / r['ttft_s'] for r in meas if 'ddr_gb' in r]) if ddr else None
            rate_h = statistics.median([r['hbm_gb'] / r['ttft_s'] for r in meas if 'hbm_gb' in r]) if hbm else None
            print(f"| {run} | {tokens} | {count} | {warm[0]['ttft_s']:.3f} | {span(ttft)} | "
                  f"{statistics.median(ddr):.1f} | {ratio if ratio is None else format(ratio, '.2f')} | "
                  f"{rate_d if rate_d is None else format(rate_d, '.1f')} | "
                  f"{statistics.median(hbm) if hbm else float('nan'):.1f} | {rate_h if rate_h is None else format(rate_h, '.1f')} | "
                  f"{statistics.median(ddrw) if ddrw else float('nan'):.2f} |")
            result[(run, workload)] = {'ttft': ttft, 'warm': warm[0]['ttft_s'], 'ddr_gb': ddr, 'hbm_gb': hbm, 'tokens': tokens}
    return result


# ---- profiled driver runs -----------------------------------------------------------------------------------------------------------
DIMS = {'70b': dict(dim=8192, hidden_dim=28672, n_layers=80, n_heads=64, n_kv_heads=8, head_dim=128, vocab_size=128256),
        '8b': dict(dim=4096, hidden_dim=14336, n_layers=32, n_heads=32, n_kv_heads=8, head_dim=128, vocab_size=128256)}


def model_of(name):
    return '70b' if '70b' in name else '8b'


def gemm_ops_for(model, rows):
    c = DIMS[model]
    kv = c['n_kv_heads'] * c['head_dim']
    macs = {'qkv': c['dim'] * (c['dim'] + 2 * kv), 'wo': c['dim'] * c['dim'], 'w1_w3_swiglu': 2 * c['dim'] * c['hidden_dim'],
            'w2': c['dim'] * c['hidden_dim']}
    ops = {k: 2.0 * v * c['n_layers'] * rows for k, v in macs.items()}
    ops['classifier'] = 2.0 * c['vocab_size'] * c['dim']
    return ops


def attention_ops_for(model, rows):
    c = DIMS[model]
    return 4.0 * (rows * rows / 2) * c['n_heads'] * c['head_dim'] * c['n_layers']


def prof_runs(out):
    """One row per (profiled run, workload): medians over the three measured requests, counters over each measured prefill window."""
    rows = []
    for stdout in sorted(out.glob('prof-*.stdout')):
        prefix = stdout.with_suffix('')
        model = model_of(prefix.name)
        requests = [json.loads(line) for line in stdout.read_text().splitlines() if line.startswith('{')]
        series, t0 = perf_series(prefix)
        for workload in ('short', 'long'):
            mine = [r for r in requests if r['workload'] == workload]
            if not mine:
                continue
            tokens = mine[0]['prompt_tokens']
            meas = [r for r in mine if not r['warmup']]
            row = {'run': prefix.name, 'workload': workload, 'tokens': tokens, 'microbatches': mine[0]['microbatches'],
                   'warm_ttft': [r['ttft_s'] for r in mine if r['warmup']][0], 'ttft': [r['ttft_s'] for r in meas]}
            phases = {}
            for name in PHASES:
                phases[name] = statistics.median(r['profile'][name] for r in meas)
            row['phases'] = phases
            row['phases_sum'] = sum(phases.values())
            row['other_s'] = statistics.median(r['ttft_s'] - sum(r['profile'].values()) for r in meas)
            ops = gemm_ops_for(model, tokens)
            row['gemm_tops'] = {p: ops[p] / phases[p] / 1e12 for p in GEMMS if phases[p]}
            row['gemm_ops_total'] = sum(ops[p] for p in GEMMS)
            row['attention_ops'] = attention_ops_for(model, tokens)
            if series:
                amx, cyc, ipc_i, ghz, cover, pct = [], [], [], [], [], []
                for r in meas:
                    first = r['start_epoch_ns']
                    last = first + round(r['ttft_s'] * 1e9)
                    counts, _, cov, pc = window(series, t0, first, last)
                    if counts.get('cycles'):
                        amx.append(counts.get('amx_busy', 0) / counts['cycles'])
                        cyc.append(counts['cycles'])
                        ipc_i.append(counts.get('instructions', 0) / counts['cycles'])
                        ghz.append(counts['cycles'] / counts['ref_cycles'] * 1.9 if counts.get('ref_cycles') else float('nan'))
                    cover.append(cov)
                    pct.append(pc)
                if amx:
                    row['counters'] = {'amx_busy_share': statistics.median(amx), 'ipc': statistics.median(ipc_i),
                                       'ghz': statistics.median(ghz), 'coverage': min(cover), 'min_pct_running': min(pct)}
            rows.append(row)
    return rows


def prof_table(rows, peak):
    print('| profiled run | prompt | batches | warm-up s | measured TTFT s (median [min-max]) | qkv | wo | w1_w3 | w2 | classifier | attention | '
          'small | other | GEMM share | GEMM TOP/s | share of peak |')
    print('| --- | ---: | ---: | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |')
    for r in rows:
        ph = r['phases']
        gemm = sum(ph[p] for p in GEMMS)
        small = sum(ph[p] for p in ('embedding', 'rope_table', 'rmsnorm', 'quantize', 'rope', 'kv_store'))
        ttft = statistics.median(r['ttft'])
        rate = r['gemm_ops_total'] / gemm / 1e12
        print(f"| {r['run']} | {r['tokens']} | {r['microbatches']} | {r['warm_ttft']:.3f} | {span(r['ttft'])} | {ph['qkv']:.3f} | {ph['wo']:.3f} | "
              f"{ph['w1_w3_swiglu']:.3f} | {ph['w2']:.3f} | {ph['classifier']:.3f} | {ph['attention']:.3f} | {small:.3f} | {r['other_s']:.3f} | "
              f"{100 * gemm / ttft:.1f}% | {rate:.1f} | {100 * rate / peak:.1f}% |")
    print()
    print('GEMM rate per matrix, TOP/s (share of peak), and attention rate:')
    print()
    print('| profiled run | prompt | qkv | wo | w1_w3 | w2 | attention TOP/s |')
    print('| --- | ---: | ---: | ---: | ---: | ---: | ---: |')
    for r in rows:
        t = r['gemm_tops']
        att = r['attention_ops'] / r['phases']['attention'] / 1e12 if r['phases']['attention'] else float('nan')
        print(f"| {r['run']} | {r['tokens']} | " + ' | '.join(f"{t[p]:.1f} ({100 * t[p] / peak:.1f}%)" for p in ('qkv', 'wo', 'w1_w3_swiglu', 'w2'))
              + f' | {att:.1f} |')
    print()
    print('Core counters over the measured prefill windows (median): AMX busy cycles / unhalted cycles, IPC, GHz')
    print()
    print('| profiled run | prompt | AMX busy / cycles | IPC | GHz | min % running | window coverage |')
    print('| --- | ---: | ---: | ---: | ---: | ---: | ---: |')
    for r in rows:
        c = r.get('counters')
        if c:
            print(f"| {r['run']} | {r['tokens']} | {100 * c['amx_busy_share']:.2f}% | {c['ipc']:.2f} | {c['ghz']:.2f} | {c['min_pct_running']:.0f} | {c['coverage']:.3f} |")


def plans(out):
    for path in sorted(out.glob('plan-*.json')):
        plan = load(path)['placement_plan']
        print(f'{path.stem}: admitted={plan.get("admitted")} packed={plan.get("packed")}')
        for node in plan['nodes']:
            print(f"  socket {node['socket']} {node['tier']} node {node['node']}: charged {node['charged_bytes']} of budget {node['budget_bytes']}")
        for arena in plan['arenas']:
            if arena['role'] in ('work', 'exchange', 'packed'):
                print(f"  socket {arena['socket']} {arena['tier']} {arena['role']}: {arena['bytes']}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('directory', type=Path)
    parser.add_argument('--peak-tops', type=float, default=450.0, help='AMX-INT8 peak of both sockets, TOP/s (2 x 225)')
    parser.add_argument('--json', type=Path)
    args = parser.parse_args()
    out = args.directory
    packed, batches = {}, {}
    for path in out.glob('drv-*.stderr'):
        m = re.search(r'-b(\d+)$', path.stem)
        if m:
            batches[path.stem] = int(m.group(1))
        # packed copy per socket, GB: 70B gs64 72,729,231,360 B on two sockets (packed_weights_peak of the job 5934864 profiled
        # report); 8B gs64 3,707,764,736 B per socket (the engine log of job 5935737, 'packed N + N bytes of int8 shard copies')
        packed[path.stem] = 72.729231360 / 2 if '70b' in path.stem else 3.707764736
    drv = drivers(out, packed)
    print('## plans\n')
    plans(out)
    print('\n## plain drivers (1 warm-up + 3 measured requests per prompt; packed copy GB per socket from the engine log)\n')
    print(json.dumps(packed))
    driver_table(drv, packed, batches)
    print('\n## profiled drivers\n')
    rows = prof_runs(out)
    prof_table(rows, args.peak_tops)
    if args.json:
        args.json.write_text(json.dumps({'drivers': drv, 'prof': rows}, indent=1, default=str))


if __name__ == '__main__':
    main()
