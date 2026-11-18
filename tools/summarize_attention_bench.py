"""Summarize the E0 attention campaign directory as Markdown: the prefill matrix, the row-block sweep (H3), half the KV
heads (H6), decode (H5), hardware counters, and the cross-check of the bench against a profiled engine run.

    summarize_attention_bench.py E0_DIR [--counters-csv PATH] [--layers 32] [--tolerance 0.20]
                                 [--peak-gflops 8800] [--peak-gbs 552]

Reads bench-prefill.jsonl, rowblock.jsonl, bench-prefill-kv4.jsonl, bench-decode.jsonl, counters-*.{jsonl,perf.csv},
engine-profile.json and the e0 row-block comparison, whichever exist. The cross-check line compares the bench's median
ms per layer times the layer count (8B, prompt 8192, M 256, 56 threads) with the attention seconds of the engine's profile.
"""
import argparse
import csv
import glob
import json
import os
import sys

CROSS_CELL = dict(shape='8b', prompt=8192, m=256, workers=56)


def load(path):
    if not os.path.exists(path):
        return []
    with open(path) as handle:
        return [json.loads(line) for line in handle if line.startswith('{')]


def pct(value, base):
    return f'{100 * value / base:.0f}%' if base and value is not None else '-'


def num(value, spec='.2f'):
    return '-' if value is None else format(value, spec)


def share(value):
    return '-' if value is None else f'{100 * value:.0f}%'


def spread(row):
    return f"{row['ms_layer_median']:.2f} ({row['ms_layer_best']:.2f}-{row['ms_layer_max']:.2f})"


def prefill_table(rows, peak):
    out = ['| shape | threads | prompt | M | ms/layer median (min-max) | GFLOP/s | % of FP32 peak | streamed/unique K/V | '
           'busy mean (min) | tail | task phase | last microbatch ms |',
           '| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |']
    for r in sorted(rows, key=lambda r: (r['shape'], r['kv_heads'], r['workers'], r['prompt'], r['m'])):
        busy = f"{share(r.get('busy_mean'))} ({share(r.get('busy_min'))})" if r.get('busy_mean') is not None else '-'
        out.append(f"| {r['shape']} kv{r['kv_heads']} | {r['workers']} | {r['prompt']} | {r['m']} | {spread(r)} | "
                   f"{r['gflops_median']:.0f} | {pct(r['gflops_median'], peak)} | {num(r.get('streamed_over_unique'), '.1f')} | "
                   f"{busy} | {share(r.get('tail_fraction'))} | {share(r.get('task_phase_fraction'))} | "
                   f"{r['last_microbatch_ms_median']:.2f} |")
    return out


def rowblock_table(rows):
    out = ['| rows per task | ms/layer median (min-max) | vs 4 | tasks/layer | streamed/unique K/V | busy mean | tail |',
           '| ---: | ---: | ---: | ---: | ---: | ---: | ---: |']
    base = next((r for r in rows if r['row_block'] == 4), None)
    for r in sorted(rows, key=lambda r: r['row_block']):
        ratio = f"{base['ms_layer_median'] / r['ms_layer_median']:.2f}x" if base else '-'
        out.append(f"| {r['row_block']} | {spread(r)} | {ratio} | {r.get('tasks', 0)} | "
                   f"{num(r.get('streamed_over_unique'), '.1f')} | {share(r.get('busy_mean'))} | {share(r.get('tail_fraction'))} |")
    verdict = ''
    if base:
        best = max(rows, key=lambda r: base['ms_layer_median'] / r['ms_layer_median'])
        gain = base['ms_layer_median'] / best['ms_layer_median']
        verdict = (f"H3: the best row block is {best['row_block']} at {gain:.2f}x over 4 "
                   f"({'at or above' if gain >= 1.3 else 'below'} the 1.3x that makes it the first step of E2).")
    return out, verdict


def half_table(full, half):
    out = ['| shape | prompt | M | ms/layer, 8 KV heads | ms/layer, 4 KV heads | ratio (2.00 = half the work in half the time) |',
           '| --- | ---: | ---: | ---: | ---: | ---: |']
    for h in sorted(half, key=lambda r: (r['shape'], r['prompt'], r['m'])):
        f = next((r for r in full if (r['shape'], r['prompt'], r['m'], r['workers']) ==
                  (h['shape'], h['prompt'], h['m'], h['workers']) and r['kv_heads'] == h['kv_heads_model']), None)
        if f:
            out.append(f"| {h['shape']} | {h['prompt']} | {h['m']} | {f['ms_layer_median']:.2f} | {h['ms_layer_median']:.2f} | "
                       f"{f['ms_layer_median'] / h['ms_layer_median']:.2f} |")
    return out


def decode_table(rows, peak_gbs, peak_gflops):
    out = ['| shape | KV heads | context | chunks | tasks | ms/layer median (min-max) | GB/s median | % of HBM peak | '
           'GFLOP/s | busy mean | tail |',
           '| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |']
    for r in sorted(rows, key=lambda r: (r['shape'], r['kv_heads'], r['context'])):
        out.append(f"| {r['shape']} | {r['kv_heads']} | {r['context']} | {r['chunks']} | {r.get('tasks', 0)} | {spread(r)} | "
                   f"{r['gbs_median']:.0f} | {pct(r['gbs_median'], peak_gbs)} | {r['gflops_median']:.0f} | "
                   f"{share(r.get('busy_mean'))} | {share(r.get('tail_fraction'))} |")
    return out


def read_perf(path):
    """{(socket, event): value} from `perf stat --per-socket -x ,` lines (S0,56,value,unit,event,...)."""
    values = {}
    with open(path) as handle:
        for fields in csv.reader(handle):
            if len(fields) < 5 or not fields[0].startswith('S'):
                continue
            try:
                values[(fields[0], fields[4].split(':')[0])] = float(fields[2])
            except ValueError:
                values[(fields[0], fields[4].split(':')[0])] = None  # <not counted> / <not supported>
    return values


def counters(directory, csv_path):
    """The counter cells as Markdown lines; writes the flat CSV (or `not available`) to csv_path."""
    cells = {}
    for path in sorted(glob.glob(os.path.join(directory, 'counters-*.perf.csv'))):
        name = os.path.basename(path)[len('counters-'):-len('.perf.csv')]
        bench = load(path[:-len('.perf.csv')] + '.jsonl')
        cells[name] = (read_perf(path), bench[0] if bench else None)
    cells = {k: v for k, v in cells.items() if v[0]}
    if csv_path:
        with open(csv_path, 'w', newline='') as handle:
            if not cells:
                handle.write('not available\n')
            else:
                writer = csv.writer(handle)
                writer.writerow(['cell', 'socket', 'event', 'value'])
                for name, (values, _) in cells.items():
                    for (socket, event), value in sorted(values.items()):
                        writer.writerow([name, socket, event, '' if value is None else f'{value:.0f}'])
    if not cells:
        return ['Counters: not available (no perf access, or no counter run).']
    out = ['Socket 0 of each counted run (the whole launch, bench start-up included). Per pass: the cell\'s FLOPs times the passes '
           'the launch made (prefill: warm-up + reps + traced; decode: 2 warm-up + reps + traced calls).', '',
           '| cell | cycles | instructions | IPC | instructions / FLOP | l1d.replacement / KFLOP | l2_rqsts.miss / KFLOP | '
           'l3_miss loads / KFLOP |', '| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |']
    for name, (values, bench) in cells.items():
        get = lambda event: values.get(('S0', event))
        flops = None
        if bench:
            passes = bench['reps'] + 2 if bench['mode'] == 'prefill' else bench['reps'] + 2 + min(bench['reps'], 16)
            flops = bench['flops'] * passes
        cycles, instructions = get('cycles'), get('instructions')
        ipc = instructions / cycles if cycles and instructions is not None else None

        def per(event, scale):
            value = get(event)
            return None if value is None or not flops else value / flops * scale
        out.append(f"| {name} | {num(cycles, '.3g')} | {num(instructions, '.3g')} | {num(ipc)} | "
                   f"{num(per('instructions', 1), '.3f')} | {num(per('l1d.replacement', 1e3), '.2f')} | "
                   f"{num(per('l2_rqsts.miss', 1e3), '.2f')} | {num(per('mem_load_retired.l3_miss', 1e3), '.3f')} |")
    return out


def cross_check(directory, prefill, layers, tolerance):
    path = os.path.join(directory, 'engine-profile.json')
    cell = [r for r in prefill if all(r.get(k) == v for k, v in CROSS_CELL.items()) and r['kv_heads'] == r['kv_heads_model']]
    if not cell:
        return ['Cross-check: no bench cell for 8b / prompt 8192 / M 256 / 56 threads.']
    bench = cell[0]['ms_layer_median'] * layers / 1e3
    if not os.path.exists(path):
        return [f'Cross-check: no engine profile (engine-profile.json); the bench predicts {bench:.2f} s of attention '
                f'({cell[0]["ms_layer_median"]:.2f} ms/layer x {layers}).']
    with open(path) as handle:
        report = json.load(handle)
    stage = (report.get('profile_seconds') or {}).get('prefill') or {}
    engine = stage.get('attention')
    if engine is None:
        return ['Cross-check: the engine profile has no prefill attention time (was LLAMA_PROFILE=1 set?).']
    ratio = bench / engine
    ok = abs(ratio - 1) <= tolerance
    return [f"Cross-check: bench {cell[0]['ms_layer_median']:.2f} ms/layer x {layers} = {bench:.2f} s against the profiled engine "
            f"attention {engine:.2f} s (forward {stage.get('forward', 0):.2f} s): ratio {ratio:.2f}, "
            f"{'PASS' if ok else 'FAIL'} (within +-{100 * tolerance:.0f}%)."]


def correctness(directory):
    out = []
    for path in sorted(glob.glob(os.path.join(directory, 'compare-*e0-rowblock*.json'))):
        with open(path) as handle:
            summary = json.load(handle).get('summary', {})
        out.append(f"{os.path.basename(path)[8:-5]}: bitwise={summary.get('bitwise')} max_abs={summary.get('max_abs')}")
    return out


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('directory')
    parser.add_argument('--counters-csv', help='write the counters of every counters-*.perf.csv here (or "not available")')
    parser.add_argument('--layers', type=int, default=32)
    parser.add_argument('--tolerance', type=float, default=0.20)
    parser.add_argument('--peak-gflops', type=float, default=8800.0, help='FP32 FMA peak of the socket (docs/hbm-roofline.md)')
    parser.add_argument('--peak-gbs', type=float, default=552.0, help='HBM read ceiling of the socket in GB/s')
    args = parser.parse_args(argv)
    d = args.directory
    prefill = load(os.path.join(d, 'bench-prefill.jsonl'))
    kv4 = load(os.path.join(d, 'bench-prefill-kv4.jsonl'))
    rowblock = load(os.path.join(d, 'rowblock.jsonl'))
    decode = load(os.path.join(d, 'bench-decode.jsonl'))
    out = ['# E0 attention measurements', '']
    first = (prefill or rowblock or decode or [None])[0]
    if first:
        out += [f"isa={first['isa']} backend={first['backend']} K/V on node {first['node']}; medians over the repetitions "
                f"(best-max in brackets); FP32 peak {args.peak_gflops:.0f} GFLOP/s, HBM {args.peak_gbs:.0f} GB/s.", '']
    out += ['## Cross-check', ''] + cross_check(d, prefill, args.layers, args.tolerance) + ['']
    if prefill:
        out += ['## Prefill, one layer, whole prompt', ''] + prefill_table(prefill, args.peak_gflops) + ['']
    if rowblock:
        table, verdict = rowblock_table(rowblock)
        out += ['## Row block sweep (H3): 8b, prompt 8192, M 256, 56 threads', ''] + table + ['', verdict, '']
    if kv4:
        out += ['## Half the KV heads (H6): one socket\'s share of the model, 56 threads', ''] + half_table(prefill, kv4) + ['']
    if decode:
        out += ['## Decode, one layer, last position (H5)', ''] + decode_table(decode, args.peak_gbs, args.peak_gflops) + ['']
    out += ['## Counters', ''] + counters(d, args.counters_csv) + ['']
    check = correctness(d)
    if check:
        out += ['## Row-block correctness (engine, bitwise)', ''] + check + ['']
    print('\n'.join(out))
    return 0


if __name__ == '__main__':
    sys.exit(main())
