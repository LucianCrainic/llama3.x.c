#!/usr/bin/env python3
"""Tables of one scripts/concurrent-tiers.sbatch STAGE=bench results directory (docs/concurrent-tiers-validation.md).

    tools/concurrent_tiers_summary.py DIR [--reference CELL] [--guard 0.05] [--json]

Per cell (median [min-max] over the interleaved repetitions): decode tokens per second, the change against the reference
cell (median ratio, and the number of repetitions in which the cell beat the reference's repetition of the same round),
and the bytes the uncore counters saw in the decode window: HBM and DDR read GB per generated token, summed over the
sockets (32-byte CAS lines on HBM, 64-byte on DDR), and the DDR share of the read bytes. Cells named rows-r<fraction>-c<cores>
are also laid out as a fraction x cores grid. The profiled runs (prof-*) are reported apart: decode seconds per token of
each phase. Perf windows come from tools/hbm_ddr_results.py (phase_epoch_ns of the report against the perf clock).
"""
import argparse
import json
import re
import statistics
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import hbm_ddr_results  # noqa: E402


def span(values, digits=2):
    if not values:
        return '-'
    return f'{statistics.median(values):.{digits}f} [{min(values):.{digits}f}-{max(values):.{digits}f}]'


def collect(directory, guard):
    """{cell: {rep: {'tps': decode tokens/s, 'hbm': GB/token, 'ddr': GB/token}}} of the plain runs, and the profiled ones."""
    runs, profiled = {}, {}
    counters = {}
    for result in hbm_ddr_results.results(directory, guard):
        decode = (result.get('counters') or {}).get('decode')
        counters[result['run']] = decode
    for path in sorted(directory.glob('*.json')):
        name = path.stem
        if name.endswith(('.plan', '.tokens')):
            continue
        match = re.match(r'^(r(\d+)|prof)-(.+)$', name)
        if not match:
            continue
        report = json.loads(path.read_text())
        if report.get('status') != 'completed':
            continue
        cell, rep = match.group(3), int(match.group(2) or 0)
        entry = {'tps': report['throughput_tokens_per_second']['decode']}
        decode = counters.get(name)
        if decode:
            for tier in ('hbm', 'ddr'):
                entry[tier] = sum(s.get(f'{tier}_gb_per_step', 0.0) for s in decode['sockets'])
        if match.group(1) == 'prof':
            entry['profile'] = report.get('profile_seconds', {}).get('decode')
            profiled[cell] = entry
        else:
            runs.setdefault(cell, {})[rep] = entry
    return runs, profiled


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('directory', type=Path)
    parser.add_argument('--reference')
    parser.add_argument('--guard', type=float, default=0.05)
    parser.add_argument('--json', action='store_true', help='print the per-cell statistics as JSON')
    args = parser.parse_args(argv)
    runs, profiled = collect(args.directory, args.guard)
    reference = args.reference
    if not reference:
        env = (args.directory/'environment.txt').read_text() if (args.directory/'environment.txt').exists() else ''
        found = re.search(r'reference=(\S+)', env)
        reference = found.group(1) if found else next(iter(runs))
    ref_runs = runs.get(reference, {})
    stats = {}
    print(f'Reference cell: {reference}. Decode tokens per second, median [min-max] over {len(ref_runs)} repetitions.')
    print()
    print('| cell | decode tok/s | vs reference | faster in | HBM GB/token | DDR GB/token | DDR share |')
    print('| --- | --- | --- | --- | --- | --- | --- |')
    ref_median = statistics.median(e['tps'] for e in ref_runs.values()) if ref_runs else None
    for cell in sorted(runs, key=lambda c: (c != reference, c)):
        reps = runs[cell]
        tps = [e['tps'] for e in reps.values()]
        change = f'{100 * (statistics.median(tps) / ref_median - 1):+.1f}%' if ref_median else '-'
        wins = sum(1 for rep, e in reps.items() if rep in ref_runs and e['tps'] > ref_runs[rep]['tps'])
        pairs = sum(1 for rep in reps if rep in ref_runs)
        hbm = [e['hbm'] for e in reps.values() if 'hbm' in e]
        ddr = [e['ddr'] for e in reps.values() if 'ddr' in e]
        share = [e['ddr'] / (e['hbm'] + e['ddr']) for e in reps.values() if 'hbm' in e and e['hbm'] + e['ddr'] > 0]
        print(f'| {cell} | {span(tps)} | {change if cell != reference else "-"} | '
              f'{f"{wins} of {pairs}" if cell != reference else "-"} | {span(hbm, 1)} | {span(ddr, 1)} | '
              f'{f"{100 * statistics.median(share):.1f}%" if share else "-"} |')
        stats[cell] = {'tps': tps, 'hbm_gb_per_token': hbm, 'ddr_gb_per_token': ddr, 'ddr_share': share,
                       'vs_reference_pct': 100 * (statistics.median(tps) / ref_median - 1) if ref_median else None,
                       'faster_in': [wins, pairs]}
    grid = {}
    for cell, reps in runs.items():
        match = re.match(r'^rows-r([0-9.]+)-c(\d+)$', cell)
        if match:
            grid[(float(match.group(1)), int(match.group(2)))] = statistics.median(e['tps'] for e in reps.values())
    if grid and ref_median:
        fractions, cores = sorted({r for r, _ in grid}), sorted({c for _, c in grid})
        print()
        print('Change of the median decode rate against the reference, by DDR fraction (rows) and DDR cores (columns):')
        print()
        print('| fraction | ' + ' | '.join(f'c{c}' for c in cores) + ' |')
        print('| --- |' + ' --- |' * len(cores))
        for r in fractions:
            print(f'| {r} | ' + ' | '.join(f'{100 * (grid[(r, c)] / ref_median - 1):+.1f}%' if (r, c) in grid else '-'
                                           for c in cores) + ' |')
        best = max(grid, key=grid.get)
        print()
        print(f'Best cell of the grid: r = {best[0]}, {best[1]} DDR cores, {100 * (grid[best] / ref_median - 1):+.1f}% '
              f'against the reference.')
    if profiled:
        print()
        print('Decode profile, milliseconds per generated token (one profiled run per cell; the counters of LLAMA_PROFILE=1):')
        print()
        phases = ['qkv', 'wo', 'w1_w3_swiglu', 'w2', 'attention', 'classifier', 'other', 'forward']
        print('| phase | ' + ' | '.join(sorted(profiled)) + ' |')
        print('| --- |' + ' --- |' * len(profiled))
        for phase in phases:
            cells = []
            for cell in sorted(profiled):
                profile = profiled[cell].get('profile')
                cells.append(f'{profile[phase] / profile["forward_passes"] * 1e3:.2f}' if profile and phase in profile else '-')
            print(f'| {phase} | ' + ' | '.join(cells) + ' |')
    if args.json:
        print(json.dumps({'reference': reference, 'cells': stats}))


if __name__ == '__main__':
    main()
