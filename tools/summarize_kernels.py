"""Summarize a kernel campaign directory as Markdown: throughput, effective weight
bandwidth (bytes read per decode token x decode tokens/s) and uncore tier traffic."""
import json
import re
import statistics
import sys
from pathlib import Path

LINE = 64  # bytes per CAS


def bytes_per_token(model):
    c = model['configuration']
    dim, hidden, layers = c['dim'], c['hidden_dim'], c['n_layers']
    head = dim // c['n_heads']
    q, kv = c['n_heads'] * head, c['n_kv_heads'] * head
    weights = layers * (dim * q + 2 * dim * kv + q * dim + 3 * dim * hidden) + c['vocab_size'] * dim
    if model['precision'] == 'fp32':
        return weights * 4.0
    return weights * (1.0 + 4.0 / model['quantization_group_size'])


def perf_traffic(path):
    totals = {}
    for line in path.read_text().splitlines():
        fields = line.split(',')
        if len(fields) < 5 or not fields[0].startswith('S'):
            continue
        try:
            value = float(fields[2])
        except ValueError:
            continue
        event = fields[4]
        totals[(fields[0], event)] = totals.get((fields[0], event), 0.0) + value
    return totals


def stats(values):
    if not values:
        return None, None
    return statistics.mean(values), statistics.stdev(values) if len(values) > 1 else None


def fmt(mean, stdev, digits=2):
    if mean is None:
        return '–'
    return f'{mean:.{digits}f}' + (f' ± {stdev:.{digits}f}' if stdev is not None else '')


def main(root):
    groups = {}
    for path in sorted(root.glob('*.json')):
        match = re.fullmatch(r'(.+)-(\d+)\.json', path.name)
        if not match or path.name.startswith('correctness'):
            continue
        groups.setdefault(match.group(1), []).append(path)
    print('| Configuration | Runs | Kernels | Prefill tok/s | Decode tok/s | Decode GB/s | HBM read GB | DDR read GB | UPI rx data flits (M) |')
    print('| --- | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: |')
    for label, paths in groups.items():
        runs = [json.loads(p.read_text()) for p in paths]
        if any(r['status'] != 'completed' for r in runs):
            print(f'| {label} | {len(runs)} | failed | | | | | | |')
            continue
        prefill = stats([r['throughput_tokens_per_second']['prefill'] for r in runs])
        decode = stats([r['throughput_tokens_per_second']['decode'] for r in runs])
        per_token = bytes_per_token(runs[0]['model']) / 1e9
        bandwidth = (decode[0] * per_token, decode[1] * per_token if decode[1] is not None else None)
        hbm = ddr = upi = '–'
        perf = paths[0].with_suffix('.perf.csv')
        if perf.exists():
            traffic = perf_traffic(perf)
            hbm = f"{sum(v for (s, e), v in traffic.items() if e == 'hbm_cas_rd') * LINE / 1e9:.1f}"
            ddr = f"{sum(v for (s, e), v in traffic.items() if e == 'ddr_cas_rd') * LINE / 1e9:.1f}"
            upi = f"{sum(v for (s, e), v in traffic.items() if 'upi' in e) / 1e6:.0f}"
        kernels = runs[0]['build'].get('kernels') or 'scalar (baseline)'
        print(f'| {label} | {len(runs)} | {kernels} | {fmt(*prefill)} | {fmt(*decode)} | {fmt(*bandwidth, 1)} '
              f'| {hbm} | {ddr} | {upi} |')
    status = root / 'correctness-status.txt'
    if status.exists():
        print('\nValidation exit codes:\n')
        for line in status.read_text().splitlines():
            print(f'- {line}')


if __name__ == '__main__':
    main(Path(sys.argv[1]))
