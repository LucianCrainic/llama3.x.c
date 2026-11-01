"""Summarize a campaign directory of schema-v2 reports as Markdown.

Run labels follow {model}-{precision}-{backend}-{placement}-s{sockets}-t{threads}[-m{M}][-p{prompt}][-c{ctx}]
[-k{kernel}]-{rep}.json. Repeats of one label are reported as the median with the full range. Schema-v1 reports
and unknown fields give '-' instead of failing. summarize_kernels.py stays for old campaigns.
"""
import json
from pathlib import Path
import re
import statistics
import sys

LABEL = re.compile(r'^(?P<model>.+?)-(?P<precision>fp32|bf16|int8|w8a32)-(?P<backend>omp|teams)-'
                   r'(?P<placement>file|ddr|hbm)-s(?P<sockets>[12])-t(?P<threads>\d+)(?:-m(?P<m>\d+))?'
                   r'(?:-p(?P<prompt>\d+))?(?:-c(?P<ctx>\d+))?(?:-k(?P<kernel>[a-z0-9_]+))?-(?P<rep>\d+)$')
CAS_LINE = 64  # bytes per uncore CAS


def parse_label(stem):
    match = LABEL.match(stem)
    return match.groupdict() if match else None


def perf_traffic(path):
    """Sum a per-socket `perf stat -x ,` CSV by event name (same layout as summarize_kernels.py)."""
    totals = {}
    for line in path.read_text().splitlines():
        fields = line.split(',')
        if len(fields) < 5 or not fields[0].startswith('S'):
            continue
        try:
            value = float(fields[2])
        except ValueError:
            continue
        totals[fields[4]] = totals.get(fields[4], 0.0) + value
    return totals


def dig(obj, *keys):
    for key in keys:
        if not isinstance(obj, dict) or key not in obj:
            return None
        obj = obj[key]
    return obj


def span(values, digits=2):
    values = [v for v in values if isinstance(v, (int, float))]
    if not values:
        return '-'
    median = statistics.median(values)
    if len(values) == 1 or min(values) == max(values):
        return f'{median:.{digits}f}'
    return f'{median:.{digits}f} [{min(values):.{digits}f}-{max(values):.{digits}f}]'


def placement_status(report):
    """Requested placement plus what the verification evidence supports."""
    placement = dig(report, 'runtime', 'placement') or '-'
    regions = report.get('regions') or []
    if placement == 'file' or placement == '-':
        return placement
    if not regions:
        return f'{placement} (unverified)'
    if any((r.get('pages_mismatched') or 0) > 0 for r in regions):
        return f'{placement} (MISMATCH)'
    if all(r.get('verified') == 'full' for r in regions):
        return f'{placement} (verified)'
    if any(r.get('verified') == 'sampled' for r in regions):
        return f'{placement} (sampled)'
    return f'{placement} (unverified)'


def phase_breakdown(report):
    stage = dig(report, 'profile_seconds', 'decode')
    if not stage or not stage.get('forward'):
        return '-'
    parts = {k: v for k, v in stage.items() if k not in ('forward_passes', 'forward') and isinstance(v, (int, float))}
    top = sorted(parts.items(), key=lambda item: -item[1])[:3]
    return ', '.join(f'{name} {100 * value / stage["forward"]:.0f}%' for name, value in top)


def reduce_wait_share(report):
    """Share of one socket's phase time spent in reduction and barrier wait (never summed across sockets)."""
    sockets = report.get('phase_seconds')
    if not isinstance(sockets, dict) or not sockets:
        return '-'
    shares = []
    for name in sorted(sockets):
        phases = {k: v for k, v in sockets[name].items() if isinstance(v, (int, float))}
        total = sum(phases.values())
        if total > 0:
            shares.append(f'{name}: {100 * (phases.get("reduce", 0) + phases.get("barrier_wait", 0)) / total:.0f}%')
    return ', '.join(shares) or '-'


def summarize(root):
    groups = {}
    for path in sorted(Path(root).glob('*.json')):
        info = parse_label(path.stem)
        if info is None:
            continue
        label = path.stem[:path.stem.rindex('-')]
        groups.setdefault(label, (info, []))[1].append(path)
    lines = ['| Configuration | Runs | Sockets | Placement | THP | M | Kernel | Precision | Prompt | TTFT s | Prefill tok/s '
             '| Decode tok/s | Decode phases | Reduce+wait | HBM read GB | DDR read GB | UPI rx data flits (M) |',
             '| --- | ---: | ---: | --- | --- | ---: | --- | --- | ---: | ---: | ---: | ---: | --- | --- | ---: | ---: | ---: |']
    for label, (info, paths) in groups.items():
        reports = [json.loads(p.read_text()) for p in paths]
        if any(r.get('status') != 'completed' for r in reports):
            lines.append(f'| {label} | {len(reports)} | failed |' + ' |' * 14)
            continue
        first = reports[0]
        kernel = info['kernel'] or dig(first, 'build', 'kernels') or '-'
        thp = dig(first, 'runtime', 'thp') or '-'
        thp_bytes = sum((r.get('thp_bytes') or 0) for r in first.get('regions') or [])
        if thp == 'on' and thp_bytes:
            thp = f'on ({thp_bytes / 2**30:.1f} GiB huge)'
        hbm = ddr = upi = '-'
        perf = paths[0].with_suffix('.perf.csv')
        if perf.exists():
            totals = perf_traffic(perf)
            hbm = f"{totals.get('hbm_cas_rd', 0) * CAS_LINE / 1e9:.1f}"
            ddr = f"{totals.get('ddr_cas_rd', 0) * CAS_LINE / 1e9:.1f}"
            upi = f"{sum(v for e, v in totals.items() if 'upi' in e) / 1e6:.0f}"
        lines.append('| ' + ' | '.join([
            label, str(len(reports)), info['sockets'], placement_status(first), thp, info['m'] or '-', kernel,
            dig(first, 'runtime', 'precision_mode') or info['precision'], info['prompt'] or '-',
            span([dig(r, 'timing_seconds', 'time_to_first_token') for r in reports], 3),
            span([dig(r, 'throughput_tokens_per_second', 'prefill') for r in reports]),
            span([dig(r, 'throughput_tokens_per_second', 'decode') for r in reports]),
            phase_breakdown(first), reduce_wait_share(first), hbm, ddr, upi]) + ' |')
    status = Path(root) / 'correctness-status.txt'
    if status.exists():
        lines += ['', 'Validation exit codes:', ''] + [f'- {line}' for line in status.read_text().splitlines()]
    return '\n'.join(lines) + '\n'


if __name__ == '__main__':
    if len(sys.argv) != 2:
        raise SystemExit('usage: summarize_campaign.py CAMPAIGN_DIR')
    print(summarize(sys.argv[1]), end='')
