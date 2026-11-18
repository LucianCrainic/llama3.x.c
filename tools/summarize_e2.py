"""Summaries of the E2 (tiled attention) cluster phase.

    summarize_e2.py checks E2_DIR     table of every compare-*.json (compare_engines.py results): case, gate, max |delta|,
                                      argmax agreement, pass; exit status 1 when any comparison failed
    summarize_e2.py ttft OUT_DIR      CSV of the engine reports whose label carries the tag attnsplit or attntiled (and the
                                      profiled repeats profsplit / proftiled): median TTFT and prefill tok/s per
                                      configuration, split against tiled
"""
import csv
import json
import re
import statistics
import sys
from pathlib import Path

LABEL = re.compile(r'^(?P<base>.+)-x(?P<tag>attnsplit|attntiled|profsplit|proftiled)-(?P<rep>\d+)$')


def gate_text(result):
    budget = result.get('budget') or {}
    if budget.get('bitwise'):
        return 'bitwise'
    parts = [f"max {budget.get('max_abs'):g}"]
    if budget.get('rms_max') is not None:
        parts.append(f"rms {budget['rms_max']:g}")
    parts.append(f"argmax {100 * budget.get('argmax_agreement', 1.0):g}%" + (' (statistical)' if budget.get('statistical') else ''))
    return ', '.join(parts)


def checks(directory):
    rows, failed = [], 0
    for path in sorted(Path(directory).glob('compare-*.json')):
        try:
            result = json.loads(path.read_text())
            summary = result['summary']
        except (ValueError, KeyError):
            continue
        passed = bool(result.get('passed'))
        failed += not passed
        rows.append(f"| {path.stem[len('compare-'):]} | {gate_text(result)} | {summary['max_abs']:.3g} | "
                    f"{100 * summary['argmax_agreement']:.1f}% | {'PASS' if passed else 'FAIL'} |")
    out = ['# E2 correctness', '', '| case | gate | max abs delta | argmax agreement | pass |', '| --- | --- | ---: | ---: | --- |']
    out += rows or ['| (no comparison results) | | | | |']
    out += ['', f'{len(rows)} comparisons, {failed} failed.']
    return '\n'.join(out), failed


def dig(obj, *keys):
    for key in keys:
        obj = obj.get(key) if isinstance(obj, dict) else None
    return obj


def ttft(root):
    groups = {}
    for path in sorted(Path(root).glob('*.json')):
        match = LABEL.match(path.stem)
        if not match:
            continue
        try:
            report = json.loads(path.read_text())
        except ValueError:
            continue
        if report.get('status') != 'completed':
            continue
        groups.setdefault((match['base'], match['tag']), []).append(report)
    cells = {}
    for (base, tag), reports in groups.items():
        times = [dig(r, 'timing_seconds', 'time_to_first_token') for r in reports]
        rates = [dig(r, 'throughput_tokens_per_second', 'prefill') for r in reports]
        times = [t for t in times if isinstance(t, (int, float))]
        rates = [t for t in rates if isinstance(t, (int, float))]
        if times:
            cells.setdefault(base, {})[tag] = (len(times), statistics.median(times), min(times), max(times),
                                               statistics.median(rates) if rates else None)
    lines = [['configuration', 'attn', 'repeats', 'ttft_median_s', 'ttft_min_s', 'ttft_max_s', 'prefill_tok_s_median',
              'speedup_vs_split']]
    for base in sorted(cells):
        split = cells[base].get('attnsplit') or cells[base].get('profsplit')
        for tag in ('attnsplit', 'attntiled', 'profsplit', 'proftiled'):
            if tag not in cells[base]:
                continue
            n, med, lo, hi, rate = cells[base][tag]
            speed = ''
            if tag in ('attntiled', 'proftiled'):
                ref = cells[base].get('attnsplit' if tag == 'attntiled' else 'profsplit')
                speed = f'{ref[1] / med:.3f}' if ref else ''
            lines.append([base, tag.replace('attn', '').replace('prof', 'profiled-'), n, f'{med:.4f}', f'{lo:.4f}', f'{hi:.4f}',
                          '' if rate is None else f'{rate:.1f}', speed])
    return lines


def main(argv=None):
    argv = sys.argv[1:] if argv is None else argv
    if len(argv) != 2 or argv[0] not in ('checks', 'ttft'):
        raise SystemExit(__doc__)
    if argv[0] == 'checks':
        text, failed = checks(argv[1])
        print(text)
        return 1 if failed else 0
    csv.writer(sys.stdout, lineterminator='\n').writerows(ttft(argv[1]))
    return 0


if __name__ == '__main__':
    sys.exit(main())
