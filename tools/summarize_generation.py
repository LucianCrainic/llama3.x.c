"""Summarize the E1 generation comparisons (tools/compare_generation.py results) of a campaign phase directory.

Reads DIR/gen-<case>-<rep>.json and prints a Markdown summary: one row per comparison (identical yes/no, first divergence,
edit distance, the reference's top-2 margin at the divergence), the item 2b verdict inputs, and the FP32 determinism gate.
The bitwise comparisons of the same directory (compare-*-e1-*.json from compare_engines.py) are listed when present.

Case names carry the meaning, as phase_E1 writes them:
    item2b-<model>-<w8a8|w8a32>-<s1-vs-s2|s1-vs-s1>   W8A8 / W8A32 from a retained context; s1-vs-s1 is the run-to-run control
    fp32-<model>-<prompt>-<s1-vs-s2|pertoken-vs-m64>   FP32 determinism-gate data

Item 2b (E1.2 item 4): a divergence between one and two sockets whose reference top-2 margin is within the W8A8 logit noise
(--noise-floor, rms 0.2) is the noise floor; a larger margin, or any divergence of W8A32 between socket counts, is a bug.
FP32 gate: identical text, or a divergence at a position whose margin is below 2x the observed max |delta logit|
(--max-dlogit, default the 1e-3 budget of the FP32 comparisons).
"""
import argparse
import glob
import json
import os
from pathlib import Path
import re
import sys

NOISE_FLOOR = 0.2
ITEM_2B = re.compile(r'^item2b-(?P<model>.+)-(?P<kind>w8a8|w8a32)-(?P<pair>s1-vs-s2|s1-vs-s1)$')
FILE = re.compile(r'^gen-(?P<case>.+)-(?P<rep>\d+)\.json$')


def load(directory):
    """{case: [(rep, result)]} from DIR/gen-*.json, repeats in order. Files that are not a generation result are skipped."""
    cases = {}
    for path in sorted(glob.glob(os.path.join(directory, 'gen-*.json'))):
        match = FILE.match(os.path.basename(path))
        if not match:
            continue
        try:
            result = json.loads(Path(path).read_text())
            result['identical'], result['first_divergence'], result['edit_distance']
        except (OSError, ValueError, KeyError, TypeError):
            continue
        cases.setdefault(match['case'], []).append((int(match['rep']), result))
    return {case: sorted(runs, key=lambda run: run[0]) for case, runs in cases.items()}


def number(value):
    return '-' if value is None else f'{value:.4g}'


def table(cases):
    lines = ['| case | rep | identical | first divergence | tokens A/B | edit distance | margin_a |',
             '| --- | ---: | --- | ---: | ---: | ---: | ---: |']
    for case in sorted(cases):
        for rep, r in cases[case]:
            where = '-' if r['identical'] else r['first_divergence']
            lines.append(f"| {case} | {rep} | {'yes' if r['identical'] else 'no'} | {where} | "
                         f"{r.get('generated_a', '-')}/{r.get('generated_b', '-')} | {r['edit_distance']} | "
                         f"{number(r.get('margin_a'))} |")
    return lines


def diverged(runs):
    return [r for _, r in runs if not r['identical']]


def item_2b(cases, noise_floor):
    """Verdict inputs and verdict per model. Returns the Markdown lines, or [] when the directory has no item 2b case."""
    models = {}
    for case, runs in cases.items():
        match = ITEM_2B.match(case)
        if match:
            models.setdefault(match['model'], {})[(match['kind'], match['pair'])] = runs
    lines = []
    for model in sorted(models):
        found = models[model]
        lines += [f'### {model}', '']
        problems, notes = [], []
        w8a8, w8a32 = found.get(('w8a8', 's1-vs-s2')), found.get(('w8a32', 's1-vs-s2'))
        control = found.get(('w8a8', 's1-vs-s1'))
        if w8a8 is None:
            lines.append('- W8A8, one socket against two: not run')
        else:
            bad = diverged(w8a8)
            margins = [r['margin_a'] for r in bad if r.get('margin_a') is not None]
            at = [r['first_divergence'] for r in bad]
            lines.append(f'- W8A8, one socket against two: diverged in {len(bad)} of {len(w8a8)} runs'
                         + (f', first divergence at token {at}, margin_a {[round(m, 4) for m in margins]} '
                            f'(noise floor {noise_floor:g})' if bad else ''))
            if bad and len(margins) < len(bad):
                notes.append('a margin is missing (no probe at the divergence)')
            elif any(m > noise_floor for m in margins):
                problems.append(f'W8A8 diverged at a margin above the noise floor {noise_floor:g}')
            stopped = [r.get('generated_a') for _, r in w8a8 if r.get('generated_a', 128) < r.get('steps', 128)]
            if stopped:
                lines.append(f'- W8A8 reference runs that stopped before the step limit: generated tokens {stopped}')
        if w8a32 is None:
            lines.append('- W8A32, one socket against two: not run')
        else:
            bad = diverged(w8a32)
            lines.append(f'- W8A32, one socket against two: diverged in {len(bad)} of {len(w8a32)} runs'
                         + (f', first divergence at token {[r["first_divergence"] for r in bad]}' if bad else ''))
            if bad:
                problems.append('W8A32 diverged between socket counts')
        if control is None:
            lines.append('- control, one socket against one socket: not run')
        else:
            bad = diverged(control)
            lines.append(f'- control, one socket against one socket (run-to-run): diverged in {len(bad)} of {len(control)} runs')
            if bad:
                notes.append('the control diverged: the runs are not deterministic even on one socket')
        if problems:
            verdict = 'BUG (blocks E2): ' + '; '.join(problems)
        elif w8a8 is None or w8a32 is None:
            verdict = 'incomplete'
        elif diverged(w8a8) and not notes:
            verdict = 'noise floor (W8A32 identical between socket counts, W8A8 margin within the floor)'
        elif diverged(w8a8):
            verdict = 'undecided: ' + '; '.join(notes)
        else:
            verdict = 'no divergence reproduced (W8A8 and W8A32 identical between socket counts)' + (
                '; ' + '; '.join(notes) if notes else '')
        lines += ['', f'Verdict ({model}): {verdict}', '']
    return lines


def fp32_gate(cases, max_dlogit):
    """The FP32 determinism gate over every fp32-* case: identical, or diverging only below 2x max |delta logit|."""
    lines = ['| case | rep | result | margin_a | 2x max dlogit |', '| --- | ---: | --- | ---: | ---: |']
    failed = rows = 0
    for case in sorted(cases):
        if not case.startswith('fp32-'):
            continue
        for rep, r in cases[case]:
            rows += 1
            margin = r.get('margin_a')
            if r['identical']:
                result = 'PASS (identical)'
            elif margin is not None and margin < 2 * max_dlogit:
                result = f"PASS (diverges at {r['first_divergence']}, margin within the tolerance)"
            else:
                failed += 1
                result = f"FAIL (diverges at {r['first_divergence']}" + (', no margin)' if margin is None else ')')
            lines.append(f'| {case} | {rep} | {result} | {number(margin)} | {2 * max_dlogit:.3g} |')
    return (lines if rows else []), failed


def bitwise_lines(directory):
    lines = []
    for path in sorted(glob.glob(os.path.join(directory, 'compare-*-e1-*.json'))):
        try:
            data = json.loads(Path(path).read_text())
            lines.append(f"| {os.path.basename(path)[8:-5]} | {'yes' if data['summary']['bitwise'] else 'no'} | "
                         f"{data['summary']['max_abs']:.3g} | {'PASS' if data['passed'] else 'FAIL'} |")
        except (OSError, ValueError, KeyError):
            lines.append(f'| {os.path.basename(path)[8:-5]} | unreadable | - | - |')
    return ['| comparison | bitwise | max abs | gate |', '| --- | --- | ---: | --- |'] + lines if lines else []


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('directory', help='phase directory with gen-*.json')
    parser.add_argument('--noise-floor', type=float, default=NOISE_FLOOR, help='W8A8 logit noise (rms bound), default 0.2')
    parser.add_argument('--max-dlogit', type=float, default=1e-3, help='observed max |delta logit| of the FP32 gate (default 1e-3)')
    parser.add_argument('--output', help='write the Markdown here as well as to stdout')
    args = parser.parse_args(argv)
    cases = load(args.directory)
    out = ['# E1 generation comparisons', '']
    if not cases:
        out += ['No generation comparison results (gen-*.json) were found.', '']
    else:
        out += ['## All comparisons', ''] + table(cases) + ['']
    lines = item_2b(cases, args.noise_floor)
    if lines:
        out += ['## Item 2b: two-socket W8A8 from a retained context', ''] + lines
    gate, failed = fp32_gate(cases, args.max_dlogit)
    if gate:
        out += ['## FP32 determinism gate (recorded)', ''] + gate + ['', f'{failed} of the FP32 comparisons outside the gate.', '']
    bitwise = bitwise_lines(args.directory)
    if bitwise:
        out += ['## Bitwise default check against the baseline build', ''] + bitwise + ['']
    text = '\n'.join(out)
    if args.output:
        Path(args.output).write_text(text)
    print(text, end='')
    return 0


if __name__ == '__main__':
    sys.exit(main())
