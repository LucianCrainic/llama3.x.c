"""Tables of one scripts/gemv-stagger.sbatch STAGE=ab results directory: decode tok/s per cell (median [min-max] over the
repetitions), the knob's effect on each page size, and the per-matrix decode profile of the profiled runs."""
import json
import re
import statistics
import sys
from pathlib import Path

CELLS = ['thp=off-stagger=0', 'thp=off-stagger=1', 'thp=on-stagger=0', 'thp=on-stagger=1']
PHASES = ['qkv', 'wo', 'w1_w3_swiglu', 'w2', 'attention', 'classifier', 'other']


def load(path):
    return json.loads(Path(path).read_text())


def span(values):
    return f'{statistics.median(values):.2f} [{min(values):.2f}-{max(values):.2f}]'


def main(directory):
    out = Path(directory)
    rates = {c: [] for c in CELLS}
    for path in sorted(out.glob('r[0-9]*-*.json')):
        cell = re.sub(r'^r\d+-', '', path.stem)
        if cell in rates:
            rates[cell].append(load(path)['throughput_tokens_per_second']['decode'])
    print('| cell | decode tok/s median [min-max] | runs |')
    print('| --- | --- | --- |')
    for cell in CELLS:
        if rates[cell]:
            print(f'| {cell} | {span(rates[cell])} | {len(rates[cell])} |')
    print()
    for page in ('off', 'on'):
        off, on = rates[f'thp={page}-stagger=0'], rates[f'thp={page}-stagger=1']
        if off and on:
            delta = 100 * (statistics.median(on) / statistics.median(off) - 1)
            pairs = sum(b > a for a, b in zip(off, on))
            print(f'THP {page}: knob on against off {delta:+.1f}% (median ratio; on faster in {pairs} of {min(len(off), len(on))} '
                  f'repetitions)')
    print()
    profiles = {}
    for cell in CELLS:
        path = out / f'prof-{cell}.json'
        if path.exists():
            profiles[cell] = load(path)['profile_seconds']['decode']
    if profiles:
        print('Decode profile, seconds per generated token (one profiled run per cell):')
        print()
        print('| phase | ' + ' | '.join(profiles) + ' |')
        print('| --- |' + ' --- |' * len(profiles))
        for phase in PHASES + ['forward']:
            cells = []
            for profile in profiles.values():
                steps = profile['forward_passes']
                cells.append(f'{profile[phase] / steps * 1e3:.3f} ms')
            print(f'| {phase} | ' + ' | '.join(cells) + ' |')


if __name__ == '__main__':
    main(sys.argv[1])
