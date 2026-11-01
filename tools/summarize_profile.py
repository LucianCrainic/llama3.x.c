"""Summarize LLAMA_PROFILE=1 reports as Markdown: decode time per token by phase,
and effective weight bandwidth of each GEMV phase. Usage: summarize_profile.py DIR."""
import json
import re
import statistics
import sys
from pathlib import Path

PHASES = ['embedding', 'rope_table', 'rmsnorm', 'quantize', 'qkv', 'rope', 'kv_store', 'attention', 'wo',
          'w1_w3_swiglu', 'w2', 'classifier', 'other']


def gemv_bytes(model):
    """Weight bytes read per token by each GEMV phase."""
    c = model['configuration']
    dim, hidden, layers = c['dim'], c['hidden_dim'], c['n_layers']
    head = dim // c['n_heads']
    q, kv = c['n_heads'] * head, c['n_kv_heads'] * head
    per = 4.0 if model['precision'] == 'fp32' else 1.0 + 4.0 / model['quantization_group_size']
    weights = {'qkv': layers * dim * (q + 2 * kv), 'wo': layers * q * dim, 'w1_w3_swiglu': layers * 2 * dim * hidden,
               'w2': layers * hidden * dim, 'classifier': c['vocab_size'] * dim}
    return {k: v * per for k, v in weights.items()}


def main(root):
    groups = {}
    for path in sorted(root.glob('*.json')):
        match = re.fullmatch(r'(.+)-(\d+)\.json', path.name)
        if match:
            groups.setdefault(match.group(1), []).append(json.loads(path.read_text()))
    for label, runs in groups.items():
        runs = [r for r in runs if r['status'] == 'completed' and r.get('profile_seconds')]
        if not runs:
            continue
        decode = [r['profile_seconds']['decode'] for r in runs]
        steps = decode[0]['forward_passes']
        token = statistics.mean(d['forward'] for d in decode) / steps
        traffic = gemv_bytes(runs[0]['model'])
        print(f'\n### {label}\n')
        print(f'{len(runs)} run(s), {steps} decode steps; {token * 1e3:.2f} ms per token '
              f'({1 / token:.2f} tok/s, kernels {runs[0]["build"].get("kernels")}).\n')
        print('| Phase | ms/token | Share | GB/s |')
        print('| --- | ---: | ---: | ---: |')
        gemv = 0.0
        for phase in PHASES:
            seconds = statistics.mean(d[phase] for d in decode) / steps
            rate = f'{traffic[phase] / seconds / 1e9:.0f}' if phase in traffic and seconds > 0 else ''
            gemv += seconds if phase in traffic else 0.0
            print(f'| {phase} | {seconds * 1e3:.3f} | {seconds / token:.1%} | {rate} |')
        total = sum(traffic.values())
        print(f'| all GEMVs | {gemv * 1e3:.3f} | {gemv / token:.1%} | {total / gemv / 1e9:.0f} |')
        print(f'| whole token | {token * 1e3:.3f} | 100% | {total / token / 1e9:.0f} |')


if __name__ == '__main__':
    main(Path(sys.argv[1]))
