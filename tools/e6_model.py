"""E6 batched-decode model: decode weight shapes, KV capacity, and the step-time model built from measured benchmark cells.

    e6_model.py shapes MODEL SOCKETS          the decode weight shapes (N x K, comma separated) gemm-bench is run on
    e6_model.py kv MODEL                      FP32 K/V bytes per token and per request at the usual contexts
    e6_model.py model E6_DIR [--runs DIR] [--json]
                                              the tables of docs/plans/E-followups/E6-batched-decode-design.md, from the
                                              output directory of `PHASE=E6 scripts/cresco8-e6.sbatch`

The model (one decode step of N requests, each at its own context C, on S sockets):

    step(N, C) = L * (qkv(N) + wo(N) + ffn(N)) + classifier(N) + N * L * attention(C) + N * other + exchange(N)

where qkv, wo, ffn and classifier are the MEASURED gemm-bench times (median of cold-weight repetitions) of the matching
decode shape at M = N rows on the kernel decode would use (they already contain the weight read: the bench streams the
weights from HBM, so the weight read is not added again), attention(C) is the MEASURED attention-bench decode time of one
layer (one request; N requests are charged N times, an upper bound), `other` is the per-token remainder of the profiled N = 1
engine run (embedding, norms, RoPE, KV store, quantization, what the profile does not name; charged per request) and
exchange(N) is the MODELED cross-socket reduction (two sockets only). Every input carries its provenance in the design
document (measured, modeled, estimated).
"""
import argparse
import json
import statistics
import sys
from pathlib import Path

# Llama 3 shapes of the repository's exporter (8B: Llama-3-8B; 1B: Llama-3.2-1B); both have 8 KV heads and a 128256-token vocabulary.
MODELS = {
    '8b': dict(dim=4096, hidden=14336, layers=32, heads=32, kv_heads=8, head_dim=128, vocab=128256),
    '1b': dict(dim=2048, hidden=8192, layers=16, heads=32, kv_heads=8, head_dim=64, vocab=128256),
}
PHASES = ('qkv', 'wo', 'w13', 'w2', 'cls')
CONTEXTS = (512, 2048, 8192, 32768)
NS = (1, 2, 4, 8, 16, 32)
HBM_SOCKET_GIB = 64  # 64 GiB of HBM per socket (docs/hbm-roofline.md)
POLICIES = (('fp32', 'avx512'), ('int8', 'avx512'), ('int8', 'amx'))
SOCKET_COUNTS = (1, 2)


def decode_shapes(model, sockets):
    """(phase, n_out, k_in) of the matrices one socket streams per layer (the classifier once per step). Two sockets:
    QKV, W1/W3 and the classifier split by output rows, Wo and W2 by input columns (the head and FFN-channel shards)."""
    c = MODELS[model]
    dim, hidden, kv_dim, vocab = c['dim'], c['hidden'], c['kv_heads'] * c['head_dim'], c['vocab']
    if sockets == 1:
        return [('qkv', dim + 2 * kv_dim, dim), ('wo', dim, dim), ('w13', 2 * hidden, dim), ('w2', dim, hidden), ('cls', vocab, dim)]
    if sockets == 2:
        return [('qkv', (dim + 2 * kv_dim) // 2, dim), ('wo', dim, dim // 2), ('w13', hidden, dim), ('w2', dim, hidden // 2),
                ('cls', vocab // 2, dim)]
    raise ValueError(f'sockets must be 1 or 2, got {sockets}')


def kv_bytes_per_token(model, kv_bytes=4):
    """FP32 K and V for every layer: 2 x layers x kv_heads x head_dim x 4 bytes (8B: 256 KiB, 1B: 64 KiB)."""
    c = MODELS[model]
    return 2 * c['layers'] * c['kv_heads'] * c['head_dim'] * kv_bytes


def weight_bytes(model, precision):
    """Checkpoint bytes read per decode token (all linear weights and the classifier); Q8_0 adds one FP32 scale per 64."""
    c = MODELS[model]
    per_layer = sum(n * k for _, n, k in decode_shapes(model, 1)[:4])
    values = c['layers'] * per_layer + c['vocab'] * c['dim']
    return values * 4 if precision == 'fp32' else values + values // 64 * 4


# ---- reading the E6 job output ---------------------------------------------------------------------------------------

def read_jsonl(path):
    path = Path(path)
    return [json.loads(line) for line in path.read_text().splitlines() if line.startswith('{')] if path.exists() else []


def gemm_cells(directory, model, sockets, tag, gemv=False):
    """{phase: {m: seconds}}: the median time of the gemm-bench cell of each decode shape of one (model, sockets, kernel) set;
    gemv=True reads the per-row GEMV baseline (m calls of the single-row kernel) instead of the batched GEMM. On two sockets both
    sockets ran at once and a phase takes the slower socket."""
    cells = {}
    for suffix in ('', '-n3'):
        rows = read_jsonl(Path(directory)/f'gemm-{tag}-{model}-s{sockets}{suffix}.out')
        for phase, n, k in decode_shapes(model, sockets):
            for r in rows:
                if (r['n'], r['k']) == (n, k) and r['kernel'].startswith('gemv') == gemv:
                    slot = cells.setdefault(phase, {})
                    slot[r['m']] = max(slot.get(r['m'], 0.0), r['median_s'])
    return {phase: slot for phase, slot in cells.items() if slot}


def best_cells(*cell_sets):
    """The cell-wise minimum of several {phase: {m: seconds}} sets: decode takes the faster kernel at each M."""
    out = {}
    for cells in cell_sets:
        for phase, slot in cells.items():
            merged = out.setdefault(phase, {})
            for m, t in slot.items():
                merged[m] = min(merged.get(m, t), t)
    return out


def decode_cells(directory, model, sockets, precision, policy):
    """The GEMM times the decode of one configuration would use. fp32: the AVX-512 FP32 GEMM or the per-row GEMV, whichever is
    faster at each M. int8: the VNNI Q8 GEMM or GEMV; policy 'amx' adds the AMX-INT8 GEMM (which needs the packed int8 copy)."""
    if precision == 'fp32':
        sets = [gemm_cells(directory, model, sockets, 'f32'), gemm_cells(directory, model, sockets, 'f32', gemv=True)]
    else:
        sets = [gemm_cells(directory, model, sockets, 'q8v'), gemm_cells(directory, model, sockets, 'q8v', gemv=True)]
        if policy == 'amx':
            sets.append(gemm_cells(directory, model, sockets, 'q8amx'))
    cells = best_cells(*sets)
    return cells if all(p in cells for p in PHASES) else {}


def attention_cells(directory):
    """{(shape, kv_heads, kernel, context): median ms per layer} from the attention-*.out files (kv_heads is the local count
    the cell ran with, --kv-heads-local; kv_heads_model is the model's)."""
    cells = {}
    for path in sorted(Path(directory).glob('attention-*.out')):
        for r in read_jsonl(path):
            if r.get('mode') == 'decode':
                cells[(r['shape'], r['kv_heads'], r['attn_decode'], r['context'])] = r['ms_layer_median']
    return cells


# ---- engine runs (N = 1): the measured token and its profile ------------------------------------------------------------

def engine_token_ms(runs_dir, label, precision, sockets, context):
    """(median, min, max, repeats) of ms per decode token over the timed repeats of one cell, or None."""
    values = []
    for path in sorted(Path(runs_dir).glob(f'{label}-{precision}-teams-hbm-s{sockets}-t56-p{context}-xe6-*.json')):
        report = json.loads(path.read_text())
        seconds = report.get('timing_seconds', {}).get('decode')
        steps = report.get('tokens', {}).get('decode_steps')
        if report.get('status') == 'completed' and seconds and steps:
            values.append(1e3 * seconds / steps)
    return (statistics.median(values), min(values), max(values), len(values)) if values else None


def engine_profile_ms(runs_dir, label, precision, sockets, context=2048):
    """{qkv, wo, ffn, classifier, attention, total}: ms per decode token of the profiled run. One socket: the report's profile
    section. Two sockets: the log lines, the slower socket per phase (wo and ffn include their reduction)."""
    stem = f'{label}-{precision}-teams-hbm-s{sockets}-t56-p{context}-xe6prof-1'
    if sockets == 1:
        path = Path(runs_dir)/f'{stem}.json'
        if not path.exists():
            return None
        d = json.loads(path.read_text()).get('profile_seconds', {}).get('decode')
        if not d:
            return None
        ms = {k: 1e3 * v / d['forward_passes'] for k, v in d.items() if isinstance(v, (int, float))}
        return {'qkv': ms['qkv'], 'wo': ms['wo'], 'ffn': ms['w1_w3_swiglu'] + ms['w2'], 'classifier': ms['classifier'],
                'attention': ms['attention'], 'total': ms['forward']}
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    import campaign_tables as ct
    profile = ct.parse_profile(Path(runs_dir)/f'{stem}.stderr')
    views = [ct.socket_view(profile[('decode', s)]) for s in (0, 1) if ('decode', s) in profile]
    if len(views) < 2:
        return None
    return {k: max(v[k] for v in views) for k in ('qkv', 'wo', 'ffn', 'classifier', 'attention', 'total')}


# ---- the model --------------------------------------------------------------------------------------------------------

def at_m(points, n):
    """Seconds at M = n from {m: seconds}; linear in M between measured values, flat outside them."""
    points = sorted(points.items())
    for (m0, t0), (m1, t1) in zip(points, points[1:]):
        if m0 <= n <= m1:
            return t0 + (t1 - t0) * (n - m0) / (m1 - m0)
    return points[-1][1] if n > points[-1][0] else points[0][1]


def gemm_seconds(model, gemm, n):
    """The weight-streaming part of one decode step at M = n: every layer's qkv, wo, w1/w3 and w2 plus the classifier."""
    c = MODELS[model]
    return c['layers'] * sum(at_m(gemm[p], n) for p in ('qkv', 'wo', 'w13', 'w2')) + at_m(gemm['cls'], n)


def step_seconds(model, gemm, attention_ms_layer, n, other_s=0.0, exchange_s=0.0):
    """One decode step of n requests at the same context: GEMM phases at M = n, n times the one-request attention, n times the
    per-request remainder, plus the exchange. `gemm` is {phase: {m: seconds}}."""
    c = MODELS[model]
    return gemm_seconds(model, gemm, n) + n * c['layers'] * attention_ms_layer * 1e-3 + n * other_s + exchange_s


def exchange_seconds(model, n, latency_us=5.0, gbps=50.0):
    """MODELED cross-socket reduction of n rows: two exchanges per layer, each a fixed latency plus n x dim x 4 bytes read from the
    peer socket at gbps (docs/hbm-roofline.md: remote HBM read 50-112 GB/s). The fixed part is what B7 measured at n = 1 (a few us)."""
    c = MODELS[model]
    return 2 * c['layers'] * (latency_us * 1e-6 + n * c['dim'] * 4 / (gbps * 1e9))


def throughput(step_s, n):
    """(aggregate tokens per second, per-request latency in seconds per token) of a step of n requests."""
    return n / step_s, step_s


def kv_gib_per_request(model, context):
    return kv_bytes_per_token(model) * context / 2**30


def fit_requests(model, precision, context, sockets, reserve_gib=4.0):
    """Requests whose KV fits in HBM next to the weights: sockets x 64 GiB minus the checkpoint (split across the sockets) and a
    reserve for the work region, exchange buffers and packed copies (ESTIMATED 4 GiB)."""
    total = sockets * HBM_SOCKET_GIB - weight_bytes(model, precision) / 2**30 - reserve_gib
    return max(0, int(total // kv_gib_per_request(model, context)))


def other_seconds(profile):
    """Per-request remainder of the token, from the profiled N = 1 run at 2048 tokens: the engine's total minus its five named
    phases (embedding, norms, RoPE, KV store, quantize and whatever the profile does not name)."""
    if not profile:
        return 0.0
    named = sum(profile[k] for k in ('qkv', 'wo', 'ffn', 'classifier', 'attention'))
    return max(0.0, 1e-3 * (profile['total'] - named))


def report(directory, runs_dir=None, ns=NS):
    directory = Path(directory)
    runs_dir = Path(runs_dir) if runs_dir else directory.parent
    attention = attention_cells(directory)
    label_of = {}
    for path in runs_dir.glob('*-teams-hbm-s*-xe6-*.json'):
        label = path.name.split('-teams-')[0].rsplit('-', 1)[0]
        for model in MODELS:
            if label.lower().endswith(model):
                label_of[model] = label
    rows, repro, phases = [], [], []
    for model in MODELS:
        label = label_of.get(model, model)
        for precision, policy in POLICIES:
            engine_precision = 'fp32' if precision == 'fp32' else 'int8'
            for sockets in SOCKET_COUNTS:
                gemm = decode_cells(directory, model, sockets, precision, policy)
                if not gemm:
                    continue
                kv = 8 if sockets == 1 else 4
                profile = engine_profile_ms(runs_dir, label, engine_precision, sockets)
                other = other_seconds(profile)
                for kernel in ('split', 'group'):
                    for context in CONTEXTS:
                        a = attention.get((model, kv, kernel, context))
                        if a is None:
                            continue
                        for n in ns:
                            ex = exchange_seconds(model, n) if sockets == 2 else 0.0
                            step = step_seconds(model, gemm, a, n, other, ex)
                            rows.append(dict(model=model, precision=precision, policy=policy, sockets=sockets, kernel=kernel,
                                             context=context, n=n, step_ms=1e3 * step, agg_tok_s=n / step, exchange_ms=1e3 * ex,
                                             gemm_ms=1e3 * gemm_seconds(model, gemm, n),
                                             attention_ms=n * MODELS[model]['layers'] * a, other_ms=1e3 * n * other))
                for context in (512, 2048):
                    token = engine_token_ms(runs_dir, label, engine_precision, sockets, context)
                    a = attention.get((model, kv, 'split', context))
                    if token and a is not None:
                        modeled = 1e3 * step_seconds(model, gemm, a, 1, other, 0.0)
                        repro.append(dict(model=model, precision=precision, policy=policy, sockets=sockets, context=context,
                                          measured_ms=token[0], range=(token[1], token[2]), modeled_ms=modeled,
                                          residual=(modeled - token[0]) / token[0]))
                if profile:
                    layers = MODELS[model]['layers']
                    a = attention.get((model, kv, 'split', 2048))
                    phases.append(dict(model=model, precision=precision, policy=policy, sockets=sockets,
                                       qkv=(1e3 * layers * at_m(gemm['qkv'], 1), profile['qkv']),
                                       wo=(1e3 * layers * at_m(gemm['wo'], 1), profile['wo']),
                                       ffn=(1e3 * layers * (at_m(gemm['w13'], 1) + at_m(gemm['w2'], 1)), profile['ffn']),
                                       classifier=(1e3 * at_m(gemm['cls'], 1), profile['classifier']),
                                       attention=(None if a is None else layers * a, profile['attention'])))
    return dict(rows=rows, repro=repro, phases=phases)


def gate(rows, model='8b', precision='int8', context=2048, n=8):
    """E6.3: aggregate tok/s at N = n over N = 1 (>= 2x) with the per-request latency at most 1.5x the single-request token.
    One result per (policy, sockets, attention kernel)."""
    out = []
    for r in rows:
        if (r['model'], r['precision'], r['context'], r['n']) != (model, precision, context, n):
            continue
        base = next(b for b in rows if (b['model'], b['precision'], b['policy'], b['sockets'], b['kernel'], b['context'], b['n']) ==
                    (model, precision, r['policy'], r['sockets'], r['kernel'], context, 1))
        gain, latency = r['agg_tok_s'] / base['agg_tok_s'], r['step_ms'] / base['step_ms']
        out.append(dict(policy=r['policy'], sockets=r['sockets'], kernel=r['kernel'], gain=gain, latency_ratio=latency,
                        throughput_ok=gain >= 2.0, latency_ok=latency <= 1.5, passes=gain >= 2.0 and latency <= 1.5))
    return out


def markdown(result):
    rows = result['rows']
    out = ['### Aggregate tok/s (and step time in ms) by N, attention `split`', '']
    for model in MODELS:
        for precision, policy in POLICIES:
            for sockets in SOCKET_COUNTS:
                sel = [r for r in rows if (r['model'], r['precision'], r['policy'], r['sockets'], r['kernel']) ==
                       (model, precision, policy, sockets, 'split')]
                if not sel:
                    continue
                out += [f'**{model} {precision} ({policy}), {sockets} socket(s)**', '',
                        '| Context | ' + ' | '.join(f'N={n}' for n in NS) + ' |', '| ---: | ' + ' | '.join('---:' for _ in NS) + ' |']
                for context in CONTEXTS:
                    cells = []
                    for n in NS:
                        r = next((x for x in sel if x['context'] == context and x['n'] == n), None)
                        cells.append(f"{r['agg_tok_s']:.1f} ({r['step_ms']:.1f})" if r else '-')
                    out.append(f'| {context} | ' + ' | '.join(cells) + ' |')
                out.append('')
    out += ['### N = 1 reproduction (model against the engine run of this job)', '',
            '| Model | Precision | Policy | Sockets | Context | Measured ms [range] | Modeled ms | Residual |',
            '| --- | --- | --- | ---: | ---: | --- | ---: | ---: |']
    for r in result['repro']:
        out.append(f"| {r['model']} | {r['precision']} | {r['policy']} | {r['sockets']} | {r['context']} | {r['measured_ms']:.2f} "
                   f"[{r['range'][0]:.2f}-{r['range'][1]:.2f}] | {r['modeled_ms']:.2f} | {100 * r['residual']:+.1f}% |")
    out += ['', '### N = 1 per phase at 2048 tokens (bench x layers / profiled engine run, ms per token)', '',
            '| Model | Precision | Policy | Sockets | qkv | wo | ffn | classifier | attention (split) |',
            '| --- | --- | --- | ---: | --- | --- | --- | --- | --- |']
    for p in result['phases']:
        cell = lambda v: '-' if v[0] is None else f'{v[0]:.2f} / {v[1]:.2f}'
        out.append(f"| {p['model']} | {p['precision']} | {p['policy']} | {p['sockets']} | {cell(p['qkv'])} | {cell(p['wo'])} | "
                   f"{cell(p['ffn'])} | {cell(p['classifier'])} | {cell(p['attention'])} |")
    out += ['', '### E6.3 gate: 8B W8A8, context 2048, N = 8', '',
            '| Policy | Sockets | Attention | Aggregate gain over N = 1 | Latency / single | Throughput >= 2x | Latency <= 1.5x | Gate |',
            '| --- | ---: | --- | ---: | ---: | --- | --- | --- |']
    for g in gate(rows):
        out.append(f"| {g['policy']} | {g['sockets']} | {g['kernel']} | {g['gain']:.2f} | {g['latency_ratio']:.2f} | "
                   f"{'yes' if g['throughput_ok'] else 'no'} | {'yes' if g['latency_ok'] else 'no'} | {'pass' if g['passes'] else 'FAIL'} |")
    return '\n'.join(out)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest='command', required=True)
    p = sub.add_parser('shapes')
    p.add_argument('model', choices=sorted(MODELS))
    p.add_argument('sockets', type=int, choices=(1, 2))
    p = sub.add_parser('kv')
    p.add_argument('model', choices=sorted(MODELS))
    p = sub.add_parser('model')
    p.add_argument('directory', help='the E6 directory of the job (bench outputs); engine reports are read from its parent')
    p.add_argument('--runs', help='directory of the engine reports (default: the parent of DIRECTORY)')
    p.add_argument('--json', action='store_true')
    args = parser.parse_args(argv)
    if args.command == 'shapes':
        print(','.join(f'{n}x{k}' for _, n, k in decode_shapes(args.model, args.sockets)))
    elif args.command == 'kv':
        print(json.dumps({'bytes_per_token': kv_bytes_per_token(args.model),
                          'gib_per_request': {str(c): kv_gib_per_request(args.model, c) for c in CONTEXTS}}))
    else:
        result = report(args.directory, args.runs)
        print(json.dumps(result, indent=1) if args.json else markdown(result))
    return 0


if __name__ == '__main__':
    sys.exit(main())
