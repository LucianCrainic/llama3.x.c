"""Tables for the decode and prefill campaigns of the NUMA / tensor-parallel / batched-prefill plans (B7, D1, D2), built
from the schema-v2 reports and the .stderr files of a campaign directory. Called by summarize_campaign.py; every function
returns Markdown lines and an empty list when the campaign has no data for it.

Definitions (docs/plans/B-tensor-parallel/B7-two-socket-decode-campaign.md section 6, study 8.3 and 10.4):

* Per-socket phase table. A profiled two-socket run (LLAMA_PROFILE=1, log level info) logs one line per stage and socket
  ("tp decode socket S (N tokens, ms/token, ...): qkv=... x_add=..."). Columns are ms per token: qkv, attention, wo, ffn
  (w1_w3 + w2), classifier, reduce (the add of the cross-socket reduction), wait (straggler + peer + close + buffer wait)
  and the total of the engine phases. wo and ffn include the reduction, reduce and wait are its components. The
  imbalance row is |socket 0 - socket 1| per column.
* Revisit triggers (study 10.4): reductions plus waits above 20% of the token time, or one socket waiting for the other
  (the peer wait) above 10%.
* Modeled against measured. Modeled token time = the sum over the compute phases of the slower socket (embedding,
  rmsnorm, qkv, rope, attention, the projections of wo and w2 without their reduction, w1_w3, classifier) plus the larger
  socket's exchange cost (all five wait/add components): "sum of the per-layer max plus exposed exchange". The bandwidth
  ceiling is the checkpoint size over the sockets' stream bandwidth (default 600 GB/s each, tests/stream_bench.c).
* "Faster beyond variability": the slowest repeat of one configuration is faster than the fastest repeat of the other.
"""
import json
from pathlib import Path
import re
import statistics

STREAM_GBPS_PER_SOCKET = 600.0
REVISIT_REDUCE_WAIT = 0.20
REVISIT_PEER_WAIT = 0.10
ATTENTION_DOMINATES = 0.5
PHASES = ('embedding', 'rope_table', 'rmsnorm', 'quantize', 'qkv', 'rope', 'kv_store', 'attention', 'wo', 'w1_w3', 'w2',
          'classifier', 'wo_mm', 'w2_mm')
WAITS = ('x_straggler', 'x_peer', 'x_close', 'x_buffer')
LINE = re.compile(r'tp (decode|prefill) socket (\d) \((\d+) tokens[^)]*\):(.*)$')


def dig(obj, *keys):
    for key in keys:
        if not isinstance(obj, dict) or key not in obj:
            return None
        obj = obj[key]
    return obj


def attention_tag(report):
    """The attention variants of a run (report build.attention for prefill, build.attention_decode for decode), written
    only when they are not the default `split`, so default runs and old reports keep their columns unchanged."""
    parts = [f'{label}={value}' for label, key in (('attn', 'attention'), ('decode-attn', 'attention_decode'))
             for value in [dig(report, 'build', key)] if value not in (None, 'split')]
    return ' '.join(parts)


def number(value):
    return value if isinstance(value, (int, float)) and not isinstance(value, bool) else None


def median(values):
    values = [v for v in values if number(v) is not None]
    return statistics.median(values) if values else None


def fmt(value, digits=2):
    return '-' if value is None else f'{value:.{digits}f}'


def span(values, digits=2):
    values = [v for v in values if number(v) is not None]
    if not values:
        return '-'
    mid = statistics.median(values)
    if len(values) == 1 or min(values) == max(values):
        return f'{mid:.{digits}f}'
    return f'{mid:.{digits}f} [{min(values):.{digits}f}-{max(values):.{digits}f}]'


def faster(a, b):
    """True when every repeat of `a` beats every repeat of `b` (rates: higher is better)."""
    a = [v for v in a if number(v) is not None]
    b = [v for v in b if number(v) is not None]
    return bool(a and b and min(a) > max(b))


def parse_profile(path):
    """{(stage, socket): {'tokens': n, name: ms_per_token}} from the log lines of a profiled run."""
    result = {}
    if not path.exists():
        return result
    for line in path.read_text(errors='replace').splitlines():
        match = LINE.search(line)
        if not match:
            continue
        stage, socket, tokens, rest = match.group(1), int(match.group(2)), int(match.group(3)), match.group(4)
        values = {name: float(value) for name, value in re.findall(r'(\w+)=(-?[0-9.]+(?:e-?\d+)?)', rest)}
        values['tokens'] = tokens
        result[(stage, socket)] = values
    return result


def token_ms(report, stage='decode'):
    """Measured ms per token of a stage from the report."""
    seconds = number(dig(report, 'timing_seconds', stage))
    count = number(dig(report, 'tokens', 'decode_steps')) if stage == 'decode' else number(dig(report, 'prefill', 'tokens'))
    return 1e3 * seconds / count if seconds and count else None


def socket_view(values):
    """The columns of the per-socket table, in ms per token."""
    get = lambda name: values.get(name, 0.0)
    ffn = get('w1_w3') + get('w2')
    reduce = get('x_add')
    wait = sum(get(name) for name in WAITS)
    engine = sum(get(name) for name in PHASES if name not in ('wo_mm', 'w2_mm'))
    return {'qkv': get('qkv'), 'attention': get('attention'), 'wo': get('wo'), 'ffn': ffn,
            'classifier': get('classifier'), 'reduce': reduce, 'wait': wait, 'total': engine}


COLUMNS = ('qkv', 'attention', 'wo', 'ffn', 'classifier', 'reduce', 'wait', 'total')


def socket_table(runs, stage='decode'):
    """runs: [(label, report, profile)] of profiled runs. One block per run: socket 0, socket 1 and the imbalance."""
    lines = []
    for label, report, profile in runs:
        sockets = {s: socket_view(profile[(stage, s)]) for s in (0, 1) if (stage, s) in profile}
        if not sockets:
            continue
        if not lines:
            lines += [f'### Per-socket {stage} phase times (ms per token)', '',
                      'wo and ffn include their reduction; reduce and wait are its components (wait = straggler + peer + '
                      'close + buffer). Imbalance = |socket 0 - socket 1|.', '',
                      '| Run | Socket | ' + ' | '.join(COLUMNS) + ' |', '| --- | ---: | ' + ' | '.join('---:' for _ in COLUMNS) + ' |']
        for s, view in sockets.items():
            lines.append(f'| {label} | {s} | ' + ' | '.join(fmt(view[c], 3) for c in COLUMNS) + ' |')
        if len(sockets) == 2:
            lines.append(f'| {label} | imbalance | ' + ' | '.join(fmt(abs(sockets[0][c] - sockets[1][c]), 3) for c in COLUMNS) + ' |')
    return lines + [''] if lines else []


def revisit_flags(report, profile, stage='decode'):
    """(reduce+wait share, peer-wait share, flags) of socket 0 against the measured token time, or None."""
    measured = token_ms(report, stage)
    view = profile.get((stage, 0))
    if not measured or not view:
        return None
    reduce_wait = (view.get('x_add', 0.0) + sum(view.get(name, 0.0) for name in WAITS)) / measured
    peer = view.get('x_peer', 0.0) / measured
    flags = []
    if reduce_wait > REVISIT_REDUCE_WAIT:
        flags.append(f'reductions+waits {100 * reduce_wait:.0f}% > {100 * REVISIT_REDUCE_WAIT:.0f}%')
    if peer > REVISIT_PEER_WAIT:
        flags.append(f'peer wait {100 * peer:.0f}% > {100 * REVISIT_PEER_WAIT:.0f}%')
    return reduce_wait, peer, flags


def modeled_ms(profile, stage='decode'):
    """Sum over the compute phases of the slower socket plus the larger socket's exchange cost, ms per token."""
    views = [profile[(stage, s)] for s in (0, 1) if (stage, s) in profile]
    if len(views) < 2:
        return None
    compute = ('embedding', 'rope_table', 'rmsnorm', 'quantize', 'qkv', 'rope', 'kv_store', 'attention', 'wo_mm', 'w1_w3',
               'w2_mm', 'classifier')
    total = sum(max(v.get(name, 0.0) for v in views) for name in compute)
    exchange = max(v.get('x_add', 0.0) + sum(v.get(name, 0.0) for name in WAITS) for v in views)
    return total + exchange


def model_file_bytes(report):
    return number(dig(report, 'memory_bytes', 'model_file'))


def modeled_table(runs, stream_gbps=STREAM_GBPS_PER_SOCKET):
    """Two-socket profiled decode runs: modeled against measured token time, and against the bandwidth ceiling."""
    rows = []
    for label, report, profile in runs:
        modeled = modeled_ms(profile)
        measured = token_ms(report)
        if modeled is None or measured is None:
            continue
        bytes_ = model_file_bytes(report)
        sockets = dig(report, 'runtime', 'sockets') or 2
        ceiling = 1e3 * bytes_ / (sockets * stream_gbps * 1e9) if bytes_ else None
        flags = revisit_flags(report, profile)
        rows.append(f'| {label} | {fmt(measured)} | {fmt(modeled)} | {fmt(100 * modeled / measured, 0)}% | {fmt(ceiling)} | '
                    f'{fmt(100 * ceiling / measured, 0) + "%" if ceiling else "-"} | '
                    f'{"; ".join(flags[2]) if flags and flags[2] else "none" if flags else "-"} |')
    if not rows:
        return []
    return ['### Modeled against measured decode time (ms per token)', '',
            f'Modeled = sum of the slower socket per compute phase + the larger socket\'s exchange cost; ceiling = checkpoint '
            f'bytes / (sockets x {stream_gbps:.0f} GB/s).', '',
            '| Run | Measured | Modeled | Modeled/measured | Bandwidth ceiling | Ceiling/measured | Revisit triggers |',
            '| --- | ---: | ---: | ---: | ---: | ---: | --- |'] + rows + ['']


def decode_scaling(groups):
    """Decode tok/s by model, precision and retained context for each configuration, with the two-socket gain.

    groups: {label: (info, [reports])}; only runs with a warm context (parameters.warm_context > 0) and no tag other than
    the control count."""
    cells = {}
    for label, (info, reports) in groups.items():
        warm = median([dig(r, 'parameters', 'warm_context') for r in reports]) or 0
        if not warm or (info['tag'] not in (None, 'ctl', 'dec', 'p4')):
            continue
        config = ('control' if info['tag'] == 'ctl' else f"{info['sockets']}x{info['threads']}" +
                  ('p4' if info['tag'] == 'p4' else ''))
        key = (info['model'], info['precision'], int(warm))
        cells.setdefault(key, {})[config] = [dig(r, 'throughput_tokens_per_second', 'decode') for r in reports]
    if not cells:
        return []
    configs = ['1x56', '2x14', '2x28', '2x42', '2x56', '2x56p4', 'control']
    lines = ['### Decode tok/s by retained context', '',
             'Median [range] of the repeats. Gain = two sockets x 56 over one socket x 56; "yes" when every repeat of the '
             'former beats every repeat of the latter.', '',
             '2x56p4 is two sockets x 56 decoding from the packed4 copy of the FP32 shards (LLAMA_FP32_LAYOUT=packed4).', '',
             '| Model | Precision | Context | ' + ' | '.join(configs) + ' | Gain 2x56 / 1x56 | Beyond variability | packed4 / rows |',
             '| --- | --- | ---: | ' + ' | '.join('---:' for _ in configs) + ' | ---: | --- | ---: |']
    for (model, precision, warm), by_config in sorted(cells.items()):
        one, two = by_config.get('1x56'), by_config.get('2x56')
        gain = median(two) / median(one) if one and two and median(one) else None
        lines.append(f'| {model} | {precision} | {warm} | ' + ' | '.join(span(by_config.get(c, [])) if c in by_config else '-'
                                                                            for c in configs) +
                     f' | {fmt(gain)} | {"-" if gain is None else "yes" if faster(two, one) else "no"} | '
                     f'{fmt(median(by_config["2x56p4"]) / median(two)) if "2x56p4" in by_config and two and median(two) else "-"} |')
    return lines + ['']


def prefill_speedup(groups):
    """Prefill tok/s and TTFT by model, precision, prompt and sockets for each rung, against the per-token baseline of one
    socket (the kernel tags per_token / per_token_q8), and where two sockets win."""
    cells = {}
    for label, (info, reports) in groups.items():
        if info['tag'] not in (None,) or not info['prompt'] or (dig(reports[0], 'parameters', 'warm_context') or 0):
            continue
        kernel = ' '.join(part for part in (info['kernel'] or '-', attention_tag(reports[0])) if part)
        key = (info['model'], info['precision'], int(info['prompt']), kernel)
        cells.setdefault(key, {})[(info['sockets'], int(info['m'] or 1))] = (
            [dig(r, 'throughput_tokens_per_second', 'prefill') for r in reports],
            [dig(r, 'timing_seconds', 'time_to_first_token') for r in reports])
    if not cells:
        return []
    baseline = {}
    for (model, precision, prompt, kernel), by in cells.items():
        if kernel in ('per_token', 'per_token_q8') and ('1', 1) in by:
            baseline[(model, precision, prompt)] = median(by[('1', 1)][0])
    lines = ['### Prefill by rung, microbatch and sockets', '',
             'Speedup = median prefill tok/s over the one-socket per-token baseline of the same model, precision and '
             'prompt; "2s wins" compares the same rung and M on two sockets with one socket, beyond variability.', '',
             '| Model | Precision | Prompt | Rung | M | Sockets | Prefill tok/s | TTFT s | Speedup vs 1-socket per-token | 2s wins |',
             '| --- | --- | ---: | --- | ---: | ---: | ---: | ---: | ---: | --- |']
    for (model, precision, prompt, kernel), by in sorted(cells.items()):
        for (sockets, m), (rates, ttfts) in sorted(by.items()):
            base = baseline.get((model, precision, prompt))
            speed = median(rates) / base if base and median(rates) else None
            other = by.get(('1', m)) if sockets == '2' else None
            wins = '-' if other is None else 'yes' if faster(rates, other[0]) else 'no'
            lines.append(f'| {model} | {precision} | {prompt} | {kernel} | {m} | {sockets} | {span(rates)} | {span(ttfts, 3)} | '
                         f'{fmt(speed)} | {wins} |')
    return lines + ['']


def attention_share(report, stage):
    forward = number(dig(report, 'profile_seconds', stage, 'forward'))
    attention = number(dig(report, 'profile_seconds', stage, 'attention'))
    return attention / forward if forward and attention is not None else None


def crossover(groups, profiled):
    """Where two sockets win for decode and for prefill, and where synchronization or attention dominates."""
    lines = []
    decode = {}
    for label, (info, reports) in groups.items():
        warm = median([dig(r, 'parameters', 'warm_context') for r in reports]) or 0
        if warm and info['tag'] in (None, 'dec') and info['threads'] == '56':
            decode.setdefault((info['model'], info['precision'], int(warm)), {})[info['sockets']] = [
                dig(r, 'throughput_tokens_per_second', 'decode') for r in reports]
    rows = []
    for key, by in sorted(decode.items()):
        if '1' in by and '2' in by:
            gain = median(by['2']) / median(by['1']) if median(by['1']) else None
            rows.append((key, gain, faster(by['2'], by['1'])))
    if rows:
        lines += ['### Crossover: decode', '', '| Model | Precision | Context | Gain | Two sockets win |', '| --- | --- | ---: | ---: | --- |']
        lines += [f'| {m} | {p} | {c} | {fmt(g)} | {"yes" if win else "no"} |' for (m, p, c), g, win in rows]
        lines.append('')
    dominated = []
    for label, report, profile in profiled:
        flags = revisit_flags(report, profile)
        share = attention_share(report, 'decode')
        note = []
        if flags and flags[2]:
            note.append('synchronization: ' + '; '.join(flags[2]))
        if share is not None and share > ATTENTION_DOMINATES:
            note.append(f'attention {100 * share:.0f}% of the token')
        if note:
            dominated.append(f'| {label} | {"; ".join(note)} |')
    if dominated:
        lines += ['### Where synchronization or attention dominates (profiled decode runs)', '', '| Run | Why |', '| --- | --- |']
        lines += dominated + ['']
    return lines


def build(groups, root):
    """All the campaign tables for `groups` ({label: (info, [reports])}) in directory `root`."""
    profiled = []
    for label, (info, reports) in groups.items():
        if info['tag'] and info['tag'].endswith('prof') and reports and number(dig(reports[0], 'schema_version')):
            profile = parse_profile(Path(root) / f'{label}-1.stderr')
            if profile:
                profiled.append((label, reports[0], profile))
    lines = []
    lines += decode_scaling(groups)
    lines += prefill_speedup(groups)
    lines += socket_table([p for p in profiled if ('decode', 0) in p[2]], 'decode')
    lines += socket_table([p for p in profiled if ('prefill', 0) in p[2]], 'prefill')
    lines += modeled_table([p for p in profiled if ('decode', 1) in p[2]])
    lines += crossover(groups, [p for p in profiled if ('decode', 0) in p[2]])
    return lines
