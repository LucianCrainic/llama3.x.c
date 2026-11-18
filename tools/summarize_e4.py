"""Summaries of the E4 (prefill attention campaign) cluster phase, as Markdown.

    summarize_e4.py checks E4_DIR    table of every compare-*.json of the check stage (same as summarize_e2.py checks, E4 title);
                                     exit status 1 when any comparison failed
    summarize_e4.py bench OUT_DIR    the bench matrix from the engine reports in OUT_DIR whose label carries the tag attn<variant>
                                     (timed repeats, 32 generated tokens) or prof<variant> (the profiled repeat), as phase_E4 writes
                                     them: per model / rung / M / prompt / sockets / attention the TTFT median and range, the speedup
                                     against split, the attention share of the profiled repeat, the decode tokens/s median and range;
                                     then the E4.3 gates of docs/plans/E-followups/README.md:
        value      8B, 8192 tokens, one socket, best variant against split >= 1.5x TTFT (headline: the AMX-BF16 rung; every rung
                   is listed; "beyond variability" when the ranges do not overlap)
        2048       not slower: the variant's median TTFT is within the split range (not above its slowest repeat)
        decode     the variant's median decode tokens/s lies inside the split range
        placement  every report's placement verified
        two sockets  the TTFT gain of two sockets over one at 8192 per rung and variant (item 2a of the plan)
    The exit status of bench is 0: a missed gate is a result, written in the text.
"""
import json
import re
import statistics
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import summarize_campaign as sc  # noqa: E402
import summarize_e2  # noqa: E402

TAG = re.compile(r'^(?P<kind>attn|prof)(?P<attn>[a-z0-9]+)$')
SPEEDUP_GATE = 1.5  # E4.3: 30.6 s -> 20.4 s at 8192 tokens
HEADLINE_RUNG = 'amx_bf16'
HEADLINE_MODEL = '8b'
ORDER = {'split': 0, 'tiled': 1, 'amx': 2}


def checks(directory):
    text, failed = summarize_e2.checks(directory)
    return text.replace('# E2 correctness', '# E4 correctness', 1), failed


def number(value):
    return value if isinstance(value, (int, float)) and not isinstance(value, bool) else None


def stats(values):
    """(median, min, max, n) of the numeric values, or None."""
    values = [v for v in values if number(v) is not None]
    return (statistics.median(values), min(values), max(values), len(values)) if values else None


def span(stat, digits=2):
    if stat is None:
        return '-'
    median, low, high, _ = stat
    if low == high:
        return f'{median:.{digits}f}'
    return f'{median:.{digits}f} [{low:.{digits}f}-{high:.{digits}f}]'


def load(root):
    """{(model, kernel, m, prompt, sockets): {attn: cell}} from the reports; cell has lists ttft, decode, share, placement and the
    count of failed runs."""
    cells, failed_runs = {}, 0
    for path in sorted(Path(root).glob('*.json')):
        info = sc.parse_label(path.stem)
        tag = TAG.match(info['tag'] or '') if info else None
        if not tag or not info['kernel'] or not info['m'] or not info['prompt']:
            continue
        try:
            report = json.loads(path.read_text())
        except ValueError:
            continue
        key = (info['model'], info['kernel'], int(info['m']), int(info['prompt']), int(info['sockets']))
        cell = cells.setdefault(key, {}).setdefault(tag['attn'], dict(ttft=[], decode=[], share=[], placement=[], failed=0))
        if report.get('status') != 'completed':
            cell['failed'] += 1
            failed_runs += 1
            continue
        cell['placement'].append(sc.placement_status(report))
        if tag['kind'] == 'prof':
            stage = sc.dig(report, 'profile_seconds', 'prefill') or {}
            forward, attention = number(stage.get('forward')), number(stage.get('attention'))
            if forward and attention is not None:
                cell['share'].append(attention / forward)
        else:
            cell['ttft'].append(sc.dig(report, 'timing_seconds', 'time_to_first_token'))
            cell['decode'].append(sc.dig(report, 'throughput_tokens_per_second', 'decode'))
    return cells, failed_runs


def ordered(attns):
    return sorted(attns, key=lambda a: (ORDER.get(a, 9), a))


def table(cells):
    lines = ['## Prefill attention matrix', '',
             'TTFT and decode as median [min-max] over the timed repeats; speedup = split median TTFT over the variant median; '
             'attention share = attention seconds over forward seconds of the profiled repeat (prefill).', '',
             '| model | rung | M | prompt | sockets | attention | repeats | TTFT s | speedup vs split | attention share | decode tok/s |',
             '| --- | --- | ---: | ---: | ---: | --- | ---: | ---: | ---: | ---: | ---: |']
    for key in sorted(cells, key=lambda k: (k[0], k[1], k[3], k[4], k[2])):
        model, kernel, m, prompt, sockets = key
        split = stats(cells[key].get('split', {}).get('ttft', []))
        for attn in ordered(cells[key]):
            cell = cells[key][attn]
            ttft, decode, share = stats(cell['ttft']), stats(cell['decode']), stats(cell['share'])
            speed = f'{split[0] / ttft[0]:.2f}x' if split and ttft and attn != 'split' else '-'
            fail = f' ({cell["failed"]} failed)' if cell['failed'] else ''
            lines.append(f'| {model} | {kernel} | {m} | {prompt} | {sockets} | {attn} | {(ttft or (0,))[-1]}{fail} | '
                         f'{span(ttft, 3)} | {speed} | {"-" if share is None else f"{100 * share[0]:.0f}%"} | {span(decode, 1)} |')
    return lines + ['']


def variants(cells_by_attn):
    """The non-split attentions with timed repeats, and split's stats (None when split has none)."""
    split = stats(cells_by_attn.get('split', {}).get('ttft', []))
    return split, [a for a in ordered(cells_by_attn) if a != 'split' and stats(cells_by_attn[a]['ttft'])]


def value_gate(cells, model_tag=HEADLINE_MODEL):
    lines = [f'### Value: {model_tag.upper()}, 8192 tokens, one socket, best variant against split (gate {SPEEDUP_GATE:g}x)', '',
             '| model | rung | M | split TTFT s | best variant | best TTFT s | speedup | beyond variability | gate |',
             '| --- | --- | ---: | ---: | --- | ---: | ---: | --- | --- |']
    verdicts = {}
    for key in sorted(cells):
        model, kernel, m, prompt, sockets = key
        if model_tag not in model.lower() or prompt != 8192 or sockets != 1:
            continue
        split, others = variants(cells[key])
        if not split or not others:
            continue
        best = min(others, key=lambda a: stats(cells[key][a]['ttft'])[0])
        stat = stats(cells[key][best]['ttft'])
        speed = split[0] / stat[0]
        beyond = stat[2] < split[1]
        verdict = 'PASS' if speed >= SPEEDUP_GATE else 'FAIL'
        verdicts[(model, kernel, m)] = verdict
        lines.append(f'| {model} | {kernel} | {m} | {span(split, 3)} | {best} | {span(stat, 3)} | {speed:.2f}x | '
                     f'{"yes" if beyond else "no"} | {verdict} |')
    if not verdicts:
        return lines + ['| (no 8192-token one-socket cell with split and another variant) | | | | | | | | |', ''], None
    headline = [v for (model, kernel, m), v in verdicts.items() if kernel == HEADLINE_RUNG]
    shown = headline or list(verdicts.values())
    overall = 'PASS' if all(v == 'PASS' for v in shown) else 'FAIL'
    scope = f'the {HEADLINE_RUNG} rung' if headline else 'every rung (no headline rung in the data)'
    lines += ['', f'Value gate ({scope}): **{overall}**.', '']
    return lines, overall


def not_slower_gate(cells):
    lines = ['### 2048 tokens: not slower than split', '',
             'The variant median TTFT must not exceed the slowest split repeat (median speed at least the lower end of the split range).',
             '', '| model | rung | M | sockets | attention | split TTFT s | variant TTFT s | gate |', '| --- | --- | ---: | ---: | --- | ---: | ---: | --- |']
    failed = rows = 0
    for key in sorted(cells):
        model, kernel, m, prompt, sockets = key
        if prompt != 2048:
            continue
        split, others = variants(cells[key])
        for attn in others if split else []:
            stat = stats(cells[key][attn]['ttft'])
            ok = stat[0] <= split[2]
            rows += 1
            failed += not ok
            lines.append(f'| {model} | {kernel} | {m} | {sockets} | {attn} | {span(split, 3)} | {span(stat, 3)} | '
                         f'{"PASS" if ok else "FAIL"} |')
    if not rows:
        return lines + ['| (no 2048-token cell with split and another variant) | | | | | | | |', ''], None
    overall = 'FAIL' if failed else 'PASS'
    return lines + ['', f'2048 gate: **{overall}** ({rows - failed} of {rows} cells).', ''], overall


def decode_gate(cells):
    lines = ['### Decode unchanged', '',
             'The variant median decode tokens/s must lie inside the split range of the same cell.', '',
             '| model | rung | M | prompt | sockets | attention | split decode tok/s | variant decode tok/s | gate |',
             '| --- | --- | ---: | ---: | ---: | --- | ---: | ---: | --- |']
    failed = rows = 0
    for key in sorted(cells):
        model, kernel, m, prompt, sockets = key
        split = stats(cells[key].get('split', {}).get('decode', []))
        for attn in ordered(cells[key]):
            stat = stats(cells[key][attn]['decode'])
            if attn == 'split' or not split or not stat:
                continue
            ok = split[1] <= stat[0] <= split[2]
            rows += 1
            failed += not ok
            lines.append(f'| {model} | {kernel} | {m} | {prompt} | {sockets} | {attn} | {span(split, 1)} | {span(stat, 1)} | '
                         f'{"PASS" if ok else "FAIL"} |')
    if not rows:
        return lines + ['| (no decode data for split and another variant) | | | | | | | | |', ''], None
    overall = 'FAIL' if failed else 'PASS'
    return lines + ['', f'Decode gate: **{overall}** ({rows - failed} of {rows} cells).', ''], overall


def placement_gate(cells):
    reports = [p for by in cells.values() for cell in by.values() for p in cell['placement']]
    bad = [p for p in reports if not p.endswith('(verified)')]
    failed = sum(cell['failed'] for by in cells.values() for cell in by.values())
    lines = ['### Placement', '', f'{len(reports) - len(bad)} of {len(reports)} completed reports have placement verified'
             + (f'; {failed} runs failed' if failed else '') + '.']
    if bad:
        counts = {p: bad.count(p) for p in sorted(set(bad))}
        lines.append('Not verified: ' + ', '.join(f'{p} x{n}' for p, n in counts.items()) + '.')
    overall = 'PASS' if reports and not bad and not failed else 'FAIL'
    return lines + ['', f'Placement gate: **{overall}**.', ''], overall


def two_socket_gain(cells):
    lines = ['### Two sockets over one (item 2a)', '',
             'TTFT gain = one-socket median over two-socket median, per rung and variant. If attention is no longer the long pole at '
             '8192 tokens, the E0 hypothesis H6 predicts a gain below the 2.05-2.15x measured with split.', '',
             '| model | rung | M | prompt | attention | 1 socket TTFT s | 2 sockets TTFT s | gain |', '| --- | --- | ---: | ---: | --- | ---: | ---: | ---: |']
    rows = 0
    for key in sorted(cells):
        model, kernel, m, prompt, sockets = key
        other = (model, kernel, m, prompt, 2)
        if sockets != 1 or other not in cells:
            continue
        for attn in ordered(cells[key]):
            one, two = stats(cells[key][attn]['ttft']), stats(cells.get(other, {}).get(attn, {}).get('ttft', []))
            if one and two:
                rows += 1
                lines.append(f'| {model} | {kernel} | {m} | {prompt} | {attn} | {span(one, 3)} | {span(two, 3)} | {one[0] / two[0]:.2f}x |')
    if not rows:
        lines.append('| (no cell with both socket counts) | | | | | | | |')
    return lines + ['']


def bench(root):
    cells, failed_runs = load(root)
    out = ['# E4 prefill attention campaign', '']
    if not cells:
        return '\n'.join(out + ['No E4 reports (labels with the tag attn<variant>) in the directory.']), None
    out += table(cells)
    out += ['## E4.3 gates', '']
    verdicts = {}
    for name, gate in (('value', value_gate), ('2048', not_slower_gate), ('decode', decode_gate), ('placement', placement_gate)):
        text, verdict = gate(cells)
        out += text
        verdicts[name] = verdict
    out += two_socket_gain(cells)
    out += ['## Verdicts', '', *(f'- {name}: {verdict or "no data"}' for name, verdict in verdicts.items()), '']
    return '\n'.join(out), verdicts


def main(argv=None):
    argv = sys.argv[1:] if argv is None else argv
    if len(argv) != 2 or argv[0] not in ('checks', 'bench'):
        raise SystemExit(__doc__)
    if argv[0] == 'checks':
        text, failed = checks(argv[1])
        print(text)
        return 1 if failed else 0
    print(bench(argv[1])[0])
    return 0


if __name__ == '__main__':
    sys.exit(main())
