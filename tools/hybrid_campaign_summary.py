#!/usr/bin/env python3
"""Tables of the results directories of scripts/hybrid-campaign.sbatch (docs/hybrid-memory-results.md).

    tools/hybrid_campaign_summary.py DIR [DIR ...] [--step STEP] [--guard 0.05] [--json]

Every directory holds runs.tsv (one line per run: step, setting, measurement, repetition), the engine reports (`--json`), the perf
CSV of each run and the token traces. Per cell (step, setting, measurement) the median [min-max] over the interleaved repetitions of:
decode tokens per second, TTFT and the prefill/warm-up/decode times, the bytes the uncore counters saw in the decode window (HBM and
DDR GB per generated token and GB/s, summed over the sockets: 32-byte CAS lines on HBM, 64-byte on DDR; tools/hbm_ddr_results.py cuts
the window from the report's phase_epoch_ns), the UPI data flits (as derived GB) and the package energy per token, the load time, the peak resident
set and the placement check. Each cell is compared with the ceiling of tools/hybrid_limits.py at the cell's own placement (peak: the
raw P1 rates; calibrated: the engine efficiency of that model's anchor); a ratio more than 30% from 1 is flagged. Synthetic-K/V
cells (measurement d<N>s) are checked against the real fill of the same setting and context, and the token traces are compared
inside each group (model, sockets, measurement, kernels) across all the directories given. The profiled runs (one per cell) are
reported apart, as milliseconds per generated token.
"""
import argparse
import csv
import hashlib
import json
import re
import statistics
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import hbm_ddr_results  # noqa: E402
from xengine_summary import UPI_DATA_FLIT_BYTES  # noqa: E402
import hybrid_limits as HL  # noqa: E402

FLAG = 0.30            # the stop rule of the campaign: a step more than 30% from its model needs a diagnosis
GIB = 2 ** 30
PHASES = ['qkv', 'wo', 'w1_w3_swiglu', 'w2', 'attention', 'classifier', 'other', 'forward']


def med(values, digits=2, spread=True):
    values = [v for v in values if v is not None]
    if not values:
        return '-'
    m = f'{statistics.median(values):.{digits}f}'
    return f'{m} [{min(values):.{digits}f}-{max(values):.{digits}f}]' if spread and len(values) > 1 else m


def median(values):
    values = [v for v in values if v is not None]
    return statistics.median(values) if values else None


# ---------------------------------------------------------------------------------------------------------------------------
# Ceilings (tools/hybrid_limits.py at the cell's own placement)
# ---------------------------------------------------------------------------------------------------------------------------

def interpolate_e1(cores):
    """(HBM GB/s, DDR GB/s) the P1 study measured with `cores` DDR cores (the other cores on HBM), linear between the table's rows."""
    rows = sorted((cd, bh, bd) for _, cd, bh, bd in HL.E1)
    for (c0, h0, d0), (c1, h1, d1) in zip(rows, rows[1:]):
        if c0 <= cores <= c1:
            t = (cores - c0) / (c1 - c0)
            return h0 + t * (h1 - h0), d0 + t * (d1 - d0)
    raise ValueError(f'{cores} DDR cores is outside the P1 table')


def model_key(report):
    layers = report['model']['configuration']['n_layers']
    return {80: '70B', 32: '8B'}[layers]


def decode_ceiling(report):
    """Peak and calibrated tok/s at the plan of `report`, and the bytes per socket and token on each tier (the model's, derived)."""
    m = HL.MODELS[model_key(report)]
    plan = report['placement_plan']
    sockets = len({node['socket'] for node in plan['nodes']})
    gs = report['model']['quantization_group_size']
    tokens = report['tokens']
    depth = report['parameters']['warm_context'] + tokens['processed_prompt'] + (tokens['decode_steps'] + 1) / 2
    eta = HL.CAL[HL.cal_key(m, sockets)][0]
    lw, cw = HL.layer_weight_bytes(m, gs, sockets), HL.classifier_bytes(m, gs, sockets)
    lk = HL.kv_bytes_per_token(m, sockets, 'fp32') / m.layers * depth
    mode = plan['mode']
    if mode == 'hbm-only':
        wh, wd, kh, kd = m.layers * lw + cw, 0.0, m.layers * lk, 0.0
    elif mode == 'ddr':
        wh, wd, kh, kd = 0.0, m.layers * lw + cw, 0.0, m.layers * lk
    else:
        nk = plan['kv_hbm_layers']
        kh, kd = nk * lk, (m.layers - nk) * lk
        if plan.get('split') == 'rows':
            frac = plan['ddr_fraction']
            wh, wd = m.layers * lw * (1 - frac) + cw, m.layers * lw * frac
        else:
            nw = plan['weights_hbm_layers']
            on_hbm = plan.get('classifier') == 'hbm'
            wh, wd = nw * lw + (cw if on_hbm else 0.0), (m.layers - nw) * lw + (0.0 if on_hbm else cw)
    out = {'depth': depth, 'sockets': sockets, 'gs': gs, 'model': m.name, 'bytes_gb': [x / HL.GB for x in (wh, kh, wd, kd)],
           'ddr_byte_fraction': (wd + kd) / (wh + kh + wd + kd), 'kind': 'serial'}
    if mode == 'hybrid' and plan.get('split') == 'rows':
        bh, bd = interpolate_e1(plan['ddr_cores'])
        out['kind'] = 'concurrent'
        for label, e in (('peak', HL.PEAK), ('cal', eta)):
            t_h = wh / (e.hw * bh * HL.GB) + kh / (e.ha * bh * HL.GB)
            t_d = wd / (e.dw * bd * HL.GB) + kd / (e.da * bd * HL.GB)
            out[label] = 1.0 / max(t_h, t_d)
        return out
    for label, e in (('peak', HL.PEAK), ('cal', eta)):
        out[label] = 1.0 / (HL.t_tier(wh, kh, HL.HBM_STREAM, e.hw, e.ha) + HL.t_tier(wd, kd, HL.DDR_STREAM, e.dw, e.da))
    return out


def ttft_ceiling(report):
    """AMX-peak TTFT, and for the 70B the TTFT predicted from the 8B in-situ rates (docs/plans/hybrid-limits.md section 5)."""
    m = HL.MODELS[model_key(report)]
    plan = report['placement_plan']
    sockets = len({node['socket'] for node in plan['nodes']})
    gs = report['model']['quantization_group_size']
    prompt = report['tokens']['processed_prompt']
    out = {'peak': HL.t_amx_peak(m, prompt, sockets), 'scaled': None}
    if m.name == '70B' and (gs, sockets) in HL.INSITU_8B:
        out['scaled'] = HL.ttft_scaled_70b(gs, sockets, prompt)['total']
    return out


# ---------------------------------------------------------------------------------------------------------------------------
# Runs
# ---------------------------------------------------------------------------------------------------------------------------

def counters_of(directory, guard):
    return {r['run']: r['counters'] for r in hbm_ddr_results.results(directory, guard)}


def tier_sum(window, key):
    if not window or not window.get('sockets'):
        return None
    values = [s.get(key) for s in window['sockets']]
    return None if any(v is None for v in values) else sum(values)


def placement_checks(path):
    if not path.exists():
        return None
    lines = path.read_text().splitlines()
    return sum(l.startswith('PASS') for l in lines), sum(l.startswith('FAIL') for l in lines)


def run_metrics(directory, row, counters):
    path = directory / f"{row['label']}.json"
    if not path.exists():
        return None
    report = json.loads(path.read_text())
    if report.get('status') != 'completed' or not report.get('placement_plan'):
        return None
    timing, tokens = report['timing_seconds'], report['tokens']
    steps = tokens['decode_steps']
    decode, prefill = counters.get('decode'), counters.get('prefill')
    run = {'row': row, 'report': report, 'tps': report['throughput_tokens_per_second']['decode'],
           'ttft': timing['time_to_first_token'], 'prefill_s': timing['prefill'], 'warm_s': timing.get('warm_context'),
           'decode_s': timing['decode'], 'load_s': timing['checkpoint_load'] + timing['runtime_init'],
           'populate_s': timing.get('placement_populate'), 'rss_gib': report['memory_bytes']['peak_resident'] / GIB,
           'prompt_tokens': tokens['processed_prompt'], 'decode_steps': steps,
           'placement': placement_checks(directory / f"{row['label']}.placement"),
           'profile': (report.get('profile_seconds') or {}).get('decode')}
    for tier in ('hbm', 'ddr'):
        run[f'{tier}_gb_per_token'] = tier_sum(decode, f'{tier}_gb_per_step')
        run[f'{tier}_gbps'] = tier_sum(decode, f'{tier}_gbps')
        run[f'{tier}_prefill_gb'] = None if tier_sum(prefill, f'{tier}_gb_per_step') is None else \
            tier_sum(prefill, f'{tier}_gb_per_step') * tokens['processed_prompt']
    rx, tx = tier_sum(decode, 'upi_rx_mflits_per_step'), tier_sum(decode, 'upi_tx_mflits_per_step')
    # derived: a data flit stands for 64/9 bytes of payload (tools/xengine_summary.py, an assumption recalled from Intel PCM)
    run['upi_gb_per_token'] = None if rx is None or tx is None else (rx + tx) * 1e6 * UPI_DATA_FLIT_BYTES / 1e9
    run['pkg_j_per_token'] = tier_sum(decode, 'pkg_j_per_step')
    run['ram_j_per_token'] = tier_sum(decode, 'ram_j_per_step')
    if run['hbm_gb_per_token'] is not None and run['hbm_gb_per_token'] + run['ddr_gb_per_token'] > 0:
        run['ddr_share'] = run['ddr_gb_per_token'] / (run['hbm_gb_per_token'] + run['ddr_gb_per_token'])
    else:
        run['ddr_share'] = None
    run['decode_ceiling'] = decode_ceiling(report) if row['kind'] == 'decode' else None
    run['ttft_ceiling'] = ttft_ceiling(report) if row['kind'] == 'ttft' else None
    return run


def collect(directories, guard):
    runs = []
    for directory in directories:
        directory = Path(directory)
        counters = counters_of(directory, guard)
        with open(directory / 'runs.tsv') as stream:
            for row in csv.DictReader(stream, delimiter='\t'):
                run = run_metrics(directory, row, counters.get(row['label'], {}))
                if run:
                    run['directory'] = directory.name
                    runs.append(run)
    return runs


def cells(runs, profiled=False):
    out = {}
    for run in runs:
        if (run['row']['profile'] == '1') != profiled:
            continue
        out.setdefault((run['row']['step'], run['row']['setting'], run['row']['meas']), []).append(run)
    return out


def trace_groups(directories):
    groups = {}
    for directory in directories:
        directory = Path(directory)
        with open(directory / 'runs.tsv') as stream:
            for row in csv.DictReader(stream, delimiter='\t'):
                path = directory / f"{row['label']}.tokens.json"
                if path.exists():
                    digest = hashlib.sha256(json.dumps(json.loads(path.read_text()), sort_keys=True).encode()).hexdigest()
                    groups.setdefault(row['group'], {})[f"{directory.name}/{row['label']}"] = digest
    return groups


# ---------------------------------------------------------------------------------------------------------------------------
# Tables
# ---------------------------------------------------------------------------------------------------------------------------

def md(header, rows):
    return ['| ' + ' | '.join(header) + ' |', '| ' + ' | '.join(['---'] + ['---:'] * (len(header) - 1)) + ' |'] + \
           ['| ' + ' | '.join(str(c) for c in r) + ' |' for r in rows]


def decode_table(groups, stats):
    rows = []
    for key in sorted(groups):
        step, setting, meas = key
        rs = groups[key]
        if rs[0]['row']['kind'] != 'decode':
            continue
        tps = median(r['tps'] for r in rs)
        c = rs[0]['decode_ceiling']
        ratio = tps / c['cal']
        flag = 'FLAG' if abs(ratio - 1) > FLAG else ''
        share = median(r['ddr_share'] for r in rs)
        rows.append([step, setting, meas, len(rs), med([r['tps'] for r in rs]), f'{1e3 / tps:.1f}',
                     '-' if share is None else f'{100 * share:.1f}%', f'{100 * c["ddr_byte_fraction"]:.1f}%',
                     med([r['hbm_gb_per_token'] for r in rs], 2, False), med([r['ddr_gb_per_token'] for r in rs], 2, False),
                     med([r['hbm_gbps'] for r in rs], 0, False), med([r['ddr_gbps'] for r in rs], 0, False),
                     med([r['upi_gb_per_token'] for r in rs], 2, False), med([r['pkg_j_per_token'] for r in rs], 1, False),
                     f'{c["peak"]:.2f}', f'{c["cal"]:.2f}', f'{ratio:.3f}', flag])
        stats[f'{step}/{setting}/{meas}'] = {'tps': [r['tps'] for r in rs], 'ratio_cal': ratio, 'ceiling': {k: c[k] for k in ('peak', 'cal', 'kind')}}
    return md(['step', 'setting', 'meas', 'n', 'decode tok/s median [min-max]', 'ms/token', 'DDR share of CAS bytes',
               'DDR share modelled', 'HBM GB/token', 'DDR GB/token', 'HBM GB/s', 'DDR GB/s', 'UPI GB/token (derived)', 'pkg J/token',
               'ceiling peak tok/s', 'ceiling cal. tok/s', 'measured / cal.', 'flag'], rows)


def ttft_table(groups, stats):
    rows = []
    for key in sorted(groups):
        step, setting, meas = key
        rs = groups[key]
        if rs[0]['row']['kind'] != 'ttft':
            continue
        t = median(r['ttft'] for r in rs)
        c = rs[0]['ttft_ceiling']
        kern = rs[0]['report']['runtime'].get('kernels_prefill', '-')
        ratio = t / c['peak']
        scaled = '-' if c['scaled'] is None else f'{c["scaled"]:.2f}'
        flag = 'FLAG' if c['scaled'] is not None and abs(t / c['scaled'] - 1) > FLAG and kern != 'avx512' else ''
        rows.append([step, setting, meas, len(rs), rs[0]['prompt_tokens'], kern, med([r['ttft'] for r in rs], 3),
                     med([r['prefill_s'] for r in rs], 3, False), med([r['hbm_prefill_gb'] for r in rs], 0, False),
                     med([r['ddr_prefill_gb'] for r in rs], 0, False), f'{c["peak"]:.3f}', f'{ratio:.1f}', scaled,
                     '-' if c['scaled'] is None else f'{t / c["scaled"]:.2f}', flag])
        stats[f'{step}/{setting}/{meas}'] = {'ttft': [r['ttft'] for r in rs], 'peak_s': c['peak'], 'scaled_s': c['scaled']}
    return md(['step', 'setting', 'meas', 'n', 'prompt tokens', 'prefill kernels', 'TTFT s median [min-max]', 'prefill s',
               'HBM GB read+written', 'DDR GB read+written', 'AMX-peak s', 'measured / peak', '8B-scaled s', 'measured / scaled',
               'flag'], rows)


def setup_table(groups):
    rows = []
    for key in sorted(groups):
        rs = groups[key]
        passes = sum(r['placement'][0] for r in rs if r['placement'])
        fails = sum(r['placement'][1] for r in rs if r['placement'])
        rows.append([*key, len(rs), med([r['load_s'] for r in rs], 1), med([r['populate_s'] for r in rs], 1),
                     med([r['warm_s'] for r in rs], 1), med([r['rss_gib'] for r in rs], 0, False), f'{passes} PASS / {fails} FAIL',
                     rs[0]['directory']])
    return md(['step', 'setting', 'meas', 'n', 'load s (file + runtime init)', 'placement populate s', 'warm-up s',
               'peak resident GiB', 'placement checks', 'results directory'], rows)


def profile_table(groups):
    rows = []
    for key in sorted(groups):
        for run in groups[key]:
            p = run['profile']
            if not p or not p.get('forward_passes'):
                continue
            rows.append([*key] + [f'{p[ph] / p["forward_passes"] * 1e3:.2f}' if ph in p else '-' for ph in PHASES])
    return md(['step', 'setting', 'meas'] + [f'{ph} ms/token' for ph in PHASES], rows)


def synthetic_check(groups):
    rows = []
    for (step, setting, meas), rs in sorted(groups.items()):
        if not re.fullmatch(r'd\d+s', meas):
            continue
        real = groups.get((step, setting, meas[:-1]))
        if not real or rs[0]['row']['kind'] != 'decode':
            continue
        a, b = median(r['tps'] for r in real), median(r['tps'] for r in rs)
        rows.append([step, setting, meas[1:-1], len(real), len(rs), f'{a:.3f}', f'{b:.3f}', f'{100 * (b / a - 1):+.1f}%',
                     med([r['ddr_gb_per_token'] for r in real], 2, False), med([r['ddr_gb_per_token'] for r in rs], 2, False)])
    return md(['step', 'setting', 'context', 'real n', 'synthetic n', 'real tok/s', 'synthetic tok/s', 'synthetic vs real',
               'real DDR GB/token', 'synthetic DDR GB/token'], rows)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('directory', type=Path, nargs='+')
    parser.add_argument('--step')
    parser.add_argument('--guard', type=float, default=0.05)
    parser.add_argument('--json', action='store_true')
    args = parser.parse_args(argv)
    runs = collect(args.directory, args.guard)
    if args.step:
        runs = [r for r in runs if r['row']['step'] == args.step]
    plain, profiled = cells(runs), cells(runs, True)
    stats = {}
    sections = [('Decode (median [min-max] over the interleaved repetitions; bytes from the CAS counters of the decode window)',
                 decode_table(plain, stats)),
                ('Time to first token', ttft_table(plain, stats)),
                ('Load, placement and memory', setup_table(plain)),
                ('Synthetic K/V against the real fill', synthetic_check(plain)),
                ('Profiled decode, milliseconds per generated token (one profiled run per cell)', profile_table(profiled))]
    for title, lines in sections:
        if len(lines) > 2:
            print(f'### {title}\n')
            print('\n'.join(lines) + '\n')
    print('### Token traces\n')
    bad = 0
    for group, traces in sorted(trace_groups(args.directory).items()):
        distinct = len(set(traces.values()))
        bad += distinct > 1
        print(f'* {group}: {len(traces)} traces, {distinct} distinct')
    if args.json:
        print(json.dumps(stats))
    return 1 if bad else 0


if __name__ == '__main__':
    sys.exit(main())
