#!/usr/bin/env python3
"""Configuration generator and analysis for the concurrent HBM+DDR bandwidth experiments (scripts/concurrent-tier.sbatch).

    concurrent_tier.py cfgs STAGE [--best-env FILE] [--results DIR ...] [--reps-of-interest N]   print stream-bench configurations
    concurrent_tier.py best RESULTS_DIR                                                           print BEST_* variables of the base sweep
    concurrent_tier.py summarize RESULTS_DIR [RESULTS_DIR ...] -o OUT_DIR                         tables, counter checks, summary.json, figure
    concurrent_tier.py replicate-check RESULTS_DIR OTHER_DIR                                      compare the key points of two jobs

STAGE is one of smoke, base, e1, e2, e3, e4, e5, e5hbm, e6, lat, rep. A results directory holds raw/<label>.jsonl (the
JSON lines of tests/stream_bench.c --exp ...), raw/<label>.perf.csv (perf stat -a --per-socket -I 100 -x,) and
raw/<label>.stderr. Labels are STAGE-PAGE-rN (N = repetition); one process runs every configuration of the stage once.

Counters: every window is a burst of memory traffic separated from the next by an idle gap, so the perf interval series
is cut into bursts (consecutive intervals above a traffic threshold) and the k-th burst belongs to the k-th window. The
burst's DDR CAS reads x 64 B and HBM CAS reads x 32 B (docs/hbm-roofline.md) are the measured bytes per tier; the window's
computed bytes are the sum over all workers of the bytes they issued (warm-up included). Stdlib plus matplotlib (for the
figure, optional).
"""
import argparse
import json
import math
import os
import re
import statistics
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import xengine_summary as xs  # noqa: E402  (the perf csv parser and the byte factors shared with the cross-engine campaign)

DDR_CAS, HBM_CAS = 64, 32
CORES = 56
BYTES_TOKEN_GS64 = 36.9e9  # per socket per decoded token, 70B Q8 gs64, 2-socket tensor parallel (task statement)
GS_FACTOR = {64: 1.0, 32: 1.125 / 1.0625}


# ---- configuration generation -----------------------------------------------------------------------------------------
def fmt(**kw):
    return ','.join(f'{k}={v}' for k, v in kw.items() if v is not None)


def read_env(path):
    env = {}
    if path and Path(path).exists():
        for line in Path(path).read_text().splitlines():
            if '=' in line and not line.startswith('#'):
                k, v = line.split('=', 1)
                env[k.strip()] = v.strip()
    return env


def tier_params(env, tier):
    """stream-bench keys for one tier's best read settings from the base sweep (HBM: S, pf, hint, nt; DDR: dS, dpf, ...)."""
    p = {'S': env.get('BEST_S', 2), 'pf': env.get('BEST_PF', 4096), 'hint': env.get('BEST_HINT', 0), 'nt': env.get('BEST_NT', 0)}
    d = {'dS': env.get('BEST_DS', p['S']), 'dpf': env.get('BEST_DPF', p['pf']), 'dhint': env.get('BEST_DHINT', p['hint']),
         'dnt': env.get('BEST_DNT', p['nt'])}
    return p, d


def split_cfg(name, ch, cd, env, **extra):
    p, d = tier_params(env, 0)
    kw = dict(name=name, kind='split', ch=ch, cd=cd, **p)
    if cd:
        kw.update(d)
    kw.update(extra)
    return fmt(**kw)


E1_CD = [0, 2, 4, 8, 12, 16, 20, 24, 28, 32, 40, 56]
E2_RATIOS = [(1, 0), (8, 1), (4, 1), (3, 1), (2, 1), (3, 2), (1, 1), (1, 2), (0, 1)]
E3_RATIOS = [(1, 0), (8, 1), (4, 1), (3, 1), (2, 1), (1, 1), (0, 1)]
E5_R = [0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.5, 1.0]


def e5_cdw_options(r):
    if r == 0:
        return [0]
    base = r * CORES
    out = sorted({min(CORES - 1, max(1, round(base * f))) for f in (0.75, 1.0, 1.25)})
    return out


def gen(stage, env, results=None):
    out = []
    add = out.append
    if stage == 'smoke':
        add(split_cfg('smoke-hbm56', 56, 0, env, warm=0.1, seconds=0.4))
        add(split_cfg('smoke-split48-8', 48, 8, env, warm=0.1, seconds=0.4))
        add(split_cfg('smoke-split28-28', 28, 28, env, warm=0.1, seconds=0.4))
        add(split_cfg('smoke-split28-28-vol', 28, 28, env, volume=2048))
        add(split_cfg('smoke-ddr56', 0, 56, env, warm=0.1, seconds=0.4))
        add(split_cfg('smoke-lat-hbm', 27, 28, env, probe=1, warm=0.1, seconds=0.4))
        add(fmt(name='smoke-shared-3-1-chunk', kind='shared', a=3, b=1, unit=128, **tier_params(env, 0)[0], warm=0.1, seconds=0.4))
        add(fmt(name='smoke-shared-3-1-line', kind='shared', a=3, b=1, unit=1, pf=tier_params(env, 0)[0]['pf'], warm=0.1, seconds=0.4))
        add(fmt(name='smoke-gemv64-r0', kind='gemv', gs=64, r=0, cdw=0))
        add(fmt(name='smoke-gemv64-r0.25', kind='gemv', gs=64, r=0.25, cdw=14))
        add(fmt(name='smoke-gemv32-r0.25-shared', kind='gemv', gs=32, r=0.25, gshared=1))
    elif stage == 'smoke-inter':
        add(fmt(name='smoke-inter-1-1', kind='inter', a=1, b=1, **tier_params(env, 0)[0], warm=0.1, seconds=0.4))
        add(fmt(name='smoke-inter-4-1', kind='inter', a=4, b=1, **tier_params(env, 0)[0], warm=0.1, seconds=0.4))
    elif stage == 'base':
        grid_pf = [0, 1024, 4096, 16384]
        for c in [8, 16, 24, 32, 40, 48, 56]:
            for s in (1, 2, 4, 8):
                for pf in grid_pf:
                    add(fmt(name=f'base-hbm-c{c}-S{s}-pf{pf}', kind='split', ch=c, cd=0, S=s, pf=pf, warm=0.15, seconds=0.5))
        for c in [8, 16, 24, 28, 32, 56]:
            for s in (1, 2, 4, 8):
                for pf in grid_pf:
                    add(fmt(name=f'base-ddr-c{c}-S{s}-pf{pf}', kind='split', ch=0, cd=c, S=s, pf=pf, warm=0.15, seconds=0.5))
        for hint, nt in ((1, 0), (2, 0), (0, 1)):
            add(fmt(name=f'base-hbm-c56-S2-pf4096-h{hint}-nt{nt}', kind='split', ch=56, cd=0, S=2, pf=4096, hint=hint, nt=nt, warm=0.15, seconds=0.5))
            add(fmt(name=f'base-ddr-c28-S2-pf4096-h{hint}-nt{nt}', kind='split', ch=0, cd=28, S=2, pf=4096, hint=hint, nt=nt, warm=0.15, seconds=0.5))
    elif stage == 'e1':
        for cd in E1_CD:
            add(split_cfg(f'e1-cd{cd}', CORES - cd, cd, env))
            add(split_cfg(f'e1v-cd{cd}', CORES - cd, cd, env, volume=6144))
        for c in (16, 24, 28, 32, 40, 44, 48, 52):  # the HBM-only and DDR-only core scans (idle cores beyond c)
            add(split_cfg(f'e1-hbmonly-c{c}', c, 0, env))
        for c in (16, 24, 28, 32, 40):
            add(split_cfg(f'e1-ddronly-c{c}', 0, c, env))
    elif stage == 'e2':
        p = tier_params(env, 0)[0]
        for variant, unit in (('chunk', 128), ('line', 1)):
            for a, b in E2_RATIOS:
                add(fmt(name=f'e2-{variant}-{a}-{b}', kind='shared', a=a, b=b, unit=unit, **p))
        for unit in (8, 32, 512):
            for a, b in ((3, 1), (1, 1)):
                add(fmt(name=f'e2-unit{unit}-{a}-{b}', kind='shared', a=a, b=b, unit=unit, **p))
        add(split_cfg('e2-ref-hbm56', CORES, 0, env))
    elif stage == 'e3':
        p = tier_params(env, 0)[0]
        for a, b in E3_RATIOS:
            add(fmt(name=f'e3-{a}-{b}', kind='inter', a=a, b=b, **p))
    elif stage == 'e4':
        best = best_points(results or [], env)
        for page_tag in ('',):
            add(split_cfg('e4-hbm56', CORES, 0, env))
            add(split_cfg(f'e4-split-cd{best["e1_cd"]}', CORES - best['e1_cd'], best['e1_cd'], env))
            p = tier_params(env, 0)[0]
            a, b = best['e2_ratio']
            add(fmt(name=f'e4-shared-{a}-{b}', kind='shared', a=a, b=b, unit=best['e2_unit'], **p))
            a, b = best['e3_ratio']
            add(fmt(name=f'e4-inter-{a}-{b}', kind='inter', a=a, b=b, **p))
            add(split_cfg('e4-ddr56', 0, CORES, env))
    elif stage == 'e5':
        for gs in (64, 32):
            for r in E5_R:
                for cdw in e5_cdw_options(r):
                    add(fmt(name=f'e5-gs{gs}-r{r}-cdw{cdw}', kind='gemv', gs=gs, r=r, cdw=cdw))
            for r in E5_R[1:]:
                add(fmt(name=f'e5-gs{gs}-r{r}-shared', kind='gemv', gs=gs, r=r, gshared=1))
    elif stage == 'e5hbm':
        for gs in (64, 32):
            for act in (32, 40, 48, 52, 56):
                add(fmt(name=f'e5hbm-gs{gs}-act{act}', kind='gemv', gs=gs, r=0, cdw=0, act=act))
    elif stage == 'e6':
        best = best_points(results or [], env)
        cd = best['e1_cd']
        p, d = tier_params(env, 0)
        base = dict(kind='split', ch=CORES - cd, cd=cd)
        for pf in (0, 512, 1024, 2048, 4096, 8192, 16384):
            add(fmt(name=f'e6-pf{pf}', **base, S=p['S'], pf=pf, hint=p['hint'], nt=p['nt'], dS=d['dS'], dpf=pf, dhint=d['dhint'], dnt=d['dnt']))
        for s in (1, 2, 4, 8):
            add(fmt(name=f'e6-S{s}', **base, S=s, pf=p['pf'], hint=p['hint'], nt=p['nt'], dS=s, dpf=d['dpf'], dhint=d['dhint'], dnt=d['dnt']))
        for hint in (0, 1, 2):
            add(fmt(name=f'e6-hint{hint}', **base, S=p['S'], pf=p['pf'], hint=hint, nt=p['nt'], dS=d['dS'], dpf=d['dpf'], dhint=hint, dnt=d['dnt']))
        for nt in (0, 1):
            add(fmt(name=f'e6-nt{nt}', **base, S=p['S'], pf=p['pf'], hint=p['hint'], nt=nt, dS=d['dS'], dpf=d['dpf'], dhint=d['dhint'], dnt=nt))
        for alt in (0, 1):
            add(fmt(name=f'e6-alt{alt}', **base, **p, **d, alt=alt))
        # HBM-only and DDR-only at the same settings, to separate a group's own limit from the interference
        for pf in (0, 1024, 4096, 16384):
            add(fmt(name=f'e6-hbmonly-pf{pf}', kind='split', ch=CORES, cd=0, S=p['S'], pf=pf, hint=p['hint'], nt=p['nt']))
        for s in (1, 2, 4, 8):
            add(fmt(name=f'e6-hbmonly-S{s}', kind='split', ch=CORES, cd=0, S=s, pf=p['pf'], hint=p['hint'], nt=p['nt']))
        # the shared-core best point under prefetch variation
        a, b = best['e2_ratio']
        for pf in (0, 1024, 4096, 16384):
            add(fmt(name=f'e6-shared-pf{pf}', kind='shared', a=a, b=b, unit=best['e2_unit'], S=p['S'], pf=pf, hint=p['hint'], nt=p['nt']))
    elif stage == 'lat':
        for probe in (1, 2):
            for ch, cd in ((1, 0), (55, 0), (47, 8), (39, 16), (27, 28), (15, 40), (0, 55)):
                add(split_cfg(f'lat-p{probe}-h{ch}-d{cd}', ch, cd, env, probe=probe))
    elif stage == 'rep':  # the key E1 points, for the replication job on the other node
        for cd in (0, 4, 8, 16, 28, 56):
            add(split_cfg(f'e1-cd{cd}', CORES - cd, cd, env))
        add(split_cfg('e1v-cd8', CORES - 8, 8, env, volume=6144))
    elif stage == 'rep-e5':
        for gs in (64, 32):
            for r, cdw in ((0, 0), (0.1, 6), (0.2, 11), (0.3, 17), (1.0, 55)):
                add(fmt(name=f'e5-gs{gs}-r{r}-cdw{cdw}', kind='gemv', gs=gs, r=r, cdw=cdw))
    else:
        raise SystemExit(f'unknown stage {stage}')
    return out


# ---- reading results --------------------------------------------------------------------------------------------------
def parse_label(label):
    m = re.match(r'(.+)-(2m|4k)-r(\d+)$', label)
    return (m.group(1), m.group(2), int(m.group(3))) if m else (label, '?', 0)


def load_windows(results):
    """[(label, start_event, [window dicts], placement events)] from RESULTS/raw/*.jsonl."""
    runs = []
    for path in sorted(Path(results, 'raw').glob('*.jsonl')):
        start, windows, placement = None, [], []
        for line in path.read_text().splitlines():
            try:
                d = json.loads(line)
            except ValueError:
                continue
            if d.get('event') == 'start':
                start = d
            elif d.get('event') == 'placement':
                placement.append(d)
            elif d.get('experiment') == 'concurrent-tier':
                windows.append(d)
        runs.append((path.stem, start, windows, placement))
    return runs


def interval_table(series):
    """Merge the per-event series into per-interval records {t0, t1, socket: {ddr, hbm, upi_rx, upi_tx, pkg_j, ram_j}}."""
    times = sorted({(t0, t1) for ser in series.values() for t0, t1, _, _ in ser})
    rows = [{'t0': a, 't1': b, 's': {0: defaultdict(float), 1: defaultdict(float)}} for a, b in times]
    index = {t: i for i, t in enumerate(times)}
    for (sock, name), ser in series.items():
        for t0, t1, v, _ in ser:
            if v is not None and sock in (0, 1):
                rows[index[(t0, t1)]]['s'][sock][name] += v
    return rows


def traffic(row):
    return sum(row['s'][s]['ddr'] * DDR_CAS + row['s'][s]['hbm'] * HBM_CAS for s in (0, 1))


def bursts(rows, threshold_gbps=2.0):
    """Maximal runs of consecutive intervals whose memory traffic exceeds threshold_gbps."""
    out, cur = [], []
    for row in rows:
        dt = row['t1'] - row['t0']
        if dt > 0 and traffic(row) / dt / 1e9 > threshold_gbps:
            cur.append(row)
        elif cur:
            out.append(cur)
            cur = []
    if cur:
        out.append(cur)
    return out


def burst_counters(burst, upi_flit_bytes):
    res = {}
    dur = burst[-1]['t1'] - burst[0]['t0']
    for s in (0, 1):
        tot = defaultdict(float)
        for row in burst:
            for k, v in row['s'][s].items():
                tot[k] += v
        interior = burst[1:-1] if len(burst) > 2 else burst
        idt = sum(r['t1'] - r['t0'] for r in interior)
        res[s] = {
            'ddr_bytes': tot['ddr'] * DDR_CAS, 'hbm_bytes': tot['hbm'] * HBM_CAS,
            'upi_rx_bytes': tot['upi_rx'] * upi_flit_bytes, 'upi_tx_bytes': tot['upi_tx'] * upi_flit_bytes,
            'pkg_j': tot['pkg_j'], 'ram_j': tot['ram_j'],
            'pkg_w': sum(r['s'][s]['pkg_j'] for r in interior) / idt if idt else 0.0,
            'ram_w': sum(r['s'][s]['ram_j'] for r in interior) / idt if idt else 0.0,
            'burst_s': dur, 'intervals': len(burst),
        }
    return res


def expected_bytes(w):
    """Computed bytes per tier of one socket's window: what the workers issued (kind=inter: split by the page ratio)."""
    if w['kind'] == 2:
        tot = w['hbm_bytes'] + w['ddr_bytes']
        a, b = w['a'], w['b']
        return tot * a / (a + b), tot * b / (a + b)
    return w['hbm_bytes'], w['ddr_bytes']


def attach(results, upi_flit_bytes=xs.UPI_DATA_FLIT_BYTES):
    """All windows of a results directory, each with its measured counters and the expected-vs-measured check."""
    windows = []
    for label, start, wins, placement in load_windows(results):
        stage, page, rep = parse_label(label)
        perf = Path(results, 'raw', label + '.perf.csv')
        by_index = defaultdict(list)
        for w in wins:
            by_index[w['window']].append(w)
        counters = None
        note = ''
        if perf.exists() and perf.stat().st_size:
            rows = interval_table(xs.parse_perf(perf))
            bs = bursts(rows)
            if len(bs) == len(by_index):
                counters = [burst_counters(b, upi_flit_bytes) for b in bs]
            else:
                note = f'{len(bs)} bursts for {len(by_index)} windows'
        else:
            note = 'no perf data'
        for k, idx in enumerate(sorted(by_index)):
            for w in by_index[idx]:
                w = dict(w, label=label, stage=stage, page_size=page, rep_index=rep, placement=placement)
                if counters:
                    c = counters[k][w['socket']]
                    exp_h, exp_d = expected_bytes(w)
                    w['counters'] = c
                    w['expected'] = {'hbm_bytes': exp_h, 'ddr_bytes': exp_d}
                    w['counter_ratio'] = {'hbm': c['hbm_bytes'] / exp_h if exp_h > 5e8 else None,
                                          'ddr': c['ddr_bytes'] / exp_d if exp_d > 5e8 else None}
                    w['counters_total_gbps'] = (c['hbm_bytes'] + c['ddr_bytes']) / c['burst_s'] / 1e9
                else:
                    w['counter_note'] = note
                windows.append(w)
    return windows


# ---- aggregation ------------------------------------------------------------------------------------------------------
def med_min_max(xs_):
    xs_ = [x for x in xs_ if x is not None]
    if not xs_:
        return None
    return {'median': statistics.median(xs_), 'min': min(xs_), 'max': max(xs_), 'n': len(xs_)}


def node_value(group, key):
    """Per-repetition sum over the sockets of a window field, aggregated over the repetitions."""
    per_rep = defaultdict(float)
    for w in group:
        per_rep[(w['label'], w['window'])] += w[key]
    return med_min_max(list(per_rep.values()))


def aggregate(windows):
    """{(name, page): {...}} medians over the repetitions of each configuration; sockets summed for a two-socket run."""
    groups = defaultdict(list)
    for w in windows:
        groups[(w['name'], w['page'], w['sockets'])].append(w)
    out = {}
    for (name, page, sockets), g in groups.items():
        s0 = [w for w in g if w['socket'] == 0]
        e = {'name': name, 'page': page, 'sockets': sockets, 'reps': len(s0), 'cfg': {k: s0[0][k] for k in (
            'kind', 'ch', 'cd', 'a', 'b', 'unit', 'streams', 'pf', 'hint', 'nt', 'dstreams', 'dpf', 'dhint', 'dnt', 'alt', 'probe',
            'volume', 'gs', 'r', 'cdw', 'act', 'gshared')}}
        for key in ('hbm_gbps', 'ddr_gbps', 'combined_gbps', 'volume_gbps', 'common_hbm_gbps', 'common_ddr_gbps', 'common_s',
                    'gemv_pass_s', 'gemv_best_pass_s'):
            e[key] = node_value(g, key)
        e['probe_ns'] = med_min_max([w['probe_ns'] for w in s0 if w['probe_ns']])
        e['socket0_combined'] = med_min_max([w['combined_gbps'] for w in s0])
        # counters (per node: both sockets summed)
        cw = [w for w in g if 'counters' in w]
        if cw:
            per = defaultdict(lambda: defaultdict(float))
            for w in cw:
                c = w['counters']
                k = (w['label'], w['window'])
                for f in ('hbm_bytes', 'ddr_bytes', 'upi_rx_bytes', 'upi_tx_bytes', 'pkg_j', 'ram_j', 'pkg_w', 'ram_w'):
                    per[k][f] += c[f]
                per[k]['burst_s'] = c['burst_s']
                per[k]['exp_h'] += w['expected']['hbm_bytes']
                per[k]['exp_d'] += w['expected']['ddr_bytes']
            e['counter_hbm_gbps'] = med_min_max([v['hbm_bytes'] / v['burst_s'] / 1e9 for v in per.values()])
            e['counter_ddr_gbps'] = med_min_max([v['ddr_bytes'] / v['burst_s'] / 1e9 for v in per.values()])
            e['upi_gbps'] = med_min_max([(v['upi_rx_bytes'] + v['upi_tx_bytes']) / v['burst_s'] / 1e9 for v in per.values()])
            e['pkg_w'] = med_min_max([v['pkg_w'] for v in per.values()])
            e['ram_w'] = med_min_max([v['ram_w'] for v in per.values()])
            ratios_h = [w['counter_ratio']['hbm'] for w in cw if w['counter_ratio']['hbm'] is not None]
            ratios_d = [w['counter_ratio']['ddr'] for w in cw if w['counter_ratio']['ddr'] is not None]
            e['ratio_hbm'] = med_min_max(ratios_h)
            e['ratio_ddr'] = med_min_max(ratios_d)
        out[(name, page, sockets)] = e
    return out


def best_hbm_only(windows):
    """Best HBM-only reading configuration of a base sweep: highest median over the repetitions."""
    agg = aggregate([w for w in windows if w['name'].startswith('base-hbm') and w['kind'] == 0 and w['cd'] == 0])
    return sorted(agg.values(), key=lambda e: -e['combined_gbps']['median'])


def best_ddr_only(windows):
    agg = aggregate([w for w in windows if w['name'].startswith('base-ddr') and w['kind'] == 0 and w['ch'] == 0])
    return sorted(agg.values(), key=lambda e: -e['combined_gbps']['median'])


def best_points(results_dirs, env):
    """Best E1 split, E2 ratio and E3 ratio of earlier jobs (defaults when nothing is there yet)."""
    best = {'e1_cd': 8, 'e2_ratio': (4, 1), 'e2_unit': 128, 'e3_ratio': (4, 1)}
    wins = []
    for d in results_dirs:
        wins += attach(d)
    if not wins:
        return best
    agg = aggregate(wins)

    def top(prefix, pred=lambda e: True):
        c = [e for e in agg.values() if e['name'].startswith(prefix) and pred(e)]
        return max(c, key=lambda e: e['combined_gbps']['median']) if c else None

    e1 = top('e1-cd')
    if e1:
        best['e1_cd'] = e1['cfg']['cd']
    e2 = top('e2-', lambda e: e['cfg']['a'] and e['cfg']['b'])
    if e2:
        best['e2_ratio'] = (e2['cfg']['a'], e2['cfg']['b'])
        best['e2_unit'] = e2['cfg']['unit']
    e3 = top('e3-', lambda e: e['cfg']['a'] and e['cfg']['b'])
    if e3:
        best['e3_ratio'] = (e3['cfg']['a'], e3['cfg']['b'])
    return best


def cmd_cfgs(args):
    env = read_env(args.best_env)
    for line in gen(args.stage, env, args.results):
        print(line)


def cmd_best(args):
    wins = attach(args.results)
    h, d = best_hbm_only(wins), best_ddr_only(wins)
    if not h or not d:
        raise SystemExit('base sweep has no HBM-only or DDR-only windows')
    bh, bd = h[0], d[0]
    print(f'BEST_S={bh["cfg"]["streams"]}\nBEST_PF={bh["cfg"]["pf"]}\nBEST_HINT={bh["cfg"]["hint"]}\nBEST_NT={bh["cfg"]["nt"]}')
    print(f'BEST_DS={bd["cfg"]["streams"]}\nBEST_DPF={bd["cfg"]["pf"]}\nBEST_DHINT={bd["cfg"]["hint"]}\nBEST_DNT={bd["cfg"]["nt"]}')
    print(f'BEST_HBM_GBPS={bh["combined_gbps"]["median"]:.1f}\nBEST_HBM_CORES={bh["cfg"]["ch"]}\nBEST_HBM_PAGE={bh["page"]}')
    print(f'BEST_DDR_GBPS={bd["combined_gbps"]["median"]:.1f}\nBEST_DDR_CORES={bd["cfg"]["cd"]}\nBEST_DDR_PAGE={bd["page"]}')


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest='cmd', required=True)
    p = sub.add_parser('cfgs')
    p.add_argument('stage')
    p.add_argument('--best-env')
    p.add_argument('--results', nargs='*', default=[])
    p.set_defaults(fn=cmd_cfgs)
    p = sub.add_parser('best')
    p.add_argument('results')
    p.set_defaults(fn=cmd_best)
    args = ap.parse_args(argv)
    args.fn(args)


if __name__ == '__main__':
    main()
