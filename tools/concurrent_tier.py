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
PERF_ORIGIN_S = 0.87  # perf's time origin precedes the 'start' event by this much (calibrated on 4590 windows)
CORES = 56
TSC_GHZ = 1.9  # reference-cycle rate of the Xeon Max 9480 (turbostat TSC_MHz 1900)
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
            for act in (14, 28, 32, 42, 48, 56):
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
    elif stage in ('fillcmp-random', 'fillcmp-zero'):  # the same reads over pseudo-random and over all-zero tier regions
        tag = stage.split('-')[1]
        p = tier_params(env, 0)[0]
        add(split_cfg(f'fc{tag}-hbm56', CORES, 0, env))
        add(split_cfg(f'fc{tag}-split48-8', 48, 8, env))
        add(split_cfg(f'fc{tag}-split28-28', 28, 28, env))
        add(split_cfg(f'fc{tag}-ddr56', 0, CORES, env))
        add(fmt(name=f'fc{tag}-shared-4-1', kind='shared', a=4, b=1, unit=128, **p))
        add(fmt(name=f'fc{tag}-inter-1-0', kind='inter', a=1, b=0, **p))
        add(fmt(name=f'fc{tag}-inter-4-1', kind='inter', a=4, b=1, **p))
        for gs in (64, 32):
            for act in (52, 56):
                add(fmt(name=f'fc{tag}-gemv{gs}-act{act}', kind='gemv', gs=gs, r=0, cdw=0, act=act))
    elif stage == 'e5c':  # confirmation of the E5 points near the all-HBM level, run in a shuffled order
        for gs in (64, 32):
            for act in (48, 52, 56):
                add(fmt(name=f'e5c-gs{gs}-act{act}', kind='gemv', gs=gs, r=0, cdw=0, act=act))
            for r, cdw in ((0.05, 3), (0.1, 6), (0.2, 14), (0.25, 18), (0.3, 21), (0.35, 24), (0.4, 28)):
                add(fmt(name=f'e5c-gs{gs}-r{r}-cdw{cdw}', kind='gemv', gs=gs, r=r, cdw=cdw))
    elif stage == 'e5off':  # alignment test of the all-HBM GEMV: every weight address moved up by `off` bytes
        for gs in (64, 32):
            for off in (0, 64, 4096, 65536, 1048640):
                add(fmt(name=f'e5off-gs{gs}-act56-off{off}', kind='gemv', gs=gs, r=0, cdw=0, act=56, off=off))
    elif stage == 'e6b':  # sensitivity at real splits (cd = 4, 8, 16): DDR core placement, DDR group settings, HBM group prefetch
        p, d = tier_params(env, 0)
        for cd in (4, 8, 16):
            base = dict(kind='split', ch=CORES - cd, cd=cd)
            for alt in (0, 1):
                add(fmt(name=f'e6b-cd{cd}-alt{alt}', **base, **p, **d, alt=alt))
            for ds in (1, 2, 4):
                add(fmt(name=f'e6b-cd{cd}-dS{ds}', **base, **p, dS=ds, dpf=d['dpf'], dhint=d['dhint'], dnt=d['dnt']))
            for dpf in (0, 4096, 16384):
                add(fmt(name=f'e6b-cd{cd}-dpf{dpf}', **base, **p, dS=d['dS'], dpf=dpf, dhint=d['dhint'], dnt=d['dnt']))
            add(fmt(name=f'e6b-cd{cd}-dnt1', **base, **p, dS=d['dS'], dpf=d['dpf'], dhint=d['dhint'], dnt=1))
            for pf in (4096, 65536):
                add(fmt(name=f'e6b-cd{cd}-hpf{pf}', **base, S=p['S'], pf=pf, hint=p['hint'], nt=p['nt'], **d))
    elif stage == 'hbmx':  # HBM-only baseline extension: longer prefetch, hint and non-temporal variants at one stream
        for pf in (16384, 32768, 65536, 131072):
            add(fmt(name=f'hbmx-S1-pf{pf}', kind='split', ch=CORES, cd=0, S=1, pf=pf, hint=0, nt=0))
        add(fmt(name='hbmx-S1-pf16384-nt1', kind='split', ch=CORES, cd=0, S=1, pf=16384, hint=0, nt=1))
        add(fmt(name='hbmx-S1-pf16384-hint1', kind='split', ch=CORES, cd=0, S=1, pf=16384, hint=1, nt=0))
        add(fmt(name='hbmx-S1-pf16384-hint2', kind='split', ch=CORES, cd=0, S=1, pf=16384, hint=2, nt=0))
    elif stage == 'e5alias':  # stripe-aliasing test of the all-HBM GEMV and of a raw read
        for gs in (64, 32):
            for act in (27, 28, 29, 53, 54, 55, 56):
                add(fmt(name=f'e5alias-gs{gs}-act{act}', kind='gemv', gs=gs, r=0, cdw=0, act=act))
            for skew in (4096, 65536):
                add(fmt(name=f'e5alias-gs{gs}-act56-skew{skew}', kind='gemv', gs=gs, r=0, cdw=0, act=56, skew=skew))
        for stag in (0, 4096, 1052672):  # worker w starts w x stag bytes into its 360 MiB stripe (stripes are multiples of 2 MiB)
            add(fmt(name=f'e5alias-raw-lock-stag{stag}', kind='split', ch=CORES, cd=0, S=1, pf=16384, stripe=360, lock=2048, stag=stag))
            add(fmt(name=f'e5alias-raw-timed-stag{stag}', kind='split', ch=CORES, cd=0, S=1, pf=16384, stripe=360, stag=stag))
        add(fmt(name='e5alias-raw-sliced', kind='split', ch=CORES, cd=0, S=1, pf=16384))
    elif stage == 'e5c2':  # E5 row split with the DDR worker count +-2 around the best point, r = 0.2 .. 0.3
        for gs in (64, 32):
            for act in (52, 56):
                add(fmt(name=f'e5c2-gs{gs}-act{act}', kind='gemv', gs=gs, r=0, cdw=0, act=act))
            for r, centre in ((0.2, 14), (0.25, 18), (0.3, 21)):
                for cdw in (centre - 2, centre, centre + 2):
                    add(fmt(name=f'e5c2-gs{gs}-r{r}-cdw{cdw}', kind='gemv', gs=gs, r=r, cdw=cdw))
    elif stage == 'repe3':  # shuffled replication on the second node
        p = tier_params(env, 0)[0]
        for a, b in ((1, 0), (4, 1), (8, 1)):
            add(fmt(name=f'e3-{a}-{b}', kind='inter', a=a, b=b, **p))
    elif stage == 'repe4':
        p = tier_params(env, 0)[0]
        add(split_cfg('e4-hbm56', CORES, 0, env))
        add(fmt(name='e4-inter-8-1', kind='inter', a=8, b=1, **p))
    elif stage == 'repe5':
        for gs in (64, 32):
            add(fmt(name=f'e5c-gs{gs}-act56', kind='gemv', gs=gs, r=0, cdw=0, act=56))
            add(fmt(name=f'e5c-gs{gs}-act52', kind='gemv', gs=gs, r=0, cdw=0, act=52))
            for r, cdw in ((0.1, 6), (0.2, 14), (0.25, 18)):
                add(fmt(name=f'e5c-gs{gs}-r{r}-cdw{cdw}', kind='gemv', gs=gs, r=r, cdw=cdw))
    elif stage == 'e1g':  # HBM-only and splits with the HBM workers on stripes of 360 MiB started stag x i bytes apart
        for stag in (0, 4096, 1052672):
            add(fmt(name=f'e1g-hbm-timed-stag{stag}', kind='split', ch=CORES, cd=0, S=1, pf=16384, stripe=360, stag=stag))
            add(fmt(name=f'e1g-hbm-lock-stag{stag}', kind='split', ch=CORES, cd=0, S=1, pf=16384, stripe=360, lock=2048, stag=stag))
        add(fmt(name='e1g-hbm-sliced', kind='split', ch=CORES, cd=0, S=1, pf=16384))
        for cd in (4, 8, 16):
            add(split_cfg(f'e1g-cd{cd}-alt1-sliced', CORES - cd, cd, env, alt=1))
            for stag in (0, 4096, 1052672):
                add(split_cfg(f'e1g-cd{cd}-alt1-stag{stag}', CORES - cd, cd, env, alt=1, stripe=360, stag=stag))
    elif stage == 'e3g':  # interleaved region, every worker's start shifted stag x i bytes into its slice
        p = tier_params(env, 0)[0]
        for a, b in ((1, 0), (4, 1), (8, 1)):
            for stag in (0, 4096, 1052672):
                add(fmt(name=f'e3g-{a}-{b}-stag{stag}', kind='inter', a=a, b=b, stag=stag, **p))
    elif stage == 'freq':  # long windows, to read the core and uncore frequencies (turbostat) under HBM-only, mixed and DDR-only load
        add(split_cfg('freq-hbm56', CORES, 0, env, warm=0.5, seconds=8))
        add(split_cfg('freq-split48-8', 48, 8, env, warm=0.5, seconds=8))
        add(split_cfg('freq-split28-28', 28, 28, env, warm=0.5, seconds=8))
        add(split_cfg('freq-ddr56', 0, CORES, env, warm=0.5, seconds=8))
    elif stage == 'rep':  # the key E1 points, for the replication job on the other node
        for cd in (0, 4, 8, 16, 28, 56):
            add(split_cfg(f'e1-cd{cd}', CORES - cd, cd, env))
        add(split_cfg('e1v-cd8', CORES - 8, 8, env, volume=6144))
    elif stage == 'rep-e5':
        for gs in (64, 32):
            for r, cdw in ((0, 0), (0.1, 6), (0.2, 11), (0.3, 17), (1.0, 55)):
                add(fmt(name=f'e5-gs{gs}-r{r}-cdw{cdw}', kind='gemv', gs=gs, r=r, cdw=cdw))
            for act in (28, 42):
                add(fmt(name=f'e5hbm-gs{gs}-act{act}', kind='gemv', gs=gs, r=0, cdw=0, act=act))
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
            # UPI over the interior intervals only: the edge intervals contain the allocation and release of the interleaved regions
            'upi_rx_bytes': sum(r['s'][s]['upi_rx'] for r in interior) * upi_flit_bytes,
            'upi_tx_bytes': sum(r['s'][s]['upi_tx'] for r in interior) * upi_flit_bytes, 'interior_s': idt,
            'pkg_j': tot['pkg_j'], 'ram_j': tot['ram_j'],
            # average core clock over the interior intervals of the burst (the first and last interval hold the idle edges where
            # the spinning workers run at another clock): unhalted cycles / reference cycles x 1.9 GHz TSC, all 56 cores of the socket
            'freq_ghz': (sum(r['s'][s]['cycles'] for r in interior) / sum(r['s'][s]['ref_cycles'] for r in interior) * TSC_GHZ
                         if sum(r['s'][s]['ref_cycles'] for r in interior) else None),
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
        align = ''
        if perf.exists() and perf.stat().st_size:
            rows = interval_table(xs.parse_perf(perf))
            bs = bursts(rows)
            order = sorted(by_index)
            if len(bs) == len(order):
                counters = [burst_counters(b, upi_flit_bytes) for b in bs]  # k-th burst = k-th window
                align = 'order'
            else:
                # Extra bursts (the kernel zeroing and touching the 32 GiB region built before an E3/E4 window): pair every
                # window with the burst whose start is nearest to where the window must start on the perf clock, perf's origin
                # being PERF_ORIGIN_S before the 'start' event. Each burst serves one window; no burst within 0.6 s = no counters.
                counters, used = [], set()
                for idx in order:
                    w0 = by_index[idx][0]
                    t_exp = w0['real_begin'] - start['real_time'] + PERF_ORIGIN_S if start else None
                    best = None
                    for j, b in enumerate(bs):
                        if j in used or t_exp is None:
                            continue
                        d = abs(b[0]['t0'] - t_exp)
                        if d < 0.6 and (best is None or d < best[0]):
                            best = (d, j)
                    if best:
                        used.add(best[1])
                        counters.append(burst_counters(bs[best[1]], upi_flit_bytes))
                    else:
                        counters.append(None)
                align = f'time ({len(bs)} bursts, {len(order)} windows)'
                if all(c is None for c in counters):
                    counters, note = None, f'{len(bs)} bursts for {len(order)} windows, none within 0.6 s'
        else:
            note = 'no perf data'
        for k, idx in enumerate(sorted(by_index)):
            for w in by_index[idx]:
                w = dict(w, label=label, stage=stage, page_size=page, rep_index=rep, placement=placement)
                if counters and counters[k] is None:
                    w['counter_note'] = 'no burst near the window'
                elif counters:
                    c = counters[k][w['socket']]
                    w['align'] = align
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
    return {'median': statistics.median(xs_), 'min': min(xs_), 'max': max(xs_), 'n': len(xs_), 'vals': list(xs_)}


def perm_p(a, b):
    """One-sided exact permutation p-value of median(a) - median(b) >= observed, over all splits of the pooled values."""
    import itertools
    pool = list(a) + list(b)
    obs = statistics.median(a) - statistics.median(b)
    n, hits, total = len(a), 0, 0
    for idx in itertools.combinations(range(len(pool)), n):
        s = set(idx)
        x = [pool[i] for i in idx]
        y = [pool[i] for i in range(len(pool)) if i not in s]
        total += 1
        hits += statistics.median(x) - statistics.median(y) >= obs - 1e-9
    return hits / total


def pcol(e, ref):
    """p-value that e's combined rate exceeds ref's by chance (permutation test over the repetitions); '-' when e is not above ref."""
    if not e or not ref or e['combined_gbps']['median'] <= ref['combined_gbps']['median']:
        return '-'
    return f'{perm_p(e["combined_gbps"]["vals"], ref["combined_gbps"]["vals"]):.3f}'


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
        # The other timer (same bytes and interval as the worker rate, only the clock differs): the bytes issued during the measured
        # interval (warm-up excluded) over the wall time from the common start of that interval to the end of the window (the join of
        # the team). Measured bytes = sum of the workers' counted bytes = rate x mean worker duration (a GEMV pass: bytes per pass x 16
        # measured passes); the measured interval starts `warm` seconds (GEMV: two warm-up passes of the measured pass time) after the
        # common start real_begin. Fixed-volume windows are not given.
        wb, wd = defaultdict(float), {}
        for w in g:
            if w['volume'] or not w['mean_worker_s']:
                continue
            gemv = w['kind'] == 3
            k = (w['label'], w['window'])
            wb[k] += w['combined_gbps'] * 1e9 * w['mean_worker_s'] * (16 if gemv else 1)
            wd[k] = w['real_end'] - w['real_begin'] - (2 * w['mean_worker_s'] if gemv else w['warm'])
        e['wall_gbps'] = med_min_max([wb[k] / wd[k] / 1e9 for k in wb if wd[k] > 0])
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
                per[k]['interior_s'] = c['interior_s']
                per[k]['exp_h'] += w['expected']['hbm_bytes']
                per[k]['exp_d'] += w['expected']['ddr_bytes']
            e['counter_hbm_gbps'] = med_min_max([v['hbm_bytes'] / v['burst_s'] / 1e9 for v in per.values()])
            e['counter_ddr_gbps'] = med_min_max([v['ddr_bytes'] / v['burst_s'] / 1e9 for v in per.values()])
            e['ddr_share'] = med_min_max([v['ddr_bytes'] / (v['hbm_bytes'] + v['ddr_bytes']) for v in per.values()])
            e['upi_gbps'] = med_min_max([(v['upi_rx_bytes'] + v['upi_tx_bytes']) / v['interior_s'] / 1e9 for v in per.values() if v['interior_s']])
            e['pkg_w'] = med_min_max([v['pkg_w'] for v in per.values()])
            e['pkg_w0'] = med_min_max([w['counters']['pkg_w'] for w in cw if w['socket'] == 0])
            e['pkg_w1'] = med_min_max([w['counters']['pkg_w'] for w in cw if w['socket'] == 1])
            e['ram_w1'] = med_min_max([w['counters']['ram_w'] for w in cw if w['socket'] == 1])
            e['upi_flits_s'] = med_min_max([(w['counters']['upi_rx_bytes'] + w['counters']['upi_tx_bytes']) / (64 / 9) / w['counters']['interior_s']
                                            for w in cw if w['socket'] == 0])
            e['ram_w0'] = med_min_max([w['counters']['ram_w'] for w in cw if w['socket'] == 0])
            e['freq_ghz'] = med_min_max([w['counters']['freq_ghz'] for w in cw if w['socket'] == 0 and w['counters']['freq_ghz']])
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

    e1 = top('e1-cd', lambda e: e['cfg']['cd'] > 0)  # a real split: at least one DDR core
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


# ---- reporting --------------------------------------------------------------------------------------------------------
def mm(m, nd=1):
    """'median [min-max]' of an aggregate, or '-'."""
    if not m:
        return '-'
    return f'{m["median"]:.{nd}f} [{m["min"]:.{nd}f}-{m["max"]:.{nd}f}]'


def is_hbm_only(e):
    c = e['cfg']
    if c['kind'] == 0:
        return c['cd'] == 0 and c['ch'] > 0 and not c['probe']
    if c['kind'] in (1, 2):
        return c['b'] == 0 and c['a'] > 0
    return False


def is_ddr_only(e):
    c = e['cfg']
    if c['kind'] == 0:
        return c['ch'] == 0 and c['cd'] > 0 and not c['probe']
    if c['kind'] in (1, 2):
        return c['a'] == 0 and c['b'] > 0
    return False


def stage_of(name):
    return name.split('-')[0]


def idle_power(results):
    """Idle package / DRAM watts per socket from the quiet interval before the first window of every perf file."""
    pkg, ram = defaultdict(list), defaultdict(list)
    for path in sorted(Path(results, 'raw').glob('*.perf.csv')):
        rows = interval_table(xs.parse_perf(path))
        bs = bursts(rows)
        if not bs:
            continue
        first = bs[0][0]['t0']
        quiet = [r for r in rows if r['t1'] <= first - 0.2 and r['t0'] > 0.2]
        dt = sum(r['t1'] - r['t0'] for r in quiet)
        if dt < 0.3:
            continue
        for s in (0, 1):
            pkg[s].append(sum(r['s'][s]['pkg_j'] for r in quiet) / dt)
            ram[s].append(sum(r['s'][s]['ram_j'] for r in quiet) / dt)
    return {s: (statistics.median(pkg[s]) if pkg[s] else None, statistics.median(ram[s]) if ram[s] else None) for s in (0, 1)}


def hbm_only_reference(agg, min_reps=7):
    # pseudo-random data only: zero-filled regions read faster under the power cap (see the data-content table)
    cands = [e for e in agg.values() if is_hbm_only(e) and e['sockets'] == 1 and e['reps'] >= min_reps and e['cfg']['kind'] != 3
             and not e['name'].startswith(('fczero', 'zfill', 'e1v')) and '-lock-' not in e['name']]  # timed windows only: never fixed-volume or barrier-lockstep windows
    return max(cands, key=lambda e: e['combined_gbps']['median']) if cands else None


def describe(e):
    c = e['cfg']
    if c['kind'] == 0:
        s = f'{c["ch"]} HBM + {c["cd"]} DDR cores'
    elif c['kind'] == 1:
        s = f'shared {c["a"]}:{c["b"]} unit {c["unit"]}'
    elif c['kind'] == 2:
        s = f'pages {c["a"]}:{c["b"]}'
    else:
        s = f'gs{c["gs"]} r={c["r"]} cdw={c["cdw"]}{" shared" if c["gshared"] else ""}'
    return f'{s}, S={c["streams"]} pf={c["pf"]} hint={c["hint"]} nt={c["nt"]}, {e["page"]}'


def table(header, rows):
    out = ['| ' + ' | '.join(header) + ' |', '|' + '|'.join('---' for _ in header) + '|']
    out += ['| ' + ' | '.join(str(x) for x in r) + ' |' for r in rows]
    return '\n'.join(out)


def e1_rows(agg, prefix, page):
    rows = []
    for e in sorted((e for e in agg.values() if e['name'].startswith(prefix) and e['page'] == page and e['sockets'] == 1),
                    key=lambda e: e['cfg']['cd']):
        rows.append((e['cfg']['ch'], e['cfg']['cd'], mm(e['hbm_gbps']), mm(e['ddr_gbps']), mm(e['combined_gbps']), e['reps'], e))
    return rows


def bytes_per_token(gs):
    return BYTES_TOKEN_GS64 * GS_FACTOR[gs]


def tables(agg, windows, idle):
    md = []
    ref = hbm_only_reference(agg)
    md.append(f'## Best HBM-only reference (>= 7 repetitions)\n\n{mm(ref["combined_gbps"]) if ref else "-"} GB/s per socket: '
              f'{describe(ref) if ref else "none"} ({ref["name"] if ref else ""}).\n')
    locks = [e for e in agg.values() if '-lock-' in e['name'] and e['sockets'] == 1 and e['reps'] >= 7]
    if locks:
        bl = max(locks, key=lambda e: e['combined_gbps']['median'])
        md.append(f'Best barrier-lockstep HBM-only read (a different measurement: fixed 1 MiB blocks, a team barrier after each): {mm(bl["combined_gbps"])} GB/s ({bl["name"]}, {bl["page"]}).\n')
    base = [e for e in agg.values() if e['name'].startswith('base-') and e['sockets'] == 1]
    for tier, pred in (('HBM-only', is_hbm_only), ('DDR-only', is_ddr_only)):
        top = sorted((e for e in base if pred(e)), key=lambda e: -e['combined_gbps']['median'])[:12]
        if top:
            md.append(f'## Base sweep, best {tier} configurations (median of the repetitions, GB/s per socket)\n')
            md.append(table(['config', 'page', 'reps', 'GB/s median [min-max]'],
                            [(describe(e).rsplit(',', 1)[0], e['page'], e['reps'], mm(e['combined_gbps'])) for e in top]))
            md.append('')
    for tier, pred, key in (('HBM-only', is_hbm_only, 'ch'), ('DDR-only', is_ddr_only, 'cd')):
        rows = []
        for e in sorted((e for e in agg.values() if e['sockets'] == 1 and e['name'].startswith('e1-') and pred(e)),
                        key=lambda e: (e['page'], e['cfg'][key])):
            rows.append((e['page'], e['cfg'][key], mm(e['combined_gbps']), e['reps']))
        if rows:
            md.append(f'## {tier} bandwidth against the number of cores (E1 reference points, GB/s)\n')
            md.append(table(['page', 'cores', 'GB/s median [min-max]', 'reps'], rows))
            md.append('')
    for page in ('2m', '4k'):
        rows = e1_rows(agg, 'e1-cd', page)
        if not rows:
            continue
        md.append(f'## E1 split teams, {page} pages (one socket; GB/s, median [min-max] of the repetitions; fixed-time windows)\n')
        body = []
        for ch, cd, h, d, c, n, e in rows:
            v = agg.get((e['name'].replace('e1-', 'e1v-'), page, 1))
            body.append((ch, cd, h, d, c, n, mm(v['volume_gbps']) if v else '-', mm(v['common_hbm_gbps']) if v else '-',
                         mm(v['common_ddr_gbps']) if v else '-', mm(e.get('freq_ghz'), 2),
                         f'{e["pkg_w0"]["median"]:.0f}' if e.get('pkg_w0') else '-', pcol(e, agg.get(('e1-cd0', page, 1)))))
        md.append(table(['HBM cores', 'DDR cores', 'HBM GB/s', 'DDR GB/s', 'combined GB/s', 'reps', 'fixed-volume run: total volume / total time',
                         'fixed-volume run: HBM GB/s over the common interval', 'DDR GB/s over the common interval', 'core GHz (counters)', 'package W (socket 0)', 'p: above e1-cd0 by chance'], body))
        best = max(rows, key=lambda r: r[6]['combined_gbps']['median'])
        e = best[6]
        gain = e['combined_gbps']['median'] / ref['combined_gbps']['median'] - 1 if ref else float('nan')
        md.append(f'\nBest split {page}: {best[0]} HBM + {best[1]} DDR cores, {mm(e["combined_gbps"])} GB/s, '
                  f'{gain * 100:+.1f}% against the best HBM-only reference.\n')
    for page in ('2m', '4k'):
        es = sorted((e for e in agg.values() if e['name'].startswith('e2-') and e['page'] == page and e['sockets'] == 1 and e['cfg']['kind'] == 1),
                    key=lambda e: (e['cfg']['unit'], e['cfg']['b'] / (e['cfg']['a'] + e['cfg']['b'])))
        if es:
            md.append(f'## E2 shared cores, {page} pages (all 56 cores read both tiers; per-socket GB/s)\n')
            md.append(table(['variant', 'HBM:DDR bytes', 'DDR share', 'HBM GB/s', 'DDR GB/s', 'combined GB/s', 'reps', 'p: above e1-cd0 by chance'],
                            [(('line interleave' if e['cfg']['unit'] == 1 else f'chunks of {e["cfg"]["unit"]} lines'),
                              f'{e["cfg"]["a"]}:{e["cfg"]["b"]}', f'{e["cfg"]["b"] / (e["cfg"]["a"] + e["cfg"]["b"]):.2f}',
                              mm(e['hbm_gbps']), mm(e['ddr_gbps']), mm(e['combined_gbps']), e['reps'], pcol(e, agg.get(('e1-cd0', page, 1)))) for e in es]))
            md.append('')
    for page in ('2m', '4k'):
        es = sorted((e for e in agg.values() if e['name'].startswith('e3-') and e['page'] == page and e['sockets'] == 1),
                    key=lambda e: e['cfg']['b'] / (e['cfg']['a'] + e['cfg']['b']))
        if es:
            md.append(f'## E3 page-interleaved region, {page} pages (per-socket GB/s; tier split from the counters)\n')
            md.append(table(['HBM:DDR pages', 'page share on DDR', 'combined GB/s', 'DDR share of the counter bytes (HBM CAS x 32 B, DDR CAS x 64 B)', 'reps',
                             'p: above e1-cd0 by chance', 'p: above this region 1:0 by chance'],
                            [(f'{e["cfg"]["a"]}:{e["cfg"]["b"]}', f'{e["cfg"]["b"] / (e["cfg"]["a"] + e["cfg"]["b"]):.3f}', mm(e['combined_gbps']),
                              mm(e.get('ddr_share'), 3), e['reps'], pcol(e, agg.get(('e1-cd0', page, 1))),
                              pcol(e, agg.get((f'e3-1-0', page, 1)))) for e in es]))
            md.append('')
    es = sorted((e for e in agg.values() if e['name'].startswith(('zfill-e3-', 'zfill-e4-'))), key=lambda e: (e['sockets'], e['page'], e['name']))
    if es:
        md.append('## E3 and E4 of the first run, whose interleaved region held zeros (NOT comparable with E1; kept for the data-content finding)\n')
        md.append(table(['config', 'page', 'combined GB/s (node total if two sockets)', 'reps'], [(e['name'], e['page'], mm(e['combined_gbps']), e['reps']) for e in es]))
        md.append('')
    es = sorted((e for e in agg.values() if e['name'].startswith('e4-') and e['sockets'] == 2), key=lambda e: (e['page'], e['name']))
    if es:
        md.append('## E4 both sockets at once (node totals, GB/s; UPI traffic from the counters)\n')
        md.append(table(['config', 'page', 'node combined GB/s', 'socket-0 GB/s', 'package W socket 0 / socket 1', 'UPI GB/s (rx+tx, 64/9 B per flit assumed)',
                         'UPI Mflit/s (rx+tx, interior intervals)', 'reps', 'p: above e4-hbm56 by chance'],
                        [(e['name'], e['page'], mm(e['combined_gbps']), mm(e['socket0_combined']),
                          f'{e["pkg_w0"]["median"]:.0f} / {e["pkg_w1"]["median"]:.0f}' if e.get('pkg_w0') else '-', mm(e.get('upi_gbps'), 2),
                          f'{e["upi_gbps"]["median"] * 1e9 * 9 / 64 / 1e6:.1f}' if e.get('upi_gbps') else '-', e['reps'],
                          pcol(e, agg.get(('e4-hbm56', e['page'], 2)))) for e in es]))
        md.append('')
    def gemv_rows(es, ref_gbps, gs, href_e):
        rows = []
        ref_wall = href_e['wall_gbps']['median']
        for e in sorted(es, key=lambda e: (e['cfg']['r'], e['cfg']['gshared'], e['cfg']['act'], e['cfg']['cdw'])):
            tok = e['combined_gbps']['median'] * 1e9 / bytes_per_token(gs)
            gain = e['combined_gbps']['median'] / ref_gbps - 1
            act = e['cfg']['act'] or 56
            who = 'every worker' if e['cfg']['gshared'] else (f'{e["cfg"]["cdw"]} of {act}' if e['cfg']['r'] else f'none ({act} workers)')
            rows.append((e['page'], e['cfg']['r'], who, mm(e['hbm_gbps']), mm(e['ddr_gbps']), mm(e['combined_gbps']), f'{tok:.2f}',
                         f'{gain * 100:+.1f}%', e['reps'], pcol(e, href_e), mm(e['wall_gbps']),
                         f'{(e["wall_gbps"]["median"] / ref_wall - 1) * 100:+.1f}%'))
        return rows

    hdr = ['page', 'DDR row fraction r', 'DDR workers', 'HBM GB/s', 'DDR GB/s', 'total GB/s', 'tok/s-eq.', 'vs best all-HBM (both pages, any worker count)', 'reps', 'p: above the best all-HBM by chance', 'wall-clock GB/s (measured bytes / wall time from the common start of the measured interval to the end of the window)', 'vs best all-HBM, wall-clock']
    for prefixes, title in ((('e5-', 'e5hbm-'), 'E5 Q8 GEMV over the 70B per-socket shard, file order'),
                            (('e5c-',), 'E5 confirmation, every process in its own shuffled order'),
                            (('e5c2-',), 'E5 row split, DDR worker count +-2 around the best point, shuffled order')):
        for gs in (64, 32):
            es = [e for e in agg.values() if e['cfg']['kind'] == 3 and e['cfg']['gs'] == gs and e['sockets'] == 1 and e['name'].startswith(prefixes)
                  and not e['name'].startswith('e5off-')]
            hb = [e for e in es if e['cfg']['r'] == 0]
            if not es or not hb:
                continue
            href = max(hb, key=lambda e: e['combined_gbps']['median'])
            md.append(f'## {title}, gs{gs} (GB/s = bytes read per pass / pass time; tok/s-eq. = GB/s / {bytes_per_token(gs) / 1e9:.1f} GB per token)\n')
            md.append(f'Best all-HBM point: {mm(href["combined_gbps"])} GB/s ({href["name"]}, {href["page"]}, {href["cfg"]["act"] or 56} workers).\n')
            md.append(table(hdr, gemv_rows(es, href['combined_gbps']['median'], gs, href)))
            md.append('')
    es = sorted((e for e in agg.values() if e['name'].startswith('e5off-')), key=lambda e: (e['cfg']['gs'], e['page'], int(e['name'].rsplit('off', 1)[1])))
    if es:
        md.append('## GEMV alignment test: every weight address moved up by `off` bytes (all-HBM, 56 workers)\n')
        md.append(table(['gs', 'page', 'offset bytes', 'GB/s', 'reps'], [(e['cfg']['gs'], e['page'], e['name'].rsplit('off', 1)[1], mm(e['combined_gbps']), e['reps']) for e in es]))
        md.append('')
    for prefix, title in (('hbmx-', 'HBM-only baseline extension (S = 1; shuffled order)'),
                          ('e6b-', 'E6 at real splits cd = 4, 8, 16 (shuffled order)'),
                          ('e5alias-', 'GEMV and raw-read stripe-aliasing tests (shuffled order)'),
                          ('e1g-', 'HBM-only and splits with the HBM workers on staggered stripes (shuffled order)'),
                          ('e3g-', 'Interleaved region with every worker started stag x i bytes into its slice (shuffled order)')):
        es = sorted((e for e in agg.values() if e['name'].startswith(prefix) and e['sockets'] == 1), key=lambda e: (e['page'], e['name']))
        if es:
            md.append(f'## {title}\n')
            md.append(table(['config', 'page', 'HBM GB/s', 'DDR GB/s', 'combined GB/s', 'reps', 'p: above e1-cd0 by chance'],
                            [(e['name'], e['page'], mm(e['hbm_gbps']), mm(e['ddr_gbps']), mm(e['combined_gbps']), e['reps'],
                              pcol(e, agg.get(('e1-cd0', e['page'], 1))) if e['cfg']['kind'] in (0, 2) else '-') for e in es]))
            md.append('')
    es = sorted((e for e in agg.values() if e['name'].startswith('e6-')), key=lambda e: (e['page'], e['name']))
    if es:
        md.append('## E6 sensitivity (per-socket GB/s)\n')
        md.append(table(['config', 'page', 'HBM GB/s', 'DDR GB/s', 'combined GB/s', 'reps'],
                        [(e['name'], e['page'], mm(e['hbm_gbps']), mm(e['ddr_gbps']), mm(e['combined_gbps']), e['reps']) for e in es]))
        md.append('')
    es = sorted((e for e in agg.values() if e['name'].startswith('lat-')), key=lambda e: (e['page'], e['name']))
    if es:
        md.append('## Loaded latency (pointer chase on the last core while the others stream; ns per dependent load)\n')
        md.append(table(['config', 'page', 'probe tier', 'streaming cores (HBM+DDR)', 'ns per load', 'streaming GB/s', 'reps'],
                        [(e['name'], e['page'], 'HBM' if e['cfg']['probe'] == 1 else 'DDR', f'{e["cfg"]["ch"]}+{e["cfg"]["cd"]}',
                          mm(e['probe_ns']), mm(e['combined_gbps']), e['reps']) for e in es]))
        md.append('')
    md.append('## Counter bytes against computed bytes, per window and tier\n')
    ratios = defaultdict(list)
    bad = []
    for w in windows:
        r = w.get('counter_ratio')
        if not r:
            continue
        for tier in ('hbm', 'ddr'):
            if r[tier] is not None:
                ratios[(stage_of(w['name']), tier)].append(r[tier])
                if abs(r[tier] - 1) > 0.05:
                    exp = w['expected']['hbm_bytes' if tier == 'hbm' else 'ddr_bytes']
                    bad.append((w['label'], w['name'], w['socket'], tier, r[tier], exp / 1e9, w.get('align', '')))
    md.append(table(['stage', 'tier', 'windows', 'min', 'median', 'max', 'outside +-5%'],
                    [(k[0], k[1], len(v), f'{min(v):.3f}', f'{statistics.median(v):.3f}', f'{max(v):.3f}', sum(abs(x - 1) > 0.05 for x in v))
                     for k, v in sorted(ratios.items())]))
    md.append('')
    unattributed = sorted({(w['label'], w.get('counter_note')) for w in windows if 'counters' not in w})
    md.append(f'Runs without counters: {len(unattributed)} {unattributed[:6]}\n')
    # absolute closure per window family: counter GB against computed GB of each tier, and the excess
    fam = defaultdict(list)
    for w in windows:
        if 'counters' not in w:
            continue
        key = w['name'] if w['stage'] in ('e3', 'e4') or w['name'].startswith(('fcrandom', 'fczero')) else stage_of(w['name'])
        fam[(key.replace('zfill-', ''), w['sockets'], w['kind'] == 2)].append(w)
    rows = []
    for (key, sockets, inter), ws in sorted(fam.items()):
        h = [w['counters']['hbm_bytes'] / 1e9 for w in ws]
        d = [w['counters']['ddr_bytes'] / 1e9 for w in ws]
        eh = [w['expected']['hbm_bytes'] / 1e9 for w in ws]
        ed = [w['expected']['ddr_bytes'] / 1e9 for w in ws]
        tot = [(a + b) / (c + e) for a, b, c, e in zip(h, d, eh, ed)]
        rows.append((key, sockets, len(ws), f'{statistics.median(eh):.1f}', f'{statistics.median(h):.1f}', f'{statistics.median(ed):.1f}',
                     f'{statistics.median(d):.1f}', f'{statistics.median(a - b for a, b in zip(d, ed)):+.2f}', f'{max(a - b for a, b in zip(d, ed)):+.2f}',
                     f'{statistics.median(tot):.3f}'))
    md.append('### CAS closure in absolute GB per window (median over windows; sockets = 2 counts each socket separately)\n')
    md.append(table(['config or stage', 'sockets', 'windows', 'computed HBM GB', 'HBM CAS GB', 'computed DDR GB', 'DDR CAS GB',
                     'DDR excess GB median', 'DDR excess GB max', '(HBM+DDR CAS)/(computed total)'], rows))
    md.append('')
    # aggregate check per run: all counter bytes of every attributed window of the process against all computed bytes
    runs = defaultdict(lambda: [0.0, 0.0, 0.0, 0.0, 0])
    for w in windows:
        if 'counters' in w:
            r = runs[w['label']]
            r[0] += w['counters']['hbm_bytes']
            r[1] += w['expected']['hbm_bytes']
            r[2] += w['counters']['ddr_bytes']
            r[3] += w['expected']['ddr_bytes']
            r[4] += 1
    per_run = [(k, v[4], v[0] / v[1] if v[1] > 5e8 else None, v[2] / v[3] if v[3] > 5e8 else None) for k, v in sorted(runs.items())]
    hb = [x[2] for x in per_run if x[2]]
    db = [x[3] for x in per_run if x[3]]
    if hb and db:
        md.append(f'Aggregate per run ({len(per_run)} processes): HBM counter/computed bytes min {min(hb):.4f} median {statistics.median(hb):.4f} '
                  f'max {max(hb):.4f}; DDR min {min(db):.4f} median {statistics.median(db):.4f} max {max(db):.4f}.\n')
        worst = [x for x in per_run if (x[2] and abs(x[2] - 1) > 0.02) or (x[3] and abs(x[3] - 1) > 0.02)]
        md.append(f'Processes with an aggregate ratio outside +-2%: {worst if worst else "none"}.\n')
    if bad:
        md.append(f'All {len(bad)} windows outside +-5% (run, config, socket, tier, counter/computed, computed GB of that tier in the window, attribution):\n')
        md.append(table(['run', 'config', 'socket', 'tier', 'ratio', 'computed GB', 'attribution'], [(a, b, c, d, f'{e:.3f}', f'{g:.1f}', h) for a, b, c, d, e, g, h in bad]))
        md.append('')
    fz = [e for e in agg.values() if e['name'].startswith(('fcrandom-', 'fczero-')) and e['sockets'] == 1]
    if fz:
        md.append('## Content of the regions: pseudo-random against all-zero data (same reads, 2m pages, per-socket GB/s)\n')
        rows = []
        for e in sorted(fz, key=lambda e: (e['name'].split('-', 1)[1], e['name'])):
            rows.append((e['name'], mm(e['combined_gbps']), mm(e.get('freq_ghz'), 2), f'{e["pkg_w0"]["median"]:.0f}' if e.get('pkg_w0') else '-', e['reps']))
        md.append(table(['config', 'combined GB/s', 'core GHz', 'package W', 'reps'], rows))
        md.append('')
    md.append('## Power and energy of socket 0 (RAPL package and DRAM energy over the interior intervals of each one-socket window)\n')
    ip = idle.get(0)
    md.append(f'Quiet interval before the first window of each process (the worker teams spin): package {ip[0] if ip else None} W, '
              f'DRAM {ip[1] if ip else None} W.\n')
    rows = []
    for name in ('e1-cd0', 'e1-cd56', 'e1-cd4', 'e1-cd8', 'e1-cd16', 'e1-cd28'):
        e = agg.get((name, '2m', 1))
        if e and e.get('pkg_w0') and ip and ip[0]:
            gbs = e['combined_gbps']['median']
            total = e['pkg_w0']['median'] + e['ram_w0']['median']
            rows.append((name, f'{gbs:.0f}', f'{e["pkg_w0"]["median"]:.0f}', f'{e["ram_w0"]["median"]:.0f}', f'{total / gbs:.3f}',
                         mm(e.get('freq_ghz'), 2)))
    if rows:
        md.append(table(['config (2m)', 'GB/s', 'package W', 'DRAM W', 'J/GB (package + DRAM of socket 0)', 'core GHz'], rows))
    return '\n'.join(md)


def model_table(agg):
    """Modelled 70B decode speed from the E1 per-tier rates: concurrent (balanced bytes), serial tiers, all-HBM."""
    ref = hbm_only_reference(agg)
    if not ref:
        return ''
    bw_h = ref['combined_gbps']['median']
    ddr_only = [e for e in agg.values() if is_ddr_only(e) and e['sockets'] == 1 and e['reps'] >= 7 and e['cfg']['kind'] == 0 and e['cfg']['cd'] == 56]
    bw_d = max((e['combined_gbps']['median'] for e in ddr_only), default=None)
    B = BYTES_TOKEN_GS64
    out = [f'Model inputs: BW_hbm(56 cores) = {bw_h:.1f} GB/s (best HBM-only), BW_ddr(56 cores) = '
           f'{"n/a" if bw_d is None else round(bw_d, 1)} GB/s, B = {B / 1e9:.1f} GB per token per socket (gs64).\n']
    rows = []
    for page in ('2m', '4k'):
        for ch, cd, h, d, c, n, e in e1_rows(agg, 'e1-cd', page):
            bh, bd = e['hbm_gbps']['median'], e['ddr_gbps']['median']
            if cd == 0 or bd <= 0 or bh <= 0 or bw_d is None:
                continue
            f = bd / (bh + bd)  # the byte fraction at which both groups finish together
            t_conc = max((1 - f) * B / (bh * 1e9), f * B / (bd * 1e9))
            t_serial = (1 - f) * B / (bw_h * 1e9) + f * B / (bw_d * 1e9)
            rows.append((page, f'{ch}+{cd}', f'{f:.3f}', f'{1 / t_conc:.2f}', f'{1 / t_serial:.2f}', f'{bw_h * 1e9 / B:.2f}'))
    out.append(table(['page', 'cores HBM+DDR', 'DDR byte fraction f (balanced)', 'tok/s concurrent (modelled)', 'tok/s serial tiers (modelled)',
                      'tok/s all-HBM (modelled)'], rows))
    return '\n'.join(out)


def make_figure(agg, path):
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
    except ImportError:
        return False
    ref = hbm_only_reference(agg)
    fig, ax = plt.subplots(2, 2, figsize=(13, 9))
    colors = {'2m': 'tab:blue', '4k': 'tab:orange'}

    def hline(a):
        if ref:
            a.axhline(ref['combined_gbps']['median'], color='k', lw=1, label='best HBM-only (56 cores)')

    a = ax[0][0]
    for page in ('2m', '4k'):
        rows = e1_rows(agg, 'e1-cd', page)
        if not rows:
            continue
        x = [r[1] for r in rows]
        es = [r[6] for r in rows]
        a.plot(x, [e['combined_gbps']['median'] for e in es], '-o', color=colors[page], label=f'combined, {page}')
        a.fill_between(x, [e['combined_gbps']['min'] for e in es], [e['combined_gbps']['max'] for e in es], color=colors[page], alpha=0.2)
        a.plot(x, [e['hbm_gbps']['median'] for e in es], '--', color=colors[page], alpha=0.7, label=f'HBM cores, {page}')
        a.plot(x, [e['ddr_gbps']['median'] for e in es], ':', color=colors[page], alpha=0.7, label=f'DDR cores, {page}')
    hline(a)
    a.set_xlabel('cores reading DDR (the other 56 - c_d read HBM)')
    a.set_ylabel('GB/s per socket')
    a.set_title('E1 split teams')
    a.legend(fontsize=7)
    a.grid(alpha=0.3)

    a = ax[0][1]
    for variant, units in (('chunks of 128 lines', (128,)), ('line interleave', (1,))):
        es = sorted((e for e in agg.values() if e['name'].startswith('e2-') and e['page'] == '2m' and e['sockets'] == 1 and e['cfg']['kind'] == 1
                     and e['cfg']['unit'] in units), key=lambda e: e['cfg']['b'] / (e['cfg']['a'] + e['cfg']['b']))
        if es:
            x = [e['cfg']['b'] / (e['cfg']['a'] + e['cfg']['b']) for e in es]
            a.plot(x, [e['combined_gbps']['median'] for e in es], '-o', label=f'combined, {variant}')
            a.fill_between(x, [e['combined_gbps']['min'] for e in es], [e['combined_gbps']['max'] for e in es], alpha=0.2)
            a.plot(x, [e['hbm_gbps']['median'] for e in es], '--', alpha=0.6, label=f'HBM part, {variant}')
            a.plot(x, [e['ddr_gbps']['median'] for e in es], ':', alpha=0.6, label=f'DDR part, {variant}')
    hline(a)
    a.set_xlabel('DDR share of the bytes each core reads (2 MiB pages)')
    a.set_ylabel('GB/s per socket')
    a.set_title('E2 shared cores')
    a.legend(fontsize=7)
    a.grid(alpha=0.3)

    a = ax[1][0]
    for page in ('2m', '4k'):
        es = sorted((e for e in agg.values() if e['name'].startswith('e3-') and e['page'] == page and e['sockets'] == 1),
                    key=lambda e: e['cfg']['b'] / (e['cfg']['a'] + e['cfg']['b']))
        if es:
            x = [e['cfg']['b'] / (e['cfg']['a'] + e['cfg']['b']) for e in es]
            a.plot(x, [e['combined_gbps']['median'] for e in es], '-o', color=colors[page], label=f'combined, {page}')
            a.fill_between(x, [e['combined_gbps']['min'] for e in es], [e['combined_gbps']['max'] for e in es], color=colors[page], alpha=0.2)
            if all(e.get('ddr_share') for e in es):  # combined rate split by the share of the bytes the counters saw on each tier
                a.plot(x, [e['combined_gbps']['median'] * (1 - e['ddr_share']['median']) for e in es], '--', color=colors[page], alpha=0.6, label=f'HBM part (counter share), {page}')
                a.plot(x, [e['combined_gbps']['median'] * e['ddr_share']['median'] for e in es], ':', color=colors[page], alpha=0.6, label=f'DDR part (counter share), {page}')
    hline(a)
    a.set_xlabel('DDR share of the pages')
    a.set_ylabel('GB/s per socket')
    a.set_title('E3 page-interleaved region')
    a.legend(fontsize=7)
    a.grid(alpha=0.3)

    a = ax[1][1]
    for gs, color in ((64, 'tab:green'), (32, 'tab:red')):
        for page, ls in (('2m', '-'), ('4k', '--')):
            es = [e for e in agg.values() if e['cfg']['kind'] == 3 and e['cfg']['gs'] == gs and e['page'] == page and e['sockets'] == 1
                  and e['name'].startswith('e5c-')]
            best = {}
            for e in es:
                r = e['cfg']['r']
                if r not in best or e['combined_gbps']['median'] > best[r]['combined_gbps']['median']:
                    best[r] = e
            if best:
                xs_ = sorted(best)
                a.plot(xs_, [best[r]['combined_gbps']['median'] * 1e9 / bytes_per_token(gs) for r in xs_], ls + 'o', color=color,
                       label=f'gs{gs}, {page} (best worker split)')
        hb = [e for e in agg.values() if e['cfg']['kind'] == 3 and e['cfg']['gs'] == gs and e['cfg']['r'] == 0 and e['sockets'] == 1
              and e['name'].startswith(('e5-', 'e5hbm-', 'e5c-'))]
        if hb:
            a.axhline(max(e['combined_gbps']['median'] for e in hb) * 1e9 / bytes_per_token(gs), color=color, lw=1, ls=':',
                      label=f'best all-HBM gs{gs}')
    a.set_xlabel("fraction r of every matrix's rows read from DDR")
    a.set_ylabel('decode tok/s-equivalent (GB/s / bytes per token)')
    a.set_title('E5 engine Q8 GEMV, 70B shard (shuffled confirmation)')
    a.legend(fontsize=7)
    a.grid(alpha=0.3)
    fig.tight_layout()
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=130)
    return True


def cmd_summarize(args):
    windows = []
    idle = {0: None, 1: None}
    zero = {str(Path(z)) for z in (args.zero_fill or [])}
    for d in args.results:
        ws = attach(d)
        if str(Path(d)) in zero:  # E3/E4 whose interleaved region was never written: kept apart, out of the headline tables
            for w in ws:
                if w['stage'] in ('e3', 'e4'):
                    w['name'] = 'zfill-' + w['name']
        windows += ws
        ip = idle_power(d)
        for s in (0, 1):
            if ip[s][0] is not None:
                idle[s] = ip[s]
    agg = aggregate(windows)
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    md = tables(agg, windows, idle) + '\n\n## Model\n\n' + model_table(agg) + '\n'
    (out / 'tables.md').write_text(md)
    ser = {f'{k[0]}|{k[1]}|{k[2]}': v for k, v in agg.items()}
    (out / 'summary.json').write_text(json.dumps(ser, indent=1, default=str))
    ok = make_figure(agg, args.figure or out / 'concurrent-tier-bandwidth.png')
    print(f'{len(windows)} windows, {len(agg)} points, figure={ok}; tables at {out / "tables.md"}')


def cmd_replicate(args):
    a = aggregate(attach(args.results))
    b = aggregate(attach(args.other))
    rows = []
    for k in sorted(set(a) & set(b)):
        x, y = a[k]['combined_gbps']['median'], b[k]['combined_gbps']['median']
        rows.append((k[0], k[1], f'{x:.1f}', f'{y:.1f}', f'{(y / x - 1) * 100:+.1f}%'))
    print(table(['config', 'page', 'this node GB/s', 'other node GB/s', 'difference'], rows))


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
    p = sub.add_parser('summarize')
    p.add_argument('results', nargs='+')
    p.add_argument('-o', '--output', required=True)
    p.add_argument('--figure')
    p.add_argument('--zero-fill', nargs='*', help='results directories whose E3/E4 regions were zero-filled')
    p.set_defaults(fn=cmd_summarize)
    p = sub.add_parser('replicate-check')
    p.add_argument('results')
    p.add_argument('other')
    p.set_defaults(fn=cmd_replicate)
    args = ap.parse_args(argv)
    args.fn(args)


if __name__ == '__main__':
    main()
