#!/usr/bin/env python3
"""Turns the raw runs of the cross-engine campaign (scripts/xengine.sbatch) into one tidy runs.csv.

    tools/xengine_summary.py OUT [OUT...] -o runs.csv [--hbm-cas-bytes 32] [--nominal-ghz 1.9]
    tools/xengine_summary.py OUT [OUT...] --calibrate

One row per run and phase (prefill, decode; a run that failed gets one row with phase=run). Each OUT is a campaign
directory with runs/<label>/{cell.json, report.json, perf.csv, markers.json, ...}; node_role comes from the manifest entry.
Stdlib only.

Perf time series (perf stat -a --per-socket -I 200 -x,): the value of a line is the count over the interval that ends at
its time stamp and starts at the previous stamp (0 for the first). A phase window [a, b] (wall clock from markers.json,
turned into perf time with perf_t0_ns) sums, for every interval that overlaps it, value x overlap / interval length:
partial intervals at the window edges are PRORATED, assuming the rate is uniform inside an interval. `window_cover` is
the covered fraction of the window (below 1 when the perf file ends or starts inside it); `min_pct_running` is the lowest
enabled/running percentage seen (below 100 means multiplexing, the numbers are then scaled estimates). Marker accuracy
is about 0.1 s (see tools/xengine_cell.py), so phases shorter than about 1 s are indicative only.

Derived columns
  *_gb_s0/_s1        GB read from DDR / HBM per socket in the window (CAS count x bytes per CAS), *_gbps_* the same / window
  gb_per_token       all DDR + HBM reads in the window / tokens of the phase (decode: decoded tokens = G - 1 steps)
  upi_*_gb_s*        UPI data flits received / transmitted at each socket x UPI_DATA_FLIT_BYTES
  pkg_w_s*, dram_w_s* average RAPL package and DRAM power per socket (joules / window)
  joules_per_token   (package + DRAM energy of both sockets) / tokens;  _net subtracts the idle baseline power of the
                     same OUT (idle-baseline.perf.csv, 10 s at job start) x window
  amx_busy_cycles    EXE.AMX_BUSY summed over sockets;  ghz_eff = cycles / ref-cycles x NOMINAL_GHZ;  ipc = insn / cycles
  cycles, ref_cycles unhalted core cycles / reference cycles summed over sockets (and over all counted cores);
                     amx_busy_share = amx_busy_cycles / cycles, the share of unhalted core cycles with the AMX unit busy
"""
import argparse
import csv
import json
from pathlib import Path
import re
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))
import xengine_cell  # noqa: E402  (marker derivation shared with the job's helper)

# ---- byte factors --------------------------------------------------------------------------------------------------
DDR_CAS_BYTES = 64  # uncore_imc event 0x05 umask 0xcf (CAS_COUNT.RD): one 64-byte line per count
# uncore_type_14 hbm_cas_rd counts 32-byte units, not 64: calibrated with `--calibrate` on c / 1S / D decode runs (check job
# 5892172): 1S-DDR reads 7.87 GB/token (64 B per ddr_cas_rd), 1S-HBM 15.67 GB/token at 64 B but 7.84 GB/token at 32 B,
# against the nominal 7.97 GB of weights. Override with --hbm-cas-bytes.
HBM_CAS_BYTES = 32
# UNC_UPI_RxL_FLITS.ALL_DATA / TxL: a 64-byte cache line travels as 9 data flits (8 payload flits + 1 header flit), so one
# data flit stands for 64/9 bytes of payload (the convention of Intel PCM for Skylake-SP and later). ASSUMPTION: recalled
# from PCM, not verified on this CPU; set --upi-flit-bytes (e.g. 8 for raw 8-byte payload per flit) to change it.
UPI_DATA_FLIT_BYTES = 64 / 9
NOMINAL_GHZ = 1.9  # Xeon Max 9480 base frequency = TSC = ref-cycles rate
NOMINAL_GB_PER_TOKEN = 7.97  # INT8 weights of the 8B model read once per decoded token (7974764544 B per socket set)

EVENT_NAMES = {
    'ddr_cas_rd': 'ddr', 'hbm_cas_rd': 'hbm', 'ddr_cas_wr': 'ddr_wr', 'hbm_cas_wr': 'hbm_wr',
    'unc_upi_rxl_flits.all_data': 'upi_rx', 'unc_upi_txl_flits.all_data': 'upi_tx',
    'power/energy-pkg/': 'pkg_j', 'power/energy-ram/': 'ram_j', 'cycles': 'cycles', 'ref-cycles': 'ref_cycles',
    'context-switches': 'context_switches', 'cpu-migrations': 'cpu_migrations',
    'instructions': 'instructions', 'exe.amx_busy': 'amx_busy', 'amx_ops_retired.int8': 'amx_ops',
}
SOCKETS = (0, 1)


# ---- perf parsing --------------------------------------------------------------------------------------------------
def _number(text):
    try:
        return float(text)
    except ValueError:
        return None


def parse_perf(path):
    """{(socket, event): [(t0, t1, value|None, pct)]}. Per-socket and plain, with and without -I, are all accepted;
    plain (system- or process-wide) counts go to socket 0."""
    series = {}
    last_t = {}
    text = Path(path).read_text() if path and Path(path).exists() else ''
    for line in text.splitlines():
        if not line.strip() or line.startswith('#'):
            continue
        f = line.split(',')
        t = None
        socket = 0
        i = 0
        if re.fullmatch(r'S\d+', f[0]):  # S0,ncpu,value,...
            socket, i = int(f[0][1:]), 2
        elif len(f) > 1 and re.fullmatch(r'S\d+', f[1]):  # time,S0,ncpu,value,...
            t, socket, i = _number(f[0]), int(f[1][1:]), 3
        elif len(f) > 1 and _number(f[0]) is not None and (_number(f[1]) is not None or f[1].startswith('<')):
            t, i = _number(f[0]), 1  # time,value,...
        try:
            value, event, run_ns, pct = f[i], f[i + 2], f[i + 3], f[i + 4]
        except IndexError:
            continue
        event = re.sub(r':[A-Za-z]+$', '', event.strip()).lower()
        name = EVENT_NAMES.get(event)
        if name is None:
            continue
        v = _number(value)
        p = _number(pct)
        key = (socket, name)
        if t is None:  # one count for the whole run: a single interval [0, run time]
            rt = (_number(run_ns) or 0) / 1e9
            series.setdefault(key, []).append((0.0, rt, v, p))
            continue
        t0 = last_t.get(key, 0.0)
        last_t[key] = t
        series.setdefault(key, []).append((t0, t, v, p))
    return series


def window_sum(series, key, a, b):
    """(prorated sum of the counts in [a, b], covered seconds, min pct running) for one series."""
    total = cover = 0.0
    pct = None
    for t0, t1, v, p in series.get(key, []):
        lo, hi = max(a, t0), min(b, t1)
        if hi <= lo or t1 <= t0:
            continue
        frac = (hi - lo) / (t1 - t0)
        cover += hi - lo
        if p is not None:
            pct = p if pct is None else min(pct, p)
        if v is not None:
            total += v * frac
    return total, cover, pct


def idle_power(out):
    """Idle watts per socket {'pkg': {s: W}, 'ram': {s: W}} from OUT/idle-baseline.perf.csv, or None."""
    path = Path(out)/'idle-baseline.perf.csv'
    if not path.exists():
        return None
    series = parse_perf(path)
    res = {'pkg': {}, 'ram': {}}
    for kind, name in (('pkg', 'pkg_j'), ('ram', 'ram_j')):
        for s in SOCKETS:
            rows = series.get((s, name))
            if rows:
                dur = rows[-1][1]
                total = sum(v for _, _, v, _ in rows if v is not None)
                if dur > 0:
                    res[kind][s] = total / dur
    return res if res['pkg'] else None


# ---- per-engine readers --------------------------------------------------------------------------------------------
def read_c_phases(report):
    """Native numbers of a llama3.x.c report: {'prefill': {...}, 'decode': {...}} (decode absent when G = 1)."""
    t = report['timing_seconds']
    thr = report['throughput_tokens_per_second']
    tok = report['tokens']
    out = {'prefill': {'tokens': report['prefill']['tokens'], 'seconds': report['prefill']['seconds'],
                       'tok_s': thr.get('prefill'), 'ttft_s': t.get('time_to_first_token')}}
    if tok.get('decode_steps'):
        out['decode'] = {'tokens': tok['decode_steps'], 'seconds': t.get('decode'), 'tok_s': thr.get('decode'),
                         'tpot_s': t.get('average_decode_step'), 'e2e_s': t.get('total_inference'),
                         'tpot_uniform_s': (t['total_inference'] - t['time_to_first_token']) / tok['decode_steps']}
    return out


def read_vllm_phases(report):
    """Native numbers of one vLLM request from the client's token arrival times (seconds since the request was sent):
    TTFT = first arrival, prefill tok/s = prompt tokens / TTFT, decode = the G - 1 tokens after the first (tok/s and TPOT
    from the arrival times). tpot_uniform_s = (end-to-end - TTFT) / (G - 1) is the within-run form of the uniform definition."""
    arr = report['token_arrivals']
    ttft = report['t_first_token']
    out = {'prefill': {'tokens': report['usage_prompt_tokens'], 'seconds': ttft, 'tok_s': report['usage_prompt_tokens'] / ttft,
                       'ttft_s': ttft}}
    if len(arr) > 1:
        dec = arr[-1] - arr[0]
        out['decode'] = {'tokens': len(arr) - 1, 'seconds': dec, 'tok_s': (len(arr) - 1) / dec if dec > 0 else None,
                         'tpot_s': dec / (len(arr) - 1), 'e2e_s': report['t_end'],
                         'tpot_uniform_s': (report['t_end'] - ttft) / (len(arr) - 1)}
    return out


def read_llamacpp_phases(report):
    """One llama-bench entry (average of its -r repetitions): pp (n_prompt > 0, n_gen = 0) is the prefill phase, TTFT =
    n_prompt / avg_ts; tg (n_gen > 0, after n_depth untimed context tokens) is the decode phase, TPOT = 1 / avg_ts. The tg
    timing covers all n_gen tokens (llama-bench has no separate first token), so decode tokens = n_gen here."""
    avg_s = report['avg_ns'] / 1e9
    if report.get('n_gen', 0) > 0:
        n = report['n_gen']
        return {'decode': {'tokens': n, 'seconds': avg_s, 'tok_s': report['avg_ts'], 'tpot_s': 1 / report['avg_ts'],
                           'e2e_s': avg_s, 'tpot_uniform_s': avg_s / n}}
    n = report['n_prompt']
    return {'prefill': {'tokens': n, 'seconds': avg_s, 'tok_s': report['avg_ts'], 'ttft_s': n / report['avg_ts']}}


PHASE_READERS = {'c': read_c_phases, 'vllm': read_vllm_phases, 'llamacpp': read_llamacpp_phases}  # llama.cpp and vLLM readers register here


def phase_windows(markers):
    """{'prefill': (a_ns, b_ns), 'decode': (a_ns, b_ns)} in wall-clock ns, for the phases the markers can place."""
    w = {}
    if markers.get('prefill_start_ns') and markers.get('first_token_ns'):
        w['prefill'] = (markers['prefill_start_ns'], markers['first_token_ns'])
    if markers.get('first_token_ns') and markers.get('decode_end_ns'):
        w['decode'] = (markers['first_token_ns'], markers['decode_end_ns'])
    return w


# ---- one phase of one run ------------------------------------------------------------------------------------------
COLUMNS = (['node_role', 'engine', 'variant', 'sockets', 'mem', 'workload', 'repeat', 'perf_group', 'phase', 'status', 'note',
            'tokens', 'seconds', 'tok_s', 'ttft_s', 'tpot_s', 'e2e_s', 'tpot_uniform_s', 'window_s', 'window_cover', 'min_pct_running',
            'perf_clock_skew_s'] +
           [f'{k}_gb_s{s}' for k in ('ddr', 'hbm') for s in SOCKETS] +
           [f'{k}_gbps_s{s}' for k in ('ddr', 'hbm') for s in SOCKETS] +
           ['gb_read', 'gbps_read', 'ddr_gb_per_token', 'hbm_gb_per_token', 'gb_per_token'] +
           [f'upi_{d}_flits_s{s}' for d in ('rx', 'tx') for s in SOCKETS] +
           [f'upi_{d}_gb_s{s}' for d in ('rx', 'tx') for s in SOCKETS] + ['upi_gbps'] +
           [f'pkg_w_s{s}' for s in SOCKETS] + [f'dram_w_s{s}' for s in SOCKETS] +
           ['joules', 'joules_per_token', 'joules_per_token_net', 'amx_busy_cycles', 'amx_ops_int8', 'cycles', 'ref_cycles',
            'amx_busy_share', 'ghz_eff', 'ipc'] +
           [f'ghz_eff_s{s}' for s in SOCKETS] + [f'ipc_s{s}' for s in SOCKETS])


def perf_metrics(series, a, b, tokens, idle, hbm_bytes, upi_bytes, nominal_ghz):
    """Counter-derived columns of one window [a, b] seconds in perf time."""
    dur = b - a
    row = {'window_s': dur}
    if dur <= 0 or not series:
        return row
    cover = []
    pcts = []

    def get(socket, name):
        total, cov, pct = window_sum(series, (socket, name), a, b)
        if cov > 0:
            cover.append(cov / dur)
        if pct is not None:
            pcts.append(pct)
        return total if (socket, name) in series else None

    gb = {}
    for s in SOCKETS:
        for name, factor in (('ddr', DDR_CAS_BYTES), ('hbm', hbm_bytes)):
            n = get(s, name)
            if n is not None:
                gb[(name, s)] = n * factor / 1e9
                row[f'{name}_gb_s{s}'] = gb[(name, s)]
                row[f'{name}_gbps_s{s}'] = gb[(name, s)] / dur
    ddr = sum(v for (k, _), v in gb.items() if k == 'ddr')
    hbm = sum(v for (k, _), v in gb.items() if k == 'hbm')
    if gb:
        row['gb_read'] = ddr + hbm
        row['gbps_read'] = (ddr + hbm) / dur
        if tokens:
            row['ddr_gb_per_token'] = ddr / tokens
            row['hbm_gb_per_token'] = hbm / tokens
            row['gb_per_token'] = (ddr + hbm) / tokens
    upi_total = 0.0
    for d in ('rx', 'tx'):
        for s in SOCKETS:
            n = get(s, f'upi_{d}')
            if n is not None:
                row[f'upi_{d}_flits_s{s}'] = n
                row[f'upi_{d}_gb_s{s}'] = n * upi_bytes / 1e9
                if d == 'rx':
                    upi_total += n * upi_bytes / 1e9
    if any(f'upi_rx_flits_s{s}' in row for s in SOCKETS):
        row['upi_gbps'] = upi_total / dur
    joules = 0.0
    have_energy = False
    for s in SOCKETS:
        for name, col in (('pkg_j', 'pkg_w'), ('ram_j', 'dram_w')):
            j = get(s, name)
            if j is not None:
                row[f'{col}_s{s}'] = j / dur
                joules += j
                have_energy = True
    if have_energy:
        row['joules'] = joules
        if tokens:
            row['joules_per_token'] = joules / tokens
            if idle:
                idle_w = sum(idle['pkg'].values()) + sum(idle['ram'].values())
                row['joules_per_token_net'] = (joules - idle_w * dur) / tokens
    cyc = {s: get(s, 'cycles') for s in SOCKETS}
    ref = {s: get(s, 'ref_cycles') for s in SOCKETS}
    ins = {s: get(s, 'instructions') for s in SOCKETS}
    amx = [get(s, 'amx_busy') for s in SOCKETS]
    ops = [get(s, 'amx_ops') for s in SOCKETS]
    if any(v is not None for v in amx):
        row['amx_busy_cycles'] = sum(v for v in amx if v is not None)
    if any(v is not None for v in ops):
        row['amx_ops_int8'] = sum(v for v in ops if v is not None)
    for s in SOCKETS:
        if cyc[s] and ref[s]:
            row[f'ghz_eff_s{s}'] = cyc[s] / ref[s] * nominal_ghz
        if cyc[s] and ins[s]:
            row[f'ipc_s{s}'] = ins[s] / cyc[s]
    cs, rs, ns = [sum(v for v in d.values() if v) for d in (cyc, ref, ins)]
    if cs:
        row['cycles'] = cs
        if row.get('amx_busy_cycles') is not None:
            row['amx_busy_share'] = row['amx_busy_cycles'] / cs
    if rs:
        row['ref_cycles'] = rs
    if cs and rs:
        row['ghz_eff'] = cs / rs * nominal_ghz
    if cs and ns:
        row['ipc'] = ns / cs
    if cover:
        row['window_cover'] = min(cover)
    if pcts:
        row['min_pct_running'] = min(pcts)
    return row


def summarize_run(out, rundir, hbm_bytes=HBM_CAS_BYTES, upi_bytes=UPI_DATA_FLIT_BYTES, nominal_ghz=NOMINAL_GHZ, idle=None):
    """Rows (dicts of COLUMNS) of one run directory."""
    cell = json.loads((rundir/'cell.json').read_text())
    base = {k: cell.get(k, '') for k in ('engine', 'variant', 'sockets', 'mem', 'workload', 'repeat', 'perf_group')}
    base['node_role'] = cell.get('node_role', '')
    status = cell.get('status', 'unknown')
    note = cell.get('status_detail', '')
    report_path = rundir/'report.json'
    report = json.loads(report_path.read_text()) if report_path.exists() else None
    reader = PHASE_READERS.get(base['engine'])
    phases = {}
    if report and reader and report.get('status') == 'completed':
        try:
            phases = reader(report)
        except (KeyError, TypeError) as e:
            status, note = 'failed', f'report unreadable: {e!r}'
    if not phases:
        return [dict(base, phase='run', status=status if status != 'ok' else 'failed', note=note or 'no phase data')]
    markers_path = rundir/'markers.json'
    markers = json.loads(markers_path.read_text()) if markers_path.exists() else None
    if markers is not None and (rundir/'anchors').exists():  # always line the perf clock up with the current rule
        markers.update(xengine_cell.perf_clock(xengine_cell.read_kv(rundir/'anchors'), markers, rundir/'perf.csv'))
    if markers is None and (rundir/'anchors').exists():  # job died before its post-processing
        anchors = xengine_cell.read_kv(rundir/'anchors')
        if 'launch_ns' in anchors and 'end_ns' in anchors and base['engine'] in xengine_cell.MARKER_DERIVERS:
            markers = xengine_cell.MARKER_DERIVERS[base['engine']](anchors, report)
            markers.update(xengine_cell.perf_clock(anchors, markers, rundir/'perf.csv'))
    windows = phase_windows(markers) if markers else {}
    series = parse_perf(rundir/'perf.csv') if (rundir/'perf.csv').exists() else {}
    rows = []
    for phase, native in phases.items():
        row = dict(base, phase=phase, status=status, note=note)
        row.update({k: v for k, v in native.items()})
        if series and phase in windows and markers.get('perf_t0_ns'):
            t0 = markers['perf_t0_ns']
            a, b = (windows[phase][0] - t0) / 1e9, (windows[phase][1] - t0) / 1e9
            row.update(perf_metrics(series, a, b, native.get('tokens'), idle, hbm_bytes, upi_bytes, nominal_ghz))
            if markers.get('perf_t0_launch_ns'):
                row['perf_clock_skew_s'] = (markers['perf_t0_launch_ns'] - t0) / 1e9
        rows.append(row)
    return rows


def summarize(outs, **kw):
    rows = []
    for out in outs:
        out = Path(out)
        idle = idle_power(out)
        for rundir in sorted((out/'runs').glob('*/')):
            if (rundir/'cell.json').exists():
                rows.extend(summarize_run(out, rundir, idle=idle, **kw))
    return rows


def write_csv(rows, path):
    def fmt(v):
        return '' if v is None else ('%.6g' % v if isinstance(v, float) else v)
    with open(path, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=COLUMNS, extrasaction='ignore')
        w.writeheader()
        for r in rows:
            w.writerow({k: fmt(r.get(k)) for k in COLUMNS})


# ---- calibration ---------------------------------------------------------------------------------------------------
def calibrate(rows, hbm_bytes):
    """Prints bytes per decoded token of the c / 1S / D decode rows next to the nominal weight bytes, with the HBM
    reads at 64 and at 32 bytes per hbm_cas_rd count (rows were computed with hbm_bytes)."""
    lines = ['bytes read per decoded token, c engine, 1 socket, workload D (nominal %.2f GB/token)' % NOMINAL_GB_PER_TOKEN,
             '%-6s %-4s %-5s %11s %11s %11s %11s %11s' % ('mem', 'rep', 'group', 'ddr GB/tok', 'hbm@64', 'hbm@32', 'x nom @64', 'x nom @32')]
    ratios = {64: [], 32: []}
    for r in rows:
        if not (r.get('engine') == 'c' and str(r.get('sockets')) == '1' and r.get('workload') == 'D' and
                r.get('phase') == 'decode' and r.get('mem') in ('hbm', 'ddr') and r.get('gb_per_token') is not None):
            continue
        ddr = r.get('ddr_gb_per_token') or 0.0
        h64 = (r.get('hbm_gb_per_token') or 0.0) * 64 / hbm_bytes
        h32 = h64 / 2
        lines.append('%-6s %-4s %-5s %11.3f %11.3f %11.3f %11.3f %11.3f' % (
            r['mem'], r['repeat'], r['perf_group'], ddr, h64, h32, (ddr + h64) / NOMINAL_GB_PER_TOKEN,
            (ddr + h32) / NOMINAL_GB_PER_TOKEN))
        if r['mem'] == 'hbm':
            ratios[64].append((ddr + h64) / NOMINAL_GB_PER_TOKEN)
            ratios[32].append((ddr + h32) / NOMINAL_GB_PER_TOKEN)
    if len(lines) == 2:
        lines.append('no c / 1S / D decode rows with a MEM-group perf series (repeat 1) found')
    elif ratios[64]:
        mean = lambda v: sum(v) / len(v)
        a, b = mean(ratios[64]), mean(ratios[32])
        lines.append('HBM runs: x nominal %.3f with 64 B per hbm CAS, %.3f with 32 B (DDR runs fix the DDR factor at 64 B).' % (a, b))
        lines.append('Pick the HBM factor whose ratio is closest to 1.0 (a few percent above: KV cache, activations, '
                     'embedding row, other traffic): %s' % ('32' if abs(b - 1) < abs(a - 1) else '64'))
    print('\n'.join(lines))


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('outs', nargs='+', metavar='OUT', help='campaign output directories')
    ap.add_argument('-o', '--output', help='runs.csv to write')
    ap.add_argument('--hbm-cas-bytes', type=int, default=HBM_CAS_BYTES, help='bytes per hbm_cas_rd count (32, calibrated)')
    ap.add_argument('--upi-flit-bytes', type=float, default=UPI_DATA_FLIT_BYTES, help='payload bytes per UPI data flit (64/9, assumption)')
    ap.add_argument('--nominal-ghz', type=float, default=NOMINAL_GHZ)
    ap.add_argument('--calibrate', action='store_true', help='print GB per decoded token of c/1S/D runs vs the nominal weight bytes')
    args = ap.parse_args(argv)
    rows = summarize(args.outs, hbm_bytes=args.hbm_cas_bytes, upi_bytes=args.upi_flit_bytes, nominal_ghz=args.nominal_ghz)
    if args.output:
        write_csv(rows, args.output)
        print('%d rows -> %s' % (len(rows), args.output), file=sys.stderr)
    if args.calibrate:
        calibrate(rows, args.hbm_cas_bytes)
    elif not args.output:
        ap.error('give -o runs.csv or --calibrate')
    return 0


if __name__ == '__main__':
    sys.exit(main())
