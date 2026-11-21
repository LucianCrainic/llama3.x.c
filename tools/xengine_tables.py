#!/usr/bin/env python3
"""Markdown tables of the cross-engine results (docs/xengine-results.md) from the runs.csv of tools/xengine_summary.py.

    tools/xengine_tables.py runs.csv [--only T1a,T2,...]

Every table is a pure function of runs.csv (stdlib only). Cells are the median over the repeats that exist for the cell
with (min-max) behind it; counter tables use the single repeat that carries the counter group (repeat 1: MEM on the Flat
nodes, USER on the Cache node; repeat 2: COMPUTE), and power uses repeats 1 and 2. Nothing here is edited by hand.
"""
import argparse
import csv
import statistics
import sys

ENG = {'c': 'llama3.x.c', 'llamacpp': 'llama.cpp', 'vllm': 'vLLM'}
TIER = {'A': 'HBM', 'B': 'DDR', 'cache': 'Cache'}
JOB = {'A': '5892201', 'B': '5892202', 'cache': '5892196'}
ROOF = {'hbm': 552.0, 'ddr': 262.0}  # GB/s per socket, docs/hbm-roofline.md (local reads)
NOMINAL_GB = 7.97  # INT8 Q8_0 weights read per decoded token (7974764544 B)
NOMINAL_VLLM_GB = 8.03  # 6.98 GB int8 layers + 1.05 GB bf16 lm_head (derived from the architecture)
KV_ELEMS = 32 * 2 * 1024  # layers x (K,V) x kv dim: elements of KV per context position
KV_BYTES = {'c': 4, 'llamacpp': 2, 'vllm': 2}  # FP32 / F16 / BF16
DEPTH = 8192 + 64  # mean context depth of the 128 decoded tokens of L


def load(path):
    rows = []
    with open(path) as f:
        for r in csv.DictReader(f):
            for k, v in list(r.items()):
                if k not in ('node_role', 'engine', 'variant', 'mem', 'workload', 'phase', 'status', 'note', 'perf_group'):
                    try:
                        r[k] = float(v) if v != '' else None
                    except ValueError:
                        pass
            r['sockets'] = int(r['sockets'])
            r['repeat'] = int(r['repeat'])
            rows.append(r)
    return rows


def config_label(c):
    node, eng, var, s, mem = c
    name = ENG[eng] + ({'': '', 'c2': '', 'patched': ' patched', 'stock-membind': ' stock + membind'}[var])
    return '%s %dS' % (name, s)


def configs(rows, node=None, anchor=False):
    """Ordered distinct (node, engine, variant, sockets, mem); the anchor cell of node B (c, 1S, hbm) only on request."""
    seen = {}
    for r in rows:
        c = (r['node_role'], r['engine'], r['variant'], r['sockets'], r['mem'])
        if c[0] == 'B' and c[4] == 'hbm' and not anchor:
            continue
        seen[c] = 1
    order = {'c': 0, 'llamacpp': 1, 'vllm': 2}
    tier = {'A': 0, 'B': 1, 'cache': 2}
    out = sorted(seen, key=lambda c: (c[3], order[c[1]], tier[c[0]], c[2] == 'patched'))
    return [c for c in out if node is None or c[0] == node]


def sel(rows, c, workload, phase, **kw):
    node, eng, var, s, mem = c
    out = [r for r in rows if (r['node_role'], r['engine'], r['variant'], r['sockets'], r['mem']) == c and
           r['workload'] == workload and r['phase'] == phase and r['status'] == 'ok']
    for k, v in kw.items():
        out = [r for r in out if r.get(k) == v]
    return out


def vals(rows, col, scale=1.0):
    return [r[col]*scale for r in rows if r.get(col) is not None]


def med(v):
    return statistics.median(v) if v else None


def cell(v, fmt='%.1f', rng=True):
    if not v:
        return '-'
    m = statistics.median(v)
    if rng and len(v) > 1:
        return (fmt + ' (' + fmt + '-' + fmt + ')') % (m, min(v), max(v))
    return fmt % m


def table(head, body):
    lines = ['| ' + ' | '.join(head) + ' |', '|' + '|'.join(' --- ' if i == 0 else ' ---: ' for i in range(len(head))) + '|']
    lines += ['| ' + ' | '.join(str(x) for x in row) + ' |' for row in body]
    return '\n'.join(lines)


def prefill8192(rows, c):
    """The 8192-token prefill: the prefill phase of L for llama3.x.c and vLLM, of LP for llama.cpp."""
    return sel(rows, c, 'LP' if c[1] == 'llamacpp' else 'L', 'prefill')


# ---- tables --------------------------------------------------------------------------------------------------------------
def t0(rows):
    """Headline: decode tok/s (D) and TTFT of the 2048-token prompt (s), median of 3, per engine x sockets x tier."""
    body = []
    for eng in ('c', 'llamacpp', 'vllm'):
        for s in (1, 2):
            row = [ENG[eng] + (' (HBM: patched)' if eng == 'vllm' else '') , s]
            for node in ('A', 'B', 'cache'):
                cs = [c for c in configs(rows, node) if c[1] == eng and c[3] == s and c[2] != 'stock-membind']
                d = med(vals(sel(rows, cs[0], 'D', 'decode'), 'tok_s'))
                t = med(vals(sel(rows, cs[0], 'P', 'prefill'), 'ttft_s'))
                row.append('%.1f / %.3f' % (d, t))
            body.append(row)
    return table(['Engine', 'Sockets', 'Flat HBM (node A)', 'Flat DDR (node B)', 'Cache (Cache node)'], body)


def t1a(rows):
    body = []
    for c in configs(rows):
        p, q = sel(rows, c, 'P', 'prefill'), prefill8192(rows, c)
        if not p:
            continue
        body.append([config_label(c), TIER[c[0]] + ' (job %s)' % JOB[c[0]], cell(vals(p, 'ttft_s'), '%.3f'), cell(vals(p, 'tok_s'), '%.0f'),
                     cell(vals(q, 'ttft_s'), '%.2f'), cell(vals(q, 'tok_s'), '%.0f')])
    return table(['Configuration', 'Tier', 'TTFT 2048 s', 'prefill tok/s (2048)', 'TTFT 8192 s', 'prefill tok/s (8192)'], body)


def t1b(rows):
    body = []
    for c in configs(rows):
        d, l = sel(rows, c, 'D', 'decode'), sel(rows, c, 'L', 'decode')
        if not l:
            body.append([config_label(c) + ' (D only)', TIER[c[0]], cell(vals(d, 'tok_s'), '%.1f'), cell(vals(d, 'tpot_s', 1e3), '%.1f'),
                         cell(vals(d, 'tpot_uniform_s', 1e3), '%.1f'), '-', '-', '-'])
            continue
        body.append([config_label(c), TIER[c[0]], cell(vals(d, 'tok_s'), '%.1f'), cell(vals(d, 'tpot_s', 1e3), '%.1f'),
                     cell(vals(d, 'tpot_uniform_s', 1e3), '%.1f'), cell(vals(l, 'tok_s'), '%.1f'), cell(vals(l, 'tpot_s', 1e3), '%.1f'),
                     '%.2f' % (med(vals(l, 'tok_s'))/med(vals(d, 'tok_s')))])
    return table(['Configuration', 'Tier', 'decode tok/s (D)', 'TPOT ms (native)', 'TPOT ms (uniform)', 'long-context tok/s (L)',
                  'TPOT ms (L)', 'L / D'], body)


def t2(rows):
    body = []
    for c in configs(rows, node=None):
        if c[0] == 'cache':
            continue
        n_sock = c[3]
        roof = ROOF[c[4]]*n_sock
        d = sel(rows, c, 'D', 'decode', repeat=1)
        l = sel(rows, c, 'L', 'decode', repeat=1)
        if not d:
            continue
        r = d[0]
        gbps = [r.get('%s_gbps_s%d' % (k, s)) or 0.0 for k in ('ddr', 'hbm') for s in (0, 1)]
        tot = r['gbps_read']
        row = [config_label(c), TIER[c[0]], '%.0f / %.0f' % (gbps[0] + gbps[2], gbps[1] + gbps[3]), '%.0f' % tot,
               '%.0f%%' % (100*tot/roof), '%.2f' % r['gb_per_token'], '%.2f' % (r['gb_per_token']/(NOMINAL_VLLM_GB if c[1] == 'vllm' else NOMINAL_GB))]
        if l:
            lr = l[0]
            row += ['%.0f' % lr['gbps_read'], '%.0f%%' % (100*lr['gbps_read']/roof), '%.2f' % lr['gb_per_token'],
                    '%.2f' % (lr['gb_per_token'] - r['gb_per_token']), '%.2f' % (DEPTH*KV_ELEMS*KV_BYTES[c[1]]/1e9)]
        else:
            row += ['-'] * 5
        body.append(row)
    return table(['Configuration', 'Tier', 'GB/s S0 / S1 (D)', 'GB/s total (D)', '% of roofline', 'GB / token (D)', 'x nominal',
                  'GB/s total (L)', '% of roofline (L)', 'GB / token (L)', 'L - D GB / token', 'KV read at depth 8192, derived GB'], body)


def t3(rows):
    body = []
    for c in configs(rows):
        if c[0] == 'cache':
            continue
        for wl, ph in (('D', 'decode'), ('P', 'prefill')):
            s = sel(rows, c, wl, ph, repeat=1)
            if not s:
                continue
            r = s[0]
            ddr = sum(r.get('ddr_gbps_s%d' % i) or 0 for i in (0, 1))
            hbm = sum(r.get('hbm_gbps_s%d' % i) or 0 for i in (0, 1))
            body.append([config_label(c), TIER[c[0]], 'decode (D)' if wl == 'D' else 'prefill (P)', '%.1f' % ddr, '%.1f' % hbm,
                         '%.1f%%' % (100*hbm/(ddr + hbm))])
    c = ('A', 'vllm', 'stock-membind', 1, 'hbm')
    return table(['Configuration', 'Asked for', 'Phase', 'DDR GB/s (sum of sockets)', 'HBM GB/s (sum of sockets)', 'HBM share of reads'], body)


def t4(rows):
    body = []
    for node in ('A', 'B', 'cache'):
        for eng, var2 in (('c', ''), ('llamacpp', ''), ('vllm', '')):
            c1 = [c for c in configs(rows, node) if c[1] == eng and c[3] == 1 and c[2] != 'stock-membind']
            c2 = [c for c in configs(rows, node) if c[1] == eng and c[3] == 2]
            if not (c1 and c2):
                continue
            c1, c2 = c1[0], c2[0]
            row = [ENG[eng] + (' patched' if c2[2] == 'patched' else ''), TIER[node]]
            for wl, ph, key in (('D', 'decode', 'tok_s'), ('L', 'decode', 'tok_s'), ('P', 'prefill', 'tok_s')):
                a, b = med(vals(sel(rows, c1, wl, ph), key)), med(vals(sel(rows, c2, wl, ph), key))
                row.append('%.2f' % (b/a))
            q1, q2 = prefill8192(rows, c1), prefill8192(rows, c2)
            row.append('%.2f' % (med(vals(q2, 'tok_s'))/med(vals(q1, 'tok_s'))))
            if node == 'cache':
                row += ['-', '-']
            else:
                u = []
                for c in (c1, c2):
                    d = sel(rows, c, 'D', 'decode', repeat=1)[0]
                    u.append((d['upi_rx_gb_s0'] + d['upi_rx_gb_s1'])/d['tokens'])
                row += ['%.3f' % u[0], '%.3f' % u[1]]
            body.append(row)
    return table(['Engine', 'Tier', '2S/1S decode D', '2S/1S long-ctx L', '2S/1S prefill 2048', '2S/1S prefill 8192', 'UPI GB/token 1S (D)',
                  'UPI GB/token 2S (D)'], body)


def t5(rows):
    body = []
    for c in configs(rows):
        grp = 'USER' if c[0] == 'cache' else 'COMPUTE'
        rep = 1 if c[0] == 'cache' else 2
        row = [config_label(c), TIER[c[0]]]
        for wl, ph in (('P', 'prefill'), ('D', 'decode')):
            if c[1] == 'llamacpp' and wl == 'L':
                continue
            s = sel(rows, c, wl, ph, repeat=rep)
            if not s:
                row += ['-', '-']
                continue
            r = s[0]
            row += ['%.2e' % r['amx_busy_cycles'] if r.get('amx_busy_cycles') is not None else '-',
                    '%.1f%%' % (100*r['amx_busy_share']) if r.get('amx_busy_share') is not None else '-']
        q = prefill8192(rows, c)
        q = [r for r in q if r['repeat'] == rep]
        row += ['%.2e' % q[0]['amx_busy_cycles'] if q and q[0].get('amx_busy_cycles') is not None else '-',
                '%.1f%%' % (100*q[0]['amx_busy_share']) if q and q[0].get('amx_busy_share') is not None else '-']
        p = sel(rows, c, 'P', 'prefill', repeat=rep)
        row.append('%.2f' % p[0]['ipc'] if p and p[0].get('ipc') is not None else '-')
        row.append('%.2f' % p[0]['ghz_eff'] if p and p[0].get('ghz_eff') is not None else '-')
        body.append(row)
    return table(['Configuration', 'Tier', 'AMX_BUSY cycles prefill 2048', 'share of cycles', 'AMX_BUSY cycles decode D', 'share',
                  'AMX_BUSY cycles prefill 8192', 'share', 'IPC (P)', 'GHz (P)'], body)


def t5b(rows):
    """Prefill cost per token (2048-token prompt, repeat 2): unhalted core cycles, instructions and AMX-busy cycles per token.
    Ideal AMX cycles/token = 6.979e9 int8 MACs of the 32 layers / 1024 MACs per AMX-busy cycle (derived)."""
    ideal = 6.979e9/1024/1e6
    body = []
    for c in configs(rows):
        if c[0] == 'cache':
            continue
        s = sel(rows, c, 'P', 'prefill', repeat=2)
        if not s:
            continue
        r = s[0]
        n = r['tokens']
        body.append([config_label(c), TIER[c[0]], '%.1f' % (r['cycles']/n/1e6), '%.1f' % (r['ipc']*r['cycles']/n/1e6),
                     '%.2f' % (r['amx_busy_cycles']/n/1e6), '%.2f' % (r['amx_busy_cycles']/n/1e6/ideal), '%.2f' % (r['ttft_s']*1e3/n)])
    return table(['Configuration', 'Tier', 'core cycles / token (M)', 'instructions / token (M)', 'AMX-busy cycles / token (M)',
                  'x ideal (%.2f M)' % ideal, 'TTFT ms / token'], body)


def idle(path):
    """(pkg W, ram W) per socket of an idle-baseline.perf.csv: [(s0 pkg, s0 ram), (s1 pkg, s1 ram)]."""
    sys.path.insert(0, __file__.rsplit('/', 1)[0])
    import xengine_summary as xs
    r = xs.idle_power(path)
    return r


def t6(rows):
    body = []
    for c in configs(rows):
        if c[0] == 'cache':
            continue
        row = [config_label(c), TIER[c[0]]]
        for wl, ph in (('P', 'prefill'), ('D', 'decode')):
            s = [r for r in sel(rows, c, wl, ph) if r['repeat'] in (1, 2) and r.get('pkg_w_s0') is not None]
            if not s:
                row += ['-'] * 4
                continue
            pkg = med([r['pkg_w_s0'] + r['pkg_w_s1'] for r in s])
            ram = med([r['dram_w_s0'] + r['dram_w_s1'] for r in s])
            jt = med(vals(s, 'joules_per_token'))
            jn = med(vals(s, 'joules_per_token_net'))
            row += ['%.0f' % pkg, '%.0f' % ram, '%.2f' % jt, '%.2f' % jn]
        body.append(row)
    return table(['Configuration', 'Tier', 'P: package W (2 sockets)', 'P: DRAM W', 'P: J/token', 'P: net J/token', 'D: package W',
                  'D: DRAM W', 'D: J/token', 'D: net J/token'], body)


def t7(rows):
    cfg_a = [c for c in configs(rows, 'A')]
    body = []
    anchor = {}
    for wl, ph in (('D', 'decode'), ('D', 'prefill')):
        a = med(vals(sel(rows, ('A', 'c', '', 1, 'hbm'), wl, ph), 'tok_s'))
        b = med(vals(sel(rows, ('B', 'c', '', 1, 'hbm'), wl, ph), 'tok_s'))
        anchor[ph] = a/b
    for ca in cfg_a:
        if ca[2] == 'stock-membind':
            continue
        cb = [c for c in configs(rows, 'B') if c[1] == ca[1] and c[3] == ca[3]][0]
        row = [config_label(ca) + (' (HBM patched vs DDR stock)' if ca[1] == 'vllm' else '')]
        for (wl, ph, key), f in ((('D', 'decode', 'tok_s'), anchor['decode']), (('L', 'decode', 'tok_s'), anchor['decode']),
                                 (('P', 'prefill', 'tok_s'), anchor['prefill'])):
            a, b = med(vals(sel(rows, ca, wl, ph), key)), med(vals(sel(rows, cb, wl, ph), key))
            row.append('%.2f (%.2f)' % (a/b, a/b/f))
        a, b = med(vals(prefill8192(rows, ca), 'tok_s')), med(vals(prefill8192(rows, cb), 'tok_s'))
        row.append('%.2f (%.2f)' % (a/b, a/b/anchor['prefill']))
        body.append(row)
    out = table(['Engine and sockets', 'decode D', 'long-context L', 'prefill 2048', 'prefill 8192'], body)
    out += '\n\nAnchor (llama3.x.c, 1S, HBM, D) node A / node B: decode %.3f, prefill (86 tokens) %.3f. Cells: raw ratio (anchor-corrected).' % (
        anchor['decode'], anchor['prefill'])
    return out


def t8(rows):
    body = []
    for eng in ('c', 'llamacpp', 'vllm'):
        for s in (1, 2):
            cs = {}
            for c in configs(rows):
                if c[1] == eng and c[3] == s and c[2] != 'stock-membind':
                    cs[c[0]] = c
            if len(cs) < 3:
                continue
            for wl, ph, name in (('D', 'decode', 'decode D tok/s'), ('L', 'decode', 'long-context L tok/s'),
                                 ('P', 'prefill', 'prefill 2048 tok/s'), ('8192', 'prefill', 'prefill 8192 tok/s')):
                v = {}
                for n in ('A', 'B', 'cache'):
                    sr = prefill8192(rows, cs[n]) if wl == '8192' else sel(rows, cs[n], wl, ph)
                    v[n] = med(vals(sr, 'tok_s'))
                body.append(['%s %dS' % (ENG[eng], s), name, '%.1f' % v['B'], '%.1f' % v['cache'], '%.1f' % v['A'],
                             '%.2f' % (v['cache']/v['B']), '%.2f' % (v['A']/v['cache'])])
    return table(['Engine', 'Metric', 'Flat DDR', 'Cache', 'Flat HBM', 'Cache / DDR', 'Flat HBM / Cache'], body)


def t9(rows):
    """Perf overhead: tok/s of the repeat with counters over repeat 3 (no perf), per cell, summarised per tier and phase."""
    body = []
    for node in ('A', 'B', 'cache'):
        for ph, wls in (('decode', ('D', 'L')), ('prefill', ('P', 'L', 'LP'))):
            r1, r2 = [], []
            for c in configs(rows, node):
                for wl in wls:
                    s = {r['repeat']: r['tok_s'] for r in sel(rows, c, wl, ph) if r.get('tok_s')}
                    if 3 in s and 1 in s:
                        r1.append(s[1]/s[3])
                    if 3 in s and 2 in s:
                        r2.append(s[2]/s[3])
            body.append([TIER[node], ph, len(r1), '%.3f (%.3f-%.3f)' % (med(r1), min(r1), max(r1)),
                         ('%.3f (%.3f-%.3f)' % (med(r2), min(r2), max(r2))) if r2 else '-'])
    return table(['Tier', 'Phase', 'cells', 'repeat 1 / repeat 3 (MEM or USER group)', 'repeat 2 / repeat 3 (COMPUTE group)'], body)


def t10(rows):
    body = []
    for node in ('A', 'B'):
        s = sel(rows, (node, 'c', '', 1, 'hbm'), 'D', 'decode')
        p = sel(rows, (node, 'c', '', 1, 'hbm'), 'D', 'prefill')
        m = [r for r in s if r['repeat'] == 1][0]
        body.append([('node A' if node == 'A' else 'node B'), '5892201' if node == 'A' else '5892202', ' / '.join('%.2f' % v for v in vals(sorted(s, key=lambda r: r['repeat']), 'tok_s')),
                     cell(vals(s, 'tok_s'), '%.2f', False), ' / '.join('%.0f' % v for v in vals(sorted(p, key=lambda r: r['repeat']), 'tok_s')),
                     '%.2f' % m['gb_per_token'], '%.0f' % (m['pkg_w_s0'] + m['pkg_w_s1'])])
    return table(['Node', 'Job', 'decode tok/s repeats 1/2/3', 'median', 'prefill (86 tokens) tok/s repeats 1/2/3', 'GB/token (repeat 1)',
                  'package W (repeat 1)'], body)


def t11(rows, outs=None):
    body = []
    for c in configs(rows):
        for wl in ('P', 'D', 'L', 'LP'):
            pass
    # spread of the repeats: relative range (max-min)/median of the headline metrics
    for c in configs(rows):
        spreads = []
        for wl, ph in (('D', 'decode'), ('L', 'decode'), ('P', 'prefill')):
            v = vals(sel(rows, c, wl, ph), 'tok_s')
            if len(v) == 3:
                spreads.append((max(v) - min(v))/med(v))
        q = vals(prefill8192(rows, c), 'tok_s')
        if len(q) == 3:
            spreads.append((max(q) - min(q))/med(q))
        body.append([config_label(c), TIER[c[0]], ' / '.join('%.1f%%' % (100*s) for s in spreads)])
    return table(['Configuration', 'Tier', 'range/median of D, L, P, 8192 (%)'], body)


TABLES = {'T0': t0, 'T1a': t1a, 'T1b': t1b, 'T2': t2, 'T3': t3, 'T4': t4, 'T5': t5, 'T5b': t5b, 'T6': t6, 'T7': t7, 'T8': t8, 'T9': t9, 'T10': t10, 'T11': t11}


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('runs', help='runs.csv')
    ap.add_argument('--only', help='comma-separated table ids (default: all)')
    args = ap.parse_args(argv)
    rows = load(args.runs)
    for name in (args.only.split(',') if args.only else TABLES):
        print('### %s\n\n%s\n' % (name, TABLES[name](rows)))
    return 0


if __name__ == '__main__':
    sys.exit(main())
