#!/usr/bin/env python3
"""Harness-side helpers of the cross-engine campaign (scripts/xengine.sbatch, scripts/lib/xengine-common.sh).

Subcommands (all stdlib only, so they run on any node):

  write-cell DIR --field K=V ... [--env VAR=VAL ...] [--command STR]
      Writes DIR/cell.json, the manifest entry of one run (called twice: status started, then the final status).
  markers DIR
      Writes DIR/markers.json from DIR/anchors (wall-clock anchors recorded by the job), DIR/report.json (the engine's
      own report) and DIR/perf.csv (only its last time stamp is read, to line the perf clock up with the wall clock).
  check-c DIR --sockets S --mem hbm|ddr|cache --generation G --prompt-tokens N
      Verifies a llama3.x.c report: completed, exactly G generated tokens (--benchmark does not ignore EOS), N prompt
      tokens, the socket count, and the memory placement of every region. Prints PASS/FAIL lines, exit 1 on any FAIL.
  assemble OUT [--campaign K=V ...]
      Merges OUT/runs/*/cell.json into OUT/manifest.json.
  llamacpp-report DIR
      Turns the JSON that llama-bench printed (DIR/stdout) into DIR/report.json (its first entry, plus a status).
  check-llamacpp DIR --n-prompt P --n-gen G --depth D --threads T --reps R
      Verifies a llama-bench report: shape of the test, thread count, number of repetitions, strict CPU mask, no mmap.
  vllm-report DIR
      Turns DIR/client.jsonl (one measured request of tools/xengine_vllm_client.py) into DIR/report.json.
  check-vllm DIR --generation G --prompt-tokens N
      Verifies a vLLM request: HTTP 200, exact prompt and completion token counts, G token arrivals.
  check-prefix CONFIG_DIR
      Asserts vllm:prefix_cache_hits_total == 0 in CONFIG_DIR/metrics-after.txt (prefix caching must stay off).
  numa-maps PID
      Summarises /proc/PID/numa_maps as JSON (pages per NUMA node, split by kind of mapping).

Marker file (epoch nanoseconds of the job's wall clock, `date +%s%N`; markers.json) and how each is derived
------------------------------------------------------------------------------------------------------------------
Anchors recorded by the job around the process (DIR/anchors, key=value):
  launch_ns         taken immediately before the engine process is started in the background
  load_end_seen_ns  first time the job's poller (20 ms period) saw the engine's end-of-load line on stderr
  end_ns            taken immediately after `wait` returned for the engine process (process exit)
  perf_start_ns     taken immediately before the system-wide perf was started; perf_mode is sys, wrap or none

llama3.x.c has no hooks, so the markers come from its JSON timings (timing_seconds) placed on the anchors:
  load_start_ns     = launch_ns (includes exec, checkpoint load, weight pack, placement populate and verify)
  load_end_seen_ns  = observed anchor above. The engine logs `placement=...` (src/app.c) right after initialisation and
                      before it tokenises the prompt, so this is the start of generate(); error is the poll period plus
                      stderr latency, below 30 ms.
  tokenize_start_ns = load_end_seen_ns
  prefill_start_ns  = load_end_seen_ns + prompt_tokenization (the engine's TTFT is tokenisation + prefill + first sample)
  first_token_ns    = load_end_seen_ns + time_to_first_token
  decode_end_ns     = first_token_ns + (total_inference - time_to_first_token), the end of the last sampling step
  end_ns            = process exit (includes freeing the regions, writing the report, process teardown)
  Fallback when the end-of-load line was never seen: the same markers placed backwards from end_ns, assuming zero
  teardown (flagged by anchor=end in markers.json). teardown_s = end_ns - decode_end_ns is recorded as a sanity value.
vLLM (markers deriver derive_markers_vllm): the client records, per request, t_first_token and every token arrival
relative to the moment it sent the request, and the wall clock when the request finished (wall_time):
  load_start_ns     = server launch (anchor load_start_ns written by the job; the client's own launch is launch_ns)
  prefill_start_ns  = request send = wall_time - t_end (the ids are sent as is, so send is the start of the prefill)
  first_token_ns    = request send + t_first_token (the first streamed token)
  decode_end_ns     = request send + last token arrival;  end_ns = client process exit
  wall_time is taken right after the request returns, so the edges are good to a few ms.
llama.cpp (derive_markers_llamacpp): llama-bench prints no load log and no per-token hooks. One invocation is a warm-up
run plus R timed repetitions (-r R), each preceded by an untimed depth prefill (-d). All it reports are the
repetitions' times (samples_ns). The rule is conservative: only the LAST repetition is placed, working backwards from
process exit:
  window_end   = end_ns - teardown_assumed_s   (anchor from the job, default 0.25 s: freeing the model and exit)
  window_start = window_end - samples_ns[-1]
  pp test (-p N -n 0):  prefill_start_ns = window_start, first_token_ns = window_end (prefill window; decode_end absent)
  tg test (-n G):       first_token_ns = window_start (start of the timed decode), decode_end_ns = window_end
  load_start_ns = launch_ns. Both windows lie inside a steady-state phase of the same kind as the repetitions before them,
  so a misjudged teardown of a few tenths of a second moves the window along the plateau, not out of it.
Perf clock: `perf -I` stamps are seconds since counting started. The start is not known to better than the perf start-up
time (tens to hundreds of ms), but perf stops within one `tail` poll (0.1 s) of the engine's exit, so the clock is lined
up at the end: perf_t0_ns = end_ns - last perf time stamp + half the poll period (system-wide perf; zero for the
wrapper mode, where perf ends with the child). The launch-based estimate is recorded next to it (perf_t0_launch_ns) and
their difference is reported by the summariser as perf_clock_skew_s. Uncertainty of any window edge: about 0.1 s.
"""
import argparse
import json
from pathlib import Path
import re
import sys

NS = 1e9
TAIL_POLL_S = 0.1  # `tail -s 0.1 --pid`: how long after the engine's exit the system-wide perf may keep running


def read_kv(path):
    out = {}
    p = Path(path)
    if not p.exists():
        return out
    for line in p.read_text().splitlines():
        if '=' in line:
            k, v = line.split('=', 1)
            out[k.strip()] = v.strip()
    return out


def load_json(path):
    p = Path(path)
    return json.loads(p.read_text()) if p.exists() else None


def last_perf_time(path):
    """Last interval time stamp (seconds) of a `perf stat -I -x,` file, or None."""
    p = Path(path)
    if not p.exists():
        return None
    last = None
    for line in p.read_text().splitlines():
        if not line or line.startswith('#'):
            continue
        head = line.split(',', 1)[0]
        try:
            last = float(head)
        except ValueError:
            continue
    return last


def derive_markers_c(anchors, report, perf_csv=None):
    """Markers of one llama3.x.c run (see the module docstring)."""
    launch = int(anchors['launch_ns'])
    end = int(anchors['end_ns'])
    seen = anchors.get('load_end_seen_ns')
    m = {'load_start_ns': launch, 'end_ns': end, 'prefill_start_ns': None, 'first_token_ns': None,
         'tokenize_start_ns': None, 'decode_end_ns': None, 'load_end_seen_ns': int(seen) if seen else None}
    t = (report or {}).get('timing_seconds') or {}
    ttft, total, tok = t.get('time_to_first_token'), t.get('total_inference'), t.get('prompt_tokenization')
    if ttft is not None and total is not None:
        if seen:
            base, anchor = int(seen), 'load_end'
            first = base + ttft * NS
        else:
            anchor = 'end'
            first = end - (total - ttft) * NS
            base = first - ttft * NS
        m['anchor'] = anchor
        m['tokenize_start_ns'] = int(base)
        m['prefill_start_ns'] = int(base + (tok or 0.0) * NS)
        m['first_token_ns'] = int(first)
        m['decode_end_ns'] = int(first + (total - ttft) * NS)
        m['teardown_s'] = (end - m['decode_end_ns']) / NS
    return m


def derive_markers_vllm(anchors, report, perf_csv=None):
    end = int(anchors['end_ns'])
    m = {'load_start_ns': int(anchors.get('load_start_ns') or anchors['launch_ns']), 'end_ns': end, 'prefill_start_ns': None,
         'first_token_ns': None, 'decode_end_ns': None}
    r = report or {}
    arrivals = r.get('token_arrivals') or []
    if r.get('wall_time') and r.get('t_end') is not None and arrivals:
        send = r['wall_time'] * NS - r['t_end'] * NS
        m['anchor'] = 'client_wall_clock'
        m['prefill_start_ns'] = int(send)
        m['first_token_ns'] = int(send + r['t_first_token'] * NS)
        m['decode_end_ns'] = int(send + arrivals[-1] * NS)
        m['teardown_s'] = (end - r['wall_time'] * NS) / NS  # client exit after the request finished
    return m


def derive_markers_llamacpp(anchors, report, perf_csv=None):
    launch = int(anchors['launch_ns'])
    end = int(anchors['end_ns'])
    m = {'load_start_ns': launch, 'end_ns': end, 'prefill_start_ns': None, 'first_token_ns': None, 'decode_end_ns': None}
    r = report or {}
    samples = r.get('samples_ns') or []
    if samples:
        td = float(anchors.get('teardown_assumed_s', 0.25))
        wend = end - td * NS
        wstart = wend - samples[-1]
        m['anchor'] = 'process_exit_minus_teardown'
        m['teardown_assumed_s'] = td
        if r.get('n_gen', 0) > 0:
            m['first_token_ns'], m['decode_end_ns'] = int(wstart), int(wend)
        else:
            m['prefill_start_ns'], m['first_token_ns'] = int(wstart), int(wend)
    return m


MARKER_DERIVERS = {'c': derive_markers_c, 'vllm': derive_markers_vllm, 'llamacpp': derive_markers_llamacpp}


def perf_clock(anchors, markers, perf_csv):
    """perf_t0_ns (wall clock of perf time 0) lined up at the end, and the launch-based estimate."""
    mode = anchors.get('perf_mode', 'none')
    last = last_perf_time(perf_csv) if perf_csv else None
    if mode == 'none' or last is None:
        return {}
    out = {'perf_mode': mode}
    wrapper = mode == 'wrap'
    out['perf_t0_ns'] = int(markers['end_ns'] - last * NS + (0.0 if wrapper else TAIL_POLL_S / 2 * NS))
    start = anchors.get('launch_ns') if wrapper else anchors.get('perf_start_ns')
    if start:
        out['perf_t0_launch_ns'] = int(start)
    return out


def cmd_markers(args):
    d = Path(args.dir)
    anchors = read_kv(d/'anchors')
    cell = load_json(d/'cell.json') or {}
    engine = cell.get('engine', args.engine or 'c')
    if 'launch_ns' not in anchors or 'end_ns' not in anchors:
        print('no launch/end anchors in %s' % d, file=sys.stderr)
        return 1
    report = load_json(d/'report.json')
    markers = MARKER_DERIVERS[engine](anchors, report)
    markers.update(perf_clock(anchors, markers, d/'perf.csv'))
    markers['engine'] = engine
    markers['clock'] = 'epoch_ns'
    markers['derivation'] = 'see tools/xengine_cell.py module docstring'
    (d/'markers.json').write_text(json.dumps(markers, indent=1) + '\n')
    return 0


def cmd_write_cell(args):
    d = Path(args.dir)
    d.mkdir(parents=True, exist_ok=True)
    entry = {}
    for item in args.field or []:
        k, v = item.split('=', 1)
        entry[k] = int(v) if re.fullmatch(r'-?\d+', v) and k in ('sockets', 'repeat', 'rc', 'generation') else v
    entry['env'] = list(args.env or [])
    entry['command'] = args.command or ''
    (d/'cell.json').write_text(json.dumps(entry, indent=1) + '\n')
    return 0


def cmd_assemble(args):
    out = Path(args.out)
    cells = []
    for path in sorted((out/'runs').glob('*/cell.json')):
        entry = json.loads(path.read_text())
        entry['dir'] = path.parent.name
        cells.append(entry)
    campaign = {}
    for item in args.campaign or []:
        k, v = item.split('=', 1)
        campaign[k] = v
    (out/'manifest.json').write_text(json.dumps({'campaign': campaign, 'cells': cells}, indent=1) + '\n')
    return 0


def check_c(report, sockets, mem, generation, prompt_tokens):
    """List of (ok, name, detail) for one llama3.x.c report."""
    res = []

    def check(name, ok, detail=''):
        res.append((bool(ok), name, str(detail)))

    if not isinstance(report, dict):
        return [(False, 'report present', 'no report.json')]
    check('run completed', report.get('status') == 'completed', report.get('status'))
    check('stopped at the generation limit', report.get('stop_reason') == 'generation_limit', report.get('stop_reason'))
    tokens = report.get('tokens') or {}
    check('generated tokens == G (EOS not hit)', tokens.get('generated_output') == generation,
          'generated=%s G=%s' % (tokens.get('generated_output'), generation))
    check('prompt tokens exact', (report.get('prefill') or {}).get('tokens') == prompt_tokens,
          'prefill.tokens=%s expected=%s' % ((report.get('prefill') or {}).get('tokens'), prompt_tokens))
    runtime = report.get('runtime') or {}
    check('sockets', runtime.get('sockets') == sockets, 'runtime.sockets=%s expected=%s' % (runtime.get('sockets'), sockets))
    regions = report.get('regions') or []
    if mem == 'cache' and sockets == 1:
        # Cache mode, no placement asked for: the nodes are recorded, not judged.
        check('placement recorded (not judged)', True, 'placement=%s nodes=%s' % (
            runtime.get('placement'), sorted({(r['role'], r.get('node')) for r in regions})))
        return res
    base = {'hbm': 2, 'ddr': 0, 'cache': 0}[mem]  # HBM node of socket s is 2+s; DDR node of socket s is s
    for s in range(sockets):
        node = base + s
        weights = [r for r in regions if r['role'] == 'weights' and r['socket'] == s]
        check('socket %d weights: one region on node %d, fully verified' % (s, node),
              len(weights) == 1 and weights[0]['node'] == node and weights[0]['verified'] == 'full' and
              (weights[0]['pages_queried'] or 0) > 0 and weights[0]['pages_mismatched'] == 0 and weights[0]['pages_missing'] == 0,
              weights)
        others = [r for r in regions if r['socket'] == s and r['role'] not in ('weights', 'embedding')]
        wrong = [r for r in others if r['node'] != node or r.get('pages_mismatched')]
        check('socket %d other regions (%s) on node %d' % (s, ','.join(sorted({r['role'] for r in others})) or 'none', node),
              not wrong, wrong)
    for r in regions:
        if r['role'] == 'embedding':
            check('embedding on DDR node 0', r['node'] == 0 and r.get('pages_mismatched') == 0, r)
    return res


def check_vllm(report, generation, prompt_tokens):
    if not isinstance(report, dict):
        return [(False, 'report present', 'no report.json')]
    res = []
    res.append((report.get('http_status') == 200, 'HTTP 200', report.get('http_status')))
    res.append((report.get('usage_prompt_tokens') == prompt_tokens, 'prompt tokens exact',
                'usage=%s expected=%s' % (report.get('usage_prompt_tokens'), prompt_tokens)))
    res.append((report.get('usage_completion_tokens') == generation, 'completion tokens == G',
                'usage=%s G=%s' % (report.get('usage_completion_tokens'), generation)))
    res.append((len(report.get('token_arrivals') or []) == generation, 'G token arrivals',
                'arrivals=%s' % len(report.get('token_arrivals') or [])))
    return [(bool(ok), name, str(detail)) for ok, name, detail in res]


def parse_bench_json(text):
    start = text.find('[')
    return json.loads(text[start:]) if start >= 0 else []


def cmd_llamacpp_report(args):
    d = Path(args.dir)
    try:
        entries = parse_bench_json((d/'stdout').read_text())
    except (OSError, ValueError):
        return 1
    if not entries:
        return 1
    r = entries[0]
    r['status'] = 'completed' if r.get('samples_ns') else 'failed'
    (d/'report.json').write_text(json.dumps(r) + '\n')
    return 0


def check_llamacpp(report, n_prompt, n_gen, depth, threads, reps):
    if not isinstance(report, dict):
        return [(False, 'report present', 'no report.json')]
    res = [(report.get('n_prompt') == n_prompt and report.get('n_gen') == n_gen and report.get('n_depth') == depth,
            'test shape', 'n_prompt=%s n_gen=%s n_depth=%s expected %s/%s/%s' % (
                report.get('n_prompt'), report.get('n_gen'), report.get('n_depth'), n_prompt, n_gen, depth)),
           (report.get('n_threads') == threads, 'threads', report.get('n_threads')),
           (len(report.get('samples_ns') or []) == reps, 'repetitions', len(report.get('samples_ns') or [])),
           (report.get('cpu_strict') is True, 'strict CPU mask', report.get('cpu_strict')),
           (report.get('load_mode') == 'none', 'weights loaded without mmap', report.get('load_mode'))]
    return [(bool(ok), n, str(dt)) for ok, n, dt in res]


def cmd_check_llamacpp(args):
    res = check_llamacpp(load_json(Path(args.dir)/'report.json'), args.n_prompt, args.n_gen, args.depth, args.threads, args.reps)
    for ok, name, detail in res:
        print('PASS' if ok else 'FAIL', name, '' if ok else detail)
    return 0 if all(ok for ok, _, _ in res) else 1


def cmd_vllm_report(args):
    d = Path(args.dir)
    records = [json.loads(ln) for ln in (d/'client.jsonl').read_text().splitlines() if ln.strip()] \
        if (d/'client.jsonl').exists() else []
    records = [r for r in records if not r.get('warmup')]
    if not records:
        return 1
    r = records[-1]
    r['status'] = 'completed' if r.get('ok') else 'failed'
    (d/'report.json').write_text(json.dumps(r) + '\n')
    return 0


def cmd_check_vllm(args):
    res = check_vllm(load_json(Path(args.dir)/'report.json'), args.generation, args.prompt_tokens)
    for ok, name, detail in res:
        print('PASS' if ok else 'FAIL', name, '' if ok else detail)
    return 0 if all(ok for ok, _, _ in res) else 1


def prefix_hits(text):
    """Sum of vllm:prefix_cache_hits_total over all label sets, or None when the metric is absent."""
    vals = [float(ln.rsplit(' ', 1)[1]) for ln in text.splitlines()
            if ln.startswith('vllm:prefix_cache_hits_total') and ' ' in ln]
    return sum(vals) if vals else None


def cmd_check_prefix(args):
    p = Path(args.dir)/'metrics-after.txt'
    hits = prefix_hits(p.read_text()) if p.exists() else None
    print('%s prefix_cache_hits_total=%s' % ('PASS' if hits == 0 else 'FAIL', hits))
    return 0 if hits == 0 else 1


def cmd_check_c(args):
    report = load_json(Path(args.dir)/'report.json')
    res = check_c(report, args.sockets, args.mem, args.generation, args.prompt_tokens)
    for ok, name, detail in res:
        print('PASS' if ok else 'FAIL', name, '' if ok else detail)
    return 0 if all(ok for ok, _, _ in res) else 1


def numa_maps_summary(text):
    """Pages-to-bytes per NUMA node, split by kind of mapping, from the text of /proc/PID/numa_maps."""
    kinds = {}
    big = []
    for line in text.splitlines():
        parts = line.split()
        if len(parts) < 2:
            continue
        kv = {}
        flags = set()
        for p in parts[2:]:
            if '=' in p:
                k, v = p.split('=', 1)
                kv[k] = v
            else:
                flags.add(p)
        kps = int(kv.get('kernelpagesize_kB', 4)) * 1024
        nodes = {int(k[1:]): int(v) * kps for k, v in kv.items() if re.fullmatch(r'N\d+', k)}
        if not nodes:
            continue
        kind = 'file' if 'file' in kv else 'heap' if 'heap' in flags else 'stack' if 'stack' in flags else 'anon'
        acc = kinds.setdefault(kind, {})
        for n, b in nodes.items():
            acc[n] = acc.get(n, 0) + b
        big.append((sum(nodes.values()), parts[0], kind, kv.get('file', ''), nodes))
    big.sort(reverse=True)
    total = {}
    for acc in kinds.values():
        for n, b in acc.items():
            total[n] = total.get(n, 0) + b
    return {'bytes_by_node': {str(n): b for n, b in sorted(total.items())},
            'bytes_by_kind_and_node': {k: {str(n): b for n, b in sorted(v.items())} for k, v in sorted(kinds.items())},
            'largest_mappings': [{'addr': a, 'kind': k, 'file': f, 'bytes': b,
                                  'bytes_by_node': {str(n): x for n, x in sorted(nodes.items())}}
                                 for b, a, k, f, nodes in big[:8]]}


def cmd_numa_maps(args):
    try:
        text = Path('/proc/%d/numa_maps' % args.pid).read_text()
    except OSError as e:
        print(json.dumps({'error': str(e)}))
        return 1
    print(json.dumps(numa_maps_summary(text), indent=1))
    return 0


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    sub = ap.add_subparsers(dest='cmd', required=True)
    p = sub.add_parser('write-cell')
    p.add_argument('dir')
    p.add_argument('--field', action='append')
    p.add_argument('--env', action='append')
    p.add_argument('--command')
    p.set_defaults(fn=cmd_write_cell)
    p = sub.add_parser('markers')
    p.add_argument('dir')
    p.add_argument('--engine')
    p.set_defaults(fn=cmd_markers)
    p = sub.add_parser('check-c')
    p.add_argument('dir')
    p.add_argument('--sockets', type=int, required=True)
    p.add_argument('--mem', choices=('hbm', 'ddr', 'cache'), required=True)
    p.add_argument('--generation', type=int, required=True)
    p.add_argument('--prompt-tokens', type=int, required=True)
    p.set_defaults(fn=cmd_check_c)
    p = sub.add_parser('llamacpp-report')
    p.add_argument('dir')
    p.set_defaults(fn=cmd_llamacpp_report)
    p = sub.add_parser('check-llamacpp')
    p.add_argument('dir')
    for name in ('n-prompt', 'n-gen', 'depth', 'threads', 'reps'):
        p.add_argument('--' + name, type=int, required=True)
    p.set_defaults(fn=cmd_check_llamacpp)
    p = sub.add_parser('vllm-report')
    p.add_argument('dir')
    p.set_defaults(fn=cmd_vllm_report)
    p = sub.add_parser('check-vllm')
    p.add_argument('dir')
    p.add_argument('--generation', type=int, required=True)
    p.add_argument('--prompt-tokens', type=int, required=True)
    p.set_defaults(fn=cmd_check_vllm)
    p = sub.add_parser('check-prefix')
    p.add_argument('dir')
    p.set_defaults(fn=cmd_check_prefix)
    p = sub.add_parser('assemble')
    p.add_argument('out')
    p.add_argument('--campaign', action='append')
    p.set_defaults(fn=cmd_assemble)
    p = sub.add_parser('numa-maps')
    p.add_argument('pid', type=int)
    p.set_defaults(fn=cmd_numa_maps)
    args = ap.parse_args(argv)
    return args.fn(args)


if __name__ == '__main__':
    sys.exit(main())
