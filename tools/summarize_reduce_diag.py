"""Summarize the Wo/W2 reduction diagnosis as Markdown: the tests/reduce_bench.c runs (reduce-bench.jsonl) and the per-
reduction traces of LLAMA_TP_TRACE (*.trace.csv) of scripts/cresco8-decode-diag.sbatch.
Usage: summarize_reduce_diag.py DIR."""
import csv
import json
import statistics
import sys
from pathlib import Path


def bench_rows(root):
    path = root / 'reduce-bench.jsonl'
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text().splitlines() if line.startswith('{')]


def med(values):
    return statistics.median(values) if values else float('nan')


def bench_tables(rows):
    runs = [r for r in rows if 'breakdown' not in r]
    groups = {}
    for r in runs:
        groups.setdefault((r['model'], r['precision']), []).append(r)
    for (model, precision), items in groups.items():
        print(f'\n### reduce-bench {model} {precision} (synthetic layers, medians over the runs; us per phase, '
              f'GB/s = the phase\'s weight bytes of all sockets / time)\n')
        print('| Sockets | Reduction | Stamps | Layer | Wo us | Wo GB/s | W1/W3 us | W1/W3 GB/s | W2 us | W2 GB/s |')
        print('| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |')
        keyed = {}
        for r in items:
            keyed.setdefault((r['sockets'], r['red'], r['stamps']), []).append(r)
        for (sockets, red, stamps), rs in sorted(keyed.items(), key=lambda kv: (kv[0][0], kv[0][2], kv[0][1])):
            for s in range(sockets):
                key = f'socket{s}'
                cell = lambda name: med([x[key][name] for x in rs])
                label = red if sockets == 1 else f'{red} (socket {s})'
                print(f'| {sockets} | {label} | {stamps} | {cell("layer_us"):.0f} | {cell("wo_us"):.0f} '
                      f'| {cell("wo_gbps"):.0f} | {cell("w1_w3_us"):.0f} | {cell("w1_w3_gbps"):.0f} '
                      f'| {cell("w2_us"):.0f} | {cell("w2_gbps"):.0f} |')
    brk = [r for r in rows if 'breakdown' in r]
    if not brk:
        return
    print('\n### reduce-bench: steps of one projection + reduction job, microseconds after the leader\'s dispatch '
          '(means; stamps on)\n')
    print('| Job | Reduction | Socket | dispatch (last worker enters) | projection first / mean / last member '
          '| barrier 1 done | peer wait (leader) | peer seen (last) | add done | barrier 2 done | join done '
          '| total p50 / p99 | skew of last member vs peer (mean / mean abs) |')
    print('| --- | --- | ---: | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- |')
    # the breakdown lines follow their run line; recover the run's red/model from the preceding run line
    context = None
    for r in rows:
        if 'breakdown' not in r:
            context = r
            continue
        if context is None or context.get('stamps') != 1 or context['sockets'] != 2:
            continue
        print(f"| {context['model']} {context['precision']} {r['breakdown']} | {context['red']} | {r['socket']} "
              f"| {r['dispatch_max_us']:.1f} | {r['proj_first_us']:.1f} / {r['proj_mean_us']:.1f} / {r['proj_last_us']:.1f} "
              f"| {r['barrier1_done_us']:.1f} | {r['peer_wait_us']:.1f} | {r['peer_seen_last_us']:.1f} "
              f"| {r['add_done_us']:.1f} | {r['barrier2_done_us']:.1f} | {r['join_done_us']:.1f} "
              f"| {r['total_p50_us']:.0f} / {r['total_p99_us']:.0f} "
              f"| {r['proj_last_skew_vs_peer_us']:+.1f} / {r['proj_last_abs_skew_us']:.1f} |")


def read_trace(path):
    records = {0: [], 1: []}
    with path.open() as f:
        for row in csv.DictReader(f):
            records[int(row['socket'])].append({k: float(v) for k, v in row.items()})
    return records


def pct(values, q):
    values = sorted(values)
    return values[min(len(values) - 1, int(len(values) * q))] if values else float('nan')


def trace_table(path):
    records = read_trace(path)
    print(f'\n### {path.name} (decode stage only; microseconds after the leader\'s dispatch; means unless noted)\n')
    print('| Job | Socket | n | dispatch | projection first / mean / last | spread (last-first) | barrier 1 '
          '(last in -> out) | peer wait | peer wait (last seen) | add | barrier 2 | join | total mean / p50 / p99 '
          '| skew vs peer: end_last (mean, mean abs) | skew vs peer: ready |')
    print('| --- | ---: | ---: | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- | --- |')
    for kind in (0, 1):
        for s in (0, 1):
            mine = [r for r in records[s] if r['stage'] == 1 and r['kind'] == kind]
            other = [r for r in records[1 - s] if r['stage'] == 1 and r['kind'] == kind]
            if not mine:
                continue
            col = lambda name: statistics.fmean(r[name] for r in mine)
            skew = [(a['disp_s'] + a['end_last'] * 1e-6) - (b['disp_s'] + b['end_last'] * 1e-6) for a, b in zip(mine, other)]
            rskew = [(a['disp_s'] + a['b1_leader'] * 1e-6) - (b['disp_s'] + b['b1_leader'] * 1e-6) for a, b in zip(mine, other)]
            totals = [r['ret'] for r in mine]
            skew_text = (f'{statistics.fmean(skew) * 1e6:+.1f}, {statistics.fmean(abs(x) for x in skew) * 1e6:.1f}'
                         if skew else '-')
            rskew_text = (f'{statistics.fmean(rskew) * 1e6:+.1f}, {statistics.fmean(abs(x) for x in rskew) * 1e6:.1f}'
                          if rskew else '-')
            print(f"| {'w2' if kind else 'wo'} | {s} | {len(mine)} | {col('start_max'):.1f} "
                  f"| {col('end_first'):.1f} / {col('end_mean'):.1f} / {col('end_last'):.1f} "
                  f"| {col('end_last') - col('end_first'):.1f} | {col('b1_last') - col('end_last'):.1f} "
                  f"| {col('peer_leader') - col('b1_leader'):.1f} | {col('peer_last') - col('b1_last'):.1f} "
                  f"| {col('add_last') - col('peer_last'):.1f} | {col('b2_last') - col('add_last'):.1f} "
                  f"| {col('ret') - col('b2_last'):.1f} | {statistics.fmean(totals):.1f} / {med(totals):.1f} / "
                  f"{pct(totals, 0.99):.1f} | {skew_text} | {rskew_text} |")
    # members that finish last
    for kind in (0, 1):
        for s in (0, 1):
            mine = [r for r in records[s] if r['stage'] == 1 and r['kind'] == kind]
            if not mine:
                continue
            counts = {}
            for r in mine:
                counts[int(r['last_worker'])] = counts.get(int(r['last_worker']), 0) + 1
            top = sorted(counts.items(), key=lambda kv: -kv[1])[:4]
            text = ', '.join(f'worker {w}: {100 * n / len(mine):.0f}%' for w, n in top)
            print(f"\n- {'w2' if kind else 'wo'} socket {s}: last to finish the projection: {text}")


def main(root):
    rows = bench_rows(root)
    if rows:
        bench_tables(rows)
    for path in sorted(root.glob('*.trace.csv')):
        trace_table(path)


if __name__ == '__main__':
    main(Path(sys.argv[1]))
