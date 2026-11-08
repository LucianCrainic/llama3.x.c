"""Summarize the decode-diagnosis job (scripts/cresco8-decode-diag.sbatch) as Markdown: decode and prefill tok/s per
configuration, and, from the /proc samples taken during the two-socket launches, which threads shared a CPU.
Usage: summarize_decode_diag.py DIR."""
import json
import re
import statistics
import sys
from pathlib import Path


def single_cpu(allowed):
    """The CPU id when the allowed list is exactly one CPU, else None."""
    return int(allowed) if re.fullmatch(r'\d+', allowed or '') else None


def sample_blocks(path):
    blocks = []
    for line in path.read_text().splitlines():
        if line.startswith('##'):
            blocks.append([])
        elif blocks and line.strip():
            blocks[-1].append(line.split())
    return blocks


def affinity_report(path):
    """Per sample: threads pinned to one CPU, CPUs claimed by more than one thread, threads running on a CPU other
    than their pinned one."""
    lines = []
    for index, block in enumerate(sample_blocks(path), 1):
        pinned = {}
        stray = 0
        for fields in block:
            if len(fields) < 3:
                continue
            cpu = single_cpu(fields[1])
            if cpu is None:
                continue
            pinned.setdefault(cpu, []).append(fields[0])
            if fields[2].isdigit() and int(fields[2]) != cpu:
                stray += 1
        shared = {cpu: tids for cpu, tids in pinned.items() if len(tids) > 1}
        total = sum(len(t) for t in pinned.values())
        lines.append(f'sample {index}: {len(block)} threads, {total} pinned to one CPU on {len(pinned)} distinct CPUs, '
                     f'{len(shared)} CPU(s) claimed by more than one thread {shared if shared else ""}, '
                     f'{stray} running off their CPU')
    return lines


def stream_table(root):
    rows = []
    for path in sorted(root.glob('stream-s*.jsonl')):
        for line in path.read_text().splitlines():
            if line.startswith('{'):
                rows.append(json.loads(line))
    if not rows:
        return
    print('\n### Streaming ceiling (GB/s, median of the repetitions; both sockets read at the same time)\n')
    print('| Experiment | Sockets | Threads/socket | Per socket | Aggregate | Min-max aggregate |')
    print('| --- | ---: | ---: | --- | ---: | --- |')
    for r in rows:
        per = ' / '.join(f'{v:.0f}' for v in r['socket_gbps'][:r['sockets']])
        print(f"| {r['experiment']} | {r['sockets']} | {r['threads']} | {per} | {r['aggregate_gbps']:.0f} "
              f"| {r['aggregate_min']:.0f}-{r['aggregate_max']:.0f} |")


def main(root):
    stream_table(root)
    groups = {}
    for path in sorted(root.glob('*.json')):
        match = re.fullmatch(r'(.+)-(\d+)\.json', path.name)
        if match:
            groups.setdefault(match.group(1), []).append(path)
    print('| Configuration | Runs | Decode tok/s (each) | Median decode | Median prefill |')
    print('| --- | ---: | --- | ---: | ---: |')
    for label, paths in groups.items():
        decode, prefill = [], []
        for path in paths:
            report = json.loads(path.read_text())
            if report.get('status') != 'completed':
                continue
            rates = report.get('throughput_tokens_per_second') or {}
            decode.append(rates.get('decode'))
            prefill.append(rates.get('prefill'))
        decode = [v for v in decode if isinstance(v, (int, float))]
        prefill = [v for v in prefill if isinstance(v, (int, float))]
        each = ', '.join(f'{v:.2f}' for v in decode) or '-'
        print(f'| {label} | {len(paths)} | {each} | {statistics.median(decode) if decode else float("nan"):.2f} '
              f'| {statistics.median(prefill) if prefill else float("nan"):.2f} |')
    for label in groups:
        for threads in sorted(root.glob(f'{label}-*.threads')):
            print(f'\n### Threads of {threads.stem}\n')
            for line in affinity_report(threads):
                print(f'- {line}')
        for stderr in sorted(root.glob(f'{label}-*.stderr')):
            if not stderr.stem.endswith('-1'):
                continue  # the first repeat is enough to read the breakdown
            notes = [line for line in stderr.read_text().splitlines()
                     if 'tp decode socket' in line or 'coordinator runs on' in line]
            if notes:
                print(f'\n### {stderr.stem}\n')
                for line in notes:
                    print(f'- {line.strip()}')


if __name__ == '__main__':
    main(Path(sys.argv[1]))
