"""Aggregate exact-prompt llama.cpp runs against phase2-summary.json.

Usage: python tools/summarize_llamacpp.py RESULTS BASELINE_SUMMARY
RESULTS contains 1b-fp32/ and 8b-fp32/ downloaded from the Slurm campaign.
"""
import json
from pathlib import Path
import statistics
import sys

root = Path(sys.argv[1])
baseline = json.loads(Path(sys.argv[2]).read_text())
rows = []
for label, threads in [('1b', 8), ('8b', 32)]:
    for batch in (512, 1):
        for workload in ('short', 'long'):
            paths = [root / f'{label}-fp32/b{batch}-{workload}-{i}.json' for i in range(1, 4)]
            runs = [json.loads(path.read_text()) for path in paths]
            ref = next(r for r in baseline if r['threads'] == threads and r['workload'] == workload)
            assert len(set(ref['input_tokens'])) == len(set(ref['output_tokens'])) == 1
            for path, run in zip(paths, runs):
                assert run['input'] == ref['input_tokens'][0]
                assert run['output'] == 128 and run['decode_steps'] == 127 and not run['stopped_eog']
                assert run['threads'] == threads and len(run['affinity_cpus']) == threads
                assert run['batch'] == batch and run['context'] == 2048
                assert run['weights'] == run['kv'] == 'f32'
                # GNU time independently observes the process high-water mark.
                time_lines = path.with_suffix('.time').read_text().splitlines()
                rss_kib = int(next(line.split(':', 1)[1] for line in time_lines if 'Maximum resident set size' in line))
                assert rss_kib * 1024 >= run['peak_rss_bytes']
            measurements = {
                key: {'mean': statistics.mean(r[key] for r in runs),
                      'stdev': statistics.stdev(r[key] for r in runs)}
                for key in ('prefill_tok_s', 'decode_tok_s')}
            # Include any later high-water mark observed during teardown too.
            peaks = []
            for path in paths:
                line = next(line for line in path.with_suffix('.time').read_text().splitlines()
                            if 'Maximum resident set size' in line)
                peaks.append(int(line.split(':', 1)[1]) * 1024)
            measurements['peak_rss_bytes'] = max(peaks)
            baseline_rss = max(json.loads(Path(p).read_text())['memory_bytes']['peak_resident'] for p in ref['runs'])
            rows.append(dict(model=ref['model']['name'], threads=threads, batch=batch, workload=workload,
                input=runs[0]['input'], output=128, runs=[str(p) for p in paths], measurements=measurements,
                baseline=dict(prefill_tok_s=ref['measurements']['prefill_tok_s']['mean'],
                    decode_tok_s=ref['measurements']['decode_tok_s']['mean'], peak_rss_bytes=baseline_rss),
                speedup={key: measurements[key]['mean'] / ref['measurements'][key]['mean']
                         for key in ('prefill_tok_s', 'decode_tok_s')},
                generated_ids_identical_across_repeats=all(r['generated_ids'] == runs[0]['generated_ids'] for r in runs)))
for row in rows:
    other_batch = 1 if row['batch'] == 512 else 512
    # Use the model's configured operating point to locate the paired run.
    other = root / f"{'1b' if row['threads'] == 8 else '8b'}-fp32/b{other_batch}-{row['workload']}-1.json"
    first = json.loads(Path(row['runs'][0]).read_text())
    row['generated_ids_match_other_batch'] = first['generated_ids'] == json.loads(other.read_text())['generated_ids']
print(json.dumps(rows, indent=2))
