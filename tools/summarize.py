"""Summarize successful short/long benchmark reports; keep individual raw runs."""
import json
from pathlib import Path
import statistics
import sys

def summarize(root):
    summaries = []
    for workload in ('short', 'long'):
        paths = sorted(root.glob(workload+'-*.json'))
        if not paths:
            continue
        runs = [json.loads(p.read_text()) for p in paths]
        if any(r['status'] != 'completed' or r['tokens']['generated_output'] == 0 for r in runs):
            raise ValueError(f'{workload}: failed or empty benchmark run')
        fields = {}
        for label, section, key in [('prefill_tok_s','throughput_tokens_per_second','prefill'),
            ('decode_tok_s','throughput_tokens_per_second','decode'), ('load_s','timing_seconds','checkpoint_load'),
            ('peak_rss_bytes','memory_bytes','peak_resident')]:
            values = [r[section][key] for r in runs if r[section][key] is not None]
            fields[label] = dict(mean=statistics.mean(values), stdev=statistics.stdev(values) if len(values)>1 else None) if values else None
        summaries.append(dict(workload=workload, runs=[str(p) for p in paths], model=runs[0]['model'],
            input_tokens=[r['tokens']['input_prompt'] for r in runs],
            output_tokens=[r['tokens']['generated_output'] for r in runs], measurements=fields))
    return summaries

if __name__ == '__main__':
    print(json.dumps(summarize(Path(sys.argv[1])), indent=2))
