"""Render original benchmark reports with the fixed HF template; verify C IDs."""
import json
from pathlib import Path
import subprocess
import sys
from transformers import AutoTokenizer

hf, baseline, probe, c_tokenizer, output = map(Path, sys.argv[1:])
output.mkdir(parents=True, exist_ok=True)
tokenizer = AutoTokenizer.from_pretrained(hf, local_files_only=True)
for workload in ('short', 'long'):
    report = json.loads((baseline / f'{workload}-1.json').read_text())
    prompt = report['parameters']['prompt']
    rendered = tokenizer.apply_chat_template(
        [{'role': 'system', 'content': ''}, {'role': 'user', 'content': prompt}],
        tokenize=False, add_generation_prompt=True, date_string='26 Jul 2024')
    ids = tokenizer.encode(rendered, add_special_tokens=False)
    c_ids = list(map(int, subprocess.check_output(
        [str(probe), 'chat', str(c_tokenizer), str(len(tokenizer)), prompt, ''], text=True).split()))
    assert ids == c_ids, 'HF/C prompt token IDs differ'
    assert len(ids) == report['tokens']['input_prompt']
    (output / f'{workload}.txt').write_text(rendered)
    (output / f'{workload}.ids').write_text(' '.join(map(str, ids)) + '\n')
    (output / f'{workload}-baseline.json').write_text(json.dumps(report, indent=2) + '\n')
    print(workload, len(ids), 'identical HF/C prompt IDs')
