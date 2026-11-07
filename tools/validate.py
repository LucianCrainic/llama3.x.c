"""Compare a converted model to local Hugging Face weights on a compute node.

No network access is needed; results include numerical errors and greedy tokens.
Use the exact FP32 reference weights used by export.py (BF16 sources are widened).
"""
# Capture the Slurm/task mask before importing Torch's OpenMP runtime. It may
# bind the launcher thread to one core; child C probes must see the full mask.
import os
INITIAL_AFFINITY = set(os.sched_getaffinity(0)) if hasattr(os, 'sched_getaffinity') else None

import argparse
import json
from pathlib import Path
import subprocess
import sys
import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from tokenizer import Tokenizer


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--hf', required=True)
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--tokenizer', required=True)
    parser.add_argument('--precision', choices=('fp32', 'int8'), default='fp32')
    parser.add_argument('--build-dir', default='build/release')
    parser.add_argument('--output', required=True)
    parser.add_argument('--max-abs', type=float, default=1e-3)
    parser.add_argument('--rtol', type=float, default=1e-3)
    parser.add_argument('--greedy-steps', type=int, default=8)
    parser.add_argument('--chat-date', default='26 Jul 2024')
    parser.add_argument('--threads', type=int, default=4)
    parser.add_argument('--batch', type=int, metavar='M',
                        help='C probe logits through the batched prefill path, microbatch M (C5/C6); the environment '
                             '(LLAMA_PRECISION, LLAMA_KERNELS, LLAMA_PREFILL_BATCH) reaches the probe unchanged')
    parser.add_argument('--record-only', action='store_true',
                        help='write the comparison, greedy tokens included, but exit 0 whatever it shows (BF16 mode)')
    args = parser.parse_args()
    torch.set_num_threads(args.threads)
    hf = AutoModelForCausalLM.from_pretrained(args.hf, torch_dtype=torch.float32,
        local_files_only=True, attn_implementation='eager', trust_remote_code=False).eval()
    ht = AutoTokenizer.from_pretrained(args.hf, local_files_only=True, trust_remote_code=False)
    reference = Tokenizer.from_hf(args.hf, args.chat_date)
    probe = Path(args.build_dir).resolve()/f'probe-{args.precision}'
    records, passed = [], True
    def probe_output(command, **kwargs):
        if INITIAL_AFFINITY is not None:
            os.sched_setaffinity(0, INITIAL_AFFINITY)
        return subprocess.check_output(command, **kwargs)
    def c_logits(tokens):
        if args.batch:
            output = probe_output([str(probe), 'batch', args.checkpoint, *map(str, tokens)],
                                  env={**os.environ, 'LLAMA_PREFILL_BATCH': str(args.batch), 'LLAMA_PREFILL_LOGITS': 'all'})
        else:
            output = probe_output([str(probe), args.checkpoint, *map(str, tokens)])
        return np.frombuffer(output, dtype='<f4').reshape(len(tokens), -1)
    def ref_logits(tokens):
        with torch.inference_mode():
            return hf(torch.tensor([tokens])).logits[0].numpy()
    prompts = ['Hello', 'Explain why the sky appears blue.',
               'A deterministic CPU inference test. ' * 5, 'é 中文 Ελληνικά العربية 🙂 1234567\n\n']
    for text in prompts:
        tokens = [reference.bos_id] + reference.model.encode_ordinary(text)
        actual_tokens = list(map(int, probe_output([str(probe), 'tokenize', args.tokenizer,
            str(hf.config.vocab_size), text], text=True).split()))
        tokenizer_match = actual_tokens == tokens == ht.encode(text, add_special_tokens=True)
        actual, expected = c_logits(tokens), ref_logits(tokens)
        delta = actual - expected
        close = bool(np.allclose(actual, expected, atol=args.max_abs, rtol=args.rtol))
        finite = bool(np.isfinite(actual).all())
        passed &= tokenizer_match and close and finite
        records.append(dict(prompt=text, input_tokens=len(tokens), tokenizer_match=tokenizer_match,
            max_abs=float(np.abs(delta).max()), rms=float(np.sqrt(np.mean(delta**2))),
            argmax_agreement=float(np.mean(actual.argmax(-1) == expected.argmax(-1))), passed=close and finite))
    messages = [{'role':'user', 'content':'Give a short definition of a CPU.'}]
    chat_tokens = ht.apply_chat_template(messages, add_generation_prompt=True, date_string=args.chat_date)
    actual_chat = list(map(int, probe_output([str(probe), 'chat', args.tokenizer,
        str(hf.config.vocab_size), messages[0]['content']], text=True).split()))
    chat_match = actual_chat == chat_tokens
    passed &= chat_match
    generation = []
    stop_ids = {reference.special_tokens[s] for s in ('<|end_of_text|>', '<|eot_id|>', '<|eom_id|>') if s in reference.special_tokens}
    for mode, prefix in [('base', [reference.bos_id]+reference.model.encode_ordinary('The capital of Italy is')),
                         ('instruct', chat_tokens)]:
        c_output = list(map(int, probe_output([str(probe), 'generate', args.checkpoint,
            str(args.greedy_steps), *map(str, prefix)], text=True).split()))
        cp, rp = prefix + c_output, prefix[:]
        cdone = bool(c_output and c_output[-1] in stop_ids)
        rdone = False
        for _ in range(args.greedy_steps):
            token = int(ref_logits(rp)[-1].argmax()); rp.append(token); rdone = token in stop_ids
            if rdone: break
        match = cp == rp
        if args.precision == 'fp32' and not args.record_only: passed &= match
        generation.append(dict(mode=mode, match=match, c_tokens=cp[len(prefix):], reference_tokens=rp[len(prefix):],
                               c_stopped=cdone, reference_stopped=rdone))
    # FP32 requires identical greedy tokens as well as the specified tensor
    # tolerance. Quantized comparisons report token divergence separately.
    result = dict(passed=bool(passed), hf=args.hf, checkpoint=args.checkpoint, precision=args.precision,
        probe_affinity=sorted(INITIAL_AFFINITY) if INITIAL_AFFINITY else None,
        torch=torch.__version__, transformers=__import__('transformers').__version__,
        tolerance=dict(atol=args.max_abs, rtol=args.rtol), chat_template_match=chat_match,
        batch=args.batch, record_only=args.record_only,
        comparisons=records, generation=generation)
    Path(args.output).write_text(json.dumps(result, indent=2, ensure_ascii=False)+'\n')
    print(json.dumps(result, indent=2, ensure_ascii=False))
    return 0 if passed or args.record_only else 1

if __name__ == '__main__':
    sys.exit(main())
