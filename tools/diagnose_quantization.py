"""Separate W8A8 approximation loss from conversion/runtime discrepancies.

Uses the established HF Llama implementation, with independently fake-quantized
weights and Linear inputs. It intentionally does not change the FP32 error budget
or authorize benchmarks when validate.py has failed. BLAS/reduction and activation
quantization boundaries can still differ from the scalar C/grouped integer path.
"""
import os
INITIAL_AFFINITY = set(os.sched_getaffinity(0)) if hasattr(os, 'sched_getaffinity') else None
import argparse
import json
from pathlib import Path
import struct
import subprocess
import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def fake_quantize(x, group_size, activation=False):
    blocks = x.float().reshape(-1, group_size)
    scale = blocks.abs().amax(dim=-1, keepdim=True) / 127.0
    scaled = blocks / torch.where(scale > 0, scale, torch.ones_like(scale))
    # Exported weights use nearest-even; C activations use roundf (away on ties).
    q = torch.sign(scaled) * torch.floor(scaled.abs() + .5) if activation else torch.round(scaled)
    return (q.clamp(-127, 127) * scale).reshape_as(x)


def difference(actual, expected):
    delta = actual - expected
    index = np.unravel_index(np.abs(delta).argmax(), delta.shape)
    return dict(max_abs=float(np.abs(delta).max()), rms=float(np.sqrt(np.mean(delta**2))),
                argmax_agreement=float(np.mean(actual.argmax(-1) == expected.argmax(-1))),
                worst_position=int(index[0]), worst_token_id=int(index[1]),
                actual_at_worst=float(actual[index]), reference_at_worst=float(expected[index]))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--hf', required=True)
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--probe', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--threads', type=int, default=8)
    args = parser.parse_args()
    torch.set_num_threads(args.threads)
    with Path(args.checkpoint).open('rb') as checkpoint:
        header = checkpoint.read(256)
    if struct.unpack_from('<II', header) != (0x616b3432, 3) or struct.unpack_from('<I', header, 36)[0] != 1:
        parser.error('requires a version 3 repository INT8 checkpoint')
    gs = struct.unpack_from('<I', header, 44)[0]
    hf = AutoModelForCausalLM.from_pretrained(args.hf, torch_dtype=torch.float32,
        local_files_only=True, trust_remote_code=False, attn_implementation='eager').eval()
    tokenizer = AutoTokenizer.from_pretrained(args.hf, local_files_only=True, trust_remote_code=False)
    tokens = tokenizer.encode('Hello', add_special_tokens=True)
    tensor = torch.tensor([tokens])
    with torch.inference_mode():
        fp32 = hf(tensor).logits[0].float().numpy().copy()
        seen = set()
        for module in hf.modules():
            if isinstance(module, (torch.nn.Linear, torch.nn.Embedding)):
                key = module.weight.data_ptr()
                if key not in seen:
                    module.weight.copy_(fake_quantize(module.weight, gs))
                    seen.add(key)
                if isinstance(module, torch.nn.Linear):
                    module.register_forward_pre_hook(lambda _module, inputs:
                        (fake_quantize(inputs[0], gs, activation=True),))
        quantized = hf(tensor).logits[0].float().numpy().copy()
    if INITIAL_AFFINITY is not None:
        os.sched_setaffinity(0, INITIAL_AFFINITY)
    raw = subprocess.check_output([args.probe, args.checkpoint, *map(str, tokens)])
    actual = np.frombuffer(raw, dtype='<f4').reshape(len(tokens), -1)
    result = dict(method='HF FP32 Llama with groupwise W8A8 fake quantization',
        group_size=gs, tokens=tokens, torch=torch.__version__,
        transformers=__import__('transformers').__version__,
        c_vs_fp32=difference(actual, fp32), fake_quant_vs_fp32=difference(quantized, fp32),
        c_vs_fake_quant=difference(actual, quantized),
        quality_budget_changed=False,
        limitation='Fake-quantized BLAS is independent of C grouped-int32 accumulation; floating reductions can change activation rounding boundaries.')
    Path(args.output).write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps(result, indent=2))

if __name__ == '__main__':
    main()
