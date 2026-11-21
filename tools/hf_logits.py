"""Hugging Face FP32 logits at every position of a token sequence, as raw little-endian float32 (positions x vocab).

The reference side of `compare_engines.py --logits-a`: the same FP32 weights export.py converts from (BF16 sources are
widened), eager attention, one forward over the whole sequence. Runs on a compute node without network access.

    hf_logits.py --hf /path/to/snapshot --tokens-file prompt-2048.ids.txt --output hf-2048.f32 --threads 56
"""
import argparse
from pathlib import Path

import numpy as np
import torch
from transformers import AutoModelForCausalLM


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--hf', required=True, help='local Hugging Face snapshot directory')
    parser.add_argument('--tokens-file', required=True, help='whitespace-separated token ids')
    parser.add_argument('--output', required=True)
    parser.add_argument('--threads', type=int, default=4)
    args = parser.parse_args()
    torch.set_num_threads(args.threads)
    tokens = [int(t) for t in Path(args.tokens_file).read_text().split()]
    if not tokens:
        raise SystemExit('no tokens given')
    model = AutoModelForCausalLM.from_pretrained(args.hf, torch_dtype=torch.float32, local_files_only=True,
                                                 attn_implementation='eager', trust_remote_code=False).eval()
    with torch.inference_mode():
        logits = model(torch.tensor([tokens])).logits[0].numpy()
    Path(args.output).write_bytes(np.ascontiguousarray(logits, dtype='<f4').tobytes())
    print(f'{args.output}: {logits.shape[0]} positions x {logits.shape[1]} logits')


if __name__ == '__main__':
    main()
