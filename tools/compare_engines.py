"""Gate C1 batched logits against token-at-a-time logits, bit for bit.

Uses binary C probes and only the Python standard library. Full logits are spooled
to temporary files (TMPDIR controls their location); comparison uses one row of
memory per engine. Run real-model comparisons on allocated compute resources.
"""
import argparse
import array
import json
import math
import os
from pathlib import Path
import subprocess
import tempfile


def compare(reference, candidate, rows):
    reference.seek(0, 2)
    size = reference.tell()
    candidate.seek(0, 2)
    if size == 0 or size != candidate.tell() or size % (4 * rows):
        raise ValueError('probe output sizes differ or do not contain complete FP32 rows')
    reference.seek(0)
    candidate.seek(0)
    width = size // rows
    bitwise, finite, max_abs, argmax_matches = True, True, 0.0, 0
    for _ in range(rows):
        left, right = reference.read(width), candidate.read(width)
        bitwise &= left == right
        a, b = array.array('f'), array.array('f')
        a.frombytes(left)
        b.frombytes(right)
        row_finite = all(math.isfinite(x) for x in a) and all(math.isfinite(x) for x in b)
        finite &= row_finite
        if row_finite:
            if left != right:
                max_abs = max(max_abs, max(abs(x-y) for x, y in zip(a, b)))
            argmax_matches += max(range(len(a)), key=a.__getitem__) == max(range(len(b)), key=b.__getitem__)
    return dict(passed=bitwise and finite, bitwise=bitwise, finite=finite,
                max_abs=max_abs if finite else None, argmax_agreement=argmax_matches/rows,
                positions=rows, vocab_size=width//4)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--probe', required=True, help='candidate probe-fp32 or probe-int8')
    parser.add_argument('--reference-probe', help='optional baseline probe from a separate build')
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--tokens', required=True, help='comma-separated token IDs, including BOS if desired')
    parser.add_argument('--batch', type=int, default=32)
    parser.add_argument('--output', required=True, help='JSON gate result')
    args = parser.parse_args()
    try:
        tokens = [int(token) for token in args.tokens.split(',')]
    except ValueError:
        parser.error('--tokens must be comma-separated integers')
    if args.batch < 1 or args.batch > 2147483647 or any(t < 0 or t > 2147483647 for t in tokens):
        parser.error('batch must be in 1..INT_MAX and tokens in 0..INT_MAX')
    probe = str(Path(args.probe).resolve())
    reference_probe = str(Path(args.reference_probe or args.probe).resolve())
    checkpoint = str(Path(args.checkpoint).resolve())
    output = Path(args.output).resolve()
    for input_path in (Path(probe), Path(reference_probe), Path(checkpoint)):
        if output == input_path or (output.exists() and output.samefile(input_path)):
            parser.error('--output must not overwrite a probe or checkpoint')
    reference_command = [reference_probe, checkpoint, *map(str, tokens)]
    candidate_command = [probe, 'batch', checkpoint, *map(str, tokens)]
    env = {**os.environ, 'LLAMA_PREFILL_LOGITS': 'all'}
    with tempfile.TemporaryFile() as reference, tempfile.TemporaryFile() as candidate:
        subprocess.run(reference_command, env={**env, 'LLAMA_PREFILL_BATCH': '1'}, stdout=reference, check=True)
        subprocess.run(candidate_command, env={**env, 'LLAMA_PREFILL_BATCH': str(args.batch)}, stdout=candidate, check=True)
        result = compare(reference, candidate, len(tokens))
    result.update(checkpoint=checkpoint, probe=probe, reference_probe=reference_probe,
                  batch=args.batch, tokens=tokens,
                  kernels=os.environ.get('LLAMA_KERNELS', 'auto'),
                  q8_activations=os.environ.get('LLAMA_Q8_ACTIVATIONS', 'int8'))
    text = json.dumps(result, indent=2, allow_nan=False) + '\n'
    output.write_text(text)
    print(text, end='')
    return 0 if result['passed'] else 1


if __name__ == '__main__':
    raise SystemExit(main())
