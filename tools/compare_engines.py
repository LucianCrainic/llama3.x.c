"""Compare two engine configurations on the same checkpoint and tokens, at every position.

Runs the raw-logits probe (tests/probe.c) twice, once under each environment, and reports per position the
maximum absolute difference, the RMS difference, the maximum relative difference and whether the argmax agrees.
Relative differences use max(|reference logit|, 1) as the denominator, so near-zero logits do not dominate.
Config A is the reference. The exit status is non-zero when any position is over budget or its argmax differs.

W8A8 requantizes its activations, so any change in FP32 operation order (another ISA, a split over two sockets) can flip
a rounding decision and move the logits by a position-independent noise floor (rms ~0.06, measured scalar against
AVX-512 on 1B and 8B) that no per-position 1e-3 budget survives. --rms-max and/or --argmax-min switch to a statistical
gate for such comparisons: passed when every logit is finite, max |delta| <= --max-abs, rms <= --rms-max and the argmax
agreement over all positions >= --argmax-min (positions are then not judged one by one).

    compare_engines.py --probe build/release/probe-fp32 --checkpoint fp32-v3.bin --tokens 1,5,9 \\
        --env-b LLAMA_BACKEND=teams --env-b LLAMA_THREADS=4 --output compare.json
"""
import argparse
import array
import json
import math
import os
from pathlib import Path
import subprocess
import sys

try:
    import numpy as np
except ImportError:  # the tests run without ML dependencies
    np = None


def parse_env(items):
    env = {}
    for item in items or []:
        key, sep, value = item.partition('=')
        if not sep or not key:
            raise SystemExit(f'invalid --env value {item!r}: expected KEY=VALUE')
        env[key] = value
    return env


def run_probe(probe, checkpoint, tokens, env, batch=None, last_only=False):
    """Raw logits at every position. batch=M runs the batched prefill path (C1) with LLAMA_PREFILL_BATCH=M; with
    last_only the batched path returns just the final position's row, which keeps long prompts affordable."""
    command = [str(probe)] + (['batch'] if batch else []) + [str(checkpoint), *map(str, tokens)]
    if batch:
        env = {**env, 'LLAMA_PREFILL_BATCH': str(batch), 'LLAMA_PREFILL_LOGITS': 'last' if last_only else 'all'}
    result = subprocess.run(command, capture_output=True, env={**os.environ, **env})
    if result.returncode != 0:
        raise SystemExit(f'probe failed (exit {result.returncode}) under {env}: {result.stderr.decode(errors="replace")}')
    rows = 1 if batch and last_only else len(tokens)
    if not result.stdout or len(result.stdout) % (rows * 4):
        raise SystemExit(f'probe wrote {len(result.stdout)} bytes, not a multiple of {rows} float32 rows')
    return result.stdout


def positions(blob_a, blob_b, count):
    vocab = len(blob_a) // 4 // count
    if np is not None:
        a = np.frombuffer(blob_a, dtype='<f4').reshape(count, vocab).astype(np.float64)
        b = np.frombuffer(blob_b, dtype='<f4').reshape(count, vocab).astype(np.float64)
        for i in range(count):
            delta = np.abs(b[i] - a[i])
            yield dict(bitwise=bool(blob_a[i * vocab * 4:(i + 1) * vocab * 4] == blob_b[i * vocab * 4:(i + 1) * vocab * 4]),
                       max_abs=float(delta.max()), rms=float(np.sqrt(np.mean(delta ** 2))),
                       max_rel=float((delta / np.maximum(np.abs(a[i]), 1.0)).max()),
                       argmax_match=bool(a[i].argmax() == b[i].argmax()), finite=bool(np.isfinite(b[i]).all()))
        return
    fa, fb = array.array('f'), array.array('f')
    fa.frombytes(blob_a)
    fb.frombytes(blob_b)
    if sys.byteorder != 'little':
        fa.byteswap(), fb.byteswap()
    for i in range(count):
        row_a, row_b = fa[i * vocab:(i + 1) * vocab], fb[i * vocab:(i + 1) * vocab]
        worst = rel = squares = 0.0
        finite = True
        for x, y in zip(row_a, row_b):
            d = abs(y - x)
            if not math.isfinite(y):
                finite = False
            worst = max(worst, d)
            rel = max(rel, d / max(abs(x), 1.0))
            squares += d * d
        yield dict(bitwise=blob_a[i * vocab * 4:(i + 1) * vocab * 4] == blob_b[i * vocab * 4:(i + 1) * vocab * 4],
                   max_abs=worst, rms=math.sqrt(squares / vocab), max_rel=rel,
                   argmax_match=max(range(vocab), key=row_a.__getitem__) == max(range(vocab), key=row_b.__getitem__),
                   finite=finite)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--probe', required=True, help='probe-fp32 or probe-int8')
    parser.add_argument('--checkpoint', required=True)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--tokens', help='comma-separated token ids')
    group.add_argument('--tokens-file', help='whitespace-separated token ids')
    parser.add_argument('--env-a', action='append', default=[], metavar='KEY=VALUE')
    parser.add_argument('--env-b', action='append', default=[], metavar='KEY=VALUE')
    parser.add_argument('--batch-a', type=int, metavar='M', help='run config A through the batched prefill path, microbatch M')
    parser.add_argument('--batch-b', type=int, metavar='M', help='run config B through the batched prefill path, microbatch M')
    parser.add_argument('--max-abs', type=float, default=1e-3, help='budget for max |delta logit| (default 1e-3)')
    parser.add_argument('--last-only', action='store_true',
                        help='compare only the final position (config B runs the batched path with LLAMA_PREFILL_LOGITS=last)')
    parser.add_argument('--rms-max', type=float, help='statistical gate: budget for the rms of the logit differences')
    parser.add_argument('--argmax-min', type=float,
                        help='statistical gate: minimum fraction of positions whose argmax agrees (default 1.0)')
    parser.add_argument('--bitwise', action='store_true', help='require bit-identical logits instead of the --max-abs budget')
    parser.add_argument('--output', help='write the JSON result here as well as to stdout')
    args = parser.parse_args(argv)
    text = args.tokens.replace(',', ' ') if args.tokens else Path(args.tokens_file).read_text()
    tokens = [int(t) for t in text.split()]
    if not tokens:
        raise SystemExit('no tokens given')
    env_a, env_b = parse_env(args.env_a), parse_env(args.env_b)
    blob_a = run_probe(args.probe, args.checkpoint, tokens, env_a, args.batch_a, args.last_only)
    blob_b = run_probe(args.probe, args.checkpoint, tokens, env_b, args.batch_b, args.last_only)
    compared = len(tokens)
    if args.last_only:
        row_bytes = len(blob_a) // (len(tokens) if not args.batch_a else 1)
        blob_a, compared = blob_a[-row_bytes:], 1
        if args.batch_a:
            raise SystemExit('--last-only needs config A to be the per-token path')
    if len(blob_a) != len(blob_b):
        raise SystemExit('the two configurations produced logits of different sizes')
    rows = []
    statistical = args.rms_max is not None or args.argmax_min is not None
    if statistical and args.bitwise:
        raise SystemExit('--bitwise cannot be combined with --rms-max or --argmax-min')
    for index, row in enumerate(positions(blob_a, blob_b, compared)):
        position = len(tokens) - 1 if args.last_only else index
        row.update(position=position, token=tokens[position])
        row['passed'] = row['finite'] and (statistical or row['argmax_match']) and (
            row['bitwise'] if args.bitwise else row['max_abs'] <= args.max_abs)
        rows.append(row)
    summary = dict(max_abs=max(r['max_abs'] for r in rows), rms=math.sqrt(sum(r['rms'] ** 2 for r in rows) / len(rows)),
                   max_rel=max(r['max_rel'] for r in rows),
                   argmax_agreement=sum(r['argmax_match'] for r in rows) / len(rows),
                   bitwise=all(r['bitwise'] for r in rows))
    argmax_min = 1.0 if args.argmax_min is None else args.argmax_min
    passed = all(r['passed'] for r in rows)
    if statistical:
        passed = passed and summary['argmax_agreement'] >= argmax_min and (
            args.rms_max is None or summary['rms'] <= args.rms_max)
    result = dict(passed=passed, probe=str(args.probe), checkpoint=str(args.checkpoint),
                  env_a=env_a, env_b=env_b, batch_a=args.batch_a, batch_b=args.batch_b, tokens=len(tokens),
                  budget=dict(max_abs=0.0 if args.bitwise else args.max_abs, argmax_agreement=argmax_min,
                              bitwise=args.bitwise, rms_max=args.rms_max, statistical=statistical),
                  summary=summary, comparisons=rows)
    out = json.dumps(result, indent=2) + '\n'
    if args.output:
        Path(args.output).write_text(out)
    print(out, end='')
    return 0 if result['passed'] else 1


if __name__ == '__main__':
    sys.exit(main())
