"""Compare the greedy generated text of two engine configurations on one prompt, and how close the reference was to a tie.

Each configuration runs `run` (or `runq`) with `-t 0` (greedy: the argmax token), `--no-text`, and LLAMA_TOKENS_OUT set, so
the engine records the token ids it forwarded before the first sample (the prompt, --warm-context ids first) and every
sampled id (the stop token included). Config A is the reference. The comparison is recorded, not gated: the exit status
is 0 whenever both runs complete.

Where the two generations diverge and --probe is given, the top-2 logit margin of config A at the divergence position
is read from the probe (tests/probe.c, `batch` mode, last row only) under config A's environment, on the token sequence
prompt + generated_a[:first_divergence]. A small margin means the reference was close to a tie there. The margin is
taken from the one-token-at-a-time path (LLAMA_PREFILL_BATCH=--probe-batch, default 1), which is the decode path.

    compare_generation.py --engine build/release/run --checkpoint fp32-v3.bin --tokenizer tokenizer-v2.bin \\
        --prompt-file prompt.txt --steps 128 --env-a LLAMA_BACKEND=omp --env-b LLAMA_BACKEND=teams \\
        --probe build/release/probe-fp32 --output result.json
"""
import argparse
import array
import json
import math
import os
from pathlib import Path
import shlex
import subprocess
import sys
import tempfile

sys.path.insert(0, str(Path(__file__).resolve().parent))
from compare_engines import parse_env, run_probe  # noqa: E402  (shared probe bridge and env parsing)


def run_engine(engine, checkpoint, tokenizer, prompt_file, steps, warm_context, engine_args, env, tokens_path):
    """One greedy generation. Returns the token lists the engine recorded: prompt (forwarded before the first sample)
    and generated (every sample). -n 0 lets the context length (--engine-args --context-length) bound the passes, so
    --warm-context is not limited by the default 4096-pass cap."""
    command = [str(engine), str(checkpoint), '-z', str(tokenizer), '--prompt-file', str(prompt_file), '-t', '0',
               '-n', '0', '--generation-length', str(steps), '--no-text']
    if warm_context:
        command += ['--warm-context', str(warm_context)]
    command += shlex.split(engine_args or '')
    result = subprocess.run(command, capture_output=True, env={**os.environ, **env, 'LLAMA_TOKENS_OUT': str(tokens_path)})
    if result.returncode != 0:
        raise SystemExit(f'engine failed (exit {result.returncode}) under {env}: {result.stderr.decode(errors="replace")}')
    try:
        data = json.loads(Path(tokens_path).read_text())
        prompt, generated = data['prompt'], data['generated']
    except (OSError, ValueError, KeyError) as error:
        raise SystemExit(f'engine did not write {tokens_path} under {env}: {error}')
    if not all(isinstance(t, int) for t in prompt + generated):
        raise SystemExit(f'LLAMA_TOKENS_OUT holds non-integer ids under {env}')
    return prompt, generated


def first_divergence(a, b):
    """Index of the first differing token; a strict prefix diverges where the shorter sequence ends. None when equal."""
    for index, (x, y) in enumerate(zip(a, b)):
        if x != y:
            return index
    return None if len(a) == len(b) else min(len(a), len(b))


def edit_distance(a, b):
    """Token-level Levenshtein distance (insertions, deletions and substitutions, each cost 1)."""
    previous = list(range(len(b) + 1))
    for i, x in enumerate(a, 1):
        current = [i]
        for j, y in enumerate(b, 1):
            current.append(min(previous[j] + 1, current[j - 1] + 1, previous[j - 1] + (x != y)))
        previous = current
    return previous[-1]


def top_two_margin(blob):
    """Top-2 logit margin (top1 - top2) and the two ids in descending logit order (ties: lower id first) of one row of
    little-endian float32 logits."""
    values = array.array('f')
    values.frombytes(blob)
    if sys.byteorder != 'little':
        values.byteswap()
    if len(values) < 2 or not all(math.isfinite(v) for v in values):
        raise SystemExit('probe returned fewer than two logits or non-finite logits')
    top1, top2 = sorted(range(len(values)), key=lambda i: (-values[i], i))[:2]
    return float(values[top1]) - float(values[top2]), [top1, top2]


def summary_line(result):
    if result['identical']:
        return f"identical: {result['generated_a']} generated tokens match"
    where = result['first_divergence']
    text = (f"diverge at token {where}: A={result['token_a']} B={result['token_b']}, "
            f"edit distance {result['edit_distance']}")
    if result['margin_a'] is not None:
        text += f", margin_a={result['margin_a']:.6g} (top2 {result['top2_a']})"
    return text


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--engine', required=True, help='run or runq')
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--tokenizer', required=True)
    parser.add_argument('--prompt-file', required=True, help='UTF-8 prompt text')
    parser.add_argument('--warm-context', type=int, default=0, metavar='N', help='forward N warm-up tokens first')
    parser.add_argument('--steps', type=int, default=128, help='maximum generated tokens (default 128)')
    parser.add_argument('--engine-args', default='', help='extra engine arguments, shell-quoted, for both runs')
    parser.add_argument('--env-a', action='append', default=[], metavar='KEY=VALUE', help='reference configuration')
    parser.add_argument('--env-b', action='append', default=[], metavar='KEY=VALUE', help='compared configuration')
    parser.add_argument('--probe', help='probe-fp32 or probe-int8: top-2 margin of A at the divergence')
    parser.add_argument('--probe-batch', type=int, default=1, metavar='M',
                        help='prefill microbatch for the probe (default 1: one token at a time, like decode)')
    parser.add_argument('--output', help='write the JSON result here as well as to stdout')
    args = parser.parse_args(argv)
    if args.steps < 1 or args.warm_context < 0 or args.probe_batch < 1:
        raise SystemExit('--steps and --probe-batch must be positive, --warm-context nonnegative')
    env_a, env_b = parse_env(args.env_a), parse_env(args.env_b)
    with tempfile.TemporaryDirectory() as temp:
        runs = []
        for name, env in (('a', env_a), ('b', env_b)):
            runs.append(run_engine(args.engine, args.checkpoint, args.tokenizer, args.prompt_file, args.steps,
                                   args.warm_context, args.engine_args, env, Path(temp)/f'tokens-{name}.json'))
    (prompt_a, generated_a), (prompt_b, generated_b) = runs
    divergence = first_divergence(generated_a, generated_b)
    result = dict(identical=divergence is None, first_divergence=divergence,
                  generated_a=len(generated_a), generated_b=len(generated_b),
                  edit_distance=edit_distance(generated_a, generated_b),
                  prompt_identical=prompt_a == prompt_b, prompt_tokens=len(prompt_a),
                  token_a=None, token_b=None, margin_a=None, top2_a=None,
                  engine=str(args.engine), checkpoint=str(args.checkpoint), warm_context=args.warm_context,
                  steps=args.steps, env_a=env_a, env_b=env_b, probe=str(args.probe) if args.probe else None)
    if divergence is not None:
        result['token_a'] = generated_a[divergence] if divergence < len(generated_a) else None
        result['token_b'] = generated_b[divergence] if divergence < len(generated_b) else None
    if args.probe and divergence is not None:
        # Logits that predict generated_a[divergence]: the sequence the engine forwarded up to that position.
        blob = run_probe(args.probe, args.checkpoint, prompt_a + generated_a[:divergence], env_a,
                         batch=args.probe_batch, last_only=True)
        result['margin_a'], result['top2_a'] = top_two_margin(blob)
    print(summary_line(result), file=sys.stderr)
    out = json.dumps(result, indent=2) + '\n'
    if args.output:
        Path(args.output).write_text(out)
    print(out, end='')
    return 0


if __name__ == '__main__':
    sys.exit(main())
