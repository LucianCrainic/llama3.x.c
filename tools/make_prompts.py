"""Build the fixed-length campaign prompts deterministically.

The text is the opening paragraph of A Tale of Two Cities (Charles Dickens, public domain), repeated. For every
target length the longest prefix whose token count (BOS included, as the engine counts it) does not exceed the
target is written to prompt-<N>.txt, and the achieved count is recorded in prompts.json with a SHA-256 so a campaign
can check it later. `short.txt` is the 86-token prompt of the kernel campaigns. Tokenization uses the repository's own
tokenizer through the raw probe, so no ML dependency is needed.

    make_prompts.py --probe build/release/probe-fp32 --tokenizer tokenizer-v2.bin --vocab 128256 --out DIR
"""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import tempfile

PARAGRAPH = ('It was the best of times, it was the worst of times, it was the age of wisdom, it was the age of '
             'foolishness, it was the epoch of belief, it was the epoch of incredulity, it was the season of Light, '
             'it was the season of Darkness, it was the spring of hope, it was the winter of despair, we had '
             'everything before us, we had nothing before us, we were all going direct to Heaven, we were all going '
             'direct the other way - in short, the period was so far like the present period, that some of its '
             'noisiest authorities insisted on its being received, for good or for evil, in the superlative degree '
             'of comparison only. ')
SHORT_SENTENCE = 'Explain how a CPU executes instructions and uses memory. '
LENGTHS = (512, 2048, 8192, 32768)


class Counter:
    def __init__(self, probe, tokenizer, vocab):
        self.probe, self.tokenizer, self.vocab = probe, tokenizer, vocab
        self.dir = tempfile.TemporaryDirectory()
        self.path = Path(self.dir.name)/'text.txt'
        self.calls = 0

    def count(self, text):
        self.path.write_text(text, encoding='utf-8')
        self.calls += 1
        out = subprocess.run([str(self.probe), 'tokenize', str(self.tokenizer), str(self.vocab), f'@{self.path}'],
                             capture_output=True, text=True)
        if out.returncode:
            raise SystemExit(f'probe tokenize failed: {out.stderr}')
        return len(out.stdout.split())


def prefix_for(counter, source, target):
    """Longest prefix of `source` with at most `target` tokens, by binary search on its length."""
    if counter.count(source) < target:
        raise SystemExit(f'source text is too short for {target} tokens')
    low, high = 0, len(source)
    while low < high:
        mid = (low + high + 1) // 2
        if counter.count(source[:mid]) <= target:
            low = mid
        else:
            high = mid - 1
    return source[:low]


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--probe', required=True)
    parser.add_argument('--tokenizer', required=True)
    parser.add_argument('--vocab', required=True, type=int)
    parser.add_argument('--out', required=True)
    parser.add_argument('--lengths', default=','.join(map(str, LENGTHS)))
    args = parser.parse_args(argv)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    counter = Counter(args.probe, args.tokenizer, args.vocab)
    manifest = {}
    short = SHORT_SENTENCE * 5
    files = {'short': (short, out/'short.txt')}
    for target in [int(n) for n in args.lengths.split(',') if n and n != 'none']:
        repeats = target * 8 // len(PARAGRAPH) + 2  # real tokenizers need ~4-5 characters per token
        files[f'prompt-{target}'] = (prefix_for(counter, PARAGRAPH * repeats, target), out/f'prompt-{target}.txt')
    for name, (text, path) in files.items():
        path.write_text(text, encoding='utf-8')
        manifest[name] = dict(file=path.name, tokens=counter.count(text), characters=len(text),
                              sha256=hashlib.sha256(text.encode('utf-8')).hexdigest())
        if name != 'short':
            manifest[name]['target'] = int(name.split('-')[1])
    (out/'prompts.json').write_text(json.dumps(manifest, indent=2) + '\n')
    print(json.dumps(manifest, indent=2))
    return 0


if __name__ == '__main__':
    sys.exit(main())
