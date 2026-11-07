"""Two-socket tensor parallelism (B5, B6) on the virtual topology: LLAMA_SOCKETS=2 against the one-socket teams
backend, on fixtures that can be split (two KV heads)."""
import array
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

from fixtures import CONFIG, TP_CONFIG, TP_ODD_CONFIG, create

BUILD = Path(sys.argv.pop(1)).resolve()
TOKENS = [65, 66, 67, 68, 69, 70, 71]
ABS, REL = 2e-6, 2e-5   # the fixture tolerance of test_runtime.py


def environment(sockets, threads=3, **extra):
    env = {**os.environ, 'LLAMA_BACKEND': 'teams', 'LLAMA_THREADS': str(threads), 'LLAMA_TP_CHECK': '1', **extra}
    env.pop('LLAMA_SOCKETS', None)
    env.pop('LLAMA_TOPOLOGY', None)
    if sockets == 2:
        env.update(LLAMA_SOCKETS='2', LLAMA_TOPOLOGY=f'virtual:2x{threads}')
    return env


class TensorParallelTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temp = tempfile.TemporaryDirectory()
        root = Path(cls.temp.name)
        cls.fixtures = {
            'shared': create(root/'shared', config=TP_CONFIG, gs=4),
            'unshared': create(root/'unshared', shared=False, config=TP_CONFIG, gs=4),
            'odd': create(root/'odd', config=TP_ODD_CONFIG, gs=4),
        }
        cls.one_kv = create(root/'one-kv', shared=True)

    @classmethod
    def tearDownClass(cls):
        cls.temp.cleanup()

    def probe(self, precision, fixture, mode, sockets, threads=3, tokens=TOKENS, **extra):
        model = self.fixtures[fixture]/('fp32.bin' if precision == 'fp32' else 'int8.bin')
        command = [str(BUILD/f'probe-{precision}')] + ([mode] if mode else []) + [str(model), *map(str, tokens)]
        if mode == 'generate':
            command = [str(BUILD/f'probe-{precision}'), 'generate', str(model), '16', *map(str, tokens)]
        result = subprocess.run(command, capture_output=True, timeout=120, env=environment(sockets, threads, **extra))
        self.assertEqual(result.returncode, 0, result.stderr.decode())
        return result.stdout

    def logits(self, *args, **kwargs):
        out = array.array('f')
        out.frombytes(self.probe(*args, **kwargs))
        return out

    def assert_agree(self, one, two, vocab):
        self.assertEqual(len(one), len(two))
        for i, (a, b) in enumerate(zip(one, two)):
            self.assertLessEqual(abs(a - b), ABS + REL * abs(a), f'logit {i}: {a} vs {b}')
        for row in range(len(one) // vocab):
            r1, r2 = one[row*vocab:(row+1)*vocab], two[row*vocab:(row+1)*vocab]
            top = max(r1)
            # argmax agreement, allowing a tie within the tolerance
            self.assertGreaterEqual(r2[r1.index(top)], max(r2) - (ABS + REL * abs(top)))

    def test_fp32_logits_agree_with_one_socket(self):
        for name, vocab in [('shared', TP_CONFIG[5]), ('unshared', TP_CONFIG[5]), ('odd', TP_ODD_CONFIG[5])]:
            one = self.logits('fp32', name, None, 1)
            for threads in (1, 3, 7):
                with self.subTest(fixture=name, threads=threads):
                    self.assert_agree(one, self.logits('fp32', name, None, 2, threads), vocab)

    def test_fp32_batched_prefill_entry_agrees(self):
        one = self.logits('fp32', 'shared', 'batch', 1)
        for batch in (1, 3):
            with self.subTest(batch=batch):
                two = self.logits('fp32', 'shared', 'batch', 2, LLAMA_PREFILL_BATCH=str(batch))
                self.assert_agree(one, two, TP_CONFIG[5])
        all_rows = self.logits('fp32', 'odd', 'batch', 2, LLAMA_PREFILL_BATCH='4', LLAMA_PREFILL_LOGITS='all')
        self.assertEqual(len(all_rows), len(TOKENS) * TP_ODD_CONFIG[5])
        self.assert_agree(self.logits('fp32', 'odd', None, 1), all_rows, TP_ODD_CONFIG[5])

    def test_greedy_generation_is_identical(self):
        one = self.probe('fp32', 'unshared', 'generate', 1)
        self.assertGreaterEqual(len(one.split()), 8)
        for threads in (1, 3):
            self.assertEqual(one, self.probe('fp32', 'unshared', 'generate', 2, threads))

    def test_cli_runs_chat_and_reports(self):
        fixture = self.fixtures['unshared']
        for sockets in (1, 2):
            report = Path(self.temp.name)/f'report{sockets}.json'
            result = subprocess.run([str(BUILD/'run'), str(fixture/'fp32.bin'), '-z', str(fixture/'tokenizer.bin'), '-i',
                                     'A', '--seed', '123', '-t', '0', '--json', str(report)],
                                    capture_output=True, text=True, timeout=60, env=environment(sockets))
            self.assertEqual(result.returncode, 0, result.stderr)
            if sockets == 1:
                expected = result.stdout
            else:
                self.assertEqual(result.stdout, expected)
                self.assertIn('"status": "completed"', report.read_text())

    def test_incompatible_configurations_are_rejected(self):
        model = self.one_kv/'fp32.bin'
        cases = [
            (model, environment(2), 'n_kv_heads=1'),
            (self.fixtures['shared']/'fp32.bin', {**environment(2), 'LLAMA_BACKEND': 'omp'}, 'requires LLAMA_BACKEND=teams'),
            (self.fixtures['shared']/'fp32.bin', {**environment(2), 'LLAMA_SOCKETS': '3'}, 'LLAMA_SOCKETS must be 1 or 2'),
            (self.fixtures['shared']/'fp32.bin', {**environment(2), 'LLAMA_TOPOLOGY': 'virtual:1x3'}, 'allowed CPUs in only 1'),
            (self.fixtures['shared']/'fp32.bin', {**environment(2), 'LLAMA_PLACEMENT': 'hbm'}, 'virtual'),
        ]
        for model, env, text in cases:
            with self.subTest(text=text):
                result = subprocess.run([str(BUILD/'probe-fp32'), str(model), '65'], capture_output=True, timeout=60, env=env)
                self.assertEqual(result.returncode, 1, result.stderr.decode())
                self.assertIn(text, result.stderr.decode())


if __name__ == '__main__':
    unittest.main()
