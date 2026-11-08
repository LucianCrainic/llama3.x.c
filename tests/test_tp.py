"""Two-socket tensor parallelism (B5, B6) on the virtual topology: LLAMA_SOCKETS=2 against the one-socket teams
backend, on fixtures that can be split (two KV heads)."""
import array
import json
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
        cls.fixtures['gs16'] = create(root/'gs16', config=TP_CONFIG, gs=16)
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

    def test_q8_logits_agree_with_one_socket(self):
        for activations in ('int8', 'fp32'):  # W8A8 and W8A32
            for name, vocab in [('shared', TP_CONFIG[5]), ('unshared', TP_CONFIG[5]), ('odd', TP_ODD_CONFIG[5])]:
                one = self.logits('int8', name, None, 1, LLAMA_Q8_ACTIVATIONS=activations)
                for threads in (1, 3, 7):
                    with self.subTest(activations=activations, fixture=name, threads=threads):
                        two = self.logits('int8', name, None, 2, threads, LLAMA_Q8_ACTIVATIONS=activations)
                        self.assert_agree(one, two, vocab)

    def test_q8_batched_entry_and_generation(self):
        for activations in ('int8', 'fp32'):
            with self.subTest(activations=activations):
                one = self.logits('int8', 'shared', 'batch', 1, LLAMA_Q8_ACTIVATIONS=activations)
                two = self.logits('int8', 'shared', 'batch', 2, LLAMA_PREFILL_BATCH='3', LLAMA_Q8_ACTIVATIONS=activations)
                self.assert_agree(one, two, TP_CONFIG[5])
                tokens = self.probe('int8', 'unshared', 'generate', 1, LLAMA_Q8_ACTIVATIONS=activations)
                self.assertGreaterEqual(len(tokens.split()), 8)
                self.assertEqual(tokens, self.probe('int8', 'unshared', 'generate', 2, LLAMA_Q8_ACTIVATIONS=activations))

    def test_fp32_batched_prefill_entry_agrees(self):
        one = self.logits('fp32', 'shared', 'batch', 1)
        for batch in (1, 3):
            with self.subTest(batch=batch):
                two = self.logits('fp32', 'shared', 'batch', 2, LLAMA_PREFILL_BATCH=str(batch))
                self.assert_agree(one, two, TP_CONFIG[5])
        all_rows = self.logits('fp32', 'odd', 'batch', 2, LLAMA_PREFILL_BATCH='4', LLAMA_PREFILL_LOGITS='all')
        self.assertEqual(len(all_rows), len(TOKENS) * TP_ODD_CONFIG[5])
        self.assert_agree(self.logits('fp32', 'odd', None, 1), all_rows, TP_ODD_CONFIG[5])

    def test_batched_gemm_knobs_do_not_reach_the_two_socket_path(self):
        """C6's batched Q8 GEMM and C2's FP32 GEMM are one-socket paths: on two sockets the knob changes nothing (the
        forward stays per token, bitwise) and the report names no batched prefill kernel."""
        for precision, name in (('int8', 'int8.bin'), ('fp32', 'fp32.bin')):
            with self.subTest(precision=precision):
                base = self.probe(precision, 'shared', 'batch', 2, LLAMA_PREFILL_BATCH='4')
                forced = self.probe(precision, 'shared', 'batch', 2, LLAMA_PREFILL_BATCH='4', LLAMA_PREFILL_GEMM='on')
                self.assertEqual(base, forced)
                fixture = self.fixtures['shared']
                report = Path(self.temp.name)/f'gemm-{precision}.json'
                engine = 'runq' if precision == 'int8' else 'run'
                result = subprocess.run([str(BUILD/engine), str(fixture/name), '-z', str(fixture/'tokenizer.bin'), '-i', 'ABCDE',
                                         '-t', '0', '--json', str(report)], capture_output=True, text=True, timeout=60,
                                        env=environment(2, LLAMA_PREFILL_BATCH='4', LLAMA_PREFILL_GEMM='on'))
                self.assertEqual(result.returncode, 0, result.stderr)
                runtime = json.loads(report.read_text())['runtime']
                self.assertEqual(runtime['sockets'], 2)
                self.assertIsNone(runtime['kernels_prefill'])

    def test_fused_and_separate_reduction_are_bitwise_equal(self):
        """The fused form (projection and reduction in one team job) only removes a dispatch: it must not change a bit."""
        for precision, activations in (('fp32', None), ('int8', 'int8'), ('int8', 'fp32')):
            extra = {'LLAMA_Q8_ACTIVATIONS': activations} if activations else {}
            for threads in (1, 3, 7):
                with self.subTest(precision=precision, activations=activations, threads=threads):
                    fused = self.probe(precision, 'shared', None, 2, threads, LLAMA_TP_FUSED='1', **extra)
                    separate = self.probe(precision, 'shared', None, 2, threads, LLAMA_TP_FUSED='0', **extra)
                    self.assertEqual(fused, separate)

    def test_fused_knob_is_validated(self):
        model = self.fixtures['shared']/'fp32.bin'
        result = subprocess.run([str(BUILD/'probe-fp32'), str(model), '1'], capture_output=True, timeout=60,
                                env=environment(2, 3, LLAMA_TP_FUSED='2'))
        self.assertNotEqual(result.returncode, 0)
        self.assertIn(b'LLAMA_TP_FUSED must be 0 or 1', result.stderr)

    def test_gemv_scheduling_agrees_and_is_validated(self):
        """Dynamic and hybrid row claiming against the static split, FP32 and Q8, on one and on two sockets, within the
        fixture budget (an AVX-512 four-row group sums in a different order than a short tail group)."""
        for precision, vocab in (('fp32', TP_CONFIG[5]), ('int8', TP_CONFIG[5])):
            for sockets in (1, 2):
                for threads in (3, 7):
                    static = self.logits(precision, 'shared', None, sockets, threads, LLAMA_GEMV_SCHED='static')
                    for mode in ('dynamic', 'hybrid', 'auto'):
                        with self.subTest(precision=precision, sockets=sockets, threads=threads, mode=mode):
                            self.assert_agree(static, self.logits(precision, 'shared', None, sockets, threads,
                                                                  LLAMA_GEMV_SCHED=mode), vocab)
        model = self.fixtures['shared']/'fp32.bin'
        result = subprocess.run([str(BUILD/'probe-fp32'), str(model), '1'], capture_output=True, timeout=60,
                                env=environment(1, 3, LLAMA_GEMV_SCHED='guided'))
        self.assertNotEqual(result.returncode, 0)
        self.assertIn(b'LLAMA_GEMV_SCHED must be static, dynamic, hybrid or auto', result.stderr)

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
        cases.append((self.fixtures['gs16']/'int8.bin', environment(2), 'group size 16 does not divide'))
        # C5 and B5 meet here: BF16 prefill packs the whole matrices, which the two-socket forward does not hold.
        cases.append((self.fixtures['shared']/'fp32.bin', {**environment(2), 'LLAMA_PRECISION': 'bf16', 'LLAMA_PREFILL_BATCH': '4'},
                      'LLAMA_PRECISION=bf16 is not supported with LLAMA_SOCKETS=2'))
        for model, env, text in cases:
            with self.subTest(text=text):
                probe = 'probe-int8' if str(model).endswith('int8.bin') else 'probe-fp32'
                result = subprocess.run([str(BUILD/probe), str(model), '65'], capture_output=True, timeout=60, env=env)
                self.assertEqual(result.returncode, 1, result.stderr.decode())
                self.assertIn(text, result.stderr.decode())


if __name__ == '__main__':
    unittest.main()
