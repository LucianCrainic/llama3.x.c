"""Tests of the 70B cross-engine harness: tools/xengine70 (prompts, summarizer, native C driver, repacker syntax) and the syntax of
scripts/xengine70.sbatch. Synthetic fixtures only; no cluster, no 70B model, no numpy."""
import hashlib
import json
import os
from pathlib import Path
import py_compile
import subprocess
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'tools/xengine70'))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import summarize  # noqa: E402
from fixtures import TP_CONFIG, create  # noqa: E402

BUILD = Path(sys.argv.pop(1)).resolve() if len(sys.argv) > 1 and not sys.argv[1].startswith('-') else None
PROMPTS = ROOT/'tools/xengine70/prompts'
# docs/xengine-70b.md: the request token IDs of the comparison, 547 and 2,113 of them.
SHA256 = {'short': '48667d28324cb4927d4d2a6b5411d4c3a717b8a028c48e19645e6ec6748b3d24',
          'long': '412858a5ba9116f557bdafd6289c47cff874f4d0f400edce5724e6fe2c9a092d'}
LENGTH = {'short': 547, 'long': 2113}


def native_rows(short=547, long=2113, generation=128):
    rows = []
    for name, length in (('short', short), ('long', long)):
        for repeat in range(4):
            rows.append({'workload': name, 'repeat': repeat, 'warmup': repeat == 0, 'prompt_tokens': length,
                         'generated_tokens': generation, 'ttft_s': 10.0 + repeat, 'decode_s': 50.0 - repeat,
                         'token_ids': list(range(generation))})
    return rows


class PromptTests(unittest.TestCase):
    def test_prompt_fixtures_are_the_documented_requests(self):
        for name in ('short', 'long'):
            path = PROMPTS/f'{name}.ids.json'
            tokens = json.loads(path.read_text())
            self.assertEqual(len(tokens), LENGTH[name])
            self.assertEqual(tokens[0], 128000)  # BOS of Llama 3
            self.assertTrue(all(isinstance(t, int) and 0 <= t < 128256 for t in tokens))
            self.assertEqual(hashlib.sha256(path.read_bytes()).hexdigest(), SHA256[name])


class SummarizeTests(unittest.TestCase):
    def native(self, rows):
        directory = Path(self.temp.name)/'c-fp32'
        directory.mkdir(exist_ok=True)
        (directory/'runs.jsonl').write_text('\n'.join(json.dumps(row) for row in rows) + '\n')
        return directory

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()

    def tearDown(self):
        self.temp.cleanup()

    def test_native_rates_use_127_steps_after_the_warm_up_request(self):
        rows = list(summarize.summarize(self.native(native_rows())))
        self.assertEqual([r['workload'] for r in rows], ['short', 'long'])
        self.assertEqual(rows[0]['prompt_tokens'], 547)
        self.assertEqual(rows[0]['ttft_s'], [11.0, 12.0, 13.0])  # the warm-up repeat (0) is dropped
        self.assertEqual(rows[0]['decode_tps'], [127 / 49.0, 127 / 48.0, 127 / 47.0])
        self.assertEqual(summarize.med_range([1, 2, 4]), '2.00 [1.00–4.00]')

    def test_wrong_request_sizes_are_refused(self):
        with self.assertRaisesRegex(ValueError, 'wrong native workload size'):
            list(summarize.summarize(self.native(native_rows(short=546))))
        rows = native_rows()
        rows[0]['warmup'] = False  # no warm-up request left
        with self.assertRaisesRegex(ValueError, 'one warm-up and three measured'):
            list(summarize.summarize(self.native(rows)))

    def test_vllm_rows_use_the_token_arrivals(self):
        directory = Path(self.temp.name)/'vllm-fp32'
        directory.mkdir()
        for name, length in LENGTH.items():
            lines = []
            for repeat in range(4):
                arrivals = [100.0 + 0.5 * i for i in range(128)]
                lines.append(json.dumps({'ok': True, 'warmup': repeat == 0, 'usage_prompt_tokens': length,
                                         'token_arrivals': arrivals, 't_first_token': 5.0, 'token_ids': list(range(128))}))
            (directory/f'{name}.jsonl').write_text('\n'.join(lines) + '\n')
        rows = list(summarize.summarize(directory))
        self.assertEqual(rows[0]['decode_tps'], [127 / 63.5] * 3)
        self.assertEqual(rows[1]['prompt_tokens'], 2113)


class DriverTests(unittest.TestCase):
    """The native driver on a tiny two-layer model: four requests per workload, the warm-up first, identical greedy output
    in every repeat. The request ids are the whitespace-delimited form of the JSON files (docs/xengine-70b.md)."""
    @classmethod
    def setUpClass(cls):
        cls.temp = tempfile.TemporaryDirectory()
        root = Path(cls.temp.name)
        cls.fixture = create(root/'model', shared=False, config=TP_CONFIG, gs=4, seq_len=512)
        (root/'short.txt').write_text(' '.join(str(65 + i % 20) for i in range(12)) + '\n')
        (root/'long.txt').write_text(' '.join(str(65 + (i * 7) % 20) for i in range(30)) + '\n')

    @classmethod
    def tearDownClass(cls):
        cls.temp.cleanup()

    def run_driver(self, binary, model, sockets):
        root = Path(self.temp.name)
        env = {k: v for k, v in os.environ.items() if not k.startswith('LLAMA_')}
        env.update(LLAMA_THREADS='3', LLAMA_PREFILL_BATCH='4', LLAMA_KERNELS='scalar', LLAMA_SOCKETS=str(sockets),
                   LLAMA_TOPOLOGY=f'virtual:{sockets}x3')
        return subprocess.run([str(BUILD/binary), str(self.fixture/model), str(root/'short.txt'), str(root/'long.txt'), '512', '4'],
                              capture_output=True, text=True, env=env, timeout=120)

    def test_driver_times_four_requests_per_workload(self):
        if BUILD is None or not all((BUILD/b).exists() for b in ('xengine70-fp32', 'xengine70-q8')):
            self.skipTest('the xengine70 drivers were not built')
        for binary, model in (('xengine70-fp32', 'fp32.bin'), ('xengine70-q8', 'int8.bin')):
            for sockets in (1, 2):
                with self.subTest(binary=binary, sockets=sockets):
                    result = self.run_driver(binary, model, sockets)
                    # A virtual topology cannot query where pages live: the final check may refuse, after every request ran.
                    if result.returncode:
                        self.assertIn('Active KV page verification failed', result.stderr)
                    rows = [json.loads(line) for line in result.stdout.splitlines()]
                    self.assertEqual([(r['workload'], r['repeat'], r['warmup']) for r in rows],
                                     [(w, i, i == 0) for w in ('short', 'long') for i in range(4)])
                    self.assertEqual({(r['workload'], r['prompt_tokens'], r['generated_tokens']) for r in rows},
                                     {('short', 12, 128), ('long', 30, 128)})
                    for workload in ('short', 'long'):
                        traces = {tuple(r['token_ids']) for r in rows if r['workload'] == workload}
                        self.assertEqual(len(traces), 1, 'greedy output differs between repeats')
                    self.assertTrue(all(r['ttft_s'] > 0 and r['decode_s'] > 0 for r in rows))

    def test_driver_refuses_a_request_that_does_not_fit_the_context(self):
        if BUILD is None or not (BUILD/'xengine70-fp32').exists():
            self.skipTest('the xengine70 drivers were not built')
        root = Path(self.temp.name)
        env = {k: v for k, v in os.environ.items() if not k.startswith('LLAMA_')}
        env.update(LLAMA_THREADS='3', LLAMA_PREFILL_BATCH='4', LLAMA_SOCKETS='1')
        (root/'big.txt').write_text(' '.join(['65'] * 400) + '\n')  # 400 + 128 generated > 512
        result = subprocess.run([str(BUILD/'xengine70-fp32'), str(self.fixture/'fp32.bin'), str(root/'big.txt'),
                                 str(root/'long.txt'), '512', '4'], capture_output=True, text=True, env=env, timeout=120)
        self.assertEqual(result.returncode, 2)


class HarnessSyntaxTests(unittest.TestCase):
    def test_script_and_repacker_are_well_formed(self):
        script = ROOT/'scripts/xengine70.sbatch'
        self.assertEqual(subprocess.run(['bash', '-n', str(script)], capture_output=True).returncode, 0)
        text = script.read_text()
        for name in ('LLAMA_KERNELS', 'LLAMA_ATTN_DECODE', 'LLAMA_KV_LAYOUT', 'LLAMA_THP'):
            # the engine defaults are not forced: each knob is only exported from an option, or from LEGACY=1
            self.assertNotIn(f'export {name}=', text.replace(f'export {name}=$', ''))
        with tempfile.TemporaryDirectory() as temp:
            py_compile.compile(str(ROOT/'tools/xengine70/repack_q8.py'), cfile=str(Path(temp)/'repack.pyc'), doraise=True)


if __name__ == '__main__':
    unittest.main()
