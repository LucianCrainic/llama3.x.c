"""Two-socket tensor parallelism (B5, B6) on the virtual topology: LLAMA_SOCKETS=2 against the one-socket teams
backend, on fixtures that can be split (two KV heads)."""
import array
import json
import os
import re
import struct
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

from fixtures import CONFIG, TP_CONFIG, TP_ODD_CONFIG, TP_WIDE_CONFIG, create, upgrade_v3

BUILD = Path(sys.argv.pop(1)).resolve()
TOKENS = [65, 66, 67, 68, 69, 70, 71]
ABS, REL = 2e-6, 2e-5   # the fixture tolerance of test_runtime.py


def environment(sockets, threads=3, **extra):
    env = {**os.environ, 'LLAMA_BACKEND': 'teams', 'LLAMA_THREADS': str(threads), 'LLAMA_TP_CHECK': '1', **extra}
    if 'LLAMA_TOPOLOGY' not in extra:
        env.pop('LLAMA_TOPOLOGY', None)
    env['LLAMA_SOCKETS'] = '1'  # explicit: unset means two sockets on a dual-socket HBM node
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
        cls.fixtures['wide'] = create(root/'wide', config=TP_WIDE_CONFIG, gs=64)
        cls.fixtures['wide-gs32'] = create(root/'wide-gs32', config=TP_WIDE_CONFIG, gs=32)
        # per-channel INT8 (version 4): one scale per output channel, the same weights as `wide`
        cls.fixtures['wide-pc'] = upgrade_v3(create(root/'wide-pc', config=TP_WIDE_CONFIG, gs=0), config=TP_WIDE_CONFIG,
                                             gs=0, int8_version=4)
        cls.one_kv = create(root/'one-kv', shared=True)

    @classmethod
    def tearDownClass(cls):
        cls.temp.cleanup()

    def probe(self, precision, fixture, mode, sockets, threads=3, tokens=TOKENS, binary=None, **extra):
        model = self.fixtures[fixture]/('fp32.bin' if precision == 'fp32' else 'int8.bin')
        exe = str(BUILD/(binary or f'probe-{precision}'))
        command = [exe] + ([mode] if mode else []) + [str(model), *map(str, tokens)]
        if mode == 'generate':
            command = [exe, 'generate', str(model), '16', *map(str, tokens)]
        result = subprocess.run(command, capture_output=True, timeout=120, env=environment(sockets, threads, **extra))
        self.assertEqual(result.returncode, 0, result.stderr.decode())
        return result.stdout

    def logits(self, *args, **kwargs):
        out = array.array('f')
        out.frombytes(self.probe(*args, **kwargs))
        return out

    def assert_agree(self, one, two, vocab, abs_tol=ABS, rel_tol=REL):
        self.assertEqual(len(one), len(two))
        for i, (a, b) in enumerate(zip(one, two)):
            self.assertLessEqual(abs(a - b), abs_tol + rel_tol * abs(a), f'logit {i}: {a} vs {b}')
        for row in range(len(one) // vocab):
            r1, r2 = one[row*vocab:(row+1)*vocab], two[row*vocab:(row+1)*vocab]
            top = max(r1)
            # argmax agreement, allowing a tie within the tolerance
            self.assertGreaterEqual(r2[r1.index(top)], max(r2) - (abs_tol + rel_tol * abs(top)))

    def assert_agree_but_flips(self, one, two, vocab, flips=1, flip_tol=1e-2):
        """assert_agree for W8A8 paths whose GEMM rescale rounds differently from the reference's: an activation that
        lands on an int8 rounding boundary can quantize one step apart, which moves that position's logits by up to about
        1e-3 and nothing else (the K/V it writes stay within the budget, so later positions agree). At most `flips` rows
        may exceed the elementwise budget, by at most flip_tol; every row keeps its argmax within the row's tolerance."""
        self.assertEqual(len(one), len(two))
        off = 0
        for row in range(len(one) // vocab):
            r1, r2 = one[row*vocab:(row+1)*vocab], two[row*vocab:(row+1)*vocab]
            worst = max(abs(a - b) - (ABS + REL * abs(a)) for a, b in zip(r1, r2))
            tol = ABS + REL * abs(max(r1))
            if worst > 0:
                off += 1
                self.assertLessEqual(worst, flip_tol, f'row {row}')
                tol = flip_tol
            self.assertGreaterEqual(r2[r1.index(max(r1))], max(r2) - tol, f'row {row}: argmax')
        self.assertLessEqual(off, flips, f'{off} rows beyond the budget')

    def test_tier_plans_preserve_logits(self):
        for sockets in (1, 2):
            for precision, activations, layout in (
                ('fp32', 'int8', 'head'), ('fp32', 'int8', 'rows'),
                ('int8', 'int8', 'head'), ('int8', 'fp32', 'rows'),
            ):
                knobs = dict(LLAMA_TOPOLOGY=f'virtual:{sockets}x3', LLAMA_KERNELS='scalar',
                             LLAMA_PREFILL_GEMM='on', LLAMA_PREFILL_BATCH='3', LLAMA_PREFILL_LOGITS='all',
                             LLAMA_Q8_ACTIVATIONS=activations, LLAMA_KV_LAYOUT=layout)
                for method in (None, 'batch'):
                    reference = self.probe(precision, 'unshared', method, sockets, **knobs)
                    for placement in ('hbm-only', 'ddr', 'hybrid'):
                        tier = {'LLAMA_PLACEMENT': placement}
                        if placement == 'hybrid':
                            tier.update(LLAMA_HYBRID_WEIGHTS_HBM_LAYERS='1', LLAMA_HYBRID_KV_HBM_LAYERS='1',
                                        LLAMA_HYBRID_CLASSIFIER='ddr')
                        with self.subTest(sockets=sockets, precision=precision, layout=layout, method=method,
                                          placement=placement):
                            self.assertEqual(reference, self.probe(precision, 'unshared', method, sockets,
                                                                   **(knobs | tier)))

    def test_tier_plans_preserve_packed_prefill(self):
        for precision, knobs in (('fp32', {'LLAMA_PRECISION': 'bf16'}), ('int8', {})):
            exe = f'probe-{precision}-amx-emul'
            common = dict(LLAMA_KERNELS='amx', LLAMA_PREFILL_BATCH='4', LLAMA_PREFILL_LOGITS='all', **knobs)
            reference = self.probe(precision, 'wide', 'batch', 2, binary=exe, **common)
            actual = self.probe(precision, 'wide', 'batch', 2, binary=exe, LLAMA_PLACEMENT='hybrid',
                                LLAMA_HYBRID_WEIGHTS_HBM_LAYERS='1', LLAMA_HYBRID_KV_HBM_LAYERS='1',
                                LLAMA_HYBRID_PACKED='ddr', **common)
            self.assertEqual(reference, actual)

    def test_planned_report_marks_prefill_and_decode_windows(self):
        data = self.report('run', 'fp32.bin', full=True, fixture='unshared',
                           LLAMA_PLACEMENT='hybrid', LLAMA_HYBRID_KV_HBM_LAYERS='1')
        phase = data['phase_epoch_ns']
        self.assertLess(phase['prefill_start'], phase['prefill_end'])
        self.assertLess(phase['prefill_end'], phase['decode_start'])
        self.assertLess(phase['decode_start'], phase['decode_end'])
        self.assertEqual(data['kv']['residency_scope'], 'active_prefix')
        self.assertTrue(data['placement_plan']['simulated'])

    def test_warm_up_stays_out_of_exchange_timing(self):
        """--warm-context runs untimed: with LLAMA_PROFILE=1 the exchange clock must not run through it. When only the
        phase profile was switched off, the first timed batch absorbed the whole warm-up (about 10x the cold value here)."""
        fixture = self.fixtures['wide']

        def exchange_ms_per_token(warm):
            command = [str(BUILD/'runq'), str(fixture/'int8.bin'), '-z', str(fixture/'tokenizer.bin'), '--benchmark',
                       '--no-text', '-t', '0', '-n', '0', '-i', 'A', '--warm-context', str(warm),
                       '--context-length', '64', '--generation-length', '2']
            result = subprocess.run(command, capture_output=True, text=True, timeout=30,
                                    env=environment(2, LLAMA_KERNELS='scalar', LLAMA_PROFILE='1',
                                                    LLAMA_PREFILL_BATCH='4'))
            self.assertEqual(result.returncode, 0, result.stderr)
            lines = [line for line in result.stderr.splitlines() if line.startswith('[INFO] tp prefill socket 0 (')]
            self.assertEqual(len(lines), 1, result.stderr)
            return sum(float(v) for v in re.findall(r' x_\w+=([0-9.]+)', lines[0]))

        cold = sorted(exchange_ms_per_token(0) for _ in range(3))[1]
        warm = sorted(exchange_ms_per_token(60) for _ in range(3))[1]
        # Each value is the median of three runs and printed to 1 us; measured here: cold <= 0.001, warm 0.000-0.001 with
        # the fix and 0.007-0.012 without it.
        self.assertLess(warm, cold + 0.004)

    def test_synthetic_kv_tracks_active_prefix_across_tiers(self):
        fixture = self.fixtures['unshared']
        report_path = Path(self.temp.name)/'synthetic-kv.json'
        trace_path = Path(self.temp.name)/'synthetic-kv-tokens.json'

        def run(mode, capacity, synthetic, layout='head', generation=4, fixture=fixture, sockets=2, prime=None):
            cmd = [str(BUILD/'runq'), str(fixture/'int8.bin'), '-z', str(fixture/'tokenizer.bin'),
                   '--benchmark', '--no-text', '-t', '0', '-n', '0', '-i', 'ABCDE', '--warm-context', '13',
                   '--context-length', str(capacity), '--generation-length', str(generation), '--json', str(report_path)]
            tier = {'LLAMA_HYBRID_KV_HBM_LAYERS': '1'} if mode == 'hybrid' else {}
            if prime is not None:
                tier['LLAMA_SYNTHETIC_KV_PRIME'] = prime
            return subprocess.run(cmd, capture_output=True, text=True, timeout=30,
                                  env=environment(sockets, LLAMA_KERNELS='scalar', LLAMA_PLACEMENT=mode,
                                                  LLAMA_TOPOLOGY=f'virtual:{sockets}x3', LLAMA_KV_LAYOUT=layout,
                                                  LLAMA_SYNTHETIC_KV=str(int(synthetic)),
                                                  LLAMA_TOKENS_OUT=str(trace_path), **tier))

        result = run('hybrid', 32, False)
        self.assertEqual(result.returncode, 0, result.stderr)
        real = json.loads(report_path.read_text())
        real_trace = json.loads(trace_path.read_text())
        if 'synthetic_kv' not in real['parameters']:
            self.skipTest('build with SYNTHETIC_KV=1')
        self.assertFalse(real['parameters']['synthetic_kv'])
        self.assertNotIn('synthetic_kv_prime', real['parameters'])
        self.assertNotIn('synthetic_kv', real_trace)
        self.assertEqual(len(real_trace['prompt']), 13 + real['tokens']['processed_prompt'])
        expected_trace = None
        for mode, capacity in (('hybrid', 32), ('ddr', 64), ('hbm-only', 64)):
            with self.subTest(mode=mode, capacity=capacity):
                result = run(mode, capacity, True)
                self.assertEqual(result.returncode, 0, result.stderr)
                data = json.loads(report_path.read_text())
                trace = json.loads(trace_path.read_text())
                self.assertTrue(data['parameters']['synthetic_kv'])
                self.assertEqual(data['parameters']['synthetic_kv_prime'], 0)
                self.assertIsNotNone(data['timing_seconds']['warm_context'])
                self.assertEqual(data['stop_reason'], 'generation_limit')
                self.assertEqual(data['tokens']['decode_steps'], 3)
                self.assertEqual(data['kv']['residency_scope'], 'active_prefix')
                dim, _, layers, heads, kv_heads, *_ = TP_CONFIG
                bytes_per_position = 2 * layers * kv_heads * (dim // heads) * 4  # both K and V, FP32
                after_prompt = 13 + data['tokens']['processed_prompt']
                self.assertEqual(data['kv']['reserved_bytes'], capacity * bytes_per_position)
                self.assertEqual(data['kv']['after_prompt']['active_bytes'], after_prompt * bytes_per_position)
                self.assertEqual(data['kv']['active_bytes'], (after_prompt + 3) * bytes_per_position)
                self.assertTrue(trace['synthetic_kv'])
                self.assertEqual(trace['synthetic_prefix_positions'], 13)
                self.assertEqual(trace['prompt'], real_trace['prompt'][13:])
                self.assertEqual(len(trace['generated']), 4)
                if expected_trace is None:
                    expected_trace = trace
                self.assertEqual(trace, expected_trace)

        for prime in ('0', '3', '13'):
            with self.subTest(prime=prime):
                result = run('hybrid', 32, True, prime=prime)
                self.assertEqual(result.returncode, 0, result.stderr)
                data = json.loads(report_path.read_text())
                self.assertEqual(data['parameters']['synthetic_kv_prime'], int(prime))
                self.assertEqual(data['tokens']['decode_steps'], 3)
                self.assertEqual(json.loads(trace_path.read_text()), expected_trace)
        for synthetic, prime in ((False, '0'), (True, ''), (True, '-1'), (True, '14'), (True, ' 1')):
            with self.subTest(synthetic=synthetic, invalid_prime=prime):
                result = run('hybrid', 32, synthetic, prime=prime)
                self.assertNotEqual(result.returncode, 0)
                self.assertIn('LLAMA_SYNTHETIC_KV_PRIME', result.stderr)

        result = run('hybrid', 32, True, layout='rows')
        self.assertNotEqual(result.returncode, 0)
        self.assertIn('requires LLAMA_KV_LAYOUT=head', result.stderr)
        result = run('hybrid', 32, True, generation=32)
        self.assertNotEqual(result.returncode, 0)
        self.assertIn('complete generation', result.stderr)

        stop_fixture = create(Path(self.temp.name)/'synthetic-stop', shared=False, stop='first')
        result = run('hybrid', 32, True, fixture=stop_fixture, sockets=1)
        self.assertEqual(result.returncode, 0, result.stderr)
        data = json.loads(report_path.read_text())
        self.assertEqual(data['tokens']['sampled_stop'], 4)
        self.assertEqual(data['tokens']['decode_steps'], 3)
        self.assertEqual(data['stop_reason'], 'generation_limit')

    def test_synthetic_kv_keeps_the_bf16_kv_copy_consistent(self):
        """LLAMA_ATTN=amx keeps a persistent BF16 copy of every layer's K/V, packed lazily from the FP32 caches. A real
        prime before the synthetic fill leaves that copy valid for the primed positions; the fill must drop that validity,
        or the batch after the fill reads BF16 values of the primed tokens where the FP32 caches hold the pattern. The
        logits of the prefill that follows the fill are therefore bitwise independent of the prime, on one and two sockets,
        with legacy and with split-tier planned KV."""
        self.need('probe-int8-amx-emul')
        if not (BUILD/'probe-int8-amx-emul').exists() or b'synthetic' not in (BUILD/'probe-int8-amx-emul').read_bytes():
            self.skipTest('build with SYNTHETIC_KV=1')
        suffix = [65 + (i * 7) % 20 for i in range(12)]
        positions = 44  # the BF16 copy packs blocks of 32 positions: a prime of 33 or more leaves block 0 valid
        planned = dict(LLAMA_PLACEMENT='hybrid', LLAMA_HYBRID_KV_HBM_LAYERS='1', LLAMA_HYBRID_WEIGHTS_HBM_LAYERS='1')
        for sockets in (1, 2):
            for name, tier in (('legacy', {}), ('split-tier', planned)):
                args = dict(binary='probe-int8-amx-emul', LLAMA_KERNELS='amx', LLAMA_ATTN='amx', LLAMA_KV_LAYOUT='head',
                            LLAMA_PREFILL_BATCH='8', LLAMA_PREFILL_LOGITS='all', LLAMA_TOPOLOGY=f'virtual:{sockets}x3', **tier)
                with self.subTest(sockets=sockets, placement=name):
                    outputs = [self.probe('int8', 'wide', 'synthetic', sockets, tokens=[prime, positions, *suffix], **args)
                               for prime in (0, 8, 33, positions)]
                    self.assertEqual(len(outputs[0]), len(suffix) * TP_WIDE_CONFIG[5] * 4)
                    for prime, output in zip((8, 33, positions), outputs[1:]):
                        self.assertEqual(output, outputs[0], f'prime {prime}')

    def test_plan_only_checks_rounded_budget_and_spills(self):
        report = Path(self.temp.name)/'tier-plan.json'
        command = [str(BUILD/'run'), str(self.fixtures['unshared']/'fp32.bin'), '--plan-only', '--json', str(report)]

        def plan(placement, **knobs):
            result = subprocess.run(command, capture_output=True, text=True, timeout=15,
                                    env=environment(2, LLAMA_PLACEMENT=placement, LLAMA_KERNELS='scalar', **knobs))
            return result, json.loads(report.read_text())

        result, report_data = plan('hbm-only')
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(report_data['stop_reason'], 'plan_only')
        self.assertEqual(report_data['regions'], [])
        placement = report_data['placement_plan']
        self.assertTrue(placement['admitted'])
        self.assertTrue(placement['simulated'])
        self.assertEqual({arena['tier'] for arena in placement['arenas']}, {'hbm'})
        charged = max(node['charged_bytes'] for node in placement['nodes'])
        cap = str(charged / (1 << 30))
        self.assertEqual(plan('hbm-only', LLAMA_HBM_BUDGET_GIB=cap)[0].returncode, 0)
        result, report_data = plan('hbm-only', LLAMA_HBM_BUDGET_GIB=str((charged - 1) / (1 << 30)))
        self.assertNotEqual(result.returncode, 0)
        self.assertIn('over its', result.stderr)
        self.assertFalse(report_data['placement_plan']['admitted'])
        result, report_data = plan('hybrid', LLAMA_HBM_BUDGET_GIB='0.001')
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(report_data['placement_plan']['classifier'], 'ddr')
        self.assertEqual({arena['tier'] for arena in report_data['placement_plan']['arenas']}, {'hbm', 'ddr'})
        result, _ = plan('hybrid', LLAMA_HYBRID_KV_HBM_LAYERS='3')
        self.assertNotEqual(result.returncode, 0)
        self.assertIn('must be auto or between 0 and 2', result.stderr)

    def test_70b_plan_only_admits_ddr_spill_without_loading_weights(self):
        model = Path(self.temp.name)/'70b-sizing.bin'
        d, hidden, layers, heads, kv_heads, vocab, capacity, group = 8192, 28672, 80, 64, 8, 128256, 131072, 64
        kv_dim = d // heads * kv_heads
        weight_count = layers * (2 * d * d + 2 * d * kv_dim + 3 * d * hidden) + 2 * vocab * d
        file_bytes = 256 + (2 * layers + 1) * d * 4 + weight_count + weight_count // group * 4
        with model.open('wb') as output:
            header = struct.pack('<Ii7iBi', 0x616b3432, 2, d, hidden, layers, heads, kv_heads, vocab, capacity, 0, group)
            output.write(header.ljust(256, b'\0'))
            output.truncate(file_bytes)  # Metadata-only fixture. The tensor payload consists entirely of holes.
        report = Path(self.temp.name)/'70b-plan.json'

        def plan(mode, context):
            proc = subprocess.run([str(BUILD/'runq'), str(model), '--context-length', str(context), '--plan-only',
                                   '--json', str(report)], capture_output=True, text=True, timeout=15,
                                  env=environment(2, LLAMA_KERNELS='scalar', LLAMA_PLACEMENT=mode,
                                                  LLAMA_HBM_BUDGET_GIB='56'))
            return proc, json.loads(report.read_text())

        fit, report_data = plan('hbm-only', 8192)
        self.assertEqual(fit.returncode, 0, fit.stderr)
        self.assertEqual(report_data['regions'], [])
        over, report_data = plan('hbm-only', 98304)
        self.assertNotEqual(over.returncode, 0)
        self.assertFalse(report_data['placement_plan']['admitted'])
        mixed, report_data = plan('hybrid', 98304)
        self.assertEqual(mixed.returncode, 0, mixed.stderr)
        placement = report_data['placement_plan']
        self.assertEqual(placement['weights_hbm_layers'], 80)
        self.assertGreater(placement['kv_hbm_layers'], 0)
        self.assertLess(placement['kv_hbm_layers'], 80)
        self.assertEqual(report_data['regions'], [])

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

    def test_w8a32_stays_per_token_on_two_sockets(self):
        """W8A32 never uses the batched GEMM, on one socket (C6) or two: the knob changes nothing, bitwise, and the
        report names no batched prefill kernel."""
        extra = {'LLAMA_Q8_ACTIVATIONS': 'fp32', 'LLAMA_PREFILL_BATCH': '4'}
        base = self.probe('int8', 'shared', 'batch', 2, **extra)
        self.assertEqual(base, self.probe('int8', 'shared', 'batch', 2, LLAMA_PREFILL_GEMM='on', **extra))
        self.assertIsNone(self.report('runq', 'int8.bin', LLAMA_PREFILL_GEMM='on', **extra)['kernels_prefill'])

    def report(self, engine, filename, sockets=2, prompt='ABCDE', fixture='shared', binary=None, full=False, **extra):
        path = Path(self.temp.name)/f'report-{engine}-{sockets}.json'
        path.unlink(missing_ok=True)
        directory = self.fixtures[fixture]
        result = subprocess.run([str(BUILD/(binary or engine)), str(directory/filename), '-z', str(directory/'tokenizer.bin'),
                                 '-i', prompt, '-t', '0', '--json', str(path)], capture_output=True, text=True, timeout=60,
                                env=environment(sockets, **extra))
        self.assertEqual(result.returncode, 0, result.stderr)
        data = json.loads(path.read_text())
        self.assertEqual(data['runtime']['sockets'], sockets)
        return data if full else data['runtime']

    def test_fp32_two_socket_batched_prefill_agrees_with_one_socket_batched(self):
        """D1 (FP32): M rows per pass over the weights on both sockets against the same batch on one socket, all
        logits, with LLAMA_TP_CHECK comparing the residual copies after every layer. The portable GEMM sums like the GEMV,
        so the one-socket batch is also the per-token reference."""
        knobs = {'LLAMA_PREFILL_GEMM': 'on', 'LLAMA_PREFILL_LOGITS': 'all'}
        for name, vocab in (('shared', TP_CONFIG[5]), ('odd', TP_ODD_CONFIG[5])):
            for prompt in (1, 5, 33):
                tokens = [65 + (i * 7) % 20 for i in range(prompt)]
                reference = self.logits('fp32', name, None, 1, tokens=tokens)
                for batch in (1, 3, 7, 32):
                    with self.subTest(fixture=name, prompt=prompt, batch=batch):
                        one = self.logits('fp32', name, 'batch', 1, tokens=tokens, LLAMA_PREFILL_BATCH=str(batch), **knobs)
                        two = self.logits('fp32', name, 'batch', 2, tokens=tokens, LLAMA_PREFILL_BATCH=str(batch), **knobs)
                        self.assert_agree(one, two, vocab)
                        self.assert_agree(reference, two, vocab)

    def test_w8a8_two_socket_batched_prefill_agrees_with_one_socket_batched(self):
        """D1 (W8A8): the batched Q8 GEMM on both sockets, full-row activation quantization for QKV/W1/W3 and the owned
        slice (on the original group boundaries) for Wo/W2, against the same batch on one socket and against the
        per-token reference; LLAMA_TP_CHECK compares the residual copies after every layer."""
        knobs = {'LLAMA_PREFILL_GEMM': 'on', 'LLAMA_PREFILL_LOGITS': 'all'}
        for name, vocab in (('shared', TP_CONFIG[5]), ('odd', TP_ODD_CONFIG[5])):
            for prompt in (1, 5, 33):
                tokens = [65 + (i * 7) % 20 for i in range(prompt)]
                reference = self.logits('int8', name, None, 1, tokens=tokens)
                for batch in (1, 3, 7, 32):
                    with self.subTest(fixture=name, prompt=prompt, batch=batch):
                        one = self.logits('int8', name, 'batch', 1, tokens=tokens, LLAMA_PREFILL_BATCH=str(batch), **knobs)
                        two = self.logits('int8', name, 'batch', 2, tokens=tokens, LLAMA_PREFILL_BATCH=str(batch), **knobs)
                        self.assert_agree(one, two, vocab)
                        self.assert_agree(reference, two, vocab)

    def test_w8a8_two_socket_batched_prefill_threads_variants_and_report(self):
        knobs = {'LLAMA_PREFILL_GEMM': 'on', 'LLAMA_PREFILL_BATCH': '8'}
        tokens = [65 + (i * 5) % 19 for i in range(33)]
        one = self.logits('int8', 'shared', 'batch', 1, tokens=tokens, **knobs)
        for threads in (1, 2, 7):
            with self.subTest(threads=threads):
                self.assert_agree(one, self.logits('int8', 'shared', 'batch', 2, threads, tokens=tokens, **knobs),
                                  TP_CONFIG[5])
        reference = self.probe('int8', 'shared', 'batch', 2, tokens=tokens, **knobs)
        self.assertEqual(reference, self.probe('int8', 'shared', 'batch', 2, tokens=tokens, **knobs, LLAMA_TEAM_FANOUT='0'))
        # Decode after a batched prefill reads the cache the batch wrote.
        generated = self.probe('int8', 'unshared', 'generate', 1, **knobs)
        self.assertGreaterEqual(len(generated.split()), 8)
        self.assertEqual(generated, self.probe('int8', 'unshared', 'generate', 2, **knobs))
        runtime = self.report('runq', 'int8.bin', **knobs)
        self.assertEqual(runtime['prefill_batch'], 8)
        self.assertIn(runtime['kernels_prefill'], ('portable-q8', 'avx512-q8'))
        # With the GEMM off the report names no batched kernel.
        self.assertIsNone(self.report('runq', 'int8.bin', LLAMA_PREFILL_BATCH='8', LLAMA_PREFILL_GEMM='off')['kernels_prefill'])

    def need(self, binary):
        if not (BUILD/binary).exists():
            self.skipTest(f'{binary} was not built')

    def test_bf16_two_socket_batched_prefill_agrees_with_one_socket(self):
        """D1 (BF16): each socket packs its own shards to BF16 and runs the BF16 GEMM, on the scalar reference and on the
        software tile model. Two sockets against one socket at the same rung, last-row and all-position logits (the latter
        through the lazily packed classifier range), then greedy generation, whose decode reads the FP32 shards."""
        for binary, kernels in (('probe-fp32', 'scalar'), ('probe-fp32-amx-emul', 'amx')):
            self.need(binary)
            knobs = {'LLAMA_PRECISION': 'bf16', 'LLAMA_KERNELS': kernels}
            for logits in ('last', 'all'):
                for prompt in (5, 33):
                    tokens = [65 + (i * 7) % 20 for i in range(prompt)]
                    for batch in (2, 5, 32):
                        with self.subTest(binary=binary, logits=logits, prompt=prompt, batch=batch):
                            args = dict(tokens=tokens, binary=binary, LLAMA_PREFILL_BATCH=str(batch),
                                        LLAMA_PREFILL_LOGITS=logits, **knobs)
                            one = self.logits('fp32', 'shared', 'batch', 1, **args)
                            two = self.logits('fp32', 'shared', 'batch', 2, **args)
                            self.assert_agree(one, two, TP_CONFIG[5])
            with self.subTest(binary=binary, what='generate'):
                args = dict(binary=binary, LLAMA_PREFILL_BATCH='4', **knobs)
                one = self.probe('fp32', 'unshared', 'generate', 1, **args)
                self.assertGreaterEqual(len(one.split()), 8)
                self.assertEqual(one, self.probe('fp32', 'unshared', 'generate', 2, **args))

    def test_bf16_two_socket_report_and_packed_regions(self):
        for binary, kernels, name in (('run', 'scalar', 'bf16-reference'), ('run-amx-emul', 'amx', 'amx-bf16')):
            self.need(binary)
            with self.subTest(binary=binary):
                data = self.report('run', 'fp32.bin', binary=binary, full=True, LLAMA_PRECISION='bf16',
                                   LLAMA_KERNELS=kernels, LLAMA_PREFILL_BATCH='4')
                self.assertEqual(data['runtime']['precision_mode'], 'bf16')
                self.assertEqual(data['runtime']['kernels_prefill'], name)
                packed = [r for r in data['regions'] if r['role'] == 'packed']
                self.assertEqual(sorted(r['socket'] for r in packed), [0, 1])
                self.assertGreater(data['memory_bytes']['packed_weights_peak'], 0)
                self.assertIsNotNone(data['timing_seconds']['weight_pack'])
                # Every shard of the seven matrices per layer, padded to 16-row tiles and 32-value K blocks, 2 bytes each.
                d, h, layers, heads, kvheads = TP_CONFIG[:5]
                kv = d // heads * kvheads
                pad = lambda rows, cols: -(-rows // 16) * 16 * (-(-cols // 32) * 32) * 2
                line = lambda n: -(-n // 64) * 64
                per_layer = [pad(d // 2, d), pad(kv // 2, d), pad(kv // 2, d), pad(d, d // 2), pad(h // 2, d),
                             pad(d, h // 2), pad(h // 2, d)]
                expected = 0
                for _ in range(layers):
                    for size in per_layer:
                        expected = line(expected) + size
                for region in packed:
                    self.assertGreaterEqual(region['bytes'], expected)
        # BF16 stays an explicit mode that needs the batched GEMM.
        for extra in ({'LLAMA_PREFILL_GEMM': 'off'}, {'LLAMA_PREFILL_BATCH': '1'}):
            result = subprocess.run([str(BUILD/'probe-fp32'), str(self.fixtures['shared']/'fp32.bin'), '65'],
                                    capture_output=True, timeout=60,
                                    env=environment(2, LLAMA_PRECISION='bf16', **{'LLAMA_PREFILL_BATCH': '4', **extra}))
            self.assertNotEqual(result.returncode, 0)
            self.assertIn(b'LLAMA_PRECISION=bf16', result.stderr)

    def test_amx_int8_two_socket_batched_prefill_agrees_with_one_socket(self):
        """D1 (AMX-INT8): int8 shard copies per socket for the tile kernel (group size 64), W8A8, against one socket at the
        same rung and against the per-token reference."""
        self.need('probe-int8-amx-emul')
        # the GEMM's budget: AMX-BF16 attention and the vector SwiGLU exponential (W8A8 defaults, judged by the W8A8
        # gate) stay on their ways back
        knobs = {'LLAMA_KERNELS': 'amx', 'LLAMA_PREFILL_LOGITS': 'all',
                 'LLAMA_ATTN': 'tiled', 'LLAMA_SWIGLU_EXP': 'libm'}
        for prompt in (5, 33):
            tokens = [65 + (i * 7) % 20 for i in range(prompt)]
            reference = self.logits('int8', 'wide', None, 1, tokens=tokens)
            for batch in (2, 5, 32):
                with self.subTest(prompt=prompt, batch=batch):
                    args = dict(tokens=tokens, binary='probe-int8-amx-emul', LLAMA_PREFILL_BATCH=str(batch), **knobs)
                    one = self.logits('int8', 'wide', 'batch', 1, **args)
                    two = self.logits('int8', 'wide', 'batch', 2, **args)
                    self.assert_agree(one, two, TP_WIDE_CONFIG[5])
                    self.assert_agree(reference, two, TP_WIDE_CONFIG[5])
        data = self.report('runq', 'int8.bin', fixture='wide', binary='runq-amx-emul', full=True, LLAMA_KERNELS='amx',
                           LLAMA_PREFILL_BATCH='4')
        self.assertEqual(data['runtime']['kernels_prefill'], 'amx-int8')
        self.assertEqual(sorted(r['socket'] for r in data['regions'] if r['role'] == 'packed'), [0, 1])
        self.assertGreater(data['memory_bytes']['packed_weights_peak'], 0)
        # A group size other than 32 or a multiple of 64 cannot use the tile kernel: the VNNI or portable rung runs, with
        # no packed copies.
        data = self.report('runq', 'int8.bin', binary='runq-amx-emul', full=True, LLAMA_KERNELS='amx',
                           LLAMA_PREFILL_BATCH='4')
        self.assertEqual(data['runtime']['kernels_prefill'], 'portable-q8')
        self.assertEqual([r for r in data['regions'] if r['role'] == 'packed'], [])
        # W8A32 never takes the tile kernel.
        data = self.report('runq', 'int8.bin', fixture='wide', binary='runq-amx-emul', full=True, LLAMA_KERNELS='amx',
                           LLAMA_PREFILL_BATCH='4', LLAMA_Q8_ACTIVATIONS='fp32')
        self.assertIsNone(data['runtime']['kernels_prefill'])
        self.assertEqual([r for r in data['regions'] if r['role'] == 'packed'], [])

    def test_amx_int8_tiled_kernels_are_bitwise_the_chunk_kernel(self):
        """LLAMA_Q8_GEMM=skewed (the default: the pipelined kernel with each tile store issued after the next half-step's
        tile products) and =pipelined (tile-order activations, transposed weight scales, overlapped rescale) give bitwise
        the logits of the chunk kernel (LLAMA_Q8_GEMM=chunk, the way back), on one and two sockets, all positions,
        batches that end mid-block, in both K-block orders (LLAMA_Q8_KB_ORDER=rows, the default, and =chunks); the mode
        is reported in build.q8_gemm and the order in build.q8_kb_order, the tiled kernels pack the transposed scales,
        and other values are refused."""
        self.need('probe-int8-amx-emul')
        knobs = {'LLAMA_KERNELS': 'amx', 'LLAMA_PREFILL_LOGITS': 'all'}
        for prompt in (5, 33):
            tokens = [65 + (i * 7) % 20 for i in range(prompt)]
            for batch in (2, 5, 32):
                for sockets in (1, 2):
                    with self.subTest(prompt=prompt, batch=batch, sockets=sockets):
                        args = dict(tokens=tokens, binary='probe-int8-amx-emul', LLAMA_PREFILL_BATCH=str(batch), **knobs)
                        chunk = self.probe('int8', 'wide', 'batch', sockets, LLAMA_Q8_GEMM='chunk', **args)
                        pipelined = self.probe('int8', 'wide', 'batch', sockets, LLAMA_Q8_GEMM='pipelined', **args)
                        skewed = self.probe('int8', 'wide', 'batch', sockets, **args)
                        chunks = self.probe('int8', 'wide', 'batch', sockets, LLAMA_Q8_KB_ORDER='chunks', **args)
                        self.assertEqual(len(chunk), prompt * TP_WIDE_CONFIG[5] * 4)
                        self.assertEqual(chunk, pipelined)
                        self.assertEqual(chunk, skewed)
                        self.assertEqual(chunk, chunks)
        for sockets in (1, 2):
            reports = {mode: self.report('runq', 'int8.bin', sockets=sockets, fixture='wide', binary='runq-amx-emul',
                                         full=True, LLAMA_KERNELS='amx', LLAMA_PREFILL_BATCH='4', LLAMA_Q8_GEMM=mode)
                       for mode in ('chunk', 'pipelined', 'skewed', '')}
            for mode in ('chunk', 'pipelined', 'skewed'):
                self.assertEqual(reports[mode]['build']['q8_gemm'], mode)
            self.assertEqual(reports['']['build']['q8_gemm'], 'skewed')
            self.assertEqual(reports['']['build']['q8_kb_order'], 'rows')
            for mode in ('pipelined', 'skewed'):
                self.assertGreater(reports[mode]['memory_bytes']['packed_weights_peak'],
                                   reports['chunk']['memory_bytes']['packed_weights_peak'])
        directory = self.fixtures['wide']
        result = subprocess.run([str(BUILD/'runq-amx-emul'), str(directory/'int8.bin'), '-z',
                                 str(directory/'tokenizer.bin'), '-i', 'A', '-t', '0'], capture_output=True, text=True,
                                timeout=60, env=environment(1, LLAMA_Q8_GEMM='bogus'))
        self.assertNotEqual(result.returncode, 0)
        self.assertIn('LLAMA_Q8_GEMM must be skewed, pipelined or chunk', result.stderr)
        result = subprocess.run([str(BUILD/'runq-amx-emul'), str(directory/'int8.bin'), '-z',
                                 str(directory/'tokenizer.bin'), '-i', 'A', '-t', '0'], capture_output=True, text=True,
                                timeout=60, env=environment(1, LLAMA_Q8_KB_ORDER='bogus'))
        self.assertNotEqual(result.returncode, 0)
        self.assertIn('LLAMA_Q8_KB_ORDER must be rows or chunks', result.stderr)

    def test_amx_int8_gs32_half_tiles(self):
        """Group size 32 (the Q8_0 GGUF block) on AMX-INT8 with half-shape tiles: the tile kernel runs (packed copies on
        both sockets), every tiled mode and K-block order gives bitwise the logits of the AMX chunk kernel, and the
        result agrees with the per-token reference on one and two sockets, up to one position whose activation quantization
        rounds one step apart (assert_agree_but_flips: position 10 of this prompt, on both kernels; eight other prompts
        showed none)."""
        self.need('probe-int8-amx-emul')
        knobs = {'LLAMA_KERNELS': 'amx', 'LLAMA_PREFILL_LOGITS': 'all', 'LLAMA_ATTN': 'tiled', 'LLAMA_SWIGLU_EXP': 'libm'}
        tokens = [65 + (i * 7) % 20 for i in range(33)]
        reference = self.logits('int8', 'wide-gs32', None, 1, tokens=tokens)
        for batch in (2, 5, 32):
            for sockets in (1, 2):
                with self.subTest(batch=batch, sockets=sockets):
                    args = dict(tokens=tokens, binary='probe-int8-amx-emul', LLAMA_PREFILL_BATCH=str(batch), **knobs)
                    chunk = self.logits('int8', 'wide-gs32', 'batch', sockets, LLAMA_Q8_GEMM='chunk', **args)
                    for mode, order in (('skewed', 'rows'), ('pipelined', 'rows'), ('skewed', 'chunks')):
                        tiled = self.logits('int8', 'wide-gs32', 'batch', sockets, LLAMA_Q8_GEMM=mode,
                                            LLAMA_Q8_KB_ORDER=order, **args)
                        self.assertEqual(chunk.tobytes(), tiled.tobytes(), (mode, order))
                    self.assert_agree_but_flips(reference, chunk, TP_WIDE_CONFIG[5])
        for sockets in (1, 2):
            data = self.report('runq', 'int8.bin', sockets=sockets, fixture='wide-gs32', binary='runq-amx-emul',
                               full=True, LLAMA_KERNELS='amx', LLAMA_PREFILL_BATCH='4')
            self.assertEqual(data['runtime']['kernels_prefill'], 'amx-int8')
            self.assertEqual(sorted(r['socket'] for r in data['regions'] if r['role'] == 'packed'), list(range(sockets)))

    def test_vnni_gemv_gs32_pairs(self):
        """Group size 32 per-token decode on the AVX-512 kernels (the scalar model of the intrinsics, so the VNNI pair kernel
        runs: one 64-byte load per two groups; every per-socket width of the wide fixture is a multiple of 64) agrees with
        the per-token reference on one and two sockets up to W8A8 rounding flips. The kernel sums each row in 16 lanes, so
        an activation within about 1e-5 of an int8 rounding tie can quantize one step apart. On two sockets that happens at
        position 0 (layer 1's QKV input at x*127/amax = -7.499987: the first and only differing byte against the same kernel
        summed in portable order), and through its K/V it moves every later position by up to 2e-2 (logits about 1.5).
        Hence every row may be off, by at most 5e-2, and keeps its argmax within that. The exact check of the kernel is
        test_q8 in tests/kernels.c; with the two scale halves swapped this test fails at 0.09, that one by whole units."""
        self.need('probe-int8-avx512-emul')
        tokens = [65 + (i * 7) % 20 for i in range(33)]
        reference = self.logits('int8', 'wide-gs32', None, 1, tokens=tokens)
        for sockets in (1, 2):
            with self.subTest(sockets=sockets):
                vnni = self.logits('int8', 'wide-gs32', None, sockets, tokens=tokens, binary='probe-int8-avx512-emul',
                                   LLAMA_KERNELS='avx512')
                self.assert_agree_but_flips(reference, vnni, TP_WIDE_CONFIG[5], flips=len(tokens), flip_tol=5e-2)

    def test_interleaved_batch_reduction_is_bitwise_the_barrier_one(self):
        """LLAMA_TP_BATCH_REDUCE=interleaved (each stored block of the two-socket W8A8 Wo/W2 partials is added into the
        residual during the projection) and =wo (Wo only) give bitwise the logits of the barrier reduction,
        on the portable and the AMX kernels (skewed, pipelined and chunk), all positions, batches that end mid-block and thread
        counts that split the blocks unevenly; LLAMA_TP_CHECK holds; it is reported in build.tp_batch_reduce and refuses
        other values."""
        self.need('probe-int8-amx-emul')
        tokens = [65 + (i * 7) % 20 for i in range(33)]
        cases = [('shared', None, {'LLAMA_PREFILL_GEMM': 'on'}), ('odd', None, {'LLAMA_PREFILL_GEMM': 'on'}),
                 ('wide', 'probe-int8-amx-emul', {'LLAMA_KERNELS': 'amx'}),
                 ('wide', 'probe-int8-amx-emul', {'LLAMA_KERNELS': 'amx', 'LLAMA_Q8_GEMM': 'pipelined'}),
                 ('wide', 'probe-int8-amx-emul', {'LLAMA_KERNELS': 'amx', 'LLAMA_Q8_GEMM': 'chunk'})]
        for fixture, binary, knobs in cases:
            for batch in (3, 32):
                for threads in (2, 5):
                    with self.subTest(fixture=fixture, knobs=knobs, batch=batch, threads=threads):
                        args = dict(tokens=tokens, binary=binary, LLAMA_PREFILL_BATCH=str(batch),
                                    LLAMA_PREFILL_LOGITS='all', LLAMA_TP_CHECK='1', **knobs)
                        barrier = self.probe('int8', fixture, 'batch', 2, threads, LLAMA_TP_BATCH_REDUCE='barrier', **args)
                        interleaved = self.probe('int8', fixture, 'batch', 2, threads, **args)
                        self.assertEqual(barrier, self.probe('int8', fixture, 'batch', 2, threads,
                                                             LLAMA_TP_BATCH_REDUCE='wo', **args))
                        self.assertEqual(len(barrier), len(tokens) * (TP_WIDE_CONFIG if fixture == 'wide' else
                                                                      TP_ODD_CONFIG if fixture == 'odd' else TP_CONFIG)[5] * 4)
                        self.assertEqual(barrier, interleaved)
        for mode in ('interleaved', 'wo', 'barrier'):
            data = self.report('runq', 'int8.bin', full=True, LLAMA_PREFILL_BATCH='4', LLAMA_TP_BATCH_REDUCE=mode)
            self.assertEqual(data['build']['tp_batch_reduce'], mode)
        self.assertEqual(self.report('runq', 'int8.bin', full=True)['build']['tp_batch_reduce'], 'interleaved')
        directory = self.fixtures['shared']
        result = subprocess.run([str(BUILD/'runq'), str(directory/'int8.bin'), '-z', str(directory/'tokenizer.bin'), '-i',
                                 'A', '-t', '0'], capture_output=True, text=True, timeout=60,
                                env=environment(2, LLAMA_TP_BATCH_REDUCE='bogus'))
        self.assertNotEqual(result.returncode, 0)
        self.assertIn('LLAMA_TP_BATCH_REDUCE must be interleaved, wo or barrier', result.stderr)

    def test_amx_bf16_attention_on_one_and_two_sockets(self):
        """LLAMA_ATTN=amx (BF16 tile products, prefill only): each socket packs its own KV heads into the BF16 copy carved
        from its work region. The kernel is partition-invariant, bitwise (tests/kernels.c: one KV head per call against
        all heads, any team against serial), but two sockets are not bitwise one socket: the row-parallel Wo/W2 partial
        sums are added across sockets, which moves FP32 values by an ulp (tiled attention: 1S against 2S 4.8e-7 on this
        fixture, hardware job 5892453), and such an ulp can flip one BF16 rounding of a Q or P element in the next layer.
        By the triangle inequality |amx 1S - amx 2S| <= |amx 1S - tiled 1S| + |tiled 1S - tiled 2S| + |tiled 2S - amx 2S|,
        i.e. twice the AMX-against-tiled budget plus the fixture tolerance, so two sockets are held to that budget below.
        So that the envelope cannot hide a partitioning bug, most rows must stay sharp: the median over rows of each
        row's worst |difference| / (fixture tolerance) is at most 1 (measured on hardware: 1S against 2S max 2.7e-3 on
        row 1, every other row within 7.2e-7, job 5892455). Against the tiled
        attention the logits stay within a noise budget: this sinusoidal fixture amplifies rounding (the BF16 GEMM alone
        moves its logits by rms 0.03, max 0.14; AMX attention by rms 0.05, max 0.26), so the budget is twice that, and
        the kernel itself is held to 0.03 in tests/kernels.c. The report names the variant. Decode is untouched (the
        knob acts on batched prefill)."""
        self.need('probe-int8-amx-emul')
        knobs = {'LLAMA_KERNELS': 'amx', 'LLAMA_PREFILL_LOGITS': 'all'}
        tokens = [65 + (i * 7) % 20 for i in range(33)]
        for batch in (5, 32):
            with self.subTest(batch=batch):
                args = dict(tokens=tokens, binary='probe-int8-amx-emul', LLAMA_PREFILL_BATCH=str(batch), **knobs)
                amx = [self.logits('int8', 'wide', 'batch', s, LLAMA_ATTN='amx', **args) for s in (1, 2)]
                tiled = self.logits('int8', 'wide', 'batch', 1, LLAMA_ATTN='tiled', **args)  # the pre-fix-7 default
                for other in (tiled, amx[1]):
                    diff = [abs(a - b) for a, b in zip(other, amx[0])]
                    self.assertLessEqual(max(diff), 0.5)
                    self.assertLessEqual((sum(d * d for d in diff) / len(diff)) ** 0.5, 0.1)
                vocab = TP_WIDE_CONFIG[5]
                worst = sorted(max(abs(a - b) / (ABS + REL * abs(a)) for a, b in
                                   zip(amx[0][r * vocab:(r + 1) * vocab], amx[1][r * vocab:(r + 1) * vocab]))
                               for r in range(len(amx[0]) // vocab))
                self.assertLessEqual(worst[len(worst) // 2], 1.0, 'most rows of two sockets must match one socket')
        for sockets in (1, 2):
            data = self.report('runq', 'int8.bin', sockets=sockets, fixture='wide', binary='runq-amx-emul', full=True,
                               LLAMA_KERNELS='amx', LLAMA_PREFILL_BATCH='4', LLAMA_ATTN='amx')
            self.assertEqual(data['build']['attention'], 'amx')

    def test_w8a8_defaults_take_amx_attention_and_vector_exp(self):
        """The W8A8 defaults of prefill-gap fixes 4 and 7, with no knob set: runq's batched prefill runs AMX-BF16
        attention where AMX is active and the vector SwiGLU exponential where AVX-512 runs (the hardware build; on a
        host without AVX-512 the emulated-AMX probe keeps libm). The default path equals LLAMA_ATTN=amx bitwise and stays
        within a budget of the ways back (LLAMA_ATTN=tiled, LLAMA_SWIGLU_EXP=libm) on one and two sockets. The budget is
        this fixture's measured AMX-attention noise (rms 0.05, max 0.25, above) scaled by the headroom the W8A8 gate
        leaves the 8B defaults: rms limit 0.2 over the worst default rms 0.147 and max limit 16 over max 9.4 (job
        5892385, docs/prefill-gap.md), i.e. rms 0.068 and max 0.43; argmax agreement >= 0.9 as in the gate."""
        self.need('probe-int8-amx-emul')
        knobs = {'LLAMA_KERNELS': 'amx', 'LLAMA_PREFILL_LOGITS': 'all'}
        tokens = [65 + (i * 7) % 20 for i in range(33)]
        vocab = TP_WIDE_CONFIG[5]
        for sockets in (1, 2):
            for batch in (5, 32):
                with self.subTest(sockets=sockets, batch=batch):
                    args = dict(tokens=tokens, binary='probe-int8-amx-emul', LLAMA_PREFILL_BATCH=str(batch), **knobs)
                    default = self.logits('int8', 'wide', 'batch', sockets, **args)
                    amx = self.logits('int8', 'wide', 'batch', sockets, LLAMA_ATTN='amx', **args)
                    self.assertEqual(default.tobytes(), amx.tobytes())
                    back = self.logits('int8', 'wide', 'batch', sockets, LLAMA_ATTN='tiled', LLAMA_SWIGLU_EXP='libm',
                                       **args)
                    diff = [abs(a - b) for a, b in zip(default, back)]
                    max_budget = 0.25 * 16 / 9.4
                    self.assertLessEqual(max(diff), max_budget)
                    self.assertLessEqual((sum(d * d for d in diff) / len(diff)) ** 0.5, 0.05 * 0.2 / 0.147)
                    # argmax agreement >= 0.9, counting a near-tie (within the max budget) of this random fixture's
                    # 128k-entry rows as agreement, the convention of assert_agree
                    rows, agree = len(back) // vocab, 0
                    for r in range(rows):
                        d, b = default[r * vocab:(r + 1) * vocab], back[r * vocab:(r + 1) * vocab]
                        agree += b[d.index(max(d))] >= max(b) - max_budget
                    self.assertGreaterEqual(agree / rows, 0.9)
            build = self.report('runq', 'int8.bin', sockets=sockets, fixture='wide', binary='runq-amx-emul', full=True,
                                LLAMA_KERNELS='amx', LLAMA_PREFILL_BATCH='4')['build']
            self.assertEqual((build['attention'], build['swiglu_exp']), ('amx', 'vector'))

    def test_per_channel_int8_checkpoint(self):
        """Fix 9 (docs/prefill-gap.md): a version 4 checkpoint holds one scale per output channel, so every product is
        one group over its whole (per-socket) input and activations get one scale per token. The batched prefill equals
        the per-token path on the same attention and SwiGLU exponential (LLAMA_ATTN=tiled, LLAMA_SWIGLU_EXP=libm: the
        W8A8 defaults round attention to BF16 in prefill only, moving this fixture by up to 0.31, hardware job
        5892453); two sockets agree with one within the W8A8 noise (each socket quantizes its own Wo/W2 input slice, as a
        tensor-parallel rank does); on the AMX tile model the pipelined kernel (its full-K path) equals the chunk kernel
        bitwise; the report names format 4 and group size 0."""
        tokens = [65 + (i * 7) % 20 for i in range(33)]
        all_logits = {'LLAMA_PREFILL_LOGITS': 'all', 'LLAMA_PREFILL_BATCH': '32', 'LLAMA_ATTN': 'tiled',
                      'LLAMA_SWIGLU_EXP': 'libm'}
        batched = {}
        for sockets in (1, 2):
            with self.subTest(sockets=sockets):
                per_token = self.logits('int8', 'wide-pc', None, sockets, tokens=tokens)
                batched[sockets] = self.logits('int8', 'wide-pc', 'batch', sockets, tokens=tokens, **all_logits)
                self.assert_agree(per_token, batched[sockets], TP_WIDE_CONFIG[5])
        diff = [abs(a - b) for a, b in zip(batched[1], batched[2])]
        self.assertLessEqual(max(diff), 0.5)
        self.assertLessEqual((sum(d * d for d in diff) / len(diff)) ** 0.5, 0.1)
        self.need('probe-int8-amx-emul')
        amx = {'LLAMA_KERNELS': 'amx', **all_logits}
        for sockets in (1, 2):
            with self.subTest(sockets=sockets, kernel='pipelined'):
                args = dict(tokens=tokens, binary='probe-int8-amx-emul', **amx)
                chunk = self.logits('int8', 'wide-pc', 'batch', sockets, LLAMA_Q8_GEMM='chunk', **args)
                pipelined = self.logits('int8', 'wide-pc', 'batch', sockets, **args)
                self.assertEqual(chunk.tobytes(), pipelined.tobytes())
                self.assert_agree(batched[sockets], chunk, TP_WIDE_CONFIG[5], abs_tol=0.05, rel_tol=0.02)
            data = self.report('runq', 'int8.bin', sockets, fixture='wide-pc', binary='runq-amx-emul', full=True,
                               LLAMA_KERNELS='amx', LLAMA_PREFILL_BATCH='4')
            self.assertEqual((data['model']['checkpoint_format'], data['model']['quantization_group_size']), (4, 0))
            self.assertEqual(data['runtime']['kernels_prefill'], 'amx-int8')

    def test_per_channel_header_is_validated(self):
        """Version 4 must say group size 0, and version 3 may not: a mislabelled file fails at load."""
        source = (self.fixtures['wide-pc']/'int8.bin').read_bytes()
        for version, gs, message in ((4, 64, b'must have group size zero'), (3, 0, b'unsupported INT8 group size 0')):
            with self.subTest(version=version, gs=gs):
                data = bytearray(source)
                data[4:8] = version.to_bytes(4, 'little')
                data[44:48] = gs.to_bytes(4, 'little')
                path = Path(self.temp.name)/f'bad-{version}-{gs}.bin'
                path.write_bytes(data)
                result = subprocess.run([str(BUILD/'probe-int8'), str(path), '65'], capture_output=True,
                                        env=environment(1))
                self.assertNotEqual(result.returncode, 0)
                self.assertIn(message, result.stderr)

    def test_auto_kernels_take_amx_for_w8a8_and_bf16_only(self):
        """D2 §7 under LLAMA_KERNELS=auto: W8A8 batched prefill runs AMX-INT8; W8A32 and plain FP32 never take AMX; the
        explicit BF16 mode takes AMX-BF16. The emulated-tile binaries emulate no AVX-512, so `auto` starts from the scalar
        kernels there unless the host has AVX-512; the GEMM and batch are named (on AVX-512 they are the default)."""
        auto = {'LLAMA_KERNELS': 'auto', 'LLAMA_PREFILL_GEMM': 'on', 'LLAMA_PREFILL_BATCH': '4'}
        for sockets in (1, 2):
            with self.subTest(sockets=sockets):
                data = self.report('runq', 'int8.bin', sockets, fixture='wide', binary='runq-amx-emul', full=True, **auto)
                self.assertEqual((data['runtime']['kernels_prefill'], data['build']['kernels']), ('amx-int8', 'amx'))
                data = self.report('runq', 'int8.bin', sockets, fixture='wide', binary='runq-amx-emul', full=True,
                                   LLAMA_Q8_ACTIVATIONS='fp32', **auto)
                self.assertIsNone(data['runtime']['kernels_prefill'])
                self.assertNotEqual(data['build']['kernels'], 'amx')  # scalar, or avx512 on an AVX-512 host
                data = self.report('run', 'fp32.bin', sockets, binary='run-amx-emul', full=True, **auto)
                self.assertNotEqual(data['build']['kernels'], 'amx')
                data = self.report('run', 'fp32.bin', sockets, binary='run-amx-emul', full=True, LLAMA_PRECISION='bf16',
                                   **auto)
                self.assertEqual((data['runtime']['kernels_prefill'], data['build']['kernels']), ('amx-bf16', 'amx'))

    def test_fp32_two_socket_batched_prefill_thread_counts_and_last_logits(self):
        knobs = {'LLAMA_PREFILL_GEMM': 'on'}
        tokens = [65 + (i * 5) % 19 for i in range(33)]
        one = self.logits('fp32', 'shared', 'batch', 1, tokens=tokens, LLAMA_PREFILL_BATCH='8', **knobs)
        for threads in (1, 2, 7):
            with self.subTest(threads=threads):
                two = self.logits('fp32', 'shared', 'batch', 2, threads, tokens=tokens, LLAMA_PREFILL_BATCH='8', **knobs)
                self.assert_agree(one, two, TP_CONFIG[5])

    def test_fp32_two_socket_batched_prefill_variants_are_bitwise_equal(self):
        """The dispatch tree only changes how the members synchronize, also for the M x dim exchange; and a decode after a
        batched prefill reads the cache the batch wrote (greedy generation equals one socket's)."""
        knobs = {'LLAMA_PREFILL_GEMM': 'on', 'LLAMA_PREFILL_BATCH': '4'}
        tokens = [65 + (i * 3) % 17 for i in range(13)]
        reference = self.probe('fp32', 'shared', 'batch', 2, tokens=tokens, **knobs)
        self.assertEqual(reference, self.probe('fp32', 'shared', 'batch', 2, tokens=tokens, **knobs, LLAMA_TEAM_FANOUT='0'))
        one = self.probe('fp32', 'unshared', 'generate', 1, **knobs)
        self.assertGreaterEqual(len(one.split()), 8)
        self.assertEqual(one, self.probe('fp32', 'unshared', 'generate', 2, **knobs))

    def test_fp32_two_socket_batched_prefill_report(self):
        runtime = self.report('run', 'fp32.bin', LLAMA_PREFILL_BATCH='4', LLAMA_PREFILL_GEMM='on')
        self.assertEqual(runtime['prefill_batch'], 4)
        self.assertIsNotNone(runtime['kernels_prefill'])
        self.assertIsNone(self.report('run', 'fp32.bin', LLAMA_PREFILL_BATCH='4', LLAMA_PREFILL_GEMM='off')['kernels_prefill'])
        # An oversized batch is capped at the context, and the exchange it needs is checked before anything is allocated.
        self.assertEqual(self.report('run', 'fp32.bin', LLAMA_PREFILL_BATCH='100000', LLAMA_PREFILL_GEMM='on')['prefill_batch'], 64)

    def test_tiled_attention_batched_prefill_agrees_with_per_token(self):
        """E2: LLAMA_ATTN=tiled on the AVX-512 kernels (the scalar model of the intrinsics, so the tile code itself runs),
        M = 64 and a smaller block, on one socket and on two (virtual 2x3, LLAMA_TP_CHECK=1), against the per-token path.
        The small fixture is held to the fixture tolerance against the scalar reference. The wide fixture has attention
        scores in the hundreds, where a softmax is conditioned about 1e-5 per unit of rounding: there the batched split
        attention itself is 2e-5 from per-token, so tiled (Q scaled before the dot, a longer dot chain) is held to 2e-4
        against the per-token path of the same kernels. W8A8 is held to the fixture tolerance against split."""
        self.need('probe-fp32-avx512-emul')
        tokens = [65 + (i * 7) % 20 for i in range(64)]
        emul = dict(binary='probe-fp32-avx512-emul', LLAMA_KERNELS='avx512')
        for name, vocab, reference, tolerance in (
                ('shared', TP_CONFIG[5], self.logits('fp32', 'shared', None, 1, tokens=tokens), (ABS, REL)),
                ('wide', TP_WIDE_CONFIG[5], self.logits('fp32', 'wide', None, 1, tokens=tokens, **emul), (2e-4, 2e-4))):
            for batch in (64, 24):
                for sockets in (1, 2):
                    with self.subTest(fixture=name, batch=batch, sockets=sockets):
                        args = dict(tokens=tokens, LLAMA_PREFILL_BATCH=str(batch), LLAMA_PREFILL_LOGITS='all',
                                    LLAMA_PREFILL_GEMM='on', LLAMA_ATTN='tiled', **emul)
                        self.assert_agree(reference, self.logits('fp32', name, 'batch', sockets, **args), vocab,
                                          *tolerance)
        self.need('probe-int8-avx512-emul')
        for sockets in (1, 2):
            args = dict(tokens=tokens, binary='probe-int8-avx512-emul', LLAMA_PREFILL_BATCH='64',
                        LLAMA_PREFILL_LOGITS='all', LLAMA_PREFILL_GEMM='on', LLAMA_KERNELS='avx512')
            with self.subTest(precision='int8', sockets=sockets):
                self.assert_agree(self.logits('int8', 'wide', 'batch', sockets, **args, LLAMA_ATTN='split'),
                                  self.logits('int8', 'wide', 'batch', sockets, **args, LLAMA_ATTN='tiled'),
                                  TP_WIDE_CONFIG[5])

    def test_group_decode_attention_agrees_with_split(self):
        """E5: LLAMA_ATTN_DECODE=group on the AVX-512 kernels (the scalar model of the intrinsics, so the group kernel itself
        runs), per-token over 64 positions (the chunk plan has two chunks past position 32 on three threads), on one socket
        and on two (virtual 2x3, LLAMA_TP_CHECK=1), against split on the same kernels. `run` (FP32) and `runq` (W8A8) are
        covered by the two probes; the wide fixture has scores in the hundreds, hence the looser budget as for tiled."""
        self.need('probe-fp32-avx512-emul')
        self.need('probe-int8-avx512-emul')
        tokens = [65 + (i * 7) % 20 for i in range(64)]
        for precision, binary in (('fp32', 'probe-fp32-avx512-emul'), ('int8', 'probe-int8-avx512-emul')):
            for name, vocab, tolerance in (('shared', TP_CONFIG[5], (ABS, REL)), ('wide', TP_WIDE_CONFIG[5], (2e-4, 2e-4))):
                for sockets in (1, 2):
                    with self.subTest(precision=precision, fixture=name, sockets=sockets):
                        args = dict(tokens=tokens, binary=binary, LLAMA_KERNELS='avx512')
                        self.assert_agree(self.logits(precision, name, None, sockets, **args, LLAMA_ATTN_DECODE='split'),
                                          self.logits(precision, name, None, sockets, **args, LLAMA_ATTN_DECODE='group'),
                                          vocab, *tolerance)
        generated = [self.probe(precision, 'unshared', 'generate', sockets, binary=binary, LLAMA_KERNELS='avx512',
                                LLAMA_ATTN_DECODE='group')
                     for precision, binary in (('fp32', 'probe-fp32-avx512-emul'),) for sockets in (1, 2)]
        self.assertGreaterEqual(len(generated[0].split()), 8)
        self.assertEqual(generated[0], generated[1])

    def test_dispatch_variants_are_bitwise_equal(self):
        """The dispatch tree against the flat dispatch only changes how the members synchronize: not a bit of the
        output."""
        variants = ({'LLAMA_TEAM_FANOUT': '0'}, {'LLAMA_TEAM_FANOUT': '2'})
        for precision, activations in (('fp32', None), ('int8', 'int8'), ('int8', 'fp32')):
            extra = {'LLAMA_Q8_ACTIVATIONS': activations} if activations else {}
            for threads in (3, 7):
                reference = self.probe(precision, 'shared', None, 2, threads, **extra)
                for variant in variants:
                    with self.subTest(precision=precision, activations=activations, threads=threads, **variant):
                        self.assertEqual(reference, self.probe(precision, 'shared', None, 2, threads, **variant, **extra))

    def test_synchronization_knobs_are_validated(self):
        model = self.fixtures['shared']/'fp32.bin'
        result = subprocess.run([str(BUILD/'probe-fp32'), str(model), '1'], capture_output=True, timeout=60,
                                env=environment(2, 3, LLAMA_TEAM_FANOUT='x'))
        self.assertNotEqual(result.returncode, 0)
        self.assertIn(b'LLAMA_TEAM_FANOUT must be 0 (flat) or a group size up to 64', result.stderr)

    def test_retired_knobs_are_refused(self):
        """A script that still sets a knob of a removed experimental path must not measure the default under the old label."""
        model = self.fixtures['shared']/'fp32.bin'
        for name, value in (('LLAMA_FP32_LAYOUT', 'packed4'), ('LLAMA_TP_FUSED', '0'), ('LLAMA_TP_REDUCE', 'barrier'),
                            ('LLAMA_TP_TRACE', '/dev/null'), ('LLAMA_TEAM_JOIN', 'counter'),
                            ('LLAMA_GEMV_SCHED', 'dynamic'), ('LLAMA_ATTN_ROW_BLOCK', '4'),
                            ('LLAMA_ATTN_TILED_QUERIES', '16'), ('LLAMA_ATTN_TILED_ORDER', 'longest'),
                            ('LLAMA_ATTN_WORKSPACE_MIB', '1'), ('LLAMA_GEMM_KC', '512')):
            with self.subTest(name=name):
                result = subprocess.run([str(BUILD/'probe-fp32'), str(model), '1'], capture_output=True, timeout=60,
                                        env=environment(2, 3, **{name: value}))
                self.assertNotEqual(result.returncode, 0)
                self.assertIn(name.encode() + b' is no longer supported', result.stderr)

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

    def test_two_sockets_imply_worker_teams(self):
        """LLAMA_BACKEND unset and LLAMA_BACKEND=teams (still accepted) run the same teams: bitwise equal on two sockets."""
        env = environment(2)
        del env['LLAMA_BACKEND']
        model = self.fixtures['shared']/'fp32.bin'
        implied = subprocess.run([str(BUILD/'probe-fp32'), str(model), *map(str, TOKENS)], capture_output=True,
                                 timeout=60, env=env)
        self.assertEqual(implied.returncode, 0, implied.stderr.decode())
        self.assertEqual(implied.stdout, self.probe('fp32', 'shared', None, 2))

    def test_kv_head_layout_is_bitwise_the_rows_layout(self):
        """E5.4: LLAMA_KV_LAYOUT=head ([kv_head][position][head_dim]) gives the very bytes of the rows layout, not merely
        values within a tolerance: per-token logits, batched prefill (last and all rows, several microbatches), greedy
        generation (batched prefill writes, then per-token decode writes), FP32 and W8A8, one and two sockets (virtual
        2x3), on the scalar kernels and on the AVX-512 kernels on the scalar intrinsic model (decode split and group,
        prefill split and tiled), and BF16 on the AMX tile model. Both layouts run the same binary, so a difference
        is the layout alone."""
        tokens = [65 + (i * 7) % 20 for i in range(64)]

        def same(precision, fixture, mode, sockets, threads=3, **args):
            rows = self.probe(precision, fixture, mode, sockets, threads, LLAMA_KV_LAYOUT='rows', **args)
            head = self.probe(precision, fixture, mode, sockets, threads, LLAMA_KV_LAYOUT='head', **args)
            self.assertGreater(len(rows), 0)
            self.assertEqual(rows, head)

        for precision in ('fp32', 'int8'):
            for fixture in ('shared', 'odd', 'wide'):
                for sockets in (1, 2):
                    with self.subTest(what='per-token', precision=precision, fixture=fixture, sockets=sockets):
                        same(precision, fixture, None, sockets, tokens=tokens)
        for precision, binary in (('fp32', 'probe-fp32-avx512-emul'), ('int8', 'probe-int8-avx512-emul')):
            self.need(binary)
            for fixture in ('shared', 'wide'):
                for sockets in (1, 2):
                    for decode in ('split', 'group'):
                        for threads in ((1, 3, 7) if sockets == 1 and fixture == 'shared' else (3,)):
                            with self.subTest(what='decode', precision=precision, fixture=fixture, sockets=sockets,
                                              decode=decode, threads=threads):
                                same(precision, fixture, None, sockets, threads, tokens=tokens, binary=binary,
                                     LLAMA_KERNELS='avx512', LLAMA_ATTN_DECODE=decode)
                    for attn in ('split', 'tiled'):
                        for batch in ('64', '24', '5'):
                            with self.subTest(what='batch', precision=precision, fixture=fixture, sockets=sockets,
                                              attn=attn, batch=batch):
                                same(precision, fixture, 'batch', sockets, tokens=tokens, binary=binary,
                                     LLAMA_KERNELS='avx512', LLAMA_ATTN=attn, LLAMA_PREFILL_BATCH=batch,
                                     LLAMA_PREFILL_LOGITS='all', LLAMA_PREFILL_GEMM='on')
            for sockets in (1, 2):
                with self.subTest(what='generate', precision=precision, sockets=sockets):
                    generated = [self.probe(precision, 'unshared', 'generate', sockets, binary=binary,
                                            LLAMA_KERNELS='avx512', LLAMA_ATTN='tiled', LLAMA_ATTN_DECODE='group',
                                            LLAMA_PREFILL_BATCH='4', LLAMA_PREFILL_GEMM='on', LLAMA_KV_LAYOUT=layout)
                                 for layout in ('rows', 'head')]
                    self.assertGreaterEqual(len(generated[0].split()), 8)
                    self.assertEqual(generated[0], generated[1])
        for precision in ('fp32', 'int8'):  # the scalar kernels: batched prefill and generation
            for sockets in (1, 2):
                with self.subTest(what='scalar batch', precision=precision, sockets=sockets):
                    same(precision, 'shared', 'batch', sockets, tokens=tokens, LLAMA_PREFILL_BATCH='24',
                         LLAMA_PREFILL_LOGITS='all', LLAMA_PREFILL_GEMM='on')
                    generated = [self.probe(precision, 'unshared', 'generate', sockets, LLAMA_PREFILL_BATCH='4',
                                            LLAMA_PREFILL_GEMM='on', LLAMA_KV_LAYOUT=layout)
                                 for layout in ('rows', 'head')]
                    self.assertGreaterEqual(len(generated[0].split()), 8)
                    self.assertEqual(generated[0], generated[1])
        binary = 'probe-fp32-amx-emul'  # BF16 prefill on the tile model: the packed projections write K/V
        self.need(binary)
        for sockets in (1, 2):
            for batch in ('2', '5', '32'):
                with self.subTest(what='bf16', sockets=sockets, batch=batch):
                    same('fp32', 'shared', 'batch', sockets, tokens=tokens[:33], binary=binary,
                         LLAMA_PRECISION='bf16', LLAMA_KERNELS='amx', LLAMA_PREFILL_BATCH=batch,
                         LLAMA_PREFILL_LOGITS='all')

    def test_kv_head_layout_is_reported_and_validated(self):
        """E5.4: the active layout is in the report (build.kv_layout, default head) and a bad value is refused."""
        for sockets in (1, 2):
            for layout, expected in ((None, 'head'), ('auto', 'head'), ('', 'head'), ('rows', 'rows'), ('head', 'head')):
                extra = {} if layout is None else {'LLAMA_KV_LAYOUT': layout}
                with self.subTest(sockets=sockets, layout=layout):
                    data = self.report('run', 'fp32.bin', sockets=sockets, full=True, **extra)
                    self.assertEqual(data['build']['kv_layout'], expected)
        result = subprocess.run([str(BUILD/'probe-fp32'), str(self.fixtures['shared']/'fp32.bin'), '65'],
                                capture_output=True, timeout=60, env=environment(2, LLAMA_KV_LAYOUT='columns'))
        self.assertEqual(result.returncode, 1, result.stderr.decode())
        self.assertIn('LLAMA_KV_LAYOUT must be rows or head', result.stderr.decode())

    def test_incompatible_configurations_are_rejected(self):
        model = self.one_kv/'fp32.bin'
        cases = [
            (model, environment(2), 'n_kv_heads=1'),
            (self.fixtures['shared']/'fp32.bin', {**environment(2), 'LLAMA_BACKEND': 'omp'}, 'the OpenMP backend was removed'),
            (self.fixtures['shared']/'fp32.bin', {**environment(2), 'LLAMA_SOCKETS': '3'}, 'LLAMA_SOCKETS must be 1 or 2'),
            (self.fixtures['shared']/'fp32.bin', {**environment(2), 'LLAMA_TOPOLOGY': 'virtual:1x3'}, 'allowed CPUs in only 1'),
            (self.fixtures['shared']/'fp32.bin', {**environment(2), 'LLAMA_PLACEMENT': 'hbm'}, 'virtual'),
        ]
        cases.append((self.fixtures['gs16']/'int8.bin', environment(2), 'group size 16 does not divide'))
        for model, env, text in cases:
            with self.subTest(text=text):
                probe = 'probe-int8' if str(model).endswith('int8.bin') else 'probe-fp32'
                result = subprocess.run([str(BUILD/probe), str(model), '65'], capture_output=True, timeout=60, env=env)
                self.assertEqual(result.returncode, 1, result.stderr.decode())
                self.assertIn(text, result.stderr.decode())

    # ---- the merge of the planned placement with the memory users added on develop (docs/hbm-ddr-placement.md) ------

    def plan(self, binary, fixture, filename, sockets=2, **knobs):
        """--plan-only under a planned placement on the virtual topology: (result, parsed report)."""
        report = Path(self.temp.name)/f'plan-{binary}-{fixture}-{sockets}.json'
        report.unlink(missing_ok=True)
        result = subprocess.run([str(BUILD/binary), str(self.fixtures[fixture]/filename), '--plan-only', '--json',
                                 str(report)], capture_output=True, text=True, timeout=30,
                                env=environment(sockets, **knobs))
        return result, (json.loads(report.read_text()) if report.exists() else None)

    @staticmethod
    def arenas(report, role=None):
        return [a for a in report['placement_plan']['arenas'] if role is None or a['role'] == role]

    def test_planned_bf16_kv_copy_has_arenas_in_both_budgets(self):
        """Hazard 1: the persistent BF16 K/V copy of LLAMA_ATTN=amx is an arena of its own (role kv.bf16), tiered per layer
        with the caches, counted in the charge of the node it sits on, and left out of the work region."""
        self.need('runq-amx-emul')
        amx = dict(LLAMA_KERNELS='amx', LLAMA_ATTN='amx', LLAMA_PREFILL_BATCH='4')
        cfg = TP_WIDE_CONFIG  # dim, hidden, layers, heads, kv_heads, vocab, seq
        hd, kl = cfg[0] // cfg[3], cfg[4] // 2
        layer_bytes = 2 * kl * ((cfg[6] + 31) // 32) * 32 * hd * 2
        for placement, knobs, hbm_layers in (('hbm-only', {}, 2), ('ddr', {}, 0),
                                             ('hybrid', dict(LLAMA_HYBRID_KV_HBM_LAYERS='1'), 1),
                                             ('hybrid', dict(LLAMA_HYBRID_KV_HBM_LAYERS='0'), 0),
                                             ('hybrid', dict(LLAMA_HYBRID_KV_HBM_LAYERS='2'), 2)):
            with self.subTest(placement=placement, hbm_layers=hbm_layers):
                result, report = self.plan('runq-amx-emul', 'wide', 'int8.bin', LLAMA_PLACEMENT=placement, **amx, **knobs)
                self.assertEqual(result.returncode, 0, result.stderr)
                plan = report['placement_plan']
                self.assertEqual(plan['kv_bf16_layer_bytes'], layer_bytes)
                copies = self.arenas(report, 'kv.bf16')
                for socket in (0, 1):
                    mine = {a['tier']: a for a in copies if a['socket'] == socket}
                    self.assertEqual({t: a['bytes'] for t, a in mine.items()},
                                     {t: n * layer_bytes for t, n in (('hbm', hbm_layers), ('ddr', 2 - hbm_layers)) if n})
                    kv = {a['tier']: a['bytes'] for a in self.arenas(report, 'kv.k') if a['socket'] == socket}
                    for tier, arena in mine.items():  # the copy sits where its layers' caches sit
                        self.assertEqual(arena['bytes'] // layer_bytes, kv[tier] // plan['kv_layer_bytes'])
                # every node is charged exactly the rounded arenas on it, the BF16 copy included
                for node in plan['nodes']:
                    rounded = sum(a['rounded_bytes'] for a in self.arenas(report)
                                  if a['socket'] == node['socket'] and a['tier'] == node['tier'])
                    self.assertEqual(node['charged_bytes'], rounded)
                # the work region no longer carries it: the same plan with the copy off has the same work arenas
                _, plain = self.plan('runq-amx-emul', 'wide', 'int8.bin', LLAMA_PLACEMENT=placement,
                                     **{**amx, 'LLAMA_ATTN': 'amx', 'LLAMA_KERNELS': 'amx'}, **knobs)
                self.assertEqual([a['bytes'] for a in self.arenas(plain, 'work')],
                                 [a['bytes'] for a in self.arenas(report, 'work')])
        # both budgets: dropping the copy's rounded bytes from a node's charge as its budget must refuse the plan
        result, report = self.plan('runq-amx-emul', 'wide', 'int8.bin', LLAMA_PLACEMENT='hybrid',
                                   LLAMA_HYBRID_KV_HBM_LAYERS='1', **amx)
        hbm = max(n['charged_bytes'] for n in report['placement_plan']['nodes'] if n['tier'] == 'hbm')
        copy = max(a['rounded_bytes'] for a in self.arenas(report, 'kv.bf16') if a['tier'] == 'hbm')
        self.assertGreater(copy, 0)
        for budget, ok in ((hbm, True), (hbm - copy, False)):
            result, _ = self.plan('runq-amx-emul', 'wide', 'int8.bin', LLAMA_PLACEMENT='hbm-only',
                                  LLAMA_HBM_BUDGET_GIB=str(budget / (1 << 30)), **amx)
            if ok:
                continue
            self.assertNotEqual(result.returncode, 0)
        ddr = max(n['charged_bytes'] for n in report['placement_plan']['nodes'] if n['tier'] == 'ddr')
        copy = max(a['rounded_bytes'] for a in self.arenas(report, 'kv.bf16') if a['tier'] == 'ddr')
        result, _ = self.plan('runq-amx-emul', 'wide', 'int8.bin', LLAMA_PLACEMENT='hybrid',
                              LLAMA_HYBRID_KV_HBM_LAYERS='1', LLAMA_DDR_BUDGET_GIB=str((ddr - copy) / (1 << 30)), **amx)
        self.assertNotEqual(result.returncode, 0)
        self.assertIn('over its', result.stderr)
        # without LLAMA_ATTN=amx there is no copy and no arena
        _, tiled = self.plan('runq-amx-emul', 'wide', 'int8.bin', LLAMA_PLACEMENT='hbm-only', LLAMA_KERNELS='amx',
                             LLAMA_ATTN='tiled')
        self.assertEqual(self.arenas(tiled, 'kv.bf16'), [])
        self.assertEqual(tiled['placement_plan']['kv_bf16_layer_bytes'], 0)

    def test_planned_bf16_kv_copy_follows_its_layer_across_tiers(self):
        """Hazard 1, results: each layer's BF16 copy lives in the tier of its caches; every split gives the logits of the
        unplanned run bitwise (the copy is keyed by the layer's cache, not by the order the layers first run in), and
        the report carries the copy."""
        self.need('probe-int8-amx-emul')
        tokens = [65 + (i * 7) % 20 for i in range(33)]
        amx = dict(tokens=tokens, binary='probe-int8-amx-emul', LLAMA_KERNELS='amx', LLAMA_ATTN='amx',
                   LLAMA_PREFILL_LOGITS='all', LLAMA_PREFILL_BATCH='5')
        for sockets in (1, 2):
            amx['LLAMA_TOPOLOGY'] = f'virtual:{sockets}x3'
            reference = self.probe('int8', 'wide', 'batch', sockets, **amx)
            for kv in ('0', '1', '2'):
                for layout in ('head', 'rows'):
                    with self.subTest(sockets=sockets, kv_hbm_layers=kv, layout=layout):
                        actual = self.probe('int8', 'wide', 'batch', sockets, LLAMA_PLACEMENT='hybrid',
                                            LLAMA_HYBRID_WEIGHTS_HBM_LAYERS='1', LLAMA_HYBRID_KV_HBM_LAYERS=kv,
                                            LLAMA_KV_LAYOUT=layout, **amx)
                        if layout == 'head':
                            self.assertEqual(reference, actual)
                        else:  # the rows layout is its own reference
                            self.assertEqual(self.probe('int8', 'wide', 'batch', sockets, LLAMA_KV_LAYOUT=layout, **amx),
                                             actual)
        data = self.report('runq', 'int8.bin', fixture='wide', binary='runq-amx-emul', full=True, LLAMA_KERNELS='amx',
                           LLAMA_ATTN='amx', LLAMA_PREFILL_BATCH='4', LLAMA_PLACEMENT='hybrid',
                           LLAMA_HYBRID_KV_HBM_LAYERS='1', prompt='ABCDEFGHIJ')
        copy = data['kv']['bf16_copy']
        plan = data['placement_plan']
        self.assertEqual(copy['reserved_bytes'], 2 * 2 * plan['kv_bf16_layer_bytes'])
        self.assertGreater(copy['active_bytes'], 0)
        self.assertLessEqual(copy['active_bytes'], copy['reserved_bytes'])

    def test_per_channel_checkpoint_is_planned_and_run(self):
        """Hazard 2: a format 4 checkpoint (gs 0) is planned without dividing by the group size: the work arena holds two
        scales per padded row (one per product input), all three placements are admitted, and the planned runs equal the
        unplanned ones bitwise, on one and two sockets, on the scalar kernels and on the AMX tile model."""
        cfg = TP_WIDE_CONFIG
        for binary, kernels in (('runq', 'scalar'), ('runq-amx-emul', 'amx')):
            self.need(binary)
            for placement in ('hbm-only', 'ddr', 'hybrid'):
                with self.subTest(binary=binary, placement=placement):
                    result, report = self.plan(binary, 'wide-pc', 'int8.bin', LLAMA_PLACEMENT=placement,
                                               LLAMA_KERNELS=kernels, LLAMA_PREFILL_BATCH='32')
                    self.assertEqual(result.returncode, 0, result.stderr)
                    self.assertTrue(report['placement_plan']['admitted'])
                    work = self.arenas(report, 'work')[0]['bytes']
                    _, gs64 = self.plan(binary, 'wide', 'int8.bin', LLAMA_PLACEMENT=placement, LLAMA_KERNELS=kernels,
                                        LLAMA_PREFILL_BATCH='32')
                    # (dim/gs + slice/gs) scales per padded row of 32 for gs 64, against 2 for per-channel
                    groups = cfg[0] // 64 + max(cfg[0], cfg[1]) // 2 // 64
                    self.assertEqual(self.arenas(gs64, 'work')[0]['bytes'] - work, 32 * (groups - 2) * 4)
        probes = dict(tokens=[65 + (i * 7) % 20 for i in range(33)], LLAMA_PREFILL_LOGITS='all', LLAMA_PREFILL_BATCH='8')
        for sockets in (1, 2):
            for binary, extra in (('probe-int8', {'LLAMA_KERNELS': 'scalar'}),
                                  ('probe-int8-amx-emul', {'LLAMA_KERNELS': 'amx'})):
                self.need(binary)
                extra = dict(extra, LLAMA_TOPOLOGY=f'virtual:{sockets}x3')
                packed = dict(LLAMA_HYBRID_PACKED='hbm' if sockets == 2 else 'ddr') if 'amx' in binary else {}
                reference = self.probe('int8', 'wide-pc', 'batch', sockets, binary=binary, **probes, **extra)
                for tier in (dict(LLAMA_PLACEMENT='hbm-only'), dict(LLAMA_PLACEMENT='ddr'),
                             dict(LLAMA_PLACEMENT='hybrid', LLAMA_HYBRID_WEIGHTS_HBM_LAYERS='1',
                                  LLAMA_HYBRID_KV_HBM_LAYERS='1', **packed)):
                    with self.subTest(sockets=sockets, binary=binary, **tier):
                        self.assertEqual(reference, self.probe('int8', 'wide-pc', 'batch', sockets, binary=binary,
                                                               **probes, **extra, **tier))

    def test_hybrid_packed_tier_with_every_group_size(self):
        """Hazard 3: LLAMA_HYBRID_PACKED=hbm|ddr plans and runs the AMX-INT8 packed copies (and their transposed scale
        streams) of group size 64, group size 32 (half tiles) and per-channel files: the packed arena sits on the
        requested tier with the same size in both, and the logits equal the unplanned run bitwise."""
        self.need('probe-int8-amx-emul')
        tokens = [65 + (i * 7) % 20 for i in range(33)]
        for fixture in ('wide', 'wide-gs32', 'wide-pc'):
            sizes = {}
            for tier in ('hbm', 'ddr'):
                result, report = self.plan('runq-amx-emul', fixture, 'int8.bin', LLAMA_PLACEMENT='hybrid',
                                           LLAMA_HYBRID_KV_HBM_LAYERS='1', LLAMA_HYBRID_PACKED=tier, LLAMA_KERNELS='amx',
                                           LLAMA_PREFILL_BATCH='32')
                self.assertEqual(result.returncode, 0, result.stderr)
                packed = self.arenas(report, 'packed')
                self.assertEqual({a['tier'] for a in packed}, {tier})
                self.assertGreater(min(a['bytes'] for a in packed), 0)
                sizes[tier] = sorted(a['bytes'] for a in packed)
            self.assertEqual(sizes['hbm'], sizes['ddr'], fixture)
            for sockets in (1, 2):
                args = dict(tokens=tokens, binary='probe-int8-amx-emul', LLAMA_KERNELS='amx', LLAMA_PREFILL_BATCH='8',
                            LLAMA_PREFILL_LOGITS='all', LLAMA_TOPOLOGY=f'virtual:{sockets}x3')
                reference = self.probe('int8', fixture, 'batch', sockets, **args)
                for tier in ('hbm', 'ddr'):
                    with self.subTest(fixture=fixture, sockets=sockets, packed=tier):
                        self.assertEqual(reference, self.probe('int8', fixture, 'batch', sockets,
                                                               LLAMA_PLACEMENT='hybrid', LLAMA_HYBRID_PACKED=tier,
                                                               LLAMA_HYBRID_WEIGHTS_HBM_LAYERS='1',
                                                               LLAMA_HYBRID_KV_HBM_LAYERS='1', **args))

    def test_fused_norm_quantizer_runs_with_a_split_kv_region(self):
        """Hazard 4: the fused RMSNorm + tile-order quantizer (develop) and the KV accessor (the branch) are both in the
        batched prefill loop. With the caches split over two tiers the pipelined (fused) and chunk (unfused) kernels give
        the same logits bitwise, as the unplanned fused run does."""
        self.need('probe-int8-amx-emul')
        tokens = [65 + (i * 7) % 20 for i in range(33)]
        for fixture in ('wide', 'wide-gs32', 'wide-pc'):
            for sockets in (1, 2):
                args = dict(tokens=tokens, binary='probe-int8-amx-emul', LLAMA_KERNELS='amx', LLAMA_PREFILL_BATCH='8',
                            LLAMA_PREFILL_LOGITS='all', LLAMA_TOPOLOGY=f'virtual:{sockets}x3')
                split = dict(LLAMA_PLACEMENT='hybrid', LLAMA_HYBRID_KV_HBM_LAYERS='1', LLAMA_HYBRID_WEIGHTS_HBM_LAYERS='1')
                with self.subTest(fixture=fixture, sockets=sockets):
                    fused = self.probe('int8', fixture, 'batch', sockets, **args)
                    self.assertEqual(fused, self.probe('int8', fixture, 'batch', sockets, **split, **args))
                    self.assertEqual(fused, self.probe('int8', fixture, 'batch', sockets, LLAMA_Q8_GEMM='chunk',
                                                       **split, **args))
        source = (Path(__file__).resolve().parent.parent/'src/tp.c').read_text()
        body = source[source.index('static void forward_batch('):source.index('One slice of a region, zeroed')]
        self.assertEqual(body.count('norm_quantize_batch(tp, sk, s, norm_weight('), 2)  # attention and FFN norms
        self.assertIn('KvLayer cache = kv_layer(sk, l);', body)

    def test_no_raw_kv_indexing_outside_the_kv_table(self):
        """Hazard 5: every K/V cache access goes through kv_layer(); the per-socket key/value pointers are gone, and
        the only place that turns a layer number into an offset is the table's construction."""
        import re
        source = (Path(__file__).resolve().parent.parent/'src/tp.c').read_text()
        self.assertEqual(re.findall(r'\b(?:key|value)_cache\b', source), [])
        self.assertEqual(len(re.findall(r'kv_layer\(sk, l\)', source)), 3)  # per-token fp32, per-token q8, batched
        lines = source.splitlines()
        offsets = [i + 1 for i, line in enumerate(lines)
                   if re.search(r'\bl \* layer_floats|\(tier == HBM \? l : l - cut\)', line)]
        self.assertEqual(len(offsets), 3, offsets)  # the legacy table, and the planned one for K/V and the BF16 copy
        for use in re.finditer(r'sk->kv_layers\[', source):
            line = source.count('\n', 0, use.start()) + 1
            self.assertTrue(any(abs(line - o) < 12 for o in offsets) or 'kv_layers[0]' in lines[line - 1]
                            or 'kv_layers[l]' in lines[line - 1] or 'kv_layers[layer]' in lines[line - 1], lines[line - 1])

    def test_planned_split_kv_prefault_and_exchange(self):
        """Hazard 6: the planned path creates what develop's legacy path creates: the team writes every KV arena (both
        tiers) at setup, the exchange arena has the interleaved reduction's flags and each socket's size, and the
        interleaved and barrier reductions agree bitwise under a split plan. Each `plan.mode` guard of tp.c is listed."""
        self.need('probe-int8-amx-emul')
        tokens = [65 + (i * 7) % 20 for i in range(33)]
        split = dict(LLAMA_PLACEMENT='hybrid', LLAMA_HYBRID_KV_HBM_LAYERS='1', LLAMA_HYBRID_WEIGHTS_HBM_LAYERS='1')
        args = dict(tokens=tokens, binary='probe-int8-amx-emul', LLAMA_KERNELS='amx', LLAMA_PREFILL_BATCH='8',
                    LLAMA_PREFILL_LOGITS='all', **split)
        runs = {mode: self.probe('int8', 'wide', 'batch', 2, LLAMA_TP_BATCH_REDUCE=mode, **args)
                for mode in ('barrier', 'wo', 'interleaved')}
        self.assertEqual(runs['barrier'], runs['wo'])
        self.assertEqual(runs['barrier'], runs['interleaved'])
        whole = {k: v for k, v in args.items() if not k.startswith('LLAMA_HYBRID')} | dict(LLAMA_PLACEMENT='hbm-only')
        self.assertEqual(runs['barrier'], self.probe('int8', 'wide', 'batch', 2, LLAMA_TP_BATCH_REDUCE='interleaved',
                                                     **whole))
        result, report = self.plan('runq-amx-emul', 'wide', 'int8.bin', LLAMA_KERNELS='amx', LLAMA_PREFILL_BATCH='32',
                                   **split)
        self.assertEqual(result.returncode, 0, result.stderr)
        exchange = self.arenas(report, 'exchange')
        self.assertEqual([a['bytes'] for a in exchange], [3 * 32 * TP_WIDE_CONFIG[0] * 4] * 2)
        source = (Path(__file__).resolve().parent.parent/'src/tp.c').read_text()
        guards = [line.strip() for line in source.splitlines() if 'tp->plan.mode' in line]
        # setup_socket (legacy 2S work memset; BF16 copy carve; planned BF16 slots; planned KV spans),
        # work_bytes (the copy is counted in the work region only without a plan), CMD_SETUP (the plan's zeroing),
        # tp_create_planned (mode), its verification loop (twice), tp_all_logits (admission), tp_kv_stats, tp_free (x2).
        self.assertEqual(guards, [
            'if (S == 2 && !tp->plan.mode)',
            'if (kv_bf16_bytes && !tp->plan.mode)',
            'else if (kv_bf16_bytes && tp->plan.mode) {',
            'if (tp->plan.mode) {',
            'return work_bytes_for(&tp->c, tp->sockets, tp->quantized, tp->gs, tp->rows, 0, !tp->plan.mode);',
            'if (tp->plan.mode) {',
            'tp->plan.mode = placement_mode();',
            'verify_bound(r, tp->plan.mode);',
            'verify_bound(r, tp->plan.mode);',
            'if (tp->plan.mode)',
            'if (tp->plan.mode) {',
            'if (tp->sockets == 1 && !tp->plan.mode) {',
            'if (!tp->plan.mode) {',
        ])

    # ---- LLAMA_HYBRID_SPLIT=rows (docs/concurrent-tiers.md) ----------------------------------------------------------

    ROLE_ORDER = ('wq', 'wk', 'wv', 'w1', 'w3', 'wo', 'w2')  # the order of a layer's shards in a socket's arenas

    def split_shapes(self, sockets, gs):
        """(rows, cols, groups) of the seven large shards of one socket of the wide fixture."""
        dim, hidden, _, heads, kv_heads = TP_WIDE_CONFIG[:5]
        kv = dim // heads * kv_heads // sockets
        gs = gs or 0
        shapes = {'wq': (dim // sockets, dim), 'wk': (kv, dim), 'wv': (kv, dim), 'w1': (hidden // sockets, dim),
                  'w3': (hidden // sockets, dim), 'wo': (dim, dim // sockets), 'w2': (dim, hidden // sockets)}
        return {name: (rows, cols, cols // gs if gs else 1) for name, (rows, cols) in shapes.items()}

    @staticmethod
    def ddr_rows(rows, ppm):
        """shards_ddr_rows: the fraction of the rows rounded down to 32, at most 15/16 of the rows."""
        n = rows * ppm // 1000000
        n -= n % 32
        most = rows * 15 // 16
        return most - most % 32 if n > most else n

    def expected_ddr_weights(self, sockets, gs, ppm):
        """The DDR weights arena of one socket for a fraction: every split part's values and scales, 64-byte aligned in
        shard order (recomputed here, independently of the planner)."""
        shapes, cursor = self.split_shapes(sockets, gs), 0
        for _ in range(TP_WIDE_CONFIG[2]):
            for name in self.ROLE_ORDER:
                rows, cols, groups = shapes[name]
                ddr = self.ddr_rows(rows, ppm)
                if ddr:
                    cursor = (cursor + 63) // 64 * 64 + ddr * cols
                    cursor = (cursor + 63) // 64 * 64 + ddr * groups * 4
        return cursor

    def test_row_split_decode_and_prefill_are_bitwise_the_unsplit_run(self):
        """LLAMA_HYBRID_SPLIT=rows reads each output row exactly as before, from either arena: the logits of a batched
        prefill (packed AMX tiles, their transposed scales, the barrier-form Wo/W2 reduction and the split GEMM calls) and
        of greedy decode (the team GEMV with members assigned to the DDR rows) are bitwise those of hbm-only, for every
        group size, per-channel files, one and two sockets, several fractions and DDR core counts."""
        tokens = [65 + (i * 7) % 20 for i in range(33)]
        for binary, kernels in (('probe-int8', 'scalar'), ('probe-int8-amx-emul', 'amx')):
            self.need(binary)
            for fixture in ('wide', 'wide-gs32', 'wide-pc'):
                for sockets in (1, 2):
                    common = dict(binary=binary, LLAMA_KERNELS=kernels, LLAMA_PREFILL_BATCH='8',
                                  LLAMA_PREFILL_LOGITS='all', LLAMA_TOPOLOGY=f'virtual:{sockets}x3')
                    reference = {mode: self.probe('int8', fixture, mode, sockets, tokens=tokens,
                                                  LLAMA_PLACEMENT='hbm-only', **common)
                                 for mode in ('batch', 'generate')}
                    for fraction, cores in (('0.25', None), ('0.5', '1'), ('0.9', '2'), ('auto', '2')):
                        knobs = dict(LLAMA_PLACEMENT='hybrid', LLAMA_HYBRID_SPLIT='rows',
                                     LLAMA_HYBRID_DDR_FRACTION=fraction, **common)
                        if cores:
                            knobs['LLAMA_HYBRID_DDR_CORES'] = cores
                        for mode in ('batch', 'generate'):
                            with self.subTest(binary=binary, fixture=fixture, sockets=sockets, fraction=fraction,
                                              cores=cores, mode=mode):
                                self.assertEqual(reference[mode], self.probe('int8', fixture, mode, sockets,
                                                                             tokens=tokens, **knobs))

    def test_row_split_with_fp32_activations_and_the_stagger(self):
        """W8A32 decode (the FP32-activation kernel) and the staggered GEMV are bitwise the unsplit run too."""
        tokens = [65 + (i * 7) % 20 for i in range(9)]
        for sockets in (1, 2):
            for extra in (dict(LLAMA_Q8_ACTIVATIONS='fp32'), dict(LLAMA_GEMV_STAGGER='1')):
                common = dict(LLAMA_KERNELS='scalar', LLAMA_TOPOLOGY=f'virtual:{sockets}x3', **extra)
                with self.subTest(sockets=sockets, **extra):
                    reference = self.probe('int8', 'wide', 'generate', sockets, tokens=tokens,
                                           LLAMA_PLACEMENT='hbm-only', **common)
                    self.assertEqual(reference, self.probe('int8', 'wide', 'generate', sockets, tokens=tokens,
                                                           LLAMA_PLACEMENT='hybrid', LLAMA_HYBRID_SPLIT='rows',
                                                           LLAMA_HYBRID_DDR_FRACTION='0.5', **common))

    def test_row_split_plan_arithmetic(self):
        """The planner's DDR weights arena is exactly the split rows' values and scales, the HBM weights arena shrinks by the
        same rows, nothing else moves, `auto` takes the smallest fraction that fits the budget (and the measured balance of
        docs/concurrent-tiers.md when everything fits), and the report names the split."""
        self.need('runq-amx-emul')
        args = dict(LLAMA_KERNELS='amx', LLAMA_PREFILL_BATCH='32')
        for fixture, gs in (('wide', 64), ('wide-gs32', 32), ('wide-pc', 0)):
            for sockets in (1, 2):
                base = dict(args, LLAMA_PLACEMENT='hybrid', LLAMA_HYBRID_KV_HBM_LAYERS='2', LLAMA_TOPOLOGY=f'virtual:{sockets}x3')
                result, whole = self.plan('runq-amx-emul', fixture, 'int8.bin', sockets, **base)
                self.assertEqual(result.returncode, 0, result.stderr)
                self.assertEqual(whole['placement_plan']['split'], 'layers')
                self.assertEqual(self.arenas(whole, 'weights')[0]['tier'], 'hbm')
                hbm_whole = {a['socket']: a['bytes'] for a in self.arenas(whole, 'weights') if a['tier'] == 'hbm'}
                charged = {(n['socket'], n['tier']): n['charged_bytes'] for n in whole['placement_plan']['nodes']}
                for fraction in ('0.25', '0.5', '0.9'):
                    ppm = round(float(fraction) * 1e6)
                    with self.subTest(fixture=fixture, sockets=sockets, fraction=fraction):
                        result, report = self.plan('runq-amx-emul', fixture, 'int8.bin', sockets,
                                                   LLAMA_HYBRID_SPLIT='rows', LLAMA_HYBRID_DDR_FRACTION=fraction,
                                                   LLAMA_HYBRID_DDR_CORES='2', **base)
                        self.assertEqual(result.returncode, 0, result.stderr)
                        plan = report['placement_plan']
                        self.assertEqual((plan['split'], plan['ddr_cores'], plan['ddr_fraction_auto']), ('rows', 2, False))
                        self.assertAlmostEqual(plan['ddr_fraction'], float(fraction))
                        want = self.expected_ddr_weights(sockets, gs, ppm)
                        ddr = [a for a in self.arenas(report, 'weights') if a['tier'] == 'ddr']
                        self.assertEqual([a['bytes'] for a in ddr], [want] * sockets)
                        hbm = {a['socket']: a['bytes'] for a in self.arenas(report, 'weights') if a['tier'] == 'hbm'}
                        for s in range(sockets):
                            self.assertLess(hbm[s], hbm_whole[s])
                        # every other arena is the layer-split plan's, and the charges move with the weights only
                        other = lambda r: sorted((a['socket'], a['tier'], a['role'], a['bytes'])
                                                 for a in self.arenas(r) if a['role'] != 'weights')
                        self.assertEqual(other(report), other(whole))
                        nodes = {(n['socket'], n['tier']): n['charged_bytes'] for n in plan['nodes']}
                        for s in range(sockets):
                            self.assertLess(nodes[(s, 'hbm')], charged[(s, 'hbm')])
                            self.assertGreater(nodes[(s, 'ddr')], charged[(s, 'ddr')])
                # auto, everything fits: the balance of docs/concurrent-tiers.md
                result, report = self.plan('runq-amx-emul', fixture, 'int8.bin', sockets, LLAMA_HYBRID_SPLIT='rows', **base)
                self.assertEqual(result.returncode, 0, result.stderr)
                self.assertEqual((report['placement_plan']['ddr_fraction'], report['placement_plan']['ddr_fraction_auto']),
                                 (0.2, True))
                # auto, a budget below what hbm-only needs: the smallest fraction (to 0.1%) that fits, and none smaller
                gib = lambda n: str(n / (1 << 30))
                for shave in (1, 20000):
                    budget = max(charged[(s, 'hbm')] for s in range(sockets)) - shave
                    result, report = self.plan('runq-amx-emul', fixture, 'int8.bin', sockets, LLAMA_HYBRID_SPLIT='rows',
                                               LLAMA_HBM_BUDGET_GIB=gib(budget), **base)
                    with self.subTest(fixture=fixture, sockets=sockets, budget=budget):
                        self.assertEqual(result.returncode, 0, result.stderr)
                        plan = report['placement_plan']
                        self.assertTrue(plan['ddr_fraction_auto'] and plan['admitted'])
                        self.assertGreater(plan['ddr_fraction'], 0)
                        self.assertTrue(all(n['charged_bytes'] <= n['budget_bytes'] for n in plan['nodes']))
                        below = plan['ddr_fraction'] - 0.001
                        if below > 0:
                            result, _ = self.plan('runq-amx-emul', fixture, 'int8.bin', sockets, LLAMA_HYBRID_SPLIT='rows',
                                                  LLAMA_HYBRID_DDR_FRACTION=f'{below:.3f}', LLAMA_HBM_BUDGET_GIB=gib(budget),
                                                  **base)
                            if self.expected_ddr_weights(sockets, gs, round(below * 1e6)) != \
                               self.expected_ddr_weights(sockets, gs, round(plan['ddr_fraction'] * 1e6)):
                                self.assertNotEqual(result.returncode, 0)
                                self.assertIn('over its', result.stderr)
                # the DDR budget bounds the DDR share
                result, _ = self.plan('runq-amx-emul', fixture, 'int8.bin', sockets, LLAMA_HYBRID_SPLIT='rows',
                                      LLAMA_HYBRID_DDR_FRACTION='0.5', LLAMA_DDR_BUDGET_GIB='0.0001', **base)
                self.assertNotEqual(result.returncode, 0)
                self.assertIn('over its', result.stderr)

    def test_row_split_knob_parsing_and_refusals(self):
        self.need('runq-amx-emul')
        base = dict(LLAMA_KERNELS='amx', LLAMA_PLACEMENT='hybrid', LLAMA_HYBRID_KV_HBM_LAYERS='2')

        def refused(text, binary='runq-amx-emul', fixture='wide', filename='int8.bin', **knobs):
            result, _ = self.plan(binary, fixture, filename, **{**base, **knobs})
            self.assertNotEqual(result.returncode, 0, knobs)
            self.assertIn(text, result.stderr, knobs)

        def accepted(**knobs):
            result, report = self.plan('runq-amx-emul', 'wide', 'int8.bin', **{**base, **knobs})
            self.assertEqual(result.returncode, 0, result.stderr)
            return report['placement_plan']

        refused('must be layers or rows', LLAMA_HYBRID_SPLIT='bogus')
        refused('requires LLAMA_HYBRID_SPLIT=rows', LLAMA_HYBRID_DDR_FRACTION='0.2')
        refused('requires LLAMA_HYBRID_SPLIT=rows', LLAMA_HYBRID_SPLIT='layers', LLAMA_HYBRID_DDR_CORES='2')
        refused('requires LLAMA_PLACEMENT=hybrid', LLAMA_PLACEMENT='hbm-only', LLAMA_HYBRID_SPLIT='rows')
        refused('requires LLAMA_PLACEMENT=hybrid', LLAMA_PLACEMENT='ddr', LLAMA_HYBRID_DDR_FRACTION='0.2')
        refused('needs a Q8_0 checkpoint', binary='run', fixture='unshared', filename='fp32.bin', LLAMA_KERNELS='scalar',
                LLAMA_HYBRID_SPLIT='rows')
        refused('cannot be combined with LLAMA_HYBRID_SPLIT=rows', LLAMA_HYBRID_SPLIT='rows',
                LLAMA_HYBRID_WEIGHTS_HBM_LAYERS='1')
        for value in ('1.5', '-0.1', 'abc', '0.95', '0.2x', 'nan'):
            refused('LLAMA_HYBRID_DDR_FRACTION must be auto or a fraction', LLAMA_HYBRID_SPLIT='rows',
                    LLAMA_HYBRID_DDR_FRACTION=value)
        for value in ('0', '-1', 'x', '3', '2.5'):  # three threads per socket: 1 or 2 members on the DDR rows
            refused('LLAMA_HYBRID_DDR_CORES must be a whole number between 1 and 2', LLAMA_HYBRID_SPLIT='rows',
                    LLAMA_HYBRID_DDR_CORES=value)
        plan = accepted(LLAMA_HYBRID_SPLIT='layers')
        self.assertEqual((plan['split'], plan['ddr_fraction'], plan['ddr_cores']), ('layers', 0.0, 0))
        plan = accepted(LLAMA_HYBRID_SPLIT='rows', LLAMA_HYBRID_DDR_FRACTION='0')
        self.assertEqual((plan['split'], plan['ddr_fraction'], plan['ddr_fraction_auto']), ('rows', 0.0, False))
        plan = accepted(LLAMA_HYBRID_SPLIT='rows', LLAMA_HYBRID_DDR_FRACTION='0.9', LLAMA_HYBRID_DDR_CORES='1')
        self.assertEqual((plan['ddr_fraction'], plan['ddr_cores']), (0.9, 1))
        plan = accepted(LLAMA_HYBRID_SPLIT='rows', LLAMA_HYBRID_KV_HBM_LAYERS='1', LLAMA_HYBRID_DDR_FRACTION='.3')
        self.assertEqual(plan['kv_hbm_layers'], 1)
        # a team of three serves the DDR rows with one member by default (14 of 56, scaled, at least one)
        self.assertEqual(accepted(LLAMA_HYBRID_SPLIT='rows')['ddr_cores'], 1)
        # auto KV with an auto fraction keeps every layer's caches in HBM; with an explicit fraction it is the usual prefix
        self.assertEqual(accepted(LLAMA_HYBRID_SPLIT='rows', LLAMA_HYBRID_KV_HBM_LAYERS='auto')['kv_hbm_layers'], 2)
        self.assertEqual(accepted(LLAMA_HYBRID_SPLIT='rows', LLAMA_HYBRID_DDR_FRACTION='0.2',
                                  LLAMA_HYBRID_KV_HBM_LAYERS='auto')['kv_hbm_layers'], 2)

    def test_hbm_ddr_script_follows_engine_defaults(self):
        """Hazard 7: scripts/hbm-ddr.sbatch forces no kernel, batch, attention, layout or page knob unless asked; LEGACY=1
        and the named options bring the old values back."""
        script = Path(__file__).resolve().parent.parent/'scripts/hbm-ddr.sbatch'
        base = {k: v for k, v in os.environ.items() if not k.startswith('LLAMA_')}
        base.update(ROOT='.', OUT=str(Path(self.temp.name)/'hbm-ddr-out'), CHECKPOINT='c', TOKENIZER='t', DRY_RUN='1')

        def knobs(**extra):
            result = subprocess.run(['bash', str(script)], env={**base, **extra}, capture_output=True, text=True, timeout=30)
            self.assertEqual(result.returncode, 0, result.stderr)
            return dict(line.split('=', 1) for line in result.stdout.splitlines())

        default = knobs()
        for name in ('KERNELS', 'PREFILL_BATCH', 'PREFILL_LOGITS', 'KV_LAYOUT', 'ATTN', 'ATTN_DECODE', 'THP'):
            self.assertNotIn(f'LLAMA_{name}', default)
        self.assertEqual(knobs(LEGACY='1'), {**default, 'LLAMA_KERNELS': 'avx512', 'LLAMA_PREFILL_BATCH': '64',
                                             'LLAMA_PREFILL_LOGITS': 'last', 'LLAMA_KV_LAYOUT': 'head',
                                             'LLAMA_ATTN': 'tiled', 'LLAMA_ATTN_DECODE': 'split', 'LLAMA_THP': 'off'})
        mixed = knobs(LEGACY='1', KERNELS='amx', PREFILL_BATCH='1024')
        self.assertEqual((mixed['LLAMA_KERNELS'], mixed['LLAMA_PREFILL_BATCH'], mixed['LLAMA_ATTN']),
                         ('amx', '1024', 'tiled'))
        self.assertEqual(knobs(ATTN='amx')['LLAMA_ATTN'], 'amx')


if __name__ == '__main__':
    unittest.main()
