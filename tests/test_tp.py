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

from fixtures import CONFIG, TP_CONFIG, TP_ODD_CONFIG, TP_WIDE_CONFIG, create

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
        cls.fixtures['wide'] = create(root/'wide', config=TP_WIDE_CONFIG, gs=64)
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
        for variant in ({'LLAMA_TP_REDUCE': 'barrier'}, {'LLAMA_TP_FUSED': '0'}):
            with self.subTest(**variant):
                self.assertEqual(reference, self.probe('int8', 'shared', 'batch', 2, tokens=tokens, **knobs, **variant))
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
        knobs = {'LLAMA_KERNELS': 'amx', 'LLAMA_PREFILL_LOGITS': 'all'}
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
        # A group size below 64 cannot use the tile kernel: the VNNI or portable rung runs, with no packed copies.
        data = self.report('runq', 'int8.bin', binary='runq-amx-emul', full=True, LLAMA_KERNELS='amx',
                           LLAMA_PREFILL_BATCH='4')
        self.assertEqual(data['runtime']['kernels_prefill'], 'portable-q8')
        self.assertEqual([r for r in data['regions'] if r['role'] == 'packed'], [])
        # W8A32 never takes the tile kernel.
        data = self.report('runq', 'int8.bin', fixture='wide', binary='runq-amx-emul', full=True, LLAMA_KERNELS='amx',
                           LLAMA_PREFILL_BATCH='4', LLAMA_Q8_ACTIVATIONS='fp32')
        self.assertIsNone(data['runtime']['kernels_prefill'])
        self.assertEqual([r for r in data['regions'] if r['role'] == 'packed'], [])

    def test_fp32_two_socket_batched_prefill_thread_counts_and_last_logits(self):
        knobs = {'LLAMA_PREFILL_GEMM': 'on'}
        tokens = [65 + (i * 5) % 19 for i in range(33)]
        one = self.logits('fp32', 'shared', 'batch', 1, tokens=tokens, LLAMA_PREFILL_BATCH='8', **knobs)
        for threads in (1, 2, 7):
            with self.subTest(threads=threads):
                two = self.logits('fp32', 'shared', 'batch', 2, threads, tokens=tokens, LLAMA_PREFILL_BATCH='8', **knobs)
                self.assert_agree(one, two, TP_CONFIG[5])

    def test_fp32_two_socket_batched_prefill_variants_are_bitwise_equal(self):
        """The reduction forms and the join only change how the sockets synchronize, also for the M x dim exchange; and
        a decode after a batched prefill reads the cache the batch wrote (greedy generation equals one socket's)."""
        knobs = {'LLAMA_PREFILL_GEMM': 'on', 'LLAMA_PREFILL_BATCH': '4'}
        tokens = [65 + (i * 3) % 17 for i in range(13)]
        reference = self.probe('fp32', 'shared', 'batch', 2, tokens=tokens, **knobs)
        for variant in ({'LLAMA_TP_REDUCE': 'barrier'}, {'LLAMA_TP_FUSED': '0'}, {'LLAMA_TEAM_JOIN': 'counter',
                                                                                 'LLAMA_TEAM_FANOUT': '0'}):
            with self.subTest(**variant):
                self.assertEqual(reference, self.probe('fp32', 'shared', 'batch', 2, tokens=tokens, **knobs, **variant))
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

    def test_packed4_decode_layout_agrees_with_the_row_layout(self):
        """B7: LLAMA_FP32_LAYOUT=packed4 streams a second, four-rows-per-stream copy of the FP32 shards in decode (the shards
        stay for the batched GEMM). Same arithmetic per row, so the logits agree to the kernel budget (the AVX-512 kernels
        to the bit, which the cluster gate checks), for every thread count and scheduling mode, fused or not, with a
        batched prefill in front of the decode, and for greedy generation."""
        for name, vocab in (('shared', TP_CONFIG[5]), ('odd', TP_ODD_CONFIG[5])):
            rows = self.logits('fp32', name, None, 2)
            for threads in (1, 3, 7):
                for sched in ('static', 'dynamic', 'hybrid'):
                    with self.subTest(fixture=name, threads=threads, sched=sched):
                        self.assert_agree(rows, self.logits('fp32', name, None, 2, threads, LLAMA_GEMV_SCHED=sched,
                                                            LLAMA_FP32_LAYOUT='packed4'), vocab)
        for variant in ({'LLAMA_TP_FUSED': '0'}, {'LLAMA_TP_REDUCE': 'barrier'}):
            with self.subTest(**variant):
                self.assert_agree(self.logits('fp32', 'shared', None, 2),
                                  self.logits('fp32', 'shared', None, 2, LLAMA_FP32_LAYOUT='packed4', **variant),
                                  TP_CONFIG[5])
        knobs = {'LLAMA_PREFILL_GEMM': 'on', 'LLAMA_PREFILL_BATCH': '4'}
        tokens = [65 + (i * 3) % 17 for i in range(13)]
        self.assert_agree(self.logits('fp32', 'shared', 'batch', 2, tokens=tokens, **knobs),
                          self.logits('fp32', 'shared', 'batch', 2, tokens=tokens, LLAMA_FP32_LAYOUT='packed4', **knobs),
                          TP_CONFIG[5])
        generated = self.probe('fp32', 'unshared', 'generate', 2)
        self.assertGreaterEqual(len(generated.split()), 8)
        self.assertEqual(generated, self.probe('fp32', 'unshared', 'generate', 2, LLAMA_FP32_LAYOUT='packed4'))

    def test_packed4_regions_and_validation(self):
        data = self.report('run', 'fp32.bin', full=True, LLAMA_FP32_LAYOUT='packed4')
        packed = [r for r in data['regions'] if r['role'] == 'packed']
        self.assertEqual(sorted(r['socket'] for r in packed), [0, 1])
        # Seven matrices per layer: the packed rows of each socket's shards, 16-column blocks, four rows per block row.
        d, h, layers, heads, kvheads = TP_CONFIG[:5]
        kv = d // heads * kvheads
        pack = lambda rows, cols: rows // 4 * (-(-cols // 16)) * 64 * 4
        line = lambda n: -(-n // 64) * 64
        expected = 0
        for _ in range(layers):
            for size in (pack(d // 2, d), pack(kv // 2, d), pack(kv // 2, d), pack(d, d // 2), pack(h // 2, d),
                         pack(d, h // 2), pack(h // 2, d)):
                expected = line(expected) + size
        for region in packed:
            self.assertGreaterEqual(region['bytes'], expected)
        self.assertGreater(data['memory_bytes']['packed_weights_peak'], 0)
        # It composes with the BF16 copies (two packed regions per socket) and is refused for Q8 and bad values.
        self.need('probe-fp32-amx-emul')
        both = self.report('run', 'fp32.bin', binary='run-amx-emul', full=True, LLAMA_FP32_LAYOUT='packed4',
                           LLAMA_PRECISION='bf16', LLAMA_KERNELS='amx', LLAMA_PREFILL_BATCH='4')
        self.assertEqual(len([r for r in both['regions'] if r['role'] == 'packed']), 4)
        for engine, extra, text in (('probe-int8', {'LLAMA_FP32_LAYOUT': 'packed4'}, b'applies to FP32 checkpoints'),
                                    ('probe-fp32', {'LLAMA_FP32_LAYOUT': 'rowz'}, b'LLAMA_FP32_LAYOUT must be rows or packed4')):
            model = self.fixtures['shared']/('int8.bin' if 'int8' in engine else 'fp32.bin')
            result = subprocess.run([str(BUILD/engine), str(model), '65'], capture_output=True, timeout=60,
                                    env=environment(2, 3, **extra))
            self.assertNotEqual(result.returncode, 0)
            self.assertIn(text, result.stderr)

    def test_fused_and_separate_reduction_are_bitwise_equal(self):
        """The fused form (projection and reduction in one team job) only removes a dispatch: it must not change a bit."""
        for precision, activations in (('fp32', None), ('int8', 'int8'), ('int8', 'fp32')):
            extra = {'LLAMA_Q8_ACTIVATIONS': activations} if activations else {}
            for threads in (1, 3, 7):
                with self.subTest(precision=precision, activations=activations, threads=threads):
                    fused = self.probe(precision, 'shared', None, 2, threads, LLAMA_TP_FUSED='1', **extra)
                    separate = self.probe(precision, 'shared', None, 2, threads, LLAMA_TP_FUSED='0', **extra)
                    self.assertEqual(fused, separate)

    def test_reduction_join_and_dispatch_variants_are_bitwise_equal(self):
        """The flag reduction against the two-barrier one, the flag join against the counter, and the dispatch tree
        against the flat one only change how the members synchronize: not a bit of the output."""
        variants = ({'LLAMA_TP_REDUCE': 'barrier', 'LLAMA_TEAM_JOIN': 'counter', 'LLAMA_TEAM_FANOUT': '0'},
                    {'LLAMA_TP_REDUCE': 'flags', 'LLAMA_TEAM_JOIN': 'counter', 'LLAMA_TEAM_FANOUT': '0'},
                    {'LLAMA_TP_REDUCE': 'barrier', 'LLAMA_TEAM_JOIN': 'flags', 'LLAMA_TEAM_FANOUT': '2'},
                    {'LLAMA_TP_REDUCE': 'flags', 'LLAMA_TEAM_JOIN': 'flags', 'LLAMA_TEAM_FANOUT': '2'})
        for precision, activations in (('fp32', None), ('int8', 'int8'), ('int8', 'fp32')):
            extra = {'LLAMA_Q8_ACTIVATIONS': activations} if activations else {}
            for threads in (3, 7):
                reference = self.probe(precision, 'shared', None, 2, threads, **extra)
                for variant in variants:
                    with self.subTest(precision=precision, activations=activations, threads=threads, **variant):
                        self.assertEqual(reference, self.probe(precision, 'shared', None, 2, threads, **variant, **extra))

    def test_synchronization_knobs_are_validated(self):
        model = self.fixtures['shared']/'fp32.bin'
        for name, message in (('LLAMA_TP_REDUCE', b'LLAMA_TP_REDUCE must be flags or barrier'),
                              ('LLAMA_TEAM_JOIN', b'LLAMA_TEAM_JOIN must be flags or counter'),
                              ('LLAMA_TEAM_FANOUT', b'LLAMA_TEAM_FANOUT must be 0 (flat) or a group size up to 64')):
            result = subprocess.run([str(BUILD/'probe-fp32'), str(model), '1'], capture_output=True, timeout=60,
                                    env=environment(2, 3, **{name: 'x'}))
            self.assertNotEqual(result.returncode, 0)
            self.assertIn(message, result.stderr)

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
        for model, env, text in cases:
            with self.subTest(text=text):
                probe = 'probe-int8' if str(model).endswith('int8.bin') else 'probe-fp32'
                result = subprocess.run([str(BUILD/probe), str(model), '65'], capture_output=True, timeout=60, env=env)
                self.assertEqual(result.returncode, 1, result.stderr.decode())
                self.assertIn(text, result.stderr.decode())


if __name__ == '__main__':
    unittest.main()
