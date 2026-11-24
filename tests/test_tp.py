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

from fixtures import CONFIG, TP_CONFIG, TP_ODD_CONFIG, TP_WIDE_CONFIG, create, upgrade_v3

BUILD = Path(sys.argv.pop(1)).resolve()
TOKENS = [65, 66, 67, 68, 69, 70, 71]
ABS, REL = 2e-6, 2e-5   # the fixture tolerance of test_runtime.py


def environment(sockets, threads=3, **extra):
    env = {**os.environ, 'LLAMA_BACKEND': 'teams', 'LLAMA_THREADS': str(threads), 'LLAMA_TP_CHECK': '1', **extra}
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


if __name__ == '__main__':
    unittest.main()
