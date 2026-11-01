"""Real CLI/process tests with dependency-free synthetic FP32/Q8_0 files."""
import json
import math
import os
from pathlib import Path
import shutil
import struct
import subprocess
import sys
import tempfile
import unittest

from fixtures import CONFIG, create, upgrade_v3

BUILD = Path(sys.argv.pop(1)).resolve()


class RuntimeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temp = tempfile.TemporaryDirectory()
        cls.root = Path(cls.temp.name)
        cls.shared = create(cls.root/'shared')
        cls.unshared = create(cls.root/'unshared', shared=False)
        cls.zero = create(cls.root/'zero', zero=True)
        cls.first_stop = create(cls.root/'first-stop', shared=False, stop='first')
        cls.later_stop = create(cls.root/'later-stop', shared=False, stop='after_one')
        cls.reference = json.loads(Path(__file__).with_name('baseline_logits.json').read_text())

    @classmethod
    def tearDownClass(cls):
        cls.temp.cleanup()

    def execute(self, engine='run', *args, fixture=None, success=True, stdin=None, env=None):
        fixture = fixture or self.shared
        report = self.root/'result.json'
        report.unlink(missing_ok=True)
        model = fixture/('fp32.bin' if engine == 'run' else 'int8.bin')
        command = [str(BUILD/engine), str(model), '-z', str(fixture/'tokenizer.bin'),
                   '-i', 'A', '--seed', '123', '-t', '0', '--json', str(report), *map(str,args)]
        result = subprocess.run(command, input=stdin, text=True, capture_output=True, timeout=30,
                                env={**os.environ, **(env or {})})
        if success:
            self.assertEqual(result.returncode, 0, result.stderr)
        else:
            self.assertGreater(result.returncode, 0, result.stderr)
            self.assertIn('[ERROR]', result.stderr)
            self.assertNotIn('AddressSanitizer', result.stderr)
            self.assertNotIn('runtime error:', result.stderr)
        data = json.loads(report.read_text())
        self.assertEqual(data['status'], 'completed' if success else 'error')
        self.assertEqual(data['schema_version'], 2)
        return result, data

    def test_baseline_numerical_regression(self):
        # The OpenMP/serial path, then worker teams with odd counts that stress range splitting.
        backends = [{}] + [{'LLAMA_BACKEND': 'teams', 'LLAMA_THREADS': str(t)} for t in (1, 2, 3, 7)]
        for env in backends:
            for engine, unit, filename in [('run','test-fp32','fp32.bin'),('runq','test-int8','int8.bin')]:
                result = subprocess.run([str(BUILD/unit), str(self.shared/filename)],capture_output=True,text=True,
                                        env={**os.environ, **env}, timeout=60)
                self.assertEqual(result.returncode,0,(env,result.stderr))
                values = list(map(float,result.stdout.splitlines()))
                self.assertEqual(len(values),48)
                for actual, expected in zip(values,self.reference[engine]):
                    self.assertTrue(math.isclose(actual,expected,rel_tol=2e-5,abs_tol=2e-6),(env,engine,actual,expected))
        for flag in ('--overflow','--add-overflow'):
            result = subprocess.run([str(BUILD/'test-fp32'),flag],capture_output=True,text=True)
            self.assertEqual(result.returncode,1)
            self.assertIn('overflow',result.stderr)

    def test_report_schema_v2(self):
        v2 = ('runtime', 'topology', 'topology_error', 'teams', 'regions', 'kv', 'phase_seconds', 'prefill')
        _, data = self.execute('run')
        for key in v2:
            self.assertIn(key, data)
        self.assertEqual(data['runtime'], {'backend': 'omp', 'sockets': 1, 'placement': 'file', 'thp': 'off',
            'precision_mode': 'fp32', 'prefill_batch': 1, 'prefill_logits': 'last', 'kernels_prefill': None})
        self.assertIsNone(data['teams'])
        self.assertEqual(data['regions'], [])
        self.assertIsNone(data['phase_seconds'])
        self.assertEqual(set(data['kv']), {'reserved_bytes', 'resident_bytes', 'active_bytes', 'node', 'verified', 'full',
                                           'pages_queried', 'pages_wrong_node', 'pages_missing', 'after_prompt'})
        self.assertEqual(set(data['prefill']), {'microbatches', 'tokens', 'seconds', 'logits_rows'})
        for key in ('weight_pack', 'placement_populate', 'placement_verify'):
            self.assertIsNone(data['timing_seconds'][key])
        self.assertIn('owned_weights_peak', data['memory_bytes'])
        self.assertIn('packed_weights_peak', data['memory_bytes'])
        _, data = self.execute('runq', env={'LLAMA_Q8_ACTIVATIONS': 'fp32'})
        self.assertEqual(data['runtime']['precision_mode'], 'int8_q8_0_w8a32')
        _, data = self.execute('run', env={'LLAMA_BACKEND': 'teams', 'LLAMA_THREADS': '3',
                                           'LLAMA_TOPOLOGY': 'virtual:1x3', 'LLAMA_THP': 'on'})
        self.assertEqual((data['runtime']['backend'], data['runtime']['thp']), ('teams', 'on'))
        self.assertEqual(data['teams'], [{'index': 0, 'threads': 3, 'cpus': None, 'affinity_verified': False}])
        topology = data['topology']
        self.assertIsNone(data['topology_error'])
        self.assertEqual(topology['sockets'], [{'package': -1, 'cpus': None, 'threads': 3, 'ddr_node': -1,
            'hbm_node': -1, 'ddr_allowed': True, 'hbm_allowed': True, 'classification': 'virtual'}])
        # A topology that cannot be discovered is reported, not fatal, and failures still write valid JSON.
        _, data = self.execute('run', env={'LLAMA_TOPOLOGY': 'explicit:0/2/0,1/3/1', 'LLAMA_SYSFS_ROOT': '/nonexistent'})
        self.assertIsNone(data['topology'])
        self.assertIn('cannot read', data['topology_error'])

    def test_backend_environment_is_validated(self):
        for env, text in [({'LLAMA_BACKEND': 'bogus'}, 'LLAMA_BACKEND must be omp or teams'),
                          ({'LLAMA_BACKEND': 'teams', 'LLAMA_THREADS': '0'}, 'LLAMA_THREADS must be'),
                          ({'LLAMA_BACKEND': 'teams', 'LLAMA_THREADS': 'x'}, 'LLAMA_THREADS must be')]:
            with self.subTest(env=env):
                self.execute('run', success=False, env=env)
                result = subprocess.run([str(BUILD/'run'), str(self.shared/'fp32.bin'), '-z', str(self.shared/'tokenizer.bin'),
                                         '-i', 'A', '-t', '0'], capture_output=True, text=True, env={**os.environ, **env})
                self.assertIn(text, result.stderr)

    def test_prefill_bitwise_and_classifier_calls(self):
        for shared in (True, False):
            fixture = create(self.root/f'prefill-{shared}', shared=shared, seq_len=64)
            for binary, filename, activations in (
                ('test-prefill-fp32', 'fp32.bin', 'int8'),
                ('test-prefill-int8', 'int8.bin', 'int8'),
                ('test-prefill-int8', 'int8.bin', 'fp32'),
            ):
                for batch in (1, 2, 3, 7, 32):
                    with self.subTest(shared=shared, binary=binary, activations=activations, batch=batch):
                        result = subprocess.run([str(BUILD/binary), str(fixture/filename)],
                                                capture_output=True, text=True, timeout=120,
                                                env={**os.environ, 'LLAMA_PREFILL_BATCH': str(batch),
                                                     'LLAMA_Q8_ACTIVATIONS': activations})
                        self.assertEqual(result.returncode, 0, result.stderr)

    def test_prefill_invalid_inputs_and_configuration(self):
        for engine, binary, filename in (('run', 'test-prefill-fp32', 'fp32.bin'),
                                          ('runq', 'test-prefill-int8', 'int8.bin')):
            for value in ('0', '-1', '+1', ' 2', '2 ', '2x', '1.5', '2147483648', '9'*40):
                _, data = self.execute(engine, success=False, env={'LLAMA_PREFILL_BATCH': value})
                self.assertIn('LLAMA_PREFILL_BATCH', data['error'])
            self.execute(engine, success=False, env={'LLAMA_PREFILL_LOGITS': 'none'})
            _, data = self.execute(engine, '--context-length', '8', '--generation-length', '1',
                                   env={'LLAMA_PREFILL_BATCH': '2147483647'})
            self.assertEqual(data['parameters']['prefill_batch'], 8)
            for case in ('null', 'zero', 'negative-count', 'large-count', 'negative-pos',
                         'large-pos', 'context', 'token', 'negative-token', 'policy'):
                result = subprocess.run([str(BUILD/binary), str(self.shared/filename), case],
                                        capture_output=True, text=True,
                                        env={**os.environ, 'LLAMA_PREFILL_BATCH': '7'})
                self.assertEqual(result.returncode, 1, (case, result.stderr))
                self.assertIn('[ERROR]', result.stderr)
                self.assertNotIn('AddressSanitizer', result.stderr)
                self.assertNotIn('runtime error:', result.stderr)

    def test_prefill_cli_equivalence(self):
        cases = [
            (self.shared, ('-i', '', '--generation-length', '1')),
            (self.shared, ('-i', 'ABCDEFGHI', '--generation-length', '4')),
            (self.shared, ('-i', 'ABCDEFGHI', '--generation-length', '0')),
            (self.shared, ('-i', 'ABCDEFGHI', '-n', '5')),
            (self.shared, ('-i', 'ABCDEFGHI', '-n', '7')),
            (self.shared, ('-i', 'ABCDEFGHI', '-n', '10')),
            (self.shared, ('-i', 'A'*31, '-n', '32')),
            (self.shared, ('-i', 'A'*32, '-n', '32')),
            (self.unshared, ('-i', 'ABC', '--generation-length', '4', '-t', '.8')),
            (self.first_stop, ('--generation-length', '5')),
            (self.later_stop, ('--generation-length', '5')),
        ]
        for engine in ('run', 'runq'):
            for fixture, args in cases:
                baseline, reference = self.execute(engine, *args, fixture=fixture,
                                                    env={'LLAMA_PREFILL_BATCH': '1', 'LLAMA_PREFILL_LOGITS': 'all'})
                for policy in ('last', 'all'):
                    actual, data = self.execute(engine, *args, fixture=fixture,
                                                env={'LLAMA_PREFILL_BATCH': '7', 'LLAMA_PREFILL_LOGITS': policy})
                    self.assertEqual(actual.stdout, baseline.stdout)
                    self.assertEqual(data['tokens'], reference['tokens'])
                    self.assertEqual(data['stop_reason'], reference['stop_reason'])
                    self.assertEqual(data['prefill']['microbatches'],
                                     (data['tokens']['processed_prompt'] + 6)//7)
                    self.assertEqual(data['parameters']['prefill_batch'], 7)
                    self.assertEqual(data['parameters']['prefill_logits'], policy)
                    self.assertEqual(data['timing_seconds']['time_to_first_token'] is None,
                                     reference['timing_seconds']['time_to_first_token'] is None)

    def test_prefill_chat_multiple_turns(self):
        # Turn the deterministic first-stop classifier into an EOT classifier.
        fixture = create(self.root/'chat-eot', shared=False, stop='first', seq_len=64)
        d, _, _, _, _, vocab, _ = CONFIG
        for filename, width in (('fp32.bin', d*4), ('int8.bin', d)):
            data = bytearray((fixture/filename).read_bytes())
            scales = vocab*(d//2)*4 if filename == 'int8.bin' else 0
            base = len(data) - vocab*width - scales
            for start, size in [(base, width)] + ([(len(data)-scales, (d//2)*4)] if scales else []):
                data[start+128009*size:start+128010*size] = data[start+128001*size:start+128002*size]
                data[start+128001*size:start+128002*size] = bytes(size)
            (fixture/filename).write_bytes(data)
        for engine in ('run', 'runq'):
            baseline, reference = self.execute(engine, '-m', 'chat', '-y', '', '-i', 'A',
                                                fixture=fixture, stdin='B\nexit\n',
                                                env={'LLAMA_PREFILL_BATCH': '1', 'LLAMA_PROFILE': '1'})
            actual, data = self.execute(engine, '-m', 'chat', '-y', '', '-i', 'A',
                                        fixture=fixture, stdin='B\nexit\n',
                                        env={'LLAMA_PREFILL_BATCH': '7', 'LLAMA_PROFILE': '1'})
            self.assertEqual(actual.stdout, baseline.stdout)
            self.assertEqual(data['tokens'], reference['tokens'])
            self.assertEqual(data['tokens']['sampled_stop'], 2)
            self.assertEqual(data['tokens']['decode_steps'], 2)  # EOT caching
            self.assertEqual(data['stop_reason'], 'user_exit')
            self.assertEqual(data['prefill']['microbatches'], 8)  # 24-token first turn, 23-token next turn

    @unittest.skipUnless('-DLLAMA_PCRE2' in (BUILD/'config').read_text(), 'new tokenizer requires PCRE2')
    def test_manifest_matches_layouts(self):
        """Manifest offsets equal the old pointer-bumping layout (oracle in tests/manifest.c)."""
        def check(*args):
            result = subprocess.run([str(BUILD/'test-manifest'), *map(str, args)], capture_output=True, text=True)
            self.assertEqual(result.returncode, 0, result.stderr)
        check()
        v3 = upgrade_v3(create(self.root/'v3-manifest', shared=False))
        v3_shared = upgrade_v3(create(self.root/'v3-manifest-shared'))
        for fixture in (self.shared, self.unshared, v3, v3_shared):
            check('fp32', fixture/'fp32.bin')
            check('int8', fixture/'int8.bin')

    def test_v3_metadata_context_and_stopping(self):
        fixture = upgrade_v3(create(self.root/'v3'))
        for engine in ('run', 'runq'):
            _, report = self.execute(engine, '--context-length', '8', '--generation-length', '3', fixture=fixture)
            self.assertEqual(report['model']['checkpoint_format'], 3)
            config = report['model']['configuration']
            self.assertEqual(config['seq_len'], 8)
            self.assertEqual(config['max_position_embeddings'], 32)
            self.assertEqual(config['head_dim'], 2)
            self.assertTrue(config['shared_embeddings'])
            self.assertEqual(report['memory_bytes']['kv_cache_allocated_peak'], 2*CONFIG[2]*8*2*4)
            self.execute(engine, '--context-length', '33', fixture=fixture, success=False)
            self.execute(engine, '--context-length', '-1', fixture=fixture, success=False)
            _, report = self.execute(engine, '--benchmark', '--chat-template', '--generation-length', '1', fixture=fixture)
            self.assertTrue(report['parameters']['chat_template'])
        # Force each of EOS, EOT and EOM as the first sampled token.
        for stop in (128001, 128009, 128008):
            fixture = upgrade_v3(create(self.root/f'v3-stop-{stop}', shared=False, stop='first'))
            if stop != 128001:
                for filename, item_size in [('fp32.bin', 4), ('int8.bin', 1)]:
                    data = bytearray((fixture/filename).read_bytes())
                    classifier_bytes = CONFIG[5]*CONFIG[0]*item_size
                    if filename == 'int8.bin':
                        # INT8 matrix data is followed by per-group scales.
                        scale_bytes = CONFIG[5]*CONFIG[0]//2*4
                        start = len(data) - classifier_bytes - scale_bytes
                        scale_start = len(data) - scale_bytes
                        for base, width in ((start, CONFIG[0]), (scale_start, CONFIG[0]//2*4)):
                            data[base+stop*width:base+(stop+1)*width] = data[base+128001*width:base+128002*width]
                            data[base+128001*width:base+128002*width] = bytes(width)
                    else:
                        start = len(data) - classifier_bytes
                        width = CONFIG[0]*4
                        data[start+stop*width:start+(stop+1)*width] = data[start+128001*width:start+128002*width]
                        data[start+128001*width:start+128002*width] = bytes(width)
                    (fixture/filename).write_bytes(data)
            for engine in ('run', 'runq'):
                _, report = self.execute(engine, '--generation-length', '5', fixture=fixture)
                self.assertEqual(report['tokens']['generated_output'], 0)
                self.assertEqual(report['tokens']['sampled_stop'], 1)
                self.assertEqual(report['stop_reason'], 'stop_token')

    def test_generation_accounting(self):
        for engine in ('run','runq'):
            result, data = self.execute(engine,'--benchmark','--generation-length','3','--no-text')
            self.assertEqual(result.stdout,'')
            self.assertEqual(data['tokens'],dict(input_prompt=2,processed_prompt=2,generated_output=3,
                                                forward_passes=4,decode_steps=2,sampled=3,sampled_stop=0))
            times = data['timing_seconds']
            self.assertGreaterEqual(times['total_inference'],times['prompt_tokenization']+times['prefill']+times['decode'])
            self.assertGreaterEqual(times['time_to_first_token'],times['prompt_tokenization']+times['prefill'])
            self.assertGreaterEqual(times['end_to_end'],times['checkpoint_load']+times['tokenizer_init']+times['runtime_init']+times['total_inference'])
            for k,count,duration in [('prefill',2,times['prefill']),('decode',2,times['decode']),('overall_forward',4,times['total_inference'])]:
                self.assertAlmostEqual(data['throughput_tokens_per_second'][k],count/duration,delta=max(1e-3,count/duration*1e-3))
            mem = data['memory_bytes']
            self.assertEqual(mem['kv_cache_allocated_peak'],2*CONFIG[2]*CONFIG[-1]*2*4)
            self.assertEqual(mem['mapped_checkpoint'],mem['model_file'])
            self.assertEqual(mem['model_file'],(self.shared/('fp32.bin' if engine=='run' else 'int8.bin')).stat().st_size)
            if engine=='runq':  # embedding rows are dequantized on demand since A4: no FP32 table exists
                self.assertEqual(mem['dequantized_embedding_allocated_peak'],0)

    # ---- A4: owned weight placement ----------------------------------------------------------------------------
    OWNED = {'LLAMA_PLACEMENT': 'ddr', 'LLAMA_PLACEMENT_STRICT': '0'}  # portable regions: no NUMA needed

    def logits(self, engine, fixture, tokens, env, batch=None):
        probe = BUILD/('probe-fp32' if engine == 'run' else 'probe-int8')
        model = fixture/('fp32.bin' if engine == 'run' else 'int8.bin')
        command = [str(probe)] + (['batch'] if batch else []) + [str(model), *map(str, tokens)]
        extra = {'LLAMA_PREFILL_BATCH': str(batch), 'LLAMA_PREFILL_LOGITS': 'all'} if batch else {}
        result = subprocess.run(command, capture_output=True, timeout=60, env={**os.environ, **extra, **env})
        self.assertEqual(result.returncode, 0, result.stderr.decode())  # exit 3 would mean a leak
        return result.stdout

    def test_owned_placement_is_bitwise_equal_to_the_file_mapping(self):
        tokens = [65, 66, 67, 68, 69, 70, 71]
        variants = [{}, {'LLAMA_LOADER_CHUNK': '7', 'LLAMA_LOADER_MAX_READ': '5'},
                    {'LLAMA_BACKEND': 'teams', 'LLAMA_THREADS': '3', 'LLAMA_LOADER_CHUNK': '4096'}]
        for fixture in (self.shared, self.unshared):
            for engine in ('run', 'runq'):
                for batch in (None, 3):
                    expected = self.logits(engine, fixture, tokens, {}, batch)
                    self.assertEqual(len(expected), len(tokens) * 128010 * 4)
                    for extra in variants:
                        actual = self.logits(engine, fixture, tokens, {**self.OWNED, **extra}, batch)
                        self.assertEqual(actual, expected, (fixture.name, engine, batch, extra))

    def test_owned_placement_report_and_accounting(self):
        for engine, name in (('run', 'fp32.bin'), ('runq', 'int8.bin')):
            for fixture in (self.shared, self.unshared):
                _, data = self.execute(engine, '--benchmark', '--generation-length', '2', '--no-text', fixture=fixture,
                                       env=self.OWNED)
                self.assertEqual(data['runtime']['placement'], 'ddr')
                mem, file_size = data['memory_bytes'], (fixture/name).stat().st_size
                self.assertEqual((mem['model_file'], mem['mapped_checkpoint']), (file_size, 0))
                everything = data['regions']
                self.assertEqual(sorted(r['role'] for r in everything if r['role'].startswith('kv') or r['role'] == 'work'),
                                 ['kv.k', 'kv.v', 'work'])
                regions = [r for r in everything if r['role'] in ('weights', 'embedding')]
                roles = sorted(r['role'] for r in regions)
                self.assertEqual(roles, ['weights'] if fixture is self.shared else ['embedding', 'weights'])
                # Portable regions (no NUMA nodes here) carry no evidence; bound ones on Linux must be fully verified.
                for r in everything:
                    if r['node'] is None:
                        self.assertEqual(r['verified'], 'unverified')
                    elif r['role'] in ('weights', 'embedding'):
                        self.assertEqual((r['verified'], r['pages_mismatched'], r['pages_missing']), ('full', 0, 0))
                total = sum(r['bytes'] for r in regions)
                header = 28 if engine == 'run' else 256
                # Everything after the header is copied; padding keeps each range's alignment modulo 64.
                self.assertGreaterEqual(total, file_size - header)
                self.assertLess(total, file_size - header + 3 * 64)
                self.assertEqual(mem['owned_weights_peak'] + mem['dequantized_embedding_allocated_peak'], total)
                self.assertEqual(data['kv']['reserved_bytes'], mem['kv_cache_allocated_peak'])
                for key in ('placement_populate', 'placement_verify'):
                    self.assertGreaterEqual(data['timing_seconds'][key], 0)
                self.assertLess(data['timing_seconds']['checkpoint_load'], 5)
        _, data = self.execute('run', '--benchmark', '--no-text')
        self.assertEqual((data['runtime']['placement'], data['regions']), ('file', []))
        self.assertIsNone(data['timing_seconds']['placement_populate'])
        self.assertEqual(data['memory_bytes']['owned_weights_peak'], 0)

    # ---- A5: KV cache and workspace placement ------------------------------------------------------------------
    def test_kv_accounting_matches_the_formula(self):
        layers, kv_dim, seq = CONFIG[2], CONFIG[0] // CONFIG[3] * CONFIG[4], CONFIG[-1]
        for env in ({}, self.OWNED):
            for engine in ('run', 'runq'):
                _, data = self.execute(engine, '--benchmark', '--generation-length', '5', '--no-text', env=env)
                kv, tokens = data['kv'], data['tokens']
                self.assertEqual(kv['reserved_bytes'], 2 * layers * seq * kv_dim * 4)
                position = tokens['processed_prompt'] + tokens['decode_steps']
                self.assertEqual(kv['active_bytes'], 2 * layers * position * kv_dim * 4)
                self.assertEqual(kv['after_prompt']['active_bytes'], 2 * layers * tokens['processed_prompt'] * kv_dim * 4)
                self.assertLessEqual(kv['active_bytes'], kv['reserved_bytes'])
                if env and kv['node'] is not None:  # bound KV on Linux: every queried page on the requested node
                    self.assertTrue(kv['verified'] and kv['pages_wrong_node'] == 0)
                    self.assertGreaterEqual(kv['resident_bytes'], kv['active_bytes'])
                    self.assertLessEqual(kv['resident_bytes'], kv['reserved_bytes'])
                else:  # residency is never invented
                    self.assertFalse(kv['verified'])
                    self.assertIsNone(kv['resident_bytes'])
                self.assertEqual(data['memory_bytes']['kv_cache_allocated_peak'], kv['reserved_bytes'])

    def test_kv_verification_is_untimed_and_can_be_disabled(self):
        _, data = self.execute('run', '--benchmark', '--no-text', env={**self.OWNED, 'LLAMA_KV_VERIFY': '0'})
        self.assertFalse(data['kv']['verified'])
        self.assertIsNone(data['kv']['resident_bytes'])

    def test_owned_placement_runs_to_the_context_limit(self):
        for engine in ('run', 'runq'):
            for env in ({}, self.OWNED):
                _, data = self.execute(engine, '--benchmark', '--no-text', '--context-length', '16', env=env)
                self.assertEqual(data['tokens']['processed_prompt'] + data['tokens']['decode_steps'], 16)
                self.assertEqual(data['kv']['active_bytes'], data['kv']['reserved_bytes'])

    def test_placement_configuration_errors(self):
        _, data = self.execute('run', env={'LLAMA_PLACEMENT': 'bogus'}, success=False)
        self.assertIn('LLAMA_PLACEMENT must be file, ddr or hbm', data['error'])
        _, data = self.execute('run', env={**self.OWNED, 'LLAMA_LOADER_CHUNK': '0'}, success=False)
        self.assertIn('LLAMA_LOADER_CHUNK', data['error'])
        # hbm needs a real HBM node; without one it must fail rather than silently use DDR (strict is the default).
        _, data = self.execute('run', env={'LLAMA_PLACEMENT': 'hbm', 'LLAMA_TOPOLOGY': 'virtual:1x2'}, success=False)
        self.assertIn('LLAMA_PLACEMENT=hbm cannot be honoured', data['error'])
        _, data = self.execute('run', '--benchmark', env={'LLAMA_PLACEMENT': 'hbm', 'LLAMA_TOPOLOGY': 'virtual:1x2',
                                                          'LLAMA_PLACEMENT_STRICT': '0'})
        self.assertEqual(data['runtime']['placement'], 'hbm')  # explicit opt-out: unbound memory, still correct
        _, data = self.execute('run', env={'LLAMA_PLACEMENT': 'hbm', 'LLAMA_TOPOLOGY': 'virtual:1x2',
                                           'LLAMA_PLACEMENT_STRICT': '0', 'LLAMA_HBM_BUDGET_GIB': '-1'}, success=False)
        self.assertIn('LLAMA_HBM_BUDGET_GIB', data['error'])

    def test_phase_profile(self):
        phases = {'embedding','rope_table','rmsnorm','quantize','qkv','rope','kv_store','attention','wo',
                  'w1_w3_swiglu','w2','classifier','other'}
        for engine in ('run','runq'):
            _, data = self.execute(engine,'--benchmark','--generation-length','3','--no-text')
            self.assertIsNone(data['profile_seconds'])
            _, data = self.execute(engine,'--benchmark','--generation-length','3','--no-text',env={'LLAMA_PROFILE':'1'})
            profile = data['profile_seconds']
            for stage, passes in [('prefill',2),('decode',2)]:
                section = profile[stage]
                self.assertEqual(section['forward_passes'],passes)
                self.assertAlmostEqual(section['forward'],data['timing_seconds'][stage],delta=1e-6)
                self.assertEqual(set(section)-{'forward_passes','forward'},phases)
                self.assertTrue(all(section[k] >= 0 for k in phases-{'other'}))
                self.assertAlmostEqual(sum(section[k] for k in phases),section['forward'],delta=1e-6)
            if engine == 'run':  # tiny phases may round to 0 on a 1 us clock, so only FP32's absence is exact
                self.assertEqual(profile['decode']['quantize'],0)
            self.execute(engine,env={'LLAMA_PROFILE':'yes'},success=False)

    def test_empty_one_output_zero_and_partial(self):
        for engine in ('run','runq'):
            _, data = self.execute(engine,'-i','','--generation-length','1')
            self.assertEqual(data['tokens']['processed_prompt'],1)
            self.assertEqual(data['tokens']['generated_output'],1)
            self.assertEqual(data['tokens']['decode_steps'],0)
            self.assertIsNone(data['timing_seconds']['decode'])
            self.assertIsNone(data['throughput_tokens_per_second']['decode'])
            _, data = self.execute(engine,'--generation-length','0')
            self.assertEqual(data['tokens']['forward_passes'],0)
            self.assertEqual(data['tokens']['input_prompt'],2)
            self.assertIsNone(data['timing_seconds']['time_to_first_token'])
            self.assertIsNone(data['throughput_tokens_per_second']['overall_forward'])
            _, data = self.execute(engine,'-n','1')
            self.assertEqual(data['tokens']['processed_prompt'],1)
            self.assertEqual(data['tokens']['generated_output'],0)
            self.assertEqual(data['stop_reason'],'step_limit')

    def test_seeded_repeatability_and_unshared(self):
        for engine in ('run','runq'):
            a, _ = self.execute(engine,'--generation-length','4','-t','.8',fixture=self.unshared)
            b, _ = self.execute(engine,'--generation-length','4','-t','.8',fixture=self.unshared)
            self.assertEqual(a.stdout,b.stdout)
            self.execute(engine,'--generation-length','3',fixture=self.zero)

    def test_stop_token_boundaries(self):
        for engine in ('run','runq'):
            for fixture, output, decode in [(self.first_stop,0,0),(self.later_stop,1,1)]:
                _, data = self.execute(engine,'--generation-length','5',fixture=fixture)
                self.assertEqual(data['stop_reason'],'stop_token')
                self.assertEqual(data['tokens']['processed_prompt'],2)
                self.assertEqual(data['tokens']['generated_output'],output)
                self.assertEqual(data['tokens']['decode_steps'],decode)
                self.assertEqual(data['tokens']['sampled_stop'],1)
                self.assertEqual(data['tokens']['sampled'],output+1)
                self.assertEqual(data['tokens']['forward_passes'],2+decode)
                self.assertEqual(data['timing_seconds']['time_to_first_token'] is None,output==0)

    def test_report_cannot_overwrite_inputs(self):
        for target in ('fp32.bin','tokenizer.bin'):
            path=self.shared/target
            before=path.read_bytes()
            result=subprocess.run([str(BUILD/'run'),str(self.shared/'fp32.bin'),'-z',str(self.shared/'tokenizer.bin'),
                                   '--json',str(path),'--generation-length','0'],capture_output=True,text=True)
            self.assertEqual(result.returncode,1)
            self.assertIn('aliases input',result.stderr)
            self.assertEqual(path.read_bytes(),before)
        alias=self.root/'alias.json'
        alias.symlink_to(self.shared/'fp32.bin')
        result=subprocess.run([str(BUILD/'run'),str(self.shared/'fp32.bin'),'--json',str(alias),'-n','invalid'],capture_output=True,text=True)
        self.assertEqual(result.returncode,1)
        self.assertIn('aliases input',result.stderr)

    def test_nonfinite_checkpoint(self):
        bad=self.root/'nonfinite';bad.mkdir(exist_ok=True)
        shutil.copy(self.shared/'tokenizer.bin',bad/'tokenizer.bin')
        for engine,filename in [('run','fp32.bin'),('runq','int8.bin')]:
            weights=bytearray((self.shared/filename).read_bytes())
            if engine=='run':
                offset=28+128000*4*4
            else:
                offset=256 # first RMSNorm weight
            struct.pack_into('<f',weights,offset,float('nan'))
            (bad/filename).write_bytes(weights)
            self.execute(engine,fixture=bad,success=False)

    def test_cli_invalid(self):
        for args in [('-n','-1'),('-n','abc'),('-s','18446744073709551616'),('-t','nan'),
                     ('-p','2'),('-m','invalid'),('--log-level','verbose'),('--unknown','1'),
                     ('-i','X','--prompt-file','none'),('--benchmark','-m','chat'),('--generation-length',)]:
            self.execute('run',*args,success=False)

    def test_bad_checkpoints(self):
        bad = self.root/'bad'
        bad.mkdir(exist_ok=True)
        shutil.copy(self.shared/'tokenizer.bin',bad/'tokenizer.bin')
        for engine,filename in [('run','fp32.bin'),('runq','int8.bin')]:
            valid = bytearray((self.shared/filename).read_bytes())
            cases = [b'',valid[:20],valid[:-1]]
            offset=0 if engine=='run' else 8
            for field,value in [(0,0),(3,3),(4,3),(5,-2147483648),(6,-1),(0,2147483646)]:
                mutated=valid.copy();struct.pack_into('<i',mutated,offset+4*field,value);cases.append(mutated)
            if engine=='runq':
                for group in (0,3,2147483647):
                    mutated=valid.copy();struct.pack_into('<i',mutated,37,group);cases.append(mutated)
                mutated=valid.copy();mutated[36]=2;cases.append(mutated)
                mutated=valid.copy();mutated[4]=1;cases.append(mutated)
            for content in cases:
                (bad/filename).write_bytes(content)
                _,data=self.execute(engine,fixture=bad,success=False)
                self.assertIsNone(data['timing_seconds']['time_to_first_token'])
            (bad/filename).unlink()
            self.execute(engine,fixture=bad,success=False)

    def test_bad_tokenizer(self):
        bad=self.root/'bad-tokenizer';bad.mkdir(exist_ok=True)
        shutil.copy(self.shared/'fp32.bin',bad/'fp32.bin')
        original=(self.shared/'tokenizer.bin').read_bytes()
        for data in (b'',struct.pack('<I',0),struct.pack('<I',0xffffffff),original[:-1],original+b'X',
                     original[:8]+struct.pack('<i',-1)+original[12:]):
            (bad/'tokenizer.bin').write_bytes(data)
            self.execute('run',fixture=bad,success=False)
        (bad/'tokenizer.bin').unlink()
        self.execute('run',fixture=bad,success=False)

    def test_prompt_file_and_json_escaping(self):
        prompt=self.root/'prompt.txt';prompt.write_text('A "quoted" \\ line\nè')
        # Direct invocation avoids the default -i supplied by execute().
        report=self.root/'unicode.json'
        command=[str(BUILD/'run'),str(self.shared/'fp32.bin'),'-z',str(self.shared/'tokenizer.bin'),
                 '--benchmark','--prompt-file',str(prompt),'--json',str(report),'--generation-length','0','--no-text']
        result=subprocess.run(command,capture_output=True,text=True)
        self.assertEqual(result.returncode,0,result.stderr)
        data=json.loads(report.read_text())
        self.assertEqual(data['parameters']['prompt'],prompt.read_text())
        self.assertEqual(data['parameters']['seed'],1)
        self.assertEqual(result.stdout,'')

    def test_chat_eof_and_long_inputs(self):
        for engine in ('run','runq'):
            self.execute(engine,'-m','chat','-y','S','-n','28',stdin='exit\n')
            self.execute(engine,'-m','chat','-y','S'*40000,'-i','U'*40000,'-n','1',stdin='')

    def test_help_and_failed_report(self):
        result=subprocess.run([str(BUILD/'run'),'--help'],capture_output=True,text=True)
        self.assertEqual(result.returncode,0)
        result=subprocess.run([str(BUILD/'run'),'missing','--json',str(self.root/'no-directory'/'x.json')],capture_output=True,text=True)
        self.assertEqual(result.returncode,1)
        self.assertIn('cannot create report',result.stderr)


if __name__ == '__main__':
    unittest.main()
