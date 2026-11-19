"""Converter smoke checks without installing large ML dependencies."""
import ast
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
from types import SimpleNamespace
from typing import List
import unittest

ROOT = Path(__file__).resolve().parents[1]
TOOLS = ROOT/'tools'
sys.path[:0] = [str(TOOLS), str(ROOT/'tests')]
BUILD = Path(os.environ['BUILD_DIR']).resolve() if os.environ.get('BUILD_DIR') else None

# Knobs the campaign scripts read from the environment. A test that runs a script starts from the caller's environment
# without them, so a run inside a campaign job (which exports BASELINE_ROOT, STAGE, MODELS and so on) sees what it sets.
CAMPAIGN_KNOBS = {'BASELINE_ROOT', 'B_EXPECT', 'C1_SPEED', 'CMP_PROBE', 'CMP_TOKENS', 'CONTEXT', 'CPU_BIND', 'D2_M1', 'D2_M2',
                  'DRY_RUN', 'D_THP', 'E0_ATTN', 'E0_DIR', 'E0_MODEL', 'E1_DIR', 'E1_MODEL', 'E1_SMALL_MODEL', 'E2_MODEL', 'E2_SWEEP_CELLS',
                  'E4_AMX', 'E4_ATTN', 'E4_RUNGS', 'E4_TILED_ENV', 'GATES_ONLY',
                  'GATE_ENV', 'GATE_TAG', 'GENERATION', 'HF_GATE', 'LOG_LEVEL', 'MEM_BIND', 'MODELS', 'OUT', 'PERF', 'PHASE',
                  'PRECISIONS', 'PYTHON', 'REP', 'REPEATS', 'ROOT', 'SKIP_PRELUDE_HF', 'SKIP_TEST', 'SLURM_JOB_ID', 'STAGE',
                  'WARM'}


def campaign_env():
    return {k: v for k, v in os.environ.items() if k not in CAMPAIGN_KNOBS and not k.startswith('LLAMA_')}


class ToolSmokeTests(unittest.TestCase):
    def test_all_sources_compile(self):
        for path in TOOLS.glob('*.py'):
            compile(path.read_text(), str(path), 'exec')

    def test_tokenizer_encode_delegates_and_adds_special_tokens(self):
        # Isolate the real method so the test needs no tiktoken installation or
        # licensed tokenizer.model. This checks the wrapper, not BPE correctness.
        source = ast.parse((TOOLS/'tokenizer.py').read_text())
        cls = next(n for n in source.body if isinstance(n, ast.ClassDef) and n.name == 'Tokenizer')
        method = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == 'encode')
        module = ast.fix_missing_locations(ast.Module(body=[method],type_ignores=[]))
        namespace = {'List':List}
        exec(compile(module,'tokenizer.py','exec'),namespace)
        calls = []
        def model_encode(text, **kwargs):
            calls.append((text, kwargs))
            return [7,8]
        instance = SimpleNamespace(model=SimpleNamespace(encode=model_encode),bos_id=128000,eos_id=128001)
        result = namespace['encode'](instance,'hello è',True,True,{'allowed'},set())
        self.assertEqual(result,[128000,7,8,128001])
        self.assertEqual(calls,[('hello è',{'allowed_special':{'allowed'},'disallowed_special':set()})])


def report(prefill, decode, ttft=0.5, status='completed', placement='file', regions=None, thp='off', kernels='avx512',
           attention=None, attention_decode=None):
    build = {'kernels': kernels}
    if attention is not None:
        build['attention'] = attention
    if attention_decode is not None:
        build['attention_decode'] = attention_decode
    return {'schema_version': 2, 'status': status, 'build': build,
            'runtime': {'placement': placement, 'thp': thp, 'precision_mode': 'fp32'}, 'regions': regions or [],
            'timing_seconds': {'time_to_first_token': ttft},
            'throughput_tokens_per_second': {'prefill': prefill, 'decode': decode},
            'profile_seconds': {'decode': {'forward_passes': 3, 'forward': 10.0, 'qkv': 2.0, 'w1_w3': 5.0, 'wo': 1.0}},
            'phase_seconds': None}


class CampaignSummaryTests(unittest.TestCase):
    def test_labels(self):
        import summarize_campaign as sc
        info = sc.parse_label('llama3-8b-fp32-teams-hbm-s2-t112-m256-p2048-c4096-kamx-3')
        self.assertEqual((info['model'], info['precision'], info['backend'], info['placement'], info['sockets'],
                          info['threads'], info['m'], info['prompt'], info['ctx'], info['kernel'], info['rep']),
                         ('llama3-8b', 'fp32', 'teams', 'hbm', '2', '112', '256', '2048', '4096', 'amx', '3'))
        info = sc.parse_label('1b-w8a32-omp-file-s1-t56-1')
        self.assertEqual((info['model'], info['precision'], info['m'], info['kernel']), ('1b', 'w8a32', None, None))
        for bad in ('correctness-1b-fp32', '1b-fp32-omp-file-s1-t56', 'x-fp16-omp-file-s1-t1-1'):
            self.assertIsNone(sc.parse_label(bad))

    def dry_run(self, script, phases, models=('m1b', 'm8b'), **extra):
        """Runs one campaign script with DRY_RUN=1 and returns its RUN lines; every run label must parse."""
        if not shutil.which('bash'):
            self.skipTest('bash is required')
        from summarize_campaign import parse_label
        with tempfile.TemporaryDirectory() as temp:
            env = {**campaign_env(), 'DRY_RUN': '1', 'SLURM_JOB_ID': '1', 'ROOT': str(ROOT), 'OUT': str(Path(temp)/'out'),
                   'PYTHON': sys.executable, 'PHASE': phases,
                   'MODELS': ' '.join(f'{m}:/a/{m}:/hf/{m}' for m in models), **extra}
            result = subprocess.run(['bash', str(ROOT/'scripts'/script)], env=env, capture_output=True, text=True, timeout=120)
            self.assertEqual(result.returncode, 0, result.stderr[-2000:])
        runs = [line for line in result.stdout.splitlines() if line.startswith('RUN ')]
        for line in runs:
            self.assertIsNotNone(parse_label(line.split()[1] + '-1'), line)
        return runs

    def test_campaign_scripts_dry_run_and_their_labels_parse(self):
        """L6: the B7, C7, D1 and D2 matrices print without running anything, with labels the summarizer understands."""
        b7 = self.dry_run('cresco8-tp.sbatch', 'B7')
        self.assertEqual(len(b7), 192)  # 96 launches per model: gates aside, see phase_B7
        self.assertTrue(all('warm=' in line and 'warm=0 ' not in line for line in b7))
        self.assertEqual({line.split('warm=')[1].split()[0] for line in b7}, {'512', '8192', '32768'})
        self.assertTrue(any('-s2-t14-' in line for line in b7) and any('xctl' in line for line in b7))
        self.assertTrue(any('xp4 ' in line and 'LLAMA_FP32_LAYOUT=packed4' in line for line in b7))
        self.assertTrue(all('LLAMA_PROFILE=1' in line for line in b7 if 'xp4prof' in line))
        self.assertTrue(all('LLAMA_PROFILE=1' in line for line in b7 if 'xprof' in line))
        d1 = self.dry_run('cresco8-prefill.sbatch', 'D1')
        self.assertEqual(len(d1), 240)
        self.assertTrue(any('-s2-' in line and 'kamx_bf16' in line for line in d1) and any('-s1-' in line for line in d1))
        self.assertEqual({line.split('warm=')[1].split()[0] for line in d1}, {'0', '2048'})
        d2 = self.dry_run('cresco8-prefill.sbatch', 'D2')
        self.assertEqual(len(d2), 288)
        self.assertEqual({line.split('warm=')[1].split()[0] for line in d2}, {'0', '512', '8192'})
        # GATES_ONLY=1 keeps the gates (and D1's placement runs) and drops the benchmark.
        self.assertEqual(self.dry_run('cresco8-tp.sbatch', 'B7', GATES_ONLY='1'), [])
        placement = self.dry_run('cresco8-prefill.sbatch', 'D1', GATES_ONLY='1')
        self.assertTrue(placement and len(placement) < 20 and all('xplace' in line for line in placement), placement)
        # Every D1 comparison of the 2048-token prompt reads only the last row, so both sides must run the batched probe
        # (job 5704685: W8A32 lost its --batch flags and the per-token side wrote every position).
        with tempfile.TemporaryDirectory() as temp:
            env = {**campaign_env(), 'DRY_RUN': '1', 'SLURM_JOB_ID': '1', 'ROOT': str(ROOT), 'OUT': str(Path(temp)/'out'),
                   'PYTHON': sys.executable, 'PHASE': 'D1', 'GATES_ONLY': '1', 'MODELS': 'm1b:/a/m1b:/hf/m1b'}
            out = subprocess.run(['bash', str(ROOT/'scripts'/'cresco8-prefill.sbatch')], env=env, capture_output=True,
                                 text=True, timeout=120).stdout
        compares = [line for line in out.splitlines() if line.startswith('EXPECT compare-') and 'prompt2048' in line]
        self.assertEqual(len(compares), 5)  # fp32, bf16, q8, int8, w8a32
        for line in compares:
            self.assertIn('--last-only', line)
            self.assertIn('--batch-a 64', line, line)
            self.assertIn('--batch-b 64', line, line)
        # C7's AMX-INT8 comparisons use the W8A8 statistical gate (rms and argmax), not the strict 1e-3 of the other rungs.
        with tempfile.TemporaryDirectory() as temp:
            env = {**campaign_env(), 'DRY_RUN': '1', 'SLURM_JOB_ID': '1', 'ROOT': str(ROOT), 'OUT': str(Path(temp)/'out'),
                   'PYTHON': sys.executable, 'PHASE': 'C7', 'MODELS': 'm1b:/a/m1b:/hf/m1b'}
            out = subprocess.run(['bash', str(ROOT/'scripts'/'cresco8-prefill.sbatch')], env=env, capture_output=True,
                                 text=True, timeout=120).stdout
        amx = [line for line in out.splitlines() if line.startswith('EXPECT compare-') and 'c7-amx-int8' in line]
        self.assertEqual(len(amx), 4)
        for line in amx:
            self.assertIn('--rms-max 0.2', line)
            self.assertNotIn('--max-abs 1e-3', line)
        for line in [l for l in out.splitlines() if l.startswith('EXPECT compare-') and 'c7-avx512-q8' in l]:
            self.assertIn('--max-abs 1e-3', line)
        self.dry_run('cresco8-prefill.sbatch', 'D3')  # the W8A8 AMX diagnostic prints its comparisons and exits 0
        c7 = self.dry_run('cresco8-prefill.sbatch', 'C7')
        decode = [line for line in c7 if 'xdec' in line]
        self.assertTrue(decode and all('warm=2048' in line for line in decode))

    def test_e0_dry_run_matrix(self):
        """E0: the attention campaign prints its bench matrix, the knob's bitwise gate and one profiled 8B engine run."""
        runs = self.dry_run('cresco8-prefill.sbatch', 'E0')  # only the 8B model of MODELS gets engine runs
        self.assertEqual(len(runs), 1)
        self.assertTrue(all(w in runs[0] for w in ('-m256-p8192-', 'kamx_bf16', 'xprof', 'LLAMA_PROFILE=1', 'warm=0')), runs)
        with tempfile.TemporaryDirectory() as temp:
            env = {**campaign_env(), 'DRY_RUN': '1', 'SLURM_JOB_ID': '1', 'ROOT': str(ROOT), 'OUT': str(Path(temp)/'out'),
                   'PYTHON': sys.executable, 'PHASE': 'E0', 'MODELS': 'm1b:/a/m1b:/hf/m1b m8b:/a/m8b:/hf/m8b'}
            out = subprocess.run(['bash', str(ROOT/'scripts'/'cresco8-prefill.sbatch')], env=env, capture_output=True,
                                 text=True, timeout=120).stdout.splitlines()
        bench = [line for line in out if line.startswith('EXPECT ') and 'attention-bench' in line]
        self.assertEqual(len(bench), 6 + 5 + 2 + 4)  # prefill, row block, half KV heads, decode
        self.assertTrue(all('--node 2' in line and '--cpunodebind=0' in line for line in bench))
        self.assertEqual(sorted(line.split('LLAMA_ATTN_ROW_BLOCK=')[1].split()[0] for line in bench if 'ROW_BLOCK' in line),
                         ['1', '16', '2', '4', '8'])
        self.assertEqual(len([line for line in out if line.startswith('COUNTERS ')]), 2)
        compare = [line for line in out if line.startswith('EXPECT compare-m8b-fp32-e0-rowblock')]
        self.assertEqual(len(compare), 1)
        for word in ('--bitwise', '--last-only', '--batch-a 64', '--batch-b 64', 'LLAMA_ATTN_ROW_BLOCK=4',
                     'LLAMA_ATTN_ROW_BLOCK=16'):
            self.assertIn(word, compare[0])
        self.assertFalse([line for line in out if 'm1b' in line and 'compare' in line])

    def e1_dry(self, **extra):
        with tempfile.TemporaryDirectory() as temp:
            env = {**campaign_env(), 'DRY_RUN': '1', 'SLURM_JOB_ID': '1', 'ROOT': str(ROOT), 'OUT': str(Path(temp)/'out'),
                   'PYTHON': sys.executable, 'PHASE': 'E1', 'MODELS': 'm1b:/a/m1b:/hf/m1b m8b:/a/m8b:/hf/m8b', 'REPEATS': '3',
                   **extra}
            result = subprocess.run(['bash', str(ROOT/'scripts'/'cresco8-prefill.sbatch')], env=env, capture_output=True,
                                    text=True, timeout=120)
            self.assertEqual(result.returncode, 0, result.stderr[-2000:])
            skipped = (Path(temp)/'out'/'E1'/'baseline-not-given.txt').exists()
        return [line for line in result.stdout.splitlines() if line.startswith('EXPECT ')], skipped

    def test_e1_dry_run_matrix(self):
        """E1: bitwise baseline comparisons on 8B only, item 2b on 1B only, FP32 determinism data on every model."""
        lines, skipped = self.e1_dry(BASELINE_ROOT='/b')
        self.assertFalse(skipped)
        bitwise = [l for l in lines if l.startswith('EXPECT compare-')]
        self.assertEqual(len(bitwise), 3)
        for line in bitwise:
            self.assertIn('m8b-fp32', line)
            for word in ('--bitwise', '--last-only', 'probe-shim-fp32.sh', 'PROBE_DIR=/b/build/release',
                         f'PROBE_DIR={ROOT}/build/release', 'LLAMA_ATTN=split'):
                self.assertIn(word, line)
        self.assertEqual(len([l for l in bitwise if 'LLAMA_TP_CHECK=1' in l and 'LLAMA_SOCKETS=2' in l]), 1)
        gen = [l for l in lines if 'compare_generation.py' in l]
        item2b = [l for l in gen if 'gen-item2b-' in l]
        self.assertEqual(len(item2b), 3 * 3)  # three cases, REPEATS = 3
        self.assertTrue(all('m1b' in l and '--warm-context 8192' in l and '--steps 128' in l and '--probe-batch 64' in l and
                            'probe-int8' in l and '--context-length 8448' in l for l in item2b), item2b)
        self.assertEqual(len([l for l in item2b if 'w8a32' in l]), 3)
        self.assertTrue(all('LLAMA_Q8_ACTIVATIONS=fp32' in l for l in item2b if 'w8a32-s1-vs-s2' in l))
        self.assertTrue(all('LLAMA_TP_CHECK=1' in l for l in item2b if 's1-vs-s2' in l))
        self.assertFalse([l for l in item2b if 's1-vs-s1' in l and 'LLAMA_SOCKETS=2' in l])
        fp32 = [l for l in gen if 'gen-fp32-' in l]
        self.assertEqual(len(fp32), 2 * 2 * 2)  # models x prompts x comparisons
        self.assertTrue(all('probe-fp32' in l and 'warm-context' not in l for l in fp32))
        self.assertEqual(len(lines), 3 + 9 + 8)
        # Without a baseline the bitwise part is skipped and recorded, and nothing else changes.
        lines, skipped = self.e1_dry()
        self.assertTrue(skipped)
        self.assertEqual(len(lines), 9 + 8)

    @staticmethod
    def v2(prefill=None, decode=None, warm=0, sockets=1, ttft=1.0, steps=128, decode_seconds=None, model_file=32e9,
           attention=None):
        """A synthetic completed schema-v2 report of a campaign run."""
        data = report(prefill or 0, decode or 0, ttft=ttft)
        data['runtime'].update(sockets=sockets, placement='hbm')
        data['parameters'] = {'warm_context': warm}
        data['tokens'] = {'decode_steps': steps}
        data['timing_seconds'].update(decode=decode_seconds if decode_seconds is not None else (steps / decode if decode else None))
        data['prefill'] = {'tokens': 86}
        data['memory_bytes'] = {'model_file': model_file}
        if attention is not None:
            data['profile_seconds'] = {'decode': {'forward': 10.0, 'attention': attention}}
        return data

    PROFILE = ('[INFO] tp decode socket {s} (128 tokens, ms/token, fused=1, reduce=flags): embedding=0.010 rmsnorm=1.000 '
               'qkv=4.000 rope=0.100 attention={att} wo=3.000 w1_w3=12.000 w2=6.000 classifier=2.000 wo_mm=2.500 '
               'w2_mm=5.000 x_straggler=0.300 x_peer={peer} x_add=0.100 x_close=0.500 x_buffer=0.000\n')

    def test_campaign_tables_decode_prefill_socket_and_modeled(self):
        import summarize_campaign as sc
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            write = lambda name, data: (root/name).write_text(json.dumps(data))
            for rep, (one, two) in enumerate([(16.0, 30.0), (16.3, 30.4), (16.1, 31.0)], 1):
                write(f'8b-fp32-teams-hbm-s1-t56-p512-{rep}.json', self.v2(decode=one, warm=512))
                write(f'8b-fp32-teams-hbm-s2-t56-p512-{rep}.json', self.v2(decode=two, warm=512, sockets=2))
            write('8b-fp32-teams-hbm-s2-t14-p512-1.json', self.v2(decode=20.0, warm=512, sockets=2))
            write('8b-fp32-omp-file-s2-t112-p512-xctl-1.json', self.v2(decode=7.0, warm=512, sockets=2))
            write('8b-fp32-teams-hbm-s2-t56-p512-xp4-1.json', self.v2(decode=33.0, warm=512, sockets=2))
            # A longer context where the gain is lost in the noise, and where the one-socket run is the faster.
            for rep, (one, two) in enumerate([(10.0, 10.1), (10.2, 9.9)], 1):
                write(f'8b-fp32-teams-hbm-s1-t56-p8192-{rep}.json', self.v2(decode=one, warm=8192))
                write(f'8b-fp32-teams-hbm-s2-t56-p8192-{rep}.json', self.v2(decode=two, warm=8192, sockets=2))
            # Profiled runs: one healthy, one whose socket 0 waits for its peer.
            for label, peer, att in (('8b-fp32-teams-hbm-s2-t56-p512-xprof', 0.2, 2.0), ('1b-fp32-teams-hbm-s2-t56-p512-xprof', 9.0, 2.0)):
                write(f'{label}-1.json', self.v2(decode=30.0, warm=512, sockets=2, decode_seconds=128 * 0.033, attention=7.0))
                (root/f'{label}-1.stderr').write_text(self.PROFILE.format(s=0, att=att, peer=peer) +
                                                      self.PROFILE.format(s=1, att=2.4, peer=4.0))
            # Prefill: the per-token one-socket baseline, the GEMM on one socket and on two.
            write('8b-fp32-teams-hbm-s1-t56-m1-p2048-kper_token-1.json', self.v2(prefill=16.0, ttft=128.0))
            for rep, (one, two) in enumerate([(174.0, 300.0), (176.0, 310.0)], 1):
                write(f'8b-fp32-teams-hbm-s1-t56-m64-p2048-kavx512_fp32-{rep}.json', self.v2(prefill=one, ttft=11.8))
                write(f'8b-fp32-teams-hbm-s2-t56-m64-p2048-kavx512_fp32-{rep}.json', self.v2(prefill=two, ttft=6.9, sockets=2))
            text = sc.summarize(root)
        self.assertIn('| Warm ctx |', text)
        lines = text.splitlines()
        row = lambda needle: next(line for line in lines if line.startswith(needle))
        # Decode by retained context: the gain 30.4 / 16.1 = 1.89, beyond variability; at 8192 it is not.
        self.assertIn('### Decode tok/s by retained context', text)
        d512 = row('| 8b | fp32 | 512 |')
        self.assertIn('16.10 [16.00-16.30]', d512)
        self.assertIn('30.40 [30.00-31.00]', d512)
        self.assertIn('| 20.00 |', d512)
        self.assertIn('| 7.00 |', d512)       # the control
        self.assertIn('| 33.00 |', d512)       # two sockets from the packed4 copy
        self.assertTrue(d512.rstrip().endswith('| 1.89 | yes | 1.09 |'), d512)    # packed4 / rows = 33.0 / 30.4
        self.assertTrue(row('| 8b | fp32 | 8192 |').rstrip().endswith('| no | - |'))
        # Prefill speedups and where two sockets win.
        self.assertIn('### Prefill by rung, microbatch and sockets', text)
        one = row('| 8b | fp32 | 2048 | avx512_fp32 | 64 | 1 |')
        two = row('| 8b | fp32 | 2048 | avx512_fp32 | 64 | 2 |')
        self.assertIn('| 10.94 |', one)         # 175 / 16
        self.assertTrue(one.rstrip().endswith('| no |') or one.rstrip().endswith('| - |'))
        self.assertTrue(two.rstrip().endswith('| yes |'), two)
        # The per-socket table, its imbalance row and the modeled-against-measured row.
        self.assertIn('### Per-socket decode phase times (ms per token)', text)
        s0 = row('| 8b-fp32-teams-hbm-s2-t56-p512-xprof | 0 |')
        self.assertIn('| 4.000 | 2.000 | 3.000 | 18.000 | 2.000 | 0.100 | 1.000 |', s0)   # qkv attn wo ffn cls reduce wait
        imbalance = row('| 8b-fp32-teams-hbm-s2-t56-p512-xprof | imbalance |')
        self.assertIn('| 0.000 | 0.400 |', imbalance)
        self.assertIn('### Modeled against measured decode time (ms per token)', text)
        healthy = row('| 8b-fp32-teams-hbm-s2-t56-p512-xprof | 33.00 |')
        self.assertIn('| 33.91 | 103% | 26.67 | 81% | none |', healthy)   # 32e9 bytes / (2 x 600e9) = 26.67 ms
        # Revisit triggers (study 10.4): socket 0 of the second run waits 9.8 ms of 33 for reductions, 9 ms of it for its peer.
        flagged = row('| 1b-fp32-teams-hbm-s2-t56-p512-xprof | 33.00 |')
        self.assertIn('reductions+waits 30% > 20%; peer wait 27% > 10%', flagged)
        self.assertIn('### Where synchronization or attention dominates', text)
        self.assertIn('attention 70% of the token', text)

    def test_results_skeletons_match_the_summarizer(self):
        """L9: the results documents of A6, B7, C7 and D2 and their validation records exist, say they are not a measurement,
        and carry the column layout the summarizers produce, so the numbers can be pasted in unchanged."""
        import summarize_campaign as sc
        import campaign_tables as ct
        with tempfile.TemporaryDirectory() as temp:
            header = sc.summarize(Path(temp)).splitlines()[0]
        names = ('numa-placement', 'tensor-parallel', 'batched-prefill', 'numa-tp-prefill-results')
        for name in names:
            text = (ROOT/'docs'/f'{name}.md').read_text()
            validation = (ROOT/'docs'/f'{name}-validation.md').read_text()
            # A skeleton says so; once the campaign has run the status line says when (and the tables hold numbers).
            for document in (text, validation):
                self.assertTrue('**Status: not run.**' in document or '**Status: run' in document, name)
            self.assertIn(header, text, name)
            self.assertIn(f'({name}-validation.md)', text)
            for section in ('## Provenance', '## Gates', '## Deviations from the plan', '## Raw data'):
                self.assertIn(section, validation, (name, section))
        tp = (ROOT/'docs'/'tensor-parallel.md').read_text()
        self.assertIn('| ' + ' | '.join(ct.COLUMNS) + ' |', tp)
        self.assertIn('Gain 2x56 / 1x56 | Beyond variability | packed4 / rows', tp)
        self.assertIn('Revisit triggers', tp)
        for name in ('batched-prefill', 'numa-tp-prefill-results'):
            self.assertIn('| Model | Precision | Prompt | Rung | M | Sockets | Prefill tok/s | TTFT s | Speedup vs 1-socket per-token '
                          '| 2s wins |', (ROOT/'docs'/f'{name}.md').read_text())

    def test_a_phase_sent_to_the_wrong_script_fails_before_the_build(self):
        """B7 lives in cresco8-tp.sbatch and D1 in cresco8-prefill.sbatch; a mismatch must not cost a build and `make test`."""
        if not shutil.which('bash'):
            self.skipTest('bash is required')
        for script, phases, missing in (('cresco8-prefill.sbatch', 'D1 B7', 'B7'), ('cresco8-tp.sbatch', 'B7 D1', 'D1'),
                                        ('cresco8-prefill.sbatch', 'C7 D4', 'D4')):
            with tempfile.TemporaryDirectory() as temp:
                env = {**campaign_env(), 'SLURM_JOB_ID': '1', 'ROOT': str(ROOT), 'OUT': str(Path(temp)/'out'),
                       'PYTHON': sys.executable, 'PHASE': phases}  # not a dry run: the check must come before check_stage
                result = subprocess.run(['bash', str(ROOT/'scripts'/script)], env=env, capture_output=True, text=True,
                                        timeout=60)
            self.assertNotEqual(result.returncode, 0, (script, phases))
            self.assertIn(f'no run matrix for phase(s) {missing}', result.stderr)
            self.assertNotIn('build', result.stdout.lower())

    def test_campaign_runs_use_the_whole_context(self):
        """The engine's default of 4096 forward passes capped A6's 8192-token point; run_engine must pass -n 0."""
        library = (ROOT/'scripts'/'lib'/'cresco8-common.sh').read_text()
        launch = library[library.index('"$ROOT/build/release/$engine" "$checkpoint"'):]
        launch = launch[:launch.index('&\n')]
        self.assertIn(' -n 0 ', launch)

    def test_small_unverified_work_region_does_not_hide_a_verified_placement(self):
        import summarize_campaign as sc
        large = {'verified': 'full', 'pages_mismatched': 0, 'bytes': 2**30}
        work = {'verified': 'unverified', 'bytes': 2**20}
        status = lambda *regions: sc.placement_status({'runtime': {'placement': 'hbm'}, 'regions': list(regions)})
        self.assertEqual(status(large, work), 'hbm (verified)')
        self.assertEqual(status(large, dict(work, bytes=2**30)), 'hbm (unverified)')  # a big unchecked region still counts
        self.assertEqual(status(work), 'hbm (unverified)')                             # nothing was verified at all
        self.assertEqual(status(large, dict(large, pages_mismatched=3)), 'hbm (MISMATCH)')

    def test_attention_variants_appear_only_when_not_default(self):
        """E1: LLAMA_ATTN / LLAMA_ATTN_DECODE are recorded in build.attention / build.attention_decode; the summary shows
        them next to the kernel when they differ from `split`, and old or default reports keep their cells."""
        import summarize_campaign as sc
        import campaign_tables as ct
        self.assertEqual(ct.attention_tag(report(1, 1)), '')
        self.assertEqual(ct.attention_tag(report(1, 1, attention='split', attention_decode='split')), '')
        self.assertEqual(ct.attention_tag(report(1, 1, attention='tiled', attention_decode='split')), 'attn=tiled')
        self.assertEqual(ct.attention_tag(report(1, 1, attention='amx', attention_decode='group')),
                         'attn=amx decode-attn=group')
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            (root/'8b-fp32-teams-hbm-s1-t56-p512-kavx512-1.json').write_text(
                json.dumps(report(1, 1, attention='split', attention_decode='split')))
            (root/'8b-fp32-teams-hbm-s1-t56-p512-kavx512_tiled-1.json').write_text(
                json.dumps(report(1, 1, attention='tiled', attention_decode='split')))
            table = sc.summarize(root)
        rows = {line.split('|')[1].strip(): line for line in table.splitlines()[2:] if line.startswith('| 8b')}
        self.assertIn('| avx512 |', rows['8b-fp32-teams-hbm-s1-t56-p512-kavx512'])
        self.assertIn('| avx512_tiled attn=tiled |', rows['8b-fp32-teams-hbm-s1-t56-p512-kavx512_tiled'])

    def test_summary_of_repeats_failures_and_old_reports(self):
        import summarize_campaign as sc
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            verified = [{'verified': 'full', 'pages_mismatched': 0, 'thp_bytes': 2**30}]
            for rep, (prefill, decode) in enumerate([(10, 4.0), (12, 4.4), (11, 4.2)], 1):
                (root/f'8b-fp32-teams-hbm-s1-t56-p512-kavx512-{rep}.json').write_text(json.dumps(
                    report(prefill, decode, placement='hbm', regions=verified, thp='on')))
            (root/'8b-fp32-teams-hbm-s1-t56-p512-kavx512-1.perf.csv').write_text(
                'S0,56,1000000000,,hbm_cas_rd,,\nS1,56,500000000,,hbm_cas_rd,,\nS0,56,2000000,,ddr_cas_rd,,\n')
            (root/'8b-fp32-omp-hbm-s2-t112-1.json').write_text(json.dumps(report(1, 1, status='error')))
            (root/'8b-fp32-omp-ddr-s1-t56-1.json').write_text(json.dumps(
                report(5, 2, placement='ddr', regions=[{'verified': 'full', 'pages_mismatched': 7}])))
            old = report(3, 1.5)
            old.pop('runtime'), old.pop('regions'), old.pop('phase_seconds')
            (root/'1b-fp32-omp-file-s1-t4-1.json').write_text(json.dumps(old))
            (root/'notes.json').write_text('{}')
            (root/'correctness-status.txt').write_text('8b fp32 exit=0\n')
            table = sc.summarize(root)
        rows = {line.split('|')[1].strip(): line for line in table.splitlines()[2:] if line.startswith('| 8b') or
                line.startswith('| 1b')}
        row = rows['8b-fp32-teams-hbm-s1-t56-p512-kavx512']
        self.assertIn('| 3 |', row)
        self.assertIn('hbm (verified)', row)
        self.assertIn('on (1.0 GiB huge)', row)
        self.assertIn('11.00 [10.00-12.00]', row)       # prefill: median with the full range
        self.assertIn('4.20 [4.00-4.40]', row)          # decode
        self.assertIn('w1_w3 50%, qkv 20%, wo 10%', row)
        self.assertIn('| 96.0 |', row.replace('96.0', '96.0'))  # 1.5e9 CAS * 64 B
        self.assertIn('failed', rows['8b-fp32-omp-hbm-s2-t112'])
        self.assertIn('ddr (MISMATCH)', rows['8b-fp32-omp-ddr-s1-t56'])
        self.assertIn('| - |', rows['1b-fp32-omp-file-s1-t4'])   # v1 report: no runtime object
        self.assertNotIn('notes', table)
        self.assertIn('- 8b fp32 exit=0', table)


@unittest.skipUnless(BUILD and (BUILD/'probe-int8').exists(), 'needs BUILD_DIR with probe-fp32/probe-int8 (make test)')
class CompareEnginesTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from fixtures import create
        cls.temp = tempfile.TemporaryDirectory()
        cls.fixture = create(Path(cls.temp.name)/'fx')

    @classmethod
    def tearDownClass(cls):
        cls.temp.cleanup()

    def compare(self, engine, *args):
        result = subprocess.run([sys.executable, str(TOOLS/'compare_engines.py'), '--probe', str(BUILD/f'probe-{engine}'),
                                 '--checkpoint', str(self.fixture/f'{engine}.bin'), '--tokens', '65,66,67,68', *args],
                                capture_output=True, text=True, timeout=120)
        return result, (json.loads(result.stdout) if result.stdout else None)

    def test_identical_configurations_agree_exactly(self):
        for engine in ('fp32', 'int8'):
            result, data = self.compare(engine, '--env-b', 'LLAMA_UNRELATED=1')  # same ISA and backend: bitwise equal
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertTrue(data['passed'])
            self.assertEqual(len(data['comparisons']), 4)
            self.assertEqual((data['summary']['max_abs'], data['summary']['rms'], data['summary']['argmax_agreement']),
                             (0.0, 0.0, 1.0))

    def test_probe_forwards_past_the_default_context(self):
        """A probe run of more than 4096 positions sizes the context to them (E1 item 2b probes about 8300 positions, E2
        compares 8192-token prompts); the engines' default would stop at position 4096. All three probe modes, both
        engines, on a fixture whose checkpoint allows 4200 positions."""
        from fixtures import create
        with tempfile.TemporaryDirectory() as temp:
            fixture = create(Path(temp)/'long', seq_len=4200)
            tokens = [str(65 + i % 20) for i in range(4100)]
            for engine in ('fp32', 'int8'):
                probe, model = str(BUILD/f'probe-{engine}'), str(fixture/f'{engine}.bin')
                for mode, command, env, values in (('batch', [probe, 'batch', model, *tokens], {'LLAMA_PREFILL_BATCH': '64'}, 1),
                                                   ('per-token', [probe, model, *tokens], {}, 4100),
                                                   ('generate', [probe, 'generate', model, '3', *tokens[:4098]], {}, None)):
                    with self.subTest(engine=engine, mode=mode):
                        result = subprocess.run(command, capture_output=True, env={**os.environ, **env}, timeout=600)
                        self.assertEqual(result.returncode, 0, result.stderr.decode(errors='replace')[-500:])
                        if values is not None:
                            self.assertEqual(len(result.stdout) % (values * 4), 0)
                            self.assertGreater(len(result.stdout), 0)
                        else:
                            self.assertGreaterEqual(len(result.stdout.split()), 1)
                result = subprocess.run([probe, 'batch', model, *tokens, *tokens[:200]], capture_output=True,
                                        env={**os.environ, 'LLAMA_PREFILL_BATCH': '64'}, timeout=600)
                self.assertNotEqual(result.returncode, 0)  # 4300 positions: beyond the checkpoint's 4200
                self.assertIn(b'requested context exceeds checkpoint context', result.stderr)

    def test_teams_backend_agrees_within_the_budget(self):
        result, data = self.compare('fp32', '--env-b', 'LLAMA_BACKEND=teams', '--env-b', 'LLAMA_THREADS=3')
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(data['env_b'], {'LLAMA_BACKEND': 'teams', 'LLAMA_THREADS': '3'})

    def test_perturbed_configuration_is_reported_and_fails_the_budget(self):
        result, data = self.compare('int8', '--env-b', 'LLAMA_Q8_ACTIVATIONS=fp32', '--max-abs', '1e-12')
        self.assertEqual(result.returncode, 1)
        self.assertFalse(data['passed'])
        self.assertGreater(data['summary']['max_abs'], 0)
        self.assertGreater(data['summary']['max_rel'], 0)
        self.assertGreater(data['summary']['rms'], 0)
        self.assertTrue(any(not row['passed'] for row in data['comparisons']))
        self.assertEqual([row['position'] for row in data['comparisons']], [0, 1, 2, 3])
        output = Path(self.temp.name)/'out.json'
        result, _ = self.compare('int8', '--env-b', 'LLAMA_Q8_ACTIVATIONS=fp32', '--output', str(output))
        self.assertEqual(json.loads(output.read_text())['tokens'], 4)

    def test_statistical_gate_for_requantizing_comparisons(self):
        """W8A8 moves by a noise floor under any FP32 reordering, so --rms-max / --argmax-min judge the whole comparison."""
        perturb = ('--env-b', 'LLAMA_Q8_ACTIVATIONS=fp32')
        result, data = self.compare('int8', *perturb, '--max-abs', '100', '--rms-max', '100', '--argmax-min', '0')
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertTrue(data['passed'])
        self.assertEqual(data['budget']['rms_max'], 100.0)
        self.assertTrue(data['budget']['statistical'])
        for failing in (('--max-abs', '1e-12', '--rms-max', '100'), ('--max-abs', '100', '--rms-max', '1e-12'),
                        ('--max-abs', '100', '--argmax-min', '1.5')):
            result, data = self.compare('int8', *perturb, *failing)
            self.assertEqual(result.returncode, 1, failing)
            self.assertFalse(data['passed'], failing)
        result, _ = self.compare('int8', *perturb, '--bitwise', '--rms-max', '1')
        self.assertNotEqual(result.returncode, 0)

    def test_batched_prefill_is_bitwise_equal_to_per_token(self):
        # split's invariant (the batched default is tiled since E4, which agrees within the FP32 gate, not bitwise,
        # where AVX-512 runs; tests/test_tp.py holds its gate)
        split = ('--env-a', 'LLAMA_ATTN=split', '--env-b', 'LLAMA_ATTN=split')
        for engine in ('fp32', 'int8'):
            for batch in ('3', '32'):
                result, data = self.compare(engine, *split, '--batch-b', batch, '--bitwise')
                self.assertEqual(result.returncode, 0, result.stderr)
                self.assertTrue(data['summary']['bitwise'])
                self.assertEqual(data['batch_b'], int(batch))

    def test_last_only_compares_the_final_position(self):
        for engine in ('fp32', 'int8'):
            result, data = self.compare(engine, '--batch-b', '3', '--last-only', '--bitwise')
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertEqual([row['position'] for row in data['comparisons']], [3])
            self.assertTrue(data['summary']['bitwise'])

    def test_last_only_with_both_sides_batched(self):
        """D1 compares two batched runs (one socket against two) on long prompts: only the final row of each is read."""
        for engine in ('fp32', 'int8'):
            result, data = self.compare(engine, '--batch-a', '3', '--batch-b', '5', '--last-only', '--env-a',
                                        'LLAMA_PREFILL_GEMM=on', '--env-b', 'LLAMA_PREFILL_GEMM=on', '--bitwise')
            self.assertEqual(result.returncode, 0, (engine, result.stderr))
            self.assertEqual([row['position'] for row in data['comparisons']], [3])

    def test_scalar_gemm_prefill_is_bitwise_equal_to_per_token(self):
        # The scalar GEMM sums each output like the GEMV (one K block, ascending k); the Q8 engine has no GEMM yet.
        # Attention is pinned to split, whose batched path is bitwise equal to per-token (see above).
        for engine in ('fp32', 'int8'):
            for batch in ('2', '7', '32'):
                result, data = self.compare(engine, '--env-a', 'LLAMA_ATTN=split', '--env-b', 'LLAMA_ATTN=split',
                                            '--env-b', 'LLAMA_PREFILL_GEMM=on', '--batch-b', batch, '--bitwise')
                self.assertEqual(result.returncode, 0, (engine, batch, result.stderr))
                self.assertTrue(data['summary']['bitwise'])
        result, _ = self.compare('fp32', '--env-b', 'LLAMA_PREFILL_GEMM=sometimes', '--batch-b', '2')
        self.assertNotEqual(result.returncode, 0)

    def test_bad_arguments(self):
        result, _ = self.compare('fp32', '--env-b', 'NOT_AN_ASSIGNMENT')
        self.assertNotEqual(result.returncode, 0)
        self.assertIn('KEY=VALUE', result.stderr)


class GenerationMathTests(unittest.TestCase):
    """compare_generation.py helpers, on synthetic sequences and logits (no engine needed)."""
    @classmethod
    def setUpClass(cls):
        import compare_generation as cg
        cls.cg = cg

    def test_first_divergence(self):
        self.assertIsNone(self.cg.first_divergence([1, 2, 3], [1, 2, 3]))
        self.assertIsNone(self.cg.first_divergence([], []))
        self.assertEqual(self.cg.first_divergence([1, 2, 3], [1, 9, 3]), 1)
        self.assertEqual(self.cg.first_divergence([1, 2], [1, 2, 3]), 2)   # a strict prefix ends where it diverges
        self.assertEqual(self.cg.first_divergence([], [7]), 0)

    def test_edit_distance(self):
        self.assertEqual(self.cg.edit_distance([], []), 0)
        self.assertEqual(self.cg.edit_distance([1, 2, 3], [1, 2, 3]), 0)
        self.assertEqual(self.cg.edit_distance([1, 2, 3], [1, 3]), 1)          # deletion
        self.assertEqual(self.cg.edit_distance([], [1, 2]), 2)                 # insertions only
        self.assertEqual(self.cg.edit_distance([1, 2, 3], [1, 9, 3]), 1)       # substitution
        self.assertEqual(self.cg.edit_distance(list('kitten'), list('sitting')), 3)
        self.assertEqual(self.cg.edit_distance([1, 2, 3, 4], [4, 3, 2, 1]), 4)

    def test_top_two_margin_on_synthetic_logits(self):
        import struct
        margin, top2 = self.cg.top_two_margin(struct.pack('<4f', 0.5, 2.5, 2.0, -1.0))
        self.assertEqual((margin, top2), (0.5, [1, 2]))
        margin, top2 = self.cg.top_two_margin(struct.pack('<3f', 1.0, 1.0, 0.0))
        self.assertEqual((margin, top2), (0.0, [0, 1]))  # a tie: the lower id ranks first
        with self.assertRaises(SystemExit):
            self.cg.top_two_margin(struct.pack('<2f', float('nan'), 1.0))
        with self.assertRaises(SystemExit):
            self.cg.top_two_margin(struct.pack('<1f', 3.0))


@unittest.skipUnless(BUILD and (BUILD/'probe-fp32').exists() and (BUILD/'run').exists(),
                     'needs BUILD_DIR with run and probe-fp32 (make test)')
class CompareGenerationTests(unittest.TestCase):
    STUB = ('#!{python}\n'
            'import json, os\n'
            'with open(os.environ["LLAMA_TOKENS_OUT"], "w") as f:\n'
            '    json.dump({{"prompt": [65, 66, 67], "generated": json.loads(os.environ["STUB_GENERATED"])}}, f)\n')

    @classmethod
    def setUpClass(cls):
        from fixtures import create
        cls.temp = tempfile.TemporaryDirectory()
        cls.fixture = create(Path(cls.temp.name)/'fx')
        cls.prompt = Path(cls.temp.name)/'prompt.txt'
        cls.prompt.write_text('A')
        cls.stub = Path(cls.temp.name)/'stub-engine'
        cls.stub.write_text(cls.STUB.format(python=sys.executable))
        cls.stub.chmod(0o755)

    @classmethod
    def tearDownClass(cls):
        cls.temp.cleanup()

    def compare(self, *args, engine=None, steps='16'):
        result = subprocess.run([sys.executable, str(TOOLS/'compare_generation.py'), '--engine', str(engine or BUILD/'run'),
                                 '--checkpoint', str(self.fixture/'fp32.bin'), '--tokenizer', str(self.fixture/'tokenizer.bin'),
                                 '--prompt-file', str(self.prompt), '--steps', steps, *args],
                                capture_output=True, text=True, timeout=120)
        return result, (json.loads(result.stdout) if result.stdout else None)

    def test_identical_configurations_generate_identical_text(self):
        result, data = self.compare('--env-a', 'LLAMA_BACKEND=omp', '--env-b', 'LLAMA_BACKEND=teams',
                                    '--env-b', 'LLAMA_THREADS=3', '--probe', str(BUILD/'probe-fp32'))
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertTrue(data['identical'])
        self.assertIsNone(data['first_divergence'])
        self.assertEqual((data['edit_distance'], data['margin_a'], data['top2_a']), (0, None, None))
        self.assertGreater(data['generated_a'], 0)
        self.assertEqual(data['generated_a'], data['generated_b'])
        self.assertTrue(data['prompt_identical'])
        self.assertEqual(data['env_b'], {'LLAMA_BACKEND': 'teams', 'LLAMA_THREADS': '3'})
        self.assertIn('identical', result.stderr)

    def test_warm_context_is_part_of_the_recorded_prompt(self):
        result, data = self.compare('--warm-context', '2')
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(data['prompt_tokens'], 2 + 2)   # BOS and 'A', forwarded once as warm-up and once as prompt
        self.assertEqual(data['warm_context'], 2)

    def test_divergence_reports_index_tokens_distance_and_margin(self):
        """A stub engine writes the chosen ids, so the divergence is known exactly; the margin is the real probe's."""
        from compare_engines import run_probe
        from compare_generation import top_two_margin
        result, data = self.compare('--env-a', 'STUB_GENERATED=[70,71,72,73]', '--env-b', 'STUB_GENERATED=[70,71,99,73]',
                                    '--probe', str(BUILD/'probe-fp32'), engine=self.stub, steps='4')
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertFalse(data['identical'])
        self.assertEqual((data['first_divergence'], data['token_a'], data['token_b']), (2, 72, 99))
        self.assertEqual((data['generated_a'], data['generated_b'], data['edit_distance']), (4, 4, 1))
        blob = run_probe(BUILD/'probe-fp32', self.fixture/'fp32.bin', [65, 66, 67, 70, 71], {}, batch=1, last_only=True)
        margin, top2 = top_two_margin(blob)
        self.assertAlmostEqual(data['margin_a'], margin, places=6)
        self.assertEqual(data['top2_a'], top2)
        self.assertIn('diverge at token 2', result.stderr)

    def test_prefix_divergence_and_identical_stubs(self):
        result, data = self.compare('--env-a', 'STUB_GENERATED=[70,71,72,73]', '--env-b', 'STUB_GENERATED=[70,71]',
                                    engine=self.stub, steps='4')
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual((data['first_divergence'], data['token_a'], data['token_b'], data['edit_distance']),
                         (2, 72, None, 2))
        self.assertIsNone(data['margin_a'])     # no --probe
        result, data = self.compare('--env-a', 'STUB_GENERATED=[70,71]', '--env-b', 'STUB_GENERATED=[70,71]',
                                    engine=self.stub, steps='4')
        self.assertTrue(data['identical'])

    def test_failed_engine_run_is_an_error(self):
        result, data = self.compare('--warm-context', '99999', steps='4')  # warm-up must leave room in the context
        self.assertNotEqual(result.returncode, 0)
        self.assertIsNone(data)


@unittest.skipUnless(BUILD and (BUILD/'probe-fp32').exists(), 'needs BUILD_DIR with probe-fp32 (make test)')
class MakePromptsTests(unittest.TestCase):
    def build(self, out, lengths):
        from fixtures import create
        fixture = create(Path(out)/'fx')
        command = [sys.executable, str(TOOLS/'make_prompts.py'), '--probe', str(BUILD/'probe-fp32'), '--tokenizer',
                   str(fixture/'tokenizer.bin'), '--vocab', '128010', '--out', str(Path(out)/'prompts'),
                   '--lengths', lengths]
        result = subprocess.run(command, capture_output=True, text=True, timeout=120)
        self.assertEqual(result.returncode, 0, result.stderr)
        return Path(out)/'prompts', json.loads((Path(out)/'prompts/prompts.json').read_text())

    def test_short_only(self):
        with tempfile.TemporaryDirectory() as temp:
            directory, manifest = self.build(temp, 'none')
            self.assertEqual(list(manifest), ['short'])
            self.assertFalse(list(directory.glob('prompt-*.txt')))

    def test_prompts_have_the_requested_token_counts_and_are_reproducible(self):
        with tempfile.TemporaryDirectory() as first, tempfile.TemporaryDirectory() as second:
            directory, manifest = self.build(first, '16,300')
            _, again = self.build(second, '16,300')
            self.assertEqual(manifest, again)
            self.assertEqual((manifest['prompt-16']['tokens'], manifest['prompt-300']['tokens']), (16, 300))
            self.assertEqual(manifest['short']['tokens'], len(('Explain how a CPU executes instructions and uses '
                                                               'memory. ' * 5)) + 1)
            text = (directory/'prompt-300.txt').read_text()
            self.assertTrue(text.startswith('It was the best of times'))
            self.assertEqual(len(text), manifest['prompt-300']['characters'])
            self.assertEqual(len(manifest['prompt-16']['sha256']), 64)


class ScriptTests(unittest.TestCase):
    scripts = sorted((ROOT/'scripts').glob('*.sbatch')) + sorted((ROOT/'scripts/lib').glob('*.sh'))

    def test_shell_syntax(self):
        self.assertTrue(self.scripts)
        for path in self.scripts:
            with self.subTest(script=path.name):
                result = subprocess.run(['bash', '-n', str(path)], capture_output=True, text=True)
                self.assertEqual(result.returncode, 0, result.stderr)

    def lib(self, script, env=None):
        return subprocess.run(['bash', '-c', f'set -euo pipefail; source "{ROOT}/scripts/lib/cresco8-common.sh"; {script}'],
                              capture_output=True, text=True, env={'PATH': os.environ['PATH'], **(env or {})})

    def test_library_validates_its_environment(self):
        base = {'SLURM_JOB_ID': '1', 'ROOT': str(ROOT), 'PYTHON': 'python3', 'MODELS': '1b:/a:/h'}
        with tempfile.TemporaryDirectory() as temp:
            out = str(Path(temp)/'out')
            result = self.lib('require_env; printf "%s %s %s %s" "$STAGE" "$REPEATS" "$PRECISIONS" "$PHASE"',
                              {**base, 'OUT': out})
            self.assertEqual((result.returncode, result.stdout), (0, 'bench 3 fp32 int8 none'), result.stderr)
            for patch, text in [({'STAGE': 'x'}, 'STAGE must be'), ({'REPEATS': '0'}, 'REPEATS must be'),
                                ({'PHASE': 'Z9'}, 'PHASE must be'), ({'MODELS': 'bad'}, 'is not NAME:ARTIFACTS:HF'),
                                ({'OUT': temp}, 'already exists'), ({'SLURM_JOB_ID': ''}, 'submit through Slurm')]:
                with self.subTest(patch=patch):
                    result = self.lib('require_env', {**base, 'OUT': out, **patch})
                    self.assertNotEqual(result.returncode, 0)
                    self.assertIn(text, result.stderr)
            result = self.lib('require_env', {k: v for k, v in {**base, 'OUT': out}.items() if k != 'ROOT'})
            self.assertIn('tree under test required', result.stderr)
            # MODELS is optional, and PHASE is a list.
            result = self.lib('require_env; printf "[%s] [%s]" "$MODELS" "$PHASE"',
                              {**{k: v for k, v in base.items() if k != 'MODELS'}, 'OUT': out, 'PHASE': 'A1 A3 F2'})
            self.assertEqual((result.returncode, result.stdout), (0, '[] [A1 A3 F2]'), result.stderr)

    def test_expect_and_unknown_phases(self):
        with tempfile.TemporaryDirectory() as temp:
            script = (f'OUT={temp}; PHASE="A9"; expect good ok {temp} -- true; expect bad fail {temp} -- false; '
                      f'expect wrong ok {temp} -- false; expect free any {temp} -- false; '
                      f'expect talk ok {temp} -- echo hello; cat {temp}/phase-results.txt; echo "overall=$OVERALL"; '
                      f'cat {temp}/talk.out; run_phases')
            result = self.lib(script)
            self.assertNotEqual(result.returncode, 0)
            self.assertEqual(result.stdout.splitlines(), [
                'PASS good exit=0 expected=ok', 'PASS bad exit=1 expected=fail', 'FAIL wrong exit=1 expected=ok',
                'PASS free exit=1 expected=any', 'PASS talk exit=0 expected=ok', 'overall=1', 'hello'])
            self.assertIn('phase A9 has no run matrix', result.stderr)

    def test_run_labels_match_the_summarizer(self):
        import summarize_campaign as sc
        result = self.lib('run_label llama3-8b fp32 teams hbm 2 112 m=256 p=2048 c=4096 k=amx')
        self.assertEqual(result.stdout.strip(), 'llama3-8b-fp32-teams-hbm-s2-t112-m256-p2048-c4096-kamx', result.stderr)
        self.assertIsNotNone(sc.parse_label(result.stdout.strip() + '-1'))
        result = self.lib('run_label 1b w8a32 omp file 1 56')
        self.assertEqual(result.stdout.strip(), '1b-w8a32-omp-file-s1-t56')
        self.assertIsNotNone(sc.parse_label(result.stdout.strip() + '-2'))

    def test_skeletons_refuse_unknown_phases_without_a_cluster(self):
        for name in ('placement', 'tp', 'prefill'):
            result = subprocess.run(['bash', str(ROOT/f'scripts/cresco8-{name}.sbatch')], capture_output=True, text=True,
                                    env={'PATH': os.environ['PATH']})
            self.assertNotEqual(result.returncode, 0)
            self.assertIn('tree under test required', result.stderr)

    @unittest.skipUnless(shutil.which('shellcheck'), 'shellcheck not installed')
    def test_shellcheck(self):
        for path in self.scripts:
            with self.subTest(script=path.name):
                result = subprocess.run(['shellcheck', '-x', '-S', 'warning', str(path)], capture_output=True, text=True)
                self.assertEqual(result.returncode, 0, result.stdout)


class SummarizeGemmTests(unittest.TestCase):
    def test_table_has_speedup_and_peak_share(self):
        line = lambda kernel, m, gf: json.dumps(dict(kernel=kernel, isa='avx512', backend='teams', node=2, n=4096, k=4096, m=m,
                                                     gflops_best=gf, gflops_mean=gf * .9, weight_gbs_best=100, kc=1024))
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp)/'bench.jsonl'
            path.write_text('\n'.join([line('gemm', 8, 2000), line('gemv_rows', 8, 500), line('gemm', 64, 4400)]) + '\n')
            result = subprocess.run([sys.executable, str(TOOLS/'summarize_gemm.py'), str(path)], capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn('| 8 | 2000 / 1800 | 100 | 23% | 500 | 4.0x |', result.stdout)
        self.assertIn('| 64 | 4400 / 3960 | 100 | 50% | - | - |', result.stdout)


class SummarizeGenerationTests(unittest.TestCase):
    @staticmethod
    def gen(identical=True, where=None, margin=None, a=128, b=128, **extra):
        return dict(identical=identical, first_divergence=where, edit_distance=0 if identical else 5, generated_a=a,
                    generated_b=b, margin_a=margin, steps=128, **extra)

    def summarize(self, files, *flags):
        with tempfile.TemporaryDirectory() as temp:
            for name, data in files.items():
                (Path(temp)/name).write_text(json.dumps(data))
            result = subprocess.run([sys.executable, str(TOOLS/'summarize_generation.py'), temp, *flags], capture_output=True,
                                    text=True)
        self.assertEqual(result.returncode, 0, result.stderr)
        return result.stdout

    def test_item_2b_verdicts(self):
        same, diverged = self.gen(), self.gen(False, 29, margin=0.05, a=29, b=40)
        base = {'gen-item2b-m1b-w8a32-s1-vs-s2-1.json': same, 'gen-item2b-m1b-w8a8-s1-vs-s1-1.json': same,
                'gen-item2b-m1b-w8a8-s1-vs-s2-1.json': diverged}
        out = self.summarize(base)
        self.assertIn('| item2b-m1b-w8a8-s1-vs-s2 | 1 | no | 29 | 29/40 | 5 | 0.05 |', out)
        self.assertIn('W8A8, one socket against two: diverged in 1 of 1 runs, first divergence at token [29]', out)
        self.assertIn('Verdict (m1b): noise floor', out)
        # a margin above the noise floor, or any W8A32 divergence, is a bug; a missing margin leaves the verdict undecided
        big = {**base, 'gen-item2b-m1b-w8a8-s1-vs-s2-1.json': self.gen(False, 29, margin=1.5)}
        self.assertIn('Verdict (m1b): BUG (blocks E2): W8A8 diverged at a margin above the noise floor 0.2', self.summarize(big))
        w8a32 = {**base, 'gen-item2b-m1b-w8a32-s1-vs-s2-1.json': self.gen(False, 70, margin=0.01)}
        self.assertIn('BUG (blocks E2): W8A32 diverged between socket counts', self.summarize(w8a32))
        self.assertIn('undecided: a margin is missing', self.summarize({**base, 'gen-item2b-m1b-w8a8-s1-vs-s2-1.json':
                                                                       self.gen(False, 29, margin=None)}))
        none = {k: same for k in base}
        self.assertIn('no divergence reproduced', self.summarize(none))
        self.assertIn('control, one socket against one socket (run-to-run): diverged in 1 of 1 runs',
                      self.summarize({**base, 'gen-item2b-m1b-w8a8-s1-vs-s1-1.json': diverged}))

    def test_repeats_and_fp32_gate(self):
        files = {f'gen-item2b-m1b-w8a8-s1-vs-s2-{r}.json': self.gen(False, 29 + r, margin=0.1) for r in (1, 2, 3)}
        files['gen-fp32-m8b-2048-s1-vs-s2-1.json'] = self.gen()
        files['gen-fp32-m8b-short-pertoken-vs-m64-1.json'] = self.gen(False, 12, margin=1e-4)
        files['gen-fp32-m1b-short-s1-vs-s2-1.json'] = self.gen(False, 7, margin=0.4)
        files['gen-broken-1.json'] = {'nonsense': 1}  # skipped, not a crash
        out = self.summarize(files)
        self.assertIn('diverged in 3 of 3 runs', out)
        self.assertIn('| fp32-m8b-2048-s1-vs-s2 | 1 | PASS (identical) |', out)
        self.assertIn('| fp32-m8b-short-pertoken-vs-m64 | 1 | PASS (diverges at 12, margin within the tolerance) |', out)
        self.assertIn('| fp32-m1b-short-s1-vs-s2 | 1 | FAIL (diverges at 7) |', out)
        self.assertIn('1 of the FP32 comparisons outside the gate', out)
        self.assertNotIn('gen-broken', out)
        # the tolerance follows the observed max |delta logit|
        self.assertIn('| fp32-m1b-short-s1-vs-s2 | 1 | PASS (diverges at 7, margin within the tolerance) |',
                      self.summarize(files, '--max-dlogit', '0.5'))

    def test_bitwise_table_and_empty_directory(self):
        compare = dict(passed=True, summary=dict(bitwise=True, max_abs=0.0))
        out = self.summarize({'compare-m8b-fp32-e1-m64-prompt2048.json': compare})
        self.assertIn('| m8b-fp32-e1-m64-prompt2048 | yes | 0 | PASS |', out)
        self.assertIn('No generation comparison results', self.summarize({}))


class SummarizeAttentionBenchTests(unittest.TestCase):
    @staticmethod
    def prefill(ms, row_block=4, m=256, kv=8, kv_model=8, **extra):
        return dict(mode='prefill', isa='avx512', backend='teams', workers=56, node=2, shape='8b', heads=4 * kv, kv_heads=kv,
                    head_dim=128, kv_heads_model=kv_model, row_block=row_block, prompt=8192, m=m, reps=3, ms_layer_median=ms,
                    ms_layer_best=ms * .98, ms_layer_max=ms * 1.03, gflops_median=770.0, last_microbatch_ms_median=ms / 20,
                    flops=5.5e11, tasks=1000, waves=40, streamed_over_unique=63.5, busy_mean=.8, busy_min=.7,
                    tail_fraction=.12, task_phase_fraction=.97, **extra)

    def run_tool(self, directory, *flags):
        result = subprocess.run([sys.executable, str(TOOLS/'summarize_attention_bench.py'), str(directory), *flags],
                                capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr)
        return result.stdout

    def test_tables_cross_check_counters_and_missing_inputs(self):
        write = lambda path, rows: path.write_text('\n'.join(json.dumps(r) for r in rows) + '\n')
        with tempfile.TemporaryDirectory() as temp:
            d = Path(temp)
            write(d/'bench-prefill.jsonl', [self.prefill(700.0), self.prefill(900.0, m=64)])
            write(d/'rowblock.jsonl', [self.prefill(700.0 * 4 / r, row_block=r) for r in (1, 4, 16)])
            write(d/'bench-prefill-kv4.jsonl', [self.prefill(330.0, kv=4, kv_model=8)])
            write(d/'bench-decode.jsonl', [dict(mode='decode', shape='1b', kv_heads=8, context=32768, chunks=7, tasks=56,
                                                ms_layer_median=.3, ms_layer_best=.29, ms_layer_max=.4, gbs_median=120.0,
                                                gflops_median=60.0, busy_mean=.9, tail_fraction=.05, isa='avx512',
                                                backend='teams', node=2)])
            write(d/'counters-prefill-8b-8192-m256.jsonl', [self.prefill(700.0)])
            (d/'counters-prefill-8b-8192-m256.perf.csv').write_text(
                'S0,56,1.0e12,,cycles,1,100.0,,\nS0,56,2.0e12,,instructions,1,100.0,,\nS1,56,5,,cycles,1,100.0,,\n'
                'S0,56,<not supported>,,l2_rqsts.miss,0,0.0,,\n')
            (d/'engine-profile.json').write_text(json.dumps({'profile_seconds': {'prefill': {'forward': 30.0, 'attention': 23.0}}}))
            out = self.run_tool(d, '--counters-csv', str(d/'counters.csv'))
            # 22.4 s from the bench against 23.0 s from the engine: inside 20%
            self.assertIn('= 22.40 s against the profiled engine attention 23.00 s', out)
            self.assertIn('PASS (within +-20%)', out)
            self.assertIn('| 4 | 700.00 (686.00-721.00) | 1.00x |', out)
            self.assertIn('| 16 | 175.00 (171.50-180.25) | 4.00x |', out)
            self.assertIn('H3: the best row block is 16 at 4.00x over 4 (at or above the 1.3x', out)
            self.assertIn('| 8b | 8192 | 256 | 700.00 | 330.00 | 2.12 |', out)
            self.assertIn('| 1b | 8 | 32768 | 7 | 56 | 0.30 (0.29-0.40) | 120 | 22% |', out)
            self.assertIn('| 8b kv8 | 56 | 8192 | 256 | 700.00 (686.00-721.00) | 770 | 9% |', out)
            self.assertIn('| prefill-8b-8192-m256 | 1e+12 | 2e+12 | 2.00 | 0.727 |', out)  # 2e12 / (5.5e11 * 5 passes)
            csv_lines = (d/'counters.csv').read_text().splitlines()
            self.assertEqual(csv_lines[0], 'cell,socket,event,value')
            self.assertIn('prefill-8b-8192-m256,S0,instructions,2000000000000', csv_lines)
            self.assertIn('prefill-8b-8192-m256,S0,l2_rqsts.miss,', csv_lines)
            # an engine run 1.5x slower than the bench's prediction fails the gate
            (d/'engine-profile.json').write_text(json.dumps({'profile_seconds': {'prefill': {'forward': 40.0, 'attention': 34.0}}}))
            self.assertIn('FAIL (within +-20%)', self.run_tool(d))
        with tempfile.TemporaryDirectory() as temp:  # nothing measured yet: no crash, counters recorded as unavailable
            out = self.run_tool(temp, '--counters-csv', str(Path(temp)/'counters.csv'))
            self.assertIn('Counters: not available', out)
            self.assertEqual((Path(temp)/'counters.csv').read_text(), 'not available\n')


class SummarizeAttentionCompareTests(unittest.TestCase):
    """--compare of tools/summarize_attention_bench.py and the E2 summaries of tools/summarize_e2.py."""

    @staticmethod
    def row(attn, ms, prompt=8192, m=256, gflops=800.0, **extra):
        return dict(SummarizeAttentionBenchTests.prefill(ms, m=m, **extra), prompt=prompt, attn=attn, gflops_median=gflops)

    @staticmethod
    def write(path, rows):
        path.write_text('\n'.join(json.dumps(r) for r in rows) + '\n')

    def run_tool(self, tool, *args):
        result = subprocess.run([sys.executable, str(TOOLS/tool), *map(str, args)], capture_output=True, text=True)
        return result.returncode, result.stdout

    def test_compare_speedups_and_gates(self):
        with tempfile.TemporaryDirectory() as temp:
            d = Path(temp)
            self.write(d/'a.jsonl', [self.row('split', 700.0), self.row('split', 40.0, prompt=2048, m=64)])
            self.write(d/'b.jsonl', [self.row('tiled', 300.0, gflops=1900.0), self.row('tiled', 30.0, prompt=2048, m=64)])
            self.write(d/'sweep.jsonl', [self.row('tiled', 310.0, tiled_queries=4, tiled_order='natural'),
                                         self.row('tiled', 290.0, tiled_queries=16, tiled_order='longest')])
            code, out = self.run_tool('summarize_attention_bench.py', '--compare', d/'a.jsonl', d/'b.jsonl', '--sweep', d/'sweep.jsonl')
            self.assertEqual(code, 0, out)
            self.assertIn('| split ms/layer | tiled ms/layer | speedup |', out)
            self.assertIn('| 256 | 700.00 (686.00-721.00) | 300.00 (294.00-309.00) | 2.33x | 800 | 1900 |', out)
            self.assertIn('| 64 | 40.00 (39.20-41.20) | 30.00 (29.40-30.90) | 1.33x |', out)
            self.assertIn('Kernel gate (8b, prompt 8192, M 256, 56 threads): 2.33x, PASS', out)
            self.assertIn('2048-token cells not slower: PASS', out)
            self.assertIn('Best for 8b prompt 8192 M 256: queries 16, order longest (290.00 ms/layer).', out)
            # a slower 2048 cell and a missed gate are reported as FAIL
            self.write(d/'b.jsonl', [self.row('tiled', 500.0), self.row('tiled', 45.0, prompt=2048, m=64)])
            code, out = self.run_tool('summarize_attention_bench.py', '--compare', d/'a.jsonl', d/'b.jsonl')
            self.assertIn('1.40x, FAIL', out)
            self.assertIn('2048-token cells not slower: FAIL', out)
            # the directory summary of any prefill matrix file, without the counters
            code, out = self.run_tool('summarize_attention_bench.py', d, '--prefill-file', d/'b.jsonl', '--no-counters')
            self.assertEqual(code, 0, out)
            self.assertIn('| 8b kv8 | 56 | 8192 | 256 | 500.00 (490.00-515.00) |', out)
            self.assertNotIn('## Counters', out)
            # a missing file is not a crash
            code, out = self.run_tool('summarize_attention_bench.py', '--compare', d/'a.jsonl', d/'none.jsonl')
            self.assertEqual(code, 0, out)
            self.assertIn('no comparison', out)

    def test_checks_table_and_ttft_csv(self):
        with tempfile.TemporaryDirectory() as temp:
            d = Path(temp)
            ok = dict(passed=True, budget=dict(max_abs=1e-3, argmax_agreement=1.0, bitwise=False, rms_max=None, statistical=False),
                      summary=dict(max_abs=4e-5, argmax_agreement=1.0, rms=1e-5))
            bad = dict(ok, passed=False, summary=dict(max_abs=0.2, argmax_agreement=0.9, rms=0.01),
                       budget=dict(max_abs=2.0, argmax_agreement=0.97, bitwise=False, rms_max=0.05, statistical=True))
            (d/'compare-m8b-fp32-e2-tiled-m64-prompt2048.json').write_text(json.dumps(ok))
            (d/'compare-m8b-fp32-e2-bf16-prompt2048.json').write_text(json.dumps(bad))
            (d/'compare-broken.json').write_text('{}')
            code, out = self.run_tool('summarize_e2.py', 'checks', d)
            self.assertEqual(code, 1, out)
            self.assertIn('| m8b-fp32-e2-tiled-m64-prompt2048 | max 0.001, argmax 100% | 4e-05 | 100.0% | PASS |', out)
            self.assertIn('| m8b-fp32-e2-bf16-prompt2048 | max 2, rms 0.05, argmax 97% (statistical) | 0.2 | 90.0% | FAIL |', out)
            self.assertIn('2 comparisons, 1 failed.', out)
        with tempfile.TemporaryDirectory() as temp:
            d = Path(temp)
            for tag, times in (('attnsplit', (10.0, 11.0, 12.0)), ('attntiled', (7.0, 8.0, 9.0)), ('proftiled', (8.5,))):
                for rep, t in enumerate(times, 1):
                    (d/f'm8b-bf16-teams-file-s1-t56-m256-p8192-c8448-kamx_bf16-x{tag}-{rep}.json').write_text(json.dumps(
                        dict(status='completed', timing_seconds=dict(time_to_first_token=t),
                             throughput_tokens_per_second=dict(prefill=8192 / t))))
            (d/'m8b-bf16-teams-file-s1-t56-m256-p8192-c8448-kamx_bf16-xattnsplit-4.json').write_text(json.dumps(dict(status='error')))
            code, out = self.run_tool('summarize_e2.py', 'ttft', d)
            self.assertEqual(code, 0, out)
            lines = out.splitlines()
            self.assertEqual(lines[0].split(',')[0:3], ['configuration', 'attn', 'repeats'])
            self.assertIn('m8b-bf16-teams-file-s1-t56-m256-p8192-c8448-kamx_bf16,split,3,11.0000,10.0000,12.0000,744.7,', out)
            self.assertIn(',tiled,3,8.0000,7.0000,9.0000,1024.0,1.375', out)
            self.assertIn(',profiled-tiled,1,8.5000,8.5000,8.5000,963.8,', out)


class CheckPlacementTests(unittest.TestCase):
    """tools/check_placement.py judges the evidence in a report; it must fail loudly on any wrong or missing page."""

    @staticmethod
    def report(**changes):
        region = lambda role, node, **kw: dict(role=role, node=node, verified='full', pages_queried=100,
                                               pages_mismatched=0, pages_missing=0, thp_bytes=0, bytes=409600, **kw)
        snapshot = dict(verified=True, node=2, pages_wrong_node=0, pages_queried=50, pages_missing=10, full=True,
                        active_bytes=1000, resident_bytes=4096, reserved_bytes=8192)
        report = dict(status='completed', runtime=dict(placement='hbm'),
                      regions=[region('weights', 2), region('embedding', 0), region('kv.k', 2), region('kv.v', 2),
                               region('work', 2)],
                      kv=dict(snapshot, after_prompt=dict(snapshot)), memory_bytes={}, timing_seconds={})
        report.update(changes)
        return report

    def run_tool(self, report, *args):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp)/'report.json'
            path.write_text(json.dumps(report))
            result = subprocess.run([sys.executable, str(TOOLS/'check_placement.py'), str(path), *args],
                                    capture_output=True, text=True)
        return result.returncode, result.stdout

    ARGS = ('--placement', 'hbm', '--weights-node', '2', '--embedding-node', '0', '--kv-node', '2', '--layers', '2')

    def test_accepts_a_fully_verified_run(self):
        code, out = self.run_tool(self.report(), *self.ARGS)
        self.assertEqual(code, 0, out)
        self.assertNotIn('FAIL', out)

    def test_rejects_every_kind_of_wrong_placement(self):
        base = self.report()
        for name, mutate in {
            'page elsewhere': lambda r: r['regions'][0].update(pages_mismatched=1),
            'missing page': lambda r: r['regions'][0].update(pages_missing=1),
            'wrong node': lambda r: r['regions'][0].update(node=3),
            'sampled': lambda r: r['regions'][0].update(verified='sampled'),
            'embedding on hbm': lambda r: r['regions'][1].update(node=2),
            'kv elsewhere': lambda r: r['kv'].update(pages_wrong_node=2),
            'kv unverified': lambda r: r['kv']['after_prompt'].update(verified=False),
            'resident below active': lambda r: r['kv'].update(resident_bytes=10),
            'resident above reserved': lambda r: r['kv'].update(resident_bytes=10 ** 9),
            'failed run': lambda r: r.update(status='error'),
            'wrong placement': lambda r: r['runtime'].update(placement='ddr'),
        }.items():
            report = json.loads(json.dumps(base))
            mutate(report)
            code, out = self.run_tool(report, *self.ARGS)
            self.assertEqual(code, 1, (name, out))
            self.assertIn('FAIL', out, name)


class SummarizeE4Tests(unittest.TestCase):
    """tools/summarize_e4.py bench and checks on small synthetic report directories."""

    @staticmethod
    def report(ttft=None, decode=None, share=None, status='completed', verified='full'):
        report = dict(status=status, runtime=dict(placement='hbm'),
                      regions=[dict(role='weights', verified=verified, pages_mismatched=0, pages_missing=0, bytes=1 << 30)],
                      timing_seconds=dict(time_to_first_token=ttft), throughput_tokens_per_second=dict(decode=decode))
        if share is not None:
            report['profile_seconds'] = dict(prefill=dict(forward=10.0, attention=10.0 * share))
        return report

    @classmethod
    def write(cls, root, model, kernel, m, prompt, sockets, attn, ttfts, decodes=None, share=None, **extra):
        base = f'{model}-fp32-teams-hbm-s{sockets}-t56-m{m}-p{prompt}-k{kernel}'
        for rep, ttft in enumerate(ttfts, 1):
            decode = (decodes or [5.0] * len(ttfts))[rep - 1]
            (Path(root)/f'{base}-xattn{attn}-{rep}.json').write_text(json.dumps(cls.report(ttft, decode, **extra)))
        if share is not None:
            (Path(root)/f'{base}-xprof{attn}-1.json').write_text(json.dumps(cls.report(1.0, share=share)))

    def bench(self, root):
        result = subprocess.run([sys.executable, str(TOOLS/'summarize_e4.py'), 'bench', str(root)], capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr)
        return result.stdout

    def populate(self, root, tiled8192=(19.2, 20.4, 20.6), tiled2048=(4.4, 4.5, 4.6), decode_tiled=(5.0, 5.1, 5.2),
                 verified='full'):
        def w(*args, **kwargs):
            self.write(root, *args, **kwargs)
        w('m8b', 'amx_bf16', 256, 8192, 1, 'split', [30.6, 30.9, 31.2], [5.0, 5.1, 5.2], share=0.75)
        w('m8b', 'amx_bf16', 256, 8192, 1, 'tiled', list(tiled8192), list(decode_tiled), share=0.40, verified=verified)
        w('m8b', 'amx_bf16', 256, 8192, 2, 'split', [14.0, 14.2, 14.4], [6.0, 6.1, 6.2])
        w('m8b', 'amx_bf16', 256, 8192, 2, 'tiled', [10.0, 10.1, 10.2], [6.0, 6.1, 6.2])
        w('m8b', 'amx_bf16', 256, 2048, 1, 'split', [4.5, 4.6, 4.7], [5.0, 5.1, 5.2], share=0.30)
        w('m8b', 'amx_bf16', 256, 2048, 1, 'tiled', list(tiled2048), [5.0, 5.1, 5.2], share=0.15)
        w('m1b', 'avx512_fp32', 64, 8192, 1, 'split', [9.0, 9.1, 9.2])
        w('m1b', 'avx512_fp32', 64, 8192, 1, 'tiled', [7.0, 7.1, 7.2])

    def test_matrix_and_passing_gates(self):
        with tempfile.TemporaryDirectory() as temp:
            self.populate(temp)
            out = self.bench(temp)
        row = [line for line in out.splitlines() if line.startswith('| m8b | amx_bf16 | 256 | 8192 | 1 | tiled')][0]
        cells = [c.strip() for c in row.strip('|').split('|')]
        self.assertEqual(cells[6], '3')
        self.assertEqual(cells[7], '20.400 [19.200-20.600]')  # median and range of TTFT
        self.assertEqual(cells[8], '1.51x')  # 30.9 / 20.4
        self.assertEqual(cells[9], '40%')  # attention share of the profiled repeat
        self.assertEqual(cells[10], '5.1 [5.0-5.2]')
        split = [line for line in out.splitlines() if line.startswith('| m8b | amx_bf16 | 256 | 8192 | 1 | split')][0]
        self.assertIn('| 75% |', split)
        self.assertIn('Value gate (the amx_bf16 rung): **PASS**', out)
        self.assertIn('| yes | PASS |', out)  # slowest tiled repeat 20.6 < fastest split repeat 30.6: beyond variability
        self.assertIn('2048 gate: **PASS**', out)
        self.assertIn('Decode gate: **PASS**', out)
        self.assertIn('Placement gate: **PASS**', out)
        self.assertIn('| m8b | amx_bf16 | 256 | 8192 | tiled | 20.400 [19.200-20.600] | 10.100 [10.000-10.200] | 2.02x |', out)
        self.assertIn('- value: PASS', out)

    def test_decode_noise_and_placement_evidence(self):
        # Job 5864134: with three repeats a variant median just outside the split range is noise when the ranges overlap; a
        # faster variant is not a regression; two-socket reports say placement 'file' and one-socket work regions are not
        # verified by design. Only non-overlapping slower decode or bad region evidence fails.
        def gates(decode_tiled, regions=None, placement='hbm'):
            with tempfile.TemporaryDirectory() as temp:
                self.populate(temp, decode_tiled=decode_tiled)
                if regions is not None:
                    for path in Path(temp).glob('m8b-*-s1-*-xattntiled-*.json'):
                        data = json.loads(path.read_text())
                        data['regions'], data['runtime']['placement'] = regions, placement
                        path.write_text(json.dumps(data))
                out = self.bench(temp)
            return ('Decode gate: **PASS**' in out, 'Placement gate: **PASS**' in out)
        full = lambda role: dict(role=role, verified='full', pages_mismatched=0, pages_missing=0, bytes=1 << 30)
        self.assertEqual(gates((5.15, 5.25, 5.3))[0], True)   # median 5.25 above split's 5.2, ranges overlap
        self.assertEqual(gates((5.3, 5.4, 5.5))[0], True)     # faster, no overlap: not a regression
        self.assertEqual(gates((4.5, 4.6, 4.7))[0], False)    # slower, no overlap
        work = dict(role='work', verified='unverified', pages_mismatched=0, pages_missing=0, bytes=70 << 20)
        self.assertEqual(gates((5.0, 5.1, 5.2), [full('weights'), full('kv.k'), work])[1], True)
        self.assertEqual(gates((5.0, 5.1, 5.2), [full('weights'), full('kv.k')], placement='file')[1], True)
        self.assertEqual(gates((5.0, 5.1, 5.2), [dict(full('weights'), pages_mismatched=3)])[1], False)
        self.assertEqual(gates((5.0, 5.1, 5.2), [dict(full('kv.k'), verified='sampled')])[1], False)
        self.assertEqual(gates((5.0, 5.1, 5.2), [])[1], False)

    def test_each_gate_fails_on_its_own_condition(self):
        for name, kwargs, text in [
                ('value', dict(tiled8192=(25.0, 25.5, 26.0)), 'Value gate (the amx_bf16 rung): **FAIL**'),
                ('2048', dict(tiled2048=(4.8, 4.9, 5.0)), '2048 gate: **FAIL**'),
                ('decode', dict(decode_tiled=(4.0, 4.1, 4.2)), 'Decode gate: **FAIL**'),
                ('placement', dict(verified='sampled'), 'Placement gate: **FAIL**')]:
            with self.subTest(gate=name), tempfile.TemporaryDirectory() as temp:
                self.populate(temp, **kwargs)
                out = self.bench(temp)
                self.assertIn(text, out)
                self.assertEqual(out.count('**FAIL**'), 1, out)  # the other gates are not affected

    def test_overlapping_ranges_are_not_beyond_variability(self):
        with tempfile.TemporaryDirectory() as temp:
            self.populate(temp, tiled8192=(19.2, 20.4, 31.0))  # the slowest tiled repeat is above the fastest split one (30.6)
            out = self.bench(temp)
        self.assertIn('| no | PASS |', out)

    def test_empty_directory_and_failed_runs(self):
        with tempfile.TemporaryDirectory() as temp:
            self.assertIn('No E4 reports', self.bench(temp))
            self.write(temp, 'm8b', 'amx_bf16', 256, 8192, 1, 'split', [30.0, 31.0])
            self.write(temp, 'm8b', 'amx_bf16', 256, 8192, 1, 'tiled', [None], status='error')
            out = self.bench(temp)
        self.assertIn('(1 failed)', out)
        self.assertIn('Placement gate: **FAIL**', out)
        self.assertIn('- value: no data', out)

    def test_checks_title_and_failure_status(self):
        with tempfile.TemporaryDirectory() as temp:
            summary = dict(max_abs=0.0005, argmax_agreement=1.0)
            good = dict(passed=True, summary=summary, budget=dict(max_abs=1e-3, argmax_agreement=1.0))
            (Path(temp)/'compare-m8b-fp32-e4-tiled-m64-prompt2048.json').write_text(json.dumps(good))

            def run():
                return subprocess.run([sys.executable, str(TOOLS/'summarize_e4.py'), 'checks', temp], capture_output=True, text=True)
            result = run()
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertIn('# E4 correctness', result.stdout)
            self.assertIn('| m8b-fp32-e4-tiled-m64-prompt2048 | max 0.001, argmax 100% |', result.stdout)
            (Path(temp)/'compare-m8b-fp32-e4-bad.json').write_text(json.dumps(dict(good, passed=False)))
            self.assertEqual(run().returncode, 1)


class PhaseE4DryRunTests(unittest.TestCase):
    """phase_E4 prints its matrix and its gates without running anything (DRY_RUN=1)."""

    def run_phase(self, stage, **extra):
        if not shutil.which('bash'):
            self.skipTest('bash is required')
        with tempfile.TemporaryDirectory() as temp:
            env = {**campaign_env(), 'DRY_RUN': '1', 'SLURM_JOB_ID': 'dry', 'ROOT': str(ROOT), 'OUT': str(Path(temp)/'out'),
                   'PYTHON': sys.executable, 'PHASE': 'E4', 'STAGE': stage, 'MODELS': 'm1b:/a/m1b:/hf/m1b m8b:/a/m8b:/hf/m8b', **extra}
            result = subprocess.run(['bash', str(ROOT/'scripts/cresco8-prefill.sbatch')], env=env, capture_output=True, text=True,
                                    timeout=120)
        self.assertEqual(result.returncode, 0, result.stderr[-2000:])
        return result.stdout.splitlines()

    def runs(self, stage='bench', **extra):
        from summarize_campaign import parse_label
        runs = [line for line in self.run_phase(stage, **extra) if line.startswith('RUN ')]
        for line in runs:
            self.assertIsNotNone(parse_label(line.split()[1] + '-1'), line)
        return runs

    def test_bench_matrix(self):
        runs = self.runs()
        self.assertEqual(len(runs), 256)  # 2 models x 8 rung-variant pairs x 2 prompts x 2 sockets x (3 repeats + 1 profile)
        timed = [line for line in runs if '-xattn' in line]
        self.assertEqual(len(timed), 192)
        self.assertTrue(all('gen=32' in line for line in timed))
        profiled = [line for line in runs if '-xprof' in line]
        self.assertEqual(len(profiled), 64)
        self.assertTrue(all('LLAMA_PROFILE=1' in line and 'gen=1 ' in line for line in profiled))
        self.assertFalse(any('LLAMA_PROFILE=1' in line for line in timed))
        self.assertEqual({line.split('-s')[1][0] for line in runs}, {'1', '2'})
        self.assertEqual({line.split('-p')[1].split('-')[0] for line in runs}, {'2048', '8192'})
        self.assertEqual({line.split('-m')[1].split('-')[0] for line in runs}, {'64', '128', '256'})
        self.assertFalse(any('LLAMA_ATTN=amx' in line for line in runs))
        self.assertTrue(all('LLAMA_SOCKETS=2' in line for line in runs if '-s2-' in line))
        self.assertTrue(all('cpu=0 ' in line for line in runs if '-s1-' in line))
        self.assertTrue(all('LLAMA_PLACEMENT=hbm' in line for line in runs))
        self.assertTrue(all('LLAMA_KERNELS=amx' in line for line in runs if 'kamx_int8' in line))

    def test_check_writes_only_into_its_own_directory(self):
        # Job 5864123: the generation comparisons wrote into E1/, which a job without phase E1 does not create.
        lines = [line for line in self.run_phase('check') if line.startswith('EXPECT ')]
        gen = [line for line in lines if line.startswith('EXPECT gen-')]
        self.assertEqual(len(gen), 12)
        self.assertTrue(all('/out/E4/gen-' in line for line in gen), gen[0])
        self.assertFalse(any('/out/E1/' in line for line in lines))

    def test_amx_only_on_the_bf16_rung_and_only_when_opted_in(self):
        runs = self.runs(E4_AMX='1')
        self.assertEqual(len(runs), 288)
        amx = [line for line in runs if 'LLAMA_ATTN=amx' in line]
        self.assertEqual(len(amx), 32)  # 2 models x 2 prompts x 2 sockets x 4 runs
        self.assertTrue(all('kamx_bf16' in line and 'LLAMA_PRECISION=bf16' in line for line in amx))
        self.assertTrue(all('-xattnamx-' not in line or 'kamx_bf16' in line for line in runs))

    def test_attention_list_and_tiled_knob_are_overridable(self):
        knob = 'LLAMA_ATTN_TILED_QUERIES=8 LLAMA_ATTN_TILED_ORDER=natural'
        runs = self.runs(E4_ATTN='tiled', E4_TILED_ENV=knob)
        self.assertEqual(len(runs), 128)
        self.assertTrue(all(line.endswith(f'LLAMA_ATTN=tiled {knob}') for line in runs))
        both = self.runs(E4_TILED_ENV='LLAMA_ATTN_TILED_QUERIES=8')
        self.assertTrue(all('TILED_QUERIES' not in line for line in both if 'LLAMA_ATTN=split' in line))
        self.assertEqual(sum('TILED_QUERIES=8' in line for line in both), 128)
        self.assertEqual(len(self.runs(REPEATS='1')), 128)  # (1 repeat + 1 profile) x 32 cells x 2 models

    def test_check_gates_every_variant(self):
        lines = self.run_phase('check')
        names = [line.split()[1] for line in lines if line.startswith('EXPECT ')]
        self.assertEqual(len(names), 42)
        for model in ('m1b', 'm8b'):
            for attn in ('split', 'tiled'):
                for case in (f'compare-{model}-fp32-e4-{attn}-m64-prompt86', f'compare-{model}-fp32-e4-{attn}-m64-prompt2048',
                             f'compare-{model}-fp32-e4-{attn}-m128-prompt2048', f'compare-{model}-fp32-e4-s2-{attn}-m64-prompt2048',
                             f'compare-{model}-fp32-e4-s2-bf16-{attn}-m256-prompt2048',
                             f'compare-{model}-int8-e4-s2-w8a8-{attn}-m256-prompt2048',
                             f'gen-fp32-e4-{model}-2048-{attn}-s1-vs-s2-1', f'gen-fp32-e4-{model}-short-{attn}-s1-vs-s2-1'):
                    self.assertIn(case, names)
            self.assertIn(f'compare-{model}-fp32-e4-bf16-tiled-vs-split-prompt86', names)
            self.assertIn(f'compare-{model}-int8-e4-w8a8-tiled-vs-split-prompt2048', names)
            self.assertIn(f'gen-fp32-e4-{model}-2048-split-vs-tiled-1', names)
            self.assertNotIn(f'compare-{model}-fp32-e4-bf16-split-vs-split-prompt86', names)
        self.assertFalse(any('amx' in name for name in names))
        self.assertNotIn('LLAMA_ATTN=amx', '\n'.join(lines))
        # FP32 strict against the one-socket per-token reference, BF16 and W8A8 their statistical gates, two sockets checked
        fp32 = [line for line in lines if line.startswith('EXPECT compare-m8b-fp32-e4-tiled-m64-prompt2048')][0]
        self.assertIn('--max-abs 1e-3 --batch-a 1 --batch-b 64 --last-only', fp32)
        two = [line for line in lines if line.startswith('EXPECT compare-m8b-fp32-e4-s2-tiled-m64')][0]
        self.assertIn('--env-b LLAMA_SOCKETS=2 --env-b LLAMA_TP_CHECK=1', two)
        self.assertIn('--env-a LLAMA_SOCKETS=1', two)
        self.assertIn('--max-abs 2 --rms-max 0.05 --argmax-min 0.97', [x for x in lines if 'e4-bf16-tiled-vs-split-prompt86' in x][0])
        self.assertIn('--max-abs 16 --rms-max 0.2', [x for x in lines if 'e4-w8a8-tiled-vs-split-prompt2048' in x][0])
        self.assertIn('--env-a LLAMA_ATTN=split', [x for x in lines if 'gen-fp32-e4-m8b-2048-split-vs-tiled' in x][0])

    def test_check_with_amx_and_the_tiled_knob(self):
        lines = self.run_phase('check', E4_AMX='1', E4_TILED_ENV='LLAMA_ATTN_TILED_QUERIES=8')
        names = [line.split()[1] for line in lines if line.startswith('EXPECT ')]
        self.assertEqual(len(names), 56)
        amx = [name for name in names if 'amx' in name]
        self.assertEqual(len(amx), 14)  # per model: bf16 against split x 2, two sockets, and 4 generation comparisons
        self.assertTrue(all('bf16' in name for name in amx))
        self.assertIn('gen-bf16-e4-m8b-2048-split-vs-amx-1', names)
        for line in lines:
            if line.startswith('EXPECT ') and 'LLAMA_ATTN=tiled' in line:
                self.assertIn('LLAMA_ATTN_TILED_QUERIES=8', line)

    def test_models_empty_is_recorded(self):
        lines = self.run_phase('bench', MODELS='')
        self.assertTrue(any('no E4 run was made' in line for line in lines))


class DecodeAttentionCampaignTests(unittest.TestCase):
    """E5: phase_E5 of scripts/cresco8-tp.sbatch (LLAMA_ATTN_DECODE=group) prints its matrix without running anything."""

    def e5(self, stage, **extra):
        if not shutil.which('bash'):
            self.skipTest('bash is required')
        with tempfile.TemporaryDirectory() as temp:
            env = {**campaign_env(), 'DRY_RUN': '1', 'SLURM_JOB_ID': 'dry', 'ROOT': str(ROOT), 'OUT': str(Path(temp)/'out'),
                   'PYTHON': sys.executable, 'PHASE': 'E5', 'STAGE': stage, 'REPEATS': '3',
                   'MODELS': 'm1b:/a/m1b:/hf/m1b m8b:/a/m8b:/hf/m8b', **extra}
            result = subprocess.run(['bash', str(ROOT/'scripts'/'cresco8-tp.sbatch')], env=env, capture_output=True,
                                    text=True, timeout=120)
            self.assertEqual(result.returncode, 0, result.stderr[-2000:])
        return result.stdout.splitlines()

    def test_check_stage_matrix(self):
        out = self.e5('check')
        compares = [l for l in out if l.startswith('EXPECT compare-')]
        self.assertEqual(len(compares), 2 * 2 * 3)  # models x precisions x (one socket, split vs two-socket group, 2048)
        for line in compares:
            self.assertIn('LLAMA_ATTN_DECODE=split', line)
            self.assertIn('LLAMA_ATTN_DECODE=group', line)
            self.assertNotIn('--batch-a', line)  # the per-token path: the knob acts in decode
            self.assertNotIn('--batch-b', line)
            self.assertNotIn('--last-only', line)
            gate = '--rms-max 0.2' if '-int8-' in line else '--max-abs 1e-3'
            self.assertIn(gate, line)
        self.assertTrue(all('LLAMA_TP_CHECK=1' in l and 'LLAMA_SOCKETS=2' in l for l in compares if 's1split-vs-s2group' in l))
        self.assertTrue(all('m8b-tokens-2048' in l or 'm1b-tokens-2048' in l for l in compares if '-2048-' in l))
        gen = [l for l in out if 'compare_generation.py' in l]
        self.assertEqual(len(gen), 2 * 2 * 2)  # models x precisions x (one socket, two sockets)
        for line in gen:
            for word in ('--steps 128', '--warm-context 8192', '--env-a LLAMA_ATTN_DECODE=split',
                         '--env-b LLAMA_ATTN_DECODE=group', '--probe-batch 64'):
                self.assertIn(word, line)
        self.assertTrue(all('LLAMA_TP_CHECK=1' in l for l in gen if '-s2-' in l))
        hf = [l for l in out if l.startswith('VALIDATE ')]
        self.assertEqual(len(hf), 2 * 2)  # FP32 only, one socket and two sockets
        self.assertTrue(all('fp32' in l and 'LLAMA_ATTN_DECODE=group' in l for l in hf))
        self.assertEqual(len([l for l in hf if 'LLAMA_SOCKETS=2' in l and 'LLAMA_TP_CHECK=1' in l]), 2)
        self.assertEqual([l for l in out if l.startswith('RUN ')], [])
        # HF_GATE=0 records that the Hugging Face gate was not run
        self.assertEqual([l for l in self.e5('check', HF_GATE='0') if l.startswith('VALIDATE ')], [])

    def test_bench_stage_matrix(self):
        from summarize_campaign import parse_label
        out = self.e5('bench')
        runs = [l for l in out if l.startswith('RUN ')]
        # models x precisions x (3 repeats x 3 contexts + 3 profiled) x two configurations x two kernels
        self.assertEqual(len(runs), 2 * 2 * (3 * 3 + 3) * 2 * 2)
        for line in runs:
            self.assertIsNotNone(parse_label(line.split()[1] + '-1'), line)
            kernel = 'group' if '-xattngroup' in line else 'split'
            self.assertIn(f'LLAMA_ATTN_DECODE={kernel}', line)
            self.assertEqual(line.count('LLAMA_ATTN_DECODE='), 1)
            self.assertIn('LLAMA_PROFILE=1' if 'prof ' in line else 'LLAMA_PREFILL_GEMM=on', line)
        self.assertEqual({l.split('warm=')[1].split()[0] for l in runs}, {'512', '8192', '32768'})
        self.assertEqual({l.split('-s')[1][0] for l in (r.split()[1] for r in runs)}, {'1', '2'})
        self.assertTrue(all('LLAMA_PROFILE=1' in l for l in runs if 'prof ' in l))
        self.assertFalse([l for l in runs if 'LLAMA_PROFILE=1' in l and 'prof ' not in l])
        self.assertFalse([l for l in runs if '-s2-t14-' in l or 'xctl' in l or 'xp4' in l])  # a subset of B7
        bench = [l for l in out if l.startswith('EXPECT bench-decode-') and 'attention-bench' in l]
        self.assertEqual(len(bench), 2 * 2 * 2)  # shapes x KV heads (all, half) x kernels
        for line in bench:
            for word in ('--mode decode', '--context 512,8192,32768', '--reps 1000', '--node 2', '--cpunodebind=0'):
                self.assertIn(word, line)
        self.assertEqual(sorted(l.split('LLAMA_ATTN_DECODE=')[1].split()[0] for l in bench), ['group'] * 4 + ['split'] * 4)
        self.assertEqual([l for l in out if l.startswith('EXPECT compare-')], [])  # the correctness gates are the check stage's


if __name__ == '__main__':
    unittest.main()
