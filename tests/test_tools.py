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


def report(prefill, decode, ttft=0.5, status='completed', placement='file', regions=None, thp='off', kernels='avx512'):
    return {'schema_version': 2, 'status': status, 'build': {'kernels': kernels},
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
            env = {**os.environ, 'DRY_RUN': '1', 'SLURM_JOB_ID': '1', 'ROOT': str(ROOT), 'OUT': str(Path(temp)/'out'),
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
            env = {**os.environ, 'DRY_RUN': '1', 'SLURM_JOB_ID': '1', 'ROOT': str(ROOT), 'OUT': str(Path(temp)/'out'),
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
            env = {**os.environ, 'DRY_RUN': '1', 'SLURM_JOB_ID': '1', 'ROOT': str(ROOT), 'OUT': str(Path(temp)/'out'),
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
            env = {**os.environ, 'DRY_RUN': '1', 'SLURM_JOB_ID': '1', 'ROOT': str(ROOT), 'OUT': str(Path(temp)/'out'),
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
                env = {**os.environ, 'SLURM_JOB_ID': '1', 'ROOT': str(ROOT), 'OUT': str(Path(temp)/'out'),
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
        for engine in ('fp32', 'int8'):
            for batch in ('3', '32'):
                result, data = self.compare(engine, '--batch-b', batch, '--bitwise')
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
        for engine in ('fp32', 'int8'):
            for batch in ('2', '7', '32'):
                result, data = self.compare(engine, '--env-b', 'LLAMA_PREFILL_GEMM=on', '--batch-b', batch, '--bitwise')
                self.assertEqual(result.returncode, 0, (engine, batch, result.stderr))
                self.assertTrue(data['summary']['bitwise'])
        result, _ = self.compare('fp32', '--env-b', 'LLAMA_PREFILL_GEMM=sometimes', '--batch-b', '2')
        self.assertNotEqual(result.returncode, 0)

    def test_bad_arguments(self):
        result, _ = self.compare('fp32', '--env-b', 'NOT_AN_ASSIGNMENT')
        self.assertNotEqual(result.returncode, 0)
        self.assertIn('KEY=VALUE', result.stderr)


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


if __name__ == '__main__':
    unittest.main()
