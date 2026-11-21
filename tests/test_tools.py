"""Tool and job-script checks without installing large ML dependencies."""
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

# Knobs the job script reads from the environment. A test that runs it starts from the caller's environment without them,
# so a run inside a cluster job (which exports STAGE, MODELS and so on) sees only what the test sets.
CAMPAIGN_KNOBS = {'BENCH_CONTEXTS', 'BENCH_PROFILE', 'BENCH_PROMPTS', 'BENCH_SOCKETS', 'CONTEXT', 'CPU_BIND', 'DRY_RUN',
                  'GENERATION', 'LOG_LEVEL', 'MEM_BIND', 'MODELS', 'OUT', 'PERF', 'PHASE', 'PRECISIONS', 'PYTHON', 'REP',
                  'REPEATS', 'ROOT', 'SKIP_TEST', 'SLURM_JOB_ID', 'STAGE', 'WARM'}


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
           attention=None, attention_decode=None, kv_layout=None):
    build = {'kernels': kernels}
    if kv_layout is not None:
        build['kv_layout'] = kv_layout
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


    def dry_run(self, stage, models=('m1b', 'm8b'), **extra):
        """Runs scripts/cresco8.sbatch with DRY_RUN=1 and returns its output lines; every run label must parse."""
        if not shutil.which('bash'):
            self.skipTest('bash is required')
        from summarize_campaign import parse_label
        with tempfile.TemporaryDirectory() as temp:
            env = {**campaign_env(), 'DRY_RUN': '1', 'SLURM_JOB_ID': '1', 'ROOT': str(ROOT), 'OUT': str(Path(temp)/'out'),
                   'PYTHON': sys.executable, 'STAGE': stage,
                   'MODELS': ' '.join(f'{m}:/a/{m}:/hf/{m}' for m in models), **extra}
            result = subprocess.run(['bash', str(ROOT/'scripts'/'cresco8.sbatch')], env=env, capture_output=True, text=True,
                                    timeout=120)
            self.assertEqual(result.returncode, 0, result.stderr[-2000:])
        lines = result.stdout.splitlines()
        for line in lines:
            if line.startswith('RUN '):
                self.assertIsNotNone(parse_label(line.split()[1] + '-1'), line)
        return lines

    def test_check_stage_gates_every_precision_at_the_defaults(self):
        lines = self.dry_run('check')
        compares = [l for l in lines if l.startswith('EXPECT compare-')]
        self.assertEqual(len(compares), 2 * 6)  # models x (fp32: 1s-vs-2s, defaults, bf16; int8: w8a8, w8a32, defaults)
        by = lambda name: next(l for l in compares if f'compare-m8b-{name} ' in l)
        strict = by('fp32-1s-vs-2s')
        self.assertIn('--max-abs 1e-3', strict)
        self.assertIn('--env-b LLAMA_SOCKETS=2 --env-b LLAMA_TP_CHECK=1', strict)
        defaults = by('fp32-reference-vs-defaults')
        # side B names no configuration but the batch: the engine's defaults (two sockets, tiled attention, head K/V)
        words = defaults.split()
        self.assertEqual([words[i + 1] for i, w in enumerate(words) if w == '--env-b'], ['LLAMA_TP_CHECK=1'])
        self.assertIn('--batch-b 64', defaults)
        bf16 = by('fp32-reference-vs-bf16')
        for word in ('--env-b LLAMA_PRECISION=bf16', '--batch-b 256', '--rms-max 0.05', '--argmax-min 0.97'):
            self.assertIn(word, bf16)
        self.assertNotIn('1e-3', bf16)
        for name in ('int8-w8a8-1s-vs-2s', 'int8-w8a8-reference-vs-defaults'):
            self.assertIn('--rms-max 0.2', by(name))
            self.assertNotIn('1e-3', by(name))
        self.assertIn('--batch-b 256', by('int8-w8a8-reference-vs-defaults'))
        self.assertIn('--env-b LLAMA_Q8_ACTIVATIONS=fp32', by('int8-w8a32-1s-vs-2s'))
        self.assertEqual(len([l for l in lines if l.startswith('VALIDATE ')]), 2 * 3)
        runs = [l for l in lines if l.startswith('RUN ')]
        self.assertEqual(sorted(l.split()[1] for l in runs if 'm8b' in l),
                         ['m8b-bf16-teams-hbm-s2-t56-xsmoke', 'm8b-fp32-teams-hbm-s2-t56-xsmoke', 'm8b-int8-teams-hbm-s2-t56-xsmoke'])
        self.assertTrue(all('LLAMA_PROFILE=1' in l and 'LLAMA_SOCKETS' not in l for l in runs))
        # PRECISIONS restricts the checkpoints, and without MODELS only the build and the tests would run.
        self.assertFalse([l for l in self.dry_run('check', PRECISIONS='int8') if 'probe-fp32 ' in l])
        self.assertFalse([l for l in self.dry_run('check', models=()) if l.startswith(('EXPECT', 'RUN', 'VALIDATE'))])

    def test_bench_stage_matrix(self):
        lines = self.dry_run('bench', models=('m8b',), REPEATS='2')
        runs = [l for l in lines if l.startswith('RUN ')]
        timed = [l for l in runs if 'prof' not in l.split()[1]]
        profiled = [l for l in runs if 'prof' in l.split()[1]]
        # per repeat: fp32 3 prompts x (fp32, bf16) + 3 contexts; int8 3 prompts + 3 contexts + w8a32 at 512 and 8192; x 2 sockets
        self.assertEqual(len(timed), 2 * 2 * (3 * 2 + 3 + 3 + 3 + 2))
        self.assertEqual(len(profiled), len(timed) // 2)
        self.assertTrue(all('LLAMA_PROFILE=1' in l for l in profiled) and not [l for l in timed if 'LLAMA_PROFILE' in l])
        decode = [l for l in timed if '-xdec' in l]
        self.assertEqual({l.split('warm=')[1].split()[0] for l in decode}, {'512', '8192', '32768'})
        self.assertFalse([l for l in decode if 'w8a32' in l and 'warm=32768' in l])
        self.assertTrue(all('LLAMA_Q8_ACTIVATIONS=fp32' in l for l in runs if '-w8a32-' in l))
        self.assertTrue(all('LLAMA_PRECISION=bf16' in l for l in runs if '-bf16-' in l))
        self.assertFalse([l for l in decode if '-bf16-' in l])  # BF16 changes the prefill only
        self.assertTrue(all('cpu=0 ' in l and 'LLAMA_SOCKETS=1' in l for l in runs if '-s1-' in l))
        self.assertTrue(all('cpu= ' in l and 'LLAMA_SOCKETS=2' in l for l in runs if '-s2-' in l))
        self.assertFalse([l for l in runs if 'LLAMA_PREFILL_BATCH' in l or 'LLAMA_KERNELS' in l or 'LLAMA_ATTN' in l])
        # the repeats interleave per checkpoint: every cell of repeat 1 before any of repeat 2
        for family in (('-fp32-', '-bf16-'), ('-int8-', '-w8a32-')):
            reps = [l.split('reps=')[1].split()[0] for l in timed if any(f in l for f in family)]
            self.assertEqual(reps, sorted(reps), family)
        # the lists are knobs
        small = [l for l in self.dry_run('bench', models=('m1b',), REPEATS='1', PRECISIONS='fp32', BENCH_SOCKETS='2',
                                         BENCH_PROMPTS='2048', BENCH_CONTEXTS='4096', BENCH_PROFILE='0') if l.startswith('RUN ')]
        self.assertEqual(sorted(l.split()[1] for l in small),
                         ['m1b-bf16-teams-hbm-s2-t56-p2048', 'm1b-fp32-teams-hbm-s2-t56-p2048', 'm1b-fp32-teams-hbm-s2-t56-p4096-xdec'])

    def test_bad_knobs_stop_before_anything_runs(self):
        if not shutil.which('bash'):
            self.skipTest('bash is required')
        for patch, text in [({'BENCH_SOCKETS': '3'}, 'BENCH_SOCKETS'), ({'BENCH_PROMPTS': 'x'}, 'token counts'),
                            ({'BENCH_PROFILE': '2'}, 'BENCH_PROFILE'), ({'PHASE': 'B7'}, 'campaigns-2026-11'),
                            ({'PRECISIONS': 'bf16'}, 'PRECISIONS'), ({'STAGE': 'campaign'}, 'STAGE must be')]:
            with tempfile.TemporaryDirectory() as temp:
                env = {**campaign_env(), 'DRY_RUN': '1', 'SLURM_JOB_ID': '1', 'ROOT': str(ROOT), 'OUT': str(Path(temp)/'out'),
                       'PYTHON': sys.executable, 'MODELS': 'm1b:/a:/h', **patch}
                result = subprocess.run(['bash', str(ROOT/'scripts'/'cresco8.sbatch')], env=env, capture_output=True,
                                        text=True, timeout=60)
                self.assertNotEqual(result.returncode, 0, patch)
                self.assertIn(text, result.stderr, patch)
                self.assertFalse((Path(temp)/'out').exists(), patch)


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
        # E5.4: the K/V layout is written only when it is not the default rows
        self.assertEqual(ct.attention_tag(report(1, 1, attention='split', attention_decode='split', kv_layout='rows')), '')
        self.assertEqual(ct.attention_tag(report(1, 1, attention='tiled', attention_decode='group', kv_layout='head')),
                         'attn=tiled decode-attn=group kv=head')
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
        self.assertIn('| 48.0 |', row)  # 1.5e9 hbm_cas_rd * 32 B (ddr 2e6 * 64 B = 0.1 GB)
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

    def test_tail_mode_compares_the_decode_path_after_a_batched_prefill(self):
        """E5.4: `--tail K` compares the logits of the last K positions, each a per-token forward after the rest of the prompt
        (batched with --batch-a/--batch-b), and the probe's `tail` mode is exactly the last K rows of the per-token run when
        nothing is batched. The K/V layouts must agree bit for bit (--bitwise); a bad K is refused."""
        tokens = [str(65 + (i * 7) % 20) for i in range(20)]
        for engine in ('fp32', 'int8'):
            probe, model = str(BUILD/f'probe-{engine}'), str(self.fixture/f'{engine}.bin')
            # Nothing batched: since F4 the prefix of `tail` is batched by default where the GEMM runs, and the tiled
            # prefill attention is not bitwise the per-token path (FP32 on a real AVX-512 host differs in the last bits).
            unbatched = {**os.environ, 'LLAMA_PREFILL_BATCH': '1'}
            every = subprocess.run([probe, model, *tokens], capture_output=True, timeout=120, env=unbatched).stdout
            tail = subprocess.run([probe, 'tail', model, '5', *tokens], capture_output=True, timeout=120, env=unbatched)
            self.assertEqual(tail.returncode, 0, tail.stderr)
            self.assertEqual(tail.stdout, every[-5 * len(every) // 20:])
            for batched in ([], ['--batch-a', '8', '--batch-b', '8']):
                with self.subTest(engine=engine, batched=bool(batched)):
                    result = subprocess.run([sys.executable, str(TOOLS/'compare_engines.py'), '--probe', probe, '--checkpoint',
                                             model, '--tokens', ','.join(tokens), '--tail', '5', '--bitwise', '--env-a',
                                             'LLAMA_KV_LAYOUT=rows', '--env-b', 'LLAMA_KV_LAYOUT=head', *batched],
                                            capture_output=True, text=True, timeout=120)
                    self.assertEqual(result.returncode, 0, result.stderr)
                    data = json.loads(result.stdout)
                    self.assertTrue(data['passed'] and data['summary']['bitwise'])
                    self.assertEqual((data['tail'], [c['position'] for c in data['comparisons']]), (5, list(range(15, 20))))
        for bad in ('0', '21'):
            result = subprocess.run([sys.executable, str(TOOLS/'compare_engines.py'), '--probe', str(BUILD/'probe-fp32'),
                                     '--checkpoint', str(self.fixture/'fp32.bin'), '--tokens', ','.join(tokens), '--tail', bad],
                                    capture_output=True, text=True, timeout=120)
            self.assertNotEqual(result.returncode, 0)

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
        result, data = self.compare('--env-a', 'LLAMA_THREADS=1', '--env-b', 'LLAMA_BACKEND=teams',
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

    def test_require_identical_turns_the_comparison_into_a_gate(self):
        for generated_b, status in (('[70,71,72,73]', 0), ('[70,71,99,73]', 1)):
            result, data = self.compare('--env-a', 'STUB_GENERATED=[70,71,72,73]', '--env-b', f'STUB_GENERATED={generated_b}',
                                        '--require-identical', engine=self.stub, steps='4')
            self.assertEqual(result.returncode, status, result.stderr)
            self.assertEqual(data['identical'], status == 0)
        # without the flag a divergence is only recorded
        result, data = self.compare('--env-a', 'STUB_GENERATED=[70,71,72,73]', '--env-b', 'STUB_GENERATED=[70,71,99,73]',
                                    engine=self.stub, steps='4')
        self.assertEqual((result.returncode, data['identical']), (0, False))

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
            result = self.lib('require_env; printf "%s %s %s" "$STAGE" "$REPEATS" "$PRECISIONS"', {**base, 'OUT': out})
            self.assertEqual((result.returncode, result.stdout), (0, 'check 3 fp32 int8'), result.stderr)
            for patch, text in [({'STAGE': 'x'}, 'STAGE must be'), ({'REPEATS': '0'}, 'REPEATS must be'),
                                ({'PHASE': 'B5'}, 'PHASE is gone'), ({'MODELS': 'bad'}, 'is not NAME:ARTIFACTS:HF'),
                                ({'PRECISIONS': 'fp16'}, 'PRECISIONS must'), ({'OUT': temp}, 'already exists'),
                                ({'SLURM_JOB_ID': ''}, 'submit through Slurm')]:
                with self.subTest(patch=patch):
                    result = self.lib('require_env', {**base, 'OUT': out, **patch})
                    self.assertNotEqual(result.returncode, 0)
                    self.assertIn(text, result.stderr)
            result = self.lib('require_env', {k: v for k, v in {**base, 'OUT': out}.items() if k != 'ROOT'})
            self.assertIn('tree under test required', result.stderr)
            # MODELS is optional.
            result = self.lib('require_env; printf "[%s]" "$MODELS"', {**{k: v for k, v in base.items() if k != 'MODELS'},
                                                                       'OUT': out})
            self.assertEqual((result.returncode, result.stdout), (0, '[]'), result.stderr)

    def test_expect(self):
        with tempfile.TemporaryDirectory() as temp:
            script = (f'OUT={temp}; expect good ok {temp} -- true; expect bad fail {temp} -- false; '
                      f'expect wrong ok {temp} -- false; expect free any {temp} -- false; '
                      f'expect talk ok {temp} -- echo hello; cat {temp}/phase-results.txt; echo "overall=$OVERALL"; '
                      f'cat {temp}/talk.out')
            result = self.lib(script)
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertEqual(result.stdout.splitlines(), [
                'PASS good exit=0 expected=ok', 'PASS bad exit=1 expected=fail', 'FAIL wrong exit=1 expected=ok',
                'PASS free exit=1 expected=any', 'PASS talk exit=0 expected=ok', 'overall=1', 'hello'])

    def test_compare_takes_flags_and_both_environments(self):
        with tempfile.TemporaryDirectory() as temp:
            result = self.lib(f'OUT={temp}; DRY_RUN=1 ROOT=/r PYTHON=py compare {temp} m fp32 /c lab --batch-b 64 -- A=1 B=2 -- C=3')
            self.assertEqual(result.returncode, 0, result.stderr)
            words = result.stdout.split()
            for pair in (['--env-a', 'A=1'], ['--env-a', 'B=2'], ['--env-b', 'C=3'], ['--batch-b', '64'], ['--max-abs', '1e-3']):
                self.assertIn(' '.join(pair), ' '.join(words))
            result = self.lib(f'OUT={temp}; DRY_RUN=1 ROOT=/r PYTHON=py compare {temp} m int8 /c lab "${{W8A8_GATE[@]}}" -- -- X=1')
            self.assertIn('--max-abs 16 --rms-max 0.2', result.stdout)
            self.assertNotIn('1e-3', result.stdout)
            self.assertNotIn('--env-a', result.stdout)

    def test_run_labels_match_the_summarizer(self):
        import summarize_campaign as sc
        result = self.lib('run_label llama3-8b fp32 teams hbm 2 112 m=256 p=2048 c=4096 k=amx')
        self.assertEqual(result.stdout.strip(), 'llama3-8b-fp32-teams-hbm-s2-t112-m256-p2048-c4096-kamx', result.stderr)
        self.assertIsNotNone(sc.parse_label(result.stdout.strip() + '-1'))
        result = self.lib('run_label 1b w8a32 teams file 1 56 x=decprof')
        self.assertEqual(result.stdout.strip(), '1b-w8a32-teams-file-s1-t56-xdecprof')
        self.assertIsNotNone(sc.parse_label(result.stdout.strip() + '-2'))

    def test_script_refuses_to_run_without_a_cluster(self):
        result = subprocess.run(['bash', str(ROOT/'scripts/cresco8.sbatch')], capture_output=True, text=True,
                                env={'PATH': os.environ['PATH']})
        self.assertNotEqual(result.returncode, 0)
        self.assertIn('tree under test required', result.stderr)

    @unittest.skipUnless(shutil.which('shellcheck'), 'shellcheck not installed')
    def test_shellcheck(self):
        for path in self.scripts:
            with self.subTest(script=path.name):
                result = subprocess.run(['shellcheck', '-x', '-S', 'warning', str(path)], capture_output=True, text=True)
                self.assertEqual(result.returncode, 0, result.stdout)



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

    def test_head_layout_slack_is_a_page_per_layer_and_kv_head(self):
        """E5.4: with LLAMA_KV_LAYOUT=head the touched pages are a prefix of each head's span, so the resident figure may
        exceed the reserved one by a page per layer and KV head (a page per layer for the rows layout)."""
        over = 8192 + 2 * 3 * 4096  # reserved + 2 layers x 3 KV heads of slack
        for layout, status in ((None, 1), ('rows', 1), ('head', 0)):
            report = self.report(model=dict(configuration=dict(n_kv_heads=3)))
            if layout:
                report['build'] = dict(kv_layout=layout)
            report['kv'].update(resident_bytes=over)
            report['kv']['after_prompt'].update(resident_bytes=over)
            code, out = self.run_tool(report, *self.ARGS)
            self.assertEqual(code, status, (layout, out))
        report = self.report(model=dict(configuration=dict(n_kv_heads=3)), build=dict(kv_layout='head'))
        report['kv'].update(resident_bytes=over + 4096)
        self.assertEqual(self.run_tool(report, *self.ARGS)[0], 1)  # beyond the slack it still fails

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


class BenchToolTests(unittest.TestCase):
    """The C benchmarks (gemm-bench, amx-bench) the cluster studies call; the JSON rows they print."""

    def bench(self, name, *args, env=None):
        if BUILD is None or not (BUILD/name).exists():
            self.skipTest(f'{name} is not built')
        result = subprocess.run([str(BUILD/name), *args], capture_output=True, text=True, timeout=120,
                                env={**os.environ, 'LLAMA_THREADS': '1', **(env or {})})
        self.assertEqual(result.returncode, 0, result.stderr[-1000:])
        return [json.loads(line) for line in result.stdout.splitlines()]

    def check_cells(self, rows, kernel, shapes, ms):
        self.assertEqual(len(rows), len(shapes) * len(ms))
        for row in rows:
            self.assertEqual(row['kernel'], kernel)
            self.assertIn((row['n'], row['k']), shapes)
            self.assertIn(row['m'], ms)
            self.assertGreater(row['median_s'], 0)
            self.assertGreaterEqual(row['median_s'], row['best_s'])
            self.assertGreater(row['weight_bytes'], 0)
        return rows

    def test_gemm_bench_kernels_report_cells_and_cold_copies(self):
        shapes, ms = [(192, 128), (64, 256)], [1, 3, 20]
        args = ['--m', '1,3,20', '--shapes', '192x128,64x256', '--reps', '3', '--cold-mib', '1']
        f32 = self.check_cells(self.bench('gemm-bench', '--kernel', 'f32', *args), 'gemm', shapes, ms)
        self.assertEqual([r['weight_bytes'] for r in f32][:1], [192 * 128 * 4])
        self.assertGreater(f32[0]['cold_copies'], 1)  # 1 MiB of copies of a 96 KiB matrix
        q8 = self.check_cells(self.bench('gemm-bench', '--kernel', 'q8', *args), 'gemm_q8', shapes, ms)
        self.assertEqual(q8[0]['weight_bytes'], 192 * 128 + 192 * 2 * 4)  # int8 values plus one FP32 scale per 64
        self.assertTrue(all(r['quant_median_s'] >= 0 for r in q8))
        self.check_cells(self.bench('gemm-bench', '--kernel', 'bf16', *args), 'gemm_bf16', shapes, ms)
        rows = self.bench('gemm-bench', '--kernel', 'q8', '--gemv', *args)
        self.assertEqual(sorted({r['kernel'] for r in rows}), ['gemm_q8', 'gemv_rows_q8'])
        self.assertEqual(len([r for r in rows if r['kernel'] == 'gemv_rows_q8']), len(shapes) * len(ms))
        rows = self.bench('gemm-bench', '--kernel', 'f32', '--gemv', '--m', '4', '--shapes', '64x64', '--reps', '2')
        self.assertEqual(sorted(r['kernel'] for r in rows), ['gemm', 'gemv_rows'])
        self.assertEqual(rows[0]['cold_copies'], 1)  # the C2 behaviour without --cold-mib
        for bad in (['--kernel', 'fp8'], ['--kernel', 'bf16', '--gemv'], ['--kernel', 'q8', '--shapes', '64x100']):
            if BUILD is not None and (BUILD/'gemm-bench').exists():
                result = subprocess.run([str(BUILD/'gemm-bench'), *bad], capture_output=True, text=True, timeout=60)
                self.assertNotEqual(result.returncode, 0, bad)

    def test_gemm_bench_amx_kernels_on_the_software_tile_model(self):
        shapes, ms = [(96, 128), (32, 192)], [1, 7, 33]
        args = ['--m', '1,7,33', '--shapes', '96x128,32x192', '--reps', '2', '--cold-mib', '1']
        env = {'LLAMA_KERNELS': 'amx'}
        q8 = self.check_cells(self.bench('gemm-bench-emul', '--kernel', 'q8', *args, env=env), 'gemm_q8', shapes, ms)
        self.assertEqual({r['impl'] for r in q8}, {'amx_int8'})
        bf16 = self.check_cells(self.bench('gemm-bench-emul', '--kernel', 'bf16', *args, env=env), 'gemm_bf16', shapes, ms)
        self.assertEqual({r['impl'] for r in bf16}, {'amx_bf16'})

    def test_amx_bench_rows(self):
        rows = self.bench('amx-bench-emul', '--k', '128', '--reps', '2', '--m', '1,16,32', env={'LLAMA_KERNELS': 'amx'})
        self.assertEqual(sorted((r['kind'], r['m']) for r in rows),
                         [(kind, m) for kind in ('bf16', 'int8') for m in (1, 16, 32)])
        self.assertTrue(all(r['tera_ops'] >= 0 and r['best_s'] > 0 for r in rows))
        result = subprocess.run([str(BUILD/'amx-bench-emul'), '--m', '33'], capture_output=True, text=True,
                                env={**os.environ, 'LLAMA_KERNELS': 'amx'}) if BUILD is not None else None
        if result is not None:
            self.assertEqual(result.returncode, 2)



if __name__ == '__main__':
    unittest.main()
