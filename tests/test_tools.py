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
            result, data = self.compare(engine, '--env-b', 'LLAMA_KERNELS=scalar')
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


if __name__ == '__main__':
    unittest.main()
