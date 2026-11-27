"""Tests of tools/hybrid_campaign_summary.py and the DRY_RUN output of scripts/hybrid-campaign.sbatch. Synthetic fixtures only: a
results directory with the files the job script writes (runs.tsv, engine reports, perf CSV, token traces); no cluster, no engine."""
import contextlib
import io
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'tools'))
import hybrid_campaign_summary as summary  # noqa: E402
import hybrid_limits as hl  # noqa: E402

SCRIPT = ROOT/'scripts/hybrid-campaign.sbatch'
T0 = 1_000_000_000_000
HEADER = ('label\tstep\tsetting\tmeas\tkind\trep\twarm\tcontext\tsynthetic\tprompt_tokens\tgroup\tmodel\tsockets\tprofile\t'
          'start_ns\tend_ns')


def plan(mode='hbm-only', weights=32, kv=32, split=None, fraction=None, cores=None):
    p = {'admitted': True, 'simulated': False, 'mode': mode, 'classifier': 'hbm', 'weights_hbm_layers': weights, 'kv_hbm_layers': kv,
         'nodes': [{'socket': 0}, {'socket': 1}]}
    if split:
        p.update(split=split, ddr_fraction=fraction, ddr_cores=cores)
    return p


def report(tps, plan_, warm=4096, steps=255, synthetic=False, profile=None, slow=0.05):
    return {'status': 'completed', 'stop_reason': 'generation_limit',
            'model': {'quantization_group_size': 64,
                      'configuration': {'dim': 4096, 'n_layers': 32, 'hidden_dim': 14336, 'n_kv_heads': 8, 'head_dim': 128,
                                        'vocab_size': 128256, 'shared_embeddings': False}},
            'system': {'hostname': 'node'}, 'build': {'kernels': 'amx'}, 'runtime': {'kernels_prefill': 'amx-int8'},
            'parameters': dict({'warm_context': warm, 'synthetic_kv': synthetic},
                               **({'synthetic_slowest_decode_step_seconds': slow} if synthetic and slow is not None else {})),
            'tokens': {'processed_prompt': 35, 'decode_steps': steps},
            'timing_seconds': {'checkpoint_load': 0.5, 'runtime_init': 9.5, 'prefill': 0.4, 'decode': steps / tps,
                               'time_to_first_token': 0.45, 'warm_context': 3.0, 'placement_populate': 7.0},
            'throughput_tokens_per_second': {'decode': tps}, 'placement_plan': plan_, 'kv': {},
            'memory_bytes': {'peak_resident': 20 * 2 ** 30}, 'profile_seconds': {'decode': profile} if profile else None,
            'phase_epoch_ns': {'prefill_start': T0 + int(1e9), 'prefill_end': T0 + int(1.4e9), 'decode_start': T0 + int(2e9),
                               'decode_end': T0 + int(10e9)}}


def perf_csv(seconds=12.0, pause=None):
    """Per-socket counts for 100 ms bins: 100 GB/s of DDR reads and 100 GB/s of HBM reads on each socket, 1.0e6 UPI flits, 5 J."""
    lines = []
    for k in range(1, int(seconds * 10) + 1):
        t = k / 10
        quiet = pause is not None and pause[0] < t <= pause[1]
        for s in (0, 1):
            for event, value in (('ddr_cas_rd', 1.5625e8), ('hbm_cas_rd', 3.125e8), ('unc_upi_rxl_flits.all_data', 1e6),
                                 ('unc_upi_txl_flits.all_data', 1e6), ('power/energy-pkg/', 5.0)):
                if quiet and event.endswith('cas_rd'):
                    value = 1000.0
                lines.append(f'{t:.3f},S{s},1,{value},,{event},{int(t * 1e9)},100.00,,')
    return '\n'.join(lines) + '\n'


def write_run(directory, label, setting, meas, rep, tps, plan_, group, warm=4096, synthetic=False, profile=None, tokens=(1, 2, 3),
              kind='decode', prof=0, pause=None, slow=0.05):
    directory = Path(directory)
    (directory/f'{label}.json').write_text(json.dumps(report(tps, plan_, warm, 255, synthetic, profile, slow)))
    (directory/f'{label}.perf.csv').write_text(perf_csv(pause=pause))
    (directory/f'{label}.perf-start').write_text(str(T0))
    (directory/f'{label}.end').write_text(str(T0 + int(12e9)))
    (directory/f'{label}.perf-end').write_text(str(T0 + int(12e9)))
    (directory/f'{label}.tokens.json').write_text(json.dumps(list(tokens)))
    (directory/f'{label}.placement').write_text('PASS run completed\nPASS physical plan admitted\n')
    return '\t'.join(map(str, (label, 'c0-8b-2s', setting, meas, kind, rep, warm, warm + 256, int(synthetic), 0, group, '8b', 2, prof,
                               T0, T0 + int(12e9))))


def fixture(directory, rows):
    (Path(directory)/'runs.tsv').write_text(HEADER + '\n' + '\n'.join(rows) + '\n')
    (Path(directory)/'environment.txt').write_text('job=1\ncommit=abc\ninterval_ms=100\n')


def run_summary(*args):
    out = io.StringIO()
    with contextlib.redirect_stdout(out):
        code = summary.main(list(map(str, args)))
    return code, out.getvalue()


class CeilingTests(unittest.TestCase):
    def report_for(self, plan_):
        return report(40.0, plan_)

    def test_hbm_only_ceiling_is_the_library_ceiling(self):
        r = self.report_for(plan('hbm-only'))
        c = summary.decode_ceiling(r)
        m = hl.MODELS['8B']
        depth = 4096 + 35 + 128
        self.assertAlmostEqual(c['depth'], depth)
        self.assertAlmostEqual(c['peak'], hl.tok(hl.t_hbm_only(m, 64, 2, depth)), places=6)
        self.assertAlmostEqual(c['cal'], hl.tok(hl.t_hbm_only(m, 64, 2, depth, eta=hl.CAL['8B-2S'][0])), places=6)
        self.assertEqual(c['ddr_byte_fraction'], 0.0)

    def test_ddr_and_all_hbm_hybrid_ceilings(self):
        m = hl.MODELS['8B']
        depth = 4096 + 35 + 128
        c = summary.decode_ceiling(self.report_for(plan('ddr', 0, 0)))
        self.assertAlmostEqual(c['peak'], hl.tok(hl.t_ddr(m, 64, 2, depth)), places=6)
        self.assertEqual(c['ddr_byte_fraction'], 1.0)
        # the serial hybrid with every layer in HBM reads what hbm-only reads
        c = summary.decode_ceiling(self.report_for(plan('hybrid', 32, 32)))
        self.assertAlmostEqual(c['peak'], hl.tok(hl.t_hbm_only(m, 64, 2, depth)), places=6)

    def test_serial_hybrid_ddr_share_grows_as_layers_leave_hbm(self):
        shares = [summary.decode_ceiling(self.report_for(plan('hybrid', n, n)))['ddr_byte_fraction'] for n in (32, 24, 16, 8, 0)]
        self.assertEqual(shares, sorted(shares))
        self.assertAlmostEqual(shares[2], 0.5, delta=0.05)
        self.assertGreater(shares[-1], 0.9)  # the classifier stays in HBM

    def test_rows_ceiling_uses_the_p1_rates_of_the_core_split(self):
        self.assertEqual(summary.interpolate_e1(12), (442.3, 95.4))
        h, d = summary.interpolate_e1(14)
        self.assertAlmostEqual(h, (442.3 + 407.6) / 2)
        self.assertAlmostEqual(d, (95.4 + 135.5) / 2)
        c = summary.decode_ceiling(self.report_for(plan('hybrid', 0, 32, 'rows', 0.2, 14)))
        self.assertEqual(c['kind'], 'concurrent')
        self.assertGreater(c['cal'], 0)
        # more DDR rows than the DDR cores can serve make the DDR group the bottleneck
        slow = summary.decode_ceiling(self.report_for(plan('hybrid', 0, 32, 'rows', 0.6, 14)))
        self.assertLess(slow['cal'], c['cal'])
        with self.assertRaises(ValueError):
            summary.interpolate_e1(60)

    def test_ttft_ceiling(self):
        c = summary.ttft_ceiling(self.report_for(plan('hbm-only')))
        self.assertAlmostEqual(c['peak'], hl.t_amx_peak(hl.MODELS['8B'], 35, 2))
        self.assertIsNone(c['scaled'])


class SummaryTests(unittest.TestCase):
    def build(self, temp):
        rows = [write_run(temp, 'r1-c0-8b-2s-hbm-only-d4096', 'hbm-only', 'd4096', 1, 80.0, plan('hbm-only'), '8b-2S-d4096-default'),
                write_run(temp, 'r2-c0-8b-2s-hbm-only-d4096', 'hbm-only', 'd4096', 2, 82.0, plan('hbm-only'), '8b-2S-d4096-default'),
                write_run(temp, 'r1-c0-8b-2s-ddr-d4096', 'ddr', 'd4096', 1, 40.0, plan('ddr', 0, 0), '8b-2S-d4096-default'),
                write_run(temp, 'r1-c0-8b-2s-hbm-only-d4096s', 'hbm-only', 'd4096s', 1, 78.0, plan('hbm-only'),
                          '8b-2S-d4096s-default', synthetic=True),
                write_run(temp, 'r1-c0-8b-2s-hbm-only-t2048', 'hbm-only', 't2048', 1, 80.0, plan('hbm-only'), '8b-2S-t2048-default',
                          warm=0, kind='ttft'),
                write_run(temp, 'prof-c0-8b-2s-hbm-only-d4096', 'hbm-only', 'd4096', 0, 70.0, plan('hbm-only'),
                          '8b-2S-d4096-default', prof=1, profile={'forward_passes': 255, 'forward': 3.0, 'attention': 0.5, 'qkv': 0.3})]
        fixture(temp, rows)

    def test_medians_bytes_and_flags(self):
        with tempfile.TemporaryDirectory() as temp:
            self.build(temp)
            runs = summary.collect([temp], 0.05)
            self.assertEqual(len(runs), 6)
            cell = summary.cells(runs)[('c0-8b-2s', 'hbm-only', 'd4096')]
            self.assertEqual(sorted(r['tps'] for r in cell), [80.0, 82.0])
            # 100 GB/s per socket per tier over the 8 s decode window, 255 steps: 3.14 GB per step per socket per tier
            self.assertAlmostEqual(cell[0]['ddr_gbps'], 200.0, delta=1.0)
            self.assertAlmostEqual(cell[0]['hbm_gbps'], 200.0, delta=1.0)
            self.assertAlmostEqual(cell[0]['ddr_gb_per_token'], 200.0 * 8 / 255, delta=0.2)
            self.assertAlmostEqual(cell[0]['ddr_share'], 0.5, delta=0.01)
            self.assertAlmostEqual(cell[0]['pkg_j_per_token'], 2 * 50.0 * 8 / 255, delta=0.2)  # 5 J per 100 ms on two sockets
            self.assertEqual(cell[0]['placement'], (2, 0))
            self.assertAlmostEqual(cell[0]['load_s'], 10.0)
            self.assertAlmostEqual(cell[0]['rss_gib'], 20.0)
            code, text = run_summary(temp)
            self.assertEqual(code, 0, text)
            self.assertIn('81.00 [80.00-82.00]', text)
            self.assertIn('Synthetic K/V against the real fill', text)
            self.assertIn('-3.7%', text)  # synthetic 78 against the real median 81
            self.assertIn('2 PASS / 0 FAIL', text)
            self.assertIn('Time to first token', text)
            self.assertIn('| c0-8b-2s | hbm-only | t2048 | 1/1 | 35 | amx-int8 | 0.450 |', text)

    def test_pause_scan_counts_idle_seconds_and_excludes_flagged_runs_from_clean_medians(self):
        with tempfile.TemporaryDirectory() as temp:
            rows = [write_run(temp, f'r{k}-a', 'hbm-only', 'd4096', k, tps, plan('hbm-only'), 'g', pause=pause)
                    for k, tps, pause in ((1, 80.0, None), (2, 82.0, None), (3, 40.0, (4.0, 5.0)), (4, 81.0, (4.0, 4.2)))]
            fixture(temp, rows)
            runs = {r['row']['label']: r for r in summary.collect([temp], 0.05)}
            self.assertAlmostEqual(runs['r1-a']['idle_decode_s'], 0.0)
            self.assertAlmostEqual(runs['r3-a']['idle_decode_s'], 1.0, delta=0.11)
            self.assertAlmostEqual(runs['r3-a']['idle_longest_s'], 1.0, delta=0.11)
            self.assertTrue(runs['r3-a']['paused'])
            self.assertFalse(runs['r4-a']['paused'])  # 0.2 s is under the 0.3 s limit
            code, text = run_summary(temp)
            self.assertIn('| 3/4 | 80.50 [40.00-82.00] | 81.00 [80.00-82.00] |', text)
            self.assertIn('| c0-8b-2s | hbm-only | d4096 | 4 | 1 |', text)

    def test_synthetic_run_with_a_slow_step_or_no_guard_field_fails(self):
        with tempfile.TemporaryDirectory() as temp:
            fixture(temp, [write_run(temp, 'r1-s', 'hbm-only', 'd4096s', 1, 80.0, plan('hbm-only'), 'g', synthetic=True),
                           write_run(temp, 'r2-s', 'hbm-only', 'd4096s', 2, 40.0, plan('hbm-only'), 'g', synthetic=True, slow=12.0),
                           write_run(temp, 'r3-s', 'hbm-only', 'd4096s', 3, 79.0, plan('hbm-only'), 'g', synthetic=True, slow=None)])
            runs = {r['row']['label']: r for r in summary.collect([temp], 0.05)}
            self.assertEqual([runs[k]['synthetic_bad'] for k in ('r1-s', 'r2-s', 'r3-s')], [False, True, True])
            code, text = run_summary(temp)
            self.assertEqual(code, 1)
            self.assertIn('r2-s: slowest decode step 12.0 s', text)
            self.assertIn('| 1/3 |', text)  # only the guarded run is clean

    def test_flag_when_a_cell_is_more_than_30_percent_from_the_model(self):
        with tempfile.TemporaryDirectory() as temp:
            fixture(temp, [write_run(temp, 'r1-c0-8b-2s-ddr-d4096', 'ddr', 'd4096', 1, 40.0, plan('ddr', 0, 0), 'g'),
                           write_run(temp, 'r1-c0-8b-2s-hbm-only-d4096', 'hbm-only', 'd4096', 1, 10.0, plan('hbm-only'), 'g')])
            _, text = run_summary(temp)
            decode = text.split('###')[1]
            lines = {l.split('|')[2].strip(): l for l in decode.splitlines() if l.startswith('| c0-8b-2s')}
            self.assertTrue(lines['hbm-only'].rstrip().endswith('| FLAG |'), lines['hbm-only'])
            self.assertFalse(lines['ddr'].rstrip().endswith('| FLAG |'), lines['ddr'])

    def test_traces_must_agree_inside_a_group_and_across_directories(self):
        with tempfile.TemporaryDirectory() as a, tempfile.TemporaryDirectory() as b:
            fixture(a, [write_run(a, 'r1-x', 'ddr', 'd4096', 1, 40.0, plan('ddr', 0, 0), 'g')])
            fixture(b, [write_run(b, 'r1-y', 'hbm-only', 'd4096', 1, 80.0, plan('hbm-only'), 'g')])
            code, text = run_summary(a, b)
            self.assertEqual(code, 0)
            self.assertIn('g: 2 traces, 1 distinct', text)
            fixture(b, [write_run(b, 'r1-y', 'hbm-only', 'd4096', 1, 80.0, plan('hbm-only'), 'g', tokens=(1, 2, 4))])
            code, text = run_summary(a, b)
            self.assertEqual(code, 1)
            self.assertIn('2 distinct', text)

    def test_failed_and_missing_runs_are_skipped(self):
        with tempfile.TemporaryDirectory() as temp:
            row = write_run(temp, 'r1-a', 'ddr', 'd4096', 1, 40.0, plan('ddr', 0, 0), 'g')
            ghost = row.replace('r1-a', 'r1-ghost')
            data = json.loads((Path(temp)/'r1-a.json').read_text())
            data['status'] = 'error'
            (Path(temp)/'r1-a.json').write_text(json.dumps(data))
            fixture(temp, [row, ghost])
            self.assertEqual(summary.collect([temp], 0.05), [])


class ScriptTests(unittest.TestCase):
    def dry_run(self, steps, **extra):
        with tempfile.TemporaryDirectory() as temp:
            env = {k: v for k, v in os.environ.items() if not k.startswith('LLAMA_')}
            env.update(ROOT=str(ROOT), OUT=str(Path(temp)/'out'), PYTHON='python3', STEPS=steps, MODEL_8B='m8', MODEL_70B='m70',
                       DRY_RUN='1')
            env.update(extra)
            result = subprocess.run(['bash', str(SCRIPT)], capture_output=True, text=True, env=env, timeout=30)
        return result

    def test_every_step_lists_its_cells(self):
        text = self.dry_run('c0-8b-1s c0-8b-2s c1 c2 c3 c4 c6 c7').stdout
        for step in ('c0-8b-1s', 'c0-8b-2s', 'c1', 'c2', 'c3', 'c4', 'c6', 'c7'):
            self.assertIn(f'step {step}:', text)
        self.assertEqual(text.count('setting hyb-'), 10)  # five serial layer counts, 1S and 2S
        self.assertEqual(text.count('setting rows-r0.'), 6)
        self.assertIn('measurement d32768s: decode warm=32768 context=33024 generation=96 synthetic=1', text)
        self.assertIn('measurement d131072s: decode warm=130816 context=131072', text)
        self.assertIn('prompt=8192', text)

    def test_the_hbm_baseline_is_the_hybrid_with_the_packed_copy_in_ddr(self):
        text = self.dry_run('c1').stdout
        self.assertIn('setting hyb80: LLAMA_PLACEMENT=hybrid LLAMA_HYBRID_CLASSIFIER=hbm LLAMA_HYBRID_PACKED=ddr '
                      'LLAMA_HYBRID_WEIGHTS_HBM_LAYERS=80 LLAMA_HYBRID_KV_HBM_LAYERS=80', text)
        self.assertNotIn('hbm-only', text)

    def test_ab_steps_use_five_repetitions_and_the_others_three(self):
        text = self.dry_run('c1 c3 c7').stdout
        reps = {line.split()[1].rstrip(':'): line.split('reps=')[1].split()[0] for line in text.splitlines()
                if line.strip().startswith('measurement') and ('c3' in text)}
        self.assertEqual(reps['d4096'], '3')
        self.assertEqual(reps['t8192'], '5')
        self.assertEqual(self.dry_run('c1', REPEATS='1').stdout.count('reps=1'), 4)  # the four TTFT cells keep 5 repetitions

    def test_ttft_cells_have_a_cold_variant_and_the_harness_check_step_lists_its_settings(self):
        text = self.dry_run('c1 h1').stdout
        for name in ('t2048', 't8192', 't2048c', 't8192c'):
            self.assertIn(f'measurement {name}: ttft', text)
        self.assertIn('setting auto-min: LLAMA_PLACEMENT=hybrid', text)
        self.assertIn('HC_PERF=min HC_TOPO=auto', text)

    def test_one_socket_cells_are_bound_to_socket_zero(self):
        text = self.dry_run('c6').stdout
        self.assertIn('bind: numactl --cpunodebind=0', text)
        self.assertIn('LLAMA_SOCKETS=1', text)

    def test_missing_options_and_models_are_refused(self):
        self.assertNotEqual(self.dry_run('c9').returncode, 0)
        env = {k: v for k, v in os.environ.items() if not k.startswith('LLAMA_') and k not in ('ROOT', 'OUT', 'STEPS', 'PYTHON')}
        result = subprocess.run(['bash', str(SCRIPT)], capture_output=True, text=True, env=env, timeout=30)
        self.assertNotEqual(result.returncode, 0)
        self.assertIn('ROOT', result.stderr)

    def test_script_names_no_host_path_or_forbidden_word(self):
        text = SCRIPT.read_text() + (ROOT/'tools/hybrid_campaign_summary.py').read_text()
        for word in ('co' + 'dex', 'lp' + 'or', '/afs/', '/PFS', 'enea', 'cresco'):
            self.assertNotIn(word, text.lower())


if __name__ == '__main__':
    unittest.main()
