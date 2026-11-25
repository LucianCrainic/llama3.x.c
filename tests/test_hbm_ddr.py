"""Tests of the 70B HBM/DDR research tools: tools/hbm_ddr_results.py, tools/hbm_worker_samples.py, the event names of
tools/xengine_summary.py and the DRY_RUN output of scripts/hbm-ddr-research.sbatch. Synthetic fixtures only; no cluster, no
engine, no numpy or matplotlib."""
from pathlib import Path
import os
import py_compile
import subprocess
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'tools'))
import hbm_ddr_results as results  # noqa: E402
import hbm_worker_samples as samples  # noqa: E402
import xengine_summary as summ  # noqa: E402

SCRIPT = ROOT/'scripts/hbm-ddr-research.sbatch'


def logical_report(sockets=2, hbm_weight_layers=40, hbm_kv_layers=40, mode='hybrid', classifier='hbm'):
    """The Llama 3.1 70B v3 Q8_0 request of the study: 512 warm positions, 35 prompt tokens, 128 generated."""
    return {'model': {'configuration': {'dim': 8192, 'n_layers': 80, 'hidden_dim': 28672, 'n_kv_heads': 8, 'head_dim': 128,
                                        'vocab_size': 128256, 'shared_embeddings': False},
                      'quantization_group_size': 64},
            'placement_plan': {'nodes': [{'socket': s} for s in range(sockets)], 'mode': mode, 'classifier': classifier,
                               'weights_hbm_layers': hbm_weight_layers, 'kv_hbm_layers': hbm_kv_layers},
            'tokens': {'processed_prompt': 35, 'decode_steps': 127}, 'parameters': {'warm_context': 512}}


class PhaseTests(unittest.TestCase):
    def test_bounds_use_the_extreme_clock_anchors(self):
        first, last = 10_000_000_000, 20_000_000_000
        a, b = results.phase_bounds(first, last, (2_000_000_000, 3_000_000_000), 0.05)
        self.assertAlmostEqual(a, 8.05)
        self.assertAlmostEqual(b, 17.0 - 0.05)

    def test_valid_bins_need_full_containment_and_95_percent_running_time(self):
        bins = [(0.0, 0.2, 5, 100), (0.2, 0.4, 5, 94.9), (0.4, 0.6, None, 100), (0.6, 0.8, 7, 95), (0.8, 1.0, 9, 100)]
        self.assertEqual(results.valid_bins(bins, 0.0, 0.8), [(0.0, 0.2, 5), (0.6, 0.8, 7)])

    def test_joint_bins_need_all_four_controllers_and_convert_cas_to_bytes(self):
        window = [(0.0, 0.2, 1e8, 100), (0.2, 0.4, 1e8, 100)]
        series = {(0, 'hbm'): window, (1, 'hbm'): window, (0, 'ddr'): window[:1], (1, 'ddr'): window[:1]}
        (start, end, hbm, ddr), = results.joint_read_bins(series, 0, 10**9, (0, 0), 0.0)
        self.assertEqual((start, end), (0.0, 0.2))
        self.assertAlmostEqual(hbm, 2e8 * summ.HBM_CAS_BYTES / 0.2 / 1e9)  # 32-byte HBM CAS
        self.assertAlmostEqual(ddr, 2e8 * summ.DDR_CAS_BYTES / 0.2 / 1e9)  # 64-byte DDR CAS

    def test_low_read_spans_merge_adjacent_bins_and_drop_short_ones(self):
        bins = [(0.0, 1.0, 100, 100), (1.0, 2.0, 1, 1), (2.0, 3.0, 1, 1), (3.0, 4.0, 1, 1), (4.0, 5.0, 100, 1),
                (5.0, 6.0, 1, 1), (6.0, 7.0, 100, 100)]
        self.assertEqual(results.low_read_spans(bins), [[1.0, 4.0]])


class LogicalBytesTests(unittest.TestCase):
    def test_study_request_matches_the_documented_volumes(self):
        logical = results.logical_decode_bytes(logical_report())
        self.assertAlmostEqual(logical['mean_positions'], 611.0)
        self.assertAlmostEqual(logical['weight_gb'], 73.856, places=2)  # docs/hbm-ddr-research.md, phase accounting
        self.assertAlmostEqual(logical['kv_gb'], 0.400, places=3)
        self.assertAlmostEqual(logical['weight_gb'] + logical['kv_gb'], 74.257, places=2)

    def test_hbm_share_follows_the_plan(self):
        hbm_only = results.logical_decode_bytes(logical_report(hbm_weight_layers=80, hbm_kv_layers=80, mode='hbm-only'))
        ddr = results.logical_decode_bytes(logical_report(hbm_weight_layers=0, hbm_kv_layers=0, mode='ddr',
                                                          classifier='ddr'))
        self.assertAlmostEqual(hbm_only['hbm_weight_gb'], hbm_only['weight_gb'], places=2)
        self.assertAlmostEqual(hbm_only['hbm_kv_gb'], hbm_only['kv_gb'])
        self.assertEqual((ddr['hbm_weight_gb'], ddr['hbm_kv_gb']), (0.0, 0.0))


class EventNameTests(unittest.TestCase):
    def test_write_and_scheduler_events_are_named(self):
        for event, name in (('ddr_cas_wr', 'ddr_wr'), ('hbm_cas_wr', 'hbm_wr'), ('context-switches', 'context_switches'),
                            ('cpu-migrations', 'cpu_migrations')):
            self.assertEqual(summ.EVENT_NAMES[event], name)

    def test_perf_csv_with_write_events_parses_per_socket(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp)/'run.perf.csv'
            path.write_text('0.200,S0,1,10,,ddr_cas_wr,200000000,100.00,,\n0.200,S1,1,20,,hbm_cas_wr,200000000,100.00,,\n')
            series = summ.parse_perf(path)
        self.assertEqual(series[(0, 'ddr_wr')], [(0.0, 0.2, 10.0, 100.0)])
        self.assertEqual(series[(1, 'hbm_wr')], [(0.0, 0.2, 20.0, 100.0)])


class WorkerSampleTests(unittest.TestCase):
    def test_snapshot_reads_state_cpu_runtime_and_wait_channel(self):
        with tempfile.TemporaryDirectory() as temp:
            thread = Path(temp)/'4242'
            thread.mkdir()
            fields = ['S'] + ['0'] * 35 + ['17'] + ['0'] * 15  # state, then field 37 of /proc/PID/task/TID/stat is the CPU
            (thread/'stat').write_text('4242 (llama worker) ' + ' '.join(fields) + '\n')
            (thread/'schedstat').write_text('1500 2500 3\n')
            (thread/'wchan').write_text('futex_wait_queue\n')
            self.assertEqual(samples.snapshot(Path(temp)), [[4242, 17, 'S', 1500, 2500, 'futex_wait_queue']])


class ToolSyntaxTests(unittest.TestCase):
    def test_plot_tool_compiles(self):
        with tempfile.TemporaryDirectory() as temp:
            py_compile.compile(str(ROOT/'tools/plot_hbm_ddr.py'), cfile=str(Path(temp)/'plot.pyc'), doraise=True)


class ResearchScriptTests(unittest.TestCase):
    def dry_run(self, **extra):
        with tempfile.TemporaryDirectory() as temp:
            env = {k: v for k, v in os.environ.items() if not k.startswith('LLAMA_')}
            env.update(ROOT=str(ROOT), OUT=str(Path(temp)/'out'), MODEL_DIR=temp, CAMPAIGN='baseline', COMMIT='test', DRY_RUN='1')
            env.update(extra)
            result = subprocess.run(['bash', str(SCRIPT)], capture_output=True, text=True, env=env, timeout=30)
        self.assertEqual(result.returncode, 0, result.stderr)
        return dict(line.split('=', 1) for line in result.stdout.splitlines() if line.startswith('LLAMA_'))

    def test_engine_defaults_are_not_forced(self):
        knobs = self.dry_run()
        for name in ('KERNELS', 'PREFILL_BATCH', 'PREFILL_LOGITS', 'KV_LAYOUT', 'ATTN', 'ATTN_DECODE', 'THP', 'Q8_ACTIVATIONS',
                     'TOPOLOGY'):
            self.assertNotIn(f'LLAMA_{name}', knobs)
        self.assertEqual((knobs['LLAMA_SOCKETS'], knobs['LLAMA_THREADS'], knobs['LLAMA_HBM_BUDGET_GIB']), ('2', '56', '56'))
        self.assertEqual(knobs['LLAMA_PLACEMENT_STRICT'], '1')
        self.assertNotIn('LLAMA_SYNTHETIC_KV', knobs)

    def test_legacy_restores_the_earlier_configuration_and_options_win(self):
        legacy = self.dry_run(LEGACY='1')
        self.assertEqual([legacy[f'LLAMA_{k}'] for k in ('KERNELS', 'PREFILL_BATCH', 'PREFILL_LOGITS', 'KV_LAYOUT', 'ATTN',
                                                         'ATTN_DECODE', 'THP', 'Q8_ACTIVATIONS')],
                         ['avx512', '128', 'last', 'head', 'tiled', 'split', 'off', 'int8'])
        mixed = self.dry_run(LEGACY='1', KERNELS='amx', PREFILL_BATCH='256', TOPOLOGY='virtual:2x4')
        self.assertEqual((mixed['LLAMA_KERNELS'], mixed['LLAMA_PREFILL_BATCH'], mixed['LLAMA_TOPOLOGY']),
                         ('amx', '256', 'virtual:2x4'))
        self.assertEqual(mixed['LLAMA_ATTN'], 'tiled')

    def test_missing_options_are_refused(self):
        env = {k: v for k, v in os.environ.items() if not k.startswith('LLAMA_') and k not in ('ROOT', 'OUT', 'MODEL_DIR', 'CAMPAIGN')}
        result = subprocess.run(['bash', str(SCRIPT)], capture_output=True, text=True, env=env, timeout=30)
        self.assertNotEqual(result.returncode, 0)
        self.assertIn('ROOT', result.stderr)


if __name__ == '__main__':
    unittest.main()
