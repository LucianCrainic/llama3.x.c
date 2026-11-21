"""Tests of the cross-engine campaign harness: tools/xengine_summary.py, tools/xengine_cell.py and the DRY_RUN matrix of
scripts/xengine.sbatch. Synthetic fixtures only; no cluster, no engine, no numpy."""
import csv
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'tools'))
import xengine_cell as cell  # noqa: E402
import xengine_summary as summ  # noqa: E402

T0 = 1_800_000_000_000_000_000  # launch of the synthetic run, epoch ns
SEEN = 3.0  # s after launch: the engine logs its end-of-load line
TOKENIZE, PREFILL, TTFT, DECODE, SAMPLING = 0.05, 0.10, 0.17, 2.0, 0.02
TOTAL = TTFT + DECODE + SAMPLING
TEARDOWN = 0.5
END = T0 + int((SEEN + TOTAL + TEARDOWN)*1e9)
PERF_T0 = T0 + int(0.04e9)  # true wall clock of perf time 0 (perf starts a little after the engine)
HBM_RATE = 2.0e9  # hbm_cas_rd counts per second on socket 0, uniform over the run
DDR_RATE = 1.0e7


def report(generated=128, steps=127, status='completed', sockets=1):
    return {'status': status, 'stop_reason': 'generation_limit', 'tokens': {'generated_output': generated, 'decode_steps': steps},
            'timing_seconds': {'prompt_tokenization': TOKENIZE, 'prefill': PREFILL, 'decode': DECODE, 'total_inference': TOTAL,
                               'time_to_first_token': TTFT, 'average_decode_step': DECODE/steps if steps else None},
            'throughput_tokens_per_second': {'prefill': 86/PREFILL, 'decode': steps/DECODE},
            'prefill': {'tokens': 86, 'seconds': PREFILL}, 'runtime': {'sockets': sockets, 'placement': 'hbm'},
            'regions': [{'role': 'weights', 'socket': 0, 'node': 2, 'verified': 'full', 'pages_queried': 10,
                         'pages_mismatched': 0, 'pages_missing': 0},
                        {'role': 'kv.k', 'socket': 0, 'node': 2, 'verified': 'full', 'pages_queried': 10,
                         'pages_mismatched': 0, 'pages_missing': 100},
                        {'role': 'embedding', 'socket': 0, 'node': 0, 'verified': 'full', 'pages_queried': 10,
                         'pages_mismatched': 0, 'pages_missing': 0}]}


def perf_lines(last, step=0.2, rates=None):
    """`perf stat -a --per-socket -I 200 -x,` lines for a run of `last` seconds, a final partial interval included."""
    rates = rates or {}
    times = []
    t = step
    while t < last:
        times.append(round(t, 9))
        t += step
    times.append(last)
    lines = ['# started on Sat Nov 21 12:47:53 2026', '']
    prev = 0.0
    for t in times:
        dt = t - prev
        prev = t
        for s in (0, 1):
            for event, unit, per_s in (('ddr_cas_rd', '', DDR_RATE), ('hbm_cas_rd', '', HBM_RATE if s == 0 else 1e3),
                                       ('unc_upi_rxl_flits.all_data', '', 9e6), ('unc_upi_txl_flits.all_data', '', 9e6),
                                       ('power/energy-pkg/', 'Joules', 300.0 if s == 0 else 100.0),
                                       ('power/energy-ram/', 'Joules', 20.0 if s == 0 else 5.0)):
                per_s = rates.get(event, per_s)
                value = per_s*dt
                text = '%.2f' % value if unit else '%d' % round(value)
                lines.append('%.9f,S%d,1,%s,%s,%s,%d,100.00,,' % (t, s, text, unit, event, int(dt*1e9)))
    return '\n'.join(lines) + '\n'


def make_run(out, label='c-s1-hbm-D-r1', mem='hbm', sockets=1, rep=1, group='MEM', rep_report=None, status='ok', rates=None):
    d = Path(out)/'runs'/label
    d.mkdir(parents=True)
    (d/'cell.json').write_text(json.dumps({'engine': 'c', 'sockets': sockets, 'mem': mem, 'workload': 'D', 'repeat': rep,
                                           'perf_group': group, 'node_role': 'A', 'status': status, 'env': [], 'command': 'x'}))
    (d/'report.json').write_text(json.dumps(rep_report or report(sockets=sockets)))
    (d/'anchors').write_text('launch_ns=%d\nend_ns=%d\nrc=0\nperf_mode=sys\nperf_start_ns=%d\nload_end_seen_ns=%d\n' % (
        T0, END, T0 + 1_000_000, T0 + int(SEEN*1e9)))
    last = (END - PERF_T0)/1e9 + 0.05  # perf stops 0.05 s after the engine exited (its tail poll)
    (d/'perf.csv').write_text(perf_lines(last, rates=rates))
    assert cell.main(['markers', str(d)]) == 0
    return d


class MarkerTests(unittest.TestCase):
    def test_markers_from_the_load_end_anchor(self):
        with tempfile.TemporaryDirectory() as tmp:
            d = make_run(tmp)
            m = json.loads((d/'markers.json').read_text())
            self.assertEqual(m['load_start_ns'], T0)
            self.assertEqual(m['anchor'], 'load_end')
            seen = T0 + int(SEEN*1e9)
            self.assertAlmostEqual((m['prefill_start_ns'] - seen)/1e9, TOKENIZE, places=6)
            self.assertAlmostEqual((m['first_token_ns'] - seen)/1e9, TTFT, places=6)
            self.assertAlmostEqual((m['decode_end_ns'] - m['first_token_ns'])/1e9, DECODE + SAMPLING, places=6)
            self.assertAlmostEqual(m['teardown_s'], TEARDOWN, places=6)
            self.assertAlmostEqual((m['perf_t0_ns'] - PERF_T0)/1e9, 0.0, places=3)  # perf clock lined up at the end

    def test_fallback_anchors_on_the_end(self):
        anchors = {'launch_ns': str(T0), 'end_ns': str(END)}
        m = cell.derive_markers_c(anchors, report())
        self.assertEqual(m['anchor'], 'end')
        self.assertAlmostEqual((END - m['decode_end_ns'])/1e9, 0.0, places=6)
        self.assertAlmostEqual((m['decode_end_ns'] - m['first_token_ns'])/1e9, DECODE + SAMPLING, places=6)

    def test_failed_run_without_report_still_has_anchor_markers(self):
        m = cell.derive_markers_c({'launch_ns': str(T0), 'end_ns': str(END)}, None)
        self.assertIsNone(m['first_token_ns'])
        self.assertEqual(m['end_ns'], END)


class PerfTests(unittest.TestCase):
    def test_parses_the_three_line_formats(self):
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp)/'p.csv'
            p.write_text('0.200218186,S0,1,5676023,,ddr_cas_rd,1761783200,100.00,,\n'
                         '0.200218186,S1,1,60.53,Joules,power/energy-pkg/,215977383,100.00,,\n'
                         '0.4,100,,cycles:u,1,100.00,,\n'
                         'S0,1,300,,hbm_cas_rd,1000000000,100.00,,\n'
                         '13960272868,,EXE.AMX_BUSY:u,163357823094,100.00,,\n'
                         '0.4,<not counted>,,instructions,0,0.00,,\n')
            s = summ.parse_perf(p)
            self.assertEqual(s[(0, 'ddr')], [(0.0, 0.200218186, 5676023.0, 100.0)])
            self.assertEqual(s[(1, 'pkg_j')][0][2], 60.53)
            self.assertEqual(s[(0, 'cycles')][0][2], 100.0)
            self.assertEqual(s[(0, 'hbm')][0][:2], (0.0, 1.0))
            self.assertEqual(s[(0, 'amx_busy')][0][2], 13960272868.0)
            self.assertIsNone(s[(0, 'instructions')][0][2])

    def test_window_sum_prorates_edges(self):
        series = {(0, 'x'): [(0.0, 0.2, 100.0, 100.0), (0.2, 0.4, 100.0, 100.0), (0.4, 0.6, 100.0, 99.0)]}
        total, cover, pct = summ.window_sum(series, (0, 'x'), 0.1, 0.5)
        self.assertAlmostEqual(total, 200.0)  # half + whole + half
        self.assertAlmostEqual(cover, 0.4)
        self.assertEqual(pct, 99.0)
        total, cover, _ = summ.window_sum(series, (0, 'x'), 0.5, 0.9)
        self.assertAlmostEqual(total, 50.0)  # the file ends at 0.6
        self.assertAlmostEqual(cover, 0.1)


class SummaryTests(unittest.TestCase):
    def rows(self, out, **kw):
        return summ.summarize([out], **kw)

    def test_one_row_per_phase_with_native_and_counter_columns(self):
        with tempfile.TemporaryDirectory() as tmp:
            make_run(tmp)
            (Path(tmp)/'idle-baseline.perf.csv').write_text(perf_lines(10.0, step=1.0, rates={
                'power/energy-pkg/': 50.0, 'power/energy-ram/': 10.0}))
            rows = self.rows(tmp)
            self.assertEqual([r['phase'] for r in rows], ['prefill', 'decode'])
            pre, dec = rows
            self.assertEqual((dec['tokens'], dec['seconds']), (127, DECODE))
            self.assertAlmostEqual(dec['tok_s'], 127/DECODE)
            self.assertAlmostEqual(dec['tpot_s'], DECODE/127)
            self.assertAlmostEqual(pre['ttft_s'], TTFT)
            self.assertEqual(dec['node_role'], 'A')
            # decode window = DECODE + SAMPLING seconds of a uniform rate; edges fall inside 200 ms intervals (prorated)
            win = DECODE + SAMPLING
            self.assertAlmostEqual(dec['window_s'], win, places=5)
            self.assertAlmostEqual(dec['hbm_gb_s0'], HBM_RATE*win*summ.HBM_CAS_BYTES/1e9, delta=1e-3)
            self.assertAlmostEqual(dec['hbm_gbps_s0'], HBM_RATE*summ.HBM_CAS_BYTES/1e9, delta=1e-3)
            self.assertAlmostEqual(dec['ddr_gb_s0'], DDR_RATE*win*64/1e9, delta=1e-4)
            self.assertAlmostEqual(dec['window_cover'], 1.0, places=2)
            self.assertAlmostEqual(dec['hbm_gb_per_token'], HBM_RATE*win*summ.HBM_CAS_BYTES/1e9/127, delta=1e-4)
            self.assertAlmostEqual(dec['upi_rx_gb_s0'], 9e6*win*summ.UPI_DATA_FLIT_BYTES/1e9, delta=1e-4)
            self.assertAlmostEqual(dec['pkg_w_s0'], 300.0, delta=0.5)
            self.assertAlmostEqual(dec['dram_w_s1'], 5.0, delta=0.1)
            total_w = 300 + 100 + 20 + 5
            self.assertAlmostEqual(dec['joules_per_token'], total_w*win/127, delta=0.05)
            self.assertAlmostEqual(dec['joules_per_token_net'], (total_w - 2*(50 + 10))*win/127, delta=0.05)
            self.assertLess(abs(dec['perf_clock_skew_s'] - (-0.04 + 0.001)), 0.06)

    def test_hbm_cas_factor_is_an_option(self):
        with tempfile.TemporaryDirectory() as tmp:
            make_run(tmp)
            a = self.rows(tmp, hbm_bytes=64)[1]['hbm_gb_s0']
            b = self.rows(tmp, hbm_bytes=32)[1]['hbm_gb_s0']
            self.assertAlmostEqual(a, 2*b, places=6)

    def test_csv_columns_and_failed_runs(self):
        with tempfile.TemporaryDirectory() as tmp:
            make_run(tmp)
            bad = Path(tmp)/'runs'/'c-s2-hbm-D-r3'
            bad.mkdir()
            (bad/'cell.json').write_text(json.dumps({'engine': 'c', 'sockets': 2, 'mem': 'hbm', 'workload': 'D', 'repeat': 3,
                                                     'perf_group': 'none', 'node_role': 'A', 'status': 'failed',
                                                     'status_detail': 'exit=1'}))
            rows = self.rows(tmp)
            out = Path(tmp)/'runs.csv'
            summ.write_csv(rows, out)
            with open(out) as f:
                table = list(csv.DictReader(f))
            self.assertEqual(list(table[0]), summ.COLUMNS)
            self.assertEqual(len(table), 3)
            self.assertEqual((table[2]['phase'], table[2]['status'], table[2]['note']), ('run', 'failed', 'exit=1'))
            self.assertEqual(table[1]['phase'], 'decode')

    def test_token_count_failure_keeps_status_failed(self):
        with tempfile.TemporaryDirectory() as tmp:
            make_run(tmp, status='failed')
            self.assertEqual({r['status'] for r in self.rows(tmp)}, {'failed'})

    def test_calibrate_prints_bytes_per_token_against_nominal(self):
        with tempfile.TemporaryDirectory() as tmp:
            make_run(tmp)
            rows = self.rows(tmp)
            r = [x for x in rows if x['phase'] == 'decode'][0]
            self.assertEqual(r['mem'], 'hbm')
            import io
            from contextlib import redirect_stdout
            buf = io.StringIO()
            with redirect_stdout(buf):
                summ.calibrate(rows, 64)
            text = buf.getvalue()
            self.assertIn('nominal 7.97 GB/token', text)
            self.assertIn('HBM runs: x nominal', text)
            self.assertIn('Pick the HBM factor', text)

    def test_command_line(self):
        with tempfile.TemporaryDirectory() as tmp:
            make_run(tmp)
            out = Path(tmp)/'runs.csv'
            self.assertEqual(summ.main([tmp, '-o', str(out), '--hbm-cas-bytes', '32']), 0)
            self.assertTrue(out.exists())


def vllm_record(gen=128, ptok=86, ttft=0.2, step=0.025, wall=END/1e9):
    arrivals = [ttft + i*step for i in range(gen)]
    return {'http_status': 200, 'usage_prompt_tokens': ptok, 'usage_completion_tokens': gen, 'token_arrivals': arrivals,
            't_first_token': ttft, 't_end': arrivals[-1] + 0.001, 'wall_time': wall, 'ok': True}


class VllmTests(unittest.TestCase):
    def test_markers_from_the_client_clock(self):
        r = vllm_record()
        m = cell.derive_markers_vllm({'launch_ns': str(T0 + 5_000_000_000), 'end_ns': str(END + 400_000_000),
                                      'load_start_ns': str(T0)}, r)
        self.assertEqual(m['load_start_ns'], T0)  # server launch, not the client's
        send = r['wall_time']*1e9 - r['t_end']*1e9
        self.assertAlmostEqual((m['first_token_ns'] - send)/1e9, 0.2, places=5)
        self.assertAlmostEqual((m['decode_end_ns'] - m['first_token_ns'])/1e9, 127*0.025, places=5)
        self.assertAlmostEqual(m['teardown_s'], 0.4, places=3)

    def test_reader_checks_and_report(self):
        phases = summ.read_vllm_phases(vllm_record())
        self.assertAlmostEqual(phases['prefill']['tok_s'], 86/0.2)
        self.assertEqual(phases['decode']['tokens'], 127)
        self.assertAlmostEqual(phases['decode']['tok_s'], 40.0, places=6)
        self.assertAlmostEqual(phases['decode']['tpot_s'], 0.025)
        self.assertNotIn('decode', summ.read_vllm_phases(vllm_record(gen=1, ptok=2048)))
        self.assertTrue(all(o for o, _, _ in cell.check_vllm(vllm_record(), 128, 86)))
        self.assertFalse(all(o for o, _, _ in cell.check_vllm(vllm_record(gen=100), 128, 86)))
        self.assertFalse(all(o for o, _, _ in cell.check_vllm(vllm_record(), 128, 87)))

    def test_prefix_hits_and_report_file(self):
        text = ('# HELP x\nvllm:prefix_cache_hits_total{engine="0",model_name="m"} 0.0\n'
                'vllm:prefix_cache_hits_created{engine="0"} 1.7e9\n')
        self.assertEqual(cell.prefix_hits(text), 0.0)
        self.assertEqual(cell.prefix_hits(text.replace('} 0.0', '} 16.0')), 16.0)
        self.assertIsNone(cell.prefix_hits('nothing'))
        with tempfile.TemporaryDirectory() as tmp:
            d = Path(tmp)
            (d/'client.jsonl').write_text(json.dumps(dict(vllm_record(), warmup=True)) + '\n' + json.dumps(vllm_record()) + '\n')
            self.assertEqual(cell.main(['vllm-report', str(d)]), 0)
            self.assertEqual(json.loads((d/'report.json').read_text())['status'], 'completed')
            (d/'metrics-after.txt').write_text(text)
            self.assertEqual(cell.main(['check-prefix', str(d)]), 0)

    def test_summary_rows_for_a_vllm_run(self):
        with tempfile.TemporaryDirectory() as tmp:
            d = Path(tmp)/'runs'/'vllm-s1-hbm-patched-D-r3'
            d.mkdir(parents=True)
            (d/'cell.json').write_text(json.dumps({'engine': 'vllm', 'variant': 'patched', 'sockets': 1, 'mem': 'hbm',
                                                   'workload': 'D', 'repeat': 3, 'perf_group': 'none', 'node_role': 'A',
                                                   'status': 'ok'}))
            (d/'report.json').write_text(json.dumps(dict(vllm_record(), status='completed')))
            rows = summ.summarize([tmp])
            self.assertEqual([r['phase'] for r in rows], ['prefill', 'decode'])
            self.assertEqual(rows[0]['variant'], 'patched')
            self.assertAlmostEqual(rows[1]['tpot_uniform_s'], (127*0.025 + 0.001)/127, places=6)


def bench_entry(n_prompt=0, n_gen=128, depth=86, samples=(4.4e9, 4.3e9, 4.2e9)):
    avg = sum(samples)/len(samples)
    work = n_gen or n_prompt
    return {'n_prompt': n_prompt, 'n_gen': n_gen, 'n_depth': depth, 'n_threads': 56, 'cpu_strict': True, 'load_mode': 'none',
            'avg_ns': avg, 'avg_ts': work/avg*1e9, 'samples_ns': list(samples), 'status': 'completed'}


class LlamaCppTests(unittest.TestCase):
    def test_markers_place_the_last_repetition_before_exit(self):
        m = cell.derive_markers_llamacpp({'launch_ns': str(T0), 'end_ns': str(END), 'teardown_assumed_s': '0.25'}, bench_entry())
        self.assertAlmostEqual((END - m['decode_end_ns'])/1e9, 0.25, places=6)
        self.assertAlmostEqual((m['decode_end_ns'] - m['first_token_ns'])/1e9, 4.2, places=6)
        self.assertIsNone(m['prefill_start_ns'])
        pp = cell.derive_markers_llamacpp({'launch_ns': str(T0), 'end_ns': str(END)}, bench_entry(2048, 0, 0))
        self.assertIsNone(pp['decode_end_ns'])
        self.assertAlmostEqual((pp['first_token_ns'] - pp['prefill_start_ns'])/1e9, 4.2, places=6)
        self.assertEqual(summ.phase_windows(pp).keys(), {'prefill'})
        self.assertEqual(summ.phase_windows(m).keys(), {'decode'})

    def test_reader_report_and_checks(self):
        tg = summ.read_llamacpp_phases(bench_entry())
        self.assertEqual(list(tg), ['decode'])
        self.assertAlmostEqual(tg['decode']['tpot_s'], 1/tg['decode']['tok_s'])
        pp = summ.read_llamacpp_phases(bench_entry(2048, 0, 0))
        self.assertAlmostEqual(pp['prefill']['ttft_s'], 2048/pp['prefill']['tok_s'])
        self.assertTrue(all(o for o, _, _ in cell.check_llamacpp(bench_entry(), 0, 128, 86, 56, 3)))
        self.assertFalse(all(o for o, _, _ in cell.check_llamacpp(bench_entry(), 0, 128, 8192, 56, 3)))
        with tempfile.TemporaryDirectory() as tmp:
            d = Path(tmp)
            (d/'stdout').write_text('noise\n' + json.dumps([bench_entry()]))
            self.assertEqual(cell.main(['llamacpp-report', str(d)]), 0)
            self.assertEqual(json.loads((d/'report.json').read_text())['status'], 'completed')

    def test_matrix_and_overrides(self):
        text, a = TestCaseHelper.cells('A', ENGINES='llamacpp')
        self.assertEqual(len(a), 2*4*3)  # {1S, 2S} x {P, D, L, LP} x 3 repeats
        self.assertIn('llamacpp-s2-hbm-LP-r3', a)
        self.assertIn('--cpunodebind=0,1 --interleave=2,3', text)
        self.assertIn('-t 112', text)
        text, _ = TestCaseHelper.cells('A', ENGINES='llamacpp', LLAMACPP_2S_HBM_NUMACTL='--cpunodebind=0,1 --interleave=all',
                                       LLAMACPP_2S_HBM_THREADS='96', LLAMACPP_2S_HBM_FLAGS='-b 1024')
        self.assertIn('--interleave=all', text)
        self.assertIn('-t 96 -b 1024', text)
        text, v = TestCaseHelper.cells('A', ENGINES='llamacpp', LLAMACPP_2S_VARIANT='c2', LLAMACPP_2S_HBM_THREADS='64',
                                       LLAMACPP_2S_HBM_MASK='0xffffffff000000ffffffff')
        self.assertIn('llamacpp-s2-hbm-c2-P-r1', v)
        self.assertIn('llamacpp-s1-hbm-P-r1', v)
        self.assertIn('-C 0xffffffff000000ffffffff -t 64', text)
        text, c = TestCaseHelper.cells('cache', ENGINES='llamacpp')
        cmds = '\n'.join(ln for ln in text.splitlines() if ln.strip().startswith('cmd:'))
        self.assertNotIn('--membind', cmds)
        self.assertIn('numactl --cpunodebind=0,1 ', cmds)
        text, _ = TestCaseHelper.cells('B', ENGINES='llamacpp')
        self.assertIn('numactl --cpunodebind=0 --membind=0 ', text)


class TestCaseHelper:
    @staticmethod
    def cells(role, **env):
        with tempfile.TemporaryDirectory() as tmp:
            result = sbatch({'OUT': str(Path(tmp)/'out'), 'NODE_ROLE': role, **env})
            assert result.returncode == 0, result.stderr
            return result.stdout, [ln.split()[1] for ln in result.stdout.splitlines() if ln.startswith('CELL ')]


class CheckTests(unittest.TestCase):
    def test_check_c(self):
        ok = cell.check_c(report(), 1, 'hbm', 128, 86)
        self.assertTrue(all(o for o, _, _ in ok), ok)
        short = cell.check_c(report(generated=40), 1, 'hbm', 128, 86)  # EOS hit: not G tokens
        self.assertIn('generated tokens == G (EOS not hit)', [n for o, n, _ in short if not o])
        wrong_node = cell.check_c(report(), 1, 'ddr', 128, 86)
        self.assertTrue(any(not o and 'weights' in n for o, n, _ in wrong_node))
        self.assertFalse(all(o for o, _, _ in cell.check_c(report(sockets=1), 2, 'hbm', 128, 86)))
        self.assertTrue(all(o for o, _, _ in cell.check_c(report(), 1, 'cache', 128, 86)))  # recorded, not judged
        self.assertFalse(cell.check_c(None, 1, 'hbm', 1, 1)[0][0])

    def test_numa_maps_summary(self):
        text = ('7f0000000000 default anon=4 dirty=4 N2=4 kernelpagesize_kB=4\n'
                '7f0000100000 default file=/dev/shm/x mapped=10 N0=10 kernelpagesize_kB=4\n'
                '55d000000000 default heap anon=2 N0=1 N1=1 kernelpagesize_kB=2048\n')
        s = cell.numa_maps_summary(text)
        self.assertEqual(s['bytes_by_node'], {'0': 40960 + 2048*1024, '1': 2048*1024, '2': 16384})
        self.assertEqual(s['bytes_by_kind_and_node']['file'], {'0': 40960})
        self.assertEqual(s['largest_mappings'][0]['kind'], 'heap')

    def test_assemble_and_write_cell(self):
        with tempfile.TemporaryDirectory() as tmp:
            d = Path(tmp)/'runs'/'c-s1-hbm-D-r1'
            self.assertEqual(cell.main(['write-cell', str(d), '--field', 'engine=c', '--field', 'sockets=1', '--field',
                                        'config=a=b c=d', '--env', 'X=1', '--command', 'run it']), 0)
            entry = json.loads((d/'cell.json').read_text())
            self.assertEqual((entry['engine'], entry['sockets'], entry['config'], entry['env']), ('c', 1, 'a=b c=d', ['X=1']))
            self.assertEqual(cell.main(['assemble', tmp, '--campaign', 'node_role=A']), 0)
            manifest = json.loads((Path(tmp)/'manifest.json').read_text())
            self.assertEqual((manifest['campaign']['node_role'], len(manifest['cells'])), ('A', 1))


def sbatch(env, **extra):
    base = {'PATH': os.environ['PATH'], 'SLURM_JOB_ID': 'dry', 'ROOT': str(ROOT), 'DRY_RUN': '1'}
    return subprocess.run(['bash', str(ROOT/'scripts/xengine.sbatch')], capture_output=True, text=True, env={**base, **env, **extra})


class DryRunTests(unittest.TestCase):
    def cells(self, role, **env):
        with tempfile.TemporaryDirectory() as tmp:
            result = sbatch({'OUT': str(Path(tmp)/'out'), 'NODE_ROLE': role, **env})
            self.assertEqual(result.returncode, 0, result.stderr)
            return result.stdout, [ln.split()[1] for ln in result.stdout.splitlines() if ln.startswith('CELL ')]

    def test_matrix_per_node_role(self):
        _, a = self.cells('A')
        self.assertEqual(len(a), 2*3*3)  # {1S, 2S} x {P, D, L} x 3 repeats, HBM
        self.assertTrue(all('-hbm-' in c for c in a))
        _, b = self.cells('B')
        self.assertEqual(len(b), 2*3*3 + 3)  # DDR cells and the anchor
        self.assertEqual([c for c in b if '-hbm-' in c], ['c-s1-hbm-D-r1', 'c-s1-hbm-D-r2', 'c-s1-hbm-D-r3'])
        text, cache = self.cells('cache')
        self.assertEqual(len(cache), 2*3*3)
        self.assertIn('LLAMA_SOCKETS=2 LLAMA_PLACEMENT=ddr', text)
        self.assertNotIn('LLAMA_TOPOLOGY', text)
        self.assertIn('perf=USER', text)

    def test_repeat_groups_and_estimate(self):
        text, _ = self.cells('A', WORKLOADS='D', SOCKETS='1')
        groups = [ln.split()[2] for ln in text.splitlines() if ln.startswith('CELL ')]
        self.assertEqual(groups, ['perf=MEM', 'perf=COMPUTE', 'perf=none'])
        self.assertIn('ESTIMATE role A', text)
        self.assertIn('--per-socket -I 200', text)
        self.assertIn('numactl --cpunodebind=0', text)
        self.assertIn('--generation-length 128', text)
        self.assertNotIn('--chat-template', text)

    def test_vllm_matrix_per_role(self):
        text, a = self.cells('A', ENGINES='vllm')
        self.assertEqual(len(a), 2*3*3 + 3)  # patched 1S, 2S x P D L x 3, plus the stock-membind control (D)
        self.assertEqual([c for c in a if 'stock-membind' in c], ['vllm-s1-hbm-stock-membind-D-r%d' % i for i in (1, 2, 3)])
        self.assertIn('VLLM_HBM_NODE_OFFSET=2', text)
        self.assertIn('--membind=2', text)
        self.assertIn('--tensor-parallel-size 2', text)
        self.assertIn('0-54|56-110', text)
        self.assertIn('--no-enable-prefix-caching', text)
        self.assertNotIn('not filled in yet', text)
        text, b = self.cells('B', ENGINES='vllm')
        self.assertEqual(len(b), 2*3*3)
        self.assertNotIn('VLLM_HBM_NODE_OFFSET', text)
        text, c = self.cells('cache', ENGINES='vllm')
        self.assertEqual(len(c), 2*3*3)
        self.assertIn('-p <worker pids>', text)
        self.assertIn('--ids-file /prompts/short-86.ids.json', text)

    def test_validation_and_unimplemented_engines(self):
        with tempfile.TemporaryDirectory() as tmp:
            for env, msg in (({'NODE_ROLE': 'X'}, 'NODE_ROLE must be'), ({'NODE_ROLE': 'A', 'MEMS': 'cache'}, 'MEMS on a Flat'),
                             ({'NODE_ROLE': 'cache', 'MEMS': 'hbm'}, 'MEMS on a cache'), ({'NODE_ROLE': 'A', 'ENGINES': 'foo'}, 'ENGINES'),
                             ({'NODE_ROLE': 'A', 'WORKLOADS': 'Q'}, 'WORKLOADS'), ({'NODE_ROLE': 'A', 'REPEATS': '0'}, 'REPEATS')):
                result = sbatch({'OUT': str(Path(tmp)/'o'), **env})
                self.assertNotEqual(result.returncode, 0)
                self.assertIn(msg, result.stderr)
            result = sbatch({'OUT': str(Path(tmp)/'o2'), 'NODE_ROLE': 'A', 'ENGINES': 'llamacpp'})
            self.assertNotIn('not filled in yet', result.stdout)
            self.assertEqual(sbatch({'OUT': tmp, 'NODE_ROLE': 'A'}).returncode, 1)  # OUT exists


@unittest.skipUnless(shutil.which('python3'), 'python3 needed')
class ExecuteCellTests(unittest.TestCase):
    """Runs xe_execute_cell for real with a fake engine (no perf), overriding the GNU-only clock."""

    def test_fake_engine_cell_end_to_end(self):
        with tempfile.TemporaryDirectory() as tmp:
            engine = Path(tmp)/'fake-engine'
            rep = json.dumps(report(sockets=1))
            engine.write_text('#!/bin/sh\nout=\nwhile [ $# -gt 0 ]; do [ "$1" = --json ] && out=$2; shift; done\n'
                              'echo "[INFO] int8_q8_0: placement=hbm; seed=42" >&2\nsleep 0.3\n'
                              "cat > \"$out\" <<'EOF'\n%s\nEOF\n" % rep)
            engine.chmod(0o755)
            out = Path(tmp)/'out'
            script = f'''
set -euo pipefail
export SLURM_JOB_ID=1 ROOT={ROOT} OUT={out} NODE_ROLE=A DRY_RUN=0
source {ROOT}/scripts/lib/xengine-common.sh
xe_now_ns() {{ python3 -c 'import time;print(time.time_ns())'; }}
mkdir -p "$OUT"; XE_PERF=0; PYTHON=python3; XE_CELL_PY=$ROOT/tools/xengine_cell.py; XE_PLACEMENT_REPS=1
CELL_ENGINE=c CELL_S=1 CELL_MEM=hbm CELL_WL=D CELL_REP=1 CELL_LABEL=c-s1-hbm-D-r1 CELL_DIR=$OUT/runs/c-s1-hbm-D-r1
CELL_PERF_GROUP=MEM CELL_ENV=(A=1) CELL_VERSION=abc CELL_CONFIG=cfg CELL_GEN=128 CELL_PROC=fake-engine
CELL_LOAD_PATTERN='placement=' CELL_TODO= CELL_VARIANT= CELL_ANCHORS=() CELL_PERF_ATTACH=
CELL_CMD=({engine} --json "$CELL_DIR/report.json")
CELL_CHECK=(python3 $ROOT/tools/xengine_cell.py check-c "$CELL_DIR" --sockets 1 --mem hbm --generation 128 --prompt-tokens 86)
c_estimate_s() {{ echo 1; }}; C_ENGINE=x
xe_execute_cell
cat "$OUT/phase-results.txt"
'''
            result = subprocess.run(['bash', '-c', script], capture_output=True, text=True)
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertIn('PASS cell-c-s1-hbm-D-r1', result.stdout)
            d = out/'runs'/'c-s1-hbm-D-r1'
            anchors = cell.read_kv(d/'anchors')
            self.assertLess(int(anchors['load_end_seen_ns']) - int(anchors['launch_ns']), 3e9)
            self.assertGreater(int(anchors['end_ns']), int(anchors['load_end_seen_ns']))
            entry = json.loads((d/'cell.json').read_text())
            self.assertEqual((entry['status'], entry['perf_group'], entry['env']), ('ok', 'none', ['A=1']))
            self.assertTrue((d/'markers.json').exists())
            self.assertTrue(json.loads((d/'markers.json').read_text())['first_token_ns'])
            self.assertEqual([r['phase'] for r in summ.summarize([out])], ['prefill', 'decode'])


if __name__ == '__main__':
    unittest.main()
