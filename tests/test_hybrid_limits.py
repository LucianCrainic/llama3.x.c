"""Tests of tools/hybrid_limits.py (the limits model of docs/plans/hybrid-limits.md): the arithmetic against numbers the sources
state, the model's internal consistency, and that the generated tables of the document are the tool's current output. Stdlib only;
no cluster, no engine."""
import contextlib
import io
from pathlib import Path
import sys
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'tools'))
import hybrid_limits as hl  # noqa: E402

DOC = ROOT/'docs'/'plans'/'hybrid-limits.md'
M70, M8 = hl.MODELS['70B'], hl.MODELS['8B']


class SizeTests(unittest.TestCase):
    def test_70b_parameters_and_bytes(self):
        # 69.5 G multiply-adds per decoded token; 73.85 GB at gs64 (docs/plans/hbm-capacity-study.md section 2), 36.9 GB per socket
        self.assertAlmostEqual(M70.decode_params() / 1e9, 69.50, places=2)
        self.assertAlmostEqual(hl.weight_bytes(M70, 64, 1) / 1e9, 73.85, delta=0.01)
        self.assertAlmostEqual(hl.weight_bytes(M70, 64, 2) / 1e9, 36.92, delta=0.01)
        self.assertAlmostEqual(hl.weight_bytes(M8, 64, 1) / 1e9, 7.97, delta=0.01)

    def test_gs32_carries_more_scale_bytes(self):
        self.assertAlmostEqual(hl.bytes_per_weight(64), 1.0625)
        self.assertAlmostEqual(hl.bytes_per_weight(32), 1.125)
        self.assertAlmostEqual(hl.weight_bytes(M70, 32, 2) / hl.weight_bytes(M70, 64, 2), 1.125 / 1.0625)

    def test_kv_bytes_per_token(self):
        kib = 1024
        self.assertEqual(hl.kv_bytes_per_token(M70, 2, 'fp32'), 320 * kib)
        self.assertEqual(hl.kv_bytes_per_token(M70, 2, 'bf16'), 160 * kib)
        self.assertEqual(hl.kv_bytes_per_token(M70, 1, 'fp32'), 640 * kib)
        self.assertEqual(hl.kv_bytes_per_token(M8, 1, 'fp32'), 256 * kib)
        # INT8: 80 KiB of values plus one FP32 scale per (K or V, layer, kv head) row = 2.5 KiB
        self.assertEqual(hl.kv_bytes_per_token(M70, 2, 'int8'), 80 * kib + 2.5 * kib)


class DecodeTests(unittest.TestCase):
    def test_anchor_peak_rates(self):
        d = hl.ANCHOR_70B['depth']
        self.assertTrue(15.0 < hl.tok(hl.t_hbm_only(M70, 64, 2, d)) < 15.5)
        self.assertTrue(7.0 < hl.tok(hl.t_ddr(M70, 64, 2, d)) < 7.2)

    def test_old_model_reproduces_the_capacity_study(self):
        old = hl.Eta(hl.OLD_RATE_HBM / hl.HBM_STREAM, hl.OLD_KV_RATE_HBM / hl.HBM_STREAM,
                     hl.OLD_RATE_DDR / hl.DDR_STREAM, hl.OLD_KV_RATE_DDR / hl.DDR_STREAM)
        self.assertAlmostEqual(hl.tok(hl.t_hbm_only(M70, 64, 2, 8192, eta=old)), 9.4, delta=0.06)
        self.assertAlmostEqual(hl.tok(hl.t_ddr(M70, 64, 2, 8192, eta=old)), 5.5, delta=0.06)
        self.assertAlmostEqual(hl.tok(hl.t_hbm_only(M70, 64, 2, 32768, eta=old)), 6.8, delta=0.1)

    def test_calibration_reproduces_its_anchors(self):
        for key, a in (('70B-2S', hl.ANCHOR_70B), ('8B-1S', hl.ANCHOR_8B_1S), ('8B-2S', hl.ANCHOR_8B_2S)):
            m, eta = hl.MODELS[a['m']], hl.CAL[key][0]
            self.assertAlmostEqual(hl.tok(hl.t_hbm_only(m, a['gs'], a['S'], a['depth'], eta=eta)), a['hbm'], delta=a['hbm'] * 1e-3)
            self.assertAlmostEqual(hl.tok(hl.t_ddr(m, a['gs'], a['S'], a['depth'], eta=eta)), a['ddr'], delta=a['ddr'] * 1e-3)

    def test_concurrent_reproduces_p1_section_7(self):
        # P1 table: 54 + 2 at f = 0.036 gives 15.24 tok/s and 48 + 8 at f = 0.136 gives 15.03 (36.9 GB per socket there, 36.92 here)
        B = hl.weight_bytes(M70, 64, 2) / hl.GB
        for ch, cd, bh, bd in hl.E1:
            if (ch, cd) == (54, 2):
                self.assertAlmostEqual((bh + bd) / B, 15.24, delta=0.03)
            if (ch, cd) == (48, 8):
                self.assertAlmostEqual((bh + bd) / B, 15.03, delta=0.03)

    def test_orderings(self):
        for model, S in (('70B', 2), ('70B', 1), ('8B', 1), ('8B', 2)):
            m = hl.MODELS[model]
            for eta in (hl.PEAK, hl.CAL['%s-%dS' % (model, S)][0]):
                prev = None
                for ctx in (4096, 32768, 65536, 131072):
                    ser = hl.t_serial(m, 64, S, ctx, eta=eta)
                    conc = hl.t_concurrent(m, 64, S, ctx, eta=eta)[0]
                    ddr = hl.t_ddr(m, 64, S, ctx, eta=eta)
                    hbm = hl.t_hbm_only(m, 64, S, ctx, eta=eta)
                    self.assertLessEqual(conc, ser * (1 + 1e-9))     # the concurrent optimum includes the serial byte split
                    self.assertLessEqual(ser, ddr * (1 + 1e-9))      # HBM first never loses to all-DDR
                    if hbm is not None:
                        self.assertAlmostEqual(ser, hbm)             # everything fits: the hybrid is the HBM-only plan
                    if prev is not None:
                        self.assertGreater(ser, prev)                # a longer context is never faster
                    prev = ser

    def test_peak_concurrent_never_beats_hbm_only_when_everything_fits(self):
        for ctx in (4096, 32768, 65536):
            self.assertAlmostEqual(hl.t_concurrent(M70, 64, 2, ctx)[0], hl.t_hbm_only(M70, 64, 2, ctx))

    def test_capacity(self):
        # 70B on two sockets fits at 65,536 positions with FP32 K/V (69k edge) and spills at 131,072; one socket never holds the weights
        self.assertTrue(hl.place(M70, 64, 2, 65536).fits)
        self.assertFalse(hl.place(M70, 64, 2, 131072).fits)
        self.assertTrue(hl.place(M70, 64, 2, 131072, 'bf16').fits)
        self.assertFalse(hl.place(M70, 64, 1, 128).fits)
        self.assertTrue(hl.place(M8, 64, 1, 131072).fits)

    def test_kv_dtype_helps(self):
        t = [hl.t_serial(M70, 64, 2, 131072, d) for d in ('fp32', 'bf16', 'int8')]
        self.assertGreater(t[0], t[1])
        self.assertGreater(t[1], t[2])


class TtftTests(unittest.TestCase):
    def test_ops_and_peak(self):
        # 2 x 69.5 G operations per prompt token: 2048 tokens is about 285 TOP; about 0.65 s at 2 x 225 TOP/s
        self.assertAlmostEqual(hl.prefill_ops(M70, 2048) / 1e12, 285.9, delta=0.2)
        self.assertTrue(0.62 < hl.t_amx_peak(M70, 2048, 2) < 0.66)
        self.assertAlmostEqual(hl.prefill_ops(M8, 2048) / 1e12, 29.7, delta=0.1)

    def test_8b_budget_closes(self):
        for gs, S in hl.INSITU_8B:
            b = hl.budget_8b(gs, S)
            self.assertAlmostEqual(b['gemm'] + b['attn'] + b['other'], b['fwd'])
            self.assertGreater(b['other'], 0)

    def test_scaled_prediction_is_below_the_measured_rows(self):
        for _, name, S, gs, P, meas, _ in hl.TTFT_CELLS:
            if name == '70B':
                self.assertLess(hl.ttft_scaled_70b(gs, S, P)['total'], meas)
                self.assertGreater(meas / hl.t_amx_peak(M70, P, S), 10)

    def test_read_floor_break_even(self):
        self.assertEqual(hl.n_batches(2113, 256), 9)
        self.assertEqual(hl.n_batches(547, 256), 3)


class DocumentTests(unittest.TestCase):
    def test_generated_blocks_match_the_tool(self):
        self.assertTrue(DOC.exists())
        self.assertEqual(hl.update_doc(DOC, check=True), [])

    def test_every_table_renders(self):
        for name in hl.TABLES:
            self.assertTrue(hl.render(name).startswith('|') or name == 'anchor')

    def test_cli(self):
        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            self.assertEqual(hl.main(['sizes']), 0)
        self.assertIn('70B', out.getvalue())


if __name__ == '__main__':
    unittest.main()
