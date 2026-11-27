#!/usr/bin/env python3
"""Limits model for HBM/DDR hybrid inference (docs/plans/hybrid-limits.md). Pure arithmetic, stdlib only, no cluster access.

    hybrid_limits.py [NAME ...]               print the named tables (default: all) as markdown
    hybrid_limits.py --update-doc DOC         rewrite the generated blocks of DOC in place
    hybrid_limits.py --check-doc DOC          exit 1 if a generated block of DOC differs from this tool's output

A generated block of the document sits between `<!-- hybrid-limits:begin NAME -->` and `<!-- hybrid-limits:end NAME -->`.

Every constant below carries its source: a P1 measurement (docs/concurrent-tier-bandwidth.md), an engine measurement (job id) or a
derivation. Units: bytes are decimal (GB = 1e9) unless GiB is written; rates are GB/s; a "socket" is one Xeon Max 9480
(56 cores, 64 GB HBM, about 512 GB DDR5).

Model (per socket, per decoded token). Bytes read: weights W (every matmul weight once, plus the classifier) and the K/V cache of
the C positions already in the context. A tier's time is bytes / (eta * BW), BW the raw P1 rate of the cores that stream the tier
and eta the efficiency of the engine against that raw rate (eta = 1 is the "peak" model; eta from an engine measurement is the
"calibrated" model).
    t_hbm_only   = bytes / BW_hbm(56 cores)                                   (everything must fit the HBM budget)
    t_ddr        = bytes / BW_ddr(56 cores)
    t_serial     = B_hbm / BW_hbm(56) + B_ddr / BW_ddr(56)                    (layer-granular hybrid, tiers read one after the other)
    t_concurrent = min over the P1 core splits (c_h + c_d = 56) and the weight-byte split of
                   max(W_hbm / BW_hbm(c_h), W_ddr / BW_ddr(c_d)) + K_hbm / BW_hbm(56) + K_ddr / BW_ddr(56)
                   (only the GEMV is split, at the rates measured while both groups run; attention reads K/V on all cores after it)
"""
import argparse
import math
import re
import sys
from collections import OrderedDict
from pathlib import Path

GB = 1e9
GIB = 2 ** 30
KIB = 1024

# ---------------------------------------------------------------------------------------------------------------------------
# Models (architecture: tools/export.py shapes; 70B = Llama 3.1 70B Instruct, 8B = Llama 3.1 8B Instruct)
# ---------------------------------------------------------------------------------------------------------------------------


class Model:
    def __init__(self, name, layers, dim, hidden, heads, kv_heads, head_dim, vocab):
        self.name, self.layers, self.dim, self.hidden = name, layers, dim, hidden
        self.heads, self.kv_heads, self.head_dim, self.vocab = heads, kv_heads, head_dim, vocab

    def matrices(self):
        """Per-layer matmul shapes as (name, rows N, cols K); W1 and W3 are one entry of twice the rows."""
        q, kv = self.heads * self.head_dim, self.kv_heads * self.head_dim
        return OrderedDict([('wqkv', (q + 2 * kv, self.dim)), ('wo', (self.dim, q)),
                            ('w13', (2 * self.hidden, self.dim)), ('w2', (self.dim, self.hidden))])

    def layer_params(self):
        return sum(n * k for n, k in self.matrices().values())

    def classifier_params(self):
        return self.vocab * self.dim

    def decode_params(self):
        return self.layers * self.layer_params() + self.classifier_params()


MODELS = OrderedDict([('70B', Model('70B', 80, 8192, 28672, 64, 8, 128, 128256)),
                      ('8B', Model('8B', 32, 4096, 14336, 32, 8, 128, 128256))])

KV_DTYPES = OrderedDict([('fp32', 4.0), ('bf16', 2.0), ('int8', 1.0)])
# INT8 KV is not in the engine (docs/plans/hbm-capacity-study.md R6). Assumption used here: one FP32 scale per (position, kv head)
# row of head_dim values, for K and for V each.
INT8_SCALE_BYTES = 4


def bytes_per_weight(gs):
    """Q8_0 v3: 1 byte of int8 plus one FP32 scale per group: 1 + 4 / gs (1.0625 at gs64, 1.125 at gs32)."""
    return 1.0 + 4.0 / gs


def layer_weight_bytes(m, gs, sockets):
    return m.layer_params() * bytes_per_weight(gs) / sockets


def classifier_bytes(m, gs, sockets):
    return m.classifier_params() * bytes_per_weight(gs) / sockets


def weight_bytes(m, gs, sockets):
    """Weight bytes one socket reads per decoded token (norms, 5.3 MB at 70B, and the one embedding row are neglected)."""
    return m.layers * layer_weight_bytes(m, gs, sockets) + classifier_bytes(m, gs, sockets)


def kv_bytes_per_token(m, sockets, dtype):
    """Bytes of K and V one socket stores (and reads) per position, all layers."""
    rows = 2 * m.layers * (m.kv_heads // sockets)          # K and V, one row of head_dim per kv head
    elems = rows * m.head_dim
    if dtype == 'int8':
        return elems * 1.0 + rows * INT8_SCALE_BYTES
    return elems * KV_DTYPES[dtype]


# ---------------------------------------------------------------------------------------------------------------------------
# Tier rates (per socket, GB/s)
# ---------------------------------------------------------------------------------------------------------------------------

HBM_STREAM = 569.1          # measured, P1 E1 56+0, 2 MiB pages, 7 reps (docs/concurrent-tier-bandwidth.md section 3)
HBM_STREAM_STAGGERED = 578.3   # measured, P1 section 3, barrier-lockstep read with 1 MiB + 4 KiB stripe stagger
DDR_STREAM = 262.5          # measured, P1 E1 0+56, 56 cores
DDR_STREAM_BEST = 266.4     # measured, P1 section 3, 28 cores
DDR_DATASHEET = 8 * 8 * 4.8  # derived: 8 channels x 8 bytes x 4.8 GT/s (DDR5-4800) = 307.2; platform spec recalled, not in the repository

# P1 E1: (HBM cores, DDR cores, HBM GB/s, DDR GB/s), measured under concurrency, 2 MiB pages, median of 7 (section 4)
E1 = [(56, 0, 569.1, 0.0), (54, 2, 542.1, 20.3), (52, 4, 518.7, 37.0), (48, 8, 479.3, 75.5), (44, 12, 442.3, 95.4),
      (40, 16, 407.6, 135.5), (36, 20, 363.5, 156.9), (32, 24, 301.1, 180.4), (28, 28, 283.3, 219.7),
      (24, 32, 210.7, 224.3), (16, 40, 123.7, 251.4), (0, 56, 0.0, 262.5)]
# The combined column of the same table (median of the per-repetition sums, which is not the sum of the two medians), by HBM cores.
E1_COMBINED = {54: 562.4, 52: 555.9, 48: 554.8, 44: 537.1, 40: 542.2, 36: 521.2, 32: 483.4, 28: 501.1, 24: 434.7, 16: 375.1}

HBM_BUDGET = 56 * GIB       # LLAMA_HBM_BUDGET_GIB default per socket (docs/plans/hbm-capacity-study.md section 2)
WORK_BYTES = 0.5 * GIB      # work buffers per socket, 70B, 256-row batch: estimated in the same study (used for 8B too)

# Old calibrated model (docs/plans/hbm-capacity-study.md section 5): weight stream HBM 397 GB/s (8B W8A8 2S, 99.1 tok/s x 7.97 GB / 2),
# DDR 238 GB/s (8B 1S DDR, 29.8 tok/s x 7.97 GB); K/V read HBM 200 GB/s (measured 186-225 on 8B 2S at 8192 and 32768 positions),
# DDR 95 = 200 x 262 / 552 (assumed).
OLD_RATE_HBM = 397.0
OLD_RATE_DDR = 238.0
OLD_KV_RATE_HBM = 200.0
OLD_KV_RATE_DDR = 95.0

# Deep-context K/V read rates of the current calibration, derived from older-tree 70B 2S runs with the grouped decode kernel:
#   HBM: job 5904763 (docs/hbm-ddr-research.md), 73,728 real positions, attention 84.84 ms per step; K/V per socket = 73728 x 320 KiB.
#   DDR: job 5904765, DDR-only at 130,944 primed positions, 2.95 tok/s: the step time less the weight stream (the DDR weight rate of the
#        short-context anchor) is the K/V read time of 131072 x 320 KiB.
DEEP_HBM_ATTN_S = 0.08484
DEEP_HBM_CTX = 73728
DEEP_DDR_TOKS = 2.95
DEEP_DDR_CTX = 131072


class Eta:
    """Efficiency of the engine against the raw rate: weights (w) and K/V reads (a), per tier."""

    def __init__(self, hw=1.0, ha=1.0, dw=1.0, da=1.0, label='peak'):
        self.hw, self.ha, self.dw, self.da, self.label = hw, ha, dw, da, label


PEAK = Eta()


# ---------------------------------------------------------------------------------------------------------------------------
# Placement (layer-granular, HBM first) and decode times
# ---------------------------------------------------------------------------------------------------------------------------

class Placement:
    def __init__(self, n_hbm, layers, Wh, Kh, Wd, Kd):
        self.n_hbm, self.layers, self.Wh, self.Kh, self.Wd, self.Kd = n_hbm, layers, Wh, Kh, Wd, Kd

    @property
    def fits(self):
        return self.n_hbm == self.layers

    @property
    def ddr_fraction(self):
        return (self.Wd + self.Kd) / (self.Wh + self.Kh + self.Wd + self.Kd)


def place(m, gs, sockets, ctx, dtype='fp32', budget=HBM_BUDGET, work=WORK_BYTES):
    """Whole layers (weights and K/V of the layer together) go to HBM until the budget is full; the classifier stays in HBM; the rest
    goes to DDR. Layer granularity is an approximation of the planner's two independent knobs (LLAMA_HYBRID_WEIGHTS_HBM_LAYERS and
    LLAMA_HYBRID_KV_HBM_LAYERS): it differs from them by less than one layer of bytes."""
    lw = layer_weight_bytes(m, gs, sockets)
    cw = classifier_bytes(m, gs, sockets)
    lk = kv_bytes_per_token(m, sockets, dtype) / m.layers * ctx
    avail = budget - work - cw
    n = max(0, min(m.layers, int(avail // (lw + lk))))
    return Placement(n, m.layers, n * lw + cw, n * lk, (m.layers - n) * lw, (m.layers - n) * lk)


def all_bytes(m, gs, sockets, ctx, dtype='fp32'):
    return weight_bytes(m, gs, sockets), kv_bytes_per_token(m, sockets, dtype) * ctx


def t_tier(W, K, bw, eta_w, eta_a):
    return W / (eta_w * bw * GB) + K / (eta_a * bw * GB)


def t_hbm_only(m, gs, S, ctx, dtype='fp32', eta=PEAK, bw=HBM_STREAM):
    p = place(m, gs, S, ctx, dtype)
    if not p.fits:
        return None
    return t_tier(p.Wh, p.Kh, bw, eta.hw, eta.ha)


def t_ddr(m, gs, S, ctx, dtype='fp32', eta=PEAK, bw=DDR_STREAM):
    W, K = all_bytes(m, gs, S, ctx, dtype)
    return t_tier(W, K, bw, eta.dw, eta.da)


def t_serial(m, gs, S, ctx, dtype='fp32', eta=PEAK, place_fn=None):
    p = (place_fn or place)(m, gs, S, ctx, dtype)
    return t_tier(p.Wh, p.Kh, HBM_STREAM, eta.hw, eta.ha) + t_tier(p.Wd, p.Kd, DDR_STREAM, eta.dw, eta.da)


def t_concurrent_split(Wh, Wd, Kh, Kd, bh, bd, eta):
    """Seconds per token of the engine's concurrent row split (LLAMA_HYBRID_SPLIT=rows) at one core split. Only the GEMV is split: the
    HBM group streams Wh at its rate bh while the DDR group streams Wd at bd (GB/s, P1 E1, measured while both run); the two finish
    together, so the weight time is the slower of the two. Attention is not concurrent with it and runs on every core of the socket:
    K/V is read after the GEMV, at the full-socket rate of the tier it sits in (HBM_STREAM, DDR_STREAM). (Until 27 November 2026 the
    K/V bytes were charged inside the group rates, as if the 42 or so HBM-group cores read the HBM-resident K/V; the engine does not,
    and the model ran 15 to 25% below the measured row-split cells of docs/hybrid-memory-results.md.) Bytes in GB / 1e9 as elsewhere."""
    t_h = Wh / (eta.hw * bh * GB) if Wh else 0.0
    t_d = Wd / (eta.dw * bd * GB) if Wd else 0.0
    t_k = Kh / (eta.ha * HBM_STREAM * GB) + Kd / (eta.da * DDR_STREAM * GB)
    return max(t_h, t_d) + t_k


def t_concurrent(m, gs, S, ctx, dtype='fp32', eta=PEAK, min_ddr_weight_fraction=None, splits=E1):
    """Best core split and weight-byte split of the row-split decode (t_concurrent_split). The K/V bytes stay where the placement put
    them; the weight bytes may move between tiers (row split) but at least the placement's DDR weight bytes stay in DDR. Returns
    (seconds, (c_h, c_d), DDR weight fraction)."""
    p = place(m, gs, S, ctx, dtype)
    W = p.Wh + p.Wd
    wd_min = p.Wd if min_ddr_weight_fraction is None else min_ddr_weight_fraction * W
    best = None
    for ch, cd, bh, bd in splits:
        if bd == 0 and wd_min > 0:
            continue                                    # the DDR group does not exist but DDR weight bytes do
        ah = 1.0 / (eta.hw * bh * GB) if bh else math.inf
        ad = 1.0 / (eta.dw * bd * GB) if bd else math.inf
        if not bh:
            wd = W                                      # no HBM group: every weight byte is read by the DDR group
        elif not bd:
            wd = 0.0
        else:
            wd = ah * W / (ah + ad)                     # balance ah (W - wd) against ad wd
        wd = min(W, max(wd_min, wd, 0.0))
        t = t_concurrent_split(W - wd, wd, p.Kh, p.Kd, bh, bd, eta)
        if best is None or t < best[0]:
            best = (t, (ch, cd), wd / W)
    return best


def tok(t):
    return None if t is None else 1.0 / t


# ---------------------------------------------------------------------------------------------------------------------------
# Calibration to the engine (anchor: measured tok/s at a known depth)
# ---------------------------------------------------------------------------------------------------------------------------

def calibrate(m, gs, S, depth, hbm_toks, ddr_toks, label, kv_h, kv_d):
    """Weight-stream rates (GB/s) that reproduce the anchors when the K/V is read at kv_h / kv_d GB/s:
    1 / tok/s = W / R_w + KV(depth) / R_a, solved for R_w; then eta = R_w / raw stream rate."""
    W, K = all_bytes(m, gs, S, depth)
    rh = W / GB / (1.0 / hbm_toks - K / GB / kv_h)
    rd = W / GB / (1.0 / ddr_toks - K / GB / kv_d)
    return Eta(rh / HBM_STREAM, kv_h / HBM_STREAM, rd / DDR_STREAM, kv_d / DDR_STREAM, label), rh, rd


# Anchors. depth = prompt + 64 (mean depth over 127 decode steps) for the xengine rows; 86 + 64 = 150 for workload D.
ANCHOR_70B = dict(m='70B', gs=64, S=2, depth=547 + 64, hbm=11.583, ddr=6.216)   # jobs 5935580 (HBM), 5929407 (DDR), docs/xengine-70b.md
ANCHOR_8B_1S = dict(m='8B', gs=64, S=1, depth=150, hbm=51.07, ddr=29.8)          # jobs 5913494 (gs64 VNNI, docs/gs32-amx.md section 6), 5892202
ANCHOR_8B_2S = dict(m='8B', gs=64, S=2, depth=150, hbm=97.23, ddr=55.9)          # same jobs


def deep_kv_rates():
    """(HBM, DDR) GB/s at which the engine read K/V in the deep-context 70B 2S runs (see DEEP_*), fixed point with the DDR weight rate."""
    m = MODELS['70B']
    kv_h = kv_bytes_per_token(m, 2, 'fp32') * DEEP_HBM_CTX / GB / DEEP_HBM_ATTN_S
    kv_d = 240.0
    for _ in range(20):
        rd = calibrate(m, ANCHOR_70B['gs'], 2, ANCHOR_70B['depth'], ANCHOR_70B['hbm'], ANCHOR_70B['ddr'], 'x', kv_h, kv_d)[2]
        kv_bytes = kv_bytes_per_token(m, 2, 'fp32') * DEEP_DDR_CTX / GB
        kv_d = kv_bytes / (1.0 / DEEP_DDR_TOKS - weight_bytes(m, 64, 2) / GB / rd)
    return kv_h, kv_d


KV_RATE_HBM, KV_RATE_DDR = deep_kv_rates()


def calibrations():
    out = OrderedDict()
    for key, a in (('70B-2S', ANCHOR_70B), ('8B-1S', ANCHOR_8B_1S), ('8B-2S', ANCHOR_8B_2S)):
        eta, rh, rd = calibrate(MODELS[a['m']], a['gs'], a['S'], a['depth'], a['hbm'], a['ddr'], key, KV_RATE_HBM, KV_RATE_DDR)
        out[key] = (eta, rh, rd)
    out['70B-1S'] = (out['70B-2S'][0], out['70B-2S'][1], out['70B-2S'][2])    # estimated: the 2S rates reused, no 1S measurement
    return out


CAL = calibrations()


def cal_key(m, S):
    return '%s-%dS' % (m.name, S)


# ---------------------------------------------------------------------------------------------------------------------------
# TTFT
# ---------------------------------------------------------------------------------------------------------------------------

AMX_PEAK_TOPS = 225.0       # measured, docs/hbm-roofline.md section 1 (AMX INT8, 56 cores, one socket), via the reference-numbers page
ONEDNN_M256_TOPS = 69.0     # measured, same source: oneDNN INT8 prefill shape, M = 256, local HBM, one socket
BATCH_XENGINE = 256         # prefill batch of the xengine rows (docs/xengine-70b.md: "a prefill batch limit of 256")


def gemm_ops(m, P):
    """Operations (2 per multiply-add) of the layer matmuls for P prompt tokens; the classifier runs on the last row only."""
    return 2.0 * P * m.layers * m.layer_params()


def attention_ops(m, P):
    """Causal attention from an empty cache: QK^T and PV, each P^2/2 position pairs x heads x head_dim multiply-adds."""
    return 2.0 * 2.0 * (P * P / 2.0) * m.heads * m.head_dim * m.layers


def prefill_ops(m, P):
    return gemm_ops(m, P) + attention_ops(m, P) + 2.0 * m.classifier_params()


def t_amx_peak(m, P, S, tops=AMX_PEAK_TOPS):
    return prefill_ops(m, P) / (S * tops * 1e12)


def n_batches(P, M):
    return -(-P // M)


def t_read_floor(m, gs, S, P, M, bw):
    """Every prefill batch streams the whole packed copy (about the size of the weights) once per socket."""
    return n_batches(P, M) * weight_bytes(m, gs, S) / (bw * GB)


# In-situ AMX GEMM rates of the 8B (TOP/s over the whole node for the matrix, one profiled P=2048 run, job 5913457, docs/gs32-amx.md
# section 4), attention seconds and forward seconds of the same run.
INSITU_8B = {(64, 1): dict(wqkv=26.9, wo=27.4, w13=36.7, w2=29.8, attn=0.164, fwd=1.136),
             (64, 2): dict(wqkv=53.4, wo=31.9, w13=75.5, w2=36.2, attn=0.080, fwd=0.703),
             (32, 1): dict(wqkv=17.9, wo=16.9, w13=22.3, w2=17.3, attn=0.162, fwd=1.719),
             (32, 2): dict(wqkv=38.8, wo=28.6, w13=44.1, w2=32.3, attn=0.087, fwd=0.912)}


def gemm_time_insitu(m, P, rates):
    return sum(2.0 * P * m.layers * n * k / (rates[name] * 1e12) for name, (n, k) in m.matrices().items())


def budget_8b(gs, S, P=2048):
    """Where the 8B forward time went at P=2048: GEMM at the in-situ rates, attention, and the rest (forward - GEMM - attention)."""
    r = INSITU_8B[(gs, S)]
    m = MODELS['8B']
    g = gemm_time_insitu(m, P, r)
    return dict(gemm=g, attn=r['attn'], other=r['fwd'] - g - r['attn'], fwd=r['fwd'], ops=gemm_ops(m, P),
                rate=gemm_ops(m, P) / g / 1e12)


def ttft_scaled_70b(gs, S, P):
    """70B TTFT predicted from the 8B in-situ budget: GEMM per matrix at the 8B in-situ rate, attention and the rest x 5.0
    (heads x head_dim x layers: 655360 / 131072), GEMM and the rest linear in P, attention quadratic in P."""
    b = budget_8b(gs, S)
    m8, m70 = MODELS['8B'], MODELS['70B']
    r = INSITU_8B[(gs, S)]
    g = gemm_time_insitu(m70, P, r)
    ratio = (m70.heads * m70.head_dim * m70.layers) / (m8.heads * m8.head_dim * m8.layers)
    attn = b['attn'] * ratio * (P / 2048.0) ** 2
    other = b['other'] * (m70.dim * m70.layers) / (m8.dim * m8.layers) * (P / 2048.0)
    return dict(gemm=g, attn=attn, other=other, total=g + attn + other)


# ---------------------------------------------------------------------------------------------------------------------------
# Measured cells (decode and TTFT). label, model, sockets, gs, depth, tier, tok/s, source
# ---------------------------------------------------------------------------------------------------------------------------

D70 = lambda P: P + 64
DECODE_CELLS = [
    ('70B hybrid 80/80, gs64, 547 prompt (anchor)', '70B', 2, 64, D70(547), 'hbm', 11.583, 'job 5935580'),
    ('70B hybrid 80/80, gs64, 2113 prompt', '70B', 2, 64, D70(2113), 'hbm', 10.709, 'job 5935580'),
    ('70B hybrid 80/80, gs32, 547 prompt', '70B', 2, 32, D70(547), 'hbm', 10.966, 'job 5935580'),
    ('70B hybrid 80/80, gs32, 2113 prompt', '70B', 2, 32, D70(2113), 'hbm', 10.120, 'job 5935580'),
    ('70B DDR, gs64, 547 prompt (anchor)', '70B', 2, 64, D70(547), 'ddr', 6.216, 'job 5929407'),
    ('70B DDR, gs64, 2113 prompt', '70B', 2, 64, D70(2113), 'ddr', 6.011, 'job 5929407'),
    ('70B DDR, gs32, 547 prompt', '70B', 2, 32, D70(547), 'ddr', 5.928, 'job 5929406'),
    ('70B DDR, gs32, 2113 prompt', '70B', 2, 32, D70(2113), 'ddr', 5.755, 'job 5929406'),
    ('70B hybrid, gs64, warm 4096, stagger off (56 threads)', '70B', 2, 64, 4096 + 64, 'hbm', 10.00, 'job 5934864'),
    ('70B hybrid, gs64, warm 4096, stagger on', '70B', 2, 64, 4096 + 64, 'hbm', 10.60, 'job 5934864'),
    ('70B hybrid, gs64, warm 4096, 4 KiB pages', '70B', 2, 64, 4096 + 64, 'hbm', 10.49, 'job 5934864'),
    ('70B weight-first, 8192 real, grouped (older tree)', '70B', 2, 64, 8192, 'hybrid', 9.75, 'job 5904761'),
    ('70B weight-first, 73,728 real, grouped (older tree)', '70B', 2, 64, 73728, 'hybrid', 5.720, 'job 5904763'),
    ('70B weight-first, 130,944 primed, grouped (older tree)', '70B', 2, 64, 131072, 'hybrid', 4.13, 'job 5904765'),
    ('70B DDR only, 130,944 primed, grouped (older tree)', '70B', 2, 64, 131072, 'ddr', 2.95, 'job 5904765'),
    ('8B 1S, gs64, D (86 + 128)', '8B', 1, 64, 150, 'hbm', 51.07, 'job 5913494'),
    ('8B 2S, gs64, D', '8B', 2, 64, 150, 'hbm', 97.23, 'job 5913494'),
    ('8B 1S, gs32, D', '8B', 1, 32, 150, 'hbm', 48.04, 'job 5913494'),
    ('8B 2S, gs32, D', '8B', 2, 32, 150, 'hbm', 91.16, 'job 5913494'),
    ('8B 1S DDR, gs64, D (older tree)', '8B', 1, 64, 150, 'ddr', 29.8, 'job 5892202'),
    ('8B 2S DDR, gs64, D (older tree)', '8B', 2, 64, 150, 'ddr', 55.9, 'job 5892202'),
    ('8B 1S, gs64, warm 4096, stagger off', '8B', 1, 64, 4096 + 64, 'hbm', 42.21, 'job 5934862'),
    ('8B 1S, gs64, warm 4096, stagger on', '8B', 1, 64, 4096 + 64, 'hbm', 44.31, 'job 5934862'),
    ('8B 2S, gs64, warm 4096, stagger off', '8B', 2, 64, 4096 + 64, 'hbm', 73.53, 'job 5934863'),
    ('8B 2S, gs64, warm 4096, stagger on', '8B', 2, 64, 4096 + 64, 'hbm', 74.95, 'job 5934863'),
    # P4 baseline campaign (docs/hybrid-memory-results.md), medians of 3 or 5 interleaved runs, depth = warm + prompt + mean decode steps
    ('8B 1S hbm-only, warm 4096 (P4)', '8B', 1, 64, 4096 + 35 + 128, 'hbm', 44.13, 'job 5950120'),
    ('8B 1S hbm-only, warm 32768 (P4)', '8B', 1, 64, 32768 + 35 + 128, 'hbm', 21.92, 'job 5950120'),
    ('8B 1S DDR, warm 4096 (P4)', '8B', 1, 64, 4096 + 35 + 128, 'ddr', 25.04, 'job 5950120'),
    ('8B 1S DDR, warm 32768 (P4)', '8B', 1, 64, 32768 + 35 + 128, 'ddr', 14.01, 'job 5950120'),
    ('8B 2S hbm-only, warm 4096 (P4)', '8B', 2, 64, 4096 + 35 + 128, 'hbm', 74.75, 'job 5950120'),
    ('8B 2S hbm-only, warm 32768 (P4)', '8B', 2, 64, 32768 + 35 + 128, 'hbm', 39.85, 'job 5950120'),
    ('8B 2S DDR, warm 32768 (P4)', '8B', 2, 64, 32768 + 35 + 128, 'ddr', 25.54, 'job 5950120'),
    ('70B hybrid 80/80 packed in DDR, warm 4096 (P4)', '70B', 2, 64, 4096 + 35 + 48, 'hbm', 10.51, 'job 5950121'),
    ('70B hybrid 80/80 packed in DDR, warm 32768 (P4)', '70B', 2, 64, 32768 + 35 + 48, 'hbm', 7.20, 'job 5950121'),
    ('70B DDR, warm 4096 (P4)', '70B', 2, 64, 4096 + 35 + 48, 'ddr', 5.89, 'job 5950121'),
    ('70B DDR, warm 32768 (P4)', '70B', 2, 64, 32768 + 35 + 48, 'ddr', 4.65, 'job 5950121'),
    ('70B DDR, warm 130816 synthetic (P4)', '70B', 2, 64, 130816 + 35 + 48, 'ddr', 2.74, 'job 5962577'),
    ('70B serial hybrid (planner), warm 65536 synthetic (P4)', '70B', 2, 64, 65536 + 35 + 48, 'hybrid', 5.29, 'job 5962577'),
    ('70B serial hybrid (weights 80, planner K/V), warm 98304 synthetic (P4)', '70B', 2, 64, 98304 + 35 + 48, 'hybrid', 4.19, 'job 5962577'),
    ('70B serial hybrid (weights 80, planner K/V), warm 130816 synthetic (P4)', '70B', 2, 64, 130816 + 35 + 48, 'hybrid', 3.42, 'job 5962577'),
]

# P1 E5: the engine's decode GEMV over the 70B per-socket shard (weights only, 40 layers per tier copy, 18.2 GB gs64), GB/s
E5_CELLS = [('E5 GEMV, 56 workers, 4 KiB pages, run 1 / run 2 (all-HBM)', 447.2, 459.0, 'jobs 5923181, 5924010'),
            ('E5 GEMV, 56 workers, 2 MiB pages, aliased', 411.1, None, 'job 5924010'),
            ('E5 GEMV, 56 workers, 2 MiB, 4 KiB stagger', 447.5, None, 'job 5924010'),
            ('E5 GEMV best row split, node A run 2 (r = 0.20, 14 DDR workers)', 469.8, None, 'job 5924010'),
            ('E5 GEMV best row split, node B (r = 0.20, 14 DDR workers)', 470.9, None, 'job 5924011')]

TTFT_CELLS = [
    ('70B hybrid 80/80 (packed in DDR), gs64', '70B', 2, 64, 547, 2.124, 'job 5935580'),
    ('70B hybrid 80/80 (packed in DDR), gs64', '70B', 2, 64, 2113, 8.009, 'job 5935580'),
    ('70B hybrid 80/80 (packed in DDR), gs32', '70B', 2, 32, 547, 2.598, 'job 5935580'),
    ('70B hybrid 80/80 (packed in DDR), gs32', '70B', 2, 32, 2113, 9.917, 'job 5935580'),
    ('70B DDR, gs64', '70B', 2, 64, 547, 2.154, 'job 5929407'),
    ('70B DDR, gs64', '70B', 2, 64, 2113, 8.065, 'job 5929407'),
    ('70B DDR, gs32', '70B', 2, 32, 547, 2.689, 'job 5929406'),
    ('70B DDR, gs32', '70B', 2, 32, 2113, 10.222, 'job 5929406'),
    ('8B HBM, gs64 (P 2048)', '8B', 1, 64, 2048, 1.170, 'job 5910223'),
    ('8B HBM, gs64 (P 2048)', '8B', 2, 64, 2048, 0.690, 'job 5910223'),
    ('8B HBM, gs64 (L 8192)', '8B', 1, 64, 8192, 6.079, 'job 5910223'),
    ('8B HBM, gs64 (L 8192)', '8B', 2, 64, 8192, 3.494, 'job 5910223'),
    ('8B HBM, gs32 (P 2048)', '8B', 1, 32, 2048, 1.763, 'job 5913457'),
    ('8B HBM, gs32 (P 2048)', '8B', 2, 32, 2048, 0.988, 'job 5913457'),
]


# ---------------------------------------------------------------------------------------------------------------------------
# Tables (markdown). Each function returns a list of lines.
# ---------------------------------------------------------------------------------------------------------------------------

def md(header, rows, align=None):
    align = align or ['---'] + ['---:'] * (len(header) - 1)
    out = ['| ' + ' | '.join(header) + ' |', '| ' + ' | '.join(align) + ' |']
    out += ['| ' + ' | '.join(str(c) for c in r) + ' |' for r in rows]
    return out


def f2(x, n=2):
    return '-' if x is None else ('%.*f' % (n, x))


def kib(x):
    return '%.1f' % (x / KIB)


def table_sizes():
    rows = []
    for name, m in MODELS.items():
        rows.append([name, '%d x %d x %d' % (m.layers, m.kv_heads, m.head_dim), '%.3f M' % (m.layer_params() / 1e6),
                     '%.4f G' % (m.layers * m.layer_params() / 1e9), '%.4f G' % (m.classifier_params() / 1e9),
                     '%.4f G' % (m.decode_params() / 1e9)])
    out = md(['model', 'layers x KV heads x head_dim', 'matmul params per layer', 'all layers', 'classifier', 'total read per token'], rows)
    out.append('')
    rows = []
    for name, m in MODELS.items():
        for gs in (64, 32):
            for S in (1, 2):
                rows.append([name, 'gs%d' % gs, '%.4f' % bytes_per_weight(gs), S, f2(weight_bytes(m, gs, S) / GB, 2),
                             f2(weight_bytes(m, gs, S) / GIB, 2)])
    out += md(['model', 'group', 'bytes per weight', 'sockets', 'GB per socket per token', 'GiB per socket per token'], rows)
    out.append('')
    rows = []
    for name, m in MODELS.items():
        for S in (1, 2):
            f, b, i = (kv_bytes_per_token(m, S, d) for d in ('fp32', 'bf16', 'int8'))
            scales = m.layers * 2 * (m.kv_heads // S) * INT8_SCALE_BYTES
            rows.append([name, S, kib(f), kib(b), kib(i), kib(scales)])
    out += md(['model', 'sockets', 'FP32 KiB per token per socket', 'BF16', 'INT8 incl. scales', 'of which INT8 scales'], rows)
    return out


def table_old_model():
    """The old calibrated model (docs/plans/hbm-capacity-study.md section 5) reproduced, 70B 2S, FP32 KV, gs64."""
    m = MODELS['70B']
    old = Eta(OLD_RATE_HBM / HBM_STREAM, OLD_KV_RATE_HBM / HBM_STREAM, OLD_RATE_DDR / DDR_STREAM, OLD_KV_RATE_DDR / DDR_STREAM, 'old')
    rows = []
    for ctx in (611, 2177, 4160, 8192, 32768, 65536):
        th, td = t_hbm_only(m, 64, 2, ctx, eta=old), t_ddr(m, 64, 2, ctx, eta=old)
        nh, nd = tok(t_hbm_only(m, 64, 2, ctx, eta=CAL['70B-2S'][0])), tok(t_ddr(m, 64, 2, ctx, eta=CAL['70B-2S'][0]))
        rows.append([ctx, f2(tok(th)), f2(tok(td)), f2(nh), f2(nd)])
    return md(['context (positions)', 'old model HBM tok/s', 'old model DDR tok/s', 'new calibration HBM tok/s', 'new calibration DDR tok/s'], rows)


def table_old_vs_new():
    m = MODELS['70B']
    old = Eta(OLD_RATE_HBM / HBM_STREAM, OLD_KV_RATE_HBM / HBM_STREAM, OLD_RATE_DDR / DDR_STREAM, OLD_KV_RATE_DDR / DDR_STREAM, 'old')
    pts = [('anchor, 547 prompt', 611, 11.583, 6.216, 'jobs 5935580, 5929407'),
           ('2113 prompt', 2177, 10.709, 6.011, 'jobs 5935580, 5929407'),
           ('warm 4096 (56 threads, stagger off)', 4160, 10.00, None, 'job 5934864'),
           ('warm 4096 (stagger on)', 4160, 10.60, None, 'job 5934864'),
           ('8192 real, grouped (older tree)', 8192, 9.75, None, 'job 5904761')]
    rows = []
    for label, d, h, dd, src in pts:
        ph, pd = tok(t_hbm_only(m, 64, 2, d, eta=old)), tok(t_ddr(m, 64, 2, d, eta=old))
        rows.append([label, f2(ph), f2(h), '%.3f' % (h / ph), f2(pd), f2(dd), '-' if dd is None else '%.3f' % (dd / pd), src])
    return md(['point', 'old model HBM', 'measured HBM', 'measured / model', 'old model DDR', 'measured DDR', 'measured / model', 'source'], rows)


def table_anchor():
    m = MODELS['70B']
    eta, rh, rd = CAL['70B-2S']
    d = ANCHOR_70B['depth']
    W, K = all_bytes(m, 64, 2, d)
    rows = []

    def row(label, t_peak, t_cal):
        rows.append([label, f2(tok(t_peak)), f2(tok(t_cal))])

    row('HBM only (569.1 GB/s)', t_hbm_only(m, 64, 2, d), t_hbm_only(m, 64, 2, d, eta=eta))
    p_s = t_hbm_only(m, 64, 2, d, bw=HBM_STREAM_STAGGERED)
    row('HBM only, staggered lockstep read (578.3 GB/s)', p_s, t_hbm_only(m, 64, 2, d, eta=eta, bw=HBM_STREAM_STAGGERED))
    row('DDR only, 56 cores (262.5 GB/s)', t_ddr(m, 64, 2, d), t_ddr(m, 64, 2, d, eta=eta))
    row('DDR only, 28 cores (266.4 GB/s)', t_ddr(m, 64, 2, d, bw=DDR_STREAM_BEST), t_ddr(m, 64, 2, d, eta=eta, bw=DDR_STREAM_BEST))
    row('DDR at the DDR5-4800 datasheet rate (307.2 GB/s, derived)', t_ddr(m, 64, 2, d, bw=DDR_DATASHEET), t_ddr(m, 64, 2, d, eta=eta, bw=DDR_DATASHEET))
    # 40/80 layer serial split, the pilot placement (older tree: 8.18 tok/s)
    def p40(mm, gs, S, ctx, dtype):
        lw = layer_weight_bytes(mm, gs, S)
        cw = classifier_bytes(mm, gs, S)
        lk = kv_bytes_per_token(mm, S, dtype) / mm.layers * ctx
        return Placement(40, mm.layers, 40 * lw + cw, 40 * lk, 40 * lw, 40 * lk)
    row('serial 40/80 layers in HBM, 40 in DDR (the pilot split)', t_serial(m, 64, 2, d, place_fn=p40), t_serial(m, 64, 2, d, eta=eta, place_fn=p40))
    tc = t_concurrent(m, 64, 2, d)
    tcc = t_concurrent(m, 64, 2, d, eta=eta)
    row('concurrent, best split with no forced spill', tc[0], tcc[0])
    f = 0.5 * 0 + 0.5
    pf = t_concurrent(m, 64, 2, d, min_ddr_weight_fraction=0.5)
    pfc = t_concurrent(m, 64, 2, d, eta=eta, min_ddr_weight_fraction=0.5)
    row('concurrent with half of the weights forced to DDR (%d HBM + %d DDR cores)' % pf[1], pf[0], pfc[0])
    out = ['Bytes per socket per token at depth %d: weights %.2f GB, K/V %.3f GB (FP32). Calibration: HBM weight stream %.1f GB/s '
           '(eta %.3f), DDR weight stream %.1f GB/s (eta %.3f); K/V read at %.0f / %.0f GB/s.' % (
               d, W / GB, K / GB, rh, eta.hw, rd, eta.dw, KV_RATE_HBM, KV_RATE_DDR), '']
    out += md(['ceiling, 70B gs64, 2S, depth %d' % d, 'peak tok/s (raw P1 rates)', 'engine-calibrated tok/s'], rows)
    return out


def ctx_table(model, S, gs=64, dtype='fp32', ctxs=(4096, 32768, 65536, 131072)):
    m = MODELS[model]
    eta = CAL[cal_key(m, S)][0]
    rows = []
    for c in ctxs:
        p = place(m, gs, S, c, dtype)
        W, K = all_bytes(m, gs, S, c, dtype)
        cells = []
        for e in (PEAK, eta):
            cells.append((tok(t_hbm_only(m, gs, S, c, dtype, eta=e)), tok(t_ddr(m, gs, S, c, dtype, eta=e)),
                          tok(t_serial(m, gs, S, c, dtype, eta=e)), tok(t_concurrent(m, gs, S, c, dtype, eta=e)[0])))
        pk, cl = cells
        rows.append([c, f2((W + K) / GIB, 1), '%d/%d' % (p.n_hbm, m.layers), f2(p.ddr_fraction, 3)] +
                    [f2(pk[i]) for i in range(4)] + [f2(cl[i]) for i in range(4)])
    header = ['context', 'GiB read per socket per token', 'layers in HBM', 'DDR byte share', 'HBM-only peak', 'DDR peak', 'serial hybrid peak',
              'concurrent peak', 'HBM-only cal.', 'DDR cal.', 'serial hybrid cal.', 'concurrent cal.']
    return md(header, rows)


def table_kv_dtype():
    m = MODELS['70B']
    eta = CAL['70B-2S'][0]
    rows = []
    for c in (32768, 65536, 131072):
        for d in ('fp32', 'bf16', 'int8'):
            p = place(m, 64, 2, c, d)
            W, K = all_bytes(m, 64, 2, c, d)
            rows.append([c, d, f2(K / GIB, 1), '%d/%d' % (p.n_hbm, m.layers), f2(tok(t_hbm_only(m, 64, 2, c, d))),
                         f2(tok(t_serial(m, 64, 2, c, d))), f2(tok(t_concurrent(m, 64, 2, c, d)[0])),
                         f2(tok(t_serial(m, 64, 2, c, d, eta=eta))), f2(tok(t_concurrent(m, 64, 2, c, d, eta=eta)[0]))])
    return md(['context', 'KV dtype', 'KV GiB per socket', 'layers in HBM', 'HBM-only peak', 'serial peak', 'concurrent peak',
               'serial cal.', 'concurrent cal.'], rows)


def table_p1_check():
    """The 70B gs64 2S weight stream (36.92 GB per socket) against P1 section 7, from the E1 per-tier rates."""
    m = MODELS['70B']
    B = weight_bytes(m, 64, 2) / GB
    rows = []
    for ch, cd, bh, bd in E1[1:-1]:
        f = bd / (bh + bd)
        t_conc = B / (bh + bd)
        t_ser = B * ((1 - f) / HBM_STREAM + f / DDR_STREAM)
        rows.append(['%d + %d' % (ch, cd), f2(f, 3), f2(1 / t_conc), f2(1 / t_ser), f2(HBM_STREAM / B)])
    return md(['HBM + DDR cores', 'DDR byte fraction f', 'tok/s concurrent', 'tok/s serial tiers', 'tok/s all-HBM'], rows)


def table_fits_split():
    """Does moving bytes to DDR beat HBM-only when everything fits? Raw P1 rates: combined(c_d) against 569.1."""
    rows = []
    for ch, cd, bh, bd in E1[1:-1]:
        comb = E1_COMBINED[ch]
        rows.append(['%d + %d' % (ch, cd), f2(bh, 1), f2(HBM_STREAM - bh, 1), f2((HBM_STREAM - bh) / cd, 1), f2(bd, 1), f2(bd / cd, 1),
                     f2(comb, 1), '%+.1f%%' % ((comb / HBM_STREAM - 1) * 100)])
    return md(['HBM + DDR cores', 'HBM GB/s', 'HBM rate lost GB/s', 'lost per core moved', 'DDR GB/s', 'DDR per core moved',
               'combined GB/s (P1, median of sums)', 'vs HBM-only 569.1'], rows)


def table_engine_split():
    base = [447.2, 459.0, 453.3]
    rows = []
    for label, a, b, gain, src in (('node A run 1, r = 0.20 / 14 DDR workers', 447.2, 453.4, '+1.4%', 'job 5923181'),
                                   ('node A run 2, r = 0.20 / 14 DDR workers', 459.0, 469.8, '+2.3%', 'job 5924010'),
                                   ('node B, r = 0.20 / 14 DDR workers', 453.3, 470.9, '+3.9%', 'job 5924011')):
        B = weight_bytes(MODELS['70B'], 64, 2) / GB
        rows.append([label, f2(a, 1), f2(b, 1), gain, f2(a / B), f2(b / B), src])
    return md(['E5 GEMV, gs64, 4 KiB pages', 'all-HBM GB/s', 'best split GB/s', 'gain stated in P1', 'tok/s-equivalent all-HBM', 'tok/s-equivalent split', 'source'], rows)


def table_ttft_ops():
    rows = []
    for name, P in (('70B', 547), ('70B', 2048), ('70B', 2113), ('70B', 8192), ('8B', 2048), ('8B', 8192)):
        m = MODELS[name]
        g, a = gemm_ops(m, P), attention_ops(m, P)
        tot = prefill_ops(m, P)
        row = [name, P, f2(g / 1e12, 1), f2(a / 1e12, 2), f2(tot / 1e12, 1), '%.1f%%' % (100 * a / tot)]
        for S in (1, 2):
            row.append(f2(t_amx_peak(m, P, S), 3))
        row.append(f2(t_amx_peak(m, P, 2, ONEDNN_M256_TOPS), 3))
        rows.append(row)
    return md(['model', 'prompt tokens', 'GEMM TOP', 'attention TOP', 'total TOP', 'attention share', 'AMX peak 1S s', 'AMX peak 2S s',
               'oneDNN M=256 rate 2S s'], rows)


def table_ttft_floor():
    rows = []
    m = MODELS['70B']
    for gs in (64, 32):
        for P in (547, 2113):
            for M in (256, 512, 1024):
                nb = n_batches(P, M)
                rows.append([gs, P, M, nb, f2(weight_bytes(m, gs, 2) / GB, 1), f2(t_read_floor(m, gs, 2, P, M, DDR_STREAM), 2),
                             f2(t_read_floor(m, gs, 2, P, M, HBM_STREAM), 2), f2(t_amx_peak(m, P, 2), 3)])
    out = md(['gs', 'prompt', 'batch M', 'batches', 'packed GB per socket', 'DDR read floor s', 'HBM read floor s', 'AMX peak s'], rows)
    out.append('')
    # break-even batch: compute per batch at peak equals the packed read
    be = []
    for gs in (64, 32):
        W = weight_bytes(m, gs, 2)
        per_row_s = 2.0 * m.layer_params() * m.layers / 2.0 / (AMX_PEAK_TOPS * 1e12)    # seconds per batch row at peak, per socket
        for tier, bw in (('DDR', DDR_STREAM), ('HBM', HBM_STREAM)):
            be.append([gs, tier, f2(W / (bw * GB), 3), f2(per_row_s * 1e3, 4), f2((W / (bw * GB)) / per_row_s, 0)])
    out += md(['gs', 'packed copy in', 'read s per batch', 'AMX-peak compute ms per batch row', 'break-even batch rows'], be)
    return out


def table_ttft_gap():
    rows = []
    m70 = MODELS['70B']
    for label, name, S, gs, P, meas, src in TTFT_CELLS:
        m = MODELS[name]
        pk = t_amx_peak(m, P, S)
        row = [label, S, P, f2(meas, 3), f2(pk, 3), '%.1f' % (meas / pk)]
        if name == '70B':
            fl = max(pk, t_read_floor(m, gs, S, P, BATCH_XENGINE, DDR_STREAM))
            row += [f2(fl, 3), '%.1f' % (meas / fl)]
            sc = ttft_scaled_70b(gs, S, P)['total']
            row += [f2(sc, 2), '%.2f' % (meas / sc)]
        else:
            row += ['-', '-', '-', '-']
        row.append(src)
        rows.append(row)
    return md(['cell', 'S', 'prompt', 'measured s', 'AMX peak ceiling s', 'measured / peak',
               'peak or DDR packed read floor at M=256, s', 'measured / that', '8B-scaled prediction s', 'measured / scaled', 'source'], rows)


def table_ttft_budget():
    rows = []
    for gs in (64, 32):
        for S in (1, 2):
            b = budget_8b(gs, S)
            peak = b['ops'] / (S * AMX_PEAK_TOPS * 1e12)
            rows.append([gs, S, f2(b['rate'], 1), '%.1f%%' % (100 * b['rate'] / (S * AMX_PEAK_TOPS)), f2(b['gemm'], 3), f2(b['attn'], 3),
                         f2(b['other'], 3), f2(b['fwd'], 3), f2(peak, 3)])
    out = md(['gs', 'S', 'GEMM TOP/s, node, time-weighted', 'share of AMX peak', 'GEMM s', 'attention s', 'rest s (forward - GEMM - attention)',
              'forward s', 'AMX peak GEMM s'], rows)
    out.append('')
    rows = []
    for gs in (64, 32):
        for P in (547, 2113, 8192):
            s = ttft_scaled_70b(gs, 2, P)
            rows.append([gs, P, f2(t_amx_peak(MODELS['70B'], P, 2), 3), f2(s['gemm'], 2), f2(s['attn'], 2), f2(s['other'], 2), f2(s['total'], 2)])
    out += md(['gs', 'prompt', '70B AMX peak s', 'GEMM at 8B in-situ rates s', 'attention x 5.0 s', 'rest x 5.0 s', 'total s'], rows)
    return out


def table_ttft_residual():
    """Measured TTFT less the 8B-scaled prediction, against the number of prefill batches (M = 256) and the packed-copy read of one batch."""
    rows = []
    for label, name, S, gs, P, meas, src in TTFT_CELLS:
        if name != '70B':
            continue
        sc = ttft_scaled_70b(gs, S, P)['total']
        nb = n_batches(P, BATCH_XENGINE)
        rd = weight_bytes(MODELS['70B'], gs, S) / (DDR_STREAM * GB)
        rows.append([label, P, nb, f2(meas, 3), f2(sc, 3), f2(meas - sc, 3), f2((meas - sc) / nb, 3), f2((meas - sc) / P * 1e3, 3), f2(rd, 3)])
    return md(['cell (2S)', 'prompt', 'batches at M=256', 'measured s', '8B-scaled s', 'residual s', 'residual per batch s',
               'residual per prompt token ms', 'packed read per batch from DDR s'], rows)


def table_decode_gap():
    rows = []
    for label, name, S, gs, depth, tier, meas, src in DECODE_CELLS:
        m = MODELS[name]
        eta = CAL[cal_key(m, S)][0]
        if tier == 'hbm':
            tp, tc = t_hbm_only(m, gs, S, depth), t_hbm_only(m, gs, S, depth, eta=eta)
        elif tier == 'ddr':
            tp, tc = t_ddr(m, gs, S, depth), t_ddr(m, gs, S, depth, eta=eta)
        else:
            tp, tc = t_serial(m, gs, S, depth), t_serial(m, gs, S, depth, eta=eta)
        W, K = all_bytes(m, gs, S, depth)
        rows.append([label, tier, depth, f2(meas, 2), f2((W + K) / GB * meas, 0), f2(tok(tp)), '%.3f' % (meas * tp),
                     f2(tok(tc)), '%.3f' % (meas * tc), src])
    out = md(['cell', 'tier', 'depth', 'measured tok/s', 'delivered GB/s per socket', 'peak ceiling tok/s', 'measured / peak',
              'calibrated ceiling tok/s', 'measured / calibrated', 'source'], rows)
    out.append('')
    rows = []
    B = weight_bytes(MODELS['70B'], 64, 2) / GB
    for label, a, b, src in E5_CELLS:
        rows.append([label, f2(a, 1) + ('' if b is None else ' / ' + f2(b, 1)), '%.3f' % (a / HBM_STREAM), f2(a / B), src])
    out += md(['P1 E5 (weights only, no attention, no classifier)', 'GB/s', 'against raw 569.1', 'tok/s-equivalent at 36.92 GB', 'source'], rows)
    return out


def table_calibration():
    rows = []
    for key in ('70B-2S', '8B-1S', '8B-2S'):
        eta, rh, rd = CAL[key]
        a = {'70B-2S': ANCHOR_70B, '8B-1S': ANCHOR_8B_1S, '8B-2S': ANCHOR_8B_2S}[key]
        rows.append([key, f2(a['hbm'], 3), f2(a['ddr'], 3), a['depth'], f2(rh, 1), '%.3f' % eta.hw, f2(rd, 1), '%.3f' % eta.dw])
    return md(['configuration', 'anchor HBM tok/s', 'anchor DDR tok/s', 'depth', 'HBM weight stream GB/s', 'eta HBM (vs 569.1)',
               'DDR weight stream GB/s', 'eta DDR (vs 262.5)'], rows)


def table_attention_rate():
    """K/V read rates the engine achieved, from end-to-end numbers (derived; none is a direct attention-kernel measurement of a clean setup)."""
    m = MODELS['70B']
    kv = kv_bytes_per_token(m, 2, 'fp32') / GB                 # GB per position per socket
    rows = [['deep, HBM: 73,728 positions, attention 84.84 ms per step (older tree, grouped)', f2(kv * DEEP_HBM_CTX / DEEP_HBM_ATTN_S, 0), 'job 5904763'],
            ['deep, DDR: 131,072 positions, DDR-only 2.95 tok/s less the weight stream (older tree)', f2(KV_RATE_DDR, 0), 'job 5904765'],
            ['4096 positions, HBM: profile attention 13.0 ms per token, stagger on', f2(kv * 4096 / 0.013, 0), 'job 5934864'],
            ['4096 positions, HBM: fixed part if K/V were read at the deep HBM rate (13.0 ms - K/V / deep rate)',
             '%.1f ms' % (1e3 * (0.013 - kv * 4096 / KV_RATE_HBM)), 'derived']]
    for label, ha, hb in (('gs64', 11.583, 10.709), ('gs32', 10.966, 10.120)):
        rows.append(['marginal 611 to 2177 positions, HBM hybrid, %s: dKV / d(1 / tok/s)' % label,
                     f2(kv * (2177 - 611) / (1 / hb - 1 / ha), 0), 'job 5935580'])
    for label, ha, hb, j in (('gs64', 6.216, 6.011, 'job 5929407'), ('gs32', 5.928, 5.755, 'job 5929406')):
        rows.append(['marginal 611 to 2177 positions, DDR, %s' % label, f2(kv * (2177 - 611) / (1 / hb - 1 / ha), 0), j])
    return md(['K/V read rate observation (70B 2S, FP32 K/V)', 'GB/s per socket (or ms)', 'source'], rows)


CKPT_BYTES = 74967187712          # 70B Q8_0 v3 file, measured (older tree, docs/plans/hbm-capacity-study.md / reference numbers 1.02)
LOAD_FIRST_S = 106.0              # first start, measured, job 5904620 (older tree)
LOAD_LATER_S = (33.0, 36.0)       # later starts, measured, same job


def table_load():
    rows = [['first start', '106', f2(CKPT_BYTES / GB / LOAD_FIRST_S, 2)],
            ['later starts', '33-36', '%.2f-%.2f' % (CKPT_BYTES / GB / LOAD_LATER_S[1], CKPT_BYTES / GB / LOAD_LATER_S[0])]]
    return md(['70B Q8_0 checkpoint, 75.0 GB', 'load seconds (measured, job 5904620, older tree)', 'derived GB/s'], rows)


TABLES = OrderedDict([
    ('attention_rate', table_attention_rate), ('load', table_load),
    ('sizes', table_sizes), ('p1_check', table_p1_check), ('calibration', table_calibration), ('anchor', table_anchor),
    ('old_vs_new', table_old_vs_new),
    ('ctx_70b_2s', lambda: ctx_table('70B', 2)), ('ctx_70b_1s', lambda: ctx_table('70B', 1)),
    ('ctx_8b_1s', lambda: ctx_table('8B', 1)), ('ctx_8b_2s', lambda: ctx_table('8B', 2)), ('kv_dtype', table_kv_dtype),
    ('ttft_ops', table_ttft_ops), ('ttft_floor', table_ttft_floor), ('ttft_budget', table_ttft_budget), ('ttft_gap', table_ttft_gap), ('ttft_residual', table_ttft_residual),
    ('fits_split', table_fits_split), ('engine_split', table_engine_split), ('decode_gap', table_decode_gap),
])


def render(name):
    return '\n'.join(TABLES[name]())


BLOCK = re.compile(r'(<!-- hybrid-limits:begin (\S+) -->\n)(.*?)(\n<!-- hybrid-limits:end \2 -->)', re.S)


def update_doc(path, check=False):
    text = Path(path).read_text()
    bad = []

    def sub(mo):
        name = mo.group(2)
        new = render(name)
        if mo.group(3) != new:
            bad.append(name)
        return mo.group(1) + new + mo.group(4)

    out = BLOCK.sub(sub, text)
    if check:
        return bad
    Path(path).write_text(out)
    return bad


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split('\n\n')[0], formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('names', nargs='*', help='table names (default all): ' + ', '.join(TABLES))
    ap.add_argument('--update-doc', metavar='DOC')
    ap.add_argument('--check-doc', metavar='DOC')
    a = ap.parse_args(argv)
    if a.update_doc:
        update_doc(a.update_doc)
        return 0
    if a.check_doc:
        bad = update_doc(a.check_doc, check=True)
        for n in bad:
            print('generated block differs from the tool: ' + n, file=sys.stderr)
        return 1 if bad else 0
    for n in a.names or TABLES:
        print('### ' + n + '\n')
        print(render(n) + '\n')
    return 0


if __name__ == '__main__':
    sys.exit(main())
