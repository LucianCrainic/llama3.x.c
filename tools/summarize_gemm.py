"""Summarize gemm-bench JSON lines as Markdown: per weight shape, GFLOP/s and weight GB/s against M for the GEMM and
the per-row GEMV, with the speedup of one over the other.

    summarize_gemm.py gemm-bench.jsonl [--peak-gflops 8800]
"""
import argparse
import json
from pathlib import Path


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('file')
    parser.add_argument('--peak-gflops', type=float, default=8800.0,
                        help='measured FP32 FMA peak of the socket in GFLOP/s (docs/hbm-roofline.md); 0 hides the column')
    args = parser.parse_args(argv)
    rows = [json.loads(line) for line in Path(args.file).read_text().splitlines() if line.startswith('{')]
    shapes = sorted({(r['n'], r['k']) for r in rows})
    out = []
    for n, k in shapes:
        out += [f'### W {n} x {k}', '', '| M | GEMM GFLOP/s (best / mean) | GEMM weight GB/s | % of peak | GEMV-rows GFLOP/s | GEMM / GEMV |',
                '| ---: | ---: | ---: | ---: | ---: | ---: |']
        for m in sorted({r['m'] for r in rows if (r['n'], r['k']) == (n, k)}):
            gemm = next((r for r in rows if (r['n'], r['k'], r['m'], r['kernel']) == (n, k, m, 'gemm')), None)
            gemv = next((r for r in rows if (r['n'], r['k'], r['m'], r['kernel']) == (n, k, m, 'gemv_rows')), None)
            if not gemm:
                continue
            peak = f"{100 * gemm['gflops_best'] / args.peak_gflops:.0f}%" if args.peak_gflops else '-'
            ratio = f"{gemm['gflops_best'] / gemv['gflops_best']:.1f}x" if gemv else '-'
            out.append(f"| {m} | {gemm['gflops_best']:.0f} / {gemm['gflops_mean']:.0f} | {gemm['weight_gbs_best']:.0f} | {peak} | "
                       f"{gemv['gflops_best']:.0f} | {ratio} |" if gemv else
                       f"| {m} | {gemm['gflops_best']:.0f} / {gemm['gflops_mean']:.0f} | {gemm['weight_gbs_best']:.0f} | {peak} | - | - |")
        out.append('')
    if rows:
        first = rows[0]
        out.append(f"isa={first['isa']} backend={first['backend']} weights on node {first['node']} kc={first['kc']}")
    print('\n'.join(out))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
