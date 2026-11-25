#!/usr/bin/env python3
"""Plot matched 70B decode controls from hbm_ddr_results.py JSON lines."""
import argparse
import json
from pathlib import Path
from statistics import median

import matplotlib.pyplot as plt


TIERS = ("hbm", "ddr")
CASES = ("HBM-only", "40/40 hybrid", "DDR-only")
COLORS = ("#267285", "#dd9b57")


def baseline_rows(rows):
    groups = {name: [] for name in CASES}
    for row in rows:
        plan = row["placement"]
        params = row["parameters"]
        if (params["warm_context"] != 512 or params["prefill_batch"] != 128
                or params["max_forward_passes"] != 8192 or row["tokens"]["processed_prompt"] != 35
                or row["build"]["kernels"] != "avx512" or row["build"]["attention_decode"] != "split"):
            continue
        if plan["mode"] == "hbm-only":
            groups["HBM-only"].append(row)
        elif plan["mode"] == "ddr":
            groups["DDR-only"].append(row)
        elif plan["mode"] == "hybrid" and plan["weights_hbm_layers"] == plan["kv_hbm_layers"] == 40:
            groups["40/40 hybrid"].append(row)
    return groups


def decode_median(rows, key):
    values = []
    for row in rows:
        phase = row["counters"].get("decode")
        if phase and phase["sockets"] and all(key in socket for socket in phase["sockets"]):
            values.append(sum(socket[key] for socket in phase["sockets"]))
    return median(values)


def plot_baseline(rows, output):
    groups = baseline_rows(rows)
    if any(not group for group in groups.values()):
        raise ValueError("The selected campaign needs HBM-only, 40/40 hybrid and DDR-only controls")
    measured = [[row["rates"]["decode"] for row in groups[name]] for name in CASES]
    hbm, _, ddr = (median(values) for values in measured)
    logical = groups["40/40 hybrid"][0]["logical_decode"]
    f_hbm = (logical["hbm_weight_gb"] + logical["hbm_kv_gb"]) / (logical["weight_gb"] + logical["kv_gb"])
    predicted = 1 / (f_hbm / hbm + (1 - f_hbm) / ddr)

    fig, (rate, traffic) = plt.subplots(1, 2, figsize=(9.4, 3.6), layout="constrained")
    labels = ["HBM-only", "40/40 hybrid", "DDR-only"]
    for index, values in enumerate(measured):
        rate.scatter([index] * len(values), values, marker="o", color="#1e3340", zorder=3)
    rate.scatter(1.18, predicted, marker="D", facecolor="white", edgecolor="#1e3340", label="Serial estimate", zorder=3)
    rate.set(xticks=range(3), xticklabels=labels, xlim=(-0.45, 2.45), ylim=(0, 12.5), ylabel="Decode tokens/s")
    rate.legend(loc="lower left", frameon=False)
    rate.set_title("Decode throughput", loc="left", fontweight="bold")

    bottom = [0] * 3
    for tier, color in zip(TIERS, COLORS):
        values = [decode_median(groups[name], f"{tier}_gbps") for name in CASES]
        traffic.bar(range(3), values, bottom=bottom, color=color, label=tier.upper(), width=0.55)
        bottom = [sum(pair) for pair in zip(bottom, values)]
    traffic.set(xticks=range(3), xticklabels=labels, ylim=(0, 930), ylabel="Read bandwidth (GB/s)")
    traffic.set_title("Controllers during decode", loc="left", fontweight="bold")
    traffic.legend(loc="upper right", frameon=False)
    for axis in (rate, traffic):
        axis.spines[["right", "top"]].set_visible(False)
        axis.grid(axis="y", alpha=0.2)
        axis.set_axisbelow(True)
    save_figure(fig, output)


def plot_read_windows(rows, output):
    row, = (row for row in rows if row["decode_read_intervals"])
    bins = row["decode_read_intervals"]
    mid = (bins[0][0] + bins[-1][1]) / 2
    sample = [b for b in bins if abs(b[0] - mid) < 0.25]
    origin = sample[0][0]
    edges = [(b[0] - origin) * 1000 for b in sample] + [(sample[-1][1] - origin) * 1000]
    fig, axis = plt.subplots(figsize=(8.2, 3.2), layout="constrained")
    for column, name, color in zip((2, 3), TIERS, COLORS):
        axis.stairs([b[column] for b in sample], edges, color=color, linewidth=1.7, label=name.upper())
    axis.set(xlabel="Time within decode excerpt (ms)", ylabel="Read bandwidth (GB/s)", xlim=(0, edges[-1]), ylim=(0, 1040))
    axis.set_title("Alternating tier reads in 70B hybrid decode", loc="left", fontweight="bold")
    axis.spines[["right", "top"]].set_visible(False)
    axis.grid(axis="y", alpha=0.2)
    axis.set_axisbelow(True)
    axis.legend(loc="upper right", frameon=False)
    save_figure(fig, output)


def plot_capacity(rows, output):
    policies = {
        "Strict HBM": ("#267285", (("budget", "hbm-short"), ("stress-primed", "near-hbm"))),
        "Weight-first, split": ("#633eb1", (("budget", "auto-short"), ("stress-primed", "validate-short"),
                                            ("stress-primed", "edge-auto"), ("edge-auto", "auto-edge"),
                                            ("stress-primed", "max-auto"), ("stress-limit", "limit-auto"))),
        "KV-heavy, split": ("#a66021", (("budget", "exchange-short"), ("stress-primed", "edge-exchange"),
                                        ("stress-primed", "max-exchange"))),
        "Weight-first, grouped": ("#277348", (("group-primed", "short-real-group"),
                                             ("group-primed", "edge-auto-group"),
                                             ("group-primed", "max-auto-group"), ("edge-group-real", "auto-edge-group"),
                                             ("stress-limit", "limit-auto-group"))),
        "DDR-only, split": ("#525b65", (("stress-primed", "edge-ddr"),)),
        "DDR-only, grouped": ("#8798a4", (("stress-limit", "limit-ddr-group"),)),
    }
    fig, axis = plt.subplots(figsize=(8.3, 4), layout="constrained")
    for label, (color, keys) in policies.items():
        values = {}
        for row in rows:
            base = row["run"].removesuffix("-2")
            if (row["campaign"], base) not in keys:
                continue
            depth = row["parameters"]["warm_context"]
            synthetic = row["parameters"].get("synthetic_kv", False)
            values.setdefault((depth, synthetic), []).append(row["rates"]["decode"])
        ordered = sorted(values.items())
        label_points = {}
        for (depth, _), samples in ordered:
            label_points.setdefault(depth, median(samples))
        if label_points:
            axis.plot([depth / 1000 for depth in label_points], list(label_points.values()),
                      color=color, linewidth=1.1, alpha=0.65, label=label)
        for (depth, synthetic), samples in sorted(ordered, key=lambda point: not point[0][1]):
            rate = median(samples)
            x = depth / 1000
            axis.errorbar(x, rate, yerr=[[rate - min(samples)], [max(samples) - rate]], fmt="o",
                          markersize=6 if synthetic else 3.7,
                          markerfacecolor="white" if synthetic else color,
                          markeredgecolor=color, ecolor=color, capsize=3, zorder=3 if synthetic else 4)

    axis.set(xlabel="Warm positions before the prompt (thousands)", ylabel="Decode tokens/s", ylim=(0, None))
    axis.set_title("70B decode approaching and exceeding the HBM budget", loc="left", fontweight="bold")
    axis.spines[["right", "top"]].set_visible(False)
    axis.grid(axis="y", alpha=0.2)
    axis.set_axisbelow(True)
    axis.legend(loc="upper right", frameon=False)
    save_figure(fig, output)


def save_figure(fig, output):
    if output.suffix.lower() == ".svg":
        fig.savefig(output, metadata={"Date": None})
        output.write_text("\n".join(line.rstrip() for line in output.read_text().splitlines()) + "\n")
    else:
        fig.savefig(output)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("summary", type=Path)
    parser.add_argument("--campaign", help="raw directory name in the JSON lines")
    parser.add_argument("--kind", choices=("baseline", "windows", "capacity"), default="baseline")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    rows = [json.loads(line) for line in args.summary.read_text().splitlines() if line.strip()]
    plt.rcParams.update({"font.size": 10, "svg.fonttype": "none", "svg.hashsalt": "hbm-ddr-decode"})
    plots = {"baseline": plot_baseline, "windows": plot_read_windows, "capacity": plot_capacity}
    if args.campaign:
        rows = [row for row in rows if row["campaign"] == args.campaign]
    plots[args.kind](rows, args.output)


if __name__ == "__main__":
    main()
