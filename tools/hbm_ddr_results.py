#!/usr/bin/env python3
"""Summarize native phase markers, NUMA plans and perf intervals from 70B tier campaigns.

Run with one or more directories of raw JSON and perf CSV. Each native report produces
one JSON line. Rates use only complete intervals inside the phase for both perf clock
anchors. Bytes and joules per step extrapolate the interior rates over the phase wall time.
"""
import argparse
import json
from pathlib import Path
from statistics import median

from xengine_summary import DDR_CAS_BYTES, HBM_CAS_BYTES, NOMINAL_GHZ, parse_perf


CAS = {"hbm": HBM_CAS_BYTES, "ddr": DDR_CAS_BYTES, "hbm_wr": HBM_CAS_BYTES, "ddr_wr": DDR_CAS_BYTES}


def phase_bounds(first, last, anchors, guard):
    return (first - min(anchors)) / 1e9 + guard, (last - max(anchors)) / 1e9 - guard


def valid_bins(bins, a, b):
    return [(start, end, value) for start, end, value, running in bins
            if a <= start < end <= b and value is not None and running is not None and running >= 95]


def joint_read_bins(series, first, last, anchors, guard):
    """Per-interval GB/s using bins present on both tiers of both sockets."""
    a, b = phase_bounds(first, last, anchors, guard)
    controllers = [{(start, end): value for start, end, value in valid_bins(series.get((socket, tier), []), a, b)}
                   for tier in ("hbm", "ddr") for socket in (0, 1)]
    common = set(controllers[0]).intersection(*controllers[1:])
    for start, end in sorted(common):
        key = (start, end)
        hbm = sum(c[key] for c in controllers[:2]) * HBM_CAS_BYTES
        ddr = sum(c[key] for c in controllers[2:]) * DDR_CAS_BYTES
        yield (start, end, hbm / (end - start) / 1e9, ddr / (end - start) / 1e9)


def low_read_spans(bins, bandwidth=8, min_seconds=2):
    """Periods of little memory traffic. These locate pauses without attributing their cause."""
    spans = []
    for start, end, hbm, ddr in bins:
        if hbm + ddr >= bandwidth:
            continue
        if spans and spans[-1][1] == start:
            spans[-1][1] = end
        else:
            spans.append([start, end])
    return [span for span in spans if span[1] - span[0] >= min_seconds]


def phase_counters(series, idle, first, last, anchors, steps, guard):
    """Use only bins fully inside the window even at the extreme anchor estimates."""
    if first is None or last is None or not series or not steps:
        return None

    a, b = phase_bounds(first, last, anchors, guard)
    duration = (last - first) / 1e9
    sockets = []
    for socket in sorted({socket for socket, _ in series}):
        rates = {}
        coverage = {}
        for (which, name), bins in series.items():
            if which != socket:
                continue
            accepted = valid_bins(bins, a, b)
            covered = sum(end - start for start, end, _ in accepted)
            if covered:
                rates[name] = sum(value for _, _, value in accepted) / covered
                coverage[name] = covered / duration

        result = {"socket": socket, "coverage": coverage}
        for name, size in CAS.items():
            if name in rates:
                result[f"{name}_gbps"] = rates[name] * size / 1e9
                result[f"{name}_gb_per_step"] = result[f"{name}_gbps"] * duration / steps
        for name in ("upi_rx", "upi_tx"):
            if name in rates:
                result[f"{name}_mflits_per_step"] = rates[name] * duration / steps / 1e6
        if rates.get("ref_cycles") and "cycles" in rates:
            result["equivalent_unhalted_cores"] = rates["ref_cycles"] / (NOMINAL_GHZ * 1e9)
            result["active_ghz"] = rates["cycles"] / rates["ref_cycles"] * NOMINAL_GHZ
        for name in ("context_switches", "cpu_migrations"):
            if name in rates:
                result[f"{name}_per_second"] = rates[name]
        for name in ("pkg_j", "ram_j"):
            if name in rates:
                result[f"{name}_per_step"] = rates[name] * duration / steps
                if (socket, name) in idle:
                    result[f"{name}_net_per_step"] = (rates[name] - idle[socket, name]) * duration / steps
        sockets.append(result)
    spans = low_read_spans(joint_read_bins(series, first, last, anchors, guard))
    return {"seconds": duration, "clock_anchor_skew_ms": (max(anchors) - min(anchors)) / 1e6,
            "sockets": sockets, "low_read_spans_s": spans}


def logical_decode_bytes(report):
    """Expected one read of original projections and each live K/V cell per step.

    Includes duplicated FP32 norms and a single embedding row. Excludes framework
    traffic, cacheline and prefetch effects, and repeated attention loads from LLC.
    """
    cfg = report["model"]["configuration"]
    placement = report["placement_plan"]
    sockets = len({node["socket"] for node in placement["nodes"]})
    group = report["model"]["quantization_group_size"]
    dim, layers, hidden = cfg["dim"], cfg["n_layers"], cfg["hidden_dim"]
    kv_dim = cfg["n_kv_heads"] * cfg["head_dim"]
    weight_layer = (2 * dim * dim + 2 * dim * kv_dim + 3 * dim * hidden) * (1 + 4 / group)
    classifier = cfg["vocab_size"] * dim * (1 + 4 / group)
    norm = (2 * layers + 1) * dim * 4 * sockets
    embedding = dim * (1 + 4 / group)
    tokens = report["tokens"]
    warm = report["parameters"]["warm_context"]
    # The first decode adds one position. Its index is warm + prompt, zero-based.
    positions = warm + tokens["processed_prompt"] + (tokens["decode_steps"] + 1) / 2
    layer_kv = 2 * kv_dim * 4 * positions
    mode = placement["mode"]
    hbm_weight = weight_layer * placement["weights_hbm_layers"]
    hbm_weight += classifier if placement["classifier"] == "hbm" else 0
    hbm_weight += norm if mode != "ddr" else 0
    hbm_weight += embedding if mode == "hbm-only" or (cfg["shared_embeddings"] and placement["classifier"] == "hbm") else 0
    hbm_kv = layer_kv * placement["kv_hbm_layers"]
    return {
        "mean_positions": positions,
        "weight_gb": (weight_layer * layers + classifier + norm + embedding) / 1e9,
        "kv_gb": layer_kv * layers / 1e9,
        "hbm_weight_gb": hbm_weight / 1e9,
        "hbm_kv_gb": hbm_kv / 1e9,
    }


def idle_rates(path):
    rates = {}
    for key, bins in parse_perf(path).items():
        usable = [(end - start, value) for start, end, value, running in bins[1:-1]
                  if end > start and value is not None and running is not None and running >= 95]
        if usable:
            rates[key] = median(value / duration for duration, value in usable)
    return rates


def results(directory, guard):
    idle_path = directory / "idle.perf.csv"
    idle = idle_rates(idle_path) if idle_path.is_file() else {}
    environment = directory / "environment.txt"
    metadata = dict(line.split("=", 1) for line in environment.read_text().splitlines()
                    if line.startswith(("commit=", "job=", "interval_ms="))) if environment.is_file() else {}

    for path in sorted(directory.glob("*.json")):
        if path.name.endswith((".plan.json", ".tokens.json")):
            continue
        report = json.loads(path.read_text())
        if report.get("status") != "completed" or not report.get("placement_plan"):
            continue
        prefix = path.with_suffix("")
        perf = Path(f"{prefix}.perf.csv")
        series = parse_perf(perf)
        markers = report["phase_epoch_ns"]
        counters = {}
        decode_reads = []
        if series:
            start = int(Path(f"{prefix}.perf-start").read_text())
            end = int(Path(f"{prefix}.end").read_text())
            final = max(bins[-1][1] for bins in series.values())
            # The tail monitor polls for child exit every 100 ms in the original
            # campaign and 50 ms in the research campaign.
            half_poll = 25_000_000 if Path(f"{prefix}.perf-end").exists() else 50_000_000
            anchors = (start, end - round(final * 1e9) + half_poll)
            warm_seconds = report["timing_seconds"]["warm_context"]
            if warm_seconds and markers["prefill_start"]:
                warm_end = markers["prefill_start"]
                warm_start = warm_end - round(warm_seconds * 1e9)
                counters["warm"] = phase_counters(
                    series, idle, warm_start, warm_end, anchors, report["parameters"]["warm_context"], guard
                )
            for phase in ("prefill", "decode"):
                steps = report["tokens"]["decode_steps" if phase == "decode" else "processed_prompt"]
                counters[phase] = phase_counters(
                    series, idle, markers[f"{phase}_start"], markers[f"{phase}_end"], anchors, steps, guard
                )

            if metadata.get("interval_ms") == "10" and markers["decode_start"] and markers["decode_end"]:
                decode_reads = list(joint_read_bins(series, markers["decode_start"], markers["decode_end"], anchors, guard))

        yield {
            "campaign": directory.name, "run": prefix.name, **metadata, "node": report["system"]["hostname"],
            "parameters": report["parameters"], "build": report["build"],
            "tokens": report["tokens"], "timing": report["timing_seconds"],
            "rates": report["throughput_tokens_per_second"], "placement": report["placement_plan"],
            "kv": report["kv"], "memory": report["memory_bytes"], "logical_decode": logical_decode_bytes(report),
            "profile": report.get("profile_seconds"), "counters": counters,
            "decode_read_intervals": decode_reads,
        }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path, nargs="+")
    parser.add_argument("--guard", type=float, default=0.05, help="seconds trimmed from each phase boundary")
    args = parser.parse_args()
    for directory in args.directory:
        for result in results(directory, args.guard):
            print(json.dumps(result))


if __name__ == "__main__":
    main()
