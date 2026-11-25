#!/usr/bin/env python3
"""Summarize 70B single-request timing and validate exact request lengths."""
import argparse
from collections import defaultdict
import json
from pathlib import Path
from statistics import median


LENGTHS = {"short": 547, "long": 2113}
GENERATION = 128


def lines(path):
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def summarize(directory):
    native = directory / "runs.jsonl"
    observed = defaultdict(list)
    if native.is_file():
        for row in lines(native):
            name = row["workload"]
            if row["prompt_tokens"] != LENGTHS[name] or row["generated_tokens"] != GENERATION:
                raise ValueError(f"{directory}: wrong native workload size")
            observed[name].append((row["warmup"], row["ttft_s"], row["decode_s"], row["token_ids"]))
    else:
        for name, length in LENGTHS.items():
            for row in lines(directory / f"{name}.jsonl"):
                arrivals = row["token_arrivals"]
                if not row["ok"] or row["usage_prompt_tokens"] != length or len(arrivals) != GENERATION:
                    raise ValueError(f"{directory}: wrong vLLM workload size")
                observed[name].append((row["warmup"], row["t_first_token"], arrivals[-1] - arrivals[0], row["token_ids"]))

    for name, length in LENGTHS.items():
        rows = observed[name]
        if len(rows) != 4 or sum(warmup for warmup, *_ in rows) != 1:
            raise ValueError(f"{directory}: expected one warm-up and three measured {name} requests")
        if any(t <= 0 or d <= 0 or len(ids) != GENERATION for _, t, d, ids in rows):
            raise ValueError(f"{directory}: invalid {name} request result")
        samples = [(t, (GENERATION - 1) / d, ids) for warmup, t, d, ids in rows if not warmup]
        yield {"engine": directory.name, "workload": name, "prompt_tokens": length, "ttft_s": [r[0] for r in samples],
               "decode_tps": [r[1] for r in samples], "trajectories": [r[2] for r in samples]}


def med_range(values):
    return f"{median(values):.2f} [{min(values):.2f}–{max(values):.2f}]"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directories", type=Path, nargs="+")
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()
    rows = [row for directory in args.directories for row in summarize(directory)]
    if args.json:
        print(json.dumps(rows, indent=2))
        return
    print("| Engine | Prompt | TTFT (s) | Decode (tok/s) |")
    print("| --- | ---: | ---: | ---: |")
    for row in rows:
        print(f"| {row['engine']} | {row['prompt_tokens']} | {med_range(row['ttft_s'])} | {med_range(row['decode_tps'])} |")


if __name__ == "__main__":
    main()
