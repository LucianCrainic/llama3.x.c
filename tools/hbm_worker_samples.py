#!/usr/bin/env python3
"""Sample scheduling state of an owned benchmark process and its workers on Linux."""
import json
from pathlib import Path
import sys
import time


def snapshot(tasks):
    workers = []
    for thread in tasks.iterdir():
        try:
            stat = (thread / "stat").read_text().rsplit(") ", 1)[1].split()
            runtime, wait, _ = map(int, (thread / "schedstat").read_text().split())
            wait_channel = (thread / "wchan").read_text().strip() if stat[0] in ("D", "S") else None
        except FileNotFoundError:
            continue
        workers.append([int(thread.name), int(stat[36]), stat[0], runtime, wait, wait_channel])
    return workers


def main():
    tasks = Path("/proc") / str(int(sys.argv[1])) / "task"
    print(json.dumps({"worker_fields": ["tid", "cpu", "state", "run_ns", "wait_ns", "wait_channel"]}), flush=True)
    while tasks.exists():
        started = time.time_ns()
        try:
            workers = snapshot(tasks)
        except FileNotFoundError:
            break
        print(json.dumps({"epoch_ns": started, "workers": workers}), flush=True)
        time.sleep(0.5)


if __name__ == "__main__":
    main()
