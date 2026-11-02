#!/usr/bin/env python3
"""Replay rank segments through fast_parse with a bounded parallel worker pool."""

import argparse
import concurrent.futures
import json
import math
import statistics
import subprocess
import sys
import time
from pathlib import Path


def percentile(values, quantile):
    ordered = sorted(values)
    if not ordered:
        return None
    position = quantile * (len(ordered) - 1)
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    fraction = position - lower
    return ordered[lower] * (1.0 - fraction) + ordered[upper] * fraction


def parse_one(binary, rank_dir):
    started = time.perf_counter()
    process = subprocess.run(
        [str(binary), "--top-k", "0", str(rank_dir)],
        check=False,
        text=True,
        capture_output=True,
    )
    elapsed = time.perf_counter() - started
    result = {
        "rank_dir": str(rank_dir),
        "returncode": process.returncode,
        "wall_seconds": elapsed,
    }
    if process.returncode == 0:
        summary = json.loads(process.stdout)
        result.update(
            {
                "parse_seconds": summary["parser"]["parse_seconds"],
                "rank": summary["source"]["rank"],
                "segment": summary["source"]["segment"],
                "stars_bytes": summary["coverage"]["stars_bytes"],
                "input_bytes": (
                    summary["coverage"]["stars_bytes"]
                    + summary["coverage"]["ts_track_bytes"]
                    + summary["coverage"]["api_event_bytes"]
                    + summary["coverage"]["task_track_bytes"]
                    + summary["coverage"]["hccl_op_info_bytes"]
                ),
                "matched_device_tasks": summary["coverage"]["matched_device_tasks"],
            }
        )
    else:
        result["stderr"] = process.stderr.strip()
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("segments_dir", type=Path)
    parser.add_argument("--jobs", type=int, default=8)
    parser.add_argument(
        "--binary",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "fast_parse",
    )
    parser.add_argument("--output", type=Path)
    parser.add_argument("--window-interval-seconds", type=float)
    args = parser.parse_args()
    if args.jobs <= 0:
        parser.error("--jobs must be positive")

    segments_dir = args.segments_dir.resolve()
    rank_dirs = sorted(
        path for path in segments_dir.glob("segment_*/rank_*")
        if (path / "profiler").is_dir()
    )
    if not rank_dirs:
        parser.error(f"no segment_*/rank_* inputs under {segments_dir}")

    started = time.perf_counter()
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.jobs) as executor:
        results = list(executor.map(lambda path: parse_one(args.binary.resolve(), path), rank_dirs))
    wall = time.perf_counter() - started
    successes = [item for item in results if item["returncode"] == 0]
    failures = [item for item in results if item["returncode"] != 0]
    durations = [item["wall_seconds"] for item in successes]
    total_bytes = sum(item.get("input_bytes", 0) for item in successes)
    window_latencies = {}
    for item in successes:
        window_latencies[item["segment"]] = max(
            window_latencies.get(item["segment"], 0.0), item["wall_seconds"]
        )
    window_values = list(window_latencies.values())
    window_p99 = percentile(window_values, 0.99)
    report = {
        "schema": "fastanalyse_replay_benchmark_v1",
        "segments_dir": str(segments_dir),
        "jobs": args.jobs,
        "inputs": len(rank_dirs),
        "successes": len(successes),
        "failures": len(failures),
        "wall_seconds": wall,
        "rank_segments_per_second": len(successes) / wall if wall else None,
        "input_mib_per_second": total_bytes / (1024 * 1024) / wall if wall else None,
        "latency_seconds": {
            "min": min(durations) if durations else None,
            "mean": statistics.fmean(durations) if durations else None,
            "p50": percentile(durations, 0.50),
            "p95": percentile(durations, 0.95),
            "p99": percentile(durations, 0.99),
            "max": max(durations) if durations else None,
        },
        "window_latency_seconds": {
            "windows": len(window_values),
            "p50": percentile(window_values, 0.50),
            "p95": percentile(window_values, 0.95),
            "p99": window_p99,
            "max": max(window_values) if window_values else None,
        },
        "window_interval_seconds": args.window_interval_seconds,
        "p99_window_budget_fraction": (
            window_p99 / args.window_interval_seconds
            if window_p99 is not None and args.window_interval_seconds
            else None
        ),
        "results": results,
    }
    rendered = json.dumps(report, ensure_ascii=False, indent=2) + "\n"
    if args.output:
        temporary = args.output.with_name(args.output.name + ".tmp")
        temporary.write_text(rendered, encoding="utf-8")
        temporary.replace(args.output)
    else:
        print(rendered, end="")
    return 0 if not failures else 1


if __name__ == "__main__":
    sys.exit(main())
