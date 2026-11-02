#!/usr/bin/env python3
"""Validate every rank segment with retained official SQLite output."""

import argparse
import concurrent.futures
import json
import math
import statistics
import sys
from pathlib import Path

from validate_against_sqlite import compare


def percentile(values, quantile):
    values = sorted(values)
    if not values:
        return None
    position = quantile * (len(values) - 1)
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return values[lower]
    fraction = position - lower
    return values[lower] * (1.0 - fraction) + values[upper] * fraction


def validate_one(rank_dir, binary):
    try:
        return compare(rank_dir, binary)
    except Exception as error:  # Keep the batch running and report the exact segment.
        return {"rank_dir": str(rank_dir), "ok": False, "error": str(error)}


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
    args = parser.parse_args()
    if args.jobs <= 0:
        parser.error("--jobs must be positive")

    rank_dirs = sorted(
        path.resolve()
        for path in args.segments_dir.resolve().glob("segment_*/rank_*")
        if (path / "analysis" / "profiler").is_dir()
    )
    if not rank_dirs:
        parser.error("no rank segments with retained analysis output found")

    with concurrent.futures.ThreadPoolExecutor(max_workers=args.jobs) as executor:
        results = list(executor.map(lambda path: validate_one(path, args.binary.resolve()), rank_dirs))

    failures = [item for item in results if not item.get("ok")]
    successes = [item for item in results if item.get("ok")]
    parse_times = [item["fast_parse_seconds"] for item in successes]
    name_coverages = [item["compute_name_coverage"] for item in successes]
    report = {
        "schema": "fastanalyse_validation_replay_v1",
        "segments_dir": str(args.segments_dir.resolve()),
        "inputs": len(rank_dirs),
        "passed": len(successes),
        "failed": len(failures),
        "parse_seconds": {
            "mean": statistics.fmean(parse_times) if parse_times else None,
            "p50": percentile(parse_times, 0.50),
            "p95": percentile(parse_times, 0.95),
            "p99": percentile(parse_times, 0.99),
            "max": max(parse_times) if parse_times else None,
        },
        "compute_name_coverage": {
            "min": min(name_coverages) if name_coverages else None,
            "mean": statistics.fmean(name_coverages) if name_coverages else None,
            "max": max(name_coverages) if name_coverages else None,
        },
        "failures": failures,
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
