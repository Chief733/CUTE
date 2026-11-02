#!/usr/bin/env python3
"""Aggregate already generated rank fast summaries for one window."""

import argparse
import json
import sys
from pathlib import Path
import re

from parse_window import aggregate_summaries, write_atomic


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("summary_dir", type=Path)
    expected = parser.add_mutually_exclusive_group(required=True)
    expected.add_argument("--expected-ranks", type=int)
    expected.add_argument("--expected-rank-ids")
    parser.add_argument("--ratio-threshold", type=float, default=1.2)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.expected_ranks is not None and args.expected_ranks <= 0:
        parser.error("--expected-ranks must be positive")
    expected_rank_ids = None
    if args.expected_rank_ids is not None:
        try:
            expected_rank_ids = [int(value) for value in args.expected_rank_ids.split(",")]
        except ValueError:
            parser.error("--expected-rank-ids must be comma-separated integers")
        if not expected_rank_ids or len(set(expected_rank_ids)) != len(expected_rank_ids):
            parser.error("--expected-rank-ids must be a non-empty unique list")

    paths = sorted(
        path
        for path in args.summary_dir.glob("rank_*.json")
        if re.fullmatch(r"rank_\d+\.json", path.name)
    )
    if not paths:
        parser.error("no rank_*.json summaries found")
    try:
        summaries = [json.loads(path.read_text()) for path in paths]
        result = aggregate_summaries(
            summaries,
            ratio_threshold=args.ratio_threshold,
            expected_ranks=args.expected_ranks,
            expected_rank_ids=expected_rank_ids,
        )
    except (OSError, ValueError, KeyError, json.JSONDecodeError) as error:
        print(f"aggregate_files: {error}", file=sys.stderr)
        return 2
    output = args.output or args.summary_dir / "window_summary.json"
    write_atomic(output, json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    return 0 if result["complete"] else 1


if __name__ == "__main__":
    sys.exit(main())
