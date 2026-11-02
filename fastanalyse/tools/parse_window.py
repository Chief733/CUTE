#!/usr/bin/env python3
"""Parse every rank in one sealed segment and build a cross-rank summary."""

import argparse
import concurrent.futures
import hashlib
import json
import math
import statistics
import subprocess
import sys
import time
from pathlib import Path


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


def parse_rank(binary, rank_dir, top_k):
    started = time.perf_counter()
    process = subprocess.run(
        [str(binary), "--top-k", str(top_k), str(rank_dir)],
        check=False,
        text=True,
        capture_output=True,
    )
    wall_seconds = time.perf_counter() - started
    if process.returncode != 0:
        return {
            "rank_dir": str(rank_dir),
            "returncode": process.returncode,
            "wall_seconds": wall_seconds,
            "stderr": process.stderr.strip(),
        }
    try:
        summary = json.loads(process.stdout)
    except json.JSONDecodeError as error:
        return {
            "rank_dir": str(rank_dir),
            "returncode": 3,
            "wall_seconds": wall_seconds,
            "stderr": f"invalid fast_parse JSON: {error}",
        }
    return {
        "rank_dir": str(rank_dir),
        "returncode": 0,
        "wall_seconds": wall_seconds,
        "summary": summary,
    }


def stable_structure_hash(summary):
    structure = sorted(
        (
            item["name"],
            item["name_source"],
            item["host_task_type"],
            item["stream_id"],
            item["device_task_type"],
            item["count"],
        )
        for item in summary["kernels"]
    )
    encoded = json.dumps(structure, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def aggregate_summaries(
    summaries,
    ratio_threshold=1.2,
    expected_ranks=None,
    expected_rank_ids=None,
    wall_seconds=None,
):
    summaries = sorted(summaries, key=lambda item: item["source"]["rank"])
    rank_ids = [item["source"]["rank"] for item in summaries]
    if len(rank_ids) != len(set(rank_ids)):
        raise ValueError("duplicate rank IDs")
    segments = {item["source"]["segment"] for item in summaries}
    if len(segments) != 1:
        raise ValueError(f"mixed segments: {sorted(segments)}")

    if expected_ranks is not None and expected_rank_ids is not None:
        raise ValueError("expected_ranks and expected_rank_ids are mutually exclusive")
    if expected_rank_ids is not None:
        expected_rank_ids = set(expected_rank_ids)
    elif expected_ranks is not None:
        expected_rank_ids = set(range(expected_ranks))
    else:
        expected_rank_ids = set(rank_ids)
    if not expected_rank_ids:
        raise ValueError("expected rank set is empty")
    complete = set(rank_ids) == expected_rank_ids
    rank_overview = []
    for summary in summaries:
        coverage = summary["coverage"]
        rank_overview.append(
            {
                "rank": summary["source"]["rank"],
                "parse_seconds": summary["parser"]["parse_seconds"],
                "steps": len(summary["steps"]),
                "matched_device_tasks": coverage["matched_device_tasks"],
                "named_device_tasks": coverage["named_device_tasks"],
                "compute_name_coverage": coverage["compute_name_coverage"],
                "structure_hash": stable_structure_hash(summary),
            }
        )

    step_groups = {}
    for summary in summaries:
        rank = summary["source"]["rank"]
        for step in summary["steps"]:
            global_step = step.get("global_step")
            if global_step is None:
                global_step = step["index_id"]
            step_groups.setdefault(global_step, {})[rank] = step["duration_ns"]
    step_comparisons = []
    for index_id, per_rank in sorted(step_groups.items()):
        values = list(per_rank.values())
        median = statistics.median(values)
        maximum = max(values)
        minimum = min(values)
        max_ratio = maximum / median if median else None
        outliers = sorted(
            rank
            for rank, value in per_rank.items()
            if median and max(value / median, median / value if value else float("inf")) >= ratio_threshold
        )
        step_comparisons.append(
            {
                "global_step": index_id,
                "index_id": index_id,
                "rank_count": len(per_rank),
                "median_ns": median,
                "min_ns": minimum,
                "max_ns": maximum,
                "max_to_median_ratio": max_ratio,
                "outlier_ranks": outliers,
                "per_rank_ns": {str(rank): value for rank, value in sorted(per_rank.items())},
            }
        )

    kernel_groups = {}
    for summary in summaries:
        rank = summary["source"]["rank"]
        for item in summary["kernels"]:
            key = (
                item["name"],
                item["name_source"],
                item["host_task_type"],
                item["stream_id"],
                item["device_task_type"],
            )
            kernel_groups.setdefault(key, {})[rank] = {
                "count": item["count"],
                "total_ns": item["total_ns"],
                "p50_ns": item["p50_ns"],
                "p95_ns": item["p95_ns"],
                "p99_ns": item["p99_ns"],
                "max_ns": item["max_ns"],
            }

    kernel_comparisons = []
    anomaly_count = 0
    for key, per_rank in sorted(kernel_groups.items()):
        totals = [item["total_ns"] for item in per_rank.values()]
        median_total = statistics.median(totals)
        counts = {item["count"] for item in per_rank.values()}
        missing_ranks = sorted(expected_rank_ids - set(per_rank))
        outlier_ranks = sorted(
            rank
            for rank, item in per_rank.items()
            if median_total
            and max(
                item["total_ns"] / median_total,
                median_total / item["total_ns"] if item["total_ns"] else float("inf"),
            )
            >= ratio_threshold
        )
        anomalous = bool(missing_ranks or len(counts) != 1 or outlier_ranks)
        anomaly_count += int(anomalous)
        kernel_comparisons.append(
            {
                "name": key[0],
                "name_source": key[1],
                "host_task_type": key[2],
                "stream_id": key[3],
                "device_task_type": key[4],
                "rank_count": len(per_rank),
                "missing_ranks": missing_ranks,
                "count_mismatch": len(counts) != 1,
                "median_total_ns": median_total,
                "max_to_median_ratio": max(totals) / median_total if median_total else None,
                "outlier_ranks": outlier_ranks,
                "anomalous": anomalous,
                "per_rank": {str(rank): value for rank, value in sorted(per_rank.items())},
            }
        )

    hccl_groups = {}
    for summary in summaries:
        rank = summary["source"]["rank"]
        for item in summary.get("hccl_ops", []):
            key = (
                item["op_name"],
                item["data_type"],
                item["algorithm"],
                item["group_hash"],
                item["element_count"],
                item["relay"],
                item["retry"],
            )
            hccl_groups.setdefault(key, {})[rank] = {
                "count": item["count"],
                "host_launch_total_ns": item["host_launch_total_ns"],
                "host_launch_p99_ns": item["host_launch_p99_ns"],
                "host_launch_max_ns": item["host_launch_max_ns"],
            }

    hccl_comparisons = []
    hccl_anomaly_count = 0
    for key, per_rank in sorted(hccl_groups.items()):
        present_totals = [
            item["host_launch_total_ns"]
            for item in per_rank.values()
            if item["host_launch_total_ns"] is not None
        ]
        median_total = statistics.median(present_totals) if present_totals else None
        counts = {item["count"] for item in per_rank.values()}
        missing_ranks = sorted(expected_rank_ids - set(per_rank))
        outlier_ranks = sorted(
            rank
            for rank, item in per_rank.items()
            if median_total
            and item["host_launch_total_ns"] is not None
            and max(
                item["host_launch_total_ns"] / median_total,
                median_total / item["host_launch_total_ns"]
                if item["host_launch_total_ns"]
                else float("inf"),
            )
            >= ratio_threshold
        )
        anomalous = bool(missing_ranks or len(counts) != 1 or outlier_ranks)
        hccl_anomaly_count += int(anomalous)
        hccl_comparisons.append(
            {
                "op_name": key[0],
                "data_type": key[1],
                "algorithm": key[2],
                "group_hash": key[3],
                "element_count": key[4],
                "relay": key[5],
                "retry": key[6],
                "rank_count": len(per_rank),
                "missing_ranks": missing_ranks,
                "count_mismatch": len(counts) != 1,
                "median_host_launch_total_ns": median_total,
                "outlier_ranks": outlier_ranks,
                "anomalous": anomalous,
                "per_rank": {str(rank): value for rank, value in sorted(per_rank.items())},
            }
        )

    parse_times = [item["parser"]["parse_seconds"] for item in summaries]
    return {
        "schema": "fast_window_summary_v1",
        "segment": next(iter(segments)),
        "complete": complete,
        "expected_ranks": len(expected_rank_ids),
        "expected_rank_ids": sorted(expected_rank_ids),
        "rank_ids": rank_ids,
        "wall_seconds": wall_seconds,
        "parse_seconds": {
            "p50": percentile(parse_times, 0.50),
            "p95": percentile(parse_times, 0.95),
            "p99": percentile(parse_times, 0.99),
            "max": max(parse_times) if parse_times else None,
        },
        "ratio_threshold": ratio_threshold,
        "rank_overview": rank_overview,
        "step_comparisons": step_comparisons,
        "kernel_group_count": len(kernel_comparisons),
        "kernel_anomaly_count": anomaly_count,
        "kernel_comparisons": kernel_comparisons,
        "hccl_group_count": len(hccl_comparisons),
        "hccl_anomaly_count": hccl_anomaly_count,
        "hccl_comparisons": hccl_comparisons,
    }


def write_atomic(path, contents):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(contents, encoding="utf-8")
    temporary.replace(path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("segment_dir", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--jobs", type=int, default=8)
    parser.add_argument("--expected-ranks", type=int)
    parser.add_argument(
        "--expected-rank-ids",
        help="comma-separated explicit rank IDs (mutually exclusive with --expected-ranks)",
    )
    parser.add_argument("--ratio-threshold", type=float, default=1.2)
    parser.add_argument("--top-k", type=int, default=20)
    parser.add_argument(
        "--binary",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "fast_parse",
    )
    args = parser.parse_args()
    if args.jobs <= 0 or args.top_k < 0:
        parser.error("--jobs must be positive and --top-k must be non-negative")
    if args.ratio_threshold <= 1.0:
        parser.error("--ratio-threshold must be greater than 1")

    rank_dirs = sorted(
        path.resolve()
        for path in args.segment_dir.resolve().glob("rank_*")
        if (path / "profiler").is_dir()
    )
    if not rank_dirs:
        parser.error("no rank_* profiler inputs found")
    if args.expected_ranks is not None and args.expected_rank_ids is not None:
        parser.error("--expected-ranks and --expected-rank-ids are mutually exclusive")
    expected_rank_ids = None
    if args.expected_rank_ids is not None:
        try:
            expected_rank_ids = [int(value) for value in args.expected_rank_ids.split(",")]
        except ValueError:
            parser.error("--expected-rank-ids must be comma-separated integers")
        if not expected_rank_ids or len(set(expected_rank_ids)) != len(expected_rank_ids):
            parser.error("--expected-rank-ids must be a non-empty unique list")
    expected_ranks = args.expected_ranks
    if expected_ranks is None and expected_rank_ids is None:
        expected_ranks = len(rank_dirs)

    started = time.perf_counter()
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.jobs) as executor:
        results = list(
            executor.map(lambda path: parse_rank(args.binary.resolve(), path, args.top_k), rank_dirs)
        )
    failures = [item for item in results if item["returncode"] != 0]
    if failures:
        report = {"schema": "fast_window_failure_v1", "failures": failures}
        write_atomic(args.output_dir / "window_failure.json", json.dumps(report, indent=2) + "\n")
        print(json.dumps(report, ensure_ascii=False, indent=2), file=sys.stderr)
        return 1

    summaries = [item["summary"] for item in results]
    wall_seconds = time.perf_counter() - started
    window_summary = aggregate_summaries(
        summaries,
        ratio_threshold=args.ratio_threshold,
        expected_ranks=expected_ranks,
        expected_rank_ids=expected_rank_ids,
        wall_seconds=wall_seconds,
    )
    output_dir = args.output_dir.resolve()
    for summary in summaries:
        rank = summary["source"]["rank"]
        write_atomic(
            output_dir / f"rank_{rank}.json",
            json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        )
    write_atomic(
        output_dir / "window_summary.json",
        json.dumps(window_summary, ensure_ascii=False, indent=2) + "\n",
    )
    print(json.dumps({
        "ok": True,
        "segment": window_summary["segment"],
        "ranks": len(summaries),
        "wall_seconds": wall_seconds,
        "output": str(output_dir / "window_summary.json"),
    }, ensure_ascii=False))
    return 0 if window_summary["complete"] else 2


if __name__ == "__main__":
    sys.exit(main())
