#!/usr/bin/env python3
"""Validate one fast_parse result against CANN-generated SQLite files."""

import argparse
import json
import sqlite3
import subprocess
import sys
from pathlib import Path


def one_path(paths, description):
    paths = list(paths)
    if len(paths) != 1:
        raise RuntimeError(f"expected one {description}, found {len(paths)}")
    return paths[0]


def compare(rank_dir, binary):
    process = subprocess.run(
        [str(binary), "--top-k", "0", str(rank_dir)],
        check=True,
        text=True,
        capture_output=True,
    )
    summary = json.loads(process.stdout)

    analysis_profiler = rank_dir / "analysis" / "profiler"
    prof_dir = one_path(analysis_profiler.glob("PROF_*"), "analysis PROF_* directory")
    device_dir = one_path(prof_dir.glob("device_*"), "analysis device directory")
    step_db = device_dir / "sqlite" / "step_trace.db"
    task_db = device_dir / "sqlite" / "ascend_task.db"
    kernel_db = device_dir / "sqlite" / "ai_core_op_summary.db"
    ge_db = prof_dir / "host" / "sqlite" / "ge_info.db"
    hccl_db = prof_dir / "host" / "sqlite" / "hccl.db"
    if (
        not step_db.is_file()
        or not task_db.is_file()
        or not kernel_db.is_file()
        or not ge_db.is_file()
        or not hccl_db.is_file()
    ):
        raise RuntimeError("missing official step/task/kernel/ge SQLite database")

    failures = []
    with sqlite3.connect(step_db) as connection:
        official_steps = connection.execute(
            "SELECT index_id, model_id, step_start, step_end "
            "FROM StepTime ORDER BY step_start"
        ).fetchall()
    actual_steps = [
        (item["index_id"], item["model_id"], item["start_syscnt"], item["end_syscnt"])
        for item in summary["steps"]
    ]
    if actual_steps != official_steps:
        failures.append(
            {"check": "steps", "actual": actual_steps, "official": official_steps}
        )

    actual_types = {item["task_type"]: item for item in summary["device_task_types"]}
    placeholders = ",".join("?" for _ in actual_types)
    query = (
        "SELECT device_task_type, COUNT(*), SUM(duration), MIN(duration), MAX(duration) "
        f"FROM AscendTask WHERE device_task_type IN ({placeholders}) "
        "GROUP BY device_task_type"
    )
    with sqlite3.connect(task_db) as connection:
        rows = connection.execute(query, tuple(actual_types)).fetchall()
    official_types = {
        name: {"count": count, "total_ns": total, "min_ns": minimum, "max_ns": maximum}
        for name, count, total, minimum, maximum in rows
    }

    type_checks = {}
    for name, actual in actual_types.items():
        official = official_types.get(name)
        if official is None:
            failures.append({"check": "missing_official_task_type", "task_type": name})
            continue
        count_ok = actual["count"] == official["count"]
        # CANN rounds each task to integer ns; fast_parse aggregates raw ticks first.
        total_tolerance_ns = max(3.0, official["count"] * 2.0)
        total_error_ns = abs(actual["total_ns"] - official["total_ns"])
        min_error_ns = abs(actual["min_ns"] - official["min_ns"])
        max_error_ns = abs(actual["max_ns"] - official["max_ns"])
        ok = count_ok and total_error_ns <= total_tolerance_ns and min_error_ns <= 3.0 and max_error_ns <= 3.0
        type_checks[name] = {
            "ok": ok,
            "count": actual["count"],
            "official_count": official["count"],
            "total_error_ns": total_error_ns,
            "min_error_ns": min_error_ns,
            "max_error_ns": max_error_ns,
        }
        if not ok:
            failures.append({"check": "task_type", "task_type": name, **type_checks[name]})

    actual_kernel_groups = {}
    for item in summary["kernels"]:
        key = (item["name"], item["stream_id"])
        group = actual_kernel_groups.setdefault(
            key,
            {"count": 0, "total_ns": 0.0, "min_ns": float("inf"), "max_ns": float("-inf")},
        )
        group["count"] += item["count"]
        group["total_ns"] += item["total_ns"]
        group["min_ns"] = min(group["min_ns"], item["min_ns"])
        group["max_ns"] = max(group["max_ns"], item["max_ns"])

    with sqlite3.connect(ge_db) as connection:
        official_host_kernel_count = connection.execute("SELECT COUNT(*) FROM TaskInfo").fetchone()[0]
        official_host_name_counts = dict(
            connection.execute("SELECT op_name, COUNT(*) FROM TaskInfo GROUP BY op_name")
        )
    actual_name_counts = {}
    for item in summary["kernels"]:
        actual_name_counts[item["name"]] = actual_name_counts.get(item["name"], 0) + item["count"]
    if actual_name_counts != official_host_name_counts:
        failures.append(
            {
                "check": "kernel_name_counts",
                "actual_names": len(actual_name_counts),
                "official_names": len(official_host_name_counts),
            }
        )

    with sqlite3.connect(kernel_db) as connection:
        official_kernel_rows = connection.execute(
            "SELECT g.op_name, g.stream_id, COUNT(*), SUM(t.duration_time), "
            "MIN(t.duration_time), MAX(t.duration_time) "
            "FROM ge_summary AS g JOIN task_time AS t "
            "ON t.stream_id=g.stream_id AND t.task_id=g.task_id AND t.batch_id=g.batch_id "
            "AND t.task_type IN ('AI_CORE', 'AI_CPU') "
            "GROUP BY g.op_name, g.stream_id"
        ).fetchall()
    official_kernel_groups = {
        (name, stream): {"count": count, "total_ns": total, "min_ns": minimum, "max_ns": maximum}
        for name, stream, count, total, minimum, maximum in official_kernel_rows
    }
    kernel_group_failures = []
    if set(actual_kernel_groups) != set(official_kernel_groups):
        kernel_group_failures.append(
            {
                "reason": "group_keys",
                "missing": len(set(official_kernel_groups) - set(actual_kernel_groups)),
                "extra": len(set(actual_kernel_groups) - set(official_kernel_groups)),
            }
        )
    for key in set(actual_kernel_groups) & set(official_kernel_groups):
        actual = actual_kernel_groups[key]
        official = official_kernel_groups[key]
        total_error = abs(actual["total_ns"] - official["total_ns"])
        min_error = abs(actual["min_ns"] - official["min_ns"])
        max_error = abs(actual["max_ns"] - official["max_ns"])
        if (
            actual["count"] != official["count"]
            or total_error > max(3.0, official["count"] * 2.0)
            or min_error > 3.0
            or max_error > 3.0
        ):
            kernel_group_failures.append(
                {"name": key[0], "stream_id": key[1], "count": actual["count"],
                 "official_count": official["count"], "total_error_ns": total_error,
                 "min_error_ns": min_error, "max_error_ns": max_error}
            )
    if kernel_group_failures:
        failures.append(
            {"check": "kernel_duration_groups", "failures": kernel_group_failures[:20]}
        )

    actual_hccl_groups = {}
    for item in summary.get("hccl_ops", []):
        key = (
            item["op_name"],
            item["data_type"],
            item["algorithm"],
            int(item["group_hash"]),
            item["element_count"],
            int(item["relay"]),
            int(item["retry"]),
        )
        actual_hccl_groups[key] = {
            "count": item["count"],
            "total_ns": item["host_launch_total_ns"],
            "min_ns": item["host_launch_min_ns"],
            "max_ns": item["host_launch_max_ns"],
        }
    official_hccl_groups = {}
    with sqlite3.connect(hccl_db) as connection:
        hccl_rows = connection.execute(
            "SELECT op_name, data_type, alg_type, group_name, count, relay, retry, begin, end FROM HCCLOP"
        ).fetchall()
    for op_name, data_type, algorithm, group_hash, elements, relay, retry, begin, end in hccl_rows:
        key = (op_name, data_type, algorithm, int(group_hash), elements, relay, retry)
        duration = end - begin
        group = official_hccl_groups.setdefault(
            key, {"count": 0, "total_ns": 0.0, "min_ns": float("inf"), "max_ns": float("-inf")}
        )
        group["count"] += 1
        group["total_ns"] += duration
        group["min_ns"] = min(group["min_ns"], duration)
        group["max_ns"] = max(group["max_ns"], duration)
    hccl_failures = []
    if set(actual_hccl_groups) != set(official_hccl_groups):
        hccl_failures.append(
            {
                "reason": "group_keys",
                "missing": len(set(official_hccl_groups) - set(actual_hccl_groups)),
                "extra": len(set(actual_hccl_groups) - set(official_hccl_groups)),
            }
        )
    for key in set(actual_hccl_groups) & set(official_hccl_groups):
        actual = actual_hccl_groups[key]
        official = official_hccl_groups[key]
        if (
            actual["count"] != official["count"]
            or abs(actual["total_ns"] - official["total_ns"]) > max(3.0, official["count"] * 2.0)
            or abs(actual["min_ns"] - official["min_ns"]) > 3.0
            or abs(actual["max_ns"] - official["max_ns"]) > 3.0
        ):
            hccl_failures.append(
                {"op_name": key[0], "data_type": key[1], "algorithm": key[2],
                 "count": actual["count"], "official_count": official["count"]}
            )
    if hccl_failures:
        failures.append({"check": "hccl_op_groups", "failures": hccl_failures[:20]})

    return {
        "rank_dir": str(rank_dir),
        "ok": not failures,
        "fast_parse_seconds": summary["parser"]["parse_seconds"],
        "step_pairs": len(actual_steps),
        "matched_device_tasks": summary["coverage"]["matched_device_tasks"],
        "named_device_tasks": summary["coverage"]["named_device_tasks"],
        "compute_name_coverage": summary["coverage"]["compute_name_coverage"],
        "official_host_kernel_count": official_host_kernel_count,
        "kernel_groups": len(actual_kernel_groups),
        "hccl_ops": summary["coverage"]["hccl_op_records"],
        "hccl_groups": len(actual_hccl_groups),
        "task_types": type_checks,
        "failures": failures,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("rank_dir", type=Path)
    parser.add_argument(
        "--binary",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "fast_parse",
    )
    args = parser.parse_args()
    try:
        report = compare(args.rank_dir.resolve(), args.binary.resolve())
    except (OSError, RuntimeError, subprocess.SubprocessError, sqlite3.Error, json.JSONDecodeError) as error:
        print(json.dumps({"ok": False, "error": str(error)}, ensure_ascii=False, indent=2))
        return 2
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0 if report["ok"] else 1


if __name__ == "__main__":
    sys.exit(main())
