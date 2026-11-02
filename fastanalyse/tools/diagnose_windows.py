#!/usr/bin/env python3
"""Replay device-step latency diagnostics from one run's fast summaries.

The input is one profile/fast directory. Runs and metric types are never mixed.
Thresholds are experimental until calibrated with longer normal training.
"""

import argparse
import json
import math
import statistics
import sys
from collections import defaultdict, deque
from pathlib import Path


SCHEMA = "fast_diagnosis_replay_v1"
METRIC = "device_step_duration_ns"


def issue(issues, seen, code, segment, rank=None, detail=None):
    key = (code, segment, rank, detail)
    if key not in seen:
        seen.add(key)
        item = {"code": code, "segment": segment}
        if rank is not None:
            item["rank"] = rank
        if detail is not None:
            item["detail"] = detail
        issues.append(item)


def read_json(path):
    with path.open(encoding="utf-8") as stream:
        return json.load(stream)


def valid_rank_steps(path, segment, rank, issues, seen):
    if not path.is_file():
        issue(issues, seen, "MISSING_RANK", segment, rank)
        return None
    try:
        summary = read_json(path)
        source = summary["source"]
        coverage = summary["coverage"]
        steps = summary["steps"]
        first = source["declared_first_step"]
        last = source["declared_last_step"]
        if source["rank"] != rank or source["segment"] != segment:
            raise ValueError("rank or segment does not match filename")
        if not isinstance(first, int) or not isinstance(last, int) or first > last:
            raise ValueError("invalid declared step interval")
        if coverage.get("step_trace_complete") is not True:
            raise ValueError("step_trace_complete is not true")
        if coverage.get("step_pairs") != last - first + 1:
            raise ValueError("step pair count does not match declared interval")
        per_step = {}
        for item in steps:
            step = item["global_step"]
            value = item["duration_ns"]
            if not isinstance(step, int) or isinstance(step, bool) or step in per_step:
                raise ValueError("missing or duplicate global_step")
            if not isinstance(value, (int, float)) or not math.isfinite(value) or value <= 0:
                raise ValueError("invalid duration_ns")
            per_step[step] = float(value)
        if set(per_step) != set(range(first, last + 1)):
            raise ValueError("global_step coverage differs from declared interval")
        return first, last, per_step
    except (OSError, ValueError, TypeError, KeyError, json.JSONDecodeError) as error:
        issue(issues, seen, "INVALID_RANK_TRACE", segment, rank, str(error))
        return None


def load_windows(fast_dir, paths=None, prior_segment=None, prior_last=None):
    paths = sorted(paths if paths is not None else fast_dir.glob("segment_*/window_summary.json"))
    if not paths:
        raise ValueError(f"no segment_*/window_summary.json in {fast_dir}")
    issues, seen, windows = [], set(), []
    for path in paths:
        summary = read_json(path)
        if summary.get("schema") != "fast_window_summary_v1":
            raise ValueError(f"unsupported window schema in {path}")
        segment = summary["segment"]
        if not isinstance(segment, int) or path.parent.name != f"segment_{segment:06d}":
            raise ValueError(f"segment name mismatch in {path}")
        if prior_segment is not None and segment <= prior_segment:
            raise ValueError("segments are duplicated or out of order")
        expected = summary["expected_rank_ids"]
        if not isinstance(expected, list) or not expected or len(expected) != len(set(expected)):
            raise ValueError(f"invalid expected_rank_ids in {path}")
        if any(not isinstance(rank, int) or isinstance(rank, bool) for rank in expected):
            raise ValueError(f"invalid rank ID in {path}")
        if set(summary["rank_ids"]) != set(expected) or summary.get("complete") is not True:
            issue(issues, seen, "INCOMPLETE_WINDOW", segment)
        ranks = {}
        intervals = set()
        for rank in expected:
            loaded = valid_rank_steps(path.parent / f"rank_{rank}.json", segment, rank, issues, seen)
            if loaded is not None:
                first, last, per_step = loaded
                intervals.add((first, last))
                ranks[rank] = per_step
        if len(intervals) > 1:
            issue(issues, seen, "RANK_STEP_INTERVAL_MISMATCH", segment)
            ranks = {}
        declared = min(intervals) if intervals else None
        if declared is None:
            issue(issues, seen, "NO_VALID_STEP_TRACE", segment)
            prior_segment = segment
            continue
        first, last = declared
        if prior_last is not None and first != prior_last + 1:
            issue(issues, seen, "STEP_DISCONTINUITY", segment, detail=f"expected {prior_last + 1}, got {first}")
        if prior_segment is not None and segment != prior_segment + 1:
            issue(issues, seen, "SEGMENT_DISCONTINUITY", segment, detail=f"previous segment {prior_segment}")

        # The cross-rank summary must agree with the authoritative rank traces.
        comparisons = summary.get("step_comparisons", [])
        by_step = {}
        try:
            for comparison in comparisons:
                step = comparison["global_step"]
                if not isinstance(step, int) or step in by_step:
                    raise ValueError("missing or duplicate window global_step")
                by_step[step] = comparison["per_rank_ns"]
            if set(by_step) != set(range(first, last + 1)):
                raise ValueError("window global_step coverage differs from rank traces")
            for rank, values in ranks.items():
                for step, value in values.items():
                    compared = by_step[step].get(str(rank))
                    if compared is None or not math.isclose(value, compared, rel_tol=1e-9, abs_tol=1e-3):
                        raise ValueError(f"rank {rank}, step {step} duration mismatch")
        except (KeyError, TypeError, ValueError) as error:
            issue(issues, seen, "WINDOW_RANK_MISMATCH", segment, detail=str(error))
            ranks = {}

        windows.append({
            "segment": segment,
            "first_step": first,
            "last_step": last,
            "expected_ranks": sorted(expected),
            "ranks": ranks,
            "gap_before": prior_last is not None and first != prior_last + 1,
        })
        prior_segment, prior_last = segment, last
    return windows, issues, seen


def parse_peer_groups(value):
    if not value:
        return []
    groups, used = [], set()
    for part in value.split(";"):
        try:
            group = [int(rank) for rank in part.split(",")]
        except ValueError as error:
            raise ValueError("peer groups must be semicolon-separated comma-separated rank IDs") from error
        if len(group) < 3 or len(set(group)) != len(group) or used.intersection(group):
            raise ValueError("peer groups need at least three unique ranks and cannot overlap")
        groups.append(sorted(group))
        used.update(group)
    return groups


def exceeds(value, baseline, relative, absolute_ns):
    return value - baseline >= absolute_ns and value / baseline - 1 >= relative


def robust_z(value, history):
    median = statistics.median(history)
    mad = statistics.median(abs(item - median) for item in history)
    return (value - median) / max(1_000_000.0, 1.4826 * mad)


def gradual_drift(values, relative, absolute_ns):
    """Prefer a sustained slope over a one-time step change in 12 observations."""
    blocks = [statistics.median(values[i:i + 4]) for i in (0, 4, 8)]
    rise1, rise2 = blocks[1] - blocks[0], blocks[2] - blocks[1]
    if (not exceeds(blocks[2], blocks[0], relative, absolute_ns)
            or min(rise1, rise2) < max(absolute_ns / 2, (blocks[2] - blocks[0]) / 5)):
        return False
    center = (len(values) - 1) / 2
    mean = statistics.mean(values)
    slope = sum((index - center) * (value - mean) for index, value in enumerate(values)) / sum(
        (index - center) ** 2 for index in range(len(values)))
    line_error = sum((value - (mean + slope * (index - center))) ** 2
                     for index, value in enumerate(values))
    step_error = min(
        sum((value - statistics.mean(values[:cut])) ** 2 for value in values[:cut])
        + sum((value - statistics.mean(values[cut:])) ** 2 for value in values[cut:])
        for cut in range(3, len(values) - 2)
    )
    return line_error < 0.75 * step_error


def finding(method, kind, segment, step, rank, observed, baseline, confidence, **extra):
    result = {
        "method": method,
        "kind": kind,
        "segment": segment,
        "global_step": step,
        "rank": rank,
        "metric": METRIC,
        "observed_ns": observed,
        "baseline_ns": baseline,
        "absolute_delta_ns": observed - baseline,
        "relative_delta": observed / baseline - 1,
        "confidence": confidence,
    }
    result.update(extra)
    return result


def replay(fast_dir, peer_groups=None, history_size=12, min_history=8,
           relative_threshold=0.10, absolute_ms=250.0, robust_z_threshold=5.0,
           windows=None, issues=None, initial_state=None, return_state=False):
    if history_size < 12 or min_history < 4 or min_history > history_size:
        raise ValueError("history_size must be >= 12 and min_history must be within 4..history_size")
    if (not all(math.isfinite(value) and value > 0 for value in
                (relative_threshold, absolute_ms, robust_z_threshold))):
        raise ValueError("thresholds must be positive finite numbers")
    fast_dir = Path(fast_dir).resolve()
    if windows is None:
        windows, issues, seen = load_windows(fast_dir)
    else:
        issues = list(issues or [])
        seen = {(item["code"], item["segment"], item.get("rank"), item.get("detail"))
                for item in issues}
    peer_groups = peer_groups or []
    expected_sets = {tuple(window["expected_ranks"]) for window in windows}
    if len(expected_sets) > 1:
        raise ValueError("expected rank set changed across windows; replay runs separately")
    prior_expected = tuple((initial_state or {}).get("expected_rank_ids", ()))
    expected = set(next(iter(expected_sets))) if expected_sets else set(prior_expected)
    if prior_expected and expected != set(prior_expected):
        raise ValueError("expected rank set changed across incremental windows")
    if any(rank not in expected for group in peer_groups for rank in group):
        raise ValueError("peer group includes a rank outside expected_rank_ids")
    absolute_ns = absolute_ms * 1_000_000.0
    initial_state = initial_state or {}
    histories = defaultdict(lambda: deque(maxlen=history_size), {
        int(rank): deque(values, maxlen=history_size)
        for rank, values in initial_state.get("histories", {}).items()
    })
    slow_streak = defaultdict(int, {int(rank): value for rank, value in
                                    initial_state.get("slow_streak", {}).items()})
    trend_pending = defaultdict(int, {int(rank): value for rank, value in
                                      initial_state.get("trend_pending", {}).items()})
    trend_pending_kind = {int(rank): value for rank, value in
                          initial_state.get("trend_pending_kind", {}).items()}
    trend_active = {int(rank): value for rank, value in initial_state.get("trend_active", {}).items()}
    findings = []
    step_count = valid_rank_steps = expected_rank_steps = 0
    prior_last = initial_state.get("prior_last")
    for window in windows:
        segment = window["segment"]
        if prior_last is not None and window["first_step"] != prior_last + 1:
            histories.clear()
            slow_streak.clear()
            trend_pending.clear()
            trend_pending_kind.clear()
            trend_active.clear()
        for missing_rank in expected - set(window["ranks"]):
            histories.pop(missing_rank, None)
            slow_streak.pop(missing_rank, None)
            trend_pending.pop(missing_rank, None)
            trend_pending_kind.pop(missing_rank, None)
            trend_active.pop(missing_rank, None)
        prior_last = window["last_step"]
        expected_rank_steps += len(expected) * (window["last_step"] - window["first_step"] + 1)
        valid_rank_steps += sum(len(values) for values in window["ranks"].values())
        for step in range(window["first_step"], window["last_step"] + 1):
            step_count += 1
            per_rank = {rank: values[step] for rank, values in window["ranks"].items()}
            boundary = step in (window["first_step"], window["last_step"])
            for rank, value in sorted(per_rank.items()):
                history = list(histories[rank])
                if len(history) >= min_history:
                    baseline = statistics.median(history)
                    z = robust_z(value, history)
                    if exceeds(value, baseline, relative_threshold, absolute_ns) and z >= robust_z_threshold:
                        slow_streak[rank] += 1
                        findings.append(finding(
                            "iteration_latency", "step_latency_spike", segment, step, rank,
                            value, baseline, "low" if boundary else "medium",
                            robust_z=z, consecutive_slow_steps=slow_streak[rank],
                            boundary_step=boundary, history_points=len(history),
                        ))
                    else:
                        slow_streak[rank] = 0
                if not boundary and len(history) >= history_size:
                    recent = history[-3:] + [value]
                    long_baseline = statistics.median(history[:-4])
                    recent_median = statistics.median(recent)
                    kind = None
                    if exceeds(recent_median, long_baseline, relative_threshold, absolute_ns):
                        values12 = history[-11:] + [value]
                        if gradual_drift(values12, relative_threshold, absolute_ns):
                            kind = "drift"
                        else:
                            kind = "regression"
                    if kind:
                        trend_pending[rank] = trend_pending[rank] + 1 if trend_pending_kind.get(rank) == kind else 1
                        trend_pending_kind[rank] = kind
                        if trend_pending[rank] >= 2 and trend_active.get(rank) != kind:
                            findings.append(finding(
                                "rolling_trend", kind, segment, step, rank,
                                recent_median, long_baseline, "medium",
                                recent_steps=4, baseline_steps=len(history[:-4]),
                                consecutive_windows=trend_pending[rank],
                            ))
                            trend_active[rank] = kind
                    else:
                        trend_pending[rank] = 0
                        trend_pending_kind.pop(rank, None)
                        trend_active.pop(rank, None)
                if not boundary:
                    histories[rank].append(value)
            for group in peer_groups:
                if not set(group).issubset(per_rank):
                    issue(issues, seen, "PEER_GROUP_INCOMPLETE", segment, detail=",".join(map(str, group)))
                    continue
                for rank in group:
                    peers = [per_rank[other] for other in group if other != rank]
                    baseline = statistics.median(peers)
                    value = per_rank[rank]
                    if exceeds(value, baseline, relative_threshold, absolute_ns):
                        findings.append(finding(
                            "peer_rank_latency", "same_step_rank_outlier", segment, step,
                            rank, value, baseline, "low" if boundary else "medium",
                            peer_group=group, peer_count=len(peers), boundary_step=boundary,
                        ))
    report = {
        "schema": SCHEMA,
        "source_fast_dir": str(fast_dir),
        "metric": METRIC,
        "scope": "one_run_offline_replay",
        "threshold_status": "experimental_uncalibrated",
        "settings": {
            "peer_groups": peer_groups,
            "history_size": history_size,
            "min_history": min_history,
            "relative_threshold": relative_threshold,
            "absolute_ms": absolute_ms,
            "robust_z_threshold": robust_z_threshold,
            "exclude_window_boundary_from_history": True,
        },
        "coverage": {
            "windows": len(windows),
            "global_steps": step_count,
            "valid_rank_steps": valid_rank_steps,
            "expected_rank_steps": expected_rank_steps,
            "quality_issue_count": len(issues),
        },
        "quality_issues": issues,
        "findings": findings,
        "finding_counts": {method: sum(item["method"] == method for item in findings)
                           for method in ("iteration_latency", "peer_rank_latency", "rolling_trend")},
        "unavailable_signals": ["host_sink_batch_duration", "progress_heartbeat"],
    }
    if not return_state:
        return report
    state = {
        "schema": "fast_diagnosis_state_v1",
        "expected_rank_ids": sorted(expected),
        "prior_last": prior_last,
        "histories": {str(rank): list(values) for rank, values in histories.items()},
        "slow_streak": {str(rank): value for rank, value in slow_streak.items()},
        "trend_pending": {str(rank): value for rank, value in trend_pending.items()},
        "trend_pending_kind": {str(rank): value for rank, value in trend_pending_kind.items()},
        "trend_active": {str(rank): value for rank, value in trend_active.items()},
    }
    return report, state


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("fast_dir", type=Path, help="one run's profile/fast directory")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--peer-groups", default="", help="same-role ranks: 0,1,2;3,4,5")
    parser.add_argument("--history-size", type=int, default=12)
    parser.add_argument("--min-history", type=int, default=8)
    parser.add_argument("--relative-threshold", type=float, default=0.10)
    parser.add_argument("--absolute-ms", type=float, default=250.0)
    parser.add_argument("--robust-z-threshold", type=float, default=5.0)
    args = parser.parse_args()
    try:
        groups = parse_peer_groups(args.peer_groups)
        result = replay(args.fast_dir, groups, args.history_size, args.min_history,
                        args.relative_threshold, args.absolute_ms, args.robust_z_threshold)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        temporary = args.output.with_name(args.output.name + ".tmp")
        temporary.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        temporary.replace(args.output)
    except (OSError, ValueError, KeyError, TypeError, json.JSONDecodeError) as error:
        print(f"diagnose_windows: {error}", file=sys.stderr)
        return 2
    print(json.dumps({"output": str(args.output), "coverage": result["coverage"],
                      "finding_counts": result["finding_counts"]}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
