"""Build a compact dashboard model from sidecar outputs without MindSpore."""

import json
import statistics
import time
from collections import defaultdict
from pathlib import Path


def _read_json(path):
    try:
        return json.loads(Path(path).read_text())
    except (OSError, ValueError, TypeError):
        return None


def _rolling_baseline(rows, value_key, warmup_key=None, size=12, minimum=8):
    history = []
    for row in rows:
        if warmup_key and row.get(warmup_key):
            row["baseline_ms"] = None
            row["threshold_ms"] = None
            continue
        if len(history) >= minimum:
            baseline = statistics.median(history[-size:])
            row["baseline_ms"] = baseline
            row["threshold_ms"] = max(baseline * 1.10, baseline + 250.0)
        else:
            row["baseline_ms"] = None
            row["threshold_ms"] = None
        history.append(row[value_key])


def _device_steps(run):
    by_step = {}
    for path in sorted((run / "fast").glob("segment_*/window_summary.json")):
        summary = _read_json(path)
        if not summary or not summary.get("complete"):
            continue
        for item in summary.get("step_comparisons", ()):
            values = {int(rank): float(value) / 1e6
                      for rank, value in item.get("per_rank_ns", {}).items()}
            if not values:
                continue
            samples = list(values.values())
            median = statistics.median(samples)
            by_step[int(item["global_step"])] = {
                "step": int(item["global_step"]),
                "mean_ms": statistics.mean(samples),
                "median_ms": median,
                "min_ms": min(samples),
                "max_ms": max(samples),
                "spread_percent": (max(samples) - min(samples)) / median * 100 if median else 0,
                "per_rank_ms": {str(rank): value for rank, value in sorted(values.items())},
                "segment": summary.get("segment"),
            }
    rows = [by_step[step] for step in sorted(by_step)]
    _rolling_baseline(rows, "mean_ms")
    rank_histories = defaultdict(list)
    for row in rows:
        baselines = {}
        for rank, value in row["per_rank_ms"].items():
            history = rank_histories[rank]
            baselines[rank] = (statistics.median(history[-12:])
                               if len(history) >= 8 else None)
            history.append(value)
        row["per_rank_baseline_ms"] = baselines
    return rows


def _host_batches(run):
    grouped = defaultdict(list)
    for path in sorted((run / "diagnosis" / "sinks").glob("step_*/rank_*.json")):
        item = _read_json(path)
        if not item or not isinstance(item.get("duration_ns"), (int, float)):
            continue
        grouped[int(item["last_step"])].append(item)
    rows = []
    for last_step in sorted(grouped):
        items = grouped[last_step]
        values = {int(item["rank"]): float(item["duration_ns"]) / 1e6 for item in items}
        samples = list(values.values())
        median = statistics.median(samples)
        injections = [item for item in items
                      if isinstance(item.get("test_injected_delay_seconds"), (int, float))]
        rows.append({
            "first_step": min(int(item["first_step"]) for item in items),
            "last_step": last_step,
            "mean_ms": statistics.mean(samples),
            "median_ms": median,
            "min_ms": min(samples),
            "max_ms": max(samples),
            "spread_percent": (max(samples) - min(samples)) / median * 100 if median else 0,
            "per_rank_ms": {str(rank): value for rank, value in sorted(values.items())},
            "warmup": any(bool(item.get("warmup_batch")) for item in items),
            "boundary": any(bool(item.get("boundary_batch")) for item in items),
            "test_injected_ranks": sorted(int(item["rank"]) for item in injections),
            "test_injected_delay_seconds": (max(float(item["test_injected_delay_seconds"])
                                                  for item in injections)
                                              if injections else None),
            "event_at_unix_ns": max(int(item.get("event_at_unix_ns", 0)) for item in items),
        })
    _rolling_baseline(rows, "mean_ms", warmup_key="warmup")
    return rows


def _alerts_and_quality(run):
    diagnosis = run / "diagnosis"
    alerts = []
    patterns = (("host_rank", "sinks/step_*/rank_*.json"),
                ("host_peer", "sinks/step_*/peer_group_*.json"),
                ("device", "windows/segment_*.json"))
    for source, pattern in patterns:
        for path in sorted(diagnosis.glob(pattern)):
            item = _read_json(path)
            if not item:
                continue
            for finding in item.get("findings", ()):
                alerts.append({"source": source, "path": str(path.relative_to(run)), **finding})
    for path in sorted((diagnosis / "progress").glob("*.json")):
        item = _read_json(path)
        if item:
            alerts.append({"source": "progress", "path": str(path.relative_to(run)), **item})
    quality = []
    for path in sorted((diagnosis / "quality").glob("*.json")):
        item = _read_json(path)
        if item:
            quality.append({"path": str(path.relative_to(run)), **item})
    for path in sorted((diagnosis / "windows").glob("segment_*.json")):
        item = _read_json(path)
        if not item:
            continue
        for issue in item.get("quality_issues", ()):
            quality.append({"path": str(path.relative_to(run)), **issue})
    def order(item):
        return (item.get("global_step", item.get("last_step", item.get("peer_max_step", -1))),
                item.get("rank", -1), item.get("kind", item.get("code", "")))
    return sorted(alerts, key=order), sorted(quality, key=order)


def _progress(run, config, status):
    state = _read_json(run / "diagnosis" / "state.json") or {}
    rank_ids = config.get("fast_expected_rank_ids") or list(range(int(config.get("world_size", 0))))
    last_steps = state.get("last_sink_step", {})
    progress_times = state.get("last_progress_unix_ns", {})
    dropped = state.get("last_dropped", {})
    producer_done = state.get("producer_done", {})
    now_ns = time.time_ns()
    complete = status.get("state") == "complete"
    result = []
    for rank in rank_ids:
        key = str(rank)
        last_time = int(progress_times.get(key, 0) or 0)
        result.append({
            "rank": int(rank),
            "last_step": int(last_steps.get(key, 0) or 0),
            "stale_seconds": None if complete or not last_time else max(0, (now_ns - last_time) / 1e9),
            "dropped": int(dropped.get(key, 0) or 0),
            "producer_done": bool(producer_done.get(key, complete)),
        })
    return result


def build_dashboard_data(run):
    run = Path(run).resolve()
    config = _read_json(run / "sidecar_config.json") or {}
    status = _read_json(run / "status.json") or {}
    device = _device_steps(run)
    host_all = _host_batches(run)
    # The first sink callback spans graph compilation plus the first device
    # execution.  Keep it available for audit/progress, but do not let that
    # startup-only interval set the scale of the steady-state latency chart.
    host_warmup = [row for row in host_all if row["warmup"]]
    host = [row for row in host_all if not row["warmup"]]
    alerts, quality = _alerts_and_quality(run)
    progress = _progress(run, config, status)
    ranks = sorted({int(rank) for row in device for rank in row["per_rank_ms"]}
                   or {item["rank"] for item in progress})
    latest = device[-1] if device else None
    telemetry_peak = None
    report = _read_json(run / "ONLINE_REPORT.json") or {}
    if report:
        telemetry_peak = report.get("telemetry_queue_max_depth")
    return {
        "schema": "argus_dashboard_v1",
        "generated_at_unix_ns": time.time_ns(),
        "run": {"name": run.name, "path": str(run), "state": status.get("state", "unknown")},
        "test_injection": config.get("test_host_delay"),
        "summary": {
            "latest_device_step": latest["step"] if latest else None,
            "latest_host_step": host[-1]["last_step"] if host else None,
            "latest_device_mean_ms": latest["mean_ms"] if latest else None,
            "latest_device_spread_percent": latest["spread_percent"] if latest else None,
            "alerts": len(alerts), "quality_issues": len(quality),
            "dropped": sum(item["dropped"] for item in progress),
            "telemetry_queue_peak": telemetry_peak,
        },
        "ranks": ranks,
        "device_steps": device,
        "host_batches": host,
        "host_warmup_batches": host_warmup,
        "progress": progress,
        "alerts": alerts,
        "quality_issues": quality,
        "health": {
            "raw_bytes": status.get("raw_bytes", 0),
            "archived_segments": status.get("archived_segments", 0),
            "fast_counts": status.get("fast_counts", {}),
            "fast_window_counts": status.get("fast_window_counts", {}),
            "fast_pending": status.get("fast_pending", 0),
            "fast_window_pending": status.get("fast_window_pending", 0),
            "diagnosis_failed": status.get("diagnosis_failed"),
            "fatal": status.get("fatal"),
            "capture_interruptions": status.get("capture_interruptions", []),
        },
    }
