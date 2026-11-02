#!/usr/bin/env python3
"""Verify an 8-rank online diagnosis run and summarize actual alert latency."""

import argparse
import json
import statistics
from pathlib import Path

from common import atomic_json, read_json


def percentile(values, fraction):
    values = sorted(values)
    if not values:
        return None
    position = fraction * (len(values) - 1)
    lower = int(position)
    upper = min(lower + 1, len(values) - 1)
    return values[lower] + (values[upper] - values[lower]) * (position - lower)


def steady_seconds_per_step(run, first_step=2, last_step=40):
    per_rank = {}
    for path in sorted((run / "events").glob("rank_*.jsonl")):
        rank = int(path.stem.split("_")[-1])
        events = [json.loads(line) for line in path.read_text().splitlines()]
        steps = {event["step"]: event["monotonic"] for event in events
                 if event.get("event") == "loss_step_end_end" and event.get("step") in
                 (first_step, last_step)}
        if set(steps) == {first_step, last_step}:
            per_rank[rank] = (steps[last_step] - steps[first_step]) / (last_step - first_step)
    return per_rank


def report(run, first_step, last_step, ranks, window_steps):
    run = Path(run).resolve()
    training = read_json(run / "training_run.json")
    status = read_json(run / "status.json")
    config = read_json(run / "sidecar_config.json")
    sink_size = config["sink_size"]
    diagnosis = run / "diagnosis"
    sink_paths = sorted(diagnosis.glob("sinks/step_*/rank_*.json"))
    peer_paths = sorted(diagnosis.glob("sinks/step_*/peer_group_*.json"))
    window_paths = sorted(diagnosis.glob("windows/segment_*.json"))
    quality_paths = sorted(diagnosis.glob("quality/*.json"))
    sink_results = [read_json(path) for path in sink_paths]
    window_results = [read_json(path) for path in window_paths]
    host_latency = [(item["published_at_unix_ns"] - item["event_at_unix_ns"]) / 1e6
                    for item in sink_results]
    window_latency = [(item["published_at_unix_ns"] / 1e9 - item["window_finished_at"]) * 1e3
                      for item in window_results]
    ready_latency = []
    for item in window_results:
        segment = item["segment"]
        ready = [read_json(path)["ready_at"] for path in
                 (run / "segments" / f"segment_{segment:06d}").glob("rank_*/ready.json")]
        if ready:
            ready_latency.append((item["published_at_unix_ns"] / 1e9 - max(ready)) * 1e3)
    telemetry = {}
    for rank in range(ranks):
        path = diagnosis / "telemetry" / f"rank_{rank}.jsonl"
        if path.is_file():
            events = [json.loads(line) for line in path.read_text().splitlines()]
            telemetry[rank] = {
                "sink_events": sum(event["type"] == "sink_complete" for event in events),
                "heartbeats": sum(event["type"] == "heartbeat" for event in events),
                "producer_done": bool(events) and events[-1]["type"] == "producer_done",
                "dropped": events[-1].get("dropped") if events else None,
                "max_queue_depth": events[-1].get("max_queue_depth") if events else None,
            }
    rank_times = steady_seconds_per_step(run, first_step=2, last_step=last_step)
    expected_sinks = (last_step // sink_size) * ranks
    expected_windows = (last_step - first_step + window_steps) // window_steps
    expected_device_rank_steps = (last_step - first_step + 1) * ranks
    actual_device_rank_steps = sum(item["coverage"]["valid_rank_steps"] for item in window_results
                                   if item.get("coverage"))
    checks = {
        "training_succeeded": training.get("returncode") == 0,
        "sidecar_complete": status.get("state") == "complete",
        "diagnosis_healthy": status.get("diagnosis_failed") is None,
        "all_sinks": len(sink_results) == expected_sinks,
        "all_peer_results": len(peer_paths) == last_step // sink_size,
        "all_windows": len(window_results) == expected_windows,
        "all_device_rank_steps": actual_device_rank_steps == expected_device_rank_steps,
        "no_quality_issues": not quality_paths and all(not item["quality_issues"] for item in window_results),
        "telemetry_complete": len(telemetry) == ranks and all(
            item["producer_done"] and item["sink_events"] == last_step // sink_size
            and item["dropped"] == 0 for item in telemetry.values()),
        "rank_timings_complete": len(rank_times) == ranks,
    }
    return {
        "schema": "online_diagnosis_run_report_v1",
        "run": str(run), "checks": checks, "passed": all(checks.values()),
        "counts": {
            "sink_results": len(sink_results), "peer_results": len(peer_paths),
            "window_results": len(window_results), "device_rank_steps": actual_device_rank_steps,
            "quality_files": len(quality_paths),
            "host_findings": sum(len(item["findings"]) for item in sink_results)
                             + sum(len(read_json(path)["findings"]) for path in peer_paths),
            "device_findings": sum(len(item["findings"]) for item in window_results),
        },
        "latency_ms": {
            "host_event_to_result_p50": percentile(host_latency, 0.50),
            "host_event_to_result_p95": percentile(host_latency, 0.95),
            "host_event_to_result_max": max(host_latency) if host_latency else None,
            "window_complete_to_diagnosis_p95": percentile(window_latency, 0.95),
            "window_complete_to_diagnosis_max": max(window_latency) if window_latency else None,
            "last_rank_ready_to_diagnosis_p95": percentile(ready_latency, 0.95),
            "last_rank_ready_to_diagnosis_max": max(ready_latency) if ready_latency else None,
        },
        "steady_seconds_per_step_by_rank": rank_times,
        "steady_seconds_per_step_job": max(rank_times.values()) if rank_times else None,
        "telemetry": telemetry,
        "telemetry_queue_max_depth": (max(item["max_queue_depth"] for item in telemetry.values())
                                      if telemetry and all(item["max_queue_depth"] is not None
                                                           for item in telemetry.values()) else None),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run", type=Path)
    parser.add_argument("--first-step", type=int, default=3)
    parser.add_argument("--last-step", type=int, default=40)
    parser.add_argument("--ranks", type=int, default=8)
    parser.add_argument("--window-steps", type=int, default=6)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    result = report(args.run, args.first_step, args.last_step, args.ranks, args.window_steps)
    output = args.output or args.run / "ONLINE_REPORT.json"
    atomic_json(output, result)
    print(json.dumps({"output": str(output), "passed": result["passed"],
                      "checks": result["checks"]}, ensure_ascii=False))
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
