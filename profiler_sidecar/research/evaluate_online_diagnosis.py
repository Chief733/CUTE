#!/usr/bin/env python3
"""Feed real fast summaries and injected sink metrics through the online manager."""

import argparse
import json
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parents[1]
FAST_TOOLS = HERE.parent / "fastanalyse" / "tools"
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(FAST_TOOLS))
from common import atomic_json, read_json
from diagnosis_worker import DiagnosisManager
from evaluate_diagnosis import copy_minimal_run


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source_fast", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    source = args.source_fast.resolve()
    output = args.output_dir.resolve()
    if output.exists():
        parser.error(f"output directory already exists: {output}")
    source_windows = sorted(source.glob("segment_*/window_summary.json"))
    if len(source_windows) < 5:
        parser.error("need at least five real windows")
    windows = [read_json(path) for path in source_windows]
    ranks = windows[0]["expected_rank_ids"]
    if any(window["expected_rank_ids"] != ranks for window in windows):
        parser.error("expected ranks change across windows")
    interior = [item["global_step"] for window in windows
                for item in window["step_comparisons"][1:-1]]
    spike_step = interior[2 * len(interior) // 3]
    if spike_step % 2:
        spike_step += 1
    spike_rank = ranks[-1]
    root, archive = output / "ram", output / "archive"
    root.mkdir(parents=True)
    copy_minimal_run(
        source, archive / "fast",
        delta=lambda rank, step: 2_000_000_000.0 if rank == spike_rank and step == spike_step else 0.0,
    )
    step_values = {}
    for path in sorted((archive / "fast").glob("segment_*/window_summary.json")):
        window = read_json(path)
        for item in window["step_comparisons"]:
            step_values[item["global_step"]] = item["per_rank_ns"]
        atomic_json(path.with_name("window_result.json"), {
            "schema": "fast_window_result_v1", "segment": window["segment"],
            "state": "complete", "finished_at": time.time(),
        })
    for rank in ranks:
        path = root / "metrics" / f"rank_{rank}.jsonl"
        path.parent.mkdir(parents=True, exist_ok=True)
        events = []
        first_step = min(step_values)
        last_step = max(step_values)
        for end in range(2, last_step + 1, 2):
            duration = (step_values[end - 1][str(rank)] + step_values[end][str(rank)]
                        if end - 1 >= first_step else 14_000_000_000.0)
            events.append({
                "schema": "sink_telemetry_v1", "rank": rank, "sequence": len(events) + 1,
                "type": "sink_complete", "first_step": end - 1, "last_step": end,
                "begin_monotonic_ns": 0, "end_monotonic_ns": duration,
                "duration_ns": duration, "event_at_unix_ns": time.time_ns(),
                "profile_start_boundary": end >= 4 and (end - 4) % 6 == 0,
                "profile_stop_boundary": end >= 8 and end % 6 == 2,
                "warmup_batch": end == 2,
            })
        events.append({
            "schema": "sink_telemetry_v1", "rank": rank, "sequence": len(events) + 1,
            "type": "producer_done", "last_completed_step": last_step,
            "event_at_unix_ns": time.time_ns(), "dropped": 0,
        })
        path.write_text("".join(json.dumps(event) + "\n" for event in events), encoding="utf-8")
    manager = DiagnosisManager(root, archive, {
        "online_diagnosis": True, "fast_expected_rank_ids": ranks,
        "diagnosis_peer_groups": [ranks], "sink_size": 2,
    })
    manager.tick()
    manager.finish()
    sink = read_json(archive / "diagnosis" / "sinks" / f"step_{spike_step:09d}" / f"rank_{spike_rank}.json")
    peer = read_json(archive / "diagnosis" / "sinks" / f"step_{spike_step:09d}" / "peer_group_0.json")
    window_path = next(path for path in (archive / "diagnosis/windows").glob("segment_*.json")
                       if any(item["global_step"] == spike_step
                              for item in read_json(path)["findings"]))
    device = read_json(window_path)
    checks = {
        "host_self": any(item["rank"] == spike_rank and item["kind"] == "sink_latency_spike"
                         for item in sink["findings"]),
        "host_peer": any(item["rank"] == spike_rank and item["kind"] == "same_batch_rank_outlier"
                         for item in peer["findings"]),
        "device_step": any(item["rank"] == spike_rank and item["global_step"] == spike_step
                           for item in device["findings"]),
        "no_pipeline_failure": manager.failed is None,
        "no_quality_issues": manager.new_quality == 0,
    }
    evaluation = {
        "schema": "online_diagnosis_stream_evaluation_v1",
        "source_fast": str(source), "spike_rank": spike_rank, "spike_step": spike_step,
        "checks": checks, "passed": all(checks.values()),
        "host_event_to_result_ms": (sink["published_at_unix_ns"] - sink["event_at_unix_ns"]) / 1e6,
        "window_result_to_diagnosis_ms":
            (device["published_at_unix_ns"] / 1e9 - device["window_finished_at"]) * 1e3,
        "note": "CPU stream replay latency; excludes training, sink, and window wait times",
        "diagnosis_status": manager.snapshot(),
    }
    atomic_json(output / "evaluation.json", evaluation)
    print(json.dumps({"output": str(output / "evaluation.json"), "passed": evaluation["passed"],
                      "checks": checks}, ensure_ascii=False))
    return 0 if evaluation["passed"] else 1


if __name__ == "__main__":
    sys.exit(main())
