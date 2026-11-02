#!/usr/bin/env python3
"""Inject latency and data-quality faults into copies of one real fast run."""

import argparse
import json
import sys
from pathlib import Path

from diagnose_windows import replay


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def copy_minimal_run(source, destination, delta=None, omit_rank=None, skip_segment=None):
    for window_path in sorted(source.glob("segment_*/window_summary.json")):
        source_window = json.loads(window_path.read_text(encoding="utf-8"))
        segment = source_window["segment"]
        if segment == skip_segment:
            continue
        expected = source_window["expected_rank_ids"]
        target_dir = destination / window_path.parent.name
        present = []
        by_step = {}
        for rank in expected:
            if omit_rank == (segment, rank):
                continue
            source_rank = json.loads((window_path.parent / f"rank_{rank}.json").read_text(encoding="utf-8"))
            steps = []
            for item in source_rank["steps"]:
                step = item["global_step"]
                value = item["duration_ns"] + (delta(rank, step) if delta else 0.0)
                steps.append({"global_step": step, "duration_ns": value})
                by_step.setdefault(step, {})[str(rank)] = value
            rank_copy = {
                "source": {key: source_rank["source"][key] for key in
                           ("rank", "segment", "declared_first_step", "declared_last_step")},
                "coverage": {key: source_rank["coverage"][key] for key in
                             ("step_trace_complete", "step_pairs")},
                "steps": steps,
            }
            write_json(target_dir / f"rank_{rank}.json", rank_copy)
            present.append(rank)
        window_copy = {
            "schema": "fast_window_summary_v1",
            "segment": segment,
            "complete": set(present) == set(expected),
            "expected_rank_ids": expected,
            "rank_ids": present,
            "step_comparisons": [
                {"global_step": step, "per_rank_ns": ranks}
                for step, ranks in sorted(by_step.items())
            ],
        }
        write_json(target_dir / "window_summary.json", window_copy)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source_fast", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--peer-groups", required=True, help="one same-role group, e.g. 0,1,...,7")
    args = parser.parse_args()
    from diagnose_windows import parse_peer_groups
    try:
        source = args.source_fast.resolve()
        output = args.output_dir.resolve()
        if output.exists():
            raise ValueError(f"output directory already exists: {output}")
        window_paths = sorted(source.glob("segment_*/window_summary.json"))
        if len(window_paths) < 5:
            raise ValueError("injection replay needs at least five windows")
        windows = [json.loads(path.read_text(encoding="utf-8")) for path in window_paths]
        steps = [item["global_step"] for window in windows for item in window["step_comparisons"]]
        if len(steps) < 30 or steps != list(range(steps[0], steps[-1] + 1)):
            raise ValueError("injection replay needs at least 30 contiguous global steps")
        expected = windows[0]["expected_rank_ids"]
        if any(window["expected_rank_ids"] != expected for window in windows):
            raise ValueError("expected ranks change across windows")
        groups = parse_peer_groups(args.peer_groups)
        last_rank = expected[-1]
        interior = [item["global_step"] for window in windows
                    for item in window["step_comparisons"][1:-1]]
        spike_step = interior[2 * len(interior) // 3]
        regression_start = steps[2 * len(steps) // 3]
        drift_start = steps[len(steps) // 3]
        affected_segment = windows[len(windows) // 2]["segment"]
        cases = {
            "baseline": {},
            "single_rank_spike": {
                "delta": lambda rank, step: 2_000_000_000.0 if rank == last_rank and step == spike_step else 0.0,
            },
            "global_regression": {
                "delta": lambda rank, step: 1_400_000_000.0 if step >= regression_start else 0.0,
            },
            "global_drift": {
                "delta": lambda rank, step: max(0, step - drift_start + 1) * 140_000_000.0,
            },
            "missing_rank": {"omit_rank": (affected_segment, last_rank)},
            "step_gap": {"skip_segment": affected_segment},
        }
        results = {}
        for name, modification in cases.items():
            case_dir = output / name
            copy_minimal_run(source, case_dir / "fast", **modification)
            report = replay(case_dir / "fast", groups)
            write_json(case_dir / "diagnosis.json", report)
            methods = {(item["method"], item["kind"], item["global_step"], item["rank"])
                       for item in report["findings"]}
            codes = {item["code"] for item in report["quality_issues"]}
            if name == "baseline":
                passed = not methods and not codes
            elif name == "single_rank_spike":
                passed = (("iteration_latency", "step_latency_spike", spike_step, last_rank) in methods
                          and ("peer_rank_latency", "same_step_rank_outlier", spike_step, last_rank) in methods)
            elif name == "global_regression":
                passed = (any(method == "rolling_trend" and kind == "regression"
                              for method, kind, _, _ in methods)
                          and not any(method == "rolling_trend" and kind == "drift"
                                      for method, kind, _, _ in methods))
            elif name == "global_drift":
                passed = (any(method == "rolling_trend" and kind == "drift"
                              for method, kind, _, _ in methods)
                          and not any(method == "rolling_trend" and kind == "regression"
                                      for method, kind, _, _ in methods))
            elif name == "missing_rank":
                passed = {"INCOMPLETE_WINDOW", "MISSING_RANK", "PEER_GROUP_INCOMPLETE"}.issubset(codes)
            else:
                passed = {"STEP_DISCONTINUITY", "SEGMENT_DISCONTINUITY"}.issubset(codes)
            results[name] = {
                "passed": passed,
                "finding_counts": report["finding_counts"],
                "quality_issue_codes": sorted(codes),
                "coverage": report["coverage"],
            }
        evaluation = {
            "schema": "diagnosis_injection_evaluation_v1",
            "source_fast": str(source),
            "peer_groups": groups,
            "injections": {
                "single_rank_spike": {"rank": last_rank, "step": spike_step, "added_ns": 2_000_000_000},
                "global_regression": {"first_step": regression_start, "added_ns": 1_400_000_000},
                "global_drift": {"first_step": drift_start, "added_ns_per_step": 140_000_000},
                "missing_rank": {"segment": affected_segment, "rank": last_rank},
                "step_gap": {"segment": affected_segment},
            },
            "cases": results,
            "passed": all(item["passed"] for item in results.values()),
        }
        write_json(output / "evaluation.json", evaluation)
        print(json.dumps({"output": str(output / "evaluation.json"), "passed": evaluation["passed"],
                          "cases": {name: item["passed"] for name, item in results.items()}},
                         ensure_ascii=False))
        return 0 if evaluation["passed"] else 1
    except (OSError, ValueError, KeyError, TypeError, json.JSONDecodeError) as error:
        print(f"evaluate_diagnosis: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
