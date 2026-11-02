import json
import tempfile
import unittest
from pathlib import Path

from tools.diagnose_windows import load_windows, parse_peer_groups, replay


def write_run(root, values, first_step=3, window_steps=6, ranks=(0, 1, 2), missing=None):
    """Write minimal fast summaries with authoritative global step alignment."""
    missing = missing or set()
    last_step = first_step + len(values) - 1
    for segment, first in enumerate(range(first_step, last_step + 1, window_steps)):
        last = min(first + window_steps - 1, last_step)
        segment_dir = root / f"segment_{segment:06d}"
        segment_dir.mkdir(parents=True)
        present = []
        for rank in ranks:
            if (segment, rank) in missing:
                continue
            present.append(rank)
            steps = [
                {"global_step": step, "duration_ns": values[step - first_step][rank]}
                for step in range(first, last + 1)
            ]
            rank_summary = {
                "source": {
                    "rank": rank, "segment": segment,
                    "declared_first_step": first, "declared_last_step": last,
                },
                "coverage": {"step_trace_complete": True, "step_pairs": len(steps)},
                "steps": steps,
            }
            (segment_dir / f"rank_{rank}.json").write_text(json.dumps(rank_summary))
        comparisons = [
            {
                "global_step": step,
                "per_rank_ns": {str(rank): values[step - first_step][rank] for rank in present},
            }
            for step in range(first, last + 1)
        ]
        window = {
            "schema": "fast_window_summary_v1", "segment": segment,
            "complete": len(present) == len(ranks),
            "expected_rank_ids": list(ranks), "rank_ids": present,
            "step_comparisons": comparisons,
        }
        (segment_dir / "window_summary.json").write_text(json.dumps(window))


class DiagnoseWindowsTest(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.normal = [{rank: 7_000_000_000.0 for rank in range(3)} for _ in range(42)]

    def test_normal_then_single_rank_spike(self):
        values = [dict(item) for item in self.normal]
        values[19 - 3][2] += 2_000_000_000.0
        write_run(self.root, values)
        report = replay(self.root, [[0, 1, 2]])
        self.assertEqual(report["coverage"]["valid_rank_steps"], 126)
        self.assertEqual(report["quality_issues"], [])
        matches = {(item["method"], item["global_step"], item["rank"])
                   for item in report["findings"]}
        self.assertIn(("iteration_latency", 19, 2), matches)
        self.assertIn(("peer_rank_latency", 19, 2), matches)
        self.assertFalse(any(item["method"] == "rolling_trend" for item in report["findings"]))

    def test_sustained_regression_and_gradual_drift(self):
        regression = [dict(item) for item in self.normal]
        for step in range(27, 45):
            for rank in range(3):
                regression[step - 3][rank] += 1_400_000_000.0
        write_run(self.root / "regression", regression)
        report = replay(self.root / "regression", [[0, 1, 2]])
        self.assertTrue(any(item["kind"] == "regression" for item in report["findings"]))
        self.assertFalse(any(item["kind"] == "drift" for item in report["findings"]))
        self.assertFalse(any(item["method"] == "peer_rank_latency" for item in report["findings"]))

        drift = [dict(item) for item in self.normal]
        for step in range(15, 45):
            for rank in range(3):
                drift[step - 3][rank] += (step - 14) * 140_000_000.0
        write_run(self.root / "drift", drift)
        report = replay(self.root / "drift", [[0, 1, 2]])
        self.assertTrue(any(item["kind"] == "drift" for item in report["findings"]))
        self.assertFalse(any(item["kind"] == "regression" for item in report["findings"]))

    def test_missing_rank_degrades_peer_comparison(self):
        write_run(self.root, self.normal, missing={(2, 2)})
        report = replay(self.root, [[0, 1, 2]])
        codes = {item["code"] for item in report["quality_issues"]}
        self.assertIn("INCOMPLETE_WINDOW", codes)
        self.assertIn("MISSING_RANK", codes)
        self.assertIn("PEER_GROUP_INCOMPLETE", codes)
        self.assertEqual(report["coverage"]["valid_rank_steps"], 120)
        self.assertFalse(any(item["segment"] == 2 and item["method"] == "peer_rank_latency"
                             for item in report["findings"]))

    def test_missing_rank_resets_its_history(self):
        values = [dict(item) for item in self.normal]
        values[22 - 3][2] += 2_000_000_000.0
        write_run(self.root, values, missing={(2, 2)})
        report = replay(self.root, [[0, 1, 2]])
        self.assertFalse(any(item["method"] == "iteration_latency"
                             and item["global_step"] == 22 and item["rank"] == 2
                             for item in report["findings"]))

    def test_rank_window_mismatch_is_not_used(self):
        write_run(self.root, self.normal)
        path = self.root / "segment_000002" / "rank_2.json"
        data = json.loads(path.read_text())
        data["steps"][0]["duration_ns"] += 1_000_000_000.0
        path.write_text(json.dumps(data))
        report = replay(self.root, [[0, 1, 2]])
        self.assertIn("WINDOW_RANK_MISMATCH", {item["code"] for item in report["quality_issues"]})
        self.assertEqual(report["coverage"]["valid_rank_steps"], 108)

    def test_group_config_is_explicit(self):
        self.assertEqual(parse_peer_groups("0,1,2;3,4,5"), [[0, 1, 2], [3, 4, 5]])
        with self.assertRaises(ValueError):
            parse_peer_groups("0,1,2;2,3,4")
        write_run(self.root, self.normal)
        report = replay(self.root)
        self.assertEqual(report["settings"]["peer_groups"], [])
        self.assertEqual(report["finding_counts"]["peer_rank_latency"], 0)

    def test_incremental_windows_match_full_replay(self):
        values = [dict(item) for item in self.normal]
        values[19 - 3][2] += 2_000_000_000.0
        for step in range(27, 45):
            values[step - 3][0] += 1_400_000_000.0
        write_run(self.root, values)
        full = replay(self.root, [[0, 1, 2]])
        state, findings, issues, previous_segment = None, [], [], None
        for path in sorted(self.root.glob("segment_*/window_summary.json")):
            windows, quality, _ = load_windows(
                self.root, [path], prior_segment=previous_segment,
                prior_last=state["prior_last"] if state else None,
            )
            part, state = replay(self.root, [[0, 1, 2]], windows=windows,
                                 issues=quality, initial_state=state, return_state=True)
            findings.extend(part["findings"])
            issues.extend(part["quality_issues"])
            previous_segment = windows[0]["segment"]
        self.assertEqual(findings, full["findings"])
        self.assertEqual(issues, full["quality_issues"])


if __name__ == "__main__":
    unittest.main()
