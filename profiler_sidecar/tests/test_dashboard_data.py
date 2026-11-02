import json
import sys
import tempfile
import unittest
from pathlib import Path

HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE))
from dashboard_data import build_dashboard_data


class DashboardDataTests(unittest.TestCase):
    def test_builds_device_host_progress_and_alerts(self):
        with tempfile.TemporaryDirectory() as temporary:
            run = Path(temporary)
            (run / "fast/segment_000000").mkdir(parents=True)
            (run / "diagnosis/sinks/step_000000002").mkdir(parents=True)
            (run / "diagnosis/sinks/step_000000004").mkdir(parents=True)
            (run / "diagnosis/windows").mkdir(parents=True)
            (run / "sidecar_config.json").write_text(json.dumps({
                "world_size": 2, "fast_expected_rank_ids": [0, 1]}))
            (run / "status.json").write_text(json.dumps({
                "state": "running", "archived_segments": 2,
                "fast_counts": {"complete": 2}, "fast_window_counts": {"complete": 1}}))
            (run / "fast/segment_000000/window_summary.json").write_text(json.dumps({
                "complete": True, "segment": 0, "step_comparisons": [{
                    "global_step": 1, "per_rank_ns": {"0": 1_000_000_000, "1": 1_100_000_000}}]}))
            for rank, duration in ((0, 2_000_000_000), (1, 2_200_000_000)):
                (run / f"diagnosis/sinks/step_000000002/rank_{rank}.json").write_text(json.dumps({
                    "rank": rank, "first_step": 1, "last_step": 2, "duration_ns": duration,
                    "warmup_batch": True, "boundary_batch": False, "findings": []}))
            for rank, duration in ((0, 1_000_000_000), (1, 1_200_000_000)):
                item = {
                    "rank": rank, "first_step": 3, "last_step": 4, "duration_ns": duration,
                    "warmup_batch": False, "boundary_batch": False, "findings": []}
                if rank == 1:
                    item["test_injected_delay_seconds"] = 2.0
                (run / f"diagnosis/sinks/step_000000004/rank_{rank}.json").write_text(json.dumps(item))
            (run / "diagnosis/state.json").write_text(json.dumps({
                "last_sink_step": {"0": 2, "1": 2}, "last_progress_unix_ns": {},
                "last_dropped": {"0": 0, "1": 1}, "producer_done": {}}))
            (run / "diagnosis/windows/segment_000000.json").write_text(json.dumps({
                "findings": [{"kind": "step_latency_spike", "global_step": 1, "rank": 1}],
                "quality_issues": []}))
            data = build_dashboard_data(run)
            self.assertEqual(data["device_steps"][0]["mean_ms"], 1050.0)
            self.assertEqual(data["host_batches"][0]["max_ms"], 1200.0)
            self.assertEqual(data["host_batches"][0]["last_step"], 4)
            self.assertEqual(data["host_batches"][0]["test_injected_ranks"], [1])
            self.assertEqual(data["host_warmup_batches"][0]["max_ms"], 2200.0)
            self.assertEqual(data["summary"]["alerts"], 1)
            self.assertEqual(data["summary"]["dropped"], 1)
            self.assertEqual([item["last_step"] for item in data["progress"]], [2, 2])

    def test_device_cells_get_per_rank_history_baselines(self):
        with tempfile.TemporaryDirectory() as temporary:
            run = Path(temporary)
            (run / "fast/segment_000000").mkdir(parents=True)
            comparisons = []
            for step in range(1, 10):
                duration = 1_300_000_000 if step == 9 else 1_000_000_000
                comparisons.append({
                    "global_step": step,
                    "per_rank_ns": {"0": duration, "1": duration},
                })
            (run / "fast/segment_000000/window_summary.json").write_text(json.dumps({
                "complete": True, "segment": 0, "step_comparisons": comparisons,
            }))
            data = build_dashboard_data(run)
            last = data["device_steps"][-1]
            self.assertEqual(last["median_ms"], 1300.0)
            self.assertEqual(last["per_rank_baseline_ms"], {"0": 1000.0, "1": 1000.0})


if __name__ == "__main__":
    unittest.main()
