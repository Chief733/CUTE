"""CPU-only tests of incremental diagnosis, telemetry, and restart recovery."""

import json
import sys
import tempfile
import threading
import time
import unittest
from pathlib import Path

HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE))
from common import atomic_json, read_json
from diagnosis_worker import DiagnosisManager
from telemetry import TelemetryEmitter


def config():
    return {"online_diagnosis": True, "fast_expected_rank_ids": [0, 1, 2],
            "sink_size": 2, "diagnosis_peer_groups": [[0, 1, 2]]}


def make_window(archive, segment, first, values):
    directory = archive / "fast" / f"segment_{segment:06d}"
    directory.mkdir(parents=True)
    last = first + len(values) - 1
    for rank in range(3):
        atomic_json(directory / f"rank_{rank}.json", {
            "source": {"rank": rank, "segment": segment,
                       "declared_first_step": first, "declared_last_step": last},
            "coverage": {"step_trace_complete": True, "step_pairs": len(values)},
            "steps": [{"global_step": first + index, "duration_ns": row[rank]}
                      for index, row in enumerate(values)],
        })
    atomic_json(directory / "window_summary.json", {
        "schema": "fast_window_summary_v1", "segment": segment, "complete": True,
        "expected_rank_ids": [0, 1, 2], "rank_ids": [0, 1, 2],
        "step_comparisons": [
            {"global_step": first + index,
             "per_rank_ns": {str(rank): row[rank] for rank in range(3)}}
            for index, row in enumerate(values)
        ],
    })
    atomic_json(directory / "window_result.json", {
        "schema": "fast_window_result_v1", "segment": segment,
        "state": "complete", "finished_at": time.time(),
    })


def append_sinks(root, start_batch, end_batch, anomaly_batch=None):
    for rank in range(3):
        path = root / "metrics" / f"rank_{rank}.jsonl"
        path.parent.mkdir(exist_ok=True)
        with path.open("a") as stream:
            for batch in range(start_batch, end_batch + 1):
                duration = 14_000_000_000 + (2_000_000_000 if rank == 2 and batch == anomaly_batch else 0)
                event = {
                    "schema": "sink_telemetry_v1", "rank": rank,
                    "sequence": batch, "type": "sink_complete",
                    "first_step": 2 * batch - 1, "last_step": 2 * batch,
                    "begin_monotonic_ns": 0, "end_monotonic_ns": duration,
                    "duration_ns": duration, "event_at_unix_ns": time.time_ns(),
                    "profile_start_boundary": False, "profile_stop_boundary": False,
                }
                if rank == 2 and batch == anomaly_batch:
                    event["test_injected_delay_seconds"] = 2.0
                stream.write(json.dumps(event) + "\n")


class DiagnosisWorkerTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name) / "ram"
        self.archive = Path(self.temporary.name) / "archive"
        self.root.mkdir()
        self.archive.mkdir()

    def test_telemetry_emitter_writes_sink_and_done(self):
        emitter = TelemetryEmitter(self.root, 0, heartbeat_seconds=0.05)
        emitter.sink_done(1, 2, 100, 200, False, True,
                          test_injected_delay_seconds=2.0)
        emitter.close()
        events = [json.loads(line) for line in (self.root / "metrics/rank_0.jsonl").read_text().splitlines()]
        self.assertEqual(events[0]["type"], "sink_complete")
        self.assertEqual(events[0]["duration_ns"], 100)
        self.assertEqual(events[0]["test_injected_delay_seconds"], 2.0)
        self.assertEqual(events[-1]["type"], "producer_done")
        self.assertEqual(emitter.dropped, 0)
        self.assertEqual(events[-1]["max_queue_depth"], 1)

    def test_telemetry_queue_overflow_is_recorded_without_blocking(self):
        gate = threading.Event()

        class BlockedEmitter(TelemetryEmitter):
            def _write(self):
                gate.wait(2)
                super()._write()

        emitter = BlockedEmitter(self.root, 0, queue_size=1)
        emitter.sink_done(1, 2, 100, 200, False, False)
        emitter.sink_done(3, 4, 200, 300, False, False)
        self.assertEqual(emitter.dropped, 1)
        gate.set()
        emitter.close()
        events = [json.loads(line) for line in (self.root / "metrics/rank_0.jsonl").read_text().splitlines()]
        self.assertEqual(events[-1]["dropped"], 1)
        self.assertEqual(events[-1]["max_queue_depth"], 1)

    def test_sink_alerts_survive_manager_restart(self):
        append_sinks(self.root, 1, 10)
        manager = DiagnosisManager(self.root, self.archive, config())
        manager.tick()
        self.assertEqual(manager.state["last_sink_step"]["2"], 20)
        append_sinks(self.root, 11, 20, anomaly_batch=15)
        recovered = DiagnosisManager(self.root, self.archive, config())
        recovered.tick()
        self.assertEqual(recovered.state["last_sink_step"]["2"], 40)
        sink = read_json(self.archive / "diagnosis/sinks/step_000000030/rank_2.json")
        peer = read_json(self.archive / "diagnosis/sinks/step_000000030/peer_group_0.json")
        self.assertEqual([item["kind"] for item in sink["findings"]], ["sink_latency_spike"])
        self.assertEqual([item["kind"] for item in peer["findings"]], ["same_batch_rank_outlier"])
        self.assertEqual(sink["test_injected_delay_seconds"], 2.0)
        self.assertEqual(peer["findings"][0]["test_injected_delay_seconds"], 2.0)
        before = recovered.snapshot()["diagnosis_findings"]
        recovered.tick()
        self.assertEqual(recovered.snapshot()["diagnosis_findings"], before)

    def test_device_windows_are_processed_incrementally(self):
        manager = DiagnosisManager(self.root, self.archive, config())
        for segment in range(7):
            first = 3 + segment * 6
            values = [{rank: 7_000_000_000.0 for rank in range(3)} for _ in range(6)]
            if segment == 4:
                values[1][2] += 2_000_000_000.0  # step 28
            make_window(self.archive, segment, first, values)
            manager.tick([(segment, "complete")])
        result = read_json(self.archive / "diagnosis/windows/segment_000004.json")
        self.assertEqual(result["coverage"]["valid_rank_steps"], 18)
        self.assertEqual({item["method"] for item in result["findings"]},
                         {"iteration_latency", "peer_rank_latency"})
        self.assertEqual(manager.state["last_segment"], 6)
        recovered = DiagnosisManager(self.root, self.archive, config())
        recovered.tick()
        self.assertEqual(recovered.state["last_segment"], 6)
        self.assertFalse(recovered.failed)

    def test_failed_window_is_quality_result(self):
        directory = self.archive / "fast/segment_000000"
        directory.mkdir(parents=True)
        atomic_json(directory / "window_result.json", {
            "schema": "fast_window_result_v1", "segment": 0,
            "state": "failed", "error": "rank 2 missing",
        })
        manager = DiagnosisManager(self.root, self.archive, config())
        manager.tick()
        result = read_json(self.archive / "diagnosis/windows/segment_000000.json")
        self.assertEqual(result["quality_issues"][0]["code"], "FAST_WINDOW_FAILED")
        self.assertFalse(manager.failed)

    def test_progress_lag_and_peer_timeout_are_deduplicated(self):
        manager = DiagnosisManager(self.root, self.archive, config())
        now = time.time_ns()
        manager.state["last_sink_step"] = {"0": 20, "1": 20, "2": 18}
        manager.state["last_progress_unix_ns"] = {
            "0": now, "1": now, "2": now - 31_000_000_000,
        }
        manager.state["pending_sinks"] = {
            "20": {"0": {"event_at_unix_ns": now - 31_000_000_000,
                           "duration_ns": 14_000_000_000}},
        }
        manager.tick()
        self.assertTrue((self.archive / "diagnosis/progress/rank_2_step_18.json").exists())
        self.assertTrue((self.archive / "diagnosis/quality/PEER_SINK_TIMEOUT_all_20_all_group_0-1-2.json").exists())
        counts = manager.snapshot()
        manager.tick()
        self.assertEqual(manager.snapshot()["diagnosis_findings"], counts["diagnosis_findings"])
        self.assertEqual(manager.snapshot()["diagnosis_quality_issues"], counts["diagnosis_quality_issues"])

    def test_missing_initial_sink_is_reported(self):
        append_sinks(self.root, 2, 2)
        manager = DiagnosisManager(self.root, self.archive, config())
        manager.tick()
        self.assertTrue((self.archive / "diagnosis/quality/SINK_STEP_DISCONTINUITY_0_4_all.json").exists())
        self.assertEqual(manager.state["last_sink_step"]["0"], 4)


if __name__ == "__main__":
    unittest.main()
