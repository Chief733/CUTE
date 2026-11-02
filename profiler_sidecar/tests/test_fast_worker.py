"""CPU-only end-to-end tests for the archive/fast-parser double latch."""
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

HERE = Path(__file__).resolve().parents[1]
FAST_ROOT = HERE.parent / "fastanalyse"
WORKER = HERE / "worker.py"
sys.path.insert(0, str(HERE))
from common import atomic_json, process_identity, read_json


def fixture_builder():
    path = FAST_ROOT / "tests" / "test_fast_parse.py"
    spec = importlib.util.spec_from_file_location("fast_parse_fixture", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.FastParseTest()


class FastWorkerTests(unittest.TestCase):
    def run_worker(self, directory, corrupt=False, missing_step_trace=False,
                   require_complete_step_trace=False):
        root = Path(directory) / "ram"
        archive = Path(directory) / "archive"
        root.mkdir()
        archive.mkdir()
        source = fixture_builder().make_fixture(root / "raw")
        if missing_step_trace:
            for track in source.glob("profiler/PROF_*/device_*/data/ts_track*"):
                track.unlink()
        if corrupt:
            stars = sorted(source.glob("profiler/PROF_*/device_*/data/stars_soc*"))[-1]
            with stars.open("ab") as stream:
                stream.write(b"x")
        config = {
            "archive": str(archive),
            "launcher": process_identity(),
            "parse": False,
            "fast_analyse": True,
            "fast_workers": 2,
            "fast_timeout_seconds": 10,
            "fast_top_k": 2,
            "fast_ratio_threshold": 1.2,
            "fast_require_complete_step_trace": require_complete_step_trace,
            "fast_expected_rank_ids": [2],
            "fast_binary": str(FAST_ROOT / "fast_parse"),
            "fast_aggregate_script": str(FAST_ROOT / "tools" / "aggregate_files.py"),
            "world_size": 3,
            "min_disk_free_gib": 0,
        }
        atomic_json(root / "config.json", config)
        atomic_json(root / "launcher_done.json", {"returncode": 0})
        process = subprocess.run(
            [sys.executable, str(WORKER), "run", str(root)],
            text=True,
            capture_output=True,
            timeout=15,
        )
        self.assertEqual(process.returncode, 0, process.stderr)
        return root, archive, source

    def test_fast_parse_and_archive_finish_before_ram_retirement(self):
        with tempfile.TemporaryDirectory() as directory:
            root, archive, source = self.run_worker(directory)
            self.assertFalse(source.exists())
            self.assertTrue(
                (archive / "segments/segment_000003/rank_2/archive.json").is_file()
            )
            summary = read_json(archive / "fast/segment_000003/rank_2.json")
            window = read_json(archive / "fast/segment_000003/window_summary.json")
            status = read_json(archive / "status.json")
            self.assertEqual(summary["source"]["rank"], 2)
            self.assertTrue(window["complete"])
            self.assertEqual(window["expected_rank_ids"], [2])
            self.assertEqual(status["state"], "complete")
            self.assertEqual(status["fast_counts"], {"complete": 1})
            self.assertEqual(status["fast_window_counts"], {"complete": 1})

    def test_parser_failure_is_durable_and_does_not_leak_tmpfs(self):
        with tempfile.TemporaryDirectory() as directory:
            root, archive, source = self.run_worker(directory, corrupt=True)
            self.assertFalse(source.exists())
            self.assertTrue(
                (archive / "segments/segment_000003/rank_2/archive.json").is_file()
            )
            result = read_json(archive / "fast/segment_000003/rank_2.result.json")
            status = read_json(archive / "status.json")
            self.assertEqual(result["state"], "failed")
            self.assertIn("trailing bytes=1", (archive / "fast/segment_000003/rank_2.log").read_text())
            self.assertEqual(status["state"], "incomplete")
            self.assertEqual(status["fast_counts"], {"failed": 1})
            self.assertEqual(status["fast_window_counts"], {"failed": 1})

    def test_required_step_trace_fails_closed_but_preserves_archive(self):
        with tempfile.TemporaryDirectory() as directory:
            root, archive, source = self.run_worker(
                directory,
                missing_step_trace=True,
                require_complete_step_trace=True,
            )
            self.assertFalse(source.exists())
            self.assertTrue(
                (archive / "segments/segment_000003/rank_2/archive.json").is_file()
            )
            result = read_json(archive / "fast/segment_000003/rank_2.result.json")
            status = read_json(archive / "status.json")
            self.assertEqual(result["state"], "failed")
            self.assertIn("incomplete per-step trace", result["error"])
            self.assertEqual(status["fast_counts"], {"failed": 1})


if __name__ == "__main__":
    unittest.main()
