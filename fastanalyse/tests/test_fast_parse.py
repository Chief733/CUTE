import json
import struct
import subprocess
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
BINARY = ROOT / "fast_parse"


def step_record(timestamp, index_id, tag_id, kind=10):
    return struct.pack(
        "<BBHIQQQHHHH",
        1,
        kind,
        40,
        0,
        timestamp,
        index_id,
        0xFFFFFFFF,
        2,
        7,
        tag_id,
        0,
    )


def stars_record(func_type, task_type, stream_id, task_id, syscnt, acc_id=0):
    header = (task_type << 10) | func_type
    return struct.pack(
        "<4HQHH11I",
        header,
        0,
        stream_id,
        task_id,
        syscnt,
        0,
        acc_id,
        *([0] * 11),
    )


def api_record(level, struct_type, thread_id, start, end, item_id):
    return struct.pack("<HHIIIQQQ", 0x5A5A, level, struct_type, thread_id, 0, start, end, item_id)


def task_track_record(thread_id, timestamp, device_id, stream_id, task_id, task_type):
    return struct.pack(
        "<HHIIIQHHHHQ24B",
        0x5A5A,
        5000,
        802,
        thread_id,
        16,
        timestamp,
        device_id,
        stream_id,
        task_id,
        0,
        task_type,
        *([0] * 24),
    )


class FastParseTest(unittest.TestCase):
    def make_fixture(self, root):
        rank = root / "segment_000003" / "rank_2"
        device = rank / "profiler" / "PROF_fixture" / "device_2"
        data = device / "data"
        data.mkdir(parents=True)
        (rank / "ready.json").write_text(
            '{"rank":2,"segment":3,"first_step":11,"last_step":11}',
            encoding="utf-8",
        )
        (device / "info.json.2").write_text(
            '{"DeviceInfo":[{"hwts_frequency":"50.0"}]}',
            encoding="utf-8",
        )
        ts_data = b"".join(
            [
                # Device index IDs can restart when collection begins after
                # warmup; ready.json is the authoritative global step range.
                step_record(1000, 1, 60000),
                step_record(1500, 0, 0, kind=14),
                step_record(2000, 1, 60001),
            ]
        )
        (data / "ts_track.data.2.slice_0").write_bytes(ts_data)

        stars_data = b"".join(
            [
                stars_record(0, 0, 3, 9, 1100),
                stars_record(1, 0, 3, 9, 1150),
                stars_record(34, 0, 0, 0, 1200),
            ]
        )
        # Deliberately split a record across slices to exercise carry handling.
        (data / "stars_soc.data.2.slice_0").write_bytes(stars_data[:70])
        (data / "stars_soc.data.2.slice_1").write_bytes(stars_data[70:])

        host_data = rank / "profiler" / "PROF_fixture" / "host" / "data"
        host_data.mkdir(parents=True)
        (host_data / "unaging.additional.type_info_dic.slice_0").write_text(
            "5000_0:KERNEL_AICORE\n5000_802:task_track\n10000_5:launch\n",
            encoding="utf-8",
        )
        (host_data / "unaging.additional.hash_dic.slice_0").write_text(
            "99:synthetic_kernel\n", encoding="utf-8"
        )
        (host_data / "aging.api_event.data.slice_0").write_bytes(
            api_record(10000, 5, 42, 1000, 2000, 99)
        )
        (host_data / "aging.compact.task_track.slice_0").write_bytes(
            task_track_record(42, 1500, 2, 3, 9, 0)
        )
        return rank

    def test_parses_steps_and_acsq_tasks(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            rank = self.make_fixture(Path(temp_dir))
            result = subprocess.run(
                [str(BINARY), "--top-k", "1", str(rank)],
                check=True,
                text=True,
                capture_output=True,
            )
            summary = json.loads(result.stdout)
            self.assertEqual(summary["source"]["rank"], 2)
            self.assertTrue(summary["coverage"]["ts_track_available"])
            self.assertTrue(summary["coverage"]["step_trace_available"])
            self.assertTrue(summary["coverage"]["step_trace_complete"])
            self.assertEqual(summary["coverage"]["expected_step_pairs"], 1)
            self.assertEqual(summary["coverage"]["step_pairs"], 1)
            self.assertEqual(summary["coverage"]["matched_device_tasks"], 1)
            self.assertEqual(summary["coverage"]["step_assigned_tasks"], 1)
            self.assertEqual(summary["coverage"]["host_kernel_records"], 1)
            self.assertEqual(summary["coverage"]["named_device_tasks"], 1)
            self.assertEqual(summary["steps"][0]["global_step"], 11)
            self.assertEqual(summary["steps"][0]["index_id"], 1)
            self.assertEqual(summary["steps"][0]["duration_ticks"], 1000)
            self.assertAlmostEqual(summary["steps"][0]["duration_ns"], 20000.0)
            self.assertEqual(summary["device_task_types"][0]["task_type"], "AI_CORE")
            self.assertAlmostEqual(summary["device_task_types"][0]["max_ns"], 1000.0)
            self.assertEqual(summary["kernels"][0]["name"], "synthetic_kernel")
            self.assertEqual(summary["kernels"][0]["count"], 1)

    def test_rejects_trailing_partial_record(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            rank = self.make_fixture(Path(temp_dir))
            data = next(rank.glob("profiler/PROF_*/device_*/data"))
            with (data / "stars_soc.data.2.slice_1").open("ab") as out:
                out.write(b"x")
            result = subprocess.run(
                [str(BINARY), str(rank)],
                check=False,
                text=True,
                capture_output=True,
            )
            self.assertEqual(result.returncode, 2)
            self.assertIn("trailing bytes=1", result.stderr)

    def test_missing_step_trace_degrades_without_losing_kernels(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            rank = self.make_fixture(Path(temp_dir))
            for path in rank.glob("profiler/PROF_*/device_*/data/ts_track*"):
                path.unlink()
            result = subprocess.run(
                [str(BINARY), str(rank)],
                check=True,
                text=True,
                capture_output=True,
            )
            summary = json.loads(result.stdout)
            self.assertEqual(summary["coverage"]["ts_track_files"], 0)
            self.assertFalse(summary["coverage"]["ts_track_available"])
            self.assertFalse(summary["coverage"]["step_trace_available"])
            self.assertFalse(summary["coverage"]["step_trace_complete"])
            self.assertEqual(summary["coverage"]["step_pairs"], 0)
            self.assertEqual(summary["steps"], [])
            self.assertEqual(summary["coverage"]["matched_device_tasks"], 1)
            self.assertEqual(summary["coverage"]["named_device_tasks"], 1)

    def test_task_flip_only_ts_track_is_not_a_step_trace(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            rank = self.make_fixture(Path(temp_dir))
            data = next(rank.glob("profiler/PROF_*/device_*/data"))
            track = next(data.glob("ts_track*"))
            track.write_bytes(step_record(1500, 0, 0, kind=14))
            result = subprocess.run(
                [str(BINARY), str(rank)],
                check=True,
                text=True,
                capture_output=True,
            )
            summary = json.loads(result.stdout)
            self.assertTrue(summary["coverage"]["ts_track_available"])
            self.assertFalse(summary["coverage"]["step_trace_available"])
            self.assertFalse(summary["coverage"]["step_trace_complete"])
            self.assertEqual(summary["coverage"]["step_pairs"], 0)

    def test_incomplete_trace_does_not_guess_global_step(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            rank = self.make_fixture(Path(temp_dir))
            (rank / "ready.json").write_text(
                '{"rank":2,"segment":3,"first_step":11,"last_step":12}',
                encoding="utf-8",
            )
            result = subprocess.run(
                [str(BINARY), str(rank)],
                check=True,
                text=True,
                capture_output=True,
            )
            summary = json.loads(result.stdout)
            self.assertEqual(summary["coverage"]["expected_step_pairs"], 2)
            self.assertEqual(summary["coverage"]["step_pairs"], 1)
            self.assertFalse(summary["coverage"]["step_trace_complete"])
            self.assertIsNone(summary["steps"][0]["global_step"])


if __name__ == "__main__":
    unittest.main()
