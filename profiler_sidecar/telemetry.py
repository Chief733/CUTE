"""Small best-effort callback telemetry, written outside the training thread."""

import json
import queue
import threading
import time
from pathlib import Path


class TelemetryEmitter:
    def __init__(self, root, rank, heartbeat_seconds=5.0, queue_size=256):
        self.rank = rank
        self.path = Path(root) / "metrics" / f"rank_{rank}.jsonl"
        self.queue = queue.Queue(maxsize=queue_size)
        self.stop = threading.Event()
        self.last_step = 0
        self.dropped = 0
        self.max_queue_depth = 0
        self.error = None
        self.heartbeat_seconds = heartbeat_seconds
        self.thread = threading.Thread(target=self._write, name=f"cyf-telemetry-{rank}", daemon=True)
        self.thread.start()

    def sink_done(self, first_step, last_step, begin_ns, end_ns, start_boundary, stop_boundary,
                  warmup_batch=False, test_injected_delay_seconds=None):
        self.last_step = last_step
        event = {
            "type": "sink_complete", "first_step": first_step, "last_step": last_step,
            "begin_monotonic_ns": begin_ns, "end_monotonic_ns": end_ns,
            "duration_ns": end_ns - begin_ns if begin_ns is not None else None,
            "event_at_unix_ns": time.time_ns(),
            "profile_start_boundary": bool(start_boundary),
            "profile_stop_boundary": bool(stop_boundary),
            "warmup_batch": bool(warmup_batch),
        }
        if test_injected_delay_seconds is not None:
            event["test_injected_delay_seconds"] = float(test_injected_delay_seconds)
        if self.error is not None:
            self.dropped += 1
            return
        try:
            self.queue.put_nowait(event)
            self.max_queue_depth = max(self.max_queue_depth, 1, self.queue.qsize())
        except queue.Full:
            self.dropped += 1

    def _write(self):
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            with self.path.open("a", encoding="utf-8", buffering=1) as stream:
                next_heartbeat = time.monotonic() + self.heartbeat_seconds
                sequence = 0
                while not self.stop.is_set() or not self.queue.empty():
                    timeout = max(0.0, next_heartbeat - time.monotonic())
                    try:
                        event = self.queue.get(timeout=timeout)
                    except queue.Empty:
                        event = None
                    if event is not None:
                        sequence += 1
                        stream.write(json.dumps({"schema": "sink_telemetry_v1", "rank": self.rank,
                                                 "sequence": sequence, **event}, separators=(",", ":")) + "\n")
                    if time.monotonic() >= next_heartbeat:
                        sequence += 1
                        stream.write(json.dumps({
                            "schema": "sink_telemetry_v1", "rank": self.rank,
                            "sequence": sequence, "type": "heartbeat",
                            "last_completed_step": self.last_step,
                            "event_at_unix_ns": time.time_ns(), "dropped": self.dropped,
                            "queue_depth": self.queue.qsize(),
                            "max_queue_depth": self.max_queue_depth,
                        }, separators=(",", ":")) + "\n")
                        next_heartbeat = time.monotonic() + self.heartbeat_seconds
                sequence += 1
                stream.write(json.dumps({
                    "schema": "sink_telemetry_v1", "rank": self.rank,
                    "sequence": sequence, "type": "producer_done",
                    "last_completed_step": self.last_step,
                    "event_at_unix_ns": time.time_ns(), "dropped": self.dropped,
                    "queue_depth": self.queue.qsize(),
                    "max_queue_depth": self.max_queue_depth,
                }, separators=(",", ":")) + "\n")
        except OSError as error:
            self.error = repr(error)

    def close(self, timeout=2.0):
        self.stop.set()
        try:
            self.queue.put_nowait(None)
        except queue.Full:
            pass
        self.thread.join(timeout=timeout)
