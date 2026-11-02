"""Incremental, restartable CPU-only diagnosis consumer for one sidecar run."""

import json
import math
import shutil
import statistics
import sys
import time
from pathlib import Path

from common import atomic_json, read_json

FAST_TOOLS = Path(__file__).resolve().parents[1] / "fastanalyse" / "tools"
if str(FAST_TOOLS) not in sys.path:
    sys.path.insert(0, str(FAST_TOOLS))
from diagnose_windows import (exceeds, gradual_drift, load_windows, replay, robust_z)


class DiagnosisManager:
    def __init__(self, root, archive, config):
        self.enabled = bool(config.get("online_diagnosis", False))
        self.root = Path(root)
        self.archive = Path(archive)
        self.output = self.archive / "diagnosis"
        self.fast_dir = self.archive / "fast"
        self.expected = tuple(int(rank) for rank in config.get("fast_expected_rank_ids", ()))
        self.sink_size = int(config.get("sink_size", 1))
        self.groups = [list(map(int, group)) for group in config.get("diagnosis_peer_groups", ())]
        self.pending_windows = {}
        self.failed = None
        self.new_findings = 0
        self.new_quality = 0
        if not self.enabled:
            return
        self.output.mkdir(parents=True, exist_ok=True)
        self.state_path = self.output / "state.json"
        self.state = read_json(self.state_path) if self.state_path.exists() else {
            "schema": "online_diagnosis_state_v1",
            "last_segment": -1, "device_state": None,
            "offsets": {}, "last_sequences": {},
            "host_histories": {}, "host_slow_streak": {},
            "host_trend_pending": {}, "host_trend_kind": {}, "host_trend_active": {},
            "last_sink_step": {}, "last_progress_unix_ns": {},
            "last_heartbeat_unix_ns": {}, "last_dropped": {},
            "producer_done": {}, "pending_sinks": {}, "peer_done": {},
            "progress_alerted": {},
        }
        if self.state.get("schema") != "online_diagnosis_state_v1":
            raise ValueError("unsupported online diagnosis state")
        self.new_findings = self.state.get("finding_total", 0)
        self.new_quality = self.state.get("quality_total", 0)
        # Read old filenames once on restart. New results arrive through events.
        for path in self.fast_dir.glob("segment_*/window_result.json"):
            result = read_json(path)
            segment = result["segment"]
            if segment > self.state["last_segment"]:
                self.pending_windows[segment] = result["state"]

    def _save(self):
        atomic_json(self.state_path, self.state)

    def _quality(self, code, **fields):
        event = {"schema": "online_diagnosis_quality_v1", "code": code,
                 "published_at_unix_ns": time.time_ns(), **fields}
        key = "_".join(str(fields.get(name, "all")) for name in
                       ("rank", "last_step", "segment"))
        if "peer_group" in fields:
            key += "_group_" + "-".join(map(str, fields["peer_group"]))
        if "dropped" in fields:
            key += "_dropped_" + str(fields["dropped"])
        path = self.output / "quality" / f"{code}_{key}.json"
        if path.exists():
            return False
        atomic_json(path, event)
        self.new_quality += 1
        self.state["quality_total"] = self.new_quality
        return True

    def _host_finding(self, method, kind, rank, first, last, value, baseline, confidence, **extra):
        return {
            "method": method, "kind": kind, "metric": "host_sink_batch_duration_ns",
            "rank": rank, "first_step": first, "last_step": last,
            "observed_ns": value, "baseline_ns": baseline,
            "absolute_delta_ns": value - baseline,
            "relative_delta": value / baseline - 1,
            "confidence": confidence, **extra,
        }

    def _process_sink(self, event):
        rank = event["rank"]
        first, last = event["first_step"], event["last_step"]
        previous = self.state["last_sink_step"].get(str(rank), 0)
        if (rank not in self.expected or last - first + 1 != self.sink_size
                or first != previous + 1):
            self._quality("SINK_STEP_DISCONTINUITY", rank=rank, last_step=last,
                          expected_first=previous + 1, observed_first=first)
            self.state["host_histories"].pop(str(rank), None)
            for name in ("host_slow_streak", "host_trend_pending",
                         "host_trend_kind", "host_trend_active"):
                self.state[name].pop(str(rank), None)
        if previous and last <= previous:
            return
        self.state["last_sink_step"][str(rank)] = last
        self.state["last_progress_unix_ns"][str(rank)] = event["event_at_unix_ns"]
        value = event.get("duration_ns")
        if not isinstance(value, (int, float)) or not math.isfinite(value) or value <= 0:
            self._quality("INVALID_SINK_DURATION", rank=rank, last_step=last)
            return
        value = float(value)
        boundary = bool(event.get("profile_start_boundary") or event.get("profile_stop_boundary"))
        warmup = bool(event.get("warmup_batch", last <= self.sink_size))
        test_delay = event.get("test_injected_delay_seconds")
        if not isinstance(test_delay, (int, float)) or not math.isfinite(test_delay) or test_delay <= 0:
            test_delay = None
        else:
            test_delay = float(test_delay)
        test_fields = ({"test_injected_delay_seconds": test_delay}
                       if test_delay is not None else {})
        history = self.state["host_histories"].setdefault(str(rank), [])
        findings = []
        if not warmup and len(history) >= 8:
            baseline = statistics.median(history)
            z = robust_z(value, history)
            if exceeds(value, baseline, 0.10, 250_000_000) and z >= 5:
                streak = self.state["host_slow_streak"].get(str(rank), 0) + 1
                self.state["host_slow_streak"][str(rank)] = streak
                findings.append(self._host_finding(
                    "batch_latency", "sink_latency_spike", rank, first, last,
                    value, baseline, "low" if boundary else "medium",
                    robust_z=z, consecutive_slow_batches=streak, boundary_batch=boundary,
                    **test_fields))
            else:
                self.state["host_slow_streak"][str(rank)] = 0
        if not warmup and len(history) >= 12:
            recent = statistics.median(history[-3:] + [value])
            baseline = statistics.median(history[:-4])
            kind = None
            if exceeds(recent, baseline, 0.10, 250_000_000):
                kind = "drift" if gradual_drift(history[-11:] + [value], 0.10, 250_000_000) else "regression"
            key = str(rank)
            if kind:
                streak = self.state["host_trend_pending"].get(key, 0) + 1 if (
                    self.state["host_trend_kind"].get(key) == kind) else 1
                self.state["host_trend_pending"][key] = streak
                self.state["host_trend_kind"][key] = kind
                if streak >= 2 and self.state["host_trend_active"].get(key) != kind:
                    findings.append(self._host_finding(
                        "batch_trend", kind, rank, first, last, recent, baseline, "medium",
                        recent_batches=4, baseline_batches=len(history[:-4])))
                    self.state["host_trend_active"][key] = kind
            else:
                self.state["host_trend_pending"].pop(key, None)
                self.state["host_trend_kind"].pop(key, None)
                self.state["host_trend_active"].pop(key, None)
        if not warmup:
            history.append(value)
            del history[:-12]
        result = {
            "schema": "online_sink_diagnosis_v1", "rank": rank,
            "first_step": first, "last_step": last,
            "metric": "host_sink_batch_duration_ns",
            "duration_ns": value, "boundary_batch": boundary,
            "warmup_batch": warmup,
            "event_at_unix_ns": event["event_at_unix_ns"],
            "published_at_unix_ns": time.time_ns(), "findings": findings,
            **test_fields,
        }
        atomic_json(self.output / "sinks" / f"step_{last:09d}" / f"rank_{rank}.json", result)
        self.new_findings += len(findings)
        self.state["finding_total"] = self.new_findings
        pending = self.state["pending_sinks"].setdefault(str(last), {})
        pending[str(rank)] = {"first_step": first, "duration_ns": value,
                              "event_at_unix_ns": event["event_at_unix_ns"],
                              "boundary": boundary, **test_fields}
        self._complete_peer_groups(last)

    def _complete_peer_groups(self, last):
        pending = self.state["pending_sinks"].get(str(last), {})
        done = self.state["peer_done"].setdefault(str(last), [])
        for index, group in enumerate(self.groups):
            if index in done or not all(str(rank) in pending for rank in group):
                continue
            findings = []
            for rank in group:
                value = pending[str(rank)]["duration_ns"]
                baseline = statistics.median(pending[str(other)]["duration_ns"]
                                             for other in group if other != rank)
                if exceeds(value, baseline, 0.10, 250_000_000):
                    test_fields = {}
                    if pending[str(rank)].get("test_injected_delay_seconds") is not None:
                        test_fields["test_injected_delay_seconds"] = pending[str(rank)]["test_injected_delay_seconds"]
                    findings.append(self._host_finding(
                        "peer_batch_latency", "same_batch_rank_outlier", rank,
                        pending[str(rank)]["first_step"], last, value, baseline,
                        "low" if pending[str(rank)]["boundary"] else "medium",
                        peer_group=group, **test_fields))
            result = {
                "schema": "online_peer_sink_diagnosis_v1", "last_step": last,
                "peer_group": group, "metric": "host_sink_batch_duration_ns",
                "published_at_unix_ns": time.time_ns(), "findings": findings,
            }
            atomic_json(self.output / "sinks" / f"step_{last:09d}" / f"peer_group_{index}.json", result)
            self.new_findings += len(findings)
            self.state["finding_total"] = self.new_findings
            done.append(index)
        if len(done) == len(self.groups):
            self.state["pending_sinks"].pop(str(last), None)
            self.state["peer_done"].pop(str(last), None)

    def _read_telemetry(self):
        changed = False
        for rank in self.expected:
            path = self.root / "metrics" / f"rank_{rank}.jsonl"
            if not path.is_file():
                continue
            offset = self.state["offsets"].get(str(rank), 0)
            with path.open("rb") as stream:
                stream.seek(offset)
                while True:
                    line = stream.readline()
                    if not line or not line.endswith(b"\n"):
                        break
                    try:
                        event = json.loads(line)
                    except json.JSONDecodeError:
                        self._quality("INVALID_TELEMETRY_JSON", rank=rank, last_step=0)
                        self.state["offsets"][str(rank)] = stream.tell()
                        changed = True
                        continue
                    if event.get("schema") != "sink_telemetry_v1" or event.get("rank") != rank:
                        self._quality("INVALID_TELEMETRY", rank=rank, last_step=0)
                    elif event["sequence"] > self.state["last_sequences"].get(str(rank), 0):
                        if event["sequence"] != self.state["last_sequences"].get(str(rank), 0) + 1:
                            self._quality("TELEMETRY_SEQUENCE_GAP", rank=rank,
                                          last_step=event.get("last_step", 0))
                        kind = event["type"]
                        if kind == "sink_complete":
                            self._process_sink(event)
                        elif kind in ("heartbeat", "producer_done"):
                            self.state["last_heartbeat_unix_ns"][str(rank)] = event["event_at_unix_ns"]
                            dropped = event.get("dropped", 0)
                            if dropped > self.state["last_dropped"].get(str(rank), 0):
                                self._quality("TELEMETRY_DROPPED", rank=rank,
                                              last_step=event["last_completed_step"], dropped=dropped)
                            self.state["last_dropped"][str(rank)] = dropped
                            if kind == "producer_done":
                                self.state["producer_done"][str(rank)] = True
                        self.state["last_sequences"][str(rank)] = event["sequence"]
                    offset = stream.tell()
                    self.state["offsets"][str(rank)] = offset
                    changed = True
        return changed

    def _check_progress(self, now_ns):
        changed = False
        for index, group in enumerate(self.groups):
            progress = {rank: self.state["last_sink_step"].get(str(rank), 0) for rank in group}
            if not all(progress.values()):
                continue
            maximum = max(progress.values())
            for rank in group:
                last_event = self.state["last_progress_unix_ns"].get(str(rank), now_ns)
                if (maximum - progress[rank] < self.sink_size
                        or now_ns - last_event < 30_000_000_000
                        or self.state["producer_done"].get(str(rank))):
                    continue
                key = str(rank)
                if self.state["progress_alerted"].get(key) == progress[rank]:
                    continue
                self.state["progress_alerted"][key] = progress[rank]
                changed = True
                event = {
                    "schema": "online_progress_diagnosis_v1", "kind": "progress_lag",
                    "rank": rank, "peer_group": group, "last_completed_step": progress[rank],
                    "peer_max_step": maximum, "stale_seconds": (now_ns - last_event) / 1e9,
                    "confidence": "low", "published_at_unix_ns": now_ns,
                }
                atomic_json(self.output / "progress" / f"rank_{rank}_step_{progress[rank]}.json", event)
                self.new_findings += 1
                self.state["finding_total"] = self.new_findings
        return changed

    def _check_peer_timeouts(self, now_ns):
        changed = False
        for last, pending in self.state["pending_sinks"].items():
            if not pending:
                continue
            first_event = min(item["event_at_unix_ns"] for item in pending.values())
            if now_ns - first_event < 30_000_000_000:
                continue
            for index, group in enumerate(self.groups):
                if index in self.state["peer_done"].get(last, []) or all(
                        str(rank) in pending for rank in group):
                    continue
                changed = self._quality("PEER_SINK_TIMEOUT", last_step=int(last),
                                        peer_group=group) or changed
        return changed

    def _process_window(self, segment, state):
        result_path = self.fast_dir / f"segment_{segment:06d}" / "window_result.json"
        window_result = read_json(result_path)
        previous = self.state["device_state"]
        if state == "complete":
            path = result_path.with_name("window_summary.json")
            windows, quality, _ = load_windows(
                self.fast_dir, [path],
                prior_segment=self.state["last_segment"] if self.state["last_segment"] >= 0 else None,
                prior_last=previous.get("prior_last") if previous else None,
            )
            report, device_state = replay(self.fast_dir, self.groups, windows=windows,
                                           issues=quality, initial_state=previous, return_state=True)
            self.state["device_state"] = device_state
            quality_issues = report["quality_issues"]
            findings = report["findings"]
        else:
            quality_issues = [{"code": "FAST_WINDOW_FAILED", "segment": segment,
                               "detail": window_result.get("error")}]
            findings = []
            report = None
            self.state["device_state"] = None
        published = time.time_ns()
        result = {
            "schema": "online_window_diagnosis_v1", "segment": segment,
            "source_window_state": state, "metric": "device_step_duration_ns",
            "window_finished_at": window_result.get("finished_at"),
            "published_at_unix_ns": published,
            "findings": findings, "quality_issues": quality_issues,
            "coverage": report["coverage"] if report else None,
        }
        atomic_json(self.output / "windows" / f"segment_{segment:06d}.json", result)
        self.new_findings += len(findings)
        self.new_quality += len(quality_issues)
        self.state["finding_total"] = self.new_findings
        self.state["quality_total"] = self.new_quality
        self.state["last_segment"] = segment

    def tick(self, window_events=()):
        if not self.enabled or self.failed:
            return
        try:
            changed = self._read_telemetry()
            for segment, state in window_events:
                if segment > self.state["last_segment"]:
                    self.pending_windows[segment] = state
            while self.state["last_segment"] + 1 in self.pending_windows:
                segment = self.state["last_segment"] + 1
                state = self.pending_windows.pop(segment)
                self._process_window(segment, state)
                changed = True
            now_ns = time.time_ns()
            changed = self._check_progress(now_ns) or changed
            changed = self._check_peer_timeouts(now_ns) or changed
            if changed:
                self._save()
        except Exception as error:
            self.failed = repr(error)
            atomic_json(self.output / "failure.json", {"error": self.failed, "at": time.time()})

    def finish(self):
        if not self.enabled or self.failed:
            return
        self.tick()
        if self.failed:
            return
        try:
            self._finish_impl()
        except Exception as error:
            self.failed = repr(error)
            atomic_json(self.output / "failure.json", {"error": self.failed, "at": time.time()})

    def _finish_impl(self):
        if self.pending_windows:
            for segment in range(self.state["last_segment"] + 1, max(self.pending_windows) + 1):
                if segment not in self.pending_windows:
                    self._quality("FAST_WINDOW_MISSING", segment=segment)
                    self.state["device_state"] = None
                    self.state["last_segment"] = segment
                else:
                    self._process_window(segment, self.pending_windows.pop(segment))
        for last, pending in list(self.state["pending_sinks"].items()):
            for index, group in enumerate(self.groups):
                if index not in self.state["peer_done"].get(last, []) and not all(
                        str(rank) in pending for rank in group):
                    self._quality("PEER_SINK_INCOMPLETE", last_step=int(last),
                                  peer_group=group)
        self.state["pending_sinks"] = {}
        self.state["peer_done"] = {}
        target = self.output / "telemetry"
        target.mkdir(exist_ok=True)
        for rank in self.expected:
            path = self.root / "metrics" / f"rank_{rank}.jsonl"
            if path.is_file():
                shutil.copyfile(path, target / path.name)
            if not self.state["producer_done"].get(str(rank)):
                self._quality("TELEMETRY_INCOMPLETE", rank=rank,
                              last_step=self.state["last_sink_step"].get(str(rank), 0))
        self._save()

    def snapshot(self):
        if not self.enabled:
            return {"diagnosis_enabled": False}
        return {"diagnosis_enabled": True, "diagnosis_failed": self.failed,
                "diagnosis_last_segment": self.state["last_segment"],
                "diagnosis_findings": self.new_findings,
                "diagnosis_quality_issues": self.new_quality,
                "diagnosis_pending_windows": len(self.pending_windows)}
