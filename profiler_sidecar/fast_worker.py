"""Bounded subprocess manager for the direct raw-profile parser.

This module intentionally has no MindSpore/CANN Python imports.  A sealed rank
directory is immutable, so the parser and the durable archive copier may read
it concurrently.  Generated summaries live outside the immutable raw archive.
"""
from collections import Counter, deque
import os
from pathlib import Path
import re
import signal
import subprocess
import sys
import time

from common import atomic_json, is_alive, process_identity, read_json


SEGMENT_RE = re.compile(r"segment_(\d+)")
RANK_RE = re.compile(r"rank_(\d+)")
SUMMARY_RE = re.compile(r"rank_(\d+)\.json")
RESULT_RE = re.compile(r"rank_(\d+)\.result\.json")
PROCESS_RE = re.compile(r"rank_(\d+)\.process\.json")


def input_key(path):
    """Return (segment, rank) from a canonical rank-segment directory."""
    path = Path(path)
    segment = SEGMENT_RE.fullmatch(path.parent.name)
    rank = RANK_RE.fullmatch(path.name)
    if not segment or not rank:
        raise ValueError(f"not a rank segment directory: {path}")
    return int(segment.group(1)), int(rank.group(1))


class FastAnalysisManager:
    """Launch/recover bounded rank parsers and cross-rank aggregators."""

    def __init__(self, root, archive, config):
        self.root = Path(root)
        self.archive = Path(archive)
        self.config = config
        self.enabled = bool(config.get("fast_analyse"))
        self.output_root = self.archive / "fast"
        self.binary = Path(config.get("fast_binary", ""))
        self.aggregate_script = Path(config.get("fast_aggregate_script", ""))
        self.max_workers = int(config.get("fast_workers", 8))
        self.timeout = float(config.get("fast_timeout_seconds", 60.0))
        self.top_k = int(config.get("fast_top_k", 20))
        self.ratio_threshold = float(config.get("fast_ratio_threshold", 1.2))
        self.require_complete_step_trace = bool(
            config.get("fast_require_complete_step_trace", False)
        )
        self.expected_ranks = tuple(
            sorted(int(rank) for rank in config.get(
                "fast_expected_rank_ids", range(int(config.get("world_size", 1)))
            ))
        )
        self.pending = deque()
        self.pending_sources = {}
        self.active = {}
        self.states = {}
        self.counts = Counter()
        self.known_segments = set()
        self.aggregate_pending = deque()
        self.aggregate_queued = set()
        self.aggregate_active = {}
        self.window_states = {}
        self.window_counts = Counter()
        self.window_events = deque()
        self.launch_index = 0
        self.cpus = sorted(os.sched_getaffinity(0))
        if not self.enabled:
            return
        if self.max_workers <= 0 or self.timeout <= 0 or self.top_k < 0:
            raise ValueError("invalid fast-analysis worker, timeout, or top-k setting")
        if not self.expected_ranks or len(set(self.expected_ranks)) != len(self.expected_ranks):
            raise ValueError("fast_expected_rank_ids must be a non-empty unique list")
        if not self.binary.is_file() or not os.access(self.binary, os.X_OK):
            raise FileNotFoundError(f"fast parser is absent or not executable: {self.binary}")
        if not self.aggregate_script.is_file():
            raise FileNotFoundError(f"fast aggregation script is absent: {self.aggregate_script}")
        self.output_root.mkdir(parents=True, exist_ok=True)
        self._bootstrap()

    def _rank_paths(self, key):
        segment, rank = key
        directory = self.output_root / f"segment_{segment:06d}"
        return {
            "directory": directory,
            "summary": directory / f"rank_{rank}.json",
            "result": directory / f"rank_{rank}.result.json",
            "process": directory / f"rank_{rank}.process.json",
            "log": directory / f"rank_{rank}.log",
        }

    def _window_paths(self, segment):
        directory = self.output_root / f"segment_{segment:06d}"
        return {
            "directory": directory,
            "summary": directory / "window_summary.json",
            "result": directory / "window_result.json",
            "process": directory / "window_process.json",
            "log": directory / "window.log",
        }

    def _archived_input(self, key):
        segment, rank = key
        return self.archive / "segments" / f"segment_{segment:06d}" / f"rank_{rank}"

    def _set_state(self, key, state):
        previous = self.states.get(key)
        if previous == state:
            return
        if previous:
            self.counts[previous] -= 1
        self.states[key] = state
        self.counts[state] += 1
        self.known_segments.add(key[0])

    def _set_window_state(self, segment, state):
        previous = self.window_states.get(segment)
        if previous == state:
            return
        if previous:
            self.window_counts[previous] -= 1
        self.window_states[segment] = state
        self.window_counts[state] += 1
        self.known_segments.add(segment)
        self.window_events.append((segment, state))

    def _validate_summary(self, key):
        path = self._rank_paths(key)["summary"]
        value = read_json(path)
        if value.get("schema") != "fast_summary_v1":
            raise ValueError(f"unexpected fast summary schema in {path}")
        source = value.get("source", {})
        if (source.get("segment"), source.get("rank")) != key:
            raise ValueError(f"fast summary identity mismatch in {path}")
        if self.require_complete_step_trace:
            first = source.get("declared_first_step")
            last = source.get("declared_last_step")
            if not isinstance(first, int) or not isinstance(last, int) or last < first:
                raise ValueError(f"invalid declared step range in {path}")
            expected_ids = list(range(first, last + 1))
            actual_ids = [step.get("global_step") for step in value.get("steps", ())]
            coverage = value.get("coverage", {})
            if (actual_ids != expected_ids
                    or coverage.get("step_pairs") != len(expected_ids)
                    or coverage.get("step_unmatched_starts") != 0
                    or coverage.get("step_unmatched_ends") != 0
                    or not coverage.get("step_trace_complete")):
                raise ValueError(
                    f"incomplete per-step trace in {path}: expected global steps "
                    f"{expected_ids}, got {actual_ids}"
                )
        return value

    def _validate_window(self, segment):
        path = self._window_paths(segment)["summary"]
        value = read_json(path)
        if value.get("schema") != "fast_window_summary_v1":
            raise ValueError(f"unexpected fast window schema in {path}")
        if value.get("segment") != segment or not value.get("complete"):
            raise ValueError(f"incomplete or mismatched fast window summary in {path}")
        if set(value.get("rank_ids", ())) != set(self.expected_ranks):
            raise ValueError(f"unexpected rank set in {path}")
        return value

    def _write_rank_result(self, key, state, **fields):
        value = dict(
            schema="fast_rank_result_v1",
            state=state,
            segment=key[0],
            rank=key[1],
            finished_at=time.time(),
            **fields,
        )
        atomic_json(self._rank_paths(key)["result"], value, durable=True)
        self._set_state(key, state)

    def _write_window_result(self, segment, state, **fields):
        value = dict(
            schema="fast_window_result_v1",
            state=state,
            segment=segment,
            finished_at=time.time(),
            **fields,
        )
        atomic_json(self._window_paths(segment)["result"], value, durable=True)
        self._set_window_state(segment, state)

    def _bootstrap(self):
        # First restore terminal states.  Exact filename matching prevents
        # process/result metadata from being mistaken for rank summaries.
        for path in sorted(self.output_root.glob("segment_*/rank_*.result.json")):
            segment_match = SEGMENT_RE.fullmatch(path.parent.name)
            rank_match = RESULT_RE.fullmatch(path.name)
            if not segment_match or not rank_match:
                continue
            key = int(segment_match.group(1)), int(rank_match.group(1))
            try:
                result = read_json(path)
                state = result["state"]
                if state not in ("complete", "failed"):
                    raise ValueError(f"invalid state {state!r}")
                if state == "complete":
                    self._validate_summary(key)
                self._set_state(key, state)
            except Exception as exc:
                self._write_rank_result(key, "failed", error=f"invalid persisted result: {exc!r}")

        # A parser can atomically publish its summary just before the coordinator
        # dies.  Recover that success rather than rerunning or calling it failed.
        for path in sorted(self.output_root.glob("segment_*/rank_*.json")):
            segment_match = SEGMENT_RE.fullmatch(path.parent.name)
            rank_match = SUMMARY_RE.fullmatch(path.name)
            if not segment_match or not rank_match:
                continue
            key = int(segment_match.group(1)), int(rank_match.group(1))
            if key in self.states:
                continue
            try:
                summary = self._validate_summary(key)
                self._write_rank_result(
                    key, "complete", recovered_after_restart=True,
                    parser_seconds=summary.get("parser", {}).get("parse_seconds"),
                )
            except Exception as exc:
                self._write_rank_result(key, "failed", error=f"invalid orphan summary: {exc!r}")

        for path in sorted(self.output_root.glob("segment_*/window_result.json")):
            segment_match = SEGMENT_RE.fullmatch(path.parent.name)
            if not segment_match:
                continue
            segment = int(segment_match.group(1))
            try:
                result = read_json(path)
                state = result["state"]
                if state not in ("complete", "failed"):
                    raise ValueError(f"invalid state {state!r}")
                if state == "complete":
                    self._validate_window(segment)
                self._set_window_state(segment, state)
            except Exception as exc:
                self._write_window_result(segment, "failed", error=f"invalid persisted result: {exc!r}")

        for path in sorted(self.output_root.glob("segment_*/window_summary.json")):
            segment_match = SEGMENT_RE.fullmatch(path.parent.name)
            if not segment_match:
                continue
            segment = int(segment_match.group(1))
            if segment in self.window_states:
                continue
            try:
                self._validate_window(segment)
                self._write_window_result(segment, "complete", recovered_after_restart=True)
            except Exception as exc:
                self._write_window_result(segment, "failed", error=f"invalid orphan summary: {exc!r}")

        # Adopt children which survived their coordinator.  A stale marker is
        # harmless; its input is queued below when no terminal result exists.
        for path in sorted(self.output_root.glob("segment_*/rank_*.process.json")):
            segment_match = SEGMENT_RE.fullmatch(path.parent.name)
            rank_match = PROCESS_RE.fullmatch(path.name)
            if not segment_match or not rank_match:
                continue
            key = int(segment_match.group(1)), int(rank_match.group(1))
            if key in self.states:
                continue
            try:
                marker = read_json(path)
                if is_alive(marker.get("process")):
                    self.active[key] = {
                        "process": None,
                        "identity": marker["process"],
                        "source": Path(marker["source"]),
                        "started_at": marker["started_at"],
                        "adopted": True,
                        "timeout_at": None,
                    }
            except (OSError, ValueError, KeyError):
                pass

        for path in sorted(self.output_root.glob("segment_*/window_process.json")):
            segment_match = SEGMENT_RE.fullmatch(path.parent.name)
            if not segment_match:
                continue
            segment = int(segment_match.group(1))
            if segment in self.window_states:
                continue
            try:
                marker = read_json(path)
                if is_alive(marker.get("process")):
                    self.aggregate_active[segment] = {
                        "process": None,
                        "identity": marker["process"],
                        "started_at": marker["started_at"],
                        "adopted": True,
                        "timeout_at": None,
                    }
            except (OSError, ValueError, KeyError):
                pass

        self.refresh([])
        for marker in sorted((self.archive / "segments").glob("segment_*/rank_*/archive.json")):
            self.enqueue(marker.parent)

    def enqueue(self, source):
        source = Path(source)
        try:
            key = input_key(source)
        except ValueError:
            return
        self.known_segments.add(key[0])
        if key in self.states or key in self.active:
            return
        if key in self.pending_sources:
            # Prefer tmpfs so online results are available before archival ends.
            if str(source).startswith(str(self.root / "raw")):
                self.pending_sources[key] = source
            return
        self.pending.append(key)
        self.pending_sources[key] = source

    def refresh(self, raw_ranks):
        if not self.enabled:
            return
        if not raw_ranks:
            raw_ranks = []
            raw_root = self.root / "raw"
            if raw_root.exists():
                for segment in raw_root.glob("segment_*"):
                    raw_ranks.extend(segment.glob("rank_*"))
        for source in raw_ranks:
            if (Path(source) / "ready.json").exists():
                self.enqueue(source)

    def _clean_environment(self):
        env = dict(
            os.environ,
            OMP_NUM_THREADS="1",
            OPENBLAS_NUM_THREADS="1",
            MKL_NUM_THREADS="1",
            ASCEND_VISIBLE_DEVICES="",
        )
        for key in (
            "MS_ROLE", "MS_NODE_ID", "MS_WORKER_NUM", "MS_SCHED_HOST",
            "MS_SCHED_PORT", "RANK_TABLE_FILE", "RANK_ID", "RANK_SIZE",
            "DEVICE_ID", "PROFILING_OPTIONS", "MS_PROFILER_OPTIONS",
        ):
            env.pop(key, None)
        return env

    def _launch_ranks(self):
        while self.pending and len(self.active) < self.max_workers:
            key = self.pending.popleft()
            source = self.pending_sources.pop(key)
            if key in self.states or key in self.active:
                continue
            if not (source / "ready.json").exists():
                source = self._archived_input(key)
            paths = self._rank_paths(key)
            paths["directory"].mkdir(parents=True, exist_ok=True)
            try:
                if not (source / "ready.json").is_file():
                    raise FileNotFoundError(f"sealed input disappeared: {source}")
                metadata = read_json(source / "ready.json")
                if (metadata.get("segment"), metadata.get("rank")) != key:
                    raise ValueError(f"ready.json identity mismatch in {source}")
                started_at = time.time()
                command = [
                    str(self.binary), "--top-k", str(self.top_k),
                    "--output", str(paths["summary"]), str(source),
                ]
                with paths["log"].open("ab") as log:
                    process = subprocess.Popen(
                        command,
                        stdin=subprocess.DEVNULL,
                        stdout=log,
                        stderr=log,
                        env=self._clean_environment(),
                        start_new_session=True,
                    )
                identity = process_identity(process.pid)
                marker = {
                    "process": identity,
                    "source": str(source),
                    "output": str(paths["summary"]),
                    "command": command,
                    "started_at": started_at,
                }
                atomic_json(paths["process"], marker, durable=True)
                try:
                    if self.cpus:
                        cpu = self.cpus[self.launch_index % len(self.cpus)]
                        os.sched_setaffinity(process.pid, {cpu})
                    os.setpriority(os.PRIO_PROCESS, process.pid, 10)
                except (OSError, ProcessLookupError):
                    pass
                self.launch_index += 1
                self.active[key] = {
                    "process": process,
                    "identity": identity,
                    "source": source,
                    "started_at": started_at,
                    "ready_at": metadata.get("ready_at"),
                    "adopted": False,
                    "timeout_at": None,
                }
            except Exception as exc:
                self._write_rank_result(
                    key, "failed", source=str(source), error=f"launch failed: {exc!r}"
                )

    @staticmethod
    def _job_alive(job):
        process = job.get("process")
        return process.poll() is None if process is not None else is_alive(job.get("identity"))

    @staticmethod
    def _signal_job(job, sig):
        identity = job.get("identity")
        if is_alive(identity):
            try:
                os.kill(identity["pid"], sig)
            except ProcessLookupError:
                pass

    def _poll_ranks(self):
        now = time.time()
        for key, job in list(self.active.items()):
            alive = self._job_alive(job)
            elapsed = now - job["started_at"]
            if alive and elapsed > self.timeout:
                if job["timeout_at"] is None:
                    job["timeout_at"] = now
                    self._signal_job(job, signal.SIGTERM)
                elif now - job["timeout_at"] > 5:
                    self._signal_job(job, signal.SIGKILL)
                continue
            if alive:
                continue
            process = job.get("process")
            returncode = process.poll() if process is not None else None
            fields = {
                "source": str(job["source"]),
                "process": job.get("identity"),
                "returncode": returncode,
                "adopted": job.get("adopted", False),
                "elapsed_seconds": max(0.0, now - job["started_at"]),
                "timed_out": job["timeout_at"] is not None,
            }
            if job.get("ready_at") is not None:
                fields["ready_to_result_seconds"] = max(0.0, now - job["ready_at"])
            try:
                summary = self._validate_summary(key)
                fields["parser_seconds"] = summary.get("parser", {}).get("parse_seconds")
                self._write_rank_result(key, "complete", **fields)
            except Exception as exc:
                fields["error"] = (
                    f"parser exited without a valid summary: {exc!r}; returncode={returncode}"
                )
                self._write_rank_result(key, "failed", **fields)
            del self.active[key]

    def _queue_aggregates(self):
        expected = set(self.expected_ranks)
        for segment in sorted(self.known_segments):
            if (segment in self.window_states or segment in self.aggregate_active
                    or segment in self.aggregate_queued):
                continue
            rank_states = {
                rank: self.states.get((segment, rank)) for rank in self.expected_ranks
            }
            if all(state == "complete" for state in rank_states.values()):
                self.aggregate_pending.append(segment)
                self.aggregate_queued.add(segment)
            elif expected and all(state in ("complete", "failed") for state in rank_states.values()):
                failed = sorted(rank for rank, state in rank_states.items() if state == "failed")
                self._write_window_result(
                    segment, "failed", error="one or more rank parsers failed", failed_ranks=failed
                )

    def _launch_aggregates(self):
        # One aggregator is enough: it only reads the compact rank JSON files.
        while self.aggregate_pending and not self.aggregate_active:
            segment = self.aggregate_pending.popleft()
            self.aggregate_queued.discard(segment)
            if segment in self.window_states:
                continue
            paths = self._window_paths(segment)
            started_at = time.time()
            command = [
                sys.executable,
                str(self.aggregate_script),
                str(paths["directory"]),
                "--expected-rank-ids", ",".join(map(str, self.expected_ranks)),
                "--ratio-threshold", str(self.ratio_threshold),
                "--output", str(paths["summary"]),
            ]
            try:
                with paths["log"].open("ab") as log:
                    process = subprocess.Popen(
                        command,
                        stdin=subprocess.DEVNULL,
                        stdout=log,
                        stderr=log,
                        env=self._clean_environment(),
                        start_new_session=True,
                    )
                identity = process_identity(process.pid)
                atomic_json(
                    paths["process"],
                    {"process": identity, "command": command, "started_at": started_at},
                    durable=True,
                )
                self.aggregate_active[segment] = {
                    "process": process,
                    "identity": identity,
                    "started_at": started_at,
                    "adopted": False,
                    "timeout_at": None,
                }
            except Exception as exc:
                self._write_window_result(segment, "failed", error=f"launch failed: {exc!r}")

    def _poll_aggregates(self):
        now = time.time()
        for segment, job in list(self.aggregate_active.items()):
            alive = self._job_alive(job)
            elapsed = now - job["started_at"]
            if alive and elapsed > self.timeout:
                if job["timeout_at"] is None:
                    job["timeout_at"] = now
                    self._signal_job(job, signal.SIGTERM)
                elif now - job["timeout_at"] > 5:
                    self._signal_job(job, signal.SIGKILL)
                continue
            if alive:
                continue
            process = job.get("process")
            returncode = process.poll() if process is not None else None
            fields = {
                "process": job.get("identity"),
                "returncode": returncode,
                "adopted": job.get("adopted", False),
                "elapsed_seconds": max(0.0, now - job["started_at"]),
                "timed_out": job["timeout_at"] is not None,
            }
            try:
                self._validate_window(segment)
                self._write_window_result(segment, "complete", **fields)
            except Exception as exc:
                fields["error"] = (
                    f"aggregator exited without a valid summary: {exc!r}; returncode={returncode}"
                )
                self._write_window_result(segment, "failed", **fields)
            del self.aggregate_active[segment]

    def tick(self):
        if not self.enabled:
            return
        self._poll_ranks()
        self._launch_ranks()
        self._queue_aggregates()
        self._poll_aggregates()
        self._launch_aggregates()

    def finish_partial_windows(self):
        """Close missing-rank windows only after all producers and jobs stop."""
        if not self.enabled or self.pending or self.active:
            return
        for segment in sorted(self.known_segments):
            if (segment in self.window_states or segment in self.aggregate_active
                    or segment in self.aggregate_queued):
                continue
            missing = [
                rank for rank in self.expected_ranks
                if self.states.get((segment, rank)) != "complete"
            ]
            if missing:
                self._write_window_result(
                    segment, "failed", error="window ended without every expected rank",
                    missing_or_failed_ranks=missing,
                )

    def state_for_input(self, source):
        if not self.enabled:
            return None
        try:
            return self.states.get(input_key(source))
        except ValueError:
            return None

    @property
    def drained(self):
        return not (self.pending or self.active or self.aggregate_pending or self.aggregate_active)

    @property
    def has_failures(self):
        return bool(self.counts["failed"] or self.window_counts["failed"])

    def snapshot(self):
        if not self.enabled:
            return {
                "fast_analyse_enabled": False,
                "fast_pending": 0,
                "fast_active": 0,
                "fast_counts": {},
                "fast_window_counts": {},
            }
        return {
            "fast_analyse_enabled": True,
            "fast_pending": len(self.pending),
            "fast_active": len(self.active),
            "fast_counts": dict(self.counts),
            "fast_window_pending": len(self.aggregate_pending),
            "fast_window_active": len(self.aggregate_active),
            "fast_window_counts": dict(self.window_counts),
            "fast_parser_processes": [
                job.get("identity") for job in self.active.values() if job.get("identity")
            ],
        }
