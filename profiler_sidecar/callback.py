"""MindSpore 2.4 producer: contiguous step windows, native flush into tmpfs only."""
import os
from pathlib import Path
import shutil
import time

import mindspore as ms
from mindspore.profiler.parser.profiler_info import ProfilerInfo
from mindspore.train.callback import Callback
from mindformers.core.callback.callback import ProfileMonitor
from mindformers.tools.utils import get_real_rank, get_pipeline_rank_ids

from common import atomic_json, capture_capacity_problem, process_identity, read_json
from native import DumperCleanup
from telemetry import TelemetryEmitter


class SidecarProfileMonitor(ProfileMonitor):
    """Same constructor as ProfileMonitor; stop_step is replaced by window_steps.

    No analysis or persistent-filesystem writes occur in this callback. Closing a
    window still synchronizes the NPU and finalizes the native collector; that
    remaining producer cost is measured in each ready.json. No step waits for a
    sidecar heartbeat, durable-write acknowledgement, or parser completion.
    """
    def __init__(self, start_step=1, stop_step=10, output_path=None,
                 start_profile=True, profile_rank_ids=None, profile_pipeline=False,
                 profile_communication=False, profile_memory=False, config=None,
                 profiler_level=0, with_stack=False, data_simplification=True, **kwargs):
        Callback.__init__(self)
        if ms.__version__ != '2.4.0':
            raise RuntimeError('This private native-finalize adapter is verified only for MindSpore 2.4.0')
        if ms.get_context('device_target') != 'Ascend':
            raise RuntimeError('Sidecar capture requires Ascend; the sidecar itself uses CPU')
        if profile_memory and not start_profile:
            raise ValueError('Memory tracing needs init_start_profile=True; use profile_memory=False for capture after warmup')
        self.root = Path(os.environ['CYF_PROFILE_ROOT'])
        self.settings = read_json(self.root/'config.json')
        self.rank = get_real_rank()
        self.sink_size = int(self.settings.get('sink_size', 1))
        self.telemetry = (TelemetryEmitter(self.root, self.rank)
                          if self.settings.get('online_diagnosis') else None)
        self.batch_start_ns = None
        self.started_this_batch = False
        self.profile_rank_ids = profile_rank_ids
        self.pipeline_rank_ids = get_pipeline_rank_ids() if profile_pipeline else None
        self.enabled = self._is_profile_required(self.rank)
        self.start_step = start_step
        self.window_steps = self.settings['window_steps']
        self.config = config
        self.segment = 0
        self.last_step = 0
        self.test_host_delay = self.settings.get('test_host_delay')
        self.test_host_delay_applied = False
        self.profiler = None
        self.profiler_active = False
        self.preinitialized = False
        self.current_path = None
        self.prepare_seconds = 0.0
        self.capture_disabled_reason = None
        self.client = {'process': process_identity(), 'rank': self.rank,
                       'enabled': self.enabled, 'done': False}
        if self.test_host_delay and self.rank == int(self.test_host_delay['rank']):
            self.client['test_host_delay'] = self.test_host_delay
        self.client_path = self.root/'clients'/f'rank_{self.rank}.json'
        atomic_json(self.client_path, self.client)
        self.options = dict(profile_communication=bool(profile_communication),
                            profile_memory=bool(profile_memory),
                            profiler_level=self._get_profiler_level(profiler_level),
                            with_stack=bool(with_stack), data_simplification=False,
                            **kwargs)
        if self.enabled:
            self.dumper_cleanup = DumperCleanup()
            if start_profile:
                self._start_segment(1)
            else:
                # Profiler initialization must happen before the first graph is
                # compiled.  In graph/sink mode this is what makes MindSpore
                # insert the 60000/60001 per-step trace points.  Collection is
                # still delayed until start_step, so warmup is not captured.
                self._prepare_first_segment()

    def _check_capacity(self):
        if self.capture_disabled_reason:
            return False
        problem = capture_capacity_problem(self.root, self.settings)
        if problem:
            # Seal data already collected; only profiling stops. There is no
            # retry loop or wait for the consumer to free space.
            if self.profiler_active:
                self._seal_segment()
            else:
                self._discard_preinitialized_segment()
            self.capture_disabled_reason = problem
            self.client.update(capture_disabled_reason=problem,
                               disabled_after_step=self.last_step,
                               capture_complete=False)
            atomic_json(self.client_path, self.client)
            return False
        return True

    def _allocate_segment_path(self):
        self.current_path = self.root/'raw'/f'segment_{self.segment:06d}'/f'rank_{self.rank}'
        self.current_path.mkdir(parents=True, exist_ok=False)

    def _prepare_first_segment(self):
        """Initialize Ascend profiling before graph compilation without collecting."""
        if not self._check_capacity():
            return
        self._allocate_segment_path()
        prepared = time.monotonic()
        self.profiler = ms.Profiler(
            start_profile=False, output_path=str(self.current_path), **self.options
        )
        self.prepare_seconds = time.monotonic() - prepared
        self.preinitialized = True
        self._record_metadata(self.config)

    def _activate_preinitialized_segment(self, first_step):
        if not self.profiler or not self.preinitialized or self.profiler_active:
            raise RuntimeError('Profiler preinitialization state is inconsistent')
        self.first_step = first_step
        started = time.monotonic()
        self.profiler.start()
        self.start_seconds = time.monotonic() - started
        self.capture_started_at = time.time()
        self.preinitialized = False
        self.profiler_active = True

    def _start_segment(self, first_step):
        if not self._check_capacity():
            return
        self.first_step = first_step
        self._allocate_segment_path()
        started = time.monotonic()
        self.profiler = ms.Profiler(start_profile=True, output_path=str(self.current_path), **self.options)
        self.start_seconds = time.monotonic() - started
        self.prepare_seconds = 0.0
        self.capture_started_at = time.time()
        self.profiler_active = True
        self.preinitialized = False
        self._record_metadata(self.config)

    def _discard_preinitialized_segment(self):
        """Finalize an inactive collector and remove its unpublished directory."""
        if not self.profiler or not self.preinitialized or self.profiler_active:
            return
        self.profiler._ascend_profiler.finalize()
        self.dumper_cleanup.close_after_stop()
        ms.Profiler._has_initialized = False
        self.profiler = None
        self.preinitialized = False
        if self.current_path:
            shutil.rmtree(self.current_path, ignore_errors=True)
        self.current_path = None

    def _seal_segment(self):
        if not self.profiler or not self.profiler_active:
            return
        begin = time.monotonic()
        self.profiler.stop()
        stopped = time.monotonic()
        self.profiler_active = False
        # analyse() would finalize the collector and then export/parse in-process.
        # Only perform its native finalize step here, before handing off ownership.
        self.profiler._ascend_profiler.finalize()
        self.dumper_cleanup.close_after_stop()
        ms.Profiler._has_initialized = False
        finalized = time.monotonic()
        ProfilerInfo.set_rank_size(self.settings['world_size'])
        ProfilerInfo.set_parallel_info(ms.get_auto_parallel_context('parallel_mode'),
                                       ms.get_auto_parallel_context('pipeline_stages'))
        ProfilerInfo.save(self.profiler._output_path)
        marker = {'rank': self.rank, 'world_size': self.settings['world_size'],
                  'segment': self.segment, 'first_step': self.first_step,
                  'last_step': self.last_step, 'capture_started_at': self.capture_started_at,
                  'capture_stopped_at': time.time(), 'prepare_seconds': self.prepare_seconds,
                  'start_seconds': self.start_seconds,
                  'stop_seconds': stopped-begin, 'finalize_seconds': finalized-stopped,
                  'handoff': 'immutable_tmpfs_segment', 'ready_at': time.time(),
                  'metadata_seconds': time.monotonic()-finalized,
                  'data_process': self.profiler._data_process}
        # This atomic RAM publication transfers ownership. Neither disk commit
        # nor analysis is acknowledged back to the training process.
        atomic_json(self.current_path/'ready.json', marker)
        self.profiler = None
        self.preinitialized = False
        self.current_path = None
        self.segment += 1

    def step_begin(self, run_context):
        self.started_this_batch = False
        if self.enabled and self._check_capacity():
            # cur_step_num is the END of the incoming sink batch (e.g. 2,4,6).
            step = int(run_context.original_args().cur_step_num)
            if step >= self.start_step and not self.profiler_active:
                if self.preinitialized:
                    self._activate_preinitialized_segment(self.last_step + 1)
                    self.started_this_batch = True
                elif not self.profiler:
                    self._start_segment(self.last_step + 1)
                    self.started_this_batch = self.profiler_active
        if self.telemetry:
            # Exclude synchronous profiler start from the sink duration.
            self.batch_start_ns = time.monotonic_ns()

    def step_end(self, run_context):
        self.last_step = int(run_context.original_args().cur_step_num)
        seal = self.profiler_active and self.last_step - self.first_step + 1 >= self.window_steps
        injected_delay_seconds = None
        if (self.test_host_delay and not self.test_host_delay_applied
                and self.rank == int(self.test_host_delay['rank'])
                and self.last_step == int(self.test_host_delay['last_step'])):
            injected_delay_seconds = float(self.test_host_delay['seconds'])
            time.sleep(injected_delay_seconds)
            self.test_host_delay_applied = True
            self.client['test_host_delay_applied_at_step'] = self.last_step
            atomic_json(self.client_path, self.client)
        end_ns = time.monotonic_ns() if self.telemetry else None
        if self.telemetry:
            self.telemetry.sink_done(self.last_step - self.sink_size + 1, self.last_step,
                                     self.batch_start_ns, end_ns, self.started_this_batch, seal,
                                     warmup_batch=self.last_step <= self.sink_size,
                                     test_injected_delay_seconds=injected_delay_seconds)
            self.batch_start_ns = None
        if seal:
            self._seal_segment()

    def end(self, run_context):
        self._seal_segment()
        self._discard_preinitialized_segment()
        if self.telemetry:
            self.telemetry.close()
        self.client.update(done=True, last_step=self.last_step, segments=self.segment,
                           capture_complete=self.enabled and not self.capture_disabled_reason,
                           finished_at=time.time())
        if self.telemetry:
            self.client.update(telemetry_dropped=self.telemetry.dropped,
                               telemetry_max_queue_depth=self.telemetry.max_queue_depth,
                               telemetry_error=self.telemetry.error,
                               telemetry_writer_alive=self.telemetry.thread.is_alive())
        atomic_json(self.client_path, self.client)


def install():
    """Opt-in entry point; does not alter the installed MindFormers sources."""
    import importlib
    module = importlib.import_module('mindformers.trainer.trainer')
    module.ProfileMonitor = SidecarProfileMonitor
