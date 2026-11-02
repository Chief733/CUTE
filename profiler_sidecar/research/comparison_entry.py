"""Identical training entry/dependency graph for every profiling comparison."""
import json
import os
from pathlib import Path
import runpy
import sys
import time

HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE))
sys.path.insert(0, os.environ['MF_DIR'])

import mindspore as ms
from mindformers.core.callback.callback import MFLossMonitor
import callback as sidecar_callback

events = []


def event(name, **values):
    events.append(dict(event=name, time=time.time(), monotonic=time.monotonic(), **values))


def wrap_method(cls, name, label, step=False):
    original = getattr(cls, name)
    def measured(self, *args, **kwargs):
        values = {'step': int(args[0].original_args().cur_step_num)} if step else {}
        event(label+'_begin', **values)
        try:
            return original(self, *args, **kwargs)
        finally:
            event(label+'_end', **values)
    setattr(cls, name, measured)


wrap_method(ms.Model, 'train', 'model_train')
wrap_method(ms.Profiler, 'stop', 'profiler_stop')
wrap_method(ms.Profiler, 'analyse', 'profiler_analyse')
wrap_method(MFLossMonitor, 'step_begin', 'loss_step_begin', step=True)
wrap_method(MFLossMonitor, 'step_end', 'loss_step_end', step=True)
if os.environ['CYF_COMPARE_MODE'] == 'offload':
    sidecar_callback.install()

event('entry_ready')
try:
    runpy.run_path(str(Path(os.environ['MF_DIR'])/'run_mindformer.py'), run_name='__main__')
    event('entry_finished')
except BaseException as exc:
    event('entry_failed', error=repr(exc))
    raise
finally:
    # Keep timing metadata in memory during training; write it only on exit.
    rank = os.environ.get('RANK_ID', 'scheduler')
    path = Path(os.environ['CYF_COMPARE_OUTPUT'])/'events'/f'rank_{rank}.jsonl'
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(''.join(json.dumps(e)+'\n' for e in events))
