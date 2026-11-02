"""Run the standard training entry with lightweight timing for comparison."""

import json
import os
from pathlib import Path
import runpy
import time

import mindspore as ms
from mindformers.core.callback.callback import MFLossMonitor, ProfileMonitor
from mindformers.tools.utils import get_real_rank


def event(name, **values):
    path = Path(os.environ['BENCHMARK_EVENT_DIR']) / f'rank_{get_real_rank()}.jsonl'
    with path.open('a') as stream:
        stream.write(json.dumps({'event': name, 'time': time.time(),
                                 'monotonic': time.monotonic(), **values}) + '\n')


original_train = ms.Model.train
original_begin = MFLossMonitor.step_begin
original_end = MFLossMonitor.step_end
original_profile_end = ProfileMonitor.step_end


def timed_train(self, *args, **kwargs):
    event('model_train_begin')
    try:
        result = original_train(self, *args, **kwargs)
    except BaseException as exc:
        event('model_train_error', error=repr(exc))
        raise
    event('model_train_end')
    return result


def timed_begin(self, context):
    event('step_begin', step=int(context.original_args().cur_step_num))
    return original_begin(self, context)


def timed_end(self, context):
    event('step_end', step=int(context.original_args().cur_step_num))
    return original_end(self, context)


def timed_profile_end(self, context):
    finalizing = context.original_args().cur_step_num == self.stop_step and self.profiler is not None
    if finalizing:
        event('profile_finalize_begin')
    result = original_profile_end(self, context)
    if finalizing:
        event('profile_finalize_end')
    return result


ms.Model.train = timed_train
MFLossMonitor.step_begin = timed_begin
MFLossMonitor.step_end = timed_end
ProfileMonitor.step_end = timed_profile_end
event('entry_ready')
runpy.run_path(str(Path(os.environ['MF_DIR']) / 'run_mindformer.py'), run_name='__main__')
event('entry_finished')
