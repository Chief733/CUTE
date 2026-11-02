"""Check the installed MindSpore/CANN collector's segment lifecycle on a real NPU."""
import json
import os
from pathlib import Path
import time

import mindspore as ms
from mindspore import Tensor, nn, ops, Profiler
from mindspore.profiler import ProfilerLevel
import numpy as np

ms.set_context(mode=ms.GRAPH_MODE, device_target="Ascend", device_id=0)

class Net(nn.Cell):
    def construct(self, x, y):
        return ops.relu(ops.matmul(x, y) + 1)

net = Net()
x = Tensor(np.ones((256, 256), np.float32))
y = Tensor(np.eye(256, dtype=np.float32))
net(x, y).asnumpy()
root = Path(os.environ['PROBE_ROOT'])
root.mkdir(parents=True, exist_ok=False)
records = []
for segment in range(3):
    path = root / f'segment_{segment:06d}' / 'rank_0'
    before = time.monotonic()
    profiler = Profiler(output_path=str(path), start_profile=True,
                        profiler_level=ProfilerLevel.Level0, profile_memory=False,
                        profile_framework=None, data_simplification=False)
    started = time.monotonic()
    for step in range(4):
        value = net(x, y).asnumpy()
        assert np.all(value == 2)
    computed = time.monotonic()
    profiler.stop()
    stopped = time.monotonic()
    profiler._ascend_profiler.finalize()
    Profiler._has_initialized = False
    finalized = time.monotonic()
    records.append({'segment': segment, 'start_seconds': started-before,
                    'compute_seconds': computed-started,
                    'stop_seconds': stopped-computed,
                    'finalize_seconds': finalized-stopped,
                    'bytes': sum(f.stat().st_size for f in path.rglob('*') if f.is_file())})
    print(json.dumps(records[-1]), flush=True)
(root / 'probe.json').write_text(json.dumps(records, indent=2))
