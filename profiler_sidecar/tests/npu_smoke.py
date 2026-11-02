"""8-rank real collector/HCCL test; each segment must contain actual NPU work."""
import argparse
import os
from pathlib import Path
import signal
import sys
import time

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
sys.path.insert(0,os.environ['MF_DIR'])
import mindspore as ms
from mindspore import Tensor, nn, ops
from mindspore.communication import init, get_rank, get_group_size
from mindspore.train.callback import RunContext
import numpy as np
import yaml
from callback import SidecarProfileMonitor
from common import atomic_json, read_json

parser = argparse.ArgumentParser()
parser.add_argument('--config')
args = parser.parse_args()
cfg = yaml.safe_load(Path(args.config).read_text())
ms.set_context(device_target='Ascend', mode=ms.GRAPH_MODE, **cfg.get('context',{}))
init()
rank = get_rank()

class Net(nn.Cell):
    def __init__(self):
        super().__init__()
        self.reduce = ops.AllReduce()
    def construct(self,x,y):
        return self.reduce(ops.relu(ops.matmul(x,y)+1))

net = Net()
x = Tensor(np.ones((256,256),np.float32))
y = Tensor(np.eye(256,dtype=np.float32))
net(x,y).asnumpy()  # Explicit warmup; measured capture covers all steps below.
root=Path(os.environ['CYF_PROFILE_ROOT'])
if rank == 0 and cfg.get('smoke_freeze_worker'):
    worker=read_json(root/'status.json')['process']
    os.kill(worker['pid'],signal.SIGSTOP)
    deadline=time.monotonic()+5
    while Path(f"/proc/{worker['pid']}/stat").read_text().rsplit(')',1)[1].split()[0] != 'T':
        if time.monotonic()>deadline:
            raise RuntimeError('Test could not suspend the sidecar')
        time.sleep(.01)
    status=read_json(root/'status.json')
    status.update(heartbeat=time.time()-120,fatal='injected persistent-storage stall')
    atomic_json(root/'status.json',status)
    atomic_json(root/'test_worker_frozen.json',worker)
monitor = SidecarProfileMonitor(start_profile=True,profile_communication=True,
                                profile_memory=cfg.get('profile_memory',False),
                                profiler_level=cfg.get('profiler_level',0))
class Parameters(dict):
    __getattr__ = dict.__getitem__
    __setattr__ = dict.__setitem__
params = Parameters(cur_step_num=0)
context = RunContext(params)
timings=[]
resources=[]
started=time.monotonic()
for step in range(1,cfg.get('smoke_steps',6)+1):
    params.cur_step_num = step
    if step == cfg.get('smoke_capacity_at_step'):
        # Exercise the actual callback's capacity path without filling tmpfs.
        monitor.settings=dict(monitor.settings,min_ram_free_gib=1e9)
    begin=time.monotonic()
    monitor.step_begin(context)
    value=net(x,y).asnumpy()
    assert np.all(value==2*get_group_size())
    time.sleep(cfg.get('smoke_step_sleep',.1))
    monitor.step_end(context)
    timings.append(time.monotonic()-begin)
    if step % read_json(Path(os.environ['CYF_PROFILE_ROOT'])/'config.json')['window_steps'] == 0:
        status = Path('/proc/self/status').read_text()
        rss = next(line for line in status.splitlines() if line.startswith('VmRSS:'))
        fds = list(Path('/proc/self/fd').iterdir())
        deleted = []
        for fd in fds:
            try:
                name = os.readlink(fd)
                if 'cyf-profile' in name and '(deleted)' in name:
                    deleted.append(name)
            except FileNotFoundError:
                pass
        resources.append({'step':step,'rss_kib':int(rss.split()[1]),'fds':len(fds),
                          'deleted_capture_fds':deleted})
monitor.end(context)
root=Path(os.environ['CYF_PROFILE_ROOT'])
out=Path(read_json(root/'config.json')['archive'])
atomic_json(out/f'smoke_rank_{rank}.json',{'rank':rank,'step_seconds':timings,
                                        'total_seconds':time.monotonic()-started,
                                        'resources':resources},durable=True)
