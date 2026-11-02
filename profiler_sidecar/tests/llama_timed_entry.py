"""Real-model validation entry with the same step timing as the earlier baseline."""
import os
from pathlib import Path
import runpy
import sys

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
sys.path.insert(0,os.environ['MF_DIR'])
from callback import install
from common import read_json
install()
root=Path(os.environ['CYF_PROFILE_ROOT'])
events=Path(read_json(root/'config.json')['archive'])/'events'
events.mkdir(exist_ok=True)
os.environ['BENCHMARK_EVENT_DIR']=str(events)
runpy.run_path('/root/work/cyf/pretrain/benchmarks/benchmark_entry.py',run_name='__main__')
