"""Install the sidecar callback before running the unmodified training entry."""
import os
from pathlib import Path
import runpy
import sys

mf_dir = Path(os.environ.get('MF_DIR', '/root/work/cyf/mindformers_r1.3.2'))
sys.path.insert(0, str(mf_dir))
from callback import install
install()
runpy.run_path(str(mf_dir/'run_mindformer.py'), run_name='__main__')
