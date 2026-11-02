"""Measure import separately from selected timeline export on the same real raw data."""
import json
from pathlib import Path
import shutil
import subprocess
import sys
import time

source=Path(sys.argv[1])
output=Path(sys.argv[2])
output.mkdir(parents=True,exist_ok=False)
raw=next((source/'profiler').glob('PROF_*'))
job=output/raw.name
started=time.monotonic()
shutil.copytree(raw,job)
record={'source':str(source),'job':str(job),'copy_seconds':time.monotonic()-started,
        'raw_bytes':sum(p.stat().st_size for p in job.rglob('*') if p.is_file())}
assert not list(job.rglob('*.db')),'Input should be unparsed original data'
here=Path(__file__).resolve().parent
commands={
    'import':[sys.executable,'/usr/local/Ascend/ascend-toolkit/latest/tools/profiler/profiler_tool/analysis/msprof/msprof.py','import','-dir',str(job)],
    'selected_export':[sys.executable,str(here/'cann_selected_export.py'),str(job)]}
for phase,command in commands.items():
    started=time.monotonic()
    with (output/(phase+'.log')).open('w') as log:
        result=subprocess.run(command,stdout=log,stderr=log)
    record[phase+'_seconds']=time.monotonic()-started
    record[phase+'_returncode']=result.returncode
    (output/'timings.json').write_text(json.dumps(record,indent=2))
    print(phase,record,flush=True)
    if result.returncode:
        raise SystemExit(result.returncode)
record['json_files']=[{'path':str(p),'bytes':p.stat().st_size} for p in job.rglob('*.json') if 'timeline' in p.parts or 'mindstudio_profiler_output' in p.parts]
(output/'timings.json').write_text(json.dumps(record,indent=2))
