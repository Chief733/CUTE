"""Prepare and run one isolated warmup/baseline/offload/default profiling case."""
import argparse
import copy
import hashlib
import json
import os
from pathlib import Path
import resource
import shlex
import shutil
import subprocess
import sys
import time
import uuid

import yaml

HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE))
from common import atomic_json, is_alive, process_identity, read_json
from launch import cache_specification


def prepare(source, output):
    output.mkdir(parents=True, exist_ok=False)
    cfg = yaml.safe_load(source.read_text())
    # Capture options are identical for offload/default. The stock callback
    # requires init_start_profile=True when communication tracing is enabled.
    cfg.update(profile=False, init_start_profile=True, profile_start_step=1,
               profile_stop_step=20, profile_communication=True,
               profile_memory=False, profiler_level=0, with_stack=False,
               data_simplification=False)
    for cb in cfg['callbacks']:
        if cb['type'] == 'TrainCallBack':
            cb['stop_step'] = 20
    assert any(c.get('stop_step') == 20 for c in cfg['callbacks'])
    spec = cache_specification(cfg, 8)
    key = hashlib.sha256(json.dumps(spec, sort_keys=True).encode()).hexdigest()[:20]
    cache = HERE.parent/'compile_cache'/'comparison'/key
    cache.mkdir(parents=True, exist_ok=True)
    cfg['context'].update(enable_compile_cache=True, compile_cache_path=str(cache))
    (output/'source_config.yaml').write_text(source.read_text())
    (output/'common.yaml').write_text(yaml.safe_dump(cfg, sort_keys=False))
    atomic_json(output/'comparison.json', dict(model=cfg['trainer']['model_name'],
                source=str(source.resolve()), cache=str(cache), cache_key=key,
                world_size=8, window_steps=4, entry=str(Path(__file__).with_name('comparison_entry.py'))), durable=True)
    atomic_json(cache/'graph_specification.json', spec, durable=True)
    print(json.dumps(read_json(output/'comparison.json')), flush=True)


def run(output, mode):
    metadata = read_json(output/'comparison.json')
    cfg = yaml.safe_load((output/'common.yaml').read_text())
    directory = output/mode
    directory.mkdir(exist_ok=False)
    cfg.update(profile=mode in ('offload','default'), output_dir=str(directory/'output'),
               profile_output=str(directory/'traces'))
    if mode.startswith('warmup'):
        cfg['callbacks'] = [c for c in cfg['callbacks'] if c['type'] != 'CheckpointMonitor']
        for cb in cfg['callbacks']:
            if cb['type'] == 'TrainCallBack':
                cb['stop_step'] = 2
    root = sidecar = None
    env = dict(os.environ, MF_DIR='/root/work/cyf/mindformers_r1.3.2',
               MS_COMPILER_CACHE_PATH=metadata['cache'],
               CYF_COMPARE_MODE=mode, CYF_COMPARE_OUTPUT=str(directory),
               LOG_MF_PATH=str(directory/'framework_logs'), PYTHONUNBUFFERED='1')
    env.pop('CYF_PROFILE_ROOT', None)
    if mode == 'offload':
        root = Path('/dev/shm/cyf-profile')/('compare_'+uuid.uuid4().hex[:12])
        root.mkdir(parents=True)
        sidecar_cfg = dict(archive=str(directory), ram_root=str(root),
                           launcher=process_identity(), window_steps=metadata['window_steps'],
                           world_size=8, max_ram_gib=64, min_ram_free_gib=8,
                           min_disk_free_gib=32, parser_memory_gib=64, parser_cpus=8,
                           tmpfs_used_at_start=shutil.disk_usage(root).used, parse=True)
        atomic_json(root/'config.json', sidecar_cfg)
        atomic_json(directory/'sidecar_config.json', sidecar_cfg, durable=True)
        with (directory/'sidecar.log').open('w') as log:
            sidecar = subprocess.Popen([sys.executable, str(HERE/'worker.py'), 'run', str(root)],
                                       stdin=subprocess.DEVNULL, stdout=log, stderr=log,
                                       start_new_session=True)
        deadline = time.monotonic()+20
        while not (root/'status.json').exists():
            if sidecar.poll() is not None or time.monotonic() > deadline:
                raise RuntimeError('Sidecar startup failed')
            time.sleep(.1)
        cfg['profile_output'] = str(root/'raw')
        env['CYF_PROFILE_ROOT'] = str(root)
    config = directory/'train_config.yaml'
    config.write_text(yaml.safe_dump(cfg, sort_keys=False))
    command = ['bash', 'scripts/msrun_launcher.sh',
               shlex.join([metadata['entry'], '--config', str(config)]),
               '8', '18418', str(directory/'log/worker'), 'True', '7200']
    _, hard = resource.getrlimit(resource.RLIMIT_NOFILE)
    resource.setrlimit(resource.RLIMIT_NOFILE, (min(2000000, hard), hard))
    record = dict(mode=mode, started_at=time.time(), command=command,
                  cache=metadata['cache'], sidecar=process_identity(sidecar.pid) if sidecar else None,
                  ram_root=str(root) if root else None)
    atomic_json(directory/'training_run.json', record, durable=True)
    begin = time.monotonic()
    try:
        with (directory/'launcher.log').open('w') as log:
            training = subprocess.Popen(command, cwd=env['MF_DIR'], env=env,
                                        stdout=log, stderr=subprocess.STDOUT)
            record['training_launcher'] = process_identity(training.pid)
            atomic_json(directory/'training_run.json', record, durable=True)
            record['returncode'] = training.wait()
    finally:
        record.update(finished_at=time.time(), elapsed_seconds=time.monotonic()-begin)
        if root and (root/'status.json').exists():
            atomic_json(directory/'status_at_training_exit.json', read_json(root/'status.json'), durable=True)
        atomic_json(directory/'training_run.json', record, durable=True)
        if root:
            atomic_json(root/'launcher_done.json', record)
    print(json.dumps(record), flush=True)
    raise SystemExit(record['returncode'])


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('action', choices=['prepare','run'])
    p.add_argument('--source', type=Path)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--mode', choices=['warmup','warmup_verify','no_profile','offload','default'])
    args = p.parse_args()
    if args.action == 'prepare':
        prepare(args.source.resolve(), args.output.resolve())
    else:
        run(args.output.resolve(), args.mode)


if __name__ == '__main__':
    main()
