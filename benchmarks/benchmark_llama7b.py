#!/usr/bin/env python3
"""Run 20-step baseline/profile jobs sequentially and retain timings and logs."""

import argparse
import copy
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import resource
import shlex
import subprocess
import time

import yaml


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--mf-dir', type=Path, default=Path('/root/work/cyf/mindformers_r1.3.2'))
    parser.add_argument('--config', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--modes', nargs='+', choices=['no_profile', 'profile'],
                        default=['no_profile', 'profile'])
    args = parser.parse_args()
    mf_dir = args.mf_dir.resolve()
    config_path = args.config or (
        Path(__file__).resolve().parent.parent
        / 'configs/llama2/pretrain_llama2_7b_20steps.yaml'
    )
    config = yaml.safe_load(config_path.read_text())
    root = args.output.resolve()
    root.mkdir(parents=True, exist_ok=False)
    (root / 'source_config.yaml').write_text(config_path.read_text())
    soft, hard = resource.getrlimit(resource.RLIMIT_NOFILE)
    resource.setrlimit(resource.RLIMIT_NOFILE, (min(2000000, hard), hard))
    results = []
    for index, mode in enumerate(args.modes):
        directory = root / mode
        directory.mkdir()
        (directory / 'events').mkdir()
        local_config = copy.deepcopy(config)
        local_config['profile'] = mode == 'profile'
        local_config['output_dir'] = str(directory / 'output')
        local_config['profile_output'] = str(directory / 'traces')
        local_config['context']['enable_compile_cache'] = True
        cache = Path(local_config['context'].get(
            'compile_cache_path',
            str(Path(__file__).resolve().parent.parent / 'compile_cache/llama7b/standard')
        )).resolve()
        cache.mkdir(parents=True, exist_ok=True)
        local_config['context']['compile_cache_path'] = str(cache)
        local_config['context']['save_graphs_path'] = str(directory / 'graphs')
        cfg = directory / 'config.yaml'
        cfg.write_text(yaml.safe_dump(local_config, sort_keys=False))
        env = os.environ.copy()
        env['MF_DIR'] = str(mf_dir)
        env['LOG_MF_PATH'] = str(directory / 'logs')
        env['BENCHMARK_EVENT_DIR'] = str(directory / 'events')
        env['MS_COMPILER_CACHE_PATH'] = str(cache)
        env['PYTHONUNBUFFERED'] = '1'
        entry = Path(__file__).with_name('benchmark_entry.py').resolve()
        train_command = shlex.join([str(entry), '--config', str(cfg)])
        command = ['bash', 'scripts/msrun_launcher.sh', train_command, '8', str(18218 + index),
                   str(directory / 'msrun_log'), 'True', '7200']
        record = {'mode': mode, 'command': command, 'config': str(cfg),
                  'started_at': datetime.now(timezone.utc).isoformat()}
        (directory / 'run.json').write_text(json.dumps(record, indent=2) + '\n')
        started = time.monotonic()
        print(f'Starting {mode}: {directory}', flush=True)
        with (directory / 'launcher.log').open('w') as log:
            process = subprocess.Popen(command, cwd=mf_dir, env=env,
                                       stdout=log, stderr=subprocess.STDOUT)
            record['launcher_pid'] = process.pid
            (directory / 'run.json').write_text(json.dumps(record, indent=2) + '\n')
            record['returncode'] = process.wait()
        record['elapsed_seconds'] = time.monotonic() - started
        record['finished_at'] = datetime.now(timezone.utc).isoformat()
        (directory / 'run.json').write_text(json.dumps(record, indent=2) + '\n')
        results.append(record)
        (root / 'runs.json').write_text(json.dumps(results, indent=2) + '\n')
        print(json.dumps(record, indent=2), flush=True)
        if record['returncode']:
            raise SystemExit(record['returncode'])


if __name__ == '__main__':
    main()
