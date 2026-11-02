#!/usr/bin/env python3
"""Run and summarize matched 40-step Llama2-7B profiler overhead trials."""

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import random
import resource
import shlex
import statistics
import subprocess
import sys
import time

import yaml

SIDECAR = Path(__file__).resolve().parents[1]
PRETRAIN = SIDECAR.parent
MF = Path('/root/work/cyf/mindformers_r1.3.2')
PYTHON = Path('/root/miniconda3/envs/mindspore2.4_python3.10/bin/python')
SOURCE = PRETRAIN / 'configs/llama2/pretrain_llama2_7b_sidecar_20steps.yaml'
ENTRY = Path(__file__).with_name('comparison_entry.py')
SEQUENCE = 'NAB B AN NBA ABN ABN'.replace(' ', '')  # Pilot NAB, then four warm A/B pairs.
WINDOW_STEPS = 6
STOP_STEP = 40
START_STEP = 3

sys.path.insert(0, str(SIDECAR))
from launch import cache_specification  # noqa: E402
from inspect_run import inspect as inspect_sidecar  # noqa: E402


def save(path, value):
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + '\n')


def prepare_no_profile(run_dir):
    cfg = yaml.safe_load(SOURCE.read_text())
    cfg['profile'] = False
    cfg['output_dir'] = str(run_dir / 'training_output')
    cfg['profile_output'] = str(run_dir / 'unused_profile_output')
    cfg['callbacks'] = [c for c in cfg['callbacks'] if c['type'] != 'CheckpointMonitor']
    for callback in cfg['callbacks']:
        if callback['type'] == 'TrainCallBack':
            callback['stop_step'] = STOP_STEP
    spec = cache_specification(cfg, 8)
    spec['sidecar_graph_instrumentation'] = 'no_profiler_control_v1'
    key = hashlib.sha256(json.dumps(spec, sort_keys=True).encode()).hexdigest()[:20]
    cache = PRETRAIN / 'compile_cache' / 'llama7b' / ('no_profile_' + key)
    cache.mkdir(parents=True, exist_ok=True)
    save(cache / 'graph_specification.json', spec)
    cfg['context']['enable_compile_cache'] = True
    cfg['context']['compile_cache_path'] = str(cache)
    cfg['context']['save_graphs_path'] = str(run_dir / 'graphs')
    config_path = run_dir / 'train_config.yaml'
    config_path.write_text(yaml.safe_dump(cfg, sort_keys=False))
    return config_path, cache


def wait_sidecar(run_dir, timeout=900):
    deadline = time.monotonic() + timeout
    status_path = run_dir / 'status.json'
    while time.monotonic() < deadline:
        if status_path.exists():
            status = json.loads(status_path.read_text())
            if status['state'] == 'complete' and status['raw_bytes'] == 0:
                result = inspect_sidecar(run_dir, first=START_STEP, last=STOP_STEP, ranks=8)
                save(run_dir / 'verification.json', result)
                if result['errors']:
                    raise RuntimeError(f'{run_dir}: sidecar verification failed: {result["errors"]}')
                return result
            if status['state'] in ('failed', 'incomplete'):
                raise RuntimeError(f'{run_dir}: sidecar state {status["state"]}')
        time.sleep(2)
    raise TimeoutError(f'{run_dir}: sidecar did not drain within {timeout}s')


def run_one(root, index, mode):
    run_dir = root / f'{index:02d}_{mode}'
    if run_dir.exists():
        record_path = run_dir / 'trial.json'
        if record_path.exists() and json.loads(record_path.read_text()).get('state') == 'complete':
            print(f'Skip completed trial {run_dir}', flush=True)
            return
        raise RuntimeError(f'Existing incomplete trial requires inspection: {run_dir}')
    run_dir.mkdir(parents=True)
    env = dict(os.environ, PYTHONUNBUFFERED='1', MF_DIR=str(MF),
               CYF_COMPARE_MODE='offload' if mode in 'AB' else 'no_profile',
               CYF_COMPARE_OUTPUT=str(run_dir), LOG_MF_PATH=str(run_dir / 'logs'))
    env.pop('CYF_PROFILE_ROOT', None)
    port = 18640 + index
    if mode == 'N':
        cfg, cache = prepare_no_profile(run_dir)
        env['MS_COMPILER_CACHE_PATH'] = str(cache)
        train = shlex.join([str(ENTRY), '--config', str(cfg)])
        command = ['bash', 'scripts/msrun_launcher.sh', train, '8', str(port),
                   str(run_dir / 'msrun_log'), 'True', '7200']
        cwd = MF
    else:
        command = [str(PYTHON), str(SIDECAR / 'launch.py'),
                   '--config', str(SOURCE), '--output', str(run_dir / 'profile'),
                   '--training-stop-step', str(STOP_STEP), '--window-steps', str(WINDOW_STEPS),
                   '--disable-checkpoint', '--port', str(port), '--entry', str(ENTRY)]
        if mode == 'A':
            command.append('--archive-only')
        else:
            command += ['--fast-analyse', '--no-official-analyse', '--fast-workers', '8',
                        '--require-complete-step-trace']
        # comparison_entry writes its timing events to the trial root.
        cwd = SIDECAR
    record = {'mode': mode, 'index': index, 'state': 'running', 'command': command,
              'started_at': time.time()}
    save(run_dir / 'trial.json', record)
    print(f'Start {index:02d} {mode}: {run_dir}', flush=True)
    started = time.monotonic()
    with (run_dir / 'launcher.log').open('w') as log:
        proc = subprocess.Popen(command, cwd=cwd, env=env, stdout=log, stderr=subprocess.STDOUT)
        record['pid'] = proc.pid
        save(run_dir / 'trial.json', record)
        code = proc.wait()
    record.update(returncode=code, elapsed_seconds=time.monotonic()-started,
                  finished_at=time.time())
    save(run_dir / 'trial.json', record)
    if code:
        raise RuntimeError(f'Trial {index:02d} {mode} exited {code}; see {run_dir}/launcher.log')
    if mode in 'AB':
        wait_sidecar(run_dir / 'profile')
    # Fail early if instrumentation or the 40-step run was incomplete.
    measure_one(run_dir, mode)
    record['state'] = 'complete'
    save(run_dir / 'trial.json', record)
    print(f'Complete {index:02d} {mode}: {record["elapsed_seconds"]:.1f}s', flush=True)


def measure_one(run_dir, mode):
    ranks = []
    for rank in range(8):
        path = run_dir / 'events' / f'rank_{rank}.jsonl'
        events = [json.loads(line) for line in path.read_text().splitlines()]
        ends = {int(e['step']): e['monotonic'] for e in events
                if e['event'] == 'loss_step_end_end'}
        if sorted(ends) != list(range(2, STOP_STEP + 1, 2)):
            raise ValueError(f'{path}: expected every sink batch through step {STOP_STEP}')
        train_begin = next(e['monotonic'] for e in events if e['event'] == 'model_train_begin')
        train_end = next(e['monotonic'] for e in events if e['event'] == 'model_train_end')
        ranks.append({'rank': rank, 'steady_seconds': ends[STOP_STEP] - ends[2],
                      'steady_seconds_per_step': (ends[STOP_STEP] - ends[2]) / (STOP_STEP - 2),
                      'model_train_seconds': train_end - train_begin,
                      'last_step': max(ends)})
    result = {'mode': mode, 'run': str(run_dir), 'ranks': ranks,
              'steady_seconds_per_step': max(r['steady_seconds_per_step'] for r in ranks),
              'mean_rank_seconds_per_step': statistics.mean(r['steady_seconds_per_step'] for r in ranks),
              'model_train_seconds': max(r['model_train_seconds'] for r in ranks)}
    if mode in 'AB':
        result['sidecar_coverage'] = json.loads((run_dir / 'profile' / 'verification.json').read_text())['coverage']
    save(run_dir / 'measurement.json', result)
    return result


def bootstrap_ci(values, statistic=statistics.mean, samples=20000):
    rng = random.Random(20261021)
    draws = sorted(statistic([rng.choice(values) for _ in values]) for _ in range(samples))
    return [draws[int(0.025 * samples)], draws[int(0.975 * samples)]]


def summarize(root, sequence):
    values = []
    for index, mode in enumerate(sequence):
        run_dir = root / f'{index:02d}_{mode}'
        record = json.loads((run_dir / 'trial.json').read_text())
        if record.get('state') != 'complete' or record.get('returncode') != 0:
            raise RuntimeError(f'Incomplete run {run_dir}')
        values.append(measure_one(run_dir, mode))
    pairs = []
    i = 0
    while i < len(sequence) - 1:
        if sequence[i:i+2] in ('AB', 'BA'):
            a = values[i] if sequence[i] == 'A' else values[i+1]
            b = values[i] if sequence[i] == 'B' else values[i+1]
            pairs.append({'indices': [i, i+1], 'order': sequence[i:i+2],
                          'a_seconds_per_step': a['steady_seconds_per_step'],
                          'b_seconds_per_step': b['steady_seconds_per_step'],
                          'fast_increment_percent': (b['steady_seconds_per_step'] /
                                                     a['steady_seconds_per_step'] - 1) * 100})
            i += 2
        else:
            i += 1
    # The first N and A were cold-cache trials. Keep the entire first N/A/B
    # block as a pilot, and use only subsequent warm-cache runs for estimates.
    pilot = {'N': values[0]['steady_seconds_per_step'],
             'A': values[1]['steady_seconds_per_step'],
             'B': values[2]['steady_seconds_per_step']}
    pilot_pair = pairs.pop(0)
    by_mode = {mode: [v['steady_seconds_per_step'] for v in values[3:] if v['mode'] == mode]
               for mode in 'NAB'}
    differences = [p['fast_increment_percent'] for p in pairs]
    if not differences or not by_mode['N'] or not by_mode['A'] or not by_mode['B']:
        raise ValueError('Summarizing requires at least one N, A and B run and one adjacent A/B pair')
    # With only four matched pairs, a Student-t interval is more candid about
    # sampling uncertainty than a percentile bootstrap over four observations.
    paired_t_95 = None
    if len(differences) == 4:
        margin = 3.182446305284263 * statistics.stdev(differences) / math.sqrt(4)
        paired_t_95 = [statistics.mean(differences) - margin,
                       statistics.mean(differences) + margin]
    result = {'sequence': sequence, 'window_steps': WINDOW_STEPS,
              'training_steps': STOP_STEP, 'measured_steps': '3-40',
              'pilot_seconds_per_step': pilot, 'pilot_pairwise': pilot_pair,
              'seconds_per_step': by_mode, 'pairwise': pairs,
              'fast_increment_mean_percent': statistics.mean(differences),
              'fast_increment_bootstrap_95_percent': bootstrap_ci(differences),
              'fast_increment_student_t_95_percent': paired_t_95,
              'full_pipeline_vs_no_profile_median_percent':
                  (statistics.median(by_mode['B']) / statistics.median(by_mode['N']) - 1) * 100,
              'archive_vs_no_profile_median_percent':
                  (statistics.median(by_mode['A']) / statistics.median(by_mode['N']) - 1) * 100}
    save(root / 'summary.json', result)
    print(json.dumps(result, indent=2, ensure_ascii=False), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=['run', 'summarize'])
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--sequence', default=SEQUENCE, help='N=no profile, A=archive only, B=fast analyse')
    args = parser.parse_args()
    sequence = args.sequence.upper()
    if any(mode not in 'NAB' for mode in sequence):
        parser.error('Sequence may contain only N, A, B')
    root = args.output.resolve()
    if args.action == 'run':
        root.mkdir(parents=True, exist_ok=True)
        _, hard = resource.getrlimit(resource.RLIMIT_NOFILE)
        resource.setrlimit(resource.RLIMIT_NOFILE, (min(2000000, hard), hard))
        save(root / 'experiment.json', {'sequence': sequence, 'source': str(SOURCE),
                                        'training_steps': STOP_STEP, 'window_steps': WINDOW_STEPS})
        for index, mode in enumerate(sequence):
            run_one(root, index, mode)
    summarize(root, sequence)


if __name__ == '__main__':
    main()
