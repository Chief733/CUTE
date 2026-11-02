"""Real two-rank NPU tests: frozen consumers and exhausted capture budget.

Run with the MindSpore interpreter after sourcing CANN. Test-only SIGSTOP is
always followed by SIGCONT, including on failure. No production delay knobs.
"""
import argparse
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

import yaml

HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE))
from common import atomic_json, is_alive, read_json, trace_events
from inspect_run import inspect


def wait_for(predicate, timeout=180):
    deadline = time.monotonic()+timeout
    while time.monotonic() < deadline:
        value = predicate()
        if value:
            return value
        time.sleep(.1)
    raise TimeoutError('Integration test deadline exceeded')


def signal_process(identity, sig):
    if is_alive(identity):
        os.kill(identity['pid'], sig)


def verify_captured_work(output):
    """Check actual device work, not only nonempty JSON or successful parsing."""
    rows = []
    for path in sorted((output/'segments').glob('segment_*/rank_*/parse_result.json')):
        parsed = read_json(path)
        captured = read_json(path.parent/'ready.json')
        steps = captured['last_step']-captured['first_step']+1
        matmuls = reductions = 0
        for event in trace_events(parsed['timeline']):
            if event.get('ph') != 'X':
                continue
            name = event.get('name', '')
            matmuls += name.endswith('_MatMulV2')
            reductions += name.startswith('hcom_allReduce__')
        assert matmuls == steps, (path, matmuls, steps)
        assert reductions == steps, (path, reductions, steps)
        rows.append({'rank': captured['rank'], 'segment': captured['segment'],
                     'steps': steps, 'device_matmul': matmuls,
                     'device_allreduce': reductions})
    return rows


def run_case(output, pressure, port):
    config = {'profile': True, 'profile_memory': False, 'profiler_level': 1,
              'context': {'enable_compile_cache': True},
              'runner_config': {'sink_size': 1}, 'smoke_steps': 6,
              'smoke_step_sleep': .1, 'smoke_freeze_worker': not pressure}
    if pressure:
        config['smoke_capacity_at_step'] = 3
    cfg = output.with_suffix('.yaml')
    cfg.write_text(yaml.safe_dump(config))
    cmd = [sys.executable, str(HERE/'launch.py'), '--config', str(cfg),
           '--output', str(output), '--workers', '2', '--window-steps', '2',
           '--port', str(port), '--entry', str(HERE/'tests/npu_smoke.py')]
    worker = parser = None
    root = None
    record = {'case': 'buffer_pressure' if pressure else 'frozen_consumers'}
    with output.with_suffix('.log').open('w') as log:
        launch = subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT)
        try:
            wait_for(lambda: (output/'sidecar_config.json').exists())
            root = Path(read_json(output/'sidecar_config.json')['ram_root'])
            wait_for(lambda: (root/'status.json').exists())
            worker = read_json(root/'status.json')['process']
            wait_for(lambda: launch.poll() is not None)
            assert launch.returncode == 0, f'Training failed; see {output}/launcher.log'
            training = read_json(output/'training_run.json')
            record['training_seconds'] = training['elapsed_seconds']
            for rank in range(2):
                assert len(read_json(output/f'smoke_rank_{rank}.json')['step_seconds']) == 6
            if not pressure:
                assert (root/'test_worker_frozen.json').exists()
                assert is_alive(worker)
                state = Path(f"/proc/{worker['pid']}/stat").read_text().rsplit(')',1)[1].split()[0]
                assert state == 'T', state
                assert len(list((root/'raw').glob('segment_*/rank_*/ready.json'))) == 6
                assert not list((output/'segments').glob('segment_*/rank_*/archive.json'))
                record['training_exited_with_writer_frozen'] = True
                print(json.dumps(record), flush=True)
                signal_process(worker, signal.SIGCONT)
                parser = wait_for(lambda: read_json(root/'status.json').get('parser_process'))
                signal_process(parser, signal.SIGSTOP)
                wait_for(lambda: read_json(root/'status.json')['archived_segments'] == 6
                         and read_json(root/'status.json')['raw_bytes'] == 0)
                assert is_alive(parser)
                assert read_json(root/'status.json')['parse_pending'] > 0
                record['archival_drained_with_parser_frozen'] = True
                print(json.dumps(record), flush=True)
                signal_process(parser, signal.SIGCONT)
            wait_for(lambda: not is_alive(worker), timeout=240)
            checked = inspect(output, verify=True, first=1, last=2 if pressure else 6, ranks=2)
            if pressure:
                assert checked['state'] == 'incomplete', checked
                assert len(checked['status']['capture_interruptions']) == 2, checked
                assert checked['errors'] == ['Sidecar not completely drained: incomplete'], checked
                record['training_continued_after_capture_disabled'] = True
            else:
                assert not checked['errors'], checked
            atomic_json(output/'verification.json', checked, durable=True)
            record.update(result='passed', raw_bytes_verified=checked['raw_bytes_verified'],
                          parse_states=checked['parse_states'],
                          captured_work=verify_captured_work(output))
            atomic_json(output/'decoupling_test.json', record, durable=True)
            print(json.dumps(record), flush=True)
            return record
        finally:
            if root and (root/'test_worker_frozen.json').exists():
                signal_process(read_json(root/'test_worker_frozen.json'), signal.SIGCONT)
            signal_process(worker, signal.SIGCONT)
            signal_process(parser, signal.SIGCONT)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--output', type=Path, required=True)
    args = ap.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    records = [run_case(args.output/'frozen', False, 18308),
               run_case(args.output/'pressure', True, 18309)]
    atomic_json(args.output/'results.json', records, durable=True)


if __name__ == '__main__':
    main()
