"""Start a detached profiler sidecar and a normal 8-NPU MindFormers training job."""
import argparse
import copy
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import shlex
import shutil
import subprocess
import sys
import time
import uuid

import yaml
from common import GIB, atomic_json, is_alive, process_identity, read_json
from dashboard_runtime import start_dashboard


def cache_specification(config, workers, entry=None, entry_environment=None):
    """Keep graph-affecting YAML changes out of a previous graph's cache."""
    spec=copy.deepcopy(config)
    for key in list(spec):
        if key.startswith('profile') or key in ('init_start_profile','with_stack','data_simplification','output_dir'):
            spec.pop(key)
    context=spec.setdefault('context',{})
    for key in ('compile_cache_path','enable_compile_cache','save_graphs_path'):
        context.pop(key,None)
    spec['sidecar_world_size']=workers
    spec['verified_mindspore_version']='2.4.0'
    # Profiler initialization state affects graph step-trace instrumentation.
    # Bump this value whenever callback initialization semantics change so a
    # graph compiled without 60000/60001 markers is never silently reused.
    spec['sidecar_graph_instrumentation']='profiler_preinit_step_trace_v1'
    if entry is not None:
        entry = Path(entry).resolve()
        spec['sidecar_training_entry'] = {
            'path': str(entry),
            'sha256': hashlib.sha256(entry.read_bytes()).hexdigest(),
            'environment': dict(sorted((entry_environment or {}).items())),
        }
    return spec


def main():
    here = Path(__file__).resolve().parent
    fast_root = here.parent/'fastanalyse'
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--mf-dir', type=Path, default=Path('/root/work/cyf/mindformers_r1.3.2'))
    parser.add_argument('--window-steps', type=int, default=100)
    parser.add_argument('--workers', type=int, default=8)
    parser.add_argument('--port', type=int, default=18228)
    parser.add_argument('--max-ram-gib', type=float, default=64)
    parser.add_argument('--min-ram-free-gib', type=float, default=8)
    parser.add_argument('--min-disk-free-gib', type=float, default=32)
    parser.add_argument('--parser-memory-gib', type=float, default=64)
    parser.add_argument('--parser-cpus', type=int, default=8)
    parser.add_argument('--compile-cache-dir',type=Path,
                        default=Path('/root/work/cyf/pretrain/compile_cache/llama7b'))
    parser.add_argument('--training-stop-step', type=int,
                        help='Short validation run: stop training at this actual step')
    parser.add_argument('--disable-checkpoint', action='store_true',
                        help='Remove CheckpointMonitor for a disposable validation run')
    parser.add_argument('--archive-only', action='store_true')
    parser.add_argument('--no-official-analyse', action='store_true',
                        help='Disable slow MindSpore analyse while retaining archival')
    parser.add_argument('--fast-analyse', action='store_true',
                        help='Directly parse sealed raw segments concurrently with archival')
    parser.add_argument('--fast-workers', type=int, default=8)
    parser.add_argument('--fast-timeout-seconds', type=float, default=60)
    parser.add_argument('--fast-top-k', type=int, default=20)
    parser.add_argument('--fast-ratio-threshold', type=float, default=1.2)
    parser.add_argument('--fast-rank-ids',
                        help='Optional comma-separated ranks expected in each fast window')
    parser.add_argument('--require-complete-step-trace', action='store_true',
                        help='Fail each fast rank result unless every declared step has a device trace pair')
    parser.add_argument('--online-diagnosis', action='store_true',
                        help='Run incremental sink and window diagnosis in the sidecar')
    parser.add_argument('--diagnosis-peer-groups', default='',
                        help="Explicit same-role rank groups, e.g. '0,1,2,3,4,5,6,7'")
    parser.add_argument('--dashboard', dest='dashboard', action='store_true',
                        help='Start a detached live dashboard before training')
    parser.add_argument('--no-dashboard', dest='dashboard', action='store_false',
                        help='Do not start the live dashboard')
    parser.set_defaults(dashboard=False)
    parser.add_argument('--dashboard-port', type=int, default=0,
                        help='Dashboard port; zero selects a new free port from 18736-18835')
    parser.add_argument('--test-host-delay-rank', type=int,
                        help='Test only: rank whose callback will inject one host delay')
    parser.add_argument('--test-host-delay-step', type=int,
                        help='Test only: sink-ending step at which to inject the host delay')
    parser.add_argument('--test-host-delay-seconds', type=float,
                        help='Test only: duration of the injected host delay')
    parser.add_argument('--fast-binary', type=Path, default=fast_root/'fast_parse')
    parser.add_argument('--fast-aggregate-script', type=Path,
                        default=fast_root/'tools/aggregate_files.py')
    parser.add_argument('--entry', type=Path, help='Optional test entry that installs the callback itself')
    args = parser.parse_args()
    if (args.window_steps <= 0 or args.workers <= 0 or args.parser_cpus <= 0
            or args.fast_workers <= 0 or args.fast_timeout_seconds <= 0 or args.fast_top_k < 0):
        parser.error('window-steps, workers, parser-cpus, fast-workers and timeout must be positive; top-k cannot be negative')
    if args.fast_ratio_threshold <= 1:
        parser.error('fast-ratio-threshold must be greater than 1')
    if args.archive_only and args.fast_analyse:
        parser.error('--archive-only and --fast-analyse are mutually exclusive')
    for name in ('max_ram_gib','min_ram_free_gib','min_disk_free_gib','parser_memory_gib'):
        if getattr(args,name) <= 0:
            parser.error(name+' must be positive')
    cfg = yaml.safe_load(args.config.read_text())
    if not cfg.get('profile'):
        parser.error('The input YAML must have profile: True')
    sink = cfg.get('runner_config', {}).get('sink_size', 1)
    if args.training_stop_step is not None:
        if args.training_stop_step <= 0 or args.training_stop_step % sink:
            parser.error('training-stop-step must be positive and divisible by sink_size')
        train_callbacks = [
            callback for callback in cfg.get('callbacks', ())
            if callback.get('type') == 'TrainCallBack'
        ]
        if len(train_callbacks) != 1:
            parser.error('training-stop-step requires exactly one TrainCallBack')
        train_callbacks[0]['stop_step'] = args.training_stop_step
    if args.disable_checkpoint:
        cfg['callbacks'] = [
            callback for callback in cfg.get('callbacks', ())
            if callback.get('type') != 'CheckpointMonitor'
        ]
    if args.window_steps % sink:
        parser.error('window-steps must be divisible by sink_size')
    test_delay_values = (args.test_host_delay_rank, args.test_host_delay_step,
                         args.test_host_delay_seconds)
    if any(value is not None for value in test_delay_values):
        if not all(value is not None for value in test_delay_values):
            parser.error('test host delay requires rank, step, and seconds together')
        if not args.online_diagnosis:
            parser.error('test host delay requires --online-diagnosis')
        if args.test_host_delay_rank < 0 or args.test_host_delay_rank >= args.workers:
            parser.error('test host delay rank must be in [0, workers)')
        if args.test_host_delay_step <= 0 or args.test_host_delay_step % sink:
            parser.error('test host delay step must be positive and divisible by sink_size')
        if (args.training_stop_step is not None
                and args.test_host_delay_step > args.training_stop_step):
            parser.error('test host delay step cannot exceed training-stop-step')
        if args.test_host_delay_seconds <= 0 or args.test_host_delay_seconds > 60:
            parser.error('test host delay seconds must be in (0, 60]')
        test_host_delay = {
            'kind': 'host_callback_delay',
            'rank': args.test_host_delay_rank,
            'last_step': args.test_host_delay_step,
            'seconds': args.test_host_delay_seconds,
        }
    else:
        test_host_delay = None
    if args.fast_rank_ids:
        try:
            fast_rank_ids = [int(value) for value in args.fast_rank_ids.split(',')]
        except ValueError:
            parser.error('--fast-rank-ids must be comma-separated integers')
    elif cfg.get('profile_rank_ids') is not None:
        fast_rank_ids = [int(value) for value in cfg['profile_rank_ids']]
    else:
        fast_rank_ids = list(range(args.workers))
    if (not fast_rank_ids or len(set(fast_rank_ids)) != len(fast_rank_ids)
            or any(rank < 0 or rank >= args.workers for rank in fast_rank_ids)):
        parser.error('fast rank IDs must be unique values in [0, workers)')
    peer_groups = []
    if args.online_diagnosis:
        if not args.fast_analyse or not args.require_complete_step_trace:
            parser.error('--online-diagnosis requires --fast-analyse and --require-complete-step-trace')
        if not args.diagnosis_peer_groups:
            parser.error('--online-diagnosis requires explicit --diagnosis-peer-groups')
        used = set()
        try:
            for text in args.diagnosis_peer_groups.split(';'):
                group = [int(rank) for rank in text.split(',')]
                if len(group) < 3 or len(set(group)) != len(group) or used.intersection(group):
                    raise ValueError('groups must have at least three unique ranks and cannot overlap')
                if not set(group).issubset(fast_rank_ids):
                    raise ValueError('groups must contain only fast rank IDs')
                peer_groups.append(sorted(group))
                used.update(group)
        except ValueError as error:
            parser.error('invalid --diagnosis-peer-groups: ' + str(error))
    if args.fast_analyse:
        if not args.fast_binary.is_file() or not os.access(args.fast_binary, os.X_OK):
            parser.error('fast parser is absent or not executable: '+str(args.fast_binary))
        if not args.fast_aggregate_script.is_file():
            parser.error('fast aggregate script is absent: '+str(args.fast_aggregate_script))
    default_entry = (here/'training_entry.py').resolve()
    entry = (args.entry or default_entry).resolve()
    if not entry.is_file():
        parser.error('training entry does not exist: '+str(entry))
    archive = args.output.resolve()
    archive.mkdir(parents=True, exist_ok=False)
    run_id = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S') + '_' + uuid.uuid4().hex[:8]
    root = Path('/dev/shm/cyf-profile')/run_id
    root.mkdir(parents=True)
    config = {'archive':str(archive), 'ram_root':str(root), 'window_steps':args.window_steps,
              'sink_size':sink, 'online_diagnosis':args.online_diagnosis,
              'diagnosis_peer_groups':peer_groups,
              'test_host_delay':test_host_delay,
              'dashboard_enabled':args.dashboard,
              'dashboard_snapshot':str(archive/'DASHBOARD.html') if args.dashboard else None,
              'tmpfs_used_at_start':shutil.disk_usage(root).used,
              'world_size':args.workers, 'max_ram_gib':args.max_ram_gib,
              'min_ram_free_gib':args.min_ram_free_gib, 'min_disk_free_gib':args.min_disk_free_gib,
              'parser_memory_gib':args.parser_memory_gib, 'parser_cpus':args.parser_cpus,
              'parse':not args.archive_only and not args.no_official_analyse,
              'fast_analyse':args.fast_analyse,
              'fast_workers':args.fast_workers,
              'fast_timeout_seconds':args.fast_timeout_seconds,
              'fast_top_k':args.fast_top_k,
              'fast_ratio_threshold':args.fast_ratio_threshold,
              'fast_require_complete_step_trace':args.require_complete_step_trace,
              'fast_expected_rank_ids':fast_rank_ids,
              'fast_binary':str(args.fast_binary.resolve()),
              'fast_aggregate_script':str(args.fast_aggregate_script.resolve()),
              'launcher':process_identity()}
    atomic_json(root/'config.json',config)
    atomic_json(archive/'sidecar_config.json',config,durable=True)
    cache=None
    if cfg.get('context',{}).get('enable_compile_cache'):
        entry_environment = {}
        if args.entry and 'CYF_COMPARE_MODE' in os.environ:
            entry_environment['CYF_COMPARE_MODE'] = os.environ['CYF_COMPARE_MODE']
        # Preserve the established default-entry cache key. Custom entries get
        # isolated keys so their different Python dependency sets cannot
        # invalidate the normal training cache.
        spec=cache_specification(cfg,args.workers,
                                 entry=entry if args.entry else None,
                                 entry_environment=entry_environment)
        key=hashlib.sha256(json.dumps(spec,sort_keys=True).encode()).hexdigest()[:20]
        cache=args.compile_cache_dir.resolve()/key
        cache.mkdir(parents=True,exist_ok=True)
        cfg['context']['compile_cache_path']=str(cache)
        atomic_json(cache/'graph_specification.json',spec,durable=True)
        atomic_json(archive/'compile_cache.json',{'enabled':True,'path':str(cache),'key':key},durable=True)
    cfg['output_dir'] = str(archive/'training_output')
    cfg['profile_output'] = str(root/'raw')
    (archive/'source_config.yaml').write_text(args.config.read_text())
    (archive/'train_config.yaml').write_text(yaml.safe_dump(cfg,sort_keys=False))
    dashboard_record = None
    if args.dashboard:
        dashboard_process, dashboard_record = start_dashboard(
            archive, here/'dashboard_server.py', archive/'dashboard.log', args.dashboard_port)
        dashboard_record['process'] = process_identity(dashboard_process.pid)
        atomic_json(archive/'dashboard.json', dashboard_record, durable=True)
        (archive/'DASHBOARD_URL.txt').write_text(dashboard_record['url']+'\n')
        link = f"\033]8;;{dashboard_record['url']}\033\\打开 ARGUS 在线仪表盘\033]8;;\033\\"
        print('ARGUS dashboard:', dashboard_record['url'], flush=True)
        print(link, flush=True)
    with (archive/'sidecar.log').open('ab') as log:
        sidecar = subprocess.Popen([sys.executable,str(here/'worker.py'),'run',str(root)],
                                   stdin=subprocess.DEVNULL,stdout=log,stderr=log,start_new_session=True)
    deadline = time.monotonic()+20
    while not (root/'status.json').exists():
        if sidecar.poll() is not None:
            raise RuntimeError('Sidecar failed to start: '+str(archive/'sidecar.log'))
        if time.monotonic()>deadline:
            raise TimeoutError('Sidecar startup timed out; see '+str(archive/'sidecar.log'))
        time.sleep(.1)
    env = dict(os.environ, CYF_PROFILE_ROOT=str(root), MF_DIR=str(args.mf_dir.resolve()),
               LOG_MF_PATH=str(archive/'logs'))
    if cache:
        env['MS_COMPILER_CACHE_PATH']=str(cache)
    train = shlex.join([str(entry),'--config',str(archive/'train_config.yaml')])
    command = ['bash','scripts/msrun_launcher.sh',train,str(args.workers),str(args.port),
               str(archive/'msrun_log'),'True','7200']
    started = time.monotonic()
    record = {'command':command,'started_at':time.time(),'sidecar':process_identity(sidecar.pid),
              'ram_root':str(root),'archive':str(archive)}
    if dashboard_record:
        record['dashboard'] = dashboard_record
    atomic_json(archive/'training_run.json',record,durable=True)
    print('RAM capture:',root,flush=True)
    print('CPU sidecar:',sidecar.pid,'archive:',archive,flush=True)
    try:
        with (archive/'launcher.log').open('w') as log:
            training = subprocess.Popen(command,cwd=args.mf_dir,env=env,stdout=log,stderr=log)
            record['training_launcher'] = process_identity(training.pid)
            code = training.wait()
        record.update(returncode=code,elapsed_seconds=time.monotonic()-started,finished_at=time.time())
    except BaseException as exc:
        record.update(error=repr(exc),finished_at=time.time())
        raise
    finally:
        atomic_json(root/'launcher_done.json',record)
        atomic_json(archive/'training_run.json',record,durable=True)
    # Deliberately no wait/join on sidecar; it drains and exits independently.
    print('Training finished:',record,flush=True)
    print('Sidecar status:',archive/'status.json',flush=True)
    raise SystemExit(code)


if __name__ == '__main__':
    main()
