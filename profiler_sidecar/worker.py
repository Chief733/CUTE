"""Detached CPU sidecar: durable archival and bounded online/offline parsing.

The coordinator never imports MindSpore. Archiving has its own I/O thread so a
slow vendor parser cannot prevent archival.  Optional direct raw parsers run in
parallel with archival; at most one vendor parser runs at a time.
"""
import argparse
from collections import Counter, deque
from concurrent.futures import ThreadPoolExecutor
import fcntl
import hashlib
import os
from pathlib import Path
import resource
import shutil
import subprocess
import sys
import time
import traceback

from common import (GIB, atomic_json, is_alive, process_identity, read_json,
                    raw_rank_paths, sync_directory, trace_events, tree_bytes)
from fast_worker import FastAnalysisManager
from diagnosis_worker import DiagnosisManager


def archive_segment(source, archive, retire_source=True):
    source, archive = Path(source), Path(archive)
    ready = read_json(source / 'ready.json')
    destination = archive / 'segments' / source.parent.name / source.name
    if destination.exists():
        saved = read_json(destination / 'archive.json')
        if saved['segment'] != ready:
            raise RuntimeError(f'Archive identity collision: {destination}')
        # A prior daemon may have died after commit and before deleting RAM.
        # Fast mode holds the immutable source until its parser is also done.
        if retire_source:
            shutil.rmtree(source)
        return dict(saved, already_committed=True)
    temporary = destination.with_name('.' + destination.name + '.copying')
    if temporary.exists():
        shutil.rmtree(temporary)
    temporary.mkdir(parents=True)
    files = []
    started = time.monotonic()
    for entry in sorted(source.rglob('*')):
        target = temporary / entry.relative_to(source)
        if entry.is_symlink():
            raise RuntimeError(f'Unexpected symlink in sealed segment: {entry}')
        if entry.is_dir():
            target.mkdir(exist_ok=True)
            continue
        digest = hashlib.sha256()
        size = 0
        with entry.open('rb') as reader, target.open('wb') as writer:
            for block in iter(lambda: reader.read(1024 * 1024), b''):
                writer.write(block)
                digest.update(block)
                size += len(block)
            writer.flush()
            os.fsync(writer.fileno())
        files.append({'path': str(entry.relative_to(source)), 'bytes': size,
                      'sha256': digest.hexdigest()})
    record = {'segment': ready, 'files': files, 'bytes': sum(f['bytes'] for f in files),
              'archive_seconds': time.monotonic() - started, 'archived_at': time.time()}
    atomic_json(temporary / 'archive.json', record, durable=True)
    for directory in sorted((p for p in temporary.rglob('*') if p.is_dir()), reverse=True):
        sync_directory(directory)
    sync_directory(temporary)
    os.replace(temporary, destination)
    sync_directory(destination.parent)
    # Without fast analysis the durable commit is the only retirement latch.
    # Fast mode lets the coordinator apply its second (parser terminal) latch.
    if retire_source:
        shutil.rmtree(source)
    return record


def parse_segment(path, config):
    path = Path(path)
    started = time.monotonic()
    result = {'process': process_identity(), 'started_at': time.time(), 'timings': {}}
    try:
        limit = int(config['parser_memory_gib'] * GIB)
        resource.setrlimit(resource.RLIMIT_AS, (limit, limit))
        os.nice(10)
        allowed = sorted(os.sched_getaffinity(0))
        os.sched_setaffinity(0, allowed[-min(len(allowed), config['parser_cpus']):])
        # Vendor analysis rewrites profiler_info and adds files beside its input.
        # Keep the committed raw archive immutable and parse a disposable copy.
        manifest = read_json(path/'archive.json')
        analysis = path/'analysis'
        phase = time.monotonic()
        for item in manifest['files']:
            target = analysis/item['path']
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(path/item['path'], target)
        result['timings']['input_copy_seconds'] = time.monotonic()-phase
        phase = time.monotonic()
        import mindspore as ms
        metadata = read_json(path / 'ready.json')
        rank = metadata['rank']
        info = read_json(path / 'profiler' / f'profiler_info_{rank}.json')
        ms.set_context(device_target='CPU', mode=(ms.GRAPH_MODE if info['context_mode'] == 'graph'
                                                else ms.PYNATIVE_MODE))
        profiler = ms.Profiler(analyse_only=True, rank_id=rank, data_simplification=False)
        options = info['profiling_options']
        profiler._profile_memory = options.get('profile_memory') == 'on'
        profiler._profile_communication = options.get('hccl') == 'on'
        profiler._data_process = metadata.get('data_process', False)
        profiler._rank_size = metadata['world_size']
        result['timings']['framework_setup_seconds'] = time.monotonic()-phase
        phase = time.monotonic()
        profiler.analyse(offline_path=str(analysis))
        result['timings']['analyse_seconds'] = time.monotonic()-phase
        phase = time.monotonic()
        timeline = analysis / 'profiler' / f'ascend_timeline_display_{rank}.json'
        names, count, durations = Counter(), 0, 0
        for event in trace_events(timeline):
            count += 1
            names[event.get('name', '')] += 1
            durations += event.get('ph') == 'X'
        if not durations:
            raise RuntimeError('Parser produced no duration events')
        statistics = list((analysis / 'profiler').rglob('op_statistic*.csv'))
        result.update(state='complete' if statistics else 'timeline_only',
                      timeline=str(timeline), events=count, duration_events=durations,
                      common_names=names.most_common(20),
                      op_statistics_present=bool(statistics),
                      trace_views=[str(p) for p in (analysis / 'profiler').rglob('trace_view.json')])
        result['timings']['validate_seconds'] = time.monotonic()-phase
        phase = time.monotonic()
        # Discard only redundant copies of input blobs, retaining the immutable
        # original and all generated outputs. Metadata JSON remains beside output.
        for item in manifest['files']:
            if not item['path'].endswith('.json'):
                (analysis/item['path']).unlink(missing_ok=True)
        result['timings']['cleanup_seconds'] = time.monotonic()-phase
    except BaseException as exc:
        result.update(state='failed', error=repr(exc), traceback=traceback.format_exc())
    result.update(finished_at=time.time(), elapsed_seconds=time.monotonic()-started)
    atomic_json(path / 'parse_result.json', result, durable=True)
    return 1 if result['state'] == 'failed' else 0


def run(root, archive_only=False):
    root = Path(root).resolve()
    config = read_json(root / 'config.json')
    if archive_only:
        config['parse']=False
        config['fast_analyse']=False
    archive = Path(config['archive'])
    lock = (root / 'worker.lock').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    status = {'process': process_identity(), 'started_at': time.time(), 'fatal': None}
    fast = FastAnalysisManager(root, archive, config)
    diagnosis = DiagnosisManager(root, archive, config)
    parser, parser_path, copy, adopted = None, None, None, None
    failed_copy = set()
    pending, counts = deque(), Counter()
    archived_total = 0
    # Discover persisted work once on startup, not an ever-growing directory
    # rescan/read of every historical result on each heartbeat.
    for marker in sorted((archive/'segments').glob('segment_*/rank_*/archive.json')):
        archived_total += 1
        result = marker.parent/'parse_result.json'
        if result.exists():
            counts[read_json(result)['state']] += 1
        elif config['parse']:
            pending.append(marker.parent)
    peak_raw_bytes, peak_pending = 0, len(pending)
    pool = ThreadPoolExecutor(max_workers=1)
    try:
        while True:
            if copy and copy.done():
                try:
                    record = copy.result()
                    if not record.get('already_committed'):
                        source = Path(status['copying'])
                        if config['parse']:
                            pending.append(archive/'segments'/source.parent.name/source.name)
                        archived_total += 1
                except Exception as exc:
                    status['fatal'] = 'Archive failed; RAM source retained: ' + repr(exc)
                    failed_copy.add(status['copying'])
                copy = None
                status['copying'] = None
            parser_finished = parser and parser.poll() is not None
            adopted_finished = adopted and not is_alive(adopted)
            if parser_finished or adopted_finished:
                if not (parser_path / 'parse_result.json').exists():
                    atomic_json(parser_path / 'parse_result.json',
                                {'state': 'failed', 'returncode': parser.returncode if parser else None,
                                 'error': 'Parser exited without result (possibly memory limit)'}, durable=True)
                counts[read_json(parser_path/'parse_result.json')['state']] += 1
                assert pending.popleft() == parser_path
                parser, parser_path, adopted = None, None, None
            raw_ranks=raw_rank_paths(root/'raw')
            ready = sorted(p/'ready.json' for p in raw_ranks if (p/'ready.json').exists())
            # Direct parsers consume the sealed tmpfs copy immediately while the
            # archive thread copies the same immutable bytes to durable storage.
            fast.refresh(raw_ranks)
            fast.tick()

            # In fast mode RAM ownership is released only after both independent
            # consumers are terminal.  A recorded parser failure is terminal so
            # it cannot leak tmpfs forever; the durable archive remains retryable.
            if fast.enabled:
                for marker in list(ready):
                    source = marker.parent
                    destination = archive/'segments'/source.parent.name/source.name
                    if ((destination/'archive.json').exists()
                            and fast.state_for_input(source) in ('complete', 'failed')):
                        try:
                            shutil.rmtree(source)
                        except OSError as exc:
                            status['fatal'] = 'Could not retire completed RAM source: ' + repr(exc)
                raw_ranks=raw_rank_paths(root/'raw')
                ready = sorted(p/'ready.json' for p in raw_ranks if (p/'ready.json').exists())
            if not copy and not status['fatal']:
                for marker in ready:
                    if str(marker.parent) in failed_copy:
                        continue
                    destination = archive/'segments'/marker.parent.parent.name/marker.parent.name
                    if (destination/'archive.json').exists():
                        continue
                    needed = tree_bytes(marker.parent)
                    if shutil.disk_usage(archive).free < needed + config['min_disk_free_gib'] * GIB:
                        status['fatal'] = 'Insufficient archive disk space; sealed RAM data retained'
                        break
                    status['copying'] = str(marker.parent)
                    copy = pool.submit(archive_segment, marker.parent, archive, not fast.enabled)
                    break
            if not parser and not adopted and pending and config['parse']:
                # On restart, do not duplicate a parser still running independently.
                parser_path = pending[0]
                previous = read_json(parser_path/'parser.json')['process'] if (parser_path/'parser.json').exists() else None
                if is_alive(previous):
                    adopted = previous
                else:
                    env = dict(os.environ, OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1',
                               MKL_NUM_THREADS='1', ASCEND_VISIBLE_DEVICES='')
                    for key in ('MS_ROLE', 'MS_NODE_ID', 'MS_WORKER_NUM', 'MS_SCHED_HOST',
                                'MS_SCHED_PORT', 'RANK_TABLE_FILE', 'RANK_ID', 'RANK_SIZE',
                                'DEVICE_ID', 'PROFILING_OPTIONS', 'MS_PROFILER_OPTIONS'):
                        env.pop(key, None)
                    with (parser_path / 'parse.log').open('ab') as log:
                        parser = subprocess.Popen([sys.executable, __file__, 'parse', str(parser_path),
                                                   '--config', str(root/'config.json')],
                                                  stdin=subprocess.DEVNULL, stdout=log, stderr=log,
                                                  env=env, start_new_session=True)
                    atomic_json(parser_path/'parser.json', {'process': process_identity(parser.pid)}, durable=True)
            launching = not (root / 'launcher_done.json').exists() and is_alive(config['launcher'])
            producers = [read_json(p) for p in (root/'clients').glob('*.json')]
            producer_alive = any(not c.get('done') and is_alive(c['process']) for c in producers)
            if not launching and not producer_alive:
                fast.finish_partial_windows()
            diagnosis.tick(fast.window_events)
            fast.window_events.clear()
            interruptions = [dict(rank=c['rank'], reason=c['capture_disabled_reason'],
                                  after_step=c.get('disabled_after_step'))
                             for c in producers if c.get('capture_disabled_reason')]
            unsealed = [str(p) for p in raw_ranks
                        if not (p/'ready.json').exists() and (not copy or str(p) != status.get('copying'))]
            raw_bytes = tree_bytes(root/'raw')
            peak_raw_bytes = max(peak_raw_bytes, raw_bytes)
            peak_pending = max(peak_pending, len(pending))
            status.update(heartbeat=time.time(), raw_bytes=raw_bytes, ready_segments=len(ready),
                          tmpfs_used_bytes=shutil.disk_usage(root).used,
                          capture_interruptions=interruptions,
                          archived_segments=archived_total, parse_pending=len(pending),
                          parse_counts=dict(counts), producer_alive=producer_alive,
                          active_unsealed_segments=unsealed,
                          parser_process=process_identity(parser.pid) if parser else adopted,
                          peak_raw_bytes=peak_raw_bytes, peak_parse_pending=peak_pending,
                          parse_enabled=config['parse'],
                          state='running')
            status.update(fast.snapshot())
            status.update(diagnosis.snapshot())
            drained = not copy and (not ready or bool(status['fatal'])) and not parser and not adopted and (
                not config['parse'] or not pending) and fast.drained
            if not launching and not producer_alive and drained:
                diagnosis.finish()
                status.update(diagnosis.snapshot())
                status['state'] = 'incomplete' if (unsealed or interruptions or status['fatal']
                                                 or counts['failed'] or fast.has_failures
                                                 or diagnosis.failed) else 'complete'
            atomic_json(root/'status.json', status)
            atomic_json(archive/'status.json', status)
            if status['state'] != 'running':
                atomic_json(archive/'capture_clients.json', producers, durable=True)
                snapshot = config.get('dashboard_snapshot')
                if snapshot:
                    try:
                        from dashboard_server import export_html
                        export_html(archive, snapshot)
                        status['dashboard_snapshot'] = snapshot
                    except Exception as error:
                        status['dashboard_snapshot_error'] = repr(error)
                    atomic_json(root/'status.json', status)
                    atomic_json(archive/'status.json', status, durable=True)
                break
            time.sleep(0.5)
    finally:
        pool.shutdown(wait=True)
        lock.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=['run', 'parse'])
    parser.add_argument('path', type=Path)
    parser.add_argument('--config', type=Path)
    parser.add_argument('--archive-only', action='store_true')
    args = parser.parse_args()
    if args.action == 'parse':
        raise SystemExit(parse_segment(args.path, read_json(args.config)))
    run(args.path,args.archive_only)


if __name__ == '__main__':
    main()
