"""Inspect real process state, step coverage, raw integrity and parsed outputs."""
import argparse
from collections import defaultdict
import hashlib
import json
from pathlib import Path
import sys

from common import atomic_json, is_alive, read_json


def inspect(archive, verify=False, first=None, last=None, ranks=None):
    archive=Path(archive)
    cfg=read_json(archive/'sidecar_config.json')
    root=Path(cfg['ram_root'])
    status=read_json(archive/'status.json')
    training=read_json(archive/'training_run.json')
    rows=defaultdict(list)
    errors=[]
    parses=[]
    fast_results=[]
    bytes_verified=0
    for marker in sorted((archive/'segments').glob('segment_*/rank_*/archive.json')):
        saved=read_json(marker)
        metadata=saved['segment']
        rows[metadata['rank']].append(metadata)
        if verify:
            for item in saved['files']:
                path=marker.parent/item['path']
                digest=hashlib.sha256()
                try:
                    with path.open('rb') as stream:
                        for block in iter(lambda: stream.read(1024*1024),b''):
                            digest.update(block)
                    if path.stat().st_size!=item['bytes'] or digest.hexdigest()!=item['sha256']:
                        errors.append('Raw archive checksum mismatch: '+str(path))
                    bytes_verified+=item['bytes']
                except OSError as exc:
                    errors.append(repr(exc))
        result=marker.parent/'parse_result.json'
        if cfg.get('parse'):
            if not result.exists():
                errors.append('Parse not complete: '+str(marker.parent))
            else:
                parsed=read_json(result)
                parses.append(parsed)
                if parsed['state'] not in ('complete','timeline_only'):
                    errors.append('Parse failed: '+str(result))
                elif not Path(parsed['timeline']).is_file():
                    errors.append('Missing parsed timeline: '+parsed['timeline'])
        if cfg.get('fast_analyse'):
            segment=metadata['segment']
            rank=metadata['rank']
            fast_dir=archive/'fast'/f'segment_{segment:06d}'
            fast_result=fast_dir/f'rank_{rank}.result.json'
            fast_summary=fast_dir/f'rank_{rank}.json'
            if not fast_result.exists() or not fast_summary.exists():
                errors.append('Fast parse not complete: '+str(marker.parent))
            else:
                result_value=read_json(fast_result)
                fast_results.append(result_value)
                if result_value.get('state')!='complete':
                    errors.append('Fast parse failed: '+str(fast_result))
                else:
                    summary=read_json(fast_summary)
                    identity=(summary.get('source',{}).get('segment'),
                              summary.get('source',{}).get('rank'))
                    if summary.get('schema')!='fast_summary_v1' or identity!=(segment,rank):
                        errors.append('Fast summary identity/schema mismatch: '+str(fast_summary))
                    if cfg.get('fast_require_complete_step_trace'):
                        expected_steps=list(range(metadata['first_step'],metadata['last_step']+1))
                        actual_steps=[item.get('global_step') for item in summary.get('steps',())]
                        trace=summary.get('coverage',{})
                        if (actual_steps!=expected_steps
                                or trace.get('step_pairs')!=len(expected_steps)
                                or not trace.get('step_trace_complete')):
                            errors.append(
                                f'Incomplete per-step trace for segment {segment} rank {rank}: '
                                f'expected {expected_steps}, got {actual_steps}'
                            )
    coverage={}
    for rank,segments in rows.items():
        segments.sort(key=lambda s:s['first_step'])
        expected=first if first is not None else segments[0]['first_step']
        for segment in segments:
            if segment['first_step']!=expected or segment['last_step']<expected:
                errors.append(f'Gap/overlap for rank {rank}: expected {expected}, got {segment}')
            expected=segment['last_step']+1
        if last is not None and expected!=last+1:
            errors.append(f'Incomplete rank {rank}: last captured step {expected-1}, expected {last}')
        coverage[rank]={'first_step':segments[0]['first_step'],'last_step':expected-1,
                        'segments':len(segments),'capture_steps':sum(s['last_step']-s['first_step']+1 for s in segments),
                        'start_seconds':sum(s['start_seconds'] for s in segments),
                        'stop_seconds':sum(s['stop_seconds'] for s in segments),
                        'finalize_seconds':sum(s['finalize_seconds'] for s in segments)}
    if ranks is not None and set(rows)!=set(range(ranks)):
        errors.append('Missing/unexpected ranks: '+str(sorted(rows)))
    if cfg.get('fast_analyse'):
        for segment in sorted({item['segment'] for values in rows.values() for item in values}):
            window_dir=archive/'fast'/f'segment_{segment:06d}'
            window_result=window_dir/'window_result.json'
            window_summary=window_dir/'window_summary.json'
            if not window_result.exists() or not window_summary.exists():
                errors.append('Fast window aggregation not complete: '+str(window_dir))
            elif read_json(window_result).get('state')!='complete':
                errors.append('Fast window aggregation failed: '+str(window_result))
            else:
                summary=read_json(window_summary)
                if summary.get('schema')!='fast_window_summary_v1' or not summary.get('complete'):
                    errors.append('Fast window summary invalid: '+str(window_summary))
    if training.get('returncode')!=0:
        errors.append('Training not completed successfully')
    if status['state']!='complete':
        errors.append('Sidecar not completely drained: '+status['state'])
    result={'state':status['state'],'sidecar_alive':is_alive(status['process']),
            'training_seconds':training.get('elapsed_seconds'),'status':status,
            'coverage':coverage,'raw_bytes_verified':bytes_verified,
            'parse_states':{name:sum(p['state']==name for p in parses)
                            for name in ('complete','timeline_only','failed')},
            'fast_parse_states':{name:sum(p.get('state')==name for p in fast_results)
                                 for name in ('complete','failed')},
            'parser_finished_after_training_seconds':max(
                (p.get('finished_at',0)-training.get('finished_at',0) for p in parses), default=None),
            'errors':errors}
    return result


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('archive',type=Path)
    parser.add_argument('--verify',action='store_true',help='SHA-256 verify every committed raw file')
    parser.add_argument('--first-step',type=int)
    parser.add_argument('--last-step',type=int)
    parser.add_argument('--ranks',type=int)
    parser.add_argument('--save',action='store_true')
    args=parser.parse_args()
    result=inspect(args.archive,args.verify,args.first_step,args.last_step,args.ranks)
    print(json.dumps(result,indent=2))
    if args.save:
        atomic_json(args.archive/'verification.json',result,durable=True)
    if args.verify and result['errors']:
        raise SystemExit(1)


if __name__=='__main__':
    main()
