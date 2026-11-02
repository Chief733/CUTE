"""Analyze worker logs first, with lifecycle timing as a separate measurement."""
import argparse
import csv
from datetime import datetime, timezone
import json
from pathlib import Path
import re
import statistics

import yaml

STEP = re.compile(r'(\d{4}-\d\d-\d\d \d\d:\d\d:\d\d,\d+).*?'
                  r'Epoch:\[\s*(\d+)/\s*(\d+)\], step:\[\s*(\d+)/\s*(\d+)\], '
                  r'loss:\s*([\d.eE+-]+|nan|inf|-inf), per_step_time:\s*(\d+)ms', re.I)
RATE = re.compile(r'([\d.]+) samples/s/p')
CACHE = re.compile(r'compil.*cache|cache.*compil|dependenc.*hash|cache.*mindir', re.I)


def read(path):
    return json.loads(path.read_text())


def analyze(directory):
    cfg = yaml.safe_load((directory/'train_config.yaml').read_text())
    run = read(directory/'training_run.json')
    sink = cfg['runner_config']['sink_size']
    ranks = []
    all_rows = []
    for path in sorted((directory/'log/worker').glob('worker_*.log')):
        rank = int(path.stem.split('_')[-1])
        rows = []
        cache_lines = []
        for line in path.read_text(errors='replace').splitlines():
            if CACHE.search(line) and ('WARNING' in line or 'INFO' in line):
                cache_lines.append(line)
            match = STEP.search(line)
            if match:
                stamp, epoch, _, step, epoch_steps, loss, ms = match.groups()
                step = (int(epoch)-1)*int(epoch_steps)+int(step)
                rows.append(dict(mode=directory.name, rank=rank, step=step,
                                 first_step=step-sink+1, sink_size=sink,
                                 logged_seconds_per_step=int(ms)/1000,
                                 logged_loss=float(loss), log_timestamp=stamp,
                                 timestamp=datetime.strptime(stamp,'%Y-%m-%d %H:%M:%S,%f').replace(
                                     tzinfo=timezone.utc).timestamp(),
                                 samples_per_second_per_device=None, source=str(path)))
            else:
                rate = RATE.search(line)
                if rate and rows:
                    rows[-1]['samples_per_second_per_device'] = float(rate.group(1))
        if not rows:
            ranks.append(dict(rank=rank, cache_lines=cache_lines, rows=0))
            continue
        steady = [r for r in rows if 4 <= r['step'] <= 18]
        by_step = {r['step']:r for r in rows}
        wall = ((by_step[18]['timestamp']-by_step[2]['timestamp'])/16
                if 18 in by_step and 2 in by_step else None)
        ranks.append(dict(rank=rank, rows=len(rows), cache_lines=cache_lines,
                          first_sink_seconds=rows[0]['logged_seconds_per_step']*sink,
                          logged_steady_seconds=statistics.mean(r['logged_seconds_per_step'] for r in steady)
                          if steady else None, wall_steady_seconds=wall,
                          final_loss=rows[-1]['logged_loss'], last_step=rows[-1]['step']))
        all_rows.extend(rows)
    steady_rows = [r for r in all_rows if 4 <= r['step'] <= 18]
    result = dict(mode=directory.name, returncode=run.get('returncode'),
                  launcher_seconds=run.get('elapsed_seconds'), ranks=ranks,
                  expected_steps=20, sink_size=sink, worker_rows=len(all_rows),
                  worker_log_dir=str(directory/'log/worker'))
    if steady_rows:
        result['logged_steady_seconds'] = statistics.mean(r['logged_seconds_per_step'] for r in steady_rows)
        result['logged_steady_samples_per_second_per_device'] = statistics.mean(
            r['samples_per_second_per_device'] for r in steady_rows
            if r['samples_per_second_per_device'] is not None)
        wall_values = [r['wall_steady_seconds'] for r in ranks
                       if r.get('wall_steady_seconds') is not None]
        result['wall_steady_seconds'] = statistics.mean(wall_values) if wall_values else None
    phases = []
    for path in sorted((directory/'events').glob('rank_*.jsonl')):
        events = [json.loads(line) for line in path.read_text().splitlines()]
        pairs = {}
        totals = {}
        for e in events:
            if e['event'].endswith('_begin'):
                pairs[e['event'][:-6]] = e['monotonic']
            elif e['event'].endswith('_end'):
                label = e['event'][:-4]
                if label in pairs:
                    totals[label] = totals.get(label,0)+e['monotonic']-pairs.pop(label)
        phases.append(dict(rank=path.stem, seconds=totals))
    result['event_phase_seconds'] = phases
    status = directory/'status.json'
    if status.exists():
        result['sidecar_status'] = read(status)
        archives = [read(p) for p in (directory/'segments').glob('segment_*/rank_*/archive.json')]
        parses = [read(p) for p in (directory/'segments').glob('segment_*/rank_*/parse_result.json')]
        result['background'] = dict(archived=len(archives), parsed=len(parses),
            raw_bytes=sum(r['bytes'] for r in archives),
            archive_seconds_sum=sum(r['archive_seconds'] for r in archives),
            parse_seconds_sum=sum(r.get('elapsed_seconds',0) for r in parses),
            last_parse_finished_at=max((r.get('finished_at',0) for r in parses),default=None),
            result_states={s:sum(r['state']==s for r in parses) for s in ('complete','timeline_only','failed')})
    return result, all_rows


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('output', type=Path)
    args = p.parse_args()
    results, rows = [], []
    for mode in ('no_profile','offload','default'):
        directory = args.output/mode
        if (directory/'training_run.json').exists():
            result, data = analyze(directory)
            results.append(result)
            rows.extend(data)
    (args.output/'summary.json').write_text(json.dumps(results,indent=2)+'\n')
    if rows:
        with (args.output/'worker_steps.csv').open('w') as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    print(json.dumps([{k:v for k,v in r.items() if k not in ('ranks','event_phase_seconds')}
                      for r in results],indent=2))


if __name__ == '__main__':
    main()
