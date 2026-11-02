#!/usr/bin/env python3
"""Summarize recorded wall time, sink batches, and profiler finalization."""

import json
from pathlib import Path
import statistics
import sys


def rank_summary(path):
    events = [json.loads(line) for line in path.read_text().splitlines()]
    by_name = {}
    for item in events:
        by_name.setdefault(item['event'], []).append(item)
    if 'model_train_end' not in by_name:
        raise ValueError(f'Incomplete training: {path}')
    finalize = 0.0
    if 'profile_finalize_end' in by_name:
        finalize = (by_name['profile_finalize_end'][0]['monotonic'] -
                    by_name['profile_finalize_begin'][0]['monotonic'])
    starts = {item['step']: item for item in by_name['step_begin']}
    batches = []
    previous_step = 0
    for end in by_name['step_end']:
        step = end['step']
        seconds = end['monotonic'] - starts[step]['monotonic']
        count = step - previous_step
        corrected = seconds - finalize if step == 20 else seconds
        batches.append({'ending_step': step, 'steps': count, 'seconds': seconds,
                        'seconds_excluding_profile_finalize': corrected,
                        'seconds_per_step_excluding_profile_finalize': corrected / count})
        previous_step = step
    if previous_step != 20:
        raise ValueError(f'Expected 20 steps, got {previous_step}: {path}')
    steady = [b for b in batches if 2 < b['ending_step'] < 20]
    return {
        'rank': path.stem,
        'model_train_seconds': (by_name['model_train_end'][0]['monotonic'] -
                                by_name['model_train_begin'][0]['monotonic']),
        'profile_finalize_seconds': finalize,
        'first_two_steps_seconds': batches[0]['seconds'],
        'steady_steps_3_to_18_mean_seconds': (
            sum(b['seconds'] for b in steady) / sum(b['steps'] for b in steady)),
        'steady_batch_median_seconds_per_step': statistics.median(
            b['seconds_per_step_excluding_profile_finalize'] for b in steady),
        'after_last_step_seconds': (by_name['model_train_end'][0]['monotonic'] -
                                    by_name['step_end'][-1]['monotonic']),
        'batches': batches,
    }


def main():
    root = Path(sys.argv[1]).resolve()
    runs = json.loads((root / 'runs.json').read_text())
    results = {}
    for run in runs:
        if run['returncode']:
            raise ValueError(f"Run {run['mode']} failed")
        ranks = [rank_summary(p) for p in sorted((root / run['mode'] / 'events').glob('rank_*.jsonl'))]
        if len(ranks) != 8:
            raise ValueError(f"Expected 8 ranks for {run['mode']}, got {len(ranks)}")
        results[run['mode']] = {
            'elapsed_seconds': run['elapsed_seconds'],
            'mean_steady_seconds_per_step_across_ranks': statistics.mean(
                r['steady_steps_3_to_18_mean_seconds'] for r in ranks),
            'profile_finalize_seconds_max_rank': max(r['profile_finalize_seconds'] for r in ranks),
            'ranks': ranks,
        }
    if 'profile' in results and 'no_profile' in results:
        base, prof = results['no_profile'], results['profile']
        results['comparison'] = {
            'wall_time_increase_seconds': prof['elapsed_seconds'] - base['elapsed_seconds'],
            'wall_time_increase_percent': (prof['elapsed_seconds'] / base['elapsed_seconds'] - 1) * 100,
            'steady_step_increase_percent': (
                prof['mean_steady_seconds_per_step_across_ranks'] /
                base['mean_steady_seconds_per_step_across_ranks'] - 1) * 100,
        }
    (root / 'summary.json').write_text(json.dumps(results, indent=2) + '\n')
    for mode, data in results.items():
        print(mode, json.dumps({k: v for k, v in data.items() if k != 'ranks'}, indent=2))


if __name__ == '__main__':
    main()
