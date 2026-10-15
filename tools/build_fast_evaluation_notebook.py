"""Build a clean, isolated fast copy while retaining the reference metric code.

Run from the repository root. The source and existing destination are never
overwritten. Assertions intentionally fail if a source edit invalidates a rewrite.
"""

from __future__ import annotations

import argparse
from copy import deepcopy
import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def build_notebook(notebook):
    notebook = deepcopy(notebook)

    def cell(needle):
        found = [c for c in notebook['cells'] if needle in ''.join(c.get('source', []))]
        if len(found) != 1:
            raise ValueError(f'Expected one cell containing {needle!r}, got {len(found)}')
        return found[0]

    def replace(target, before, after):
        text = ''.join(target['source'])
        if text.count(before) != 1:
            raise ValueError(f'Expected one occurrence of {before!r}')
        target['source'] = text.replace(before, after).splitlines(keepends=True)

    title = notebook['cells'][0]
    title['source'] = (''.join(title['source']) + '''

## Fast-copy implementation

This copy preserves the reference grid, populations, metrics, selectors, table
columns, and plots. Pair replay searches precomputed crossing frames and lookahead
peaks instead of scanning quiet frames in Python. Full normal-use sweeps retain
only counts and reuse them across population subsets. Detailed tables are still
built for selected operating points. Input probabilities must not be mutated in
place; rerun the detector and pair-sweep cells after changing upstream inputs.

Pair metrics are cached in memory and (optionally) on disk using actual trace,
truth, exposure, metric-code and timing fingerprints. Budgets/selection modes can
be changed without repeating the sweep. `PAIR_SWEEP_CACHE_DIR=None` disables disk
caching. Exports are **off by default** and use an isolated `fast_evaluation/`
subdirectory. The original notebook and its exports are never overwritten.
''').splitlines(keepends=True)

    setup = cell('from notebooks_analysis.normal_usage_evaluation import ')
    replace(setup, 'from notebooks_analysis.normal_usage_evaluation import load_normal_usage, evaluate_normal_usage\n',
            'from notebooks_analysis.normal_usage_evaluation import load_normal_usage, evaluate_normal_usage\n'
            'from notebooks_analysis.fast_pair_evaluation import (\n'
            '    TracePool, normal_usage_counts, PairMetricsCache, sweep_context,\n'
            ')\n'
            'from functools import partial\n'
            'from time import perf_counter\n'
            "PAIR_SWEEP_CACHE_DIR = ROOT / '.cache/evaluate_fast'  # None disables disk caching.\n")
    replace(setup, 'model_cfg = dict(checkpoint[\'model_config\'])\n',
            "# Keep fast-copy exports separate from the reference notebook.\n"
            "EVALUATION_OUTPUT_DIR = EVALUATION_OUTPUT_DIR / 'fast_evaluation'\n"
            "model_cfg = dict(checkpoint['model_config'])\n")

    inference = cell('validation_crop_normal_usage_probabilities = {}')
    replace(inference, '''    validation_crop_normal_usage_probabilities[rec.path.name] = normal_usage_probabilities.get(
        rec.path.name,
        predict_recording(rec, normal_usage_model, False, NORMAL_USAGE_CHUNK_FRAMES),
    )''', '''    if rec.path.name in normal_usage_probabilities:
        validation_crop_normal_usage_probabilities[rec.path.name] = normal_usage_probabilities[rec.path.name]
    else:
        validation_crop_normal_usage_probabilities[rec.path.name] = predict_recording(
            rec, normal_usage_model, False, NORMAL_USAGE_CHUNK_FRAMES)''')

    detector = cell('def emit_alarms_pair(probs, left_threshold, right_threshold, look, refractory):')
    original_detector = ''.join(detector['source'])
    function, remainder = original_detector.split('# At equal thresholds', 1)
    detector['source'] = ('''# Recreate after changing upstream probabilities; prepared traces are immutable snapshots.
fast_traces = TracePool()
emit_alarms_pair = fast_traces.emit
fast_normal_usage_counts = partial(normal_usage_counts, fast_traces)

# At equal thresholds''' + remainder).splitlines(keepends=True)

    pair = cell('def evaluate_pair_events(left_threshold, right_threshold, policy):')
    pair_source = ''.join(pair['source'])
    function, sweep = pair_source.split('sweep_rows = []', 1)
    # Hash unmodified metric/matching/detector code so copied caches cannot outlive
    # semantic changes to the notebook. Runtime constants are fingerprinted below.
    metric_source = function + original_detector + ''.join(cell('def match_alarms(')['source'])
    metric_hash = hashlib.sha256(metric_source.encode()).hexdigest()
    function = function.replace('evaluate_normal_usage(', 'fast_normal_usage_counts(')
    before = f'''pair_cache = PairMetricsCache(PAIR_SWEEP_CACHE_DIR, sweep_context(
    recordings, probabilities, duration_audit,
    [(normal_usage_recordings, normal_usage_probabilities),
     (validation_crop_normal_usage_recordings, validation_crop_normal_usage_probabilities)],
    dict(policies=POLICIES, deadlines=DEADLINES, timely_budget_frames=TIMELY_BUDGET_FRAMES,
         early_allowance_frames=EARLY_ALLOWANCE_FRAMES, max_late_frames=MAX_LATE_FRAMES,
         metric_source_sha256={metric_hash!r}),
))

@pair_cache.memoize
'''
    after = '''sweep_finished = perf_counter()
print(f'Pair sweep: {sweep_finished - sweep_started:.2f} s | '
      f'computed={pair_cache.misses}, disk hits={pair_cache.disk_hits}, memory hits={pair_cache.hits}')
pair_sweep
'''
    sweep = 'sweep_rows = []' + sweep
    assert sweep.endswith('pair_sweep') or sweep.endswith('pair_sweep\n')
    sweep = sweep.rstrip().removesuffix('pair_sweep') + after
    pair['source'] = (before + function + 'sweep_started = perf_counter()\n' + sweep).splitlines(keepends=True)

    exports = cell('SAVE_COMPARISON = True')
    replace(exports, '\nSAVE_COMPARISON = True\n',
            '\nSAVE_COMPARISON = False  # Opt in; exports go only to fast_evaluation/.\n')
    for index, target in enumerate(notebook['cells']):
        if target['cell_type'] == 'code':
            target['outputs'] = []
            target['execution_count'] = None
            compile(''.join(target['source']), f'evaluate_fast:cell-{index}', 'exec')
    return notebook


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, default=ROOT / 'main_notebooks/evaluate.ipynb')
    parser.add_argument('--output', type=Path, default=ROOT / 'main_notebooks/evaluate_fast.ipynb')
    args = parser.parse_args()
    notebook = build_notebook(json.loads(args.source.read_text()))
    # Exclusive creation protects all user-edited evaluation copies.
    with args.output.open('x', encoding='utf-8') as handle:
        json.dump(notebook, handle, indent=1, ensure_ascii=False)
        handle.write('\n')
    print(f'Created {args.output}; source unchanged.')


if __name__ == '__main__':
    main()
