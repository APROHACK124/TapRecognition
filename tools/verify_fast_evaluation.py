"""Compare reference/fast sweeps and export schemas on the configured real data.

No notebook, checkpoint or historical export is modified. Full cached probability
traces are shared, so this measures detector/sweep work rather than model inference.
The reference's 100-pair Python scan can take several minutes.
"""

from __future__ import annotations

import argparse
from copy import deepcopy
from datetime import datetime
import hashlib
import json
from pathlib import Path
import sys
from time import perf_counter

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--report-dir', type=Path, default=Path('/tmp/opencode') /
                        f'fast_evaluation_verification_{datetime.now():%Y%m%d_%H%M%S}')
    args = parser.parse_args()
    args.report_dir.mkdir(parents=True, exist_ok=False)
    source_path = ROOT / 'main_notebooks/evaluate.ipynb'
    fast_path = ROOT / 'main_notebooks/evaluate_fast.ipynb'
    unchanged = {p: hashlib.sha256(p.read_bytes()).hexdigest()
                 for p in (source_path, *source_path.parent.glob('evaluate copy*.ipynb'))}
    original = json.loads(source_path.read_text())
    fast_notebook = json.loads(fast_path.read_text())
    codes = lambda n: [''.join(c['source']) for c in n['cells'] if c['cell_type'] == 'code']
    source_codes, fast_codes = codes(original), codes(fast_notebook)
    find = lambda collection, needle: next(c for c in collection if needle in c)

    def execute(code, env):
        exec(compile(code, '<evaluation-verification>', 'exec'), env)
        plt.close('all')

    reference = dict(__name__='__main__', display=lambda *args, **kwargs: None)
    # Use the fast copy's inference cell solely to avoid its fixed eager-cache
    # fallback bug. Both evaluators receive these very same model probabilities.
    for code in source_codes:
        if 'def evaluate_pair_events(' in code:
            break
        if '# FILE = next(' in code or code.startswith('selected  #'):
            continue  # Optional one-file visual inspection has no role in metrics.
        if 'validation_crop_normal_usage_probabilities = {}' in code:
            code = find(fast_codes, 'validation_crop_normal_usage_probabilities = {}')
        execute(code, reference)
    print('Probabilities prepared; running complete reference sweep.', flush=True)

    pair_code = find(source_codes, 'def evaluate_pair_events(')
    definitions, sweep = pair_code.split('sweep_rows = []', 1)
    execute(definitions, reference)
    reference_function = reference['evaluate_pair_events']
    reference_results = {}

    def capture(left, right, policy):
        key = (float(left), float(right), policy)
        if key not in reference_results:
            reference_results[key] = reference_function(*key)
        return deepcopy(reference_results[key])

    reference['evaluate_pair_events'] = capture
    started = perf_counter()
    execute('sweep_rows = []' + sweep, reference)
    reference_seconds = perf_counter() - started
    reference['pair_sweep'].to_csv(args.report_dir / 'reference_pair_sweep.csv', index=False)
    print(f'Reference sweep completed in {reference_seconds:.2f} s.', flush=True)

    fast = dict(reference)
    execute('from notebooks_analysis.fast_pair_evaluation import *\n'
            'from functools import partial\nfrom time import perf_counter', fast)
    fast['PAIR_SWEEP_CACHE_DIR'] = args.report_dir / 'pair_cache'
    execute(find(fast_codes, 'fast_traces = TracePool()'), fast)
    started = perf_counter()
    fast_pair_code = find(fast_codes, 'def evaluate_pair_events(')
    execute(fast_pair_code, fast)
    fast_seconds = perf_counter() - started
    pd.testing.assert_frame_equal(reference['pair_sweep'], fast['pair_sweep'], check_exact=True)
    for key, (overall, sides) in reference_results.items():
        actual, actual_sides = fast['evaluate_pair_events'](*key)
        pd.testing.assert_frame_equal(pd.DataFrame([overall]), pd.DataFrame([actual]), check_exact=True)
        pd.testing.assert_frame_equal(pd.DataFrame(sides), pd.DataFrame(actual_sides), check_exact=True)
    fast['pair_sweep'].to_csv(args.report_dir / 'fast_pair_sweep.csv', index=False)
    print('Every sweep row and side-metric dictionary matches exactly.', flush=True)

    # Fresh metric-cache object: require full-grid disk hits and equal dtypes/values.
    started = perf_counter()
    execute(fast_pair_code, fast)
    warm_seconds = perf_counter() - started
    assert fast['pair_cache'].misses == 0
    assert fast['pair_cache'].disk_hits == len(reference_results)
    pd.testing.assert_frame_equal(reference['pair_sweep'], fast['pair_sweep'], check_exact=True)

    # Run downstream selectors, comparisons, and all per-file/type/export tables.
    # Disable writing in the execution string, never in the reference notebook.
    for env, collection in ((reference, source_codes), (fast, fast_codes)):
        in_downstream = False
        for code in collection:
            if 'def select_operating_point(' in code:
                in_downstream = True
            if not in_downstream:
                continue
            code = code.replace('\nSAVE_COMPARISON = True\n', '\nSAVE_COMPARISON = False\n')
            execute(code, env)
    names = ['pair_sweep', 'selected_thresholds', 'comparison_summary', 'comparison_metrics',
             'recording_metrics', 'type_metrics', 'type_confusion', 'recording_metrics_all_modes',
             'type_metrics_all_modes', 'normal_usage_metrics', 'normal_usage_recording_metrics',
             'normal_usage_alarm_details', 'duration_audit', 'normal_usage_manifest']
    formats = []
    for name in names:
        expected, actual = reference[name], fast[name]
        pd.testing.assert_frame_equal(expected, actual, check_exact=True)
        assert expected.to_csv(index=False) == actual.to_csv(index=False), f'{name}: CSV differs'
        formats.append(dict(table=name, rows=len(actual), columns=len(actual.columns),
                            dtypes={str(k): str(v) for k, v in actual.dtypes.items()}))
    assert all(hashlib.sha256(p.read_bytes()).hexdigest() == digest for p, digest in unchanged.items())
    report = dict(model=str(reference['RUN_DIR'].relative_to(ROOT)),
                  dataset=str(reference['evaluation_data_root'].relative_to(ROOT)),
                  validation_recordings=len(reference['recordings']),
                  normal_usage_originals=len(reference['normal_usage_recordings']),
                  normal_usage_minutes=float(reference['normal_usage_manifest'].minutes.sum()),
                  validation_source_normal_originals=len(reference['validation_crop_normal_usage_recordings']),
                  pairs=len(reference_results), reference_seconds=reference_seconds,
                  fast_seconds=fast_seconds, warm_cache_seconds=warm_seconds,
                  speedup=reference_seconds / fast_seconds,
                  exact_values_and_csv_formats=True, tables=formats,
                  unchanged_notebooks={str(p.relative_to(ROOT)): digest for p, digest in unchanged.items()})
    (args.report_dir / 'verification.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps({k: v for k, v in report.items() if k not in ('tables', 'unchanged_notebooks')}, indent=2))
    print(f'All {len(names)} table schemas/dtypes and CSV values match. Report: {args.report_dir}', flush=True)


if __name__ == '__main__':
    main()
