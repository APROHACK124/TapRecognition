"""Benchmark reference/cold/warm comparison and check exact tables and traces.

Uses the current compare.ipynb RUNS and frozen operating points. Writes reports
and disposable prediction caches under /tmp/opencode, never historical exports.
"""

from __future__ import annotations

import argparse
from datetime import datetime
import hashlib
import json
from pathlib import Path
import sys
from time import perf_counter
from unittest.mock import patch

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from notebooks_analysis.fast_comparison import ComparisonRuntime
from notebooks_analysis import model_comparison_examples as examples
from notebooks_analysis import normal_usage_evaluation as normal


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--report-dir', type=Path, default=Path('/tmp/opencode') /
                        f'fast_comparison_verification_{datetime.now():%Y%m%d_%H%M%S}')
    args = parser.parse_args()
    args.report_dir.mkdir(parents=True, exist_ok=False)
    paths = [ROOT / 'main_notebooks/compare.ipynb', *sorted((ROOT / 'main_notebooks').glob('evaluate*.ipynb'))]
    snapshots = {p: hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}
    original = json.loads(paths[0].read_text())
    fast = json.loads((ROOT / 'main_notebooks/fast_comparison.ipynb').read_text())
    torch.set_num_threads(2)  # Same CPU setting for BOTH paths, not a speedup trick.
    namespaces = []
    timings, formats = {}, []

    def timed(label, fn):
        start = perf_counter()
        result = fn()
        timings[label] = perf_counter() - start
        print(f'{label}: {timings[label]:.3f} s', flush=True)
        return result

    def same_table(label, expected, actual):
        pd.testing.assert_frame_equal(expected, actual, check_exact=True)
        assert expected.to_csv(index=False) == actual.to_csv(index=False), f'{label}: CSV mismatch'
        formats.append(dict(table=label, rows=len(actual), columns=len(actual.columns)))

    for label, notebook in [('setup_reference', original), ('setup_fast', fast)]:
        ns = dict(__name__='__main__', display=lambda *args, **kwargs: None)
        code = ''.join(notebook['cells'][1]['source'])
        # Keep verification data out of the user's persistent notebook cache.
        code = code.replace("ROOT / '.cache/fast_comparison'", repr(str(args.report_dir / 'cache')))
        timed(label, lambda: exec(compile(code, label, 'exec'), ns))
        namespaces.append(ns)
    reference, copied = namespaces
    for name in ('comparison_sources', 'operating_points', 'scores', 'type_scores', 'file_scores', 'recording_duration_audit'):
        same_table(name, reference[name], copied[name])
    runtime = copied['comparison_runtime']
    points = reference['operating_points'].merge(reference['comparison_sources'][[
        'model', 'checkpoint_path', 'evaluation_data', 'normal_usage_export_sha256']], on='model', validate='one_to_one')
    normal_hashes = {}
    original_predict = normal.predict_recording

    def record_prediction(rec, spec, *args):
        p = original_predict(rec, spec, *args)
        normal_hashes[(spec.sha256, rec.sha256)] = hashlib.sha256(p.tobytes()).hexdigest()
        return p

    call = lambda rt=None: normal.replay_normal_usage(ROOT, points, reference['NORMAL_USAGE_SOURCE_DIR'],
                    reference['NORMAL_USAGE_FILES'], reference['NORMAL_USAGE_CHUNK_FRAMES'], runtime=rt)
    with patch.object(normal, 'predict_recording', side_effect=record_prediction):
        expected = timed('normal_reference', call)
    for mode, rt in [('cold', runtime), ('warm', ComparisonRuntime(args.report_dir / 'cache'))]:
        actual = timed(f'normal_{mode}', lambda: call(rt))
        for index, name in enumerate(('summary', 'files', 'alarms', 'manifest')):
            same_table(f'normal_{mode}_{name}', expected[index], actual[index])
        assert expected[4] == actual[4]
        for row in points.to_dict('records'):
            recs, _, _ = rt.load_normal_usage(reference['NORMAL_USAGE_SOURCE_DIR'], reference['NORMAL_USAGE_FILES'])
            ckpt = rt.checkpoint(ROOT / row['checkpoint_path'], row['checkpoint_sha256'])
            for rec in recs:
                p = rt.normal_prediction(ROOT / row['checkpoint_path'], ckpt, row['checkpoint_sha256'],
                                         row['evaluation_data'], rec, reference['NORMAL_USAGE_CHUNK_FRAMES'])
                assert normal_hashes[(row['checkpoint_sha256'], rec.sha256)] == hashlib.sha256(p.tobytes()).hexdigest()
        if mode == 'warm':
            assert rt.stats['model_forwards'] == rt.stats['prediction_computes'] == 0
    del expected, actual, rt
    print(f'All {len(normal_hashes)} full-normal probability traces are bitwise identical.', flush=True)

    detail_points = reference['operating_points'].merge(reference['comparison_sources'][[
        'model', 'folder', 'checkpoint_path', 'evaluation_data', 'evaluation_labels', 'evaluation_seed']],
        on='model', validate='one_to_one')
    detail_points['early_allowance_frames'] = reference['MATCH_EARLY_FRAMES']
    detail_points['max_late_frames'] = reference['MATCH_LATE_FRAMES']
    for split in ('valid', 'train'):
        call = lambda rt=None: examples.replay_examples(ROOT, detail_points, split=split,
                                                       source_dir=reference['MINING_SOURCE_DIR'], runtime=rt)
        expected = timed(f'{split}_reference', call)
        for mode, rt in [('cold', runtime), ('warm', ComparisonRuntime(args.report_dir / 'cache'))]:
            actual = timed(f'{split}_{mode}', lambda: call(rt))
            assert expected[1].keys() == actual[1].keys()
            for key in expected[1]:
                np.testing.assert_array_equal(expected[1][key], actual[1][key])
            for index, name in enumerate(('segments', 'files', 'alarms', 'status'), start=2):
                same_table(f'{split}_{mode}_{name}', expected[index], actual[index])
            if mode == 'warm':
                assert rt.stats['model_forwards'] == rt.stats['prediction_computes'] == 0
        print(f'{split}: {len(expected[1])} traces, tables, alarms and CSV formats match exactly.', flush=True)
        del expected, actual, rt
    # Exercise every plotting/matrix/inspection cell on the fast implementation.
    # Fresh runtime must hit the disk cache rather than repeat inference.
    new_runtime = ComparisonRuntime(args.report_dir / 'cache')
    copied['comparison_runtime'] = new_runtime
    from functools import partial
    copied['replay_normal_usage'] = partial(normal.replay_normal_usage, runtime=new_runtime)
    for index, cell in enumerate(fast['cells'][2:], start=2):
        if cell['cell_type'] == 'code':
            exec(compile(''.join(cell['source']), f'fast-comparison:cell{index}', 'exec'), copied)
            plt.close('all')
    assert new_runtime.stats['model_forwards'] == 0
    assert all(hashlib.sha256(p.read_bytes()).hexdigest() == digest for p, digest in snapshots.items())
    report = dict(models=detail_points.model.tolist(), matching_interval=[-reference['MATCH_EARLY_FRAMES'], reference['MATCH_LATE_FRAMES']],
                  timings=timings, tables=formats, normal_traces_verified=len(normal_hashes),
                  cpu_threads=2, exact_tables_csvs_and_probability_traces=True,
                  original_notebooks_unchanged=True, full_fast_notebook_executed=True)
    (args.report_dir / 'verification.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps({k: v for k, v in report.items() if k not in ('models', 'tables')}, indent=2))
    print('PASS. Report:', args.report_dir, flush=True)


if __name__ == '__main__':
    main()
