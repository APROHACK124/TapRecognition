"""Golden-reference alarm, metric, selection, schema, and cache equivalence."""

import ast
import contextlib
from functools import partial
import io
import json
from pathlib import Path
import tempfile
from time import perf_counter
from types import SimpleNamespace
import unittest

import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment

from notebooks_analysis.fast_pair_evaluation import (
    PairMetricsCache, PreparedTrace, TracePool, normal_usage_counts, sweep_context,
)
from notebooks_analysis.model_comparison_examples import emit_alarms_pair as reference_emit
from notebooks_analysis.normal_usage_evaluation import evaluate_normal_usage
from tools.build_fast_evaluation_notebook import build_notebook


ROOT = Path(__file__).resolve().parents[1]


def notebook_code(notebook, needle):
    return next(''.join(c['source']) for c in notebook['cells']
                if c['cell_type'] == 'code' and needle in ''.join(c['source']))


def definitions(code, names, env):
    nodes = [node for node in ast.parse(code).body
             if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name in names]
    exec(compile(ast.Module(body=nodes, type_ignores=[]), '<reference-definitions>', 'exec'), env)


class PreparedTraceTests(unittest.TestCase):
    def test_random_quantized_and_boundary_traces_match_exact_alarms(self):
        rng = np.random.default_rng(51)
        for dtype in (np.float32, np.float64):
            for frames in (0, 1, 12, 13, 14, 59, 301):
                scores = rng.choice([0., .1, .3, .5, .7, .9, 1.], (frames, 3)).astype(dtype)
                prepared = PreparedTrace(scores)
                for look in (0, 1, 12):
                    for refractory in (0, 1, 5, 50):
                        for left, right in ((.1, .9), (.5, .5), (.7, .3), (.9, .9), (1., 1.)):
                            reference = reference_emit(scores, left, right, look, refractory, [(0, frames)])
                            self.assertEqual(prepared.emit(left, right, look, refractory), reference)
                            self.assertEqual(prepared.counts(left, right, look, refractory),
                                             tuple(sum(a['class_id'] == c for a in reference) for c in (1, 2)))

    def test_earliest_frame_wins_cross_side_ties_and_nonqualifying_peak_is_ignored(self):
        p = np.zeros((40, 3), dtype=np.float32)
        p[0, 1] = .6
        p[2, 2] = .8
        p[4, 1] = .8
        prepared = PreparedTrace(p)
        self.assertEqual(prepared.emit(.5, .5, 12, 50)[0]['class_id'], 2)
        self.assertEqual(prepared.emit(.5, .9, 12, 50)[0]['class_id'], 1)
        p[2, 1] = .8  # Same frame and score: left wins.
        self.assertEqual(PreparedTrace(p).emit(.5, .5, 12, 50)[0]['class_id'], 1)

    def test_float32_threshold_promotion_matches_array_reference(self):
        p = np.zeros((30, 3), dtype=np.float32)
        p[0, 1] = np.float32(.7)  # Slightly below float64 .7.
        p[15, 2] = np.nextafter(np.float32(.7), np.float32(1))
        self.assertEqual(PreparedTrace(p).emit(.7, .7, 0, 1),
                         reference_emit(p, .7, .7, 0, 1, [(0, len(p))]))

    def test_dense_continuous_probabilities(self):
        rng = np.random.default_rng(17)
        for dtype in (np.float32, np.float64):
            p = rng.dirichlet([1, 1, 1], 2500).astype(dtype)
            prepared = PreparedTrace(p)
            for left in (.1, .4, .8):
                for right in (.1, .55, .9):
                    for look in (0, 12):
                        self.assertEqual(prepared.emit(left, right, look, 50),
                                         reference_emit(p, left, right, look, 50, [(0, len(p))]))


class FastSweepTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.original = json.loads((ROOT / 'main_notebooks/evaluate.ipynb').read_text())
        cls.fast = build_notebook(cls.original)

    def fixture(self):
        rng = np.random.default_rng(14)
        recordings, traces = [], {}
        for i, events in enumerate(([(100, 1), (350, 2)], [(120, 2), (420, 1)], [])):
            path = Path(f'fixture_{i}.csv')
            p = np.zeros((600, 3), dtype=np.float32)
            p[:, 0] = 1
            for frame, cls in events:
                p[frame - 5:frame + 8, cls] = rng.uniform(.5, .95)
            p[40:45, 1] = .75
            p[520:530, 2] = .6
            recordings.append(SimpleNamespace(path=path, fs=100., events=events, imu=np.zeros((600, 6))))
            traces[path.name] = p
        normal, normal_traces = [], {}
        for i in range(3):
            rec = SimpleNamespace(path=Path(f'normal_{i}.csv'), fs=100., events=[],
                                  sha256=f'fixture-{i}', data=np.zeros((3000, 1)))
            p = np.zeros((3000, 3), dtype=np.float32)
            p[:, 0] = 1
            if i != 2:  # Include a zero-alarm original's entire exposure.
                p[299, 1] = .9
                p[301, 2] = .95  # Shared cooldown crosses a would-be session boundary.
                p[2100:2105, 2] = .7
            normal.append(rec)
            normal_traces[rec.path.name] = p
        audit = pd.DataFrame([dict(file=r.path.name, minutes=.1, evaluated_minutes=.1) for r in recordings])
        return dict(np=np, pd=pd, perf_counter=perf_counter, recordings=recordings, probabilities=traces,
                    duration_lookup=audit.set_index('file'), duration_audit=audit,
                    POLICIES={'instant': 0, 'lookahead_12f': 12}, DEADLINES=(5, 10, 15, 20, 30),
                    EARLY_ALLOWANCE_FRAMES=25, MAX_LATE_FRAMES=40, TIMELY_BUDGET_FRAMES=10,
                    normal_usage_recordings=normal, normal_usage_probabilities=normal_traces,
                    validation_crop_normal_usage_recordings=normal[1:],
                    validation_crop_normal_usage_probabilities={r.path.name: normal_traces[r.path.name] for r in normal[1:]},
                    evaluate_normal_usage=evaluate_normal_usage, linear_sum_assignment=linear_sum_assignment,
                    divide=lambda a, b: float(a / b) if b else float('nan'),
                    PAIR_THRESHOLDS=np.array([.5, .7, .9]), THRESHOLDS=np.array([.5, .7, .9]),
                    PAIR_SWEEP_CACHE_DIR=None, PairMetricsCache=PairMetricsCache, sweep_context=sweep_context)

    def environments(self):
        reference = self.fixture()
        definitions(notebook_code(self.original, 'def match_alarms('), {'match_alarms'}, reference)
        definitions(notebook_code(self.original, 'def emit_alarms_pair('), {'emit_alarms_pair'}, reference)
        code = notebook_code(self.original, 'def evaluate_pair_events(')
        exec(code.split('sweep_rows = []')[0], reference)
        event_rows = []
        for policy in reference['POLICIES']:
            for threshold in reference['THRESHOLDS']:
                overall, _ = reference['evaluate_pair_events'](threshold, threshold, policy)
                event_rows.append(dict(policy=policy, threshold=threshold, **overall))
        reference['event_summary'] = pd.DataFrame(event_rows)
        fast = dict(reference)
        pool = TracePool()
        fast.update(fast_traces=pool, emit_alarms_pair=pool.emit,
                    fast_normal_usage_counts=partial(normal_usage_counts, pool))
        return reference, fast

    def test_full_sweep_side_metrics_and_selected_thresholds_match_values_and_format(self):
        reference, fast = self.environments()
        with contextlib.redirect_stdout(io.StringIO()):
            exec(notebook_code(self.original, 'def evaluate_pair_events('), reference)
            exec(notebook_code(self.fast, 'def evaluate_pair_events('), fast)
        pd.testing.assert_frame_equal(reference['pair_sweep'], fast['pair_sweep'], check_exact=True)
        for policy in reference['POLICIES']:
            for left in reference['PAIR_THRESHOLDS']:
                for right in reference['PAIR_THRESHOLDS']:
                    expected = reference['evaluate_pair_events'](left, right, policy)
                    actual = fast['evaluate_pair_events'](left, right, policy)
                    pd.testing.assert_frame_equal(pd.DataFrame([expected[0]]), pd.DataFrame([actual[0]]), check_exact=True)
                    pd.testing.assert_frame_equal(pd.DataFrame(expected[1]), pd.DataFrame(actual[1]), check_exact=True)
        # All selector modes and all three FP-budget populations retain the same winners/schema.
        modes = ['max_f1', 'recall_target_max_threshold', 'recall_target_min_fp',
                 'fp_budget_max_recall', 'timely_recall_fp_budget']
        for metric in ('negative_fp_per_min', 'fp_per_min', 'normal_usage_fp_per_min'):
            for env, notebook in ((reference, self.original), (fast, self.fast)):
                env.update(SELECTION_MODES=modes, RECALL_TARGET=.5, FP_PER_MIN_LIMIT=10., FP_SELECTION_METRIC=metric)
                exec(notebook_code(notebook, 'def select_operating_point('), env)
            pd.testing.assert_frame_equal(reference['selected_thresholds'], fast['selected_thresholds'], check_exact=True)

    def test_normal_subsets_reuse_counts_and_preserve_empty_rates(self):
        env = self.fixture()
        pool = TracePool()
        for recs in (env['normal_usage_recordings'], env['validation_crop_normal_usage_recordings'], []):
            expected = evaluate_normal_usage(recs, env['normal_usage_probabilities'], .7, .7, 12)
            actual = normal_usage_counts(pool, recs, env['normal_usage_probabilities'], .7, .7, 12)
            pd.testing.assert_frame_equal(pd.DataFrame([expected[0]]), pd.DataFrame([actual[0]]), check_exact=True)
            pd.testing.assert_frame_equal(pd.DataFrame(expected[1]), pd.DataFrame(actual[1]), check_exact=True)
        self.assertEqual(len(pool.traces), 3)
        self.assertTrue(all(len(prepared.count_cache) == 1 for _, prepared in pool.traces.values()))

    def test_persistent_cache_round_trip_and_invalidation(self):
        reference, _ = self.environments()
        result = reference['evaluate_pair_events'](.5, .7, 'instant')
        with tempfile.TemporaryDirectory() as temp:
            first = PairMetricsCache(temp, 'context')
            cached = first.memoize(lambda *_: result)
            cached(.5, .7, 'instant')[0]['tp'] = -999  # A caller must not corrupt cached results.
            pd.testing.assert_frame_equal(pd.DataFrame([cached(.5, .7, 'instant')[0]]), pd.DataFrame([result[0]]), check_exact=True)
            second = PairMetricsCache(temp, 'context')
            restored = second.memoize(lambda *_: self.fail('Expected a disk hit'))(.5, .7, 'instant')
            pd.testing.assert_frame_equal(pd.DataFrame([restored[0]]), pd.DataFrame([result[0]]), check_exact=True)
            pd.testing.assert_frame_equal(pd.DataFrame(restored[1]), pd.DataFrame(result[1]), check_exact=True)
            self.assertEqual(second.disk_hits, 1)
            other = PairMetricsCache(temp, 'new-context')
            other.memoize(lambda *_: result)(.5, .7, 'instant')
            self.assertEqual(other.misses, 1)
        kwargs = dict(recordings=reference['recordings'], probabilities=reference['probabilities'],
                      duration_audit=reference['duration_audit'], populations=[], settings=dict(look=12))
        before = sweep_context(**kwargs)
        reference['recordings'][0].events.append((550, 1))
        self.assertNotEqual(before, sweep_context(**kwargs))
        reference['recordings'][0].events.pop()
        reference['duration_audit'].loc[0, 'minutes'] += .1
        self.assertNotEqual(before, sweep_context(**kwargs))
        reference['duration_audit'].loc[0, 'minutes'] -= .1
        kwargs['settings']['look'] = 0
        self.assertNotEqual(before, sweep_context(**kwargs))
        kwargs['settings']['look'] = 12
        reference['probabilities']['fixture_0.csv'][0, 1] = .1
        self.assertNotEqual(before, sweep_context(**kwargs))

    def test_copy_preserves_metric_body_columns_and_safe_export_defaults(self):
        original = notebook_code(self.original, 'def evaluate_pair_events(')
        fast = notebook_code(self.fast, 'def evaluate_pair_events(')
        original_body = original.split('def evaluate_pair_events', 1)[1].split('sweep_rows = []')[0]
        fast_body = fast.split('def evaluate_pair_events', 1)[1].split('sweep_started = ')[0]
        self.assertEqual(original_body, fast_body.replace('fast_normal_usage_counts(', 'evaluate_normal_usage('))
        self.assertEqual(notebook_code(self.original, 'def select_operating_point('),
                         notebook_code(self.fast, 'def select_operating_point('))
        for c in self.fast['cells']:
            if c['cell_type'] == 'code':
                compile(''.join(c['source']), 'fast-notebook', 'exec')
                self.assertEqual(c['outputs'], [])
        self.assertIn('SAVE_COMPARISON = False', notebook_code(self.fast, 'SAVE_COMPARISON = False'))

    def test_inference_reuses_existing_normal_traces_without_eager_fallback(self):
        inference = notebook_code(self.fast, 'validation_crop_normal_usage_probabilities = {}')
        snippet = 'validation_crop_normal_usage_probabilities = {}' + inference.split(
            'validation_crop_normal_usage_probabilities = {}', 1)[1].split(
                "print('Full normal_usage originals referenced by validation crops:'", 1)[0]
        calls = []
        cached, new = np.ones((3, 3)), np.zeros((3, 3))
        recs = [SimpleNamespace(path=Path('cached.csv')), SimpleNamespace(path=Path('new.csv'))]

        def predict(rec, *_):
            calls.append(rec.path.name)
            return new

        env = dict(validation_crop_normal_usage_recordings=recs,
                   normal_usage_probabilities={'cached.csv': cached}, predict_recording=predict,
                   normal_usage_model=None, NORMAL_USAGE_CHUNK_FRAMES=4096)
        exec(snippet, env)
        self.assertEqual(calls, ['new.csv'])
        self.assertIs(env['validation_crop_normal_usage_probabilities']['cached.csv'], cached)


if __name__ == '__main__':
    unittest.main()
