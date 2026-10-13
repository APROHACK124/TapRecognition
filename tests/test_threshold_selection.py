"""Exercise the evaluator's actual notebook selector and FP-rate calculations."""

import json
from pathlib import Path
from types import SimpleNamespace
import unittest

import numpy as np
import pandas as pd

from notebooks_analysis.recording_durations import durations_for_recordings


class ThresholdSelectionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        root = Path(__file__).resolve().parents[1]
        cls.notebook = json.loads((root / 'main_notebooks/evaluate.ipynb').read_text())
        cls.code = [''.join(cell['source']) for cell in cls.notebook['cells']
                    if cell['cell_type'] == 'code']
        cls.selector_cell = next(s for s in cls.code if 'def select_operating_point(' in s)

    def setUp(self):
        self.namespace = dict(np=np, pd=pd, FP_SELECTION_METRIC='negative_fp_per_min',
                              RECALL_TARGET=.94, FP_PER_MIN_LIMIT=.7)
        exec(self.selector_cell.split('selected_rows = []')[0], self.namespace)

    def candidates(self):
        return pd.DataFrame([
            dict(policy='lookahead_12f', left_threshold=.5, right_threshold=.5,
                 left_recall=.96, right_recall=.96, macro_recall=.96, macro_f1=.94,
                 macro_timely_recall=.9, fp_per_min=2., negative_fp_per_min=.2),
            dict(policy='lookahead_12f', left_threshold=.6, right_threshold=.6,
                 left_recall=.99, right_recall=.99, macro_recall=.99, macro_f1=.95,
                 macro_timely_recall=.95, fp_per_min=.1, negative_fp_per_min=.8),
        ])

    def test_budget_selects_negative_rate_not_total_rate(self):
        for mode in ('fp_budget_max_recall', 'timely_recall_fp_budget'):
            with self.subTest(mode=mode):
                winner = self.namespace['select_operating_point'](self.candidates(), mode)
                self.assertEqual(winner.left_threshold, .5)
                self.assertGreater(winner.fp_per_min, self.namespace['FP_PER_MIN_LIMIT'])
                self.assertLess(winner.negative_fp_per_min, self.namespace['FP_PER_MIN_LIMIT'])

    def test_total_rate_option_restores_old_budget(self):
        self.namespace['FP_SELECTION_METRIC'] = 'fp_per_min'
        winner = self.namespace['select_operating_point'](self.candidates(), 'fp_budget_max_recall')
        self.assertEqual(winner.left_threshold, .6)

    def test_limit_is_strict_and_undefined_rates_are_infeasible(self):
        candidates = self.candidates()
        candidates['negative_fp_per_min'] = [.7, np.nan]
        self.assertIsNone(self.namespace['select_operating_point'](candidates, 'fp_budget_max_recall'))
        candidates.loc[0, 'negative_fp_per_min'] = .69
        self.assertEqual(self.namespace['select_operating_point'](candidates, 'fp_budget_max_recall').left_threshold, .5)

    def test_recall_constrained_min_fp_and_ties_use_negative_rate(self):
        select = self.namespace['select_operating_point']
        candidates = self.candidates()
        self.assertEqual(select(candidates, 'recall_target_min_fp').left_threshold, .5)
        candidates['macro_f1'] = .95
        self.assertEqual(select(candidates, 'max_f1').left_threshold, .5)
        candidates['left_recall'] = [.93, .99]
        self.assertEqual(select(candidates, 'recall_target_min_fp').left_threshold, .6)

    def test_max_f1_primary_objective_is_unchanged(self):
        winner = self.namespace['select_operating_point'](self.candidates(), 'max_f1')
        self.assertEqual(winner.left_threshold, .6)
        self.assertGreater(winner.negative_fp_per_min, self.namespace['FP_PER_MIN_LIMIT'])

    def test_selected_table_records_metric_and_reports_infeasibility(self):
        self.namespace.update(pair_sweep=self.candidates(), POLICIES={'lookahead_12f': 12},
                              SELECTION_MODES=['max_f1', 'fp_budget_max_recall'])
        exec(self.selector_cell, self.namespace)
        table = self.namespace['selected_thresholds']
        self.assertTrue(table.feasible.all())
        self.assertTrue(table.fp_selection_metric.eq('negative_fp_per_min').all())
        self.assertIn('negative_fp_per_min', table)
        self.namespace['pair_sweep']['negative_fp_per_min'] = np.nan
        exec(self.selector_cell, self.namespace)
        budget = self.namespace['selected_thresholds'].iloc[1]
        self.assertFalse(budget.feasible)
        self.assertIn('undefined', budget.infeasible_reason)
        self.assertIn('negative_fp_per_min', budget.infeasible_reason)

    def test_negative_rate_uses_only_raw_negative_recording_minutes(self):
        source = next(s for s in self.code if 'def evaluate_pair_events(' in s)
        function_source = source.split('sweep_rows = []')[0]
        positive = SimpleNamespace(path=Path('positive.csv'), imu=np.zeros((6000, 6)),
                                   fs=100., events=[(100, 1), (200, 2)])
        negative = SimpleNamespace(path=Path('negative.csv'), imu=np.zeros((300, 6)),
                                   fs=100., events=[])
        alarms = {
            'positive.csv': [dict(alarm_frame=100, class_id=1), dict(alarm_frame=200, class_id=2),
                             dict(alarm_frame=300, class_id=1)],
            'negative.csv': [dict(alarm_frame=50, class_id=1)],
        }
        ns = dict(np=np, POLICIES={'lookahead_12f': 12}, DEADLINES=(5, 10, 20, 30),
                  TIMELY_BUDGET_FRAMES=10, recordings=[positive, negative], probabilities=alarms,
                  emit_alarms_pair=lambda probabilities, *_: probabilities,
                  match_alarms=lambda truth, alarms, **_: [(0, 0), (1, 1)] if truth else [],
                  divide=lambda a, b: a / b if b else np.nan)
        ns['duration_audit'] = durations_for_recordings(Path.cwd(), ns['recordings'])
        ns['duration_lookup'] = ns['duration_audit'].set_index('file')
        exec(function_source, ns)
        overall, _ = ns['evaluate_pair_events'](.5, .5, 'lookahead_12f')
        self.assertEqual(overall['fp'], 2)
        self.assertAlmostEqual(overall['negative_minutes'], .05)
        self.assertAlmostEqual(overall['negative_fp_per_min'], 20.)
        self.assertAlmostEqual(overall['fp_per_min'], 2 / 1.05)
        # More positive duration changes total FP/min but not negative-only FP/min.
        positive.imu = np.zeros((12000, 6))
        ns['duration_audit'] = durations_for_recordings(Path.cwd(), ns['recordings'])
        ns['duration_lookup'] = ns['duration_audit'].set_index('file')
        again, _ = ns['evaluate_pair_events'](.5, .5, 'lookahead_12f')
        self.assertAlmostEqual(again['negative_fp_per_min'], overall['negative_fp_per_min'])
        self.assertLess(again['fp_per_min'], overall['fp_per_min'])

    def test_notebook_compiles_and_metadata_preserves_selection_metric(self):
        for index, source in enumerate(self.code):
            compile(source, f'evaluate.ipynb:cell{index}', 'exec')
        settings = next(s for s in self.code if 'FP_SELECTION_METRIC =' in s)
        self.assertIn("FP_SELECTION_METRIC = 'negative_fp_per_min'", settings)
        metadata = next(s for s in self.code if 'comparison_metadata = dict(' in s)
        self.assertIn('fp_selection_metric=FP_SELECTION_METRIC', metadata)
        long_format = next(s for s in self.code if 'comparison_summary.melt' in s)
        self.assertIn("'fp_selection_metric'", long_format)


if __name__ == '__main__':
    unittest.main()
