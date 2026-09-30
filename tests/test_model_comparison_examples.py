"""Checks for per-example replay alarm semantics."""

import unittest

import numpy as np

from notebooks_analysis.model_comparison_examples import emit_alarms_pair, match_alarms


class ModelComparisonExamplesTests(unittest.TestCase):
    def test_excluded_gap_resets_lookahead_and_refractory(self):
        probabilities = np.zeros((12, 3), dtype=float)
        probabilities[:, 0] = 1
        probabilities[3, 1] = .9
        probabilities[4, 1] = .99  # Excluded frame must not affect the candidate peak.
        probabilities[7, 2] = .9
        alarms = emit_alarms_pair(probabilities, .8, .8, 1, 50, [(0, 4), (6, 12)])
        self.assertEqual([(alarm['alarm_frame'], alarm['peak_frame'], alarm['class_id'])
                          for alarm in alarms], [(8, 7, 2)])

    def test_strict_timing_and_wrong_side_are_unmatched(self):
        truth = [(100, 1)]
        alarms = [dict(alarm_frame=98, class_id=1),
                  dict(alarm_frame=100, class_id=2)]
        self.assertEqual(match_alarms(truth, alarms, 1, 30), [])
        self.assertEqual(match_alarms(truth, alarms, 1, 30, class_aware=False), [(0, 1)])
        alarms.append(dict(alarm_frame=101, class_id=1))
        self.assertEqual(match_alarms(truth, alarms, 1, 30), [(0, 2)])


if __name__ == '__main__':
    unittest.main()
