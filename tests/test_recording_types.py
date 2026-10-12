"""Tests for stable per-recording report type labels."""

import unittest

from notebooks_analysis.recording_types import recording_type_from_filename


class RecordingTypesTests(unittest.TestCase):
    def test_standard_and_mined_types(self):
        cases = {
            'imu_CC_arbitrary_20260901_154914.csv': 'arbitrary',
            'imu_CC_knock_twice_left_20260901_153013.csv': 'knock_twice',
            'imuStrict_Miguel_knock_twice_right_20260929_115836.csv': 'knock_twice',
            'imu_Miguel_switch_doubletap_20261012_170000.csv': 'switch_doubletap',
            'imu_Miguel_negative_mining_20261009_164938.csv': 'hard_negative',
            'imu_Miguel_negative_mining_20261009_164938__crop_f000495_f000795.csv': 'hard_negative',
        }
        for name, expected in cases.items():
            with self.subTest(name=name):
                self.assertEqual(recording_type_from_filename(name), expected)

    def test_malformed_or_non_negative_crop_is_rejected(self):
        for name in (
            'imu_Miguel_knock_once_20261009_164938__crop_f000495_f000795.csv',
            'imu_Miguel_switch_doubletap_20261012_170000__crop_f000495_f000795.csv',
            'imu_Miguel_negative_mining_20261009_164938__crop_f000795_f000495.csv',
            'not-a-recording.csv',
            'imu_Miguel_negative_mining_20261009_164938.txt',
        ):
            with self.subTest(name=name):
                with self.assertRaises(ValueError):
                    recording_type_from_filename(name)

    def test_action_fallback_keeps_nonstandard_replay_files_usable(self):
        self.assertEqual(recording_type_from_filename('fixture.csv', 'knock_twice_left'), 'knock_twice')
        self.assertEqual(recording_type_from_filename('fixture.csv', 'negative_mining'), 'hard_negative')
        self.assertEqual(recording_type_from_filename('fixture.csv', 'switch_doubletap'), 'switch_doubletap')


if __name__ == '__main__':
    unittest.main()
