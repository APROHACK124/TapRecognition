"""Strict spike-window behavior for auto-labeling."""

import unittest
from unittest.mock import patch

import numpy as np

from tap_recognition.labels import detect_second_tap_frame


class AutoLabelWindowTest(unittest.TestCase):
    def detect_pair(self, peaks: list[int]) -> tuple[int, int] | None:
        energy = np.zeros(400, dtype=np.float32)
        energy[peaks] = 1.0
        with (
            patch("tap_recognition.labels.acc_energy_delta", return_value=energy),
            patch("tap_recognition.labels._local_peaks", return_value=peaks),
        ):
            return detect_second_tap_frame(
                np.zeros((400, 3), dtype=np.float32),
                sample_rate=100.0,
                first_tap_after_sec=2.0,
                second_tap_before_sec=2.85,
            )

    def test_strict_window_accepts_interior_pair(self) -> None:
        self.assertEqual(self.detect_pair([201, 230]), (201, 230))

    def test_strict_window_excludes_start_and_end_boundaries(self) -> None:
        self.assertIsNone(self.detect_pair([200, 230]))
        self.assertIsNone(self.detect_pair([250, 285]))

    def test_strict_window_rejects_invalid_bounds(self) -> None:
        with self.assertRaisesRegex(ValueError, "must be before"):
            detect_second_tap_frame(
                np.zeros((1, 3), dtype=np.float32),
                first_tap_after_sec=2.85,
                second_tap_before_sec=2.0,
            )


if __name__ == "__main__":
    unittest.main()
