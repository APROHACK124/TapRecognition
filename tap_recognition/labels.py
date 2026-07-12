"""Gaussian second-tap soft labels (mathematical_model.md Section 5.4)."""

from __future__ import annotations

import numpy as np

DT_MIN = 0.12
DT_MAX = 0.45


def is_valid_delta_t(
    delta_t: float,
    dt_min: float = DT_MIN,
    dt_max: float = DT_MAX,
) -> bool:
    """Return True when the inter-tap interval is within the allowed range."""
    return dt_min <= delta_t <= dt_max


def gaussian_second_tap_labels(
    n_samples: int,
    sample_rate: float,
    t2_sec: float | None,
    *,
    sigma: float = 0.04,
    peak_offset: float = 0.02,
) -> np.ndarray:
    """
    Build Gaussian soft labels centered on the second tap.

    Returns:
        labels: float32 array [n_samples]
    """
    labels = np.zeros(n_samples, dtype=np.float32)
    if t2_sec is None:
        return labels

    t = np.arange(n_samples, dtype=np.float32) / sample_rate
    t_peak = t2_sec + peak_offset
    labels = np.exp(-0.5 * ((t - t_peak) / sigma) ** 2)
    return labels.astype(np.float32)


def estimate_first_tap_frame(imu: np.ndarray, t2_frame: int) -> int | None:
    """
    Estimate the first tap as the strongest acc-energy delta before t2.

    Used to filter recorded training windows by inter-tap interval.
    """
    if t2_frame <= 0:
        return None

    acc_mag = np.linalg.norm(imu[:, :3], axis=1)
    acc_delta = np.abs(np.diff(acc_mag, prepend=acc_mag[0]))
    if t2_frame >= len(acc_delta):
        return None

    t1_frame = int(np.argmax(acc_delta[:t2_frame]))
    if t1_frame <= 0 or t1_frame >= t2_frame:
        return None
    return t1_frame
