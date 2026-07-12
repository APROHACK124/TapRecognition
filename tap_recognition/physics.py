"""
Physics-based IMU signal synthesis for PC double-tap recognition.

Implements the mathematical model in docs/mathematical_model.md:
  y(t) = R^T g + sum_k h(t - t_k; p_k) + background + noise
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
from scipy import signal

from .labels import gaussian_second_tap_labels


@dataclass
class TapParams:
    """Parameters for a single tap impulse response."""

    time: float  # tap onset (seconds)
    amplitude: float
    frequency: float  # Hz, chassis resonance
    damping: float  # zeta in (0, 1)
    direction: np.ndarray  # unit vector in R^6 (acc + gyro coupling)


@dataclass
class SampleMeta:
    """Metadata for one generated window."""

    label: int  # 1 = valid double-tap, 0 = negative
    tap_times: list[float]
    sample_rate: float
    delta_t: float | None = None  # inter-tap interval when two taps present


class IMUSimulator:
    """
    Synthesize 6-DoF IMU windows for double-tap / single-tap / background.

    Channels: [ax, ay, az, gx, gy, gz]
    """

    GRAVITY = 9.81
    DT_MIN = 0.12
    DT_MAX = 0.45

    def __init__(
        self,
        sample_rate: float = 100.0,
        window_samples: int = 64,
        seed: int | None = None,
        label_sigma: float = 0.04,
        label_peak_offset: float = 0.02,
    ):
        self.sample_rate = sample_rate
        self.window_samples = window_samples
        self.label_sigma = label_sigma
        self.label_peak_offset = label_peak_offset
        self.rng = np.random.default_rng(seed)

    def _damped_oscillation(
        self,
        t: np.ndarray,
        onset: float,
        amplitude: float,
        frequency: float,
        damping: float,
        direction: np.ndarray,
    ) -> np.ndarray:
        """h(t) = A * exp(-zeta * omega * tau) * sin(omega * tau + phi) * u."""
        tau = t - onset
        active = tau >= 0
        omega = 2 * np.pi * frequency
        phase = self.rng.uniform(0, 2 * np.pi)
        ring = np.zeros_like(t)
        ring[active] = (
            amplitude
            * np.exp(-damping * omega * tau[active])
            * np.sin(omega * tau[active] + phase)
        )
        # Short contact impulse (rect window ~3 ms)
        impulse_width = 0.003
        impulse = (tau >= 0) & (tau < impulse_width)
        ring[impulse] += amplitude * 3.0 * (1.0 - tau[impulse] / impulse_width)

        return ring[:, None] * direction[None, :]

    def _random_direction(self) -> np.ndarray:
        d = self.rng.normal(size=6)
        d[:3] *= 1.0  # accel dominant
        d[3:] *= 0.4  # gyro weaker
        norm = np.linalg.norm(d)
        return d / (norm + 1e-8)

    def _random_tap(self, onset: float) -> TapParams:
        return TapParams(
            time=onset,
            amplitude=self.rng.uniform(0.8, 2.5),
            frequency=self.rng.uniform(35, 95),
            damping=self.rng.uniform(0.08, 0.35),
            direction=self._random_direction(),
        )

    def _background(self, n: int) -> np.ndarray:
        t = np.arange(n) / self.sample_rate
        # Gravity in body frame (slowly varying tilt)
        tilt = self.rng.uniform(-0.15, 0.15, size=2)
        g = np.zeros((n, 3))
        g[:, 0] = self.GRAVITY * np.sin(tilt[0])
        g[:, 1] = self.GRAVITY * np.sin(tilt[1])
        g[:, 2] = -self.GRAVITY * np.cos(tilt[0]) * np.cos(tilt[1])

        # Fan / typing low-amplitude noise
        fan = 0.05 * np.sin(2 * np.pi * 45 * t)[:, None] * self.rng.normal(size=3)
        typing = np.zeros((n, 3))
        for _ in range(self.rng.integers(0, 4)):
            idx = self.rng.integers(0, n)
            typing[idx] += self.rng.normal(scale=0.15, size=3)

        gyro_bg = 0.02 * self.rng.normal(size=(n, 3))
        gyro_bg += 0.03 * np.gradient(g, axis=0)  # coupling from accel motion

        bg = np.hstack([g + fan + typing, gyro_bg])
        return bg

    def _synthesize(
        self,
        taps: list[TapParams],
        n: int,
    ) -> np.ndarray:
        t = np.arange(n) / self.sample_rate
        y = self._background(n)
        for tap in taps:
            y += self._damped_oscillation(
                t,
                tap.time,
                tap.amplitude,
                tap.frequency,
                tap.damping,
                tap.direction,
            )
        noise = self.rng.normal(scale=0.04, size=y.shape)
        return y + noise

    def _gaussian_second_tap_labels(
        self,
        tap_times: list[float],
        n: int,
    ) -> np.ndarray:
        if len(tap_times) < 2:
            return np.zeros(n, dtype=np.float32)
        return gaussian_second_tap_labels(
            n,
            self.sample_rate,
            tap_times[1],
            sigma=self.label_sigma,
            peak_offset=self.label_peak_offset,
        )

    def generate_window(
        self,
        kind: Literal["double", "single", "background"] | None = None,
    ) -> tuple[np.ndarray, np.ndarray, SampleMeta]:
        """Return (imu [T,6], soft_labels [T], meta)."""
        n = self.window_samples
        duration = n / self.sample_rate

        if kind is None:
            kind = self.rng.choice(
                ["double", "single", "background"],
                p=[0.50, 0.25, 0.25],
            )

        taps: list[TapParams] = []
        if kind == "double":
            t1 = self.rng.uniform(0.05, duration * 0.35)
            dt = self.rng.uniform(self.DT_MIN, self.DT_MAX)
            t2 = t1 + dt
            if t2 > duration - 0.05:
                t2 = duration - 0.05
                t1 = max(0.05, t2 - dt)
            taps = [self._random_tap(t1), self._random_tap(t2)]
        elif kind == "single":
            t1 = self.rng.uniform(0.08, duration * 0.6)
            taps = [self._random_tap(t1)]

        y = self._synthesize(taps, n)
        tap_times = [t.time for t in taps]
        labels = self._gaussian_second_tap_labels(tap_times, n)
        delta_t = (tap_times[1] - tap_times[0]) if len(tap_times) >= 2 else None
        label = int(labels.max() > 0.5)

        return y.astype(np.float32), labels, SampleMeta(
            label=label,
            tap_times=tap_times,
            sample_rate=self.sample_rate,
            delta_t=delta_t,
        )

    def highpass(self, y: np.ndarray, cutoff: float = 0.5) -> np.ndarray:
        """Remove gravity drift (causal-friendly one-pole high-pass)."""
        fs = self.sample_rate
        b, a = signal.butter(2, cutoff / (fs / 2), btype="high", analog=False)
        # filtfilt is non-causal; use lfilter for online-compatible offline prep
        out = np.zeros_like(y)
        for ch in range(y.shape[1]):
            out[:, ch] = signal.lfilter(b, a, y[:, ch])
        return out.astype(np.float32)
