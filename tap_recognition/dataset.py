"""PyTorch dataset for synthetic and recorded IMU double-tap windows."""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np
import torch
from torch.utils.data import Dataset

from .labels import gaussian_second_tap_labels, estimate_first_tap_frame, is_valid_delta_t
from .physics import IMUSimulator
from .recording import IMU_COLUMNS, LabelParams, estimate_sample_rate_hz


@dataclass(frozen=True)
class RecordedWindowMeta:
    source_file: str
    trigger_frame: int | None = None


def load_trigger_frames(label_path: Path) -> list[int]:
    if not label_path.exists() or label_path.stat().st_size == 0:
        return []

    frames: list[int] = []
    for line in label_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        frames.append(int(line.split(",")[0].strip()))
    return sorted(set(frames))


def load_full_recording(csv_path: Path) -> tuple[np.ndarray, float]:
    with csv_path.open(encoding="utf-8-sig", newline="") as f:
        rows = list(csv.DictReader(f))

    if not rows:
        raise ValueError(f"No rows in {csv_path}")

    timestamp_ms = np.array([float(r["timestamp_ms"]) for r in rows], dtype=np.float64)
    imu = np.array(
        [[float(r[c]) for c in IMU_COLUMNS] for r in rows],
        dtype=np.float32,
    )
    sample_rate = estimate_sample_rate_hz(timestamp_ms)
    return imu, sample_rate


class IMUDoubleTapDataset(Dataset):
    def __init__(
        self,
        num_samples: int = 4000,
        window_samples: int = 64,
        sample_rate: float = 100.0,
        seed: int = 42,
        highpass: bool = True,
    ):
        self.sim = IMUSimulator(
            sample_rate=sample_rate,
            window_samples=window_samples,
            seed=seed,
        )
        self.highpass = highpass
        self.samples: list[tuple] = []
        for i in range(num_samples):
            y, labels, meta = self.sim.generate_window()
            if highpass:
                y = self.sim.highpass(y)
            self.samples.append((y, labels, meta.label))

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        y, labels, window_label = self.samples[idx]
        return {
            "imu": torch.from_numpy(y),
            "frame_labels": torch.from_numpy(labels),
            "window_label": torch.tensor(window_label, dtype=torch.float32),
        }


class RecordedIMUDataset(Dataset):
    """Sliding-window dataset built from labeled CSV recordings."""

    def __init__(
        self,
        data_dir: str | Path,
        window_samples: int = 64,
        label_params: LabelParams | None = None,
        highpass: bool = True,
        negative_windows_per_file: int = 20,
        trigger_context_samples: int = 48,
        exclusion_margin: int = 80,
        dt_min: float = 0.12,
        dt_max: float = 0.45,
        seed: int = 42,
        csv_paths: Sequence[str | Path] | None = None,
    ):
        self.window_samples = window_samples
        self.label_params = label_params or LabelParams()
        self.highpass = highpass
        self.trigger_context_samples = trigger_context_samples
        self.exclusion_margin = exclusion_margin
        self.dt_min = dt_min
        self.dt_max = dt_max
        self.rng = np.random.default_rng(seed)
        self.sim = IMUSimulator(
            sample_rate=100.0,
            window_samples=window_samples,
            seed=seed,
        )
        self.samples: list[tuple[np.ndarray, np.ndarray, int]] = []
        self.window_meta: list[RecordedWindowMeta] = []

        csv_files = sorted(Path(data_dir).glob("*.csv"))
        if csv_paths is not None:
            allowed = {Path(p).resolve() for p in csv_paths}
            csv_files = [p for p in csv_files if p.resolve() in allowed]
        if not csv_files:
            raise ValueError(f"No CSV files found in {data_dir}")

        for csv_path in csv_files:
            imu, sample_rate = load_full_recording(csv_path)
            if highpass:
                self.sim.sample_rate = sample_rate
                imu = self.sim.highpass(imu)

            label_path = csv_path.with_suffix(".txt")
            trigger_frames = load_trigger_frames(label_path)

            for trigger_frame in trigger_frames:
                window, labels = self._window_for_trigger(
                    imu, trigger_frame, sample_rate
                )
                if window is not None:
                    self.samples.append((window, labels, 1))
                    self.window_meta.append(
                        RecordedWindowMeta(csv_path.name, trigger_frame)
                    )

            negative_starts = self._sample_negative_starts(
                len(imu),
                trigger_frames,
                negative_windows_per_file,
            )
            for start in negative_starts:
                window = imu[start : start + window_samples]
                if len(window) < window_samples:
                    continue
                self.samples.append(
                    (window.astype(np.float32), np.zeros(window_samples, np.float32), 0)
                )
                self.window_meta.append(RecordedWindowMeta(csv_path.name, None))

        if not self.samples:
            raise ValueError(f"No training windows built from {data_dir}")

    def _window_for_trigger(
        self,
        imu: np.ndarray,
        trigger_frame: int,
        sample_rate: float,
    ) -> tuple[np.ndarray | None, np.ndarray | None]:
        if trigger_frame < 0 or trigger_frame >= len(imu):
            return None, None

        start = trigger_frame - self.trigger_context_samples
        end = start + self.window_samples

        if start < 0:
            start = 0
            end = self.window_samples
        if end > len(imu):
            end = len(imu)
            start = max(0, end - self.window_samples)

        window = imu[start:end]
        if len(window) < self.window_samples:
            return None, None

        t2_frame_in_window = trigger_frame - start
        t1_frame_in_window = estimate_first_tap_frame(window, t2_frame_in_window)
        if t1_frame_in_window is not None:
            delta_t = (t2_frame_in_window - t1_frame_in_window) / sample_rate
            if not is_valid_delta_t(delta_t, self.dt_min, self.dt_max):
                return None, None

        t2_sec = t2_frame_in_window / sample_rate
        labels = gaussian_second_tap_labels(
            self.window_samples,
            sample_rate,
            t2_sec,
            sigma=self.label_params.sigma,
            peak_offset=self.label_params.peak_offset,
        )
        return window.astype(np.float32), labels

    def _sample_negative_starts(
        self,
        n_samples: int,
        trigger_frames: list[int],
        count: int,
    ) -> list[int]:
        if n_samples <= self.window_samples:
            return []

        forbidden: list[tuple[int, int]] = []
        for frame in trigger_frames:
            forbidden.append(
                (
                    max(0, frame - self.exclusion_margin),
                    min(n_samples, frame + self.exclusion_margin),
                )
            )

        starts: list[int] = []
        attempts = 0
        max_attempts = count * 50
        while len(starts) < count and attempts < max_attempts:
            attempts += 1
            start = int(self.rng.integers(0, n_samples - self.window_samples + 1))
            end = start + self.window_samples
            if any(not (end <= lo or start >= hi) for lo, hi in forbidden):
                continue
            starts.append(start)
        return starts

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        y, labels, window_label = self.samples[idx]
        return {
            "imu": torch.from_numpy(y),
            "frame_labels": torch.from_numpy(labels),
            "window_label": torch.tensor(window_label, dtype=torch.float32),
        }
