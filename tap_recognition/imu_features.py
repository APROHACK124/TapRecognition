"""Opt-in causal IMU features; no label access or data-dependent fitting here."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import math

import torch
from torch import nn
import torch.nn.functional as F

from .recording import IMU_COLUMNS


@dataclass(frozen=True)
class IMUFeatureConfig:
    """Keep signed axes and optionally append finite-history features.

    Differences are per sample (not divided by dt). RMS uses a fixed-length
    trailing window with zero prehistory. No future frames or labels are used.
    """

    version: int = 1
    magnitudes: bool = True
    differences: bool = True
    difference_magnitudes: bool = True
    rms_windows: tuple[int, ...] = (5, 15)

    def __post_init__(self):
        if self.version != 1:
            raise ValueError(f"Unsupported IMU feature version: {self.version}")
        windows = tuple(self.rms_windows)
        if any(type(w) is not int or w < 1 for w in windows):
            raise ValueError("rms_windows must contain positive integer sample counts")
        if len(set(windows)) != len(windows):
            raise ValueError("rms_windows must be unique")
        object.__setattr__(self, "rms_windows", windows)

    def to_dict(self):
        payload = asdict(self)
        payload["rms_windows"] = list(self.rms_windows)
        return payload

    @property
    def history_samples(self):
        return max((int(self.differences or self.difference_magnitudes),
                    *(w - 1 for w in self.rms_windows)))

    @property
    def names(self):
        names = list(IMU_COLUMNS)
        if self.magnitudes:
            names += ["acc_magnitude", "gyro_magnitude"]
        if self.differences:
            names += [f"delta_{name}" for name in IMU_COLUMNS]
        if self.difference_magnitudes:
            names += ["delta_acc_magnitude", "delta_gyro_magnitude"]
        for window in self.rms_windows:
            names += [f"acc_rms_{window}f", f"gyro_rms_{window}f"]
        return names


class CausalIMUFeatures(nn.Module):
    def __init__(self, config: IMUFeatureConfig):
        super().__init__()
        self.config = config

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 3 or x.shape[-1] != 6 or x.shape[1] == 0:
            raise ValueError("IMU features expect a nonempty [batch, time, 6] tensor")
        parts = [x]
        vectors = x.reshape(*x.shape[:2], 2, 3)
        if self.config.magnitudes:
            parts.append(torch.linalg.vector_norm(vectors, dim=-1))
        if self.config.differences or self.config.difference_magnitudes:
            # No previous observation at a window/recording reset: first delta is zero.
            delta = torch.cat((torch.zeros_like(x[:, :1]), x[:, 1:] - x[:, :-1]), dim=1)
            if self.config.differences:
                parts.append(delta)
            if self.config.difference_magnitudes:
                parts.append(torch.linalg.vector_norm(delta.reshape(*delta.shape[:2], 2, 3), dim=-1))
        if self.config.rms_windows:
            energy = vectors.square().sum(dim=-1).transpose(1, 2)
            for window in self.config.rms_windows:
                mean_energy = F.avg_pool1d(F.pad(energy, (window - 1, 0)), window, stride=1)
                # Finite derivatives at zero while mapping zero-energy history to zero.
                rms = (mean_energy + 1e-8).sqrt() - math.sqrt(1e-8)
                parts.append(rms.transpose(1, 2))
        return torch.cat(parts, dim=-1)

    def step(self, x, history, ready):
        """Fixed-shape export: [B,1,6], [B,history_samples,6], [1].

        Zero all states at a stream boundary. ready distinguishes first-frame
        zero differences from an actual previous observation of zero. RMS
        retains zero prehistory and a fixed divisor even during startup.
        """
        raw = torch.cat((history, x), dim=1)
        parts = [x]
        vectors = x.reshape(x.shape[0], 1, 2, 3)
        if self.config.magnitudes:
            parts.append(vectors.square().sum(-1).sqrt())
        if self.config.differences or self.config.difference_magnitudes:
            delta = (x - history[:, -1:]) * ready
            if self.config.differences:
                parts.append(delta)
            if self.config.difference_magnitudes:
                parts.append(delta.reshape(x.shape[0], 1, 2, 3).square().sum(-1).sqrt())
        for window in self.config.rms_windows:
            energy = raw[:, -window:].reshape(x.shape[0], window, 2, 3).square().sum(-1)
            parts.append((energy.sum(1, keepdim=True) / window + 1e-8).sqrt() - 1e-4)
        return torch.cat(parts, dim=-1), raw[:, 1:], torch.ones_like(ready)


class FeatureProjection(nn.Module):
    """Feature extraction -> frozen training-set normalization -> learned projection."""

    def __init__(self, config: IMUFeatureConfig, channels: int):
        super().__init__()
        self.features = CausalIMUFeatures(config)
        n = len(config.names)
        self.register_buffer("mean", torch.zeros(n))
        self.register_buffer("scale", torch.ones(n))
        self.register_buffer("normalization_fitted", torch.tensor(False))
        self.linear = nn.Linear(n, channels)

    def forward(self, x):
        return self.linear((self.features(x) - self.mean) / self.scale)

    @torch.no_grad()
    def fit_normalization(self, training_batches, min_scale: float = 1e-3):
        """Fit population mean/std using only the supplied raw training batches.

        Uses a stable merge of batch variances in float64. Statistics are frozen
        buffers afterwards, including during validation and positive augmentation.
        """
        if not math.isfinite(min_scale) or min_scale <= 0:
            raise ValueError("min_scale must be finite and positive")
        count, mean, m2 = 0, None, None
        for batch in training_batches:
            x = batch["imu"] if isinstance(batch, dict) else batch
            x = x.to(device=self.mean.device, dtype=self.mean.dtype)
            values = self.features(x).reshape(-1, self.mean.numel()).double()
            if not torch.isfinite(values).all():
                raise ValueError("Training features contain non-finite values")
            n = values.shape[0]
            batch_mean = values.mean(dim=0)
            batch_m2 = (values - batch_mean).square().sum(dim=0)
            if count == 0:
                mean, m2 = batch_mean, batch_m2
            else:
                delta = batch_mean - mean
                m2 = m2 + batch_m2 + delta.square() * (count * n / (count + n))
                mean = mean + delta * (n / (count + n))
            count += n
        if count == 0:
            raise ValueError("Cannot fit feature normalization on an empty training loader")
        self.mean.copy_(mean)
        self.scale.copy_((m2 / count).clamp_min(0).sqrt().clamp_min(min_scale))
        self.normalization_fitted.fill_(True)
        return count
