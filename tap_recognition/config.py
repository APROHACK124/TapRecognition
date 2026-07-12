"""Load training hyperparameters from YAML config files."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from .recording import LabelParams

try:
    import yaml
except ImportError as exc:  # pragma: no cover - optional at import time
    yaml = None
    _YAML_IMPORT_ERROR = exc
else:
    _YAML_IMPORT_ERROR = None


@dataclass
class DataConfig:
    mode: str = "recorded"
    train_dir: str = "data/train_data"
    val_dir: str = "data/valid_data"
    window_samples: int = 64
    sample_rate: float = 100.0
    highpass: bool = True
    negative_windows_per_file: int = 20
    trigger_context_samples: int = 48
    exclusion_margin: int = 80
    dt_min: float = 0.12
    dt_max: float = 0.45
    synthetic_samples: int = 8000
    val_split: float = 0.15


@dataclass
class LabelConfig:
    sigma: float = 0.04
    peak_offset: float = 0.02

    def to_label_params(self) -> LabelParams:
        return LabelParams(
            sigma=self.sigma,
            peak_offset=self.peak_offset,
        )


@dataclass
class ModelConfig:
    input_dim: int = 6
    cnn_channels: int = 32
    gru_hidden: int = 64
    gru_layers: int = 1
    kernel_size: int = 5
    dilations: tuple[int, ...] = (1, 2, 4)
    dropout: float = 0.1

    def to_model_kwargs(self) -> dict[str, Any]:
        return {
            "input_dim": self.input_dim,
            "cnn_channels": self.cnn_channels,
            "gru_hidden": self.gru_hidden,
            "gru_layers": self.gru_layers,
            "kernel_size": self.kernel_size,
            "dilations": self.dilations,
            "dropout": self.dropout,
        }


@dataclass
class EarlyStoppingConfig:
    enabled: bool = True
    patience: int = 10
    min_delta: float = 0.001
    monitor: str = "val_window_acc"


@dataclass
class TrainingConfig:
    epochs: int = 50
    batch_size: int = 16
    lr: float = 1e-3
    weight_decay: float = 1e-4
    grad_clip: float = 1.0
    prediction_threshold: float = 0.5
    num_workers: int = 0
    early_stopping: EarlyStoppingConfig = field(default_factory=EarlyStoppingConfig)


@dataclass
class TrainConfig:
    seed: int = 42
    device: str = "auto"
    out_dir: str = "checkpoints"
    data: DataConfig = field(default_factory=DataConfig)
    labels: LabelConfig = field(default_factory=LabelConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["model"]["dilations"] = list(self.model.dilations)
        return payload


def _merge_dataclass(cls, data: dict[str, Any]):
    fields = {f.name for f in cls.__dataclass_fields__.values()}  # type: ignore[attr-defined]
    kwargs = {key: value for key, value in data.items() if key in fields}
    if cls is ModelConfig and "dilations" in kwargs:
        kwargs["dilations"] = tuple(kwargs["dilations"])
    if cls is TrainingConfig and "early_stopping" in kwargs:
        es = kwargs["early_stopping"]
        if isinstance(es, dict):
            kwargs["early_stopping"] = _merge_dataclass(EarlyStoppingConfig, es)
    return cls(**kwargs)


def load_train_config(path: str | Path) -> TrainConfig:
    if yaml is None:
        raise ImportError(
            "PyYAML is required to load config files. Install with: pip install pyyaml"
        ) from _YAML_IMPORT_ERROR

    path = Path(path)
    with path.open(encoding="utf-8") as f:
        raw = yaml.safe_load(f) or {}

    return TrainConfig(
        seed=raw.get("seed", 42),
        device=raw.get("device", "auto"),
        out_dir=raw.get("out_dir", "checkpoints"),
        data=_merge_dataclass(DataConfig, raw.get("data", {})),
        labels=_merge_dataclass(LabelConfig, raw.get("labels", {})),
        model=_merge_dataclass(ModelConfig, raw.get("model", {})),
        training=_merge_dataclass(TrainingConfig, raw.get("training", {})),
    )
