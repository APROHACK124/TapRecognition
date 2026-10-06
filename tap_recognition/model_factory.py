"""Construct baseline or opt-in feature LSTM models from checkpoint configuration."""

from __future__ import annotations

from typing import Any

from .model import CausalCNNGRU
from .model_lstm import CausalCNNLSTM
from .model_feature_lstm import FeatureCNNLSTM


def build_model(
    model_config: dict[str, Any], model_type: str | None = None
) -> CausalCNNGRU | CausalCNNLSTM:
    """Keep legacy GRU checkpoints usable without architecture metadata."""
    config = dict(model_config)
    if "dilations" in config:
        config["dilations"] = tuple(config["dilations"])
    if model_type is None:
        if "feature_config" in config:
            model_type = "feature_lstm"
        else:
            model_type = "lstm" if "lstm_hidden" in config or "lstm_layers" in config else "gru"
    if model_type == "feature_lstm":
        return FeatureCNNLSTM(**config)
    if model_type == "gru":
        return CausalCNNGRU(**config)
    if model_type == "lstm":
        return CausalCNNLSTM(**config)
    raise ValueError(f"Unsupported model_type: {model_type!r}")
