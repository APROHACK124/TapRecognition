"""Experimental engineered-feature LSTM, isolated from the raw-IMU baseline."""

from __future__ import annotations

import torch

from .imu_features import FeatureProjection, IMUFeatureConfig
from .model_lstm import CausalCNNLSTM, LSTMState


class FeatureCNNLSTM(CausalCNNLSTM):
    """Accept six raw/high-passed channels through the baseline's public interface.

    Feature history is included in receptive_field so the independent overlap
    streaming reference used by evaluate.ipynb remains valid. Single-sample step
    carries raw feature history separately from the per-block CNN histories.
    """

    def __init__(self, feature_config: dict | None = None, **model_kwargs):
        if model_kwargs.get("input_dim", 6) != 6:
            raise ValueError("FeatureCNNLSTM takes six IMU channels, not precomputed features")
        super().__init__(**model_kwargs)
        self.feature_config = IMUFeatureConfig(**(feature_config or {}))
        self.input_proj = FeatureProjection(self.feature_config, self.input_proj.out_features)

    @property
    def receptive_field(self):
        return self._receptive_field + self.feature_config.history_samples

    def step(self, x_t: torch.Tensor, state: LSTMState | None, cnn_buffer: dict | None):
        """Return probabilities, LSTM state, and a {raw, cnn} history buffer.

        Pass None for both state and buffer to reset at a new recording. There is
        no mutable cross-call state in the model, so independent streams stay isolated.
        """
        if x_t.ndim == 2:
            x_t = x_t.unsqueeze(1)
        if x_t.ndim != 3 or x_t.shape[1:] != (1, 6):
            raise ValueError("step() expects [batch, 6] or [batch, 1, 6]")
        if self.training:
            raise ValueError("step() requires eval mode")
        raw = x_t if cnn_buffer is None else torch.cat((cnn_buffer["raw"], x_t), dim=1)
        current = self.input_proj(raw)[:, -1:].transpose(1, 2)
        buffer = (current.new_zeros(current.shape[0], current.shape[1], self._receptive_field)
                  if cnn_buffer is None else cnn_buffer["cnn"])
        prob, next_state, next_cnn = self._step_projected(current, state, buffer)
        keep = self.feature_config.history_samples
        next_buffer = {
            "raw": raw[:, -keep:] if keep else raw[:, :0],
            "cnn": next_cnn,
        }
        return prob, next_state, next_buffer

    def step_fixed(self, x_t, state, cnn_buffer, feature_history, feature_ready):
        """Tensor-only export contract with frozen feature normalization.

        Unlike receptive_field (43 for defaults), cnn_buffer has only the
        CNN history (29). Feature history (14) is carried separately.
        """
        features, history, ready = self.input_proj.features.step(x_t, feature_history, feature_ready)
        projected = self.input_proj.linear((features - self.input_proj.mean) / self.input_proj.scale)
        prob, state, buffer = self._step_projected(projected.transpose(1, 2), state, cnn_buffer)
        return prob, state, buffer, history, ready

    def _step_projected(self, current, state, buffer):
        offset, histories = 0, []
        for block in self.cnn:
            n_past = block.conv.left_pad
            if n_past:
                past = buffer[:, :, offset:offset + n_past]
                histories.append(torch.cat((past[:, :, 1:], current), dim=2))
                current = block(torch.cat((past, current), dim=2))[:, :, -1:]
            else:
                current = block(current)
            offset += n_past
        next_buffer = torch.cat((*histories, buffer[:, :, -1:]), dim=2)
        features = self.dropout(current.transpose(1, 2))
        output, next_state = self._lstm_one_step(features[:, 0], state)
        return self.head(output).softmax(-1), next_state, next_buffer
