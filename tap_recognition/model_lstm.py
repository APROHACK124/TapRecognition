"""Causal CNN + LSTM model for streaming double-tap detection."""

from __future__ import annotations

import torch
import torch.nn as nn

from .model import CausalConvBlock


LSTMState = tuple[torch.Tensor, torch.Tensor]


class CausalCNNLSTM(nn.Module):
    """The GRU baseline's CNN and head with a unidirectional LSTM backbone.

    Input: [B, T, F], with F=6 IMU channels.
    Output: [B, T, C] logits and (hidden, cell), each state [L, B, H].
    """

    def __init__(
        self,
        input_dim: int = 6,
        num_classes: int = 3,
        cnn_channels: int = 32,
        lstm_hidden: int = 64,
        lstm_layers: int = 1,
        kernel_size: int = 5,
        dilations: tuple[int, ...] = (1, 2, 4),
        dropout: float = 0.1,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.num_classes = num_classes
        self.lstm_hidden = lstm_hidden
        self.lstm_layers = lstm_layers

        self.input_proj = nn.Linear(input_dim, cnn_channels)
        self.cnn = nn.Sequential(*[
            CausalConvBlock(cnn_channels, kernel_size, d) for d in dilations
        ])
        self.dropout = nn.Dropout(dropout)
        self.lstm = nn.LSTM(
            input_size=cnn_channels,
            hidden_size=lstm_hidden,
            num_layers=lstm_layers,
            bias=True,
            batch_first=True,
            dropout=0.0,
            bidirectional=False,
        )
        # As in the GRU baseline, dropout is outside the recurrent layer.
        self.head = nn.Sequential(
            nn.Linear(lstm_hidden, lstm_hidden // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(lstm_hidden // 2, num_classes),
        )
        self._receptive_field = 1 + sum(
            (kernel_size - 1) * d for d in dilations
        )

    @property
    def receptive_field(self) -> int:
        return self._receptive_field

    def forward(
        self,
        x: torch.Tensor,
        state: LSTMState | None = None,
    ) -> tuple[torch.Tensor, LSTMState]:
        """Return sequence logits and the final hidden/cell state pair.

        Omitting state resets both recurrent states, as in baseline training.
        Carrying state between forward calls does not carry CNN context; use
        step() for continuous sample-by-sample inference in eval mode.
        """
        cnn_in = self.input_proj(x).transpose(1, 2)
        cnn_out = self.dropout(self.cnn(cnn_in).transpose(1, 2))
        output, next_state = self.lstm(cnn_out, state)
        return self.head(output), next_state

    def step(
        self,
        x_t: torch.Tensor,
        state: LSTMState | None,
        cnn_buffer: torch.Tensor | None,
    ) -> tuple[torch.Tensor, LSTMState, torch.Tensor]:
        """Process one IMU sample in eval mode, carrying hidden AND cell state.

        x_t is [B, F] or [B, 1, F]. The CNN buffer has the same packed
        per-block history layout as CausalCNNGRU.step(): [B, C_cnn, R].
        Returns probabilities [B, 1, C], (hidden, cell), and the new buffer.
        """
        if x_t.dim() == 2:
            x_t = x_t.unsqueeze(1)
        elif x_t.dim() != 3 or x_t.shape[1] != 1:
            raise ValueError(
                f"step() expects x_t shape [B, 1, F] or [B, F], got {tuple(x_t.shape)}"
            )

        current = self.input_proj(x_t).transpose(1, 2)
        if cnn_buffer is None:
            cnn_buffer = current.new_zeros(
                current.shape[0], current.shape[1], self.receptive_field
            )

        # Each convolution needs its own input history to match left padding.
        offset = 0
        histories = []
        for block in self.cnn:
            n_past = block.conv.left_pad
            if n_past:
                past = cnn_buffer[:, :, offset : offset + n_past]
                histories.append(torch.cat([past[:, :, 1:], current], dim=2))
                current = block(torch.cat([past, current], dim=2))[:, :, -1:]
            else:
                current = block(current)
            offset += n_past

        buffer_new = torch.cat([*histories, cnn_buffer[:, :, -1:]], dim=2)
        features = self.dropout(current.transpose(1, 2))
        output, next_state = self.lstm(features, state)
        return self.head(output).softmax(dim=-1), next_state, buffer_new
