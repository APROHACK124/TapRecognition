"""Causal CNN + GRU model for streaming double-tap detection."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class CausalConv1d(nn.Module):
    """
    1D convolution with strict causality: output at time t
    depends only on inputs at times <= t.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        dilation: int = 1,
    ):
        super().__init__()
        self.kernel_size = kernel_size
        self.dilation = dilation
        self.conv = nn.Conv1d(
            in_channels,
            out_channels,
            kernel_size,
            dilation=dilation,
        )
        self.left_pad = (kernel_size - 1) * dilation

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, C, T]
        x = F.pad(x, (self.left_pad, 0))
        return self.conv(x)


class CausalConvBlock(nn.Module):
    def __init__(self, channels: int, kernel_size: int, dilation: int):
        super().__init__()
        self.conv = CausalConv1d(channels, channels, kernel_size, dilation)
        self.norm = nn.BatchNorm1d(channels)
        self.act = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(self.norm(self.conv(x)))


class CausalCNNGRU(nn.Module):
    """
    Causal CNN front-end + GRU for IMU double-tap recognition.

    Input:  [B, T, F]  F=6 IMU channels
    Output: [B, T, C]  per-frame logits, C=3 (none, left, right)
    """

    def __init__(
        self,
        input_dim: int = 6,
        num_classes: int = 3,
        cnn_channels: int = 32,
        gru_hidden: int = 64,
        gru_layers: int = 1,
        kernel_size: int = 5,
        dilations: tuple[int, ...] = (1, 2, 4),
        dropout: float = 0.1,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.num_classes = num_classes
        self.gru_hidden = gru_hidden
        self.gru_layers = gru_layers

        self.input_proj = nn.Linear(input_dim, cnn_channels)

        blocks = []
        for d in dilations:
            blocks.append(CausalConvBlock(cnn_channels, kernel_size, d))
        self.cnn = nn.Sequential(*blocks)

        self.dropout = nn.Dropout(dropout)
        self.gru = nn.GRU(
            cnn_channels,
            gru_hidden,
            num_layers=gru_layers,
            batch_first=True,
        )
        self.head = nn.Sequential(
            nn.Linear(gru_hidden, gru_hidden // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(gru_hidden // 2, num_classes),
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
        h0: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x:  [B, T, F]
            h0: [L, B, H] optional initial GRU state

        Returns:
            logits: [B, T, C]
            h_n:    [L, B, H] final GRU state
        """
        b, t, _ = x.shape
        cnn_in = self.input_proj(x)  # [B, T, C]
        cnn_in = cnn_in.transpose(1, 2)  # [B, C, T]
        cnn_out = self.cnn(cnn_in).transpose(1, 2)  # [B, T, C]
        cnn_out = self.dropout(cnn_out)

        gru_out, h_n = self.gru(cnn_out, h0)
        logits = self.head(gru_out)
        return logits, h_n

    def step(
        self,
        x_t: torch.Tensor,
        h: torch.Tensor | None,
        cnn_buffer: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Process a single time step for online inference.

        Args:
            x_t: [B, 1, F] or [B, F] — one IMU sample
            h:   [L, B, H] GRU state
            cnn_buffer: [B, C, R] past projected samples for causal conv

        Returns:
            prob: [B, 1, C] softmax P(none, left, right)
            h_new: [L, B, H]
            buffer_new: [B, C_cnn, R]
        """
        if x_t.dim() == 2:
            x_t = x_t.unsqueeze(1)
        elif x_t.dim() != 3 or x_t.shape[1] != 1:
            raise ValueError(
                f"step() expects x_t shape [B, 1, F] or [B, F], got {tuple(x_t.shape)}"
            )

        b = x_t.shape[0]
        proj = self.input_proj(x_t[:, 0, :]).unsqueeze(1)  # [B, 1, C]
        c = proj.shape[-1]
        r = self._receptive_field

        if cnn_buffer is None:
            cnn_buffer = torch.zeros(b, c, r, device=x_t.device, dtype=x_t.dtype)

        buffer_new = torch.cat([cnn_buffer[:, :, 1:], proj.transpose(1, 2)], dim=2)
        cnn_out = self.cnn(buffer_new).transpose(1, 2)  # [B, 1, C]
        gru_out, h_new = self.gru(cnn_out, h)
        logit = self.head(gru_out[:, -1:, :])  # [B, 1, num_classes]
        prob = torch.softmax(logit, dim=-1)
        return prob, h_new, buffer_new
