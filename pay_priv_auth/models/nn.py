"""Simple neural network models."""

from __future__ import annotations

import torch
from torch import nn


class MLP(nn.Module):
    def __init__(self, in_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, 64),
            nn.GELU(),
            nn.LayerNorm(64),
            nn.Linear(64, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)


class TCNBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, kernel_size: int = 3):
        super().__init__()
        pad = kernel_size // 2
        self.conv = nn.Conv1d(in_ch, out_ch, kernel_size, padding=pad)
        self.norm = nn.LayerNorm(out_ch)
        self.act = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.conv(x)
        y = self.norm(y.transpose(1, 2)).transpose(1, 2)
        return self.act(y)
