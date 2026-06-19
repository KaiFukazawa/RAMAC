# Copyright 2025 University of California, Davis, and Kai Fukazawa.
# SPDX-License-Identifier: Apache-2.0
"""Shared actor utilities owned by the RAMAC codebase."""

from __future__ import annotations

import math

import torch
import torch.nn as nn


class SinusoidalTimeEmbedding(nn.Module):
    """Map scalar timesteps to smooth sinusoidal features."""

    def __init__(self, dim: int):
        super().__init__()
        if dim < 2:
            raise ValueError('time embedding dimension must be at least 2')
        self.dim = dim

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        if t.dim() == 1:
            t = t.unsqueeze(-1)
        t = t.float()
        half_dim = self.dim // 2
        exponent = math.log(10000.0) / max(half_dim - 1, 1)
        freqs = torch.exp(-exponent * torch.arange(half_dim, device=t.device, dtype=t.dtype))
        angles = t * freqs.unsqueeze(0)
        embedding = torch.cat((angles.sin(), angles.cos()), dim=-1)
        if embedding.shape[-1] < self.dim:
            padding = torch.zeros(*embedding.shape[:-1], self.dim - embedding.shape[-1], device=t.device, dtype=t.dtype)
            embedding = torch.cat((embedding, padding), dim=-1)
        return embedding


class ExponentialMovingAverage:
    """Utility for maintaining an exponential moving average of model weights."""

    def __init__(self, beta: float):
        self.beta = beta

    def update_average(self, old: torch.Tensor, new: torch.Tensor) -> torch.Tensor:
        if old is None:
            return new
        return old * self.beta + (1.0 - self.beta) * new

    def update_model_average(self, averaged_model: nn.Module, current_model: nn.Module) -> None:
        for current_params, averaged_params in zip(current_model.parameters(), averaged_model.parameters()):
            averaged_params.data = self.update_average(averaged_params.data, current_params.data)
