# Copyright 2025 University of California, Davis, and Kai Fukazawa.
# SPDX-License-Identifier: Apache-2.0
"""Implicit-quantile critic networks owned by the RAMAC codebase."""

from __future__ import annotations

import math
from typing import Iterable

import torch
import torch.nn as nn


class ImplicitQuantileCritic(nn.Module):
    """Continuous-action IQN critic used by RADAC and RAFMAC."""

    def __init__(
        self,
        dim_state: int,
        dim_action: int,
        layers_state=None,
        layers_action=None,
        layers_f=None,
        embedding_dim: int | None = None,
        tau_embed_dim: int = 1,
        biased_head: bool = True,
        non_linearity: str = 'ReLU',
        tau: float = 1.0,
    ):
        super().__init__()
        self.dim_state = dim_state
        self.dim_action = dim_action
        self.embedding_dim = embedding_dim
        self.tau_embed_dim = tau_embed_dim
        self.tau = tau

        self.fc_state, state_out_dim = _build_mlp(layers_state, dim_state, non_linearity, normalized=True)
        self.fc_action, action_out_dim = _build_mlp(layers_action, dim_action, non_linearity, normalized=True)
        self.fc_state_action, _ = _build_mlp(embedding_dim, state_out_dim + action_out_dim, non_linearity, normalized=True)

        if tau_embed_dim > 1:
            self.register_buffer('tau_basis', torch.arange(tau_embed_dim, dtype=torch.float32))
        else:
            self.tau_basis = None

        self.head_tau, _ = _build_mlp(embedding_dim, tau_embed_dim, non_linearity, normalized=True)
        self.hidden_layers_f, fused_dim = _build_mlp(layers_f, embedding_dim, non_linearity, normalized=True)
        self.head = nn.Linear(fused_dim, 1, bias=biased_head)
        self.output_layer = nn.Sequential(self.hidden_layers_f, self.head)

        for name, param in self.named_parameters():
            if name.endswith('head.weight'):
                torch.nn.init.uniform_(param, -3e-4, 3e-4)
            if name.endswith('head.bias'):
                torch.nn.init.zeros_(param)

    def forward(self, state: torch.Tensor, tau_quantile: torch.Tensor, action: torch.Tensor | None = None) -> torch.Tensor:
        state_features = self.fc_state(state)
        action_features = self.fc_action(action)
        fused_features = self.fc_state_action(torch.cat((state_features, action_features), dim=-1))

        if self.tau_embed_dim > 1:
            tau_input = torch.cos(math.pi * self.tau_basis.to(tau_quantile.device) * tau_quantile)
        else:
            tau_input = tau_quantile
        tau_features = self.head_tau(tau_input)
        return self.output_layer(fused_features * tau_features).view(-1, 1)

    def get_sampled_Z(self, state: torch.Tensor, confidences: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        num_quantiles = confidences.size(0)
        batch_size = state.size(0) if state.dim() > 1 else 1
        repeated_state = state.repeat(1, num_quantiles).view(-1, self.dim_state)
        repeated_action = action.repeat(1, num_quantiles).view(-1, self.dim_action)
        repeated_tau = confidences.repeat(batch_size, 1).view(num_quantiles * batch_size, 1)
        return self(state=repeated_state, tau_quantile=repeated_tau, action=repeated_action).view(batch_size, num_quantiles)

    @property
    def params(self):
        return self.parameters()

    @params.setter
    def params(self, new_params: Iterable[torch.nn.Parameter]):
        for target_param, new_param in zip(self.params, new_params):
            if target_param is new_param:
                continue
            updated = ((1.0 - self.tau) * target_param.data.detach() + self.tau * new_param.data.detach())
            target_param.data.copy_(updated)



def _resolve_nonlinearity(name: str):
    if hasattr(nn, name):
        return getattr(nn, name)
    if hasattr(nn, name.capitalize()):
        return getattr(nn, name.capitalize())
    if hasattr(nn, name.upper()):
        return getattr(nn, name.upper())
    raise NotImplementedError(f'non-linearity {name} not implemented')


def _build_mlp(layers, in_dim: int, non_linearity: str, normalized: bool = False):
    if layers is None:
        layers = []
    elif isinstance(layers, int):
        layers = [layers]

    nonlinearity = _resolve_nonlinearity(non_linearity)
    modules = []
    for width in layers:
        linear = nn.Linear(in_dim, width)
        if normalized:
            linear.weight.data.normal_(0, 0.1)
        modules.append(linear)
        modules.append(nonlinearity())
        in_dim = width
    return nn.Sequential(*modules), in_dim
