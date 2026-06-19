# Copyright 2025 University of California, Davis, and Kai Fukazawa.
# SPDX-License-Identifier: Apache-2.0
"""Offline dataset utilities owned by the RAMAC codebase."""

from __future__ import annotations

import torch


class OfflineReplayDataset:
    """Tensor-backed sampler for static offline RL datasets."""

    def __init__(self, data: dict, device):
        self.state = torch.from_numpy(data['observations']).float()
        self.action = torch.from_numpy(data['actions']).float()
        self.next_state = torch.from_numpy(data['next_observations']).float()
        self.reward = torch.from_numpy(data['rewards']).view(-1, 1).float()
        self.not_done = 1.0 - torch.from_numpy(data['terminals']).view(-1, 1).float()

        self.size = self.state.shape[0]
        self.state_dim = self.state.shape[1]
        self.action_dim = self.action.shape[1]
        self.device = device

    def sample(self, batch_size: int):
        indices = torch.randint(0, self.size, (batch_size,))
        return (
            self.state[indices].to(self.device),
            self.action[indices].to(self.device),
            self.next_state[indices].to(self.device),
            self.reward[indices].to(self.device),
            self.not_done[indices].to(self.device),
        )
