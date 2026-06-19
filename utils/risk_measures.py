# Copyright 2025 University of California, Davis, and Kai Fukazawa.
# SPDX-License-Identifier: Apache-2.0
"""Risk-distortion samplers used by RAMAC-family agents."""

from __future__ import annotations

import numpy as np
import torch
from torch.distributions import Uniform
from torch.distributions.normal import Normal


class WangDistortionSampler:
    """Sample quantile levels under Wang distortion."""

    def __init__(self, eta: float = -0.7):
        self.eta = eta
        self.normal = Normal(loc=torch.tensor([0.0]), scale=torch.tensor([1.0]))

    def sample(self, num_samples):
        taus = Uniform(0.0, 1.0).sample(num_samples)
        return self.normal.cdf(self.normal.icdf(taus) + self.eta)


class CumulativeProspectWeightSampler:
    """Sample quantile levels under cumulative prospect weighting."""

    def __init__(self, eta: float = 0.71):
        self.eta = eta

    def sample(self, num_samples):
        taus = Uniform(0.0, 1.0).sample(num_samples)
        tau_eta = taus ** self.eta
        one_minus_tau_eta = (1.0 - taus) ** self.eta
        return tau_eta / ((tau_eta + one_minus_tau_eta) ** (1.0 / self.eta))


class PowerDistortionSampler:
    """Sample quantile levels under power distortion."""

    def __init__(self, eta: float = -2.0):
        self.eta = eta
        self.exponent = 1.0 / (1.0 + np.abs(eta))

    def sample(self, num_samples):
        taus = Uniform(0.0, 1.0).sample(num_samples)
        if self.eta > 0:
            return taus ** self.exponent
        return 1.0 - (1.0 - taus) ** self.exponent

