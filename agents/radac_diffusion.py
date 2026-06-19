# Copyright 2025 University of California, Davis, and Kai Fukazawa.
# SPDX-License-Identifier: Apache-2.0
"""RADAC-owned diffusion actor implementation."""

from __future__ import annotations

import math
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from agents.common import SinusoidalTimeEmbedding


def _extract(buffer: torch.Tensor, timesteps: torch.Tensor, target_shape: torch.Size) -> torch.Tensor:
    batch_size = timesteps.shape[0]
    gathered = buffer.gather(0, timesteps)
    return gathered.reshape(batch_size, *((1,) * (len(target_shape) - 1)))


def _linear_beta_schedule(num_steps: int, beta_start: float = 1e-4, beta_end: float = 2e-2) -> torch.Tensor:
    return torch.linspace(beta_start, beta_end, num_steps, dtype=torch.float32)


def _cosine_beta_schedule(num_steps: int, s: float = 0.008) -> torch.Tensor:
    steps = num_steps + 1
    x = torch.linspace(0, num_steps, steps, dtype=torch.float64)
    alphas_cumprod = torch.cos(((x / num_steps) + s) / (1 + s) * math.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1.0 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return betas.clamp(0.0, 0.999).to(dtype=torch.float32)


def _variance_preserving_beta_schedule(num_steps: int) -> torch.Tensor:
    t = torch.arange(1, num_steps + 1, dtype=torch.float64)
    b_min = 0.1
    b_max = 10.0
    alpha = torch.exp(-b_min / num_steps - 0.5 * (b_max - b_min) * (2 * t - 1) / (num_steps ** 2))
    betas = 1.0 - alpha
    return betas.to(dtype=torch.float32)


class _WeightedL1Loss(nn.Module):
    def forward(self, pred: torch.Tensor, target: torch.Tensor, weights: float | torch.Tensor = 1.0) -> torch.Tensor:
        return (torch.abs(pred - target) * weights).mean()


class _WeightedL2Loss(nn.Module):
    def forward(self, pred: torch.Tensor, target: torch.Tensor, weights: float | torch.Tensor = 1.0) -> torch.Tensor:
        return (F.mse_loss(pred, target, reduction='none') * weights).mean()


_LOSS_BY_NAME = {
    'l1': _WeightedL1Loss,
    'l2': _WeightedL2Loss,
}


class RADACDenoiser(nn.Module):
    """State-conditioned denoiser used by the RADAC diffusion actor."""

    def __init__(self, state_dim: int, action_dim: int, hidden_dim: int = 256, t_dim: int = 16):
        super().__init__()
        self.time_encoder = nn.Sequential(
            SinusoidalTimeEmbedding(t_dim),
            nn.Linear(t_dim, t_dim * 2),
            nn.Mish(),
            nn.Linear(t_dim * 2, t_dim),
        )
        input_dim = state_dim + action_dim + t_dim
        self.backbone = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.Mish(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Mish(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Mish(),
        )
        self.output_layer = nn.Linear(hidden_dim, action_dim)

    def forward(self, noisy_action: torch.Tensor, timestep: torch.Tensor, state: torch.Tensor) -> torch.Tensor:
        time_features = self.time_encoder(timestep)
        features = torch.cat((noisy_action, time_features, state), dim=1)
        return self.output_layer(self.backbone(features))


class RADACDiffusionActor(nn.Module):
    """Diffusion actor specialized for RADAC."""

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        denoiser: nn.Module,
        max_action: float,
        beta_schedule: str = 'linear',
        n_timesteps: int = 100,
        loss_type: str = 'l2',
        clip_denoised: bool = True,
        predict_epsilon: bool = True,
    ):
        super().__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.max_action = max_action
        self.denoiser = denoiser
        self.n_timesteps = int(n_timesteps)
        self.clip_denoised = clip_denoised
        self.predict_epsilon = predict_epsilon

        if beta_schedule == 'linear':
            betas = _linear_beta_schedule(self.n_timesteps)
        elif beta_schedule == 'cosine':
            betas = _cosine_beta_schedule(self.n_timesteps)
        elif beta_schedule == 'vp':
            betas = _variance_preserving_beta_schedule(self.n_timesteps)
        else:
            raise ValueError(f'unsupported beta schedule: {beta_schedule}')

        alphas = 1.0 - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)
        alphas_cumprod_prev = torch.cat((torch.ones(1, dtype=alphas.dtype), alphas_cumprod[:-1]), dim=0)
        posterior_variance = betas * (1.0 - alphas_cumprod_prev) / (1.0 - alphas_cumprod)

        self.register_buffer('betas', betas)
        self.register_buffer('alphas_cumprod', alphas_cumprod)
        self.register_buffer('alphas_cumprod_prev', alphas_cumprod_prev)
        self.register_buffer('sqrt_alphas_cumprod', torch.sqrt(alphas_cumprod))
        self.register_buffer('sqrt_one_minus_alphas_cumprod', torch.sqrt(1.0 - alphas_cumprod))
        self.register_buffer('sqrt_recip_alphas_cumprod', torch.sqrt(1.0 / alphas_cumprod))
        self.register_buffer('sqrt_recipm1_alphas_cumprod', torch.sqrt(1.0 / alphas_cumprod - 1.0))
        self.register_buffer('posterior_variance', posterior_variance)
        self.register_buffer('posterior_log_variance_clipped', torch.log(torch.clamp(posterior_variance, min=1e-20)))
        self.register_buffer('posterior_mean_coef1', betas * torch.sqrt(alphas_cumprod_prev) / (1.0 - alphas_cumprod))
        self.register_buffer('posterior_mean_coef2', (1.0 - alphas_cumprod_prev) * torch.sqrt(alphas) / (1.0 - alphas_cumprod))

        try:
            self.loss_fn = _LOSS_BY_NAME[loss_type]()
        except KeyError as exc:
            raise ValueError(f'unsupported loss type: {loss_type}') from exc

    def predict_start_from_noise(self, noisy_action: torch.Tensor, timesteps: torch.Tensor, predicted_noise: torch.Tensor) -> torch.Tensor:
        if not self.predict_epsilon:
            return predicted_noise
        return (
            _extract(self.sqrt_recip_alphas_cumprod, timesteps, noisy_action.shape) * noisy_action
            - _extract(self.sqrt_recipm1_alphas_cumprod, timesteps, noisy_action.shape) * predicted_noise
        )

    def q_posterior(self, x_start: torch.Tensor, x_t: torch.Tensor, timesteps: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        posterior_mean = (
            _extract(self.posterior_mean_coef1, timesteps, x_t.shape) * x_start
            + _extract(self.posterior_mean_coef2, timesteps, x_t.shape) * x_t
        )
        posterior_variance = _extract(self.posterior_variance, timesteps, x_t.shape)
        posterior_log_variance = _extract(self.posterior_log_variance_clipped, timesteps, x_t.shape)
        return posterior_mean, posterior_variance, posterior_log_variance

    def p_mean_variance(self, noisy_action: torch.Tensor, timesteps: torch.Tensor, state: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        predicted_noise = self.denoiser(noisy_action, timesteps, state)
        x_start = self.predict_start_from_noise(noisy_action, timesteps, predicted_noise)
        if self.clip_denoised:
            x_start = x_start.clamp(-self.max_action, self.max_action)
        return self.q_posterior(x_start=x_start, x_t=noisy_action, timesteps=timesteps)

    def p_sample(self, noisy_action: torch.Tensor, timesteps: torch.Tensor, state: torch.Tensor) -> torch.Tensor:
        batch_size = noisy_action.shape[0]
        model_mean, _, model_log_variance = self.p_mean_variance(noisy_action, timesteps, state)
        noise = torch.randn_like(noisy_action)
        nonzero_mask = (timesteps != 0).float().reshape(batch_size, *((1,) * (noisy_action.dim() - 1)))
        return model_mean + nonzero_mask * torch.exp(0.5 * model_log_variance) * noise

    def p_sample_loop(
        self,
        state: torch.Tensor,
        shape: Tuple[int, int],
        verbose: bool = False,
        return_diffusion: bool = False,
    ) -> torch.Tensor | Tuple[torch.Tensor, torch.Tensor]:
        _ = verbose
        sample = torch.randn(shape, device=self.betas.device)
        diffusion_trace = [sample] if return_diffusion else None
        for step in reversed(range(self.n_timesteps)):
            timesteps = torch.full((shape[0],), step, device=self.betas.device, dtype=torch.long)
            sample = self.p_sample(sample, timesteps, state)
            if return_diffusion:
                diffusion_trace.append(sample)
        if not return_diffusion:
            return sample
        return sample, torch.stack(diffusion_trace, dim=1)

    def sample(self, state: torch.Tensor, *args, **kwargs) -> torch.Tensor:
        shape = (state.shape[0], self.action_dim)
        action = self.p_sample_loop(state, shape, *args, **kwargs)
        if isinstance(action, tuple):
            final_action, diffusion_trace = action
            return final_action.clamp(-self.max_action, self.max_action), diffusion_trace
        return action.clamp(-self.max_action, self.max_action)

    def q_sample(self, action: torch.Tensor, timesteps: torch.Tensor, noise: Optional[torch.Tensor] = None) -> torch.Tensor:
        if noise is None:
            noise = torch.randn_like(action)
        return (
            _extract(self.sqrt_alphas_cumprod, timesteps, action.shape) * action
            + _extract(self.sqrt_one_minus_alphas_cumprod, timesteps, action.shape) * noise
        )

    def p_losses(self, action: torch.Tensor, state: torch.Tensor, timesteps: torch.Tensor, weights: float | torch.Tensor = 1.0) -> torch.Tensor:
        noise = torch.randn_like(action)
        noisy_action = self.q_sample(action, timesteps, noise=noise)
        predicted_noise = self.denoiser(noisy_action, timesteps, state)
        target = noise if self.predict_epsilon else action
        return self.loss_fn(predicted_noise, target, weights)

    def loss(self, action: torch.Tensor, state: torch.Tensor, weights: float | torch.Tensor = 1.0) -> torch.Tensor:
        batch_size = action.shape[0]
        timesteps = torch.randint(0, self.n_timesteps, (batch_size,), device=action.device, dtype=torch.long)
        return self.p_losses(action, state, timesteps, weights=weights)

    def forward(self, state: torch.Tensor, *args, **kwargs) -> torch.Tensor:
        return self.sample(state, *args, **kwargs)
