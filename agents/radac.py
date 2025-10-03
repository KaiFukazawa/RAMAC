# Copyright 2025 University of California, Davis, and Kai Fukazawa.
# SPDX-License-Identifier: Apache-2.0
import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim.lr_scheduler import CosineAnnealingLR
from utils.neural_networks import DeterministicNN_IQN
from agents.diffusion import Diffusion
from agents.model import MLP
from utils.data_sampler import Data_Sampler
from utils.sampling import Wang_distortion, CPW, Power
from torch.distributions import uniform
from agents.helpers import EMA

class RADAC:
    """
    RADAC with Double Distributional Critic + min-target (critic side)
    and CVaR-based actor update (risk-averse)

    * Critic: Double distributional (IQN), use min-target to reduce overestimation
    * Actor: if risk_distortion='cvar', update with a CVaR-based loss
      (focus on risk aversion over extrapolation; no ratio trick)
    * Additional techniques:
      - Clip critic target values (self.q_clip_range)
      - Use min of critic1/critic2 for CVaR in actor update
      - BC loss coefficient (self.lambda_bc)
      - Warmup period with self.eta=0 early in training (self.eta_warmup_steps)
    """
    def __init__(self,
                 state_dim,
                 action_dim,
                 max_action,
                 device,
                 discount,
                 tau,
                 n_quantiles,
                 embedding_dim,
                 beta_schedule='linear',
                 n_timesteps=100,
                 lr_actor=3e-4,
                 lr_critic=3e-4,
                 risk_distortion='cvar',
                 alpha_cvar=0.1,
                 ema_decay=0.995,
                 lr_maxt=1000,
                 grad_norm=1.0,
                 eta=1.0,
                 lr_decay=False,
                 step_start_ema=1000,
                 update_ema_every=5,
                 q_clip_range=None,
                 lambda_bc=1.0, 
                 eta_warmup_steps=0,
                 eta_ramp_steps=1e5
                 ):
        self.device = device
        self.discount = discount
        self.tau = tau
        self.n_quantiles = n_quantiles
        self.risk_distortion = risk_distortion
        self.alpha_cvar = alpha_cvar  # Used when risk_distortion='cvar'
        self.grad_norm = grad_norm
        self.eta = eta
        self.step = 0
        self.lr_decay = lr_decay
        self.q_clip_range = q_clip_range
        self.lambda_bc = lambda_bc
        self.eta_warmup_steps = eta_warmup_steps
        self.eta_ramp_steps = eta_ramp_steps

        # ----------------------------
        # 1. Actor (Diffusion) Setup
        # ----------------------------
        self.policy_model = MLP(
            state_dim=state_dim,
            action_dim=action_dim,
            device=device
        )
        self.actor = Diffusion(
            state_dim=state_dim,
            action_dim=action_dim,
            model=self.policy_model,
            max_action=max_action,
            beta_schedule=beta_schedule,
            n_timesteps=n_timesteps
        ).to(device)

        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=lr_actor)
        if lr_decay:
            self.actor_lr_scheduler = CosineAnnealingLR(
                self.actor_optimizer, T_max=lr_maxt, eta_min=0.)

        # EMA for actor
        self.ema = EMA(ema_decay)
        self.ema_model = copy.deepcopy(self.actor)
        self.step_start_ema = step_start_ema
        self.update_ema_every = update_ema_every

        # ----------------------------
        # 2. Double Critic (IQN) and Target Critic
        # ----------------------------
        self.critic1 = DeterministicNN_IQN(
            dim_state=state_dim,
            dim_action=action_dim,
            layers_state=[256],
            layers_action=[256],
            layers_f=[256],
            embedding_dim=embedding_dim,
            tau_embed_dim=n_quantiles
        ).to(device)
        self.critic2 = DeterministicNN_IQN(
            dim_state=state_dim,
            dim_action=action_dim,
            layers_state=[256],
            layers_action=[256],
            layers_f=[256],
            embedding_dim=embedding_dim,
            tau_embed_dim=n_quantiles
        ).to(device)

        self.critic1_target = copy.deepcopy(self.critic1)
        self.critic2_target = copy.deepcopy(self.critic2)

        self.critic1_optimizer = torch.optim.Adam(self.critic1.parameters(), lr=lr_critic)
        self.critic2_optimizer = torch.optim.Adam(self.critic2.parameters(), lr=lr_critic)

        if lr_decay:
            self.critic1_lr_scheduler = CosineAnnealingLR(self.critic1_optimizer, T_max=lr_maxt, eta_min=0.)
            self.critic2_lr_scheduler = CosineAnnealingLR(self.critic2_optimizer, T_max=lr_maxt, eta_min=0.)

        # Tau distribution (Uniform) for both critic and actor
        self.distr_taus_uniform = uniform.Uniform(0., 1.)

        # Tau distribution for CVaR computation
        if risk_distortion == 'cvar':
            self.alpha_cvar = alpha_cvar
            self.distr_taus_risk = uniform.Uniform(0., self.alpha_cvar)
        elif risk_distortion == 'wang':
            self.distr_taus_risk = Wang_distortion()
        elif risk_distortion == 'cpw':
            self.distr_taus_risk = CPW()
        elif risk_distortion == 'power':
            self.distr_taus_risk = Power()
        else:
            raise ValueError("Invalid risk distortion function specified.")
        self.max_action = max_action
        self.action_dim = action_dim

    def train(self, replay_buffer: Data_Sampler, iterations, batch_size=100, log_writer=None):
        """Run training updates for critic and actor and log metrics.

        Args:
            replay_buffer (ExperienceReplay): Replay buffer with sample(batch_size).
            iterations (int): Number of gradient update iterations.
            batch_size (int): Batch size for sampling.
            log_writer: Optional TensorBoard SummaryWriter.

        Returns:
            dict: Metric lists for losses and statistics.
        """
        metric = {
            'bc_loss': [],
            'actor_loss': [],
            'critic_loss': [],
            'cvar_val': [],
            'Q_mean': []
        }

        for _ in range(iterations):
            # ================
            # 1. Sample batch
            # ================
            state, action, next_state, reward, not_done = replay_buffer.sample(batch_size)

            # ================
            # 2. Critic update (Double Critic)
            # ================
            tau_k  = self.distr_taus_uniform.sample((self.n_quantiles, 1)).to(self.device)
            tau_k_ = self.distr_taus_uniform.sample((self.n_quantiles, 1)).to(self.device)

            current_q1 = self.critic1.get_sampled_Z(state, tau_k, action)
            current_q2 = self.critic2.get_sampled_Z(state, tau_k, action)

            with torch.no_grad():
                # Sample next_action from EMA actor
                next_action = self.ema_model.sample(next_state)

                target_q1 = self.critic1_target.get_sampled_Z(next_state, tau_k_, next_action)
                target_q2 = self.critic2_target.get_sampled_Z(next_state, tau_k_, next_action)

                reward = reward.view(batch_size, 1).expand(batch_size, self.n_quantiles)
                not_done = not_done.view(batch_size, 1).expand(batch_size, self.n_quantiles)

                # Use min-target to reduce critic overestimation
                target_min = torch.min(target_q1, target_q2)
                target_min = reward + not_done * self.discount * target_min

                # Optional clipping
                if self.q_clip_range is not None:
                    low_clip, high_clip = self.q_clip_range
                    target_min = torch.clamp(target_min, min=low_clip, max=high_clip)

            loss1 = self.quantile_huber_loss(target_min, current_q1, tau_k)
            loss2 = self.quantile_huber_loss(target_min, current_q2, tau_k)
            critic_loss = 0.5 * (loss1 + loss2)

            # Optimize Critic1
            self.critic1_optimizer.zero_grad()
            loss1.backward()
            if self.grad_norm > 0:
                nn.utils.clip_grad_norm_(self.critic1.parameters(), max_norm=self.grad_norm, norm_type=2)
            self.critic1_optimizer.step()
            if self.lr_decay:
                self.critic1_lr_scheduler.step()

            # Optimize Critic2
            self.critic2_optimizer.zero_grad()
            loss2.backward()
            if self.grad_norm > 0:
                nn.utils.clip_grad_norm_(self.critic2.parameters(), max_norm=self.grad_norm, norm_type=2)
            self.critic2_optimizer.step()
            if self.lr_decay:
                self.critic2_lr_scheduler.step()

            # ================
            # 3. Actor update (Risk Averse via CVaR)
            # ================
            sampled_action = self.actor(state)

            # BC Loss
            bc_loss = self.actor.loss(action, state)

            # Calculate CVaR from distribution (min of Double Critic)
            tau_cvar = self.distr_taus_risk.sample((self.n_quantiles, 1)).to(self.device)
            actor_q1_cvar = self.critic1.get_sampled_Z(state, tau_cvar, sampled_action)
            actor_q2_cvar = self.critic2.get_sampled_Z(state, tau_cvar, sampled_action)

            # Take min here to be more conservative in risk evaluation
            actor_q_cvar = torch.min(actor_q1_cvar, actor_q2_cvar)
            cvar_val = actor_q_cvar.mean()

            # During warmup, use eta=0
            if self.step < self.eta_warmup_steps:
                local_eta = 0.0
            else:
                denom = max(1.0, float(self.eta_ramp_steps))   # Guard against zero
                progress = min(1.0, (self.step - self.eta_warmup_steps) / denom)
                local_eta = self.eta * progress

            cvar_loss = - cvar_val  # Increase CVaR (risk aversion)

            # Apply coefficient to BC loss as well
            actor_loss = self.lambda_bc * bc_loss + local_eta * cvar_loss

            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            if self.grad_norm > 0:
                nn.utils.clip_grad_norm_(self.actor.parameters(), max_norm=self.grad_norm, norm_type=2)
            self.actor_optimizer.step()
            if self.lr_decay:
                self.actor_lr_scheduler.step()

            # ================
            # 4. Update EMA Model
            # ================
            if self.step >= self.step_start_ema and self.step % self.update_ema_every == 0:
                self.ema.update_model_average(self.ema_model, self.actor)

            # ================
            # 5. Soft Update Target Critic
            # ================
            for param, target_param in zip(self.critic1.parameters(), self.critic1_target.parameters()):
                target_param.data.copy_(
                    self.tau * param.data + (1 - self.tau) * target_param.data
                )
            for param, target_param in zip(self.critic2.parameters(), self.critic2_target.parameters()):
                target_param.data.copy_(
                    self.tau * param.data + (1 - self.tau) * target_param.data
                )

            self.step += 1

            # ================
            # 6. Logging
            # ================
            q_mean_val = 0.5 * (current_q1.mean().item() + current_q2.mean().item())

            if log_writer is not None:
                if self.grad_norm > 0:
                    actor_grad_norm = nn.utils.clip_grad_norm_(self.actor.parameters(), max_norm=self.grad_norm, norm_type=2)
                    critic1_grad_norm = nn.utils.clip_grad_norm_(self.critic1.parameters(), max_norm=self.grad_norm, norm_type=2)
                    critic2_grad_norm = nn.utils.clip_grad_norm_(self.critic2.parameters(), max_norm=self.grad_norm, norm_type=2)

                    log_writer.add_scalar('Actor Grad Norm', actor_grad_norm.max().item(), self.step)
                    log_writer.add_scalar('Critic1 Grad Norm', critic1_grad_norm.max().item(), self.step)
                    log_writer.add_scalar('Critic2 Grad Norm', critic2_grad_norm.max().item(), self.step)

                log_writer.add_scalar('BC Loss', bc_loss.item(), self.step)
                log_writer.add_scalar('Actor Loss', actor_loss.item(), self.step)
                log_writer.add_scalar('Critic Loss', critic_loss.item(), self.step)
                log_writer.add_scalar('CVaR Val', cvar_val.item(), self.step)
                log_writer.add_scalar('Q Mean', q_mean_val, self.step)

            metric['bc_loss'].append(bc_loss.item())
            metric['actor_loss'].append(actor_loss.item())
            metric['critic_loss'].append(critic_loss.item())
            metric['cvar_val'].append(cvar_val.item())
            metric['Q_mean'].append(q_mean_val)

        return metric

    def quantile_huber_loss(self, target, current, tau_k):
        """Compute quantile Huber loss used by distributional critics.

        Args:
            target (torch.Tensor): Target quantiles, shape (B, K).
            current (torch.Tensor): Predicted quantiles, shape (B, K).
            tau_k (torch.Tensor): Sampled quantile fractions, shape (K, 1).

        Returns:
            torch.Tensor: Scalar loss.
        """
        batch_size, num_quantiles = target.size()
        target_ = target.unsqueeze(2).expand(batch_size, -1, num_quantiles)
        current_ = current.unsqueeze(1).expand(batch_size, num_quantiles, -1)
        tau_ = tau_k.unsqueeze(0).expand(batch_size, num_quantiles, num_quantiles)
        huber_loss = F.smooth_l1_loss(current_, target_, reduction='none')
        quantile_loss = torch.abs(tau_ - (target_ - current_).detach().le(0).float()) * huber_loss
        return quantile_loss.sum(dim=1).mean()

    def sample_action(self, state):
        """Sample an action from the current policy given a state.

        Args:
            state (np.ndarray | torch.Tensor): Environment state.

        Returns:
            np.ndarray: Action vector.
        """
        if not isinstance(state, torch.Tensor):
            state = torch.FloatTensor(state).to(self.device)
        else:
            state = state.to(self.device)
        state = state.reshape(1, -1)

        with torch.no_grad():
            action = self.actor.sample(state)
        return action.cpu().data.numpy().flatten()

    def save_model(self, dir, id=None):
        """Save actor and critic parameters to directory.

        Args:
            dir (str): Target directory path.
            id (int | None): Optional suffix for checkpoint naming.
        """
        if id is not None:
            torch.save(self.actor.state_dict(), f'{dir}/actor_{id}.pth')
            torch.save(self.critic1.state_dict(), f'{dir}/critic1_{id}.pth')
            torch.save(self.critic2.state_dict(), f'{dir}/critic2_{id}.pth')
        else:
            torch.save(self.actor.state_dict(), f'{dir}/actor.pth')
            torch.save(self.critic1.state_dict(), f'{dir}/critic1.pth')
            torch.save(self.critic2.state_dict(), f'{dir}/critic2.pth')

    def load_model(self, dir, id=None):
        """Load actor and critic parameters from directory.

        Args:
            dir (str): Source directory path.
            id (int | None): Optional suffix for checkpoint naming.
        """
        if id is not None:
            self.actor.load_state_dict(torch.load(f'{dir}/actor_{id}.pth'))
            self.critic1.load_state_dict(torch.load(f'{dir}/critic1_{id}.pth'))
            self.critic2.load_state_dict(torch.load(f'{dir}/critic2_{id}.pth'))
        else:
            self.actor.load_state_dict(torch.load(f'{dir}/actor.pth'))
            self.critic1.load_state_dict(torch.load(f'{dir}/critic1.pth'))
            self.critic2.load_state_dict(torch.load(f'{dir}/critic2.pth'))
