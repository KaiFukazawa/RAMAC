# Copyright 2025 University of California, Davis, and Kai Fukazawa.
# SPDX-License-Identifier: Apache-2.0
"""
RAFMAC – Risk‑Averse Flow‑Matching Actor–Critic 
=========================================================
• Double IQN critics + Flow‑Matching actor (Euler K‑step, EMA target)
• Distillation: teacher (K‑step) → student one‑step policy (α controls weight)
• Risk distortions: 'cvar' | 'wang' | 'cpw' | 'power'
• Switch inference cost vs fidelity with `flow_steps` = 1/5/20...
"""
from __future__ import annotations
import copy, torch, torch.nn as nn, torch.nn.functional as F
from torch.distributions import Uniform, Normal
from typing import Tuple, List, Dict

from utils.neural_networks import DeterministicNN_IQN # critics

from agents.helpers import EMA, SinusoidalPosEmb # time‑embed

# ---------------------------------------------------------------------------
# Velocity‑field with explicit time embed (teacher)
# ---------------------------------------------------------------------------
class VF_MLP(nn.Module):
    def __init__(self, s_dim:int, a_dim:int, h:int=128, t_dim:int=4):
        super().__init__()
        self.t_proj = nn.Sequential(
            SinusoidalPosEmb(t_dim),
            nn.Linear(t_dim, t_dim*2), nn.Mish(),
            nn.Linear(t_dim*2, t_dim))
        inp = s_dim + a_dim + t_dim
        self.net = nn.Sequential(
            nn.Linear(inp, h), nn.Mish(),
            nn.Linear(h, h),   nn.Mish(),
            nn.Linear(h, h),   nn.Mish(),
            nn.Linear(h, a_dim))

    def forward(self, x_t, t, s):
        x_t = x_t.flatten(1)
        s   = s.flatten(1)
        t   = self.t_proj(t).flatten(1)
        return self.net(torch.cat([x_t, t, s], -1))

# ---------------------------------------------------------------------------
# One‑step student policy
# ---------------------------------------------------------------------------
class OneStepMLP(nn.Module):
    def __init__(self, s_dim:int, a_dim:int, h:int=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(s_dim + a_dim, h), nn.Mish(),
            nn.Linear(h, h),            nn.Mish(),
            nn.Linear(h, a_dim))
    def forward(self, s, z):
        return self.net(torch.cat([s.flatten(1), z], -1))

# ---------------------------------------------------------------------------
class RAFMAC:
    """Risk‑Averse Flow‑Matching Actor–Critic with distillation (Toy style)."""
    def __init__(self,
                 state_dim:int, action_dim:int, max_action:float, device:str,
                 discount:float=0.99, tau:float=5e-3,
                 eta:float=2.5,                 # risk weight
                 alpha:float=3.0,               # distill weight
                 risk_dist:str='cvar', alpha_cvar:float=0.1,
                 xi:float=0.2, delta:float=0.71, gamma:float=0.7,
                 hidden_dim:int=128, lr:float=3e-4,
                 flow_steps:int=10, q_agg:str='mean',
                 ema_decay:float=0.995, step_start_ema:int=500,
                 update_ema_every:int=5, bandit:bool=True,
                 normalize_q_loss:bool=False,   # Q loss normalization
                 grad_norm:float=0.6,           # gradient clipping
                 use_distillation:bool=False,   # Enable distillation training
                 **extra_kwargs):

        """NOTE: Accept `hidden` keyword from main.py as an alias for `hidden_dim`."""
        # allow alias `hidden`
        if 'hidden' in extra_kwargs and extra_kwargs['hidden'] is not None:
            hidden_dim = extra_kwargs.pop('hidden')
        # --------------------------------------------------------------
        self.d      = torch.device(device)
        self.A      = action_dim
        self.max_a  = max_action
        self.K      = max(1, flow_steps)
        self.discount, self.tau = discount, tau
        self.eta, self.alpha = eta, alpha
        self.bandit, self.q_agg = bandit, q_agg
        self.normalize_q_loss = normalize_q_loss
        self.grad_norm = grad_norm
        self.use_distillation = use_distillation

        # teacher / student ----------------------------------------------
        self.vf     = VF_MLP(state_dim, action_dim, hidden_dim).to(self.d)
        self.one    = OneStepMLP(state_dim, action_dim, hidden_dim).to(self.d)
        self.opt_v  = torch.optim.Adam(self.vf.parameters(),  lr=lr)
        self.opt_one= torch.optim.Adam(self.one.parameters(), lr=lr)

        # EMA teacher for bootstrap
        self.ema        = EMA(ema_decay)
        self.ema_v      = copy.deepcopy(self.vf)
        self.step_ctr   = 0
        self.step_start = step_start_ema
        self.update_every = update_ema_every

        # critics ---------------------------------------------------------
        def make_q():
            return DeterministicNN_IQN(state_dim, action_dim,
                                        layers_state=[hidden_dim],
                                        layers_action=[hidden_dim],
                                        layers_f=[hidden_dim],
                                        embedding_dim=hidden_dim,
                                        tau_embed_dim=32).to(self.d)
        # Double IQN critics
        self.q1     = make_q()
        self.q2     = make_q()
        self.q1_tgt = copy.deepcopy(self.q1)
        self.q2_tgt = copy.deepcopy(self.q2)
        self.opt_q1 = torch.optim.Adam(self.q1.parameters(), lr=lr)
        self.opt_q2 = torch.optim.Adam(self.q2.parameters(), lr=lr)

        # distributions ---------------------------------------------------
        self.u = Uniform(0.,1.)
        self.risk_dist, self.alpha_cvar = risk_dist, alpha_cvar
        self.xi, self.delta, self.gamma = xi, delta, gamma

    # ---------- helpers --------------------------------------------------
    def _rollout(self, s:torch.Tensor, z:torch.Tensor)->torch.Tensor:
        x = z
        B = x.size(0)
        for k in range(self.K):
            t = torch.full((B,1), k/self.K, device=self.d)
            v = self.vf(x, t, s)
            x = x + v / self.K
        return torch.clamp(x, -self.max_a, self.max_a)

    def _risk_map(self, tau):
        if self.risk_dist=='cvar':  return tau*self.alpha_cvar
        if self.risk_dist=='wang':  return Normal(0,1).cdf(Normal(0,1).icdf(tau)+self.xi)
        if self.risk_dist=='cpw':   d=self.delta; num=tau**d; return num/(num+(1-tau)**d)
        if self.risk_dist=='power': return tau**self.gamma
        return tau

    @staticmethod
    def _huber(target, current, tau, kappa=1.):
        td = target-current
        hub= torch.where(td.abs()<=kappa,0.5*td.pow(2),kappa*(td.abs()-0.5*kappa))
        tau= tau.view(1,-1)
        return ((tau-(td.detach()<0).float()).abs()*hub).mean()

    @staticmethod
    def _unpack(batch:Tuple):
        """Organize Data_Sampler returned tuple to (s,a,ns,r,d)
        * If r has shape other than (B,) or (B,1) (e.g., (B,17)), treat first component as reward.
        """
        if len(batch)==5:
            s,a,ns,r,d = batch
        elif len(batch)==3:
            s,a,r = batch
            ns, d = s, torch.zeros_like(r) 
        else:
            raise ValueError("ReplayBuffer must return 3 or 5 tensors")

        # --- ensure reward and done shape = (B,1) ----------------------------------
        if r.dim()==1:
            r = r.unsqueeze(1)
        elif r.dim()==2 and r.size(1) != 1:
            # For matrices containing features other than reward → treat first column as reward
            r = r[:, :1]
        else:
            # (B,1) or unexpected high dimension → keep as is / warning
            pass

        if d.dim()==1:
            d = d.unsqueeze(1)

        # OGBench is basically {0,1}. No scaling needed.
        r = torch.clamp(r, -1.0, 1.0)

        return s,a,ns,r,d

    # --------------------------------------------------------------------
    def sample(self, s, use_student:bool=False):
        if not torch.is_tensor(s):
            s = torch.tensor(s, dtype=torch.float32, device=self.d)
        single = s.dim()==1
        if single: s=s.unsqueeze(0)
        z = torch.randn(s.size(0), self.A, device=self.d)
        with torch.no_grad():
            a = self.one(s,z) if use_student else self._rollout(s,z)
            # Ensure action clipping for student policy
            if use_student:
                a = torch.clamp(a, -self.max_a, self.max_a)
        return a.squeeze(0).cpu().numpy() if single else a.cpu().numpy()

    sample_action = sample

    # --------------------------------------------------------------------
    def train(self, replay_buffer, iterations:int, batch_size:int=256, log_writer=None)->Dict[str,List[float]]:
        log = {'bc_loss':[], 'actor_loss':[], 'critic_loss':[], 'q_abs_mean':[], 'td_target_mean':[], 'a_student_max':[], 'not_done_mean':[]}
        for _ in range(iterations):
            batch = replay_buffer.sample(batch_size)
            s,a,ns,r,d = self._unpack(batch)
            s,a,ns,r,d = [x.to(self.d) for x in (s,a,ns,r,d)]
            # Critic ---------------------------------------------------
            tau = self.u.sample((32,)).to(self.d)
            tau_ = self.u.sample((32,)).to(self.d)

            # Current Q values
            q1 = self.q1.get_sampled_Z(s, tau, a)
            q2 = self.q2.get_sampled_Z(s, tau, a)

            if self.bandit:
                tgt = r.expand(r.size(0), 32)
            else:
                with torch.no_grad():
                    na = self._rollout(ns, torch.randn_like(a))
                    q1_t = self.q1_tgt.get_sampled_Z(ns, tau_, na)
                    q2_t = self.q2_tgt.get_sampled_Z(ns, tau_, na)
                    # min target for Double Q-learning
                    q_t = torch.min(q1_t, q2_t)
                    not_done = (1.0 - d.float())
                    tgt = r + not_done * self.discount * q_t

            # Critic losses
            c_loss1 = self._huber(tgt, q1, tau)
            c_loss2 = self._huber(tgt, q2, tau)
            c_loss = 0.5 * (c_loss1 + c_loss2)

            # Update critic1
            self.opt_q1.zero_grad()
            c_loss1.backward()
            if self.grad_norm > 0:
                nn.utils.clip_grad_norm_(self.q1.parameters(), max_norm=self.grad_norm, norm_type=2)
            self.opt_q1.step()

            # Update critic2
            self.opt_q2.zero_grad()
            c_loss2.backward()
            if self.grad_norm > 0:
                nn.utils.clip_grad_norm_(self.q2.parameters(), max_norm=self.grad_norm, norm_type=2)
            self.opt_q2.step()

            # Update target networks
            with torch.no_grad():
                for p, tp in zip(self.q1.parameters(), self.q1_tgt.parameters()):
                    tp.data.mul_(1-self.tau).add_(self.tau*p.data)
                for p, tp in zip(self.q2.parameters(), self.q2_tgt.parameters()):
                    tp.data.mul_(1-self.tau).add_(self.tau*p.data)
            # Actor ----------------------------------------------------
            z  = torch.randn_like(a)
            t_u= torch.rand((batch_size,1), device=self.d)
            x0 = z; x_t=(1-t_u)*x0 + t_u*a
            v_tgt=a-x0; v_pred=self.vf(x_t,t_u,s)
            bc = F.mse_loss(v_pred, v_tgt)   

            # Actor training with optional distillation
            if self.use_distillation:
                # Option 1: Distillation On
                with torch.no_grad():
                    a_teacher = self._rollout(s, z)
                
                a_student = self.one(s, z)
                dist = F.mse_loss(a_student, a_teacher)
                
                # Risk-aware Q evaluation for distillation
                tau_r = self._risk_map(self.u.sample((32,)).to(self.d))
                q_s = self.q1.get_sampled_Z(s, tau_r, a_student)  # Use single Q for distillation
                q_term = q_s.min(-1, True).values.mean() if self.q_agg=='min' else q_s.mean()
                
                actor_loss = bc + self.alpha * dist - self.eta * q_term
            else:
                # Option 2: Distillation Off (standard RAFMAC)
                with torch.no_grad():
                    a_teacher = self._rollout(s, z)

                a_student = self.one(s, z)
                a_student = torch.clamp(a_student, -self.max_a, self.max_a)  # Mandatory clipping

                # Q(pi) evaluation with Double IQN
                q1_s = self.q1.get_sampled_Z(s, tau, a_student)
                q2_s = self.q2.get_sampled_Z(s, tau, a_student)
                q_s = torch.min(q1_s, q2_s)

                # Q loss calculation
                q_term = q_s.min(-1, True).values.mean() if self.q_agg=='min' else q_s.mean()

                # Normalize Q loss if enabled
                if self.normalize_q_loss:
                    lam = (1.0 / (q_s.abs().mean() + 1e-8)).detach()  # Prevent divergence with ε
                    actor_loss = bc - self.eta * (lam * q_term)
                else:
                    actor_loss = bc - self.eta * q_term

            # Gradient clipping
            self.opt_v.zero_grad()
            self.opt_one.zero_grad()
            actor_loss.backward()

            if self.grad_norm > 0:
                nn.utils.clip_grad_norm_(self.vf.parameters(), max_norm=self.grad_norm, norm_type=2)
                nn.utils.clip_grad_norm_(self.one.parameters(), max_norm=self.grad_norm, norm_type=2)

            self.opt_v.step()
            self.opt_one.step()
            # EMA ------------------------------------------------------
            self.step_ctr += 1
            if self.step_ctr>self.step_start and self.step_ctr%self.update_every==0:
                self.ema.update_model_average(self.ema_v, self.vf)
            # Debug logging --------------------------------------------------
            q_abs_mean = q_s.abs().mean().item()
            td_target_mean = tgt.mean().item()
            a_student_max = a_student.abs().max().item()
            not_done_mean = not_done.mean().item() if not self.bandit else 0.0

            # Logging --------------------------------------------------
            for k,v in zip(['bc_loss','actor_loss','critic_loss','q_abs_mean','td_target_mean','a_student_max','not_done_mean'],
                            [bc,actor_loss,c_loss,q_abs_mean,td_target_mean,a_student_max,not_done_mean]):
                log[k].append(v)
                if log_writer: log_writer.add_scalar(k, v, self.step_ctr)
        return log

    # ---------------- save / load -------------------------
    def save_model(self, d:str, it:int|None=None):
        tag=f"_{it}" if it is not None else ""
        torch.save(self.vf.state_dict(),    f"{d}/vf{tag}.pth")
        torch.save(self.one.state_dict(),   f"{d}/one{tag}.pth")
        torch.save(self.q1.state_dict(),    f"{d}/critic1{tag}.pth")
        torch.save(self.q2.state_dict(),    f"{d}/critic2{tag}.pth")
        torch.save(self.ema_v.state_dict(), f"{d}/ema_v{tag}.pth")
        torch.save(self.q1_tgt.state_dict(), f"{d}/critic1_tgt{tag}.pth")
        torch.save(self.q2_tgt.state_dict(), f"{d}/critic2_tgt{tag}.pth")

    def load_model(self, d:str, it:int|None=None):
        tag=f"_{it}" if it is not None else ""
        self.vf.load_state_dict(torch.load(f"{d}/vf{tag}.pth",     map_location=self.d))
        self.one.load_state_dict(torch.load(f"{d}/one{tag}.pth",    map_location=self.d))
        self.q1.load_state_dict(torch.load(f"{d}/critic1{tag}.pth",   map_location=self.d))
        self.q2.load_state_dict(torch.load(f"{d}/critic2{tag}.pth",   map_location=self.d))
        self.ema_v.load_state_dict(torch.load(f"{d}/ema_v{tag}.pth",map_location=self.d))
        self.q1_tgt.load_state_dict(torch.load(f"{d}/critic1_tgt{tag}.pth", map_location=self.d))
        self.q2_tgt.load_state_dict(torch.load(f"{d}/critic2_tgt{tag}.pth", map_location=self.d))
