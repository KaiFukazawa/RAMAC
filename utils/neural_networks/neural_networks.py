"""Implementation of different Neural Networks with pytorch."""
import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from .utilities import update_parameters, parse_layers

__all__ = ['DeterministicNN_IQN']

class DeterministicNN_IQN(nn.Module):
    """Deterministic NN Implementation for Implicit Quantile Network
    with continuous actions.
    Returns Q function given the triplet (state,action, tau) where tau
    is the confidence level.

    Parameters
    ----------
    dim_state: int
        dimension of state input to neural network.
    dim_action: int
        dimension of action input to neural network.
    layers_*: list of int, optional
        list of width of neural network layers, each separated with a
        'non_linearity' type non-linearity.
        *==state: layers mapping state input
        *==action: layers mapping action input
        *==f: layers mapping all 3 inputs together
    embedding_dim: dimension to map cat(state,action) to, and tau to.
    tau_embed_dim: int, optional, default 1
        if >1 map tau to a learned linear function of
        tau_embed_dim cosine basis functions of the form cos(pi*i*tau); where
        i = 1... tau_embed_dim. As in paper.

    biased_head: bool, optional, default = True
        flag that indicates if head of NN has a bias term or not.
    non_linearity: str, optional, default = 'ReLU'
        type of nonlinearity between layers

    tau: float [0,1], optional, default 1.0
        Regulates soft update of target parameters.
        % of new parameters used to update target parameters

     References
    ----------
    Will Dabney and Georg Ostrovski and David Silver and Rémi Munos
    Implicit Quantile Networks for Distributional Reinforcement Learning
    2018
    """

    def __init__(self, dim_state, dim_action,
                 layers_state: list = None,
                 layers_action: list = None,
                 layers_f: list = None,
                 embedding_dim=None,
                 tau_embed_dim=1,
                 biased_head=True,
                 non_linearity='ReLU',
                 tau=1.0):

        super().__init__()
        self.dim_state = dim_state
        self.dim_action = dim_action
        self.layers_state = layers_state or list()
        self.layers_action = layers_action or list()
        self.layers_f = layers_f or list()
        self.embedding_dim = embedding_dim
        self.tau_embed_dim = tau_embed_dim
        self.tau = tau

        # Map state:
        self.fc_state, state_out_dim = parse_layers(
            layers_state, self.dim_state, non_linearity, normalized=True)
        # Map action:
        self.fc_action, action_out_dim = parse_layers(
            layers_action, self.dim_action, non_linearity, normalized=True)

        # Map cat(state,action) to embedding_dim
        self.fc_state_action, _ = parse_layers(
            self.embedding_dim,
            state_out_dim + action_out_dim,
            non_linearity,
            normalized=True)

        # Prepare to map with cosine basis functions
        if self.tau_embed_dim > 1:
            self.i_ = torch.Tensor(np.arange(tau_embed_dim))

        # Map tau to embedding_dim
        self.head_tau, _ = parse_layers(self.embedding_dim,
                                        tau_embed_dim, non_linearity,
                                        normalized=True)

        # Map [state,action,tau] to in_dim:
        self.hidden_layers_f, in_dim = parse_layers(
            layers_f, self.embedding_dim, non_linearity, normalized=True)
        # Layer mapping to 1-dim value function. No non-linearity added.
        self.head = nn.Linear(in_dim, 1, bias=biased_head)
        self.output_layer = nn.Sequential(self.hidden_layers_f, self.head)

        for name, param in self.named_parameters():
            if 'head.weight' in name:
                torch.nn.init.uniform_(param, -3e-4, 3e-4)
            if 'head.bias' in name:
                torch.nn.init.zeros_(param)

    def forward(self, state, tau_quantile, action=None):
        """Execute forward computation of the Neural Network.

        Parameters
        ----------
        state: torch.Tensor
            Tensor of size [batch_size x dim_state]
        action: torch.Tensor
            Tensor of size [batch_size x dim_action]
        tau_quantile: torch.Tensor
        Tensor of size [batch_size x 1]

        Returns
        -------
        output: torch.Tensor
            [batch_size x 1] (Q_function for triplet (state,action, tau)
        """

        state_output = self.fc_state(state)  # [batch_size x state_layer]
        action_output = self.fc_action(action)  # [batch_size x action_layer]
        # print(f"Input state shape: {state.shape}")
        # print(f"Input action shape: {action.shape}")
        # print(f"Input tau shape: {tau_quantile.shape}")
        state_action_output = self.fc_state_action(
            torch.cat((state_output, action_output), dim=-1))
        # [batch_size x  embedding_dim]

        # Cosine basis functions of the form cos(pi*i*tau)
        if self.tau_embed_dim > 1:
            a = torch.cos(torch.Tensor([math.pi]).to(tau_quantile.device)*self.i_.to(tau_quantile.device)*tau_quantile)
        else:
            a = tau_quantile
        tau_output = self.head_tau(a)  # [batch_size x embedding_dim]

        output = self.output_layer(
            torch.mul(state_action_output, tau_output)
        ).view(-1, 1)
        return output

    def get_sampled_Z(self, state, confidences, action):
        """Runs IQN for K different confidence levels
        Parameters
        ----------
        state: torch.Tensor [batch_size x dim_state]
        confidences: torch.Tensor. [1 x K]
        Returns
        -------
        Z_tau_K: torch.Tensor [batch_size x K]

        """
        # print(f"Input state shape: {state.shape}")
        # print(f"Input action shape: {action.shape}")
        # print(f"Input tau_N shape: {confidences.shape}")
        # print(f"Confidence levels: {confidences}")
        K = confidences.size(0)  # number of confidence levels to evaluate
        batch_size = state.size(0) if state.dim() > 1 else 1
        # print(f"Batch size: {batch_size}")
        # Reorganize so that the NN runs per one quantile at a time. Repeat
        # all batch_size block "num_quantiles" times:
        # [batch_size * K, dim_state]
        x = state.repeat(1, K).view(-1, self.dim_state)
        # [batch_size * K, dim_state]
        a = action.repeat(1, K).view(-1, self.dim_action)
        y = confidences.repeat(batch_size, 1).view(K*batch_size, 1) # [batch_size * K, 1]
        # print(f"Input x shape: {x.shape}")
        # print(f"Input a shape: {a.shape}")
        # print(f"Input y shape: {y.shape}")
        Z_tau_K = self(state=x, tau_quantile=y, action=a).view(batch_size, K)
        # print(f"Output Z_tau_K shape: {Z_tau_K.shape}")
        return Z_tau_K

    @property
    def params(self):  #  do not call it 'parameters' (already a default func)
        """Get iterator of NN parameters."""
        return self.parameters()

    @params.setter
    def params(self, new_params):
        """Set parameters softly."""
        update_parameters(self.params, new_params, self.tau)
