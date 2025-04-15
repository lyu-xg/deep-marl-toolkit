"""
Author: jianzhnie
LastEditors: jianzhnie
Description: RLToolKit is a flexible and high-efficient reinforcement learning framework.
Copyright (c) 2022 by jianzhnie@126.com, All Rights Reserved.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class QMixerModel(nn.Module):
    """Implementation of the QMixer network.

    Args:
        num_agents (int): The number of agents.
        state_dim (int): The shape of the state.
        hypernet_layers (int): The number of layers in the hypernetwork.
        mixing_embed_dim (int): The dimension of the mixing embedding.
        hypernet_embed_dim (int): The dimension of the hypernetwork embedding.
    """

    def __init__(
        self,
        num_agents: int = None,
        state_dim: int = None,
        hypernet_layers: int = 2,
        mixing_embed_dim: int = 32,
        hypernet_embed_dim: int = 64,
    ):
        super(QMixerModel, self).__init__()

        self.num_agents = num_agents
        self.state_dim = state_dim
        self.mixing_embed_dim = mixing_embed_dim
        if hypernet_layers == 1:
            self.hyper_w_1 = nn.Linear(state_dim, mixing_embed_dim * num_agents)
            self.hyper_w_2 = nn.Linear(state_dim, mixing_embed_dim)
        elif hypernet_layers == 2:
            self.hyper_w_1 = nn.Sequential(
                nn.Linear(state_dim, hypernet_embed_dim),
                nn.ReLU(inplace=True),
                nn.Linear(hypernet_embed_dim, mixing_embed_dim * num_agents),
            )
            self.hyper_w_2 = nn.Sequential(
                nn.Linear(state_dim, hypernet_embed_dim),
                nn.ReLU(inplace=True),
                nn.Linear(hypernet_embed_dim, mixing_embed_dim),
            )
        else:
            raise ValueError('hypernet_layers should be "1" or "2"!')

        # State dependent bias for hidden layer
        self.hyper_b_1 = nn.Linear(state_dim, mixing_embed_dim)
        self.hyper_b_2 = nn.Sequential(
            nn.Linear(state_dim, mixing_embed_dim),
            nn.ReLU(inplace=True),
            nn.Linear(mixing_embed_dim, 1),
        )

    def forward(self, agent_qs: torch.Tensor, states: torch.Tensor):
        """
        Args:
            agent_qs (torch.Tensor): (batch_size, T, num_agents)
            states (torch.Tensor):   (batch_size, T, state_dim)
        Returns:
            q_total (torch.Tensor):  (batch_size, T, 1)
        """
        batch_size = agent_qs.size(0)
        # states : (batch_size * T, state_dim)
        states = states.reshape(-1, self.state_dim)
        # agent_qs: (batch_size * T, 1, num_agents)
        agent_qs = agent_qs.view(-1, 1, self.num_agents)

        # First layer w and b
        w1 = torch.abs(self.hyper_w_1(states))
        # w1: (batch_size * T, num_agents, embed_dim)
        w1 = w1.view(-1, self.num_agents, self.mixing_embed_dim)
        b1 = self.hyper_b_1(states)
        b1 = b1.view(-1, 1, self.mixing_embed_dim)

        # Second layer w and b
        # w2 : (batch_size * T, embed_dim)
        w2 = torch.abs(self.hyper_w_2(states))
        # w2 : (batch_size * T, embed_dim, 1)
        w2 = w2.view(-1, self.mixing_embed_dim, 1)
        # State-dependent bias
        b2 = self.hyper_b_2(states).view(-1, 1, 1)

        # First hidden layer
        # hidden: (batch_size * T,  1, embed_dim)
        hidden = F.elu(torch.bmm(agent_qs, w1) + b1)
        # Compute final output
        # y: (batch_size * T,  1, 1)
        y = torch.bmm(hidden, w2) + b2
        # Reshape and return
        # q_total: (batch_size, T, 1)
        q_total = y.view(batch_size, -1, 1)
        return q_total

    def update(self, model):
        self.load_state_dict(model.state_dict())


class QMixerCentralFF(nn.Module):

    def __init__(
        self, num_agents, state_dim, central_mixing_embed_dim, central_action_embed
    ):
        super(QMixerCentralFF, self).__init__()

        self.num_agents = num_agents
        self.state_dim = state_dim
        self.input_dim = num_agents * central_action_embed + state_dim
        self.central_mixing_embed_dim = central_mixing_embed_dim
        self.central_action_embed = central_action_embed

        self.net = nn.Sequential(
            nn.Linear(self.input_dim, central_mixing_embed_dim),
            nn.ReLU(inplace=True),
            nn.Linear(central_mixing_embed_dim, central_mixing_embed_dim),
            nn.ReLU(inplace=True),
            nn.Linear(central_mixing_embed_dim, central_mixing_embed_dim),
            nn.ReLU(inplace=True),
            nn.Linear(central_mixing_embed_dim, 1),
        )

        # V(s) instead of a bias for the last layers
        self.vnet = nn.Sequential(
            nn.Linear(state_dim, central_mixing_embed_dim),
            nn.ReLU(inplace=True),
            nn.Linear(central_mixing_embed_dim, 1),
        )

    def forward(self, agent_qs, states):
        bs = agent_qs.size(0)
        states = states.reshape(-1, self.state_dim)
        agent_qs = agent_qs.reshape(-1, self.num_agents * self.central_action_embed)

        inputs = torch.cat([states, agent_qs], dim=1)

        advs = self.net(inputs)
        vs = self.vnet(states)
        y = advs + vs
        q_tot = y.view(bs, -1, 1)
        return q_tot
