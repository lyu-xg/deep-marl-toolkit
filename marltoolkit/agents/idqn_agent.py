from copy import deepcopy

import numpy as np
import torch
import torch.nn as nn
from torch.distributions import Categorical

from marltoolkit.utils import (LinearDecayScheduler, MultiStepScheduler,
                               check_model_method, hard_target_update)

from .qmix_agent import QMixAgent


class IDQNAgent(QMixAgent):
    """IDQN algorithm
    Args:
        actor_model (nn.Model): agents' local q network for decision making.
        mixer_model (nn.Model): A mixing network which takes local q values as input
            to construct a global Q network.
        double_q (bool): Double-DQN.
        gamma (float): discounted factor for reward computation.
        lr (float): learning rate.
        clip_grad_norm (None, or float): clipped value of gradients' global norm.
    """

    def __init__(
        self,
        actor_model: nn.Module = None,
        mixer_model: nn.Module = None,
        num_agents: int = None,
        double_q: bool = True,
        total_steps: int = 1e6,
        gamma: float = 0.99,
        learning_rate: float = 0.0005,
        min_learning_rate: float = 0.0001,
        exploration_start: float = 1.0,
        min_exploration: float = 0.01,
        update_target_interval: int = 100,
        update_learner_freq: int = 1,
        clip_grad_norm: float = 10,
        optim_alpha: float = 0.99,
        optim_eps: float = 0.00001,
        device: str = 'cpu',
    ):
        check_model_method(actor_model, 'init_hidden', self.__class__.__name__)
        check_model_method(actor_model, 'forward', self.__class__.__name__)
        assert isinstance(gamma, float)
        assert isinstance(learning_rate, float)

        self.num_agents = num_agents
        self.double_q = double_q
        self.gamma = gamma
        self.learning_rate = learning_rate
        self.min_learning_rate = min_learning_rate
        self.clip_grad_norm = clip_grad_norm
        self.global_steps = 0
        self.exploration = exploration_start
        self.min_exploration = min_exploration
        self.target_update_count = 0
        self.update_target_interval = update_target_interval
        self.update_learner_freq = update_learner_freq

        self.device = device
        self.actor_model = actor_model
        self.target_actor_model = deepcopy(self.actor_model)
        self.actor_model.to(device)
        self.target_actor_model.to(device)

        self.params = list(self.actor_model.parameters())

        self.mixer_model = None
        if mixer_model is not None:
            self.mixer_model = mixer_model
            self.mixer_model.to(device)
            self.target_mixer_model = deepcopy(self.mixer_model)
            self.target_mixer_model.to(device)
            self.params += list(self.mixer_model.parameters())

        self.optimizer = torch.optim.RMSprop(params=self.params,
                                             lr=self.learning_rate,
                                             alpha=optim_alpha,
                                             eps=optim_eps)

        self.ep_scheduler = LinearDecayScheduler(exploration_start,
                                                 total_steps * 0.8)

        lr_steps = [total_steps * 0.5, total_steps * 0.8]
        self.lr_scheduler = MultiStepScheduler(start_value=learning_rate,
                                               max_steps=total_steps,
                                               milestones=lr_steps,
                                               decay_factor=0.5)

    def init_hidden_states(self, batch_size: int = 1) -> None:
        self.hidden_states = self.actor_model.init_hidden()
        if self.hidden_states is not None:
            self.hidden_states = self.hidden_states.unsqueeze(0).expand(
                batch_size, self.num_agents, -1)

        self.target_hidden_states = self.target_actor_model.init_hidden()
        if self.target_hidden_states is not None:
            self.target_hidden_states = self.target_hidden_states.unsqueeze(
                0).expand(batch_size, self.num_agents, -1)

    def sample(self, obs, available_actions):
        """sample actions via epsilon-greedy
        Args:
            obs (np.ndarray):               (num_agents, obs_shape)
            available_actions (np.ndarray): (num_agents, n_actions)
        Returns:
            actions (np.ndarray): sampled actions of agents
        """
        epsilon = np.random.random()
        if epsilon < self.exploration:
            available_actions = torch.tensor(available_actions,
                                             dtype=torch.float32)
            actions_dist = Categorical(available_actions)
            actions = actions_dist.sample().long().cpu().detach().numpy()

        else:
            actions = self.predict(obs, available_actions)

        # update exploration
        self.exploration = max(self.ep_scheduler.step(), self.min_exploration)
        return actions

    def predict(self, obs, available_actions):
        """take greedy actions
        Args:
            obs (np.ndarray):               (num_agents, obs_shape)
            available_actions (np.ndarray): (num_agents, n_actions)
        Returns:
            actions (np.ndarray):           (num_agents, )
        """
        obs = torch.tensor(obs, dtype=torch.float32, device=self.device)
        available_actions = torch.tensor(available_actions,
                                         dtype=torch.long,
                                         device=self.device)
        agents_q, self.hidden_states = self.actor_model(
            obs, self.hidden_states)
        # mask unavailable actions
        agents_q[available_actions == 0] = -1e10
        actions = agents_q.max(dim=1)[1].detach().cpu().numpy()
        return actions

    def update_target(self):
        hard_target_update(self.actor_model, self.target_actor_model)
        if self.mixer_model is not None:
            hard_target_update(self.mixer_model, self.target_mixer_model)

    def learn(self, state_batch, actions_batch, reward_batch, terminated_batch,
              obs_batch, available_actions_batch, filled_batch, **kwargs):
        """
        Args:
            state (np.ndarray):                   (batch_size, T, state_shape)
            actions (np.ndarray):                 (batch_size, T, num_agents)
            reward (np.ndarray):                  (batch_size, T, 1)
            terminated (np.ndarray):              (batch_size, T, 1)
            obs (np.ndarray):                     (batch_size, T, num_agents, obs_shape)
            available_actions_batch (np.ndarray): (batch_size, T, num_agents, n_actions)
            filled_batch (np.ndarray):            (batch_size, T, 1)
        Returns:
            mean_loss (float): train loss
            mean_td_error (float): train TD error
        """
        # update target model
        if self.global_steps % self.update_target_interval == 0:
            self.update_target()
            self.target_update_count += 1

        self.global_steps += 1

        # set the actions to torch.Long
        actions_batch = actions_batch.to(self.device, dtype=torch.long)
        # get the batch_size and episode_length
        batch_size = state_batch.shape[0]
        episode_len = state_batch.shape[1]

        # get the relevant quantitles
        reward_batch = reward_batch[:, :-1, :]
        actions_batch = actions_batch[:, :-1, :].unsqueeze(-1)
        terminated_batch = terminated_batch[:, :-1, :]
        filled_batch = filled_batch[:, :-1, :]

        mask = (1 - filled_batch) * (1 - terminated_batch)

        # Calculate estimated Q-Values
        local_qs = []
        target_local_qs = []
        self.init_hidden_states(batch_size)
        for t in range(episode_len):
            obs = obs_batch[:, t, :, :]
            # obs: (batch_size * num_agents, obs_shape)
            obs = obs.reshape(-1, obs_batch.shape[-1])
            # Calculate estimated Q-Values
            local_q, self.hidden_states = self.actor_model(
                obs, self.hidden_states)
            #  local_q: (batch_size * num_agents, n_actions) -->  (batch_size, num_agents, n_actions)
            local_q = local_q.reshape(batch_size, self.num_agents, -1)
            local_qs.append(local_q)

            # Calculate the Q-Values necessary for the target
            target_local_q, self.target_hidden_states = self.target_actor_model(
                obs, self.target_hidden_states)
            # target_local_q: (batch_size * num_agents, n_actions) -->  (batch_size, num_agents, n_actions)
            target_local_q = target_local_q.view(batch_size, self.num_agents,
                                                 -1)
            target_local_qs.append(target_local_q)

        # Concat over time
        local_qs = torch.stack(local_qs, dim=1)
        # We don't need the first timesteps Q-Value estimate for calculating targets
        target_local_qs = torch.stack(target_local_qs[1:], dim=1)

        # Pick the Q-Values for the actions taken by each agent
        # Remove the last dim
        chosen_action_local_qs = torch.gather(local_qs[:, :-1, :, :],
                                              dim=3,
                                              index=actions_batch).squeeze(3)

        # mask unavailable actions
        target_local_qs[available_actions_batch[:, 1:, :] == 0] = -1e10

        # Max over target Q-Values
        if self.double_q:
            # Get actions that maximise live Q (for double q-learning)
            local_qs_detach = local_qs.clone().detach()
            local_qs_detach[available_actions_batch == 0] = -1e10
            cur_max_actions = local_qs_detach[:, 1:].max(dim=3,
                                                         keepdim=True)[1]
            target_local_max_qs = torch.gather(
                target_local_qs, dim=3, index=cur_max_actions).squeeze(3)
        else:
            # idx0: value, idx1: index
            target_local_max_qs = target_local_qs.max(dim=3)[0]

        # Mixing network
        # mix_net, input: ([Q1, Q2, ...], state), output: Q_total
        if self.mixer_model is not None:
            chosen_action_global_qs = self.mixer_model(chosen_action_local_qs,
                                                       state_batch[:, :-1, :])
            target_global_max_qs = self.target_mixer_model(
                target_local_max_qs, state_batch[:, 1:, :])

        if self.mixer_model is None:
            target_max_qvals = target_local_max_qs
            chosen_action_qvals = chosen_action_local_qs
        else:
            target_max_qvals = target_global_max_qs
            chosen_action_qvals = chosen_action_global_qs

        # Calculate 1-step Q-Learning targets
        target = reward_batch + self.gamma * (
            1 - terminated_batch) * target_max_qvals
        #  Td-error
        td_error = target.detach() - chosen_action_qvals

        #  0-out the targets that came from padded data
        masked_td_error = td_error * mask
        mean_td_error = masked_td_error.sum() / mask.sum()
        # Normal L2 loss, take mean over actual data
        loss = (masked_td_error**2).sum() / mask.sum()

        # Optimise
        self.optimizer.zero_grad()
        loss.backward()
        if self.clip_grad_norm:
            torch.nn.utils.clip_grad_norm_(self.params, self.clip_grad_norm)
        self.optimizer.step()

        for param_group in self.optimizer.param_groups:
            param_group['lr'] = self.learning_rate

        return loss.item(), mean_td_error.item()
