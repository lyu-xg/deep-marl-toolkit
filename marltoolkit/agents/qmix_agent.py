import os
from copy import deepcopy
from typing import Dict

import numpy as np
import torch
import torch.nn as nn
from torch.distributions import Categorical

from marltoolkit.agents.base_agent import BaseAgent
from marltoolkit.utils import (
    LinearDecayScheduler,
    MultiStepScheduler,
    check_model_method,
    hard_target_update,
    soft_target_update,
)


class QMixAgent(BaseAgent):
    """QMIX algorithm
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
        num_envs: int = 1,
        num_agents: int = None,
        double_q: bool = True,
        total_steps: int = 1e6,
        gamma: float = 0.99,
        optimizer_type: str = "rmsprop",
        learning_rate: float = 0.0005,
        min_learning_rate: float = 0.0001,
        egreedy_exploration: float = 1.0,
        min_exploration: float = 0.01,
        target_update_tau: float = 0.01,
        target_update_interval: int = 100,
        learner_update_freq: int = 1,
        clip_grad_norm: float = 10,
        optim_alpha: float = 0.99,
        optim_eps: float = 0.00001,
        device: str = "cpu",
    ) -> None:
        check_model_method(actor_model, "init_hidden", self.__class__.__name__)
        check_model_method(actor_model, "forward", self.__class__.__name__)
        if mixer_model is not None:
            check_model_method(mixer_model, "forward", self.__class__.__name__)
            assert hasattr(mixer_model, "num_agents") and not callable(
                getattr(mixer_model, "num_agents", None)
            ), "mixer_model needs to have attribute num_agents"
        assert isinstance(gamma, float)
        assert isinstance(learning_rate, float)

        self.num_envs = num_envs
        self.num_agents = num_agents
        self.double_q = double_q
        self.gamma = gamma
        self.optimizer_type = optimizer_type
        self.learning_rate = learning_rate
        self.min_learning_rate = min_learning_rate
        self.clip_grad_norm = clip_grad_norm
        self.global_steps = 0
        self.exploration = egreedy_exploration
        self.min_exploration = min_exploration
        self.target_update_count = 0
        self.target_update_tau = target_update_tau
        self.target_update_interval = target_update_interval
        self.learner_update_freq = learner_update_freq

        self.device = device
        self.actor_model = actor_model
        self.target_actor_model = deepcopy(self.actor_model)
        self.actor_model.to(device)
        self.target_actor_model.to(device)

        self.params = list(self.actor_model.parameters())

        self.mixer_model = None
        if mixer_model is not None:
            self.mixer_model = mixer_model
            self.target_mixer_model = deepcopy(self.mixer_model)
            self.mixer_model.to(device)
            self.target_mixer_model.to(device)
            self.params += list(self.mixer_model.parameters())

        if self.optimizer_type == "adam":
            self.optimizer = torch.optim.Adam(params=self.params, lr=self.learning_rate)
        else:
            self.optimizer = torch.optim.RMSprop(
                params=self.params,
                lr=self.learning_rate,
                alpha=optim_alpha,
                eps=optim_eps,
            )

        self.ep_scheduler = LinearDecayScheduler(egreedy_exploration, total_steps * 0.8)

        lr_milstons = [total_steps * 0.5, total_steps * 0.8]
        self.lr_scheduler = MultiStepScheduler(
            start_value=learning_rate,
            max_steps=total_steps,
            milestones=lr_milstons,
            decay_factor=0.1,
        )

        # 执行过程中，要为每个agent都维护一个 hidden_state
        # 学习过程中，要为每个agent都维护一个 hidden_state、target_hidden_state
        self.hidden_state = None
        self.target_hidden_state = None

    def init_hidden_states(self, batch_size: int = 1) -> None:
        """Initialize hidden states for each agent.

        Args:
            batch_size (int): batch size
        """
        self.hidden_state = self.actor_model.init_hidden()
        self.hidden_state = self.hidden_state.unsqueeze(0).expand(
            batch_size, self.num_agents, -1
        )

        self.target_hidden_state = self.target_actor_model.init_hidden()
        if self.target_hidden_state is not None:
            self.target_hidden_state = self.target_hidden_state.unsqueeze(0).expand(
                batch_size, self.num_agents, -1
            )

    def sample(self, obs: torch.Tensor, available_actions: torch.Tensor) -> np.ndarray:
        """sample actions via epsilon-greedy
        Args:
            obs (np.ndarray):               (num_agents, obs_shape)
            available_actions (np.ndarray): (num_agents, n_actions)
        Returns:
            actions (np.ndarray): sampled actions of agents
        """
        epsilon = np.random.random()
        if epsilon < self.exploration:
            available_actions = torch.tensor(available_actions)
            actions_dist = Categorical(available_actions)
            actions = actions_dist.sample().numpy()

        else:
            actions = self.predict(obs, available_actions)

        # update exploration
        self.exploration = max(self.ep_scheduler.step(), self.min_exploration)
        return actions

    def predict(self, obs: torch.Tensor, available_actions: torch.Tensor) -> np.ndarray:
        """take greedy actions
        Args:
            obs (np.ndarray):               (num_agents, obs_shape)
            available_actions (np.ndarray): (num_agents, n_actions)
        Returns:
            actions (np.ndarray):           (num_agents, )
        """
        obs = torch.tensor(obs, dtype=torch.float32, device=self.device)
        available_actions = torch.tensor(
            available_actions, dtype=torch.long, device=self.device
        )
        agents_q, self.hidden_state = self.actor_model(obs, self.hidden_state)

        # mask unavailable actions
        agents_q[available_actions == 0] = -1e10
        actions = agents_q.max(dim=1)[1].detach().cpu().numpy()
        return actions

    def update_target(self, target_update_tau: float = 0.05) -> None:
        """Update target network with the current network parameters.

        Args:
            target_update_tau (float): target update tau
        """
        if target_update_tau:
            soft_target_update(
                self.actor_model, self.target_actor_model, target_update_tau
            )
            soft_target_update(
                self.mixer_model, self.target_mixer_model, target_update_tau
            )
        else:
            hard_target_update(self.actor_model, self.target_actor_model)
            hard_target_update(self.mixer_model, self.target_mixer_model)

    def learn(self, episode_data: Dict[str, np.ndarray]):
        """Update the model from a batch of experiences.

        Args: episode_data (dict) with the following:

            - obs (np.ndarray):                     (batch_size, T, num_agents, obs_shape)
            - state (np.ndarray):                   (batch_size, T, state_shape)
            - actions (np.ndarray):                 (batch_size, T, num_agents)
            - rewards (np.ndarray):                 (batch_size, T, 1)
            - dones (np.ndarray):                   (batch_size, T, 1)
            - available_actions (np.ndarray):       (batch_size, T, num_agents, n_actions)
            - filled (np.ndarray):                  (batch_size, T, 1)

        Returns:
            - mean_loss (float): train loss
            - mean_td_error (float): train TD error
        """
        # get the data from episode_data buffer
        obs_episode = episode_data["obs"]
        state_episode = episode_data["state"]
        actions_episode = episode_data["actions"]
        available_actions_episode = episode_data["available_actions"]
        rewards_episode = episode_data["rewards"]
        dones_episode = episode_data["dones"]
        filled_episode = episode_data["filled"]

        # update target model
        if self.global_steps % self.target_update_interval == 0:
            self.update_target(self.target_update_tau)
            self.target_update_count += 1

        self.global_steps += 1

        # set the actions to torch.Long
        actions_episode = torch.tensor(
            actions_episode, dtype=torch.long, device=self.device
        )
        # get the batch_size and episode_length
        batch_size, episode_len, _ = state_episode.shape

        # get the relevant quantitles
        actions_episode = actions_episode[:, :-1, :].unsqueeze(-1)
        rewards_episode = rewards_episode[:, :-1, :]
        dones_episode = dones_episode[:, :-1, :].float()
        filled_episode = filled_episode[:, :-1, :].float()

        mask = (1 - dones_episode) * (1 - filled_episode)

        # Calculate estimated Q-Values
        local_qs = []
        target_local_qs = []
        self.init_hidden_states(batch_size)
        for t in range(episode_len):
            obs = obs_episode[:, t, :, :]
            # obs: (batch_size * num_agents, obs_shape)
            obs = obs.reshape(-1, obs.shape[-1])
            # Calculate estimated Q-Values
            local_q, self.hidden_state = self.actor_model(obs, self.hidden_state)
            # local_q: (batch_size * num_agents, n_actions) -->  (batch_size, num_agents, n_actions)
            local_q = local_q.reshape(batch_size, self.num_agents, -1)
            local_qs.append(local_q)

            # Calculate the Q-Values necessary for the target
            target_local_q, self.target_hidden_state = self.target_actor_model(
                obs, self.target_hidden_state
            )
            # target_local_q: (batch_size * num_agents, n_actions) -->  (batch_size, num_agents, n_actions)
            target_local_q = target_local_q.view(batch_size, self.num_agents, -1)
            target_local_qs.append(target_local_q)

        # Concat over time
        local_qs = torch.stack(local_qs, dim=1)
        # We don't need the first timesteps Q-Value estimate for calculating targets
        target_local_qs = torch.stack(target_local_qs[1:], dim=1)

        # Pick the Q-Values for the actions taken by each agent
        # Remove the last dim
        chosen_action_local_qs = torch.gather(
            local_qs[:, :-1, :, :], dim=3, index=actions_episode
        ).squeeze(3)

        # mask unavailable actions
        target_local_qs[available_actions_episode[:, 1:, :] == 0] = -1e10

        # Max over target Q-Values
        if self.double_q:
            # Get actions that maximise live Q (for double q-learning)
            local_qs_detach = local_qs.clone().detach()
            local_qs_detach[available_actions_episode == 0] = -1e10
            cur_max_actions = local_qs_detach[:, 1:].max(dim=3, keepdim=True)[1]
            target_local_max_qs = torch.gather(
                target_local_qs, dim=3, index=cur_max_actions
            ).squeeze(3)
        else:
            # idx0: value, idx1: index
            target_local_max_qs = target_local_qs.max(dim=3)[0]

        # Mixing network
        # mix_net, input: ([Q1, Q2, ...], state), output: Q_total
        if self.mixer_model is not None:
            chosen_action_global_qs = self.mixer_model(
                chosen_action_local_qs, state_episode[:, :-1, :]
            )
            target_global_max_qs = self.target_mixer_model(
                target_local_max_qs, state_episode[:, 1:, :]
            )

        if self.mixer_model is None:
            target_max_qvals = target_local_max_qs
            chosen_action_qvals = chosen_action_local_qs
        else:
            target_max_qvals = target_global_max_qs
            chosen_action_qvals = chosen_action_global_qs

        # Calculate 1-step Q-Learning targets
        target = rewards_episode + self.gamma * (1 - dones_episode) * target_max_qvals
        #  Td-error
        td_error = target.detach() - chosen_action_qvals

        # 0-out the targets that came from padded data
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
            param_group["lr"] = self.learning_rate

        results = {
            "loss": loss.item(),
            "mean_td_error": mean_td_error.item(),
        }
        return results

    def save_model(
        self,
        save_dir: str = None,
        actor_model_name: str = "actor_model.th",
        mixer_model_name: str = "mixer_model.th",
        opt_name: str = "optimizer.th",
    ):
        if not os.path.exists(save_dir):
            os.mkdir(save_dir)
        actor_model_path = os.path.join(save_dir, actor_model_name)
        mixer_model_path = os.path.join(save_dir, mixer_model_name)
        optimizer_path = os.path.join(save_dir, opt_name)
        torch.save(self.actor_model.state_dict(), actor_model_path)
        torch.save(self.mixer_model.state_dict(), mixer_model_path)
        torch.save(self.optimizer.state_dict(), optimizer_path)
        print("save model successfully!")

    def load_model(
        self,
        save_dir: str = None,
        actor_model_name: str = "actor_model.th",
        mixer_model_name: str = "mixer_model.th",
        opt_name: str = "optimizer.th",
    ):
        actor_model_path = os.path.join(save_dir, actor_model_name)
        mixer_model_path = os.path.join(save_dir, mixer_model_name)
        optimizer_path = os.path.join(save_dir, opt_name)
        self.actor_model.load_state_dict(torch.load(actor_model_path))
        self.mixer_model.load_state_dict(torch.load(mixer_model_path))
        self.optimizer.load_state_dict(torch.load(optimizer_path))
        print("restore model successfully!")
