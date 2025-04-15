import os
from copy import deepcopy
from typing import Dict

import numpy as np
import torch
import torch.nn.functional as F
from torch.distributions import Categorical

from marltoolkit.agents.base_agent import BaseAgent
from marltoolkit.modules.actors.rnn import RNNActorModel
from marltoolkit.modules.critics.coma import MLPCriticModel
from marltoolkit.utils import (
    LinearDecayScheduler,
    MultiStepScheduler,
    hard_target_update,
    soft_target_update,
)


class ComaAgent(BaseAgent):
    """Coma algorithm
    Args:
        actor_model (nn.Model): agents' local q network for decision making.
        critic_model (nn.Model): A mixing network which takes local q values as input
            to construct a global Q network.
        double_q (bool): Double-DQN.
        td_lambda (float): lambda of td-lambda return
        gamma (float): discounted factor for reward computation.
        actor_lr (float): actor network learning rate
        critic_lr (float): critic network learning rate
        clip_grad_norm (None, or float): clipped value of gradients' global norm.
    """

    def __init__(
        self,
        actor_model: RNNActorModel = None,
        critic_model: MLPCriticModel = None,
        num_envs: int = 1,
        num_agents: int = None,
        n_actions: int = None,
        double_q: bool = True,
        total_steps: int = 1e6,
        gamma: float = 0.99,
        nstep_return: int = 3,
        td_lambda: float = 0.8,
        entropy_coef: float = 0.01,
        actor_lr: float = 0.0005,
        critic_lr: float = 0.0001,
        egreedy_exploration: float = 1.0,
        min_exploration: float = 0.01,
        update_target_method: str = "hard",
        target_update_interval: int = 100,
        learner_update_freq: int = 1,
        optimizer_type: str = "rmsprop",
        add_value_last_step: bool = True,
        clip_grad_norm: float = 10,
        optim_alpha: float = 0.99,
        optim_eps: float = 0.00001,
        device: str = "cpu",
    ) -> None:
        self.num_envs = num_envs
        self.num_agents = num_agents
        self.n_actions = n_actions
        self.double_q = double_q
        self.gamma = gamma
        self.nstep_return = nstep_return
        self.td_lambda = td_lambda
        self.entropy_coef = entropy_coef
        self.actor_lr = actor_lr
        self.critic_lr = critic_lr
        self.optimizer_type = optimizer_type
        self.add_value_last_step = add_value_last_step
        self.clip_grad_norm = clip_grad_norm
        self.global_steps = 0
        self.exploration = egreedy_exploration
        self.min_exploration = min_exploration
        self.target_update_count = 0
        self.update_target_method = update_target_method
        self.target_update_interval = target_update_interval
        self.learner_update_freq = learner_update_freq
        self.device = device

        self.actor_model = actor_model.to(device)
        self.target_actor_model = deepcopy(actor_model).to(device)

        self.critic_model = critic_model.to(device)
        self.target_critic_model = deepcopy(critic_model).to(device)

        self.actor_params = list(self.actor_model.parameters())
        self.critic_params = list(self.critic_model.parameters())

        if self.optimizer_type == "adam":
            self.agent_optimiser = torch.optim.Adam(
                params=self.actor_params, lr=self.actor_lr
            )
            self.critic_optimiser = torch.optim.Adam(
                params=self.critic_params, lr=self.critic_lr
            )
        else:
            self.actor_optimizer = torch.optim.RMSprop(
                params=self.actor_params,
                lr=self.actor_lr,
                alpha=optim_alpha,
                eps=optim_eps,
            )

            self.critic_optimizer = torch.optim.RMSprop(
                params=self.actor_params,
                lr=self.critic_lr,
                alpha=optim_alpha,
                eps=optim_eps,
            )

        self.ep_scheduler = LinearDecayScheduler(egreedy_exploration, total_steps * 0.8)

        lr_steps = [total_steps * 0.5, total_steps * 0.8]
        self.actor_lr_scheduler = MultiStepScheduler(
            start_value=self.actor_lr,
            max_steps=total_steps,
            milestones=lr_steps,
            decay_factor=0.5,
        )
        self.critic_lr_scheduler = MultiStepScheduler(
            start_value=self.critic_lr,
            max_steps=total_steps,
            milestones=lr_steps,
            decay_factor=0.5,
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

    def sample(self, obs: np.array, available_actions: np.array):
        """sample actions via epsilon-greedy
        Args:
            obs (np.ndarray):               (num_agents, obs_shape)
            available_actions (np.ndarray): (num_agents, n_actions)
        Returns:
            actions (np.ndarray): sampled actions of agents
        """
        epsilon = np.random.random()
        if epsilon < self.exploration:
            available_actions = torch.tensor(available_actions, dtype=torch.float32)
            actions_dist = Categorical(available_actions)
            actions = actions_dist.sample().numpy()

        else:
            actions = self.predict(obs, available_actions)

        # update exploration
        self.exploration = max(self.ep_scheduler.step(), self.min_exploration)
        return actions

    def predict(self, obs: torch.Tensor, available_actions: torch.Tensor):
        """Predict actions for each agent.

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
        policy_logits, self.hidden_state = self.actor_model(obs, self.hidden_state)
        # mask unavailable actions
        policy_logits[available_actions == 0] = -1e10
        pi_logits = F.softmax(policy_logits, dim=-1)
        actions = pi_logits.max(dim=1)[1].detach().cpu().numpy()
        return actions

    def update_target(self, update_method: str = "hard") -> None:
        """Update target network with the current network parameters.

        Args:
            update_method (str): update method, hard or soft
        """
        if update_method == "soft":
            soft_target_update(self.actor_model, self.target_actor_model)
            soft_target_update(self.critic_model, self.target_critic_model)
        elif update_method == "hard":
            hard_target_update(self.actor_model, self.target_actor_model)
            hard_target_update(self.critic_model, self.target_critic_model)
        else:
            raise ValueError("update method not supported")

    def learn(self, episode_data: Dict[str, np.ndarray]):
        """Update the model from a batch of experiences.

        Args: episode_data (dict) with the following:

            - obs (np.ndarray):                     (batch_size, T, num_agents, obs_shape)
            - state (np.ndarray):                   (batch_size, T, state_shape)
            - actions (np.ndarray):                 (batch_size, T, num_agents)
            - rewards (np.ndarray):                  (batch_size, T, 1)
            - dones (np.ndarray):                    (batch_size, T, 1)
            - available_actions (np.ndarray):        (batch_size, T, num_agents, n_actions)
            - filled (np.ndarray):                   (batch_size, T, 1)

        Returns:
            - mean_loss (float): train loss
            - mean_td_error (float): train TD error
        """
        # get the data from episode_data buffer
        obs_episode = episode_data["obs"]
        actions_episode = episode_data["actions"]
        available_actions_episode = episode_data["available_actions"]
        rewards_episode = episode_data["rewards"]
        dones_episode = episode_data["dones"].float()
        filled_episode = episode_data["filled"].float()

        # update target model
        if self.global_steps % self.target_update_interval == 0:
            self.update_target(update_method=self.update_target_method)
            self.target_update_count += 1

        self.global_steps += 1

        # get the batch_size and episode_length
        batch_size, episode_len, num_agents, _ = obs_episode.shape
        # set the actions to torch.Long
        actions_episode = actions_episode.to(self.device, dtype=torch.long)
        mask = (1 - filled_episode) * (1 - dones_episode)
        critic_mask = mask.clone()
        mask = mask.repeat(1, 1, self.num_agents).view(-1)

        # Calculate estimated Q-Values
        local_qs = []
        self.init_hidden_states(batch_size)
        for t in range(episode_len):
            obs = obs_episode[:, t, :, :]
            # obs: (batch_size * num_agents, obs_shape)
            obs = obs.reshape(-1, obs_episode.shape[-1])
            # Calculate estimated Q-Values
            local_q, self.hidden_state = self.actor_model(obs)
            # local_q: (batch_size * num_agents, n_actions) -->  (batch_size, num_agents, n_actions)
            local_q = local_q.reshape(batch_size, self.num_agents, -1)
            local_qs.append(local_q)

        # Concat over time
        # local_qs: (batch_size, T, num_agents, n_actions)
        local_qs = torch.stack(local_qs, dim=1)
        local_qs[available_actions_episode == 0] = -1e10

        logits_pi = F.softmax(local_qs, dim=-1)
        # Calculate q_vals and critic_loss
        q_vals, critic_loss = self.learn_critic(
            obs_episode, actions_episode, rewards_episode, critic_mask
        )

        # Calculate the baseline
        # q_vals: (batch_size * T * num_agents, n_actions)
        q_vals = q_vals.reshape(-1, self.n_actions)
        # pi: (batch_size * T * num_agents, n_actions)
        pi = logits_pi.view(-1, self.n_actions)
        # Calculate the baseline
        # baseline: (batch_size * T * num_agents)
        baseline = (pi * q_vals).sum(-1).detach()

        # Calculate policy grad with mask
        q_taken = torch.gather(
            q_vals, dim=1, index=actions_episode.reshape(-1, 1)
        ).squeeze(1)
        pi_taken = torch.gather(
            pi, dim=1, index=actions_episode.reshape(-1, 1)
        ).squeeze(1)
        pi_taken[mask == 0] = 1.0
        log_pi_taken = torch.log(pi_taken)

        advantagees = (q_taken - baseline).detach()
        entropy = -torch.sum(pi * torch.log(pi + 1e-10), dim=-1)
        actor_loss = (
            -((advantagees * log_pi_taken + self.entropy_coef * entropy) * mask).sum()
            / mask.sum()
        )

        # Optimise actor model
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        if self.clip_grad_norm:
            torch.nn.utils.clip_grad_norm_(self.actor_params, self.clip_grad_norm)
        self.actor_optimizer.step()

        for param_group in self.actor_optimizer.param_groups:
            param_group["lr"] = self.actor_lr

        results = {
            "actor_loss": actor_loss.item(),
            "critic_loss": critic_loss.item(),
        }
        return results

    def learn_critic(
        self,
        obs_episode: torch.Tensor,
        actions_episode: torch.Tensor,
        rewards_episode: torch.Tensor,
        mask_episode: torch.Tensor,
    ) -> None:
        """Update the critic model from a batch of experiences.

        Args: episode_data (dict) with the following:

            - obs_episode (np.ndarray):                     (batch_size, T, num_agents, obs_shape)
            - actions_episode (np.ndarray):                 (batch_size, T, num_agents)
            - rewards_episode (np.ndarray):                  (batch_size, T, 1)
            - mask_episode (np.ndarray):                     (batch_size, T, 1)

        Returns:
            - critic_loss (float): train loss
        """
        # Optimise critic
        # target_q_vals: (batch_size, T, num_agents, n_actions)
        with torch.no_grad():
            target_q_vals = self.target_critic_model(obs_episode)

        # targets_taken: (batch_size, T, num_agents)
        targets_taken = torch.gather(
            target_q_vals, dim=3, index=actions_episode
        ).squeeze(3)

        # Calculate n-step returns
        targets = self.nstep_returns(
            rewards_episode, targets_taken, mask_episode, self.nstep_return
        )
        targets = targets_taken
        # q_vals: (batch_size, T, num_agents)
        q_vals = self.critic_model(obs_episode)
        # q_taken: (batch_size, T, num_agents)
        q_taken = torch.gather(q_vals, dim=3, index=actions_episode)
        # td_error: (batch_size, T, num_agents)
        td_error = q_taken - targets.detach()
        # masked_td_error: (batch_size, T, num_agents)
        masked_td_error = td_error * mask_episode
        critic_loss = (masked_td_error**2).sum() / mask_episode.sum()
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        if self.clip_grad_norm:
            torch.nn.utils.clip_grad_norm_(self.critic_params, self.clip_grad_norm)
        self.critic_optimizer.step()

        return q_vals, critic_loss

    def build_td_lambda_targets(
        self, rewards, dones, mask, target_qs, gamma, td_lambda
    ):
        """Build TD-lambda targets for Q-learning.

        Args:
            rewards (torch.Tensor): rewards tensor, in shape (batch_size, T, 1)
            dones (torch.Tensor): dones tensor, in shape (batch_size, T, 1)
            mask (torch.Tensor): mask tensor, in shape (batch_size, T, 1)
            target_qs (torch.Tensor): target Q values tensor, in shape (batch_size, T, num_agents)
            gamma (float): discount factor
            td_lambda (float): lambda for TD-lambda return
        Returns:
            ret (torch.Tensor): lambda-return from t=0 to t=T-1, in shape (batch_size, T-1, num_agents)
        """
        # Initialise  last  lambda -return  for  not  terminated  episodes
        ret = target_qs.new_zeros(*target_qs.shape)
        ret[:, -1] = target_qs[:, -1] * (1 - torch.sum(dones, dim=1))
        # Backwards  recursive  update  of the "forward  view"
        for t in range(ret.shape[1] - 2, -1, -1):
            ret[:, t] = td_lambda * gamma * ret[:, t + 1] + mask[:, t] * (
                rewards[:, t]
                + (1 - td_lambda) * gamma * target_qs[:, t + 1] * (1 - dones[:, t])
            )
        # Returns lambda-return from t=0 to t=T-1, i.e. in B*T-1*A
        return ret[:, 0:-1]

    def nstep_returns(
        self,
        rewards: torch.Tensor,
        values: torch.Tensor,
        mask: torch.Tensor,
        nsteps: torch.Tensor,
    ) -> torch.Tensor:
        """Calculate n-step returns.

        Args:
            rewards (torch.Tensor): rewards tensor, in shape (batch_size, T, 1)
            values (torch.Tensor): values tensor, in shape (batch_size, T, num_agents)
            mask (torch.Tensor): mask tensor, in shape (batch_size, T, 1)
            nsteps (int): n-step return
        Returns:
            torch.Tensor: n-step returns, in shape (batch_size, T-1, num_agents)
        """
        (batch_size, episode_len, num_agents) = values.shape
        nstep_values = torch.zeros_like(values[:, :-1, :], dtype=values.dtype)
        for t_start in range(episode_len):
            nstep_return_t = torch.zeros_like(values[:, :0, :])
            for step in range(nsteps + 1):
                t = t_start + step
                if t >= episode_len:
                    break
                elif step == nsteps:
                    nstep_return_t += self.gamma**step * values[:, t, :] * mask[:, t]
                elif t == episode_len - 1 and self.add_value_last_step:
                    nstep_return_t += self.gamma**step * rewards[:, t] * mask[:, t]
                    nstep_return_t += self.gamma ** (step + 1) * values[:, t + 1, :]
                else:
                    nstep_return_t += self.gamma ** (step) * rewards[:, t] * mask[:, t]
            nstep_values[:, t_start, :] = nstep_return_t
        return nstep_values

    def save_model(
        self,
        save_dir: str = None,
        actor_model_name: str = "actor_model.th",
        critic_model_name: str = "critic_model.th",
        opt_name: str = "optimizer.th",
    ) -> None:
        if not os.path.exists(save_dir):
            os.mkdir(save_dir)
        actor_model_path = os.path.join(save_dir, actor_model_name)
        critic_model_path = os.path.join(save_dir, critic_model_name)
        optimizer_path = os.path.join(save_dir, opt_name)
        torch.save(self.actor_model.state_dict(), actor_model_path)
        torch.save(self.critic_model.state_dict(), critic_model_path)
        torch.save(self.critic_optimizer.state_dict(), optimizer_path)
        print("save model successfully!")

    def load_model(
        self,
        save_dir: str = None,
        actor_model_name: str = "actor_model.th",
        critic_model_name: str = "critic_model.th",
        opt_name: str = "optimizer.th",
    ) -> None:
        actor_model_path = os.path.join(save_dir, actor_model_name)
        critic_model_path = os.path.join(save_dir, critic_model_name)
        optimizer_path = os.path.join(save_dir, opt_name)
        self.actor_model.load_state_dict(torch.load(actor_model_path))
        self.critic_model.load_state_dict(torch.load(critic_model_path))
        self.critic_optimizer.load_state_dict(torch.load(optimizer_path))
        print("restore model successfully!")
