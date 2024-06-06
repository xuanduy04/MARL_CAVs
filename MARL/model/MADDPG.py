"""
'Almost' single-file implementation of MADDPG, with it's ReplayBuffer on a seperate file

based on cleanrl's DDPG implementation and openai's original MADDPG implementation
"""

from typing import Tuple, List, Union

import numpy as np
import torch
import torch.nn as nn
from numpy import ndarray
from torch import Tensor
from torch.optim import Adam
from torch.distributions.categorical import Categorical

# replay buffer
from MARL.common.replay_buffer import ReplayBuffer

# for type hints
from config import Config
from highway_env.envs import AbstractEnv
# for standardization between multiple algorithms
from MARL.model import BaseModel
# for quick network initialization
from MARL.common.network import layer_init

# noinspection PyUnresolvedReferences
# debug utilities
from MARL.utils.debug_utils import checknan, checknan_Sequential, analyze, printd


# ALGO LOGIC: initialize agent here (similar to cleanrl, but uses nn.Sequential)
class QNetwork(nn.Module):
    def __init__(self, state_dim: int, num_agents: int, hidden_size: int):
        super(QNetwork, self).__init__()
        self.fc = nn.Sequential(
            layer_init(nn.Linear(state_dim + num_agents, hidden_size)),
            nn.ReLU(),
            layer_init(nn.Linear(hidden_size, hidden_size)),
            nn.ReLU(),
            layer_init(nn.Linear(hidden_size, 1)),
        )

    def forward(self, state: Tensor, action: Tensor) -> Tensor:
        x = torch.cat([state.view(state.shape[0], -1), action], dim=1)
        x = self.fc(x)
        return x


class Actor(nn.Module):
    """DDPG actor network, similar to that of cleanrl, but for descrete action space"""

    def __init__(self, state_dim: int, action_dim: int, hidden_size: int):
        super(Actor, self).__init__()
        self.fc = nn.Sequential(
            layer_init(nn.Linear(state_dim, hidden_size)),
            nn.ReLU(),
            layer_init(nn.Linear(hidden_size, hidden_size)),
            nn.ReLU(),
            layer_init(nn.Linear(hidden_size, action_dim)),
            nn.Tanh()
        )

    def forward(self, state: Tensor) -> Tensor:
        logits = self.fc(state)
        probs = Categorical(logits=logits)
        action = probs.sample()
        return action


# noinspection PyUnusedLocal
class MADDPG(BaseModel):
    def __init__(self, config: Config):
        super(MADDPG, self).__init__(config)

        # replay buffer
        self.rb = ReplayBuffer(config.model.buffer_size,
                               config.env.state_dim,
                               config.env.num_CAV,
                               config.device)
        # updates are based off timesteps, not episode, so this needs to be stored.
        self.current_step = 0  # init = 0 as it's numbered from 1

        # save action_dim for random action
        self.action_dim = config.env.action_dim
        # save number of agents.
        self.num_agents = config.env.num_CAV

        actor_args = (config.env.state_dim, config.env.action_dim, config.model.hidden_size)
        qnet_args = (config.env.state_dim, config.env.num_CAV, config.model.hidden_size)
        device = config.device
        # init networks
        self.actor = Actor(*actor_args).to(device)
        self.qnet = QNetwork(*qnet_args).to(device)
        # init target networks
        self.actor_target = Actor(*actor_args).to(device)
        self.qnet_target = QNetwork(*qnet_args).to(device)
        self.actor_target.load_state_dict(self.actor.state_dict())
        self.qnet_target.load_state_dict(self.qnet.state_dict())
        # optimizer
        self.actor_optimizer = Adam(self.actor.parameters(), lr=config.model.learning_rate)
        self.qnet_optimizer = Adam(self.qnet.parameters(), lr=config.model.learning_rate)

    def _random_action(self) -> Union[int, ndarray]:
        """Takes a random exploration action."""
        return np.random.randint(0, self.action_dim, (self.num_agents,))

    def train(self, env: AbstractEnv, curriculum_training: bool = False, global_episode: int = 0):
        """
        Interacts with the environment once (till termination).
        Update model parameters once every (config.model.policy_frequency) timesteps.
        """
        # set up variables
        device = self.config.device
        args = self.config.model
        begin_step = self.current_step

        # TRY NOT TO MODIFY: start the game
        obs, _ = env.reset(curriculum_training=curriculum_training)
        done = False
        # NOTE: current_step is count from 0
        while not done:
            self.current_step += 1
            # ALGO LOGIC: put actions logic here
            if self.current_step > args.learning_starts:
                with torch.no_grad():
                    actions = self.actor(torch.Tensor(obs).to(device)).cpu().numpy()
            else:
                actions = self._random_action()

            # TRY NOT TO MODIFY: execute the game and log data.
            next_obs, rewards, next_done, infos = env.step(actions)

            # Save data to reply buffer; handle `final_observation`
            self.rb.add(obs, actions, rewards, next_obs, infos["agents_dones"])

            # TRY NOT TO MODIFY: CRUCIAL step easy to overlook
            obs = next_obs
            done = next_done

            # ALGO LOGIC: training.
            if self.current_step > args.learning_starts:
                printd("BEGIN TRAINING MODEL")
                for agent_id in range(self.num_agents):
                    # Sample random batch from replay buffer
                    b_obs, b_actions, b_rewards, b_next_obs, b_dones = self.rb.sample(args.batch_size)
                    
                    # Update Q-network
                    with torch.no_grad():
                        next_state_actions = self.actor_target(b_next_obs)
                        qnet_next_target = self.qnet_target(b_next_obs, next_state_actions)
                        next_q_value = b_rewards.flatten() + (1 - b_dones.flatten()) * args.gamma * (qnet_next_target.view(-1))

                    qnet_a_values = self.qnet(b_obs, b_actions).view(-1)
                    qnet_loss = ((next_q_value - qnet_a_values) ** 2).mean()

                    self.qnet_optimizer.zero_grad()
                    qnet_loss.backward()
                    self.qnet_optimizer.step()

                    if (self.current_step - begin_step) >= args.policy_frequency:
                        # Update Actor network
                        actor_loss = -self.qnet(b_obs, self.actor(b_obs)).mean()
                        self.actor_optimizer.zero_grad()
                        actor_loss.backward()
                        self.actor_optimizer.step()

                        # update the target network
                        for param, target_param in zip(self.actor.parameters(), self.actor_target.parameters()):
                            target_param.data.copy_(args.tau * param.data + (1 - args.tau) * target_param.data)
                        for param, target_param in zip(self.qnet.parameters(), self.qnet_target.parameters()):
                            target_param.data.copy_(args.tau * param.data + (1 - args.tau) * target_param.data)

                        begin_step = self.current_step

        printd(f'At end of episode {global_episode}, total number of steps = {self.current_step}')

    def _act(self, obs: Tensor) -> np.ndarray:
        action = self.actor(obs)
        return action.cpu().numpy()
