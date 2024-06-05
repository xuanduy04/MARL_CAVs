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

import os
import imageio

# replay buffer
from MARL_redux.common.replay_buffer import ReplayBuffer

# for type hints
from config import Config
from highway_env.envs import AbstractEnv
# for standardization between multiple algorithms
from MARL_redux.model import BaseModel
# for quick network initialization
from MARL_redux.common.network import layer_init
# debug utilities
from MARL_redux.utils.debug_utils import checknan, checknan_Sequential, analyze, printd


# ALGO LOGIC: initialize agent here (similar to cleanrl, but uses nn.Sequential)
class QNetwork(nn.Module):
    def __init__(self, state_dim: int, action_dim: int, hidden_size: int):
        super(QNetwork, self).__init__()
        self.fc = nn.Sequential(
            layer_init(nn.Linear(state_dim + action_dim, hidden_size)),
            nn.ReLU(),
            layer_init(nn.Linear(hidden_size, hidden_size)),
            nn.ReLU(),
            layer_init(nn.Linear(hidden_size, 1)),
        )

    def forward(self, state: Tensor, action: Tensor) -> Tensor:
        x = torch.cat([state, action], dim=1)
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
        return self.fc(state)


class MADDPG(BaseModel):
    def __init__(self, config: Config):
        super(MADDPG, self).__init__(config)

        # replay buffer
        self.rb = ReplayBuffer(config.model.buffer_size)
        # updates are based off timesteps, not episode, so this needs to be stored.
        self.current_step = 0  # init as 0 as it's numbered from 1

        self.action_dim = config.env.action_dim
        network_args = (config.env.state_dim, config.env.action_dim, config.model.hidden_size)
        device = config.device
        # init networks
        self.actor = Actor(*network_args).to(device)
        self.qnet = QNetwork(*network_args).to(device)
        # init target networks
        self.actor_target = Actor(*network_args).to(device)
        self.qnet_target = QNetwork(*network_args).to(device)
        self.actor_target.load_state_dict(self.actor.state_dict())
        self.qnet_target.load_state_dict(self.qnet.state_dict())
        # optimizer
        self.actor_optimizer = Adam(self.actor.parameters(), lr=config.model.learning_rate)
        self.qnet_optimizer = Adam(self.qnet.parameters(), lr=config.model.learning_rate)

    def _random_action(self, num_CAV: int) -> Union[int, ndarray]:
        """Takes a random exploration action."""
        return np.random.randint(0, self.action_dim, (num_CAV,))

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
        obs, (num_CAV, _) = env.reset(curriculum_training=curriculum_training)
        done = False
        # NOTE: current_step is count from 0
        while not done:
            self.current_step += 1
            # ALGO LOGIC: put actions logic here
            if self.current_step > args.learning_starts:
                with torch.no_grad():
                    actions = self.actor(torch.Tensor(obs).to(device))
            else:
                actions = self._random_action(num_CAV)

            # TRY NOT TO MODIFY: execute the game and log data.
            next_obs, rewards, next_done, infos = env.step(actions)

            # Save data to reply buffer; handle `final_observation`
            self.rb.add(obs, actions, rewards, next_obs, infos["agents_dones"])

            # TRY NOT TO MODIFY: CRUCIAL step easy to overlook
            obs = next_obs
            done = next_done

            # ALGO LOGIC: training.
            if self.current_step > args.learning_starts:
                for agent_id in range(num_CAV):
                    # Sample random batch from replay buffer
                    b_obs, b_actions, b_rewards, b_next_obs, b_dones = self.rb.sample(args.batch_size)
                    with torch.no_grad():
                        next_state_actions = self.actor_target(b_next_obs)
                        qnet_next_target = self.qnet_target(b_next_obs, next_state_actions)
                        next_q_value = b_rewards.flatten() + (1 - b_dones.flatten()) * args.gamma * (qnet_next_target.view(-1))

                    qnet_a_values = self.qnet(b_obs, b_actions).view(-1)
                    qnet_loss = ((qnet_a_values - next_q_value) ** 2).mean()

                    # optimize the model
                    self.qnet_optimizer.zero_grad()
                    qnet_loss.backward()
                    self.qnet_optimizer.step()

                    if (self.current_step - begin_step) >= args.policy_frequency:
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

    def evaluate(self, env: AbstractEnv, output_dir: str, global_episode: int) \
            -> Tuple[List[List[float]], List[List[dict]]]:
        # set up variables
        device = self.config.device

        infos = []
        rewards = []
        for i, seed in enumerate(self.config.model.test_seeds):
            # set up variables
            infos_i = []
            rewards_i = []
            Recorded_frames = []

            # TRY NOT TO MODIFY: start the game
            next_obs, (num_CAV, _) = env.reset(is_training=False, testing_seeds=seed)
            next_obs = torch.Tensor(next_obs).to(device)
            next_done = torch.zeros(1).to(device)

            # TRY NOT TO MODIFY: init video recorder
            rendered_frame = env.render(mode="rgb_array")
            video_filename = os.path.join(output_dir, f"testing_episode{global_episode + 1}_{i}.mp4")
            print("Recording video to {} ({}x{}x{}@{}fps)".format(video_filename, *rendered_frame.shape, 5))

            for step in range(0, 1_000):
                # ALGO LOGIC: action logic
                with torch.no_grad():
                    # TODO: if it's so similar, might as well just overrides this to BaseModel.
                    action = self.actor(torch.Tensor(next_obs).to(device))

                # TRY NOT TO MODIFY: execute the game and log data.
                next_obs, reward, next_done, info = env.step(action.cpu().numpy())

                if video_filename is not None:
                    rendered_frame = env.render(mode="rgb_array")
                    Recorded_frames.append(rendered_frame)

                rewards_i.append(reward)
                infos_i.append(info)

                if next_done:
                    break
                next_obs, next_done = torch.Tensor(next_obs).to(device), torch.Tensor([float(next_done)]).to(device)

            # records final frame
            if video_filename is not None:
                rendered_frame = env.render(mode="rgb_array")
                Recorded_frames.append(rendered_frame)

            rewards.append(rewards_i)
            infos.append(infos_i)

            if video_filename is not None:
                imageio.mimsave(video_filename, [np.array(frame) for frame in Recorded_frames], fps=5)
                # Alternate writer:
                # writer = imageio.get_writer(video_filename, fps=5)
                # for frame in Recorded_frames:
                #     writer.append_data(np.array(frame))
                # writer.close()

        env.close()
        return rewards, infos
