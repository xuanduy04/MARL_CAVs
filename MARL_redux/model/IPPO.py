"""
'Almost' single-file implementation of IPPO

based on cleanrl's PPO implementation.
"""

from typing import List, Tuple, Optional

import numpy as np
import torch
import torch.nn as nn
from torch import Tensor
from torch.optim import Adam
from torch.distributions.categorical import Categorical

import os
import math
import imageio

# for type hints
from config import Config
from highway_env.envs import AbstractEnv
# for standardization between multiple algorithms
from MARL_redux.model import BaseModel
# for quick network initialization
from MARL_redux.common.network import layer_init
# debug utilities
from MARL_redux.utils.debug_utils import checknan, checknan_Sequential, analyze, printd


class ActorCriticNetwork(nn.Module):
    """An actor critic network, similar to that of cleanrl"""
    def __init__(self, state_dim: int, action_dim: int, hidden_size: int):
        super(ActorCriticNetwork, self).__init__()
        self.actor = nn.Sequential(
            layer_init(nn.Linear(state_dim, hidden_size)),
            nn.Tanh(),
            layer_init(nn.Linear(hidden_size, hidden_size)),
            nn.Tanh(),
            layer_init(nn.Linear(hidden_size, action_dim)),
            nn.Tanh(),
        )

        self.critic = nn.Sequential(
            layer_init(nn.Linear(state_dim, hidden_size)),
            nn.Tanh(),
            layer_init(nn.Linear(hidden_size, hidden_size)),
            nn.Tanh(),
            layer_init(nn.Linear(hidden_size, 1))
        )

    def get_value(self, state: Tensor) -> Tensor:
        return self.critic(state)

    def get_action_and_value(self, state: Tensor, action: Optional[Tensor] = None):
        logits = self.actor(state)
        probs = Categorical(logits=logits)
        if action is None:
            # Sample an action
            action = probs.sample()
        return action, probs.log_prob(action), probs.entropy(), self.critic(state)


class IPPO(BaseModel):
    def __init__(self, config: Config):
        super(IPPO, self).__init__(config)
        self.config.model.batch_size = self.config.model.num_steps  # on-policy so use all data we get.
        # batch_size is "per agent", 'cause IPPO.

        # Actor & Critic
        self.network = ActorCriticNetwork(config.env.state_dim, config.env.action_dim,
                                          config.model.hidden_size)
        self.optimizer = Adam(self.network.parameters(), lr=config.model.learning_rate)

        # self.model_path = ""
        # self.actor_path = os.path.join(self.model_path, "actor.pth")
        # self.critic_path = os.path.join(self.model_path, "critic.pth")

    def train(self, env: AbstractEnv, curriculum_training: bool = False, global_episode: int = 0):
        # printd(f'Begin training for episode {global_episode + 1}')
        # set up variables
        device = self.config.device
        num_steps = self.config.model.num_steps
        args = self.config.model

        # Annealing the rate if instructed to do so.
        if args.anneal_lr:
            frac = 1.0 - (global_episode / args.train_episodes)
            lrnow = frac * args.learning_rate
            self.optimizer.param_groups[0]["lr"] = lrnow

        # TRY NOT TO MODIFY: start the game
        next_obs, (num_CAV, _) = env.reset(curriculum_training=curriculum_training)
        next_obs = torch.Tensor(next_obs).to(device)
        next_done = torch.zeros(1).to(device)

        # ALGO Logic: Storage setup
        memory_shape = (num_steps, num_CAV)
        obs = torch.zeros(memory_shape + (self.config.env.state_dim,)).to(device)
        actions = torch.zeros(memory_shape).to(device)
        logprobs = torch.zeros(memory_shape).to(device)
        rewards = torch.zeros(memory_shape).to(device)
        dones = torch.zeros(memory_shape[0]).to(device)
        values = torch.zeros(memory_shape).to(device)

        for step in range(0, num_steps):
            obs[step] = next_obs
            dones[step] = next_done

            # ALGO LOGIC: action logic
            with torch.no_grad():
                action, logprob, _, value = self.network.get_action_and_value(next_obs)
                values[step] = value.flatten()
            actions[step] = action
            logprobs[step] = logprob

            # TRY NOT TO MODIFY: execute the game and log data.
            next_obs, reward, next_done, info = env.step(action.cpu().numpy())
            rewards[step] = torch.tensor(reward / args.reward_scale).to(device).view(-1)

            next_obs, next_done = torch.Tensor(next_obs).to(device), torch.Tensor([float(next_done)]).to(device)
            if next_done:
                num_steps = step
                break

        # bootstrap value if not done
        with torch.no_grad():
            next_value = self.network.get_value(next_obs).reshape(1, -1)
            advantages = torch.zeros_like(rewards).to(device)
            lastgaelam = 0
            for t in reversed(range(num_steps)):
                if t == num_steps - 1:
                    nextnonterminal = 1.0 - next_done
                    nextvalues = next_value
                else:
                    nextnonterminal = 1.0 - dones[t + 1]
                    nextvalues = values[t + 1]
                delta = rewards[t] + args.gamma * nextvalues * nextnonterminal - values[t]
                advantages[t] = lastgaelam = delta + args.gamma * args.gae_lambda * nextnonterminal * lastgaelam
            returns = advantages + values

        batch_size = min(args.batch_size, num_steps)
        minibatch_size = math.ceil(batch_size / args.num_minibatches)
        b_inds = np.arange(batch_size)

        for agent_id in range(num_CAV):
            # flatten the agent's batch
            b_obs = obs[:, agent_id, :].reshape((-1, self.config.env.state_dim))
            b_logprobs = logprobs[:, agent_id].reshape(-1)
            b_actions = actions[:, agent_id].reshape(-1)
            b_advantages = advantages[:, agent_id].reshape(-1)
            b_returns = returns[:, agent_id].reshape(-1)
            b_values = values[:, agent_id].reshape(-1)

            # Optimizing the agent's policy and value network
            for epoch in range(args.update_epochs):
                # printd(f'Epoch {epoch}:')
                np.random.shuffle(b_inds)
                for start in range(0, batch_size, minibatch_size):
                    end = min(start + minibatch_size, batch_size)
                    mb_inds = b_inds[start:end]

                    _, newlogprob, entropy, newvalue = \
                        self.network.get_action_and_value(b_obs[mb_inds], b_actions.long()[mb_inds])
                    logratio = newlogprob - b_logprobs[mb_inds]
                    ratio = logratio.exp()

                    mb_advantages = b_advantages[mb_inds]
                    if args.norm_adv:
                        mb_advantages = (mb_advantages - mb_advantages.mean()) / (mb_advantages.std() + 1e-8)

                    # Policy loss
                    pg_loss1 = -mb_advantages * ratio
                    pg_loss2 = -mb_advantages * torch.clamp(ratio, 1 - args.clip_coef,
                                                            1 + args.clip_coef)
                    pg_loss = torch.max(pg_loss1, pg_loss2).mean()

                    # Value loss
                    newvalue = newvalue.view(-1)
                    if args.clip_vloss:
                        v_loss_unclipped = (newvalue - b_returns[mb_inds]) ** 2
                        v_clipped = b_values[mb_inds] + torch.clamp(
                            newvalue - b_values[mb_inds],
                            -args.clip_coef,
                            args.clip_coef,
                        )
                        v_loss_clipped = (v_clipped - b_returns[mb_inds]) ** 2
                        v_loss_max = torch.max(v_loss_unclipped, v_loss_clipped)
                        v_loss = 0.5 * v_loss_max.mean()
                    else:
                        v_loss = 0.5 * ((newvalue - b_returns[mb_inds]) ** 2).mean()

                    entropy_loss = entropy.mean()
                    loss = pg_loss + args.vf_coef * v_loss - args.ent_coef * entropy_loss

                    self.optimizer.zero_grad()
                    loss.backward()
                    nn.utils.clip_grad_norm_(self.network.parameters(), args.max_grad_norm)
                    self.optimizer.step()

    def evaluate(self, env: AbstractEnv, output_dir: str, global_episode: int)\
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
                    action, _, _, _ = self.network.get_action_and_value(next_obs)

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

    def save_model(self, model_dir: str, global_episode: int):
        file_path = model_dir + 'checkpoint-{:d}.pt'.format(global_episode)
        torch.save({'global_step': global_episode,
                    'model_state_dict': self.network.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict()},
                   file_path)

    def load_model(self, model_dir: str):
        pass
