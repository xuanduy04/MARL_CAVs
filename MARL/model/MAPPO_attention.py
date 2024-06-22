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
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.distributions.categorical import Categorical
from torch.utils.tensorboard import SummaryWriter

import math
import time

# for type hints
from config import Config
from highway_env.envs import AbstractEnv
# for standardization between multiple algorithms
from MARL.model import BaseModel
# for quick network initialization
from MARL.common.network import layer_init
# attention module
from MARL.common.Attention_Feed_Forward import Encoder

# noinspection PyUnresolvedReferences
# debug utilities
from MARL.utils.debug_utils import checknan, checknan_Sequential, analyze, printd


class ActorCriticNetwork(nn.Module):
    """An actor critic network, similar to that of cleanrl"""
    def __init__(self, state_dim: int, action_dim: int, hidden_size: int,
                 d_model: int, num_heads: int, dropout_p: int):
        super(ActorCriticNetwork, self).__init__()
        self.encoder = Encoder(
            d_model=d_model,
            num_heads=num_heads,
            dropout_p=dropout_p,
            state_dim=state_dim
        )

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
        state, attn = self.encoder(state)
        return self.critic(state)

    def get_action_and_value(self, state: Tensor, action: Optional[Tensor] = None):
        state, attn = self.encoder(state)
        logits = self.actor(state)
        probs = Categorical(logits=logits)
        if action is None:
            # Sample an action
            action = probs.sample()
        return action, probs.log_prob(action), probs.entropy(), self.critic(state)


# noinspection PyUnusedLocal
class MAPPO_attention(BaseModel):
    def __init__(self, config: Config):
        super(MAPPO_attention, self).__init__(config)
        config_attention = config.model.attention
        self.seq_len = config_attention.seq_len
        self.d_model = config_attention.d_model
        assert self.seq_len * self.d_model == config.env.state_dim, \
        f'seq_len * d_model != state_dim, {self.seq_len} * {self.d_model} != {config.env.state_dim}'
        assert self.d_model % config_attention.num_heads == 0, \
        f'd_model % num_heads = {self.d_model} * {config_attention.num_heads} != 0'

        # Actor & Critic
        self.network = ActorCriticNetwork(
            config.env.state_dim, config.env.action_dim, config.model.hidden_size,
            config_attention.d_model, config_attention.num_heads, config_attention.dropout_p,
        ).to(config.device)
        self.optimizer = Adam(self.network.parameters(), lr=config.model.learning_rate, weight_decay=config.model.weight_decay)
        # TODO: compare & contrast w/ normal annealing
        self.scheduler = ReduceLROnPlateau(self.optimizer, 
            patience=200,
            factor=0.5,
            min_lr=config.model.learning_rate / 10_000,
            verbose=True
        )

    def train(self, env: AbstractEnv, curriculum_training: bool, writer: SummaryWriter, global_episode: int):
        # printd(f'Begin training for episode {global_episode + 1}')
        # set up variables
        start_time = time.time()
        device = self.config.device
        num_steps = self.config.model.num_steps
        args = self.config.model
        # variables for logging
        overall_losses = []
        v_losses = []
        pg_losses = []
        entropy_losses = []
        old_approx_kls = []
        approx_kls = []

        # Annealing the rate if instructed to do so.
        # if args.anneal_lr:
        #     frac = 1.0 - (global_episode / args.train_episodes)
        #     lrnow = frac * args.learning_rate
        #     self.optimizer.param_groups[0]["lr"] = lrnow

        # TRY NOT TO MODIFY: start the game
        next_obs, (num_CAV, _) = env.reset(curriculum_training=curriculum_training)
        next_obs = torch.Tensor(next_obs).to(device)
        next_done = torch.zeros(1).to(device)

        # ALGO Logic: Storage setup
        memory_shape = (num_steps, num_CAV)
        obs = torch.zeros(memory_shape + (self.seq_len, self.d_model, )).to(device)
        actions = torch.zeros(memory_shape).to(device)
        logprobs = torch.zeros(memory_shape).to(device)
        rewards = torch.zeros(memory_shape).to(device)
        dones = torch.zeros(memory_shape).to(device)
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

            next_obs, next_done = torch.Tensor(next_obs).to(device), float(next_done)
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

        batch_size = num_steps * num_CAV # for all agents
        minibatch_size = math.ceil(batch_size / args.num_minibatches)
        b_inds = np.arange(batch_size)

        # flatten the batch
        b_obs = obs.reshape((-1, self.seq_len, self.d_model))
        b_logprobs = logprobs.reshape(-1)
        b_actions = actions.reshape(-1)
        b_advantages = advantages.reshape(-1)
        b_returns = returns.reshape(-1)
        b_values = values.reshape(-1)

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

                with torch.no_grad():
                    # calculate approx_kl http://joschu.net/blog/kl-approx.html
                    old_approx_kl = (-logratio).mean()
                    approx_kl = ((ratio - 1) - logratio).mean()

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

            overall_losses.append(loss.item())
            v_losses.append(v_loss.item())
            pg_losses.append(pg_loss.item())
            entropy_losses.append(entropy_loss.item())
            old_approx_kls.append(old_approx_kl.item())
            approx_kls.append(approx_kl.item())

        clipped_losses = [np.clip(loss, -args.max_grad_norm, args.max_grad_norm) \
            for loss in overall_losses]
        self.scheduler.step(np.asarray(clipped_losses).mean())

        # TODO: DEBUG THIS, TEST IF WRITER ACTUALLY WORKS
        # TRY NOT TO MODIFY: record rewards for plotting purposes
        writer.add_scalar("charts/learning_rate", self.optimizer.param_groups[0]["lr"], global_episode)
        writer.add_scalar("losses/overall_loss", np.asarray(overall_losses).mean(), global_episode)
        writer.add_scalar("losses/clipped_overall_loss", np.asarray(clipped_losses).mean(), global_episode)
        writer.add_scalar("losses/value_loss", np.asarray(v_losses).mean(), global_episode)
        writer.add_scalar("losses/policy_loss", np.asarray(pg_losses).mean(), global_episode)
        writer.add_scalar("losses/entropy", np.asarray(entropy_losses).mean(), global_episode)
        writer.add_scalar("losses/old_approx_kl", np.asarray(old_approx_kls).mean(), global_episode)
        writer.add_scalar("losses/approx_kl", np.asarray(approx_kls).mean(), global_episode)
        # writer.add_scalar("losses/clipfrac", np.mean(clipfracs), global_episode)
        # writer.add_scalar("losses/explained_variance", explained_var, global_episode)
        writer.add_scalar("charts/train_episode_per_sec", int(global_episode / (time.time() - start_time)), global_episode)

    def _act(self, obs: Tensor) -> np.ndarray:
        action, _, _, _ = self.network.get_action_and_value(obs)
        return action.cpu().numpy()
