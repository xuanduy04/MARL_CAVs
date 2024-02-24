import torch
from torch import nn
import configparser
import imageio

config_dir = 'configs/configs_ppo.ini'
config = configparser.ConfigParser()
config.read(config_dir)
torch_seed = config.getint('MODEL_CONFIG', 'torch_seed')
torch.manual_seed(torch_seed)
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True

from torch.optim import Adam, RMSprop

from MAPPO import MAPPO

import numpy as np
import os, logging
from copy import deepcopy
from single_agent.Attention_Feed_Forward import Attention_Feed_Forward
from single_agent.Attention import MultiHeadAttention
from common.utils import index_to_one_hot, to_tensor_var


class MAPPO_attention(MAPPO):
    """
    A multi-agent learned with PPO
    
    Input states are passed through an attention layer before going through the actor/ critic.
    """
    def __init__(self,
                 env, state_dim, action_dim,
                 num_heads, dropout_p,
                 memory_capacity=10000, max_steps=None,
                 roll_out_n_steps=1, target_tau=1.,
                 target_update_steps=5, clip_param=0.2,
                 reward_gamma=0.99, reward_scale=20,
                 actor_hidden_size=128, critic_hidden_size=128,
                 actor_output_act=nn.functional.log_softmax, critic_loss="mse",
                 actor_lr=0.0001, critic_lr=0.0001, test_seeds=0,
                 optimizer_type="rmsprop", entropy_reg=0.01,
                 max_grad_norm=0.5, batch_size=100, episodes_before_train=100,
                 use_cuda=True, traffic_density=1, reward_type="global_R"):

        super().__init__(env, state_dim, action_dim,
                 memory_capacity, max_steps,
                 roll_out_n_steps, target_tau,
                 target_update_steps, clip_param,
                 reward_gamma, reward_scale,
                 actor_hidden_size, critic_hidden_size,
                 actor_output_act, critic_loss,
                 actor_lr, critic_lr, test_seeds,
                 optimizer_type, entropy_reg,
                 max_grad_norm, batch_size, episodes_before_train,
                 use_cuda, traffic_density, reward_type)
        
        self.attention = MultiHeadAttention(d_model=state_dim, num_heads=num_heads, dropout_p=dropout_p)
        self.actor  = Attention_Feed_Forward(self.attention, self.actor)
        self.critic = Attention_Feed_Forward(self.attention, self.critic)

        # to ensure target network and learning network has the same weights
        self.actor_target = deepcopy(self.actor)
        self.critic_target = deepcopy(self.critic)

        if self.optimizer_type == "adam":
            self.actor_optimizer = Adam(self.actor.parameters(), lr=self.actor_lr)
            self.critic_optimizer = Adam(self.critic.parameters(), lr=self.critic_lr)
        elif self.optimizer_type == "rmsprop":
            self.actor_optimizer = RMSprop(self.actor.parameters(), lr=self.actor_lr)
            self.critic_optimizer = RMSprop(self.critic.parameters(), lr=self.critic_lr)

        if self.use_cuda:
            self.actor.cuda()
            self.critic.cuda()
            self.actor_target.cuda()
            self.critic_target.cuda()

    # train on a roll out batch
    def train(self):
        if self.n_episodes <= self.episodes_before_train:
            pass

        batch = self.memory.sample(self.batch_size)
        states_var = to_tensor_var(batch.states, self.use_cuda).view(-1, self.n_agents, self.state_dim)
        actions_var = to_tensor_var(batch.actions, self.use_cuda).view(-1, self.n_agents, self.action_dim)
        rewards_var = to_tensor_var(batch.rewards, self.use_cuda).view(-1, self.n_agents, 1)

        for agent_id in range(self.n_agents):
            # update actor network
            self.actor_optimizer.zero_grad()
            values, attn = self.critic_target(states_var[:, agent_id, :], actions_var[:, agent_id, :]).detach()
            advantages = rewards_var[:, agent_id, :] - values

            action_log_probs, attn = self.actor(states_var[:, agent_id, :])
            action_log_probs = torch.sum(action_log_probs * actions_var[:, agent_id, :], 1)
            old_action_log_probs, attn = self.actor_target(states_var[:, agent_id, :]).detach()
            old_action_log_probs = torch.sum(old_action_log_probs * actions_var[:, agent_id, :], 1)
            ratio = torch.exp(action_log_probs - old_action_log_probs)
            surr1 = ratio * advantages
            surr2 = torch.clamp(ratio, 1.0 - self.clip_param, 1.0 + self.clip_param) * advantages
            # PPO's pessimistic surrogate (L^CLIP)
            actor_loss = -torch.mean(torch.min(surr1, surr2))
            actor_loss.backward()
            if self.max_grad_norm is not None:
                nn.utils.clip_grad_norm_(self.actor.parameters(), self.max_grad_norm)
            self.actor_optimizer.step()

            # update critic network
            self.critic_optimizer.zero_grad()
            target_values = rewards_var[:, agent_id, :]
            values, attn = self.critic(states_var[:, agent_id, :], actions_var[:, agent_id, :])
            if self.critic_loss == "huber":
                critic_loss = nn.functional.smooth_l1_loss(values, target_values)
            else:
                critic_loss = nn.MSELoss()(values, target_values)
            critic_loss.backward()
            if self.max_grad_norm is not None:
                nn.utils.clip_grad_norm_(self.critic.parameters(), self.max_grad_norm)
            self.critic_optimizer.step()

        # update actor target network and critic target network
        if self.n_episodes % self.target_update_steps == 0 and self.n_episodes > 0:
            self._soft_update_target(self.actor_target, self.actor)
            self._soft_update_target(self.critic_target, self.critic)

    # predict softmax action based on state
    def _softmax_action(self, state, n_agents):
        state_var = to_tensor_var([state], self.use_cuda)

        softmax_action = []
        for agent_id in range(n_agents):
            softmax_action_var, attn = torch.exp(self.actor(state_var[:, agent_id, :]))

            if self.use_cuda:
                softmax_action.append(softmax_action_var.data.cpu().numpy()[0])
            else:
                softmax_action.append(softmax_action_var.data.numpy()[0])
        return softmax_action

    # evaluate value for a state-action pair
    def value(self, state, action):
        state_var = to_tensor_var([state], self.use_cuda)
        action = index_to_one_hot(action, self.action_dim)
        action_var = to_tensor_var([action], self.use_cuda)

        values = [0] * self.n_agents
        for agent_id in range(self.n_agents):
            value_var, attn = self.critic(state_var[:, agent_id, :], action_var[:, agent_id, :])

            if self.use_cuda:
                values[agent_id] = value_var.data.cpu().numpy()[0]
            else:
                values[agent_id] = value_var.data.numpy()[0]
        return values
