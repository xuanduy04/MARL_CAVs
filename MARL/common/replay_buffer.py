from typing import Union

import numpy as np
import random

import torch
from numpy import ndarray
from MARL.utils.debug_utils import DEBUG

class ReplayBuffer(object):
    def __init__(self, size):
        """Create Prioritized Replay buffer.
        (modified from https://github.com/openai/maddpg.git)

        Parameters
        ----------
        size: int
            Max number of transitions to store in the buffer. When the buffer
            overflows the old memories are dropped.
        """
        self._storage = []
        self._maxsize = int(size)
        self._next_idx = 0

    def __len__(self):
        return len(self._storage)

    def clear(self):
        self._storage = []
        self._next_idx = 0

    def add(self, obs: ndarray, actions: tuple, reward: float, next_obs: ndarray, dones: tuple):
        if DEBUG:
            assert obs.shape[0] == len(actions)
            assert len(actions) == len(dones)
            assert len(dones) == next_obs.shape[0]
        for i in range(obs.shape[0]):
            data = (obs[i], actions[i], reward, next_obs[i], dones[i])
            if self._next_idx >= len(self._storage):
                self._storage.append(data)
            else:
                self._storage[self._next_idx] = data
            self._next_idx = (self._next_idx + 1) % self._maxsize

    def _encode_sample(self, idxes):
        obs, actions, rewards, next_obs, dones = [], [], [], [], []
        for i in idxes:
            data = self._storage[i]
            obs_i, action, reward, next_obs_i, done = data
            obs.append(np.array(obs_i, copy=False))
            actions.append(np.array(action, copy=False))
            rewards.append(reward)
            next_obs.append(np.array(next_obs_i, copy=False))
            dones.append(done)
        return np.array(obs), np.array(actions), np.array(rewards), np.array(next_obs), np.array(dones)

    def make_index(self, batch_size):
        return [random.randint(0, len(self._storage) - 1) for _ in range(batch_size)]

    def make_latest_index(self, batch_size):
        idx = [(self._next_idx - 1 - i) % self._maxsize for i in range(batch_size)]
        np.random.shuffle(idx)
        return idx

    def sample_index(self, idxes):
        return self._encode_sample(idxes)

    def sample(self, batch_size: int, as_tensor: bool = False, device: torch.DeviceObjType = 'cpu'):
        """Sample a batch of experiences.

        Parameters
        ----------
        batch_size: int
            How many transitions to sample.
        as_tensor: bool
            should the return type be a torch.Tensor
        device:
            when `as_tensor` is True, move the output to device.

        Returns
        -------
        obs_batch: np.array
            batch of observations
        act_batch: np.array
            batch of actions executed given obs_batch
        rew_batch: np.array
            rewards received as results of executing act_batch
        next_obs_batch: np.array
            next set of observations seen after executing act_batch
        done_mask: np.array
            done_mask[i] = 1 if executing act_batch[i] resulted in
            the end of an episode and 0 otherwise.
        """
        if batch_size > 0:
            idxes = self.make_index(batch_size)
        else:
            idxes = range(0, len(self._storage))
        if as_tensor:
            obs, act, rew, next_obs, done = self._encode_sample(idxes)
            obs = torch.Tensor(obs).to(device)
            act = torch.Tensor(act).to(device)
            rew = torch.Tensor(rew).to(device)
            next_obs = torch.Tensor(next_obs).to(device)
            done = torch.Tensor(done).to(device)
            return obs, act, rew, next_obs, done
        return self._encode_sample(idxes)

