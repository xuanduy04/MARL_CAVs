import random
import torch
from numpy import ndarray
from MARL.utils.debug_utils import DEBUG


class ReplayBuffer(object):
    def __init__(self, size: int, state_dim: int, num_agents: int, device: torch.DeviceObjType = 'cpu'):
        """Create Prioritized Replay buffer.
        (modified from https://github.com/openai/maddpg.git)

        Parameters
        ----------
        size: int
            Max number of transitions to store in the buffer. When the buffer
            overflows the old memories are dropped.
        state_dim: int
            Shape of the observation space.
        action_dim: int
            Shape of the action space.
        num_agents: int
            number of agents
        device: torch.DeviceObjType
            The device on which to store the tensors.
        """
        self._maxsize = int(size)
        self.device = device

        # Pre-allocate storage tensors
        memory_shape = (size, num_agents)
        self.obs = torch.zeros(memory_shape + (state_dim,)).to(device)
        self.actions = torch.zeros(memory_shape).to(device)
        self.rewards = torch.zeros((size,)).to(device)
        self.next_obs = torch.zeros(memory_shape + (state_dim,)).to(device)
        self.dones = torch.zeros(memory_shape).to(device)

        self._next_idx = 0
        self._size = 0

    def __len__(self):
        return self._size

    def clear(self):
        self._next_idx = 0
        self._size = 0

    def add(self, obs: ndarray, actions: ndarray, rewards: float, next_obs: ndarray, dones: ndarray):
        if DEBUG:
            assert obs.shape[0] == actions.shape[0] == dones.shape[0] == next_obs.shape[0]

        batch_size = obs.shape[0]
        idxs = (self._next_idx + torch.arange(batch_size)) % self._maxsize

        self.obs[idxs] = torch.tensor(obs, device=self.device)
        self.actions[idxs] = torch.tensor(actions, device=self.device)
        self.rewards[idxs] = torch.tensor(rewards, device=self.device)
        self.next_obs[idxs] = torch.tensor(next_obs, device=self.device)
        self.dones[idxs] = torch.tensor(dones, device=self.device)

        self._next_idx = (self._next_idx + batch_size) % self._maxsize
        self._size = min(self._size + batch_size, self._maxsize)

    def _encode_sample(self, idxes):
        obs = self.obs[idxes]
        actions = self.actions[idxes]
        rewards = self.rewards[idxes]
        next_obs = self.next_obs[idxes]
        dones = self.dones[idxes]
        return obs, actions, rewards, next_obs, dones

    def _make_index(self, batch_size):
        return torch.randint(0, self._size, (batch_size,), dtype=torch.long)

    def sample(self, batch_size: int):
        """Sample a batch of experiences.

        Parameters
        ----------
        batch_size: int
            How many transitions to sample.

        Returns
        -------
        obs_batch: torch.Tensor
            batch of observations
        act_batch: torch.Tensor
            batch of actions executed given obs_batch
        rew_batch: torch.Tensor
            rewards received as results of executing act_batch
        next_obs_batch: torch.Tensor
            next set of observations seen after executing act_batch
        done_mask: torch.Tensor
            done_mask[i] = 1 if executing act_batch[i] resulted in
            the end of an episode and 0 otherwise.
        """
        idxes = self._make_index(batch_size)
        return self._encode_sample(idxes)
