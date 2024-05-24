import torch
import torch.nn as nn

from MARL.single_agent.Memory import ReplayMemory


class BasePolicy(object):
    def __init__(self, memory: ReplayMemory = ReplayMemory()):
        super(BasePolicy, self).__init__()
        self.memory = memory

    # choose an action based on state with random noise added for exploration in training
    def exploration_action(self, state, n_agents):
        """
        Chose an action, for exploration in training.
        """
        raise NotImplementedError

    def action(self, state, n_agents):
        """
        Chose an action, directly sampled from policy.
        """
        raise NotImplementedError

    def train(self):
        raise NotImplementedError

    def load_model(self):
        raise NotImplementedError

    def save_model(self):
        raise NotImplementedError
