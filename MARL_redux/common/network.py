from typing import Optional

import numpy as np
import torch
from torch import nn
from torch import Tensor
from torch.distributions.categorical import Categorical

from MARL_redux.utils.debug_utils import checknan, checknan_Sequential, analyze


def layer_init(layer: nn.Linear, method: str = 'xavier', **kwargs) -> nn.Linear:
    if method == 'xavier':
        nn.init.xavier_uniform_(layer.weight)
    elif method == 'orthogonal':
        std = kwargs.get('std', np.sqrt(2))
        nn.init.orthogonal_(layer.weight, std)
    else:
        raise ValueError("Invalid initialization method")
    nn.init.zeros_(layer.bias)
    return layer


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
        # if checknan(Input=state, print_when_false=True):
        #     print(state)
        #     exit(0)
        # checknan_Sequential(self.actor)
        # checknan_Sequential(self.critic)
        logits = self.actor(state)
        # analyze(logits)
        probs = Categorical(logits=logits)
        if action is None:
            # Sample an action
            action = probs.sample()
        return action, probs.log_prob(action), probs.entropy(), self.critic(state)
