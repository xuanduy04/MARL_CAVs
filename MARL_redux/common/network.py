from typing import Optional

import numpy as np
import torch
from torch import nn
from torch import Tensor
from torch.distributions.categorical import Categorical


def layer_init(layer: nn.Linear, method: str = 'orthogonal', **kwargs) -> nn.Linear:
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
            layer_init(nn.Linear(hidden_size, action_dim), std=0.01),
        )

        self.critic = nn.Sequential(
            layer_init(nn.Linear(state_dim, hidden_size)),
            nn.Tanh(),
            layer_init(nn.Linear(hidden_size, hidden_size)),
            nn.Tanh(),
            layer_init(nn.Linear(hidden_size, 1), std=1.0)
        )

    def get_value(self, state: Tensor) -> Tensor:
        return self.critic(state)

    def get_action_and_value(self, state: Tensor, action: Optional[Tensor] = None):
        if torch.isnan(state).any():
          print("Input contains NaN values")
        else:
          print("Input is ok")
        nn_check_nan(self.actor)
        nn_check_nan(self.critic)
        logits = self.actor(state)
        analyze(logits)
        probs = Categorical(logits=logits)
        if action is None:
            # Sample an action
            action = probs.sample()
        return action, probs.log_prob(action), probs.entropy(), self.critic(state)

def analyze(tensor: Tensor):
    positive_count, negative_count, zero_count, nan_count = torch.sum(tensor > 0).item(), torch.sum(tensor < 0).item(), torch.sum(tensor == 0).item(), torch.sum(torch.isnan(tensor)).item()
    print(positive_count, negative_count, zero_count, nan_count)

def nn_check_nan(network: nn.Linear):
    contains_nan = False
    for name, param in network.named_parameters():
        if torch.isnan(param).any():
            contains_nan = True
            print(f"Parameter {name} contains NaNs")
        if param.grad is not None and torch.isnan(param.grad).any():
            contains_nan = True
            print(f"Gradient of {name} contains NaNs")
    if contains_nan is False:
        print(f"No parameter contains NaNs")
