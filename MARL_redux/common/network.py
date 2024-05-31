import torch
from torch import nn
from torch.distributions.categorical import Categorical


def layer_init(layer: nn.Linear) -> nn.Linear:
    nn.init.xavier_uniform_(layer.weight)
    nn.init.zeros_(layer.bias)
    return layer


class ActorCriticNetwork(nn.Module):
    """An actor critic network, similar to that of cleanrl"""
    def __init__(self, state_dim: int, action_dim: int, hidden_size: int):
        super(ActorCriticNetwork, self).__init__()
        self.actor = nn.Sequential(
            layer_init(nn.Linear(state_dim, hidden_size)),
            nn.ReLU(),
            layer_init(nn.Linear(hidden_size, hidden_size)),
            nn.ReLU(),
            layer_init(nn.Linear(hidden_size, action_dim)),
            nn.LogSigmoid()
        )

        self.critic = nn.Sequential(
            layer_init(nn.Linear(state_dim, hidden_size)),
            nn.ReLU(),
            layer_init(nn.Linear(hidden_size, hidden_size)),
            nn.ReLU(),
            layer_init(nn.Linear(hidden_size, 1))
        )

    def get_value(self, state):
        return self.critic(state)

    def get_action_and_value(self, state, action=None):
        logits = self.actor(state)
        probs = Categorical(logits=logits)
        if action is None:
            # Sample an action
            action = probs.sample()
        return action, probs.log_prob(action), probs.entropy(), self.critic(state)
