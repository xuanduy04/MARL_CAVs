import torch
from torch import nn


def layer_init(layer: nn.Linear) -> nn.Linear:
    nn.init.xavier_uniform_(layer.weight)
    nn.init.zeros_(layer.bias)
    return layer


class ActorNetwork(nn.Module):
    """
    A network for actor
    """
    def __init__(self, state_dim, hidden_size, output_size, output_act):
        super(ActorNetwork, self).__init__()
        self.in_features = state_dim
        self.fc1 = layer_init(nn.Linear(state_dim, hidden_size))
        self.fc2 = layer_init(nn.Linear(hidden_size, hidden_size))
        self.fc3 = layer_init(nn.Linear(hidden_size, output_size))
        # activation function for the output
        self.output_act = output_act

    def __call__(self, state):
        out = nn.functional.relu(self.fc1(state))
        out = nn.functional.relu(self.fc2(out))
        out = self.output_act(self.fc3(out), -1) 
        # -1 for the last layer, as what is previously done.
        return out


class CriticNetwork(nn.Module):
    """
    A network for critic
    """
    def __init__(self, state_dim, action_dim, hidden_size, output_size=1):
        super(CriticNetwork, self).__init__()
        self.in_features = state_dim
        self.fc1 = layer_init(nn.Linear(state_dim, hidden_size))
        self.fc2 = layer_init(nn.Linear(hidden_size + action_dim, hidden_size))
        self.fc3 = layer_init(nn.Linear(hidden_size, output_size))

    def __call__(self, state, action):
        out = nn.functional.relu(self.fc1(state))
        out = torch.cat([out, action], 1)
        out = nn.functional.relu(self.fc2(out))
        out = self.fc3(out)
        return out


class ActorCriticNetwork(nn.Module):
    """
    An actor-critic network that shared lower-layer representations but
    have distinct output layers
    """
    def __init__(self, state_dim, action_dim, hidden_size,
                 actor_output_act, critic_output_size=1):
        super(ActorCriticNetwork, self).__init__()
        self.fc1 = nn.Linear(state_dim, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.actor_linear = nn.Linear(hidden_size, action_dim)
        self.critic_linear = nn.Linear(hidden_size, critic_output_size)
        self.actor_output_act = actor_output_act

    def __call__(self, state):
        out = nn.functional.relu(self.fc1(state))
        out = nn.functional.relu(self.fc2(out))
        act = self.actor_output_act(self.actor_linear(out))
        val = self.critic_linear(out)
        return act, val
