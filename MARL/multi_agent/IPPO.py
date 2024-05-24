import os

from BasePolicy import BasePolicy
from torch.optim import AdamW
# from MARL.single_agent.Memory import OnPolicyReplayMemory


class IPPO(BasePolicy):
    def __init__(self, train_config: dict, env_config: dict):
        # super().__init__(memory=OnPolicyReplayMemory())
        self.train_config = train_config
        self.env_config = env_config

        # Actor & Critic
        self.actor = None
        self.critic = None
        self.actor_optimizer = AdamW(self.actor.parameters(),
                                     lr=self.train_config.actor_lr)
        self.critic_optimizer = AdamW(self.critic.parameters(),
                                      lr=self.train_config.critic_lr)

        self.model_path = ""
        self.actor_path = os.path.join(self.model_path, "actor.pth")
        self.critic_path = os.path.join(self.model_path, "critic.pth")

    def train(self):
        pass

    def evaluate(self):
        pass
