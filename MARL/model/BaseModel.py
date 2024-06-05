from typing import List, Tuple

from config import Config
from highway_env.envs import AbstractEnv


class BaseModel(object):
    def __init__(self, config: Config):
        super(BaseModel, self).__init__()
        self.config = config

    def train(self, env: AbstractEnv, curriculum_training: bool = False, global_episode: int = 0):
        """Interacts with the environment and trains the model, once (i.e 1 episode)."""
        raise NotImplementedError

    def evaluate(self, env: AbstractEnv, output_dir: str, global_episode: int)\
            -> Tuple[List[List[float]], List[List[dict]]]:
        """Evaluates the model, returns (rewards, infos)"""
        raise NotImplementedError

    def save_model(self, model_dir: str, global_episode: int):
        """Saves the model"""
        raise NotImplementedError

    def load_model(self, model_dir: str):
        """Loads a pre-trained model"""
        raise NotImplementedError
