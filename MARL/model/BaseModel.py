"""
BaseModel implementation, for standardization and streamlined evaluation methods
"""
from typing import List, Tuple

import imageio
import numpy as np
import torch
from torch import Tensor
from torch.utils.tensorboard import SummaryWriter

import os

# for type hints
from config import Config
from highway_env.envs import AbstractEnv
# noinspection PyUnresolvedReferences
# debug utilities
from MARL.utils.debug_utils import analyze, checknan, checknan_Sequential, printd


class BaseModel(object):
    def __init__(self, config: Config):
        super(BaseModel, self).__init__()
        self.config = config

    def train(self, env: AbstractEnv, curriculum_training: bool, writer: SummaryWriter, global_episode: int):
        """Interacts with the environment and trains the model, once (i.e 1 episode)."""
        raise NotImplementedError

    def _act(self, obs: Tensor) -> np.ndarray:
        """Samples action from policy, required for self.evaluate() to work"""
        raise NotImplementedError

    def evaluate(self, env: AbstractEnv, output_dir: str, global_episode: int) \
            -> Tuple[List[List[float]], List[List[dict]]]:
        """
        Evaluates the model, returns (rewards, infos)
        Uses self._act() to sample model actions.

        MUST NOT BE OVERRIDEN IN ANY WAY
        """
        # set up variables
        device = self.config.device

        infos = []
        rewards = []
        for i, seed in enumerate(self.config.model.test_seeds):
            # set up variables
            infos_i = []
            rewards_i = []
            Recorded_frames = []

            # TRY NOT TO MODIFY: start the game
            next_obs, (num_CAV, _) = env.reset(is_training=False, testing_seeds=seed)
            next_obs = torch.Tensor(next_obs).to(device)

            # TRY NOT TO MODIFY: init video recorder
            rendered_frame = env.render(mode="rgb_array")
            video_filename = os.path.join(output_dir, f"testing_episode{global_episode + 1}_{i}.mp4")
            print("Recording video to {} ({}x{}x{}@{}fps)".format(video_filename, *rendered_frame.shape, 5))

            for step in range(0, 1_000):
                # ALGO LOGIC: action logic
                with torch.no_grad():
                    action = self._act(next_obs)

                # TRY NOT TO MODIFY: execute the game and log data.
                next_obs, reward, next_done, info = env.step(action)

                if video_filename is not None:
                    rendered_frame = env.render(mode="rgb_array")
                    Recorded_frames.append(rendered_frame)

                rewards_i.append(reward)
                infos_i.append(info)

                if next_done:
                    break
                next_obs = torch.Tensor(next_obs).to(device)

            # records final frame
            if video_filename is not None:
                rendered_frame = env.render(mode="rgb_array")
                Recorded_frames.append(rendered_frame)

            rewards.append(rewards_i)
            infos.append(infos_i)

            if video_filename is not None:
                imageio.mimsave(video_filename, [np.array(frame) for frame in Recorded_frames], fps=5)
                # Alternate writer:
                # writer = imageio.get_writer(video_filename, fps=5)
                # for frame in Recorded_frames:
                #     writer.append_data(np.array(frame))
                # writer.close()

        env.close()
        return rewards, infos

    def save_model(self, model_dir: str, global_episode: int):
        """Saves the model"""
        raise NotImplementedError

    def load_model(self, model_dir: str):
        """Loads a pre-trained model"""
        raise NotImplementedError
