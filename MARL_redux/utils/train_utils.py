from typing import List

import numpy as np
import torch

import random
import os

from highway_env.envs import AbstractEnv
from config import Config


def init_env(env: AbstractEnv, config: Config) -> AbstractEnv:
    env.config['flatten_obs'] = 'use_attention_module' not in config.model
    return env


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def init_dir(base_dir: str, paths: List[str] = ['train_videos', 'configs', 'models', 'eval_videos', 'eval_logs']) -> dict:
    if not os.path.exists("./results/"):
        os.mkdir("./results/")
    if not os.path.exists(base_dir):
        os.mkdir(base_dir)
    dirs = {}
    for path in paths:
        cur_dir = base_dir + '/%s/' % path
        if not os.path.exists(cur_dir):
            os.mkdir(cur_dir)
        dirs[path] = cur_dir
    return dirs
