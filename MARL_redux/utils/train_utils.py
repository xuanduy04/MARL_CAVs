from typing import Dict, List

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


def init_dir(base_dir: str) -> Dict[str, str]:
    paths = ['train_videos', 'configs', 'models', 'eval_videos', 'eval_logs']
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

def extract_data(infos: List[List[dict]], config: Config):
    assert len(infos) == len(config.model.test_seeds)
    avg_speeds = []
    avg_steps = 0
    crash_rate = 0.0
    for i, infos_i in enumerate(infos):
        steps = len(infos_i)
        avg_speed = 0.0
        for info in infos_i:
            avg_speed += info["average_speed"]

        avg_speeds.append(avg_speed / steps)
        avg_steps += steps
        crash_rate += (infos[-1]["crashed"] / infos[-1]["vehicles_count"])
        # TODO: maybe extract vehicle count from config directly? would that fasten the code? is it needed?
    
    avg_steps /= len(infos)
    return avg_steps, avg_speeds, crash_rate
