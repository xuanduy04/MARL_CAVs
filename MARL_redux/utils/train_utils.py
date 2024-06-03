from typing import Dict, List, Tuple

import numpy as np
import torch

import random
import os

from highway_env.envs import AbstractEnv
from config import Config

ROUND_NDIGITS = 2


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


def get_mean_std(l: List[List[float]]) -> Tuple[float, float]:
    # l: [ [...], [...], [...] ]
    # l_i: result of each step in the i-th episode
    s = [np.sum(np.array(l_i), 0) for l_i in l]
    s_mu = np.mean(np.array(s), 0)
    s_std = np.std(np.array(s), 0)
    s_mu, s_std = round(s_mu, ROUND_NDIGITS), round(s_std, ROUND_NDIGITS)
    return s_mu, s_std


def extract_data(infos: List[List[dict]], config: Config):
    """returns (avg_steps, avg_speeds_mean, crash_rate) from infos"""
    assert len(infos) == len(config.model.test_seeds)
    avg_speeds = []
    avg_steps = 0
    crash_rate = 0.0
    for i, infos_i in enumerate(infos):
        steps = len(infos_i)
        avg_speed = []
        for info in infos_i:
            avg_speed.append(info["average_speed"])

        avg_speeds.append(avg_speed)
        avg_steps += steps
        crash_rate += (infos_i[-1]["crashed"] / infos_i[-1]["vehicle_count"])
        # TODO: maybe extract vehicle count from config directly? would that fasten the code? is it needed?

    avg_steps /= len(infos)
    avg_speeds_mean, _ = get_mean_std(avg_speeds)
    return avg_steps, avg_speeds_mean, crash_rate
