from typing import Dict, List, Tuple

import numpy as np
import torch

import random
import os

from highway_env.envs import AbstractEnv
from config import Config

# Number of digits after the comma to round all data
ROUND_NDIGITS = 2


def init_env(env: AbstractEnv, config: Config, is_eval_env: bool = False) -> AbstractEnv:
    env.config['seed'] = config.seed + int(is_eval_env)
    econfig = config.env
    env.config['simulation_frequency'] = econfig.simulation_frequency
    env.config['policy_frequency'] = econfig.policy_frequency
    env.config['duration'] = econfig.duration

    env.config['COLLISION_COST'] = econfig.COLLISION_COST
    env.config['HIGH_SPEED_REWARD'] = econfig.HIGH_SPEED_REWARD
    env.config['HEADWAY_COST'] = econfig.HEADWAY_COST
    env.config['HEADWAY_TIME'] = econfig.HEADWAY_TIME
    env.config['MERGING_LANE_COST'] = econfig.MERGING_LANE_COST
    env.config['PRIORITY_LANE_COST'] = econfig.PRIORITY_LANE_COST
    env.config['LANE_CHANGE_COST'] = econfig.LANE_CHANGE_COST

    env.config['num_CAV'] = econfig.num_CAV
    env.config['num_HDV'] = econfig.num_HDV

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
    paths = ['train_videos', 'configs', 'models', 'eval_videos', 'eval_logs', 'runs']
    if not os.path.exists(base_dir):
        os.mkdir(base_dir)
    dirs = {}
    for path in paths:
        cur_dir = base_dir + '/%s/' % path
        if not os.path.exists(cur_dir):
            os.mkdir(cur_dir)
        dirs[path] = cur_dir
    return dirs


def reward_mean_std(rewards: List[List[float]]) -> Tuple[float, float]:
    # l: [ [...], [...], [...] ]
    # l_i: result of each step in the i-th episode
    s = [np.sum(np.array(rewards_i), 0) for rewards_i in rewards]
    s_mu = np.mean(np.array(s), 0)
    s_std = np.std(np.array(s), 0)
    s_mu, s_std = round(s_mu, ROUND_NDIGITS), round(s_std, ROUND_NDIGITS)
    return s_mu, s_std


def extract_data(infos: List[List[dict]], config: Config):
    """returns (avg_step, avg_speed_mean, crash_rate) from infos"""
    assert len(infos) == len(config.model.test_seeds)

    avg_step = 0.0
    avg_speed_mean = 0.0
    crash_rate = 0.0
    for i, infos_i in enumerate(infos):
        steps = len(infos_i)
        avg_speed_i = 0.0
        for info in infos_i:
            avg_speed_i += info["average_speed"]

        avg_step += steps
        avg_speed_mean += (avg_speed_i / steps)
        crash_rate += (infos_i[-1]["crashed"] / infos_i[-1]["vehicle_count"])
        # TODO: maybe extract vehicle count from config directly? would that fasten the code? is it needed?

    avg_step = round(avg_step / len(infos), ROUND_NDIGITS)
    avg_speed_mean = round(avg_speed_mean / len(infos), ROUND_NDIGITS)
    crash_rate = round(crash_rate / len(infos), ROUND_NDIGITS)
    return avg_step, avg_speed_mean, crash_rate
