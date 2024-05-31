import numpy as np
import torch

import random


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
