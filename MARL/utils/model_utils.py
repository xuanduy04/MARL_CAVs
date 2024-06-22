from typing import List

from MARL.model import *
from config import Config

model_list = {
    'ippo': IPPO,
    # 'maddpg': MADDPG,
    'mappo': MAPPO,
    'maa2c': MAA2C,
}


def supported_models() -> List[str]:
    return list(model_list.keys())


def init_model(model_name: str, config: Config) -> BaseModel:
    if model_name not in model_list:
        raise ValueError(f'Unsupported model ({model_name}), '
                         f'supported models: {supported_models()}')
    return model_list[model_name](config)
