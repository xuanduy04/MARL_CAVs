from typing import List

from MARL.model import *
from config import Config

model_list = {
    'ippo': IPPO,
    'maddpg': MADDPG,
}


def supported_models() -> List[str]:
    return list(model_list.keys())


def init_model(model_name: str, config: Config) -> BaseModel:
    if model_name not in model_list:
        raise ValueError(f'Unsupported model ({model_name})')
    return model_list[model_name](config)
