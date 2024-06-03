from MARL_redux.model import *
from config import Config


def init_model(model_name: str, config: Config) -> BaseModel:
    if model_name == 'ippo':
        return IPPO(config)
    else:
        raise ValueError(f'Unsupported model ({model_name})')