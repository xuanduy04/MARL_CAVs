from typing import List

import difflib

from MARL.model import *
from config import Config, import_config

model_list = {
    'maa2c': MAA2C,
    'ippo': IPPO,
    'mappo': MAPPO,
    'mappo_attention': MAPPO_attention,
    'ippo_attention': IPPO_attention,
    'ippo_attention_patience': IPPO_attention_patience,
}


def supported_models() -> List[str]:
    return list(model_list.keys())


def init_model(model_name: str, config: Config) -> BaseModel:
    if model_name not in model_list:
        raise ValueError(f'Unsupported model ({model_name}), '
                         f'supported models: {supported_models()}')
    return model_list[model_name](config)


def verify_consistancy(current_algo: str):
    """verifies consistancy between configs of models of the same base algorithm to the current alg"""
    base_algo = None
    for algo in ['ippo', 'mappo']:
        if algo in current_algo:
            base_algo = algo
    
    if base_algo is None:
        return
    
    print(f'Verifying consistancy between {current_algo} and all {base_algo} algs')
    
    configs = dict()
    for model in supported_models():
        if algo in model:
            configs[model] = import_config(model)[0]._config_dict

    model_names = list(configs.keys())
    compared_keys = set()

    reference_model = current_algo
    reference_config = configs[current_algo]

    for key in reference_config:
        if key not in compared_keys:
            for model in model_names[i+1:]:
                current_config = configs[model]
                if key in current_config:
                    if reference_config[key] != current_config[key]:
                        print(f"Difference in {algo}: {reference_model} vs {model}")
                        print(f"Parameter: {key}")
                        print(f"{reference_model}: {reference_config[key]}")
                        print(f"{model}: {current_config[key]}")
                        print("-" * 30)
            compared_keys.add(key)

    # for i in range(0, len(model_names)-1):
    #     reference_model = model_names[i]
    #     reference_config = configs[reference_model]

    #     for key in reference_config:
    #         if key not in compared_keys:
    #             for model in model_names[i+1:]:
    #                 current_config = configs[model]
    #                 if key in current_config:
    #                     if reference_config[key] != current_config[key]:
    #                         print(f"Difference in {algo}: {reference_model} vs {model}")
    #                         print(f"Parameter: {key}")
    #                         print(f"{reference_model}: {reference_config[key]}")
    #                         print(f"{model}: {current_config[key]}")
    #                         print("-" * 30)
    #             compared_keys.add(key)

    # check if config params are consistant across models with the same algo, print all differences to stdout
    # (e.g the 'learning_rate' of mappo_attention, mappo, mappo_lmao, mappo_attention2 are the same.)
    # do not report "missing" configs. 
    # (e.g. the 'attention' parameter is not present in mappo while it is on mappo_attention and mappo_attention2,
    #  ---> do not report the missing parameter of 'mappo'
    #       do compare the attention parameter of both mappo_attention and mappo_attention2 as usual)
