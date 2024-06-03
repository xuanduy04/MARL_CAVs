import numpy as np
import torch

import random

import sys
import highway_env
import gym
import argparse
import logging
from datetime import datetime

from MARL_redux.utils.train_utils import init_env, set_seed, init_dir, extract_data
from MARL_redux.utils.model_utils import init_model
from config import import_config


def train(args):
    config = import_config(args.algorithm)
    print(f'seed: {config.seed}')
    set_seed(config.seed)

    # update configs
    config.device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # create an experiment folder
    now = datetime.now().strftime("%b_%d_%H_%M_%S")
    output_dir = args.base_dir + now
    dirs = init_dir(output_dir)

    # init envs
    env_train = gym.make('merge-multilane-priority-multi-agent-v0')
    env_train = init_env(env_train, config)

    env_eval = gym.make('merge-multilane-priority-multi-agent-v0')
    env_eval = init_env(env_eval, config)
    config.env.state_dim = env_train.state_dim
    config.env.action_dim = env_train.action_dim
    print(f'Env has {config.env.num_CAV} CAVs, {config.env.num_HDV} HDVs and 1 PV')

    # init model
    model = init_model(model_name=args.algorithm, config=config)
    print(f'Training {args.algorithm} model\n')

    # Training loop
    results = []
    for episode in range(0, config.model.train_episodes):
        # Model interacts with env and trains
        model.train(env_train, curriculum_training=episode < config.model.curriculum_episodes, global_episode=episode)

        if (episode + 1) % config.model.eval_interval == 0:
            # evaluate the model
            eval_result, eval_infos = model.evaluate(env_eval, output_dir, global_episode=episode)
            print(f"Episode {episode + 1}:\n{eval_result}")
            print(extract_data(eval_infos, config), sep='\n', end='\n\n')

            results.append(eval_result)
            # Save the model.
            model.save_model(dirs['models'], global_episode=episode)
    return results


def parse_args():
    default_base_dir = "./results/"
    implemented_algorithm_list = ['ippo']

    parser = argparse.ArgumentParser(description='Train or evaluate policy on RL environment')
    parser.add_argument('--base-dir', type=str, required=False,
                        default=default_base_dir,
                        help="experiment base dir")
    parser.add_argument('--algorithm', type=str, required=False,
                        choices=implemented_algorithm_list,
                        default='ippo',
                        help='which algorithm to use (in all lowercase)')
    parser.add_argument('--option', type=str, required=False,
                        choices=['train', 'evaluate'],
                        default='train',
                        help="whether to train or evaluate")
    parser.add_argument('--log-to-stdout', type=bool, required=False,
                        default=True)
    # parser.add_argument('--model-dir', type=str, required=False,
    #                     default='',
    #                     help="pretrained model path")
    # parser.add_argument('--evaluation-seeds', type=str, required=False,
    #                     default=','.join([str(i) for i in range(0, 600, 20)]),
    #                     help="random seeds for evaluation, split by ,")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    logging.getLogger().setLevel(logging.INFO)
    if args.log_to_stdout:
        logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))
    # train or eval
    if args.option == 'train':
        train(args)
    else:
        raise NotImplementedError("evaluate pretrained model not implemented")
