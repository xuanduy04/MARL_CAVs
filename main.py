import numpy as np
import torch

import gym
import argparse
import logging
from datetime import datetime

from MARL.common.utils import init_dir
from MARL_redux.utils.train_utils import init_env
from config import import_config

from MARL_redux.model import IPPO


def train(args):
    config = import_config(args.model)
    # set seed


    # update configs
    config.device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # create an experiment folder
    now = datetime.now().strftime("%b_%d_%H_%M_%S")
    output_dir = args.base_dir + now
    dirs = init_dir(output_dir)

    # init envs
    env_train = gym.make('merge-multilane-priority-multi-agent-v0')
    env_train = init_env(config.env, env_train)

    env_eval = gym.make('merge-multilane-priority-multi-agent-v0')
    env_eval = init_env(config.env, env_eval)
    config.env.state_dim = env_train.state_dim
    config.env.action_dim = env_train.action_dim

    # init model
    on_policy = True
    if on_policy:
        config.model.num_steps = None
    policy = IPPO(config.model)

    # Training loop
    for episode in range(0, config.model.train_episodes):
        num_CAV, num_HDV = config.env.num_CAV, config.env.num_HDV
        if episode < config.model.curriculum_episodes:
            # Simulates curriculum training
            num_CAV = np.random.choice(np.arange(max(min(3, num_CAV), num_CAV - 2), num_CAV + 1), 1)[0]
            num_HDV = np.random.choice(np.arange(max(1, num_HDV - 2), num_HDV + 1), 1)[0]

        # Interacts and trains the policy
        policy.train(env_train, num_CAV, num_HDV)

        if (episode + 1) % config.model.eval_interval == 0:
            # evaluate the model
            result = policy.evaluate(env_eval, num_CAV, num_HDV)
            print(result)
        results.append(result)
        return results  #whatever they are

    policy.save_model()
    # Save the model.
    pass

def parse_args():
    default_base_dir = "./results/"
    parser = argparse.ArgumentParser(description='Train or evaluate policy on RL environment')
    parser.add_argument('--base-dir', type=str, required=False,
                        default=default_base_dir,
                        help="experiment base dir")
    parser.add_argument('--algorithm', type=str, required=False,
                        choices=['mappo', 'ippo'],
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
        import sys
        logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))
    # train or eval
    if args.option == 'train':
        train(args)
    else:
        raise NotImplementedError("evaluate pretrained model not implemented")