import numpy as np
import torch

import sys
import highway_env
import gym
import argparse
import logging
import warnings
from datetime import datetime

from MARL.utils.train_utils import init_env, set_seed, init_dir, extract_data, reward_mean_std
from MARL.utils.model_utils import init_model, supported_models
from config import import_config

warnings.simplefilter("ignore")


def train(args):
    config = import_config(args.algorithm)
    set_seed(config.seed)

    # update configs
    config.device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f'Device = {config.device}')
    print(f'Seed = {config.seed}')

    # create an experiment folder
    now = datetime.now().strftime("%b_%d_%H_%M_%S")
    output_dir = args.base_dir + f'{args.algorithm}-{config.seed}-{now}'
    dirs = init_dir(output_dir)

    # init envs
    env_train = gym.make('merge-multilane-priority-multi-agent-v0')
    env_train = init_env(env_train, config, is_eval_env=False)

    env_eval = gym.make('merge-multilane-priority-multi-agent-v0')
    env_eval = init_env(env_eval, config, is_eval_env=True)

    config.env.state_dim = env_train.state_dim
    config.env.action_dim = env_train.action_dim
    print(f'Env has {config.env.num_CAV} CAVs, {config.env.num_HDV} HDVs and 1 PV')

    # init model
    model = init_model(model_name=args.algorithm, config=config)
    print(f'Training {args.algorithm} model\n')

    # Training loop
    results = []
    avg_steps = []
    avg_speeds = []
    crash_rates = []
    for episode in range(0, config.model.train_episodes):
        # Model interacts with env, and trains (when valid)
        model.train(env_train, curriculum_training=episode < config.model.curriculum_episodes, global_episode=episode)

        if (episode + 1) % config.model.eval_interval == 0:
            # evaluate the model
            eval_rewards, eval_infos = model.evaluate(env_eval, dirs['train_videos'], global_episode=episode)

            # Saves & logs results
            eval_rewards_mean, _ = reward_mean_std(eval_rewards)
            avg_step, avg_speed_mean, crash_rate = extract_data(eval_infos, config)
            results.append(eval_rewards_mean)
            avg_steps.append(avg_step)
            avg_speeds.append(avg_speed_mean)
            crash_rates.append(crash_rate)

            print("Episode %d, Average Reward %.2f, Average Speed %.2f, Crash rate %.2f"
                  % (episode + 1, eval_rewards_mean, avg_speed_mean, crash_rate))
            print("Average rewards:", results)
            print("  Average steps:", avg_steps)
            print(" Average speeds:", avg_speeds)
            print("    Crash rates:", crash_rates, end='\n\n')

            # Save the model.
            # model.save_model(dirs['models'], global_episode=episode)

    print("Average rewards:", results,
          "Average steps:", avg_steps,
          "Average speeds:", avg_speeds,
          "Crash rates", crash_rates,
          "Output_dir:", output_dir,
          sep='\n')


def parse_args():
    default_base_dir = "./results/"

    parser = argparse.ArgumentParser(description='Train or evaluate policy on RL environment')
    parser.add_argument('--base-dir', type=str, required=False,
                        default=default_base_dir,
                        help="experiment base dir")
    parser.add_argument('--algorithm', type=str, required=False,
                        choices=supported_models(),
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
