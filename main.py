import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter

import sys
import highway_env
import gym
import argparse
import logging
import warnings
from datetime import datetime
from shutil import copy
# noinspection PyUnresolvedReferences
from tqdm.auto import tqdm

from MARL.utils.train_utils import DEFAULT_BASE_DIR
from MARL.utils.train_utils import init_env, set_seed, init_dir, extract_data, reward_mean_std, obs_feature_type_to_list
from MARL.utils.model_utils import init_model, supported_models, verify_consistancy
from config import import_config

warnings.filterwarnings("ignore", message="Please also save or load the state of the optimizer when saving or loading the scheduler.")


def train(args):
    verify_consistancy(args.algorithm)
    config, config_files = import_config(args.algorithm)
    # update missing configs
    config.device = 'cuda' if torch.cuda.is_available() else 'cpu'
    try:
        config.model.curriculum_episodes = 0
    except Exception:
        pass
    try:
        config.env.N = 6 # auto set to 6 for this ablation
    except Exception:
        pass

    assert config.model.curriculum_episodes == 0
    assert config.env.N >= 1

    # create an experiment folder
    N_ = f"_N{config.env.N}" if config.env.N != 6 else ""
    Xtype_ = f"_Xtype{config.env.obs_feature_type}" if config.env.obs_feature_type != 0 else ""
    eps_ = f"_eps{config.model.train_episodes}" if config.model.train_episodes != 1000 else ""
    pte_ = f"_p{config.model.patience}" if 'patience' in config.model else ""
    try:
        drop_ = f"_d{config.model.attention.dropout_p}" if config.model.attention.dropout_p != 0.3 else ""
    except Exception:
        drop_ = ""
    try:
        warmup_ = f"_w{config.model.warmup_steps}" if config.model.warmup_steps > 0 else ""
    except Exception:
        warmup_ = ""
    run_date = datetime.now().strftime("%b_%d_%H_%M_%S")

    env_name = f'({config.env.num_CAV},{config.env.num_HDV}){eps_}'
    alg_name = f'{args.algorithm}{drop_}{pte_}{warmup_}' + N_ + Xtype_
    run_name = f'{env_name}-{alg_name}-{config.seed}-{run_date}'
    output_dir = args.base_dir + run_name
    dirs = init_dir(output_dir)
    for file in config_files:
        copy(file, dirs["configs"])

    writer = SummaryWriter(dirs["runs"] + run_name)
    writer.add_text(
        "hyperparameters",
        "|param|value|\n|-|-|\n%s" % ("\n".join([f"|{key}|{value}|" for key, value in vars(config).items()])),
    )
    set_seed(config.seed)

    print(f'Device = {config.device}')
    print(f'Seed = {config.seed}')
    print(f'Trains for {config.model.train_episodes} episodes')

    # init envs
    env_train = gym.make('merge-multilane-priority-multi-agent-v0')
    env_train = init_env(env_train, config, is_eval_env=False)

    env_eval = gym.make('merge-multilane-priority-multi-agent-v0')
    env_eval = init_env(env_eval, config, is_eval_env=True)

    config.env.state_dim = config.env.N * len(obs_feature_type_to_list(config.env.obs_feature_type))
    config.env.action_dim = env_train.action_dim
    if 'attention' in config.model:
        config.model.attention.seq_len = config.env.N
        config.model.attention.d_model = len(obs_feature_type_to_list(config.env.obs_feature_type))
        config.model.attention.num_heads = config.model.attention.d_model // 2
        assert config.model.attention.num_heads * 2 == config.model.attention.d_model, \
            "obs feature list should be divisible by 2"

    print(f'Env has {config.env.num_CAV} CAVs, {config.env.num_HDV} HDVs and 1 PV')

    if config.env.N != 6:
        print(f'Testing with N={config.env.N}, state_dim becomes {config.env.state_dim}')
    if config.env.obs_feature_type != 0:
        obs_feature_list = obs_feature_type_to_list(config.env.obs_feature_type)
        print(f'Testing with features type {config.env.obs_feature_type}: {obs_feature_list}')
        if 'attention' in config.model:
            print(f'Num_heads={config.model.attention.num_heads}')

    # init model
    model = init_model(model_name=args.algorithm, config=config)
    print(f'Run name: {run_name}')
    print(f'Begin training {args.algorithm} model\n')

    # Training loop
    results = []
    avg_steps = []
    avg_speeds = []
    crash_rates = []
    for episode in tqdm(range(0, config.model.train_episodes)):
        # Model interacts with env, and trains (when valid)
        model.train(env_train, curriculum_training=episode < config.model.curriculum_episodes, writer=writer, global_episode=episode)

        if (episode + 1) % config.model.eval_interval == 0:
            # evaluate the model
            eval_rewards, eval_infos = model.evaluate(env_eval, dirs['train_videos'], global_episode=episode)

            # Save & log results
            eval_rewards_mean, _ = reward_mean_std(eval_rewards)
            avg_step, avg_speed_mean, crash_rate = extract_data(eval_infos, config)
            results.append(eval_rewards_mean)
            avg_steps.append(avg_step)
            avg_speeds.append(avg_speed_mean)
            crash_rates.append(crash_rate)
            writer.add_scalar("charts/reward", eval_rewards_mean, episode)
            writer.add_scalar("charts/length", avg_step, episode)
            writer.add_scalar("charts/speed", avg_speed_mean, episode)
            writer.add_scalar("charts/crash_rate", crash_rate, episode)

            print("Episode %d, Average Reward %.2f, Average Speed %.2f, Crash rate %.2f"
                  % (episode + 1, eval_rewards_mean, avg_speed_mean, crash_rate))
            print("Average rewards:", results)
            print("  Average steps:", avg_steps)
            print(" Average speeds:", avg_speeds)
            print("    Crash rates:", crash_rates, end='\n\n')

            # Save the model.
            model.save_model(dirs['models'], global_episode=episode)

    print("Average rewards:", results,
          "Average steps:", avg_steps,
          "Average speeds:", avg_speeds,
          "Crash rates:", crash_rates,
          "Output_dir:", output_dir,
          "Run date:", run_date,
          sep='\n')

    writer.close()
    env_train.close()
    env_eval.close()


def parse_args():
    parser = argparse.ArgumentParser(description='Train or evaluate policy on '
                                     'a merging-ramp environment with a priority vehicle')
    parser.add_argument('--base-dir', type=str, required=False,
                        default=DEFAULT_BASE_DIR,
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
