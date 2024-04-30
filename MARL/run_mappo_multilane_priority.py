from typing import Union

from MAPPO import MAPPO
from MAPPO_attention import MAPPO_attention
from common.utils import agg_double_list, copy_file_ppo, init_dir
import sys
sys.path.append("../highway-env")

import gym
import numpy as np
import matplotlib.pyplot as plt
import highway_env
import argparse
import configparser
import os
from datetime import datetime


def create_model(config, env) -> Union[MAPPO, MAPPO_attention]:
    # model configs
    BATCH_SIZE = config.getint('MODEL_CONFIG', 'BATCH_SIZE')
    MEMORY_CAPACITY = config.getint('MODEL_CONFIG', 'MEMORY_CAPACITY')
    ROLL_OUT_N_STEPS = config.getint('MODEL_CONFIG', 'ROLL_OUT_N_STEPS')
    reward_gamma = config.getfloat('MODEL_CONFIG', 'reward_gamma')
    actor_hidden_size = config.getint('MODEL_CONFIG', 'actor_hidden_size')
    critic_hidden_size = config.getint('MODEL_CONFIG', 'critic_hidden_size')
    MAX_GRAD_NORM = config.getfloat('MODEL_CONFIG', 'MAX_GRAD_NORM')
    ENTROPY_REG = config.getfloat('MODEL_CONFIG', 'ENTROPY_REG')
    reward_type = config.get('MODEL_CONFIG', 'reward_type')
    TARGET_UPDATE_STEPS = config.getint('MODEL_CONFIG', 'TARGET_UPDATE_STEPS')
    TARGET_TAU = config.getfloat('MODEL_CONFIG', 'TARGET_TAU')

    use_xavier_initialization = config.getboolean('MODEL_CONFIG','use_xavier_initialization')
    use_attention_module = config.getboolean('MODEL_CONFIG','use_attention_module')

    # train configs
    actor_lr = config.getfloat('TRAIN_CONFIG', 'actor_lr')
    critic_lr = config.getfloat('TRAIN_CONFIG', 'critic_lr')
    EPISODES_BEFORE_TRAIN = config.getint('TRAIN_CONFIG', 'EPISODES_BEFORE_TRAIN')
    reward_scale = config.getfloat('TRAIN_CONFIG', 'reward_scale')
    

    state_dim = env.n_s
    action_dim = env.n_a
    test_seeds = args.evaluation_seeds

    if use_attention_module:
        d_model = env.n_obs_features
        num_heads = config.getint('MODEL_CONFIG','num_heads')
        dropout_p = config.getfloat('MODEL_CONFIG','dropout_p')
        return MAPPO_attention(env=env, memory_capacity=MEMORY_CAPACITY,
                    d_model=d_model, num_heads=num_heads, dropout_p=dropout_p, 
                    use_xavier_initialization=use_xavier_initialization,
                    state_dim=state_dim, action_dim=action_dim,
                    batch_size=BATCH_SIZE, entropy_reg=ENTROPY_REG,
                    roll_out_n_steps=ROLL_OUT_N_STEPS,
                    actor_hidden_size=actor_hidden_size, critic_hidden_size=critic_hidden_size,
                    actor_lr=actor_lr, critic_lr=critic_lr, reward_scale=reward_scale,
                    target_update_steps=TARGET_UPDATE_STEPS, target_tau=TARGET_TAU,
                    reward_gamma=reward_gamma, reward_type=reward_type,
                    max_grad_norm=MAX_GRAD_NORM, test_seeds=test_seeds,
                    episodes_before_train=EPISODES_BEFORE_TRAIN
                    )
    else:
        return MAPPO(env=env, memory_capacity=MEMORY_CAPACITY,
                    use_xavier_initialization=use_xavier_initialization,
                    state_dim=state_dim, action_dim=action_dim,
                    batch_size=BATCH_SIZE, entropy_reg=ENTROPY_REG,
                    roll_out_n_steps=ROLL_OUT_N_STEPS,
                    actor_hidden_size=actor_hidden_size, critic_hidden_size=critic_hidden_size,
                    actor_lr=actor_lr, critic_lr=critic_lr, reward_scale=reward_scale,
                    target_update_steps=TARGET_UPDATE_STEPS, target_tau=TARGET_TAU,
                    reward_gamma=reward_gamma, reward_type=reward_type,
                    max_grad_norm=MAX_GRAD_NORM, test_seeds=test_seeds,
                    episodes_before_train=EPISODES_BEFORE_TRAIN
                    )


def init_env(config, env):
    env.config['seed'] = config.getint('ENV_CONFIG', 'seed')
    env.config['simulation_frequency'] = config.getint('ENV_CONFIG', 'simulation_frequency')
    env.config['duration'] = config.getint('ENV_CONFIG', 'duration')
    env.config['policy_frequency'] = config.getint('ENV_CONFIG', 'policy_frequency')
    env.config['COLLISION_COST'] = config.getfloat('ENV_CONFIG', 'COLLISION_COST')
    env.config['HIGH_SPEED_REWARD'] = config.getfloat('ENV_CONFIG', 'HIGH_SPEED_REWARD')
    env.config['PRIORITY_SPEED_COST'] = config.getfloat('ENV_CONFIG', 'PRIORITY_SPEED_COST')
    env.config['HEADWAY_COST'] = config.getfloat('ENV_CONFIG', 'HEADWAY_COST')
    env.config['HEADWAY_TIME'] = config.getfloat('ENV_CONFIG', 'HEADWAY_TIME')
    env.config['MERGING_LANE_COST'] = config.getfloat('ENV_CONFIG', 'MERGING_LANE_COST')
    env.config['PRIORITY_LANE_COST'] = config.getfloat('ENV_CONFIG', 'PRIORITY_LANE_COST')
    env.config['LANE_CHANGE_COST'] = config.getfloat('ENV_CONFIG', 'LANE_CHANGE_COST')
    env.config['action_masking'] = config.getboolean('MODEL_CONFIG', 'action_masking')
    env.config['num_CAV'] = config.getint('ENV_CONFIG', 'num_CAV')
    env.config['num_HDV'] = config.getint('ENV_CONFIG', 'num_HDV')
    
    use_attention_module = config.getboolean('MODEL_CONFIG','use_attention_module')
    env.config['flatten_obs'] = not use_attention_module
    return env


def parse_args():
    """
    Description for this experiment:
        + easy: globalR
        + seed = 0
    """
    default_base_dir = "./results/"
    default_config_dir = 'configs/configs_ppo.ini'
    parser = argparse.ArgumentParser(description=('Train or evaluate policy on RL environment '
                                                  'using mappo'))
    parser.add_argument('--base-dir', type=str, required=False,
                        default=default_base_dir, help="experiment base dir")
    parser.add_argument('--option', type=str, required=False,
                        default='train', help="train or evaluate")
    parser.add_argument('--config-dir', type=str, required=False,
                        default=default_config_dir, help="experiment config path")
    parser.add_argument('--model-dir', type=str, required=False,
                        default='', help="pretrained model path")
    parser.add_argument('--evaluation-seeds', type=str, required=False,
                        default=','.join([str(i) for i in range(0, 600, 20)]),
                        help="random seeds for evaluation, split by ,")
    args = parser.parse_args()
    return args


def train(args):
    base_dir = args.base_dir
    config_dir = args.config_dir
    config = configparser.ConfigParser()
    config.read(config_dir)

    # create an experiment folder
    now = datetime.now().strftime("%b_%d_%H_%M_%S_multilane_priority")
    output_dir = base_dir + now
    dirs = init_dir(output_dir)
    copy_file_ppo(dirs['configs'])

    if os.path.exists(args.model_dir):
        model_dir = args.model_dir
    else:
        model_dir = dirs['models']

    # train configs
    MAX_EPISODES = config.getint('TRAIN_CONFIG', 'MAX_EPISODES')
    EPISODES_BEFORE_TRAIN = config.getint('TRAIN_CONFIG', 'EPISODES_BEFORE_TRAIN')
    EVAL_INTERVAL = config.getint('TRAIN_CONFIG', 'EVAL_INTERVAL')
    EVAL_EPISODES = config.getint('TRAIN_CONFIG', 'EVAL_EPISODES')

    # init env
    env = gym.make('merge-multilane-priority-multi-agent-v0')
    env = init_env(config=config, env=env)

    # ROLL_OUT_N_STEPS = config.getint('MODEL_CONFIG', 'ROLL_OUT_N_STEPS')
    # assert env.T % ROLL_OUT_N_STEPS == 0

    env_eval = gym.make('merge-multilane-priority-multi-agent-v0')
    env_eval = init_env(config=config, env=env_eval)
    env_eval.config['seed'] = config.getint('ENV_CONFIG', 'seed') + 1

    num_CAV = config.getint('ENV_CONFIG', 'num_CAV')
    num_HDV = config.getint('ENV_CONFIG', 'num_HDV')

    # initialize model
    mappo = create_model(config=config, env=env)
    
    # print comments.
    use_xavier_initialization = config.getboolean('MODEL_CONFIG','use_xavier_initialization')
    model_type_name = "MAPPO" + (" with attention" if isinstance(mappo, MAPPO_attention) else "")
    init_method = "using xavier_uniform_" if use_xavier_initialization else "randomly"

    print(f"Environment initialized with {num_CAV} CAVs, {num_HDV} HDVs and 1 PV.")
    print(f"{model_type_name} model initialized {init_method}.\n")

    # load the model if exist
    mappo.load(model_dir, train_mode=True)
    env.seed = env.config['seed']
    env.unwrapped.seed = env.config['seed']
    eval_rewards = []
    avg_speeds = []
    crash_rates = []
    evaluated_episodes = []

    while mappo.n_episodes < MAX_EPISODES:
        mappo.interact()
        if mappo.n_episodes >= EPISODES_BEFORE_TRAIN:
            mappo.train()
        if mappo.episode_done and ((mappo.n_episodes + 1) % EVAL_INTERVAL == 0):
            rewards, _, _, avg_speed, crash_rate = \
                mappo.evaluation(env_eval, dirs['train_videos'], EVAL_EPISODES)
            rewards_mu, rewards_std = agg_double_list(rewards)
            avg_speed_mu, avg_speed_std = agg_double_list(avg_speed)
            eval_rewards.append(round(rewards_mu,2))
            avg_speeds.append(round(avg_speed_mu,2))
            crash_rates.append(round(crash_rate,2))
            evaluated_episodes.append(mappo.n_episodes + 1)
            # save the model
            mappo.save(dirs['models'], mappo.n_episodes + 1)
            # Outputs:
            print("Episode %d, Average Reward %.2f, Average Speed %.2f, Crash rate %.2f" \
                  % (mappo.n_episodes + 1, rewards_mu, avg_speed_mu, crash_rate))
            print("Average rewards:", eval_rewards)
            print( "Average speeds:", avg_speeds)
            print(    "Crash rates:", crash_rates)

    # save the model
    mappo.save(dirs['models'], MAX_EPISODES + 2)

    plt.figure()
    plt.plot(eval_rewards)
    plt.xlabel("Episode")
    plt.ylabel("Average Reward")
    plt.legend([model_type_name])
    plt.show()
    print("Evaluated episodes:", evaluated_episodes,
          "Average rewards:", eval_rewards,
          "Average speeds:", avg_speeds,
          "Crash rate:", round(np.mean(np.array(crash_rates)),2),
          "Output_dir:", output_dir,
          sep='\n')


def evaluate(args):
    if os.path.exists(args.model_dir):
        model_dir = args.model_dir + '/models/'
    else:
        raise Exception("Sorry, no pretrained models")
    config_dir = args.model_dir + '/configs/configs_ppo.ini'
    config = configparser.ConfigParser()
    config.read(config_dir)

    video_dir = args.model_dir + '/eval_videos'

    # init env
    env = gym.make('merge-multilane-priority-multi-agent-v0')
    env = init_env(config=config, env=env)

    ROLL_OUT_N_STEPS = config.getint('MODEL_CONFIG', 'ROLL_OUT_N_STEPS')
    assert env.T % ROLL_OUT_N_STEPS == 0

    test_seeds = args.evaluation_seeds
    seeds = [int(s) for s in test_seeds.split(',')]

    mappo = create_model(args, env)

    # load the model if exist
    mappo.load(model_dir, train_mode=False)
    rewards, _, steps, avg_speeds, _ = mappo.evaluation(env, video_dir, len(seeds), is_train=False)


if __name__ == "__main__":
    args = parse_args()
    # train or eval
    if args.option == 'train':
        train(args)
    else:
        evaluate(args)
