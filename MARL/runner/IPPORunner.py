import os
import imageio

import numpy as np

from MARL.model import IPPO
from config import Config


class IPPORunner(object):
    def __init__(self, model: IPPO, config: Config, env_train, env_eval):
        self.config = config
        self.model = model
        self.env_train = env_train
        self.env_eval = env_eval

        self.current_episode = 0
        self.episodes = self.config.model.episodes

        self.reward_scale = self.config.model.reward_scale

    def collect_exp(self):
        """
        Policy interacts with environment to collect experience
        """
        if self.current_episode >= self.episodes:
            return

        # if self.current_episode < self.curriculum_episodes:

        env = self.env_train
        curr_state, _ = env.reset()

        n_agents = len(env.controlled_vehicles)

        states = []
        actions = []
        rewards = []
        average_speed = 0
        done = False

        while not done:
            states.append(curr_state)
            action = self.model.exploration_action(curr_state, n_agents)
            next_state, global_reward, done, info = env.step(tuple(action))
            actions.append([index_to_one_hot(a, self.action_dim) for a in action])
            reward = [global_reward] * n_agents
            rewards.append(reward)
            average_speed += info["average_speed"]
            curr_state = next_state

        # reward scaling
        if self.reward_scale > 0:
            rewards = np.array(rewards) / self.reward_scale

        # discount reward
        final_value = [0.0] * n_agents
        for agent_id in range(n_agents):
            rewards[:, agent_id] = self.model.discount_reward(rewards[:, agent_id],
                                                              final_value[agent_id])

        rewards = rewards.tolist()
        self.model.memory.push(states, actions, rewards)
        self.current_episode += 1

    def evaluate(self, output_dir: str):
        """
        Evaluates the policy.
        """
        env = self.env_eval
        rewards = []
        # infos = []
        vehicle_speed = []
        vehicle_position = []
        steps = []
        avg_speeds = []
        crash_rate = 0.0
        seeds = self.config['test_seeds']

        for i, seed in enumerate(seeds):
            rewards_i = []
            # infos_i = []
            step = 0
            avg_speed = 0
            Recorded_frames = []
            done = False
            state, action_mask = env.reset(is_training=False, testing_seeds=seed)

            n_agents = len(env.controlled_vehicles)
            rendered_frame = env.render(mode="rgb_array")
            video_filename = os.path.join(output_dir,
                                          f"testing_episode{self.current_episode + 1}_{i}.mp4")
            # Init video recording
            if video_filename is not None:
                print("Recording video to {} ({}x{}x{}@{}fps)".format(video_filename,
                                                                      *rendered_frame.shape,
                                                                      5))
            while not done:
                step += 1
                action = self.model.action(state, n_agents)
                state, reward, done, info = env.step(action)
                avg_speed += info["average_speed"]

                if video_filename is not None:
                    rendered_frame = env.render(mode="rgb_array")
                    Recorded_frames.append(rendered_frame)

                rewards_i.append(reward)
                # infos_i.append(info)

            if video_filename is not None:
                rendered_frame = env.render(mode="rgb_array")
                Recorded_frames.append(rendered_frame)

            rewards.append(rewards_i)
            # infos.append(infos_i)
            vehicle_speed.append(info["vehicle_speed"])
            vehicle_position.append(info["vehicle_position"])
            steps.append(step)
            avg_speeds.append(avg_speed / step)
            crash_rate += info["crashed"] / n_agents

            if video_filename is not None:
                imageio.mimsave(video_filename, [np.array(frame) for frame in Recorded_frames],
                                fps=5)

        env.close()
        crash_rate /= eval_episodes
        return rewards, (vehicle_speed, vehicle_position), steps, avg_speeds, crash_rate
