import numpy as np
import torch
import torch.nn as nn
from torch.optim import Adam

import os
import imageio

from config import Config
from highway_env.envs import AbstractEnv
from MARL_redux.common.network import ActorCriticNetwork


class IPPO(object):
    def __init__(self, config: Config):
        self.config = config

        # Actor & Critic
        self.network = ActorCriticNetwork(config.env.state_dim, config.env.action_dim,
                                          config.model.hidden_size)
        self.optimizer = Adam(self.network.parameters(), lr=config.learning_rate)

        # self.model_path = ""
        # self.actor_path = os.path.join(self.model_path, "actor.pth")
        # self.critic_path = os.path.join(self.model_path, "critic.pth")

    def train(self, env: AbstractEnv, curriculum_training: bool = False):
        """
        Interacts with the environment and trains the model, once (i.e 1 episode).
        """
        # set up variables
        device = self.config.device
        rollout_steps = self.config.model.rollout_steps
        args = self.config.model

        # Annealing the rate if instructed to do so.
        # if args.anneal_lr:
        #     frac = 1.0 - (iteration - 1.0) / args.num_iterations
        #     lrnow = frac * args.learning_rate
        #     self.optimizer.param_groups[0]["lr"] = lrnow

        # TRY NOT TO MODIFY: start the game
        next_obs, (num_CAV, _) = env.reset(curriculum_training)
        next_obs = torch.Tensor(next_obs).to(device)
        next_done = torch.zeros(num_CAV).to(device)

        # ALGO Logic: Storage setup
        memory_shape = (rollout_steps, num_CAV)
        obs = torch.zeros(memory_shape + self.config.env.state_dim).to(device)
        actions = torch.zeros(memory_shape + self.config.env.action_dim).to(device)
        logprobs = torch.zeros(memory_shape).to(device)
        rewards = torch.zeros(memory_shape).to(device)
        dones = torch.zeros(memory_shape).to(device)
        values = torch.zeros(memory_shape).to(device)

        for step in range(0, rollout_steps):
            obs[step] = next_obs
            dones[step] = next_done

            # ALGO LOGIC: action logic
            with torch.no_grad():
                action, logprob, _, value = self.network.get_action_and_value(next_obs)
                values[step] = value.flatten()
            actions[step] = action
            logprobs[step] = logprob

            # TRY NOT TO MODIFY: execute the game and log data.
            next_obs, reward, next_done, infos = env.step(action.cpu().numpy())
            rewards[step] = torch.tensor(reward).to(device).view(-1)

            if next_done:
                rollout_steps = step
                break
            next_obs, next_done = torch.Tensor(next_obs).to(device), torch.Tensor(next_done).to(device)

            # bootstrap value if not done
        with torch.no_grad():
            next_value = self.network.get_value(next_obs).reshape(1, -1)
            advantages = torch.zeros_like(rewards).to(device)
            lastgaelam = 0
            for t in reversed(range(rollout_steps)):
                if t == rollout_steps - 1:
                    nextnonterminal = 1.0 - next_done
                    nextvalues = next_value
                else:
                    nextnonterminal = 1.0 - dones[t + 1]
                    nextvalues = values[t + 1]
                delta = rewards[t] + args.gamma * nextvalues * nextnonterminal - values[t]
                advantages[t] = lastgaelam = delta + args.gamma * args.gae_lambda * nextnonterminal * lastgaelam
            returns = advantages + values

            # flatten the batch
        b_obs = obs.reshape((-1,) + self.config.env.state_dim)
        b_logprobs = logprobs.reshape(-1)
        b_actions = actions.reshape((-1,) + self.config.env.action_dim)
        b_advantages = advantages.reshape(-1)
        b_returns = returns.reshape(-1)
        b_values = values.reshape(-1)

        # Optimizing the policy and value network
        batch_size = min(args.batch_size, rollout_steps)
        minibatch_size = min(args.minibatch_size, batch_size)
        b_inds = np.arange(batch_size)
        clipfracs = []
        for epoch in range(args.update_epochs):
            np.random.shuffle(b_inds)
            for start in range(0, batch_size, minibatch_size):
                end = start + minibatch_size
                mb_inds = b_inds[start:end]

                _, newlogprob, entropy, newvalue = \
                    self.network.get_action_and_value(b_obs[mb_inds], b_actions.long()[mb_inds])
                logratio = newlogprob - b_logprobs[mb_inds]
                ratio = logratio.exp()

                with torch.no_grad():
                    # calculate approx_kl http://joschu.net/blog/kl-approx.html
                    # old_approx_kl = (-logratio).mean()
                    # approx_kl = ((ratio - 1) - logratio).mean()
                    clipfracs += [((ratio - 1.0).abs() > args.clip_coef).float().mean().item()]

                mb_advantages = b_advantages[mb_inds]
                if args.norm_adv:
                    mb_advantages = (mb_advantages - mb_advantages.mean()) / (mb_advantages.std() + 1e-8)

                # Policy loss
                pg_loss1 = -mb_advantages * ratio
                pg_loss2 = -mb_advantages * torch.clamp(ratio, 1 - args.clip_coef,
                                                        1 + args.clip_coef)
                pg_loss = torch.max(pg_loss1, pg_loss2).mean()

                # Value loss
                newvalue = newvalue.view(-1)
                if args.clip_vloss:
                    # TODO: change clip value loss to the one in IPPO.
                    v_loss_unclipped = (newvalue - b_returns[mb_inds]) ** 2
                    v_clipped = b_values[mb_inds] + torch.clamp(
                        newvalue - b_values[mb_inds],
                        -args.clip_coef,
                        args.clip_coef,
                    )
                    v_loss_clipped = (v_clipped - b_returns[mb_inds]) ** 2
                    v_loss_max = torch.max(v_loss_unclipped, v_loss_clipped)
                    v_loss = 0.5 * v_loss_max.mean()
                else:
                    v_loss = 0.5 * ((newvalue - b_returns[mb_inds]) ** 2).mean()

                entropy_loss = entropy.mean()
                loss = pg_loss - args.ent_coef * entropy_loss + v_loss * args.vf_coef

                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.network.parameters(), args.max_grad_norm)
                self.optimizer.step()

    def evaluate(self, env: AbstractEnv, output_dir: str, global_episode: int):
        # set up variables
        rewards = []
        vehicle_speed = []
        vehicle_position = []
        steps = []
        avg_speeds = []
        crash_rate = 0.0

        device = self.config.device
        args = self.config.model

        for i, seed in enumerate(self.config.model.test_seeds):
            # set up variables
            rewards_i = []
            step = 0
            avg_speed = 0
            Recorded_frames = []
            done = False

            # TRY NOT TO MODIFY: start the game
            next_obs, (num_CAV, _) = env.reset(is_training=False, testing_seeds=seed)
            next_obs = torch.Tensor(next_obs).to(device)
            next_done = torch.zeros(num_CAV).to(device)

            # TRY NOT TO MODIFY: init video recorder
            rendered_frame = env.render(mode="rgb_array")
            video_filename = os.path.join(output_dir, f"testing_episode{global_episode + 1}_{i}.mp4")
            # Init video recording
            if video_filename is not None:
                print("Recording video to {} ({}x{}x{}@{}fps)".format(video_filename,
                                                                      *rendered_frame.shape,
                                                                      5))

            for step in range(0, 10_000):
                # ALGO LOGIC: action logic
                with torch.no_grad():
                    action, _, _, _ = self.network.get_action_and_value(next_obs)

                # TRY NOT TO MODIFY: execute the game and log data.
                next_obs, reward, next_done, infos = env.step(action.cpu().numpy())

                if video_filename is not None:
                    rendered_frame = env.render(mode="rgb_array")
                    Recorded_frames.append(rendered_frame)

                avg_speed += infos["average_speed"]
                rewards_i.append(reward)
                # infos_i.append(infos)

                if next_done:
                    break
                next_obs, next_done = torch.Tensor(next_obs).to(device), torch.Tensor(next_done).to(device)

            # records final frame
            if video_filename is not None:
                rendered_frame = env.render(mode="rgb_array")
                Recorded_frames.append(rendered_frame)

            rewards.append(rewards_i)
            # infos.append(infos_i)
            vehicle_speed.append(infos["vehicle_speed"])
            vehicle_position.append(infos["vehicle_position"])
            steps.append(step)
            avg_speeds.append(avg_speed / step)
            crash_rate += infos["crashed"] / num_CAV

            if video_filename is not None:
                imageio.mimsave(video_filename, [np.array(frame) for frame in Recorded_frames],
                                fps=5)
                # Alternate writer.
                # writer = imageio.get_writer(video_filename, fps=5)
                # for frame in Recorded_frames:
                #     writer.append_data(np.array(frame))
                # writer.close()

        env.close()
        crash_rate /= len(self.config.model.test_seeds)
        return rewards, ((vehicle_speed, vehicle_position), steps, avg_speeds, crash_rate)
    
    def save_model():
        pass

    def load_model():
        pass
