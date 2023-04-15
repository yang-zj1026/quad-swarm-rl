import random
from collections import deque
from copy import deepcopy

import gym
import numpy as np
import torch

from swarm_rl.env_wrappers.quad_experience_replay import ReplayBuffer, ExperienceReplayWrapper
from swarm_rl.env_wrappers.rl_utils import gae_advantage, score_distribution, staleness_distribution


class CurriculumReplayBufferEvent:
    def __init__(self, env, obs, score, ep_counter):
        self.env = env
        self.obs = obs
        self.score = score
        self.last_replayed = ep_counter
        self.num_replayed = 0


class CurriculumReplayBuffer:
    def __init__(self, buffer_size=100, beta=0.1, rho=0.1):
        self.buffer_idx = 0
        self.buffer = deque([], maxlen=buffer_size)

        self.beta = beta  # temperature in computing score distribution
        self.rho = rho  # staleness coefficient in mixing two distributions

    def write_cp_to_buffer(self, env, obs, score, ep_counter):
        """
        A collision was found, and we want to load the corresponding checkpoint from X seconds ago into the buffer to be sampled later on
        """
        env.saved_in_replay_buffer = True

        # For example, replace the item with the lowest number of collisions in the last 10 replays
        evt = CurriculumReplayBufferEvent(env, obs, score, ep_counter)
        if len(self.buffer) < self.buffer.maxlen:
            self.buffer.append(evt)
        else:
            self.buffer[self.buffer_idx] = evt

        print(f"Added new collision event to buffer at {self.buffer_idx}")
        self.buffer_idx = (self.buffer_idx + 1) % self.buffer.maxlen

    def sample_event(self, episode_counter):
        """
        Sample an event according to the mixture of score distribution and staleness distribution.
        See https://arxiv.org/pdf/2010.03934.pdf for more details
        """
        p_score = self.get_score_distribution()
        p_staleness = self.get_staleness_distribution(episode_counter)
        p_replay = (1 - self.rho) * p_score + self.rho * p_staleness

        # Sample events based on the mixture of two distributions
        idx = np.random.choice(range(len(self.buffer)), p=p_replay)
        # idx = random.randint(0, len(self.buffer) - 1)
        print(f'Replaying event at idx {idx}')
        self.buffer[idx].num_replayed += 1
        return self.buffer[idx], idx

    def cleanup(self):
        new_buffer = deque([], maxlen=self.buffer.maxlen)
        for event in self.buffer:
            if event.num_replayed < 10:
                new_buffer.append(event)

        self.buffer = new_buffer

    def avg_num_replayed(self):
        replayed_stats = [e.num_replayed for e in self.buffer]
        if not replayed_stats:
            return 0
        return np.mean(replayed_stats)

    def update_event(self, idx, score, episode_counter):
        """
        Update the score of given event
        """
        event = self.buffer[idx]
        event.score = score
        event.last_replayed = episode_counter

    def get_score_distribution(self):
        """
        Calculate the probabilities based on scores
        """
        scores = np.array([evt.score for evt in self.buffer])
        scores_dist = score_distribution(scores, self.beta)
        return scores_dist

    def get_staleness_distribution(self, episode_counter):
        last_replayed_eps = np.array([evt.last_replayed for evt in self.buffer])
        staleness_dist = staleness_distribution(last_replayed_eps, episode_counter)
        return staleness_dist

    def avg_score(self):
        scores = [event.score for event in self.buffer]
        if not scores:
            return 0
        return np.mean(scores)

    def __len__(self):
        return len(self.buffer)


class CurriculumReplayWrapper(ExperienceReplayWrapper):
    def __init__(self, env, gamma, gae_lambda, replay_buffer_sample_prob=0.0):
        super().__init__(env, replay_buffer_sample_prob)
        self.replay_buffer = ReplayBuffer(env.envs[0].control_freq)
        self.replay_buffer_sample_prob = replay_buffer_sample_prob
        self.replay_buffer = CurriculumReplayBuffer()
        self.curr_evt_idx = 0

        self.gamma = gamma
        self.gae_lambda = gae_lambda

        self.rewards = []
        self.value_preds = []
        self.dones = []

        self.init_obs = None
        self.saved_in_buffer = False

        # variables for tensorboard
        self.replayed_events = 0
        self.episode_counter = 0

    def reset(self):
        """Do the default reset and save the first obs"""
        obs = self.env.reset()
        self.init_obs = obs
        return obs

    def step(self, action, values_pred=None):
        obs, rewards, dones, infos = self.env.step(action)
        if values_pred:
            self.rewards.append(rewards)
            self.value_preds.append(values_pred)
            self.dones.append(dones)

        if any(dones):
            if self.saved_in_buffer:
                # Calculate score with the given score function
                ep_rewards = torch.tensor(self.rewards)
                ep_dones = torch.tensor(self.dones).float()

                # since we cannot get the last value from sample factory, set it to be same as the previous one
                self.value_preds.append(self.value_preds[-1])
                value_preds = torch.tensor(self.value_preds)
                valids = torch.ones_like(value_preds).float()

                gae = gae_advantage(ep_rewards, ep_dones, value_preds, valids, self.gamma, self.gae_lambda)
                score = torch.mean(gae)

                if not self.env.saved_in_replay_buffer:
                    # Append this trajectory to the replay buffer
                    self.replay_buffer.write_cp_to_buffer(self.env, self.init_obs, score.item(), self.episode_counter)
                else:
                    # Update the event's score
                    self.replay_buffer.update_event(self.curr_evt_idx, score.item(), self.episode_counter)

            # Reset values, rewards and dones
            self.rewards = []
            self.value_preds = []
            self.dones = []

            # Sample from replay buffer
            self.init_obs = self.new_episode()
            avg_num_replayed = self.replay_buffer.avg_num_replayed()
            for i in range(len(infos)):
                if not infos[i]["episode_extra_stats"]:
                    infos[i]["episode_extra_stats"] = dict()

                tag = "replay"
                infos[i]["episode_extra_stats"].update({
                    f"{tag}/replay_rate": self.replayed_events / self.episode_counter,
                    f"{tag}/new_episode_rate": (self.episode_counter - self.replayed_events) / self.episode_counter,
                    f"{tag}/replay_buffer_size": len(self.replay_buffer),
                    f"{tag}/avg_replayed": self.replay_buffer.avg_num_replayed(),
                    f"{tag}/avg_replayed_score": self.replay_buffer.avg_score(),
                })

            if avg_num_replayed > 10:
                self.replay_buffer.cleanup()

        else:

            collision_flag = self.env.last_step_unique_collisions.any()
            if self.env.use_obstacles:
                collision_flag = collision_flag or len(self.env.curr_quad_col) > 0

            if collision_flag and self.env.use_replay_buffer and self.env.activate_replay_buffer and not self.saved_in_buffer:
                self.save_to_buffer = True

        return obs, rewards, dones, infos

    def new_episode(self):
        """
        Normally this would go into reset(), but MultiQuadEnv is a multi-agent env that automatically resets.
        This means that reset() is never actually called externally and we need to take care of starting our new episode.
        """
        self.episode_counter += 1
        self.last_tick_added_to_buffer = -1e9
        self.episode_checkpoints = deque([], maxlen=self.max_episode_checkpoints_to_keep)

        if np.random.uniform(0, 1) < self.replay_buffer_sample_prob and self.replay_buffer and self.env.activate_replay_buffer \
                and len(self.replay_buffer) > 0:
            self.replayed_events += 1
            event, self.curr_evt_idx = self.replay_buffer.sample_event(self.episode_counter)
            env = event.env
            obs = event.obs
            replayed_env = deepcopy(env)
            replayed_env.scene = self.env.scene

            # we want to use these for tensorboard, so reset them to zero to get accurate stats
            replayed_env.collisions_per_episode = replayed_env.collisions_after_settle = 0
            replayed_env.obst_quad_collisions_per_episode = replayed_env.obst_quad_collisions_after_settle = 0
            self.env = replayed_env
            self.saved_in_buffer = True

            return obs
        else:
            obs = self.env.reset()
            self.env.saved_in_replay_buffer = False
            self.saved_in_buffer = False
            return obs
