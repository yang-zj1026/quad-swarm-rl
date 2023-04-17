import random
from collections import deque
from copy import deepcopy

import gym
import numpy as np
import torch

from swarm_rl.env_wrappers.rl_utils import gae_advantage, score_distribution, staleness_distribution


class ReplayBufferEvent:
    def __init__(self, env, obs):
        self.env = env
        self.obs = obs
        self.num_replayed = 0
        self.score = None
        self.last_played_episode = None
        self.env_tick = None


class ReplayBuffer:
    def __init__(self, control_frequency, cp_step_size=0.5, buffer_size=20):
        self.control_frequency = control_frequency
        self.cp_step_size_sec = cp_step_size  # how often (seconds) a checkpoint is saved
        self.cp_step_size_freq = self.cp_step_size_sec * self.control_frequency
        self.buffer_idx = 0
        self.buffer = deque([], maxlen=buffer_size)

        self.beta = 0.1  # temperature in computing score distribution
        self.rho = 0.1  # staleness coefficient in mixing two distributions

    def write_cp_to_buffer(self, env, obs):
        """
        A collision was found and we want to load the corresponding checkpoint from X seconds ago into the buffer to be sampled later on
        """
        env.saved_in_replay_buffer = True

        # For example, replace the item with the lowest number of collisions in the last 10 replays
        evt = ReplayBufferEvent(env, obs)
        if len(self.buffer) < self.buffer.maxlen:
            self.buffer.append(evt)
        else:
            self.buffer[self.buffer_idx] = evt
        print(f"Added new collision event to buffer at idx {self.buffer_idx}")
        curr_event_idx = self.buffer_idx
        self.buffer_idx = (self.buffer_idx + 1) % self.buffer.maxlen
        return curr_event_idx

    def sample_event(self, use_curriculum=False, episode_counter=None):
        """
        Sample an event to replay
        """
        if use_curriculum:
            if not episode_counter:
                raise ValueError("Episode counter cannot be none when sampling based on PLR")

            # Sample events based on the mixture of two distributions
            p_score = self.get_score_distribution()
            p_staleness = self.get_staleness_distribution(episode_counter)
            p_replay = (1 - self.rho) * p_score + self.rho * p_staleness

            idx = np.random.choice(range(len(self.buffer)), p=p_replay)
        else:
            idx = random.randint(0, len(self.buffer) - 1)

        print(f'Replaying event at idx {idx}')
        self.buffer[idx].num_replayed += 1
        return self.buffer[idx], idx

    def cleanup(self):
        new_buffer = deque([], maxlen=self.buffer.maxlen)
        for event in self.buffer:
            if event.num_replayed < 10:
                new_buffer.append(event)

        self.buffer = new_buffer
        if len(new_buffer) == self.buffer.maxlen:
            self.buffer_idx = self.buffer.maxlen - 1
        else:
            self.buffer_idx = len(new_buffer)

    def avg_num_replayed(self):
        replayed_stats = [e.num_replayed for e in self.buffer]
        if not replayed_stats:
            return 0
        return np.mean(replayed_stats)

    def update_event(self, idx, score, episode_counter, env_tick):
        """
        Update the score of given event
        """
        if not score:
            raise ValueError("Score cannot be None")
        event = self.buffer[idx]
        event.score = score
        event.last_played_episode = episode_counter
        event.env_tick = env_tick

    def get_score_distribution(self):
        """
        Calculate the probabilities based on scores
        """
        scores = np.array([evt.score for evt in self.buffer])
        # if None in scores:
        #     raise ValueError("Episode score should not be None")
        scores_dist = score_distribution(scores, self.beta)
        return scores_dist

    def get_staleness_distribution(self, episode_counter):
        last_replayed_eps = np.array([evt.last_played_episode for evt in self.buffer])
        staleness_dist = staleness_distribution(last_replayed_eps, episode_counter)
        return staleness_dist

    def __len__(self):
        return len(self.buffer)


class ExperienceReplayWrapper(gym.Wrapper):
    def __init__(self, env, replay_buffer_sample_prob=0.0, use_curriculum=False, gamma=None, gae_lambda=None):
        super().__init__(env)
        self.replay_buffer = ReplayBuffer(env.envs[0].control_freq)
        self.replay_buffer_sample_prob = replay_buffer_sample_prob

        self.max_episode_checkpoints_to_keep = int(3.0 / self.replay_buffer.cp_step_size_sec)  # keep only checkpoints from the last 3 seconds
        self.episode_checkpoints = deque([], maxlen=self.max_episode_checkpoints_to_keep)

        self.save_time_before_collision_sec = 1.5
        self.last_tick_added_to_buffer = -1e9

        # variables for tensorboard
        self.replayed_events = 0
        self.episode_counter = 0

        self.use_curriculum = use_curriculum
        if self.use_curriculum:
            # store rewards, value predictions and env ticks to calculate GAE
            self.rewards = []
            self.values_pred = []
            self.dones = []
            self.curr_event_ids = []
            self.event_ticks = []
            self.saved_in_buffer = False  # True if new events are saved to replay buffer
            self.gamma = gamma
            self.gae_lambda = gae_lambda

    def save_checkpoint(self, obs):
        """
        Save a checkpoint every X steps so that we may load it later if a collision was found. This is NOT the same as the buffer
        Checkpoints are added to the buffer only if we find a collision and want to replay that event later on
        """
        self.episode_checkpoints.append((deepcopy(self.env), deepcopy(obs)))

    def reset(self):
        """For reset we just use the default implementation."""
        return self.env.reset()

    def step(self, action, values_pred=None):
        obs, rewards, dones, infos = self.env.step(action)

        if values_pred and self.use_curriculum:
            self.rewards.append(rewards)
            self.values_pred.append(values_pred)
            self.dones.append(dones)

        if any(dones):
            if self.use_curriculum:
                if self.env.activate_replay_buffer:
                    # Calculate score with the given score function
                    ep_rewards = torch.tensor(self.rewards)
                    ep_dones = torch.tensor(self.dones).float()

                    # since we cannot get the last value from sample factory, set it to be same as the previous one
                    self.values_pred.append(self.values_pred[-1])
                    values_pred = torch.tensor(self.values_pred)
                    valids = torch.ones_like(values_pred).float()

                    gae = gae_advantage(ep_rewards, ep_dones, values_pred, valids, self.gamma, self.gae_lambda)

                    # Calculate score for each event
                    for event_idx, env_tick in zip(self.curr_event_ids, self.event_ticks):
                        traj_gae = gae[env_tick:]
                        traj_score = torch.mean(torch.abs(traj_gae))
                        # Update event info
                        self.replay_buffer.update_event(event_idx, traj_score.item(), self.episode_counter, env_tick)

                # Reset
                self.rewards = []
                self.values_pred = []
                self.dones = []
                self.curr_event_ids = []
                self.event_ticks = []

            # Cleanup replay buffer after each episode ends
            self.replay_buffer.cleanup()

            # Sample from replay buffer
            obs = self.new_episode()
            for i in range(len(infos)):
                if not infos[i]["episode_extra_stats"]:
                    infos[i]["episode_extra_stats"] = dict()

                tag = "replay"
                infos[i]["episode_extra_stats"].update({
                    f"{tag}/replay_rate": self.replayed_events / self.episode_counter,
                    f"{tag}/new_episode_rate": (self.episode_counter - self.replayed_events) / self.episode_counter,
                    f"{tag}/replay_buffer_size": len(self.replay_buffer),
                    f"{tag}/avg_replayed": self.replay_buffer.avg_num_replayed(),
                })

        else:
            if self.env.use_replay_buffer and self.env.activate_replay_buffer and not self.env.saved_in_replay_buffer \
                    and self.env.envs[0].tick % self.replay_buffer.cp_step_size_freq == 0:
                # Save a deepcopy of current env so that we can add a copy of this episode once again
                # if another collision occurs
                self.save_checkpoint(obs)

            collision_flag = self.env.last_step_unique_collisions.any()
            if self.env.use_obstacles:
                collision_flag = collision_flag or len(self.env.curr_quad_col) > 0

            if collision_flag and self.env.use_replay_buffer and self.env.activate_replay_buffer \
                    and self.env.envs[0].tick > self.env.collisions_grace_period_seconds * self.env.envs[0].control_freq and not self.env.saved_in_replay_buffer:

                if self.env.envs[0].tick - self.last_tick_added_to_buffer > 5 * self.env.envs[0].control_freq:
                    # added this check to avoid adding a lot of collisions from the same episode to the buffer

                    steps_ago = int(self.save_time_before_collision_sec / self.replay_buffer.cp_step_size_sec)
                    if steps_ago > len(self.episode_checkpoints):
                        print(f"Tried to read past the boundary of checkpoint_history. Steps ago: {steps_ago}, episode checkpoints: {len(self.episode_checkpoints)}, {self.env.envs[0].tick}")
                        raise IndexError
                    else:
                        env, obs = self.episode_checkpoints[-steps_ago]
                        # self.replay_buffer.write_cp_to_buffer(env, obs)
                        # self.env.collision_occurred = False  # this allows us to add a copy of this episode to the buffer once again if another collision happens
                        curr_event_idx = self.replay_buffer.write_cp_to_buffer(env, obs)
                        if self.use_curriculum:
                            self.curr_event_ids.append(curr_event_idx)
                            self.event_ticks.append(env.envs[0].tick)

                        self.last_tick_added_to_buffer = self.env.envs[0].tick

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
            event, event_idx = self.replay_buffer.sample_event()
            env = event.env
            obs = event.obs
            replayed_env = deepcopy(env)

            # we want to use these for tensorboard, so reset them to zero to get accurate stats
            replayed_env.collisions_per_episode = replayed_env.collisions_after_settle = 0
            replayed_env.obst_quad_collisions_per_episode = replayed_env.obst_quad_collisions_after_settle = 0
            self.env = replayed_env

            # Since we replay this episode, env tick should be set to 0 otherwise the score may be wrong
            self.curr_event_ids.append(event_idx)
            self.event_ticks.append(0)

            return obs
        else:
            obs = self.env.reset()
            self.env.saved_in_replay_buffer = False
            return obs
