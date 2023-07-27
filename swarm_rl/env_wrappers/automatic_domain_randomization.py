from collections import deque

import numpy as np
import gym
from gym.utils.step_api_compatibility import convert_to_terminated_truncated_step_api


# Wrapper for Automatic Domain Randomization
class QuadEnvADR(gym.Wrapper):
    def __init__(self, env, obst_size_init, obst_size_step, buffer_max_len=10, perf_threshold_low=0.25,
                 perf_threshold_high=0.75):
        """A wrapper which implements automatic domain randomization."""
        super().__init__(env)
        self.obst_size_init = obst_size_init
        self.obst_size_low, self.obst_size_high = obst_size_init, obst_size_init + 0.05
        self.obst_size_step = obst_size_step
        # self.obst_density_low, self.obst_density_high = obst_density_init, obst_density_init
        # self.obst_density_step = obst_density_step

        self.buffer_max_len = buffer_max_len
        self.obst_size_buffer = deque(maxlen=self.buffer_max_len)
        self.perf_threshold_low = perf_threshold_low
        self.perf_threshold_high = perf_threshold_high

        self.curr_obst_size = None

    def reset(self):
        """Resets obstacle parameters of the environment.

        Args:
            seed: the seed to reset the environment with
            options: the options to reset the environment with

        Returns:
            (observation, info)
        """
        if np.random.uniform(0, 1) < 0.5:
            self.curr_obst_size = self.obst_size_low
        else:
            self.curr_obst_size = self.obst_size_high
        print("Obstacle size in the new episode: ", self.curr_obst_size)
        return self.env.reset(obst_size=self.curr_obst_size)

    def step(self, action):
        """Steps through the environment.

        Args:
            action: action to step through the environment with

        Returns:
            (observation, reward, terminated, truncated, info)
        """
        obs, reward, dones, infos = self.env.step(action)

        if any(dones):
            for i in range(len(infos)):
                infos[i]['episode_extra_stats']['obst_size'] = self.curr_obst_size

            if 'metric/agent_success_rate' in infos[0]['episode_extra_stats']:
                self.obst_size_buffer.append(infos[0]['episode_extra_stats']['metric/agent_success_rate'])
            if len(self.obst_size_buffer) == self.buffer_max_len:
                avg_perf = np.mean(self.obst_size_buffer)

                if avg_perf <= self.perf_threshold_low:
                    self.obst_size_low = max(self.obst_size_low - self.obst_size_step, self.obst_size_init)
                    self.obst_size_high = max(self.obst_size_high - self.obst_size_step, self.obst_size_init + 0.05)
                elif avg_perf >= self.perf_threshold_high:
                    self.obst_size_low = self.obst_size_low + self.obst_size_step
                    self.obst_size_high = self.obst_size_high + self.obst_size_step

                self.obst_size_buffer.clear()

                # Send a signal to replay buffer to clear it
                infos[0]['episode_extra_stats']['clear_replay_buffer'] = True

        return obs, reward, dones, infos
