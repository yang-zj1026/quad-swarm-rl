from collections import deque

import numpy as np
import gym
from gym.utils.step_api_compatibility import convert_to_terminated_truncated_step_api


# Wrapper for Automatic Domain Randomization
class QuadEnvADR(gym.Wrapper):
    def __init__(self, env, obst_size_init, obst_size_step):
        """A wrapper which implements automatic domain randomization."""
        super().__init__(env)
        self.obst_size_init = obst_size_init
        self.obst_size_low, self.obst_size_high = obst_size_init, obst_size_init + 0.05
        self.obst_size_step = obst_size_step
        # self.obst_density_low, self.obst_density_high = obst_density_init, obst_density_init
        # self.obst_density_step = obst_density_step

        self.buffer_max_len = 30
        self.obst_size_buffer = deque(maxlen=self.buffer_max_len)
        self.perf_threshold_low = 0.8
        self.perf_threshold_high = 0.9

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

        return obs, reward, dones, infos
