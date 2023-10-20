from collections import deque

import numpy as np
import gym
from gym.utils.step_api_compatibility import convert_to_terminated_truncated_step_api


# Wrapper for Automatic Domain Randomization
class ADRNoiseWrapper(gym.Wrapper):
    def __init__(self, env, pos_noise_init, pos_std_step, buffer_max_len=10, perf_threshold_low=0.8,
                 perf_threshold_high=0.9):
        """A wrapper which implements automatic domain randomization."""
        super().__init__(env)
        self.pos_std_init = pos_noise_init
        self.pos_std_low, self.pos_std_high = pos_noise_init, pos_noise_init + 0.05
        self.pos_std_step = pos_std_step

        self.buffer_max_len = buffer_max_len
        self.perf_buffer = deque(maxlen=self.buffer_max_len)
        self.perf_threshold_low = perf_threshold_low
        self.perf_threshold_high = perf_threshold_high

        self.curr_pos_std = None

    def reset(self, obst_density=None, obst_size=None):
        """Resets pos noise parameters of the environment.

        Args:
            obst_density: the density of obstacles to reset the environment with
            obst_size: the size of obstacles to reset the environment with
            seed: the seed to reset the environment with
            options: the options to reset the environment with

        Returns:
            (observation, info)
        """
        self.curr_pos_std = self.pos_std_low if np.random.random() < 0.5 else self.pos_std_high
        print("Pos Noise Std in the new episode: {:.4f}".format(self.curr_pos_std))
        # TODO: set the pos noise std in the environment
        env_noise_shaping = self.env.unwrapped.noise_coeff
        env_noise_shaping['pos_norm_std'] = self.curr_pos_std
        return self.env.reset()

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
                infos[i]['episode_extra_stats']['pos_std'] = self.curr_pos_std

            if 'metric/agent_success_rate' in infos[0]['episode_extra_stats']:
                self.perf_buffer.append(infos[0]['episode_extra_stats']['metric/agent_success_rate'])

            if len(self.perf_buffer) == self.buffer_max_len:
                avg_perf = np.mean(self.perf_buffer)

                if avg_perf <= self.perf_threshold_low:
                    self.pos_std_low = max(self.pos_std_low - self.pos_std_step, self.pos_std_init)
                    self.pos_std_high = max(self.pos_std_high - self.pos_std_step, self.pos_std_init + 0.05)

                elif avg_perf >= self.perf_threshold_high:
                    self.pos_std_low = self.pos_std_low + self.pos_std_step
                    self.pos_std_high = self.pos_std_high + self.pos_std_step

                self.perf_buffer.clear()

            # # Reset the noise std in the environment
            # self.curr_pos_std = self.pos_std_low if np.random.random() < 0.5 else self.pos_std_high
            # env_noise_shaping = self.env.unwrapped.noise_coeff
            # env_noise_shaping['pos_norm_std'] = self.curr_pos_std

        return obs, reward, dones, infos
