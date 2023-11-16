from collections import deque

import numpy as np
import gym
from gym.utils.step_api_compatibility import convert_to_terminated_truncated_step_api


# Wrapper for Automatic Domain Randomization
class ADRNoiseWrapper(gym.Wrapper):
    def __init__(self, env, adr_cfg):
        """A wrapper which implements automatic domain randomization."""
        super().__init__(env)
        self.adr_cfg = adr_cfg
        self.adr_threshold_low = adr_cfg['threshold_low']
        self.adr_threshold_high = self.adr_cfg["threshold_high"]
        self.adr_queue_length = self.adr_cfg['adr_queue_length']

        self.adr_params = self.adr_cfg["params"]
        self.adr_ranges = {}
        self.curr_adr_values = {}
        self.adr_step_sizes = {}
        for k in self.adr_params:
            self.adr_ranges[k] = self.adr_params[k]["init_range"]
            self.curr_adr_values[k] = self.adr_params[k]["init_range"][0]
            self.adr_step_sizes[k] = self.adr_params[k]["step_size"]

        self.num_adr_params = len(self.adr_params)
        self.adr_params_keys = list(self.adr_params.keys())
        self.adr_perf_queue = [deque(maxlen=self.adr_queue_length) for _ in range(self.num_adr_params)]

        self.curr_adr_param_index = None
        self.curr_adr_param_name = None

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
        # Sample which noise parameters to use
        self.curr_adr_param_index = np.random.randint(self.num_adr_params)
        self.curr_adr_param_name = self.adr_params_keys[self.curr_adr_param_index]
        # Set the value to be the upper bound of the range
        adr_param_value = self.adr_ranges[self.curr_adr_param_name][1]
        self.curr_adr_values[self.curr_adr_param_name] = adr_param_value
        print("{} in the new episode: {:.4f}".format(self.curr_adr_param_name, adr_param_value))

        env_noise_shaping = self.env.unwrapped.noise_coeff
        for k in self.adr_params:
            env_noise_shaping[k] = self.curr_adr_values[k]

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
                for k in self.adr_params:
                    infos[i]['episode_extra_stats'][k] = self.curr_adr_values[k]

            if 'metric/agent_success_rate' in infos[0]['episode_extra_stats']:
                adr_param_index = self.curr_adr_param_index
                adr_param_name = self.curr_adr_param_name
                self.adr_perf_queue[adr_param_index].append(
                    infos[0]['episode_extra_stats']['metric/agent_success_rate'])

                if len(self.adr_perf_queue[adr_param_index]) == self.adr_queue_length:
                    avg_perf = np.mean(self.adr_perf_queue[adr_param_index])

                    if avg_perf <= self.adr_threshold_low:
                        current_range = self.adr_ranges[adr_param_name]
                        init_range = self.adr_params[adr_param_name]["init_range"]
                        step_size = self.adr_step_sizes[adr_param_name]

                        self.adr_ranges[adr_param_name][0] = max(current_range[0] - step_size, init_range[0])
                        self.adr_ranges[adr_param_name][1] = max(current_range[1] - step_size, init_range[1])

                    elif avg_perf >= self.perf_threshold_high:
                        current_range = self.adr_ranges[adr_param_name]
                        step_size = self.adr_step_sizes[adr_param_name]

                        self.adr_ranges[adr_param_name][0] = current_range[0] + step_size
                        self.adr_ranges[adr_param_name][1] = current_range[1] + step_size

                    self.adr_perf_queue[adr_param_index].clear()

            # # Reset the noise std in the environment
            # self.curr_pos_std = self.pos_std_low if np.random.random() < 0.5 else self.pos_std_high
            # env_noise_shaping = self.env.unwrapped.noise_coeff
            # env_noise_shaping['pos_norm_std'] = self.curr_pos_std

        return obs, reward, dones, infos
