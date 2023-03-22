import numpy as np
import copy

from gym_art.quadrotor_multi.scenarios.obstacles.o_base import Scenario_o_base


class Scenario_o_static_same_goal(Scenario_o_base):
    def __init__(self, quads_mode, envs, num_agents, room_dims, room_dims_callback, rew_coeff, quads_formation,
                 quads_formation_size):
        super().__init__(quads_mode, envs, num_agents, room_dims, room_dims_callback, rew_coeff, quads_formation,
                         quads_formation_size)
        # teleport every [4.0, 6.0] secs
        duration_time = 5.0
        self.control_step_for_sec = int(duration_time * self.envs[0].control_freq)

    def update_formation_size(self, new_formation_size):
        pass

    def step(self):
        tick = self.envs[0].tick

        if tick <= int(self.duration_time * self.envs[0].control_freq):
            return

        self.duration_time += self.envs[0].ep_time + 1
        for i, env in enumerate(self.envs):
            env.goal = self.end_point

        return

    def reset(self, obst_map=None, cell_centers=None):
        self.start_point = []

        # Update duration time
        self.duration_time = np.random.uniform(low=4.0, high=6.0)
        self.control_step_for_sec = int(self.duration_time * self.envs[0].control_freq)

        self.obstacle_map = obst_map
        self.cell_centers = cell_centers
        if obst_map is None or cell_centers is None:
            raise NotImplementedError

        obst_map_locs = np.where(self.obstacle_map == 0)
        self.free_space = list(zip(*obst_map_locs))

        for i in range(self.num_agents):
            self.start_point.append(self.generate_pos_obst_map())

        self.end_point = self.generate_pos_obst_map(check_surroundings=True)

        # Reset formation and related parameters
        self.update_formation_and_relate_param()

        # Reassign goals
        self.goals = copy.deepcopy(self.start_point)
