import numpy as np
import copy

from gym_art.quadrotor_multi.scenarios.obstacles.o_base import Scenario_o_base


class Scenario_o_random(Scenario_o_base):
    def __init__(self, quads_mode, envs, num_agents, room_dims, room_dims_callback, rew_coeff, quads_formation,
                 quads_formation_size):
        super().__init__(quads_mode, envs, num_agents, room_dims, room_dims_callback, rew_coeff, quads_formation,
                         quads_formation_size)
        self.free_x = [-self.room_dims[0] / 2 + 2, self.room_dims[0] / 2 - 2,
                       -self.room_dims[0] / 2 + 2, self.room_dims[0] / 2 - 2]

        self.free_y = [-self.room_dims[1] / 2 + 2, -self.room_dims[1] / 2 + 2,
                       self.room_dims[1] / 2 - 2, self.room_dims[1] / 2 - 2]

    def update_formation_size(self, new_formation_size):
        pass

    def step(self, infos, rewards):
        tick = self.envs[0].tick

        if tick <= int(self.duration_time * self.envs[0].control_freq):
            return infos, rewards

        self.duration_time += self.envs[0].ep_time + 1
        for i, env in enumerate(self.envs):
            env.goal = self.end_point[i]

        return infos, rewards

    def reset(self, obst_map=None, cell_centers=None):
        self.start_point = []
        self.end_point = []

        self.obstacle_map = obst_map
        self.cell_centers = cell_centers
        if obst_map is None or cell_centers is None:
            raise NotImplementedError

        obst_map_locs = np.where(self.obstacle_map == 0)
        self.free_space = list(zip(*obst_map_locs))

        for i in range(self.num_agents):
            start_goal, end_goal = self.generate_pos_obst_map(), self.generate_pos_obst_map()
            # while np.linalg.norm(start_goal - end_goal) < self.room_dims[0] / 2:
            end_goal = self.generate_pos_obst_map()

            self.start_point.append(start_goal)
            self.end_point.append(end_goal)

        self.start_point = np.array(self.start_point)
        self.end_point = np.array(self.end_point)

        self.duration_time = np.random.uniform(low=4.0, high=6.0)
        self.update_formation_and_relate_param()

        self.formation_center = np.array((0., 0., 2.))
        self.goals = copy.deepcopy(self.start_point)
