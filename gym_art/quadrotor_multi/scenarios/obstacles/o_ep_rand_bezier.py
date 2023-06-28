import numpy as np
import copy

from gym_art.quadrotor_multi.scenarios.obstacles.o_base import Scenario_o_base


class Scenario_o_rand_bezier(Scenario_o_base):
    def __init__(self, quads_mode, envs, num_agents, room_dims):
        super().__init__(quads_mode, envs, num_agents, room_dims)
        # teleport every [4.0, 6.0] secs
        duration_time = 0.3
        self.control_step_for_sec = int(duration_time * self.envs[0].control_freq)
        self.approch_goal_metric = 1.0

    def step(self):
        tick = self.envs[0].tick

        if tick % int(self.duration_time * self.envs[0].control_freq) == 0:
            idx = tick // int(self.duration_time * self.envs[0].control_freq)
            for i, env in enumerate(self.envs):
                env.goal = self.traj_points_smooth[idx]

        return

    def reset(self, obst_map=None, cell_centers=None):
        # Update duration time
        self.duration_time = 0.01
        self.control_step_for_sec = int(self.duration_time * self.envs[0].control_freq)

        self.obstacle_map = obst_map
        self.cell_centers = cell_centers
        if obst_map is None or cell_centers is None:
            raise NotImplementedError

        obst_map_locs = np.where(self.obstacle_map == 0)
        self.free_space = list(zip(*obst_map_locs))

        self.start_point = self.generate_pos_obst_map_2(num_agents=self.num_agents)
        self.end_point = self.max_square_area_center()

        # Generate obstacle-free trajectory points
        num_samples = 10
        max_dist = 4.0
        sampled_points_idx = []
        while len(sampled_points_idx) < num_samples:
            # Randomly select a point
            point_idx = np.random.choice(len(self.free_space))

            # Check if the distance constraint is satisfied with the previously sampled points
            if len(sampled_points_idx) > 0:
                distances = np.array([np.linalg.norm(self.cell_centers[sampled_point_idx] - self.cell_centers[point_idx])
                                      for sampled_point_idx in sampled_points_idx])
                if np.any(distances > max_dist):
                    continue

            # Add the point to the sampled trajectory and remove it from the free space
            sampled_points_idx.append(point_idx)
            self.free_space.pop(point_idx)

        # Separate x and y coordinates of the sampled points
        sampled_points = self.cell_centers[sampled_points_idx]

        # Reset formation and related parameters
        self.update_formation_and_relate_param()

        # Reassign goals
        self.spawn_points = copy.deepcopy(self.start_point)
        self.goals = np.array([self.end_point for _ in range(self.num_agents)])
