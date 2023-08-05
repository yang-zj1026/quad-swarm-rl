import numpy as np
import copy

from gym_art.quadrotor_multi.scenarios.obstacles.o_base import Scenario_o_base

change_rules = {
    0: 6,  # 0 -> 6
    1: 4,  # 1 -> 4
    2: 7,  # 2 -> 7
    3: 2,  # 3 -> 2
    4: 5,  # 4 -> 5
    5: 0,  # 5 -> 0
    6: 3,  # 6 -> 3
    7: 1  # 7 -> 1
}


class Scenario_o_swap_goals(Scenario_o_base):
    def __init__(self, quads_mode, envs, num_agents, room_dims):
        super().__init__(quads_mode, envs, num_agents, room_dims)
        # teleport every [4.0, 6.0] secs
        duration_time = 6.0
        self.control_step_for_sec = int(duration_time * self.envs[0].control_freq)

    def update_goals(self):
        # np.random.shuffle(self.goals)
        current_goals = np.array([env.goal for env in self.envs])
        sorted_goals = np.array(sorted(self.goals, key=lambda pos: (pos[0], pos[1], pos[2])))

        distances = np.linalg.norm(current_goals[:, None] - current_goals, axis=2)
        max_distance_index = np.argmax(distances, axis=1)

        for i, env in enumerate(self.envs):
            # find the index of current goal in sorted goals
            current_goal_index = np.where((sorted_goals == current_goals[i]).all(axis=1))[0][0]
            env.goal = sorted_goals[change_rules[current_goal_index]]
            # env.goal = current_goals[max_distance_index[i]]

    def step(self):
        tick = self.envs[0].tick
        # Switch every [6, 8] seconds
        if tick % self.control_step_for_sec == 0 and tick > 0:
            self.update_goals()

        return

    def reset(self, obst_map=None, cell_centers=None):
        self.obstacle_map = obst_map
        self.cell_centers = cell_centers
        if obst_map is None:
            raise NotImplementedError

        obst_map_locs = np.where(self.obstacle_map == 0)
        self.free_space = list(zip(*obst_map_locs))

        # Update duration time
        duration_time = 10.0
        self.control_step_for_sec = int(duration_time * self.envs[0].control_freq)

        # Reset formation and related parameters
        self.update_formation_and_relate_param()

        self.start_point = self.generate_pos_obst_map_2(num_agents=self.num_agents)
        self.spawn_points = copy.deepcopy(self.start_point)

        self.formation_center = self.max_square_area_center()

        # Regenerate goals, we don't have to assign goals to the envs,
        # the reset function in quadrotor_multi.py would do that
        self.goals = self.generate_goals(num_agents=self.num_agents, formation_center=self.formation_center,
                                         layer_dist=self.layer_dist)
        np.random.shuffle(self.goals)

        # Generate an obstacle in the center of max area
        return self.max_square_area_center(return_index=True)
