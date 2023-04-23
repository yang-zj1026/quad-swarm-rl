import numpy as np

from gym_art.quadrotor_multi.scenarios.base import QuadrotorScenario


class Scenario_o_base(QuadrotorScenario):
    def __init__(self, quads_mode, envs, num_agents, room_dims):
        super().__init__(quads_mode, envs, num_agents, room_dims)
        self.start_point = np.array([0.0, -3.0, 2.0])
        self.end_point = np.array([0.0, 3.0, 2.0])
        self.room_dims = room_dims
        self.duration_step = 0
        self.quads_mode = quads_mode

        self.cell_centers = None
        self.obstacle_map = None
        self.free_space = []

    def generate_pos(self):
        half_room_length = self.room_dims[0] / 2
        half_room_width = self.room_dims[1] / 2

        x = np.random.uniform(low=-1.0 * half_room_length + 2.0, high=half_room_length - 2.0)
        y = np.random.uniform(low=-1.0 * half_room_width + 2.0, high=half_room_width - 2.0)

        z = np.random.uniform(low=1.0, high=4.0)

        return np.array([x, y, z])

    def step(self):
        tick = self.envs[0].tick

        if tick <= self.duration_step:
            return

        self.duration_step += int(self.envs[0].ep_time * self.envs[0].control_freq)
        self.goals = self.generate_goals(num_agents=self.num_agents, formation_center=self.end_point, layer_dist=0.0)

        for i, env in enumerate(self.envs):
            env.goal = self.goals[i]

        return

    def reset(self, obst_map, cell_centers):
        self.start_point = self.generate_pos()
        self.end_point = self.generate_pos()
        self.duration_step = int(np.random.uniform(low=2.0, high=4.0) * self.envs[0].control_freq)
        self.standard_reset(formation_center=self.start_point)

    def generate_pos_obst_map(self, check_surroundings=False):
        idx = np.random.choice(a=len(self.free_space), replace=True)
        x, y = self.free_space[idx][0], self.free_space[idx][1]
        if check_surroundings:
            surroundings_free = self.check_surroundings(x, y)
            while not surroundings_free:
                idx = np.random.choice(a=len(self.free_space), replace=True)
                x, y = self.free_space[idx][0], self.free_space[idx][1]
                surroundings_free = self.check_surroundings(x, y)

        width = self.obstacle_map.shape[0]
        index = x + (width * y)
        pos_x, pos_y = self.cell_centers[index]
        z_list_start = np.random.uniform(low=0.5, high=3.0)
        # xy_noise = np.random.uniform(low=-0.2, high=0.2, size=2)
        return np.array([pos_x, pos_y, z_list_start])

    def check_surroundings(self, row, col):
        length, width = self.obstacle_map.shape[0], self.obstacle_map.shape[1]
        obstacle_map = self.obstacle_map
        # Check if the given position is out of bounds
        if row < 0 or row >= width or col < 0 or col >= length:
            raise ValueError("Invalid position")

        # Check if the surrounding cells are all 0s
        check_pos_x, check_pos_y = [], []
        if row > 0:
            check_pos_x.append(row - 1)
            check_pos_y.append(col)
            if col > 0:
                check_pos_x.append(row - 1)
                check_pos_y.append(col - 1)
        if row < width - 1:
            check_pos_x.append(row + 1)
            check_pos_y.append(col)

        if col > 0:
            check_pos_x.append(row)
            check_pos_y.append(col - 1)
        if col < length - 1:
            check_pos_x.append(row)
            check_pos_y.append(col + 1)
            if row < length - 1:
                check_pos_x.append(row + 1)
                check_pos_y.append(col + 1)

        check_pos = ([check_pos_x, check_pos_y])
        # Get the values of the adjacent cells
        adjacent_cells = obstacle_map[check_pos]

        return np.any(adjacent_cells != 0)
