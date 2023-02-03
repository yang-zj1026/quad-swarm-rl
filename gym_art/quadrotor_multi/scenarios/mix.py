import numpy as np

from gym_art.quadrotor_multi.scenarios.base import QuadrotorScenario
from gym_art.quadrotor_multi.scenarios.utils import QUADS_MODE_LIST_SINGLE, QUADS_MODE_LIST, \
    QUADS_MODE_LIST_OBSTACLES, create_scenario


class Scenario_mix(QuadrotorScenario):
    def __init__(self, quads_mode, envs, num_agents, room_dims, room_dims_callback, rew_coeff, quads_formation,
                 quads_formation_size):
        super().__init__(quads_mode, envs, num_agents, room_dims, room_dims_callback, rew_coeff, quads_formation,
                         quads_formation_size)

        self.room_dims_callback = room_dims_callback

        # Once change the parameter here, should also update QUADS_PARAMS_DICT to make sure it is same as run a
        # single scenario key: quads_mode value: 0. formation, 1: [formation_low_size, formation_high_size],
        # 2: episode_time
        if num_agents == 1:
            self.quads_mode_list = QUADS_MODE_LIST_SINGLE
        elif num_agents > 1 and not envs[0].use_obstacles:
            self.quads_mode_list = QUADS_MODE_LIST
        elif envs[0].use_obstacles:
            self.quads_mode_list = QUADS_MODE_LIST_OBSTACLES

            # Add parameters
            self.start_point = np.array([-3.0, -3.0, 2.0])
            self.end_point = np.array([3.0, 3.0, 2.0])
            self.scenario_mode = 'o_dynamic_same_goal'

        # actual scenario being used
        self.scenario = None

    def name(self):
        """
        :return: the name of the actual scenario used in this episode
        """
        return self.scenario.__class__.__name__

    def step(self, infos, rewards, pos):
        infos, rewards = self.scenario.step(infos=infos, rewards=rewards, pos=pos)
        # This is set for obstacle mode
        self.goals = self.scenario.goals
        self.formation_size = self.scenario.formation_size
        return infos, rewards

    def reset(self):
        mode_index = np.random.randint(low=0, high=len(self.quads_mode_list))
        mode = self.quads_mode_list[mode_index]

        # Init the scenario
        self.scenario = create_scenario(quads_mode=mode, envs=self.envs, num_agents=self.num_agents,
                                        room_dims=self.room_dims, room_dims_callback=self.room_dims_callback,
                                        rew_coeff=self.rew_coeff, quads_formation=self.formation,
                                        quads_formation_size=self.formation_size)

        self.scenario.reset()
        self.goals = self.scenario.goals
        self.formation_size = self.scenario.formation_size

        if self.envs[0].use_obstacles:
            self.start_point = self.scenario.start_point
            self.end_point = self.scenario.end_point
