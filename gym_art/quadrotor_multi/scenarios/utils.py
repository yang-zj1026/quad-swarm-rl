import numpy as np
from gym_art.quadrotor_multi.quad_utils import get_circle_radius, get_sphere_radius, get_grid_dim_number

QUADS_MODE_LIST = ['static_same_goal', 'static_diff_goal',  # static formations
                   'ep_lissajous3D', 'ep_rand_bezier',  # evader pursuit
                   'dynamic_same_goal', 'dynamic_diff_goal', 'dynamic_formations', 'swap_goals',  # dynamic formations
                   'swarm_vs_swarm']  # only support >=2 drones

QUADS_MODE_LIST_SINGLE = ['static_same_goal', 'static_diff_goal',  # static formations
                          'ep_lissajous3D', 'ep_rand_bezier',  # evader pursuit
                          'dynamic_same_goal',  # dynamic formations
                          ]

QUADS_MODE_LIST_OBSTACLES = ['o_random', 'o_static_same_goal']

QUADS_MODE_LIST_OBSTACLES_SINGLE = ['o_random']


QUADS_FORMATION_LIST = ['circle_horizontal', 'circle_vertical_xz', 'circle_vertical_yz', 'sphere', 'grid_horizontal',
                        'grid_vertical_xz', 'grid_vertical_yz', 'cube']

# key: quads_mode
# value: 0. formation, 1: [formation_low_size, formation_high_size], 2: episode_time
quad_arm_size = 0.05
QUADS_PARAMS_DICT = {
    'static_same_goal': [['circle_horizontal'], [0.0, 0.0]],
    'dynamic_same_goal': [['circle_horizontal'], [0.0, 0.0]],
    'ep_lissajous3D': [['circle_horizontal'], [0.0, 0.0]],
    'ep_rand_bezier': [['circle_horizontal'], [0.0, 0.0]],
    'static_diff_goal': [QUADS_FORMATION_LIST, [5 * quad_arm_size, 10 * quad_arm_size]],
    'dynamic_diff_goal': [QUADS_FORMATION_LIST, [5 * quad_arm_size, 10 * quad_arm_size]],
    'swarm_vs_swarm': [QUADS_FORMATION_LIST, [5 * quad_arm_size, 10 * quad_arm_size]],
    'swap_goals': [QUADS_FORMATION_LIST, [8 * quad_arm_size, 16 * quad_arm_size]],
    'dynamic_formations': [QUADS_FORMATION_LIST, [0.0, 20 * quad_arm_size]],
    'run_away': [QUADS_FORMATION_LIST, [5 * quad_arm_size, 10 * quad_arm_size]],

    # For obstacles
    'o_uniform_same_goal_spawn': [['circle_horizontal'], [0.0, 0.0]],
    'o_uniform_diff_goal_spawn': [QUADS_FORMATION_LIST, [0.4, 0.8]],
    'o_uniform_swarm_vs_swarm': [QUADS_FORMATION_LIST, [0.4, 0.8]],
    'o_test': [['circle_horizontal'], [0.0, 0.0]],
    'o_random': [QUADS_FORMATION_LIST, [5 * quad_arm_size, 10 * quad_arm_size]],
    'o_static_same_goal': [['circle_horizontal'], [0.0, 0.0]],
    'o_dynamic_same_goal': [['circle_horizontal'], [0.0, 0.0]],
    'o_dynamic_diff_goal': [QUADS_FORMATION_LIST, [5 * quad_arm_size, 10 * quad_arm_size]],
    'o_diagonal': [['circle_horizontal'], [0.0, 0.0]],

}


def create_scenario(quads_mode, envs, num_agents, room_dims, room_dims_callback, rew_coeff, quads_formation,
                    quads_formation_size):
    cls = eval('Scenario_' + quads_mode)
    scenario = cls(quads_mode, envs, num_agents, room_dims, room_dims_callback, rew_coeff, quads_formation,
                   quads_formation_size)
    return scenario


def update_formation_and_max_agent_per_layer(mode):
    formation_index = np.random.randint(low=0, high=len(QUADS_PARAMS_DICT[mode][0]))
    formation = QUADS_FORMATION_LIST[formation_index]
    if formation.startswith("circle"):
        num_agents_per_layer = 8
    elif formation.startswith("grid"):
        num_agents_per_layer = 50
    else:
        # for 3D formations. Specific formations override this
        num_agents_per_layer = 8

    return formation, num_agents_per_layer


def update_layer_dist(low, high):
    layer_dist = np.random.uniform(low=low, high=high)
    return layer_dist


def get_formation_range(mode, formation, num_agents, low, high, num_agents_per_layer):
    if mode == 'swarm_vs_swarm':
        n = num_agents // 2
    else:
        n = num_agents

    if formation.startswith("circle"):
        formation_size_low = get_circle_radius(num_agents_per_layer, low)
        formation_size_high = get_circle_radius(num_agents_per_layer, high)
    elif formation.startswith("grid"):
        formation_size_low = low
        formation_size_high = high
    elif formation.startswith("sphere"):
        formation_size_low = get_sphere_radius(n, low)
        formation_size_high = get_sphere_radius(n, high)
    elif formation.startswith("cube"):
        formation_size_low = low
        formation_size_high = high
    else:
        raise NotImplementedError(f'{formation} is not supported!')

    return formation_size_low, formation_size_high


def get_goal_by_formation(formation, pos_0, pos_1, layer_pos=0.):
    if formation.endswith("horizontal"):
        goal = np.array([pos_0, pos_1, layer_pos])
    elif formation.endswith("vertical_xz"):
        goal = np.array([pos_0, layer_pos, pos_1])
    elif formation.endswith("vertical_yz"):
        goal = np.array([layer_pos, pos_0, pos_1])
    else:
        raise NotImplementedError("Unknown formation")

    return goal


def get_z_value(num_agents, num_agents_per_layer, box_size, formation, formation_size):
    z = np.random.uniform(low=-0.5 * box_size, high=0.5 * box_size) + 2.0
    z_lower_bound = 0.25
    if formation == "sphere" or formation.startswith("circle_vertical"):
        z_lower_bound = formation_size + 0.25
    elif formation.startswith("grid_vertical"):
        real_num_per_layer = np.minimum(num_agents, num_agents_per_layer)
        dim_1, _ = get_grid_dim_number(real_num_per_layer)
        z_lower_bound = dim_1 * formation_size + 0.25

    z = max(z_lower_bound, z)
    return z
