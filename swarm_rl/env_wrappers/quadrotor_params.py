from sample_factory.utils.utils import str2bool


def quadrotors_override_defaults(env, parser):
    parser.set_defaults(
        encoder_type='mlp',
        encoder_subtype='mlp_quads',
        rnn_size=256,
        encoder_extra_fc_layers=0,
        env_frameskip=1,
    )


# noinspection PyUnusedLocal
def add_quadrotors_env_args(env, parser):
    p = parser

    p.add_argument('--quads_episode_duration', default=15.0, type=float, help='Override default value for episode duration')
    p.add_argument('--quads_num_agents', default=8, type=int, help='Override default value for the number of quadrotors')
    p.add_argument('--quads_neighbor_hidden_size', default=256, type=int, help='The hidden size for the neighbor encoder')
    p.add_argument('--quads_neighbor_encoder_type', default='attention', type=str, choices=['attention', 'mean_embed', 'mlp', 'no_encoder'], help='The type of the neighborhood encoder')
    p.add_argument('--quads_encoder_type', default="corl", type=str, help='The type of the neighborhood encoder')

    # TODO: better default values for collision rewards
    p.add_argument('--quads_collision_reward', default=0.0, type=float, help='Override default value for quadcol_bin reward, which means collisions between quadrotors')
    p.add_argument('--quads_collision_hitbox_radius', default=2.0, type=float, help='When the distance between two drones are less than N arm_length, we would view them as collide.')
    p.add_argument('--quads_collision_falloff_radius', default=0.0, type=float, help='The falloff radius for the smooth penalty. 0: radius is 0 arm_length, which means we would not add extra penalty except drones collide')
    p.add_argument('--quads_collision_smooth_max_penalty', default=10.0, type=float, help='The upper bound of the collision function given distance among drones')

    p.add_argument('--quads_obst_hidden_size', default=256, type=int, help='The hidden size for the neighbor encoder')
    p.add_argument('--quads_obst_encoder_type', default='attention', type=str, choices=['attention', 'mean_embed', 'mlp', 'no_encoder'], help='The type of the neighborhood encoder')
    p.add_argument('--quads_collision_reward_obst', default=0.0, type=float, help='Override default value for quadcol_bin reward, which means collisions between quadrotors')
    p.add_argument('--quads_collision_obst_falloff_radius', default=0.0, type=float, help='The falloff radius for the smooth penalty. 0: radius is 0 arm_length, which means we would not add extra penalty except drones collide')
    p.add_argument('--quads_collision_obst_smooth_max_penalty', default=10.0, type=float, help='The upper bound of the collision function given distance among drones')
    p.add_argument('--use_obstacles', default=False, type=str2bool, help='Use Obstacles or not')
    p.add_argument('--quads_obstacle_mode', default='no_obstacles', type=str, choices=['no_obstacles', 'static'], help='Choose which obstacle mode to run')
    p.add_argument('--quads_obstacle_num', default=0, type=int, help='Set obstacle number')
    p.add_argument('--quads_obstacle_size', default=1.0, type=float, help='The radius of obstacles')
    p.add_argument('--quads_obstacle_density', default=0.2, type=float, help='Obstacle density in the map')
    p.add_argument('--quads_obstacle_spawn_area', nargs='+', default=[6.0, 6.0], type=float, help='The spawning area of obstacles')
    p.add_argument('--quads_obstacle_shape', default="cube", type=str, choices=["cube", "cylinder"], help='The shape of obstacles')
    p.add_argument('--quads_obst_obs_clip', default=10.0, type=float, help='Clip distance to the closest obstacle in octomap')

    p.add_argument('--quads_obst_collision_reward', default=0.0, type=float, help='Override default value for quadcol_bin_obst reward, which means collisions between quadrotor and obstacles')
    p.add_argument('--quads_obst_collision_smooth_max_penalty', default=10.0, type=float, help='The upper bound of the collision function given distance between drones and obstacles')
    p.add_argument('--quads_collision_coeff', default=1.0, type=float, help='The coefficient for collision simulation between drones')
    p.add_argument('--use_downwash', default=False, type=bool, help='Apply downwash or not')

    p.add_argument('--neighbor_obs_type', default='none', type=str, choices=['none', 'pos_vel', 'pos_vel_goals', 'pos_vel_goals_ndist_gdist'], help='Choose what kind of obs to send to encoder.')
    p.add_argument('--quads_use_numba', default=False, type=str2bool, help='Whether to use numba for jit or not')
    p.add_argument('--quads_local_obs', default=-1, type=int, help='Number of neighbors to consider. -1=all neighbors. 0=blind agents, 0<n<num_agents-1 = nonzero number of agents')

    p.add_argument('--quads_view_mode', default='local', type=str, choices=['local', 'global'], help='Choose which kind of view/camera to use')

    p.add_argument('--quads_mode', default='static_same_goal', type=str, choices=['static_same_goal', 'static_diff_goal', 'dynamic_same_goal', 'dynamic_diff_goal', 'ep_lissajous3D', 'ep_rand_bezier', 'swarm_vs_swarm', 'swap_goals', 'dynamic_formations', 'mix', 'tunnel', 'o_uniform_same_goal_spawn', 'mix_test', 'o_test', 'o_random', 'o_dynamic_diff_goal', 'o_dynamic_same_goal', 'o_diagonal', 'o_static_same_goal'], help='Choose which scenario to run. Ep = evader pursuit')
    p.add_argument('--quads_formation', default='circle_horizontal', type=str, choices=['circle_xz_vertical', 'circle_yz_vertical', 'circle_horizontal', 'sphere', 'grid_xz_vertical', 'grid_yz_vertical', 'grid_horizontal'], help='Choose the swarm formation at the goal')
    p.add_argument('--quads_formation_size', default=-1.0, type=float, help='The size of the formation, interpreted differently depending on the formation type. Default (-1) means it is determined by the mode')
    p.add_argument('--room_dims', nargs='+', default=[10, 10, 10], type=float, help='Length, width, and height dimensions respectively of the quadrotor env')
    p.add_argument('--quads_obs_repr', default='xyz_vxyz_R_omega', type=str, help='obs space for drone itself')
    p.add_argument('--replay_buffer_sample_prob', default=0.0, type=float, help='Probability at which we sample from it rather than resetting the env. Set to 0.0 (default) to disable the replay. Set to value in (0.0, 1.0] to use replay buffer')

    p.add_argument('--anneal_collision_steps', default=0.0, type=float, help='Anneal collision penalties over this many steps. Default (0.0) is no annealing')
    p.add_argument('--anneal_collision_sim_steps', default=0.0, type=float, help='Anneal simulation of collision over this many steps. Default (0.0) is no annealing')
    p.add_argument('--use_spectral_norm', default=False, type=str2bool, help="Use spectral normalization to smoothen the gradients and stabilize training. Only supports fully connected layers")
