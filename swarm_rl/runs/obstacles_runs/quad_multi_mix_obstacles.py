from sample_factory.launcher.run_description import Experiment, ParamGrid, RunDescription
from swarm_rl.runs.obstacles_runs.quad_obstacle_baseline import QUAD_BASELINE_CLI_8

_params = ParamGrid(
    [
        ("seed", [0000, 1111, 2222, 3333]),
    ]
)

OBSTACLE_MODEL_CLI = QUAD_BASELINE_CLI_8 + (
    '--train_for_env_steps=1000000000 --num_workers=36 --num_envs_per_worker=4 '
    '--use_obstacles=True --quads_obstacle_spawn_area 8 8 --quads_obstacle_density=0.2 --quads_obstacle_size=0.6 '
    '--quads_obst_collision_smooth_max_penalty=0.0 --quads_obst_collision_reward=5.0 '
    '--quads_neighbor_hidden_size=256 --quads_local_obs=6 --quads_num_agents=8 '
    '--with_wandb=True --wandb_project=Quad-Swarm-RL --wandb_group=obstacles_multi --wandb_user=multi-drones '
    '--anneal_collision_steps=300000000 --replay_buffer_sample_prob=0.75'
)

_experiment = Experiment(
    "obstacles_multi_agents",
    OBSTACLE_MODEL_CLI,
    _params.generate_params(randomize=False),
)

RUN_DESCRIPTION = RunDescription("obstacles_multi", experiments=[_experiment])
# python -m sample_factory.launcher.run --run=swarm_rl.runs.sf2_multi_drone --runner=processes --max_parallel=1 --pause_between=1 --experiments_per_gpu=4 --num_gpus=1
