from sample_factory.launcher.run_description import Experiment, ParamGrid, RunDescription
from swarm_rl.runs.obstacles.quad_obstacle_baseline import QUAD_BASELINE_CLI_8

_params = ParamGrid(
    [
        ("seed", [0000, 1111]),
        ("quads_obst_density", [0.2, 0.4]),
    ]
)

OBSTACLE_MODEL_CLI = QUAD_BASELINE_CLI_8 + (
    ' --num_workers=36 --num_envs_per_worker=4 --quads_num_agents=8 --quads_use_downwash=True '
    '--quads_neighbor_visible_num=2 --quads_neighbor_obs_type=pos_vel --quads_encoder_type=attention '
    '--quads_sim2real=True --rnn_size=10 '
    '--with_wandb=True --wandb_project=Quad-Swarm-RL --wandb_user=multi-drones '
    '--wandb_group=sim2real_obst_density'
)

_experiment = Experiment(
    "sim2real_obst_density",
    OBSTACLE_MODEL_CLI,
    _params.generate_params(randomize=False),
)

RUN_DESCRIPTION = RunDescription("obstacles_multi", experiments=[_experiment])