from sample_factory.launcher.run_description import Experiment, ParamGrid, RunDescription
from swarm_rl.runs.obstacles.quad_obstacle_baseline import QUAD_BASELINE_CLI_8

_params = ParamGrid(
    [
        ("with_pbt", ["True"]),
    ]
)

OBSTACLE_MODEL_CLI = QUAD_BASELINE_CLI_8 + (
    # PBT
    ' --num_policies=4 --pbt_mix_policies_in_one_env=True --pbt_period_env_steps=50000000 '
    '--pbt_start_mutation=300000000 --pbt_replace_reward_gap=0.05 --pbt_replace_reward_gap_absolute=2.0 '
    '--pbt_optimize_gamma=False --pbt_perturb_max=1.2 --pbt_mix_policies_in_one_env=False '
    # Pre-set hyperparameters
    #'--exploration_loss_coeff=0.0003 --max_entropy_coeff=0.0005 '
    '--anneal_collision_steps=300000000 --train_for_env_steps=15000000000 '
    # Num workers
    '--num_workers=96 --num_envs_per_worker=8 --quads_num_agents=8 '
    # Neighbor & General Encoder for obst & neighbor
    '--quads_neighbor_visible_num=6 --quads_neighbor_obs_type=pos_vel --quads_encoder_type=attention '
    # WandB
    '--with_wandb=True --wandb_project=Quad-Swarm-RL --wandb_user=multi-drones '
    '--wandb_group=pbt_obstacle_multi_attn_new_objective --max_policy_lag=100'
)

_experiment = Experiment(
    "pbt_obstacle_multi_attn_new_objective",
    OBSTACLE_MODEL_CLI,
    _params.generate_params(randomize=False),
)

RUN_DESCRIPTION = RunDescription("obstacles_multi", experiments=[_experiment])
