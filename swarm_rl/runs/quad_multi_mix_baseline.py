from sample_factory.launcher.run_description import RunDescription, Experiment, ParamGrid

_params = ParamGrid([
    ('quads_collision_reward', [5.0]),
])

QUAD_BASELINE_CLI_8 = (
    'python -m swarm_rl.train --env=quadrotor_multi --train_for_env_steps=500000000 --algo=APPO --use_rnn=False '
    '--num_workers=36 --num_envs_per_worker=4 --learning_rate=0.0001 --ppo_clip_value=5.0 --recurrence=1 '
    '--nonlinearity=tanh --actor_critic_share_weights=False --policy_initialization=xavier_uniform '
    '--adaptive_stddev=False --with_vtrace=False --max_policy_lag=100000000 --rnn_size=256 --with_pbt=False '
    '--gae_lambda=1.00 --max_grad_norm=5.0 --exploration_loss_coeff=0.0 --rollout=128 --batch_size=1024 '
<<<<<<< HEAD
    '--quads_use_numba=True --quads_mode=mix --quads_episode_duration=15.0 --quads_formation_size=0.0 '
    '--with_pbt=False --quads_collision_reward=5.0 '
    '--quads_neighbor_hidden_size=256 --neighbor_obs_type=pos_vel '
    '--quads_settle_reward=0.0 --quads_collision_hitbox_radius=2.0 --quads_collision_falloff_radius=4.0 '
    '--quads_local_obs=6 --quads_local_metric=dist '
    '--quads_local_coeff=1.0 --quads_num_agents=8 '
    '--quads_collision_reward=5.0 '
    '--quads_collision_smooth_max_penalty=10.0 '
    '--quads_neighbor_encoder_type=attention '
    '--replay_buffer_sample_prob=0.75 --save_milestones_sec=900 '
    '--anneal_collision_steps=300000000 --normalize_input=False --normalize_returns=False --reward_clip=10 '
    '--decorrelate_experience_max_seconds=10 --force_envs_single_thread=True '
    '--with_wandb=True --wandb_project=Quad-Swarm-RL --wandb_group=floor'
=======
    '--quads_use_numba=True --quads_num_agents=8 --quads_mode=mix --quads_episode_duration=15.0 '
    '--quads_neighbor_encoder_type=attention --quads_neighbor_hidden_size=256 --quads_neighbor_obs_type=pos_vel '
    '--quads_collision_reward=5.0 --quads_collision_hitbox_radius=2.0 --quads_collision_falloff_radius=4.0 '
    '--quads_collision_smooth_max_penalty=10.0 --quads_neighbor_visible_num=6 '
    '--replay_buffer_sample_prob=0.75 --anneal_collision_steps=300000000 --normalize_input=False '
    '--normalize_returns=False --reward_clip=10.0 --save_milestones_sec=3600'
>>>>>>> 89f822fc157b97de6c65525b944e0fcf0df7f3e7
)

_experiment = Experiment(
    'quad_mix_baseline-8_mixed',
    QUAD_BASELINE_CLI_8,
    _params.generate_params(randomize=False),
)

RUN_DESCRIPTION = RunDescription('quads_multi_mix_baseline_8a_local_v116', experiments=[_experiment])