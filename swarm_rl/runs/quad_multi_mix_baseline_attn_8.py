from sample_factory.launcher.run_description import RunDescription, Experiment, ParamGrid
from swarm_rl.runs.quad_multi_mix_baseline import QUAD_BASELINE_CLI_8

from swarm_rl.utils import timeStamped, generate_seeds

_params = ParamGrid([
<<<<<<< HEAD
    ('quads_neighbor_encoder_type', ['attention']),
    ('seed', generate_seeds(4)),
    ('num_workers', [12]),
=======
    ('seed', [0000, 1111, 2222, 3333]),
>>>>>>> 89f822fc157b97de6c65525b944e0fcf0df7f3e7
])

_experiment = Experiment(
    'quad_mix_baseline-8_mixed_attn',
    QUAD_BASELINE_CLI_8,
    _params.generate_params(randomize=False),
)

run_name = timeStamped("floor", fmt="{fname}_%Y%m%d_%H%M")

RUN_DESCRIPTION = RunDescription(run_name, experiments=[_experiment])

# For scale, need to change
# num_workers / num_envs_per_worker && quads_num_agents
# num_workers * num_envs_per_worker * quads_num_agents should not change