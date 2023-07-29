import sys
import torch
import torch.nn as nn

from sample_factory.algo.utils.make_env import NonBatchedVecEnv
from sample_factory.cfg.arguments import parse_sf_args
from sample_factory.model.actor_critic import default_make_actor_critic_func
from swarm_rl.env_wrappers.quad_utils import make_quadrotor_env
from swarm_rl.env_wrappers.quadrotor_params import add_quadrotors_env_args, quadrotors_override_defaults
from swarm_rl.models.quad_multi_model import register_models


def parse_swarm_cfg(argv=None, evaluation=False):
    parser, partial_cfg = parse_sf_args(argv=argv, evaluation=evaluation)
    add_quadrotors_env_args(partial_cfg.env, parser)
    quadrotors_override_defaults(partial_cfg.env, parser)
    # final_cfg = parse_full_cfg(parser, argv)
    if argv is None:
        argv = sys.argv[1:]
    final_cfg, _ = parser.parse_known_args(argv)
    # final_cfg = postprocess_args(args, argv, parser)
    return final_cfg


def make_env_non_batched(cfg):
    """
        Make NonBatched Environment with configuration
    """
    env = make_quadrotor_env(cfg.env, cfg)
    env = NonBatchedVecEnv(env)

    return env


def make_model(cfg, obs_space, action_space, sim2real=False):
    """
        Initialize the model with configuration and obs_space, action space of env
    """
    register_models()

    if sim2real:
        cfg.quads_sim2real = True

    model = default_make_actor_critic_func(cfg, obs_space, action_space)

    return model


class DistillationLoss(nn.Module):
    def __init__(self, temperature=1.0):
        super(DistillationLoss, self).__init__()
        self.temperature = temperature

    def forward(self, student_probs, teacher_probs):
        # Compute the distillation loss using the KL divergence
        # between student and teacher policy probabilities
        kl_loss = -torch.sum(teacher_probs * torch.log(student_probs / (teacher_probs + 1e-8)), dim=-1)
        loss = torch.mean(kl_loss * self.temperature * self.temperature)
        return loss
