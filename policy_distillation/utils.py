import sys
import torch
from torch.distributions import Normal
from torch.distributions.kl import kl_divergence

from sample_factory.algo.utils.make_env import NonBatchedVecEnv
from sample_factory.cfg.arguments import parse_sf_args
from sample_factory.model.actor_critic import default_make_actor_critic_func, create_actor_critic
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
        cfg.rnn_size = 10

    model = create_actor_critic(cfg, obs_space, action_space)

    return model


def _kl(teacher_dist_params, student_dist_params):
    pi = Normal(loc=teacher_dist_params[0], scale=teacher_dist_params[1])
    pi_new = Normal(loc=student_dist_params[0], scale=student_dist_params[1])
    kl = torch.mean(kl_divergence(pi, pi_new))
    return kl


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.count = 0
        self.sum = 0.
        self.avg = 0.
        self.val = 0.

    def reset(self):
        self.val = 0.
        self.avg = 0.
        self.sum = 0.
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count
