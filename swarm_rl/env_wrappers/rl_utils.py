import numpy as np
import torch
from sample_factory.algo.utils.rl_utils import calculate_discounted_sum_torch
from torch import Tensor


@torch.jit.script
def gae_advantage(rewards: Tensor, dones: Tensor, values: Tensor, valids: Tensor, gamma: float, gae_lambda: float):
    # section 3 in GAE paper: calculating advantages
    deltas = (rewards - values[:-1]) * valids[:-1] + (1 - dones) * (gae_lambda * values[1:] * valids[1:])
    advantages = calculate_discounted_sum_torch(deltas, dones, valids[:-1], gamma * gae_lambda)
    return advantages


def score_distribution(scores, beta):
    """
    Calculate the score distribution for PLR
    """
    order = scores.argsort()
    ranks = order.argsort()
    desc_ranks = len(scores) - ranks
    h_s = (1 / desc_ranks) ** (1 / beta)
    return h_s / np.sum(h_s)


def staleness_distribution(staleness, episode_counter):
    """
    Calculate the staleness distribution for PLR
    """
    staleness = episode_counter - staleness
    return staleness / np.sum(staleness)