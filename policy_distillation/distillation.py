import argparse
import os
import random
import time
from distutils.util import strtobool
import logging
import numpy as np

import torch
import torch.nn.functional as F
import torch.optim as optim
from policy_distillation.utils import parse_swarm_cfg, make_env_non_batched, make_model, DistillationLoss
from torch.utils.tensorboard import SummaryWriter

from swarm_rl.utils import timeStamped


def args_parse():
    parser = argparse.ArgumentParser(description='Policy distillation')
    # Network, env, MDP, seed
    parser.add_argument('--env', default="QuadSwarm", metavar='G',
                        help='name of the environment to run')
    parser.add_argument('--gamma', type=float, default=0.995, metavar='G',
                        help='discount factor (default: 0.995)')
    parser.add_argument('--tau', type=float, default=0.97, metavar='G',
                        help='gae (default: 0.97)')
    parser.add_argument('--seed', type=int, default=1, metavar='N',
                        help='random seed (default: 1)')
    parser.add_argument('--load-models', action='store_true',
                        help='load_pretrained_models')

    # Teacher policy training
    parser.add_argument('--agent-count', type=int, default=10, metavar='N',
                        help='number of agents (default: 100)')
    parser.add_argument('--num-teachers', type=int, default=1, metavar='N',
                        help='number of teacher policies (default: 1)')
    parser.add_argument('--max-kl', type=float, default=1e-2, metavar='G',
                        help='max kl value (default: 1e-2)')
    parser.add_argument('--cg-damping', type=float, default=1e-2, metavar='G',
                        help='damping for conjugate gradient (default: 1e-2)')
    parser.add_argument('--cg-iter', type=int, default=10, metavar='G',
                        help='maximum iteration of conjugate gradient (default: 1e-1)')
    parser.add_argument('--l2-reg', type=float, default=1e-3, metavar='G',
                        help='l2 regularization parameter for critics (default: 1e-3)')
    parser.add_argument('--teacher-batch-size', type=int, default=1000, metavar='N',
                        help='per-iteration batch size for each agent (default: 1000)')
    parser.add_argument('--sample-batch-size', type=int, default=10000, metavar='N',
                        help='expert batch size for each teacher (default: 10000)')
    parser.add_argument('--render', action='store_true',
                        help='render the environment')
    parser.add_argument('--log-interval', type=int, default=1, metavar='N',
                        help='interval between training status logs (default: 10)')
    parser.add_argument('--device', type=str, default='cpu',
                        help='set the device (cpu or cuda)')
    parser.add_argument('--num-workers', type=int, default=10,
                        help='number of workers for parallel computing')
    parser.add_argument('--num-teacher-episodes', type=int, default=10, metavar='N',
                        help='num of teacher training episodes (default: 100)')

    # Student policy training
    parser.add_argument('--lr', type=float, default=1e-3, metavar='G',
                        help='adam learnig rate (default: 1e-3)')
    parser.add_argument('--test-interval', type=int, default=10, metavar='N',
                        help='interval between training status logs (default: 10)')
    parser.add_argument('--student-batch-size', type=int, default=1000, metavar='N',
                        help='per-iteration batch size for student (default: 1000)')
    parser.add_argument('--sample-interval', type=int, default=10, metavar='N',
                        help='frequency to update expert data (default: 10)')
    parser.add_argument('--testing-batch-size', type=int, default=10000, metavar='N',
                        help='batch size for testing student policy (default: 10000)')
    parser.add_argument('--num-student-episodes', type=int, default=1000, metavar='N',
                        help='num of teacher training episodes (default: 1000)')
    parser.add_argument('--loss-metric', type=str, default='kl',
                        help='metric to build student objective')
    parser.add_argument('--algo', type=str, default='sgd',
                        help='update method')

    args = parser.parse_args()
    return args


def policy_distillation(args, teacher_model, student_model, env, epochs=1000, batch_size=64, lr=0.001, distillation_weight=0.1):
    """Distill policy from teacher to student"""
    optimizer = optim.Adam(student_model.parameters(), lr=args.learning_rate, eps=1e-5)
    distillation_criterion = DistillationLoss()

    total_reward = 0
    obs = env.reset()
    done = False
    for epoch in range(epochs):
        while not done:
            observations, actions, rewards, dones, log_probs = [], [], [], [], []
            for _ in range(batch_size):
                obs = torch.FloatTensor(obs).unsqueeze(0)
                with torch.no_grad():
                    teacher_probs = F.softmax(teacher_model(obs), dim=1)
                student_probs = F.softmax(student_model(obs), dim=1)
                action = torch.multinomial(student_probs, 1).item()  # Sample an action from student's policy

                next_obs, reward, done, _ = env.step(action)

                observations.append(obs)
                actions.append(torch.LongTensor([action]))
                rewards.append(reward)
                dones.append(done)
                log_probs.append(torch.log(student_probs[0, action]))

                total_reward += reward
                obs = next_obs

                if done:
                    break

            observations = torch.cat(observations)
            rewards = torch.FloatTensor(rewards)
            dones = torch.FloatTensor(dones)

            # TODO: Unsqueeze obs, actions, rewards, dones


            with torch.no_grad():
                teacher_probs = F.softmax(teacher_model(observations), dim=1)

            student_probs = F.softmax(student_model(observations), dim=1)

            optimizer.zero_grad()
            distillation_loss = distillation_criterion(student_probs, teacher_probs)
            distillation_loss.backward()
            optimizer.step()


def main():
    args = args_parse()
    cfg = parse_swarm_cfg()
    run_name = timeStamped(args.experiment, "{fname}_%Y%m%d_%H%M")
    if args.track:
        import wandb
        wandb.init(
            project=args.wandb_project_name,
            entity=args.wandb_entity,
            group=args.wandb_group,
            job_type=args.wandb_job_type,
            sync_tensorboard=True,
            config=vars(args),
            name=run_name,
            monitor_gym=False,
            save_code=True,
        )

    # Setup logger
    exp_dir = os.path.join(args.train_dir, run_name)
    writer = SummaryWriter(exp_dir)
    writer.add_text(
        "hyperparameters",
        "|param|value|\n|-|-|\n%s" % ("\n".join([f"|{key}|{value}|" for key, value in vars(args).items()])),
    )

    if args.use_sf:
        from sample_factory.utils.utils import log
    else:
        log = logging.getLogger('rl')
        log.setLevel(logging.DEBUG)

    # TRY NOT TO MODIFY: seeding
    if args.seed is None:
        log.debug("Starting seed is not provided, set to 0")
        torch.manual_seed(0)
        np.random.seed(0)
        random.seed(0)
    else:
        log.debug(f"Setting fixed seed {args.seed}")
        torch.manual_seed(cfg.seed)
        np.random.seed(cfg.seed)
        random.seed(cfg.seed)

    torch.backends.cudnn.deterministic = args.torch_deterministic

    device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")

    # Model setup
    env_t = make_env_non_batched(cfg)
    teacher_agent = make_model(cfg, env_t.observation_space, env_t.action_space).to(device)
    teacher_agent.load_state_dict(torch.load(args.teacher_model_path, map_location=device))
    student_agent = make_model(cfg, env_t.observation_space, env_t.action_space, sim2real=True).to(device)
    log.debug(teacher_agent)
    log.debug(student_agent)

    env_t.close()
    del env_t








