import argparse
import os
import random
import logging
import numpy as np

import torch
from policy_distillation.student import Student
from policy_distillation.teacher import Teacher
from policy_distillation.utils import parse_swarm_cfg, make_env_non_batched, make_model
from swarm_rl.env_wrappers.quad_utils import make_quadrotor_env
from sample_factory.utils.utils import str2bool
from torch.utils.tensorboard import SummaryWriter

from swarm_rl.utils import timeStamped


def args_parse():
    parser = argparse.ArgumentParser(description='Policy distillation')
    # Network, env, MDP, seed
    parser.add_argument('--env', default="quadrotor_multi", help='name of the environment to run')
    parser.add_argument('--gamma', type=float, default=0.995, help='discount factor (default: 0.995)')
    parser.add_argument('--tau', type=float, default=0.97, help='gae (default: 0.97)')
    parser.add_argument('--seed', type=int, default=1, help='random seed (default: 1)')
    parser.add_argument('--load-models', action='store_true', help='load_pretrained_models')

    # Teacher policy
    parser.add_argument('--teacher_model_path', default='teacher.pth',
                        help='The path to load teacher model')
    parser.add_argument('--num_agents', type=int, default=12, metavar='N',
                        help='number of agents (default: 10)')
    parser.add_argument('--num_teachers', type=int, default=1, metavar='N',
                        help='number of teacher policies (default: 1)')
    parser.add_argument('--sample_batch_size', type=int, default=10000, metavar='N',
                        help='expert batch size for each teacher (default: 10000)')
    parser.add_argument('--num_workers', type=int, default=12,
                        help='number of workers for parallel computing')

    # Student policy training
    parser.add_argument('--lr', type=float, default=1e-3, metavar='G',
                        help='learnig rate (default: 1e-3)')
    parser.add_argument('--test_interval', type=int, default=5, metavar='N',
                        help='interval between training status logs (default: 10)')
    parser.add_argument('--student_batch_size', type=int, default=1000, metavar='N',
                        help='per-iteration batch size for student (default: 1000)')
    parser.add_argument('--sample_interval', type=int, default=10, metavar='N',
                        help='frequency to update expert data (default: 10)')
    parser.add_argument('--testing_batch_size', type=int, default=10000, metavar='N',
                        help='batch size for testing student policy (default: 10000)')
    parser.add_argument('--num_student_episodes', type=int, default=1000, metavar='N',
                        help='num of student training episodes (default: 1000)')

    # Training
    parser.add_argument('--train_dir', default='train_dir', help='directory to save agent logs (default: train_dir)')
    parser.add_argument('--experiment', default='policy_distillation', help='name of the experiment')
    parser.add_argument('--track', default=False, type=str2bool, help='whether to log using wandb')

    args, _ = parser.parse_known_args()
    return args


def policy_distillation(teacher, student, writer, args):
    for ep in range(args.num_student_episodes):
        if ep % args.sample_interval == 0:
            expert_data, expert_reward = teacher.get_expert_sample()
        loss = student.train(expert_data)
        writer.add_scalar('KL loss', loss.data, ep)
        print('Episode {}, KL loss: {:.2f}'.format(ep, loss.data))

        if ep % args.test_interval == 0:
            average_reward = student.test()
            writer.add_scalar('Students_average_reward', average_reward, ep)
            writer.add_scalar('teacher_reward', expert_reward, ep)
            print("Students_average_reward: {:.3f} (teacher_reaward:{:3f})".format(average_reward, expert_reward))

    print('Training student policy finished!')

    # Save student policy
    student.save_model()


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
    exp_dir = os.path.join(os.getcwd(), args.train_dir, run_name)
    if not os.path.isdir(exp_dir):
        os.makedirs(exp_dir)

    writer = SummaryWriter(exp_dir)
    writer.add_text(
        "hyperparameters",
        "|param|value|\n|-|-|\n%s" % ("\n".join([f"|{key}|{value}|" for key, value in vars(args).items()])),
    )

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
        torch.manual_seed(args.seed)
        np.random.seed(args.seed)
        random.seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Model setup
    env_t = make_env_non_batched(cfg)
    teacher_agent = make_model(cfg, env_t.observation_space, env_t.action_space).to(device)
    log.debug(teacher_agent)
    teacher_model_path = os.path.join(os.getcwd(), args.teacher_model_path)
    checkpoint_dict = torch.load(teacher_model_path, map_location=device)
    teacher_agent.load_state_dict(checkpoint_dict["model"])

    student_agent = make_model(cfg, env_t.observation_space, env_t.action_space, sim2real=True).to(device)
    log.debug(student_agent)

    env_t.close()
    del env_t

    import multiprocessing as mp
    mp.set_start_method('spawn')

    # Make quad envs
    envs = [make_quadrotor_env(cfg.env, cfg) for _ in range(args.num_agents)]
    teachers = Teacher(envs, teacher_agent, args, cfg, device)

    test_env = make_quadrotor_env(cfg.env, cfg)

    optimizer = torch.optim.SGD(student_agent.parameters(), lr=args.lr)
    students = Student(test_env, student_agent, args, cfg, optimizer, device)

    policy_distillation(teachers, students, writer, args)


if __name__ == '__main__':
    main()
