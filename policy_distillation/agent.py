import multiprocessing as mp
import numpy as np

import torch
import math

from sample_factory.algo.utils.action_distributions import argmax_actions
from swarm_rl.env_wrappers.quad_utils import make_quadrotor_env
from policy_distillation.replay_memory import Memory


def collect_samples(pid, queue, policy, min_batch_size, agent_count, device, cfg):
    env = make_quadrotor_env(cfg.env, cfg)
    for id_a in range(agent_count):
        torch.randn(pid * agent_count + id_a)
        log = dict()
        memory = Memory()
        num_epsidoes = 0
        num_steps = 0
        total_reward = np.zeros(cfg.quads_num_agents)

        while num_steps < min_batch_size:
            obs, _ = env.reset()

            for t in range(10000):
                obs_tensor = torch.tensor(obs).to(device).float()
                with torch.no_grad():
                    model_input = {'obs': obs_tensor}
                    model_outputs = policy(model_input, None)
                    action_distribution = policy.action_distribution()
                    action = argmax_actions(action_distribution).cpu().numpy()
                    action_means, action_stds = torch.chunk(model_outputs['action_logits'], 2, dim=1)

                next_obs, rewards, dones, truncated, infos = env.step(action)
                total_reward += np.array(rewards)

                memory.push(obs, action, action_means.cpu().numpy(), action_stds.cpu().numpy(), rewards)

                if any(dones):
                    break

                obs = next_obs

            # log stats
            num_steps += (t + 1)
            num_epsidoes += 1

        log['num_steps'] = num_steps
        log['num_epsidoes'] = num_epsidoes
        log['total_reward'] = np.mean(total_reward)
        log['avg_reward'] = np.mean(total_reward) / num_epsidoes

        if queue is not None:
            queue.put([pid * agent_count + id_a, memory, log])

        else:
            return memory, log


class AgentCollection:
    def __init__(self, envs, policy, device, cfg, num_agents=1, num_parallel_workers=1):
        self.envs = envs
        self.policy = policy
        self.device = device
        self.cfg = cfg
        self.num_parallel_workers = num_parallel_workers
        self.num_agents = num_agents

    def collect_samples(self, min_batch_size):
        process_agent_count = int(math.floor(self.num_agents / self.num_parallel_workers))
        queue = mp.Queue()
        workers = []

        for i in range(self.num_parallel_workers):
            worker_args = (i, queue, self.policy, min_batch_size, process_agent_count, self.device,
                           self.cfg)
            workers.append(mp.Process(target=collect_samples, args=worker_args))

        for worker in workers:
            worker.start()

        worker_logs = [None] * self.num_agents
        worker_memories = [None] * self.num_agents
        for z in range(self.num_agents):
            pid, worker_memory, worker_log = queue.get()
            worker_memories[pid] = worker_memory
            worker_logs[pid] = worker_log
            # print("pid {}. {}".format(pid, worker_log['total_reward']))

        # worker_memories.append(memory)
        # worker_logs.append(log)

        # log['action_mean'] = np.mean(np.vstack(batch.action), axis=0)
        # log['action_min'] = np.min(np.vstack(batch.action), axis=0)
        # log['action_max'] = np.max(np.vstack(batch.action), axis=0)
        return worker_memories, worker_logs

    def get_expert_samples(self, batch_size):
        memories, logs = self.collect_samples(batch_size)
        teacher_rewards = [log['avg_reward'] for log in logs]
        teacher_average_reward = np.mean(teacher_rewards)

        dataset = []
        for memory in memories:
            batch = memory.sample()
            states = torch.from_numpy(np.stack(batch.state))
            action_means = torch.from_numpy(np.stack(batch.action_mean))
            action_stds = torch.from_numpy(np.stack(batch.action_std))
            dataset += [(state, mean, std) for state, mean, std in zip(states, action_means, action_stds)]

        return dataset, teacher_average_reward

