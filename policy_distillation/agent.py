import multiprocessing
from policy_distillation.replay_memory import Memory
import numpy as np

import torch
import math


def collect_samples(pid, queue, env, policy, mean_action, min_batch_size, agent_count):
    for id_a in range(agent_count):
        torch.randn(pid * agent_count + id_a)
        log = dict()
        memory = Memory()
        num_steps = 0
        total_reward = 0
        min_reward = 1e6
        max_reward = -1e6
        num_episodes = 0

        while num_steps < min_batch_size:
            obs = env.reset()
            reward_episode = 0

            for t in range(10000):
                state_var = torch.tensor(obs).unsqueeze(0)
                with torch.no_grad():
                    # TODO: select action deterministicly
                    action = policy.select_action(state_var, deterministic=True)

                next_obs, reward, done, _ = env.step(action)
                reward_episode += reward

                mask = 0 if done else 1

                memory.push(obs, action, mask, next_obs, reward)

                if done:
                    break

                obs = next_obs

            # log stats
            num_steps += (t + 1)
            num_episodes += 1
            total_reward += reward_episode
            min_reward = min(min_reward, reward_episode)
            max_reward = max(max_reward, reward_episode)

        log['num_steps'] = num_steps
        log['num_episodes'] = num_episodes
        log['total_reward'] = total_reward
        log['avg_reward'] = total_reward / num_episodes
        log['max_reward'] = max_reward
        log['min_reward'] = min_reward

        if queue is not None:
            queue.put([pid * agent_count + id_a, memory, log])

        else:
            return memory, log


def merge_log(log_list):
    log = dict()
    log['total_reward'] = sum([x['total_reward'] for x in log_list])
    log['num_episodes'] = sum([x['num_episodes'] for x in log_list])
    log['num_steps'] = sum([x['num_steps'] for x in log_list])
    log['avg_reward'] = log['total_reward'] / log['num_episodes']
    log['max_reward'] = max([x['max_reward'] for x in log_list])
    log['min_reward'] = min([x['min_reward'] for x in log_list])

    return log


class AgentCollection:
    def __init__(self, envs, policy, device, mean_action=False, render=False, running_state=None,
                 num_agents=1, num_parallel_workers=1):
        self.envs = envs
        self.policy = policy
        self.device = device
        self.mean_action = mean_action
        self.running_state = running_state
        self.render = render
        self.num_parallel_workers = num_parallel_workers
        self.num_agents = num_agents

    def collect_samples(self, min_batch_size):
        process_agent_count = int(math.floor(self.num_agents / self.num_parallel_workers))
        queue = multiprocessing.Queue()
        workers = []

        for i in range(self.num_parallel_workers):
            worker_args = (i, queue, self.envs[i], self.policy, self.mean_action,
                           min_batch_size, process_agent_count)
            workers.append(multiprocessing.Process(target=collect_samples, args=worker_args))

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
        teacher_average_reward = np.array(teacher_rewards).mean()

        dataset = []
        for memory, in memories:
            batch = memory.sample()
            states = torch.from_numpy(np.stack(batch.state)).to(torch.double).to('cpu')
            action_probs = torch.from_numpy(np.stack(batch.action_probs)).to(torch.double).to('cpu')
            dataset += [(state, mean, std) for state, mean, std in zip(states, action_probs)]

        return dataset, teacher_average_reward

