from policy_distillation.agent import AgentCollection


class Teacher:
    def __init__(self, envs, policy, args, cfg, device):
        self.envs = envs
        self.num_envs = len(envs)
        self.policy = policy
        self.expert_batch_size = args.sample_batch_size
        self.agent = AgentCollection(envs, policy, device, cfg, num_agents=args.num_agents,
                                     num_parallel_workers=args.num_workers)

    def get_expert_sample(self):
        return self.agent.get_expert_samples(self.expert_batch_size)

