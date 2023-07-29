from policy_distillation.agent import AgentCollection


class ExpertCollection:
    def __init__(self, envs, policy, memory, args):
        self.envs = envs
        self.num_envs = len(envs)
        self.policy = policy
        self.memory = memory
        self.expert_batch_size = args.sample_batch_size
        self.agent = AgentCollection(envs, policy, args.device, 'gpu')

    def get_expert_sample(self):
        return self.agent.get_expert_samples(self.expert_batch_size)

