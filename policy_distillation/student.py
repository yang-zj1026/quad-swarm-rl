import os
import random
import torch
import numpy as np

from policy_distillation.agent import AgentCollection
from policy_distillation.utils import _kl, AverageMeter
from sample_factory.algo.utils.action_distributions import argmax_actions


class Student:
    def __init__(self, env, model, args, cfg, optimizer, device):
        self.env = env
        self.training_batch_size = args.student_batch_size
        self.testing_batch_size = args.testing_batch_size
        self.policy = model
        self.optimizer = optimizer
        self.device = device
        self.cfg = cfg

    def train(self, expert_data):
        total_loss = AverageMeter()
        np.random.shuffle(expert_data)
        for start in range(0, len(expert_data), self.training_batch_size):
            m_batch = expert_data[start:start + self.training_batch_size]
            states = torch.stack([x[0] for x in m_batch]).to(self.device).float()
            means_teacher = torch.stack([x[1] for x in m_batch]).to(self.device).float()
            logstds_teacher = torch.stack([x[2] for x in m_batch]).to(self.device).float()
            states = states.reshape(-1, states.shape[-1])
            means_teacher = means_teacher.reshape(-1, means_teacher.shape[-1])
            stds_teacher = torch.exp(logstds_teacher).reshape(-1, logstds_teacher.shape[-1])

            student_input = {'obs': states}
            students_outputs = self.policy(student_input, None)
            means_student, logstds_student = torch.chunk(students_outputs['action_logits'], 2, dim=1)
            stds_student = torch.exp(logstds_student)

            self.optimizer.zero_grad()
            # kl-divergence loss
            kl_loss = _kl([means_teacher, stds_teacher], [means_student, stds_student])
            kl_loss.backward()
            self.optimizer.step()
            total_loss.update(kl_loss.item(), len(m_batch))

        return total_loss.avg

    def test(self):
        obs, _ = self.env.reset()
        done = False
        total_rewards = np.zeros(self.cfg.quads_num_agents)
        while not done:
            obs_tensor = torch.tensor(obs).to(self.device).float()
            with torch.no_grad():
                model_input = {'obs': obs_tensor}
                model_outputs = self.policy(model_input, None)
                action_distribution = self.policy.action_distribution()
                action = argmax_actions(action_distribution).cpu().numpy()

            next_obs, rewards, dones, truncated, infos = self.env.step(action)
            total_rewards += np.array(rewards)
            obs = next_obs
            done = any(dones)
        return np.mean(total_rewards)

    def save_model(self, run_name):
        save_path = os.path.join(os.getcwd(), 'train_dir', run_name, 'checkpoint_pd.pth')
        checkpoint_dict = {
            "model": self.policy.state_dict(),
        }
        torch.save(checkpoint_dict, save_path)
        print(f"Saved model to {save_path}")
