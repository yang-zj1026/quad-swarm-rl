import numpy as np
import torch

from glas.learning.barrier_fncs import Barrier_Fncs


class Empty_Net_wAPF:

    def __init__(self, param, env, empty):

        self.env = env
        self.empty = empty
        self.param = param
        self.bf = Barrier_Fncs(param)
        self.device = 'cpu'
        self.dim_neighbor = param.il_phi_network_architecture[0].in_features
        self.dim_action = param.il_psi_network_architecture[-1].out_features
        self.dim_state = param.il_psi_network_architecture[0].in_features - \
                         param.il_rho_network_architecture[-1].out_features - \
                         param.il_rho_obs_network_architecture[-1].out_features

    def to(self, device):
        self.device = device
        self.bf.to(device)
        self.empty.to(device)

    def __call__(self, x):

        if type(x) == torch.Tensor:

            if self.param.safety == "potential":
                P, H = self.bf.torch_get_relative_positions_and_safety_functions(x)
                barrier_action = -1 * self.param.kp * self.bf.torch_get_grad_phi(x, P, H)

                empty_action = self.empty(x)
                empty_action = self.bf.torch_scale(empty_action, self.param.pi_max)

                adaptive_scaling = self.bf.torch_get_adaptive_scaling_si(x, empty_action, barrier_action, P, H)
                action = torch.mul(adaptive_scaling, empty_action) + barrier_action
                action = self.bf.torch_scale(action, self.param.a_max)

            elif self.param.safety == "fdbk_si":
                P, H = self.bf.torch_get_relative_positions_and_safety_functions(x)
                barrier_action = self.bf.torch_fdbk_si(x, P, H)

                empty_action = self.empty(x)
                empty_action = self.bf.torch_scale(empty_action, self.param.pi_max)

                adaptive_scaling = self.bf.torch_get_adaptive_scaling_si(x, empty_action, barrier_action, P, H)
                action = torch.mul(adaptive_scaling, empty_action) + barrier_action
                action = self.bf.torch_scale(action, self.param.a_max)

            elif self.param.safety == "fdbk_di":

                P, H = self.bf.torch_get_relative_positions_and_safety_functions(x)
                barrier_action = self.bf.torch_fdbk_di(x, P, H)

                empty_action = self.empty(x)
                empty_action = self.bf.torch_scale(empty_action, self.param.pi_max)

                adaptive_scaling = self.bf.torch_get_adaptive_scaling_di(x, empty_action, barrier_action, P, H)
                action = torch.mul(adaptive_scaling, empty_action) + barrier_action
                action = self.bf.torch_scale(action, self.param.a_max)

            elif self.param.safety == "cf_si":

                P, H = self.bf.torch_get_relative_positions_and_safety_functions(x)
                barrier_action = self.bf.torch_fdbk_si(x, P, H)

                empty_action = self.empty(x)
                empty_action = self.bf.torch_scale(empty_action, self.param.pi_max)

                cf_alpha = self.bf.torch_get_cf_si(x, P, H, empty_action, barrier_action)
                action = torch.mul(cf_alpha, empty_action) + torch.mul(1 - cf_alpha, barrier_action)
                action = self.bf.torch_scale(action, self.param.a_max)

            elif self.param.safety == "cf_si_2":

                P, H = self.bf.torch_get_relative_positions_and_safety_functions(x)
                barrier_action = self.bf.torch_fdbk_si(x, P, H)

                empty_action = self.empty(x)
                empty_action = self.bf.torch_scale(empty_action, self.param.pi_max)

                cf_alpha = self.bf.torch_get_cf_si_2(x, empty_action, barrier_action, P, H)
                action = torch.mul(cf_alpha, empty_action) + torch.mul(1 - cf_alpha, barrier_action)
                action = self.bf.torch_scale(action, self.param.a_max)

            elif self.param.safety == "cf_di":

                P, H = self.bf.torch_get_relative_positions_and_safety_functions(x)
                barrier_action = self.bf.torch_fdbk_di(x, P, H)

                empty_action = self.empty(x)
                empty_action = self.bf.torch_scale(empty_action, self.param.pi_max)

                cf_alpha = self.bf.torch_get_cf_di(x, P, H, empty_action, barrier_action)
                action = torch.mul(cf_alpha, empty_action) + torch.mul(1 - cf_alpha, barrier_action)
                action = self.bf.torch_scale(action, self.param.a_max)

            elif self.param.safety == "cf_di_2":

                P, H = self.bf.torch_get_relative_positions_and_safety_functions(x)
                barrier_action = self.bf.torch_fdbk_di(x, P, H)

                empty_action = self.empty(x)
                empty_action = self.bf.torch_scale(empty_action, self.param.pi_max)

                cf_alpha = self.bf.torch_get_cf_di_2(x, empty_action, barrier_action, P, H)
                action = torch.mul(cf_alpha, empty_action) + torch.mul(1 - cf_alpha, barrier_action)
                action = self.bf.torch_scale(action, self.param.a_max)

            else:
                exit('self.param.safety: {} not recognized'.format(self.param.safety))


        elif type(x) is np.ndarray:

            if self.param.safety == "potential":
                P, H = self.bf.numpy_get_relative_positions_and_safety_functions(x)
                barrier_action = -1 * self.param.b_gamma * self.bf.numpy_get_grad_phi(x, P, H)

                empty_action = self.empty(torch.tensor(x).float()).detach().numpy()
                empty_action = self.bf.numpy_scale(empty_action, self.param.pi_max)

                adaptive_scaling = self.bf.numpy_get_adaptive_scaling_si(x, empty_action, barrier_action, P, H)
                action = adaptive_scaling * empty_action + barrier_action
                action = self.bf.numpy_scale(action, self.param.a_max)

            elif self.param.safety == "fdbk_si":

                P, H = self.bf.numpy_get_relative_positions_and_safety_functions(x)
                barrier_action = self.bf.numpy_fdbk_si(x, P, H)

                empty_action = self.empty(torch.tensor(x).float()).detach().numpy()
                empty_action = self.bf.numpy_scale(empty_action, self.param.pi_max)

                adaptive_scaling = self.bf.numpy_get_adaptive_scaling_si(x, empty_action, barrier_action, P, H)
                action = adaptive_scaling * empty_action + barrier_action
                action = self.bf.numpy_scale(action, self.param.a_max)

            elif self.param.safety == "fdbk_di":

                P, H = self.bf.numpy_get_relative_positions_and_safety_functions(x)
                barrier_action = self.bf.numpy_fdbk_di(x, P, H)

                empty_action = self.empty(torch.tensor(x).float()).detach().numpy()
                empty_action = self.bf.numpy_scale(empty_action, self.param.pi_max)

                adaptive_scaling = self.bf.numpy_get_adaptive_scaling_di(x, empty_action, barrier_action, P, H)
                action = adaptive_scaling * empty_action + barrier_action
                action = self.bf.numpy_scale(action, self.param.a_max)

            elif self.param.safety == "cf_si":

                P, H = self.bf.numpy_get_relative_positions_and_safety_functions(x)
                barrier_action = self.bf.numpy_fdbk_si(x, P, H)

                empty_action = self.empty(torch.tensor(x).float()).detach().numpy()
                empty_action = self.bf.numpy_scale(empty_action, self.param.pi_max)

                cf_alpha = self.bf.numpy_get_cf_si(x, P, H, empty_action, barrier_action)
                action = cf_alpha * empty_action + (1 - cf_alpha) * barrier_action
                action = self.bf.numpy_scale(action, self.param.a_max)

            elif self.param.safety == "cf_si_2":

                P, H = self.bf.numpy_get_relative_positions_and_safety_functions(x)
                barrier_action = self.bf.numpy_fdbk_si(x, P, H)

                empty_action = self.empty(torch.tensor(x).float()).detach().numpy()
                empty_action = self.bf.numpy_scale(empty_action, self.param.pi_max)

                cf_alpha = self.bf.numpy_get_cf_si_2(x, P, H, empty_action, barrier_action)
                action = cf_alpha * empty_action + (1 - cf_alpha) * barrier_action
                action = self.bf.numpy_scale(action, self.param.a_max)

            elif self.param.safety == "cf_di":

                P, H = self.bf.numpy_get_relative_positions_and_safety_functions(x)
                barrier_action = self.bf.numpy_fdbk_di(x, P, H)

                empty_action = self.empty(torch.tensor(x).float()).detach().numpy()
                empty_action = self.bf.numpy_scale(empty_action, self.param.pi_max)

                cf_alpha = self.bf.numpy_get_cf_di(x, P, H, empty_action, barrier_action)
                action = cf_alpha * empty_action + (1 - cf_alpha) * barrier_action
                action = self.bf.numpy_scale(action, self.param.a_max)

            elif self.param.safety == "cf_di_2":

                P, H = self.bf.numpy_get_relative_positions_and_safety_functions(x)
                barrier_action = self.bf.numpy_fdbk_di(x, P, H)

                empty_action = self.empty(torch.tensor(x).float()).detach().numpy()
                empty_action = self.bf.numpy_scale(empty_action, self.param.pi_max)

                cf_alpha = self.bf.numpy_get_cf_di_2(x, P, H, empty_action, barrier_action)
                action = cf_alpha * empty_action + (1 - cf_alpha) * barrier_action
                action = self.bf.numpy_scale(action, self.param.a_max)

            else:
                exit('self.param.safety: {} not recognized'.format(self.param.safety))

        else:
            exit('type(x) not recognized: ', type(x))

        return action

    def policy(self, x, device):

        if self.param.rollout_batch_on:
            grouping = dict()
            for i, x_i in enumerate(x):
                key = (int(x_i[0][0]), x_i.shape[1])
                if key in grouping:
                    grouping[key].append(i)
                else:
                    grouping[key] = [i]

            if len(grouping) < len(x):
                A = np.empty((len(x), self.dim_action))
                for key, idxs in grouping.items():
                    batch = torch.tensor(np.array([x[idx][0] for idx in idxs]), device=device)
                    a = self(batch)
                    a = a.detach().cpu().numpy()
                    for i, idx in enumerate(idxs):
                        A[idx, :] = a[i]

                return A

            else:
                A = np.empty((len(x), self.dim_action))
                for i, x_i in enumerate(x):
                    a_i = self(x_i)
                    A[i, :] = a_i
                return A
        else:
            A = np.empty((len(x), self.dim_action))
            for i, x_i in enumerate(x):
                a_i = self(x_i)
                A[i, :] = a_i
            return A