import numpy as np
import torch
import torch.nn as nn


class DoubleIntegratorParam:
    def __init__(self):
        super().__init__()
        self.env_name = 'DoubleIntegrator'
        self.env_case = None

        # flags
        self.pomdp_on = True
        self.single_agent_sim = False
        self.multi_agent_sim = True
        self.il_state_loss_on = False
        self.sim_render_on = False

        # orca param
        self.n_agents = 8
        self.r_comm = 3.  # 0.5
        self.r_obs_sense = 3.
        self.r_agent = 0.15  # 0.2
        self.r_obstacle = 0.5
        self.v_max = 0.5
        self.a_max = 2.0  # .0 #2.0 # 7.5
        # self.v_max = 100
        # self.a_max = 100
        self.v_min = -1 * self.v_max
        self.a_min = -1 * self.a_max

        # sim
        self.sim_t0 = 0
        self.sim_dt = 0.025
        self.sim_tf = 100  # 2*self.sim_dt # 0.1
        self.sim_times = np.arange(self.sim_t0, self.sim_tf, self.sim_dt)
        self.sim_nt = len(self.sim_times)
        self.plots_fn = 'plots_double.pdf'

        self.max_neighbors = 6
        self.max_obstacles = 6

        self.safety = "cf_di_2"  # potential, fdbk_di, cf_di, cf_di_2
        self.rollout_batch_on = False
        self.default_instance = "map_8by8_obst6_agents8_ex0000.yaml"
        self.current_model = 'il_current.pt'

        if self.safety == "cf_di_2":  # 'working di 2' parameters
            self.pi_max = 1.5  # 0.05
            self.kp = 0.025  # 0.01
            self.kv = 1.0  # 2.0
            self.cbf_kp = 0.035  # 0.5
            self.cbf_kd = 0.5  # 2.0

        self.Delta_R = 2 * (0.5 * 0.05 + 0.5 ** 2 / (2 * 2.0))
        self.epsilon = 0.01

        # obsolete parameters
        self.b_gamma = .05
        self.b_eps = 100
        self.b_exph = 1.0
        self.D_robot = 1. * (self.r_agent + self.r_agent)
        self.D_obstacle = 1. * (self.r_agent + self.r_obstacle)
        self.circle_obstacles_on = True

        # learning hyperparameters
        n, m, h, l, p = 4, 2, 64, 16, 16  # state dim, action dim, hidden layer, output phi, output rho
        self.il_phi_network_architecture = nn.ModuleList([
            nn.Linear(4, h),
            nn.Linear(h, h),
            nn.Linear(h, l)])

        self.il_phi_obs_network_architecture = nn.ModuleList([
            nn.Linear(4, h),
            nn.Linear(h, h),
            nn.Linear(h, l)])

        self.il_rho_network_architecture = nn.ModuleList([
            nn.Linear(l, h),
            nn.Linear(h, h),
            nn.Linear(h, p)])

        self.il_rho_obs_network_architecture = nn.ModuleList([
            nn.Linear(l, h),
            nn.Linear(h, h),
            nn.Linear(h, p)])

        self.il_psi_network_architecture = nn.ModuleList([
            nn.Linear(2 * p + 4, h),
            nn.Linear(h, h),
            nn.Linear(h, m)])

        self.il_network_activation = torch.relu

        # plots
        self.vector_plot_dx = 0.3


class SingleIntegratorParam:
    def __init__(self):
        super().__init__()
        self.env_name = 'SingleIntegrator'
        self.env_case = None

        # flags
        self.pomdp_on = True
        self.single_agent_sim = False
        self.multi_agent_sim = True
        self.il_state_loss_on = False
        self.sim_render_on = False

        # orca param
        self.n_agents = 1
        self.r_comm = 3
        self.r_obs_sense = 3.
        self.r_agent = 0.15
        self.r_obstacle = 0.5

        # sim
        self.sim_t0 = 0
        self.sim_tf = 100
        self.sim_dt = 0.025
        self.sim_times = np.arange(self.sim_t0, self.sim_tf, self.sim_dt)
        self.sim_nt = len(self.sim_times)

        # safety
        self.safety = "cf_si_2"  # "potential", "fdbk_si", "cf_si"
        self.default_instance = "map_8by8_obst6_agents8_ex0003.yaml"
        self.rollout_batch_on = False

        self.max_neighbors = 6
        self.max_obstacles = 6

        self.plots_fn = 'plots_single_{}.pdf'.format(self.default_instance.split('.')[0])

        # Barrier function
        if self.safety == "cf_si_2":
            self.a_max = 0.5  # 0.5
            self.pi_max = 0.8  # 0.5
            self.kp = 1.5  # 1.0
            self.cbf_kp = 0.5

            pi_max_thresh = self.kp / (0.2 - self.r_agent) * 0.01  # 0.01 = epsilon
            print('pi_max_thresh = ', pi_max_thresh)

        self.Delta_R = 2 * self.a_max * self.sim_dt
        self.a_min = -self.a_max
        self.pi_min = -self.pi_max

        # obsolete
        self.b_gamma = 0.005
        self.b_exph = 1.0

        # old
        self.D_robot = 1. * (self.r_agent + self.r_agent)
        self.D_obstacle = 1. * (self.r_agent + self.r_obstacle)
        self.circle_obstacles_on = True  # square obstacles batch not implemented

        # learning hyperparameters
        n, m, h, l, p = 2, 2, 64, 16, 16  # state dim, action dim, hidden layer, output phi, output rho
        self.il_phi_network_architecture = nn.ModuleList([
            nn.Linear(2, h),
            nn.Linear(h, h),
            nn.Linear(h, l)])

        self.il_phi_obs_network_architecture = nn.ModuleList([
            nn.Linear(2, h),
            nn.Linear(h, h),
            nn.Linear(h, l)])

        self.il_rho_network_architecture = nn.ModuleList([
            nn.Linear(l, h),
            nn.Linear(h, h),
            nn.Linear(h, p)])

        self.il_rho_obs_network_architecture = nn.ModuleList([
            nn.Linear(l, h),
            nn.Linear(h, h),
            nn.Linear(h, p)])

        self.il_psi_network_architecture = nn.ModuleList([
            nn.Linear(2 * p + 2, h),
            nn.Linear(h, h),
            nn.Linear(h, m)])

        self.il_network_activation = torch.relu

        # plots
        self.vector_plot_dx = 0.3
