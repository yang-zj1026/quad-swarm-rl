import torch
import os

from glas.env import DoubleIntegrator
from glas.learning.empty_net import Empty_Net
from glas.model import Empty_Net_wAPF
from glas.params import DoubleIntegratorParam
from glas.sim import load_instance_double, run


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--rl", action='store_true', help="Run Reinforcement Learning")
    parser.add_argument("--il", action='store_true', help="Run Imitation Learning")
    parser.add_argument("--rrt", action='store_true')
    parser.add_argument("--scp", action='store_true')
    parser.add_argument("--animate", action='store_true')
    parser.add_argument("-i", "--instance", help="File instance to run simulation on")
    parser.add_argument("--batch", action='store_true',
                        help="use batch (npy) output instead of interactive (pdf) output")
    parser.add_argument("--controller", action='append', help="controller(s) to run")
    parser.add_argument("--Rsense", type=float, help="sensing radius")
    parser.add_argument("--maxNeighbors", type=int, help="maximum neighbors")

    parser.add_argument("--export", action='store_true', help="export IL model to onnx")

    parser.add_argument('--disable-cuda', action='store_true', help='Disable CUDA')

    args = parser.parse_args()
    param = DoubleIntegratorParam()
    env = DoubleIntegrator(param)

    # ckpt = torch.load('./double_integrator/il_current.pt')
    # model = Empty_Net_wAPF(param, ckpt)
    # model.empty.save_weights('./double_integrator/empty_net.pt')
    empty_net = Empty_Net(param, "DeepSet")
    empty_net.load_weights('./double_integrator/empty_net.pt')
    controllers = {
        'emptywapf': Empty_Net_wAPF(param, empty_net),
    }

    torch.set_num_threads(1)

    for file in os.listdir("./instances/test"):
        if file.endswith(".yaml") and "agents8" in file:
            map_instance_name = file.split('.')[0]
            print("Instance: ", map_instance_name)
            param.default_instance = file
            param.plots_fn = 'plots_double_{}.pdf'.format(map_instance_name)
            s0 = load_instance_double(param, env)
            run(args, param, env, controllers, s0)
