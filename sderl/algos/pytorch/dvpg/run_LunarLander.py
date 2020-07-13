from sderl.utils.run_utils import ExperimentGrid
from sderl import dvpg_pytorch
import torch

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--cpu', type=int, default=2)
    parser.add_argument('--num_runs', type=int, default=10)
    args = parser.parse_args()

    eg = ExperimentGrid(name='dvpg_LunarLander-v2')
    eg.add('env_name', 'LunarLander-v2', '', True)
    eg.add('seed', [10*i for i in range(args.num_runs)])
    eg.add('epochs', 500)
    eg.add('epsilon', [0.01, 0.1, 1, 10])
    eg.add('pi_lr', [3e-4, 3e-3, 3e-2])
    eg.add('steps_per_epoch', 4000)
    eg.add('ac_kwargs:hidden_sizes', [(32,32)], 'hid')
    eg.add('ac_kwargs:activation', [torch.nn.Tanh], '')
    eg.run(dvpg_pytorch, num_cpu=args.cpu)
