import os
os.environ['KMP_DUPLICATE_LIB_OK']='True' # This is to fix libiomp5.dylib problem running OMP on mac

import logging
import collections

import itertools
import numpy as np
import torch
from torch.optim import Adam
import time
import sderl.algos.pytorch.wind.core as core
from sderl.utils.logx import EpochLogger
from sderl.utils.mpi_pytorch import setup_pytorch_for_mpi, sync_params, mpi_avg_grads
from sderl.utils.mpi_tools import mpi_fork, mpi_avg, proc_id, mpi_statistics_scalar, num_procs
from config import get_config
from equation import get_equation
from scipy.stats import multivariate_normal as normal

MOMENTUM = 0.99
EPSILON = 1e-6
DELTA_CLIP = 50.0


def mc(bsde, config):

    # Special function to avoid certain slowdowns from PyTorch + MPI combo.
    setup_pytorch_for_mpi()

    # make sure consistent with FBSDE equation
    dim = bsde.dim

    # Main loop: collect experience in env and update/log each epoch
    start_time = time.time()
    print('I am here 2', config.num_iterations)

    lo = -10 * torch.ones(dim, dtype=torch.float64)
    hi =  10 * torch.ones(dim, dtype=torch.float64)
    x_init = torch.distributions.uniform.Uniform(lo, hi, validate_args=None)

    sqrtT = np.sqrt(bsde._total_time)
    print('total_time = %.4e' % bsde._total_time)
    sqrt2 = np.sqrt(2)
    bsde_lambda = 1.0
    coeff = sqrtT * sqrt2
    N = int(1e7)

    results = np.zeros((config.num_iterations, dim + 1))

    for epoch in range(config.num_iterations):

        #x0 = np.zeros(bsde._dim) # x_init.sample().numpy()
        if epoch == 0:
            x0 = np.zeros(bsde._dim) # x_init.sample().numpy()
        else:
            x0 = x_init.sample().numpy()
        res = 0

        for itr in range(N):

            dw_sample = normal.rvs(size=[bsde._dim]) * coeff

            xwt = x0 + dw_sample

            g = np.log((1 + np.inner(xwt,xwt)) / 2.0)

            res += np.exp(-bsde_lambda * g)

            if itr % 100000 == 0:
                elapsed_time = time.time() - start_time
                u0 = -1./bsde_lambda * np.log(res/N)
                print("itr: %5u, Y0: %.4e, elapsed_time: %3u" % (itr, u0, elapsed_time))

        u0 = -1./bsde_lambda * np.log(res/N)
        results[epoch][:dim] = x0
        results[epoch][dim] = u0
        if epoch % 1 == 0:
            print("Epoch %5u, MC for x0: %.4e" % (epoch, u0))
            np.save('mc_results.npy', results)

    elapsed_time = time.time() - start_time
    print('Time elapsed: %3u' % elapsed_time )


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--env', type=str, default='HJB')
    parser.add_argument('--seed', '-s', type=int, default=0)
    parser.add_argument('--cpu', type=int, default=4)
    parser.add_argument('--steps', type=int, default=4000)
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--exp_name', type=str, default='wind')
    args = parser.parse_args()

    mpi_fork(args.cpu)  # run parallel code with mpi

    from sderl.utils.run_utils import setup_logger_kwargs
    logger_kwargs = setup_logger_kwargs(args.exp_name, args.seed)

    from config import get_config
    config = get_config(args.env)

    bsde = get_equation(args.env, config.dim, config.total_time, config.num_time_interval)
    mc(bsde, config)
