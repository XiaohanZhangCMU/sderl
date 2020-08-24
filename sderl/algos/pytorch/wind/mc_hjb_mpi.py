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
from sderl.utils.run_utils import seed_torch
from sderl.utils.mpi_tools import mpi_fork, mpi_avg, proc_id, mpi_statistics_scalar, num_procs

import random


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
    my_proc_id = proc_id()
    n_proc = num_procs()
    print('I am here 2', my_proc_id)

    seed_torch(int(my_proc_id))
    lo = -1. * torch.ones(dim, dtype=torch.float64)
    hi =  1. * torch.ones(dim, dtype=torch.float64)
    x_init_un = torch.distributions.uniform.Uniform(lo, hi, validate_args=None)

    loc = torch.zeros(bsde._dim)
    scale = torch.ones(bsde._dim)
    x_init_gs = torch.distributions.normal.Normal(loc, scale)

    sqrtT = np.sqrt(bsde._total_time)
    #print('total_time = %.4e' % bsde._total_time)
    sqrt2 = np.sqrt(2)
    bsde_lambda = 1.0
    coeff = sqrtT * sqrt2

    N = int(1e6)
    chunk_size = 10000
    num_chunks = N//chunk_size
    assert(N % chunk_size == 0)

    num_iterations = config.num_iterations//n_proc
    results = np.zeros((num_iterations, dim + 1))

    for epoch in range(num_iterations):

        # A werid way to get a valid dataset with large variation in u(x)
        d = np.random.randint(0,bsde._dim)
        e = np.random.uniform(-10,10, size=1)
        x0 = np.ones(bsde._dim)
        x0 = x0 *e

        # uniform sampling
        # if epoch == 0:
        #     x0 = np.zeros(bsde._dim)
        # else:
        #     x0 = x_init.sample().numpy()

        x0s = np.repeat(x0.reshape(1,-1),  chunk_size, axis=0)
        res = 0
        for itr in range(num_chunks):
            xwt = x0s + normal.rvs(size=[chunk_size, bsde._dim]) * coeff
            res += np.sum(np.exp(-np.log((1 + np.sum(xwt*xwt, axis=1)) / 2.0)))

        u0 = -1./bsde_lambda * np.log(res/N)
        results[epoch][:dim] = x0
        results[epoch][dim] = u0
        if epoch % 1 == 0:
            if proc_id() == 0:
                print("proc: %2u: Epoch %5u, MC for x0: %.4e" % (proc_id(), epoch, u0))
            np.save('mc_results_'+str(proc_id())+'.npy', results)

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
