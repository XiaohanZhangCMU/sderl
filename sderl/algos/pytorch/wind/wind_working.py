import os
os.environ['KMP_DUPLICATE_LIB_OK']='True' # This is to fix libiomp5.dylib problem running OMP on mac

import pickle
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

MOMENTUM = 0.99
EPSILON = 1e-6
DELTA_CLIP = 50.0


def wind(bsde, config):

    # Special function to avoid certain slowdowns from PyTorch + MPI combo.
    setup_pytorch_for_mpi()

    # make sure consistent with FBSDE equation
    dim = bsde.dim

    # Set up function for computing Wind policy loss
    def compute_loss(dw, x):
        time_stamp = np.arange(0, bsde.num_time_interval) * bsde.delta_t

        y = ac.v(x[:,:,0])

        for t in range(0, bsde.num_time_interval-1):
            z = (ac.pi(x[:, :, t]))[0] / dim
            y = y - bsde.delta_t * (
                bsde.f_tf(time_stamp[t], x[:, :, t], y, z)
            ) + torch.sum(z * dw[:, :, t], 1, keepdim=True)

        # terminal time
        y = y - bsde.delta_t * bsde.f_tf(
            time_stamp[-1], x[:, :, -2], y, z
        ) + torch.sum(z * dw[:, :, -1], 1, keepdim=True)
        delta = y - bsde.g_tf(bsde.total_time, x[:, :, -1])
        # use linear approximation outside the clipped range
        loss = torch.mean(torch.where(torch.abs(delta) < DELTA_CLIP, delta**2, 2*DELTA_CLIP * torch.abs(delta) - DELTA_CLIP**2))
        return loss

    def update(dw_train, x_train):
        optimizer.zero_grad()
        loss = compute_loss(dw_train, x_train)
        loss.backward()
        optimizer.step()
        scheduler.step()

    def accuracy():
        with torch.no_grad():
            preds = (ac.v(torch.as_tensor(mc_x[:300, :], dtype=torch.float64))).numpy()
            error = np.mean(np.abs(preds - mc_y))
            return error

    # Set up optimizers for policy and value function
    ac = core.MLPActorCritic(config)
    sync_params(ac)

    lr_lambda = lambda epoch: config.lr_values[0] if epoch < config.lr_boundaries[0] else config.lr_values[1]

    print('number of params = ', core.count_parameters(ac))
    core.print_parameters(ac)

    params = itertools.chain(ac.pi.parameters(), ac.v.parameters())
    optimizer = torch.optim.Adam(params, lr=1.)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda) # what is initial learning rate?

    # Set up model saving
    training_history = collections.defaultdict()

    mc_valid = np.load('mc_results_0.npy')

    mc_x, mc_y = mc_valid[:300,:dim], mc_valid[:300,-1]

    # Main loop: collect experience in env and update/log each epoch
    start_time = time.time()

    lo = -1. * torch.ones(dim, dtype=torch.float64)
    hi =  1. * torch.ones(dim, dtype=torch.float64)
    x_init = torch.distributions.uniform.Uniform(lo, hi, validate_args=None)

    for epoch in range(config.num_iterations+1):

        if epoch == 0:
            x0 = np.zeros(dim)
        else:
            x0 = x_init.sample().numpy()

        dw_valid, x_valid = bsde.sample(config.valid_size, x0)

        for itr in range(config.train_iters):
            dw_train, x_train = bsde.sample(config.batch_size, x0)

            # Perform Wind update!
            update(dw_train, x_train)

            if itr % config.logging_frequency == 0:
                loss = compute_loss(dw_valid, x_valid)
                loss_val = loss.item()
                init = ac.v(x_valid[:,:,0]).mean(axis=0).item()
                elapsed_time = time.time()-start_time
                training_history[epoch]=[itr, loss, init, elapsed_time]
                if config.verbose:
                    print("\titer: %5u, loss: %.4e, Y0: %.4e, elapsed time %3u" % (
                        itr, loss, init, elapsed_time))
        print("epoch: %5u, error: %.4e, mc_y: %.4e, init: %.4e" % (epoch, accuracy(), mc_y[epoch], init))
        with open('training_history.pickle', 'wb') as handle:
            pickle.dump(training_history, handle, protocol=pickle.HIGHEST_PROTOCOL)


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
    wind(bsde, config)
