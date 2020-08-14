import os
os.environ['KMP_DUPLICATE_LIB_OK']='True' # This is to fix libiomp5.dylib problem running OMP on mac

import logging

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

    lo = torch.tensor([config.y_init_range[0]], dtype=torch.float64)
    hi = torch.tensor([config.y_init_range[1]], dtype=torch.float64)
    y_init = torch.distributions.uniform.Uniform(lo, hi, validate_args=None)

    print('minval = {0}, maxval = {1}'.format(config.y_init_range[0], config.y_init_range[1]))

    lo = -.1 * torch.ones(1,dim, dtype=torch.float64)
    hi =  .1 * torch.ones(1,dim, dtype=torch.float64)
    z_init = torch.distributions.uniform.Uniform(lo, hi, validate_args=None)

    y_init_val = y_init.sample()
    y_init_val.requires_grad=True
    z_init_val = z_init.sample()
    z_init_val.requires_grad=True

    # Set up function for computing Wind policy loss
    def compute_loss(ac, dw, x):
        time_stamp = np.arange(0, bsde.num_time_interval) * bsde.delta_t

        all_one_vec = torch.ones([dw.shape[0],1], dtype=torch.float64)
        y = all_one_vec * y_init_val
        z = all_one_vec * z_init_val

        for t in range(0, bsde.num_time_interval-1):
            y = y - bsde.delta_t * (
                bsde.f_tf(time_stamp[t], x[:, :, t], y, z)
            ) + torch.sum(z * dw[:, :, t], 1, keepdim=True)
            z = ac.pi(x[:, :, t + 1])[0] / dim
        # terminal time
        y = y - bsde.delta_t * bsde.f_tf(
            time_stamp[-1], x[:, :, -2], y, z
        ) + torch.sum(z * dw[:, :, -1], 1, keepdim=True)
        delta = y - bsde.g_tf(bsde.total_time, x[:, :, -1])
        # use linear approximation outside the clipped range
        loss = torch.mean(torch.where(torch.abs(delta) < DELTA_CLIP, delta**2, 2*DELTA_CLIP * torch.abs(delta) - DELTA_CLIP**2))
        return loss

    def update(dw_train, x_train):
        loss = compute_loss(ac, dw_train, x_train)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        scheduler.step()

    # Set up optimizers for policy and value function
    ac = core.MLPActorCritic(config)
    lr_lambda = lambda epoch: config.lr_values[0] if epoch < config.lr_boundaries[0] else config.lr_values[1]
    print('number of params = ', core.count_parameters(ac))
    core.print_parameters(ac)
    optimizer = torch.optim.Adam([y_init_val, z_init_val]+list(ac.parameters()), lr=1)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda) # what is initial learning rate?

    # Set up model saving
    training_history = []

    print('I am here 1')

    dw_valid, x_valid = bsde.sample(config.valid_size)

    # Main loop: collect experience in env and update/log each epoch
    start_time = time.time()
    print('I am here 2', config.num_iterations)
    for epoch in range(config.num_iterations+1):

        dw_train, x_train = bsde.sample(config.batch_size)

        # Perform Wind update!
        update(dw_train, x_train)

        if epoch % config.logging_frequency == 0:
            loss = compute_loss(ac, dw_valid, x_valid)
            loss_val = loss.item()
            init = y_init_val.item()
            elapsed_time = time.time()-start_time
            training_history.append([epoch, loss, init, elapsed_time])
            if config.verbose:
                print("epoch: %5u,    loss: %.4e,   Y0: %.4e,  elapsed time %3u" % (
                    epoch, loss, init, elapsed_time))


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--env', type=str, default='HJB')
    parser.add_argument('--hid', type=int, default=64)
    parser.add_argument('--l', type=int, default=2)
    parser.add_argument('--gamma', type=float, default=0.99)
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
