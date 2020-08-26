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


def wind(env, n_validation_points, path_prefix, config, idx_run):

    bsde = get_equation(env, config.dim, config.total_time, config.num_time_interval)
    dim = bsde.dim

    # Special function to avoid certain slowdowns from PyTorch + MPI combo.
    setup_pytorch_for_mpi()

    # make sure consistent with FBSDE equation

    def backward_integrate(dw, x):
        time_stamp = np.arange(0, bsde.num_time_interval) * bsde.delta_t

        y = bsde.g_tf(bsde.total_time, x[:, :, -1])

        for t in range(bsde.num_time_interval-1, -1, -1):
            z = (ac.pi(x[:, :, t]))[0] / dim
            y = y + bsde.delta_t * (
                bsde.f_tf(time_stamp[t], x[:, :, t], y, z)
            ) - torch.sum(z * dw[:, :, t], 1, keepdim=True)
        return y


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

    def accuracy(mc_x, mc_y):

        preds = np.zeros(mc_y.shape)

        with torch.no_grad():
            for i in range(mc_x.shape[0]):
                x0 = mc_x[i,:]
                dw_valid, x_valid = bsde.sample(config.valid_size, x0)
                y0 = backward_integrate(dw_valid, x_valid)
                y0 = y0.numpy().squeeze().mean()
                preds[i] = y0

            mse = ((mc_y - preds)**2).sum(axis=0).squeeze()
            bse = ((np.average(mc_y, axis=0) - preds)**2).sum(axis=0).squeeze()
            r2 = 1.0 - mse / bse
            return mse/mc_y.shape[0], r2, preds

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
    training_history = []

    start_time = time.time()

    lo = -10. * torch.ones(dim, dtype=torch.float64)
    hi =  10. * torch.ones(dim, dtype=torch.float64)
    x_init = torch.distributions.uniform.Uniform(lo, hi, validate_args=None)

    # Set up validation set
    if "y_init_analytical" in dir(bsde):
        print("Analytical solution is available for " + env)
        mc_x = np.zeros((n_validation_points, dim))
        for i in range(n_validation_points):
            x0 = x_init.sample().numpy()
            mc_x[i,:] = x0
        mc_y = bsde.y_init_analytical(mc_x)
    else:
        # If no analytical solution available, get ground truth from Monte Carlo. E.g., HJB
        print("Analytical solution is NOT available for " + env)
        print("Use hjb_mc_results_0.npy for validating HJB.")
        mc_valid = np.load('hjb_mc_results_0.npy')
        mc_x, mc_y = mc_valid[:n_validation_points,:dim], mc_valid[:n_validation_points,-1]
    assert(mc_y[-1]!=0)

    for pt in range(config.num_pts):

        if pt == 0:
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
                yinit = ac.v(x_valid[:,:,0]).mean(axis=0).item()
                mse, r2, _ = accuracy(mc_x, mc_y)
                elapsed_time = time.time()-start_time
                training_history.append([itr, loss, yinit, elapsed_time])
                if config.verbose:
                    print("\titer: %5u, loss: %.4e, Y0: %.4e, mse: %.4e, r2: %.4e, elapsed time %3u" % (
                        itr, loss, yinit, mse, r2, elapsed_time))

    np.savetxt(os.path.join(path_prefix, 'training_history_{}.csv'.format(idx_run)),
               np.array(training_history),
               fmt=['%d', '%.5e', '%.5e', '%d'],
               delimiter=",",
               header="itr,loss,target_value,elapsed_time",
               comments='')

    mse, r2, preds = accuracy(mc_x, mc_y)

    validation_results = np.zeros((mc_x.shape[0], mc_x.shape[1]+2))
    validation_results[:,:dim] = mc_x
    validation_results[:, dim] = mc_y
    validation_results[:, dim+1] = preds

    np.save(os.path.join(path_prefix, 'validation_results_{}.npy'.format(idx_run)), validation_results)


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--env', type=str, default='HJB')
    parser.add_argument('--seed', '-s', type=int, default=0)
    parser.add_argument('--cpu', type=int, default=4)
    parser.add_argument('--steps', type=int, default=4000)
    parser.add_argument('--nvp', type=int, default=4000)
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--exp_name', type=str, default='wind')
    args = parser.parse_args()

    mpi_fork(args.cpu)  # run parallel code with mpi

    from sderl.utils.run_utils import setup_logger_kwargs
    logger_kwargs = setup_logger_kwargs(args.exp_name, args.seed)

    from config import get_config
    config = get_config(args.env)

    num_run = 1

    path_prefix = os.path.join('./data/wind/', args.env)
    if not os.path.exists(path_prefix):
        os.makedirs(path_prefix,0o777)

    for idx_run in range(1, num_run+1):
        wind(args.env, args.nvp, path_prefix, config, idx_run)


