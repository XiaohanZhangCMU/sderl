import os
os.environ['KMP_DUPLICATE_LIB_OK']='True' # This is to fix libiomp5.dylib problem running OMP on mac

import time
import pickle
import logging
import itertools
import numpy as np
import torch
from torch.optim import Adam
import sderl.algos.pytorch.wind.core as core
from sderl.utils.logx import EpochLogger
from sderl.utils.mpi_pytorch import setup_pytorch_for_mpi, sync_params, mpi_avg_grads
from sderl.utils.mpi_tools import mpi_fork, mpi_avg, proc_id, mpi_statistics_scalar, num_procs, mpi_sum
from config import get_config
from equation import get_equation
from sderl.utils.pytorchtools import EarlyStopping


MOMENTUM = 0.99
EPSILON = 1e-6
DELTA_CLIP = 50.0


def wind(env, n_train_points, n_valid_points, n_test_points, n_cpu, path_prefix, config):
    device = (torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu"))
    if torch.cuda.is_available():
        print('\nuse GPU')

    bsde = get_equation(env, config.dim, config.total_time, config.num_time_interval)
    dim = bsde.dim

    # Special function to avoid certain slowdowns from PyTorch + MPI combo.
    setup_pytorch_for_mpi()

    def backward_integrate(dw, x):
        time_stamp = np.arange(0, bsde.num_time_interval) * bsde.delta_t

        y = bsde.g_tf(bsde.total_time, x[:, :, -1])

        for t in range(bsde.num_time_interval-1, -1, -1):
            z = (ac.pi(x[:, :, t]))[0] / dim
            y = y + bsde.delta_t * (
                bsde.f_tf(time_stamp[t], x[:, :, t], y, z, device=device)
            ) - torch.sum(z * dw[:, :, t], 1, keepdim=True)
        return y

    # Set up function for computing Wind policy loss
    def compute_loss(dw, x):
        time_stamp = np.arange(0, bsde.num_time_interval) * bsde.delta_t

        y = ac.v(x[:,:,0])

        for t in range(0, bsde.num_time_interval-1):
            z = (ac.pi(x[:, :, t]))[0] / dim
            y = y - bsde.delta_t * (
                bsde.f_tf(time_stamp[t], x[:, :, t], y, z, device=device)
            ) + torch.sum(z * dw[:, :, t], 1, keepdim=True)

        # terminal time
        y = y - bsde.delta_t * bsde.f_tf(
            time_stamp[-1], x[:, :, -2], y, z, device=device
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
        return loss.cpu().item()

    def accuracy(mc_x, mc_y):

        preds = np.zeros(mc_y.shape)

        with torch.no_grad():
            #y0 = ac.v(torch.as_tensor(mc_x, dtype=torch.float64).to(device))
            dw_valid, x_valid = bsde.sample(mc_x.shape[0], mc_x)
            y0 = backward_integrate(dw_valid.to(device), x_valid.to(device))
            preds = y0.cpu().numpy().squeeze()

        mse = ((mc_y - preds)**2).sum(axis=0).squeeze()
        bse = ((np.average(mc_y, axis=0) - mc_y)**2).sum(axis=0).squeeze()
        r2 = 1.0 - mse / bse
        return mse/mc_y.shape[0], r2, preds

    # Set up optimizers for policy and value function
    ac = core.MLPActorCritic(config)
    sync_params(ac)
    ac = ac.to(device)

    lr_lambda = lambda epoch: config.lr_values[0] if epoch < config.lr_boundaries[0] else config.lr_values[1]

    print('number of params = ', core.count_parameters(ac))
    core.print_parameters(ac)

    params = itertools.chain(ac.pi.parameters(), ac.v.parameters())
    optimizer = torch.optim.Adam(params, lr=1.)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)

    if env == 'HJB':
        v_lo = -10.
        v_hi = 10.
    elif env == 'QuadraticGradients':
        v_lo = -5.
        v_hi = 5.
    elif env == 'BurgesType':
        v_lo = -5.
        v_hi = 5.
    else:
        v_lo = -1.
        v_hi = 1.

    lo = v_lo * torch.ones(dim, dtype=torch.float64)
    hi = v_hi * torch.ones(dim, dtype=torch.float64)
    x_init = torch.distributions.uniform.Uniform(lo, hi, validate_args=None)

    # Set up training set for initial x0
    train_x = np.array([ x_init.sample().numpy() for i in range(n_train_points) ])

    # Set up valida nd test set with initial x0 and ground truth
    if "y_init_analytical" in dir(bsde):
        print("Analytical solution is available for " + env)
        vc_x = np.array([x_init.sample().numpy() for i in range(n_valid_points)])
        vc_y = bsde.y_init_analytical(vc_x)
        tc_x = np.array([x_init.sample().numpy() for i in range(n_test_points)])
        tc_y = bsde.y_init_analytical(tc_x)
    else:
        print("Analytical solution is NOT available for " + env)
        print("Use hjb_mc_results_0.npy for validating HJB.")
        mc_result = np.load('hjb_mc_results_0.npy')
        vc_x, vc_y = mc_result[:n_valid_points,:dim], mc_result[:n_valid_points,-1]
        tc_x, tc_y = mc_result[n_valid_points:n_valid_points+n_test_points:,:dim], mc_result[n_valid_points:n_valid_points+n_test_points:,-1]

    # Some assumptions on inputs to make sure the code flows as expected.
    assert(vc_y[-1]!=0)
    assert(tc_y[-1]!=0)
    assert(n_train_points % config.batch_size == 0)

    vc_bse = ((np.average(vc_y, axis=0) - vc_y)**2).sum(axis=0).squeeze()
    tc_bse = ((np.average(tc_y, axis=0) - tc_y)**2).sum(axis=0).squeeze()
    print('valid bse = %.4e' % (vc_bse))
    print('test bse = %.4e' % (tc_bse))

    # Set up model saving
    training_history = []
    early_stopping = EarlyStopping(patience=config.patience, verbose=True)

    start_time = time.time()

    for epoch in range(config.train_iters):

        np.random.shuffle(train_x) # shuffle training set along first axis

        for batch_idx in range(n_train_points // config.batch_size):
            #x0_samples = np.ones((config.batch_size, dim))
            x0_samples = train_x[batch_idx * config.batch_size : (batch_idx+1) * config.batch_size, : ]
            dw_train, x_train = bsde.sample(config.batch_size, x0_samples)
            loss = update(dw_train.to(device), x_train.to(device))

        # Validation
        #dw_valid, x_valid = bsde.sample(config.batch_size, x0_samples)
        #loss = compute_loss(dw_valid, x_valid)
        #yinit = ac.v(x_valid[:,:,0].to(device)).mean(axis=0).cpu().item()
        #with torch.no_grad():
        #    yinit = backward_integrate(dw_valid.to(device), x_valid.to(device)).cpu().numpy().mean()
        #print("Y0: %.4e" % (yinit))

        mse, r2, _ = accuracy(vc_x, vc_y)
        elapsed_time = time.time()-start_time
        training_history.append([epoch, loss, mse, r2, elapsed_time])

        print("\tEpoch: %5u, loss: %.4e, mse: %.4e, r2: %.4e, elapsed time %3u" %(epoch, loss, mse, r2, elapsed_time))

        early_stopping(mse, ac)

        if early_stopping.early_stop:
            print("Early stopping")
            break

    np.savetxt(os.path.join(path_prefix,
        'progress.csv'),
               np.array(training_history),
               fmt=['%d', '%.5e', '%.5e', '%.5e', '%d'],
               delimiter=",",
               header="Epoch,loss,mse,r2,elapsed_time",
               comments='')

    mse, r2, preds = accuracy(tc_x, tc_y)
    print("\tHoldout Test: mse: %.4e, r2: %.4e" %(mse, r2))

    #validation_results = np.zeros((mc_x.shape[0], mc_x.shape[1]+2))
    #validation_results[:,:dim] = mc_x
    #validation_results[:, dim] = mc_y
    #validation_results[:, dim+1] = preds

    #np.save(os.path.join(path_prefix, 'validation_results_{}.npy'.format(idx_run)), validation_results)


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--env', type=str, default='HJB')
    parser.add_argument('--seed', '-s', type=int, default=0)
    parser.add_argument('--cpu', type=int, default=4)
    parser.add_argument('--steps', type=int, default=4000)
    parser.add_argument('--n_valid_points', type=int, default=2560)
    parser.add_argument('--n_train_points', type=int, default=10240)
    parser.add_argument('--n_test_points', type=int, default=5120)
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--exp_name', type=str, default='wind')
    parser.add_argument('--num_runs', type=int, default=1)
    args = parser.parse_args()

    entry_time = str(int(time.time()))

    mpi_fork(args.cpu)  # run parallel code with mpi

    from sderl.utils.run_utils import setup_logger_kwargs
    logger_kwargs = setup_logger_kwargs(args.exp_name, args.seed)

    from config import get_config
    config = get_config(args.env)

    for idx_run in range(1, args.num_runs+1):
        path_prefix = os.path.join('./data/wind/', args.env, 'run_'+str(idx_run)+'_'+entry_time)
        if not os.path.exists(path_prefix):
            os.makedirs(path_prefix,0o777)
        wind(args.env, args.n_train_points, args.n_valid_points, args.n_test_points, num_procs(), path_prefix, config)

