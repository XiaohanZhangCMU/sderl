import logging
import time
import numpy as np

import torch
import torch.nn as nn
from torch.optim import Adam

MOMENTUM = 0.99
EPSILON = 1e-6
DELTA_CLIP = 50.0

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def print_parameters(model):
    for name, param in model.named_parameters():
        if param.requires_grad:
            print(name, param.shape)

class Actor(nn.Module):
    def forward(self, obs, act=None):
        raise NotImplementedError

class FeedForwardModel(object):
    """The fully connected neural network model."""
    def __init__(self, config, bsde):
        self._config = config

        self._bsde = bsde
        # make sure consistent with FBSDE equation
        self._dim = bsde.dim
        self._num_time_interval = bsde.num_time_interval
        self._total_time = bsde.total_time
        # ops for statistics update of batch normalization
        self._extra_train_ops = []

        lo = torch.tensor([self._config.y_init_range[0]], dtype=torch.float64)
        hi = torch.tensor([self._config.y_init_range[1]], dtype=torch.float64)
        y_init = torch.distributions.uniform.Uniform(lo, hi, validate_args=None)

        print('minval = {0}, maxval = {1}'.format(self._config.y_init_range[0], self._config.y_init_range[1]))

        lo = -.1 * torch.ones(1,self._dim, dtype=torch.float64)
        hi =  .1 * torch.ones(1,self._dim, dtype=torch.float64)
        z_init = torch.distributions.uniform.Uniform(lo, hi, validate_args=None)

        self.y_init_val = y_init.sample()
        self.y_init_val.requires_grad=True
        self.z_init_val = z_init.sample()
        self.z_init_val.requires_grad=True

    def train(self):
        start_time = time.time()
        # to save iteration results
        training_history = []
        # for validation
        dw_valid, x_valid = self._bsde.sample(self._config.valid_size)

        # initialization
        lr_lambda = lambda epoch: self._config.lr_values[0] if epoch < self._config.lr_boundaries[0] else self._config.lr_values[1]
        ac = self._subnetwork(self._config)
        print('number of params = ', count_parameters(ac))
        print_parameters(ac)
        optimizer = torch.optim.Adam([self.y_init_val, self.z_init_val]+list(ac.parameters()), lr=1)
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda) # what is initial learning rate?

        # begin sgd iteration
        for step in range(self._config.num_iterations+1):
            dw_train, x_train = self._bsde.sample(self._config.batch_size)
            loss = self.compute_loss(ac, dw_train, x_train)

            if step % self._config.logging_frequency == 0:
                loss = self.compute_loss(ac, dw_valid, x_valid)
                loss_val = loss.item()
                init = self.y_init_val.item()
                elapsed_time = time.time()-start_time
                training_history.append([step, loss, init, elapsed_time])
                if self._config.verbose:
                    logging.info("step: %5u,    loss: %.4e,   Y0: %.4e,  elapsed time %3u" % (
                        step, loss, init, elapsed_time))

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()

        return np.array(training_history)


    def compute_loss(self, ac, dw, x):
        time_stamp = np.arange(0, self._bsde.num_time_interval) * self._bsde.delta_t

        all_one_vec = torch.ones([dw.shape[0],1], dtype=torch.float64)
        y = all_one_vec * self.y_init_val
        z = all_one_vec * self.z_init_val

        for t in range(0, self._num_time_interval-1):
            y = y - self._bsde.delta_t * (
                self._bsde.f_tf(time_stamp[t], x[:, :, t], y, z)
            ) + torch.sum(z * dw[:, :, t], 1, keepdim=True)
            z = ac(x[:, :, t + 1]) / self._dim
        # terminal time
        y = y - self._bsde.delta_t * self._bsde.f_tf(
            time_stamp[-1], x[:, :, -2], y, z
        ) + torch.sum(z * dw[:, :, -1], 1, keepdim=True)
        delta = y - self._bsde.g_tf(self._total_time, x[:, :, -1])
        # use linear approximation outside the clipped range
        loss = torch.mean(torch.where(torch.abs(delta) < DELTA_CLIP, delta**2, 2*DELTA_CLIP * torch.abs(delta) - DELTA_CLIP**2))

        return loss


    class _subnetwork(Actor):
        def __init__(self, config):
            super().__init__()
            layers = []
            activation = nn.ReLU
            output_activation = nn.Identity
            self._config = config
            sizes = self._config.num_hiddens

            #layers += [nn.BatchNorm1d(sizes[0], affine=True)]

            for j in range(len(sizes)-1):
                act = activation if j < len(sizes)-2 else output_activation
                layers += [nn.BatchNorm1d(sizes[j], affine=True), nn.Linear(sizes[j], sizes[j+1], bias=False), act()]
                #layers += [nn.Linear(sizes[j], sizes[j+1]), act()]

            layers += [nn.BatchNorm1d(sizes[-1], affine=True)]

            self.net = nn.Sequential(*layers).double()

            def init_weights(m):
                if type(m) == nn.Linear:
                    torch.nn.init.xavier_uniform(m.weight)
                    #m.bias.data.fill_(0.01)
            self.net.apply(init_weights)

        def forward(self, x):
            z = self.net(x)
            return z

