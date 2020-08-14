import numpy as np
import scipy.signal
import torch
import torch.nn as nn
from torch.distributions.normal import Normal
from torch.distributions.categorical import Categorical


def combined_shape(length, shape=None):
    if shape is None:
        return (length,)
    return (length, shape) if np.isscalar(shape) else (length, *shape)

def count_vars(module):
    return sum([np.prod(p.shape) for p in module.parameters()])

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def print_parameters(model):
    for name, param in model.named_parameters():
        if param.requires_grad:
            print(name, param.shape)

def discount_cumsum(x, discount):
    """
    magic from rllab for computing discounted cumulative sums of vectors.

    input:
        vector x,
        [x0,
         x1,
         x2]

    output:
        [x0 + discount * x1 + discount^2 * x2,
         x1 + discount * x2,
         x2]
    """
    return scipy.signal.lfilter([1], [1, float(-discount)], x[::-1], axis=0)[::-1]

def mlp(sizes, activation, output_activation=nn.Identity):
    layers = []
    for j in range(len(sizes)-1):
        act = activation if j < len(sizes)-2 else output_activation
        layers += [nn.Linear(sizes[j], sizes[j+1]), act()]
    return nn.Sequential(*layers).double()


#class MLPCritic(nn.Module):
#    def __init__(self, config):
#        super().__init__()
#        sizes = config.num_hiddens + [1]
#        layers = []
#        for j in range(len(sizes)-1):
#            act = nn.ReLU if j < len(sizes)-2 else nn.Identity
#            layers += [nn.BatchNorm1d(sizes[j], affine=True), nn.Linear(sizes[j], sizes[j+1], bias=False), act()]
#        layers += [nn.BatchNorm1d(sizes[-1], affine=True)]
#
#        self.v_net = nn.Sequential(*layers).double()
#
#    def forward(self, obs):
#        return torch.squeeze(self.v_net(obs), -1) # Critical to ensure v has right shape.


#class MLPCritic(nn.Module):
#    def __init__(self, config):
#        super().__init__()
#        sizes = config.num_hiddens + [1]
#        self.mu_net = mlp(sizes, nn.ReLU)
#        log_std = -3.0 * np.ones(1, dtype=np.double)
#        self.log_std = torch.nn.Parameter(torch.as_tensor(log_std))
#
#    def forward(self, obs):
#        mu = self.mu_net(obs)
#        std = torch.exp(self.log_std)
#        return Normal(mu, std).sample()

class MLPCritic(nn.Module):
    def __init__(self, config):
        super().__init__()
        sizes = config.num_hiddens + [1]
        self.v_net = mlp(sizes, nn.ReLU)

    def forward(self, obs):
        return self.v_net(obs)


class MLPActor(nn.Module):
    def __init__(self, config):
        super().__init__()

        sizes = config.num_hiddens
        layers = []
        for j in range(len(sizes)-1):
            act = nn.ReLU if j < len(sizes)-2 else nn.Identity
            layers += [nn.BatchNorm1d(sizes[j], affine=True), nn.Linear(sizes[j], sizes[j+1], bias=False), act()]
        layers += [nn.BatchNorm1d(sizes[-1], affine=True)]
        self.pi_net = nn.Sequential(*layers).double()

        def init_weights(m):
            if type(m) == nn.Linear:
                torch.nn.init.xavier_uniform(m.weight)
        self.pi_net.apply(init_weights)

    def forward(self, obs):
        return self.pi_net(obs), 0

#class MLPActor(nn.Module):
#    def __init__(self, config):
#        super().__init__()
#
#        sizes = config.num_hiddens
#        layers = []
#        for j in range(len(sizes)-1):
#            act = nn.ReLU if j < len(sizes)-2 else nn.Identity
#            layers += [nn.BatchNorm1d(sizes[j], affine=True), nn.Linear(sizes[j], sizes[j+1], bias=False), act()]
#        layers += [nn.BatchNorm1d(sizes[-1], affine=True)]
#        self.mu_net = nn.Sequential(*layers).double()
#        log_std = -5 * np.ones(sizes[0], dtype=np.double)
#        self.log_std = torch.nn.Parameter(torch.as_tensor(log_std))
#
#        def init_weights(m):
#            if type(m) == nn.Linear:
#                torch.nn.init.xavier_uniform(m.weight)
#        self.mu_net.apply(init_weights)
#
#    def forward(self, obs):
#        mu = self.mu_net(obs)
#        std = torch.exp(self.log_std)
#        dist = Normal(mu, std)
#        act = dist.sample()
#        return act, dist.log_prob(act).sum(axis=-1)

class MLPActorCritic(nn.Module):

    def __init__(self, config):
        super().__init__()

        self.pi = MLPActor(config)
        self.v = MLPCritic(config)




