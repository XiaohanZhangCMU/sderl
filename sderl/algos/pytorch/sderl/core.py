import numpy as np
import scipy.signal
from gym.spaces import Box, Discrete

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


class Actor(nn.Module):

    def _distribution(self, obs):
        raise NotImplementedError

    def _log_prob_from_distribution(self, pi, act):
        raise NotImplementedError

    def forward(self, obs, act=None):
        # Produce action distributions for given observations, and
        # optionally compute the log likelihood of given actions under
        # those distributions.
        pi = self._distribution(obs)
        logp_a = None
        if act is not None:
            logp_a = self._log_prob_from_distribution(pi, act)
        return pi, logp_a

class MLPGaussianActor_1(Actor): # gives y_init

    def __init__(self, config):
        super().__init__()

        self._config = config

        self.lo = torch.tensor([self._config.y_init_range[0]], dtype=torch.float64)
        self.hi = torch.tensor([self._config.y_init_range[1]], dtype=torch.float64)

    def _distribution(self, obs):
        return torch.distributions.uniform.Uniform(self.lo, self.hi, validate_args=None)

    def _log_prob_from_distribution(self, pi, act):
        return pi.log_prob(act).sum(axis=-1)    # Last axis sum needed for Torch Normal distribution

class MLPGaussianActor_2(Actor): # gives z
    def __init__(self, config):
        super().__init__()
        layers = []
        activation = nn.ReLU
        output_activation = nn.Identity
        self._config = config
        sizes = self._config.num_hiddens

        for j in range(len(sizes)-1):
            act = activation if j < len(sizes)-2 else output_activation
            layers += [nn.BatchNorm1d(sizes[j], affine=True), nn.Linear(sizes[j], sizes[j+1], bias=False), act()]

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

class MLPActorCritic(nn.Module):


    def __init__(self, config):
        super().__init__()

        self.pi = MLPGaussianActor_1(config)

        self.nu = MLPGaussianActor_2(config)

    def step(self, obs):
        with torch.no_grad():
            pi = self.pi._distribution(obs)
            a = pi.sample()
            logp_a = self.pi._log_prob_from_distribution(pi, a)
            z = self.nu(obs)
        return a.numpy(), z.numpy(), logp_a.numpy()

    def act(self, obs):
        return self.step(obs)[0]




