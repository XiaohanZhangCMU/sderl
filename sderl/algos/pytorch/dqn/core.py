import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from gym.spaces import Box, Discrete


def combined_shape(length, shape=None):
    if shape is None:
        return (length,)
    return (length, shape) if np.isscalar(shape) else (length, *shape)

def mlp(sizes, activation, output_activation=nn.Identity):
    layers = []
    for j in range(len(sizes)-1):
        act = activation if j < len(sizes)-2 else output_activation
        layers += [nn.Linear(sizes[j], sizes[j+1]), act()]
    return nn.Sequential(*layers)

def count_vars(module):
    return sum([np.prod(p.shape) for p in module.parameters()])


"""
Actor-Critics
"""
class MLPQFunction(nn.Module):

    def __init__(self, obs_dim, act_dim, hidden_sizes, activation=nn.ReLU):
        super().__init__()
        self.q = mlp([obs_dim] + list(hidden_sizes) + [act_dim], activation)

    def forward(self, obs):
        return self.q(obs)


class MLPActorCritic(nn.Module):

    def __init__(self, observation_space, action_space, hidden_sizes=(64, 64),
                 activation=nn.ReLU):
        super().__init__()

        obs_dim = observation_space.shape[0]
        act_dim = action_space.n

        # build policy and value functions
        self.q = MLPQFunction(obs_dim, act_dim, hidden_sizes, activation)


