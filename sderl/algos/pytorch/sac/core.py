import numpy as np
import scipy.signal
from gym.spaces import Box, Discrete

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions.normal import Normal
from torch.distributions.categorical import Categorical

def initialize_weights_he(m):
    if isinstance(m, nn.Linear) or isinstance(m, nn.Conv2d):
        torch.nn.init.kaiming_uniform_(m.weight)
        if m.bias is not None:
            torch.nn.init.constant_(m.bias, 0)

class Flatten(nn.Module):
    def forward(self, x):
        return x.view(x.size(0), -1)

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

def cnn(obs_dim, channels=[16,32,32], activation=nn.ReLU()):
    layers = []
    sizes = [obs_dim[0]] + channels
    for j in range(len(sizes)-1):
        layers += [nn.Conv2d(sizes[j], sizes[j+1], kernel_size=3, stride=1, padding=(1,1)), activation]
    layers += [ nn.AdaptiveAvgPool2d(4), Flatten() ]  # 4 * 4 * 32 = 512
    return nn.Sequential(*layers), 4 * 4 * 32

#def cnn(obs_dim):
#    return nn.Sequential(
#            nn.Conv2d(obs_dim, 32, kernel_size=8, stride=4, padding=0),
#            nn.ReLU(),
#            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=0),
#            nn.ReLU(),
#            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=0),
#            nn.ReLU(),
#            Flatten()).apply(initialize_weights_he)

def count_vars(module):
    return sum([np.prod(p.shape) for p in module.parameters()])


LOG_STD_MAX = 2
LOG_STD_MIN = -20

class SquashedGaussianMLPActor(nn.Module):

    def __init__(self, obs_dim, act_dim, hidden_sizes, activation, act_limit, feng):
        super().__init__()
        if feng == 'mlp':
            self.fe_net = Flatten() # nn.Identity()
            feat_dim = np.prod(obs_dim)
        elif feng == 'cnn':
            self.fe_net, feat_dim = cnn(obs_dim)
        self.net = mlp([feat_dim] + list(hidden_sizes), activation, activation)
        self.mu_layer = nn.Linear(hidden_sizes[-1], act_dim)
        self.log_std_layer = nn.Linear(hidden_sizes[-1], act_dim)
        self.act_limit = act_limit

    def forward(self, obs, deterministic=False, with_logprob=True):
        net_out = self.net(self.fe_net(obs))
        mu = self.mu_layer(net_out)
        log_std = self.log_std_layer(net_out)
        log_std = torch.clamp(log_std, LOG_STD_MIN, LOG_STD_MAX)
        std = torch.exp(log_std)

        # Pre-squash distribution and sample
        pi_distribution = Normal(mu, std)
        if deterministic:
            # Only used for evaluating policy at test time.
            pi_action = mu
        else:
            pi_action = pi_distribution.rsample()

        if with_logprob:
            # Compute logprob from Gaussian, and then apply correction for Tanh squashing.
            # NOTE: The correction formula is a little bit magic. To get an understanding
            # of where it comes from, check out the original SAC paper (arXiv 1801.01290)
            # and look in appendix C. This is a more numerically-stable equivalent to Eq 21.
            # Try deriving it yourself as a (very difficult) exercise. :)
            logp_pi = pi_distribution.log_prob(pi_action).sum(axis=-1)
            logp_pi -= (2*(np.log(2) - pi_action - F.softplus(-2*pi_action))).sum(axis=1)
            #logp_pi -= (2*(np.log(2) + pi_action - F.softplus(2*pi_action))).sum(axis=1) # Use parity to simplify a bit log_pi
        else:
            logp_pi = None

        pi_action = torch.tanh(pi_action)
        pi_action = self.act_limit * pi_action

        return pi_action, logp_pi, -1


class CategoricalMLPActor(nn.Module):

    def __init__(self, obs_dim, act_dim, hidden_sizes, activation, feng):
        super().__init__()
        if feng == 'mlp':
            self.fe_net = Flatten() # nn.Identity()
            feat_dim = np.prod(obs_dim)
        elif feng == 'cnn':
            self.fe_net, feat_dim = cnn(obs_dim)

        self.logits_net = mlp([feat_dim] + list(hidden_sizes) + [act_dim], activation)

    def forward(self, obs, deterministic=False, with_logprob=True):
        logits = self.logits_net(self.fe_net(obs))
        pi_distribution = Categorical(logits=logits)

        if deterministic:
            # Only used for evaluating policy at test time.
            pi_action = torch.squeeze(torch.argmax(logits, dim=-1, keepdim=True), -1)
        else:
            pi_action = torch.squeeze(pi_distribution.sample(), -1)

        #if with_logprob:
        #    #logp_pi = pi_distribution.log_prob(pi_action)
        #    z = logits == 0.0
        #    z = z.float() * 1e-8
        #    logp_pi = torch.log(logits + z)
        #else:
        #    logp_pi = None
        if with_logprob:
            #logp_pi = pi_distribution.log_prob(pi_action)
            logp_pi = F.log_softmax(logits, dim=-1)
            return pi_action, logp_pi, torch.exp(logp_pi)
        else:
            return pi_action, None, None

        #return pi_action, logp_pi, logits

class MLPQFunction(nn.Module):

    def __init__(self, obs_dim, act_dim, hidden_sizes, activation, feng):
        super().__init__()
        if feng == 'mlp':
            self.fe_net = Flatten() # nn.Identity()
            feat_dim = np.prod(obs_dim)
        elif feng == 'cnn':
            self.fe_net, feat_dim = cnn(obs_dim)

        self.q = mlp([feat_dim+act_dim] + list(hidden_sizes) + [1], activation)
        self.act_dim = act_dim

    def forward(self, obs, act):
        q = self.q(torch.cat([self.fe_net(obs), act], dim=-1))
        return torch.squeeze(q, -1)   # Critical to ensure q has right shape.

class MLPQFunctionDiscrete(nn.Module):

    def __init__(self, obs_dim, act_dim, hidden_sizes, activation, feng):
        super().__init__()
        if feng == 'mlp':
            self.fe_net = Flatten() # nn.Identity()
            feat_dim = np.prod(obs_dim)
        elif feng == 'cnn':
            self.fe_net, feat_dim = cnn(obs_dim)

        self.q = mlp([feat_dim] + list(hidden_sizes) + [act_dim], activation)
        self.act_dim = act_dim

    def forward(self, obs, act):
        #if len(act.shape) == 1: # discrete action
        #    act = (F.one_hot(act.to(torch.int64),num_classes=self.act_dim)).to(torch.float32)
        pvec = self.q(self.fe_net(obs))
        return pvec


class MLPActorCritic(nn.Module):

    def __init__(self, observation_space, action_space, hidden_sizes=(256,256),
                 activation=nn.ReLU, feng='mlp'):
        super().__init__()

        obs_dim = observation_space.shape # [0]

        if isinstance(action_space, Box):
            act_dim = action_space.shape[0]
            act_limit = action_space.high[0]
        elif isinstance(action_space, Discrete):
            act_dim = action_space.n

        # build policy and value functions
        if isinstance(action_space, Box):
            self.pi = SquashedGaussianMLPActor(obs_dim, act_dim, hidden_sizes, activation, act_limit, feng)
            self.q1 = MLPQFunction(obs_dim, act_dim, hidden_sizes, activation, feng)
            self.q2 = MLPQFunction(obs_dim, act_dim, hidden_sizes, activation, feng)

        elif isinstance(action_space, Discrete):
            self.pi = CategoricalMLPActor(obs_dim, act_dim, hidden_sizes, activation, feng)
            self.q1 = MLPQFunctionDiscrete(obs_dim, act_dim, hidden_sizes, activation, feng)
            self.q2 = MLPQFunctionDiscrete(obs_dim, act_dim, hidden_sizes, activation, feng)

    def act(self, obs, deterministic=False):
        with torch.no_grad():
            a, _, _ = self.pi(obs, deterministic, False)
            return a.cpu().numpy()
