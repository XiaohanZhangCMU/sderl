# Demonstrate:
# Pearlmutter hvp works for core.mlp

import numpy as np
import torch
import gym
from torch.optim import Adam
import time
import sderl.algos.pytorch.dppo.core as core
from sderl.utils.mpi_pytorch import setup_pytorch_for_mpi, sync_params, mpi_avg_grads
from sderl.utils.mpi_tools import mpi_fork, mpi_avg, proc_id, mpi_statistics_scalar, num_procs

def torchwNet1(): # Add 2nd order gradients
    class net(torch.nn.Module):
        def __init__(self): #, D_in, H, D_out):
            super(net, self).__init__()

        def forward(self, x):
            return torch.sum(torch.sin(x))

    x = torch.randn(8,)
    v = torch.randn(8,)
    x.requires_grad=True
    fun = net()
    L = fun(x)
    y, = torch.autograd.grad(L, x, create_graph=True, retain_graph=False)
    w = torch.zeros(y.size(), requires_grad=True)
    g = torch.autograd.grad(y, x, grad_outputs = w, create_graph = True)
    r = torch.autograd.grad(g, w, grad_outputs = v, create_graph = False)


def torchwNet(epsilon = 0.1): # Add 2nd order gradients
    for l in ac.pi.logits_net:
        for x in l.parameters():
            y, = torch.autograd.grad(loss_pi, x, create_graph=True, retain_graph=True)
            w = torch.zeros(y.size(), requires_grad=True)
            g, = torch.autograd.grad(y, x, grad_outputs = w, create_graph = True)
            r, = torch.autograd.grad(g, w, grad_outputs = y, create_graph = False)
            x.grad = y + epsilon * r

# Test Pytorch layer schema for mannually interfering parameters` gradient

observation_space = gym.spaces.Box(-np.inf, np.inf, shape=(8,), dtype=np.float32)
action_space = gym.spaces.Discrete(4)

ac_kwargs=dict()

ac = core.MLPActorCritic(observation_space, action_space, **ac_kwargs)
sync_params(ac)

#obs = 3* torch.rand(1,8)
#act = 3* torch.rand(1)
#torch.save(obs, 'obs.pt')
#torch.save(act, 'act.pt')
obs = torch.load('obs.pt')
act = torch.load('act.pt')

pi_optimizer = Adam(ac.pi.parameters(), lr=0.001)
pi_optimizer.zero_grad()

pi, logp = ac.pi(obs, act)
loss_pi = logp.mean()

print('Before Backward Propagation')
for l in ac.pi.logits_net:
    for x in l.parameters():
        print(x.data[0][0])
        print(x.grad)
        break
    break

# Compute gradients d loss_pi/dw
#loss_pi.backward(retain_graph=True)

print('Before Maunipulating Gradients')
for l in ac.pi.logits_net:
    for x in l.parameters():
        print(x.data[0][0])
        print(x.grad)
        break
    break

# Modify gradients with customized values
torchwNet(epsilon = 0)

# Update weights according to Adam schema
pi_optimizer.step()

# Print out updated weights
print('After One Step of Minimization')
for l in ac.pi.logits_net:
    for x in l.parameters():
        print(x.data[0][0])
        print(x.grad[0][0])
        break
    break
