# Demonstrate
# 1. Pearlmutter hvp theory works
# 2. j-towns 3 backward prop trick works

from autograd import numpy as np
import autograd.numpy.random as npr
from autograd.test_util import check_grads, check_equivalent
from autograd import (grad, elementwise_grad, jacobian, value_and_grad,
                      hessian_tensor_product, hessian, make_hvp,
                      tensor_jacobian_product, checkpoint, make_jvp,
                      make_ggnvp, grad_and_aux)
import torch
import timeit

npr.seed(1)

def jvp():
    A = np.random.randn(2, 2)
    def f(x):
        return np.dot(A, x)
    x = np.zeros(2)
    jvp_f_x = make_jvp(f)(x)
    print(jvp_f_x(np.array([1, 0])))  # f(0) and first column of f's Jacobian at 0
    print(jvp_f_x(np.array([0, 1])))  # f(0) and second column of f's Jacobian at 0


def hvp():
    hvp = make_hvp(fun)(a)[0]
    s = hvp(u)
    return s

def hessian1():
    H = hessian(fun)(a)
    s = np.dot(H, u)
    return s

# Adapted using the trick: https://j-towns.github.io/2017/06/12/A-new-trick.html
def torchhvp():
    L = torch.sum(torch.sin(x))
    y, = torch.autograd.grad(L, x, create_graph=True, retain_graph=False)
    w = torch.zeros(y.size(), requires_grad=True)
    g = torch.autograd.grad(y, x, grad_outputs = w, create_graph = True)
    r = torch.autograd.grad(g, w, grad_outputs = v, create_graph = False)
    return r

class net(torch.nn.Module):
    def __init__(self): #, D_in, H, D_out):
        super(net, self).__init__()

    def forward(self, x):
        return torch.sum(torch.sin(x))

def torchwNet():
    fun = net()
    L = fun(x)
    y, = torch.autograd.grad(L, x, create_graph=True, retain_graph=False)
    w = torch.zeros(y.size(), requires_grad=True)
    g = torch.autograd.grad(y, x, grad_outputs = w, create_graph = True)
    r = torch.autograd.grad(g, w, grad_outputs = v, create_graph = False)
    return r

fun = lambda a: np.sum(np.sin(a))
for size in range(50, 50000, 50):
    x = torch.randn(size,)
    v = torch.randn(size,)
    x.requires_grad=True

    a = x.detach().numpy()
    u = v.detach().numpy()
    print([timeit.timeit(hvp, number=10),timeit.timeit(hessian1, number=10), timeit.timeit(torchhvp, number=10), timeit.timeit(torchwNet, number=10)])

#print(fun(a))
#print(hvp())
#print(hessian1())
#print(torchhvp())
#print(torchwNet())
