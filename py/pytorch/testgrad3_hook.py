"""
Try using the gradient hook
"""
import torch
from torch import autograd

input = autograd.Variable(torch.rand(1, 3), requires_grad=True)
h1 = input * 3
out = h1.sum()

print('h1', h1)

out.backward(torch.ones(1))
print('input.grad', input.grad)

print('h1.grad', h1.grad)

print('=== with hook ===')

def retain_grad(var):
    def save_grad(grad):
        print('return_grad running()')
        var._grad = grad
        return grad
    var.register_hook(save_grad)

# h1.register_hook(h1, return_grad)

retain_grad(h1)
input.grad.data.zero_()
out.backward(torch.ones(1))
print('input.grad', input.grad)
print('h1.grad', h1.grad)
