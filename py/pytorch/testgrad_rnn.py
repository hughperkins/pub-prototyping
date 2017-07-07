import torch
from torch import autograd
import numpy as np
import math

torch.manual_seed(123)

a = autograd.Variable(torch.rand(1, 3))
print('a', a)

W = autograd.Variable(torch.rand(1, 3), requires_grad=True)
print('W', W)

print('=============')
print('multiply once')

out = a * W
print('out1', out)

out.backward(torch.ones(1, 3))

print('W.grad', W.grad)
# out[0] = a[0] * W[0]
# => d(out[0]) / d(W[0]) = a[0]
print('a', a)

print('=============')
print('multiply twice')

out = a * W
print('out1', out)
out = out * W
print('out2', out)

W.grad.data.fill_(0)
out.backward(torch.ones(1, 3))

print('W.grad', W.grad)
# out[0] = a[0] * W[0] * W[0]
# => d(out[0]) / d(W[0]) = 2 * a[0] * W[0]
print('2 * a * W', 2 * a * W)


print('=============')
print('multiply 50 times')

out = a
for it in range(50):
    out = out * W
print('out', out)

W.grad.data.fill_(0)
out.backward(torch.ones(1, 3))

print('W.grad', W.grad)
# out[0] = a[0] * W[0]^50
# => d(out[0]) / d(W[0]) = 50 * a[0] * W[0]^49
print('50 * a * W.pow(49)', 50 * a * W.pow(49))
