import torch
from torch import autograd
import math


torch.manual_seed(123)
input = autograd.Variable(torch.rand(1, 3))

seq_len = 50

state = input
W = autograd.Variable(torch.rand(1, 3), requires_grad=True)
for t in range(seq_len):
    state = W * state
out = state

print('input', input)
print('W', W)
print('state', state)

out.backward(torch.ones(1, 3))
print('W.grad', W.grad)

print('W.data[0]', W.data[0])
print('input.data[0]', input.data[0])
print('50 * input[0] * W[0].pow(49)', 50 * input.data[0] * W.data[0].pow(49))

"""
seq_len 1

out = W * input
dout / dW = input

seq_len 2

out = W * W * input
dout / dW = 2 * input * W

seq_len 50

out = input * W^50
dout /dW = 50 * input * W^49

"""
