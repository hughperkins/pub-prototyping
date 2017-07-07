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

print('=============')
print('multiply, add arbitrary tensor')

torch.manual_seed(123)
state = autograd.Variable(torch.zeros(1))
W = autograd.Variable(torch.rand(1), requires_grad=True)
print('W', W)
inputs = []
outputs = []
seq_len = 50
for t in range(seq_len):
    input_t = autograd.Variable(torch.rand(1),
            requires_grad=True)
    state = (state + input_t) * W

    inputs.append(input_t.clone())
    outputs.append(state.clone())
out = state

out.backward(torch.ones(1))

print('W.grad', W.grad)
print('inputs', [input.data[0] for input in inputs])
print('outputs', [output.data[0] for output in outputs])

# now calcualte manually, or rather, get the computer to
# calculate it manually

# for each iteration we have, :
# out[t] = (out[t-1] + input[t]) * W
# => d(out[t]) / d(out[t-1]) = W
#    d(out[t]) / dW = out[t-1] + input[t]

# out[t] = f(out[t-1])
# out[t+1] = f(out[t])
# d(out[t])/d(out[t-1]) = d(f(out[t-1]))/d() d(out[t-1])/

# lets say we have d(out[T]) / d(out[t + 1])
# we want: d(out[T]) / d(out[t])
# we have:
# out[t + 1] = f(out[t])
#
# d(out[T]) / d(out[t]) = d(out[T]) / d(out[t+1])  d(out[t+1]) / d(out[t])
#                         ^^^ known                ^^^ calculate
# d(out[t+1]) d(out[t]) = d/d(out[t]) f(out[t])
#
# in our case: out[t+1] = f(out[t]) = (out[t] + input[t+1]) * W
# so, d/d(out[t]) = W
# so, d(out[T]) / d(out[t]) = d(out[T])/d(out[t+1]) * W

# let's calculate these:

gradOutputs = [None for i in range(seq_len)]
gradOutputs[seq_len - 1] = torch.ones(1)
for t in range(seq_len - 2, -1, -1):
    gradOutputs[t] = gradOutputs[t + 1] * W.data
print('gradOutputs', [grad[0] for grad in gradOutputs])

# out[t] = (out[t - 1] + input[t]) * W
# we want: d(out[T])/dW, for this layer
# we have:
# - d(out[T]) / d(out[t])
# - d(out[T]) / d(out[t - 1])
# - out[t] = (out[t - 1] + input[t]) * W

# by chain rule:
# d(out[T]) / dW = d(out[T]) / d(out[t]) d(out[t]) / dW
#                  ^^^ we have this        ^^^ calculate
# d(out[t]) / dW = d /dW( (out[t - 1] + input[t]) * W )
#                = out[t - 1] + input[t]

# so, for time t:
# d(out[T]) / dW = gradOutputs[t] * (outputs[t - 1] + inputs[t])
# call this: gradWeights[t]

gradWeights = [None for t in range(seq_len)]
for t in range(30, seq_len):
    output = torch.zeros(1)
    if t - 1 >= 0 and outputs[t - 1] is not None:
        output = outputs[t - 1].data
    print('t', t)
    print('output[t - 1]', output[0])
    print('input[t]', inputs[t].data[0])
    gradWeights[t] = gradOutputs[t] * (output + inputs[t].data)
print([gradWeight[0] for gradWeight in gradWeights if gradWeight is not None])
print('gradWeight', np.sum([gradWeight[0] for gradWeight in gradWeights if gradWeight is not None]))
