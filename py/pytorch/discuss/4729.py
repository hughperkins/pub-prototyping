import torch
from torch import autograd


threshold = autograd.Variable(torch.rand(1), requires_grad=True)
print('threshold', threshold)
# m = torch.nn.Threshold(threshold, threshold)
input = autograd.Variable(torch.rand(1, 5), requires_grad=True) - 0.5
print('input', input)
# out = m(input)
# out = torch.clamp(input, min=threshold)
# out = torch.cla(input, min=threshold)
out = input.max(threshold)
print('out', out)
out.backward(torch.ones(1, 5))
print('threshold.grad.data', threshold.grad.data)
