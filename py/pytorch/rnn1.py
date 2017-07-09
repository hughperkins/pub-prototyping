

import torch
from torch import autograd
import numpy as np
import math


torch.manual_seed(123)

seq_len = 3
W = autograd.Variable(torch.rand(1), requires_grad=True)
inputs = [None for t in range(seq_len)]
for t in range(seq_len):
    inputs[t] = autograd.Variable(torch.rand(1), requires_grad=True)

print('W', W.data[0])
print('inputs', [input.data[0] for input in inputs])

outputs = [None for t in range(seq_len)]
for t in range(seq_len):
    prev_output = autograd.Variable(torch.zeros(1), requires_grad=True)
    if t > 0:
        prev_output = outputs[t - 1]
    outputs[t] = (prev_output + inputs[t]) * W
print('outputs', [output.data[0] for output in outputs])

gradOutputs = [None for t in range(seq_len)]
for t in range(seq_len - 1, -1, -1):
    if t == seq_len - 1:
        gradOutputs[t] = autograd.Variable(torch.ones(1))
    else:
        gradOutputs[t] = gradOutputs[t + 1] * W
print('gradOutputs', [gradOutput.data[0] for gradOutput in gradOutputs])

gradWeights = [None for t in range(seq_len)]
for t in range(seq_len - 1, -1, -1):
    prev_output = autograd.Variable(torch.zeros(1), requires_grad=True)
    if t > 0:
        prev_output = outputs[t - 1]    
    gradWeights[t] = gradOutputs[t] * (prev_output + inputs[t])
print('gradWeights', [gradWeight.data[0] for gradWeight in gradWeights])
gradWeights_overall = np.sum([gradWeight.data[0] for gradWeight in gradWeights])
print('gradWeights_overall', gradWeights_overall)
