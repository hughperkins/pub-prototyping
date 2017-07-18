import torch
from torch import autograd





"""

t in {0, ... seq_len - 1}
- for each of these t, exists:
   - input[t]
   - output[t]
   

out[T]

for each layer, we want:
- d(out[T])/d(W[t])

we have:
- out[t] = (out[t-1] + input[t]) * W

first calcualte:
- d(out[T])/d(out[t-1])

d(out[T])/d(out[t-1]) = d(out[T])/d(out[t]) d(out[t])/d(out[t-1])
                        ^^^ from layer above  ^^^ calculate

d(out[t])/d(out[t-1]) = W


d(out[T])/ d(W[t]) = d(out[T]) / d(out[t]) d(out[t]) / d(W[t])
                      ^^^ have this already  ^^^ calculate

d(out[T]) / d(W[t]) = d/dW((out[t-1] + input[t]) * W)
                    = out[t-1] + input[t]
"""


torch.manual_seed(123)
seq_len = 2

inputs = [None for t in range(seq_len)]
for t in range(1, seq_len):
    inputs[t] = autograd.Variable(torch.rand(1))

states = [None for t in range(seq_len)]
states[0] = autograd.Variable(torch.zeros(1))
W = autograd.Variable(torch.rand(1), requires_grad=True)
for t in range(1, seq_len):
    states[t] = (states[t-1] + inputs[t]) * W
out = states[seq_len - 1]

print('out', out)

out.backward(torch.ones(1))
print('W.grad', W.grad)

print('W', W)
gradOutputs = [None for t in range(seq_len)]   # gradOutputs[t] = d(out[T]) / d(out[t])
gradWeights = [None for t in range(seq_len)]   # gradWeights[t] = d(out[T]) / d(W[t])
gradOutputs[seq_len - 1] = torch.ones(1)
for t in range(seq_len - 2, -1, -1):
    gradOutputs[t] = gradOutputs[t + 1] * W.data
    prevOutput = 
    gradWeights[t] = gradOutputs[t] * (states[t-1] + inputs[t].data)
# print('gradOutputs', [gradOutput[0] for gradOutput in gradOutputs])

print('gradWeights', [gradWeight[0] for gradWeight in gradWeights if gradWeight is not None])
