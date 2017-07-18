import torch
from torch import nn, autograd
import torch.nn.functional as F

a = autograd.Variable(torch.rand(1000, 10000))
while True:
    print('.')
    a = F.tanh(a)
