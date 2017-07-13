import torch
from torch import Tensor
from torch.autograd import Variable
from torch.nn import functional as F


target = Variable(torch.LongTensor([2, 1]))

v_a = Variable(Tensor([[1,2,3], [3,4,5]]),requires_grad=True)
# t_a = v_a.data
# t_a = v_a
v_b = v_a * 2
# v_b = Variable(v_b, requires_grad=True)
# loss = Crit(v_b, target)
loss = F.cross_entropy(v_b, target)
loss.backward()

print('v_a.grad.data', v_a.grad.data)
