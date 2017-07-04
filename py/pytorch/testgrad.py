import torch
from torch import autograd

torch.manual_seed(123)
a = torch.rand(3, 2)
print('a', a)

# a_var = autograd.Variable(a)
a_var = autograd.Variable(a, requires_grad=True)
print('a_var', a_var)
print('a_var.creator', a_var.creator)
print('a_var.grad', a_var.grad)
print('a_var.data', a_var.data)
print('a_var.requires_grad', a_var.requires_grad)

b_var = a_var + 3.5
print('b_var', b_var)
print('b_var.creator', b_var.creator)
print('b_var.grad', b_var.grad)
print('b_var.data', b_var.data)
print('b_var.requires_grad', b_var.requires_grad)

grad_out = torch.rand(3, 2)
b_var.backward(grad_out)
print(b_var.grad)
print(a_var.grad)

# W = torch.rand(3, 2)
# W_var = autograd.Variable()

W_init = torch.rand(3, 2)
W = autograd.Variable(W_init.clone(), requires_grad=True)
print('W', W)
x = autograd.Variable(torch.rand(1, 3))
h1 = x @ W
print('h1', h1)
h2 = h1 @ W.transpose(0, 1)
print('h2', h2)

grad_out = torch.rand(1, 3)
h2.backward(grad_out)
print('W.grad', W.grad)


h1 = x @ W
# print('h1', h1)
h2 = h1 @ W.transpose(0, 1)

W.grad.data.fill_(0)
h2.backward(grad_out)
print('W.grad', W.grad)

W1 = autograd.Variable(W_init.clone(), requires_grad=True)
W2 = autograd.Variable(W_init.clone(), requires_grad=True)
h1 = x @ W1
# print('h1', h1)
h2 = h1 @ W2.transpose(0, 1)

h2.backward(grad_out)
print('W1.grad', W1.grad)
print('W2.grad', W2.grad)
print('W1.grad + W2.grad', W1.grad + W2.grad)
W1.grad.data.fill_(0)
W2.grad.data.zero_()
