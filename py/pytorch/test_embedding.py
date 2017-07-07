import torch
from torch import autograd
import torch.nn.functional as F


torch.manual_seed(123)
# a = autograd.Variable(torch.rand(2, 4))
a = autograd.Variable(torch.LongTensor([
    [1, 3, 2],
    [0, 2, 1]
]))
embeddings = autograd.Variable(torch.rand(4, 3), requires_grad=True)

embed_old = torch.nn.Embedding(4, 3)
print('embed_old', embed_old)
print('embed_old(a)', embed_old(a))
embed_old.weight.data = embeddings.data
print('embed_old(a)', embed_old(a))

res = F.embedding(a, embeddings)
print('res', res)
