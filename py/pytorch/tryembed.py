import torch
from torch import nn, autograd


# batch_size = 5
# input_size = 3
hidden_size = 4
vocab_size = 7
model = nn.Embedding(vocab_size, hidden_size, padding_idx=0)

input = autograd.Variable(torch.LongTensor([0, 1, 2, 3, 4, 5, 6, 7, 8]))
print(model(input))
