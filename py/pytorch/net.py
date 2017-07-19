import torch
from torch import autograd, nn
import torch.nn.functional as F

batch_size = 5
input_size = 3
hidden_size = 4
num_classes = 2

torch.manual_seed(123)
input = autograd.Variable(torch.rand(batch_size, input_size))
print('input', input)


class Model(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super().__init__()
        self.h1 = nn.Linear(input_size, hidden_size)
        self.h2 = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        x = self.h1(x)
        x = F.tanh(x)
        x = self.h2(x)
        return x


model = Model(input_size=input_size, hidden_size=hidden_size, num_classes=num_classes)
out = model(input)
print('out', out)
