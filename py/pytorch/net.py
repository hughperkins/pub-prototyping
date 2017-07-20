import torch
from torch import autograd, nn, optim
import torch.nn.functional as F

batch_size = 5
input_size = 4
hidden_size = 4
num_classes = 4
learning_rate = 0.001

torch.manual_seed(123)
input = autograd.Variable(torch.rand(batch_size, input_size) - 0.5)
target = autograd.Variable((torch.rand(batch_size) * num_classes).long())


class Model(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super().__init__()
        self.h1 = nn.Linear(input_size, hidden_size)
        self.h2 = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        x = self.h1(x)
        x = F.tanh(x)
        x = self.h2(x)
        x = F.log_softmax(x)
        return x


model = Model(input_size=input_size, hidden_size=hidden_size, num_classes=num_classes)
opt = optim.Adam(params=model.parameters(), lr=learning_rate)


for epoch in range(1000):
    out = model(input)
    _, pred = out.max(1)
    print('target', str(target.view(1, -1)).split('\n')[1])
    print('pred', str(pred.view(1, -1)).split('\n')[1])
    loss = F.nll_loss(out, target)
    print('loss', loss.data[0])

    model.zero_grad()
    loss.backward()
    opt.step()
