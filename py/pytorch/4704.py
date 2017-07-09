import torch
from torch.autograd import Variable
from torch import nn


class LSTMNet(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(LSTMNet, self).__init__()
        self.hidden_dim = hidden_dim
        self.lstm = nn.LSTM(input_dim, hidden_dim)
        self.linear = nn.Linear(hidden_dim, output_dim, bias=False)

    def forward(self, x):
        batch_size = x.size()[1]
        h0 = Variable(torch.zeros([1, batch_size, self.hidden_dim]), requires_grad=False)
        c0 = Variable(torch.zeros([1, batch_size, self.hidden_dim]), requires_grad=False)
        fx, _ = self.lstm.forward(x, (h0, c0))
        return self.linear.forward(fx[-1])

seq_length = 5   # Number of timesteps for prediction.
input_dim = 1    # Number of features
hidden_dim = 128 
batch_size = 20
output_dim = 1   # Predict a real-value feature

x = Variable(torch.rand(seq_length, batch_size, input_dim), requires_grad=False)
model = LSTMNet(seq_length, hidden_dim, output_dim)
model.forward(x)
