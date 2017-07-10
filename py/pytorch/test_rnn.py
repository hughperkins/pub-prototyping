import torch
from torch import autograd, nn


batch_size = 1
seq_len = 7
input_size = 6
hidden_size = 4

example = [3, 2, 0, 0, 4, 5, 1, 1]

# input = autograd.Variable(torch.rand(seq_len, batch_size, input_size))
# print('input.size()', input.size())

embedding = nn.Embedding(input_size, hidden_size)
rnn = torch.nn.RNN(
    input_size=hidden_size,
    hidden_size=hidden_size,
    num_layers=1,
    nonlinearity='tanh'
)
# criterion = torch.nn.
# print('rnn', rnn)

input = autograd.Variable(
    torch.LongTensor(example[:-1]).view(seq_len, batch_size)
)
target = autograd.Variable(
    torch.LongTensor(example[1:]).view(seq_len, batch_size)
)
print('input', input)

parameters = [p for p in rnn.parameters()] + [p for p in embedding.parameters()]
optimizer = torch.optim.Adam(parameters)

epoch = 0
while True:
    embedded_input = embedding(input)
    state = autograd.Variable(torch.zeros(1, batch_size, hidden_size))
    out, state = rnn(embedded_input, state)
    # print('out.size()', out.size())
    # print('embedding.weight.transpose(0, 1).size()', embedding.weight.transpose(0, 1).size())
    out_unembedded = out.view(-1, hidden_size) @ embedding.weight.transpose(0, 1)
    _, pred = out_unembedded.max(1)
    # out_unembedded = out_unembedded.view(seq_len, batch_size, input_size)
    # print('out_unembedded.size()', out_unembedded.size())
    # print('target.size()', target.size())
    loss = torch.nn.functional.nll_loss(out_unembedded, target.view(-1))
    # print('epoch %s loss %s' % (epoch, loss.data[0]))
    if epoch % 500 == 0:
        print('epoch', epoch)
        print('input', input.data.view(1, -1))
        print('target', target.data.view(1, -1))
        print('pred', pred.data.view(1, -1))
    # print('out', out.data.view(1, -1))
    rnn.zero_grad()
    embedding.zero_grad()
    loss.backward()
    optimizer.step()
    # print('out.size()', out.size())
    # print('state.size()', state.size())
    epoch += 1
