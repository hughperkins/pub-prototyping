import torch
from torch import nn, optim
import torch.nn.functional as F

vocab_size = 24
embedding_size = 32

"""
inputs discrete, bunch of integers
=> embedding  nn.Embedding
=> RNN        nn.RNN
=> e2v        nn.Linear
=> outputs    .max
=> discrete predictions
"""

embedding = nn.Embedding(vocab_size, embedding_size)
rnn = nn.LSTM(embedding_size, embedding_size, batch_first=True)
e2v = nn.Linear(embedding_size, vocab_size)

integer_sequence = [23, 15, 3, 11, 12, 19, 14]
inputs = integer_sequence[: -1]
gold_outputs = integer_sequence[1:]

batch_size = 1
seq_len = len(gold_outputs)

print('inputs', inputs)
print('gold_outputs', gold_outputs)

inputs_t = torch.tensor([inputs])
opt = optim.Adam(lr=0.02, params=list(
    embedding.parameters()) + list(rnn.parameters()) + list(e2v.parameters()))

for epoch in range(10):
    x = embedding(inputs_t)
    emb_out, (h, c) = rnn(x)
    outputs = e2v(emb_out)

    outputs_flat = outputs.view(batch_size * seq_len, vocab_size)
    gold_outputs_flat = torch.tensor([gold_outputs]).view(
        batch_size * seq_len
    )
    loss = F.cross_entropy(outputs_flat, gold_outputs_flat)
    print('loss %.4f' % loss)
    opt.zero_grad()
    loss.backward()
    opt.step()

    _, preds = outputs.max(dim=-1)
    print('preds', preds)
