"""
pytorch seq2seq

designed for rapid prototyping, so trains super quickly, on just a few examples
"""
import torch
from torch import nn, autograd, optim
import numpy as np
import math
import sys
import encoding
# import data_starredwords as data
import data_anki as data


N = 100
# N = 8
N = 16
max_sentence_len = 10
# N = 4
print_every = 8  # should be even, so matches teacher_forcing == False
hidden_size = 16
# hidden_size = 1024
hidden_size = 256
# num_epochs = 16
# N = 10


training = data.Data().get_training(N=N)
training = [
    {'input': ex['first'][:max_sentence_len], 'target': ex['second'][:max_sentence_len]}
    for ex in training
]
for n in range(min(N, 16)):
    print(n, training[n])

for i, example in enumerate(training):
    example['input_encoded'] = encoding.encode_passage(example['input'])
    example['target_encoded'] = encoding.encode_passage(example['target'])

V = len(encoding.char_by_idx)
print('vocab size %s' % V)

batch_size = N  # since N is no greater than 256 for these toy models anyway
seq_len = max_sentence_len + 2  # add 2 for start/end tokens
encoder_batch = torch.LongTensor(seq_len, batch_size)
decoder_batch = torch.LongTensor(seq_len, batch_size)
print('encoder_batch.size()', encoder_batch.size())
print('decoder_batch.size()', decoder_batch.size())
for n in range(N):
    encoder_batch[:, n] = training[n]['input_encoded']
    decoder_batch[:, n] = training[n]['target_encoded']

torch.manual_seed(123)
np.random.seed(123)


class Encoder(nn.Module):
    def __init__(self, embedding):
        super().__init__()
        self.input_size = embedding.weight.size()[0]
        self.hidden_size = embedding.weight.size()[1]
        self.embedding = embedding
        self.rnn_enc = nn.RNN(
            input_size=self.hidden_size,
            hidden_size=self.hidden_size,
            num_layers=1,
            nonlinearity='tanh'
        )

    def forward(self, x, state):
        x = self.embedding(x)
        x, state = self.rnn_enc(x, state)
        return x, state


class Decoder(nn.Module):
    def __init__(self, embedding):
        super().__init__()
        self.input_size = embedding.weight.size()[0]
        self.hidden_size = embedding.weight.size()[1]
        self.embedding = embedding
        self.rnn_dec = nn.RNN(
            input_size=self.hidden_size,
            hidden_size=self.hidden_size,
            num_layers=1,
            nonlinearity='tanh'
        )

    def forward(self, x, state):
        x = self.embedding(x)
        x, state = self.rnn_dec(x, state)
        return x, state


optimizer_fn = optim.Adam
# optimizer_fn = optim.SGD

embedding = nn.Embedding(V, hidden_size)
encoder = Encoder(embedding=embedding)
decoder = Decoder(embedding=embedding)
embedding_matrix = embedding.weight

parameters = (
    set(encoder.parameters()) |
    set(decoder.parameters()) |
    set(embedding.parameters()))
opt = optimizer_fn(parameters, lr=0.001)

epoch = 0
while True:
    encoder_debug = ''
    decoder_debug = ''

    # teacher_forcing = (epoch % print_every) != 0
    teacher_forcing = epoch % 2 != 0
    printing = epoch % print_every == 0

    loss = 0
    criterion = torch.nn.NLLLoss()

    # encode
    def encode(encoder_batch, state):
        global encoder_debug

        pred_embedded, state = encoder(autograd.Variable(encoder_batch), state)
        # pred_embedded is: [seq_len][batch_size][hidden_size]
        # embedding: [input_size][hidden_size]
        pred_flat = pred_embedded.view(-1, hidden_size) @ embedding_matrix.transpose(0, 1)
        pred = pred_flat.view(seq_len, batch_size, V)
        _, v_flat = pred_flat.max(-1)
        v = v_flat.view(seq_len, batch_size)

        enc_loss = criterion(pred[:-1].view(-1, V), autograd.Variable(
            encoder_batch[1:].view(-1)))

        if printing:
            encoder_debug += 'epoch %s encoder:\n' % epoch
            for n in range(min(4, N)):
                input_sentence_verify = encoding.decode_passage(encoder_batch[:, n])
                sentence = encoding.decode_passage(v.data.cpu()[:, n])
                encoder_debug += '    [%s] => [%s]\n' % (input_sentence_verify, sentence)
        return state, enc_loss

    state = autograd.Variable(torch.zeros(1, batch_size, hidden_size))
    state, enc_loss = encode(encoder_batch, state)
    loss += enc_loss

    # decode
    if True:
        output_sentences = ['' for n in range(batch_size)]

        prev_c_batch = decoder_batch[0].view(1, -1)
        # prev_c_batch [seq_len=1][batch_size]
        for t in range(1, seq_len):
            target_c_batch = decoder_batch[t]

            pred_c_embedded_batch, state = decoder(
                autograd.Variable(prev_c_batch), state)
            pred_c_batch = pred_c_embedded_batch.view(-1, hidden_size) @ embedding_matrix.transpose(0, 1)
            _, v_batch = pred_c_batch.max(-1)
            v_batch = v_batch.data.view(1, -1)
            if printing:
                for n in range(batch_size):
                    output_sentences[n] += encoding.char_by_idx[v_batch[0][n]]
            loss += criterion(pred_c_batch, autograd.Variable(target_c_batch))

            if teacher_forcing:
                prev_c_batch = target_c_batch.view(1, -1)
            else:
                prev_c_batch = v_batch
        if printing:
            decoder_debug += 'epoch %s decoder:\n' % epoch
            if not teacher_forcing:
                for n in range(min(4, N)):
                    ex = training[n]
                    decoder_debug += '    [%s] => [%s] [%s]\n' % (
                        ex['input'], ex['target'], output_sentences[n])
    embedding.zero_grad()
    encoder.zero_grad()
    decoder.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_norm(parameters, 4.0)
    opt.step()

    if encoder_debug != '':
        print(encoder_debug)
    if decoder_debug != '':
        print(decoder_debug)

    epoch += 1
