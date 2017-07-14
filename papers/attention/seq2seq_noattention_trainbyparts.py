"""
In this version, we'll train each language model separately,
or at least teacher-force-train the encoder, not just the decoder

The link between encoder and decoder does seem pretty brittle though...
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
N = 4
print_every = 2
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
    for n, ex in enumerate(training):
        input_encoded = ex['input_encoded']
        target_encoded = ex['target_encoded']
        input_len = len(input_encoded)
        target_len = len(target_encoded)

        teacher_forcing = (epoch % print_every) != 0

        loss = 0
        criterion = torch.nn.NLLLoss()

        # encode
        def encode(input_encoded, state):
            global encoder_debug
            enc_loss = 0
            prev_c = encoding.start_code
            input_sentence_verify = ''
            sentence = ''
            # [1:] is to cut off the start token
            # [:-1] is to cut off end token too :-)
            for t, input_c in enumerate(input_encoded[1:]):
                input_c = input_c.item()
                input_sentence_verify += encoding.char_by_idx[input_c]
                pred_c_embedded, state = encoder(autograd.Variable(torch.LongTensor([[prev_c]])), state)
                pred_c = pred_c_embedded.view(-1, hidden_size) @ embedding_matrix.transpose(0, 1)
                _, v = pred_c.max(-1)
                v = v.data[0][0]
                sentence += encoding.char_by_idx[v]
                # want to force encoder to build language model a bit faster than
                # if it has to wait only for gradient from decoder:
                enc_loss += criterion(pred_c, autograd.Variable(
                    torch.LongTensor([input_c])))
                prev_c = input_c
            if n <= 4 and epoch % print_every == 0:
                if n == 0:
                    encoder_debug += 'epoch %s encoder:\n' % epoch
                encoder_debug += '    [%s] => [%s]\n' % (input_sentence_verify, sentence)
            return state, enc_loss

        state = autograd.Variable(torch.zeros(1, 1, hidden_size))
        state, enc_loss = encode(input_encoded, state)
        loss += enc_loss

        # decode
        if True:
            prev_c = encoding.start_code

            output_sentence = ''
            for t, target_c in enumerate(target_encoded[1:]):
                target_c = target_c.item()

                pred_c_embedded, state = decoder(
                    autograd.Variable(torch.LongTensor([[prev_c]])), state)
                pred_c = pred_c_embedded.view(-1, hidden_size) @ embedding_matrix.transpose(0, 1)
                _, v = pred_c.max(-1)
                v = v.data[0][0]
                output_sentence += encoding.char_by_idx[v]
                loss += criterion(pred_c, autograd.Variable(torch.LongTensor(
                    [target_c])))

                if teacher_forcing:
                    prev_c = target_c
                else:
                    # if we're already wrong, let's just abandon...
                    if target_c != v:
                        break
                    prev_c = v
            if n <= 1 and epoch % print_every == 0:
                if n == 0:
                    decoder_debug += 'epoch %s decoder:\n' % epoch
                if not teacher_forcing:
                    decoder_debug += '    [%s] => [%s] [%s]\n' % (
                        ex['input'], ex['target'], output_sentence)
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
