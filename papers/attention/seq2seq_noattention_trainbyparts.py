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
# import data_starredwords as data
import data_anki as data


# try normal rnn first
# input: "this is a *test of *some stuff"
# target: "test some"
# (basically, we want any word that was prefixed by '*')

N = 100
N = 8
max_sentence_len = 10
# N = 4
print_every = 5
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
# print(training[0])
# if N > 1:
#     print(training[1])

char_by_idx = {}
idx_by_char = {}


def add_char(c):
    if c in idx_by_char:
        return idx_by_char[c]
    idx = len(idx_by_char)
    char_by_idx[idx] = c
    idx_by_char[c] = idx
    return idx


def encode_char(c):
    return idx_by_char.get(c, unk_code)


def encode_passage(sentence):
    encoded = np.zeros((len(sentence) + 2,), dtype=np.int32)
    encoded[0] = start_code
    for i, c in enumerate(sentence):
        encoded[i + 1] = add_char(c)
    encoded[len(sentence) + 1] = end_code
    return encoded


# training_encoded = []
add_char('<start>')
add_char('<end>')
add_char('<unk>')
unk_code = idx_by_char['<unk>']
start_code = encode_char('<start>')
end_code = encode_char('<end>')
for i, example in enumerate(training):
    example['input_encoded'] = encode_passage(example['input'])
    example['target_encoded'] = encode_passage(example['target'])

V = len(char_by_idx)
print('vocab size %s' % V)

# based very loosely on
# http://pytorch.org/tutorials/intermediate/seq2seq_translation_tutorial.html#the-encoder

torch.manual_seed(123)
np.random.seed(123)
embedding = nn.Embedding(V, hidden_size)
print('V', V, 'hidden_size', hidden_size)
rnn_enc = nn.RNN(
    input_size=hidden_size,
    hidden_size=hidden_size,
    num_layers=1,
    nonlinearity='tanh',
    bias=True,
)
rnn_dec = nn.RNN(
    input_size=hidden_size,
    hidden_size=hidden_size,
    num_layers=1,
    nonlinearity='tanh',
    bias=True,
)
# rnn_dec = rnn_enc

optimizer_fn = optim.Adam
# optimizer_fn = optim.SGD
opt = optimizer_fn(
    [p for p in embedding.parameters()] +
    [p for p in rnn_enc.parameters()]
    # + [p for p in rnn_dec.parameters()]
, lr=0.001)

# for epoch in range(num_epochs):
epoch = 0
while True:
    # print('epoch', epoch)
    for n, ex in enumerate(training):
        input_encoded = ex['input_encoded']
        target_encoded = ex['target_encoded']
        input_len = len(input_encoded)
        target_len = len(target_encoded)

        teacher_forcing = np.random.random() < 0.5

        loss = 0
        criterion = torch.nn.NLLLoss()

        # encode first
        def encode(input_encoded):
            enc_loss = 0
            initial_state = autograd.Variable(torch.zeros(1, 1, hidden_size))
            prev_c_encoded = autograd.Variable(
                torch.from_numpy(np.array([start_code], np.int32)).long().view(1, 1)
            )
            state = initial_state
            input_sentence_verify = ''
            sentence = ''
            # [1:] is to cut off the start token
            for t, input_c_encoded in enumerate(input_encoded[1:]):
                input_sentence_verify += char_by_idx[input_c_encoded]
                prev_c_embedded = embedding(prev_c_encoded)
                pred_c_embedded, state = rnn_enc(prev_c_embedded, state)
                pred_c = pred_c_embedded.view(-1, hidden_size) @ embedding.weight.transpose(0, 1)
                _, v = pred_c.max(-1)
                v = v.data[0][0]
                sentence += char_by_idx[v]
                if teacher_forcing or True:
                    enc_loss += criterion(pred_c, autograd.Variable(torch.LongTensor([input_c_encoded.item()])))
                prev_c_encoded = autograd.Variable(
                    torch.from_numpy(np.array([input_c_encoded], np.int32)).long().view(1, 1)
                )
            if n <= 4 and epoch % print_every == 0:
                if n == 0:
                    print('epoch', epoch, 'encoder:')
                print('    [%s] => [%s]' % (input_sentence_verify, sentence))
            return state, enc_loss

        state, enc_loss = encode(input_encoded)
        loss += enc_loss

        # now decode
        if False:
            prev_c_encoded = autograd.Variable(
                torch.from_numpy(np.array([start_code], np.int32)).long().view(1, 1)
            )

            output_sentence = ''
            # teacher_forcing = False
            for t, target_c_encoded in enumerate(target_encoded[1:]):
                # this is going to correspond approximately to
                # 'teacher forcing' in the seq2seq example
                # on the pytorch website
                prev_c_embedded = embedding(prev_c_encoded)
                pred_c_embedded, state = rnn_dec(prev_c_embedded, state)
                pred_c = pred_c_embedded.view(-1, hidden_size) @ embedding.weight.transpose(0, 1)
                _, v = pred_c.max(-1)
                v = v.data[0][0]
                output_sentence += char_by_idx[v]
                loss += criterion(pred_c, autograd.Variable(torch.LongTensor([target_c_encoded.item()])))

                if teacher_forcing:
                    prev_c_encoded = autograd.Variable(
                        torch.from_numpy(np.array([target_c_encoded], np.int32)).long().view(1, 1)
                    )
                else:
                    prev_c_encoded = autograd.Variable(
                        torch.from_numpy(np.array([v], np.int32)).long().view(1, 1)
                    )
            if n <= 1 and epoch % print_every == 0:
                if n == 0:
                    print('epoch', epoch)
                print(output_sentence)
        embedding.zero_grad()
        rnn_enc.zero_grad()
        loss.backward()
        opt.step()


        def predict_on(input_encoded):
            state = encode(input_encoded=input_encoded)

            prev_c_encoded = autograd.Variable(
                torch.from_numpy(np.array([start_code], np.int32)).long().view(1, 1)
            )
            output_sentence = ''
            # for t, _ in enumerate(target_encoded):
            for t in range(20):  # hardcode length for now...
                # this is going to correspond approximately to
                # 'teacher forcing' in the seq2seq example
                # on the pytorch website
                prev_c_embedded = embedding(prev_c_encoded)
                pred_c_embedded, state = rnn_dec(prev_c_embedded, state)
                pred_c = pred_c_embedded.view(-1, hidden_size) @ embedding.weight.transpose(0, 1)
                _, v = pred_c.max(-1)
                v = v.data[0][0]
                output_sentence += char_by_idx[v]

                prev_c_encoded = autograd.Variable(
                    torch.from_numpy(np.array([v], np.int32)).long().view(1, 1)
                )
            return output_sentence

    # if epoch % print_every == 0:
    #     print(predict_on(input_encoded=training[0]['input_encoded']))
    #     print(predict_on(input_encoded=training[1]['input_encoded']))

    epoch += 1

# if __name__ == '__main__':
#     run()
