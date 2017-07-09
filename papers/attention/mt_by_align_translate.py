import torch
from torch import nn, autograd
import numpy as np
import math
import sys


# try normal rnn first
# input: "this is a *test of *some stuff"
# target: "test some"
# (basically, we want any word that was prefixed by '*')

words = 'this is a test of some foo bar paris london whatever near far'.split(' ')

# training_sources = [
#     {'input': 'this is a test of some stuff']}
# ]
training = []
num_available_words = len(words)
N = 100
hidden_size = 16
num_epochs = 16
# N = 10

class Rnn(nn.Module):
    def __init__(self):
        pass

    def forward(self, x):
        pass


torch.manual_seed(123)
np.random.seed(123)
for n in range(N):
    num_words = np.random.choice(range(3,8))
    chosen_word_idxes = np.random.choice(num_available_words, num_words, replace=True)
    chosen_words = []
    for i in chosen_word_idxes:
        chosen_words.append(words[i])
    num_starred = np.random.choice(range(1, num_words + 1))
    starred = set(np.random.choice(num_words, num_starred, replace=False))
    for i in starred:
        chosen_words[i] = '*' + chosen_words[i]
    sentence = ' '.join(chosen_words)
    # print('sentence [%s]' % sentence)
    target_words = []
    for i in starred:
        target_words.append(chosen_words[i][1:])
    target_sentence = ' '.join(target_words)
    # print('target [%s]' % target_sentence)
    training.append({'input': sentence, 'target': target_sentence})

char_by_idx = {}
idx_by_char = {}


def add_char(c):
    if c in idx_by_char:
        return idx_by_char[c]
    idx = len(idx_by_char)
    char_by_idx[idx] = c
    idx_by_char[c] = idx
    return idx


def encode(sentence):
    encoded = np.zeros((len(sentence),), dtype=np.int32)
    for i, c in enumerate(sentence):
        encoded[i] = add_char(c)
    return encoded


training_encoded = []
for i, example in enumerate(training):
    training_encoded.append({
        'input': encode(example['input']), 'target': encode(example['target'])})

# print(training_encoded[0])
# print(training_encoded[1])

# def run():

V = len(char_by_idx)
print('vocab size %s' % V)

# W_enc = autograd.Variable(torch.rand(num_hidden, num_hidden) - 0.5)
# b_enc = autograd.Variable(torch.rand(num_hidden) - 0.5) * 0.05

# W_dec = autograd.Variable(torch.rand(num_hidden, num_hidden) - 0.5)
# b_dec = autograd.Variable(torch.rand(num_hidden) - 0.5) * 0.05

# embedding = autograd.Variable(torch.rand(V, num_hidden))

# based very loosely on
# http://pytorch.org/tutorials/intermediate/seq2seq_translation_tutorial.html#the-encoder

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

for epoch in range(num_epochs):
    for n, ex in enumerate(training_encoded):
        input = ex['input']
        target = ex['target']
        input_len = len(input)
        target_len = len(target)
        print('input_len', input_len)
        # encode first
        initial_state = autograd.Variable(torch.zeros(1, 1, hidden_size))
        # input_tensor = autograd.Variable(torch.zeros(input_len, 1))
        input_tensor = autograd.Variable(torch.from_numpy(input).long().view(-1, 1))
        print('input_tensor.size()', input_tensor.size())
        # input_tensor[:, 0, :] = input
        # input_tensor = autograd.Variable(torch.Tensor(
        #     input.reshape(input_len, 1, V)
        # ))
        input_embedded = embedding(input_tensor)
        print('input_embedded.size()', input_embedded.size())
        # print('input_tensor.size()', input_tensor.size())
        print('initial_state.size()', initial_state.size())
        out, hn = rnn_enc.forward(input_embedded, initial_state)
        print('out.size()', out.size())
        print('hn.size()', hn.size())
        # for i, v in enumerate(input):
        # state = autograd.Variable(torch.zeros(num_hidden))
        # for t, c in enumerate(input):
        #     c_embedded = embedding[c]
        #     state = c_embedded @ W_enc + b_enc
        #     state = nn.functional.tanh(state)
        # print('encoder output', state[:10].view(1, -1))
        # if n >= 3:
        #     sys.exit(0)

        # now decode
        # for t, target_c in enumerate(target):
        #     target_c_embedded = embedding[target_c]
        #     state = c_embedded @ W_dec + b_dec
        #     state = nn.functional.tanh(state)
        # out = state @ embedding.transpose(0, 1)
        # print('out.shape', out.shape)
        sys.exit(0)


# if __name__ == '__main__':
#     run()
