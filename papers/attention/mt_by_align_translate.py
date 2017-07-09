import torch
from torch import nn, autograd, optim
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
# N = 16
hidden_size = 16
# hidden_size = 1024
hidden_size = 128
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
print(training[0])
if N > 1:
    print(training[1])

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


training_encoded = []
add_char('<start>')
add_char('<end>')
add_char('<unk>')
unk_code = idx_by_char['<unk>']
start_code = encode_char('<start>')
end_code = encode_char('<end>')
for i, example in enumerate(training):
    training_encoded.append({
        'input_encoded': encode_passage(example['input']),
        'target_encoded': encode_passage(example['target'])})

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
# enc_opt = optim.Adam([embedding.parameters()] + [, lr=0.001)
# dec_opt = optim.Adam(, lr=0.001)
opt = optim.Adam(
    [p for p in embedding.parameters()] +
    [p for p in rnn_enc.parameters()] +
    [p for p in rnn_dec.parameters()], lr=0.001)

# for epoch in range(num_epochs):
epoch = 0
while True:
    print('epoch', epoch)
    for n, ex in enumerate(training_encoded):
        input_encoded = ex['input_encoded']
        target_encoded = ex['target_encoded']
        input_len = len(input_encoded)
        target_len = len(target_encoded)
        # print('input_len', input_len)
        # encode first
        initial_state = autograd.Variable(torch.zeros(1, 1, hidden_size))
        # input_tensor = autograd.Variable(torch.zeros(input_len, 1))
        input_tensor = autograd.Variable(torch.from_numpy(input_encoded).long().view(-1, 1))
        # print('input_tensor.size()', input_tensor.size())
        # input_tensor[:, 0, :] = input
        # input_tensor = autograd.Variable(torch.Tensor(
        #     input.reshape(input_len, 1, V)
        # ))
        input_embedded = embedding(input_tensor)
        # print('input_embedded.size()', input_embedded.size())
        # print('input_tensor.size()', input_tensor.size())
        # print('initial_state.size()', initial_state.size())
        out, hn = rnn_enc(input_embedded, initial_state)
        # print('out.size()', out.size())
        # print('hn.size()', hn.size())
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
        prev_c_encoded = autograd.Variable(
            torch.from_numpy(np.array([start_code], np.int32)).long().view(1, 1)
        )
        # print('prev_c_encoded.size()', prev_c_encoded.size())
        loss = 0
        criterion = torch.nn.NLLLoss()

        output_sentence = ''
        for t, target_c_encoded in enumerate(target_encoded):
            # print('target_c', target_c)
            # this is going to correspond approximately to
            # 'teacher forcing' in the seq2seq example
            # on the pytorch website
            prev_c_embedded = embedding(prev_c_encoded)
            # print('prev_c_embedded.size()', prev_c_embedded.size())
            pred_c_embedded, hn = rnn_dec(prev_c_embedded, hn)
            # print('pred_c_embedded.size()', pred_c_embedded.size())
            # print('embedding.weight.size()', embedding.weight.size())
            pred_c = pred_c_embedded.view(-1, hidden_size) @ embedding.weight.transpose(0, 1)
            # print('pred_c.size()', pred_c.size())
            _, v = pred_c.max(-1)
            # print('v', v.data[0][0])
            # print(char_by_idx[v.data[0][0]])
            output_sentence += char_by_idx[v.data[0][0]]
            loss += criterion(pred_c, autograd.Variable(torch.LongTensor([target_c_encoded.item()])))

            prev_c_encoded = autograd.Variable(
                torch.from_numpy(np.array([target_c_encoded], np.int32)).long().view(1, 1)
            )
            # print('loss', loss)
            # pred_c_embedded = rnn_dec()
        # print('loss', loss)
        if n == 0:
            print(output_sentence)

        embedding.zero_grad()
        rnn_enc.zero_grad()
        rnn_dec.zero_grad()
        loss.backward()
        opt.step()
        # print(dir(rnn_enc))
        # print('rnn_enc.weight.grad', rnn_enc.weight_hh_l0.grad)
        # sys.exit(0)

        #     target_c_embedded = embedding[target_c]
        #     state = c_embedded @ W_dec + b_dec
        #     state = nn.functional.tanh(state)
        # out = state @ embedding.transpose(0, 1)
        # print('out.shape', out.shape)
        # sys.exit(0)
    epoch += 1

# if __name__ == '__main__':
#     run()
