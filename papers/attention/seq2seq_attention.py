import torch
from torch import nn, autograd, optim
import torch.nn.functional as F
import numpy as np
import math
import sys
import time
import encoding
# import data_starredwords as data
import data_anki as data

# mapping of notation in 'align and translate' paper to variables in this code:
#   i => t for decoder
#   j => t for encoder
#   T => seq_len - 1 (our t is 0-based)
#   t => t (but ours is 0-based, not 1-based)
#   h => encoder state
#   s => decoder state

# mapping of variable names here, to notation in 'align and translate' paper:
#   hidden_size => n
#   N => N/A (I dont think they use batching, or consider multiple examples)
#   seq_len - 1 => T (but t=1...T is 1-based, whereas our t=0...,seq_len-1 is 0-based)
#

max_sentence_len = 20
max_sentence_len = 6
N = 64
N = 4
print_every = 16  # should be even, so matches teacher_forcing == False
hidden_size = 128
num_layers = 2


training = []
while len(training) < N:
    new_training = data.Data().get_training(N=N-len(training))
    new_training = [
        {'input': ex['first'][:max_sentence_len], 'target': ex['second'][:max_sentence_len]}
        for ex in new_training
        if len(ex['first']) >= max_sentence_len and len(ex['second']) >= max_sentence_len
    ]
    print('len(new_training)', len(new_training))
    training += new_training
print('len(training)', len(training))
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


def cudafy(x):
    if torch.cuda.is_available():
        return x.cuda()
    return x


encoder_batch = cudafy(encoder_batch)
decoder_batch = cudafy(decoder_batch)


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
            num_layers=num_layers,
            nonlinearity='tanh',
            bidirectional=True
        )

    def forward(self, x, state):
        x = self.embedding(x)
        x, state = self.rnn_enc(x, state)
        return x, state


class AlignmentModel(nn.Module):
    def __init__(self, seq_len, hidden_size):
        super().__init__()
        self.seq_len = seq_len
        self.hidden_size = hidden_size

        self.W = nn.Parameter(
            torch.rand(hidden_size, hidden_size) * 0.1)
        self.U = nn.Parameter(
            torch.rand(hidden_size, 2 * hidden_size) * 0.1)
        self.v = nn.Parameter(
            torch.rand(hidden_size) * 0.1)

    def forward(self, enc_out, prev_dec_state):
        """
        Input to this is:
        - from the encoder, we have a batch of outputs, ie:
            [seq_len, batch_size, hidden_size * 2]
            (* 2, because bidirectional)
        - from the decoder, we have the previous state, also batched:
            [batch_size, hidden_size]
            (just batch_size, since not bidirectional)

        Then, conceptually, we want to do something like:
        - for each of the timesteps, t=0, ..., seq_len - 1, from the encoder
          outputs, do something like dot product with the decoder state
        - then, we'll get a distribution over the timesteps t of the encoder
        - and we softmax it, so it really is a probability distribution

        We do this for each member of the batch

        So, the output will have dimensions, to within a transpose, of:
          [seq_len][batch_size]

        In terms of the decoder, we'll only evaluate a single timestep here,
        although can be batched. Therefore decoder timestep doesnt figure in the
        output dimensinos

        In terms of hidden_size, since we're dot-producting that out, and getting
        a distribution over seq_len, for each of the batch_size, it doesnt
        figure in the output dimensions

        So, overall we're going to have:

        [seq_len][batch_size] = f(
            [seq_len, batch_size, hidden_size * 2],
            [batch_size, hidden_size]

        The dot products are between, for each member of batch_size, a single
        timestep from the encoder, and the decoder previous state vector,
        which is hidden_size long. there is no seq_len in the decoder input to
        this forward

        We're going to call this forward for each timestep in the decoder,
        producing a distribution over the seq_len timesteps t of the encoder
        each time, and also batched

        As far as flattening, impelmenting this, we can flatten out the
        incoming encoder outputs to:
        [seq_len * batch_size][hidden_size * 2]

        ... then concatenate with the decoder prev state, and pass through
        a Linear layer ... oh, not sure if that'll work
        Lets just use the formula in the paper :-)

        Formula in the paper is:

        x[t_enc] = v dot tanh(W @ dec_prev_state + U @ enc_out[t_enc])

        Then:
        x = soft_max(x)

        For us, this is all batched, so:
        - dec_prev_state is [batch_size][hidden_size]
        - enc_out is [seq_len][batch_size][hidden_size * 2]

        And we have:
        - W is [hidden_size][hidden_size]
        - U is [hidden_size][hidden_size * 2]
        - v is [hidden_size]

        So, for W @ dec_prev_state, we can do:
        - dec_prev_state @ W
          => [batch_size][hidden_size]
        For U @ enc_out:
        - enc_out @ U.transpose(0, 1)   (flattening enc_out appropriatley, then unflattening)
          => [seq_len][batch_size][hidden_size]
        - to add these, we need to expand out dec_prev_state @ W to:
              [seq_len][batch_size][hidden_size]
          ... then we can add
          ... take tanh
          ... dot with v giving:
             [seq_len][batch_size]
          Then we need to transpose to:
              [batch_size][seq_len]
            ... take softmax
         We are done :-)
        """

        prev_dec_state = prev_dec_state.view(batch_size, hidden_size)
        prev_dec_state_W = prev_dec_state @ self.W
        enc_out_U = enc_out.view(seq_len * batch_size, hidden_size * 2) @ \
            self.U.transpose(0, 1)
        enc_out_U = enc_out_U.view(seq_len, batch_size, hidden_size)
        prev_dec_state_W_exp = prev_dec_state_W \
            .view(1, batch_size, hidden_size) \
            .expand(seq_len, batch_size, hidden_size)
        x = F.tanh(enc_out_U + prev_dec_state_W_exp)
        x = x.view(seq_len * batch_size, hidden_size) @ self.v.view(-1, 1)
        x = x.view(seq_len, batch_size)
        x = x.transpose(0, 1)
        x = F.softmax(x)
        return x


class Decoder(nn.Module):
    def __init__(self, embedding):
        super().__init__()
        self.input_size = embedding.weight.size()[0]
        self.hidden_size = embedding.weight.size()[1]
        self.embedding = embedding
        self.context_layer = nn.Linear(self.hidden_size * 2, self.hidden_size)
        self.rnn_dec = nn.RNN(
            input_size=self.hidden_size,
            hidden_size=self.hidden_size,
            num_layers=num_layers,
            nonlinearity='tanh'
        )

    def forward(self, input, context, state):
        input = self.embedding(input)
        context = self.context_layer(context)
        dec_out, state = self.rnn_dec(input + context, state)
        return dec_out, state


optimizer_fn = optim.Adam
# optimizer_fn = optim.SGD

embedding = nn.Embedding(V, hidden_size)
encoder = Encoder(embedding=embedding)
alignment_model = AlignmentModel(seq_len=seq_len, hidden_size=hidden_size)
decoder = Decoder(embedding=embedding)
embedding_matrix = embedding.weight

cudafy(embedding)
cudafy(encoder)
cudafy(alignment_model)
cudafy(decoder)

parameters = (
    set(encoder.parameters()) |
    set(decoder.parameters()) |
    set(embedding.parameters()))
opt = optimizer_fn(parameters, lr=0.001)

epoch = 0
last_print = time.time()
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

        pred_embedded, state = encoder(cudafy(autograd.Variable(encoder_batch)), state)

        enc_loss = 0

        # calc loss for forward direction:
        pred_flat = pred_embedded[:, :, :hidden_size].contiguous().view(
            -1, hidden_size) @ embedding_matrix.transpose(0, 1)
        pred = pred_flat.view(seq_len, batch_size, V)
        _, v_flat = pred_flat.max(-1)
        v_forward = v_flat.view(seq_len, batch_size)
        # loss is based on comparing:
        # - prediction for timestep t, with
        # - input for timestep t + 1
        # so if we have timesteps t:
        # 0  1  2  3
        # i0 i1 i2 i3  i is 'input'
        # p0 p1 p2 p3  p is prediction
        # we should match:
        # - p0 == i1
        # - p1 == i2
        # - ...
        # - p[seq_len - 2] == i[seq_len-1]
        enc_loss += criterion(pred[:-1].view(-1, V), cudafy(autograd.Variable(
            encoder_batch[1:].view(-1))))

        # and backward...
        pred_flat = pred_embedded[:, :, hidden_size:].contiguous().view(
            -1, hidden_size) @ embedding_matrix.transpose(0, 1)
        pred = pred_flat.view(seq_len, batch_size, V)
        _, v_flat = pred_flat.max(-1)
        v_backward = v_flat.view(seq_len, batch_size)
        enc_loss += criterion(pred[1:].view(-1, V), cudafy(autograd.Variable(
            encoder_batch[:-1].view(-1))))

        if printing:
            encoder_debug += 'epoch %s encoder:\n' % epoch
            for n in range(min(4, N)):
                input_sentence_verify = encoding.decode_passage(encoder_batch[:, n])
                sentence = encoding.decode_passage(v_forward.data.cpu()[:, n][:-1])
                encoder_debug += '    forward [%s] => [%s]\n' % (input_sentence_verify, sentence)
            for n in range(min(4, N)):
                input_sentence_verify = encoding.decode_passage(encoder_batch[:, n])
                sentence = encoding.decode_passage(v_backward.data.cpu()[:, n][1:])
                encoder_debug += '    back [%s] => [%s]\n' % (input_sentence_verify, sentence)
        return pred_embedded, state, enc_loss

    enc_state = cudafy(autograd.Variable(torch.zeros(2 * num_layers, batch_size, hidden_size)))
    enc_out, enc_state, enc_loss = encode(encoder_batch, enc_state)
    loss += enc_loss

    # decode
    if True:
        output_sentences = ['' for n in range(batch_size)]

        dec_state = cudafy(autograd.Variable(torch.zeros(1 * num_layers, batch_size, hidden_size)))
        prev_c_batch = decoder_batch[0].view(1, -1)
        for t in range(1, seq_len):
            target_c_batch = decoder_batch[t]

            alignment_out = alignment_model(
                enc_out=enc_out, prev_dec_state=dec_state)

            """
            alignment_out is [batch_size][seq_len]
            enc_out is [seq_len][batch_size][hidden_size * 2]

            we want to Hadamand multiply alignment_out with enc_out,
            so we need to expand alignment_out to correct dimensions
            probably need a transpose too :-)
            """
            alignment_out = alignment_out.transpose(0, 1).contiguous().view(
                seq_len, batch_size, 1).expand(
                seq_len, batch_size, hidden_size * 2)

            context = (alignment_out * enc_out)
            context = context.sum(0)
            context = context.view(batch_size, hidden_size * 2)

            dec_out, dec_state = decoder(cudafy(autograd.Variable(prev_c_batch)), context, dec_state)

            pred_c_embedded_batch = dec_out
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
    if printing:
        print('time per epoch', (time.time() - last_print) / print_every)
        last_print = time.time()

    epoch += 1
