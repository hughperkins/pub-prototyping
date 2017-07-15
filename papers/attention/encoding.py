"""
Handles converting from sentences <=> numpy arrays of integers
"""
import torch


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
    encoded = torch.LongTensor(len(sentence) + 2)
    encoded[0] = start_code
    for i, c in enumerate(sentence):
        encoded[i + 1] = add_char(c)
    encoded[len(sentence) + 1] = end_code
    return encoded


def decode_passage(encoded):
    decoded = [char_by_idx[c_enc] for c_enc in encoded.contiguous().view(-1)]
    return ''.join(decoded)


# training_encoded = []
add_char('<start>')
add_char('<end>')
add_char('<unk>')
unk_code = idx_by_char['<unk>']
start_code = encode_char('<start>')
end_code = encode_char('<end>')
