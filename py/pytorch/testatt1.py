import torch
import math
import numpy as np
from torch import autograd


def attention(Q, K, V):
    num_key_features = K.size()[-1]
    print('num_key_features', num_key_features)
    print('Q.shape', Q.size())
    print('K.shape', K.size())
    # print('K.transpose(0,1).shape', K.transpose(0,1).shape)
    QK = Q @ K.transpose(0, 1)
    print('QK', QK)
    # x = Q @ K
    QK /= math.sqrt(num_key_features)
    print('QK norm', QK)
    QK_prob = torch.nn.functional.softmax(QK)
    print('QK_prob', QK_prob)
    return QK_prob @ V


if __name__ == '__main__':
    d = 2
    V = autograd.Variable(torch.Tensor([
        [0.7, 0, 0],
        [0, 0.3, 0],
        [0, 0, 0.2]
    ]))
    K = autograd.Variable(torch.Tensor([
            [1, 0, 0],
            [0, 1, 0],
            [0, 0, 1]
        ]))
    Q = autograd.Variable(torch.Tensor([
            [1, 0, 0],
            [5, 0, 0],
            [0.1, 0, 0],
            [0.1, 0.1, 0],
            [0, 0.1, 0]
        ]))
    print('V', V)
    print('K', K)
    print('Q', Q)
    res = attention(Q, K, V)
    print('res', res)
