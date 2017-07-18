"""
Try birectional rnn, check sizes etc

'test' here means 'prototype', 'try'.

These are not actually tests, as in 'unit tests', or similar.

This runs using pytest, so can choose a function using `-k [function name]`
"""
import torch
from torch import nn, autograd


def test_bidirectional_outsize():
    """
    each number below is unique, so we can tell which is which later:
    """
    # input_size = 3
    batch_size = 5
    seq_len = 7
    hidden_size = 4

    rnn = nn.RNN(
      input_size=hidden_size, hidden_size=hidden_size, bidirectional=True)
    state = autograd.Variable(torch.zeros(2, batch_size, hidden_size))

    input = autograd.Variable(torch.rand(seq_len, batch_size, hidden_size))
    output, state = rnn(input, state)

    print('output.size()', output.size())
    print('state.size()', state.size())


def test_output_layout():
    """
    try to verify the layout of the output, ie is it with one direction
    first, then the other direction?
    or is it interleaved: one value from one direction, then the same value
    from the other direction, and so on?

    idea: turn off bias, feed through some values are 1s, some are zeros
    check which output values are non-zero.
    """
    batch_size = 1
    seq_len = 4
    hidden_size = 3
    rnn = nn.RNN(
        input_size=hidden_size, hidden_size=hidden_size, bidirectional=True,
        bias=False)
    state = autograd.Variable(torch.zeros(2, batch_size, hidden_size))

    input = autograd.Variable(torch.rand(seq_len, batch_size, hidden_size))
    input[2:].data.zero_()
    output, state = rnn(input, state)

    print('output.size()', output.size())
    print('state.size()', state.size())
    print('output', output)

    """
    Example output:

    output Variable containing:
    (0 ,.,.) =
     -0.5898 -0.4658  0.3640 -0.3346  0.3044  0.2513

    (1 ,.,.) =
     -0.7412 -0.8000  0.3279 -0.0022 -0.0207  0.5351

    (2 ,.,.) =
     -0.0832 -0.4698 -0.0491  0.0000  0.0000  0.0000

    (3 ,.,.) =
     -0.1565 -0.0716 -0.0565  0.0000  0.0000  0.0000
    [torch.FloatTensor of size 4x1x6]

    => looks like first values of output are forwards direction, and the
    remaining values are backwards direction
    """


def test_is_inplace():
    """
    Test if state gets modified in-place
    """
    batch_size = 1
    seq_len = 4
    hidden_size = 3
    rnn = nn.RNN(
        input_size=hidden_size, hidden_size=hidden_size, bidirectional=True,
        bias=False)
    state = autograd.Variable(torch.zeros(2, batch_size, hidden_size))
    old_state = state.data.clone()  # call data to make sure dont invoke any autograd bits

    input = autograd.Variable(torch.rand(seq_len, batch_size, hidden_size))
    output, new_state = rnn(input, state)

    print('old_state', old_state)
    print('state', state)
    print('new_state', new_state)

    """
    Output:

    old_state
    (0 ,.,.) =
      0  0  0

    (1 ,.,.) =
      0  0  0
    [torch.FloatTensor of size 2x1x3]

    state Variable containing:
    (0 ,.,.) =
      0  0  0

    (1 ,.,.) =
      0  0  0
    [torch.FloatTensor of size 2x1x3]

    new_state Variable containing:
    (0 ,.,.) =
      0.3964 -0.1368 -0.0647

    (1 ,.,.) =
     -0.0837 -0.0157  0.0158
    [torch.FloatTensor of size 2x1x3]

    Looks like original state is left unmodified by running rnn
    """


def test_cat():
    """
    Test torch.cat
    """
    a = autograd.Variable(torch.rand(3), requires_grad=True)
    b = autograd.Variable(torch.rand(3), requires_grad=True)
    c = torch.cat([a, b])
    print('c.size()', c.size())
    print('a', a)
    print('b', b)
    print('c', c)
    c.backward(torch.ones(6))
    print('a.grad', a.grad)
    """
    Example output:

    test_bidirectional_rnn.py::test_cat c.size() torch.Size([6])
    a Variable containing:
     0.7181
     0.2566
     0.7274
    [torch.FloatTensor of size 3]

    b Variable containing:
     0.0344
     0.4115
     0.4244
    [torch.FloatTensor of size 3]

    c Variable containing:
     0.7181
     0.2566
     0.7274
     0.0344
     0.4115
     0.4244
    [torch.FloatTensor of size 6]

    a.grad Variable containing:
     1
     1
     1
    [torch.FloatTensor of size 3]
    """


def test_broadcast():
    """
    Test broadcast, for calculating context from alignment model output,
    and annotations
    """
    batch_size = 4
    hidden_size = 3
    seq_len = 5
    alignment_model_out = autograd.Variable(torch.rand(batch_size, seq_len)).transpose(0, 1)
    if True:  # so flake8 doesnt complain...
        import torch.nn.functional as F
    alignment_model_out = F.softmax(alignment_model_out)
    print('alignment_model_out.size()', alignment_model_out.size())
    # alignment_model_out = alignment_model_out.expand([seq_len, batch_size, hidden_size * 2])

    annotations = autograd.Variable(torch.rand(seq_len, batch_size, hidden_size * 2))
    print('annotations.size()', annotations.size())

    alignment_model_out = alignment_model_out.view(seq_len, batch_size, 1).expand_as(annotations)
    print('alignment_model_out.size()', alignment_model_out.size())
    context_batch = alignment_model_out * annotations
    print('context_batch', context_batch)
    """
    Example output:

    test_bidirectional_rnn.py::test_broadcast alignment_model_out.size() torch.Size([5, 4])
    annotations.size() torch.Size([5, 4, 6])
    alignment_model_out.size() torch.Size([5, 4, 6])
    context_batch Variable containing:
    (0 ,.,.) =
      0.1022  0.1132  0.1601  0.0611  0.2285  0.1575
      0.2123  0.2267  0.0893  0.1335  0.0335  0.1995
      0.0155  0.1366  0.2541  0.1953  0.2041  0.0426
      0.0912  0.0425  0.0717  0.1340  0.1652  0.0844

    (1 ,.,.) =
      0.1621  0.0480  0.0605  0.2651  0.0883  0.3060
      0.0246  0.1094  0.0436  0.1717  0.1513  0.0447
      0.1352  0.0682  0.0305  0.1157  0.1091  0.1287
      0.1899  0.0615  0.0508  0.0073  0.1758  0.0705

    (2 ,.,.) =
      0.0726  0.3085  0.1380  0.3315  0.2373  0.0565
      0.0441  0.1485  0.0491  0.1673  0.0962  0.1091
      0.1936  0.1272  0.1602  0.0245  0.1501  0.0131
      0.1505  0.1404  0.0480  0.1648  0.0105  0.1524

    (3 ,.,.) =
      0.0441  0.1943  0.0656  0.1068  0.1067  0.1805
      0.1257  0.2300  0.2095  0.2372  0.0297  0.1942
      0.1981  0.0825  0.0269  0.0458  0.1375  0.1087
      0.0750  0.0466  0.1428  0.2158  0.1639  0.2473

    (4 ,.,.) =
      0.0011  0.0626  0.1492  0.0922  0.1730  0.1997
      0.1288  0.0232  0.2257  0.2502  0.0179  0.3118
      0.0906  0.0890  0.1441  0.1715  0.2131  0.1903
      0.1362  0.1810  0.0591  0.0735  0.1121  0.0387
    [torch.FloatTensor of size 5x4x6]
    """
