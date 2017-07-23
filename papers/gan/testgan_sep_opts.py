import torch
# import numpy as np
from torch import nn, autograd, optim
from matplotlib import pyplot as plt


num_data_samples = 16
hidden_size = 8
batch_size = 16
# num_critic_steps = 5
# weight_clipping = 0.01

data_samples = torch.rand(num_data_samples, 1)


class Discriminator(nn.Module):
    def __init__(self, hidden_size):
        super().__init__()
        self.h1 = nn.Linear(1, hidden_size)
        self.h2 = nn.Linear(hidden_size, 1)

    def forward(self, x):
        x = self.h1(x)
        x = nn.functional.tanh(x)
        x = self.h2(x)
        x = nn.functional.sigmoid(x)
        return x


class Generator(nn.Module):
    def __init__(self, hidden_size):
        super().__init__()
        self.h1 = nn.Linear(1, hidden_size)
        self.h2 = nn.Linear(hidden_size, 1)

    def forward(self, x):
        x = self.h1(x)
        x = nn.functional.tanh(x)
        x = self.h2(x)
        return x


def run():
    d = Discriminator(hidden_size=hidden_size)
    g = Generator(hidden_size=hidden_size)

    opt_d = optim.Adam(params=d.parameters(), lr=0.001)
    opt_g = optim.Adam(params=g.parameters(), lr=0.001)
    epoch = 0
    while True:
        # generator
        g_loss = 0
        g.zero_grad()
        z = autograd.Variable(torch.rand(batch_size, 1))
        sample = g(z)
        d_out = d(sample)
        g_loss += -0.5 * d_out.log().sum()
        g_loss.backward()
        opt_g.step()

        # discriminator
        d_loss = 0
        d.zero_grad()

        # generated
        z = autograd.Variable(torch.rand(batch_size, 1))
        sample = g(z)
        d_out = d(sample)
        d_loss += -0.5 * (1 - d_out).log().sum()

        # samples
        d_out = d(autograd.Variable(data_samples))
        d_loss += -0.5 * d_out.log().sum()

        d_loss.backward()
        opt_d.step()

        if epoch % 2000 == 0:
            plt.clf()
            plt.scatter(data_samples.numpy(), [1] * num_data_samples, label='truth')
            plt.scatter(sample.data.numpy(), [0] * num_data_samples, label='generated')
            x_min = min(data_samples.min(), sample.data.min())
            x_max = max(data_samples.max(), sample.data.max())
            interval = (x_max - x_min) / 100

            X = torch.arange(x_min - interval * 50, x_max + interval * 51, interval)
            d_graph_out = d(autograd.Variable(X.view(-1, 1))).data.view(-1)
            d_graph_real_loss = -0.5 * d_graph_out.log()
            d_graph_fake_loss = -0.5 * (1 - d_graph_out).log()
            g_graph_loss = -0.5 * d_graph_out.log()

            plt.plot(X.numpy(), d_graph_out.numpy(), label='discriminator')
            # plt.plot(X.numpy(), d_graph_real_loss.numpy(), label='discriminator real loss')
            plt.plot(X.numpy(), d_graph_fake_loss.numpy(), label='discriminator fake loss')
            plt.plot(X.numpy(), g_graph_loss.numpy(), label='Generator loss, disc real loss')
            plt.legend()
            plt.savefig('/tmp/data.png')

            print('e=%s d_loss %.3f g_loss %.3f' % (epoch, d_loss.data[0], g_loss.data[0]))

        epoch += 1


if __name__ == '__main__':
    run()
