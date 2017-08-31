import gym
import time
import torch
from torch import nn, autograd, optim
import torch.nn.functional as F
import numpy as np


class Policy(nn.Module):
    """
    Given parameters, chooses an action, possibly deterministically,
    ie no stochastic sampling
    given any state as input
    """
    def __init__(self, num_inputs, num_actions):
        super().__init__()
        self.num_actions = num_actions
        self.num_inputs = num_inputs
        self.h1 = nn.Linear(num_inputs, num_actions)

    def forward(self, x):
        # print('x.data.shape', x.data.shape)
        # print('self.num_inputs', self.num_inputs)
        # print('self.num_actions', self.num_actions)
        x = self.h1(x)
        _, x = x.max(dim=1)
        # print('x', x)
        return x


class ES(nn.Module):
    def __init__(self, num_params):
        self.num_params = num_params

    def forward(self, x):
        return x


env = gym.make('CartPole-v0')


def run_episode(policy):
    x = env.reset()
    reward = 0
    for _ in range(1000):
        env.render()
        a_idx = policy(autograd.Variable(torch.from_numpy(x.astype(np.float32)).view(1, -1)))
        # a_idx = a_idx.data[0]
        # print('a_idx', a_idx.data[0])
        # a = env.action_space.sample()
        x, r, done, info = env.step(a_idx.data[0])
        # print('a', a, 'x', x, 'r', r, 'done', done)
        reward += r
        if done:
            break
        time.sleep(0.1)
    return reward


print('action_space', env.action_space)
print(dir(env.action_space))
print('num_actions', env.action_space.n)
print('num_inputs', env.observation_space.shape[0])
policy = Policy(num_inputs=env.observation_space.shape[0], num_actions=env.action_space.n)
# opt = optim.Adam(params=model.parameters(), lr=0.001)
print('observation_space', env.observation_space)

episode = 0
while True:
    # opt.zero_grad()
    reward = run_episode(policy=policy)
    loss = - reward
    # loss.backward()
    # help(opt.step)
    # opt.step()
    print('episode %s reward %s' % (episode, reward))
    episode += 1
