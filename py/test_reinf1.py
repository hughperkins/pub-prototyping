import gym
import time
import torch
from torch import nn, autograd, optim
import torch.nn.functional as F
import numpy as np


class Policy(nn.Module):
    """
    Given parameters, calculates probability of any action,
    given any state as input
    samples from these probabilities
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
        x = F.softmax(x)
        multinomial_res = torch.multinomial(x, num_samples=1)
        # x =
        # _, x = x.max(dim=1)
        # print('x', x)
        return multinomial_res


def run_episode(policy, render):
    x = env.reset()
    reward = 0
    multinomial_res_nodes = []
    for _ in range(1000):
        if render:
            env.render()
        a_idx = policy(autograd.Variable(torch.from_numpy(x.astype(np.float32)).view(1, -1)))
        multinomial_res_nodes.append(a_idx)
        # a_idx = a_idx.data[0]
        # print('a_idx', a_idx.data[0])
        # a = env.action_space.sample()
        x, r, done, info = env.step(a_idx.data[0][0])
        # print('a', a, 'x', x, 'r', r, 'done', done)
        reward += r
        if done:
            break
        # time.sleep(0.02)

    avg_reward = reward / len(multinomial_res_nodes)
    for node in multinomial_res_nodes:
        node.reinforce(avg_reward)
    return reward


env = gym.make('CartPole-v0')
print('action_space', env.action_space)
print(dir(env.action_space))
print('num_actions', env.action_space.n)
print('num_inputs', env.observation_space.shape[0])
policy = Policy(num_inputs=env.observation_space.shape[0], num_actions=env.action_space.n)
opt = optim.Adam(params=policy.parameters(), lr=0.001)
print('observation_space', env.observation_space)

episode = 0
sum_reward = 0
print_every = 1000
while True:
    opt.zero_grad()
    render = episode % print_every == 0
    reward = run_episode(policy=policy, render=render)
    sum_reward += reward
    # loss = - reward
    # loss.backward()
    # help(opt.step)
    opt.step()
    if render:
        print('episode %s avg_reward %s reward %s' % (episode, sum_reward / print_every, reward))
        sum_reward = 0
    episode += 1
