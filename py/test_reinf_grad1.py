import gym
import time
import torch
from torch import nn, autograd, optim
import torch.nn.functional as F
import numpy as np


num_hidden = 4
print_every = 100
max_steps_per_episode = 2000


class Policy(nn.Module):
    def __init__(self, num_inputs, num_hidden, num_actions):
        super().__init__()
        self.num_actions = num_actions
        self.num_inputs = num_inputs
        self.num_hidden = num_hidden
        # self.h1 = nn.Linear(num_inputs, num_hidden)
        # self.h2 = nn.Linear(num_hidden, num_actions)
        self.h1 = nn.Linear(num_inputs, num_actions)

    def forward(self, x):
        x = self.h1(x)
        x = F.softmax(x)
        multinomial_res = torch.multinomial(x, num_samples=1)
        return multinomial_res


def run_episode(policy, render):
    x = env.reset()
    reward = 0
    actions = []
    for _ in range(max_steps_per_episode):
        if render:
            env.render()
        a = policy(autograd.Variable(torch.from_numpy(x.astype(np.float32)).view(1, -1)))
        actions.append(a)
        x, r, done, info = env.step(a.data[0][0])
        reward += r
        if done:
            break

    return actions, reward


env = gym.make('CartPole-v0')
policy = Policy(
    num_inputs=env.observation_space.shape[0],
    num_hidden=num_hidden,
    num_actions=env.action_space.n)
opt = optim.Adam(params=policy.parameters(), lr=0.001)

# episode = 0
iteration = 0
sum_reward = 0

while True:
    res = []
    actions, reward = run_episode(policy=policy, render=False)
    opt.zero_grad()
    for a in actions:
        a.reinforce(reward)
    autograd.backward(actions, [None for _ in actions])
    opt.step()
    sum_reward += reward

    if iteration % print_every == 0:
        print('iteration %s avg_reward %s' % (iteration, sum_reward / print_every))
        run_episode(policy=policy, render=True)
        sum_reward = 0
    iteration += 1
