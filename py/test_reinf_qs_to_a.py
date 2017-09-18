import gym
import time
import torch
from torch import nn, autograd, optim
import torch.nn.functional as F
import numpy as np


num_hidden = 8
print_every = 100
max_steps_per_episode = 2000
exploration_prob = 0.1


class QEstimator(nn.Module):
    """
    Given current observation, predict q-value of all actions
    """
    def __init__(self, num_obs_features, num_actions):
        super().__init__()
        self.num_actions = num_actions
        self.num_obs_features = num_obs_features

        self.h1 = nn.Linear(num_obs_features, num_hidden)
        self.h2 = nn.Linear(num_hidden, num_actions)

    def forward(self, x):
        # a_hot = torch.zeros(1, self.num_actions)
        # a_hot[0][a] = 1.0
        # x = torch.cat((x, autograd.Variable(a_hot)), 1)
        x = self.h1(x)
        x = F.tanh(x)
        x = self.h2(x)
        return x


def run_episode(opt, q_estimator, render, num_actions):
    """
    at each step, we have:

    - s_t
    - a function estimator, for q(s, a)

    - we choose an action a_t, eg by e-greedy selection over actions for current state
    - we take this action, giving:
       - reward r_t
       - new state s_{t+1}

    To backprop, we need to know:
    - state s_t
    - action a_t
    - predicted q(s_t, a_t)
    - reward r_t
    - q(s_{t+1}, a_{t+1})
    - ... and we'll backprop the loss from:
       - predicted q(s_t, a_t), versus
       - r_t + q(s_{t+1}, a_{t+1})
    """
    x_t = torch.from_numpy(env.reset().astype(np.float32))

    sum_reward = 0
    sum_loss = 0
    prev = None
    for _ in range(max_steps_per_episode):
        if render:
            env.render()
        qv_t = q_estimator(autograd.Variable(x_t.view(1, -1)))[0]
        if np.random.rand() <= exploration_prob:
            # explore
            a_t = np.random.randint(num_actions)
        else:
            # exploit
            _, a_t = qv_t.data.max(dim=-1)
            a_t = a_t[0]
            # print('a_t', a_t)

        x_next_t, r_t, done, info = env.step(a_t)
        x_next_t = torch.from_numpy(x_next_t.astype(np.float32))
        if prev is not None:
            opt.zero_grad()
            prev['qv_next_t'] = qv_t
            # print('prev qv_t', prev['qv_t'])
            # print('prev a_t', prev['a_t'])
            # print('type prev a_t', type(prev['a_t']))
            # print('qv_next_d.data', prev['qv_next_t'].data)
            error = prev['r_t'] + prev['qv_next_t'].data[a_t] \
                - prev['qv_t'][prev['a_t']]
            loss = error * error
            sum_loss += loss.data[0]
            loss.backward()
            opt.step()

        sum_reward += r_t
        prev = {
            's_t': x_t.clone(),
            'qv_t': qv_t,
            'a_t': a_t,
            'r_t': r_t,
            'x_next_t': x_next_t.clone()
        }

        if done:
            break

        x_t = x_next_t

    opt.zero_grad()
    error = prev['r_t'] + 0 - prev['qv_t'][prev['a_t']]
    loss = error * error
    sum_loss += loss.data[0]
    loss.backward()
    opt.step()

    return sum_reward, sum_loss


env = gym.make('CartPole-v0')
q_estimator = QEstimator(
    num_obs_features=env.observation_space.shape[0], num_actions=env.action_space.n)
opt = optim.Adam(params=q_estimator.parameters(), lr=0.001)

iteration = 0
sum_reward = 0
sum_loss = 0

while True:
    reward, loss = run_episode(
        opt=opt,
        q_estimator=q_estimator,
        render=False, num_actions=env.action_space.n)
    sum_loss += loss
    sum_reward += reward

    if iteration % print_every == 0:
        print('iteration %s avg_reward %s loss %s' % (iteration, sum_reward / print_every, sum_loss))
        sum_reward = 0
        sum_loss = 0
    iteration += 1
