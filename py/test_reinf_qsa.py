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


class QSAEstimator(nn.Module):
    """
    Given current observation, and an action, predict q-value
    """
    def __init__(self, num_obs_features, num_actions):
        super().__init__()
        self.num_actions = num_actions
        self.num_obs_features = num_obs_features

        self.h1 = nn.Linear(num_obs_features + num_actions, num_hidden)
        self.h2 = nn.Linear(num_hidden, 1)

    def forward(self, x, a):
        a_hot = torch.zeros(1, self.num_actions)
        a_hot[0][a] = 1.0
        x = torch.cat((x, autograd.Variable(a_hot)), 1)
        x = self.h1(x)
        x = F.tanh(x)
        x = self.h2(x)
        return x


def run_episode(opt, qsa_estimator, render, num_actions):
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
        if np.random.rand() <= exploration_prob:
            # explore
            a_t = np.random.randint(num_actions)
            q_t = qsa_estimator(autograd.Variable(x_t.view(1, -1)), a_t)
        else:
            # exploit
            best_a = -1
            best_q = None
            # pred_q = None
            for candidate_a in range(num_actions):
                candidate_q = qsa_estimator(autograd.Variable(x_t.view(1, -1)), candidate_a)
                if best_a == -1 or candidate_q.data[0][0] > best_q.data[0][0]:
                    best_a = candidate_a
                    best_q = candidate_q
            a_t = best_a
            q_t = best_q

        x_next_t, r_t, done, info = env.step(a_t)
        x_next_t = torch.from_numpy(x_next_t.astype(np.float32))
        if prev is not None:
            opt.zero_grad()
            prev['q_next_t'] = q_t
            error = prev['r_t'] + prev['q_next_t'].data[0][0] - prev['q_t']
            loss = error * error
            sum_loss += loss.data[0]
            loss.backward()
            opt.step()

        sum_reward += r_t
        prev = {
            's_t': x_t.clone(),
            'q_t': q_t,
            'a_t': a_t,
            'r_t': r_t,
            'x_next_t': x_next_t.clone()
        }

        if done:
            break

        x_t = x_next_t

    opt.zero_grad()
    error = prev['r_t'] + 0 - prev['q_t']
    loss = error * error
    sum_loss += loss.data[0]
    loss.backward()
    opt.step()

    return sum_reward, sum_loss


env = gym.make('CartPole-v0')
qsa_estimator = QSAEstimator(
    num_obs_features=env.observation_space.shape[0], num_actions=env.action_space.n)
opt = optim.Adam(params=qsa_estimator.parameters(), lr=0.001)

iteration = 0
sum_reward = 0
sum_loss = 0

while True:
    reward, loss = run_episode(
        opt=opt,
        qsa_estimator=qsa_estimator,
        render=False, num_actions=env.action_space.n)
    sum_loss += loss
    sum_reward += reward

    if iteration % print_every == 0:
        print('iteration %s avg_reward %s loss %s' % (iteration, sum_reward / print_every, sum_loss))
        sum_reward = 0
        sum_loss = 0
    iteration += 1
