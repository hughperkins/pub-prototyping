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
        # self.num_actions = num_actions
        # self.num_inputs = num_inputs
        # self.num_hidden = num_hidden
        # self.h1 = nn.Linear(num_inputs, num_hidden)
        # self.h2 = nn.Linear(num_hidden, num_actions)
        self.num_actions = num_actions
        self.num_obs_features = num_obs_features
        # self.h1 = nn.Linear(num_obs_features + num_actions, 1)

        self.h1 = nn.Linear(num_obs_features + num_actions, num_hidden)
        self.h2 = nn.Linear(num_hidden, 1)

    def forward(self, x, a):
        # a_hot = torch.zeros(self.num_obs_features + self.num_actions)
        a_hot = torch.zeros(1, self.num_actions)
        a_hot[0][a] = 1.0
        # a_hot[self.num_obs_features + a] = 1.0
        # a_hot[:self.num_obs_features] = x
        # print(torch.__version__)
        # print('x', x)
        # print('a_hot', a_hot)
        x = torch.cat((x, autograd.Variable(a_hot)), 1)
        # print('x', x)
        x = self.h1(x)
        x = F.tanh(x)
        x = self.h2(x)
        # x = F.softmax(x)
        # multinomial_res = torch.multinomial(x, num_samples=1)
        # return multinomial_res
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
    # x_t = autograd.Variabl
    # x_torch = autograd.Variable(x_t.view(1, -1))

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
                # next_q_s = next_q.data
                # print('candidate_q', candidate_q)
                # print('best_q', best_q)
                if best_a == -1 or candidate_q.data[0][0] > best_q.data[0][0]:
                    best_a = candidate_a
                    best_q = candidate_q
                    # pred_q = next_q
            a_t = best_a
            q_t = best_q

        x_next_t, r_t, done, info = env.step(a_t)
        x_next_t = torch.from_numpy(x_next_t.astype(np.float32))
        if prev is not None:
            opt.zero_grad()
            prev['q_next_t'] = q_t
            error = prev['r_t'] + prev['q_next_t'].data[0][0] - prev['q_t']
            # print('next %.3f this %.3f' % (prev['r_t'] + prev['q_next_t'].data[0][0], prev['q_t'].data[0][0]))
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

        # rewards.append(r)
        x_t = x_next_t

    opt.zero_grad()
    # prev['q_next_t'] = q_t
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

# episode = 0
iteration = 0
sum_reward = 0
sum_loss = 0

while True:
    # states, actions, rewards, reward = run_episode(
    #     qsa_estimator=qsa_estimator,
    #     render=False, num_actions=env.action_space.n)
    reward, loss = run_episode(
        opt=opt,
        qsa_estimator=qsa_estimator,
        render=False, num_actions=env.action_space.n)
    sum_loss += loss
    sum_reward += reward
    # reward_from_here = 0
    # for t in range(len(states) - 1, -1, -1):
    #     # reward_from_here += rewards[t]
    #     state = states[t]

    #     # opt.zero_grad()

    #     # pred_value = value_estimator(autograd.Variable(state))
    #     # loss = ((pred_value - reward_from_here) * (pred_value - reward_from_here)).sqrt()
    #     # sum_v_loss += loss.data[0]
    #     # loss.backward()
    #     # v_opt.step()

    #     if t > 0:
    #         s_opt.zero_grad()
    #         pred_s = state_estimator(autograd.Variable(states[t - 1].view(1, -1)), actions[t - 1])
    #         # print('pred_s', pred_s)
    #         # print('state', state)
    #         loss = ((pred_s - autograd.Variable(state)) * (pred_s - autograd.Variable(state))).sum().sqrt()
    #         sum_s_loss += loss.data[0]
    #         loss.backward()
    #         opt.step()
    # sum_reward += reward

    if iteration % print_every == 0:
        print('iteration %s avg_reward %s loss %s' % (iteration, sum_reward / print_every, sum_loss))
    #     run_episode(qsa_estimator=qsa_estimator, render=True, num_actions=env.action_space.n)
        sum_reward = 0
        sum_loss = 0
    iteration += 1
