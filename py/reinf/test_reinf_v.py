import gym
import time
import torch
from torch import nn, autograd, optim
import torch.nn.functional as F
import numpy as np


num_hidden = 4
print_every = 100
max_steps_per_episode = 2000


class ValueEstimator(nn.Module):
    def __init__(self, num_obs_features):
        super().__init__()
        # self.num_actions = num_actions
        # self.num_inputs = num_inputs
        # self.num_hidden = num_hidden
        # self.h1 = nn.Linear(num_inputs, num_hidden)
        # self.h2 = nn.Linear(num_hidden, num_actions)
        self.h1 = nn.Linear(num_obs_features, 1)

    def forward(self, x):
        x = self.h1(x)
        # x = F.softmax(x)
        # multinomial_res = torch.multinomial(x, num_samples=1)
        # return multinomial_res
        return x


class StateEstimator(nn.Module):
    """
    Given current observation, and an action, predict next observation
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
        self.h1 = nn.Linear(num_obs_features + num_actions, num_obs_features)

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
        # x = F.softmax(x)
        # multinomial_res = torch.multinomial(x, num_samples=1)
        # return multinomial_res
        return x


def run_episode(state_estimator, value_estimator, render, num_actions):
    x = torch.from_numpy(env.reset().astype(np.float32))
    reward = 0
    actions = []
    states = []
    rewards = []
    for _ in range(max_steps_per_episode):
        if render:
            env.render()
        best_a = -1
        best_value = None
        for a in range(num_actions):
            x_torch = autograd.Variable(x.view(1, -1))
            next_state = state_estimator(x_torch, a).data
            next_value = value_estimator(autograd.Variable(next_state)).data[0][0]
            # print('next_value', next_value)
            if best_value is None or next_value > best_value:
                best_a = a
                best_value = next_value
        if np.random.uniform() < 0.3:
            best_a = np.random.choice(num_actions, 1)[0]
        a = best_a
        # a = policy(autograd.Variable(torch.from_numpy(x.astype(np.float32)).view(1, -1)))
        actions.append(a)
        states.append(x.clone())
        x, r, done, info = env.step(a)
        x = torch.from_numpy(x.astype(np.float32))
        rewards.append(r)
        reward += r
        if done:
            break

    return states, actions, rewards, reward


env = gym.make('CartPole-v0')
value_estimator = ValueEstimator(
    num_obs_features=env.observation_space.shape[0])
state_estimator = StateEstimator(
    num_obs_features=env.observation_space.shape[0], num_actions=env.action_space.n)
v_opt = optim.Adam(params=value_estimator.parameters(), lr=0.001)
s_opt = optim.Adam(params=state_estimator.parameters(), lr=0.001)

# episode = 0
iteration = 0
sum_reward = 0
sum_s_loss = 0
sum_v_loss = 0

while True:
    states, actions, rewards, reward = run_episode(
        state_estimator=state_estimator, value_estimator=value_estimator,
        render=False, num_actions=env.action_space.n)
    reward_from_here = 0
    for t in range(len(states) - 1, -1, -1):
        reward_from_here += rewards[t]
        state = states[t]

        v_opt.zero_grad()
        pred_value = value_estimator(autograd.Variable(state))
        loss = ((pred_value - reward_from_here) * (pred_value - reward_from_here)).sqrt()
        sum_v_loss += loss.data[0]
        loss.backward()
        v_opt.step()

        if t > 0:
            s_opt.zero_grad()
            pred_s = state_estimator(autograd.Variable(states[t - 1].view(1, -1)), actions[t - 1])
            # print('pred_s', pred_s)
            # print('state', state)
            loss = ((pred_s - autograd.Variable(state)) * (pred_s - autograd.Variable(state))).sum().sqrt()
            sum_s_loss += loss.data[0]
            loss.backward()
            s_opt.step()
    sum_reward += reward

    if iteration % print_every == 0:
        print('iteration %s avg_reward %s s_loss %s v_loss %s' % (iteration, sum_reward / print_every, sum_s_loss, sum_v_loss))
        run_episode(state_estimator=state_estimator, value_estimator=value_estimator, render=True, num_actions=env.action_space.n)
        sum_reward = 0
        sum_s_loss = 0
        sum_v_loss = 0
    iteration += 1
