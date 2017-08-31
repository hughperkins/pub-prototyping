import gym
import time
import torch
from torch import nn, autograd, optim
import torch.nn.functional as F
import numpy as np


num_hidden = 4
print_every = 100
num_theta_samples = 100
num_samples_per_theta = 100
initial_variance = 10
num_elite = 10


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
    # multinomial_res_nodes = []
    for _ in range(1000):
        if render:
            env.render()
        a_idx = policy(autograd.Variable(torch.from_numpy(x.astype(np.float32)).view(1, -1)))
        # multinomial_res_nodes.append(a_idx)
        # a_idx = a_idx.data[0]
        # print('a_idx', a_idx.data[0])
        # a = env.action_space.sample()
        x, r, done, info = env.step(a_idx.data[0][0])
        # print('a', a, 'x', x, 'r', r, 'done', done)
        reward += r
        if done:
            break
        # time.sleep(0.02)

    # avg_reward = reward / len(multinomial_res_nodes)
    # reward = reward - 18
    # reward = reward * reward
    # for node in multinomial_res_nodes:
        # node.reinforce(reward)
    return reward


env = gym.make('CartPole-v0')
print('action_space', env.action_space)
print(dir(env.action_space))
print('num_actions', env.action_space.n)
print('num_inputs', env.observation_space.shape[0])
policy = Policy(
    num_inputs=env.observation_space.shape[0],
    num_hidden=num_hidden,
    num_actions=env.action_space.n)
# opt = optim.Adam(params=policy.parameters(), lr=0.001)
print('observation_space', env.observation_space)


def tensor_list_to_tensor(l):
    shape = l[0].size()
    N = len(l)
    t = torch.zeros(N, *shape)
    # print('t.size()', t.size())
    return t
    # asdfsdf


episode = 0
sum_reward = 0
# theta = [p for p in policy.parameters()]
theta_shapes = [p.data.shape for p in policy.parameters()]
theta_mu = []
theta_var = []
for shape in theta_shapes:
    theta_mu.append(torch.zeros(shape))
    var = torch.ones(shape) * 10
    theta_var.append(var)
# print('theta_shapes', theta_shapes)
# print(theta_mu)
# print(theta_var)
# asdf
while True:
    # opt.zero_grad()
    res = []
    for n in range(num_samples):
        render = episode % print_every == 0 and n % (num_samples // 3) == 0
        theta = []
        for i, p in enumerate(policy.parameters()):
            this_theta = torch.normal(theta_mu[i], theta_var[i].sqrt())
            p.data = this_theta
            theta.append(this_theta)
        reward = run_episode(policy=policy, render=render)
        res.append({'theta': theta, 'reward': reward})
    sum_reward += reward
    # print(res)
    res.sort(key=lambda x: x['reward'], reverse=True)
    elite_samples = res[:num_elite]
    for i, p in enumerate(policy.parameters()):
        shape = p.data.shape
        # this_new_theta = torch.zeros(shape)
        this_elite_thetas = tensor_list_to_tensor([result['theta'][i] for result in elite_samples])
        this_new_mu = this_elite_thetas.mean(dim=0)
        # elite_vars = tensor_list_to_tensor([result['theta'][i] for result in elite_samples])
        this_new_var = this_elite_thetas.var(dim=0)
        theta_mu[i] = this_new_mu
        theta_var[i] = this_new_var
        # this_new_var = torch.

    # print(res[:3])
    # asdfad
    # loss = - reward
    # loss.backward()
    # help(opt.step)
    # opt.step()
    if episode % print_every == 0:
        print('episode %s avg_reward %s reward %s' % (episode, sum_reward / print_every, reward))
        sum_reward = 0
    episode += 1
