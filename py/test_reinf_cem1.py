import gym
import time
import torch
from torch import nn, autograd, optim
import torch.nn.functional as F
import numpy as np


num_hidden = 4
print_every = 1
num_theta_samples = 10
num_samples_per_theta = 10
initial_variance = 10
num_elite = 3


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
# print('action_space', env.action_space)
# print(dir(env.action_space))
# print('num_actions', env.action_space.n)
# print('num_inputs', env.observation_space.shape[0])
policy = Policy(
    num_inputs=env.observation_space.shape[0],
    num_hidden=num_hidden,
    num_actions=env.action_space.n)
# opt = optim.Adam(params=policy.parameters(), lr=0.001)
# print('observation_space', env.observation_space)


def tensor_list_to_tensor(l):
    shape = l[0].size()
    N = len(l)
    t = torch.zeros(N, *shape)
    # print('t.size()', t.size())
    for i in range(N):
        t[i] = l[i]
    return t
    # asdfsdf


def draw_theta(theta_mu, theta_var):
    """
    assumes theta_mu and theta_var are both lists of torch tensors
    thus, returns also a list, of torch tensors
    """
    theta = []
    # for i, p in enumerate(policy.parameters()):
    for i, mu in enumerate(theta_mu):
        var = theta_var[i]
        _theta = torch.normal(mu, var.sqrt())
        # p.data = this_theta
        theta.append(_theta)
    return theta


def assign_theta_to_params(params, theta):
    for i, p in enumerate(params):
        p.data.copy_(theta[i])
# def render_example():


# episode = 0
iteration = 0
sum_reward = 0
# theta = [p for p in policy.parameters()]
theta_shapes = [p.data.shape for p in policy.parameters()]
theta_mu = []
theta_var = []
for shape in theta_shapes:
    theta_mu.append(torch.zeros(shape))
    var = torch.ones(shape)
    var *= initial_variance
    print('var', var)
    theta_var.append(var)
# print('theta_shapes', theta_shapes)
# print(theta_mu)
# print(theta_var)
# asdf
while True:
    # opt.zero_grad()
    res = []
    for n in range(num_theta_samples):
        theta = draw_theta(theta_mu=theta_mu, theta_var=theta_var)
        assign_theta_to_params(params=policy.parameters(), theta=theta)
        sum_reward_this_theta = 0
        for j in range(num_samples_per_theta):
            # render = episode % print_every == 0 and n % (num_samples // 3) == 0
            reward = run_episode(policy=policy, render=False)
            sum_reward_this_theta += reward
        avg_reward_this_theta = sum_reward_this_theta / num_samples_per_theta
        res.append({'theta': theta, 'reward': avg_reward_this_theta})
    sum_reward += avg_reward_this_theta
    # print(res)
    res.sort(key=lambda x: x['reward'], reverse=True)
    elite_samples = res[:num_elite]
    # print('elite_samples', elite_samples)
    for i, p in enumerate(policy.parameters()):
        this_elite_thetas = tensor_list_to_tensor([result['theta'][i] for result in elite_samples])
        # print('this_elite_thetas', this_elite_thetas)
        this_new_mu = this_elite_thetas.mean(dim=0)
        this_new_var = this_elite_thetas.var(dim=0)
        theta_mu[i] = this_new_mu
        theta_var[i] = this_new_var

    if iteration % print_every == 0:
        print('theta_mu', theta_mu)
        print('theta_var', theta_var)
        print('iteration %s avg_reward %s' % (iteration, sum_reward / print_every))
        # render_example()
        assign_theta_to_params(params=policy.parameters(), theta=theta_mu)
        run_episode(policy=policy, render=True)
        sum_reward = 0
    iteration += 1
    # asdfasd
