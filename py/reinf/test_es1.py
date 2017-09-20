import argparse
import time
import torch
from torch import nn, autograd, optim
import torch.nn.functional as F
import numpy as np
import myenvs


class Policy(nn.Module):
    """
    Given parameters, chooses an action, possibly deterministically,
    ie no stochastic sampling
    given any state as input
    """
    def __init__(self, num_inputs, num_actions, num_hidden=8):
        super().__init__()
        self.num_actions = num_actions
        self.num_inputs = num_inputs
        self.h1 = nn.Linear(num_inputs, num_hidden)
        self.h2 = nn.Linear(num_hidden, num_actions)

    def forward(self, x):
        # print('x.data.shape', x.data.shape)
        # print('self.num_inputs', self.num_inputs)
        # print('self.num_actions', self.num_actions)
        x = self.h1(x)
        x = F.tanh(x)
        x = self.h2(x)
        # _, x = x.max(dim=1)
        x = F.softmax(x)
        a = torch.multinomial(x)
        # print('x', x)
        return a


class ES(nn.Module):
    def __init__(self, num_params):
        self.num_params = num_params

    def forward(self, x):
        return x


def run_episode(env, policy, render=False):
    x = env.reset()
    # reward = 0
    actions = []
    rewards = []
    states = []
    for _ in range(1000):
        if render:
            env.render()
        a = policy(autograd.Variable(torch.from_numpy(x.astype(np.float32)).view(1, -1)))
        # a = a_node.data[0]
        states.append(x)
        actions.append(a)
        # a_idx = a_idx.data[0]
        # print('a_idx', a_idx.data[0])
        # a = env.action_space.sample()
        # print('a', a)
        x, r, done, info = env.step(a.data[0][0])
        # print('a', a, 'x', x, 'r', r, 'done', done)
        # reward += r
        rewards.append(r)
        if done:
            break
        # time.sleep(0.1)
    return states, actions, rewards


def run(env):
    env = myenvs.get_env_by_name(env_name=env)

    print('action_space', env.action_space)
    print(dir(env.action_space))
    print('num_actions', env.action_space.n)
    print('num_inputs', env.observation_space.shape[0])
    policy = Policy(num_inputs=env.observation_space.shape[0], num_actions=env.action_space.n)
    opt = optim.Adam(params=policy.parameters(), lr=0.001)
    print('observation_space', env.observation_space)

    episode = 0
    last = time.time()
    sum_epochs = 0
    sum_rewards = 0
    start = time.time()
    while True:
        opt.zero_grad()
        states, actions, rewards = run_episode(env=env, policy=policy)
        total_reward = np.sum(rewards)
        sum_rewards += total_reward
        sum_epochs += 1
        # print('total_reward %s' % total_reward)
        for a in actions:
            a.reinforce(total_reward)
        autograd.backward(actions, [None] * len(actions))
        # loss = - reward
        # loss.backward()
        # help(opt.step)
        opt.step()
        if time.time() - last >= 1.0:
            print('episode %s elapsed %ss avg reward %.1f' % (episode, int(time.time() - start), sum_rewards / sum_epochs))
            sum_epochs = 0
            sum_rewards = 0
            last = time.time()
        episode += 1

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--env', type=str, default='cartpole')
    args = parser.parse_args()
    run(**args.__dict__)
