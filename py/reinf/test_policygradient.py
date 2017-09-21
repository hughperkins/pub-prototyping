import argparse
import time
import torch
from torch import nn, autograd, optim
import torch.nn.functional as F
import numpy as np
import myenvs


class Policy(nn.Module):
    def __init__(self, num_inputs, num_actions, num_hidden=8):
        super().__init__()
        self.num_actions = num_actions
        self.num_inputs = num_inputs
        self.h1 = nn.Linear(num_inputs, num_actions)
        # self.h1 = nn.Linear(num_inputs, num_hidden)
        # self.h2 = nn.Linear(num_hidden, num_actions)

    def forward(self, x):
        x = self.h1(x)
        # x = F.tanh(x)
        # x = self.h2(x)
        x = F.softmax(x)
        a = torch.multinomial(x)
        return a


def run_episode(env, policy, render=False):
    x = env.reset()
    # print('x', x)
    # print('type(x)', type(x))
    actions = []
    rewards = []
    states = []
    abandoned = False
    step = 0
    while True:
    # for _ in range(1000000):
        if render:
            env.render()
        a = policy(autograd.Variable(torch.from_numpy(x.astype(np.float32)).view(1, -1)))
        states.append(x)
        actions.append(a)
        x, r, done, info = env.step(a.data[0][0])
        # print('x', x)
        # print('x.shape', x.shape)
        # print('type(x)', type(x))
        # print('r', r)
        # print('type(r)', type(r))
        rewards.append(r)
        if done:
            break
        step += 1
        if step >= 20000:
            abandoned = True
            break
    return states, actions, rewards, abandoned


def run(env):
    env = myenvs.get_env_by_name(env_name=env)

    print('action_space', env.action_space)
    print(dir(env.action_space))
    print('num_actions', env.action_space.n)
    print('num_inputs', env.observation_space.shape[0])
    policy = Policy(num_inputs=env.observation_space.shape[0], num_actions=env.action_space.n)
    # opt = optim.Adam(params=policy.parameters(), lr=0.001)
    opt = optim.RMSprop(params=policy.parameters(), lr=0.001)
    print('observation_space', env.observation_space)

    episode = 0
    last = time.time()
    sum_epochs = 0
    sum_rewards = 0
    abandoned_count = 0
    start = time.time()
    while True:
        print('run')
        states, actions, rewards, abandoned = run_episode(env=env, policy=policy)
        total_reward = np.sum(rewards)
        sum_rewards += total_reward
        sum_epochs += 1
        if not abandoned:
            print('learn')
            opt.zero_grad()
            sample = True
            if sample:
                sample_idxes = np.random.choice(len(actions), 32, replace=False)
                states = [states[i] for i in sample_idxes]
                actions = [actions[i] for i in sample_idxes]
                # states = [states[i] for i in sample_idxes]
            for a in actions:
                a.reinforce(total_reward)
            # print('actions', actions)
            autograd.backward(actions, [None] * len(actions))
            opt.step()
            print('done')
        else:
            abandoned_count += 1
        if time.time() - last >= 1.0:
            print('episode %s elapsed %ss avg reward %.1f abondoned_prob %.2f' % (
                episode, int(time.time() - start), sum_rewards / sum_epochs, abandoned_count / sum_epochs))
            sum_epochs = 0
            sum_rewards = 0
            abandoned_count = 0
            last = time.time()
        episode += 1

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--env', type=str, default='cartpole')
    args = parser.parse_args()
    run(**args.__dict__)
