import numpy as np
import math
from collections import namedtuple


class MountainCar(object):
    def __init__(self):
        self.observation_space = namedtuple('ObservationSpace', ['shape'])([2])
        self.action_space = namedtuple('ActionSpace', ['n'])(3)
        self.reset()

    def reset(self):
        # self.s = np.zeros(-0.6 + np.random.random() * 0.2, 0.0)
        self.x = -0.6 + np.random.random() * 0.2
        self.v = 0.0
        return np.array([self.x, self.v])

    def render(self):
        # print(self.x, self.v)
        x_c = (self.x - (-1.2)) / 1.8 * 80
        print('|' + ' ' * int(x_c) + '*' + (80 - int(x_c)) * ' ' + '| %.1f' % self.x)

    def step(self, a):
        a -= 1
        self.v += 0.001 * a - 0.0025 * math.cos(3 * self.x)
        self.v = max(-0.07, self.v)
        self.v = min(self.v, 0.06999999)
        self.x += self.v
        if self.x < -1.2:
            self.x = -1.2
            self.v = 0.0
        done = self.x >= 0.5
        r = 0 if done else -1
        return np.array([self.x, self.v]), r, done, None


def get_env_by_name(env_name):
    if env_name == 'cartpole':
        import gym
        return gym.make('CartPole-v0')
    elif env_name == 'mountaincar':
        return MountainCar()
    else:
        raise Exception('env %s not recognized' % env_name)
