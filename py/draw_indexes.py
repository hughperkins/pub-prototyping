import random
import numpy as np


if __name__ == '__main__':
    idxs = np.random.choice(12, replace=False, size=(4, 3))
    print('idxs', idxs)
