from __future__ import print_function
import numba
import numpy as np
import time
import dis


def bubblesort(X):
    N = len(X)
    for end in range(N, 1, -1):
        for i in range(end - 1):
            cur = X[i]
            if cur > X[i + 1]:
                tmp = X[i]
                X[i] = X[i + 1]
                X[i + 1] = tmp


if __name__ == '__main__':
    print(numba.__version__)
    original = np.arange(0.0, 10.0, 0.01, dtype='f4')
    shuffled = original.copy()
    np.random.seed(123)
    np.random.shuffle(shuffled)
    times = []
    for it in range(5):
        sorted = shuffled.copy()
        start = time.time()
        bubblesort(sorted)
        elapsed = time.time() - start
        times.append(elapsed)
    assert np.array_equal(sorted, original)
    print('')
    print('cpython times:')
    for _time in times:
        print(_time)

    # bubblesort_jit = numba.jit('void(f4[:])')(bubblesort)
    bubblesort_jit = numba.jit(nopython=True)(bubblesort)
    times = []
    for it in range(5):
        sorted = shuffled.copy()
        start = time.time()
        bubblesort_jit(sorted)
        elapsed = time.time() - start
        times.append(elapsed)
    assert np.array_equal(sorted, original)
    print('')
    print('numba jit times:')
    for _time in times:
        print(_time)

    # dis.dis(bubblesort)
