import numpy as np

a = np.zeros((32, 32), dtype=np.uint8)
a.fill(3)
print('a', a)

b = np.zeros((64, 8, 4), dtype = np.float32)
print('b', b)

a = a.reshape(32,8,4)
print('a', a)
b[0:32] = a
print('b', b)

b /= 7
print('b', b)

labels = [1] * 32
c = np.zeros(64, dtype = np.uint8)
c[0:32] = labels
print('c', c)

b[0:32] = a
b[16:32] = 6

mean = b.mean()
print('mean', mean)

std = b.std()
print('std', std)

