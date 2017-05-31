import pycuda.driver as cuda
import pycuda.autoinit
from pycuda.compiler import SourceModule
import time
import numpy as np

print('np.float32.itemsize', np.float32.itemsize)
print('np.float32.itemsize * 8', np.float32.itemsize * 8)
#a = np.random.randn(N).astype(np.float32)
#print('a.shape', a.shape)
a_gpu = cuda.mem_alloc(1024*1024*128)

