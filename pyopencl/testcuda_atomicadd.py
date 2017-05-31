import pycuda.driver as cuda
import pycuda.autoinit
from pycuda.compiler import SourceModule
import time
import numpy as np

N = 32
its = 1

#a = np.random.randn(N).astype(np.float32)
a = np.zeros((N,), dtype=np.float32)
print('a.shape', a.shape)
a_gpu = cuda.mem_alloc(a.nbytes)
print('copy to gpu...')
cuda.memcpy_htod(a_gpu, a)

mod = SourceModule("""
  __global__ void doublify(float *a)
  {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    
//    float source = 3.0f;
    atomicAdd(&a[0], 3.0f);
  }
  """, keep=True, cache_dir='/tmp/cudaptx')
func = mod.get_function('doublify')
cuda.Context.synchronize()
start = time.time()
print('run kernel...')
blockSize = 32
numBlocks = N // blockSize
for it in range(its):
  func(a_gpu, block=(blockSize,1,1), grid=(numBlocks,1,1))
cuda.Context.synchronize()
end = time.time()
print('kernel done')
print('diff', end - start)
a_doubled = np.empty_like(a)
cuda.memcpy_dtoh(a_doubled, a_gpu)
print('a_doubled', a_doubled)

