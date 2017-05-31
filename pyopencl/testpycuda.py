import pycuda.driver as cuda
import pycuda.autoinit
from pycuda.compiler import SourceModule
import time
import numpy as np

div = int(1e2)
N = (int(1e8//div) // 1024) * 1024
its = 10 * div

a = np.random.randn(N).astype(np.float32)
print('a.shape', a.shape)
a_gpu = cuda.mem_alloc(a.nbytes)
print('copy to gpu...')
cuda.memcpy_htod(a_gpu, a)

mod = SourceModule("""
  __global__ void doublify(float *a)
  {
    int foo = (int)a[blockIdx.x * blockDim.x + threadIdx.x];
    int ballot_res = __ballot(foo);
    int count = __popc(ballot_res);
    a[blockIdx.x * blockDim.x + threadIdx.x] = count;
  }
  """, keep=True, cache_dir='/tmp/cudaptx')
func = mod.get_function('doublify')
cuda.Context.synchronize()
start = time.time()
print('run kernel...')
blockSize = 1024
numBlocks = N // blockSize
for it in range(its):
  func(a_gpu, block=(blockSize,1,1), grid=(numBlocks,1,1))
cuda.Context.synchronize()
end = time.time()
print('kernel done')
print('diff', end - start)
a_doubled = np.empty_like(a)
cuda.memcpy_dtoh(a_doubled, a_gpu)


