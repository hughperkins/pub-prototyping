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
typedef union Matrix {
    float4 f4;
    float f[4];
} Matrix;

  __global__ void doublify(Matrix *a)
  {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    
//    Matrix foo[4] = {0};
    float b[4];
    b[0] = 5;
    b[1] = 3;
    b[2] = 4;
    b[3] = 7;
    Matrix foo[4];
    foo[1].f[0] = 3.0f;
    if(tid == 0) {
       for(int i=0; i < 4; i++) {
          a[i] = foo[i];
       }
    }
    
    //a[blockIdx.x * blockDim.x + threadIdx.x] = count;
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

