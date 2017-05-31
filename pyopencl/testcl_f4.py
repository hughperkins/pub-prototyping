import time
import numpy as np
import pyopencl as cl

N = 32
its = 1

#a = np.random.rand(N).astype(np.float32)
a = np.zeros((N,), dtype=np.float32)
a.fill(-1)

gpu_idx = 0

platforms = cl.get_platforms()
i = 0
for platform in platforms:
   gpu_devices = platform.get_devices(device_type=cl.device_type.GPU)
   if gpu_idx < i + len(gpu_devices):
       ctx = cl.Context(devices=[gpu_devices[gpu_idx-i]])
       break
   i += len(gpu_devices)

print('context', ctx)
#ctx = cl.create_some_context()
q = cl.CommandQueue(ctx)

mf = cl.mem_flags
a_gpu = cl.Buffer(ctx, mf.READ_WRITE | mf.COPY_HOST_PTR, hostbuf=a)

prg = cl.Program(ctx, """
typedef union Matrix {
    float4 f4;
    float f[4];
} Matrix;

__kernel void sum(__global Matrix *a_g) {
  if(get_global_id(0) == 1) {
    Matrix foo[4]; // = {(float4)0.0f,(float4)0.0f,(float4)0.0f,(float4)0.0f};
    for(int i = 0; i < 4; i++) {
       foo[i].f4 = (float4)0.0f;
    }
    foo[1].f4 = (float4)3;
    for(int i = 0; i < 4; i++) {
       a_g[i].f4 = foo[i].f4;
    }
//     a_g[get_global_id(0)].f4 = (float4)0.0f;
 //    a_g[get_global_id(0)].f[1] = 4.0f;
  }
}
""").build()

# run once, just to make sure the buffer copied to gpu
prg.sum(q, (N,), (32,), a_gpu)

q.finish()
start = time.time()
print('run kernel...')

q.finish()
end = time.time()
print('kernel done')
print('diff', end - start)

a_doubled = np.empty_like(a)
cl.enqueue_copy(q, a_doubled, a_gpu)

#print('a', a)
print('a_doubled', a_doubled)

