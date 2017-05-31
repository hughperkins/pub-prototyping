import time
import numpy as np
import pyopencl as cl

N = 32
its = 1

a = np.random.rand(N).astype(np.float32) - 0.5
b = np.zeros((N,), dtype=np.uint32)

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
b_gpu = cl.Buffer(ctx, mf.READ_WRITE | mf.COPY_HOST_PTR, hostbuf=b)

#prg = cl.Program(ctx, """
#inline uint popcnt(const uint i) {
#  uint n;
#  asm("popc.b32 %0, %1;" : "=r"(n) : "r" (i));
#  return n;
#}

#__kernel void sum(__global float *a_g, __global float *b_g) {
#  int res = a_g[get_global_id(0)] > 0.0f;
#  uint popc_res = popc(res);
#  b_g[get_global_id(0)] = popc_res;
#//  a_g[get_global_id(0)] += 1;
#}
#""").build()

prg = cl.Program(ctx, """
inline uint popcnt(const uint i) {
  uint n;
  asm("popc.b32 %0, %1;" : "=r"(n) : "r" (i));
  return n;
}

inline uint ballot(const uint i) {
  uint n;
  asm(
    "{\\n\\t"
    "setp.ne.u32 %%p1, %1, 0;\\n\\t"
    "vote.ballot.b32 %0, %%p1;\\n\\t"
    "}"
     : "=r"(n)
     : "r" (i)
  );
  return n;
}

__kernel void sum(__global float *a_g, __global unsigned int *b_g) {
  uint res = 0;
  asm("mov.u32 %0, %%laneid;" : "=r"(res));
  unsigned int res2;
//  uint comp;
  res = a_g[0];
 // res += 23;
//  res = res > 37 ? 5 : 99;
  asm(
  //".reg .pred %%p<2>;"
  "mov.u32 %0, %1;" 
  //"add.u32 %0, %0, 7;"
  //"mov.u32 %0, %%laneid;"
//  "setp.gt.u32 %%p1, %0, 12;"
 // "@%%p1 mov.u32 %0, 33;"
    : "=r"(res2)
    : "r"(res)
  );
  res2 = a_g[get_global_id(0)] > 0 ? 1 : 0;
  res = ballot(res2) & 0xffffffff;
  //res2 = popcnt(res2);
  b_g[get_global_id(0)] = get_global_id(0) == 31 ? res : res2;
//  if(get_global_id(0) == 0) {
 //    a_g[31] = res;
  //}
}
""").build()

# run once, just to make sure the buffer copied to gpu
q.finish()
start = time.time()

prg.sum(q, (N,), (N,), a_gpu, b_gpu)

q.finish()

q.finish()
end = time.time()
print('kernel done')
print('diff', end - start)

a_doubled = np.empty_like(a)
cl.enqueue_copy(q, a_doubled, a_gpu)

b_doubled = np.empty_like(b, dtype=np.uint32)
cl.enqueue_copy(q, b_doubled, b_gpu)

print('a', a)
print('a_doubled', a_doubled)
print('b_doubled', b_doubled)

