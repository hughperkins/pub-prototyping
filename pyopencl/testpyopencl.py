import numpy as np
import pyopencl as cl

N = 32
its = 3

a = np.random.rand(N).astype(np.float32)

gpu_idx = 0

platforms = cl.get_platforms()
i = 0
for platform in platforms:
    gpu_devices = platform.get_devices(device_type=cl.device_type.GPU)
    if gpu_idx < i + len(gpu_devices):
        ctx = cl.Context(devices=[gpu_devices[gpu_idx - i]])
        break
    i += len(gpu_devices)

print('context', ctx)
q = cl.CommandQueue(ctx)

mf = cl.mem_flags
a_gpu = cl.Buffer(ctx, mf.READ_WRITE | mf.COPY_HOST_PTR, hostbuf=a)

prg = cl.Program(ctx, """
__kernel void mykernel(global float *data) {
    int tid = get_global_id(0);
    data[tid] = 123;
}
""").build()

print('run kernel...')
workgroupsize = 32
global_size = ((N + workgroupsize - 1) // workgroupsize) * workgroupsize
for it in range(its):
    prg.mykernel(q, (global_size,), (workgroupsize,), a_gpu)


a_res = np.empty_like(a)
cl.enqueue_copy(q, a_res, a_gpu)

q.finish()
print('kernel done')

print('a_res[:5]', a_res[:5])
