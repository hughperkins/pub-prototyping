import time
import numpy as np
import pyopencl as cl


N = 1024

dtype = np.float32
a = np.random.rand(N).astype(dtype)

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
    long tid = get_global_id(0);
    data[0] = -INFINITY;
    data[1] = INFINITY;
}
""").build()

prg.mykernel(q, (32,), (32,), a_gpu)
q.finish()
print('kernel done')

a_doubled = np.empty_like(a)
cl.enqueue_copy(q, a_doubled, a_gpu)
print(a_doubled[0])
print(a_doubled[1])
