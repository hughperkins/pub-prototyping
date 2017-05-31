import numpy as np
import pyopencl as cl
import argparse
import time


parser = argparse.ArgumentParser()
parser.add_argument('--file', type=str, required=True)
parser.add_argument('--kernelname', type=str, required=True)
args = parser.parse_args()

N = 128
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

with open(args.file, 'r') as f:
    source = f.read()

# prg = cl.Program(ctx, """
# __kernel void sum(__global float *a_g) {
#   a_g[get_global_id(0)] += 1;
# }

# __kernel void testloop(global int *data) {
#     int p = get_global_id(0);
# mylabel0:
#     p++;
#     if(data[p] > 5) {
#         goto mylabel0;
#     }
# }
# """).build()

prg = cl.Program(ctx, source).build()

# run once, just to make sure the buffer copied to gpu
start = time.time()
prg.__getattr__(args.kernelname)(q, (N,), (32,), a_gpu, np.int32(N), np.float32(123.0))
print('queued')

q.finish()
print('done')

print('time', time.time() - start)
