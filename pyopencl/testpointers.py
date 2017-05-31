import time
import numpy as np
import pyopencl as cl


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

source = """
struct MyStruct {
    int myint;
};

struct S2 {
    constant struct MyStruct *s;
};

struct S3 {
    struct MyStruct *s;
};

struct S4 {
    global struct MyStruct *s;
};

constant struct MyStruct foo = { 345 };
constant struct S2 bar = { &foo };
constant struct S3 bar2 = { 0 };

kernel void mykernel(global float *float_data, global struct S3 *s3, global struct S4 *s4, global struct MyStruct *astruct) {
    constant struct MyStruct *f = &foo;
    constant struct MyStruct **g = &f;
    constant struct MyStruct *h = bar.s;
    struct MyStruct *i = bar2.s;

    struct MyStruct *j = s3->s;
    global struct S3 *k = s3;
    global struct S3 **l = &k;
    struct MyStruct **m = &j;

    s4->s = astruct;
}
"""

for i, line in enumerate(source.split('\n')):
    print(i + 1, line)

prg = cl.Program(ctx, source).build()
