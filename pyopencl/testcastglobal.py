import subprocess
import pyopencl as cl


gpu_idx = 0
platforms = cl.get_platforms()
i = 0
for platform in platforms:
    gpu_devices = platform.get_devices(device_type=cl.device_type.GPU)
    if gpu_idx < i + len(gpu_devices):
        context = cl.Context(devices=[gpu_devices[gpu_idx - i]])
        break
    i += len(gpu_devices)

print('context', context)
q = cl.CommandQueue(context)

mf = cl.mem_flags


kernel_source = """struct Eigen__TensorEvaluator {
    global float* f0;
    float f1;
};

kernel void foo(global float *floats0, global float* floats1) {
    global struct Eigen__TensorEvaluator *pstruct = (global struct Eigen__TensorEvaluator *)floats0;
    pstruct[0].f0 = floats1;
    pstruct[0].f0[0] = pstruct[0].f1;
}
"""

k2 = """
kernel void foo(global long *data, global float *floats0) {
    global float *global*data_as_ppfloat = (global float *global*)data;
    data_as_ppfloat[0] = floats0;
}
"""


def run():
    cl.Program(context, k2).build()
    print('compiled ok')

if __name__ == '__main__':
    run()
