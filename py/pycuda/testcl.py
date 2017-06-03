from __future__ import print_function

import pyopencl as cl

#import pyopencl.driver as cl
#import pyopencl.autoinit
#from pyopencl.compiler import SourceModule
import numpy as np

ctx = cl.create_some_context()
queue = cl.CommandQueue(ctx)

mf = cl.mem_flags

a = np.random.randn(4,4).astype(np.float32)
print('a', a)

#a_gpu = cl.mem_alloc(a.nbytes)
a_gpu = cl.Buffer(ctx, mf.

cl.memcpy_htod(a_gpu, a)

mod = SourceModule("""
kernel void doublify(float *a)
{
  int idx = get_global_id(0);
  a[idx] *= 2;
}
""")

func = mod.get_function("doublify")
func(a_gpu, block=(4,4,1))
a_doubled = np.empty_like(a)
cl.memcpy_dtoh(a_doubled, a_gpu)
print('a_doubled', a_doubled)

