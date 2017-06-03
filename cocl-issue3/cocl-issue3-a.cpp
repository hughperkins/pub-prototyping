#include "EasyCL/EasyCL.h"

#include <iostream>
#include <stdexcept>

using namespace std;

string kernelSource = R"(
kernel void _z8setValuePfif(global float* data, long data_offset, int idx, float value) {
   data = (global float*)((global char *)data + data_offset);

   label0:;
   int v1 = get_local_id(0);
    bool v2 = v1 == 0;
    if(v2) {
        goto v4;
    } else {
        goto v5;
    }
    v4:;
    long v6 = idx;
    global float* v7 = (&data[v6]);
    v7[0] = value;
        goto v5;
    v5:;
    return;
}
)";

int main(int argc, char *argv[]) {
    easycl::EasyCL *cl = easycl::EasyCL::createForFirstGpuOtherwiseCpu();
    cl_int err;

    easycl::CLKernel *kernel = cl->buildKernelFromString(kernelSource, "_z8setValuePfif", "");

    int N = 32;
    cl_float *a = new cl_float[N];
    cl_long offset = 0;
    cl_int pos = 2;
    cl_float value = 123;

    for(int i = 0; i < N; i++) {
        a[i] = 555;
    }
    cl_mem a_gpu = clCreateBuffer(*cl->context, CL_MEM_READ_WRITE, sizeof(float) * N, 0, &err);
    easycl::EasyCL::checkError(err);

    err = clEnqueueWriteBuffer(*cl->queue, a_gpu, CL_TRUE, 0,
                                         sizeof(cl_float) * N, a, 0, NULL, NULL);
    easycl::EasyCL::checkError(err);

    kernel
        ->inout(&a_gpu)
        ->in(offset)
        ->in(pos)
        ->in(value);
    kernel->run_1d(N, 32);

    err = clEnqueueReadBuffer(*cl->queue, a_gpu, CL_TRUE, 0,
                                         sizeof(cl_float) * N, a, 0, NULL, NULL);
    easycl::EasyCL::checkError(err);
    cl->finish();
    cout << "clfinish finished ok" << endl;

    for(int i = 0; i < 5; i++) {
        cout << "a[" << i << "]=" << a[i] << endl;
    }

    return 0;
}
