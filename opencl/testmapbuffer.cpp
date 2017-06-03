#include "EasyCL/EasyCL.h"

#include <iostream>
#include <stdexcept>

using namespace std;

string getValueSrc = R"(
kernel void getValue(float *data, int idx, float *out) {
    out[0] = data[idx];
}
)";

string setValueSrc = R"(
kernel void setValue(float *data, int idx, float val) {
    data[idx] = val;
}
)";

string sqrtSrc = R"(
kernel void unarysqrt(global float *d0_, uint offset0, global float *d1_, uint offset1, int N) {
    global float *d0 = d0_ + offset0;
    global float *d1 = d0_ + offset1;
    int tid = get_global_id(0);
    if(tid < N) {
        d0[tid] = sqrt(d1[tid]);
    }
}
)";

int main(int argc, char *argv[]) {
    easycl::EasyCL *cl = easycl::EasyCL::createForFirstGpuOtherwiseCpu();
    cl_int err;

    int bufsize = 32 * 1024 * 1024;

    int N = 10;
    int d0_offset = 10 * 4;
    int d1_offset = 20 * 4;

    cl_mem a_gpu = clCreateBuffer(*cl->context, CL_MEM_READ_WRITE, bufsize, 0, &err);
    easycl::EasyCL::checkError(err);
    cout << "a_gpu " << (void *)a_gpu << endl;

    char *a = (char *)clEnqueueMapBuffer(
        *cl->queue,
        a_gpu, CL_TRUE, CL_MEM_READ_WRITE, 0, bufsize,
        0, 0, 0, &err);
    easycl::EasyCL::checkError(err);
    cout << "a " << (void *)a << endl;

    err= clEnqueueUnmapMemObject(*cl->queue, a_gpu, a, 0, 0, 0);
    easycl::EasyCL::checkError(err);

    a = (char *)clEnqueueMapBuffer(
        *cl->queue,
        a_gpu, CL_TRUE, CL_MEM_READ_WRITE, 0, bufsize,
        0, 0, 0, &err);
    easycl::EasyCL::checkError(err);
    cl->finish();
    cout << "a " << (void *)a << endl;

    float *d0 = (float *)(a + d0_offset);
    float *d1 = (float *)(a + d1_offset);

    d1[0] = 4;
    d1[1] = 3;
    d1[2] = 8;
    d1[3] = 14;
    d1[4] = 19;

    err= clEnqueueUnmapMemObject(*cl->queue, a_gpu, a, 0, 0, 0);
    easycl::EasyCL::checkError(err);
    cl->finish();

    easycl::CLKernel *kernel = cl->buildKernelFromString(sqrtSrc, "unarysqrt", "");

    long o0 = d0_offset / 4;
    long o1 = d1_offset / 4;
    cout << "o0 " << o0 << " o1 " << o1 << endl;

    kernel->inout(&a_gpu);
    kernel->in_uint32(o0);
    //kernel->in((int)(20));

    kernel->inout(&a_gpu);
    kernel->in_uint32(o1);

    kernel->in(N);

    kernel->run_1d(32, 32);
    cl->finish();

    for(int i = 0; i < 5; i++) {
        cout << i << " " << d0[i] << endl;
    }

    for(int i = 0; i < 30; i++) {
        cout << "a[" << i << "] " << ((float *)a)[i] << endl;
    }

    a = (char *)clEnqueueMapBuffer(
        *cl->queue,
        a_gpu, CL_TRUE, CL_MEM_READ_WRITE, 0, bufsize,
        0, 0, 0, &err);
    easycl::EasyCL::checkError(err);
    cout << "a " << (void *)a << endl;
    cl->finish();

    for(int i = 0; i < 5; i++) {
        cout << i << " " << d0[i] << endl;
    }

    for(int i = 0; i < 30; i++) {
        cout << "a[" << i << "] " << ((float *)a)[i] << endl;
    }


    // int N = 32;
    // cl_float *a = new cl_float[N];
    // cl_long offset = 0;
    // cl_int pos = 2;
    // cl_float value = 123;

    // for(int i = 0; i < N; i++) {
    //     a[i] = 555;
    // }
    // cl_mem a_gpu = clCreateBuffer(*cl->context, CL_MEM_READ_WRITE, sizeof(float) * N, 0, &err);
    // easycl::EasyCL::checkError(err);

    // err = clEnqueueWriteBuffer(*cl->queue, a_gpu, CL_TRUE, 0,
    //                                      sizeof(cl_float) * N, a, 0, NULL, NULL);
    // easycl::EasyCL::checkError(err);

    // kernel
    //     ->inout(&a_gpu)
    //     ->in(offset)
    //     ->in(pos)
    //     ->in(value);
    // kernel->run_1d(N, 32);

    // err = clEnqueueReadBuffer(*cl->queue, a_gpu, CL_TRUE, 0,
    //                                      sizeof(cl_float) * N, a, 0, NULL, NULL);
    // easycl::EasyCL::checkError(err);
    // cl->finish();
    // cout << "clfinish finished ok" << endl;

    // for(int i = 0; i < 5; i++) {
    //     cout << "a[" << i << "]=" << a[i] << endl;
    // }

    return 0;
}


