#include <easycl/EasyCL.h>

#include <memory>

#include "OpenCL/cl.h"

using namespace easycl;

struct GlobalInfo {
    unsigned long buffer_addresses[10];
};

static std::string kernelSource = R"(
struct GlobalInfo {
    unsigned long buffer_addresses[10];
};

kernel void addBuffer(global struct GlobalInfo *globalInfo, int index, global float *buffer) {
    globalInfo[0].buffer_addresses[index] = (unsigned long)buffer;
}

kernel void useBuffers(global struct GlobalInfo *globalInfo) {
    global float *b0 = (global float *)(globalInfo[0].buffer_addresses[0]);
    //global float *b1 = (global float *)(globalInfo[0].buffer_addresses[1]);
    //b0[0] = b1[0] + 3.0f;
    b0[0] = 2.34f;
}
)";

int main(int argc, char *argv[]) {
    const int numArgs = 4;

    std::unique_ptr<EasyCL> cl(EasyCL::createForFirstGpuOtherwiseCpu());
    std::unique_ptr<CLKernel> addBuffer(cl->buildKernelFromString(kernelSource, "addBuffer", "", "conststring"));
    std::unique_ptr<CLKernel> useBuffers(cl->buildKernelFromString(kernelSource, "useBuffers", "", "conststring"));

    cl_int err;
    const int N = 1024;

    cl_mem clmem0 = clCreateBuffer(*cl->context, CL_MEM_READ_WRITE, N * 4, 0, &err);
    EasyCL::checkError(err);

    cl_mem clmem1 = clCreateBuffer(*cl->context, CL_MEM_READ_WRITE, N * 4, 0, &err);
    EasyCL::checkError(err);

    cl_mem globalInfo = clCreateBuffer(*cl->context, CL_MEM_READ_WRITE, sizeof(struct GlobalInfo), 0, &err);
    EasyCL::checkError(err);

    addBuffer->inout(&globalInfo);
    addBuffer->in((int)0);
    addBuffer->inout(&clmem0);
    addBuffer->run_1d(cl->queue, 32, 32);

    addBuffer->inout(&globalInfo);
    addBuffer->in((int)1);
    addBuffer->inout(&clmem1);
    addBuffer->run_1d(cl->queue, 32, 32);
    cl->finish();

    float *b1 = new float[N];
    float *b0 = new float[N];

    b1[0] = 2.0f;
    err = clEnqueueWriteBuffer(*cl->queue, clmem1, CL_TRUE, 0,
                                      N * 4, b1, 0, NULL, NULL);
    EasyCL::checkError(err);
    cl->finish();

    useBuffers->inout(&globalInfo);
    useBuffers->run_1d(cl->queue, 32, 32);
    cl->finish();

    err = clEnqueueReadBuffer(*cl->queue, clmem0, CL_TRUE, 0,
                                         sizeof(cl_float) * N, b0, 0, NULL, NULL);
    easycl::EasyCL::checkError(err);

    err = clEnqueueReadBuffer(*cl->queue, clmem1, CL_TRUE, 0,
                                         sizeof(cl_float) * N, b1, 0, NULL, NULL);
    easycl::EasyCL::checkError(err);

    cl->finish();

    std::cout << "b1[0]" << b1[0] << std::endl;

    std::cout << "b0[0]" << b0[0] << std::endl;
    assert(b0[0] == 5.0f);

    err = clReleaseMemObject(clmem0);
    clReleaseMemObject(clmem1);
    clReleaseMemObject(globalInfo);
    EasyCL::checkError(err);

    return 0;
}
