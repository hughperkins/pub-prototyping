#include <easycl/EasyCL.h>

#include <memory>

#include "OpenCL/cl.h"

using namespace easycl;

// std::string someKernelSource = R"(
// kernel void myKernel(global float *d0, global float *d1, global float *d2, global float *d3, global float *d4) {
//     d0[0] = 123.0f;
//     d1[0] = 123.0f;
//     d2[0] = 123.0f;
//     d3[0] = 123.0f;
//     d4[0] = 123.0f;
// }
// )";

std::string createKernelOne(int numArgs) {
    std::ostringstream oss;
    oss << "kernel void myKernel(\n";
    for(int i = 0; i < numArgs; i++) {
        if(i > 0) {
            oss << ", ";
        }
        oss << "global float *d" << i;
    }
    oss << ") {\n";
    for(int i = 0; i < numArgs; i++) {
        oss << "    d" << i << "[0] = 123.0f;\n";
    }
    oss << "}\n";
    return oss.str();    
}

std::string createKernelTwo(int numArgs) {
    std::ostringstream oss;
    oss << "kernel void myKernel(global float *d";
    for(int i = 0; i < numArgs; i++) {
        oss << ", int offset_d" << i;
    }
    oss << ") {\n";
    for(int i = 0; i < numArgs; i++) {
        oss << "    global float *d" << i << " = d + offset_d" << i << ";\n";
    }
    for(int i = 0; i < numArgs; i++) {
        oss << "    d" << i << "[0] = 123.0f;\n";
    }
    oss << "}\n";
    std::string source = oss.str();
    std::cout << source << std::endl;
    return source;    
}

int main(int argc, char *argv[]) {
    const int numArgs = 4;

    // std::string someKernelSource = createKernelOne(numArgs);
    std::string someKernelSource = createKernelTwo(numArgs);

    std::unique_ptr<EasyCL> cl(EasyCL::createForFirstGpuOtherwiseCpu());

    std::unique_ptr<CLKernel> kernel(cl->buildKernelFromString(someKernelSource, "myKernel", "", "conststring"));

    cl_int err;
    const int bufferSizeMegs = 512;
    const int bufferSize = bufferSizeMegs * 1024 * 1024;
    cl_mem clmem = clCreateBuffer(*cl->context, CL_MEM_READ_WRITE, bufferSize, 0, &err);
    EasyCL::checkError(err);

    for(int i = 0; i < 1000; i++) {
        std::cout << "i=" << i << std::endl;
        // for(int i = 0; i < numArgs; i++) {
            kernel->inout(&clmem);
        // }
        for(int i = 0; i < numArgs; i++) {
            kernel->in(i);
        }
        // kernel->inout(&clmem);
        // kernel->inout(&clmem);
        // kernel->inout(&clmem);
        // kernel->inout(&clmem);
        size_t global[3] = {1024, 1, 1};
        size_t block[3] = {256, 1, 1};
        kernel->run(cl->queue, 3, global, block);
    }

    err = clReleaseMemObject(clmem);
    EasyCL::checkError(err);

    return 0;
}


    // cl_int err;
    // cl_mem gpu_struct = clCreateBuffer(*ctx, CL_MEM_READ_WRITE, structAllocateSize,
    //                                        NULL, &err);
    // EasyCL::checkError(err);
    // err = clEnqueueWriteBuffer(launchConfiguration.queue->queue, gpu_struct, CL_TRUE, 0,
    //                                   structAllocateSize, pCpuStruct, 0, NULL, NULL);
    // EasyCL::checkError(err);
    // launchConfiguration.kernelArgsToBeReleased.push_back(gpu_struct);
    // launchConfiguration.kernel->inout(&launchConfiguration.kernelArgsToBeReleased[launchConfiguration.kernelArgsToBeReleased.size() - 1]);
 

    //  size_t global[3];
    //  COCL_PRINT(cout << "<<< global=dim3(");
    // for(int i = 0; i < 3; i++) {
    //     global[i] = launchConfiguration.grid[i] * launchConfiguration.block[i];
    //     COCL_PRINT(cout << global[i] << ",");
    // }
    // COCL_PRINT(cout << "), workgroupsize=dim3(");
    // for(int i = 0; i < 3; i++) {
    //     COCL_PRINT(cout << launchConfiguration.block[i] << ",");
    // }
    // COCL_PRINT(cout << ")>>>" << endl);
    // // cout << "launching kernel, using OpenCL..." << endl;
    // int workgroupSize = launchConfiguration.block[0] * launchConfiguration.block[1] * launchConfiguration.block[2];
    // COCL_PRINT(cout << "workgroupSize=" << workgroupSize << endl);
    // launchConfiguration.kernel->localInts(workgroupSize);

        // launchConfiguration.kernel->run(launchConfiguration.queue, 3, global, launchConfiguration.block);
