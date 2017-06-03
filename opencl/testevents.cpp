#include "CL/cl.h"
#include <iostream>
#include <cstdio>


using namespace std;

string kernelSource = R"(
kernel void mykernel(global float *data, int N, float value) {
    for(int i = 0; i < N; i++) {
        data[i] = value;
    }
}
)";

int main(int argc, char *argv[]) {
    cl_int err;
    cl_platform_id platform = 0;
    cl_device_id device = 0;
    cl_context_properties props[3] = { CL_CONTEXT_PLATFORM, 0, 0 };
    cl_context ctx = 0;
    cl_command_queue queue = 0;
    cl_mem bufA, bufB, bufC;
    cl_event event = NULL;

    err = clGetPlatformIDs(1, &platform, NULL);
    if (err != CL_SUCCESS) {
        printf( "clGetPlatformIDs() failed with %d\n", err );
        return 1;
    }
    cout << "got platforms" << endl;

    err = clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU, 1, &device, NULL);
    if (err != CL_SUCCESS) {
        printf( "clGetDeviceIDs() failed with %d\n", err );
        return 1;
    }

    props[1] = (cl_context_properties)platform;
    ctx = clCreateContext(props, 1, &device, NULL, NULL, &err);
    if (err != CL_SUCCESS) {
        printf( "clCreateContext() failed with %d\n", err );
        return 1;
    }

    queue = clCreateCommandQueue(ctx, device, 0, &err);
    if (err != CL_SUCCESS) {
        printf( "clCreateCommandQueue() failed with %d\n", err );
        clReleaseContext(ctx);
        return 1;
    }

    size_t src_size = kernelSource.size();
    const char *source_pchar = kernelSource.c_str();
    cl_program program = clCreateProgramWithSource(ctx, 1, &source_pchar, &src_size, &err);
    if (err != CL_SUCCESS) {
        printf( "clCreateProgramWithSource() failed with %d\n", err );
        clReleaseContext(ctx);
        return 1;
    }

//    error = clBuildProgram(program, 1, &device, "-cl-opt-disable", NULL, NULL);
//    std::cout << "options: [" << options.c_str() << "]" << std::endl;
    string options = "";
    err = clBuildProgram(program, 1, &device, options.c_str(), NULL, NULL);

    char* build_log;
    size_t log_size;
    err = clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG, 0, NULL, &log_size);
    if (err != CL_SUCCESS) {
        printf( "clGetProgramBuildInfo() failed with %d\n", err );
        clReleaseContext(ctx);
        return 1;
    }
    build_log = new char[log_size+1];
    err = clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG, log_size, build_log, NULL);
    if (err != CL_SUCCESS) {
        printf( "clGetProgramBuildInfo() failed with %d\n", err );
        clReleaseContext(ctx);
        return 1;
    }
    build_log[log_size] = '\0';
    string buildLogMessage = "";
    string sourceFilename = "filename";
    if(log_size > 2) {
        buildLogMessage = sourceFilename + " build log: "  + "\n" + build_log;
        cout << buildLogMessage << endl;
    }
    delete[] build_log;
    // checkError(error);

    string kernelName = "mykernel";
    cl_kernel kernel = clCreateKernel(program, kernelName.c_str(), &err);
    if(err != CL_SUCCESS) {
        // vector<std::string> splitSource = easycl::split(source, "\n");
        // std::string sourceWithNumbers = "\nkernel source:\n";
        // for(int i = 0; i < (int)splitSource.size(); i++) {
        //     sourceWithNumbers += toString(i + 1) + ": " + splitSource[i] + "\n";
        // }
        // sourceWithNumbers += "\n";
        // std::string exceptionMessage = "";
        // switch(error) {
        //     case -46:
        //         exceptionMessage = sourceWithNumbers + "\nInvalid kernel name, code -46, kernel " + kernelname + "\n" + buildLogMessage;
        //         break;
        //     default:
        //         exceptionMessage = sourceWithNumbers + "\nSomething went wrong with clCreateKernel, OpenCL erorr code " + toString(error) + "\n" + buildLogMessage;
        //         break;
        // }
        // cout << "kernel build error:\n" << exceptionMessage << endl;
        cout << "kernel build error" << endl;
        return 1;
    }
    // checkError(error);


    int N = 32;
    float value = 123.0f;
    cl_mem floatsbuf = clCreateBuffer(ctx, CL_MEM_ALLOC_HOST_PTR, N * sizeof(float),
                                           NULL, &err);
    if(err != CL_SUCCESS) {
        throw runtime_error("floatsbuf create failed");
    }

    float *floats = 0;
   if(false) {
    floats = (float *)clEnqueueMapBuffer(
        queue,
        floatsbuf,
        CL_FALSE,
        CL_MAP_READ | CL_MAP_WRITE,
        0,
        N * sizeof(float),
        0,
        0,
        0,
        &err);
    if(err != CL_SUCCESS) {
        throw runtime_error("map1 failed");
    }

     err = clEnqueueUnmapMemObject (
        queue,
        floatsbuf,
        floats,
        0,
        0,
        0
    );
    if(err != CL_SUCCESS) {
        throw runtime_error("unmap1 failed");
    }
    }

    // cl_mem Nbuf = clCreateBuffer(*ctx, CL_MEM_ALLOC_HOST_PTR, bytes,
    //                                        NULL, &err);
    // if(err != CL_SUCCESS) {
    //     throw runtime_error("Nbuf create failed");
    // }
    // cl_mem valuebuf = clCreateBuffer(*ctx, CL_MEM_ALLOC_HOST_PTR, sizeof(float),
    //                                        &value, &err);
    // if(err != CL_SUCCESS) {
    //     throw runtime_error("valuebuf create failed");
    // }

    err = clSetKernelArg(kernel, 0, sizeof(cl_mem), &floatsbuf);
    if(err != CL_SUCCESS) {
        throw runtime_error("arg0 failed");
    }
    err = clSetKernelArg(kernel, 1, sizeof(int), &N);
    if(err != CL_SUCCESS) {
        throw runtime_error("arg1 failed");
    }
    err = clSetKernelArg(kernel, 2, sizeof(float), &value);
    if(err != CL_SUCCESS) {
        throw runtime_error("arg2 failed");
    }

    size_t global_ws[3];
    size_t local_ws[3];
    for(int i = 0; i < 3; i++) {
        global_ws[i] = 0;
        local_ws[i] = 0;
    }
    global_ws[0] = 32;
    local_ws[0] = 32;
    err = clEnqueueNDRangeKernel(queue, kernel, 1, NULL, global_ws, local_ws, 0, NULL, 0);
    if(err != CL_SUCCESS) {
        cout << "err" << err << endl;
        throw runtime_error("clenqueuendrangekernel failed");
    }
    cout << "queued kernel ok" << endl;

    floats = (float *)clEnqueueMapBuffer(
        queue,
        floatsbuf,
        CL_FALSE,
        CL_MAP_READ | CL_MAP_WRITE,
        0,
        N * sizeof(float),
        0,
        0,
        0,
        &err);
    if(err != CL_SUCCESS) {
        throw runtime_error("map2 failed");
    }
    cout << "map2 returned" << endl;

    err = clFinish(queue);
    if(err != CL_SUCCESS) {
        throw runtime_error("finish failed");
    }
    cout << "clfinish finished ok" << endl;

    return 0;
}
