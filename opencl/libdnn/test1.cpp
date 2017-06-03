#include <iostream>
#include <sstream>
#include <stdexcept>
using namespace std;

#include "CL/cl.h"

#include "greentea/libdnn.hpp"

template<typename T>
std::string toString(T val ) { // not terribly efficient, but works...
   std::ostringstream myostringstream;
   myostringstream << val;
   return myostringstream.str();
}

void checkError( cl_int err ) {
    if (err != CL_SUCCESS) {
       throw std::runtime_error( "Error: " + toString(err) );
    }
}

int main( int argc, char *argv[] ) {

     cl_int err;  

    cl_device_id *device_ids;

    cl_uint num_platforms;
    cl_uint num_devices;

    cl_platform_id platform_id;
    cl_device_id device;

    cl_context ctx;
    cl_command_queue queue;
    // cl_program program;

    checkError( clGetPlatformIDs(1, &platform_id, &num_platforms) );
    checkError(  clGetDeviceIDs(platform_id, CL_DEVICE_TYPE_GPU, 1, &device, &num_devices) );
    device_ids = new cl_device_id[num_devices];
    checkError( clGetDeviceIDs(platform_id, CL_DEVICE_TYPE_GPU, num_devices, device_ids, &num_devices) );
    device = device_ids[0];
    ctx = clCreateContext(0, 1, &device, NULL, NULL, &err);
    checkError(err);
    queue = clCreateCommandQueue(ctx, device, 0, &err);
    checkError(err);

    const int batchSize = 32;
    const int size = 5;
    const int planes = 4;
    const int kernelSize = 3;
    float *input = new float[batchSize * size * size * planes];
    float *output = new float[batchSize * size * size * planes];
    float *filters = new float[kernelSize * kernelSize * planes * planes];

    cl_mem bufInput = clCreateBuffer(ctx, CL_MEM_READ_ONLY, batchSize * size * size * planes * sizeof(float),
                        NULL, &err);
    cl_mem bufOutput = clCreateBuffer(ctx, CL_MEM_READ_WRITE, batchSize * size * size * planes * sizeof(float),
                        NULL, &err);
    cl_mem bufFilters = clCreateBuffer(ctx, CL_MEM_READ_ONLY, kernelSize * kernelSize * planes * planes * sizeof(float),
                        NULL, &err);
     err = clEnqueueWriteBuffer(queue, bufInput, CL_TRUE, 0,
      batchSize * size * size * planes * sizeof(float), input, 0, NULL, NULL);
     err = clEnqueueWriteBuffer(queue, bufOutput, CL_TRUE, 0,
      batchSize * size * size * planes * sizeof(float), output, 0, NULL, NULL);
     err = clEnqueueWriteBuffer(queue, bufFilters, CL_TRUE, 0,
      kernelSize * kernelSize * planes * planes * sizeof(float), filters, 0, NULL, NULL);

    int gpu_id = 0;
    greentea::device::setupViennaCLContext(gpu_id, ctx, device, queue);

    checkError( clFinish( queue ) );
    // checkError( clEnqueueReadBuffer( queue, bufData, CL_TRUE, 0, sizeof(float) * N, data, 0, NULL, NULL) );    
    // checkError( clEnqueueReadBuffer( queue, bufOut, CL_TRUE, 0, sizeof(float) * outN, out, 0, NULL, NULL) );    
    // checkError( clFinish( queue ) );

    // for( int i = 0; i < N; i++ ) {
    //    cout << data[i] << " ";
    // }
    // cout << endl;
    // for( int i = 0; i < outN; i++ ) {
    //    cout << out[i] << " ";
    // }
    // cout << endl;

    return 0;
}
