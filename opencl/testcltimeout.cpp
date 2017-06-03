#include <iostream>
#include <sstream>
#include <stdexcept>
using namespace std;

#include "CL/cl.hpp"

template<typename T>
std::string toString(T val ) { // not terribly efficient, but works...
   std::ostringstream myostringstream;
   myostringstream << val;
   return myostringstream.str();
}

void checkError( cl_int error ) {
    if (error != CL_SUCCESS) {
       throw std::runtime_error( "Error: " + toString(error) );
    }
}

int main( int argc, char *argv[] ) {

     cl_int error;  

    cl_device_id *device_ids;

    cl_uint num_platforms;
    cl_uint num_devices;

    cl_platform_id platform_id;
    cl_device_id device;

    cl_context context;
    cl_command_queue queue;
    cl_program program;

    checkError( clGetPlatformIDs(1, &platform_id, &num_platforms) );
    checkError(  clGetDeviceIDs(platform_id, CL_DEVICE_TYPE_GPU, 1, &device, &num_devices) );
    device_ids = new cl_device_id[num_devices];
    checkError( clGetDeviceIDs(platform_id, CL_DEVICE_TYPE_GPU, num_devices, device_ids, &num_devices) );
    device = device_ids[0];
    context = clCreateContext(0, 1, &device, NULL, NULL, &error);
    checkError(error);
    queue = clCreateCommandQueue(context, device, 0, &error);
    checkError(error);

    string kernel_source = string( "kernel void test_read( const int one,  const int two, global int *out) {\n" ) +
    "    const int globalid = get_global_id(0);\n" +
    "    int sum = 0;\n" +
    "    int n = 0;\n" +
    "    while( n < 100000 ) {\n" +
    "        sum = (sum + one ) % 1357 * two;\n" +
    "        n++;\n" +
    "    }\n" +
    "    out[globalid] = sum;\n" +
    "}\n";
    const char *source_char = kernel_source.c_str();
    size_t src_size = strlen( source_char );
    program = clCreateProgramWithSource(context, 1, &source_char, &src_size, &error);
    checkError(error);

    checkError( clBuildProgram(program, 1, &device, 0, NULL, NULL) );

    cl_kernel kernel = clCreateKernel(program, "test_read", &error);
    checkError(error);

    const int N = 4500000;
    int *out = new int[N];
    if( out == 0 ) throw runtime_error("couldnt allocate array");

    int c1 = 3;
    int c2 = 7;
    checkError( clSetKernelArg(kernel, 0, sizeof(int), &c1 ) );
    checkError( clSetKernelArg(kernel, 1, sizeof(int), &c2 ) );
    cl_mem outbuffer = clCreateBuffer(context, CL_MEM_WRITE_ONLY, sizeof(int) * N, 0, &error);
    checkError(error);
    checkError( clSetKernelArg(kernel, 2, sizeof(cl_mem), &outbuffer) );

    size_t globalSize = N;
    size_t workgroupsize = 512;
    globalSize = ( ( globalSize + workgroupsize - 1 ) / workgroupsize ) * workgroupsize;
    checkError( clEnqueueNDRangeKernel( queue, kernel, 1, NULL, &globalSize, &workgroupsize, 0, NULL, NULL) );
    checkError( clFinish( queue ) );
    checkError( clEnqueueReadBuffer( queue, outbuffer, CL_TRUE, 0, sizeof(int) * N, out, 0, NULL, NULL) );    
    checkError( clFinish( queue ) );

    for( int i = 0; i < N; i++ ) {
       if( out[i] != 4228 ) {
           cout << "out[" << i << "] != 4228: " << out[i] << endl;
           exit(-1);
       }
    }

    return 0;
}

