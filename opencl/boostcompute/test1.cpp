#include <iostream>
#include <sstream>
#include <stdexcept>
using namespace std;

#include "CL/cl.h"

#include <boost/compute/core.hpp>
#include <boost/compute/iterator/buffer_iterator.hpp>
#include <boost/compute/algorithm/reverse.hpp>
#include <boost/compute/algorithm/transform_if.hpp>
#include <boost/compute/lambda.hpp>
#include <boost/compute/iterator/zip_iterator.hpp>

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
    cl_program program;

    checkError( clGetPlatformIDs(1, &platform_id, &num_platforms) );
    checkError(  clGetDeviceIDs(platform_id, CL_DEVICE_TYPE_GPU, 1, &device, &num_devices) );
    device_ids = new cl_device_id[num_devices];
    checkError( clGetDeviceIDs(platform_id, CL_DEVICE_TYPE_GPU, num_devices, device_ids, &num_devices) );
    device = device_ids[0];
    ctx = clCreateContext(0, 1, &device, NULL, NULL, &err);
    checkError(err);
    queue = clCreateCommandQueue(ctx, device, 0, &err);
    checkError(err);

     const int N = 5;
     const int outN = 3;
    float mask[N] = {1,1,0,0,1};
    float data[N] = {3,4,5,1,2};
    float out[outN] = {0,0,0};
    cl_mem bufMask = clCreateBuffer(ctx, CL_MEM_READ_ONLY, N * sizeof(*mask),
                        NULL, &err);
    cl_mem bufData = clCreateBuffer(ctx, CL_MEM_READ_ONLY, N * sizeof(*data),
                        NULL, &err);
    cl_mem bufOut = clCreateBuffer(ctx, CL_MEM_READ_WRITE, outN * sizeof(*out),
                        NULL, &err);
     err = clEnqueueWriteBuffer(queue, bufMask, CL_TRUE, 0,
      N * sizeof(*mask), mask, 0, NULL, NULL);
     err = clEnqueueWriteBuffer(queue, bufData, CL_TRUE, 0,
      N * sizeof(*data), data, 0, NULL, NULL);
     err = clEnqueueWriteBuffer(queue, bufOut, CL_TRUE, 0,
      outN * sizeof(*data), out, 0, NULL, NULL);

    boost::compute::context boost_context(ctx);
    boost::compute::command_queue boost_queue(queue);

    boost::compute::buffer boostData(bufData);
    boost::compute::buffer boostMask(bufMask);
    boost::compute::buffer boostOut(bufOut);

    // // reverse the values in the buffer
    // boost::compute::reverse(
    //     boost::compute::make_buffer_iterator<float>(boostData, 0),
    //     boost::compute::make_buffer_iterator<float>(boostData, 5),
    //     boost_queue
    // );

    transform_if(
      make_zip_iterator(
        boost::make_tuple(
        boost::compute::make_buffer_iterator<float>(boostData, 0),
        boost::compute::make_buffer_iterator<float>(boostMask, 0)
        )
      ),
      make_zip_iterator(
        boost::make_tuple(
        boost::compute::make_buffer_iterator<float>(boostData, 5),
        boost::compute::make_buffer_iterator<float>(boostMask, 5)
        )
      ),
      boost::compute::make_buffer_iterator<float>(boostOut, 0),
      boost::compute::get<0>(), // function that return input value
      boost::compute::lambda::get<1>(boost::compute::_1) == 1, // lambda function that checks if mask is 1
      boost_queue // command queue (boost::compute::command_queue object)
    );

    checkError( clFinish( queue ) );
    checkError( clEnqueueReadBuffer( queue, bufData, CL_TRUE, 0, sizeof(float) * N, data, 0, NULL, NULL) );    
    checkError( clEnqueueReadBuffer( queue, bufOut, CL_TRUE, 0, sizeof(float) * outN, out, 0, NULL, NULL) );    
    checkError( clFinish( queue ) );

    for( int i = 0; i < N; i++ ) {
       cout << data[i] << " ";
    }
    cout << endl;
    for( int i = 0; i < outN; i++ ) {
       cout << out[i] << " ";
    }
    cout << endl;

    return 0;
}



