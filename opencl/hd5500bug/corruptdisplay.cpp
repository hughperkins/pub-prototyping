#include <sys/types.h>
#include <stdio.h>
#include <string.h>

#include "CL/cl.h"

#include <string>
#include <iostream>
using namespace std;

string kernelSource = R"DELIM(
kernel void corruptDisplay(local float *_buffer) {
}
)DELIM";

string getDeviceInfoString( cl_device_id deviceId, cl_device_info name ) {
    char buffer[256];
    buffer[0] = 0;
    cl_int error = clGetDeviceInfo(deviceId, name, 256, buffer, 0);
    if( error != CL_SUCCESS ) {
        if( error == CL_INVALID_DEVICE ) {
            throw runtime_error("Failed to obtain info for device id " + EasyCL::toString( deviceId ) + ": invalid device" );
        } else if( error == CL_INVALID_VALUE ) {
            throw runtime_error("Failed to obtain device info " + EasyCL::toString( name ) + " for device id " + EasyCL::toString( deviceId ) + ": invalid value" );
        } else {
            throw runtime_error("Failed to obtain device info " + EasyCL::toString( name ) + " for device id " + EasyCL::toString( deviceId ) + ": unknown error code: " + EasyCL::toString( error ) );
        }
    }
    return string( buffer );
}
int main(int argc, char *argv[])
{
    string platformName = "Intel Gen OCL Driver";
    string deviceName = "Intel(R) HD Graphics BroadWell U-Processor GT2";

    cl_int error;
    cl_platform_id platform_ids[10];
    cl_uint num_platforms;
    error = clGetPlatformIDs(10, platform_ids, &num_platforms);
    if (error != CL_SUCCESS) {
       throw std::runtime_error( "Error getting platforms ids");
    }
    if( num_platforms == 0 ) {
       throw std::runtime_error("Error: no platforms available");
    }
    bool foundPlatform = false;
    cl_platform_id platform_id;
    for( int platform =  0; platform < (int)num_platforms; platform++ ) {
        platform_id = platform_ids[platform];
        string thisPlatformName = getPlatformInfoString(platform_id, CL_PLATFORM_NAME );
        if(thisPlatformName == platformName) {
            cout << "Detected platform [" << platformName << "]" << endl;
            foundPlatform = true;
            break;
        }
    }
    if(!foundPlatform) {
        throw std::runtime_error("failed to find platform " + platformName);
    }
    cl_device_id device_ids[100];
    cl_uint num_devices;
    cl_device_id device_id;
    bool foundDevice = false;
    error = clGetDeviceIDs(platform_id, CL_DEVICE_TYPE_GPU, 100, device_ids, &num_devices);
    if (error != CL_SUCCESS) {
         throw std::runtime_error("Error getting device ids for platform");
    }
    for(int i = 0; i < (int)num_devices; i++ ) {
        device_id = device_ids[i];
        string thisDeviceName = getDeviceInfoString(device_id, CL_DEVICE_NAME);
        if(deviceName == thisDeviceName) {
            foundDevice = true;
            cout << "Detected device [" << deviceName << "]" << endl;
            break;
        }
    }
    if(!foundDevice) {
        throw std::runtime_error("failed to find device " + deviceName);
    }


    cl_context context;
    context = clCreateContext(0, 1, &device_id, NULL, NULL, &error);
    if (error != CL_SUCCESS) {
       throw std::runtime_error("Error creating context");
    }
    cl_command_queue queue;
    queue = clCreateCommandQueue(context, device, 0, &error);
    if (error != CL_SUCCESS) {
       throw std::runtime_error("Error creating command queue");
    }


    

    clReleaseCommandQueue(queue);
    clReleaseContext(context);
    return 0;
}

