
#include "clew.h"
#include <iostream>
#include <clBLAS.h>

using namespace std;

int main( int argc, char *argv[] ) {
    cl_int err;
    cl_platform_id platform = 0;
    cl_device_id device = 0;
    cl_context_properties props[3] = { CL_CONTEXT_PLATFORM, 0, 0 };
    cl_context ctx = 0;
    cl_command_queue queue = 0;
    cl_mem bufA, bufB, bufC;
    cl_event event = NULL;
//    int ret = 0;

    clewInit();

    /* Setup OpenCL environment. */
    err = clGetPlatformIDs(1, &platform, NULL);
    if (err != CL_SUCCESS) {
        printf( "clGetPlatformIDs() failed with %d\n", err );
        return 1;
    }

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

    /* Setup clblas. */
    err = clblasSetup();
    if (err != CL_SUCCESS) {
        printf("clblasSetup() failed with %d\n", err);
        clReleaseCommandQueue(queue);
        clReleaseContext(ctx);
        return 1;
    }

    float A[] = {
        1,2,-1,
        3,4,0,
    };
    float B[] = {
        0,1,
        1,2,
        4,5
    };
//    float *C = new float[2 * 3];
    float C[] = {
        1,3,
        4,-2
    };

    int M = 2;
    int K = 3;
    int N = 2;

    /* Prepare OpenCL memory objects and place matrices inside them. */
    bufA = clCreateBuffer(ctx, CL_MEM_READ_ONLY, M * K * sizeof(*A),
                          NULL, &err);
    bufB = clCreateBuffer(ctx, CL_MEM_READ_ONLY, K * N * sizeof(*B),
                          NULL, &err);
    bufC = clCreateBuffer(ctx, CL_MEM_READ_WRITE, M * N * sizeof(*C),
                          NULL, &err);

    err = clEnqueueWriteBuffer(queue, bufA, CL_TRUE, 0,
        M * K * sizeof(*A), A, 0, NULL, NULL);
    err = clEnqueueWriteBuffer(queue, bufB, CL_TRUE, 0,
        K * N * sizeof(*B), B, 0, NULL, NULL);
    err = clEnqueueWriteBuffer(queue, bufC, CL_TRUE, 0,
        M * N * sizeof(*C), C, 0, NULL, NULL);

    size_t lda = K;        /* i.e. lda = K */
    size_t ldb = N;        /* i.e. ldb = N */
    size_t ldc = N;        /* i.e. ldc = N */
//    size_t off  = 0;
//    size_t offA = K + off;   /* K + off */
//    size_t offB = N + off;   /* N + off */
//    size_t offC = N + off;   /* N + off */
//    size_t offA = 0;
//    size_t offB = 0;
//    size_t offC = 0;
    err = clblasSgemm(clblasRowMajor, clblasNoTrans, clblasNoTrans, M, N, K,
                         1, bufA, 0, lda,
                         bufB, 0, ldb, 0,
                         bufC, 0, ldc,
                         1, &queue, 0, NULL, &event);
    if (err != CL_SUCCESS) {
        printf("clblasSgemmEx() failed with %d\n", err);
        return 1;
//        ret = 1;
    }
    else {
        /* Wait for calculations to be finished. */
        err = clWaitForEvents(1, &event);

        /* Fetch results of calculations from GPU memory. */
        err = clEnqueueReadBuffer(queue, bufC, CL_TRUE, 0,
                                  M * N * sizeof(*C),
                                  C, 0, NULL, NULL);
        for( int row = 0; row < M; row++ ) {
            for( int col = 0; col < N; col++ ) {
                cout << C[row * N + col] << " ";
            }
            cout << endl;
        }

        /* At this point you will get the result of SGEMM placed in 'result' array. */
//        puts("");
//        printResult("clblasSgemmEx result");
    }

    /* Finalize work with clblas. */
    clblasTeardown();

    /* Release OpenCL working objects. */
    clReleaseCommandQueue(queue);
    clReleaseContext(ctx);

    return 0;
}

