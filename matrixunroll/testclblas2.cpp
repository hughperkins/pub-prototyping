#include "clew.h"
#include <iostream>
#include <clBLAS.h>
#include "OpenCLHelper.h"

using namespace std;

int main( int argc, char *argv[] ) {
    if( !OpenCLHelper::isOpenCLAvailable() ) {
        cout << "opencl library not found" << endl;
        exit(1);
    }
    cout << "found opencl library" << endl;
    OpenCLHelper *cl = OpenCLHelper::createForFirstGpuOtherwiseCpu();

    cl_int err;

    err = clblasSetup();
    if (err != CL_SUCCESS) {
        printf("clblasSetup() failed with %d\n", err);
        delete cl;
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
    float C[] = {
        1,3,
        4,-2
    };

    int M = 2;
    int K = 3;
    int N = 2;

    CLFloatWrapper *Awrap = cl->wrap( M * K, A );
    CLFloatWrapper *Bwrap = cl->wrap( K * N, B );
    CLFloatWrapper *Cwrap = cl->wrap( M * N, C );
    Awrap->copyToDevice();
    Bwrap->copyToDevice();
    Cwrap->copyToDevice();

    size_t lda = K;        /* i.e. lda = K */
    size_t ldb = N;        /* i.e. ldb = N */
    size_t ldc = N;        /* i.e. ldc = N */

    cl_event event = NULL;
    err = clblasSgemm(clblasRowMajor, clblasNoTrans, clblasNoTrans, M, N, K,
                         1, Awrap->getBuffer(), 0, lda,
                         Bwrap->getBuffer(), 0, ldb, 0,
                         Cwrap->getBuffer(), 0, ldc,
                         1, cl->queue, 0, NULL, &event);
    if (err != CL_SUCCESS) {
        printf("clblasSgemmEx() failed with %d\n", err);
        delete cl;
        return 1;
    }
    else {
        /* Wait for calculations to be finished. */
        err = clWaitForEvents(1, &event);

        Cwrap->copyToHost();
        for( int row = 0; row < M; row++ ) {
            for( int col = 0; col < N; col++ ) {
                cout << C[row * N + col] << " ";
            }
            cout << endl;
        }
    }

    /* Finalize work with clblas. */
    clblasTeardown();

    delete cl;

    return 0;
}

