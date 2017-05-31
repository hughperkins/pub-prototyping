#include <iostream>
using namespace std;

#include <omp.h>
#include <CL/cl.h>

#include "sleep.h"

int main( int argc, char *argv[] ) {
    const char *myname = "hello";
    int values[4];
    int count = 0;
    #pragma omp parallel num_threads(4)
    {
        int numthreads = omp_get_num_threads();
        int rank = omp_get_thread_num();
        values[rank] = rank * 3;
        #pragma omp critical
            cout << myname << " thread " << rank << " / " << numthreads << endl;
        count += 1;

        #pragma omp critical
        {
        cl_int error = 0;
        cl_uint num_platforms;
        cl_platform_id platform_id;
        error = clGetPlatformIDs(1, &platform_id, &num_platforms);
        cout << "num platforms: " << num_platforms << endl;
        //assert (num_platforms == 1);
        //assert (error == CL_SUCCESS);
        }

        sleepmillis(1000);
        #pragma omp critical
            cout << "thread " << rank << " / " << numthreads << " done" << endl;
    }
    cout << "count: " << count << endl;
    for( int i = 0; i < 4; i++ ) {
        cout << values[i] << endl;
    }
    return 0;
}

