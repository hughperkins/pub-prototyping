#include <sys/time.h>
#include <stdio.h>
#include <unistd.h>
#include <iostream>
using namespace std;

#include "EasyCL.h"
#include "CLKernel_structs.h"

#define MAX_CLTORCH_DIMS 25

typedef struct TensorInfoCl {
  unsigned int sizes[MAX_CLTORCH_DIMS]; // note: this is redundant between a/b
  unsigned int strides[MAX_CLTORCH_DIMS];
  int offset;
  int dims; //redundant
} TensorInfoCl;

double getSystemMilliseconds() {
    #ifdef WINNOCHRONO
      DWORD thistime = timeGetTime();
      return thistime;
    #else // linux etc
      struct timeval now;
      gettimeofday(&now, NULL);
      double mtime = now.tv_sec * 1000.0 + now.tv_usec/1000.0;
      return mtime;
    #endif
}

int main(int argc, char *argv[] ) {
  EasyCL *cl = EasyCL::createForIndexedGpu(0);

  CLKernel *kernel = cl->buildKernel("../teststructperfnostruct.cl", "test");

  TensorInfoCl info;
  const int its = 100;
  info.dims = 4;
  info.offset = 0;
  info.sizes[0] = 50;
  info.sizes[1] = 400;
  info.sizes[2] = 5;
  info.sizes[3] = 6;
  int stride = 1;
  for( int dim = info.dims - 1; dim >= 0; dim-- ) {
    info.strides[dim] = stride;
    stride *= info.sizes[dim];
    cout << "sizes[" << dim << "]=" << info.sizes[dim] << " strides[" << dim << "]=" << info.strides[dim] << endl;
  }
  int totalElements = 0;
  for( int dim = info.dims - 1; dim >= 0; dim-- ) {
    int thisElements = info.sizes[dim] * info.strides[dim];
    totalElements = thisElements > totalElements ? thisElements : totalElements;
  }
  float *data = new float[totalElements];
  for( int i = 0; i < totalElements; i++ ) {
    data[i] = i + 1;
  }
  cout << "N=" << totalElements << endl;
  double start = getSystemMilliseconds();
  int workgroupSize = 64;
  int numWorkgroups = (totalElements + workgroupSize - 1) / workgroupSize;
  cout << "numWorkgroups " << numWorkgroups << endl;
  for( int it = 0; it < its; it++ ) {
    kernel->in(totalElements);
//    kernel->in( 1, &info );
    kernel->inout( totalElements, data );
    kernel->run_1d( numWorkgroups * workgroupSize, workgroupSize );
    // cl->finish();
  }
  cl->finish();
  double end = getSystemMilliseconds();
  cout << "time " << (end-start)/1000 << endl;
//  for( int i = 0; i < totalElements; i++ ) {
//    cout << data[i] << " ";
//  }
  for( int i = 0; i < 10; i++ ) {
    cout << data[i] << " ";
  }
  cout << endl;

  delete[] data;
  delete cl;

  return 0;
}

