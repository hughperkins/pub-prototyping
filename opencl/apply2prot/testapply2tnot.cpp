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

void transpose(TensorInfoCl *info, int dim0, int dim1) {
  int tempStride = info->strides[dim0];
  int tempSize = info->sizes[dim0];
  info->strides[dim0] = info->strides[dim1];
  info->sizes[dim0] = info->sizes[dim1];
  info->strides[dim1] = tempStride;
  info->sizes[dim1] = tempSize;
}

int main(int argc, char *argv[] ) {
  EasyCL *cl = EasyCL::createForIndexedGpu(0);

  CLKernel *kernel = cl->buildKernel("../testapply2tnot.cl", "test");

  TensorInfoCl a_info;
  const int its = 10000;
  a_info.dims = 4;
  a_info.offset = 0;
  a_info.sizes[0] = 500;
  a_info.sizes[1] = 50;
  a_info.sizes[2] = 1;
  a_info.sizes[3] = 1;
  int stride = 1;
  for( int dim = a_info.dims - 1; dim >= 0; dim-- ) {
    a_info.strides[dim] = stride;
    stride *= a_info.sizes[dim];
    cout << "sizes[" << dim << "]=" << a_info.sizes[dim] << " strides[" << dim << "]=" << a_info.strides[dim] << endl;
  }
  int totalElements = 0;
  for( int dim = a_info.dims - 1; dim >= 0; dim-- ) {
    int thisElements = a_info.sizes[dim] * a_info.strides[dim];
    totalElements = thisElements > totalElements ? thisElements : totalElements;
  }
  TensorInfoCl b_info;
  b_info.dims = a_info.dims;
  b_info.offset = a_info.offset;
  for( int dim = 0; dim < a_info.dims; dim++ ) {
    b_info.sizes[dim] = a_info.sizes[dim];
    b_info.strides[dim] = a_info.strides[dim];
  }
  transpose(&a_info, 2, 3);

  float *a = new float[totalElements];
  float *b = new float[totalElements];

  for( int i = 0; i < totalElements; i++ ) {
    a[i] = i + 1;
    b[i] = i * 2;
  }
  CLWrapper *awrapper = cl->wrap( totalElements, a );
  CLWrapper *bwrapper = cl->wrap( totalElements, b );
  awrapper->copyToDevice();
  bwrapper->copyToDevice();
  cl->finish();
  cout << "N=" << totalElements << endl;
  double start = getSystemMilliseconds();
  for( int it = 0; it < its; it++ ) {
    kernel->in(totalElements);
    kernel->in( 1, &a_info );
    kernel->inout( awrapper );
    kernel->in( 1, &b_info );
    kernel->in( bwrapper );
    int workgroupSize = 64;
    int numWorkgroups = (totalElements + workgroupSize - 1) / workgroupSize;
    kernel->run_1d( numWorkgroups * workgroupSize, workgroupSize );
    // cl->finish();
  }
  cl->finish();
  double end = getSystemMilliseconds();
  cout << "time " << (end-start)/1000 << endl;
//  for( int i = 0; i < totalElements; i++ ) {
//    cout << data[i] << " ";
//  }
  awrapper->copyToHost();
  for( int i = 0; i < 10; i++ ) {
    cout << a[i] << " ";
  }
  cout << endl;

  delete[] a;
  delete[] b;
  delete cl;

  return 0;
}

