#include <sys/time.h>
#include <stdio.h>
#include <unistd.h>
#include <iostream>
using namespace std;

#include "EasyCL.h"
#include "CLKernel_structs.h"

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

  CLKernel *kernel = cl->buildKernel("../testfloat4.cl", "test");

  const int its = 200;
  int totalElements = 600000;
  float *data = new float[totalElements];
  for( int i = 0; i < totalElements; i++ ) {
    data[i] = i + 1;
  }
  cout << "N=" << totalElements << endl;
  int workgroupSize = 64;
  int numWorkgroups = (totalElements / 4 + workgroupSize - 1) / workgroupSize;
  cout << "numWorkgroups " << numWorkgroups << endl;
  kernel->in(totalElements / 4);
  kernel->inout( totalElements, data );
  kernel->run_1d( numWorkgroups * workgroupSize, workgroupSize );
  cl->finish();
  double start = getSystemMilliseconds();

  for( int it = 0; it < its; it++ ) {
    kernel->in(totalElements / 4);
    kernel->inout( totalElements, data );
    kernel->run_1d( numWorkgroups * workgroupSize, workgroupSize );
  }
  cl->finish();
  double end = getSystemMilliseconds();
  cout << "time " << (end-start)/1000 << endl;
  for( int i = 0; i < 10; i++ ) {
    cout << data[i] << " ";
  }
  cout << endl;

  delete[] data;
  delete cl;

  return 0;
}

