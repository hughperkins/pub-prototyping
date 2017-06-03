#include <iostream>
using namespace std;
#include "EasyCL.h"
#include "CLKernel_structs.h"
#include "util/StatefulTimer.h"
#include "util/easycl_stringhelper.h"

static const char *kernelSource = R"DELIM(
  kernel void test(int totalN,
      global float *out,
      global float *in1,
      global float *in2,
      global float *in3,
      global float *in4,
      global float *in5,
      global float *in6
      ) {
    int linearId = get_global_id(0);
    if(linearId < totalN) {
      out[linearId] = in1[linearId] + in2[linearId] + in3[linearId] + in4[linearId] + in5[linearId] + in6[linearId];
    }
  }
)DELIM";

void test(EasyCL *cl, int its, int size, bool reuseStructBuffers) {
  int totalN = size;
  string templatedSource = kernelSource;
  CLKernel *kernel = cl->buildKernelFromString(templatedSource, "test", "");
  const int workgroupSize = 64;
  int numWorkgroups = (totalN + workgroupSize - 1) / workgroupSize;

  float *out = new float[totalN];
  float *in = new float[totalN];
  for( int i = 0; i < totalN; i++ ) {
      in[i] = (i + 4) % 1000000;
  }
  CLWrapper *outwrap = cl->wrap(totalN, out);
  CLWrapper *inwrap = cl->wrap(totalN, in);
  inwrap->copyToDevice();
  outwrap->createOnDevice();

  cl->finish();
  cl->dumpProfiling();

  double start = StatefulTimer::instance()->getSystemMilliseconds();
  for(int it = 0; it < its; it++) {
    kernel->in(totalN);
    kernel->out(outwrap);
    kernel->in(inwrap);
    kernel->in(inwrap);
    kernel->in(inwrap);
    kernel->in(inwrap);
    kernel->in(inwrap);
    kernel->in(inwrap);

    kernel->run_1d(numWorkgroups * workgroupSize, workgroupSize);
  }
  cl->finish();
  double end = StatefulTimer::instance()->getSystemMilliseconds();
  cl->dumpProfiling();
  cout << "its=" << its << " size=" << size << " time=" << (end - start) << "ms" << endl;
  outwrap->copyToHost();
  cl->finish();
  int errorCount = 0;
  for( int i = 0; i < totalN; i++ ) {
    float targetValue = in[i] * 6;
    if(abs(out[i] - targetValue)> 0.1f) {
      errorCount++;
      if( errorCount < 20 ) {
        cout << "out[" << i << "]" << " != " << targetValue << endl;
        cout << abs(out[i] - targetValue) << endl;
      }
    }
  }
//  cout << endl;
  if( errorCount > 0 ) {
    cout << "errors: " << errorCount << " out of totalN=" << totalN << endl;
  } else {
  }

  delete outwrap;
  delete inwrap;
  delete[] in;
  delete[] out;
  delete kernel;
}

int main(int argc, char *argv[]) {
  int gpu = 0;
  if( argc == 2 ) {
    gpu = atoi(argv[1]);
  }
  cout << "using gpu " << gpu << endl;
  EasyCL *cl = EasyCL::createForIndexedGpu(gpu);
  cl->setProfiling(true);
  test(cl, 900, 6400, true);
  test(cl, 9000, 6400, true);
  cl->dumpProfiling();
  delete cl;
  return 0;
}


