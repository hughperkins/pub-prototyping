#include "EasyCL/EasyCL.h"

#include <iostream>
#include <stdexcept>

#include "cocl/cocl.h"

using namespace std;

string kernelSource = R"(
kernel void _z8setValuePfif(global float* data, long data_offset, int idx, float value) {
   data = (global float*)((global char *)data + data_offset);

   label0:;
   int v1 = get_local_id(0);
    bool v2 = v1 == 0;
    if(v2) {
        goto v4;
    } else {
        goto v5;
    }
    v4:;
    long v6 = idx;
    global float* v7 = (&data[v6]);
    v7[0] = value;
        goto v5;
    v5:;
    return;
}
)";

namespace cocl {
    easycl::CLKernel *getKernelForName(string name, string sourcecode);
}
void configureKernel(const char *kernelName, const char *devicellsourcecode, const char *clSourcecodeString);

int main(int argc, char *argv[]) {
    cl_int err;

    cocl::ThreadVars *v = cocl::getThreadVars();
    cocl::Context *coclContext = v->getContext();
    easycl::EasyCL *cl = coclContext->cl.get();

    int N = 32;
    cl_float *a = new cl_float[N];
    cl_long offset = 0;
    cl_int pos = 2;
    cl_float value = 123;

    for(int i = 0; i < N; i++) {
        a[i] = 555;
    }

    cocl::Memory *a_memory = cocl::Memory::newDeviceAlloc(N * sizeof(float));

    err = clEnqueueWriteBuffer(*cl->queue, a_memory->clmem, CL_TRUE, 0,
                                         sizeof(cl_float) * N, a, 0, NULL, NULL);
    easycl::EasyCL::checkError(err);

    easycl::CLKernel *kernel = cocl::getKernelForName("_z8setValuePfif", kernelSource);

    // string kernelName = "_z8setValuePfif";
    // configureKernel(kernelName.c_str(), "", kernelSource.c_str());
    // setKernelArgCharStar((char *)
    
    // easycl::CLKernel *kernel = cl->buildKernelFromString(kernelSource, "_z8setValuePfif", "");

    kernel
        ->inout(&a_memory->clmem)
        ->in(offset)
        ->in(pos)
        ->in(value);
    kernel->run_1d(N, 32);

    err = clEnqueueReadBuffer(*cl->queue, a_memory->clmem, CL_TRUE, 0,
                                         sizeof(cl_float) * N, a, 0, NULL, NULL);
    easycl::EasyCL::checkError(err);
    cl->finish();
    cout << "clfinish finished ok" << endl;

    for(int i = 0; i < 5; i++) {
        cout << "a[" << i << "]=" << a[i] << endl;
    }

    return 0;
}
