#include <iostream>
#include <sys/types.h>
#include <stdio.h>
#include <string.h>
#include <clBLAS.h>
#include <stdlib.h>
using namespace std;

//extern void *sgemm_Col_NN_B0_MX016_NX016_KX01_clKernel;
//extern void *sgemm_Col_NN_B0_ML016_NX016_KX01_clKernel;
//extern void *sgemm_Col_NN_B0_MX016_NL016_KX01_clKernel;
//extern void *sgemm_Col_NN_B0_ML016_NL016_KX01_clKernel;
extern cl_kernel sgemm_Col_NN_B0_MX016_NX016_KX01_clKernel;
extern cl_kernel sgemm_Col_NN_B0_ML016_NX016_KX01_clKernel;
extern cl_kernel sgemm_Col_NN_B0_MX016_NL016_KX01_clKernel;
extern cl_kernel sgemm_Col_NN_B0_ML016_NL016_KX01_clKernel;

void prompt(const char *label) {
    char in[256];
    printf("%s\n", label);
    fgets(in, sizeof(in), stdin);
}

  cl_int err;
  cl_platform_id platform = 0;
  cl_device_id device = 0;
  cl_context_properties props[3] = { CL_CONTEXT_PLATFORM, 0, 0 };
  cl_context ctx = 0;
  cl_command_queue queue = 0;
  cl_mem bufA, bufB, bufC;
  cl_event event = NULL;
  int ret = 0;

void clgemm(int colmaj, char transAchar, char transBchar, int M, int N, int K, float alpha, float *A, int lda,
     float *B, int ldb, float beta, float *C, int ldc, float *result) {
clblasTranspose transA = transAchar == 'n' ? clblasNoTrans : clblasTrans;
clblasTranspose transB = transBchar == 'n' ? clblasNoTrans : clblasTrans;

size_t off = 0;
size_t offA = 0;
size_t offB = 0;
size_t offC = 0;

clblasOrder order;
if(colmaj == 1 ) {
  order = clblasColumnMajor;
} else {
  order = clblasRowMajor;
}

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

  err = clblasSgemm(order, transA, transB, M - off, N - off, K - off,
                       alpha, bufA, offA, lda,
                       bufB, offB, ldb, beta,
                       bufC, offC, ldc,
                       1, &queue, 0, NULL, &event);
  if (err != CL_SUCCESS) {
      printf("clblasSgemmEx() failed with %d\n", err);
      ret = 1;
      exit(1);
  }
  else {
      err = clWaitForEvents(1, &event);
      err = clEnqueueReadBuffer(queue, bufC, CL_TRUE, 0,
                                M * N * sizeof(*result),
                                result, 0, NULL, NULL);
      clReleaseEvent(event);
  }

  clReleaseMemObject(bufC);
  clReleaseMemObject(bufB);
  clReleaseMemObject(bufA);

}

void checkError(cl_int err) {
  if(err != CL_SUCCESS) {
    cout << "error" << endl;
    exit(1);
  }
}

bool test1(int colmaj, int M, int N, int K, int transAint, int transBint) {
  char transa = transAint == 1 ? 't' : 'n';
  char transb = transBint == 1 ? 't' : 'n';

  float alpha = 1; int lda = 1; int ldb = 1; int ldc = 1; float beta = 0;
  float *A = new float[M * K];
  float *B = new float[K * N];
  float *C = new float[M * N];
  float *clout = new float[M * N];
  clgemm(colmaj, transa, transb, M, N, K, alpha, A, lda,
     B, ldb, beta, C, ldc, clout);
  delete[] A;
  delete[] B;
  delete[] C;
  delete[] clout;
}

void printRefCount(cl_context ctx) {
   cl_uint refCount = 123;
   size_t retsize;
  checkError(clGetContextInfo(ctx, CL_CONTEXT_REFERENCE_COUNT, sizeof(refCount), &refCount, &retsize));
   cout << " refCount " << refCount << endl;
}

int main(int argc, char *argv[]) {
  clewInit();

  err = clGetPlatformIDs(1, &platform, NULL);
  if (err != CL_SUCCESS) {
      printf( "clGetPlatformIDs() failed with %d\n", err );
      return 1;
  }
  cout << "got platforms" << endl;

  err = clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU, 1, &device, NULL);
  if (err != CL_SUCCESS) {
      printf( "clGetDeviceIDs() failed with %d\n", err );
      return 1;
  }

  props[1] = (cl_context_properties)platform;
  for(int i = 0; i < 100000; i++ ) {
    cout << "i " << i << endl;
    ctx = clCreateContext(props, 1, &device, NULL, NULL, &err);
   checkError(err);
   prompt("created context");
   printRefCount(ctx);

    queue = clCreateCommandQueue(ctx, device, 0, &err);
   checkError(err);
   prompt("created commandqueue");
   printRefCount(ctx);

    clblasSetup();
   prompt("setup blas");
   printRefCount(ctx);
    test1(1, 1, 1, 1, 0, 0);
   prompt("run gemm");
   printf("sgemm_Col_NN_B0_MX016_NX016_KX01_clKernel %li\n", (long)sgemm_Col_NN_B0_MX016_NX016_KX01_clKernel);
   printf("sgemm_Col_NN_B0_ML016_NX016_KX01_clKernel %li\n", (long)sgemm_Col_NN_B0_ML016_NX016_KX01_clKernel);
   printf("sgemm_Col_NN_B0_MX016_NL016_KX01_clKernel %li\n", (long)sgemm_Col_NN_B0_MX016_NL016_KX01_clKernel);
   printf("sgemm_Col_NN_B0_ML016_NL016_KX01_clKernel %li\n", (long)sgemm_Col_NN_B0_ML016_NL016_KX01_clKernel);

   printRefCount(ctx);

    clblasTeardown();
   prompt("torn down");

   printRefCount(ctx);

    checkError(clReleaseCommandQueue(queue));
   prompt("relesed queue");

   printRefCount(ctx);

    checkError(clReleaseContext(ctx));
   prompt("relesed context");

//              *tileKernelSource       =  sgemm_Col_NN_B0_MX016_NX016_KX01_src;
//              *rowKernelSource        =  sgemm_Col_NN_B0_ML016_NX016_KX01_src;
//              *colKernelSource        =  sgemm_Col_NN_B0_MX016_NL016_KX01_src;
//              *cornerKernelSource     =  sgemm_Col_NN_B0_ML016_NL016_KX01_src;
//              *sourceBuildOptions     =  sgemm_Col_NN_B0_MX016_NX016_KX01_srcBuildOptions;
//              *tileKernelBinary       =  sgemm_Col_NN_B0_MX016_NX016_KX01_bin;
//              *rowKernelBinary        =  sgemm_Col_NN_B0_ML016_NX016_KX01_bin;
//              *colKernelBinary        =  sgemm_Col_NN_B0_MX016_NL016_KX01_bin;
//              *cornerKernelBinary     =  sgemm_Col_NN_B0_ML016_NL016_KX01_bin;
//              *tileKernelBinarySize   = &sgemm_Col_NN_B0_MX016_NX016_KX01_binSize;
//              *rowKernelBinarySize    = &sgemm_Col_NN_B0_ML016_NX016_KX01_binSize;
//              *colKernelBinarySize    = &sgemm_Col_NN_B0_MX016_NL016_KX01_binSize;
//              *cornerKernelBinarySize = &sgemm_Col_NN_B0_ML016_NL016_KX01_binSize;
//              *binaryBuildOptions     =  sgemm_Col_NN_B0_MX016_NX016_KX01_binBuildOptions;
//              *tileClKernel           = &sgemm_Col_NN_B0_MX016_NX016_KX01_clKernel;
//              *rowClKernel            = &sgemm_Col_NN_B0_ML016_NX016_KX01_clKernel;
//              *colClKernel            = &sgemm_Col_NN_B0_MX016_NL016_KX01_clKernel;
//              *cornerClKernel         = &sgemm_Col_NN_B0_ML016_NL016_KX01_clKernel;
//              *workGroupNumRows       =  sgemm_Col_NN_B0_MX016_NX016_KX01_workGroupNumRows;
//              *workGroupNumCols       =  sgemm_Col_NN_B0_MX016_NX016_KX01_workGroupNumCols;
//              *microTileNumRows       =  sgemm_Col_NN_B0_MX016_NX016_KX01_microTileNumRows;
//              *microTileNumCols       =  sgemm_Col_NN_B0_MX016_NX016_KX01_microTileNumRows;
//              *unroll                 =  sgemm_Col_NN_B0_MX016_NX016_KX01_unroll;
   printf("sgemm_Col_NN_B0_MX016_NX016_KX01_clKernel %li\n", (long)sgemm_Col_NN_B0_MX016_NX016_KX01_clKernel);
   printf("sgemm_Col_NN_B0_ML016_NX016_KX01_clKernel %li\n", (long)sgemm_Col_NN_B0_ML016_NX016_KX01_clKernel);
   printf("sgemm_Col_NN_B0_MX016_NL016_KX01_clKernel %li\n", (long)sgemm_Col_NN_B0_MX016_NL016_KX01_clKernel);
   printf("sgemm_Col_NN_B0_ML016_NL016_KX01_clKernel %li\n", (long)sgemm_Col_NN_B0_ML016_NL016_KX01_clKernel);

   prompt("cycle");
  }

  return 0;
}

