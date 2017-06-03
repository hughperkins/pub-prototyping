#include <iostream>
#include <sys/types.h>
#include <stdio.h>
#include <string.h>
#include <clBLAS.h>
#include <stdlib.h>
using namespace std;

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

bool boolxor(bool a, bool b) {
  return (!a) != (!b);
}

bool test1(int colmaj, int M, int N, int K, int transAint, int transBint) {
  char transa = transAint == 1 ? 't' : 'n';
  char transb = transBint == 1 ? 't' : 'n';

  float alpha = 1;

  bool flipa = boolxor(colmaj != 1, transAint != 0);
  bool flipb = boolxor(colmaj != 1, transBint != 0);

  int lda = flipa ? K : M;
  int ldb = flipb ? N : K;
  int ldc = (colmaj == 1) ? M : N;

  float beta = 0;
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
  ctx = clCreateContext(props, 1, &device, NULL, NULL, &err);
  queue = clCreateCommandQueue(ctx, device, 0, &err);
  clblasSetup();

  for(int it = 0; it < 3; it++) {
    test1(1, 4, 3, 16, 1, 0);
//    test1(1, 4, 16, 3, 1, 0);
  }

  clblasTeardown();
  checkError(clReleaseCommandQueue(queue));
  checkError(clReleaseContext(ctx));

  return 0;
}

