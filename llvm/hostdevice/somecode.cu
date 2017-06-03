#include "myincludes.h"

__device__ void devicefunc() {

}

__host__ void hostfunc() {

}

__host__ __device__ void hostanddevicefunc() {

}

__global__ void kernelfunc() {

}

__device__ int bitcasti(float in) {
    return *(int *)&in;
}

__device__ float bitcastf(int in) {
    return *(float *)&in;
}

__global__ void hasshared() {
    __shared__ float foo[32];
}
