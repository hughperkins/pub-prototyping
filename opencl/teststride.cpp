#include <iostream>
using namespace std;

void swap(int *array, int p1, int p2) {
  int old = array[p1];
  array[p1] = array[p2];
  array[p2] = old;
}
void fill(int N, int *array, int value) {
  for(int i=0; i < N; i++) {
    array[i] = value;
  }
}
void walk(int N, int *array, int dims, int *sizes, int *strides, bool transpose) {
  int *x = new int[dims];

  if(transpose) {
    swap(strides, dims-2, dims-1);
    swap(sizes, dims-2, dims-1);
  }

  for(int n = 0; n < N; n++) {
    int thisN = n;
    for(int d = dims - 1; d >= 0; d--) {
      x[d] = thisN % sizes[d];
      thisN /= sizes[d];
    }
    int storageOffset = 0;
    for(int d=0; d < dims; d++) {
      storageOffset = x[d] * strides[d];
    }
    array[storageOffset] += 1;
    cout << "n=" << n << " x=" << x[0] << "," << x[1] << " o=" << storageOffset << endl;
  }

  if(transpose) {
    swap(strides, dims-2, dims-1);
    swap(sizes, dims-2, dims-1);
  }

  delete[] x;
}

void check(int N, int *array) {
  int violationCount = 0;
  for(int i = 0; i < N; i++) {
    if(array[i] != 1) {
      cout << "violation: array[" << i << "]=" << array[i] << endl;
      violationCount++;
    }
  }
  cout << "violations: " << violationCount << endl;
}

int main(int argc, char *argv[] ) {
  const int nDims = 2;
  int sizes[] = {32, 4};
  int strides[] = {4, 1};
  int numElements = 1;
  for(int d = 0; d < nDims; d++ ) {
    numElements *= sizes[d];
  }
  cout << "numElements " << numElements << endl;
  int *array = new int[numElements];
  fill(numElements, array, 0);
  walk(numElements, array, nDims, sizes, strides, true);
  delete[] array;
}

