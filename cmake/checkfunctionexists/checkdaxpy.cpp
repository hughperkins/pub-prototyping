#include <iostream>
using namespace std;

extern "C" {
    void daxpy_(int *n, double *a, double *X, int *incx, double *Y, int *incy );
}

int main( int argc, char *argv[] ) {
    int a = 1;
    double d = 1;
    double A[5];
    double B[5];
    daxpy_( &a, &d, A, &a, B, &a );
    return 0;
}


