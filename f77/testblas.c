#include <stdio.h>

#include <cblas_f77.h>

int main( int argc, char *argv[] ) {
    double A[4];
    A[0] = 3; A[2] = 2;
    A[1] = 4; A[3] = 1;
    double B[4];
    B[0] = 2; B[2] = 0;
    B[1] = 0; B[3] = 3;
    int m = 2;
    int n = 2;
    int k = 2;
    double alpha = 1;
    double beta = 0;
    double C[4];
    dgemm_("n","n",&m,&n,&k, &alpha, A, &m, B, &k, &beta, C, &m );
    printf("%g %g\n", C[0], C[2] );
    printf("%g %g\n", C[1], C[3] );
    return 0;
}



